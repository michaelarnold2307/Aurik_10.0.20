"""§SOTA-ML-V6 — Whisper-Tiny-Torch-ROCm-Encoder: Unit-Tests (CPU, Fake-Core).

Der echte Device-Pfad bleibt im GPU-Benchmark/Produktionslauf abgedeckt;
hier wird mit einem Fake-Core (Identity-Encoder) und einem optionalen
CPU-Paritätstest gegen den ONNX-Encoder geprüft. Geprüft: Form/NaN-Guard,
Fail-closed (§V6 (copilot-instructions.md)), Parität (max|Δ| ≤ 5e-3 bei
lokal gecachtem HF-Snapshot — sonst skip).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from backend.core.dsp import whisper_torch_rocm as wtr


def _fake_core():
    """Identity-Encoder-Bundle: last_hidden_state = Eingangs-Mel."""

    class _IdentityEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._p = torch.nn.Parameter(torch.zeros(1))

        def forward(self, mel):
            # [1,80,T] → [1,T,384]-Form nachahmen (deterministisch, billig)
            t = mel.transpose(1, 2).repeat(1, 1, 5)[:, :, :384]
            return type("Hidden", (), {"last_hidden_state": t})()

    class _ModelShell(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = _IdentityEncoder()

        def get_encoder(self):
            return self.encoder

    return {"model": _ModelShell().eval(), "extractor": None}


def test_encode_mel_shape_and_nan_guard():
    core = _fake_core()
    rng = np.random.default_rng(3)
    mel = rng.standard_normal((1, 80, 3000)).astype(np.float32)
    out = wtr.encode_whisper_torch_mel(core, mel)
    assert out.shape == (1, 3000, 384)
    assert out.dtype == np.float32
    mel_bad = mel.copy()
    mel_bad[0, 0, 0] = np.nan
    out_bad = wtr.encode_whisper_torch_mel(core, mel_bad)
    assert np.isfinite(out_bad).all()


def test_encode_mel_rejects_wrong_shape():
    core = _fake_core()
    with pytest.raises(ValueError):
        wtr.encode_whisper_torch_mel(core, np.zeros((80, 3000), dtype=np.float32))


def test_fail_closed_without_cuda(monkeypatch):
    """Ohne CUDA/ROCm muss get_whisper_torch_core() None liefern (ONNX-CPU-Pfad bleibt)."""
    if torch.cuda.is_available():
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(wtr, "_core", None)
    monkeypatch.setattr(wtr, "_core_resolved", False)
    assert wtr.get_whisper_torch_core() is None


def test_fail_closed_without_snapshot(monkeypatch):
    """Fehlender HF-Snapshot → None (fail-closed, kein Netz zur Laufzeit)."""
    monkeypatch.setattr(wtr, "_core", None)
    monkeypatch.setattr(wtr, "_core_resolved", False)
    monkeypatch.setattr(wtr, "_find_snapshot", lambda: None)
    assert wtr.get_whisper_torch_core() is None


def test_transcribe_word_grouping(monkeypatch):
    """§SOTA-ML-V9: Token-Timestamps → Wort-Liste (Stub-generate)."""

    class _Tok:
        eos_token_id = 6

        @staticmethod
        def convert_ids_to_tokens(tid):
            return {
                0: "<|startoftranscript|>",
                1: "<|en|>",
                2: "<|0.00|>",
                3: "Ġhello",
                4: "Ġworld",
                5: "<|2.00|>",
                6: "<|endoftext|>",
            }[tid]

        @staticmethod
        def decode(ids, skip_special_tokens=False):
            _m = {3: "hello", 4: "world"}
            return " ".join(_m[i] for i in ids if i in _m)

    class _Feats:
        def __init__(self):
            self.input_features = torch.zeros(1, 80, 3000)

    class _Proc:
        tokenizer = _Tok()

        def __call__(self, x, sampling_rate=16000, return_tensors=None):
            return _Feats()

    class _Gen:
        sequences = torch.tensor([[0, 1, 2, 3, 4, 5, 6]])
        token_timestamps = [torch.tensor([[0.0, 0.0, 0.0, 0.2, 0.4, 2.0, 2.0]])]
        scores = [torch.zeros(1, 51864) for _ in range(4)]

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._p = torch.nn.Parameter(torch.zeros(1))

        def generate(self, *a, **kw):
            return _Gen()

    core = {"model": _Model().eval(), "processor": _Proc()}
    words = wtr.transcribe_whisper_torch(core, np.zeros(16000 * 5, dtype=np.float32), language="en")
    assert len(words) == 2
    assert words[0]["word"] == "hello"
    assert abs(words[0]["start"] - 0.2) < 1e-6 and abs(words[0]["end"] - 0.2) < 1e-6
    assert words[1]["word"] == "world"
    assert abs(words[1]["start"] - 0.4) < 1e-6 and abs(words[1]["end"] - 0.4) < 1e-6


def test_transcribe_empty_transcript(monkeypatch):
    """Leeres Transkript (EOT sofort) → leere Liste, keine Exception."""

    class _Tok:
        eos_token_id = 6

        @staticmethod
        def convert_ids_to_tokens(tid):
            return {0: "<|startoftranscript|>", 1: "<|en|>", 6: "<|endoftext|>"}[tid]

        @staticmethod
        def decode(ids, skip_special_tokens=False):
            return ""

    class _Feats:
        input_features = torch.zeros(1, 80, 3000)

    class _Proc:
        tokenizer = _Tok()

        def __call__(self, x, sampling_rate=16000, return_tensors=None):
            return _Feats()

    class _Gen:
        sequences = torch.tensor([[0, 1, 6]])
        token_timestamps = [torch.tensor([[0.0, 0.0, 0.0]])]
        scores = [torch.zeros(1, 51864)]

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._p = torch.nn.Parameter(torch.zeros(1))

        def generate(self, *a, **kw):
            return _Gen()

    core = {"model": _Model().eval(), "processor": _Proc()}
    assert wtr.transcribe_whisper_torch(core, np.zeros(16000, dtype=np.float32)) == []


def test_cpu_parity_vs_onnx():
    """Torch-Encoder == ONNX-CPU (max|Δ| ≤ 5e-3). Skip ohne HF-Snapshot."""
    onnxruntime = pytest.importorskip("onnxruntime")
    snapshot = wtr._find_snapshot()
    if snapshot is None:
        pytest.skip("Whisper-Tiny-Snapshot nicht lokal gecacht")
    from pathlib import Path

    import librosa
    import soundfile as sf
    from transformers import WhisperFeatureExtractor

    core = wtr._build_core()
    extractor = WhisperFeatureExtractor.from_pretrained(snapshot)
    audio, sr = sf.read(Path("models/banquet/test_input.wav"))
    mono16 = librosa.resample(audio.mean(axis=1), orig_sr=sr, target_sr=16000)
    mel = extractor(mono16, sampling_rate=16000, return_tensors="pt").input_features.numpy()
    out = wtr.encode_whisper_torch_mel(core, mel)
    sess = onnxruntime.InferenceSession("models/whisper/whisper_tiny.onnx", providers=["CPUExecutionProvider"])
    ref = sess.run(None, {sess.get_inputs()[0].name: mel})[0]
    assert np.abs(out - ref).max() <= 5e-3
