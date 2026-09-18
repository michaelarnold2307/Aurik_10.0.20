"""§PERF-R P9 (2026-09-18) — HF-Whisper-Gerätewahl (Torch-ROCm) mit Fail-Closed.

Der HF-Decoder-Pfad der LGE-Transkription wählt jetzt ein Zielgerät:
GPU (cuda:0), wenn torch.cuda verfügbar ist und AURIK_WHISPER_GPU != 0;
sonst CPU. Jeder GPU-Fehler (Transfer ODER Generierung) fällt sichtbar
(§V6 (copilot-instructions.md)) EINMAL auf CPU zurück. Alle Tests laufen ohne
echte GPU (Fakes + gemocktes torch.cuda).
"""

from __future__ import annotations

import contextlib
import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.core.lyrics_guided_enhancement import LyricsGuidedEnhancement


class _FakeFeatures:
    """Fake-Input-Tensor: to()/cpu() identität, device.type steuerbar."""

    def __init__(self, device_type: str = "cpu"):
        self.device = MagicMock(type=device_type)

    def to(self, _device):
        return self

    def cpu(self):
        self.device = MagicMock(type="cpu")
        return self


class _FakeProcessor:
    def __init__(self):
        self.features = _FakeFeatures()

    def __call__(self, audio, sampling_rate=16_000, return_tensors="pt"):
        return MagicMock(input_features=self.features)

    def batch_decode(self, ids, skip_special_tokens=True):
        return ["Test"]


class _FakeModel:
    def __init__(self, fail_transfer: bool = False, fail_generate_first: bool = False):
        self.to_calls: list[str] = []
        self._fail_transfer = fail_transfer
        self._fail_generate_first = fail_generate_first
        self.generate_calls = 0

    def eval(self):
        return self

    def to(self, device: str):
        self.to_calls.append(device)
        if self._fail_transfer and device != "cpu":
            raise RuntimeError("simulierter GPU-Transfer-Fehler")
        return self

    def generate(self, input_features, **kwargs):
        self.generate_calls += 1
        if self._fail_generate_first and self.generate_calls == 1:
            raise RuntimeError("simulierter GPU-Generierungs-Fehler")
        return MagicMock(device=MagicMock(type="cpu"), tolist=lambda: [], dim=lambda: 1)


def _make_lge() -> LyricsGuidedEnhancement:
    lge = LyricsGuidedEnhancement.__new__(LyricsGuidedEnhancement)
    lge._whisper_hf_processor = None
    lge._whisper_hf_model = None
    lge._whisper_hf_device = "cpu"
    return lge


@contextlib.contextmanager
def _load_patches(model: _FakeModel, processor: _FakeProcessor, cuda_available: bool):
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            patch(
                "transformers.WhisperForConditionalGeneration.from_pretrained",
                return_value=model,
            )
        )
        stack.enter_context(patch("transformers.WhisperProcessor.from_pretrained", return_value=processor))
        stack.enter_context(patch("torch.cuda.is_available", return_value=cuda_available))
        stack.enter_context(patch("backend.core.ml_memory_budget.try_allocate", return_value=True))
        stack.enter_context(patch("backend.core.plugin_lifecycle_manager.register_plugin", return_value=None))
        # Hosts ohne Modell-Blob-Store: config.json-Symlink-Existenz erzwingen —
        # der Gerätewahl-Test braucht den Ladepfad, nicht die echten Gewichte.
        stack.enter_context(patch("pathlib.Path.exists", return_value=True))
        yield


@pytest.mark.unit
def test_gpu_available_moves_model_to_cuda(monkeypatch):
    """P9: GPU verfügbar → Modell auf cuda:0, Geräte-Flag gesetzt."""
    model = _FakeModel()
    lge = _make_lge()
    with _load_patches(model, _FakeProcessor(), cuda_available=True):
        lge._try_load_hf_whisper()
    assert lge._whisper_hf_device == "cuda:0"
    assert "cuda:0" in model.to_calls


@pytest.mark.unit
def test_kill_switch_forces_cpu(monkeypatch):
    """P9: AURIK_WHISPER_GPU=0 erzwingt CPU trotz verfügbarer GPU."""
    monkeypatch.setenv("AURIK_WHISPER_GPU", "0")
    model = _FakeModel()
    lge = _make_lge()
    with _load_patches(model, _FakeProcessor(), cuda_available=True):
        lge._try_load_hf_whisper()
    assert lge._whisper_hf_device == "cpu"
    assert model.to_calls == []


@pytest.mark.unit
def test_no_gpu_keeps_cpu(monkeypatch):
    """P9: Kein CUDA → CPU (bisheriges Verhalten)."""
    model = _FakeModel()
    lge = _make_lge()
    with _load_patches(model, _FakeProcessor(), cuda_available=False):
        lge._try_load_hf_whisper()
    assert lge._whisper_hf_device == "cpu"
    assert model.to_calls == []


@pytest.mark.unit
def test_gpu_transfer_error_falls_back_to_cpu_with_warning(monkeypatch, caplog):
    """P9: Transfer-Fehler → sichtbare Warnung (§V6 (copilot-instructions.md)) + CPU-Rückfall."""
    model = _FakeModel(fail_transfer=True)
    lge = _make_lge()
    with caplog.at_level(logging.WARNING):
        with _load_patches(model, _FakeProcessor(), cuda_available=True):
            lge._try_load_hf_whisper()
    assert lge._whisper_hf_device == "cpu"
    assert model.to_calls[-1] == "cpu"
    assert any("GPU-Transfer fehlgeschlagen" in r.message for r in caplog.records)


@pytest.mark.unit
def test_transcribe_gpu_generation_error_retries_on_cpu(monkeypatch, caplog):
    """P9: GPU-Generierung wirft → EINMAL sichtbar auf CPU wiederholen."""
    model = _FakeModel(fail_generate_first=True)
    processor = _FakeProcessor()
    lge = _make_lge()
    lge._whisper_hf_processor = processor
    lge._whisper_hf_model = model
    lge._whisper_hf_device = "cuda:0"

    # Rest der Pipeline abklemmen — Fokus auf Geräte-Rückfall:
    lge._resample = lambda mono, src, tgt: np.zeros(1600, dtype=np.float32)
    lge._parse_hf_tokens_to_words = lambda ids, mono, dur: []
    lge._align_phonemes = lambda words, mono, sr: words
    lge._detect_language_from_mono = lambda mono, sr: ("unknown", 0.0)

    with caplog.at_level(logging.WARNING):
        result = lge._transcribe_hf(np.zeros(4800, dtype=np.float32), 48_000, 0.1)

    assert model.generate_calls == 2, "Erst GPU (Fehler), dann CPU-Wiederholung"
    assert lge._whisper_hf_device == "cpu"
    assert model.to_calls[-1] == "cpu"
    assert any("GPU-Generierung fehlgeschlagen" in r.message for r in caplog.records)
    assert result.fallback_used is False


@pytest.mark.unit
def test_transcribe_cpu_generation_error_propagates(monkeypatch):
    """P9: CPU-Generierungsfehler wird NICHT verschluckt (kein Endlos-Retry)."""
    model = _FakeModel(fail_generate_first=True)
    processor = _FakeProcessor()
    lge = _make_lge()
    lge._whisper_hf_processor = processor
    lge._whisper_hf_model = model
    lge._whisper_hf_device = "cpu"
    lge._resample = lambda mono, src, tgt: np.zeros(1600, dtype=np.float32)

    with pytest.raises(RuntimeError, match="GPU-Generierungs-Fehler"):
        lge._transcribe_hf(np.zeros(4800, dtype=np.float32), 48_000, 0.1)
