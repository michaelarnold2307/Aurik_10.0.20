"""Cantus Vocal Restorer — Norm-Suite (CPU, tiny-Preset).

Sichert die Cantus-Pipeline (models/cantus/, scripts/generate_synthetic_degraded_vocals.py,
scripts/train_cantus.py, scripts/export_cantus_onnx.py, plugins/cantus_plugin.py):
  - Modell: Shape-Integrität, Determinismus (§G5 (GEBOTE.md)), Multi-Scale
    Bandaufteilung mit perfekter Rekonstruktion (low + high ≡ x)
  - Losses: Zero bei perfektem v, Gradientenfluss, SingMOS-Learned-Loss
  - Datensatz: deterministische Degradation inkl. §V5 (VERBOTEN.md) TPDF-Dither
  - ONNX-Export: Paritäts-Gate rel ≤ 1e-3 (§III.9 (copilot-instructions.md))
  - Plugin: §V6 (VERBOTEN.md) Ersatzpfad, Layout-Roundtrip (Stereo-Layout-
    Invariante), Material-Gate, CPU-Ende-zu-Ende-Inferenz auf Sample-Vocal
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample_vocal(n: int = 24576, sr: int = 48000, seed: int = 0) -> np.ndarray:
    """Deterministischer Sample-Vocal-Track: Vibrato + Harmonische + Hüllkurve."""
    t = np.arange(n, dtype=np.float32) / sr
    f0 = 220.0 + 15.0 * np.sin(2.0 * np.pi * 5.5 * t)
    phase = 2.0 * np.pi * np.cumsum(f0) / sr
    wave = (0.5 * np.sin(phase) + 0.25 * np.sin(2.0 * phase) + 0.1 * np.sin(3.0 * phase)).astype(np.float32)
    rng = np.random.default_rng(seed)
    noise: np.ndarray = np.asarray(rng.standard_normal(n), dtype=np.float32)
    signal: np.ndarray = wave + (0.01 * noise)
    return signal.astype(np.float32, copy=False)


@pytest.fixture(scope="module")
def tiny_model():
    from models.cantus.cantus_model import CantusConfig, create_cantus

    torch.manual_seed(20261004)
    config = CantusConfig(
        dim=32,
        depth=2,
        heads=2,
        patch_size=256,
        mlp_ratio=2.0,
        dropout=0.0,
        band_kernel_size=65,
        high_hidden=8,
        high_depth=1,
        high_kernel=5,
        mert_dim=1024,
        pitch_dim=2,
        harmonic_dim=768,
        cond_dropout=0.0,
        max_tokens=512,
    )
    model = create_cantus(config)
    model.eval()
    return model


@pytest.fixture(scope="module")
def tiny_feeds():
    from scripts.export_cantus_onnx import make_feeds

    return make_feeds(seed=7, t_samples=8192, f_frames=8)


# ── Modell-Architektur ───────────────────────────────────────────────────────


def test_model_shape_and_finite(tiny_model, tiny_feeds):
    with torch.no_grad():
        v = tiny_model(**tiny_feeds)
    assert v.shape == tiny_feeds["x"].shape, "Shape-Integrität der Velocity verletzt"
    assert bool(torch.isfinite(v).all()), "Nicht-finite Modellausgabe (§0a (copilot-instructions.md))"


def test_model_deterministic(tiny_model, tiny_feeds):
    with torch.no_grad():
        a = tiny_model(**tiny_feeds)
        b = tiny_model(**tiny_feeds)
    assert torch.equal(a, b), "Determinismus verletzt (§G5 (GEBOTE.md))"


def test_model_unconditioned_null_tokens(tiny_model, tiny_feeds):
    feeds = dict(tiny_feeds)
    feeds["use_cond"] = torch.tensor([0.0])
    feeds["mert"] = torch.zeros(1, 1, 1024)
    feeds["pitch"] = torch.zeros(1, 1, 2)
    feeds["harm"] = torch.zeros(1, 768)
    with torch.no_grad():
        v = tiny_model(**feeds)
    assert v.shape == feeds["x"].shape
    assert bool(torch.isfinite(v).all())


def test_band_split_perfect_reconstruction(tiny_model):
    from models.cantus.cantus_model import FIRBandSplit

    splitter = FIRBandSplit(kernel_size=65, cutoff_hz=2000.0, sample_rate=48000)
    x = torch.from_numpy(_sample_vocal(4096)).reshape(1, -1, 1)
    low, high = splitter(x)
    err = float(torch.max(torch.abs((low + high) - x)))
    assert err < 1e-5, f"Komplementäre Bänder rekonstruieren nicht (max err {err:.2e})"


# ── Multi-Objective-Losses ───────────────────────────────────────────────────


def test_losses_zero_at_perfect_velocity():
    tc = _load_script("train_cantus")
    torch.manual_seed(0)
    y = torch.randn(2, 1, 16384) * 0.1
    x = y + torch.randn_like(y) * 0.05
    t = torch.rand(2)
    t_col = t.reshape(-1, 1, 1)
    x_t = (1.0 - t_col) * x + t_col * y
    v = (y - x).clone()
    loss_fn = tc.MultiObjectiveLoss(use_singmos=False)
    _, comps, y_hat = loss_fn(v, y - x, x_t, y, t)
    assert torch.allclose(y_hat, y, atol=1e-6), "Rekonstruktionsformel ŷ = x_t + (1−t)·v̂ verletzt"
    for key in ("flow_matching", "mel_spectral", "stft_phase", "pitch_preservation", "temporal"):
        assert float(comps[key]) < 1e-4, f"{key} nicht Null bei perfektem v"


def test_losses_gradient_flow_and_singmos():
    tc = _load_script("train_cantus")
    torch.manual_seed(1)
    y = torch.randn(1, 1, 16384) * 0.1
    x = y + torch.randn_like(y) * 0.05
    t = torch.rand(1)
    t_col = t.reshape(-1, 1, 1)
    x_t = (1.0 - t_col) * x + t_col * y
    v = (y - x).clone().requires_grad_(True)
    loss_fn = tc.MultiObjectiveLoss(use_singmos=True)
    total, comps, _ = loss_fn(v, y - x, x_t, y, t)
    assert float(comps["singmos_perceptual"]) >= 0.0, "SingMOS-Hinge darf nicht negativ sein"
    total.backward()
    assert v.grad is not None and bool(torch.isfinite(v.grad).all()), "Gradientenfluss verletzt"


# ── Synthetische Degradation (§V5 (VERBOTEN.md)) ─────────────────────────────


def test_degradations_deterministic_and_finite():
    gen = _load_script("generate_synthetic_degraded_vocals")
    clean = np.stack([_sample_vocal(8192, seed=3)])
    for kind in gen.KINDS:
        np.random.seed(1234)
        rng_a = np.random.default_rng(42)
        out_a, params_a = gen.degrade_one(clean, kind, rng_a)
        np.random.seed(1234)
        rng_b = np.random.default_rng(42)
        out_b, params_b = gen.degrade_one(clean, kind, rng_b)
        assert params_a == params_b, f"{kind}: Parameter nicht deterministisch"
        assert np.array_equal(out_a, out_b), f"{kind}: Ausgabe nicht deterministisch (§G5 (GEBOTE.md))"
        assert np.all(np.isfinite(out_a)), f"{kind}: nicht finite (§0a (copilot-instructions.md))"
        assert out_a.shape == clean.shape, f"{kind}: Shape verletzt"


def test_synthetic_generator_resume_preserves_complete_tracks(tmp_path):
    from scipy.io import wavfile

    gen = _load_script("generate_synthetic_degraded_vocals")
    musdb_root = tmp_path / "musdb"
    out_root = tmp_path / "pairs"
    for track in ("Track A", "Track B"):
        source = musdb_root / "train" / track / "vocals.wav"
        source.parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(source, 48000, np.zeros(4096, dtype=np.int16))

    existing_dir = out_root / "train"
    existing_dir.mkdir(parents=True)
    clean_path = existing_dir / "Track A__clean.wav"
    degraded_path = existing_dir / "Track A__v00__add_noise.wav"
    clean_path.touch()
    degraded_path.touch()
    existing_row = {
        "track": "Track A",
        "split": "train",
        "variant": 0,
        "kind": "add_noise",
        "params": {"snr_db": 5.0, "noise_kind": "white"},
        "clean": "train/Track A__clean.wav",
        "degraded": "train/Track A__v00__add_noise.wav",
        "sr": 48000,
        "seed": gen._seed_for(gen.DEFAULT_SEED, "Track A", 0),
        "channels": 1,
    }
    manifest = out_root / "manifest.jsonl"
    manifest.write_text(json.dumps(existing_row) + "\n", encoding="utf-8")

    gen.generate(musdb_root, out_root, variants_per_track=1, resume=True)
    first_manifest = manifest.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in first_manifest.splitlines()]
    assert [row["track"] for row in rows] == ["Track A", "Track B"]
    assert rows[0] == existing_row, "Vollstaendiger Track wurde veraendert"
    assert rows[1]["seed"] == gen._seed_for(gen.DEFAULT_SEED, "Track B", 0)

    gen.generate(musdb_root, out_root, variants_per_track=1, resume=True)
    assert manifest.read_text(encoding="utf-8") == first_manifest, "Resume ist nicht idempotent"


def test_bitcrush_uses_dither_not_naked_cast():
    gen = _load_script("generate_synthetic_degraded_vocals")
    clean = np.stack([np.full(2048, 0.3173, dtype=np.float32)])
    np.random.seed(7)
    crushed, params = gen.degrade_one(clean, "bitcrush", np.random.default_rng(7))
    assert params["dither"] == "tpdf", "§V5 (VERBOTEN.md): Bitcrush braucht TPDF-Dither"
    levels = np.unique(crushed)
    assert levels.size > 1, "Dithering wirkungslos — nacktes astype-Verdacht"


# ── ONNX-Export + Paritäts-Gate (§III.9 (copilot-instructions.md)) ──────────


def test_export_parity_gate(tmp_path):
    tc = _load_script("export_cantus_onnx")
    out = tmp_path / "cantus_tiny.onnx"
    rc = tc.export_model("tiny", Path("/nonexistent/ckpt.pt"), out, allow_random_init=True, seed=7)
    assert rc == 0, "Paritäts-Gate verfehlt oder Export fehlgeschlagen (fail-closed)"
    assert out.is_file()
    report = out.with_suffix(".parity.json")
    assert report.is_file()
    import json

    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["passed"] is True
    assert all(f["rel_error"] <= 1e-3 for f in data["feeds"])


# ── Plugin: Guards + CPU-Ende-zu-Ende auf Sample-Vocal ───────────────────────


def test_plugin_dsp_fallback_without_weights():
    from plugins.cantus_plugin import CantusPlugin

    model_path: Path = Path("/nonexistent/cantus_dit.onnx")
    plugin = CantusPlugin(model_path=model_path)
    vocal = _sample_vocal(12288)
    result = plugin.enhance(vocal, 48000, "mp3_low")
    assert result.model_used == "dsp_fallback", "§V6 (VERBOTEN.md): Ersatzpfad ohne Gewichte erwartet"
    assert result.applied is True
    assert result.audio.shape == vocal.shape
    assert bool(np.isfinite(result.audio).all())


def test_plugin_material_gate():
    from plugins.cantus_plugin import CantusPlugin

    plugin = CantusPlugin(model_path=Path("/nonexistent/cantus_dit.onnx"))
    vocal = _sample_vocal(4096)
    result = plugin.enhance(vocal, 48000, "clean_studio", restorability_score=90.0)
    assert result.model_used == "none" and result.applied is False
    assert np.array_equal(result.audio, vocal), "Ungateiertes Material muss unverändert bleiben"


def test_plugin_restores_vocal_stem_without_changing_instrumental_mix(monkeypatch: pytest.MonkeyPatch):
    """Ein vorhandener Stem steuert Cantus; nur dessen Delta darf den Mix ändern."""
    from plugins.cantus_plugin import CantusPlugin

    plugin = CantusPlugin(model_path=Path("/nonexistent/cantus_dit.onnx"))
    plugin._model_loaded = True
    plugin._fallback_active = False
    monkeypatch.setattr(plugin, "_restore_single", lambda stem: stem * 0.5)
    monkeypatch.setattr(plugin, "_spectral_novelty", lambda _before, _after, _sr: 0.0)
    monkeypatch.setattr(plugin, "_singmos_scores", lambda _before, _after: (None, None))

    vocal = _sample_vocal(8192)
    vocal = vocal - np.mean(vocal, dtype=np.float32)
    instrumental = 0.1 * _sample_vocal(8192, seed=9)
    mix = vocal + instrumental
    result = plugin.enhance(mix, 48000, "mp3_low", vocal_stem=vocal)

    assert result.metadata is not None and result.metadata["input_source"] == "vocal_stem"
    assert np.allclose(result.audio, instrumental + 0.5 * vocal, atol=1e-5)


def test_plugin_cpu_end_to_end_sample_vocal(tmp_path):
    """Ende-zu-Ende CPU-Inferenz: Export → ORT → Guards → Layout-Roundtrip."""
    tc = _load_script("export_cantus_onnx")
    from plugins.cantus_plugin import CantusPlugin

    onnx_path = tmp_path / "cantus_tiny.onnx"
    assert tc.export_model("tiny", Path("/nonexistent/ckpt.pt"), onnx_path, allow_random_init=True, seed=7) == 0

    plugin = CantusPlugin(model_path=onnx_path)
    assert plugin._model_loaded, "Tiny-ONNX musste laden"
    vocal = _sample_vocal(24576)

    r_a = plugin.enhance(vocal, 48000, "mp3_low")
    r_b = plugin.enhance(vocal, 48000, "mp3_low")
    assert r_a.audio.shape == vocal.shape
    assert bool(np.isfinite(r_a.audio).all())
    assert np.array_equal(r_a.audio, r_b.audio), "Inferenz nicht deterministisch (§G5 (GEBOTE.md))"
    assert r_a.metadata is not None, "Enhance-Ergebnis muss Metadaten liefern"
    assert r_a.metadata["use_cond"] in (0.0, 1.0)

    stereo_n2 = np.stack([vocal, vocal * 0.8], axis=-1)
    out_n2 = plugin.enhance(stereo_n2, 48000, "streaming")
    out_cn = plugin.enhance(stereo_n2.T, 48000, "streaming")
    assert out_n2.audio.shape == stereo_n2.shape, "Layout (N, 2) nicht wiederhergestellt"
    assert out_cn.audio.shape == (2, vocal.size), "Layout (2, N) nicht wiederhergestellt"
    assert bool(np.isfinite(out_n2.audio).all()) and bool(np.isfinite(out_cn.audio).all())

    # 44.1 kHz → Resample-Pfad
    r_44 = plugin.enhance(vocal[:16384], 44100, "mp3_low")
    assert r_44.audio.ndim == 1 and r_44.audio.size > 0
    assert bool(np.isfinite(r_44.audio).all())


def test_plugin_retries_ort_inference_on_cpu_after_provider_failure(monkeypatch, caplog, tmp_path):
    """Ein ROCm-Laufzeitfehler muss protokolliert und auf CPU wiederholt werden (§V6 (copilot-instructions.md))."""
    import logging

    import plugins.cantus_plugin as cantus_module

    plugin = cantus_module.CantusPlugin(model_path=tmp_path / "missing.onnx")
    plugin._model_loaded = True
    plugin._fallback_active = False

    class GpuSession:
        def run(self, _outputs, _feeds):
            raise RuntimeError("HIPBLAS_STATUS_ALLOC_FAILED")

    plugin._ort_session = GpuSession()
    plugin.__dict__["_extract_conditions"] = lambda _mono: {
        "mert": np.zeros((1, 1024), dtype=np.float32),
        "pitch": np.zeros((1, 2), dtype=np.float32),
        "harm": np.zeros((768,), dtype=np.float32),
        "use_cond": np.asarray(0.0, dtype=np.float32),
    }

    class CpuSession:
        def run(self, _outputs, feeds):
            return [np.zeros_like(feeds["x"])]

    created_providers = []

    def create_cpu_session(_path, *, providers):
        created_providers.append(providers)
        return CpuSession()

    monkeypatch.setattr(cantus_module.ort, "InferenceSession", create_cpu_session)
    mono = _sample_vocal(4096)
    with caplog.at_level(logging.WARNING):
        restored = plugin._restore_single(mono)

    assert created_providers == [["CPUExecutionProvider"]]
    assert np.array_equal(restored, mono)
    assert "§V6 (VERBOTEN.md)" in caplog.text
    assert "HIPBLAS_STATUS_ALLOC_FAILED" in caplog.text


def test_plugin_prefers_torch_rocm_primary_path(monkeypatch, tmp_path):
    """Der paritätsverifizierte Torch-ROCm-Kern ist vor ONNX-CPU primär (§III.9)."""
    from plugins.cantus_plugin import CantusPlugin

    plugin = CantusPlugin(model_path=tmp_path / "missing.onnx")
    plugin._model_loaded = True
    plugin._fallback_active = False
    plugin._torch_model = object()
    plugin._inference_backend = "torch_rocm"
    plugin.__dict__["_extract_conditions"] = lambda _mono: {
        "mert": np.zeros((1, 1024), dtype=np.float32),
        "pitch": np.zeros((1, 2), dtype=np.float32),
        "harm": np.zeros((768,), dtype=np.float32),
        "use_cond": np.asarray(0.0, dtype=np.float32),
    }
    monkeypatch.setattr(plugin, "_run_torch_rocm", lambda feeds: np.zeros_like(feeds["x"]))
    monkeypatch.setattr(plugin, "_run_ort_with_cpu_fallback", lambda _feeds: pytest.fail("ONNX darf nicht laufen"))

    mono = _sample_vocal(4096)
    restored = plugin._restore_single(mono)

    assert np.array_equal(restored, mono)
    assert plugin._inference_backend == "torch_rocm"


def test_plugin_falls_back_from_torch_rocm_to_onnx_cpu(monkeypatch, caplog, tmp_path):
    """Ein GPU-Laufzeitfehler wechselt mit §V6-Warnung auf ONNX-CPU, nie still auf DSP."""
    import logging

    from plugins.cantus_plugin import CantusPlugin

    plugin = CantusPlugin(model_path=tmp_path / "missing.onnx")
    plugin._model_loaded = True
    plugin._fallback_active = False
    plugin._torch_model = object()
    plugin.__dict__["_extract_conditions"] = lambda _mono: {
        "mert": np.zeros((1, 1024), dtype=np.float32),
        "pitch": np.zeros((1, 2), dtype=np.float32),
        "harm": np.zeros((768,), dtype=np.float32),
        "use_cond": np.asarray(0.0, dtype=np.float32),
    }
    monkeypatch.setattr(plugin, "_run_torch_rocm", lambda _feeds: (_ for _ in ()).throw(RuntimeError("ROCm alloc")))
    monkeypatch.setattr(plugin, "_run_ort_with_cpu_fallback", lambda feeds: [np.zeros_like(feeds["x"])])

    mono = _sample_vocal(4096)
    with caplog.at_level(logging.WARNING):
        restored = plugin._restore_single(mono)

    assert np.array_equal(restored, mono)
    assert plugin._torch_model is None
    assert plugin._inference_backend == "onnx_cpu"
    assert "Cantus-Torch-ROCm-Inferenz fehlgeschlagen" in caplog.text
    assert "§V6 (copilot-instructions.md)" in caplog.text


def test_plugin_uses_dsp_fallback_when_cpu_retry_also_fails(monkeypatch, caplog, tmp_path):
    """Auch ein CPU-ORT-Fehler darf nie einen unbehandelten ML-Abbruch auslösen."""
    import logging

    import plugins.cantus_plugin as cantus_module

    plugin = cantus_module.CantusPlugin(model_path=tmp_path / "missing.onnx")
    plugin._model_loaded = True
    plugin._fallback_active = False

    class BrokenSession:
        def run(self, _outputs, _feeds):
            raise RuntimeError("ORT-Ausführung nicht verfügbar")

    plugin._ort_session = BrokenSession()
    plugin.__dict__["_extract_conditions"] = lambda _mono: {
        "mert": np.zeros((1, 1024), dtype=np.float32),
        "pitch": np.zeros((1, 2), dtype=np.float32),
        "harm": np.zeros((768,), dtype=np.float32),
        "use_cond": np.asarray(0.0, dtype=np.float32),
    }
    monkeypatch.setattr(cantus_module.ort, "InferenceSession", lambda _path, *, providers: BrokenSession())

    with caplog.at_level(logging.WARNING):
        result = plugin.enhance(_sample_vocal(4096), 48000, "mp3_low")

    assert result.model_used == "dsp_fallback"
    assert bool(np.isfinite(result.audio).all())
    assert "ORT-Inferenz auch auf CPU fehlgeschlagen" in caplog.text


def test_smoke_dataset_deterministic():
    tc = _load_script("train_cantus")
    ds = tc.SmokeDataset(4, chunk_samples=8192, seed=20261004)
    ds.set_epoch(2)
    a = ds[1]
    b = ds[1]
    assert torch.equal(a["clean"], b["clean"]), "SmokeDataset nicht deterministisch"
    assert torch.equal(a["degraded"], b["degraded"])
    ds.set_epoch(3)
    c = ds[1]
    assert torch.equal(a["clean"], c["clean"]), "clean muss epoch-stabil sein (Ground Truth)"
    assert not torch.equal(a["degraded"], c["degraded"]), "Epoch-Seed der Degradation wirkt nicht"
