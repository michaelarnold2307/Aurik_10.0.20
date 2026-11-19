"""tests/unit/test_phase_09_crackle_removal.py — §v10.700 I1."""

import numpy as np
import pytest

from backend.core.phases.phase_09_crackle_removal import CrackleRemovalPhase


@pytest.fixture
def phase():
    return CrackleRemovalPhase()


@pytest.fixture
def audio():
    rng = np.random.RandomState(42)
    t = np.linspace(0, 1, 48000, endpoint=False, dtype=np.float32)
    return (np.sin(2 * np.pi * 440 * t) * 0.5 + rng.randn(48000) * 0.01).astype(np.float32)


def test_returns_ndarray(phase, audio):
    result = phase.process(audio, sample_rate=48000, material_type="vinyl")
    assert isinstance(result.audio, np.ndarray)


def test_no_nan_inf(phase, audio):
    result = phase.process(audio, sample_rate=48000, material_type="vinyl")
    assert np.isfinite(result.audio).all()


def test_not_silent(phase, audio):
    result = phase.process(audio, sample_rate=48000, material_type="vinyl")
    assert float(np.sqrt(np.mean(result.audio**2))) > 1e-10


def test_length_preserved(phase, audio):
    result = phase.process(audio, sample_rate=48000, material_type="vinyl")
    assert len(result.audio) == len(audio)


def test_stereo_layout_is_restored_and_crackle_score_uses_sample_count(phase, monkeypatch, audio):
    stereo = np.stack([audio, audio * 0.8], axis=0).astype(np.float32)
    monkeypatch.setattr(phase, "_compute_crackle_regions_with_protection", lambda *_args: ([], [], [], [(10, 20)]))
    monkeypatch.setattr(phase, "_union_scanner_crackle_regions", lambda _audio, regions, _kwargs: regions)
    monkeypatch.setattr(phase, "_model_background_texture", lambda *_args: None)
    monkeypatch.setattr(phase, "_remove_crackle_spectral", lambda mono, *_args: mono.copy())

    result_cf = phase.process(stereo, sample_rate=48000, material_type="unknown")
    result_sf = phase.process(stereo.T.copy(), sample_rate=48000, material_type="unknown")

    assert result_cf.audio.shape == stereo.shape
    assert result_sf.audio.shape == (stereo.shape[1], 2)
    assert result_cf.resolved_defects["CRACKLE"] == pytest.approx(0.3)
    assert result_sf.resolved_defects["CRACKLE"] == pytest.approx(0.3)
    assert np.isfinite(result_cf.audio).all()
    assert np.isfinite(result_sf.audio).all()


@pytest.mark.parametrize("shape", [(1, 48000), (48000, 1)])
def test_two_dimensional_mono_uses_mono_processing_and_preserves_layout(phase, monkeypatch, shape):
    mono = np.zeros(shape, dtype=np.float32)
    observed_shapes = []

    def fake_regions(audio, _params):
        observed_shapes.append(audio.shape)
        return [], [], [], []

    monkeypatch.setattr(phase, "_compute_crackle_regions_with_protection", fake_regions)
    monkeypatch.setattr(phase, "_model_background_texture", lambda *_args: None)
    monkeypatch.setattr(phase, "_remove_crackle_spectral", lambda audio, *_args: audio.copy())

    result = phase.process(mono, sample_rate=48000, material_type="unknown")

    assert observed_shapes == [(48000,)]
    assert result.audio.shape == mono.shape


@pytest.mark.parametrize("shape", [(3, 48000), (48000, 3), (2, 48000, 1)])
def test_unsupported_channel_layout_raises(phase, shape):
    with pytest.raises(ValueError, match="erwartet Mono oder Stereo"):
        phase.process(np.zeros(shape, dtype=np.float32), sample_rate=48000, material_type="unknown")


def test_banquet_fallback_receives_channels_first_and_restores_layout(phase, monkeypatch, audio):
    import backend.core.phases.phase_09_crackle_removal as phase09

    stereo = np.stack([audio, audio * 0.8], axis=0).astype(np.float32)
    received_shapes = []

    monkeypatch.setattr(phase09, "QUALITY_MODE_AVAILABLE", True)
    monkeypatch.setattr(phase09, "is_phase_ml_enabled", lambda _phase: True, raising=False)

    def fail_direct(*_args):
        raise RuntimeError("force the documented Docker fallback")

    monkeypatch.setattr(phase, "_remove_crackle_onnx_direct", fail_direct)
    monkeypatch.setattr(phase, "_get_banquet_plugin", lambda: object())

    def fake_ml(audio_cf, *_args):
        received_shapes.append(audio_cf.shape)
        return audio_cf * 0.9

    monkeypatch.setattr(phase, "_remove_crackle_ml", fake_ml)
    result = phase.process(stereo, sample_rate=48000, material_type="vinyl", strength=1.0)

    assert received_shapes == [stereo.shape]
    assert result.audio.shape == stereo.shape


def test_banquet_stereo_per_channel_layout(phase, monkeypatch, audio):
    """BANQUET-Direktpfad delegiert an die kanonische Plugin-Pipeline
    (§V7 (copilot-instructions.md)): Layout identisch zum Input, kanalweise
    verarbeitet (Fake-Plugin)."""
    import plugins.banquet_vinyl_plugin as _bp

    class _FakePlugin:
        _model_ok = True

        def process(self, x, sr):
            # Plugin-Kontrakt: ndim-erhaltend, leichte deterministische Dämpfung.
            return np.asarray(x, dtype=np.float32) * 0.9

    monkeypatch.setattr(_bp, "get_banquet_plugin", lambda: _FakePlugin())

    stereo_cf = np.stack([audio, audio], axis=0).astype(np.float32)
    out_cf = phase._remove_crackle_onnx_direct(stereo_cf, 48000, {})
    assert out_cf.shape == (2, len(audio))
    np.testing.assert_allclose(out_cf[0], out_cf[1], atol=1e-6)

    stereo_cl = stereo_cf.T.copy()
    out_cl = phase._remove_crackle_onnx_direct(stereo_cl, 48000, {})
    assert out_cl.shape == (len(audio), 2)
    np.testing.assert_allclose(out_cl[:, 0], out_cl[:, 1], atol=1e-6)


def test_banquet_direct_raises_without_model(phase, monkeypatch, audio):
    """Ohne Modell: RuntimeError → Aufrufer fällt auf den Plugin-/DSP-Pfad zurück."""
    import plugins.banquet_vinyl_plugin as _bp

    class _NoModelPlugin:
        _model_ok = False

        def process(self, x, sr):
            raise AssertionError("darf ohne Modell nicht aufgerufen werden")

    monkeypatch.setattr(_bp, "get_banquet_plugin", lambda: _NoModelPlugin())
    with pytest.raises(RuntimeError):
        phase._remove_crackle_onnx_direct(audio, 48000, {})


def test_banquet_file_fallback_uses_process_files(phase, audio):
    """Fallback-Pfad nutzt die Datei-API (process_files) — der frühere Aufruf
    process(tmp_in, tmp_out) mit Pfaden ergab 'str' object has no attribute
    'astype' (Produktionsfehler 2026-09-13)."""
    import soundfile as sf

    class _FakePlugin:
        def __init__(self):
            self.calls = []

        def process_files(self, input_wav, output_wav, strength=1.0):
            self.calls.append((input_wav, output_wav, strength))
            x, sr = sf.read(input_wav, always_2d=False, dtype="float32")
            sf.write(output_wav, x * 0.9, sr, subtype="FLOAT")

    plugin = _FakePlugin()
    out = phase._remove_crackle_ml(audio, plugin, {})
    assert len(plugin.calls) == 1
    assert out.shape == audio.shape
    # Blend: 0,85 Original + 0,15 * (0,9 × Original) = 0,985 × Original
    # (load_audio_file reskaliert minimal — 1e-4-Toleranz deckt den Loader ab)
    np.testing.assert_allclose(out, audio * 0.985, atol=1e-4)


def test_banquet_file_fallback_stereo_layout(phase, audio):
    """Stereo (C, N): Schreiben/Lesen erhält das Kanal-Layout (kein Vertauschen)."""
    import soundfile as sf

    class _FakePlugin:
        def process_files(self, input_wav, output_wav, strength=1.0):
            x, sr = sf.read(input_wav, always_2d=False, dtype="float32")
            sf.write(output_wav, x * 0.9, sr, subtype="FLOAT")

    stereo_cf = np.stack([audio, audio * 0.5], axis=0).astype(np.float32)
    out = phase._remove_crackle_ml(stereo_cf, _FakePlugin(), {})
    assert out.shape == (2, len(audio))
    np.testing.assert_allclose(out[0], audio * 0.985, atol=1e-4)
    np.testing.assert_allclose(out[1], audio * 0.5 * 0.985, atol=1e-4)
