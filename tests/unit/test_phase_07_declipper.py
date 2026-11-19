"""tests/unit/test_phase_07_declipper.py — §v10.700 I1."""

import numpy as np
import pytest

from backend.core.phases.phase_07_declipper import DeclipperPhase


@pytest.fixture
def phase():
    return DeclipperPhase()


@pytest.fixture
def audio():
    rng = np.random.RandomState(42)
    t = np.linspace(0, 1, 48000, endpoint=False, dtype=np.float32)
    clipped = np.clip(np.sin(2 * np.pi * 440 * t) * 1.5, -1.0, 1.0)
    return clipped.astype(np.float32)


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


def _clipped_bursts(amp: float) -> np.ndarray:
    """0.8-Sinus mit drei 25-ms-Bursts ×amp, hart bei ±1.0 geclippt."""
    sr = 48000
    t = np.linspace(0, 1, sr, endpoint=False)
    x = np.sin(2 * np.pi * 440 * t) * 0.8
    burst = int(sr * 0.025)
    for onset_s in (0.25, 0.5, 0.75):
        s0 = int(onset_s * sr)
        x[s0 : s0 + burst] *= amp
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def test_mild_clipping_sparse_rejected_pchip_stays(phase):
    """Evidenz (DECLIPPER_SOTA_PLAN.md Slice A): mildes 2.5× — Sparse verschlechtert, PCHIP bleibt."""
    x = _clipped_bursts(2.5)
    result = phase.process(x, sample_rate=48000)
    assert result.metrics.get("declip_applied") is True
    assert result.metrics.get("sparse_used") is False


def test_unvalidated_sparse_path_is_not_applied(phase, monkeypatch):
    """Ein Proxy-Gewinn reicht ohne Ground-Truth-/Hörabnahme nicht zur Übernahme."""
    import backend.core.dsp.sparse_declipper as sparse_mod

    def _unexpected_call(*_args, **_kwargs):
        pytest.fail("Der nicht validierte Sparse-Declipper darf nicht ausgeführt werden.")

    monkeypatch.setattr(sparse_mod, "sparse_declip", _unexpected_call)
    x = _clipped_bursts(6.0)
    result = phase.process(x, sample_rate=48000)
    assert result.metrics.get("declip_applied") is True
    assert result.metrics.get("sparse_used") is False


def test_phase_output_deterministic(phase):
    x = _clipped_bursts(6.0)
    r1 = phase.process(x, sample_rate=48000)
    r2 = phase.process(x, sample_rate=48000)
    np.testing.assert_array_equal(r1.audio, r2.audio)  # §G5 (copilot-instructions.md)


def test_sparse_proxy_gain_cannot_cause_large_sdr_regression(phase, monkeypatch):
    """Ground-truth regression guard: reduced harmonic proxy had hidden a 24.63 dB loss."""
    import backend.core.phases.phase_07_declipper as mod

    sr = 48000
    t = np.linspace(0, 1, sr, endpoint=False)
    clean = 0.72 * np.sin(2 * np.pi * 440 * t) + 0.18 * np.sin(2 * np.pi * 880 * t)
    clean += 0.08 * np.sin(2 * np.pi * 1320 * t)
    for onset in (0.25, 0.5, 0.75):
        start = int(onset * sr)
        clean[start : start + int(0.025 * sr)] *= 1.8
    clean = (0.95 * clean / np.max(np.abs(clean))).astype(np.float32)
    clipped = np.clip(clean, -0.85, 0.85)
    left = (0.6 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    monkeypatch.setattr(mod, "_histogram_clip_threshold", lambda *_args, **_kwargs: 0.85)

    result = phase.process(np.stack((left, clipped)), sample_rate=sr, material="cd_digital")

    def _sdr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
        ref = np.asarray(reference, dtype=np.float64)
        err = np.mean((np.asarray(estimate, dtype=np.float64) - ref) ** 2)
        return float(10.0 * np.log10(np.mean(ref**2) / max(float(err), 1e-12)))

    input_sdr = _sdr_db(clean, clipped)
    output_sdr = _sdr_db(clean, np.asarray(result.audio[1]))
    assert result.metrics.get("n_clips", 0) > 0
    assert result.metrics.get("sparse_used") is False
    assert output_sdr > 30.0, f"Sparse candidate caused severe SDR regression: {input_sdr:.2f} → {output_sdr:.2f} dB"
    np.testing.assert_array_equal(result.audio[0], left)


def _saettigung_tanh8() -> np.ndarray:
    """tanh(8x)-Mastering-Sättigung (Schadens-Rezept 'clipping'): flaches
    Plateau bei ±0.9 — KEIN Hard-Clipping, aber der Histogramm-Detektor
    markiert die Decke (§D-K3-97)."""
    sr = 48000
    t = np.linspace(0, 1, sr, endpoint=False)
    x = np.sin(2 * np.pi * 440 * t) * 1.5
    y = np.tanh(8.0 * x) / np.tanh(8.0)
    return (y / (float(np.abs(y).max()) + 1e-9) * 0.9).astype(np.float32)


def test_saettigung_plateau_wird_nicht_verschlechtert(phase):
    """§D-K3-97: tanh(8x)-Sättigung darf Phase 07 NICHT verschlechtern —
    defektlast steigt höchstens um JND 0,05."""
    from backend.core.dsp.wohlklang_zeuge import defektlast

    x = _saettigung_tanh8()
    result = phase.process(x, sample_rate=48000, material="cd_digital")
    d_in = float(defektlast(x, 48000))
    d_out = float(defektlast(np.asarray(result.audio, dtype=np.float32), 48000))
    assert d_out <= d_in + 0.05, f"defektlast verschlechtert: {d_in:.4f} → {d_out:.4f}"


def test_pchip_ist_formtreu_unter_der_eingangs_decke():
    """§D-K3-97: PCHIP ist formtreu — es gibt KEINE Überschwinger über das
    Eingangs-Maximum. Damit ist die frühere Hypothese „das Clipping auf ±1.0
    erzeugt neue Flat-Tops“ widerlegt (gemessen: der Clamp griff nie), und der
    Test hält die Eigenschaft fest, auf die sich der Ersatzpfad verlässt."""
    from backend.core.phases.phase_07_declipper import _declip_pchip, _histogram_clip_threshold

    x = _saettigung_tanh8()
    th = _histogram_clip_threshold(x, material="cd_digital")
    y = _declip_pchip(x, th)
    assert float(np.max(np.abs(y))) <= float(np.max(np.abs(x))) + 1e-6


def test_phase07_senkt_den_harmonik_proxy_nie(phase):
    """§D-K3-97: Never-worsen-Kontrakt der ganzen Phase — die Ausgabe darf den
    Harmonik-Proxy gegen den Eingang nie erhöhen. Motiv ist gemessen: PCHIP auf
    SPARSE Hard-Clips kostet 6,84 dB SDR (Decke 0,90) bis 115,97 dB (0,99)."""
    from backend.core.phases.phase_07_declipper import _harmonic_distortion_proxy

    rng = np.random.default_rng(42)
    x = np.clip(rng.normal(0.0, 0.4, 48000), -0.9, 0.9).astype(np.float32)
    result = phase.process(x, sample_rate=48000, material="cd_digital")
    p_in = _harmonic_distortion_proxy(x.astype(np.float64), 48000)
    p_out = _harmonic_distortion_proxy(np.asarray(result.audio, dtype=np.float64), 48000)
    assert p_out <= p_in * 1.001, f"Harmonik-Proxy stieg: {p_in:.4f} → {p_out:.4f}"


def test_passthrough_meldet_keine_reparatur(phase, monkeypatch):
    """§D-K3-97: Wird kein Kandidat übernommen, ist das Signal unverändert —
    dann MUSS die Phase `declip_applied: false` melden und KEINEN Defekt als
    behoben verbuchen (sonst zählt CLIPPING downstream als repariert)."""
    import backend.core.dsp.sparse_declipper as sparse_mod
    import backend.core.phases.phase_07_declipper as mod

    def _kein_gewinn(audio, _threshold):
        return np.array(audio, copy=True)

    def _kein_gewinn_sparse(audio, _sample_rate, _threshold):
        return np.array(audio, copy=True)

    monkeypatch.setattr(mod, "_declip_pchip", _kein_gewinn)
    monkeypatch.setattr(sparse_mod, "sparse_declip", _kein_gewinn_sparse)
    x = _clipped_bursts(6.0)
    result = phase.process(x, sample_rate=48000, material="cd_digital")
    assert result.metrics.get("declip_applied") is False
    assert result.metrics.get("reason") == "never_worsen_proxy"
    assert result.resolved_defects == {}
    assert np.array_equal(np.asarray(result.audio), x)


def test_stereo_clipping_in_second_channel_is_detected(phase, monkeypatch):
    """Ein unbeschädigter erster Kanal darf Clipping im zweiten nicht maskieren."""
    sr = 48000
    t = np.linspace(0, 1, sr, endpoint=False, dtype=np.float32)
    left = 0.6 * np.sin(2 * np.pi * 440 * t)
    right = _clipped_bursts(6.0)
    monkeypatch.setattr(
        "backend.core.phases.phase_07_declipper._histogram_clip_threshold",
        lambda *_args, **_kwargs: 0.88,
    )
    stereo = np.stack((left, right))

    result = phase.process(stereo, sample_rate=sr, material="cd_digital")

    assert result.metrics.get("n_clips", 0) > 0
    np.testing.assert_array_equal(result.audio[0], left)
    assert result.metrics.get("sparse_used") is False
    assert np.isfinite(result.audio).all()


def test_no_clip_passthrough_is_bit_identical(phase, monkeypatch):
    """Ein echter Passthrough darf nicht durch PhaseResult-Softclip verändert werden."""
    import backend.core.phases.phase_07_declipper as mod

    monkeypatch.setattr(mod, "_histogram_clip_threshold", lambda *_args, **_kwargs: 0.999)
    t = np.linspace(0, 1, 48000, endpoint=False, dtype=np.float32)
    audio = np.stack((0.98 * np.sin(2 * np.pi * 440 * t), 0.98 * np.sin(2 * np.pi * 660 * t)))

    result = phase.process(audio, sample_rate=48000, material="cd_digital")

    assert result.metrics.get("reason") == "no_clipping_detected"
    np.testing.assert_array_equal(result.audio, audio)


def test_strength_bypass_is_bit_identical(phase):
    t = np.linspace(0, 1, 48000, endpoint=False, dtype=np.float32)
    audio = np.sin(2 * np.pi * 440 * t)

    result = phase.process(audio, sample_rate=48000, strength=0.0)

    assert result.metrics.get("reason") == "strength_bypass"
    np.testing.assert_array_equal(result.audio, audio)
