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


def test_strong_clipping_uses_sparse_path(phase):
    """Evidenz: starkes 6× — Sparse übernimmt (Harmonik-Proxy sinkt)."""
    x = _clipped_bursts(6.0)
    result = phase.process(x, sample_rate=48000)
    assert result.metrics.get("declip_applied") is True
    assert result.metrics.get("sparse_used") is True


def test_sparse_path_deterministic(phase):
    x = _clipped_bursts(6.0)
    r1 = phase.process(x, sample_rate=48000)
    r2 = phase.process(x, sample_rate=48000)
    np.testing.assert_array_equal(r1.audio, r2.audio)  # §G5 (copilot-instructions.md)


def test_sparse_path_reduces_thd(phase):
    """THD-Verhältnis (Harmonische 2–10 vs. Grundton) sinkt durch den Sparse-Zweig."""
    x = _clipped_bursts(6.0)
    result = phase.process(x, sample_rate=48000)
    assert result.metrics.get("sparse_used") is True

    def _thd(y: np.ndarray) -> float:
        spec = np.abs(np.fft.rfft(y.astype(np.float64)))
        freqs = np.fft.rfftfreq(len(y), 1.0 / 48000)
        fund = float(spec[np.argmin(np.abs(freqs - 440))]) + 1e-12
        harm = 0.0
        for h in range(2, 11):
            _i = int(np.argmin(np.abs(freqs - 440 * h)))
            harm += float(spec[_i]) ** 2
        return float(np.sqrt(harm) / fund)

    assert _thd(result.audio) < _thd(x)


def test_sparse_fallback_to_pchip(phase, monkeypatch):
    """Sparse-Modul nicht verfügbar → PCHIP bleibt, kein Crash (§V6 (copilot-instructions.md))."""
    x = _clipped_bursts(6.0)

    def _boom(_a, _sr, _t, **_kw):
        raise RuntimeError("sparse unavailable")

    monkeypatch.setattr("backend.core.dsp.sparse_declipper.sparse_declip", _boom)
    result = phase.process(x, sample_rate=48000)
    assert result.metrics.get("declip_applied") is True
    assert result.metrics.get("sparse_used") is False
    assert np.isfinite(result.audio).all()


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
    x = _saettigung_tanh8()
    result = phase.process(x, sample_rate=48000, material="cd_digital")
    assert result.metrics.get("declip_applied") is False
    assert result.metrics.get("reason") == "never_worsen_proxy"
    assert result.resolved_defects == {}
    assert np.array_equal(np.asarray(result.audio), x)
