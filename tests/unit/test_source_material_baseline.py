"""
Tests für §2.50 Material-Adaptive Gate Baseline —
ArtifactFreedomGate.measure_source_baseline / SourceMaterialBaseline.

Normative Anforderungen (§2.50):
- measure_source_baseline() misst Quellmaterial-Artefakte VOR der Pipeline.
- Gibt SourceMaterialBaseline mit 5 Feldern zurück.
- has_critical_stereo_issue = True wenn > 20 % Frames mono-inkompatibel.
- has_anti_phase_region = True wenn ein Frame lr_corr < 0.
- hf_loss_db: geschätzter HF-Verlust (0 = kein Verlust).
- Gate darf Eigenschaften des Quellmaterials nicht bestrafen (§0 Primum non nocere).
"""

from typing import cast

import numpy as np
import pytest


@pytest.fixture
def gate():
    from backend.core.artifact_freedom_gate import ArtifactFreedomGate

    return ArtifactFreedomGate()


@pytest.fixture
def sr():
    return 48000


def _make_stereo(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Erstellt 2xN Stereo-Array."""
    n = min(len(left), len(right))
    stereo = np.asarray(np.stack([left[:n], right[:n]], axis=0), dtype=np.float32)
    return cast(np.ndarray, stereo)


# ── Grundlegende Rückgabe-Invarianten ─────────────────────────────────────


@pytest.mark.unit
def test_returns_source_material_baseline(gate, sr):
    """measure_source_baseline gibt immer ein SourceMaterialBaseline zurück."""
    from backend.core.artifact_freedom_gate import SourceMaterialBaseline

    mono = np.random.randn(sr).astype(np.float32) * 0.1
    result = gate.measure_source_baseline(mono, sr, "vinyl")
    assert isinstance(result, SourceMaterialBaseline)


def test_all_five_fields_present(gate, sr):
    """Alle 5 normativen Felder (§2.50) sind in der Baseline."""
    audio = np.random.randn(2, sr * 3).astype(np.float32) * 0.1
    baseline = gate.measure_source_baseline(audio, sr, "vinyl")
    assert hasattr(baseline, "phase_cancellation_ratio")
    assert hasattr(baseline, "stereo_mono_compat_mean")
    assert hasattr(baseline, "has_critical_stereo_issue")
    assert hasattr(baseline, "has_anti_phase_region")
    assert hasattr(baseline, "hf_loss_db")


def test_material_type_stored(gate, sr):
    """material_type wird korrekt normiert und gespeichert."""
    mono = np.random.randn(sr).astype(np.float32) * 0.1
    baseline = gate.measure_source_baseline(mono, sr, "shellac")
    assert baseline.material_type == "shellac"


# ── Stereo-Feldanalyse ─────────────────────────────────────────────────────


def test_good_stereo_no_critical_issue(gate, sr):
    """Normales Stereo → has_critical_stereo_issue = False."""
    t = np.arange(sr * 3) / sr
    left = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    right = (np.sin(2 * np.pi * 440 * t + 0.1) * 0.5).astype(np.float32)
    audio = _make_stereo(left, right)
    baseline = gate.measure_source_baseline(audio, sr, "vinyl")
    assert not baseline.has_critical_stereo_issue
    assert baseline.phase_cancellation_ratio < 0.20


def test_anti_phase_stereo_detected(gate, sr):
    """Vollständig anti-phasiges Stereo → has_anti_phase_region = True."""
    t = np.arange(sr * 3) / sr
    sig = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    # L und R exakt gespiegelt = anti-phase
    audio = _make_stereo(sig, -sig)
    baseline = gate.measure_source_baseline(audio, sr, "vinyl")
    assert baseline.has_anti_phase_region


def test_critical_stereo_issue_above_20_percent(gate, sr):
    """Mehr als 20 % mono-inkompatible Frames → has_critical_stereo_issue = True."""
    # 80 % der Zeit anti-phasig: viele Frames mit lr_corr < 0
    n = sr * 5
    t = np.linspace(0, 5, n)
    sig = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5
    # Anti-Phase Signal
    anti = -sig.copy()
    # 80 % anti-phase + 20 % normal
    left = np.concatenate([sig[: n // 5], anti[n // 5 :]])
    right = np.concatenate([sig[: n // 5], sig[n // 5 :]])
    audio = _make_stereo(left[:n], right[:n])
    baseline = gate.measure_source_baseline(audio, sr, "vinyl")
    assert baseline.has_critical_stereo_issue


def test_critical_stereo_issue_when_interchannel_lag_exceeds_1ms(gate, sr):
    """§2.51a: Bereits > 1 ms Quell-L/R-Lag muss den Stereo-Hard-Fail triggern."""
    n = sr * 3
    lag_samples = int(sr * 0.079)
    rng = np.random.default_rng(42)
    # Breitbandiges Rauschen statt Sinus: GCC-PHAT ist für Einzeltöne ungeeignet
    # (periodische Kreuzkorrelation → kein eindeutiger Peak).
    sig = rng.standard_normal(n).astype(np.float32) * 0.5
    delayed = np.concatenate([np.zeros(lag_samples, dtype=np.float32), sig[:-lag_samples]])
    audio = _make_stereo(sig, delayed)

    baseline = gate.measure_source_baseline(audio, sr, "vinyl")

    assert abs(baseline.interchannel_lag_samples) > int(sr * 0.001)
    assert baseline.has_critical_stereo_issue


def test_mono_audio_no_crash(gate, sr):
    """Mono-Audio → keine Exception, Stereo-Felder bleiben auf Default."""
    mono = np.random.randn(sr * 3).astype(np.float32) * 0.1
    baseline = gate.measure_source_baseline(mono, sr, "digital")
    # Mono: keine Stereo-Analyse. Defaults erwartet.
    assert baseline.phase_cancellation_ratio == 0.0
    assert not baseline.has_critical_stereo_issue
    assert not baseline.has_anti_phase_region


def test_silence_frames_ignored(gate, sr):
    """Stille-Frames (rms < 1e-6) werden ignoriert, kein Division-by-Zero."""
    silence = np.zeros((2, sr * 3), dtype=np.float32)
    baseline = gate.measure_source_baseline(silence, sr, "vinyl")
    assert baseline.phase_cancellation_ratio == 0.0
    assert not baseline.has_critical_stereo_issue


# ── HF-Loss-Analyse ────────────────────────────────────────────────────────


def test_hf_loss_low_for_full_bandwidth_signal(gate, sr):
    """Breitband-Signal ~ flaches Spektrum → hf_loss_db nahe 0."""
    white = (np.random.randn(sr * 3) * 0.1).astype(np.float32)
    baseline = gate.measure_source_baseline(white, sr, "digital")
    # Weißes Rauschen hat gleichmäßiges Spektrum → kein HF-Verlust
    assert baseline.hf_loss_db < 10.0, f"hf_loss_db={baseline.hf_loss_db} unerwartet hoch für White Noise"


def test_hf_loss_high_for_lowpassed_signal(gate, sr):
    """Tiefpass-gefiltertes Signal → hf_loss_db deutlich > 0."""
    from scipy.signal import butter, filtfilt

    white = (np.random.randn(sr * 4) * 0.1).astype(np.float32)
    b, a = butter(6, 3000 / (sr / 2), btype="low")
    lp = filtfilt(b, a, white).astype(np.float32)
    baseline = gate.measure_source_baseline(lp, sr, "cassette")
    assert baseline.hf_loss_db > 5.0, f"hf_loss_db={baseline.hf_loss_db} zu niedrig für ~3 kHz-Tiefpass-Signal"


# ── Robustheit ─────────────────────────────────────────────────────────────


def test_nan_input_safe(gate, sr):
    """NaN-Input → kein Absturz, saubere Baseline."""
    bad = np.full((2, sr * 2), np.nan, dtype=np.float32)
    baseline = gate.measure_source_baseline(bad, sr, "vinyl")
    assert isinstance(baseline.phase_cancellation_ratio, float)
    assert not np.isnan(baseline.hf_loss_db)


def test_very_short_audio_no_crash(gate, sr):
    """Sehr kurzes Audio (< 100 ms) → kein Absturz."""
    tiny = np.random.randn(int(sr * 0.05)).astype(np.float32) * 0.1
    baseline = gate.measure_source_baseline(tiny, sr, "vinyl")
    assert isinstance(baseline, object)  # kein Crash


def test_unknown_material_fallback(gate, sr):
    """Unbekannter Material-Typ → Fallback auf digital, kein Absturz."""
    mono = np.random.randn(sr * 2).astype(np.float32) * 0.1
    baseline = gate.measure_source_baseline(mono, sr, "zeppelin_acetate")
    assert baseline.hf_loss_db >= 0.0


# ── §2.50 Gate-Paradoxon-Invariante (keine doppelte Bestrafung) ────────────


def test_source_baseline_does_not_penalize_input_artifacts(gate, sr):
    """
    §2.50: Das Gate bestraft keine Eigenschaften, die im Input bereits
    vorhanden waren (Primum non nocere).
    Wenn measure_source_baseline ein kritisches Stereoproblem detektiert,
    liefert evaluate_phase() beim Delta-Vergleich gegen sich selbst KEINE
    Phase-Cancellation-Artefakte (kein Delta → kein Artefakt eingeführt).
    """
    t = np.arange(sr * 3) / sr
    sig = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5
    audio = _make_stereo(sig, -sig)  # anti-phase input

    baseline = gate.measure_source_baseline(audio, sr, "vinyl")
    assert baseline.has_anti_phase_region  # Input-Problem korrekt erkannt

    # Wenn Phase Input unveränderter weitergibt → keine neuen Artefakte
    # evaluate() per-phase-Modus: original=before, restored=after, phase_id gesetzt
    result_same_phase = gate.evaluate(
        original=audio,
        restored=audio,  # Identisch → kein Delta
        sr=sr,
        material_type="vinyl",
        phase_id="phase_14",
    )
    # delta=0 → keine phase_cancellation-Artefakte durch Phase eingeführt
    assert result_same_phase.artifact_freedom >= 0.95, (
        f"Identisches before/after sollte keine neuen Artefakte zeigen, "
        f"aber artifact_freedom={result_same_phase.artifact_freedom:.3f}"
    )
