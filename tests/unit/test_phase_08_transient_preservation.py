"""tests/unit/test_phase_08_transient_preservation.py — §v10.700 I1."""

import numpy as np
import pytest

from backend.core.phases.phase_08_transient_preservation import TransientPreservationPhase


@pytest.fixture
def phase():
    return TransientPreservationPhase()


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


# ---------------------------------------------------------------------------
# §SOTA-PSY-A1: subaudible Transport-Bumps werden übersprungen (Muster A)
# ---------------------------------------------------------------------------


def _make_transient_audio(sr: int = 48000, duration: float = 1.0) -> np.ndarray:
    """Signal mit Transienten + Grundton (für echte Onset-/Bump-Detektion)."""
    rng = np.random.RandomState(42)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    x = (np.sin(2 * np.pi * 440 * t) * 0.2 + rng.randn(int(sr * duration)) * 0.005).astype(np.float32)
    for i in range(int(0.1 * sr), int(sr * duration), int(0.2 * sr)):
        x[i : i + 30] += 0.6
    return x


def test_psy_a1_bump_gate_uses_correct_band_and_skips(monkeypatch):
    """Muster A: Gate meldet skippable ⇒ Bump-Verarbeitung wird übersprungen.

    Verifiziert die Band-Wahl (200–8000 Hz) über einen Recording-Monkeypatch.
    """
    import backend.core.dsp.audibility_gate as ag

    sr = 48000
    phase = TransientPreservationPhase()
    audio = _make_transient_audio(sr)
    calls: list[dict] = []

    def _fake(x, _sr, s, e, lo_hz=None, hi_hz=None, **kw):
        calls.append({"s": s, "e": e, "lo_hz": lo_hz, "hi_hz": hi_hz})
        return {"audible": False, "delta_db": 0.0, "threshold_db": 0.0, "skippable": True}

    monkeypatch.setattr(ag, "defect_audibility", _fake)

    result = phase.process(
        audio.copy(),
        sample_rate=sr,
        material_type="vinyl",
        defect_locations={"transport_bump": [(0.1, 0.12), (0.3, 0.32)]},
    )
    assert calls, "defect_audibility wurde nicht aufgerufen"
    assert all(c["lo_hz"] == 200.0 and c["hi_hz"] == 8000.0 for c in calls), f"Band-Wahl falsch: {calls}"
    # Kein Bump wurde repariert (alle subaudible → übersprungen).
    assert result.modifications.get("transport_bumps_repaired", 0) == 0
    assert np.isfinite(result.audio).all()


def test_psy_a1_real_band_does_not_deactivate_phase():
    """Realer Gegentest: hörbares Transienten-Signal im richtigen Band (200–8000 Hz)
    ⇒ Bump-Gate crasht nicht und deaktiviert die Phase nicht.
    """
    sr = 48000
    phase = TransientPreservationPhase()
    audio = _make_transient_audio(sr)
    result = phase.process(
        audio.copy(),
        sample_rate=sr,
        material_type="vinyl",
        defect_locations={"transport_bump": [(0.1, 0.12), (0.3, 0.32)]},
    )
    # Phase lief erfolgreich durch (transient_preserved=True bedeutet der Hauptpfad
    # wurde erreicht, kein Gate-Crash und kein falscher Gesamtausstieg).
    assert result.success is True
    assert np.isfinite(result.audio).all()


def test_psy_a1_bump_gate_stereo_runs_without_collapse(monkeypatch):
    """Stereo-Robustheit (Muster A): das Gate nutzt die ravelte Original-Eingabe und
    überspringt den Bump; der Rückgabe-Pfad restauriert das (C, N)-Layout.

    Befund 2026-09-14: der Hauptpfad gab Stereo zuvor channels-last (N, 2)
    zurück (kein restore_layout) — mit dem _restore_08-Fix muss die exakte
    (2, N)-Form erhalten bleiben (AGENTS.md Stereo-Layout-Invariante).
    """
    import backend.core.dsp.audibility_gate as ag

    sr = 48000
    phase = TransientPreservationPhase()
    mono = _make_transient_audio(sr)
    audio = np.vstack([mono, mono * 0.9]).astype(np.float32)  # channels-first (2, N)

    monkeypatch.setattr(
        ag,
        "defect_audibility",
        lambda *a, **kw: {"audible": False, "delta_db": 0.0, "threshold_db": 0.0, "skippable": True},
    )

    result = phase.process(
        audio.copy(),
        sample_rate=sr,
        material_type="vinyl",
        defect_locations={"transport_bump": [(0.1, 0.12), (0.3, 0.32)]},
    )
    assert result.success is True
    assert result.audio.shape == audio.shape, f"Layout-Kollaps: {result.audio.shape} statt {audio.shape}"
    assert result.audio.size == audio.size, f"Gesamtsamplezahl kollabiert: {result.audio.size} statt {audio.size}"
    assert np.isfinite(result.audio).all()
    assert result.modifications.get("transport_bumps_repaired", 0) == 0
