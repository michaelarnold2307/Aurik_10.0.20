from __future__ import annotations

import pytest

"""Tests für backend/core/phases/phase_65_vocal_naturalness_restoration.py (§7.10)."""


import numpy as np

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_vocal_audio(sr: int = 48000, duration: float = 1.0) -> np.ndarray:
    """Einfaches Sinus-Signal als Vokal-Dummy."""
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    return (0.25 * np.sin(2 * np.pi * 220 * t)).reshape(1, -1)  # type: ignore[no-any-return]  # (1, N) channels-first


def _make_phase():
    from backend.core.phases.phase_65_vocal_naturalness_restoration import get_phase_65

    return get_phase_65()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_singleton():
    from backend.core.phases.phase_65_vocal_naturalness_restoration import get_phase_65

    a = get_phase_65()
    b = get_phase_65()
    assert a is b


# ---------------------------------------------------------------------------
# §0a: Studio-2026-Bypass
# ---------------------------------------------------------------------------


def test_studio2026_bypass():
    """quality_mode='studio_2026' MUSS passthrough (§0a) liefern."""
    phase = _make_phase()
    sr = 48000
    audio = _make_vocal_audio(sr)
    result = phase.process(
        audio.copy(),
        sr,
        quality_mode="studio_2026",
        panns_singing=0.8,
        pre_nr_audio=audio.copy(),
    )
    np.testing.assert_array_almost_equal(result.audio, audio, decimal=6)
    assert result.metadata.get("bypassed_reason") is not None or result.metadata.get("activation_triggered") is False


# ---------------------------------------------------------------------------
# Vokal-Gate: panns_singing < 0.25 → Passthrough
# ---------------------------------------------------------------------------


def test_vocal_gate_below_threshold():
    """panns_singing < 0.25 → passthrough, kein Processing."""
    phase = _make_phase()
    sr = 48000
    audio = _make_vocal_audio(sr)
    result = phase.process(
        audio.copy(),
        sr,
        quality_mode="restoration",
        panns_singing=0.10,
        pre_nr_audio=audio.copy(),
    )
    np.testing.assert_array_almost_equal(result.audio, audio, decimal=6)


# ---------------------------------------------------------------------------
# Kein pre_nr_audio → Passthrough
# ---------------------------------------------------------------------------


def test_no_pre_nr_audio_passthrough():
    """Fehlendes pre_nr_audio kwarg → passthrough (kein Crash)."""
    phase = _make_phase()
    sr = 48000
    audio = _make_vocal_audio(sr)
    result = phase.process(
        audio.copy(),
        sr,
        quality_mode="restoration",
        panns_singing=0.9,
        # pre_nr_audio FEHLT absichtlich
    )
    np.testing.assert_array_almost_equal(result.audio, audio, decimal=6)


# ---------------------------------------------------------------------------
# Aktivierungs-Gate: delta_hnr ≤ 2.5 AND |tilt_delta| ≤ 1.5 → Passthrough
# ---------------------------------------------------------------------------


def test_activation_gate_no_trigger_returns_passthrough():
    """Wenn keine signifikante Veränderung → kein Eingriff."""
    phase = _make_phase()
    sr = 48000
    audio = _make_vocal_audio(sr)
    # pre_nr_audio = identisch zum Input → kein HNR-Delta
    result = phase.process(
        audio.copy(),
        sr,
        quality_mode="restoration",
        panns_singing=0.9,
        pre_nr_audio=audio.copy(),  # identisch
    )
    # Kein signifikanter Eingriff erwartet (Aktivierungs-Gate nicht ausgelöst)
    assert result is not None
    assert not np.any(np.isnan(result.audio))


# ---------------------------------------------------------------------------
# NaN-Sicherheit: kein NaN im Output
# ---------------------------------------------------------------------------


def test_no_nan_in_output():
    phase = _make_phase()
    sr = 48000
    audio = _make_vocal_audio(sr)
    # Rauschen als pre_nr (leicht anders → könnte Gate triggern)
    rng = np.random.default_rng(42)
    pre_nr = (audio + rng.normal(0, 0.01, audio.shape)).astype(np.float32)
    result = phase.process(
        audio.copy(),
        sr,
        quality_mode="restoration",
        panns_singing=0.8,
        pre_nr_audio=pre_nr,
    )
    assert not np.any(np.isnan(result.audio)), "Output enthält NaN"
    assert not np.any(np.isinf(result.audio)), "Output enthält Inf"


# ---------------------------------------------------------------------------
# Clip-Invariante: Output ∈ [-1, 1]
# ---------------------------------------------------------------------------


def test_output_clipped():
    phase = _make_phase()
    sr = 48000
    audio = _make_vocal_audio(sr) * 0.9
    result = phase.process(
        audio.copy(),
        sr,
        quality_mode="restoration",
        panns_singing=0.7,
        pre_nr_audio=audio.copy(),
    )
    assert np.max(np.abs(result.audio)) <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# get_metadata()
# ---------------------------------------------------------------------------


def test_get_metadata():
    from backend.core.phases.phase_interface import PhaseCategory

    phase = _make_phase()
    meta = phase.get_metadata()
    assert meta.phase_id == "phase_65_vocal_naturalness_restoration"
    assert meta.category == PhaseCategory.RESTORATION
    # §0a: KEINE studio_2026-Nutzung → category muss RESTORATION sein
    assert meta.category != getattr(
        __import__("backend.core.phases.phase_interface", fromlist=["PhaseCategory"]).PhaseCategory,
        "STUDIO",
        None,
    )


# ---------------------------------------------------------------------------
# Shape-Preservation
# ---------------------------------------------------------------------------


def test_shape_preserved_mono():
    phase = _make_phase()
    sr = 48000
    audio = np.zeros((1, sr), dtype=np.float32)
    audio[0, :100] = 0.5
    result = phase.process(audio.copy(), sr, quality_mode="restoration", panns_singing=0.6)
    assert result.audio.shape == audio.shape, f"Shape-Abweichung: Input {audio.shape} → Output {result.audio.shape}"


def test_shape_preserved_stereo():
    phase = _make_phase()
    sr = 48000
    audio = np.zeros((2, sr), dtype=np.float32)
    audio[:, :200] = 0.3
    result = phase.process(audio.copy(), sr, quality_mode="restoration", panns_singing=0.5)
    assert result.audio.shape == audio.shape


def test_phase_locality_factor_scales_effective_strength(monkeypatch):
    """phase_locality_factor MUSS die Eingriffsstaerke in Phase 65 reduzieren."""
    import backend.core.dsp.hallucination_guard as _hg_mod
    import backend.core.dsp.hnr_guard as _hnr_guard_mod
    import backend.core.phases.phase_65_vocal_naturalness_restoration as p65

    # compute_hnr kann intern ein RNN (GRU) triggern — in Unit-Tests mocken,
    # damit kein PyTorch-Warmup-Timeout entsteht (§tests.instructions.md).
    monkeypatch.setattr(_hnr_guard_mod, "compute_hnr", lambda *a, **kw: 5.0)

    # HallucinationGuard mocken: kein Rollback, damit d_full und d_local
    # deterministisch vergleichbar sind (nichtlinearer Guard verfälscht sonst
    # den locality-factor-Vergleich — spectral_novelty=0 durch Mock, V41-Guard OK).
    _ok_halluc = _hg_mod.HallucinationCheckResult(
        spectral_novelty=0.0,
        requires_rollback=False,
        score_penalty=0.0,
        harmonic_ceiling_violation=False,
        metadata={},
    )
    monkeypatch.setattr(_hg_mod, "check_hallucination", lambda *a, **kw: _ok_halluc)

    phase = _make_phase()
    sr = 48000
    audio = _make_vocal_audio(sr)
    pre_nr = (audio * 0.1).astype(np.float32)

    def _fake_tilt(arr: np.ndarray, _sr: int) -> float:
        return 0.0 if float(np.mean(np.abs(arr))) < 0.08 else 4.0

    def _fake_shelf(arr: np.ndarray, _sr: int, _hz: float, boost_db: float, _stype: str = "low") -> np.ndarray:
        return (arr * (1.0 + 0.05 * float(boost_db))).astype(arr.dtype)

    monkeypatch.setattr(p65, "_estimate_spectral_tilt_db", _fake_tilt)
    monkeypatch.setattr(p65, "_apply_shelving_eq", _fake_shelf)

    res_full = phase.process(
        audio.copy(),
        sr,
        quality_mode="restoration",
        panns_singing=0.9,
        pre_nr_audio=pre_nr,
        strength=1.0,
        phase_locality_factor=1.0,
    )
    res_local = phase.process(
        audio.copy(),
        sr,
        quality_mode="restoration",
        panns_singing=0.9,
        pre_nr_audio=pre_nr,
        strength=1.0,
        phase_locality_factor=0.35,
    )

    d_full = float(np.mean(np.abs(res_full.audio - audio)))
    d_local = float(np.mean(np.abs(res_local.audio - audio)))

    assert d_local <= d_full + 1e-8
    assert float(res_full.metadata.get("phase_locality_factor", 0.0)) == 1.0
    assert float(res_local.metadata.get("phase_locality_factor", 0.0)) <= 0.35 + 1e-6
    assert float(res_local.metadata.get("effective_strength", 0.0)) < float(
        res_full.metadata.get("effective_strength", 1.0)
    )


# ---------------------------------------------------------------------------
# §SOTA-PSY-A1: subaudibles Vokal-Delta wird gegated (Dry-Passthrough)
# ---------------------------------------------------------------------------


def test_psy_a1_subaudible_delta_passthrough(monkeypatch):
    """Gate meldet skippable ⇒ Phase gibt Dry zurück (Verifikation aller Stufen)."""
    phase = _make_phase()
    sr = 48000
    audio = _make_vocal_audio(sr)
    pre_nr = np.asarray(audio * 0.1, dtype=np.float32)

    # Erzwinge ein audibles Delta durch Tilt-Änderung, aber Gate auf skippable.
    def _fake_tilt(arr, _sr):
        return 4.0 if float(np.mean(np.abs(arr))) > 0.08 else 0.0

    def _fake_shelf(arr, _sr, _hz, boost_db, _stype="low"):
        return (arr * (1.0 + 0.05 * float(boost_db))).astype(arr.dtype)

    import backend.core.phases.phase_65_vocal_naturalness_restoration as p65

    monkeypatch.setattr(p65, "_estimate_spectral_tilt_db", _fake_tilt)
    monkeypatch.setattr(p65, "_apply_shelving_eq", _fake_shelf)

    import backend.core.dsp.audibility_gate as ag

    monkeypatch.setattr(
        ag,
        "defect_audibility",
        lambda *a, **kw: {"audible": False, "delta_db": 0.0, "threshold_db": 0.0, "skippable": True},
    )

    result = phase.process(
        audio.copy(),
        sr,
        quality_mode="restoration",
        panns_singing=0.8,
        pre_nr_audio=pre_nr,
        strength=1.0,
        phase_locality_factor=1.0,
    )
    assert result.metadata.get("subaudible_defects_skipped") is True
    np.testing.assert_array_almost_equal(result.audio, audio, decimal=5)
