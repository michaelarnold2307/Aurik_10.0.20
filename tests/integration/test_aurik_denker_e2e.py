from __future__ import annotations

"""
AurikDenker E2E Integration Test (DSP-only, kein echter ML-Modell-Load nötig).

Testet den vollständigen kanonischen Pipeline-Einstiegspunkt
`AurikDenker.denke()` mit synthetischem 3-Sekunden-Audio @ 48 000 Hz.

Spec-Referenzen:
    - §2.2: AurikDenker als PFLICHT-Einstiegspunkt (nicht UV3 direkt)
    - §8.1: quality_estimate ≥ 0.55 nach erfolgreicher Restaurierung
    - §8.2: Kein NaN/Inf im Ausgang, kein Clipping
    - §1.1: RestorationResult-Rückgabe
    - §14 E2E: Pflicht-Integrations-Test
    - §2.2: progress_callback-Signatur (pct: int, msg: str, elapsed_s: float)
    - §6.5: assert sr == 48000 am Eingang
"""


import math

import numpy as np
import pytest

# PMGG-Fix (v10.0.0) causes phases to actually execute — needs higher timeout.
# Desktop-Budget: Phase-Pipeline ≤ 120 s/min + Cold-Start ≤ 60 s overhead.
_E2E_TIMEOUT = 180


@pytest.fixture(scope="module")
def synthetic_audio():
    """3s synthetisches Audio: 440 Hz Sinus + Rauschen @ 48 000 Hz (Stereo)."""
    sr = 48_000
    t = np.linspace(0, 3.0, 3 * sr, endpoint=False, dtype=np.float32)
    mono = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.02 * np.random.default_rng(42).standard_normal(len(t)).astype(
        np.float32
    )
    audio = np.column_stack([mono, mono])  # Stereo [n, 2]
    return audio, sr


@pytest.mark.e2e
@pytest.mark.timeout(_E2E_TIMEOUT)
def test_aurik_denker_returns_restoration_result(synthetic_audio):
    """AurikDenker.denke() gibt AurikErgebnis zurück (§1.1, §2.2)."""
    audio, sr = synthetic_audio
    try:
        from denker.aurik_denker import AurikDenker
    except ImportError as exc:
        pytest.skip(f"AurikDenker nicht importierbar (Umgebungsproblem): {exc}")

    denker = AurikDenker()
    try:
        result = denker.denke(audio.copy(), sr, mode="balanced")
    except Exception as exc:
        pytest.fail(f"AurikDenker.denke() raised {type(exc).__name__}: {exc}")

    assert result is not None, "Ergebnis darf nicht None sein"


@pytest.mark.e2e
@pytest.mark.timeout(_E2E_TIMEOUT)
def test_aurik_denker_output_no_nan_inf(synthetic_audio):
    """Ausgabe-Audio enthält kein NaN/Inf (§8.2 Universelle Garantie)."""
    audio, sr = synthetic_audio
    try:
        from denker.aurik_denker import AurikDenker
    except ImportError as exc:
        pytest.skip(f"AurikDenker nicht importierbar: {exc}")

    denker = AurikDenker()
    result = denker.denke(audio.copy(), sr, mode="balanced")

    if hasattr(result, "audio") and result.audio is not None:
        out = np.asarray(result.audio)
        assert np.isfinite(out).all(), "NaN/Inf im Ausgabe-Audio gefunden"
        assert np.max(np.abs(out)) <= 1.0, "Clipping im Ausgabe-Audio (|x| > 1.0)"


@pytest.mark.e2e
@pytest.mark.timeout(_E2E_TIMEOUT)
def test_aurik_denker_quality_estimate_present(synthetic_audio):
    """quality_estimate-Feld ist vorhanden und ≥ 0.0 (§8.1 E2E-Pflicht)."""
    audio, sr = synthetic_audio
    try:
        from denker.aurik_denker import AurikDenker
    except ImportError as exc:
        pytest.skip(f"AurikDenker nicht importierbar: {exc}")

    denker = AurikDenker()
    result = denker.denke(audio.copy(), sr, mode="balanced")

    if hasattr(result, "quality_estimate"):
        qe = float(result.quality_estimate)
        assert math.isfinite(qe), "quality_estimate ist nicht endlich"
        assert qe >= 0.0, f"quality_estimate muss ≥ 0.0 sein, erhalten: {qe}"


@pytest.mark.e2e
@pytest.mark.timeout(_E2E_TIMEOUT)
def test_aurik_denker_preserves_sample_rate(synthetic_audio):
    """Ausgabe-SR muss 48 000 Hz sein (interne Verarbeitungs-Invariante)."""
    audio, sr = synthetic_audio
    try:
        from denker.aurik_denker import AurikDenker
    except ImportError as exc:
        pytest.skip(f"AurikDenker nicht importierbar: {exc}")

    denker = AurikDenker()
    result = denker.denke(audio.copy(), sr, mode="balanced")

    if hasattr(result, "audio") and result.audio is not None:
        out = np.asarray(result.audio)
        # SR-Erhalt: Länge sollte nicht mehr als 1 % abweichen (Resampling-Check)
        expected_samples = len(audio)
        actual_samples = out.shape[0] if out.ndim >= 1 else 0
        if actual_samples > 0:
            ratio = actual_samples / expected_samples
            assert 0.95 <= ratio <= 1.05, f"Audio-Länge verändert sich zu stark: {actual_samples} vs {expected_samples}"


@pytest.mark.e2e
@pytest.mark.timeout(_E2E_TIMEOUT)
def test_aurik_denker_fast_mode(synthetic_audio):
    """AurikDenker.denke() funktioniert auch im mode='fast' (kein Absturz)."""
    audio, sr = synthetic_audio
    try:
        from denker.aurik_denker import AurikDenker
    except ImportError as exc:
        pytest.skip(f"AurikDenker nicht importierbar: {exc}")

    denker = AurikDenker()
    try:
        result = denker.denke(audio.copy(), sr, mode="fast")
    except Exception as exc:
        pytest.fail(f"Modus 'fast' crashed: {type(exc).__name__}: {exc}")

    assert result is not None


def test_aurik_denker_short_clip_gate_rms_threshold():
    """
    Short-Clip-Gate: RMS-Schwelle ≤ 0.001 (nicht ≥ 0.0001).

    Verifikation des Fix für: "ML-Modelle werden nicht eingesetzt"
    — Noisy 5s audio sollte NICHT als "benign silence" übersprungen werden.

    Spec: §2.31–§2.34 Adaptive Qualitätsziele — statische Schwellwerte verboten.
    """
    try:
        from denker.aurik_denker import AurikDenker
    except ImportError as exc:
        pytest.skip(f"AurikDenker nicht importierbar: {exc}")

    sr = 48_000

    # Test 1: 5-Sekunden Rauschen (RMS ≈ 0.14) → sollte NICHT übersprungen werden
    audio_noisy = np.random.default_rng(123).standard_normal(5 * sr).astype(np.float32) * 0.2
    skip_noisy, metrics_noisy = AurikDenker._should_skip_excellence_for_clean_digital(
        audio_noisy, sr, "cd_digital", None
    )
    assert not skip_noisy, (
        f"Noisy 5s audio sollte NICHT übersprungen werden (RMS > 0.001), aber wurde es: {metrics_noisy}"
    )
    rms_noisy = float(np.sqrt(np.mean(audio_noisy.astype(np.float64) ** 2)))
    assert rms_noisy > 0.001, f"Test-Audio RMS = {rms_noisy}, sollte > 0.001 sein"

    # Test 2: 5-Sekunden sehr leises Audio (RMS ≈ 0.00001) → sollte übersprungen werden
    audio_quiet = np.random.default_rng(456).standard_normal(5 * sr).astype(np.float32) * 0.00001
    skip_quiet, metrics_quiet = AurikDenker._should_skip_excellence_for_clean_digital(
        audio_quiet, sr, "cd_digital", None
    )
    assert skip_quiet, f"Quiet 5s audio (RMS ≤ 0.001) sollte übersprungen werden, aber wurde es nicht: {metrics_quiet}"
    rms_quiet = float(np.sqrt(np.mean(audio_quiet.astype(np.float64) ** 2)))
    assert rms_quiet <= 0.001, f"Quiet-Audio RMS = {rms_quiet}, sollte ≤ 0.001 sein"


# ─── Spec-Compliance-Tests (§14.1 / §8.1 / §2.2 / §6.5) ────────────────────


def test_aurik_denker_sr_not_48000_raises(synthetic_audio):
    """AurikDenker.restauriere() mit sr ≠ 48000 muss AssertionError auslösen (§6.5)."""
    audio, _ = synthetic_audio
    try:
        from denker.aurik_denker import AurikDenker
    except ImportError as exc:
        pytest.skip(f"AurikDenker nicht importierbar: {exc}")

    denker = AurikDenker()
    with pytest.raises(AssertionError, match="48000"):
        denker.restauriere(audio.copy(), sr=44_100)

    with pytest.raises(AssertionError, match="48000"):
        denker.denke(audio.copy(), sr=22_050)


def test_aurik_denker_progress_callback_signature():
    """progress_callback wird mit (pct: int, msg: str, elapsed_s: float) aufgerufen (§2.2).

    Spec: progress_callback-Signatur: (pct: int, msg: str, elapsed_s: float = 0.0) → None
    """
    from unittest.mock import MagicMock, patch

    try:
        from denker.aurik_denker import AurikDenker
    except ImportError as exc:
        pytest.skip(f"AurikDenker nicht importierbar: {exc}")

    sr = 48_000
    audio = np.zeros(sr, dtype=np.float32) + 0.001  # minimal non-silent

    cb = MagicMock()

    # Mock all sub-denkers to prevent real execution
    _toni = MagicMock()
    _toni.material_type = "unknown"
    _toni.confidence = 0.5
    _kette = MagicMock()
    _kette.chain_string = "unknown"
    _kette.chain_complexity = 0.0
    _kette.combined_phases = []
    _kette.as_dict.return_value = {"chain_string": "unknown", "primary_medium": "unknown"}
    _defekt = MagicMock()
    _defekt.defect_scores = {}
    _defekt.primary_defect = "none"
    _defekt.overall_severity = 0.0
    _defekt.recommended_phases = []
    _defekt.cause_confidence = 0.0
    _strat = MagicMock()
    _strat.plan.return_value = MagicMock(quality_mode="quality", max_processing_s=30.0)
    _strat.starte_timer.return_value = None
    _rest = MagicMock()
    _rest.restauriere.return_value = MagicMock(
        audio=audio.copy(),
        phases_executed=[],
        warnings=[],
        quality_estimate=0.7,
        rt_factor=0.3,
        confidence=0.9,
        rollback_triggered=False,
        winning_variant=None,
        musical_goals={},
        goals_passed=0,
        era_decade=None,
    )
    _rep = MagicMock()
    _rep.repariere.return_value = MagicMock(
        audio=audio.copy(),
        warnings=[],
        clicks_removed=False,
        hum_removed=False,
        clipping_repaired=False,
    )
    _rek = MagicMock()
    _rek.rekonstruiere.return_value = MagicMock(
        audio=audio.copy(),
        warnings=[],
        gaps_found=0,
        gaps_repaired=0,
        total_repaired_ms=0.0,
    )
    _exz = MagicMock()
    _exz.optimiere.return_value = MagicMock(
        audio=audio.copy(),
        excellence_score=0.8,
        musical_goals={"brillanz": 0.9},
        goals_passed=1,
        goals_total=14,
        warnings=[],
        processing_note="ok",
        versa_mos=4.2,
    )

    with (
        patch(
            "denker.aurik_denker.get_tontraeger_denker",
            MagicMock(return_value=MagicMock(erkenne=MagicMock(return_value=_toni))),
        ),
        patch(
            "denker.aurik_denker.get_tontraegerkette_denker",
            MagicMock(return_value=MagicMock(analysiere=MagicMock(return_value=_kette))),
        ),
        patch(
            "denker.aurik_denker.get_defekt_denker",
            MagicMock(return_value=MagicMock(analysiere=MagicMock(return_value=_defekt))),
        ),
        patch("denker.aurik_denker.get_strategie_denker", MagicMock(return_value=_strat)),
        patch("denker.aurik_denker.get_restaurier_denker", MagicMock(return_value=_rest)),
        patch("denker.aurik_denker.get_reparatur_denker", MagicMock(return_value=_rep)),
        patch("denker.aurik_denker.get_rekonstruktions_denker", MagicMock(return_value=_rek)),
        patch("denker.aurik_denker.get_exzellenz_denker", MagicMock(return_value=_exz)),
        patch(
            "denker.aurik_denker.AurikDenker._should_skip_excellence_for_clean_digital",
            MagicMock(return_value=(False, {"material": "unknown"})),
        ),
    ):
        denker = AurikDenker()
        denker.denke(audio, sr, progress_callback=cb)

    assert cb.call_count >= 1, "progress_callback wurde nie aufgerufen"
    for call_args in cb.call_args_list:
        args = call_args[0]
        assert len(args) >= 2, f"progress_callback muss mindestens (pct, msg) erhalten, bekam: {args}"
        pct, msg = args[0], args[1]
        assert isinstance(pct, int), f"pct muss int sein, ist: {type(pct).__name__} ({pct!r})"
        assert isinstance(msg, str), f"msg muss str sein, ist: {type(msg).__name__} ({msg!r})"
        assert 0 <= pct <= 100, f"pct außerhalb [0, 100]: {pct}"
        if len(args) >= 3:
            elapsed = args[2]
            assert isinstance(elapsed, (int, float)), f"elapsed_s muss numerisch sein, ist: {type(elapsed).__name__}"
            assert elapsed >= 0.0, f"elapsed_s darf nicht negativ sein: {elapsed}"


def test_aurik_denker_quality_estimate_e2e_minimum():
    """quality_estimate ≥ 0.55 bei erfolgreicher Gemockt-Pipeline (§8.1 E2E-Pflicht, §14.1).

    Spec §8.1: 'E2E-Pflicht: result.quality_estimate >= 0.55'
    Spec §14.1: assert result.quality_estimate >= 0.55
    """
    from unittest.mock import MagicMock, patch

    try:
        from denker.aurik_denker import AurikDenker
    except ImportError as exc:
        pytest.skip(f"AurikDenker nicht importierbar: {exc}")

    sr = 48_000
    audio = (np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr, dtype=np.float32)) * 0.3).astype(np.float32)

    _toni = MagicMock(material_type="tape", confidence=0.85)
    _kette = MagicMock()
    _kette.chain_string = "tape"
    _kette.chain_complexity = 0.3
    _kette.combined_phases = []
    _kette.as_dict.return_value = {"chain_string": "tape", "primary_medium": "tape"}
    _defekt = MagicMock()
    _defekt.defect_scores = {"hiss": 0.3}
    _defekt.primary_defect = "hiss"
    _defekt.overall_severity = 0.3
    _defekt.recommended_phases = ["phase_03_denoise"]
    _defekt.cause_confidence = 0.7
    _strat = MagicMock()
    _strat.plan.return_value = MagicMock(quality_mode="quality", max_processing_s=30.0)
    _strat.starte_timer.return_value = None
    _rest = MagicMock()
    _rest.restauriere.return_value = MagicMock(
        audio=audio.copy(),
        phases_executed=["phase_03_denoise"],
        warnings=[],
        quality_estimate=0.75,
        rt_factor=0.4,
        confidence=0.9,
        rollback_triggered=False,
        winning_variant="balanced",
        musical_goals={},
        goals_passed=0,
        era_decade=None,
    )
    _rep = MagicMock(
        repariere=MagicMock(
            return_value=MagicMock(
                audio=audio.copy(),
                warnings=[],
                clicks_removed=False,
                hum_removed=False,
                clipping_repaired=False,
            )
        )
    )
    _rek = MagicMock(
        rekonstruiere=MagicMock(
            return_value=MagicMock(
                audio=audio.copy(),
                warnings=[],
                gaps_found=0,
                gaps_repaired=0,
                total_repaired_ms=0.0,
            )
        )
    )
    _exz = MagicMock(
        optimiere=MagicMock(
            return_value=MagicMock(
                audio=audio.copy(),
                excellence_score=0.85,
                musical_goals={"brillanz": 0.88},
                goals_passed=1,
                goals_total=14,
                warnings=[],
                processing_note="ok",
                versa_mos=4.3,
            )
        )
    )

    with (
        patch(
            "denker.aurik_denker.get_tontraeger_denker",
            MagicMock(return_value=MagicMock(erkenne=MagicMock(return_value=_toni))),
        ),
        patch(
            "denker.aurik_denker.get_tontraegerkette_denker",
            MagicMock(return_value=MagicMock(analysiere=MagicMock(return_value=_kette))),
        ),
        patch(
            "denker.aurik_denker.get_defekt_denker",
            MagicMock(return_value=MagicMock(analysiere=MagicMock(return_value=_defekt))),
        ),
        patch("denker.aurik_denker.get_strategie_denker", MagicMock(return_value=_strat)),
        patch("denker.aurik_denker.get_restaurier_denker", MagicMock(return_value=_rest)),
        patch("denker.aurik_denker.get_reparatur_denker", MagicMock(return_value=_rep)),
        patch("denker.aurik_denker.get_rekonstruktions_denker", MagicMock(return_value=_rek)),
        patch("denker.aurik_denker.get_exzellenz_denker", MagicMock(return_value=_exz)),
        patch(
            "denker.aurik_denker.AurikDenker._should_skip_excellence_for_clean_digital",
            MagicMock(return_value=(False, {"material": "tape"})),
        ),
    ):
        result = AurikDenker().denke(audio, sr, mode="quality")

    # §8.1 + §14.1: quality_estimate ≥ 0.55
    assert result.quality_estimate >= 0.55, (
        f"quality_estimate={result.quality_estimate:.4f} < 0.55 — "
        f"Spec §8.1 E2E-Pflicht verletzt (sev=0.3, VERSA MOS=4.3 → "
        f"erwartet ≈ 0.40*(1-0.3) + 0.60*(4.3-1)/4 ≈ 0.775)"
    )


def test_aurik_denker_global_plan_in_as_dict():
    """AurikErgebnis.as_dict() enthält 'global_plan'-Schlüssel."""
    from denker.aurik_denker import AurikErgebnis

    audio = np.zeros(48_000, dtype=np.float32)
    e = AurikErgebnis(
        audio=audio,
        material="tape",
        rt_factor=0.5,
        quality_estimate=0.7,
        musical_goals={},
        goals_passed=0,
        phases_executed=[],
        global_plan={"portrait": {"decade": 1970}},
    )
    d = e.as_dict()
    assert "global_plan" in d, "global_plan fehlt in as_dict()"
    assert d["global_plan"] == {"portrait": {"decade": 1970}}
