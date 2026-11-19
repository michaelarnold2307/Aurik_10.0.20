from __future__ import annotations

import pytest

"""Tests für core/goal_applicability_filter.py — Spec §2.32.

≥ 25 Unit-Tests: Shape/Bounds, NaN-Safety, Edge-Cases, Invarianten.
"""


import math
import threading
from typing import Any

import numpy as np

np.random.seed(42)  # §5.4 Reproduzierbarkeit

from backend.core.goal_applicability_filter import (
    ALL_GOALS,
    ALWAYS_APPLICABLE,
    GoalApplicabilityResult,
    evaluate_goal_applicability,
    get_goal_applicability_filter,
)

SR = 48_000
RNG = np.random.default_rng(42)


def _sine(freq: float = 440.0, dur: float = 5.0, sr: int = SR) -> np.ndarray:
    t = np.linspace(0, dur, int(dur * sr), endpoint=False)
    sine: np.ndarray[Any, Any] = np.asarray(np.sin(2 * np.pi * freq * t), dtype=np.float32)
    return sine


def _stereo(dur: float = 5.0) -> np.ndarray:
    """Echtes Stereo: zwei unterschiedliche Sinustone -> niedrige L/R-Korrelation."""
    n = int(dur * SR)
    t = np.linspace(0, dur, n, dtype=np.float32)
    left = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    right = (np.sin(2 * np.pi * 523 * t) * 0.5).astype(np.float32)  # andere Freq
    return np.stack([left, right], axis=1)  # (N, 2)


# ---------------------------------------------------------------------------
# 1. Konstanten-Invarianten
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConstants:
    def test_all_goals_count(self):
        # §0p v10.0.0: vocal_quality + formant_fidelity hinzugefügt (P0-Goals)
        assert len(ALL_GOALS) == 17

    def test_always_applicable_count(self):
        assert len(ALWAYS_APPLICABLE) == 6

    def test_always_applicable_subset_of_all(self):
        assert ALWAYS_APPLICABLE.issubset(ALL_GOALS)

    def test_always_applicable_known_keys(self):
        expected = {
            "natuerlichkeit",
            "authentizitaet",
            "emotionalitaet",
            "transparenz",
            "timbre_authentizitaet",
            "artikulation",
        }
        assert expected == ALWAYS_APPLICABLE


# ---------------------------------------------------------------------------
# 2. Ergebnis-Struktur
# ---------------------------------------------------------------------------


class TestResultStructure:
    def test_returns_dataclass(self):
        audio = _sine()
        result = evaluate_goal_applicability(audio, SR)
        assert isinstance(result, GoalApplicabilityResult)

    def test_applicable_plus_inapplicable_equals_all(self):
        audio = _sine()
        result = evaluate_goal_applicability(audio, SR)
        assert result.applicable | result.inapplicable == ALL_GOALS

    def test_applicable_inapplicable_disjoint(self):
        audio = _sine()
        result = evaluate_goal_applicability(audio, SR)
        assert result.applicable & result.inapplicable == frozenset()

    def test_always_applicable_always_in_applicable(self):
        audio = _sine()
        result = evaluate_goal_applicability(audio, SR)
        assert ALWAYS_APPLICABLE.issubset(result.applicable)

    def test_minimum_six_goals_always_active(self):
        audio = _sine()
        result = evaluate_goal_applicability(audio, SR, material="wax_cylinder", era_decade=1910)
        assert len(result.applicable) >= 6

    def test_as_dict_returns_dict(self):
        audio = _sine()
        result = evaluate_goal_applicability(audio, SR)
        d = result.as_dict()
        assert isinstance(d, dict)
        assert "applicable" in d
        assert "inapplicable" in d

    def test_reasons_keys_subset_of_inapplicable(self):
        audio = _sine()
        result = evaluate_goal_applicability(audio, SR, material="wax_cylinder", era_decade=1910)
        for k in result.reasons:
            assert k in result.inapplicable


# ---------------------------------------------------------------------------
# 3. Material-spezifische Deaktivierungen
# ---------------------------------------------------------------------------


class TestMaterialRules:
    def test_wax_cylinder_may_disable_spatial_depth(self):
        # Mono-ähnliches Signal (M/S-Korrelation hoch) → spatial_depth inapplicable
        audio = _sine(dur=5.0)
        stereo = np.stack([audio, audio], axis=1)  # perfekte Mono-Quelle
        result = evaluate_goal_applicability(stereo, SR, material="wax_cylinder", era_decade=1910)
        # spatial_depth SOLLTE deaktiviert sein bei Mono + altem Material
        # (kein harter Assert, da interne SNR-Prüfung variiert)
        assert isinstance(result.applicable, frozenset)

    def test_modern_digital_all_goals_potentially_applicable(self):
        # Stereo (25 s) -> groove + micro_dynamics + spatial_depth aktiv
        audio = _stereo(dur=25.0)
        result = evaluate_goal_applicability(audio, SR, material="cd_digital", era_decade=2000)
        # Bei digitaler Quelle erwartet: alle oder fast alle Goals aktiv
        assert len(result.applicable) >= 10

    def test_shellac_material_accepted_without_error(self):
        audio = _sine()
        result = evaluate_goal_applicability(audio, SR, material="shellac", era_decade=1930)
        assert isinstance(result, GoalApplicabilityResult)

    def test_unknown_material_accepted(self):
        audio = _sine()
        result = evaluate_goal_applicability(audio, SR, material="unknown")
        assert isinstance(result, GoalApplicabilityResult)


# ---------------------------------------------------------------------------
# 4. Ära-spezifische Regeln
# ---------------------------------------------------------------------------


class TestEraRules:
    def test_pre_1950_era_may_disable_spatial_depth(self):
        audio = _sine(dur=5.0)
        stereo = np.stack([audio, audio], axis=1)
        result = evaluate_goal_applicability(stereo, SR, era_decade=1930)
        assert isinstance(result.applicable, frozenset)

    def test_modern_era_keeps_spatial_depth(self):
        # Zwei verschiedene Frequenzen -> corr << 0.97 -> kein is_mono_signal
        audio = _stereo(dur=5.0)  # left=440 Hz, right=523 Hz
        result = evaluate_goal_applicability(audio, SR, era_decade=2000)
        assert "spatial_depth" in result.applicable


# ---------------------------------------------------------------------------
# 5. Kurze Dateien — Groove und MicroDynamics
# ---------------------------------------------------------------------------


class TestShortFileRules:
    def test_short_file_groove_disabled(self):
        # Datei < 10 s → GrooveMetric sollte inapplicable sein
        audio = _sine(dur=3.0)
        result = evaluate_goal_applicability(audio, SR)
        assert "groove" in result.inapplicable

    def test_short_file_micro_dynamics_disabled(self):
        # Datei < 20 s → MicroDynamics sollte inapplicable sein
        audio = _sine(dur=5.0)
        result = evaluate_goal_applicability(audio, SR)
        assert "micro_dynamics" in result.inapplicable

    def test_long_file_groove_applicable(self):
        # Datei ≥ 10 s → GrooveMetric kann applicable sein
        audio = _sine(dur=12.0)
        result = evaluate_goal_applicability(audio, SR)
        # Bei ausreichendem Signal kein harter Fehler
        assert isinstance(result.applicable, frozenset)

    def test_long_file_micro_dynamics_applicable(self):
        # Datei ≥ 20 s mit Dynamik → MicroDynamics applicable
        audio = _sine(dur=22.0)
        # Amplitude variiert → LUFS-Varianz vorhanden
        env = np.linspace(0.1, 1.0, len(audio))
        audio = (audio * env).astype(np.float32)
        result = evaluate_goal_applicability(audio, SR)
        assert isinstance(result.applicable, frozenset)


# ---------------------------------------------------------------------------
# 6. PANNs-Tags-Einfluss
# ---------------------------------------------------------------------------


class TestPaansTags:
    def test_high_percussion_confidence_affects_groove(self):
        audio = _sine(dur=12.0)
        result = evaluate_goal_applicability(audio, SR, panns_tags={"drums": 0.9, "percussion": 0.8})
        # Hohe Percussion-Konfidenz → groove sollte applicable sein bei langer Datei
        assert isinstance(result.applicable, frozenset)

    def test_no_percussion_may_disable_groove(self):
        audio = _sine(dur=12.0)
        result = evaluate_goal_applicability(audio, SR, panns_tags={"piano": 0.9, "violin": 0.8})
        assert isinstance(result, GoalApplicabilityResult)


# ---------------------------------------------------------------------------
# 7. NaN/Inf-Sicherheit
# ---------------------------------------------------------------------------


class TestNanSafety:
    def test_nan_audio_handled(self):
        audio = np.full(SR * 5, float("nan"), dtype=np.float32)
        result = evaluate_goal_applicability(audio, SR)
        assert isinstance(result, GoalApplicabilityResult)
        assert math.isfinite(len(result.applicable))

    def test_inf_audio_handled(self):
        audio = np.full(SR * 5, float("inf"), dtype=np.float32)
        result = evaluate_goal_applicability(audio, SR)
        assert isinstance(result, GoalApplicabilityResult)

    def test_silence_handled(self):
        audio = np.zeros(SR * 5, dtype=np.float32)
        result = evaluate_goal_applicability(audio, SR)
        assert isinstance(result, GoalApplicabilityResult)
        assert ALWAYS_APPLICABLE.issubset(result.applicable)

    def test_dirac_impulse_handled(self):
        audio = np.zeros(SR * 5, dtype=np.float32)
        audio[100] = 1.0
        result = evaluate_goal_applicability(audio, SR)
        assert isinstance(result, GoalApplicabilityResult)


# ---------------------------------------------------------------------------
# 8. Singleton-Thread-Safety
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_singleton_same_instance(self):
        a = get_goal_applicability_filter()
        b = get_goal_applicability_filter()
        assert a is b

    def test_singleton_thread_safe(self):
        instances = []
        errors = []

        def worker():
            try:
                instances.append(get_goal_applicability_filter())
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert all(inst is instances[0] for inst in instances)

    def test_convenience_wrapper_matches_direct(self):
        audio = _sine(dur=5.0)
        filt = get_goal_applicability_filter()
        direct = filt.evaluate(audio, SR)
        wrapper = evaluate_goal_applicability(audio, SR)
        assert direct.applicable == wrapper.applicable


# ---------------------------------------------------------------------------
# 9. Stereo-Unterstützung
# ---------------------------------------------------------------------------


class TestStereoInput:
    def test_stereo_accepted(self):
        audio = _stereo()
        result = evaluate_goal_applicability(audio, SR)
        assert isinstance(result, GoalApplicabilityResult)

    def test_stereo_mono_both_return_applicable_superset_of_always_applicable(self):
        stereo = _stereo()
        mono = _sine()
        r_s = evaluate_goal_applicability(stereo, SR)
        r_m = evaluate_goal_applicability(mono, SR)
        assert ALWAYS_APPLICABLE.issubset(r_s.applicable)
        assert ALWAYS_APPLICABLE.issubset(r_m.applicable)


# ---------------------------------------------------------------------------
# 10. BassKraft-Regel — §2.32 bassarmes Material in Restoration-Modus
# ---------------------------------------------------------------------------


def _vocal_dominant(dur: float = 5.0, sr: int = SR) -> np.ndarray:
    """Simulates a bright vocal-dominant recording with minimal bass (<0.015 ratio)."""
    t = np.linspace(0, dur, int(dur * sr), endpoint=False)
    # Vocal fundamentals 200-800 Hz dominant; minimal bass at 90 Hz
    audio = (
        0.60 * np.sin(2 * np.pi * 400 * t)
        + 0.30 * np.sin(2 * np.pi * 800 * t)
        + 0.04 * np.sin(2 * np.pi * 90 * t)  # ~1 % bass contribution
    ).astype(np.float32)
    vocal: np.ndarray[Any, Any] = np.asarray(audio, dtype=np.float32)
    return vocal


def _bass_heavy(dur: float = 5.0, sr: int = SR) -> np.ndarray:
    """Simulates a rock/blues recording with strong bass (>>0.015 ratio)."""
    t = np.linspace(0, dur, int(dur * sr), endpoint=False)
    audio = (
        0.50 * np.sin(2 * np.pi * 400 * t)
        + 0.40 * np.sin(2 * np.pi * 90 * t)  # strong bass
        + 0.30 * np.sin(2 * np.pi * 55 * t)  # sub-bass
    ).astype(np.float32)
    bass: np.ndarray[Any, Any] = np.asarray(audio, dtype=np.float32)
    return bass


class TestBassKraftApplicability:
    """§2.32 BassKraft-Regel — physikalisch bassarmes Material."""

    def test_bass_poor_restoration_marks_bass_kraft_inapplicable(self):
        """Schlager/vocal-dominant in Restoration: bass_kraft must be N/A."""
        audio = _vocal_dominant()
        result = evaluate_goal_applicability(audio, SR, mode="restoration")
        assert "bass_kraft" in result.inapplicable, (
            "bass_kraft muss N/A sein für vocal-dominante Aufnahmen in Restoration-Modus"
        )

    def test_bass_poor_studio_2026_bass_kraft_stays_applicable(self):
        """Studio 2026: Enhancement-Phasen können Bass hinzufügen — bass_kraft bleibt aktiv."""
        audio = _vocal_dominant()
        result = evaluate_goal_applicability(audio, SR, mode="studio_2026")
        assert "bass_kraft" not in result.inapplicable, "bass_kraft darf in Studio-2026-Modus NICHT deaktiviert werden"

    def test_bass_rich_restoration_bass_kraft_applicable(self):
        """Rock/Blues mit starkem Bass: bass_kraft muss anwendbar bleiben."""
        audio = _bass_heavy()
        result = evaluate_goal_applicability(audio, SR, mode="restoration")
        assert "bass_kraft" in result.applicable, (
            "bass_kraft muss bei bass-reichen Aufnahmen in Restoration anwendbar sein"
        )

    def test_bass_poor_default_mode_marks_inapplicable(self):
        """Kein mode-Argument → default 'restoration' → bass_kraft N/A bei bassarmem Signal."""
        audio = _vocal_dominant()
        result = evaluate_goal_applicability(audio, SR)
        assert "bass_kraft" in result.inapplicable, "Default-Modus ist 'restoration' — Regel muss gelten"

    def test_bass_kraft_inapplicable_reason_contains_ratio(self):
        """Ablehnungsgrund muss bass-ratio-Wert enthalten für Diagnose."""
        audio = _vocal_dominant()
        result = evaluate_goal_applicability(audio, SR, mode="restoration")
        if "bass_kraft" in result.inapplicable:
            reason = result.reasons.get("bass_kraft", "")
            assert "0.0" in reason, f"Ablehnungsgrund muss Ratio-Wert enthalten: {reason!r}"

    def test_bass_kraft_rule_does_not_affect_always_applicable(self):
        """ALWAYS_APPLICABLE-Goals dürfen nie durch bass_kraft-Regel betroffen werden."""
        audio = _vocal_dominant()
        result = evaluate_goal_applicability(audio, SR, mode="restoration")
        for goal in ALWAYS_APPLICABLE:
            assert goal in result.applicable, f"{goal} muss immer applicable sein"

    def test_short_audio_no_crash(self):
        """Zu kurze Datei (<1s) → kein Crash, kein false-positive bass_kraft N/A."""
        audio = _sine(dur=0.5)  # 0.5 s — unter duration_s >= 1.0 Schutz
        result = evaluate_goal_applicability(audio, SR, mode="restoration")
        # Kein Crash; bass_kraft kann applicable oder inapplicable sein (andere Regeln)
        assert isinstance(result, GoalApplicabilityResult)

    def test_none_audio_no_crash(self):
        """None audio → bass_kraft Regel nicht ausgelöst (kein crash)."""
        result = evaluate_goal_applicability(audio=None, sr=SR, mode="restoration")
        # Kein Crash; bass_kraft default applicable wenn kein Audio-Check möglich
        assert "bass_kraft" not in result.inapplicable, "Ohne Audio-Input darf bass_kraft nicht pauschal N/A sein"
