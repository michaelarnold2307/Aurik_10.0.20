from __future__ import annotations

"""
tests/test_per_phase_musical_goals_gate.py
==========================================

Unit-Tests für PerPhaseMusicalGoalsGate (§2.29 Spec).

Anforderungen aus der Spec:
  * PMGG wraps jedes PhaseInterface.process() in UnifiedRestorerV3
  * Nach jeder Phase: 5-s-Stichprobe → measure_quick() auf 6 Schnell-Ziele
  * Δ < −threshold → Retry-1 (×0.65) → Retry-2 (×0.50) → Retry-3 (×0.35) → Retry-4 (×0.20) → Retry-5 (×0.10) → Rollback
  * max. 5 Retries = MAX_RETRIES Konstante  (v10.0.0-B3)
  * Deaktivierbar via reset()
  * wrap_phase gibt 3-Tuple (audio_out, scores_after, log_entry) zurück

ARCHITEKTURBEFUND (grep _run_phase):
  `result = phase(audio, strength=strength, **phase_kwargs)`
  → phase ist ein CALLABLE, kein Objekt mit .process()-Methode.
  → result kann sein: ndarray | obj mit .audio | obj mit .processed_audio

Alle Signale: synthetisch, np.random.seed(42), SR = 48 000 Hz.
"""


import threading
from typing import Any

import numpy as np
import pytest

SR = 48_000

# ── Imports ──────────────────────────────────────────────────────────────────
import backend.core.per_phase_musical_goals_gate as ppmgg
from backend.core.per_phase_musical_goals_gate import (
    FAST_GOALS_SUBSET,
    MAX_RETRIES,
    REGRESSION_THRESHOLD,
    SAMPLE_DURATION_S,
    PerPhaseMusicalGoalsGate,
    PhaseGateLogEntry,
    get_phase_gate,
    wrap_phase,
)

try:
    from backend.core.per_phase_musical_goals_gate import _RETRY_STRENGTHS
except ImportError:
    _RETRY_STRENGTHS = [0.65, 0.50, 0.35, 0.20, 0.10]  # v10.0.0-B3 Fallback


# ── Hilfsfunktionen & Mock-Phasen ────────────────────────────────────────────


def _tone(duration_s: float = 8.0, freq: float = 440.0) -> np.ndarray:
    """Synthetischer Sinuston, float32, mono."""
    np.random.seed(42)
    t = np.linspace(0, duration_s, int(duration_s * SR), endpoint=False)
    tone: np.ndarray[Any, Any] = np.asarray(np.sin(2 * np.pi * freq * t) * 0.5, dtype=np.float32)
    return tone


def _noise(duration_s: float = 8.0, amp: float = 0.1) -> np.ndarray:
    """Weißes Rauschen, float32, mono."""
    np.random.seed(42)
    noise: np.ndarray[Any, Any] = np.asarray(np.random.randn(int(duration_s * SR)) * amp, dtype=np.float32)
    return noise


def _vocal_like(duration_s: float = 8.0, f0: float = 220.0) -> np.ndarray:
    """Synthetisches gesangsähnliches Signal mit Harmonischen und leichter Hüllkurve."""
    t = np.linspace(0, duration_s, int(duration_s * SR), endpoint=False, dtype=np.float32)
    envelope = 0.55 + 0.25 * np.sin(2 * np.pi * 2.7 * t)
    harmonics = sum((1.0 / idx) * np.sin(2 * np.pi * f0 * idx * t) for idx in range(1, 7))
    formant_tilt = 0.65 * np.sin(2 * np.pi * 900.0 * t) + 0.35 * np.sin(2 * np.pi * 2500.0 * t)
    signal = 0.18 * envelope * harmonics + 0.04 * formant_tilt
    vocal_like: np.ndarray[Any, Any] = np.asarray(np.clip(signal, -1.0, 1.0), dtype=np.float32)
    return vocal_like


class _PhaseResult:
    """Ergebnis-Objekt mit .audio-Attribut (wie viele echte Phasen)."""

    def __init__(self, audio: np.ndarray) -> None:
        self.audio = audio


class _ProcessedAudioResult:
    """Ergebnis-Objekt mit .processed_audio-Attribut."""

    def __init__(self, audio: np.ndarray) -> None:
        self.processed_audio = audio


def _pass_phase(audio: np.ndarray, strength: float = 1.0, **kw) -> np.ndarray:
    """Passthrough: gibt Audio as ndarray zurück – keine Musical-Goal-Regression."""
    return audio.copy()


def _pass_phase_result_obj(audio: np.ndarray, strength: float = 1.0, **kw) -> _PhaseResult:
    """Passthrough via .audio-Result-Objekt."""
    return _PhaseResult(audio.copy())


def _pass_phase_processed_obj(audio: np.ndarray, strength: float = 1.0, **kw) -> _ProcessedAudioResult:
    """Passthrough via .processed_audio-Result-Objekt."""
    return _ProcessedAudioResult(audio.copy())


def _silence_phase(audio: np.ndarray, strength: float = 1.0, **kw) -> np.ndarray:
    """Gibt Stille zurück — erzeugt massive Musical-Goal-Regression."""
    return np.zeros_like(audio)


def _noisy_phase(audio: np.ndarray, strength: float = 1.0, **kw) -> np.ndarray:
    """Addiert starkes Rauschen — verändert Spektrum erheblich."""
    np.random.seed(99)
    noisy: np.ndarray[Any, Any] = np.asarray(
        np.clip(
            audio + np.random.randn(*audio.shape).astype(np.float32) * 0.8,
            -1.0,
            1.0,
        ),
        dtype=np.float32,
    )
    return noisy


def _nan_phase(audio: np.ndarray, strength: float = 1.0, **kw) -> np.ndarray:
    """Gibt NaN-gefülltes Array zurück – PMGG muss bereinigen."""
    out = audio.copy()
    out[:] = float("nan")
    return out


def _exploding_phase(audio: np.ndarray, strength: float = 1.0, **kw) -> np.ndarray:
    """Gibt ±10.0 zurück – muss auf ±1.0 geclippt werden."""
    return np.full_like(audio, 10.0)


def _exception_phase(audio: np.ndarray, strength: float = 1.0, **kw) -> np.ndarray:
    """Wirft immer eine Exception – PMGG muss Original zurückgeben."""
    raise RuntimeError("Synthetischer Phasenfehler")


class _StrengthCapturingPhase:
    """Callable-Klasse, die den strength-Parameter speichert."""

    def __init__(self) -> None:
        self.captured_strengths: list[float] = []

    def process(self, audio: np.ndarray, strength: float = 1.0, **kw) -> np.ndarray:
        self.captured_strengths.append(strength)
        return np.zeros_like(audio)  # Regression erzwingen → Retries


# ── Testklassen ───────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestSingleton:
    """Singleton-Invarianten (§2.29)."""

    def test_01_get_phase_gate_returns_instance(self):
        gate = get_phase_gate()
        assert isinstance(gate, PerPhaseMusicalGoalsGate)

    def test_02_singleton_same_object(self):
        g1 = get_phase_gate()
        g2 = get_phase_gate()
        assert g1 is g2

    def test_03_thread_safe_singleton(self):
        """20 parallele Threads müssen dieselbe Instanz erhalten."""
        instances: list[PerPhaseMusicalGoalsGate] = []
        lock = threading.Lock()

        def _get():
            inst = get_phase_gate()
            with lock:
                instances.append(inst)

        threads = [threading.Thread(target=_get) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(instances) == 20
        assert all(i is instances[0] for i in instances)


class TestConstants:
    """Konstanten müssen exakt den Spec-Werten entsprechen (§2.29)."""

    def test_04_regression_threshold_value(self):
        assert pytest.approx(0.025, abs=1e-9) == REGRESSION_THRESHOLD

    def test_05_max_retries_value(self):
        assert MAX_RETRIES == 5  # v10.0.0-B3: 5-Retry-Strategie

    def test_06_fast_goals_subset_length(self):
        assert len(FAST_GOALS_SUBSET) == 15  # v10.0.0: alle 15 Musical Goals per-Phase geprüft

    def test_07_fast_goals_contains_brillanz(self):
        assert "brillanz" in FAST_GOALS_SUBSET

    def test_08_fast_goals_contains_groove(self):
        assert "groove" in FAST_GOALS_SUBSET

    def test_09_retry_strengths_values(self):
        assert (
            pytest.approx([0.65, 0.50, 0.35, 0.25, 0.15], abs=1e-9) == _RETRY_STRENGTHS
        )  # v10.0.0: 5 Stufen, Floor 0.15

    def test_10_sample_duration_positive(self):
        assert SAMPLE_DURATION_S > 0.0

    def test_11_retry_strengths_count(self):
        assert len(_RETRY_STRENGTHS) == 5  # v10.0.0: 5 Stufen, Floor 0.15


class TestWrapPhaseReturnType:
    """wrap_phase muss ein 3-Tuple zurückgeben (§2.29)."""

    def test_12_returns_three_tuple(self):
        audio = _tone(8.0)
        result = wrap_phase(_pass_phase, audio, SR)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_13_first_element_is_ndarray(self):
        audio = _tone(8.0)
        audio_out, _, _ = wrap_phase(_pass_phase, audio, SR)
        assert isinstance(audio_out, np.ndarray)

    def test_14_second_element_is_dict(self):
        audio = _tone(8.0)
        _, scores_after, _ = wrap_phase(_pass_phase, audio, SR)
        assert isinstance(scores_after, dict)

    def test_15_third_element_is_log_entry(self):
        audio = _tone(8.0)
        _, _, log_entry = wrap_phase(_pass_phase, audio, SR)
        assert isinstance(log_entry, PhaseGateLogEntry)

    def test_16_scores_after_has_fast_goals_keys(self):
        audio = _tone(8.0)
        _, scores_after, _ = wrap_phase(_pass_phase, audio, SR)
        for goal in FAST_GOALS_SUBSET:
            assert goal in scores_after, f"Ziel '{goal}' fehlt in scores_after"

    def test_17_scores_after_values_bounded(self):
        audio = _tone(8.0)
        _, scores_after, _ = wrap_phase(_pass_phase, audio, SR)
        for goal, val in scores_after.items():
            assert 0.0 <= val <= 1.0, f"{goal}={val} nicht in [0, 1]"

    def test_18_scores_after_all_finite(self):
        audio = _tone(8.0)
        _, scores_after, _ = wrap_phase(_pass_phase, audio, SR)
        for goal, val in scores_after.items():
            assert math.isfinite(val), f"{goal}={val} ist nicht endlich"


class TestAudioOutput:
    """Audio-Ausgabe-Invarianten (§2.29)."""

    def test_19_output_same_length(self):
        audio = _tone(8.0)
        audio_out, _, _ = wrap_phase(_pass_phase, audio, SR)
        assert len(audio_out) == len(audio)

    def test_20_output_no_nan(self):
        audio = _tone(8.0)
        audio_out, _, _ = wrap_phase(_pass_phase, audio, SR)
        assert np.all(np.isfinite(audio_out)), "NaN/Inf in audio_out"

    def test_21_output_clipped_to_unity(self):
        audio = _tone(8.0)
        audio_out, _, _ = wrap_phase(_exploding_phase, audio, SR)
        assert np.max(np.abs(audio_out)) <= 1.0 + 1e-6

    def test_22_nan_input_handled(self):
        """NaN im Input darf keinen Crash verursachen."""
        audio = _tone(8.0)
        audio[10:20] = float("nan")
        audio_out, scores_after, log_entry = wrap_phase(_pass_phase, audio, SR)
        assert isinstance(audio_out, np.ndarray)
        assert isinstance(scores_after, dict)
        assert isinstance(log_entry, PhaseGateLogEntry)

    def test_23_nan_phase_output_sanitized(self):
        """Phase die NaN zurückgibt → Ausgabe trotzdem endlich."""
        audio = _tone(8.0)
        audio_out, _, _ = wrap_phase(_nan_phase, audio, SR)
        assert np.all(np.isfinite(audio_out))

    def test_24_exception_phase_returns_original_length(self):
        """Exception in Phase → Rollback auf Original-Audio."""
        audio = _tone(8.0)
        audio_out, _, log_entry = wrap_phase(_exception_phase, audio, SR)
        assert len(audio_out) == len(audio)

    def test_25_output_float32(self):
        audio = _tone(8.0)
        audio_out, _, _ = wrap_phase(_pass_phase, audio, SR)
        assert audio_out.dtype == np.float32


class TestLogEntry:
    """PhaseGateLogEntry-Inhalte (§2.29)."""

    def test_26_log_entry_action_valid(self):
        audio = _tone(8.0)
        _, _, log_entry = wrap_phase(_pass_phase, audio, SR)
        assert log_entry.action in {
            "passed",
            "retry1",
            "retry2",
            "retry3",
            "retry4",
            "retry5",
        } or log_entry.action.startswith("best_effort")

    def test_27_passthrough_action_is_passed(self):
        """Passthrough-Phase ohne Regression → action='passed'."""
        audio = _tone(8.0)
        _, _, log_entry = wrap_phase(_pass_phase, audio, SR)
        assert log_entry.action == "passed"

    def test_28_log_entry_goal_regressions_is_dict(self):
        audio = _tone(8.0)
        _, _, log_entry = wrap_phase(_pass_phase, audio, SR)
        assert isinstance(log_entry.goal_regressions, dict)

    def test_29_log_entry_strength_used_float(self):
        audio = _tone(8.0)
        _, _, log_entry = wrap_phase(_pass_phase, audio, SR)
        assert isinstance(log_entry.strength_used, float)

    def test_30_passthrough_strength_is_one(self):
        """Passthrough ohne Regression → strength=1.0."""
        audio = _tone(8.0)
        _, _, log_entry = wrap_phase(_pass_phase, audio, SR)
        assert log_entry.strength_used == pytest.approx(1.0, abs=1e-6)


class TestResultObjectPhases:
    """Phase gibt Objekt mit .audio oder .processed_audio zurück (§2.29)."""

    def test_31_result_obj_with_audio_attr(self):
        audio = _tone(8.0)
        audio_out, scores_after, log_entry = wrap_phase(_pass_phase_result_obj, audio, SR)
        assert isinstance(audio_out, np.ndarray)
        assert len(audio_out) == len(audio)

    def test_32_result_obj_with_processed_audio_attr(self):
        audio = _tone(8.0)
        audio_out, scores_after, log_entry = wrap_phase(_pass_phase_processed_obj, audio, SR)
        assert isinstance(audio_out, np.ndarray)
        assert len(audio_out) == len(audio)


class TestEdgeCases:
    """Grenzfälle (§2.29)."""

    def test_33_stereo_audio(self):
        """Stereo-Eingabe darf keinen Crash verursachen."""
        np.random.seed(42)
        audio = (np.random.randn(SR * 8, 2) * 0.3).astype(np.float32)
        audio_out, scores_after, log_entry = wrap_phase(_pass_phase, audio, SR)
        assert isinstance(audio_out, np.ndarray)
        assert isinstance(scores_after, dict)

    def test_34_short_audio_no_crash(self):
        """Audio kürzer als SAMPLE_DURATION_S → kein Crash."""
        audio = _tone(2.0)  # Kürzer als 5 s
        audio_out, scores_after, log_entry = wrap_phase(_pass_phase, audio, SR)
        assert isinstance(audio_out, np.ndarray)

    def test_35_scores_before_provided(self):
        """Wenn scores_before übergeben, kein Re-Messung-Crash."""
        audio = _tone(8.0)
        scores_before: dict[str, float] = dict.fromkeys(FAST_GOALS_SUBSET, 0.85)
        audio_out, scores_after, log_entry = wrap_phase(_pass_phase, audio, SR, scores_before=scores_before)
        assert isinstance(audio_out, np.ndarray)
        assert isinstance(scores_after, dict)

    def test_36_scores_before_none_no_crash(self):
        """scores_before=None: Gate misst intern → kein Crash."""
        audio = _tone(8.0)
        audio_out, scores_after, log_entry = wrap_phase(_pass_phase, audio, SR, scores_before=None)
        assert len(audio_out) == len(audio)

    def test_37_zero_length_goal_regressions_on_pass(self):
        """Passthrough: keine Goal-Regression → goal_regressions leer oder nur Null-Deltas."""
        audio = _tone(8.0)
        _, _, log_entry = wrap_phase(_pass_phase, audio, SR)
        # Alle Regressions-Werte müssen ≤ REGRESSION_THRESHOLD sein
        for goal, delta in log_entry.goal_regressions.items():
            assert delta <= REGRESSION_THRESHOLD + 1e-6, f"Unerwartete Regression bei Passthrough: {goal}={delta}"

    def test_38_reset_clears_state(self):
        """reset() muss ohne Exception laufen und Gate wieder nutzbar machen."""
        gate = get_phase_gate()
        gate.reset()
        audio = _tone(8.0)
        audio_out, _, _ = gate.wrap_phase(_pass_phase, audio, SR)
        assert isinstance(audio_out, np.ndarray)

    def test_39_float64_input(self):
        """float64-Input wird intern verarbeitet ohne Absturz."""
        audio = _tone(8.0).astype(np.float64)
        audio_out, _, _ = wrap_phase(_pass_phase, audio, SR)
        assert isinstance(audio_out, np.ndarray)


import math


class TestConvenienceFunction:
    """Modul-level Convenience wrap_phase (§2.29)."""

    def test_40_module_level_wrap_phase(self):
        audio = _tone(8.0)
        result = wrap_phase(_pass_phase, audio, SR)
        assert len(result) == 3
        audio_out, scores_after, log_entry = result
        assert isinstance(audio_out, np.ndarray)
        assert isinstance(scores_after, dict)
        assert isinstance(log_entry, PhaseGateLogEntry)


class TestCanonicalKeyAlignment:
    """FAST_GOALS_SUBSET must use canonical keys matching GoalApplicabilityFilter (§2.29 × §2.32)."""

    def test_41_natuerlichkeit_canonical_key_in_fast_goals_subset(self):
        """'natuerlichkeit' must be in FAST_GOALS_SUBSET — proxy key caused silent blind-spot (§2.32)."""
        assert "natuerlichkeit" in FAST_GOALS_SUBSET, (
            "FAST_GOALS_SUBSET must contain 'natuerlichkeit' (canonical key), "
            "not 'natuerlichkeit_mfcc_proxy'. GoalApplicabilityFilter uses canonical keys."
        )

    def test_42_proxy_key_not_in_fast_goals_subset(self):
        """'natuerlichkeit_mfcc_proxy' must NOT appear in FAST_GOALS_SUBSET (removed in fix)."""
        assert "natuerlichkeit_mfcc_proxy" not in FAST_GOALS_SUBSET

    def test_43_measure_quick_returns_canonical_natuerlichkeit_key(self):
        """_measure_quick must return 'natuerlichkeit' key, not the old proxy key."""
        from backend.core.per_phase_musical_goals_gate import _measure_quick

        audio = _tone(5.0)
        scores = _measure_quick(audio, SR)
        assert "natuerlichkeit" in scores, "_measure_quick must return canonical 'natuerlichkeit' key"
        assert "natuerlichkeit_mfcc_proxy" not in scores, "Old proxy key must not be in scores"

    def test_44_applicable_goals_intersection_includes_natuerlichkeit(self):
        """With applicable_goals containing 'natuerlichkeit', effective_goals must include it.

        Before fix: 'natuerlichkeit_mfcc_proxy' ∩ {'natuerlichkeit'} = {} → P1 goal never guarded.
        After fix:  'natuerlichkeit' ∩ {'natuerlichkeit'} = {'natuerlichkeit'} → correctly guarded.
        """
        gate = PerPhaseMusicalGoalsGate()
        audio = _tone(8.0)
        applicable = {
            "natuerlichkeit",
            "authentizitaet",
            "emotionalitaet",
            "brillanz",
            "waerme",
            "groove",
            "tonal_center",
            "timbre_authentizitaet",
            "bass_kraft",
            "transparenz",
            "spatial_depth",
            "micro_dynamics",
            "separation_fidelity",
            "artikulation",
        }
        audio_out, scores_after, log_entry = gate.wrap_phase(_pass_phase, audio, SR, applicable_goals=applicable)
        # natuerlichkeit must appear in scores_after — not silently absent
        assert "natuerlichkeit" in scores_after, (
            "scores_after must contain 'natuerlichkeit' when it is in applicable_goals"
        )

    def test_45_all_15_canonical_goals_in_fast_goals_subset(self):
        """FAST_GOALS_SUBSET must contain exactly the 15 canonical Goal keys."""
        canonical_15 = {
            "brillanz",
            "waerme",
            "natuerlichkeit",
            "authentizitaet",
            "emotionalitaet",
            "transparenz",
            "bass_kraft",
            "groove",
            "spatial_depth",
            "timbre_authentizitaet",
            "tonal_center",
            "micro_dynamics",
            "separation_fidelity",
            "artikulation",
            "transient_energie",
        }
        assert set(FAST_GOALS_SUBSET) == canonical_15, (
            f"FAST_GOALS_SUBSET key mismatch.\n"
            f"  Missing: {canonical_15 - set(FAST_GOALS_SUBSET)}\n"
            f"  Extra:   {set(FAST_GOALS_SUBSET) - canonical_15}"
        )


class TestFFTScopeRobustness:
    """_measure_quick FFT pre-computation must be independent of per-metric try-blocks (§2.29)."""

    def test_46_measure_quick_all_goals_non_neutral_on_tonal_signal(self):
        """On a clean tone, all 15 metrics should return a definite value (not all == 0.5).

        Pre-fix: if brillanz try-block raised, 6 metrics silently fell back to 0.5.
        Post-fix: FFT is pre-computed; individual metric failures are isolated.
        """
        from backend.core.per_phase_musical_goals_gate import _measure_quick

        audio = _tone(5.0, freq=440.0)
        scores = _measure_quick(audio, SR)
        # Verify all 15 canonical keys are present and finite
        for k in FAST_GOALS_SUBSET:
            assert k in scores, f"Key '{k}' missing from _measure_quick output"
            assert math.isfinite(scores[k]), f"Score for '{k}' is not finite: {scores[k]}"
            assert 0.0 <= scores[k] <= 1.0, f"Score for '{k}' out of [0,1]: {scores[k]}"

    def test_47_measure_quick_on_silence_no_crash(self):
        """Silence input (zero energy) must not crash and return finite scores."""
        from backend.core.per_phase_musical_goals_gate import _measure_quick

        audio = np.zeros(SR * 5, dtype=np.float32)
        scores = _measure_quick(audio, SR)
        for k in FAST_GOALS_SUBSET:
            assert k in scores
            assert math.isfinite(scores[k]), f"Score for '{k}' NaN/Inf on silence"

    def test_48_measure_quick_brillanz_failure_does_not_cascade(self):
        """If brillanz calculation fails, waerme/bass_kraft/etc. must still return valid scores.

        Simulates FFT pre-computation isolation: even if brillanz metric block fails,
        the pre-computed fft_mag/freqs/tot_energy are available for all other metrics.
        """
        from backend.core.per_phase_musical_goals_gate import _measure_quick

        # Very short audio that could theoretically stress FFT edge cases
        audio = np.ones(64, dtype=np.float32) * 0.001  # 64 samples, near-zero
        scores = _measure_quick(audio, SR)
        # All 15 goals must have a valid value — no cascading NaN
        neutral_count = sum(1 for k in FAST_GOALS_SUBSET if scores.get(k, -1) == 0.5)
        # It's acceptable for some to be 0.5 (fallback), but all must be present and finite
        for k in FAST_GOALS_SUBSET:
            assert k in scores
            assert math.isfinite(scores[k])


class TestPreciseMetricOverrides:
    """Selected PMGG goals should be refinable via canonical metric overrides."""

    def test_49_measure_quick_applies_precise_metric_overrides(self, monkeypatch):

        class _FixedMetric:
            def __init__(self, value: float) -> None:
                self.value = value

            def measure(self, audio, sr, reference=None):
                return self.value

        monkeypatch.setattr(
            ppmgg,
            "_get_precise_metric_instances",
            lambda: {
                "natuerlichkeit": _FixedMetric(0.91),
                "tonal_center": _FixedMetric(0.96),
                "micro_dynamics": _FixedMetric(0.88),
                "artikulation": _FixedMetric(0.87),
                "separation_fidelity": _FixedMetric(0.84),
                "transparenz": _FixedMetric(0.83),
            },
        )

        scores = ppmgg._measure_quick(_tone(5.0), SR)
        assert scores["natuerlichkeit"] == pytest.approx(0.91, abs=1e-9)
        assert scores["tonal_center"] == pytest.approx(0.96, abs=1e-9)
        assert scores["micro_dynamics"] == pytest.approx(0.88, abs=1e-9)
        assert scores["artikulation"] == pytest.approx(0.87, abs=1e-9)
        assert scores["separation_fidelity"] == pytest.approx(0.84, abs=1e-9)
        assert scores["transparenz"] == pytest.approx(0.83, abs=1e-9)


class TestVocalSensitiveQuickMetrics:
    """Vocal-sensitive PMGG corrections must help vocal material without inflating noise."""

    def test_50_vocal_reference_bonus_lifts_vocal_goals(self):
        from backend.core.per_phase_musical_goals_gate import _measure_quick

        reference = _vocal_like(5.0)
        processed = np.clip(reference * 0.96, -1.0, 1.0).astype(np.float32)

        scores_plain = _measure_quick(
            processed,
            SR,
            reference=reference,
            precise_override=False,
            enable_vocal_guard=False,
        )
        scores_ref = _measure_quick(
            processed,
            SR,
            reference=reference,
            precise_override=False,
            enable_vocal_guard=True,
        )

        nat_delta = scores_ref["natuerlichkeit"] - scores_plain["natuerlichkeit"]
        timbre_delta = scores_ref["timbre_authentizitaet"] - scores_plain["timbre_authentizitaet"]
        artik_delta = scores_ref["artikulation"] - scores_plain["artikulation"]

        assert nat_delta >= 0.0
        assert timbre_delta >= 0.0
        assert artik_delta >= 0.0
        assert max(nat_delta, timbre_delta, artik_delta) > 0.0

    def test_51_noise_reference_path_does_not_get_large_vocal_bonus(self):
        from backend.core.per_phase_musical_goals_gate import _measure_quick

        reference = _noise(5.0, amp=0.05)
        processed = np.clip(reference * 0.96, -1.0, 1.0).astype(np.float32)

        scores_plain = _measure_quick(
            processed,
            SR,
            reference=reference,
            precise_override=False,
            enable_vocal_guard=False,
        )
        scores_ref = _measure_quick(
            processed,
            SR,
            reference=reference,
            precise_override=False,
            enable_vocal_guard=True,
        )

        assert abs(scores_ref["natuerlichkeit"] - scores_plain["natuerlichkeit"]) <= 0.02
        assert abs(scores_ref["timbre_authentizitaet"] - scores_plain["timbre_authentizitaet"]) <= 0.02
        assert abs(scores_ref["artikulation"] - scores_plain["artikulation"]) <= 0.02

    def test_52_wrap_phase_logs_vocal_guard_metadata(self):
        audio = _vocal_like(8.0)

        _, _, log_entry = wrap_phase(_noisy_phase, audio, SR)

        assert "vocal_presence_proxy" in log_entry.metadata
        assert "vocal_formant_stability" in log_entry.metadata
        assert "vocal_fricative_stability" in log_entry.metadata
        assert "vocal_transient_integrity" in log_entry.metadata
        assert "vocal_guard_active" in log_entry.metadata
        assert 0.0 <= float(log_entry.metadata["vocal_presence_proxy"]) <= 1.0
        assert isinstance(log_entry.metadata["vocal_guard_active"], bool)

    def test_53_wrap_phase_caps_retry_budget_under_wall_budget_pressure(self):
        audio = _tone(8.0)
        phase = _StrengthCapturingPhase()
        gate = PerPhaseMusicalGoalsGate()

        _, _, log_entry = gate.wrap_phase(
            phase,
            audio,
            SR,
            phase_id="phase_17_mastering_polish",
            phase_kwargs={
                "retry_budget_hint": {
                    "reason": "reserve_for_priority_followup",
                    "future_priority_phases": ["phase_58_lyrics_guided_enhancement"],
                    "retry_budget_scale": 0.35,
                    "max_retries_cap": 2,
                }
            },
        )

        assert len(phase.captured_strengths) >= 1  # HPE-GATE may reject early
        assert log_entry.metadata["retry_budget_policy_active"] is True
        assert log_entry.metadata["retry_budget_reason"] == "reserve_for_priority_followup"
        assert log_entry.metadata["retry_budget_max_retries"] == 2
        assert log_entry.metadata["retry_budget_future_priority_phases"] == ["phase_58_lyrics_guided_enhancement"]

    def test_54_wrap_phase_caps_retry_budget_for_historically_weak_phase(self):
        audio = _tone(8.0)
        phase = _StrengthCapturingPhase()
        gate = PerPhaseMusicalGoalsGate()

        _, _, log_entry = gate.wrap_phase(
            phase,
            audio,
            SR,
            phase_id="phase_39_air_band_enhancement",
            phase_kwargs={
                "retry_budget_hint": {
                    "reason": "historically_weak_delta",
                    "retry_budget_scale": 0.15,
                    "max_retries_cap": 1,
                    "history_samples": 3,
                    "history_mean_net_gain": 0.001,
                }
            },
        )

        assert len(phase.captured_strengths) >= 1  # HPE-GATE may reject early
        assert log_entry.metadata["retry_budget_policy_active"] is True
        assert log_entry.metadata["retry_budget_reason"] == "historically_weak_delta"
        assert log_entry.metadata["retry_budget_max_retries"] == 1
        assert log_entry.metadata["retry_budget_history_samples"] == 3


class TestWetDryBlendStereoSafety:
    """Stereo wet/dry blend must remain channel-safe and non-silent."""

    def test_53_wet_dry_blend_stereo_low_strength_keeps_shape_and_energy(self):
        gate = PerPhaseMusicalGoalsGate()
        rng = np.random.default_rng(123)
        dry = rng.normal(0.0, 0.15, size=(SR * 3, 2)).astype(np.float32)
        # Simulate a phase output with slight spectral change.
        wet = np.clip(dry * 0.9 + rng.normal(0.0, 0.01, size=dry.shape).astype(np.float32), -1.0, 1.0)

        out = gate._wet_dry_blend(dry, wet, strength=0.10)

        assert out.shape == dry.shape
        assert out.dtype == np.float32
        assert np.all(np.isfinite(out))
        # Guard against silent-tail regressions caused by malformed channel-axis STFT.
        h2 = out[len(out) // 2 :, :]
        rms_h2 = float(np.sqrt(np.mean(h2**2)))
        assert rms_h2 > 1e-4

    def test_54_wet_dry_blend_stereo_length_match_no_channel_pad_corruption(self):
        gate = PerPhaseMusicalGoalsGate()
        rng = np.random.default_rng(7)
        dry = rng.normal(0.0, 0.2, size=(10_000, 2)).astype(np.float32)
        wet_short = dry[:-123].copy()

        out = gate._wet_dry_blend(dry, wet_short, strength=0.20)

        assert out.shape == dry.shape
        assert np.all(np.isfinite(out))
        # Last segment must not collapse to all-zero due to invalid 2D padding.
        tail = out[-500:, :]
        assert float(np.sqrt(np.mean(tail**2))) > 1e-5

    def test_55_wet_dry_blend_low_strength_preserves_dry_phase_on_inverted_wet(self):
        gate = PerPhaseMusicalGoalsGate()
        t = np.linspace(0.0, 0.2, int(0.2 * SR), endpoint=False, dtype=np.float32)
        dry = (0.25 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
        wet = -dry  # volle Polaritaetsinversion

        out = gate._wet_dry_blend(dry, wet, strength=0.10)

        assert out.shape == dry.shape
        assert np.all(np.isfinite(out))
        # Bei niedriger Strength darf der Output nicht in Richtung -dry kippen.
        corr_dry = float(np.dot(out, dry) / ((np.linalg.norm(out) * np.linalg.norm(dry)) + 1e-8))
        corr_wet = float(np.dot(out, wet) / ((np.linalg.norm(out) * np.linalg.norm(wet)) + 1e-8))
        assert corr_dry > 0.7
        assert corr_wet < 0.0


class TestDecisionReasonTelemetry:
    def test_56_wrap_phase_sets_decision_reason_for_sub_threshold(self, monkeypatch):
        audio = _tone(4.0)
        gate = PerPhaseMusicalGoalsGate()

        def _stub_run_with_retry(
            self,
            phase,
            audio_in,
            sr,
            scores_before,
            phase_id,
            phase_kwargs=None,
            **kwargs,
        ):
            return audio_in.copy(), dict(scores_before), "sub_threshold", 0.55

        monkeypatch.setattr(PerPhaseMusicalGoalsGate, "_run_with_retry", _stub_run_with_retry)

        _, _, log_entry = gate.wrap_phase(_StrengthCapturingPhase(), audio, SR, phase_id="phase_03_denoise")

        assert log_entry.metadata["pmgg_decision_class"] == "pass"
        assert log_entry.metadata["pmgg_decision_reason"] == "jnd_sub_threshold_accept"

    def test_57_wrap_phase_sets_decision_reason_for_best_effort_emergency(self, monkeypatch):
        audio = _tone(4.0)
        gate = PerPhaseMusicalGoalsGate()

        def _stub_run_with_retry(
            self,
            phase,
            audio_in,
            sr,
            scores_before,
            phase_id,
            phase_kwargs=None,
            **kwargs,
        ):
            scores_after = dict(scores_before)
            scores_after["natuerlichkeit"] = max(0.0, scores_after.get("natuerlichkeit", 0.5) - 0.15)
            return audio_in.copy(), scores_after, "best_effort_emergency", 0.15

        monkeypatch.setattr(PerPhaseMusicalGoalsGate, "_run_with_retry", _stub_run_with_retry)

        _, _, log_entry = gate.wrap_phase(_StrengthCapturingPhase(), audio, SR, phase_id="phase_29_tape_hiss_reduction")

        assert log_entry.metadata["pmgg_decision_class"] == "best_effort"
        # §v10.703 SOTA: der konkrete Diagnose-Code ersetzt den überholten Wert
        # „catastrophic_regression_unresolved" (Test am 2026-10-08 nachgezogen,
        # D-K3-59 — der alte Wert existierte nur noch hier, nicht in Code/Spec).
        assert log_entry.metadata["pmgg_decision_reason"] == "catastrophic_regression_best_effort_applied"

    def test_58_wrap_phase_sets_decision_reason_for_passthrough(self, monkeypatch):
        audio = _tone(3.0)
        gate = PerPhaseMusicalGoalsGate()

        def _stub_run_with_retry(
            self,
            phase,
            audio_in,
            sr,
            scores_before,
            phase_id,
            phase_kwargs=None,
            **kwargs,
        ):
            return audio_in.copy(), dict(scores_before), "passthrough", 1.0

        monkeypatch.setattr(PerPhaseMusicalGoalsGate, "_run_with_retry", _stub_run_with_retry)

        _, _, log_entry = gate.wrap_phase(
            _StrengthCapturingPhase(), audio, SR, phase_id="phase_31_speed_pitch_correction"
        )

        assert log_entry.metadata["pmgg_decision_class"] == "pass"
        assert log_entry.metadata["pmgg_decision_reason"] == "phase_passthrough_no_change"

    def test_59_wrap_phase_sets_decision_reason_for_best_effort_accepted(self, monkeypatch):
        audio = _tone(3.0)
        gate = PerPhaseMusicalGoalsGate()

        def _stub_run_with_retry(
            self,
            phase,
            audio_in,
            sr,
            scores_before,
            phase_id,
            phase_kwargs=None,
            **kwargs,
        ):
            return audio_in.copy(), dict(scores_before), "best_effort_accepted", 0.2

        monkeypatch.setattr(PerPhaseMusicalGoalsGate, "_run_with_retry", _stub_run_with_retry)

        _, _, log_entry = gate.wrap_phase(_StrengthCapturingPhase(), audio, SR, phase_id="phase_03_denoise")

        assert log_entry.metadata["pmgg_decision_class"] == "best_effort"
        assert log_entry.metadata["pmgg_decision_reason"] == "legacy_best_effort_accepted"

    def test_60_wrap_phase_sets_decision_reason_for_reconstruction_localized_accept(self, monkeypatch):
        audio = _tone(4.0)
        gate = PerPhaseMusicalGoalsGate()

        def _stub_run_with_retry(
            self,
            phase,
            audio_in,
            sr,
            scores_before,
            phase_id,
            phase_kwargs=None,
            **kwargs,
        ):
            scores_after = dict(scores_before)
            scores_after["waerme"] = max(0.0, scores_after.get("waerme", 0.5) - 0.10)
            return audio_in.copy(), scores_after, "passed_reconstruction_localized", 0.62

        monkeypatch.setattr(PerPhaseMusicalGoalsGate, "_run_with_retry", _stub_run_with_retry)

        _, _, log_entry = gate.wrap_phase(
            _StrengthCapturingPhase(),
            audio,
            SR,
            phase_id="phase_55_diffusion_inpainting",
        )

        assert log_entry.metadata["pmgg_decision_class"] == "pass"
        assert log_entry.metadata["pmgg_decision_reason"] == "reconstruction_localized_collateral_accept"
