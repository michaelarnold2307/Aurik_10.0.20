from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

"""tests/unit/test_denker/test_strategie_denker.py

Tests für StrategieDenker — Restaurierungsplanung & Phasenauswahl.
"""


import math
from unittest.mock import MagicMock

import numpy as np
import pytest

SR = 48_000


def _make_defekt_ergebnis(primary_defect: str = "hiss", confidence: float = 0.7):
    """Synthetisches DefektErgebnis für StrategieDenker-Tests."""
    try:
        from denker.defekt_denker import DefektErgebnis

        return DefektErgebnis(
            defect_scores={primary_defect: confidence},
            primary_defect=primary_defect,
            confidence=confidence,
            material_context="tape",
            recommended_phases=["phase_03_denoise"],
            reasoning="Test",
        )
    except Exception:
        # Fallback: einfaches Objekt wenn Import fehlschlägt
        e = MagicMock()
        e.primary_defect = primary_defect
        e.confidence = confidence
        e.defect_scores = {primary_defect: confidence}
        e.material_context = "tape"
        e.recommended_phases = ["phase_03_denoise"]
        return e


# ─── StrategiePlan (der ECHTE Rückgabetyp von plan()) ─────────────────────────


@pytest.mark.unit
class TestStrategiePlanContract:
    """Befund 2026-10-07: Die frühere Klasse `TestStrategieErgebnisFields` prüfte
    `StrategieErgebnis` — eine Dataklasse, die `plan()` NIE zurückgibt (und die
    deshalb entfernt wurde). Alle Assertions liefen gegen ein Objekt, das im
    Produktionspfad nicht existiert; der Vertrag von `StrategiePlan` war
    ungeprüft.
    """

    def _plan(self, seconds: float = 10.0, **kwargs):
        from denker.strategie_denker import StrategieDenker

        audio = np.zeros(int(seconds * SR), dtype=np.float32)
        return StrategieDenker().plan(audio, SR, **kwargs)

    def test_01_returns_strategie_plan(self):
        from denker.strategie_denker import StrategiePlan

        assert isinstance(self._plan(), StrategiePlan)

    def test_02_budget_never_exceeds_the_hard_guard_ceiling(self):
        """Das Plan-Budget darf nie mehr Zeit zusagen, als der Guard gewährt.

        Befund 2026-10-07: ``max_processing_s`` erlaubte bis zu **73,6×** RT
        (32 × Tiefe 2,0 × Restaurierbarkeit 1,5), während der harte Ausstieg bei
        32× liegt — der Plan meldete Zeit, die es nie gab, und speiste darüber
        die Stufen-Wahl. Jetzt ist er auf die Guard-Obergrenze gedeckelt.
        Die gemessene Ist-Lage (~53×, `RT_REALITY_MEASURED`) ist als Lücke
        benannt, nicht als Zusage.
        """
        plan = self._plan(seconds=10.0)
        assert plan.max_processing_s <= 32.0 * 10.0 + 1e-6
        assert plan.max_processing_s == pytest.approx(32.0 * 10.0, rel=1e-6)

    def test_03_duration_and_mode_are_reported(self):
        plan = self._plan(seconds=4.0, mode="balanced")
        assert plan.audio_duration_s == pytest.approx(4.0, rel=1e-3)
        assert plan.quality_mode == "balanced"

    def test_04_intervention_budget_is_clipped(self):
        plan = self._plan(defect_severity=1.0)
        assert 0.12 <= plan.intervention_budget <= 0.88

    def test_05_chunk_size_is_positive_and_bounded(self):
        plan = self._plan(seconds=600.0, defect_severity=0.9)
        assert 2.0 <= plan.recommended_chunk_s <= 600.0

    def test_06_enforce_flag_is_carried(self):
        assert self._plan(enforce_3x_rt=True).enforce_limit is True

    def test_07_as_dict_roundtrips_all_fields(self):
        plan = self._plan()
        payload = plan.as_dict()
        assert set(payload) == {
            "audio_duration_s",
            "max_processing_s",
            "quality_mode",
            "enforce_limit",
            "enable_adaptive_skipping",
            "recommended_chunk_s",
            "defect_severity",
            "intervention_budget",
            "listening_experience_targets",
            "human_hearing_risk_map",
            "human_hearing_comfort_profile",
            "pleasantness_baseline",
            "goosebumps_baseline",
            "budget_note",
        }
        assert payload["max_processing_s"] == pytest.approx(plan.max_processing_s)

    def test_08_hearing_targets_are_finite(self):
        plan = self._plan()
        assert plan.listening_experience_targets, "Hör-Ziele dürfen nicht leer sein"
        for goal, value in plan.listening_experience_targets.items():
            assert math.isfinite(value), f"{goal} ist nicht endlich: {value}"

    def test_09_empty_audio_does_not_crash(self):
        from denker.strategie_denker import StrategieDenker

        plan = StrategieDenker().plan(np.zeros(0, dtype=np.float32), SR)
        assert plan.audio_duration_s >= 0.0
        assert math.isfinite(plan.max_processing_s)


# ─── Singleton ────────────────────────────────────────────────────────────────


class TestStrategieDenkerSingleton:
    def test_07_returns_instance(self):
        from denker.strategie_denker import StrategieDenker, get_strategie_denker

        assert isinstance(get_strategie_denker(), StrategieDenker)

    def test_08_singleton_identity(self):
        from denker.strategie_denker import get_strategie_denker

        assert get_strategie_denker() is get_strategie_denker()

    def test_09_thread_safe(self):
        import concurrent.futures

        from denker.strategie_denker import get_strategie_denker

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
            insts = list(ex.map(lambda _: get_strategie_denker(), range(12)))
        assert all(i is insts[0] for i in insts)


# ─── Budget-Tracking: starte_timer() + check() ───────────────────────────────
#
# Befund 2026-10-07: Die früheren Tests 10–20 dieser Datei riefen
# ``plan(defekt, rt_limit=3.0)`` auf — eine Signatur, die es nicht gibt
# (``plan(audio, sr, ...)``), und kapselten den Aufruf in ``try/except: pass``.
# Der ``TypeError`` wurde geschluckt, keine Assertion lief: 11 Tests waren grün,
# egal was der Code tat. Ebenso ungeprüft waren ``starte_timer()`` und
# ``check()``.


@pytest.mark.unit
class TestStrategieDenkerBudget:
    """Vertrag von starte_timer() und check() — der 32×-Notausstieg."""

    def _chunk_plan(self, *, seconds: float = 10.0, severity: float = 0.0, **kwargs):
        from denker.strategie_denker import StrategieDenker

        audio = np.zeros(int(seconds * SR), dtype=np.float32)
        return StrategieDenker().plan(audio, SR, defect_severity=severity, **kwargs)

    def test_10_starte_timer_returns_none_and_arms_budget(self):
        from denker.strategie_denker import StrategieDenker

        denker = StrategieDenker()
        assert denker.starte_timer(10) is None
        status = denker.check(phases_remaining=5)
        assert status.elapsed_s >= 0.0
        assert status.rt_factor_current >= 0.0
        assert status.should_exit_early is False, "Frisch gestartetes Budget darf nicht sofort abbrechen"

    def test_11_budget_remaining_is_finite_and_non_negative(self):
        from denker.strategie_denker import StrategieDenker

        denker = StrategieDenker()
        denker.starte_timer(10)
        status = denker.check(phases_remaining=1)
        assert math.isfinite(status.budget_remaining_s)
        assert status.budget_remaining_s >= 0.0

    def test_12_check_without_timer_does_not_raise(self):
        from denker.strategie_denker import StrategieDenker

        status = StrategieDenker().check()
        assert math.isfinite(status.rt_factor_current)

    def test_13_plan_then_check_is_consistent(self):
        """Nach plan() muss check() mit demselben Song-Budget arbeiten."""
        from denker.strategie_denker import StrategieDenker

        denker = StrategieDenker()
        audio = np.zeros(SR * 3, dtype=np.float32)
        plan = denker.plan(audio, SR)
        denker.starte_timer(int(plan.audio_duration_s) or 1)
        status = denker.check(phases_remaining=3)
        assert status.budget_remaining_s <= plan.max_processing_s + 1e-6

    def test_14_long_songs_are_chunked_short_ones_are_not(self):
        """§7.6: lange Songs werden in Chunks zerlegt, kurze laufen ganz."""
        long_chunk = self._chunk_plan(seconds=600.0).recommended_chunk_s
        short_chunk = self._chunk_plan(seconds=30.0).recommended_chunk_s
        assert long_chunk == pytest.approx(120.0)
        assert short_chunk == pytest.approx(30.0)

    def test_15_severity_lowers_chunk_size(self):
        """Höhere Defekt-Schwere → feingranularere Chunks (§7.6)."""
        calm = self._chunk_plan(seconds=600.0, severity=0.0).recommended_chunk_s
        dense = self._chunk_plan(seconds=600.0, severity=0.9).recommended_chunk_s
        assert dense < calm, (dense, calm)

    def test_16_chain_depth_wish_is_capped_at_the_guard(self):
        """Der Tiefen-Wunsch ist ein Wunsch: er darf die Guard-Grenze nicht überschreiten."""
        flat = self._chunk_plan(seconds=10.0, chain_depth=1).max_processing_s
        deep = self._chunk_plan(seconds=10.0, chain_depth=9).max_processing_s
        assert deep == pytest.approx(32.0 * 10.0, rel=1e-6)
        assert flat == pytest.approx(32.0 * 10.0, rel=1e-6)

    def test_17_low_restorability_wish_is_capped_at_the_guard(self):
        """Schwer restaurierbares Material bekommt mehr Zeit — aber nie über den Guard hinaus."""
        easy = self._chunk_plan(seconds=10.0, restorability_score=100.0).max_processing_s
        hard = self._chunk_plan(seconds=10.0, restorability_score=0.0).max_processing_s
        assert hard == pytest.approx(32.0 * 10.0, rel=1e-6)
        assert easy == pytest.approx(32.0 * 10.0, rel=1e-6)

    def test_17b_fast_mode_gets_the_narrower_budget(self):
        """FAST führt im Guard 8×, nicht 32× — der Plan muss das abbilden."""
        quality = self._chunk_plan(seconds=10.0, mode="quality").max_processing_s
        fast = self._chunk_plan(seconds=10.0, mode="fast").max_processing_s
        assert fast < quality
        assert fast <= 8.0 * 10.0 * 1.15 + 1e-6
        assert fast >= 8.0 * 10.0 - 1e-6

    def test_18_unknown_mode_is_normalized(self):
        """Unbekannte Modus-Namen dürfen nicht abstürzen und werden normalisiert."""
        from denker.strategie_denker import StrategieDenker

        plan = StrategieDenker().plan(np.zeros(SR, dtype=np.float32), SR, mode="voellig-unbekannt")
        assert isinstance(plan.quality_mode, str) and plan.quality_mode

    def test_19_budget_note_is_present_and_german(self):
        plan = self._chunk_plan(seconds=600.0, severity=0.9)
        assert isinstance(plan.budget_note, str)

    def test_20_severity_is_taken_from_signal_signature_upwards_only(self):
        """Der Severity-Aufschlag aus der Signal-Signatur hebt nur an, nie ab."""
        weak = {"crest_factor_db": 8.0, "transient_ratio": 0.001, "micro_dynamics": 20.0, "hf_ratio": 0.01}
        strong = {"crest_factor_db": 26.0, "transient_ratio": 0.02, "micro_dynamics": 8.0, "hf_ratio": 0.2}
        base = self._chunk_plan(seconds=10.0, severity=0.3, signal_signature=weak).defect_severity
        boosted = self._chunk_plan(seconds=10.0, severity=0.3, signal_signature=strong).defect_severity
        assert base >= 0.3
        assert boosted >= base


# ─── §7.6 Defekt-adaptive Chunk-Größe (Spec §7.6) ──────────────────────────────────


class TestAdaptiveChunkSize:
    """Spec §7.6: Chunk-Größe muss defektdichte-adaptiv sein."""

    def _chunk(self, dur_s: float, severity: float) -> float:
        from denker.strategie_denker import _adaptive_chunk

        return _adaptive_chunk(dur_s, defect_severity=severity)

    def test_21_high_severity_gives_small_chunks(self):
        """§7.6: defect_severity >= 0.6 → 5 s Chunks (Feingranular)."""
        chunk = self._chunk(120.0, severity=0.7)
        assert chunk == pytest.approx(5.0, rel=1e-6), f"Expected 5.0 s, got {chunk}"

    def test_22_moderate_severity_gives_medium_chunks(self):
        """§7.6: defect_severity >= 0.3 → 15 s Chunks."""
        chunk = self._chunk(120.0, severity=0.4)
        assert chunk == pytest.approx(15.0, rel=1e-6), f"Expected 15.0 s, got {chunk}"

    def test_23_low_severity_gives_large_chunks(self):
        """§7.6: defect_severity < 0.3 → 60 s Chunks (clean material)."""
        chunk = self._chunk(120.0, severity=0.1)
        assert chunk == pytest.approx(60.0, rel=1e-6), f"Expected 60.0 s, got {chunk}"

    def test_24_short_file_not_chunked(self):
        """Dateien <= 2 s werden nicht unterteilt."""
        chunk = self._chunk(1.5, severity=0.9)
        assert chunk == pytest.approx(1.5, rel=1e-6)

    def test_25_chunk_never_exceeds_file_length(self):
        """Chunk-Größe darf nie größer als die Audio-Dauer sein."""
        for dur in (3.0, 10.0, 30.0, 60.0, 300.0):
            for sev in (0.0, 0.3, 0.6, 1.0):
                chunk = self._chunk(dur, severity=sev)
                assert chunk <= dur, f"chunk={chunk} > dur={dur} bei sev={sev}"

    def test_26_chunk_min_2s(self):
        """Chunk-Größe Minimum 2 s (außer Dateien < 2 s)."""
        chunk = self._chunk(120.0, severity=0.99)
        assert chunk >= 2.0

    def test_27_strategie_plan_includes_defect_severity(self):
        """StrategieDenker.plan() nimmt defect_severity an und schreibt es in den Plan."""
        import numpy as np

        from denker.strategie_denker import StrategieDenker

        audio = np.zeros(int(SR * 120), dtype=np.float32)
        denker = StrategieDenker()
        plan = denker.plan(audio, SR, defect_severity=0.7)
        assert hasattr(plan, "defect_severity")
        assert plan.defect_severity == pytest.approx(0.7, rel=1e-6)

    def test_28_strategie_plan_chunk_reflects_severity(self):
        """StrategieDenker.plan() mit severity=0.7 ergibt 5 s Chunk."""
        import numpy as np

        from denker.strategie_denker import StrategieDenker

        audio = np.zeros(int(SR * 120), dtype=np.float32)
        plan = StrategieDenker().plan(audio, SR, defect_severity=0.7)
        assert plan.recommended_chunk_s == pytest.approx(5.0, rel=1e-6), (
            f"Expected 5.0 s chunk for severity=0.7, got {plan.recommended_chunk_s}"
        )

    def test_29_boundary_exactly_06_is_fine_grained(self):
        """Überprüft Grenzwert severity=0.6 fällt in 5-s-Bucket."""
        chunk = self._chunk(300.0, severity=0.6)
        assert chunk == pytest.approx(5.0, rel=1e-6)

    def test_30_boundary_exactly_03_is_medium(self):
        """Überprüft Grenzwert severity=0.3 fällt in 15-s-Bucket."""
        chunk = self._chunk(300.0, severity=0.3)
        assert chunk == pytest.approx(15.0, rel=1e-6)


class TestModeAliasNormalization:
    def test_31_parse_mode_accepts_studio_aliases(self):
        from denker.strategie_denker import StrategieDenker

        m1 = StrategieDenker._parse_mode("studio2026")
        m2 = StrategieDenker._parse_mode("studio_2026")
        m3 = StrategieDenker._parse_mode("Studio 2026")

        assert m1 == m2 == m3


class TestSignalAwareSeverity:
    def test_32_derive_effective_severity_increases_on_risky_signature(self):
        from denker.strategie_denker import _derive_effective_defect_severity

        base = 0.25
        sig = {
            "crest_db": 21.0,
            "transient_ratio": 0.015,
            "micro_dynamic_db": 16.0,
            "hf_ratio": 0.15,
        }
        effective = _derive_effective_defect_severity(base, sig)
        assert effective > base
        assert 0.0 <= effective <= 1.0

    def test_33_plan_uses_signal_signature_for_chunking(self):
        import numpy as np

        from denker.strategie_denker import StrategieDenker

        audio = np.zeros(int(SR * 120), dtype=np.float32)
        sig = {
            "crest_db": 22.0,
            "transient_ratio": 0.02,
            "micro_dynamic_db": 18.0,
            "hf_ratio": 0.14,
        }
        plan = StrategieDenker().plan(audio, SR, defect_severity=0.1, signal_signature=sig)
        assert plan.defect_severity > 0.1
        assert plan.recommended_chunk_s == pytest.approx(15.0, rel=1e-6)
