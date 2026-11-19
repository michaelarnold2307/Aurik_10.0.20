"""Vertrags-Tests §PERF-R21 — RT-Budget-Ledger (D-K3-75).

Der Ledger macht das 32×-Budget **allozierbar**: Soll-Anteil je Phase aus der
normativen Hör-Impact-Ordnung (``PHASE_PRIORITIES``), Ist-Anteil aus den
gemessenen Phasenzeiten, plus Reserve/Überziehung in Summe.

Die Tests pinnen vier Dinge:
1. Die Rechnung stimmt (Summen, Anteile, Status, Reserve).
2. Der Ledger ist deterministisch (§G5 (copilot-instructions.md)) und zero-safe
   (vor ``start_monitoring``).
3. Er ist **reine Buchführung** — kein Skip-/Deferral-Pfad liest ihn
   (§V7 (copilot-instructions.md): kein stiller Verhaltens-Flip).
4. Er ist verdrahtet (UV3-Metadaten), sonst wäre er die nächste Phantom-Norm.
"""

from __future__ import annotations

import inspect

import pytest

from backend.core.performance_guard import PerformanceGuard, PhasePerformance, QualityMode

EPS = 1e-6


def _guard(duration_s: float = 60.0, mode: QualityMode = QualityMode.QUALITY) -> PerformanceGuard:
    guard = PerformanceGuard(mode=mode, enforce_limit=False, enable_adaptive_skipping=False)
    guard.start_monitoring(duration_s)
    return guard


def _add_phase(guard: PerformanceGuard, phase_id: str, duration_s: float) -> None:
    """Misst eine Phase deterministisch (ohne echte Zeit) in den Guard hinein."""
    audio = float(guard.audio_duration or 0.0)
    guard.phase_performances.append(
        PhasePerformance(
            phase_id=phase_id,
            start_time=0.0,
            end_time=duration_s,
            duration_seconds=duration_s,
            audio_duration_seconds=audio,
            rt_factor=(duration_s / audio) if audio else 0.0,
            is_critical=guard._is_phase_critical(phase_id),
            skipped=False,
        )
    )


class TestLedgerArithmetik:
    def test_total_budget_is_target_rt_times_budget_duration(self) -> None:
        guard = _guard(60.0)
        ledger = guard.get_budget_allocation()
        assert ledger["schema"] == "aurik.rt_budget_ledger/1"
        assert ledger["target_rt_factor"] == pytest.approx(32.0)
        assert ledger["audio_duration_s"] == pytest.approx(60.0)
        assert ledger["total_budget_s"] == pytest.approx(32.0 * 60.0)

    def test_soll_anteile_folgen_der_normativen_prioritaet(self) -> None:
        """Soll = PHASE_PRIORITIES normalisiert; die Summe schöpft das Budget aus."""
        guard = _guard(60.0)
        _add_phase(guard, "click_removal", 10.0)  # Priorität 10
        _add_phase(guard, "final_polish", 10.0)  # Priorität 3
        ledger = guard.get_budget_allocation()

        rows = {row["phase_id"]: row for row in ledger["phases"]}
        assert rows["click_removal"]["priority"] == 10
        assert rows["final_polish"]["priority"] == 3
        assert rows["click_removal"]["share"] == pytest.approx(10 / 13, abs=1e-3)
        assert rows["final_polish"]["share"] == pytest.approx(3 / 13, abs=1e-3)
        assert sum(row["share"] for row in ledger["phases"]) == pytest.approx(1.0, abs=1e-3)
        assert ledger["allocated_s"] == pytest.approx(ledger["total_budget_s"], abs=1e-2)

    def test_ist_ueber_soll_wird_als_ueberzogen_ausgewiesen(self) -> None:
        guard = _guard(60.0)
        _add_phase(guard, "click_removal", 10.0)
        _add_phase(guard, "final_polish", 10.0)
        # click_removal darf 1920 * 10/13 = 1476,9 s kosten — 1500 s sind zu viel.
        _add_phase(guard, "click_removal", 1500.0)

        rows = {row["phase_id"]: row for row in guard.get_budget_allocation()["phases"]}
        assert rows["click_removal"]["measured_s"] == pytest.approx(1510.0)
        assert rows["click_removal"]["status"] == "ueberzogen"
        assert rows["click_removal"]["overrun_s"] > 0.0
        assert rows["final_polish"]["status"] in {"im_budget", "knapp"}

    def test_reserve_und_ueberziehung_in_summe(self) -> None:
        guard = _guard(60.0)
        _add_phase(guard, "click_removal", 100.0)
        ledger = guard.get_budget_allocation()
        assert ledger["measured_total_s"] == pytest.approx(100.0)
        assert ledger["reserve_s"] == pytest.approx(ledger["total_budget_s"] - 100.0)
        assert ledger["budget_exceeded"] is False
        assert ledger["rt_measured"] == pytest.approx(100.0 / 60.0, abs=1e-3)

        _add_phase(guard, "click_removal", ledger["total_budget_s"])
        over = guard.get_budget_allocation()
        assert over["budget_exceeded"] is True
        assert over["reserve_s"] < 0.0
        assert over["overrun_total_s"] > 0.0

    def test_top_consumers_sind_deterministisch_sortiert(self) -> None:
        guard = _guard(60.0)
        _add_phase(guard, "final_polish", 50.0)
        _add_phase(guard, "click_removal", 50.0)  # Gleichstand → alphabetisch
        top = guard.get_budget_allocation()["top_consumers"]
        assert [row["phase_id"] for row in top] == ["click_removal", "final_polish"]

    def test_mehrfach_gelaufene_phase_wird_summiert(self) -> None:
        """Deckungsgleichheit mit current_rt_factor (kumulativ), nicht je Aufruf."""
        guard = _guard(60.0)
        _add_phase(guard, "phase_12_wow_flutter_fix", 20.0)
        _add_phase(guard, "phase_12_wow_flutter_fix", 30.0)
        rows = {row["phase_id"]: row for row in guard.get_budget_allocation()["phases"]}
        # Zeilen-ID bleibt der echte Kurzname; die Priorität kommt über den Alias.
        assert rows["wow_flutter_fix"]["measured_s"] == pytest.approx(50.0)
        assert rows["wow_flutter_fix"]["priority"] == 9


class TestPrioritaetsDrift:
    """§PERF-R21b (D-K3-76): zwei CRITICAL-Phasen liefen als MEDIUM.

    ``_normalize_phase_id`` erzeugt aus den Modulnamen ``wow_flutter_fix`` /
    ``crackle_removal``; die normative Tabelle kannte nur ``wow_flutter`` /
    ``decracklé``. Folge: Default 5 statt 9 ⇒ ``should_skip_phase`` durfte die
    teuerste Reparaturphase unter RT-Druck überspringen, obwohl die Norm
    „Priority ≥ 9 wird NIEMALS übersprungen" genau sie meint.
    """

    @pytest.mark.parametrize(
        ("phase_id", "table_key", "priority"),
        [
            ("phase_12_wow_flutter_fix", "wow_flutter", 9),
            ("phase_09_crackle_removal", "decracklé", 9),
        ],
    )
    def test_alias_loest_auf_die_normative_prioritaet_auf(self, phase_id: str, table_key: str, priority: int) -> None:
        assert PerformanceGuard._priority_key(phase_id) == table_key
        assert PerformanceGuard.PHASE_PRIORITIES[table_key] == priority

    @pytest.mark.parametrize("phase_id", ["phase_12_wow_flutter_fix", "phase_09_crackle_removal"])
    def test_kritische_phase_gilt_jetzt_als_kritisch(self, phase_id: str) -> None:
        guard = _guard(60.0)
        assert guard._is_phase_critical(phase_id) is True

    def test_skip_entscheidung_verweigert_fuer_die_teuerste_phase(self) -> None:
        """Realitätsnaher Fall: Budget bereits zu 95 % verbraucht, Phase teuer."""
        guard = _guard(60.0)
        guard.enable_adaptive_skipping = True
        # 95 % des 32×-Budgets sind aufgebraucht (1920 s * 0.95 = 1824 s).
        guard.current_rt_factor = 30.4
        assert (
            guard.should_skip_phase("phase_12_wow_flutter_fix", estimated_time_seconds=120.0, remaining_phases=5)
            is False
        )
        assert (
            guard.should_skip_phase("phase_09_crackle_removal", estimated_time_seconds=120.0, remaining_phases=5)
            is False
        )

    def test_unprofilierte_phase_ist_kritisch_und_nie_skipbar_m6(self) -> None:
        """M6 (2026-11-17): Phasen ohne PHASE_PRIORITIES-Eintrag sind CRITICAL (10).

        D-K3-76-Lehre: zwei CRITICAL-Phasen fielen als „unbekannt“ auf MEDIUM
        und waren unter Budgetdruck skipbar — Budget-Spielraum entsteht durch
        Profilierung, nicht durch das Skip unprofilierter Phasen.
        """
        assert PerformanceGuard.BUDGET_ALLOCATION_DEFAULT_PRIORITY == 10
        guard = _guard(60.0)
        guard.enable_adaptive_skipping = True
        guard.current_rt_factor = 30.4  # 95 % Budget verbraucht

        assert guard._is_phase_critical("phase_99_voellig_unbekannt") is True
        assert (
            guard.should_skip_phase("phase_99_voellig_unbekannt", estimated_time_seconds=120.0, remaining_phases=5)
            is False
        )

    def test_aliasse_zeigen_auf_existierende_tabelleneintraege(self) -> None:
        """Kein Phantom-Alias: jedes Ziel existiert, jeder Schlüssel ist ein echter Kurzname."""
        for alias, target in PerformanceGuard.PHASE_PRIORITY_ALIASES.items():
            assert target in PerformanceGuard.PHASE_PRIORITIES, f"Ziel {target!r} fehlt in PHASE_PRIORITIES"
            assert PerformanceGuard._normalize_phase_id(f"phase_00_{alias}") == alias
            assert alias != target

    def test_jede_prioritaetsabfrage_nutzt_die_alias_aufloesung(self) -> None:
        """Muster C: die drei Entscheidungspfade dürfen NICHT an der Tabelle vorbeilesen."""
        src = inspect.getsource(PerformanceGuard)
        for method in ("should_skip_phase", "_is_phase_critical", "_calculate_quality_degradation"):
            start = src.index(f"def {method}(")
            end = src.index("\n    def ", start + 10)
            body = src[start:end]
            assert "_priority_key(" in body, f"{method} liest die Priorität ohne Alias-Auflösung"
            assert "PHASE_PRIORITIES.get(self._normalize_phase_id" not in body


class TestLedgerScopeUndRobustheit:
    def test_planned_scope_alloziert_vorab_ohne_messung(self) -> None:
        guard = _guard(60.0)
        ledger = guard.get_budget_allocation(planned_phase_ids=["phase_03_denoise", "phase_50_ml_refine"])
        assert ledger["scope"] == "planned"
        rows = {row["phase_id"]: row for row in ledger["phases"]}
        assert rows["denoise"]["priority"] == 8
        assert rows["denoise"]["status"] == "nicht_gelaufen"
        assert ledger["measured_total_s"] == 0.0
        assert ledger["reserve_s"] == pytest.approx(ledger["total_budget_s"])

    def test_unbekannte_phase_bekommt_median_prioritaet(self) -> None:
        guard = _guard(60.0)
        _add_phase(guard, "brandneue_phase", 10.0)
        rows = {row["phase_id"]: row for row in guard.get_budget_allocation()["phases"]}
        assert rows["brandneue_phase"]["priority"] == PerformanceGuard.BUDGET_ALLOCATION_DEFAULT_PRIORITY

    def test_zero_safe_vor_start_monitoring(self) -> None:
        guard = PerformanceGuard(mode=QualityMode.QUALITY)
        ledger = guard.get_budget_allocation()
        assert ledger["total_budget_s"] == 0.0
        assert ledger["phases"] == []
        assert ledger["budget_exceeded"] is False
        assert ledger["skipped_phases"] == []

    def test_deterministisch_bei_identischem_zustand(self) -> None:
        guard = _guard(60.0)
        _add_phase(guard, "click_removal", 12.0)
        _add_phase(guard, "denoise", 40.0)
        assert guard.get_budget_allocation() == guard.get_budget_allocation()

    def test_skipped_phases_werden_ausgewiesen(self) -> None:
        guard = _guard(60.0)
        guard.skipped_phases.append("phase_55_ml_refine")
        assert guard.get_budget_allocation()["skipped_phases"] == ["phase_55_ml_refine"]


class TestPhaseBudgetDelegation:
    def test_bekannte_phase_delegiert_an_die_allokation(self) -> None:
        guard = _guard(60.0)
        _add_phase(guard, "click_removal", 10.0)
        _add_phase(guard, "final_polish", 10.0)
        rows = {row["phase_id"]: row for row in guard.get_budget_allocation()["phases"]}
        assert guard.get_phase_budget_seconds("phase_01_click_removal", 2, 1) == pytest.approx(
            rows["click_removal"]["budget_s"]
        )

    def test_unbekannte_phase_faellt_auf_flaches_restbudget_zurueck(self) -> None:
        """Signatur-Vertrag: keine Funktion ohne Verhalten (AGENTS.md §5)."""
        guard = _guard(60.0)
        value = guard.get_phase_budget_seconds("nie_gelaufene_phase", total_phases=4, completed_phases=0)
        assert value > 0.0
        assert value <= 32.0 * 60.0


class TestLedgerIstReineBuchfuehrung:
    """§V7 (copilot-instructions.md): der Ledger darf keine Entscheidung treiben."""

    @pytest.mark.parametrize("method", ["should_skip_phase", "check_early_exit", "_get_status"])
    def test_entscheidungspfade_lesen_den_ledger_nicht(self, method: str) -> None:
        source = inspect.getsource(getattr(PerformanceGuard, method))
        assert "get_budget_allocation" not in source
        assert "_budget_row_status" not in source

    def test_ledger_veraendert_den_guard_zustand_nicht(self) -> None:
        guard = _guard(60.0)
        _add_phase(guard, "click_removal", 12.0)
        before = (
            len(guard.phase_performances),
            list(guard.skipped_phases),
            guard.current_rt_factor,
            guard.warnings.copy(),
        )
        guard.get_budget_allocation()
        after = (
            len(guard.phase_performances),
            list(guard.skipped_phases),
            guard.current_rt_factor,
            guard.warnings.copy(),
        )
        assert before == after


class TestLedgerVerdrahtung:
    def test_uv3_metadaten_tragen_das_ledger(self) -> None:
        """Wirkungs-Pin: ohne diese Verdrahtung wäre der Ledger ein Phantom."""
        from pathlib import Path

        source = Path("backend/core/unified_restorer_v3.py").read_text(encoding="utf-8")
        assert '"rt_budget_ledger"' in source
        assert "get_budget_allocation()" in source
        # Beide Budget-Sichten stehen nebeneinander (P0-3-Wahrheit + Allokation).
        assert source.index('"performance_guard_report"') < source.index('"rt_budget_ledger"')
