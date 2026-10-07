"""Vertragstests der Denker-SOTA-Korrekturen (2026-10-07).

Jeder Test sichert genau eine Zusage, die vorher **nicht** galt oder deren
Fehlen unbemerkt blieb. Die Befunde stammen aus der Tiefenanalyse aller 13
Denker (Register D-K3-14 … D-K3-17).

* §2.59 Chirurgischer Schutz: eine Phase, die einen chirurgischen Defekt
  behandelt, wird von der Konflikt-Supprimierung ausgenommen.
* Hörordnung Ebene 3: kein Gewinn auf Kosten eines höherrangigen Ziels.
* §G5 (copilot-instructions.md): die Band-Budgets hängen nicht von der
  Hash-Reihenfolge einer Menge ab.
* §V8 (copilot-instructions.md): kein Song erbt den Koordinator-Zustand des
  Vorsongs; kein Modul-globaler Lernkanal; die Re-Pass-Dämpfung mutiert nicht
  die gecachte Optimizer-Instanz.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit


# ── §2.59 Chirurgischer Schutz ──────────────────────────────────────────────


def test_surgical_phases_protects_click_family() -> None:
    """Klicks/Knistern schützen die Impuls- und Oberflächenphasen."""
    from denker.phase_interaction_denker import PhaseInteractionDenker

    protected = PhaseInteractionDenker._surgical_phases({"surgical_defect_types": ["clicks", "crackle"]})
    assert "phase_01_click_removal" in protected
    assert "phase_09_crackle_removal" in protected
    assert "phase_27_click_pop_removal" in protected


def test_surgical_phases_protects_dropout_family() -> None:
    """Dropouts schützen die Gap-/Inpainting-Phasen."""
    from denker.phase_interaction_denker import PhaseInteractionDenker

    protected = PhaseInteractionDenker._surgical_phases({"surgical_defect_types": ["dropouts", "dropout_oxide"]})
    assert "phase_24_dropout_repair" in protected


def test_surgical_phases_is_empty_without_hint_or_match() -> None:
    """Ohne Hint oder ohne Treffer wird nichts geschützt (kein Blindschutz)."""
    from denker.phase_interaction_denker import PhaseInteractionDenker

    assert PhaseInteractionDenker._surgical_phases(None) == frozenset()
    assert PhaseInteractionDenker._surgical_phases({}) == frozenset()
    assert PhaseInteractionDenker._surgical_phases({"surgical_defect_types": ["voellig_unbekannt"]}) == frozenset()


def test_surgical_phase_survives_conflict_resolution() -> None:
    """Die Zusage in der Praxis: eine geschützte Phase steht danach noch im Plan."""
    from denker.phase_interaction_denker import PhaseInteractionDenker

    denker = PhaseInteractionDenker()
    phases = ["phase_35_multiband_compression", "phase_16_stereo_width", "phase_01_click_removal"]
    annotations = denker._annotate(phases)
    resolved, suppressed, _notes = denker._resolve_conflicts(
        phases, annotations, protected=frozenset({"phase_01_click_removal"})
    )
    assert "phase_01_click_removal" in resolved
    assert "phase_01_click_removal" not in suppressed


# ── Hörordnung Ebene 3 ──────────────────────────────────────────────────────


def test_hearing_order_rejects_gain_at_cost_of_higher_tier() -> None:
    """Brillanz (Stufe 4) steigt, Natürlichkeit (Stufe 1) fällt → abgelehnt."""
    from denker.exzellenz_denker import hearing_order_violation

    initial = {"natuerlichkeit": 0.72, "waerme": 0.70, "transparenz": 0.66, "brillanz": 0.62}
    candidate = {"natuerlichkeit": 0.65, "waerme": 0.70, "transparenz": 0.66, "brillanz": 0.75}
    violation = hearing_order_violation(initial, candidate)
    assert violation is not None
    assert "natuerlichkeit" in violation


def test_hearing_order_allows_improvement_on_all_tiers() -> None:
    """Steigen alle Ziele, liegt kein Verstoß vor."""
    from denker.exzellenz_denker import hearing_order_violation

    initial = {"natuerlichkeit": 0.72, "waerme": 0.70, "brillanz": 0.62}
    candidate = {"natuerlichkeit": 0.74, "waerme": 0.72, "brillanz": 0.65}
    assert hearing_order_violation(initial, candidate) is None


def test_hearing_order_ignores_measurement_noise() -> None:
    """Deltas unterhalb der Rauschgrenze gelten weder als Gewinn noch als Verlust."""
    from denker.exzellenz_denker import _HEARING_NOISE_EPS_DEFAULT, hearing_order_violation

    eps = _HEARING_NOISE_EPS_DEFAULT
    initial = {"natuerlichkeit": 0.72, "brillanz": 0.62}
    candidate = {"natuerlichkeit": 0.72 - eps / 2, "brillanz": 0.62 + eps * 3}
    assert hearing_order_violation(initial, candidate) is None


def test_hearing_order_ignores_inapplicable_goals() -> None:
    """Ein als nicht anwendbar markiertes Ziel darf keinen Verstoß erzeugen."""
    from denker.exzellenz_denker import hearing_order_violation

    initial = {"natuerlichkeit": 0.72, "brillanz": 0.62}
    candidate = {"natuerlichkeit": 0.60, "brillanz": 0.75}
    assert hearing_order_violation(initial, candidate, inapplicable=frozenset({"natuerlichkeit"})) is None


def test_hearing_order_skips_non_finite_values() -> None:
    """NaN-Kandidaten werden ignoriert statt als Verstoß fehlgedeutet."""
    from denker.exzellenz_denker import hearing_order_violation

    initial = {"natuerlichkeit": 0.72, "brillanz": 0.62}
    candidate = {"natuerlichkeit": float("nan"), "brillanz": 0.75}
    assert hearing_order_violation(initial, candidate) is None


# ── §G5 (copilot-instructions.md) Determinismus der Band-Budgets ──────────────────────────────────────


def test_band_budgets_do_not_depend_on_input_order() -> None:
    """Dieselben Phasen in anderer Reihenfolge liefern bit-identische Kappungen —
    §G5 (copilot-instructions.md)."""
    from denker.cross_phase_coordinator import CrossPhaseCoordinator

    phases = [
        "phase_03_denoise",
        "phase_19_de_esser",
        "phase_38_presence_boost",
        "phase_29_tape_hiss_reduction",
        "phase_06_bw_extension",
        "phase_23_spectral_repair",
    ]
    first = CrossPhaseCoordinator.analyze(phase_plan=list(phases), material="vinyl", decade=1972)._last_result
    second = CrossPhaseCoordinator.analyze(
        phase_plan=list(reversed(phases)), material="vinyl", decade=1972
    )._last_result
    assert first is not None and second is not None
    assert first.capped_strengths == second.capped_strengths


# ── §V8 (copilot-instructions.md) Song-Isolation ──────────────────────────────────────────────────────


def test_coordinator_reset_clears_previous_song_caps() -> None:
    """Ohne Reset lieferte der Koordinator im Folge-Song die Caps des Vorsongs."""
    from denker.cross_phase_coordinator import CrossPhaseCoordinator

    CrossPhaseCoordinator.analyze(phase_plan=["phase_19_de_esser", "phase_38_presence_boost"], material="vinyl")
    assert CrossPhaseCoordinator.get_capped_strength("phase_19_de_esser", 0.8, material="vinyl") is not None

    CrossPhaseCoordinator.reset_session()
    assert CrossPhaseCoordinator.get_capped_strength("phase_19_de_esser", 0.8, material="vinyl") is None


def test_low_confidence_cross_song_channel_is_gone() -> None:
    """Der song-übergreifende Lernkanal darf nicht zurückkehren (§V8/§G1 (copilot-instructions.md))."""
    import denker.phase_interaction_denker as module

    assert not hasattr(module, "record_low_confidence_stripped_families")
    assert not hasattr(module, "get_low_confidence_stripped_families")


def test_optimizer_cache_is_not_mutated_by_goal_repair(monkeypatch: pytest.MonkeyPatch) -> None:
    """Der Re-Pass dämpft auf einer FRISCHEN Instanz, nicht auf der gecachten —
    §V8 (copilot-instructions.md)."""
    from denker.exzellenz_denker import ExzellenzDenker

    denker = ExzellenzDenker()
    audio = np.sin(np.linspace(0.0, 200.0, 48000, dtype=np.float32)) * 0.3
    audio = audio.astype(np.float32)

    built: list[int] = []

    class _Opt:
        def __init__(self) -> None:
            self.sample_rate = 48_000
            self.material = "vinyl"
            self._harm_boost_db = 1.0
            self._modulation_strength = 1.0

        def optimize(self, arr: np.ndarray):
            return arr, None

    def _factory(sr: int, material: str):
        built.append(1)
        return _Opt()

    monkeypatch.setattr(type(denker), "_build_optimizer", staticmethod(_factory))
    # Ein Ziel verletzt (erzwingt den Re-Pass), danach alles erfüllt.
    monkeypatch.setattr(type(denker), "messe_ziele", lambda self, a, s, reference=None, **kw: {"natuerlichkeit": 0.10})

    cached = denker._get_optimizer(sr=48_000, material="vinyl")
    _cached_harm = float(cached._harm_boost_db)

    denker.messe_und_repariere(audio, 48_000, material="vinyl")

    assert built, "Der Re-Pass muss eine frische Instanz bauen"
    assert float(cached._harm_boost_db) == _cached_harm, "Die gecachte Instanz darf nicht gedämpft werden"
    assert cached._modulation_strength == 1.0
