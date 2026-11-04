"""§Ebene-3 Wohlklang-Ordnungs-Enforcer — Unit Tests (Hörordnung §5/§2).

Prüft die ZENTRALE Durchsetzung der lexikografischen Wohlklang-Ordnung:
  - Nicht-Verletzungsfall: Kandidat bit-identisch unverändert (ACCEPT).
  - Verletzungsfall: verletzender Anteil wird gekappt (CAP, feste Blend-Stufen)
    oder zurückgeworfen (ROLLBACK, Muster: ExcellenceOptimizer-Rollback).
  - Ebene-1-Schutz: „nie gegen Ebene 1" — Stufe-1-Verluste werden nie akzeptiert.
  - Never-worsen: keine Kappung, die ein anderes Ziel unter seine Basis drückt.
  - Determinismus (§G5 (copilot-instructions.md)): gleiche Eingabe → gleiches
    Ergebnis; kein Zeit-/Sampling-Bezug in der Entscheidung.
  - Zentrale Stärke-Logik (§V7 (copilot-instructions.md)): feste Blend-Stufen,
    keine phasen-individuellen Schwellwerte.

Synthetische Score-Vektoren; kein Audio-I/O. Die Hörstufen werden wie im Gate
aus GoalPriorityProtocol recycelt (Stufe 1 Natürlichkeit/… > 2 Wärme > 3 Klarheit
> 4 Brillanz).
"""

from __future__ import annotations

import numpy as np

from backend.core.wohlklang_ordnung_gate import (
    ACCEPT,
    CAP,
    DEFAULT_BLEND_ALPHAS,
    ROLLBACK,
    OrdnungEnforcementResult,
    WohlklangOrdnungEnforcer,
    blend_with_reference,
    compute_goal_deltas,
)


def make_enforcer() -> WohlklangOrdnungEnforcer:
    return WohlklangOrdnungEnforcer()


# Verletzender Kandidat: brillanz (Stufe 4) verbessert auf Kosten von
# natuerlichkeit (Stufe 1) — das Grundmuster der B3-Verletzungspaare.
VIOLATING_DELTAS: dict[str, float] = {"brillanz": 0.10, "natuerlichkeit": -0.08}


class TestNichtVerletzung:
    def test_reine_verbesserung_wird_uebernommen(self) -> None:
        res = make_enforcer().enforce({"natuerlichkeit": 0.05, "waerme": 0.03})
        assert res.action == ACCEPT
        assert res.blend_factor == 1.0
        assert res.events == []
        assert res.residual_violations == 0

    def test_tie_innerhalb_derselben_stufe_ist_erlaubt(self) -> None:
        # transparenz & separation_fidelity sind beide Stufe 3 → kein Verstoß.
        res = make_enforcer().enforce({"transparenz": 0.05, "separation_fidelity": -0.04})
        assert res.action == ACCEPT

    def test_keine_deltas_kein_enforcement(self) -> None:
        for leer in (None, {}, {"brillanz": float("nan")}):
            res = make_enforcer().enforce(leer)
            assert res.action == ACCEPT
            assert res.blend_factor == 1.0

    def test_nicht_verletzungsfall_bit_identisch_unveraendert(self) -> None:
        """Ohne Verstoß gibt blend_with_reference EXAKT den Kandidaten zurück."""
        kandidat = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        referenz = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        res = make_enforcer().enforce({"natuerlichkeit": 0.01})
        assert res.action == ACCEPT
        ergebnis = blend_with_reference(kandidat, referenz, res.blend_factor)
        assert ergebnis is kandidat  # bit-identisch (§G5 (copilot-instructions.md))

    def test_enforced_deltas_entsprechen_dem_kandidaten(self) -> None:
        res = make_enforcer().enforce({"brillanz": 0.02})
        assert res.enforced_deltas == {"brillanz": 0.02}


class TestVerletzungKappen:
    def test_verletzungsfall_wird_gekappt(self) -> None:
        """Erste Blend-Stufe ohne Verstoß gewinnt — der Rest wird gekappt."""
        probes = {
            0.5: {"brillanz": 0.05, "natuerlichkeit": -0.04},  # weiter Verstoß
            0.25: {"brillanz": 0.02, "natuerlichkeit": 0.001},  # ordnungskonform
        }
        res = make_enforcer().enforce(VIOLATING_DELTAS, blend_probe=probes.__getitem__)
        assert res.action == CAP
        assert res.blend_factor == 0.25
        assert res.enforced_deltas == {"brillanz": 0.02, "natuerlichkeit": 0.001}
        assert res.residual_violations == 0
        assert len(res.events) == 1
        assert res.events[0].startswith("Ordnung erzwungen:")
        assert "gekappt" in res.events[0]
        assert res.violated_goals == ["natuerlichkeit"]

    def test_kappung_behaelt_groesste_konforme_stufe(self) -> None:
        probes = {alpha: {"brillanz": 0.02, "natuerlichkeit": 0.001} for alpha in DEFAULT_BLEND_ALPHAS}
        res = make_enforcer().enforce(VIOLATING_DELTAS, blend_probe=probes.__getitem__)
        assert res.action == CAP
        assert res.blend_factor == DEFAULT_BLEND_ALPHAS[0]

    def test_kappte_stufe_ist_gate_konform(self) -> None:
        """Jede akzeptierte Stufe ist gegen das Gate verifiziert (Rest = 0)."""
        probes = {
            0.5: {"brillanz": 0.05, "natuerlichkeit": -0.04},
            0.25: {"brillanz": 0.02, "natuerlichkeit": 0.0},
            0.125: {"brillanz": 0.01, "natuerlichkeit": 0.0},
            0.0625: {"brillanz": 0.005, "natuerlichkeit": 0.0},
        }
        res = make_enforcer().enforce(VIOLATING_DELTAS, blend_probe=probes.__getitem__)
        assert res.action == CAP
        assert res.blend_factor == 0.25
        assert res.residual_violations == 0


class TestRollback:
    def test_rollback_ohne_messprobe(self) -> None:
        """Ohne Probe greift das Muster ExcellenceOptimizer-Rollback."""
        res = make_enforcer().enforce(VIOLATING_DELTAS)
        assert res.action == ROLLBACK
        assert res.blend_factor == 0.0
        assert res.enforced_deltas == {"brillanz": 0.0, "natuerlichkeit": 0.0}
        assert res.residual_violations == 0
        assert "zurückgeworfen" in res.events[0]
        assert "ExcellenceOptimizer" in res.events[0]

    def test_rollback_wenn_keine_stufe_konform(self) -> None:
        probes = {alpha: {"brillanz": 0.05, "natuerlichkeit": -0.04} for alpha in DEFAULT_BLEND_ALPHAS}
        res = make_enforcer().enforce(VIOLATING_DELTAS, blend_probe=probes.__getitem__)
        assert res.action == ROLLBACK
        assert res.blend_factor == 0.0

    def test_rollback_wenn_proben_unbrauchbar(self) -> None:
        res = make_enforcer().enforce(VIOLATING_DELTAS, blend_probe=lambda _a: None)
        assert res.action == ROLLBACK

    def test_rollback_ist_never_worsen(self) -> None:
        res = make_enforcer().enforce(VIOLATING_DELTAS)
        assert all(d == 0.0 for d in res.enforced_deltas.values())


class TestEbene1Schutz:
    def test_ebene1_verletzung_wird_nie_akzeptiert(self) -> None:
        """„nie gegen Ebene 1": Stufe-1-Verlust + Probe ohne Erholung → ROLLBACK."""
        probes = {
            0.5: {"groove": 0.025, "natuerlichkeit": -0.01},
            0.25: {"groove": 0.012, "natuerlichkeit": -0.005},
            0.125: {"groove": 0.006, "natuerlichkeit": -0.002},
            0.0625: {"groove": 0.003, "natuerlichkeit": -0.001},
        }
        res = make_enforcer().enforce({"groove": 0.05, "natuerlichkeit": -0.02}, blend_probe=probes.__getitem__)
        assert res.action == ROLLBACK
        assert res.blend_factor == 0.0
        assert res.tier1_violated is True
        assert "Ebene-1" in res.events[0]
        assert "natuerlichkeit" in res.events[0]

    def test_ebene1_kappung_nur_mit_vollstaendiger_erholung(self) -> None:
        """CAP ist erlaubt, sobald das Stufe-1-Ziel nicht mehr unter der Basis liegt."""
        probes = {
            0.5: {"groove": 0.025, "natuerlichkeit": -0.01},
            0.25: {"groove": 0.012, "natuerlichkeit": 0.001},
        }
        res = make_enforcer().enforce({"groove": 0.05, "natuerlichkeit": -0.02}, blend_probe=probes.__getitem__)
        assert res.action == CAP
        assert res.blend_factor == 0.25
        assert res.tier1_violated is True  # Befund bleibt sichtbar (§G8 (copilot-instructions.md))
        assert "Ebene-1-Schutz" in res.events[0]

    def test_kein_accept_bei_ebene1_verletzung(self) -> None:
        res = make_enforcer().enforce({"groove": 0.05, "natuerlichkeit": -0.02})
        assert res.action != ACCEPT
        assert res.tier1_violated is True


class TestNeverWorsen:
    def test_keine_kappung_auf_kosten_anderer_ziele(self) -> None:
        """Eine Probe ohne Verstoß, die aber Ziele unter die Basis drückt,
        wird NICHT akzeptiert („höhere Ebenen verlieren nie")."""
        probes = {alpha: {"brillanz": -0.01, "natuerlichkeit": -0.02} for alpha in DEFAULT_BLEND_ALPHAS}
        res = make_enforcer().enforce(VIOLATING_DELTAS, blend_probe=probes.__getitem__)
        assert res.action == ROLLBACK
        assert all(d == 0.0 for d in res.enforced_deltas.values())


class TestDeterminismus:
    def test_gleiche_eingabe_gleiches_ergebnis(self) -> None:
        probes = {
            0.5: {"brillanz": 0.05, "natuerlichkeit": -0.04},
            0.25: {"brillanz": 0.02, "natuerlichkeit": 0.001},
        }
        a = make_enforcer().enforce(VIOLATING_DELTAS, blend_probe=probes.__getitem__)
        b = make_enforcer().enforce(dict(VIOLATING_DELTAS), blend_probe=probes.__getitem__)
        assert a.to_dict() == b.to_dict()

    def test_stufen_sind_fest_und_absteigend(self) -> None:
        """Zentrale Stärke-Logik (§V7 (copilot-instructions.md)): keine phasen-individuellen Schwellwerte."""
        from itertools import pairwise

        assert all(0.0 < a < 1.0 for a in DEFAULT_BLEND_ALPHAS)
        assert all(groesser > kleiner for groesser, kleiner in pairwise(DEFAULT_BLEND_ALPHAS))


class TestSchwellwertDurchreichung:
    def test_keine_neue_schwelle(self) -> None:
        """Die optionale Schwelle wird unverändert an das Gate durchgereicht."""
        res = make_enforcer().enforce({"brillanz": 0.10, "waerme": -0.001}, threshold=0.05)
        assert res.action == ACCEPT


class TestBlendWithReference:
    def test_alpha_1_und_darueber_bit_identisch(self) -> None:
        kandidat = np.array([1.0, 2.0], dtype=np.float32)
        referenz = np.array([0.0, 0.0], dtype=np.float32)
        assert blend_with_reference(kandidat, referenz, 1.0) is kandidat
        assert blend_with_reference(kandidat, referenz, 1.5) is kandidat

    def test_alpha_0_ist_rollback_auf_referenzkopie(self) -> None:
        kandidat = np.array([1.0, 2.0], dtype=np.float32)
        referenz = np.array([0.5, -0.5], dtype=np.float32)
        ergebnis = blend_with_reference(kandidat, referenz, 0.0)
        assert ergebnis is not referenz
        np.testing.assert_array_equal(ergebnis, referenz)

    def test_alpha_0_5_ist_mittelpunkt(self) -> None:
        kandidat = np.array([1.0, 2.0], dtype=np.float32)
        referenz = np.array([0.0, 0.0], dtype=np.float32)
        np.testing.assert_allclose(
            blend_with_reference(kandidat, referenz, 0.5),
            np.array([0.5, 1.0]),
        )


class TestComputeGoalDeltas:
    def test_delta_ist_after_minus_before(self) -> None:
        deltas = compute_goal_deltas({"a": 0.5, "b": 0.2}, {"a": 0.7, "b": 0.1})
        assert set(deltas) == {"a", "b"}
        assert abs(deltas["a"] - 0.2) < 1e-9
        assert abs(deltas["b"] + 0.1) < 1e-9

    def test_fehlende_keys_werden_mit_null_beziffert(self) -> None:
        deltas = compute_goal_deltas({"a": 0.5}, {"a": 0.5, "b": 0.3})
        assert deltas == {"a": 0.0, "b": 0.3}

    def test_unsaubere_werte_uebersprungen(self) -> None:
        deltas = compute_goal_deltas(
            {"a": 0.5, "nan": float("nan"), "txt": "x"},
            {"a": 0.6, "nan": 0.1, "txt": 0.1},
        )
        assert set(deltas) == {"a"}
        assert abs(deltas["a"] - 0.1) < 1e-9


class TestErgebnisSerialisierung:
    def test_to_dict_alle_felder(self) -> None:
        d = make_enforcer().enforce(VIOLATING_DELTAS).to_dict()
        assert d["action"] == ROLLBACK
        assert d["blend_factor"] == 0.0
        assert d["tier1_violated"] is True
        assert d["violated_goals"] == ["natuerlichkeit"]
        assert isinstance(d["violations"], list) and d["violations"]
        assert d["residual_violations"] == 0
        assert d["events"][0].startswith("Ordnung erzwungen:")

    def test_default_ergebnis(self) -> None:
        r = OrdnungEnforcementResult()
        assert r.action == ACCEPT
        assert r.blend_factor == 1.0
        assert r.residual_violations == 0
