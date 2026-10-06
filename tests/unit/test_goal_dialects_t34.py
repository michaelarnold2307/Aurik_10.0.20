"""Tranche 3.4 — Ziel-Dialekte, kanonische Zuordnung und §V6-Sichtbarkeit.

Befund 2026-10-06: Das Projekt führt historisch **vier** Ziel-Vokabulare; kanonisch
ist allein ``ALL_GOAL_NAMES``. Gemessen:

* **12 Fremdschlüssel**, davon nur 2 wörtlich abbildbar
  (``mikrodynamik`` → ``micro_dynamics``, ``raeumlichkeit`` → ``spatial_depth``).
* ``goal_budget.create_goal_budget()`` verwarf pro Genre **5 von 15 Zellen STILL**
  (§V6-Verstoß) — bei ``metal`` ausgerechnet die stärksten Signale
  (``punch`` 2.0, ``bass_praesenz`` 1.9).

Diese Suite pinnt: Deklarations-Vollständigkeit, Auflösungsverhalten,
**Verhaltensgleichheit** des Budgets (kein Klang-Eingriff) und die neue
Sichtbarkeit des Verwerfens.
"""

from __future__ import annotations

import logging

import pytest

from backend.core import goal_budget as gb
from backend.core.song_goal_importance import (
    ALL_GOAL_NAMES,
    GOAL_DIALECT_MAP,
    GOAL_DIALECT_NOTES,
    canonical_goal_name,
)

#: Die 12 Fremdschlüssel, die am 2026-10-06 gemessen wurden
#: (Vereinigung aus ``genre_goal_profile`` und ``goal_budget._DEFAULT_GOAL_BUDGET``).
MEASURED_FOREIGN_KEYS: tuple[str, ...] = (
    "bass_praesenz",
    "bassdefinition",
    "durchschlagskraft",
    "dynamik",
    "hoehen_luft",
    "klangbalance",
    "makrodynamik",
    "mikrodynamik",
    "punch",
    "raeumlichkeit",
    "stimmklarheit",
    "textverstaendlichkeit",
)

#: Wörtlich abbildbare Fremdschlüssel (der Rest ist bewusst offen).
UNAMBIGUOUS: tuple[str, ...] = ("mikrodynamik", "raeumlichkeit")

#: Wirksame Budget-Zellen für ``metal`` VOR der T3.4-Änderung — Regressionsanker.
METAL_TARGETS_BEFORE: dict[str, float] = {
    "artikulation": 0.255,
    "authentizitaet": 0.135,
    "brillanz": 0.18,
    "emotionalitaet": 0.135,
    "groove": 0.24,
    "mikrodynamik": 0.21,
    "natuerlichkeit": 0.105,
    "raeumlichkeit": 0.15,
    "transparenz": 0.195,
    "waerme": 0.12,
}

#: Ziele, die ``goal_budget`` für ``metal`` verwerfen muss (Reihenfolge des Profils).
METAL_DROPPED: tuple[str, ...] = (
    "punch",
    "bass_praesenz",
    "makrodynamik",
    "textverstaendlichkeit",
    "hoehen_luft",
)


@pytest.fixture(autouse=True)
def _clear_dialect_warn_cache():
    """Die §V6-Warnung wird je (Genre, Ziel) nur einmal gemeldet — Cache leeren."""
    gb._WARNED_DIALECT_KEYS.clear()
    yield
    gb._WARNED_DIALECT_KEYS.clear()


class TestKanonischeAufloesung:
    @pytest.mark.parametrize("goal", ALL_GOAL_NAMES)
    def test_kanonische_namen_sind_identitaet(self, goal):
        assert canonical_goal_name(goal) == goal

    @pytest.mark.parametrize("key", UNAMBIGUOUS)
    def test_woertliche_entsprechungen_sind_abgebildet(self, key):
        mapped = GOAL_DIALECT_MAP[key]
        assert mapped in ALL_GOAL_NAMES
        assert canonical_goal_name(key) == mapped

    @pytest.mark.parametrize("key", [k for k in MEASURED_FOREIGN_KEYS if k not in UNAMBIGUOUS])
    def test_offene_dialekte_liefern_none(self, key):
        assert GOAL_DIALECT_MAP[key] is None
        assert canonical_goal_name(key) is None

    def test_unbekannter_name_liefert_none(self):
        assert canonical_goal_name("gibt_es_nicht") is None


class TestDialektInventar:
    def test_inventar_ist_vollstaendig_und_stabil(self):
        assert set(GOAL_DIALECT_MAP) == set(MEASURED_FOREIGN_KEYS)
        assert len(ALL_GOAL_NAMES) == 15

    def test_jeder_fremdschluessel_hat_eine_begruendung(self):
        assert set(GOAL_DIALECT_NOTES) == set(MEASURED_FOREIGN_KEYS)
        assert all(GOAL_DIALECT_NOTES[k].strip() for k in GOAL_DIALECT_NOTES)

    def test_abgebildete_ziele_sind_kanonisch(self):
        for key, target in GOAL_DIALECT_MAP.items():
            if target is not None:
                assert target in ALL_GOAL_NAMES, f"{key} -> {target} ist nicht kanonisch"

    def test_offene_zuordnungen_sind_dokumentiert_statt_geraten(self):
        """Kein Raten (§V7 copilot-instructions.md): offen bleibt offen + begründet."""
        open_keys = sorted(k for k, v in GOAL_DIALECT_MAP.items() if v is None)
        assert len(open_keys) == 10
        for key in open_keys:
            assert GOAL_DIALECT_NOTES[key].strip(), f"{key} ohne Begründung"


class TestBudgetVerhalten:
    def test_wirksame_ziele_sind_unveraendert(self):
        """T3.4 verändert den KLANG nicht — nur die Sichtbarkeit (§0/Gesamtkonzept)."""
        budget = gb.create_goal_budget(genre_key="metal").to_dict()["budget"]
        applied = {g: v for g, v in budget.items() if g in METAL_TARGETS_BEFORE}
        assert applied == METAL_TARGETS_BEFORE

    def test_verworfene_ziele_werden_gemeldet(self, caplog):
        """§V6 copilot-instructions.md: kein stilles Verwerfen."""
        with caplog.at_level(logging.WARNING, logger="backend.core.goal_budget"):
            gb.create_goal_budget(genre_key="metal")

        meldungen = [r.getMessage() for r in caplog.records if "außerhalb des Budget" in r.getMessage()]
        assert len(meldungen) == len(METAL_DROPPED)
        for goal in METAL_DROPPED:
            assert any(f"'{goal}'" in m for m in meldungen), f"{goal} fehlt in den Meldungen"

    def test_warnung_wird_je_genre_und_ziel_nur_einmal_gemeldet(self, caplog):
        """Erste Nennung sichtbar, Folgenennungen als Debug (keine Log-Flut im Batch)."""
        with caplog.at_level(logging.WARNING, logger="backend.core.goal_budget"):
            gb.create_goal_budget(genre_key="metal")
            erste = len([r for r in caplog.records if r.levelno == logging.WARNING])
            gb.create_goal_budget(genre_key="metal")
            zweite = len([r for r in caplog.records if r.levelno == logging.WARNING])

        assert erste == len(METAL_DROPPED)
        assert zweite == erste

    def test_ohne_genre_bleiben_die_defaults(self):
        budget = gb.create_goal_budget(genre_key="").to_dict()["budget"]
        assert budget == gb._DEFAULT_GOAL_BUDGET
