"""Genre-Single-Source-Guard (§G9 copilot-instructions.md) — Regressionen.

Deckt ab:
  * R1  Genre-Alias-Tabelle nur in ``backend/core/genre_registry.py``
  * R2  alle Schlüssel beider Ziel-Gewichtstabellen lösen über die Registry auf
  * R3  XOR-Invariante ``goal_profile_key(g) is None`` ⟺ ``g in _GOAL_KEY_ABSENT``
  * R4  Divergenz-Bericht (Transparenz, kein Fail)
  * Verhaltensgleichheit der konsolidierten Anzeige-Taxonomie (Tranche 3.3)

Befund 2026-10-06: Vor dieser Tranche existierten FÜNF parallele
Genre-Auflösungen; die fünfte (``phase_53_semantic_audio._GENRE_ALIAS_MAP``)
wurde verhaltensgleich in die Registry verschoben (47 Referenz-Proben).
"""

from __future__ import annotations

import pytest

from scripts import genre_single_source_check as guard

# ---------------------------------------------------------------------------
# R1 — einzige Alias-Quelle
# ---------------------------------------------------------------------------


class TestR1AliasSingleSource:
    def test_keine_zweite_alias_tabelle_im_repo(self):
        """Der Ist-Stand muss konform sein (fail-closed-Basis für den Hook)."""
        assert guard._check_alias_single_source() == []

    def test_zweite_alias_tabelle_wird_erkannt(self, tmp_path, monkeypatch):
        """Negativtest: eine neue Alias-Tabelle in einem Scan-Verzeichnis greift."""
        fake_repo = tmp_path
        target = fake_repo / "backend" / "core"
        target.mkdir(parents=True)
        (target / "probe.py").write_text('GENRE_ALIASES_EXTRA = {"techno": "electronic"}\n', encoding="utf-8")
        monkeypatch.setattr(guard, "_REPO", fake_repo)
        monkeypatch.setattr(guard, "_SCAN_DIRS", ("backend",))

        problems = guard._check_alias_single_source()

        assert len(problems) == 1
        assert "GENRE_ALIASES_EXTRA" in problems[0]
        assert "genre_registry.py" in problems[0]

    def test_registry_selbst_ist_ausgenommen(self, tmp_path, monkeypatch):
        """Die Registry darf ihre eigene Tabelle natürlich führen."""
        target = tmp_path / "backend" / "core"
        target.mkdir(parents=True)
        (target / "genre_registry.py").write_text('GENRE_ALIASES = {"klassik": "klassik"}\n', encoding="utf-8")
        monkeypatch.setattr(guard, "_REPO", tmp_path)
        monkeypatch.setattr(guard, "_SCAN_DIRS", ("backend",))

        assert guard._check_alias_single_source() == []


# ---------------------------------------------------------------------------
# R2 / R3 — Schlüsselraum und deklarierte Ausnahmen
# ---------------------------------------------------------------------------


class TestR2R3Struktur:
    def test_alle_profil_schluessel_aufloesbar(self):
        assert guard._check_profile_keys_resolvable() == []

    def test_ausnahmeliste_konsistent(self):
        """XOR: fehlendes Ziel-Profil ⟺ deklariert in _GOAL_KEY_ABSENT."""
        assert guard._check_absent_xor() == []

    def test_ausnahmeliste_ist_nicht_leer(self):
        """Schutz gegen einen Check, der nur zufällig grün ist."""
        from backend.core import genre_registry as registry

        assert len(registry._GOAL_KEY_ABSENT) == 11
        assert len(registry.CANONICAL_GENRES) == 21

    def test_kanonische_genres_loesen_auf_sich_selbst_auf(self):
        from backend.core import genre_registry as registry

        for genre in registry.CANONICAL_GENRES:
            assert registry.normalize_genre(genre) == genre


# ---------------------------------------------------------------------------
# R4 — Divergenz-Bericht ist vorhanden und stabil
# ---------------------------------------------------------------------------


class TestR4DivergenzBericht:
    def test_bericht_meldet_die_dokumentierte_divergenz(self):
        cells, lines = guard._divergence_report()

        assert cells > 0, "Die Rollen-Divergenz muss sichtbar bleiben (§G8 copilot-instructions.md)"
        assert len(lines) == 7, "7 gemeinsam geführte Genres"
        assert cells == 115, "dokumentierter Stand 2026-10-06"


# ---------------------------------------------------------------------------
# Konsolidierte Anzeige-Taxonomie — verhaltensgleich, Verluste dokumentiert
# ---------------------------------------------------------------------------


class TestSemanticHintLabel:
    @pytest.mark.parametrize(
        ("label", "expected"),
        [
            (None, "Unbekannt"),
            ("", "Unbekannt"),
            ("unknown", "Unbekannt"),
            ("classical", "Klassik"),
            ("orchestra", "Klassik"),
            ("opera", "Oper"),
            ("jazz_acoustic", "Jazz"),
            ("hip_hop", "Hip-Hop"),
            ("rnb", "Soul/R&B"),
            ("schlager", "Schlager"),
            ("Techno", "Electronic"),
            ("big band jazz", "Jazz"),
        ],
    )
    def test_tabelle_und_varianten(self, label, expected):
        from backend.core.genre_registry import semantic_hint_label

        assert semantic_hint_label(label) == expected

    @pytest.mark.parametrize(
        ("label", "expected"),
        [
            # Bewusst konservierte, DOKUMENTIERTE Verluste gegenüber der Registry.
            # Eine Korrektur wäre eine Verhaltensänderung → A/B + Sign-off nötig.
            ("metal", "Rock"),
            ("country", "Folk"),
            ("ambient", "Electronic"),
            ("latin", "Unbekannt"),
            # Substring-Reihenfolge: "class" wird vor "rock" geprüft.
            ("classic_rock", "Klassik"),
        ],
    )
    def test_dokumentierte_verluste_sind_gepinnt(self, label, expected):
        from backend.core.genre_registry import semantic_hint_label

        assert semantic_hint_label(label) == expected

    def test_phase_53_delegiert_verhaltensgleich(self):
        from backend.core.genre_registry import semantic_hint_label
        from backend.core.phases.phase_53_semantic_audio import _canonicalize_genre_hint

        probe = ["classic_rock", "heavy metal", "soul", "", None, "trailer", "DUB"]
        for label in probe:
            assert _canonicalize_genre_hint(label) == semantic_hint_label(label)


# ---------------------------------------------------------------------------
# R5 — Ziel-Dialekte (T3.4-Befund 2026-10-06)
# ---------------------------------------------------------------------------


class TestR5ZielDialekte:
    def test_alle_ziel_schluessel_des_repos_sind_deklariert(self):
        assert guard._check_goal_dialects_declared() == []

    def test_neuer_undeklarierter_dialekt_wird_erkannt(self):
        """Ein fünftes Vokabular darf nicht unbemerkt entstehen."""
        problems = guard._undeclared_goal_keys({"probe_tabelle": {"voellig_neues_ziel"}})

        assert len(problems) == 1
        assert "voellig_neues_ziel" in problems[0]
        assert "GOAL_DIALECT_MAP" in problems[0]

    def test_kanonische_und_deklarierte_schluessel_sind_erlaubt(self):
        from backend.core.song_goal_importance import ALL_GOAL_NAMES, GOAL_DIALECT_MAP

        erlaubt = set(ALL_GOAL_NAMES) | set(GOAL_DIALECT_MAP)
        assert erlaubt, "Schlüsselmengen dürfen nicht leer sein (sonst grüner Leerlauf)"
        assert guard._undeclared_goal_keys({"probe": erlaubt}) == []
