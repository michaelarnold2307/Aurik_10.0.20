"""Regressionstest: Sprache-Guard — kanonische Phasen-IDen bleiben unübersetzt.

§5 AGENTS.md: kanonische Phasen-IDs (`phase_<Ziffer>`, z.B. `phase_20`) sind
Vertragstext und dürfen nicht in `Verarbeitungsschritt_<Ziffer>` übersetzt
werden. Der Sprache-Guard (`scripts/pre_commit_language_guard.py`) hatte sie
wortweise ersetzt (3. Vorkommen: 64d95d08, b785c434, 655a145b). Die Härtung in
`_replace_forbidden_terms` schließt die Phasen-ID-Form via `(?!_\\d)` aus.
"""

from __future__ import annotations

from scripts.pre_commit_language_guard import _replace_forbidden_terms


class TestKanonischePhasenIDsNichtUebersetzt:
    """Phase_<Ziffer> ist Vertragstext und bleibt unverändert."""

    def test_phase_20_rollback_keyword_bleibt(self) -> None:
        out = _replace_forbidden_terms("§0p Formant drift phase_20 (F1 200→150 Hz) — rollback")
        assert "phase_20" in out
        assert "Verarbeitungsschritt_20" not in out

    def test_phase_49_rollback_keyword_bleibt(self) -> None:
        out = _replace_forbidden_terms("§0p Formant drift phase_49 (F1 200→150 Hz) — rollback")
        assert "phase_49" in out
        assert "Verarbeitungsschritt_49" not in out

    def test_phase_67_textur_keyword_bleibt(self) -> None:
        out = _replace_forbidden_terms("phase_67 crackle texture removal")
        assert "phase_67" in out

    def test_kanonische_id_mit_leading_underscore_bleibt(self) -> None:
        out = _replace_forbidden_terms("Verarbeitungsschritt_20 already translated")
        assert out == "Verarbeitungsschritt_20 already translated"


class TestFreistehendesPhaseWirdUebersetzt:
    """Allgemeines 'phase' (ohne ID-Form) wird weiterhin übersetzt."""

    def test_freistehendes_phase_wird_verarbeitungsschritt(self) -> None:
        out = _replace_forbidden_terms("the phase is complete")
        assert "Verarbeitungsschritt" in out
        assert " phase " not in f" {out} "

    def test_mischung_id_bleibt_wort_wird_uebersetzt(self) -> None:
        out = _replace_forbidden_terms("phase_20: the phase runs")
        assert "phase_20" in out
        assert "Verarbeitungsschritt" in out
