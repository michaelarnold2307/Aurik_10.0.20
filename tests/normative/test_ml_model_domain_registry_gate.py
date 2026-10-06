"""Normativ-Gate: ML-Modell-Domänen-Registry (§III.13 copilot-instructions.md).

Zweck: Der häufigste KI-Fehlschluss in diesem Repo — „ein Großteil der lokal
verfügbaren ML-Modelle sind Sprachmodelle" — darf nicht wieder in die
Dokumentation zurückkehren. Dieses Gate hält die kanonische Registry, ihre
Verankerung in der normativen Kette und die belegte Musik-Dominanz fest.

Quelle: `.github/ML_MODEL_DOMAIN_REGISTRY.md`,
`.github/instructions/ml_domain.instructions.md`,
`scripts/model_inventory.py::domain_report`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_REGISTRY = _ROOT / ".github" / "ML_MODEL_DOMAIN_REGISTRY.md"
_INSTRUCTION = _ROOT / ".github" / "instructions" / "ml_domain.instructions.md"
_AGENTS = _ROOT / "AGENTS.md"
_COPILOT = _ROOT / ".github" / "copilot-instructions.md"
_CLAUDE = _ROOT / "CLAUDE.md"

_FORBIDDEN_CLAIM = "die meisten modelle sind sprache"
_MUSIC_DOMINANCE_MARKER = "musik-trainiert"


@pytest.fixture(scope="module")
def registry_text() -> str:
    assert _REGISTRY.is_file(), f"Kanonische Registry fehlt: {_REGISTRY}"
    return _REGISTRY.read_text(encoding="utf-8")


def test_registry_states_the_counter_fact(registry_text: str) -> None:
    """Die Registry benennt den Fehlschluss als FALSCH und belegt Musik-Dominanz."""
    lowered = registry_text.lower()
    assert "falsch" in lowered, "Die Registry muss den Sprach-Fehlschluss als FALSCH markieren"
    assert _MUSIC_DOMINANCE_MARKER in lowered, "Die Registry muss die Musik-Dominanz belegen"
    assert "37" in registry_text and "61" in registry_text, "Kopfzahlen (37 von 61) müssen belegt sein"
    assert "sprach-trainiert" in lowered, "Die Registry muss auch die Sprach-Modelle ausweisen"


def test_registry_explicitly_forbids_the_claim(registry_text: str) -> None:
    """Der Fehlschluss ist als quotierbares Verbot dokumentiert."""
    lowered = registry_text.lower()
    assert _FORBIDDEN_CLAIM in lowered, "Der Fehlschluss muss wörtlich zitiert und widerlegt werden"
    assert "verbotene fehlschl" in lowered, "Es braucht einen Abschnitt 'Verbotene Fehlschlüsse'"


def test_instruction_file_autoloads_for_ml_code() -> None:
    """Die auto-geladene Instruktion deckt ML-Pfade ab (plugins/backend/denker/models)."""
    assert _INSTRUCTION.is_file(), f"Instruktion fehlt: {_INSTRUCTION}"
    text = _INSTRUCTION.read_text(encoding="utf-8")
    assert text.startswith("---"), "Die Instruktion braucht YAML-Frontmatter mit applyTo"
    frontmatter = text.split("---", 2)[1]
    for scope in ("plugins/", "backend/", "denker/", "models/"):
        assert scope in frontmatter, f"applyTo muss '{scope}' abdecken"


@pytest.mark.parametrize("path", [_AGENTS, _COPILOT, _CLAUDE])
def test_registry_is_anchored_in_normative_chain(path: Path) -> None:
    """Registry ist aus der normativen Kette erreichbar (AGENTS/§III.13/CLAUDE)."""
    assert path.is_file(), f"Normative Datei fehlt: {path}"
    text = path.read_text(encoding="utf-8")
    assert "ML_MODEL_DOMAIN_REGISTRY" in text, f"{path.name} muss auf die Registry verweisen"


def test_normative_rule_iii13_exists() -> None:
    """§III.13 (copilot-instructions.md) benennt den Fehlschluss als VERBOTEN."""
    text = _COPILOT.read_text(encoding="utf-8")
    assert "ML-Domänen-Registry" in text, "§III.13 fehlt in copilot-instructions.md"
    assert "Sprachmodelle" in text, "§III.13 muss den Fehlschluss benennen"


def test_domain_report_music_dominates() -> None:
    """Maschinen-Sicht: Musik dominiert klar gegenüber Sprache."""
    from scripts.model_inventory import domain_report

    report = domain_report()
    totals = report["totals_by_domain"]
    assert totals["musik"] > totals["sprache"], "Musik muss Sprache klar dominieren (§III.13)"
    assert sum(totals.values()) == report["totals_all_dirs"], "Domänen-Summe muss den Gesamtbestand treffen"
    assert report["registry"] == ".github/ML_MODEL_DOMAIN_REGISTRY.md"


def test_domain_report_has_evidence_for_every_model() -> None:
    """Jedes kuratierte Modell hat eine nicht-leere Domäne + Evidenz (§G8 copilot-instructions.md)."""
    from scripts.model_inventory import domain_report

    for path, entry in domain_report()["models"].items():
        assert entry["domain"], f"{path} ohne Domäne"
        assert entry["evidence"], f"{path} ohne Evidenz"
