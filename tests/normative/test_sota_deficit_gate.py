"""Normativ-Gate: SOTA-Defizit-Register (WP-0 der SOTA-Roadmap).

Hält die drei Akzeptanzkriterien des Registers fest und prüft sie gegen
synthetische Register-Texte (Regel-Nachweis) sowie gegen das echte Register
(Bestands-Nachweis). Verankert außerdem den Domänen-Abgleich Register ↔
kuratiertes Manifest (§G9 (copilot-instructions.md)) und die Verdrahtung des
Pre-Commit-Gates.

Quelle: `.github/SOTA_DEFICIT_REGISTER.md`, `scripts/sota_deficit_gate.py`,
`docs/TODOS_SOTA_ROADMAP.md` (Arbeitspakete WP-0…WP-7).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.sota_deficit_gate import (
    GENERIC_EVIDENCE_MARKER,
    REGISTER,
    REQUIRED_CLASSES,
    ROOT,
    audit,
    domain_evidence_ok,
    entry_evidence_ok,
    main,
    parse_register,
    validate,
)

_COPILOT = ROOT / ".github" / "copilot-instructions.md"
_PRECOMMIT = ROOT / ".pre-commit-config.yaml"
_ROADMAP = ROOT / "docs" / "TODOS_SOTA_ROADMAP.md"
_FILE_REGISTRY = ROOT / ".github" / "FILE_REGISTRY.md"
_GATE_SCRIPT = ROOT / "scripts" / "sota_deficit_gate.py"


def _row(
    entry_id: str,
    artefakt: str,
    klasse: str,
    domaene: str,
    evidenz: str,
    status: str,
    blocker: str = "Blocker noch offen",
    beleg: str = "—",
) -> str:
    """Baut eine Registerzeile mit genau acht Zellen."""
    return f"| {entry_id} | {artefakt} | {klasse} | {domaene} | {evidenz} | {status} | {blocker} | {beleg} |"


def _class_row(klasse: str) -> str:
    """Baut eine Klassentabellen-Zeile (Deklaration)."""
    return f"| {klasse} | Testklasse | Definition | Belegquelle |"


def _register_text(rows: list[str], classes: tuple[str, ...] = REQUIRED_CLASSES) -> str:
    """Baut einen minimalen Register-Text mit Klassendeklaration + Einträgen."""
    lines = ["# Test-Register", "", "## Defizit-Klassen", "", "| Klasse | Name | Definition | Quelle |"]
    lines += [_class_row(klasse) for klasse in classes]
    lines.append("")
    lines.append("| ID | Artefakt | Klasse | Domäne | Evidenz | Status | Blocker | Beleg |")
    lines += rows
    return "\n".join(lines) + "\n"


def _valid_row(entry_id: str = "D-K0-1", **overrides: str) -> str:
    """Eine per Default regelkonforme Zeile (Evidenz verweist auf ein echtes Artefakt)."""
    fields = {
        "artefakt": "`models/hifi_gan/hifi_gan.onnx`",
        "klasse": "K0",
        "domaene": "unbekannt",
        "evidenz": "Beleg über `.github/SOTA_DEFICIT_REGISTER.md`",
        "status": "offen",
        "blocker": "Abarbeitung ausstehend",
        "beleg": "—",
    }
    fields.update(overrides)
    return _row(entry_id, **fields)


# ── Bestands-Nachweis: das echte Register ist sauber ─────────────────────────


def test_register_exists_and_passes_the_gate() -> None:
    """Das reale Register erfüllt alle Regeln (Exit 0)."""
    assert REGISTER.is_file(), f"Register fehlt: {REGISTER}"
    result = audit()
    assert result["violations"] == [], f"Register-Verstöße: {result['violations']}"
    assert result["entries"] >= 12, "Das Register muss die belegten Defizite führen (K0–K3)"


@pytest.mark.parametrize("klasse", REQUIRED_CLASSES)
def test_every_deficit_class_is_declared_and_has_entries(klasse: str) -> None:
    """Jede Klasse K0–K3 ist deklariert und hat mindestens einen Eintrag (Regel c)."""
    entries, declared = parse_register(REGISTER.read_text(encoding="utf-8"))
    assert klasse in declared, f"Klasse {klasse} fehlt in der Klassentabelle"
    assert any(entry.klass == klasse for entry in entries), f"Klasse {klasse} hat keinen Eintrag"


def test_every_entry_cites_a_resolvable_path() -> None:
    """Jeder Eintrag zitiert mindestens einen existierenden Repo-Pfad
    (§G8 (copilot-instructions.md))."""
    entries, _ = parse_register(REGISTER.read_text(encoding="utf-8"))
    assert entries, "Register ohne Einträge"
    for entry in entries:
        assert entry_evidence_ok(entry), f"{entry.id}: kein auflösbarer Beleg-Pfad"


def test_music_domain_entries_carry_concrete_evidence() -> None:
    """Domänen musik/sprache/gemischt sind im echten Register konkret belegt (Regel b)."""
    entries, _ = parse_register(REGISTER.read_text(encoding="utf-8"))
    checked = 0
    for entry in entries:
        if entry.domain not in ("musik", "sprache", "gemischt"):
            continue
        checked += 1
        assert domain_evidence_ok(entry), f"{entry.id}: Domänen-Evidenz nicht konkret"
        assert GENERIC_EVIDENCE_MARKER not in entry.evidence, f"{entry.id}: generische Evidenz"
    assert checked >= 8, "Die Domänen-Belege dürfen nicht leerlaufen"


# ── Regel-Nachweis: das Gate schlägt bei Verstößen an ───────────────────────


def test_rule_c_missing_class_declaration_is_reported() -> None:
    """Fehlt eine Klassendeklaration, meldet das Gate (c)."""
    text = _register_text([_valid_row()], classes=("K0", "K1", "K3"))
    entries, declared = parse_register(text)
    problems = validate(entries, declared, curated_domains={})
    assert any("K2" in p and "(c)" in p for p in problems), problems


def test_rule_c_class_without_entry_is_reported() -> None:
    """Eine deklarierte Klasse ohne Eintrag wird gemeldet (c)."""
    text = _register_text([_valid_row("D-K1-1", klasse="K1")])
    entries, declared = parse_register(text)
    problems = validate(entries, declared, curated_domains={})
    assert any("K0" in p and "keinen Registereintrag" in p for p in problems), problems


def test_rule_a_closed_status_requires_existing_beleg() -> None:
    """Status 'geschlossen' ohne existierenden Beleg-Pfad ist ein Verstoß (a)."""
    text = _register_text([_valid_row(status="geschlossen", beleg="—")])
    entries, declared = parse_register(text)
    problems = validate(entries, declared, curated_domains={})
    assert any(p.startswith("(a)") for p in problems), problems


def test_rule_a_closed_status_with_existing_beleg_passes() -> None:
    """Status 'geschlossen' mit existierendem Beleg-Pfad ist zulässig."""
    text = _register_text([_valid_row(status="geschlossen", beleg="`scripts/sota_deficit_gate.py`")])
    entries, declared = parse_register(text)
    problems = validate(entries, declared, curated_domains={})
    assert not [p for p in problems if p.startswith("(a)")], problems


def test_rule_b_music_domain_requires_concrete_evidence() -> None:
    """Domäne 'musik' mit generischer Evidenz wird gemeldet (b)."""
    text = _register_text([_valid_row(domaene="musik", evidenz=GENERIC_EVIDENCE_MARKER)])
    entries, declared = parse_register(text)
    problems = validate(entries, declared, curated_domains={})
    assert any(p.startswith("(b)") for p in problems), problems


def test_rule_b_music_domain_with_missing_path_is_reported() -> None:
    """Domäne 'sprache' mit nicht existierendem Evidenz-Pfad wird gemeldet (b)."""
    text = _register_text([_valid_row(domaene="sprache", evidenz="belegt in `models/gibts_nicht/weg.onnx`")])
    entries, declared = parse_register(text)
    problems = validate(entries, declared, curated_domains={})
    assert any(p.startswith("(b)") for p in problems), problems


def test_row_without_any_resolvable_path_is_reported() -> None:
    """Ohne auflösbaren Pfad in der Zeile greift die Evidenz-Regel
    (§G8 (copilot-instructions.md))."""
    text = _register_text(
        [
            _row(
                "D-K0-1",
                "Phantom-Artefakt",
                "K0",
                "unbekannt",
                "kein Pfad, keine Quelle",
                "offen",
                "Blocker",
                "—",
            )
        ]
    )
    entries, declared = parse_register(text)
    problems = validate(entries, declared, curated_domains={})
    assert any("(Evidenz)" in p for p in problems), problems


def test_unknown_status_and_domain_are_reported() -> None:
    """Unbekannter Status/Domäne wird gemeldet (Enum-Zwang)."""
    text = _register_text([_valid_row(status="erledigt", domaene="sprachmodell")])
    entries, declared = parse_register(text)
    problems = validate(entries, declared, curated_domains={})
    assert any(p.startswith("(Status)") for p in problems), problems
    assert any(p.startswith("(Domäne)") for p in problems), problems


def test_id_class_mismatch_is_reported() -> None:
    """ID-Präfix und Klasse-Spalte müssen zusammenpassen."""
    text = _register_text([_valid_row("D-K2-9", klasse="K0")])
    entries, declared = parse_register(text)
    problems = validate(entries, declared, curated_domains={})
    assert any("ID-Klasse" in p for p in problems), problems


def test_duplicate_ids_are_reported() -> None:
    """Doppelte Eintrags-IDs werden gemeldet."""
    text = _register_text([_valid_row("D-K0-1"), _valid_row("D-K0-1")])
    entries, declared = parse_register(text)
    problems = validate(entries, declared, curated_domains={})
    assert any("doppelte Eintrags-ID" in p for p in problems), problems


def test_accept_status_requires_reason() -> None:
    """'bewusst-akzeptiert' ohne Begründung im Blocker-Feld wird gemeldet."""
    text = _register_text([_valid_row(status="bewusst-akzeptiert", blocker="n/a")])
    entries, declared = parse_register(text)
    problems = validate(entries, declared, curated_domains={})
    assert any("bewusst-akzeptiert" in p for p in problems), problems


def test_domain_crosscheck_against_curated_manifest() -> None:
    """Widerspruch zur kuratierten Manifest-Domäne wird gemeldet
    (§G9 (copilot-instructions.md) — eine Quelle)."""
    text = _register_text([_valid_row(artefakt="`models/hifi_gan/hifi_gan.onnx`", domaene="musik")])
    entries, declared = parse_register(text)
    problems = validate(entries, declared, curated_domains={"models/hifi_gan/hifi_gan.onnx": "sprache"})
    assert any("§G9 (copilot-instructions.md)" in p for p in problems), problems


def test_malformed_row_is_rejected() -> None:
    """Eine Zeile mit falscher Zellenzahl ist ein Formatfehler."""
    broken = "| D-K0-1 | Phantom | K0 | unbekannt | Evidenz `models/hifi_gan/hifi_gan.onnx` | offen | Blocker |"
    text = _register_text([broken])  # 7 Zellen statt 8 (Beleg-Spalte fehlt)
    with pytest.raises(ValueError):
        parse_register(text)


def test_missing_register_is_a_violation(tmp_path: Path) -> None:
    """Fehlt das Register, liefert audit() einen Formatverstoß."""
    result = audit(tmp_path / "fehlt.md")
    assert result["violations"] and "Register fehlt" in result["violations"][0]


# ── Domänen-Ehrlichkeit: unbelegte Sprach-Zuordnung ist gesperrt ────────────


def test_curated_manifest_marks_unproven_vocoders_as_unknown() -> None:
    """bigvgan/hifi_gan sind 'unbekannt' — die Sprach-Zuordnung ist am Artefakt
    widerlegt und der Korpus ist lokal nicht belegbar.

    Messung 2026-10-06 (§III.13 (copilot-instructions.md) — Evidenzpflicht):
    `bigvgan_v2.onnx` hat 128 Mel-Bänder/512×/1536 Kanäle (44,1-kHz-Konfiguration,
    nicht LibriTTS = 100 Bänder/256×/512 Kanäle); `hifi_gan.onnx` hat 80 Mel-Bänder,
    256× und nur 128/64 Kanäle (0,93 M Parameter — kein offizieller V1 mit
    512 Kanälen/13,9 M). Damit darf weder 'sprache' noch 'musik' behauptet werden.
    """
    from scripts.model_inventory import domain_report

    models = domain_report()["models"]
    for path in (
        "models/hifi_gan/hifi_gan.onnx",
        "models/bigvgan/bigvgan_v2.onnx",
    ):
        assert models[path]["domain"] == "unbekannt", path
        assert models[path]["domain"] != "sprache", path
        # Drift-Fix gesperrt: die Evidenz MUSS die Artefakt-Messung nennen.
        assert "Mel" in models[path]["evidence"], path


# ── Verankerung + CLI ──────────────────────────────────────────────────────


def test_gate_is_wired_into_pre_commit() -> None:
    """Das Gate ist als Pre-Commit-Hook verdrahtet (maschinelle Wahrheit)."""
    assert _PRECOMMIT.is_file()
    text = _PRECOMMIT.read_text(encoding="utf-8")
    assert "aurik-sota-deficit-gate" in text, "Hook-ID fehlt"
    assert "scripts/sota_deficit_gate.py" in text, "Hook ruft das Gate nicht auf"


def test_gate_script_is_registered_in_file_registry() -> None:
    """Neue Code-Datei braucht einen FILE_REGISTRY-Eintrag (Write-Gate)."""
    assert _GATE_SCRIPT.is_file()
    text = _FILE_REGISTRY.read_text(encoding="utf-8")
    assert "scripts/sota_deficit_gate.py" in text, "FILE_REGISTRY-Eintrag fehlt"


def test_register_is_reachable_from_roadmap_and_anchors() -> None:
    """Register nennt seine normativen Anker; die Roadmap nennt das Register (WP-0)."""
    register_text = REGISTER.read_text(encoding="utf-8")
    assert "ML_MODEL_DOMAIN_REGISTRY.md" in register_text, "Evidenz-Anker fehlt"
    assert "sota_deficit_gate.py" in register_text, "Gate-Verweis fehlt"
    assert _ROADMAP.is_file()
    assert "SOTA_DEFICIT_REGISTER" in _ROADMAP.read_text(encoding="utf-8"), "Roadmap nennt das Register nicht"
    assert "ML-Domänen-Registry" in _COPILOT.read_text(encoding="utf-8"), "§III.13 (copilot-instructions.md) fehlt"


def test_cli_exit_codes(capsys: pytest.CaptureFixture[str]) -> None:
    """CLI: sauberes Register → 0; fehlendes Register → 1."""
    assert main(["--quiet"]) == 0
    assert main(["--quiet", "--register", str(ROOT / "gibts_nicht.md")]) == 1
    assert main(["--json"]) == 0
    out = capsys.readouterr().out
    assert '"violations"' in out
