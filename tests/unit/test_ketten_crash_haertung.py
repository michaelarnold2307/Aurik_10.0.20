"""Vertragstests: Crash-Härtung der Mess-Ketten (2026-11-19).

Befund: Ein Neustart der Kun-Laufzeit riss alle Hintergrund-Ketten mit (keine
Reboots/OOM/Tracebacks — Prozesse wurden extern terminiert). Härtung:
(1) Resume-Sicherheit — fertige Paare/Titel werden übersprungen, ein
abgebrochener Lauf setzt ohne Zeitverlust fort; (2) Faulthandler — native
Crashes (SIGSEGV) landen im Log statt spurlos zu verschwinden; (3) die
Ketten laufen detached (setsid/nohup, Init-Reparenting) — das ist eine
Start-Konvention und wird hier als Runbook-Pin dokumentiert.
"""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _src(name: str) -> str:
    return (_ROOT / "scripts" / name).read_text(encoding="utf-8")


def test_erstelle_resume_ueberspringt_fertige_paare() -> None:
    import re

    q = _src("erstelle_schadens_proben.py")
    assert "_rest_check = ausgang /" in q
    assert "schaden_pfad.stat().st_size > 0" in q
    m = re.search(
        r"_rest_check = ausgang /.*?\n"
        r"\s+if schaden_pfad\.is_file\(\) and _rest_check\.is_file\(\) and schaden_pfad\.stat\(\)\.st_size > 0:\n"
        r"\s+.*?manifest\[\"songs\"\]\.append\(\n"
        r"\s+\{\n"
        r"\s+\"quelle\": str\(stueck\),\n"
        r"\s+\"stamm\": stamm,\n"
        r"\s+\"rezept\": rezept,\n"
        r"\s+\"schaden\": str\(schaden_pfad\),\n"
        r"\s+\"restauriert\": str\(_rest_check\),\n"
        r"\s+\}\n"
        r"\s+\)\n"
        r"\s+continue",
        q,
        re.S,
    )
    assert m, "Resume-Skip-Block fehlt oder falsch geordnet (Manifest-Nachtrag vor continue fehlt)"


def test_crackle_resume_ueberspringt_fertige_titel() -> None:
    q = _src("erstelle_crackle_expansion.py")
    assert "bereits fertig, übersprungen (Resume)" in q
    assert "schaden_pfad.is_file() and restauriert_pfad.is_file()" in q


def test_faulthandler_in_allen_ketten_skripten() -> None:
    for name in ("erstelle_schadens_proben.py", "bewerte_schadens_proben.py", "erstelle_crackle_expansion.py"):
        q = _src(name)
        assert "faulthandler" in q, name
        assert "python_faulthandler_ketten.log" in q, name
        assert "all_threads=True" in q, name


def test_ketten_runbook_detached_start() -> None:
    """Die Ketten müssen setsid/nohup-detached starten (überlebt Laufzeit-Neustarts)."""
    handover = (_ROOT / "docs" / "SOTA_HANDOVER.md").read_text(encoding="utf-8")
    assert "setsid nohup" in handover
    assert "Init-Reparenting" in handover or "detached" in handover
