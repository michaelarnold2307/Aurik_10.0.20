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
    assert "schaden_pfad.is_file()" in q
    assert "restauriert_pfad.is_file()" in q
    assert "restauriert_pfad.stat().st_size > 0" in q


def test_crackle_resume_rebuilds_report_and_clean_links(tmp_path, monkeypatch) -> None:
    import json

    import numpy as np

    from scripts import erstelle_corpus_vinyl as corpus
    from scripts import erstelle_crackle_expansion as crackle
    from scripts import erstelle_schadens_proben as schadens_proben

    root = tmp_path
    quellen = root / "quellen"
    ausgang = root / "ausgang"
    clean_links = root / "clean"
    bank = root / "bank"
    report_dir = root / "reports"
    for idx in range(3):
        titel = f"Titel {idx}"
        quelle = quellen / f"{titel}.wav"
        quelle.parent.mkdir(parents=True, exist_ok=True)
        quelle.write_bytes(b"source")
        for suffix, payload in (("(vinyl).wav", b"degraded"), ("(vinyl)_restauriert.wav", b"restored")):
            datei = ausgang / f"{titel} {suffix}"
            datei.parent.mkdir(parents=True, exist_ok=True)
            datei.write_bytes(b"" if idx == 0 and suffix == "(vinyl)_restauriert.wav" else payload)
    for idx in range(2):
        (bank / f"noise_{idx}.wav").parent.mkdir(parents=True, exist_ok=True)
        (bank / f"noise_{idx}.wav").write_bytes(b"noise")

    monkeypatch.setattr(crackle, "_ROOT", root)
    monkeypatch.setattr(crackle, "REPORT_DIR", report_dir)
    monkeypatch.setattr(
        crackle,
        "_lade_mono_48k",
        lambda _quelle: (np.ones(8, dtype=np.float32), crackle.SR),
    )
    monkeypatch.setattr(
        corpus,
        "_vinyl_rezept",
        lambda clean, _bank, _rng: (clean, {"scratch": False, "staerke": np.float32(0.5)}),
    )
    monkeypatch.setattr(schadens_proben, "restauriere_schnipsel", lambda audio, *_args: audio)
    report_dir.mkdir()
    assert (
        crackle.main(
            [
                "--titel",
                "3",
                "--seed",
                "42",
                "--quellen",
                str(quellen),
                "--ausgang",
                str(ausgang),
                "--clean-links",
                str(clean_links),
                "--bank",
                str(bank),
            ]
        )
        == 0
    )

    reports = list(report_dir.glob("*_crackle_expansion.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert len(report["eintraege"]) == 3
    assert {entry["titel"] for entry in report["eintraege"]} == {f"Titel {idx}" for idx in range(3)}
    assert all(entry["rezept_params"]["staerke"] == 0.5 for entry in report["eintraege"])
    assert (ausgang / "Titel 0 (vinyl)_restauriert.wav").stat().st_size > 0
    assert all(
        (clean_links / f"Titel {idx} original.wav").resolve() == (quellen / f"Titel {idx}.wav").resolve()
        for idx in range(3)
    )


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
