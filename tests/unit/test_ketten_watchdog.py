"""Vertragstests: Ketten-Watchdog (scripts/ketten_watchdog.py, 2026-11-19).

Härtungs-Schicht 4: der Watchdog prüft je Kette (Prozess läuft? END-Marker im
Log?) und startet nur bei „tot UND nicht fertig“ detached neu. Modell-frei:
Entscheidungslogik wird ohne echte Prozesse gepinnt.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import ketten_watchdog as wd


def test_end_marker_erkannt(tmp_path) -> None:
    log = tmp_path / "kette.log"
    log.write_text("läuft noch\n", encoding="utf-8")
    assert not wd._end_marker_da(str(log), "KETTE_END=")
    log.write_text("KETTE_END=0\n", encoding="utf-8")
    assert wd._end_marker_da(str(log), "KETTE_END=")


def test_end_status_fehlender_marker_liefert_none(tmp_path) -> None:
    log = tmp_path / "kette.log"
    log.write_text("läuft noch\n", encoding="utf-8")
    assert wd._end_status(str(log), "KETTE_END=") is None
    assert wd._end_status(str(tmp_path / "fehlt.log"), "KETTE_END=") is None


def test_end_status_unterscheidet_sauber_von_abgesturzt(tmp_path) -> None:
    """Härtungs-Schicht 4b: der ursprüngliche Bug behandelte JEDEN Marker als
    'fertig' — ein Absturzstatus (z. B. 139 = SIGSEGV) muss als Absturz, nicht
    als sauberer Abschluss erkannt werden."""
    log = tmp_path / "kette.log"
    log.write_text("KETTE_END=0\n", encoding="utf-8")
    assert wd._end_status(str(log), "KETTE_END=") == 0
    log.write_text("KETTE_END=139\n", encoding="utf-8")
    assert wd._end_status(str(log), "KETTE_END=") == 139


def test_end_status_nimmt_den_letzten_versuch(tmp_path) -> None:
    """Mehrere Neustarts hängen je eine eigene Markerzeile an — maßgeblich
    für den aktuellen Zustand ist die ZULETZT geloggte."""
    log = tmp_path / "kette.log"
    log.write_text("KETTE_END=139\nweiter...\nKETTE_END=0\n", encoding="utf-8")
    assert wd._end_status(str(log), "KETTE_END=") == 0


def test_neuer_start_verwirft_alten_erfolg_bis_zum_abschluss(tmp_path) -> None:
    log = tmp_path / "kette.log"
    marker = "KETTE_END="
    log.write_text(f"{marker}0\n{wd._start_marker(marker)}\n", encoding="utf-8")
    assert wd._end_status(str(log), marker) is None
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"{marker}139\n")
    assert wd._end_status(str(log), marker) == 139
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"{wd._start_marker(marker)}\n{marker}0\n")
    assert wd._end_status(str(log), marker) == 0


def test_warten_ignoriert_alten_erfolg_waehrend_neuem_lauf(tmp_path, monkeypatch) -> None:
    log = tmp_path / "kette.log"
    log.write_text("KETTE_END=0\n", encoding="utf-8")
    monkeypatch.setattr(wd, "KETTEN_CONFIG", [("kette", "prozess", "", str(log), "KETTE_END=")])
    wartezyklen = []
    monkeypatch.setattr(wd, "_prozess_laeuft", lambda muster: not wartezyklen)

    def schlafen(_sekunden):
        wartezyklen.append(1)
        if len(wartezyklen) == 2:
            with log.open("a", encoding="utf-8") as fh:
                fh.write("KETTE_END=0\n")
        elif len(wartezyklen) > 2:
            raise AssertionError("Frischer Erfolgsmarker wurde nicht erkannt")

    monkeypatch.setattr(wd.time, "sleep", schlafen)
    wd._warte_auf_vorgaenger("kette")
    assert len(wartezyklen) == 2


def test_unbekannter_vorgaenger_wird_abgewiesen() -> None:
    with pytest.raises(ValueError, match="Unbekannte Vorgängerkette"):
        wd._warte_auf_vorgaenger("nicht-vorhanden")


def test_neustart_markiert_versuch_vor_dem_prozessstart(tmp_path, monkeypatch) -> None:
    log = tmp_path / "kette.log"
    log.write_text("KETTE_END=1\n", encoding="utf-8")
    monkeypatch.setattr(wd, "KETTEN_CONFIG", [("kette", "prozess", "kommando", str(log), "KETTE_END=")])
    monkeypatch.setattr(wd, "LOCKFILE", tmp_path / "watchdog.lock")
    monkeypatch.setattr(wd, "LOGFILE", tmp_path / "watchdog.log")
    monkeypatch.setattr(wd, "_prozess_laeuft", lambda muster: False)

    def starten(*args, **kwargs):
        assert wd._end_status(str(log), "KETTE_END=") is None

    def ende(_sekunden):
        raise StopIteration

    monkeypatch.setattr(wd.subprocess, "Popen", starten)
    monkeypatch.setattr(wd.time, "sleep", ende)
    with pytest.raises(StopIteration):
        wd.main()


def test_anzahl_abgestuerzter_versuche_zaehlt_fehlschlaege_seit_letztem_erfolg(tmp_path) -> None:
    log = tmp_path / "kette.log"
    log.write_text("KETTE_END=139\nKETTE_END=137\nKETTE_END=0\n", encoding="utf-8")
    assert wd._anzahl_abgestuerzter_versuche(str(log), "KETTE_END=") == 0
    log.write_text("KETTE_END=139\nKETTE_END=0\nKETTE_END=139\n", encoding="utf-8")
    assert wd._anzahl_abgestuerzter_versuche(str(log), "KETTE_END=") == 1
    log.write_text("KETTE_END=0\n", encoding="utf-8")
    assert wd._anzahl_abgestuerzter_versuche(str(log), "KETTE_END=") == 0


def test_anzahl_abgestuerzter_versuche_fehlende_datei_ist_null(tmp_path) -> None:
    assert wd._anzahl_abgestuerzter_versuche(str(tmp_path / "fehlt.log"), "KETTE_END=") == 0


def test_max_crash_restarts_ist_endlich_begrenzt() -> None:
    """Keine Endlosschleife bei dauerhaft kaputtem Kommando — der Guard muss
    eine endliche, positive Obergrenze sein."""
    assert isinstance(wd.MAX_CRASH_RESTARTS, int)
    assert 0 < wd.MAX_CRASH_RESTARTS < 100


def test_prozess_check_ohne_self_match(monkeypatch) -> None:
    import subprocess

    eigene = str(12345)
    monkeypatch.setattr(wd.os, "getpid", lambda: 12345)
    monkeypatch.setattr(
        wd.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(args=[], returncode=0, stdout=f"{eigene}\n"),
    )
    assert not wd._prozess_laeuft("muster"), "eigene PID darf nicht als Ketten-Prozess zählen"


def test_prozess_check_fremde_pid_zaehlt(monkeypatch) -> None:
    import subprocess

    monkeypatch.setattr(wd.os, "getpid", lambda: 1)
    monkeypatch.setattr(
        wd.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(args=[], returncode=0, stdout="9999\n"),
    )
    assert wd._prozess_laeuft("muster")


def test_prozess_check_fail_safe_bei_fehler(monkeypatch) -> None:
    """Unklare Prozess-Sicht (pgrep-Fehler) ⇒ KEIN Neustart (fail-safe True)."""

    def _kaputt(*a, **k):
        raise FileNotFoundError("pgrep fehlt")

    monkeypatch.setattr(wd.subprocess, "run", _kaputt)
    assert wd._prozess_laeuft("muster") is True


def test_prozess_check_fail_safe_bei_pgrep_fehlerstatus(monkeypatch) -> None:
    import subprocess

    monkeypatch.setattr(
        wd.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(args=[], returncode=2, stdout="", stderr="Fehler"),
    )
    assert wd._prozess_laeuft("muster") is True


def test_konfiguration_komplett_und_marker_konsistent() -> None:
    """9 Ketten, je mit eigenem Log + eigenem END-Marker (keine Kollision)."""
    assert len(wd.KETTEN_CONFIG) == 11
    marker = {e[4] for e in wd.KETTEN_CONFIG}
    logs = {e[3] for e in wd.KETTEN_CONFIG}
    assert len(marker) == 11 and len(logs) == 11
    for e in wd.KETTEN_CONFIG:
        assert e[4].endswith("END="), e[0]


def test_gegatete_ketten_warten_auf_vorgaenger() -> None:
    """Abhängige Ketten warten auf den zuletzt erfolgreichen Vorgänger-Lauf."""
    voll = next(e for e in wd.KETTEN_CONFIG if e[0] == "kanalsatz-voll")
    b4 = next(e for e in wd.KETTEN_CONFIG if e[0] == "signal10-batch4")
    vh_ = next(e for e in wd.KETTEN_CONFIG if e[0] == "vocal-harshness")
    for command, vorgaenger in (
        (voll[2], "kanalsatz-smoke"),
        (b4[2], "signal10-resume"),
        (vh_[2], "crackle-kette"),
    ):
        wait = wd._wait_for_success(vorgaenger)
        assert wait in command
    # Beleg-Kollisionsschutz: voll schreibt in eine eigene Datei (nicht den
    # datumskanonischen Pfad des Smoke).
    assert "_kanalsatz_ab_voll.json" in voll[2]


def test_manifest_nachbehandlung_propagiert_vorgaengerfehler() -> None:
    """Ein erfolgreicher Manifest-Anhänger darf einen fehlgeschlagenen Lauf
    nicht als END=0 maskieren."""
    crackle = next(e for e in wd.KETTEN_CONFIG if e[0] == "crackle-kette")[2]
    vh_ = next(e for e in wd.KETTEN_CONFIG if e[0] == "vocal-harshness")[2]
    assert "--out data/klassen_paare_crackle_expansion && " in crackle
    assert "--out data/klassen_paare_vh && " in vh_
    assert "--klasse crackle && " in vh_


def test_manifest_anhaengen_in_kette() -> None:
    """Crackle-Kette endet mit dem Manifest-Anhänger (Self-Healing: der
    laufende Alt-Lauf ohne Anhänger wird vom vh-Ketten-Anfang geheilt)."""
    crackle = next(e for e in wd.KETTEN_CONFIG if e[0] == "crackle-kette")
    vh_ = next(e for e in wd.KETTEN_CONFIG if e[0] == "vocal-harshness")
    assert "haenge_klassen_manifest_an.py" in crackle[2]
    assert "--klasse crackle" in crackle[2]
    assert "haenge_klassen_manifest_an.py" in vh_[2]
    assert "--klasse crackle" in vh_[2] and "--klasse vocal_harshness" in vh_[2]


def test_muster_sind_teilzeichenketten_der_startkommandos() -> None:
    """Jedes pgrep-Muster muss eine zusammenhängende Teilzeichenkette des
    Start-Kommandos sein — sonst sieht der Watchdog die eigene Kette nie
    laufen und startet jede Runde Duplikate (Produktionsbefund 2026-11-19:
    `messe_kanalsatz_ab.py --paare 45` passte nie, `--verzeichnis` stand
    dazwischen; `de_crackle/train.py` ebenso — 4 Duplikat-Waiter + 2
    GPU-Trainings waren die Folge)."""
    for e in wd.KETTEN_CONFIG:
        assert e[1] in e[2], f"{e[0]}: Muster {e[1]!r} nicht im Start-Kommando"


def test_muster_kollidieren_nicht_mit_fremden_kommandos() -> None:
    """Kein Muster darf Teilzeichenkette eines FREMDEN Start-Kommandos sein
    (Produktionsbefund: das Smoke-Muster `messe_kanalsatz_ab.py` matchte den
    kanalsatz-voll-Waiter ⇒ die tote Smoke-Kette wurde nie neu gestartet;
    `--paare 4` ist Präfix von `--paare 45`)."""
    for i, a in enumerate(wd.KETTEN_CONFIG):
        for j, b in enumerate(wd.KETTEN_CONFIG):
            if i == j:
                continue
            assert a[1] not in b[2], f"{a[0]}-Muster {a[1]!r} matcht das {b[0]}-Kommando"


def test_restart_befehl_leitet_in_geprueftes_log() -> None:
    """Der Restart hängt stdout/stderr an DAS geprüfte Log und der END-Marker
    trägt den ECHTEN Ketten-Status (sonst Endlos-Restart bzw. stille Fehlschläge)."""
    quelle = Path(wd.__file__).read_text(encoding="utf-8")
    assert "cmd = f\"{start_cmd} >> {log_pfad} 2>&1; STATUS=$?; echo '{marker}'$STATUS >> {log_pfad}\"" in quelle
    assert "start_new_session=True" in quelle  # setsid-Semantik zusätzlich abgesichert
    # Kein interner echo-Befehl in den Start-Kommandos mehr (Doppel-Marker mit
    # immer 0 war der VinylNet-Stillfehler: fehlgeschlagene Neustarts sahen
    # „fertig“ aus).
    for e in wd.KETTEN_CONFIG:
        assert "echo '" not in e[2], f"{e[0]}: interner echo-Marker im Start-Kommando"
