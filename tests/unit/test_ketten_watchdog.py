"""Vertragstests: Ketten-Watchdog (scripts/ketten_watchdog.py, 2026-11-19).

Härtungs-Schicht 4: der Watchdog prüft je Kette (Prozess läuft? END-Marker im
Log?) und startet nur bei „tot UND nicht fertig“ detached neu. Modell-frei:
Entscheidungslogik wird ohne echte Prozesse gepinnt.
"""

from __future__ import annotations

from pathlib import Path

from scripts import ketten_watchdog as wd


def test_end_marker_erkannt(tmp_path) -> None:
    log = tmp_path / "kette.log"
    log.write_text("läuft noch\n", encoding="utf-8")
    assert not wd._end_marker_da(str(log), "KETTE_END=")
    log.write_text("KETTE_END=0\n", encoding="utf-8")
    assert wd._end_marker_da(str(log), "KETTE_END=")


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


def test_konfiguration_komplett_und_marker_konsistent() -> None:
    """8 Ketten, je mit eigenem Log + eigenem END-Marker (keine Kollision)."""
    assert len(wd.KETTEN_CONFIG) == 8
    marker = {e[4] for e in wd.KETTEN_CONFIG}
    logs = {e[3] for e in wd.KETTEN_CONFIG}
    assert len(marker) == 8 and len(logs) == 8
    for e in wd.KETTEN_CONFIG:
        assert e[4].endswith("END="), e[0]


def test_gegatete_ketten_warten_auf_vorgaenger() -> None:
    """kanalsatz-voll, signal10-batch4 und vocal-harshness dürfen erst nach dem
    END-Marker des Vorgängers rechnen (Warte-Schleife im Start-Kommando;
    sonst Datei-Kollision bzw. CPU-Überlast)."""
    voll = next(e for e in wd.KETTEN_CONFIG if e[0] == "kanalsatz-voll")
    b4 = next(e for e in wd.KETTEN_CONFIG if e[0] == "signal10-batch4")
    vh_ = next(e for e in wd.KETTEN_CONFIG if e[0] == "vocal-harshness")
    assert "grep -q 'KANALSATZ_END=' /tmp/kanalsatz_ab_smoke.log" in voll[2]
    assert "grep -q 'SIGNAL10_RESUME_END=' /tmp/signal10_resume.log" in b4[2]
    assert "grep -q 'CRACKLE_CHAIN_END=' /tmp/crackle_chain.log" in vh_[2]
    # Beleg-Kollisionsschutz: voll schreibt in eine eigene Datei (nicht den
    # datumskanonischen Pfad des Smoke).
    assert "_kanalsatz_ab_voll.json" in voll[2]


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
