"""Vertragstests: DRT-12-Kanalsatz-A/B (scripts/messe_kanalsatz_ab.py, 2026-11-19).

Modell-frei: Flag-Aktivierung ist mess-only (prozess-lokal, keine Flag-Datei),
die Paar-Auswahl ist deterministisch (Seed 42), die A/B-Zuordnung nutzt die
bewertung.json-Schlüssel (song + rezept), und das Skript initialisiert kein
CUDA (CPU-only-Vertrag, stört die GPU-Kette nicht).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from scripts import messe_kanalsatz_ab as ab

_ROOT = Path(__file__).resolve().parents[2]


def test_kanalsatz_flags_vollstaendig() -> None:
    """Alle 14 gate-bestandenen Kanal-Flags sind gelistet (12 Alt + hum/wow 2026-11-21)."""
    assert len(ab.KANALSAETZ_FLAGS) == 14
    assert "use_witness_kopf_clicks" in ab.KANALSAETZ_FLAGS
    assert "use_witness_kopf_groove_echo" in ab.KANALSAETZ_FLAGS
    assert "use_witness_kopf_dynamic_compression_excess" in ab.KANALSAETZ_FLAGS
    assert "use_witness_kopf_dropout_head_contact" in ab.KANALSAETZ_FLAGS
    assert "use_witness_kopf_hum" in ab.KANALSAETZ_FLAGS
    assert "use_witness_kopf_wow" in ab.KANALSAETZ_FLAGS


def test_flag_aktivierung_nur_prozess_lokal() -> None:
    """_aktiviere_kanalsatz setzt Attribute, schreibt aber keine Flag-Datei (§G1 (copilot-instructions.md))."""
    import backend.core.music_model_flags as flags

    ab._aktiviere_kanalsatz()
    assert flags.use_witness_kopf_clicks is True
    assert flags.use_witness_kopf_dropout_head_contact is True
    # Kein Datei-Effekt: die Defaults des Moduls sind Quelltext — es gibt keinen
    # Persistenz-Aufruf im Skript.
    quelle = (ab._ROOT / "scripts" / "messe_kanalsatz_ab.py").read_text(encoding="utf-8")
    assert "flag_freigabe" not in quelle
    assert (
        "open(" not in quelle.split("def main")[1] or ".json" not in quelle.split("def main")[1].split("REPORT_DIR")[0]
    )


def test_aktivierung_eines_kanals_deaktiviert_die_restlichen(monkeypatch) -> None:
    import backend.core.music_model_flags as flags

    for name in ab.KANALSAETZ_FLAGS:
        monkeypatch.setattr(flags, name, False)

    ausgewaehlt = ab.KANALSAETZ_FLAGS[2]
    ab._aktiviere_kanalsatz(ausgewaehlt)

    assert getattr(flags, ausgewaehlt) is True
    assert all(not getattr(flags, name) for name in ab.KANALSAETZ_FLAGS if name != ausgewaehlt)


def test_flags_override_aktiviert_nur_den_uebergebenen_satz(monkeypatch) -> None:
    import backend.core.music_model_flags as flags

    for name in ab.KANALSAETZ_FLAGS:
        monkeypatch.setattr(flags, name, False)
    monkeypatch.setattr(flags, "use_witness_kopf_hum", False)
    monkeypatch.setattr(flags, "use_witness_kopf_wow", False)

    satz = ["use_witness_kopf_clicks", "use_witness_kopf_hum"]
    ab._aktiviere_kanalsatz(kanalsatz_flags=satz)

    assert flags.use_witness_kopf_clicks is True
    assert flags.use_witness_kopf_hum is True
    assert flags.use_witness_kopf_wow is False
    assert all(not getattr(flags, name) for name in ab.KANALSAETZ_FLAGS if name not in satz)


def test_parse_flags_dedup_fail_closed() -> None:
    assert ab._parse_flags("use_witness_kopf_clicks, use_witness_kopf_clicks,use_witness_kopf_hum") == [
        "use_witness_kopf_clicks",
        "use_witness_kopf_hum",
    ]
    with pytest.raises(ValueError):
        ab._parse_flags("")
    with pytest.raises(ValueError):
        ab._parse_flags("use_gibts_nicht")


def test_flags_fehlen_im_modul_wirft_runtime_error(monkeypatch) -> None:
    with pytest.raises(RuntimeError):
        ab._aktiviere_kanalsatz(kanalsatz_flags=["use_gibts_nicht"])


def test_kanal_muss_im_uebergebenen_satz_liegen() -> None:
    with pytest.raises(ValueError):
        ab._aktiviere_kanalsatz("use_witness_kopf_clicks", kanalsatz_flags=["use_witness_kopf_hum"])


def test_zeugen_traegt_kanonische_ueberrestauration(monkeypatch) -> None:
    from backend.core.dsp import wohlklang_zeuge

    monkeypatch.setattr(
        wohlklang_zeuge,
        "messe_wohlklang_zeuge",
        lambda *_args: {
            "delta": {"musik_mos": -0.1, "defektlast": -0.2},
            "ueberrestauration": True,
        },
    )

    result = ab._zeugen(np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.float32), 48000)

    assert result["ueberrestauration"] is True


def test_paar_auswahl_deterministisch(monkeypatch, tmp_path) -> None:
    for i in range(8):
        (tmp_path / f"song{i} (vinyl).wav").write_bytes(b"x")
    a = ab._paare_auswaehlen(tmp_path, 3, 42)
    b = ab._paare_auswaehlen(tmp_path, 3, 42)
    assert a == b and len(a) == 3
    assert all("_restauriert" not in p.name for _, p in a)


def test_bestand_key_song_plus_rezept(tmp_path) -> None:
    """A-Seite: bewertung.json-Schlüssel = 'song (rezept)' — deckungsgleich mit den Paar-Stems."""
    roh = {
        "zeilen": [
            {
                "song": "Titel A",
                "rezept": "vinyl",
                "delta_musik_mos": 0.1,
                "delta_defektlast": -0.2,
                "ueberrestauration": True,
            }
        ]
    }
    beleg = tmp_path / "bewertung.json"
    beleg.write_text(json.dumps(roh), encoding="utf-8")

    bestand = ab._bestand_laden(beleg)

    assert bestand["Titel A (vinyl)"]["ueberrestauration"] is True


def test_schema_pin_und_cpu_only() -> None:
    assert ab.SCHEMA == "aurik.kanalsatz_ab/1"
    quelle = (ab._ROOT / "scripts" / "messe_kanalsatz_ab.py").read_text(encoding="utf-8")
    assert 'os.environ.setdefault("AURIK_FORCE_CPU", "1")' in quelle
    assert ".cuda(" not in quelle and "cuda()" not in quelle


def test_subprozess_isolation_pro_paar() -> None:
    """Crash-Härtung (Produktionsbefund 2026-11-19: SIGABRT in phase_10/
    temporal_loudness riss den 4-Paar-Smoke): JEDES Paar läuft im eigenen
    Kindprozess, ein Absturz erzeugt einen `abgestuerzt`-Eintrag statt den
    Lauf zu beenden; der Beleg wird nach jedem Paar inkrementell geschrieben."""
    quelle = (ab._ROOT / "scripts" / "messe_kanalsatz_ab.py").read_text(encoding="utf-8")
    assert "subprocess.run(kind_argv, check=False" in quelle
    assert '"status": "abgestuerzt"' in quelle
    assert '"returncode": abgeschlossen.returncode' in quelle
    assert "_beleg_schreiben(ziel, paare, verzeichnis, args.seed, args.bestand, zeilen, kanal_flags)" in quelle
    assert "_fh_mod.enable(" in quelle  # Kind trägt den Crash ins Faulthandler-Log


def test_resume_pro_zeilen_datei() -> None:
    """Erfolgreiche Paar-Zeilen werden beim Resume übersprungen."""
    quelle = (ab._ROOT / "scripts" / "messe_kanalsatz_ab.py").read_text(encoding="utf-8")
    assert "vorhanden (Resume)" in quelle
    assert "if zeile_pfad.is_file():" in quelle


def test_resume_wiederholt_abgestuerzte_zeile(monkeypatch, tmp_path) -> None:
    quelle = tmp_path / "quelle"
    quelle.mkdir()
    (quelle / "Titel (hum).wav").write_bytes(b"fixture")
    beleg = tmp_path / "kanalsatz.json"
    zeile = ab._zeilen_dir(beleg) / "000_Titel (hum).json"
    zeile.parent.mkdir()
    zeile.write_text(json.dumps({"stamm": "Titel (hum)", "status": "abgestuerzt", "returncode": -11}))
    aufrufe: list[list[str]] = []

    def run(argv: list[str], check: bool, cwd: str) -> subprocess.CompletedProcess:
        assert check is False
        aufrufe.append(argv)
        zeilen_index = argv.index("--zeile-datei") + 1
        zeile_pfad = Path(argv[zeilen_index])
        assert json.loads(zeile_pfad.read_text())["status"] == "abgestuerzt"
        zeile_pfad.write_text(json.dumps({"stamm": "Titel (hum)", "status": "ok"}))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(ab.subprocess, "run", run)

    assert ab.main(["--verzeichnis", str(quelle), "--paare", "1", "--beleg", str(beleg)]) == 0
    assert len(aufrufe) == 1
    assert json.loads(zeile.read_text())["status"] == "ok"


def test_resume_aktualisiert_aggregat_aus_vorhandenen_zeilen(monkeypatch, tmp_path) -> None:
    quelle = tmp_path / "quelle"
    quelle.mkdir()
    (quelle / "A.wav").write_bytes(b"fixture")
    (quelle / "B.wav").write_bytes(b"fixture")
    paare = ab._paare_auswaehlen(quelle, 2, 42)
    beleg = tmp_path / "kanalsatz.json"
    zdir = ab._zeilen_dir(beleg)
    zdir.mkdir()
    first_stamm = paare[0][0]
    (zdir / f"000_{first_stamm[:40]}.json").write_text(
        json.dumps({"stamm": first_stamm, "status": "ok"}), encoding="utf-8"
    )

    def run(argv: list[str], check: bool, cwd: str) -> subprocess.CompletedProcess:
        aggregate = json.loads(beleg.read_text(encoding="utf-8"))
        assert aggregate["n_paare"] == 1
        assert aggregate["zeilen"][0]["stamm"] == first_stamm
        zeilen_index = argv.index("--zeile-datei") + 1
        zeile_pfad = Path(argv[zeilen_index])
        zeile_pfad.write_text(json.dumps({"stamm": paare[1][0], "status": "ok"}), encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(ab.subprocess, "run", run)

    assert ab.main(["--verzeichnis", str(quelle), "--paare", "2", "--beleg", str(beleg)]) == 0
    report = json.loads(beleg.read_text(encoding="utf-8"))
    assert report["kanalsatz_flags"] == ab.KANALSAETZ_FLAGS
    assert report["n_paare"] == 2
    assert [row["stamm"] for row in report["zeilen"]] == [stamm for stamm, _ in paare]


def test_resume_findet_passenden_unvollstaendigen_voll_beleg(tmp_path) -> None:
    quelle = Path("data/schadens_proben")
    beleg = tmp_path / "2026-11-19_kanalsatz_ab_voll.json"
    beleg.write_text(
        json.dumps(
            {
                "schema": ab.SCHEMA,
                "seed": 42,
                "verzeichnis": str(quelle),
                "bestand_beleg": "data/schadens_proben/bewertung.json",
                "n_geplant": 2,
                "zeilen": [{"stamm": "Titel (hum)", "status": "abgestuerzt"}],
            }
        ),
        encoding="utf-8",
    )

    resumed = ab._unvollstaendigen_beleg_finden(
        [("Titel (hum)", quelle / "Titel (hum).wav"), ("Titel (vinyl)", quelle / "Titel (vinyl).wav")],
        quelle,
        42,
        Path("data/schadens_proben/bewertung.json"),
        tmp_path,
    )

    assert resumed == beleg


def test_resume_ignoriert_beleg_mit_anderen_kanal_flags(tmp_path) -> None:
    quelle = Path("data/schadens_proben")
    beleg = tmp_path / "2026-11-19_kanalsatz_ab_voll.json"
    beleg.write_text(
        json.dumps(
            {
                "schema": ab.SCHEMA,
                "seed": 42,
                "verzeichnis": str(quelle),
                "bestand_beleg": "data/schadens_proben/bewertung.json",
                "n_geplant": 2,
                "kanalsatz_flags": ab.KANALSAETZ_FLAGS,
                "zeilen": [{"stamm": "Titel (hum)", "status": "abgestuerzt"}],
            }
        ),
        encoding="utf-8",
    )

    resumed = ab._unvollstaendigen_beleg_finden(
        [("Titel (hum)", quelle / "Titel (hum).wav"), ("Titel (vinyl)", quelle / "Titel (vinyl).wav")],
        quelle,
        42,
        Path("data/schadens_proben/bewertung.json"),
        tmp_path,
        kanalsatz_flags=[ab.KANALSAETZ_FLAGS[0]],
    )

    assert resumed is None


def test_einzelkanal_lauf_persistiert_flags_und_reicht_sie_weiter(monkeypatch, tmp_path) -> None:
    quelle = tmp_path / "quelle"
    quelle.mkdir()
    (quelle / "Titel (hum).wav").write_bytes(b"fixture")
    beleg = tmp_path / "einzelkanal.json"
    kanal = ab.KANALSAETZ_FLAGS[0]
    aufrufe: list[list[str]] = []

    def run(argv: list[str], check: bool, cwd: str) -> subprocess.CompletedProcess:
        assert check is False
        aufrufe.append(argv)
        zeilen_index = argv.index("--zeile-datei") + 1
        zeile_pfad = Path(argv[zeilen_index])
        zeile_pfad.write_text(
            json.dumps({"stamm": "Titel (hum)", "status": "ok", "kanalsatz_flags": [kanal]}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(ab.subprocess, "run", run)

    assert ab.main(["--verzeichnis", str(quelle), "--paare", "1", "--beleg", str(beleg), "--kanal", kanal]) == 0
    report = json.loads(beleg.read_text(encoding="utf-8"))
    assert report["kanalsatz_flags"] == [kanal]
    assert report["zeilen"][0]["kanalsatz_flags"] == [kanal]
    assert aufrufe[0][aufrufe[0].index("--kanal") + 1] == kanal


def test_resume_kanalmodus_weist_unpassende_zeile_ab(monkeypatch, tmp_path) -> None:
    quelle = tmp_path / "quelle"
    quelle.mkdir()
    (quelle / "Titel (hum).wav").write_bytes(b"fixture")
    beleg = tmp_path / "einzelkanal.json"
    kanal = ab.KANALSAETZ_FLAGS[0]
    stamm = ab._paare_auswaehlen(quelle, 1, 42)[0][0]
    zdir = ab._zeilen_dir(beleg)
    zdir.mkdir()
    (zdir / f"000_{stamm[:40]}.json").write_text(
        json.dumps({"stamm": stamm, "status": "ok", "kanalsatz_flags": ab.KANALSAETZ_FLAGS}),
        encoding="utf-8",
    )
    monkeypatch.setattr(ab.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("must not reuse another flag set"))

    assert ab.main(["--verzeichnis", str(quelle), "--paare", "1", "--beleg", str(beleg), "--kanal", kanal]) == 2


def test_beleg_mit_anderen_messparametern_wird_abgewiesen(monkeypatch, tmp_path) -> None:
    quelle = tmp_path / "quelle"
    quelle.mkdir()
    (quelle / "Titel (hum).wav").write_bytes(b"fixture")
    beleg = tmp_path / "bestand.json"
    beleg.write_text(
        json.dumps(
            {
                "schema": ab.SCHEMA,
                "seed": 7,
                "verzeichnis": str(quelle),
                "n_geplant": 1,
                "bestand_beleg": None,
                "kanalsatz_flags": ab.KANALSAETZ_FLAGS,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(ab.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("must not mix evidence"))

    assert ab.main(["--verzeichnis", str(quelle), "--paare", "1", "--beleg", str(beleg)]) == 2


def test_unvollstaendige_kanalsatzmessung_liefert_fehlerstatus(monkeypatch, tmp_path) -> None:
    quelle = tmp_path / "quelle"
    quelle.mkdir()
    (quelle / "Titel (hum).wav").write_bytes(b"fixture")
    beleg = tmp_path / "kanalsatz.json"

    def run(argv: list[str], check: bool, cwd: str) -> subprocess.CompletedProcess:
        assert check is False
        return subprocess.CompletedProcess(argv, -11)

    monkeypatch.setattr(ab.subprocess, "run", run)

    assert ab.main(["--verzeichnis", str(quelle), "--paare", "1", "--beleg", str(beleg)]) == 1
    report = json.loads(beleg.read_text())
    assert report["n_paare"] == report["n_geplant"] == 1
    assert report["zeilen"][0]["status"] == "abgestuerzt"


def test_kind_dispatch_und_modus_marker() -> None:
    """`--index`/`--einzel` dispatcht in den Kind-Modus; der Beleg trägt
    `modus: subprozess-isoliert` (Schema-additiv, DRT-12-Nachvollziehbarkeit)."""
    quelle = (ab._ROOT / "scripts" / "messe_kanalsatz_ab.py").read_text(encoding="utf-8")
    assert 'if "--index" in argv or "--einzel" in argv:' in quelle
    assert '"modus": "subprozess-isoliert"' in quelle
    assert "tmp.replace(args.zeile_datei)" in quelle  # atomare Zeilen-Datei


def test_numba_bootstrap_vor_denker_thread() -> None:
    """Exit-139-Fix (2026-11-19): numba/librosa-JIT-Kompilierung wird VOR dem
    Bridge-Import (Denker-Thread) im Hauptthread erzwungen — lazy njit-Erst-
    Kompilierung im Worker-Thread segfaultete den Prozess (Faulthandler:
    <no Python frame> + numba-Module)."""
    quelle = (ab._ROOT / "scripts" / "erstelle_schadens_proben.py").read_text(encoding="utf-8")
    i_def = quelle.index("def restauriere_schnipsel")
    i_bridge = quelle.index("from backend.api.bridge import", i_def)
    assert "ensure_librosa_ready()" in quelle[i_def:i_bridge]
    boot = (ab._ROOT / "backend" / "core" / "librosa_bootstrap.py").read_text(encoding="utf-8")
    assert "_pghi_heap_kernel_nb" in boot
    assert "_qp_envelope_numba" in boot
    assert "kalman_smooth_warp" in boot
