"""Vertragstests Ohr-Panel (M6-Komfort, 2026-11-09).

Modell-frei: Paar-Erkennung, interaktive Abfrage und der Ein-Kommando-E2E-Pfad
werden ohne GPU/Modell-Ladung gepinnt (§G5 (GEBOTE.md)).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ohr_panel import (
    defekt_variante,
    erfrage_verzeichnis,
    erkenne_paare,
    erzeuge_paare,
    finde_player,
    frage_verdikte,
    spiele_ab,
)


def _schreibe(pfad: Path, sr: int = 48000, seed: int = 1) -> None:
    import numpy as np
    import soundfile as sf

    rng = np.random.default_rng(seed)
    sf.write(pfad, (rng.standard_normal(sr).astype(np.float32) * 0.1), sr)


def test_erkenne_paare_a_b_ordner(tmp_path: Path) -> None:
    (tmp_path / "A").mkdir()
    (tmp_path / "B").mkdir()
    _schreibe(tmp_path / "A" / "song1.wav")
    _schreibe(tmp_path / "B" / "song1.wav")
    _schreibe(tmp_path / "A" / "song2.wav")
    _schreibe(tmp_path / "B" / "song2.wav")
    _schreibe(tmp_path / "B" / "nur_in_b.wav")
    paare = erkenne_paare(tmp_path)
    assert [p["id"] for p in paare] == ["song1", "song2"], paare
    assert paare[0]["a"].endswith("A/song1.wav") and paare[0]["b"].endswith("B/song1.wav")


def test_erkenne_paare_suffix_layout(tmp_path: Path) -> None:
    _schreibe(tmp_path / "song.a.wav")
    _schreibe(tmp_path / "song.b.wav")
    paare = erkenne_paare(tmp_path)
    assert [p["id"] for p in paare] == ["song"]


def test_erkenne_paare_unterordner_layout(tmp_path: Path) -> None:
    (tmp_path / "paar1").mkdir()
    _schreibe(tmp_path / "paar1" / "a.wav")
    _schreibe(tmp_path / "paar1" / "b.wav")
    paare = erkenne_paare(tmp_path)
    assert [p["id"] for p in paare] == ["paar1"]


def test_erkenne_paare_restauriert_layout(tmp_path: Path) -> None:
    """Echte Aurik-Paare: <stamm>_restauriert.wav ↔ <stamm>.wav bzw. _ORIGINAL."""
    _schreibe(tmp_path / "lied.wav")
    _schreibe(tmp_path / "lied_restauriert.wav")
    _schreibe(tmp_path / "song.wav")
    _schreibe(tmp_path / "song_restauriert_27.09.2026.wav")
    _schreibe(tmp_path / "andere.wav")
    _schreibe(tmp_path / "andere_ORIGINAL.wav")
    _schreibe(tmp_path / "andere_restauriert.wav")
    _schreibe(tmp_path / "nur_ORIGINAL.wav")
    _schreibe(tmp_path / "nur_restauriert.wav")
    paare = erkenne_paare(tmp_path)
    ids = {p["id"] for p in paare}
    assert ids == {"lied", "song", "andere", "nur"}, ids
    # a = Original, b = restauriert; exakter Name hat Vorrang vor _ORIGINAL
    by_id = {p["id"]: p for p in paare}
    assert by_id["lied"]["a"].endswith("lied.wav")
    assert by_id["lied"]["b"].endswith("lied_restauriert.wav")
    assert by_id["andere"]["a"].endswith("andere.wav")
    assert by_id["nur"]["a"].endswith("nur_ORIGINAL.wav"), "_ORIGINAL-Fallback ohne exakten Namen"


def test_erstelle_trials_kuerzt_immer_auf_10s(tmp_path: Path) -> None:
    """Auch ERKANNTE Paare (restauriert) werden auf den 10-s-Ausschnitt gekürzt —
    nie ganze Songs abspielen (2026-11-10).
    """
    import numpy as np
    import soundfile as sf

    from scripts.ohr_panel import erstelle_trials

    for name in ("lied.wav", "lied_restauriert.wav"):
        sf.write(
            tmp_path / name,
            (np.sin(2 * np.pi * 440 * np.arange(48000 * 120) / 48000) * 0.4).astype(np.float32),
            48000,
        )
    paare = [{"id": "lied", "a": str(tmp_path / "lied.wav"), "b": str(tmp_path / "lied_restauriert.wav")}]
    sheet, _key = erstelle_trials(paare, tmp_path / "work", seed=7, ausschnitt_s=10.0)
    for trial in sheet["trials"]:
        for slot in ("left", "right"):
            d, sr = sf.read(str(tmp_path / "work" / trial["stimuli"][slot]))
            assert round(len(d) / sr) == 10.0, f"{slot} muss 10 s sein: {len(d) / sr}"


def test_erkenne_paare_leer_und_fehler(tmp_path: Path) -> None:
    assert erkenne_paare(tmp_path) == []
    try:
        erkenne_paare(tmp_path / "gibt_es_nicht")
        raise AssertionError("FileNotFoundError erwartet")
    except FileNotFoundError:
        pass


def test_frage_verdikte_abbruch_und_ties() -> None:
    trials = [
        {"trial_id": "t0", "stimuli": {"left": "l.wav", "right": "r.wav"}},
        {"trial_id": "t1", "stimuli": {"left": "l.wav", "right": "r.wav"}},
    ]
    v = frage_verdikte(trials, iter(["l", "="]))
    assert v == [
        {"trial_id": "t0", "preferred": "left"},
        {"trial_id": "t1", "preferred": "tie"},
    ]
    # Abbruch: leere Eingaben → bisherige Verdikte bleiben
    v2 = frage_verdikte(trials, iter(["r"]))
    assert v2 == [{"trial_id": "t0", "preferred": "right"}]


def test_panel_e2e_ein_kommando(tmp_path: Path) -> None:
    """E2E: Verzeichnis + 2 Eingabezeichen → Artefakt models/ohr/ohr_prior.json."""
    (tmp_path / "A").mkdir()
    (tmp_path / "B").mkdir()
    _schreibe(tmp_path / "A" / "song.wav")
    _schreibe(tmp_path / "B" / "song.wav")
    artefakt = PROJECT_ROOT / "models" / "ohr" / "ohr_prior.json"
    artefakt.unlink(missing_ok=True)
    r = subprocess.run(
        [
            sys.executable,
            "-B",
            "scripts/ohr_panel.py",
            "--verzeichnis",
            str(tmp_path),
            "--arbeits-verzeichnis",
            str(tmp_path / "work"),
            "--seed",
            "7",
            "--kein-player",
        ],
        cwd=PROJECT_ROOT,
        input="r\n",
        text=True,
        capture_output=True,
        timeout=180,
    )
    assert r.returncode == 0, r.stderr[-500:]
    assert artefakt.is_file(), "Panel muss das Ohr-Artefakt schreiben"
    import json

    d = json.loads(artefakt.read_text(encoding="utf-8"))
    assert d["schema"].startswith("aurik.ohr_prior/"), d["schema"]
    assert d["n_labels"] == 1
    artefakt.unlink(missing_ok=True)  # Test-Artefakt nicht als echtes Ohr zurücklassen


def test_defekt_varianten_deterministisch_und_unterscheidbar() -> None:
    """Defekt-Varianten: deterministisch (gleicher Seed ⇒ bit-identisch),
    hörbar verschieden vom Original; unbekannte Variante wirft (§V6 (copilot-instructions.md)).
    """
    import numpy as np

    rng = np.random.default_rng(1)
    x = (np.sin(2 * np.pi * 440 * np.arange(48000) / 48000) * 0.4).astype(np.float32)
    for variante in (
        "rausch",
        "bandlimit",
        "saettigung",
        "kammfilter",
        "wowflutter",
        "kompression",
        "brummen",
        "tremolo",
        "verzerrung",
        "lautheit",
    ):
        v1 = defekt_variante(x, variante, np.random.default_rng(7))
        v2 = defekt_variante(x, variante, np.random.default_rng(7))
        assert np.array_equal(v1, v2), f"{variante} nicht deterministisch"
        assert float(np.abs(v1 - x).mean()) > 1e-4, f"{variante} zu schwach"
        assert np.isfinite(v1).all()
    try:
        defekt_variante(x, "gibt_es_nicht", np.random.default_rng(0))
        raise AssertionError("ValueError erwartet")
    except ValueError:
        pass


def test_erzeuge_paare_aus_musik_ordner(tmp_path: Path) -> None:
    """Ein Musik-Ordner + 3 Varianten ⇒ 3 A/B-Paare im A/-B/-Layout."""
    musik = tmp_path / "musik"
    musik.mkdir()
    _schreibe(musik / "lied.wav")
    paare = erzeuge_paare(musik, tmp_path / "paare", ["rausch", "bandlimit", "saettigung"], seed=5)
    assert [p["id"] for p in paare] == ["lied__rausch", "lied__bandlimit", "lied__saettigung"]
    for p in paare:
        assert Path(p["a"]).is_file() and Path(p["b"]).is_file()
    # Deterministisch: zweiter Lauf gleiche Dateien überschreibt identisch
    paare2 = erzeuge_paare(musik, tmp_path / "paare", ["rausch", "bandlimit", "saettigung"], seed=5)
    assert paare == paare2


def test_erzeuge_paare_mittel_ausschnitt(tmp_path: Path) -> None:
    """Standard: nur 20 s aus der Song-Mitte werden verglichen (kein Ganztrack-Hören)."""
    import numpy as np
    import soundfile as sf

    musik = tmp_path / "musik"
    musik.mkdir()
    # 90 s Signal mit Markierung: 440 Hz überall, 880 Hz nur im Mittelteil
    sr = 48000
    t = np.arange(sr * 90) / sr
    x = (np.sin(2 * np.pi * 440 * t) * 0.4).astype(np.float32)
    mitte = slice(sr * 35, sr * 55)
    x[mitte] = (np.sin(2 * np.pi * 880 * t[mitte]) * 0.4).astype(np.float32)
    sf.write(musik / "lang.wav", x, sr)

    paare = erzeuge_paare(musik, musik / "paare", ["rausch"], seed=3, ausschnitt_s=20.0)
    a_daten, a_sr = sf.read(paare[0]["a"])
    assert round(len(a_daten) / a_sr, 1) == 20.0, f"Ausschnitt muss 20 s sein: {len(a_daten) / a_sr}"
    # Mittel-Ausschnitt enthält den 880-Hz-Ton (35–55 s) — Beweis der Stelle
    spektrum = np.abs(np.fft.rfft(np.asarray(a_daten, dtype=np.float64)))
    freqs = np.fft.rfftfreq(len(a_daten), 1 / a_sr)
    assert spektrum[(freqs > 800) & (freqs < 960)].max() > spektrum[(freqs > 400) & (freqs < 480)].max() * 0.5
    # Ganz-Song-Option bleibt: ausschnitt_s=0 ⇒ volle Länge
    paare_voll = erzeuge_paare(musik, musik / "paare_voll", ["rausch"], seed=3, ausschnitt_s=0.0)
    voll, _ = sf.read(paare_voll[0]["a"])
    assert round(len(voll) / sr, 1) == 90.0


def test_erfrage_verzeichnis_enter_nimmt_vorschlag() -> None:
    """Ohne Argumente: Enter nimmt den vorhandenen Vorschlag; ungültige Eingabe fragt erneut."""
    import tempfile
    from unittest import mock

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        # Vorschlag existiert nicht in HOME — Eingabe liefert den Ordner
        with mock.patch.object(Path, "home", return_value=td_path):
            erg = erfrage_verzeichnis(iter([str(td_path), ""]))
            assert erg == td_path

    # Ungültige Eingabe wird wiederholt, dann gültig
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        with mock.patch.object(Path, "home", return_value=td_path):
            erg = erfrage_verzeichnis(iter(["gibt_es_nicht", str(td_path), ""]))
            assert erg == td_path


def test_panel_e2e_auto_erzeugt_ohne_flag(tmp_path: Path) -> None:
    """Der Normalfall für den Nutzer: Musik-Ordner, KEIN --erzeuge-paare —
    das Panel erzeugt die Paare automatisch und läuft komplett durch.
    """
    musik = tmp_path / "musik"
    musik.mkdir()
    _schreibe(musik / "lied.wav")
    artefakt = PROJECT_ROOT / "models" / "ohr" / "ohr_prior.json"
    artefakt.unlink(missing_ok=True)
    r = subprocess.run(
        [
            sys.executable,
            "-B",
            "scripts/ohr_panel.py",
            "--verzeichnis",
            str(musik),
            "--arbeits-verzeichnis",
            str(tmp_path / "work"),
            "--kein-player",
        ],
        cwd=PROJECT_ROOT,
        input="l\nl\nl\n",
        text=True,
        capture_output=True,
        timeout=180,
    )
    assert r.returncode == 0, r.stderr[-500:]
    assert "automatisch aus der Musik erzeugt" in r.stdout, r.stdout[-400:]
    assert artefakt.is_file()
    artefakt.unlink(missing_ok=True)


def test_player_erkennung_und_fallback() -> None:
    """finde_player findet einen Player ODER None; spiele_ab wirft nie (§V6 (copilot-instructions.md))."""
    assert finde_player() in ("ffplay", "aplay", "paplay", None)
    # Nicht-existenter PLAYER ⇒ deterministisch False, kein Wurf
    assert spiele_ab(Path("egal.wav"), "gibt_es_nicht") is False


def test_frage_verdikte_mit_player_enter_wiederholt_und_q_bricht_ab() -> None:
    """Mit player: Trials werden abgespielt, Enter = nochmal, q = Abbruch."""
    trials = [
        {"trial_id": "t0", "stimuli": {"left": "l.wav", "right": "r.wav"}},
        {"trial_id": "t1", "stimuli": {"left": "l.wav", "right": "r.wav"}},
    ]
    gespielt: list[str] = []

    def _spiel(pfad) -> bool:
        gespielt.append(str(pfad))
        return True

    v = frage_verdikte(
        trials,
        iter(["", "l", "q"]),
        player="ffplay",
        basis=Path("/basis"),
        spieler_fn=_spiel,
    )
    assert v == [{"trial_id": "t0", "preferred": "left"}], v
    assert gespielt[:4] == ["/basis/l.wav", "/basis/r.wav"] * 2, gespielt  # Enter ⇒ beide nochmal
    assert len(gespielt) == 6  # t1: links+rechts gespielt, dann q
