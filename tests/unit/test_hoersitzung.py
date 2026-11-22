from __future__ import annotations

"""Tests für scripts/hoersitzung.py — die unterbrechungsfreie, verblindete Hörsitzung.

Alle Pakete entstehen synthetisch in ``tmp_path`` im Layout der echten Pakete. Der
„Orakel-Hörer“ ist ein reines Testwerkzeug: Er liest den versiegelten Schlüssel, läuft nur
in ``tmp_path``-Sitzungen und erzeugt nie einen Beleg im Repository.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import hoersitzung as hs
from scripts.flag_freigabe import verdikt_ok

SR = 16000
DAUER_S = 6.0


# ── synthetische Pakete ──────────────────────────────────────────────────────


def _ton(seed: int, dauer: float = DAUER_S, klick: bool = False) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * dauer)) / SR
    x = (
        0.3 * np.sin(2 * np.pi * (220 + 7 * seed) * t)
        + 0.1 * np.sin(2 * np.pi * 3100 * t)
        + 0.02 * rng.standard_normal(t.size)
    )
    if klick:
        x[::4000] += 0.6
    return np.clip(x, -1, 1).astype(np.float32)


def _wav(pfad: Path, x: np.ndarray) -> Path:
    pfad.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(pfad), x, SR, subtype="PCM_24")
    return pfad


def _baue_pakete(proben: Path) -> Path:
    # C4/M5: manifest.json, A = Bestand, B = Kandidat (absolute Pfade wie im echten Paket)
    sets: dict[str, dict] = {}
    for name in ("c4_vocos", "m5_rekonstruktion"):
        eintraege = []
        for i in range(6):
            a = _wav(proben / name / f"s{i}__A.wav", _ton(10 + i, klick=True))
            b = _wav(proben / name / f"s{i}__B.wav", _ton(10 + i))
            eintraege.append({"stamm": f"{name}-{i}", "a": str(a), "b": str(b)})
        sets[name] = {"entscheidung": f"Entscheidung {name}", "proben": eintraege}
    (proben / "manifest.json").write_text(json.dumps({"sets": sets}), encoding="utf-8")

    # Declip: Schlüssel je Titel („A=flag_an“ → A ist Kandidat)
    declip = proben / "declip_produktionspfad"
    key = {}
    for i, zuordnung in enumerate(("A=flag_aus", "A=flag_an", "A=flag_aus")):
        stamm = f"Titel_{i}"
        a_ist_kandidat = zuordnung == "A=flag_an"
        _wav(declip / f"{stamm}_A.wav", _ton(40 + i) if a_ist_kandidat else _ton(40 + i, klick=True))
        _wav(declip / f"{stamm}_B.wav", _ton(40 + i, klick=True) if a_ist_kandidat else _ton(40 + i))
        key[stamm] = zuordnung
    (declip / "schluessel.json").write_text(json.dumps(key), encoding="utf-8")

    # Stereo: Schlüssel je Datei
    stereo = proben / "stereo_sinnbeweis"
    key = {}
    for i, (ka, kb) in enumerate((("restauriert", "schaden"), ("schaden", "restauriert"), ("restauriert", "schaden"))):
        stamm = f"Stereo {i}"
        _wav(stereo / f"{stamm}_A.wav", _ton(60 + i) if ka == "restauriert" else _ton(60 + i, klick=True))
        _wav(stereo / f"{stamm}_B.wav", _ton(60 + i) if kb == "restauriert" else _ton(60 + i, klick=True))
        key[f"{stamm}_A.wav"], key[f"{stamm}_B.wav"] = ka, kb
    (stereo / "schluessel.json").write_text(json.dumps(key), encoding="utf-8")

    # Witness: A = Schaden (Bestand), B = repariert (Kandidat); 1-s-Clips
    for klasse, n in (("clicks", 6), ("jitter_artifacts", 1)):
        ordner = proben / "witness_kopf" / klasse
        zeilen = []
        for i in range(1, n + 1):
            a = _wav(ordner / f"probe{i:02d}_a.wav", _ton(80 + i, 1.0, klick=True))
            b = _wav(ordner / f"probe{i:02d}_b.wav", _ton(80 + i, 1.0))
            zeilen.append(json.dumps({"probe": i, "a": str(a), "b": str(b), "titel": f"{klasse}-{i}"}))
        (ordner / "manifest.jsonl").write_text("\n".join(zeilen) + "\n", encoding="utf-8")
    return proben


@pytest.fixture
def proben(tmp_path: Path) -> Path:
    return _baue_pakete(tmp_path / "proben")


def _plan(proben: Path, ziel: Path, **kw) -> dict:
    return hs.baue_plan(proben, ziel, **kw)


def _lade(ziel: Path) -> tuple[dict, dict]:
    plan = json.loads((ziel / "plan.json").read_text(encoding="utf-8"))
    key = json.loads((ziel / "versiegelt" / "schluessel.json").read_text(encoding="utf-8"))
    return plan, key


class Orakel:
    """Test-Hörer: urteilt nach dem versiegelten Schlüssel (nie im Produktionspfad)."""

    def __init__(self, ziel: Path, modus: str = "kandidat") -> None:
        self.key = json.loads((ziel / "versiegelt" / "schluessel.json").read_text(encoding="utf-8"))["versuche"]
        self.modus = modus
        self.aktuell = ""

    def spiele(self, pfad: Path) -> bool:
        self.aktuell = Path(pfad).stem
        return True

    def taste(self) -> str:
        e = self.key.get(self.aktuell)
        if e is None:  # Kalibrierung bzw. Runden-Pause: beliebige Taste
            return "x"
        if self.modus == "rate1":
            return "1"
        if e["art"] == "identisch":
            return "="
        if e["art"] == "anker":
            return "1" if e["erstes"] == "original" else "2"
        if self.modus == "rate1":
            return "1"
        if self.modus == "gleich":
            return "="
        wunsch_kandidat = self.modus == "kandidat"
        return "1" if (e["erstes"] == "kandidat") == wunsch_kandidat else "2"


def _hoere(ziel: Path, hoerer: str, modus: str = "kandidat") -> dict:
    o = Orakel(ziel, modus)
    return hs.hoeren(ziel, hoerer, spiele_fn=o.spiele, taste_fn=o.taste, ausgabe=lambda _t: None)


# ── Statistik ────────────────────────────────────────────────────────────────


def test_vorzeichentest_und_machbarkeit() -> None:
    assert hs.vorzeichentest_p(6, 6) == pytest.approx(0.03125)
    assert hs.vorzeichentest_p(3, 3) == pytest.approx(0.25)
    assert hs.vorzeichentest_p(5, 6) == pytest.approx(0.21875)
    assert hs.vorzeichentest_p(0, 0) == 1.0
    assert hs.p_minimal(3) == pytest.approx(0.25)
    assert hs.n_fuer_signifikanz() == 6
    assert hs.p_minimal(5) >= 0.05 > hs.p_minimal(6)


# ── Pakete laden / Auflösung Bestand vs. Kandidat ────────────────────────────


def test_pakete_werden_auf_bestand_und_kandidat_aufgeloest(proben: Path) -> None:
    bloecke = {b.id: b for b in hs.lade_bloecke(proben)}
    assert set(bloecke) == {
        "declip",
        "witness_clicks",
        "witness_jitter_artifacts",
        "stereo",
        "c4_vocos",
        "m5_rekonstruktion",
    }
    declip = {p.titel: p for p in bloecke["declip"].paare}
    assert declip["Titel_1"].kandidat.name == "Titel_1_A.wav"  # A=flag_an
    assert declip["Titel_0"].kandidat.name == "Titel_0_B.wav"  # A=flag_aus
    stereo = {p.titel: p for p in bloecke["stereo"].paare}
    assert stereo["Stereo 0"].kandidat.name == "Stereo 0_A.wav"
    assert stereo["Stereo 1"].kandidat.name == "Stereo 1_B.wav"
    w = bloecke["witness_clicks"].paare[0]
    assert w.bestand.name.endswith("_a.wav") and w.kandidat.name.endswith("_b.wav")
    assert bloecke["witness_clicks"].flag == "use_witness_kopf_clicks"
    assert bloecke["declip"].flag == "use_aspade_declipper"
    c4 = bloecke["c4_vocos"].paare[0]
    assert c4.bestand.name.endswith("__A.wav") and c4.kandidat.name.endswith("__B.wav")


def test_widerspruechlicher_schluessel_wird_abgelehnt(proben: Path) -> None:
    schluessel = proben / "stereo_sinnbeweis" / "schluessel.json"
    key = json.loads(schluessel.read_text(encoding="utf-8"))
    key["Stereo 0_B.wav"] = "restauriert"  # beide „restauriert“
    schluessel.write_text(json.dumps(key), encoding="utf-8")
    with pytest.raises(ValueError, match="nicht eindeutig"):
        hs.lade_bloecke(proben)


def test_unbekannte_declip_zuordnung_wird_abgelehnt(proben: Path) -> None:
    schluessel = proben / "declip_produktionspfad" / "schluessel.json"
    key = json.loads(schluessel.read_text(encoding="utf-8"))
    key["Titel_0"] = "A=irgendwas"
    schluessel.write_text(json.dumps(key), encoding="utf-8")
    with pytest.raises(ValueError, match="unbekannte Zuordnung"):
        hs.lade_bloecke(proben)


# ── Plan: Machbarkeit, Verblindung, Balance, Pegel, Kontrollen ───────────────


def test_unentscheidbare_bloecke_werden_vor_dem_hoeren_zurueckgestellt(proben: Path, tmp_path: Path) -> None:
    r = _plan(proben, tmp_path / "s")
    zurueck = {z["block"]: z for z in r["zurueckgestellt"]}
    assert set(zurueck) == {"declip", "stereo", "witness_jitter_artifacts"}
    assert zurueck["declip"]["p_minimal"] == pytest.approx(0.25)
    assert zurueck["witness_jitter_artifacts"]["p_minimal"] == pytest.approx(1.0)
    assert set(r["bloecke"]) == {"witness_clicks", "c4_vocos", "m5_rekonstruktion"}
    assert r["machbarkeit"]["declip"]["fehlende_paare"] == 3
    assert r["machbarkeit"]["c4_vocos"]["entscheidbar"] is True


def test_nicht_entscheidbare_koennen_bewusst_mitgehoert_werden(proben: Path, tmp_path: Path) -> None:
    r = _plan(proben, tmp_path / "s", mit_nicht_entscheidbaren=True)
    assert set(r["bloecke"]) == {
        "declip",
        "stereo",
        "witness_clicks",
        "witness_jitter_artifacts",
        "c4_vocos",
        "m5_rekonstruktion",
    }
    assert r["zurueckgestellt"] == []


def test_plan_ist_deterministisch_und_seed_abhaengig(proben: Path, tmp_path: Path) -> None:
    a, b = _plan(proben, tmp_path / "a"), _plan(proben, tmp_path / "b")
    assert a["sitzung_id"] == b["sitzung_id"]
    assert (tmp_path / "a" / "plan.json").read_text() == (tmp_path / "b" / "plan.json").read_text()
    assert _plan(proben, tmp_path / "c", seed=7)["sitzung_id"] != a["sitzung_id"]


def test_oeffentlicher_plan_verraet_nichts(proben: Path, tmp_path: Path) -> None:
    ziel = tmp_path / "s"
    _plan(proben, ziel)
    text = (ziel / "plan.json").read_text(encoding="utf-8").lower()
    for verboten in (
        "kandidat",
        "bestand",
        "witness",
        "c4_vocos",
        "m5_",
        "declip",
        "stereo",
        "clicks",
        "block",
        "erstes",
    ):
        assert verboten not in text, verboten
    plan, _ = _lade(ziel)
    assert all(set(v) == {"nr", "id", "datei", "dauer_s", "runde"} for v in plan["versuche"])


def test_position_ist_je_block_exakt_ausgewogen(proben: Path, tmp_path: Path) -> None:
    ziel = tmp_path / "s"
    _plan(proben, ziel)
    _, key = _lade(ziel)
    je_block: dict[str, list[str]] = {}
    for e in key["versuche"].values():
        if e["art"] == "paar":
            je_block.setdefault(e["block"], []).append(e["erstes"])
    assert set(je_block) == {"witness_clicks", "c4_vocos", "m5_rekonstruktion"}
    for erstes in je_block.values():
        assert erstes.count("kandidat") == erstes.count("bestand") == 3


def test_pegelunterschied_wird_nur_abgeschwaecht(proben: Path, tmp_path: Path) -> None:
    b = _ton(5)
    laut = np.clip(b * 2.0, -1, 1).astype(np.float32)  # +6 dB
    g_b, g_k, status = hs.pegel_gains(b, laut)
    assert status == "kandidat_abgeschwaecht"
    assert g_b == 0.0 and g_k == pytest.approx(-6.0, abs=0.3)
    g_b2, g_k2, status2 = hs.pegel_gains(laut, b)
    assert status2 == "bestand_abgeschwaecht" and g_k2 == 0.0 and g_b2 < 0.0
    assert hs.pegel_gains(b, b * 1.02)[2] == "ausgeglichen"
    assert hs.pegel_gains(np.zeros(100, np.float32), b)[2] == "pegel_nicht_bestimmbar"


def test_gerenderter_versuch_hat_ausgeglichene_pegel(proben: Path, tmp_path: Path) -> None:
    for i in range(6):  # Kandidat durchgehend +6 dB lauter
        k = proben / "c4_vocos" / f"s{i}__B.wav"
        x, _ = sf.read(str(k), dtype="float32")
        sf.write(str(k), np.clip(x * 2.0, -1, 1), SR, subtype="PCM_24")
    ziel = tmp_path / "s"
    _plan(proben, ziel)
    plan, key = _lade(ziel)
    gesehen = 0
    for v in plan["versuche"]:
        e = key["versuche"][v["id"]]
        if e["art"] == "paar" and e["block"] == "c4_vocos":
            assert e["gain_kandidat_db"] == pytest.approx(-6.0, abs=0.4) and e["gain_bestand_db"] == 0.0
            gesehen += 1
    assert gesehen == 6


def test_kontrollen_sind_vorhanden_verteilt_und_nie_zuerst(proben: Path, tmp_path: Path) -> None:
    ziel = tmp_path / "s"
    r = _plan(proben, ziel)
    plan, key = _lade(ziel)
    arten = [key["versuche"][v["id"]]["art"] for v in plan["versuche"]]
    assert arten.count("anker") >= hs.KONTROLL_MIN and arten.count("identisch") >= hs.KONTROLL_MIN
    assert r["n_kontrollen"] == arten.count("anker") + arten.count("identisch")
    assert arten[0] == "paar"
    positionen = [i for i, a in enumerate(arten) if a != "paar"]
    assert max(positionen) > len(arten) // 2 and min(positionen) < len(arten) // 2  # über die Sitzung verteilt


def test_anker_ist_dunkler_als_das_original(proben: Path, tmp_path: Path) -> None:
    x, sr = hs._lade_audio(proben / "c4_vocos" / "s0__B.wav")
    anker = hs.anker_tiefpass(x, sr)
    assert anker.shape == x.shape
    ab_5khz = int(len(x) * 5000 / sr)
    hoch = lambda y: float(np.sum(np.abs(np.fft.rfft(y[:, 0])[ab_5khz:]) ** 2))
    assert hoch(anker) < 0.001 * hoch(x)
    with pytest.raises(ValueError):
        hs.anker_tiefpass(x, 6000)


def test_defekte_pakete_werden_mit_grund_ausgewiesen(proben: Path, tmp_path: Path) -> None:
    (proben / "c4_vocos" / "s0__B.wav").unlink()  # fehlt
    _wav(proben / "c4_vocos" / "s1__B.wav", _ton(11, klick=True))  # identisch zu A → kein Unterschied
    sf.write(str(proben / "c4_vocos" / "s2__B.wav"), np.zeros(SR, np.float32), 8000)  # andere Abtastrate
    r = _plan(proben, tmp_path / "s", mit_nicht_entscheidbaren=True)
    gruende = {a["paar"]: a["grund"] for a in r["ausgeschlossen"]}
    assert "Datei fehlt" in gruende["c4_vocos__1"]
    assert "identisch" in gruende["c4_vocos__2"]
    assert "Format verschieden" in gruende["c4_vocos__3"]
    assert r["bloecke"]["c4_vocos"] == 3


def test_ohne_hoerbare_bloecke_wird_laut_abgebrochen(tmp_path: Path) -> None:
    leer = tmp_path / "leer"
    leer.mkdir()
    with pytest.raises(ValueError, match="Keine hörbaren Blöcke"):
        _plan(leer, tmp_path / "s")


def test_runden_respektieren_die_dauergrenze(proben: Path, tmp_path: Path) -> None:
    r = _plan(proben, tmp_path / "s", runde_min=1.0)
    plan, _ = _lade(tmp_path / "s")
    assert r["n_runden"] == plan["n_runden"] > 1
    for runde in range(1, plan["n_runden"] + 1):
        summe = sum(v["dauer_s"] + hs.ANTWORTZEIT_S for v in plan["versuche"] if v["runde"] == runde)
        assert summe <= 60.0 + max(v["dauer_s"] for v in plan["versuche"]) + hs.ANTWORTZEIT_S


# ── Sitzung: Bedienung, Wiederaufnahme, Zurück ───────────────────────────────


def test_session_ausgabe_zeigt_nur_fortschritt(proben: Path, tmp_path: Path) -> None:
    ziel = tmp_path / "s"
    _plan(proben, ziel)
    zeilen: list[str] = []
    o = Orakel(ziel)
    hs.hoeren(ziel, "ich", spiele_fn=o.spiele, taste_fn=o.taste, ausgabe=zeilen.append)
    text = "\n".join(zeilen).lower()
    for verboten in ("kandidat", "bestand", "witness", "c4_vocos", "declip", "stereo", ".wav", "anker"):
        assert verboten not in text, verboten
    assert "Probe 1/" in "\n".join(zeilen)


def test_abbruch_und_wiederaufnahme_verlieren_nichts(proben: Path, tmp_path: Path) -> None:
    ziel = tmp_path / "s"
    _plan(proben, ziel)
    o = Orakel(ziel)
    zaehler = {"n": 0}

    def taste() -> str:
        if o.aktuell.startswith("t"):
            zaehler["n"] += 1
            if zaehler["n"] > 4:
                return "q"
        return o.taste()

    erst = hs.hoeren(ziel, "ich", spiele_fn=o.spiele, taste_fn=taste, ausgabe=lambda _t: None)
    assert erst["status"] == "abgebrochen" and erst["beantwortet"] == 4
    zweit = _hoere(ziel, "ich")
    assert zweit["abgeschlossen"] and zweit["beantwortet"] == zweit["gesamt"]
    assert hs._lade_antworten(ziel / "antworten" / "ich.jsonl").keys() == {
        v["id"] for v in json.loads((ziel / "plan.json").read_text())["versuche"]
    }
    assert _hoere(ziel, "ich")["status"] == "fertig"  # nichts mehr offen


def test_zurueck_nimmt_die_letzte_antwort_zurueck_und_fragt_erneut(proben: Path, tmp_path: Path) -> None:
    ziel = tmp_path / "s"
    _plan(proben, ziel)
    o = Orakel(ziel)
    skript = iter(["x", "1", "z", "2"])  # Kalibrierung, antworten, zurück, neu antworten
    gespielt: list[str] = []

    def spiele(pfad: Path) -> bool:
        gespielt.append(Path(pfad).stem)
        return o.spiele(pfad)

    def taste() -> str:
        try:
            return next(skript)
        except StopIteration:
            return "q"

    ergebnis = hs.hoeren(ziel, "ich", spiele_fn=spiele, taste_fn=taste, ausgabe=lambda _t: None)
    assert ergebnis["status"] == "abgebrochen"
    antworten = hs._lade_antworten(ziel / "antworten" / "ich.jsonl")
    assert antworten["t001"]["antwort"] == "2"  # die widerrufene „1“ zählt nicht, gültig ist die neue Antwort
    assert gespielt.count("t001") == 2  # nach „zurück“ erneut gespielt
    zeilen = (ziel / "antworten" / "ich.jsonl").read_text().splitlines()
    assert any(json.loads(z).get("widerrufen") for z in zeilen)


def test_wiederholen_pause_und_ungueltige_tasten(proben: Path, tmp_path: Path) -> None:
    ziel = tmp_path / "s"
    _plan(proben, ziel)
    o = Orakel(ziel)
    skript = iter(["x", "q"[:0] or "9", "w", "p", "x", "2"])
    gespielt: list[str] = []

    def spiele(pfad: Path) -> bool:
        gespielt.append(Path(pfad).stem)
        return o.spiele(pfad)

    def taste() -> str:
        return next(skript, "q")

    hs.hoeren(ziel, "ich", spiele_fn=spiele, taste_fn=taste, ausgabe=lambda _t: None)
    eintrag = hs._lade_antworten(ziel / "antworten" / "ich.jsonl")["t001"]
    assert eintrag["antwort"] == "2" and eintrag["wiederholungen"] == 1
    assert gespielt.count("t001") == 3  # erstes Spielen, nach „w“, nach „p“; „9“ spielt nicht neu


def test_player_fehler_wird_gemeldet_nicht_verschluckt(proben: Path, tmp_path: Path) -> None:
    ziel = tmp_path / "s"
    _plan(proben, ziel)
    meldungen: list[str] = []
    r = hs.hoeren(ziel, "ich", spiele_fn=lambda _p: False, taste_fn=lambda: "x", ausgabe=meldungen.append)
    assert r["status"] == "player_fehler"


@pytest.mark.parametrize("name", ["", "a b", "../x", "x" * 41, "ä"])
def test_hoerer_kennung_wird_validiert(proben: Path, tmp_path: Path, name: str) -> None:
    ziel = tmp_path / "s"
    _plan(proben, ziel)
    with pytest.raises(ValueError):
        hs.hoeren(ziel, name, spiele_fn=lambda _p: True, taste_fn=lambda: "q", ausgabe=lambda _t: None)


# ── Auswertung: Richtung, Validität, Vollständigkeit, Schnittstelle zu flag_freigabe ──


def test_aufmerksamer_hoerer_fuer_den_kandidaten_ist_buchbar(proben: Path, tmp_path: Path) -> None:
    ziel = tmp_path / "s"
    _plan(proben, ziel)
    _hoere(ziel, "ich", "kandidat")
    erg = hs.auswerten(ziel)
    assert erg["gueltige_hoerer"] == ["ich"] and not erg["simuliert"]
    v = erg["verdikte"]["witness_clicks"]
    assert (v["siege_kandidat"], v["siege_bestand"], v["gleichstaende"]) == (6, 0, 0)
    assert v["p_wert_vorzeichen_test"] == pytest.approx(0.03125) and v["signifikant"] and v["buchbar"]
    assert erg["verdikte"]["c4_vocos"]["buchbar"] is False  # signifikant, aber kein Flag zu buchen
    beleg = json.loads((ziel / "ergebnis" / "hoer_verdikt_witness_clicks.json").read_text())
    assert beleg["schema"] == "aurik.ab_stats/2" and beleg["flag"] == "use_witness_kopf_clicks"
    assert verdikt_ok(beleg)[0]  # direkt von flag_freigabe konsumierbar
    entwurf = (ziel / "ergebnis" / "signoff_entwurf_witness_clicks.md").read_text(encoding="utf-8")
    assert "Maintainer Sign-off: AUSSTEHEND" in entwurf and "use_witness_kopf_clicks" in entwurf
    assert not (ziel / "ergebnis" / "signoff_entwurf_c4_vocos.md").exists()


def test_votum_fuer_den_bestand_ist_nie_buchbar(proben: Path, tmp_path: Path) -> None:
    ziel = tmp_path / "s"
    _plan(proben, ziel)
    _hoere(ziel, "ich", "bestand")
    erg = hs.auswerten(ziel)
    v = erg["verdikte"]["witness_clicks"]
    assert (v["siege_kandidat"], v["siege_bestand"]) == (0, 6) and v["richtung"] == "bestand"
    assert v["signifikant"] and not v["buchbar"]
    beleg = json.loads((ziel / "ergebnis" / "hoer_verdikt_witness_clicks.json").read_text())
    ok, grund = verdikt_ok(beleg)
    assert not ok and "Kandidat" in grund
    assert "Signifikant **für den Bestand**" in (ziel / "ergebnis" / "bericht.md").read_text(encoding="utf-8")


def test_unaufmerksamer_hoerer_wird_ueber_die_anker_ausgeschlossen(proben: Path, tmp_path: Path) -> None:
    ziel = tmp_path / "s"
    _plan(proben, ziel)
    _hoere(ziel, "rater", "rate1")
    erg = hs.auswerten(ziel)
    assert erg["gueltige_hoerer"] == [] and "rater" in erg["ausgeschlossene_hoerer"]
    v = erg["verdikte"]["witness_clicks"]
    assert v["gueltig"] is False and v["buchbar"] is False
    assert not verdikt_ok(json.loads((ziel / "ergebnis" / "hoer_verdikt_witness_clicks.json").read_text()))[0]


def test_gleichstaende_verbessern_nichts(proben: Path, tmp_path: Path) -> None:
    ziel = tmp_path / "s"
    _plan(proben, ziel)
    _hoere(ziel, "ich", "gleich")
    v = hs.auswerten(ziel)["verdikte"]["witness_clicks"]
    assert (v["siege_kandidat"], v["siege_bestand"], v["gleichstaende"]) == (0, 0, 6)
    assert v["richtung"] == "unentschieden" and not v["signifikant"] and not v["buchbar"]


def test_unvollstaendige_sitzung_ist_nicht_buchbar(proben: Path, tmp_path: Path) -> None:
    ziel = tmp_path / "s"
    _plan(proben, ziel)
    o = Orakel(ziel)
    n = {"n": 0}

    def taste() -> str:
        if o.aktuell.startswith("t"):
            n["n"] += 1
            if n["n"] > 20:
                return "q"
        return o.taste()

    hs.hoeren(ziel, "ich", spiele_fn=o.spiele, taste_fn=taste, ausgabe=lambda _t: None)
    erg = hs.auswerten(ziel)
    unvollstaendig = [b for b, v in erg["verdikte"].items() if not v["vollstaendig"]]
    assert unvollstaendig  # mindestens ein Block ist nicht komplett beurteilt
    assert all(not erg["verdikte"][b]["buchbar"] for b in unvollstaendig)


def test_mehrere_hoerer_zaehlen_je_paar_per_mehrheit(proben: Path, tmp_path: Path) -> None:
    ziel = tmp_path / "s"
    _plan(proben, ziel)
    _hoere(ziel, "eins", "kandidat")
    _hoere(ziel, "zwei", "bestand")
    v = hs.auswerten(ziel)["verdikte"]["witness_clicks"]
    assert v["hoerer"] == ["eins", "zwei"]
    assert (v["siege_kandidat"], v["siege_bestand"], v["gleichstaende"]) == (0, 0, 6)  # je Paar Patt → Gleichstand
    assert not v["signifikant"]


def test_simulierter_lauf_ist_nie_buchbar(proben: Path, tmp_path: Path) -> None:
    ziel = tmp_path / "s"
    _plan(proben, ziel, simuliert=True)
    _hoere(ziel, "ich", "kandidat")
    erg = hs.auswerten(ziel)
    assert erg["simuliert"] and not erg["verdikte"]["witness_clicks"]["buchbar"]
    beleg = json.loads((ziel / "ergebnis" / "hoer_verdikt_witness_clicks.json").read_text())
    ok, grund = verdikt_ok(beleg)
    assert not ok and "simuliert" in grund
    assert "SIMULIERTER LAUF" in (ziel / "ergebnis" / "bericht.md").read_text(encoding="utf-8")


def test_manipulierter_plan_wird_bei_der_auswertung_abgelehnt(proben: Path, tmp_path: Path) -> None:
    ziel = tmp_path / "s"
    _plan(proben, ziel)
    plan = json.loads((ziel / "plan.json").read_text())
    plan["plan_sha256"] = "0" * 64
    (ziel / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="plan_sha256"):
        hs.auswerten(ziel)


def test_bericht_nennt_zurueckgestellte_bloecke_mit_naechstem_schritt(proben: Path, tmp_path: Path) -> None:
    ziel = tmp_path / "s"
    _plan(proben, ziel)
    _hoere(ziel, "ich")
    hs.auswerten(ziel)
    bericht = (ziel / "ergebnis" / "bericht.md").read_text(encoding="utf-8")
    assert "Zurückgestellt" in bericht and "declip" in bericht and "kleinstmögliches p=0.25" in bericht
    assert "erstelle_aspade_hoerpaare.py" in bericht


# ── CLI ──────────────────────────────────────────────────────────────────────


def test_cli_bauen_und_status(proben: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ziel = tmp_path / "s"
    assert hs.main(["bauen", "--proben", str(proben), "--sitzung", str(ziel)]) == 0
    ausgabe = capsys.readouterr().out
    assert "Zurückgestellt" in ausgabe and "declip" in ausgabe
    assert hs.main(["status", "--sitzung", str(ziel)]) == 0
    assert "Proben" in capsys.readouterr().out


def test_cli_los_ohne_hoerer_antworten_bricht_sauber_ab(
    proben: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ziel = tmp_path / "s"
    monkeypatch.setattr(hs, "spiele", lambda _p, _pl=None: True)
    monkeypatch.setattr(hs, "lese_taste", lambda: "q")
    assert hs.main(["los", "--proben", str(proben), "--sitzung", str(ziel), "--hoerer", "ich"]) == 0
    assert (ziel / "plan.json").is_file()
    assert not list((ziel / "ergebnis").glob("*.json"))  # abgebrochen ⇒ keine Auswertung
