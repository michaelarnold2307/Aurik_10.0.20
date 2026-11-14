"""Vertragstests Defekt-Delta-Witness (§G191 (GEBOTE.md), Spec 07 §8.6).

Modell-frei und deterministisch (§G5 (GEBOTE.md)): alle Fälle laufen über
synthetische Signale mit injizierten Metriken — kein Modell, keine GPU,
keine Umgebungsabhängigkeit. Zitierdisziplin AGENTS.md §5: IDs immer mit
Quelle qualifiziert.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent

from backend.core.dsp import defekt_delta as dd
from backend.core.dsp.wohlklang_zeuge import JND_TOLERANZ, defektlast  # EINE Wahrheit


def _rng_seg(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.05, size=n).astype(np.float32)


def _metrik_rms(audio: object, sample_rate: int) -> float:
    """Synthetische Metrik: RMS in dB — deterministisch, injizierbar."""
    x = np.asarray(audio, dtype=np.float64).reshape(-1)
    rms = float(np.sqrt(np.mean(np.square(x)) + 1e-12))
    return float(np.log10(rms + 1e-12))


def _regionen(typ: str, n_reg: int = 2, laenge: int = 8192) -> list[dict]:
    return [{"start": i * laenge, "end": (i + 1) * laenge, "typ": typ, "staerke": 0.5} for i in range(n_reg)]


def test_defektlast_lang_ist_auf_dem_zertifikatsfenster_bitidentisch() -> None:
    """§G191 (GEBOTE.md): die Voll-Fenster-Variante darf die zertifizierte
    Referenz auf deren 16384-Samples-Fenster nicht verändern (Äquivalenz-Pin)."""
    for seed in (42, 7):
        x = _rng_seg(16384, seed=seed)
        assert dd.defektlast_lang(x, 44100) == defektlast(x, 44100)


def test_defektlast_lang_sieht_spaete_clips_die_die_referenz_verpasst() -> None:
    """D-K3-95/97-Grenze: die Referenz liest nur 16384 Samples — ein Clip
    hinter dem Fenster ist für sie unsichtbar, für `defektlast_lang` nicht."""
    x = _rng_seg(65536, seed=42)
    basis = dd.defektlast_lang(x, 44100)
    x_late = x.copy()
    x_late[40000:40800] = 1.0  # harter Clip weit hinter dem Referenzfenster
    assert dd.defektlast_lang(x_late, 44100) > basis
    assert defektlast(x_late, 44100) == defektlast(x, 44100)  # Referenz bleibt blind


def test_pegel_matching_gleicht_restauriert_auf_den_schaden() -> None:
    """D-K3-95-Lehre: ohne Matching misst man den Loudness-Regelkreis — mit
    identischen Regeln liefert die RMS-Metrik identische Werte je Region."""
    schaden = _rng_seg(16384, seed=1) * 0.5
    restauriert = _rng_seg(16384, seed=1) * 0.25  # Gain 0,5-Regelkreis
    reg = [{"start": 0, "end": 16384, "typ": "vinyl_noise", "staerke": 0.5}]
    beleg = dd.berechne_defekt_delta(
        regionen=reg,
        schaden=schaden,
        restauriert=restauriert,
        clean=None,
        sample_rate=44100,
        metrik=_metrik_rms,
        metrik_name="rms_test",
        metrik_richtung=dd.RICHTUNG_NIEDRIGER,
    )
    werte = beleg["regionen"][0]
    assert abs(werte["metrik_schaden"] - werte["metrik_restauriert"]) < 1e-6


def test_reparaturquote_exakt() -> None:
    """Q = Δ(schaden→restauriert) / Δ(schaden→clean) — exakte Rechnung mit
    einer formbasierten Metrik (Vorzeichen-Anteil), die das Pegel-Matching
    nicht nivelliert (D-K3-95-Lehre: Pegel-Metriken messen den Regelkreis)."""
    n = 16384
    reg = [{"start": 0, "end": n, "typ": "clipping", "staerke": 0.8}]

    def metrik_pos(audio: object, sample_rate: int) -> float:
        x = np.asarray(audio, dtype=np.float64).reshape(-1)
        return float(np.mean(x > 0.0))

    # gleiche RMS je Signal ⇒ Gain-Matching 1,0; Formen: 100 % / 50 % / 0 % positiv
    schaden = np.full(n, 0.6, dtype=np.float32)  # metrik 1.0
    restauriert = np.tile(np.array([0.6, -0.6], dtype=np.float32), n // 2)  # metrik 0.5
    clean = np.full(n, -0.6, dtype=np.float32)  # metrik 0.0
    beleg = dd.berechne_defekt_delta(
        regionen=reg,
        schaden=schaden,
        restauriert=restauriert,
        clean=clean,
        sample_rate=44100,
        metrik=metrik_pos,
        metrik_name="pos_test",
        metrik_richtung=dd.RICHTUNG_NIEDRIGER,
    )
    werte = beleg["regionen"][0]
    # Δ = 1.0 − 0.5 = 0.5; Nenner = 1.0 − 0.0 = 1.0 ⇒ Q = 0.5
    assert werte["delta"] == pytest.approx(0.5, abs=1e-5)
    assert werte["reparaturquote_q"] == pytest.approx(0.5, abs=1e-5)


def test_jnd_grenzen() -> None:
    """JND-Klassifikation mit der einen Wahrheit `JND_TOLERANZ` (0,05) —
    formbasierte Metrik (Vorzeichen-Anteil), die das Pegel-Matching nicht
    nivelliert (D-K3-95-Lehre)."""
    n = 16384
    eps = 0.001

    def metrik_pos(audio: object, sample_rate: int) -> float:
        x = np.asarray(audio, dtype=np.float64).reshape(-1)
        return float(np.mean(x > 0.0))

    def _flip(x: np.ndarray, k: int) -> np.ndarray:
        y = x.copy()
        y[:k] = -y[:k]
        return y

    schaden = np.full(n, 0.6, dtype=np.float32)  # metrik 1.0

    for d, erwartet in (
        (JND_TOLERANZ + eps, "ueber_jnd"),
        (JND_TOLERANZ - eps, "unter_jnd"),
    ):
        k = int(round(d * n))  # delta = k/n (Vorzeichen-Wechsel erhalten die RMS)
        restauriert = _flip(schaden, k)  # metrik = 1 − k/n
        beleg = dd.berechne_defekt_delta(
            regionen=[{"start": 0, "end": n, "typ": "kassette", "staerke": 0.5}],
            schaden=schaden,
            restauriert=restauriert,
            clean=None,
            sample_rate=44100,
            metrik=metrik_pos,
            metrik_name="pos_jnd_test",
            metrik_richtung=dd.RICHTUNG_NIEDRIGER,
        )
        assert beleg["regionen"][0]["jnd_klasse"] == erwartet, (
            d,
            beleg["regionen"][0]["delta"],
        )

    # verschlechtert: der Schaden hat FEWIGER positives Signal als die Restauration
    schaden_halb = _flip(schaden, n // 2)  # metrik 0.5
    restauriert_schlecht = np.full(n, 0.6, dtype=np.float32)  # metrik 1.0
    beleg = dd.berechne_defekt_delta(
        regionen=[{"start": 0, "end": n, "typ": "kassette", "staerke": 0.5}],
        schaden=schaden_halb,
        restauriert=restauriert_schlecht,
        clean=None,
        sample_rate=44100,
        metrik=metrik_pos,
        metrik_name="pos_jnd_test",
        metrik_richtung=dd.RICHTUNG_NIEDRIGER,
    )
    assert beleg["regionen"][0]["jnd_klasse"] == "verschlechtert"


def test_stereo_layout_invariante() -> None:
    """AGENTS.md §3: (C, N) und (N, C) müssen identische Ergebnisse liefern."""
    n = 16384
    l = _rng_seg(n, seed=3)
    r = _rng_seg(n, seed=4)
    reg = [{"start": 0, "end": n, "typ": "mp3_128", "staerke": 0.4}]
    cn = np.stack([l, r], axis=0)
    nc = np.stack([l, r], axis=1)
    b1 = dd.berechne_defekt_delta(regionen=reg, schaden=cn, restauriert=cn, clean=cn, sample_rate=44100)
    b2 = dd.berechne_defekt_delta(regionen=reg, schaden=nc, restauriert=nc, clean=nc, sample_rate=44100)
    assert b1["aggregate"] == b2["aggregate"]


def test_stille_region_erhaelt_gain_1_ohne_nan() -> None:
    """§V6 (copilot-instructions.md): Stille darf nicht verstärkt werden —
    Gain 1,0 und keine NaN/Inf im Beleg."""
    n = 16384
    reg = [{"start": 0, "end": n, "typ": "dropouts", "staerke": 0.2}]
    beleg = dd.berechne_defekt_delta(
        regionen=reg,
        schaden=np.zeros(n, dtype=np.float32),
        restauriert=np.zeros(n, dtype=np.float32),
        clean=np.zeros(n, dtype=np.float32),
        sample_rate=44100,
    )
    w = beleg["regionen"][0]
    assert w["gain_restauriert"] == 1.0
    assert w["gain_clean"] == 1.0
    assert np.isfinite(w["delta"])
    assert np.isfinite(w["metrik_schaden"])


def test_schema_fail_closed() -> None:
    """Fremdes Schema wird abgelehnt, /1 (RMS-Matching) als veraltet markiert
    statt still interpretiert (Muster ohr_prior/2-Ablehnung, D-K3-89)."""
    gueltig = {"schema": dd.SCHEMA, "jnd": 0.05, "aggregate": {}}
    assert dd.lade_defekt_delta(gueltig)["verfuegbar"] is True
    alt = {"schema": "aurik.defektdelta/1", "aggregate": {}}
    geladen_alt = dd.lade_defekt_delta(alt)
    assert geladen_alt["verfuegbar"] is False
    assert "veraltet" in geladen_alt["grund"]
    fremd = {"schema": "aurik.defektdelta/9", "aggregate": {}}
    geladen = dd.lade_defekt_delta(fremd)
    assert geladen["verfuegbar"] is False
    assert "fremdes" in geladen["grund"]


def test_flat_top_saettigung_wird_relativ_erkannt() -> None:
    """Erstbeleg §3 (D-K3-97-Klasse): die relative Flat-Top-Statistik muss
    Sättigung sehen, die das Body-Matching nicht nivelliert — verschlechtert
    in jnd_klasse_flat_top bei hart geclippter Variante."""
    n = 16384
    t = np.linspace(0.0, 2.0 * np.pi * 256.0, n, endpoint=False)
    schaden = (0.5 * np.sin(t)).astype(np.float32)  # sauberer Sinus
    variant = np.clip(schaden, -0.35, 0.35).astype(np.float32)  # Flat-Tops bei ±0,35
    reg = [{"start": 0, "end": n, "typ": "soft_saturation", "staerke": 0.7}]
    beleg = dd.berechne_defekt_delta(
        regionen=reg,
        schaden=schaden,
        restauriert=variant,
        clean=None,
        sample_rate=44100,
    )
    w = beleg["regionen"][0]
    # p25-Matching lässt den Body gleich (Gain ≈ 1) — die Decke bleibt relativ sichtbar
    assert w["gain_restauriert"] == pytest.approx(1.0, abs=0.05)
    assert w["flat_top_schaden"] < 0.05
    assert w["flat_top_restauriert"] > 0.3
    assert w["delta_flat_top"] < -0.25
    assert w["jnd_klasse_flat_top"] == "verschlechtert"
    # Zusatzkomponente aggregiert je Typ
    assert beleg["aggregate"]["soft_saturation"]["flat_top_delta_median"] < -0.25


def test_aggregat_nach_defektklasse() -> None:
    """Aggregat je DefectType (n, Median Δ, Median Q) — formbasierte Metrik,
    die das Pegel-Matching nicht nivelliert (D-K3-95-Lehre)."""
    n = 8192

    def metrik_pos(audio: object, sample_rate: int) -> float:
        x = np.asarray(audio, dtype=np.float64).reshape(-1)
        return float(np.mean(x > 0.0))

    reg = [
        {"start": 0, "end": n, "typ": "clipping", "staerke": 0.9},
        {"start": n, "end": 2 * n, "typ": "clipping", "staerke": 0.9},
        {"start": 2 * n, "end": 3 * n, "typ": "vinyl_noise", "staerke": 0.3},
    ]
    schaden = np.full(3 * n, 0.6, dtype=np.float32)  # metrik 1.0
    restauriert = np.tile(np.array([0.6, -0.6], dtype=np.float32), (3 * n) // 2)  # metrik 0.5
    clean = np.full(3 * n, -0.6, dtype=np.float32)  # metrik 0.0
    beleg = dd.berechne_defekt_delta(
        regionen=reg,
        schaden=schaden,
        restauriert=restauriert,
        clean=clean,
        sample_rate=44100,
        metrik=metrik_pos,
        metrik_name="pos_agg_test",
        metrik_richtung=dd.RICHTUNG_NIEDRIGER,
    )
    agg = beleg["aggregate"]
    assert agg["clipping"]["n"] == 2
    assert agg["vinyl_noise"]["n"] == 1
    assert agg["clipping"]["delta_median"] == pytest.approx(0.5, abs=1e-5)
    # Q = Δ(1,0 → 0,5) / Δ(1,0 → 0,0) = 0,5 / 1,0
    assert agg["clipping"]["reparaturquote_median"] == pytest.approx(0.5, abs=1e-3)
    # §G13 (copilot-instructions.md): Q >= REPARATURQUOTE_ZIEL (0,5) erfüllt.
    assert agg["clipping"]["q_ziel_erfuellt"] is True


def test_q_ziel_erfuellt_verfehlt_und_nicht_bewertbar() -> None:
    """§G13 (copilot-instructions.md): q_ziel_erfuellt je Defektklasse —
    True ab Q >= 0,5, False bei keiner Rückgewinnung, None ohne Clean."""
    n = 4096

    def metrik_pos(audio: object, sample_rate: int) -> float:
        x = np.asarray(audio, dtype=np.float64).reshape(-1)
        return float(np.mean(x > 0.0))

    reg = [
        {"start": 0, "end": n, "typ": "repariert", "staerke": 0.9},
        {"start": n, "end": 2 * n, "typ": "unrepariert", "staerke": 0.9},
    ]
    schaden = np.full(2 * n, 0.6, dtype=np.float32)  # metrik 1.0
    restauriert = np.tile(np.array([0.6, -0.6], dtype=np.float32), n)  # metrik 0.5
    clean = np.full(2 * n, -0.6, dtype=np.float32)  # metrik 0.0
    beleg = dd.berechne_defekt_delta(
        regionen=reg,
        schaden=schaden,
        restauriert=restauriert,
        clean=clean,
        sample_rate=44100,
        metrik=metrik_pos,
        metrik_name="pos_q_test",
        metrik_richtung=dd.RICHTUNG_NIEDRIGER,
    )
    agg = beleg["aggregate"]
    assert agg["repariert"]["q_ziel_erfuellt"] is True
    # ohne Clean: Q nicht bewertbar (None), nie still True
    beleg_ohne_clean = dd.berechne_defekt_delta(
        regionen=reg[:1],
        schaden=schaden[:n],
        restauriert=restauriert[:n],
        clean=None,
        sample_rate=44100,
        metrik=metrik_pos,
        metrik_name="pos_q_test",
        metrik_richtung=dd.RICHTUNG_NIEDRIGER,
    )
    assert beleg_ohne_clean["aggregate"]["repariert"]["q_ziel_erfuellt"] is None
    # unrepariert: nur 1/5 des Schadens zurückgewonnen => Q = 0,2 < 0,5
    restauriert_schwach = np.tile(np.array([0.6, 0.6, 0.6, 0.6, -0.6], dtype=np.float32), n // 5)  # metrik 0.8
    beleg_flat = dd.berechne_defekt_delta(
        regionen=[{"start": 0, "end": n, "typ": "unrepariert", "staerke": 0.9}],
        schaden=schaden[:n],
        restauriert=restauriert_schwach,
        clean=clean[:n],
        sample_rate=44100,
        metrik=metrik_pos,
        metrik_name="pos_q_test",
        metrik_richtung=dd.RICHTUNG_NIEDRIGER,
    )
    assert beleg_flat["aggregate"]["unrepariert"]["reparaturquote_median"] == pytest.approx(0.2, abs=1e-2)
    assert beleg_flat["aggregate"]["unrepariert"]["q_ziel_erfuellt"] is False


def test_regionen_aus_defekt_scores() -> None:
    """Spec 07 §8.6: Locations (Sekunden) → Regionen (Samples), Schwelle 0,1,
    sortiert, staerke = severity — deterministisch (§G5 (GEBOTE.md))."""

    class _Score:
        def __init__(self, value: str, severity: float, locations: list[tuple[float, float]]) -> None:
            self.value = value
            self.severity = severity
            self.locations = locations

    scores = {
        "clipping": _Score("clipping", 0.8, [(1.0, 2.0), (10.0, 10.5)]),
        "vinyl_noise": _Score("vinyl_noise", 0.05, [(5.0, 6.0)]),  # unter Schwelle
        "dropouts": _Score("dropouts", 0.4, []),
    }
    regionen = dd.regionen_aus_defekt_scores(scores, sample_rate=44100)
    assert regionen == [
        {"start": 44100, "end": 88200, "typ": "clipping", "staerke": 0.8},
        {"start": 441000, "end": 463050, "typ": "clipping", "staerke": 0.8},
    ]


def test_kontrollregionen_deterministisch_ohne_ueberlappung() -> None:
    """Spec 07 §8.6: Kontrollregionen bei festen Anteilen, maximal 3, nie
    überlappend mit Defektregionen, wiederholbar identisch."""
    sr = 44100
    dauer = 100 * sr
    # Defekt deckt den 0,1-Kandidaten (9–11 s) ab ⇒ der wird übersprungen
    defekt = [{"start": 8 * sr, "end": 12 * sr, "typ": "kassette", "staerke": 0.5}]
    kontrollen = dd.kontrollregionen_aus_abdeckung(defekt, dauer, sr)
    assert 0 < len(kontrollen) <= 3
    assert kontrollen == dd.kontrollregionen_aus_abdeckung(defekt, dauer, sr)  # deterministisch
    for k in kontrollen:
        assert k["typ"] == "kontrolle"
        assert k["staerke"] == 0.0
        assert k["end"] > k["start"] + 2048
        assert not (k["start"] < defekt[0]["end"] and defekt[0]["start"] < k["end"])
    anteil_10 = int(round(dauer * 0.1))
    assert not any(k["start"] <= anteil_10 < k["end"] for k in kontrollen)


def test_saettigungskanal_gain_invariant() -> None:
    """D-K3-95/97-Lücke: 1/crest muss Skalierung kürzen (relativ, gain-invariant)."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(8192).astype(np.float32) + 0.5 * np.sin(np.linspace(0, 3000 * np.pi, 8192))
    x = (x / np.abs(x).max() * 0.8).astype(np.float32)
    assert dd.saettigung_kanal(x) == pytest.approx(dd.saettigung_kanal(x * 0.1), abs=1e-3)
    assert dd.saettigung_kanal(x) == pytest.approx(dd.saettigung_kanal(x * 2.5), abs=1e-3)


def test_saettigungskanal_sieht_tanh_saettigung() -> None:
    """k398-Clipping-Rezeptur: tanh-Sättigung senkt den Crest-Faktor — der
    Kanal muss sie auf beliebigem Material sehen (universelle Eigenschaft)."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal(16384).astype(np.float64) + 0.8 * np.sin(np.linspace(0, 4000 * np.pi, 16384))
    x = (x / np.abs(x).max()).astype(np.float32)
    gesaettigt = (np.tanh(8 * x) / np.tanh(8)).astype(np.float32)
    assert dd.saettigung_kanal(gesaettigt) > dd.saettigung_kanal(x) + 0.05


_MUSDB_MIXTURES = ROOT / "data" / "musdb_mixtures"


def _musdb_beispiel() -> np.ndarray | None:
    """Erste MUSDB-Mixture (falls lokal vorhanden), mono, 20 s."""
    titel = sorted(_MUSDB_MIXTURES.glob("*.wav"))
    if not titel:
        return None
    from scipy.io import wavfile

    sr, d = wavfile.read(titel[0])
    x = np.asarray(d, np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = x / (np.abs(x).max() + 1e-12)
    return np.ascontiguousarray(x[: 20 * sr])


@pytest.mark.skipif(_MUSDB_MIXTURES is None or not _MUSDB_MIXTURES.exists(), reason="MUSDB-Mixtures nicht vorhanden")
def test_saettigungskanal_sieht_was_defektlast_uebersieht_auf_musik() -> None:
    """Echte Musik (D-K3-95/97-Befund 2026-11-14): die k398-Rezeptur ändert
    defektlast nicht messbar (blind), der Sättigungskanal sieht sie deutlich."""
    x = _musdb_beispiel()
    assert x is not None
    gesaettigt = (np.tanh(8 * x) / np.tanh(8)).astype(np.float32)
    sr = 48000
    # defektlast-Delta unter JND 0,05 = Blindheit (gemessener Befund)
    delta_last = abs(dd.defektlast_lang(x, sr) - dd.defektlast_lang(gesaettigt, sr))
    assert delta_last < 0.05
    # Sättigungskanal sieht die Rezeptur deutlich
    assert dd.saettigung_kanal(gesaettigt) > dd.saettigung_kanal(x) + 0.1


def test_saettigungskanal_sinus_normiert_und_stille_sicher() -> None:
    n = 8192
    t = np.linspace(0.0, 2.0 * np.pi * 100.0, n, endpoint=False)
    sinus = np.sin(t).astype(np.float32)
    assert dd.saettigung_kanal(sinus) == pytest.approx(1.0 / np.sqrt(2.0), abs=0.02)
    assert dd.saettigung_kanal(np.zeros(n, dtype=np.float32)) == 0.0


def test_schema_3_region_enthaelt_saettigungskanal() -> None:
    """Schema /3: jede Region trägt den Sättigungs-Zweitkanal + Aggregat."""
    n = 16384
    t = np.linspace(0.0, 2.0 * np.pi * 256.0, n, endpoint=False)
    schaden = (0.5 * np.sin(t)).astype(np.float32)
    gesaettigt = (np.tanh(6 * schaden) / np.tanh(6) * 0.5).astype(np.float32)
    beleg = dd.berechne_defekt_delta(
        regionen=[{"start": 0, "end": n, "typ": "soft_saturation", "staerke": 0.7}],
        schaden=gesaettigt,
        restauriert=schaden,
        clean=schaden,
        sample_rate=44100,
    )
    assert beleg["schema"] == "aurik.defektdelta/3"
    w = beleg["regionen"][0]
    assert "delta_saettigung" in w and "jnd_klasse_saettigung" in w
    assert w["delta_saettigung"] > 0.02
    assert w["jnd_klasse_saettigung"] == "ueber_jnd"
    agg = beleg["aggregate"]["soft_saturation"]
    assert agg["saettigung_delta_median"] > 0.02
    assert agg["anteil_ueber_jnd_saettigung"] > 0.5
