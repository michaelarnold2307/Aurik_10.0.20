"""Defekt-Delta-Witness — Reparatur-Nachweis je Defektklasse (§G191 (GEBOTE.md)).

Die zertifizierten Ganzclip-Metriken sehen Reparatur in einzelnen
Defektregionen nicht (defektlast: 0,34-s-Fenster + Sättigungs-Blindheit,
D-K3-95/97; musik_mos integriert über den Clip). Dieses Modul misst deshalb
Metriken NUR in den annotierten Defektregionen, vorher (Schaden) und
nachher (restauriert), pegel-gematcht, und weist die **Reparaturquote**
Q = Δ(schaden→restauriert) / Δ(schaden→clean) je Region und Defektklasse aus.

Status: **Report-Only-Zeuge** (Hörordnung §7) — kein Gate, kein Rollback,
keine Export-Blockade, kein Flag. Der Witness trägt **kein**
Richtungszertifikat, solange seine Regionen-Urteile nicht gegen Hör-Labels
kalibriert sind (D-K4-9); Stärke-/Schwellen-Nutzung NUR über das
§4-Verfahren. Determinismus: §G5 (copilot-instructions.md) — kein
`time.time()`, Seed-frei (reine Funktionen), sha256 der Eingänge im Beleg.

Schema: ``aurik.defektdelta/1`` — fail-closed (Muster ohr_prior/2-Ablehnung).
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from backend.core.dsp.wohlklang_zeuge import JND_TOLERANZ  # EINE Wahrheit

logger = logging.getLogger(__name__)

SCHEMA = "aurik.defektdelta/2"
_SCHEMA_VERALTET = "aurik.defektdelta/1"  # RMS-Matching-Definition (Belege veraltet, §G191 (GEBOTE.md))
# defektlast ist "höher = mehr Defekte"; musik_mos "höher = besser".
RICHTUNG_NIEDRIGER = "niedriger_besser"
RICHTUNG_HOEHER = "hoeher_besser"
_METRIK_RICHTUNGEN: dict[str, str] = {
    "defektlast_lang": RICHTUNG_NIEDRIGER,
    "musik_mos": RICHTUNG_HOEHER,
}
_MIN_SAMPLES = 2048  # Untergrenze der Metrik-Definition (2048er-FFT)
_JND = float(JND_TOLERANZ)


def _mono(audio: Any) -> np.ndarray:
    """Stereo-Layout-Invariante (AGENTS.md §3): (C, N) und (N, C) ⇒ mono.

    Gleiche Regel wie ``wohlklang_zeuge._mono`` (Kanäle ≤ 8 und längste Achse
    hinten ⇒ transponieren, dann Kanal-Mittelung) — bewusst ohne dtype-Zwang,
    damit der Wert bit-identisch zur Mess-Referenz bleibt.
    """
    x: np.ndarray = np.asarray(audio)
    if x.ndim == 2:
        if x.shape[0] <= 8 and x.shape[-1] > 8:
            x = x.T
        x = np.asarray(x.mean(axis=1), dtype=x.dtype)
    _mono_out: np.ndarray = np.ascontiguousarray(x.reshape(-1))
    return _mono_out


def defektlast_lang(audio: Any, sample_rate: int) -> float:
    """Voll-Fenster-Verallgemeinerung der zertifizierten Referenz.

    Identische Konstruktion wie ``wohlklang_zeuge.defektlast`` (Clip-Statistik,
    Spektral-Flatness, HF-Verlust), aber OHNE die 16384-Samples-Kappung:
    die Clip-Statistik läuft über die gesamte Region (die Spektral-Komponenten
    nutzen weiterhin das erste 2048er-Fenster — wie die Referenz).

    Auf Eingängen ≤ 16384 Samples ist der Wert **bit-identisch** zur
    zertifizierten Referenz (Test-gepinnt). Trägt **nicht** das
    Richtungszertifikat; Grenzen der Referenz (Sättigungs-Blindheit,
    D-K3-97) gelten sinngemäß und sind im Zertifikat zu dokumentieren.
    """
    seg = _mono(audio)
    clip = float(np.mean(np.abs(seg) > 0.98))
    spec = np.abs(np.fft.rfft(seg, n=2048)) ** 2 + 1e-18
    flatness = float(np.clip(np.exp(np.mean(np.log(spec))) / (np.mean(spec) + 1e-12), 0.0, 1.0))
    freqs = np.fft.rfftfreq(2048, d=1.0 / max(int(sample_rate), 1))
    hf = float(np.sum(spec[freqs > 8000]) / (np.sum(spec) + 1e-12))
    hf_verlust = max(0.0, 1.0 - min(1.0, hf / 0.02))
    return round(clip * 2.0 + flatness + hf_verlust * 0.5, 4)


def pegel_gain(signal: np.ndarray, referenz: np.ndarray) -> float:
    """Body-Perzentil-Matching (p25 von |x|), um ``signal`` auf den Pegel der
    ``referenz`` zu heben.

    D-K3-95-Lehre: ungematcht misst man den Loudness-Regelkreis, nicht die
    Reparatur. RMS ist dafür **ungeeignet**: Sättigung hebt die RMS an, das
    RMS-Matching zieht den Sättigungs-Beweis wieder unter die Schwelle
    (Erstbeleg 2026-11-11_b3_defektdelta_erstbelege.md §3). Das p25-Body-
    Perzentil ist ausreißer-/Flat-Top-robust. Stille-Referenz (p25 unter
    1e-6) ⇒ Gain 1,0 — §V6 (copilot-instructions.md): kein stiller Ausfall,
    Warnung wird geloggt.
    """
    a = np.abs(np.asarray(signal, dtype=np.float64).reshape(-1))
    b = np.abs(np.asarray(referenz, dtype=np.float64).reshape(-1))
    if a.size < 4 or b.size < 4:
        logger.warning("§V6 (copilot-instructions.md) pegel_gain: Signal zu kurz — Gain 1,0")
        return 1.0
    p25_sig = float(np.percentile(a, 25.0))
    p25_ref = float(np.percentile(b, 25.0))
    if p25_ref < 1e-6:
        logger.warning("§V6 (copilot-instructions.md) pegel_gain: Referenz ist Stille — Gain 1,0")
        return 1.0
    if p25_sig < 1e-6:
        logger.warning("§V6 (copilot-instructions.md) pegel_gain: Signal ist Stille — Gain 1,0")
        return 1.0
    return min(max(p25_ref / p25_sig, 0.25), 4.0)


def flat_top_anteil(audio: Any, band: float = 0.002) -> float:
    """Sättigungs-Zusatzkomponente: Anteil der Samples an der EIGENEN
    Hüllen-Decke (|x| ≥ max(|x|) − band).

    Relativ und gain-invariant: Skalierung verschiebt die Decke mit — die
    absolute 0,98-Schwelle der zertifizierten Referenz kann genau das nicht
    (Erstbeleg §3, D-K3-97-Klasse). Höher = mehr Sättigung.
    **Nicht zertifiziert** — Report-Only-Zeuge (§G191 (GEBOTE.md)).
    """
    x = _mono(audio)
    if x.size == 0:
        return 0.0
    m = float(np.max(np.abs(x)))
    if m < 1e-6:
        return 0.0
    return float(np.mean(np.abs(x) >= m - band))


def _delta_qualitaet(wert_schaden: float, wert_restauriert: float, richtung: str) -> float:
    """Delta in Qualitätsrichtung: > 0 bedeutet besser."""
    if richtung == RICHTUNG_NIEDRIGER:
        return float(wert_schaden - wert_restauriert)
    return float(wert_restauriert - wert_schaden)


def _jnd_klasse(delta: float) -> str:
    if delta <= -_JND:
        return "verschlechtert"
    if delta >= _JND:
        return "ueber_jnd"
    return "unter_jnd"


def _slice_region(audio_mono: np.ndarray, region: dict[str, Any]) -> np.ndarray:
    start = int(region["start"])
    end = int(region["end"])
    _region: np.ndarray = np.ascontiguousarray(audio_mono[start:end])
    return _region


def berechne_defekt_delta(
    regionen: list[dict[str, Any]],
    schaden: Any,
    restauriert: Any,
    clean: Any | None,
    sample_rate: int,
    *,
    metrik: Callable[[Any, int], float] | None = None,
    metrik_name: str = "defektlast_lang",
    metrik_richtung: str | None = None,
) -> dict[str, Any]:
    """Berechnet Regionen-Deltas, Reparaturquote und Aggregat je DefectType.

    Raises:
        ValueError: unbekannte Metrik-Richtung oder zu kurzes Signal.
    """
    richtung = metrik_richtung or _METRIK_RICHTUNGEN.get(metrik_name)
    if richtung not in (RICHTUNG_NIEDRIGER, RICHTUNG_HOEHER):
        raise ValueError(f"Unbekannte Metrik-Richtung für {metrik_name!r}")
    messe = metrik if metrik is not None else (defektlast_lang if metrik_name == "defektlast_lang" else None)
    if messe is None:
        raise ValueError(f"Keine Metrik-Funktion für {metrik_name!r} — metrik= explizit übergeben")

    mono_schaden = _mono(schaden)
    mono_rest = _mono(restauriert)
    mono_clean = _mono(clean) if clean is not None else None
    if mono_schaden.size < _MIN_SAMPLES:
        raise ValueError("Signal zu kurz für die Metrik-Definition (< 2048 Samples)")

    ergebnisse: list[dict[str, Any]] = []
    for region in regionen:
        r_schaden = _slice_region(mono_schaden, region)
        r_rest = _slice_region(mono_rest, region)
        if r_schaden.size < _MIN_SAMPLES or r_rest.size < _MIN_SAMPLES:
            # §V6 (copilot-instructions.md): kein stiller Ausfall.
            logger.warning(
                "§V6 (copilot-instructions.md) Region %s zu kurz (%d Samples) — übersprungen",
                region.get("typ", "?"),
                min(r_schaden.size, r_rest.size),
            )
            continue
        gain_rest = pegel_gain(r_rest, r_schaden)
        w_schaden = float(messe(r_schaden, int(sample_rate)))
        w_rest = float(messe(r_rest * gain_rest, int(sample_rate)))
        delta = _delta_qualitaet(w_schaden, w_rest, richtung)
        ft_schaden = flat_top_anteil(r_schaden)
        ft_rest = flat_top_anteil(r_rest * gain_rest)
        eintrag: dict[str, Any] = {
            "start": int(region["start"]),
            "end": int(region["end"]),
            "typ": str(region.get("typ", "unbekannt")),
            "staerke": region.get("staerke"),
            "metrik_schaden": w_schaden,
            "metrik_restauriert": w_rest,
            "delta": round(delta, 4),
            "jnd_klasse": _jnd_klasse(delta),
            "gain_restauriert": round(gain_rest, 4),
            "flat_top_schaden": round(ft_schaden, 4),
            "flat_top_restauriert": round(ft_rest, 4),
            "delta_flat_top": round(ft_schaden - ft_rest, 4),
            "jnd_klasse_flat_top": _jnd_klasse(ft_schaden - ft_rest),
            "reparaturquote_q": None,
        }
        if mono_clean is not None:
            r_clean = _slice_region(mono_clean, region)
            if r_clean.size >= _MIN_SAMPLES:
                gain_clean = pegel_gain(r_clean, r_schaden)
                w_clean = float(messe(r_clean * gain_clean, int(sample_rate)))
                delta_clean = _delta_qualitaet(w_schaden, w_clean, richtung)
                eintrag["metrik_clean"] = w_clean
                eintrag["gain_clean"] = round(gain_clean, 4)
                eintrag["flat_top_clean"] = round(flat_top_anteil(r_clean * gain_clean), 4)
                if delta_clean > _JND:
                    eintrag["reparaturquote_q"] = round(delta / delta_clean, 4)
                else:
                    eintrag["reparaturquote_q"] = None
        ergebnisse.append(eintrag)

    return {
        "schema": SCHEMA,
        "metrik": metrik_name,
        "metrik_richtung": richtung,
        "jnd": _JND,
        "sample_rate": int(sample_rate),
        "regionen": ergebnisse,
        "aggregate": _aggregiere(ergebnisse),
        "hinweis": (
            "Report-Only-Zeuge (§G191 (GEBOTE.md), Hörordnung §7): kein Gate, "
            "kein Rollback, kein Flag. Richtungs-Zertifizierung über Hör-Labels "
            "(D-K4-9) erforderlich, bevor der Beleg B3-Beschlüsse stützt."
        ),
    }


def _aggregiere(ergebnisse: list[dict[str, Any]]) -> dict[str, Any]:
    typen: dict[str, list[float]] = {}
    quoten: dict[str, list[float]] = {}
    flattops: dict[str, list[float]] = {}
    for e in ergebnisse:
        typen.setdefault(str(e["typ"]), []).append(float(e["delta"]))
        flattops.setdefault(str(e["typ"]), []).append(float(e.get("delta_flat_top", 0.0)))
        if e.get("reparaturquote_q") is not None:
            quoten.setdefault(str(e["typ"]), []).append(float(e["reparaturquote_q"]))
    aggregat: dict[str, Any] = {}
    for typ, deltas in typen.items():
        aggregat[typ] = {
            "n": len(deltas),
            "delta_median": round(float(np.median(deltas)), 4),
            "anteil_ueber_jnd": round(float(np.mean([d >= _JND for d in deltas])), 4),
            "anteil_verschlechtert": round(float(np.mean([d <= -_JND for d in deltas])), 4),
            "flat_top_delta_median": round(float(np.median(flattops[typ])), 4),
            "anteil_verschlechtert_flat_top": round(float(np.mean([d <= -_JND for d in flattops[typ]])), 4),
            "reparaturquote_median": (round(float(np.median(quoten[typ])), 4) if quoten.get(typ) else None),
        }
    return aggregat


def sha256_datei(pfad: str) -> str:
    """Deterministischer Herkunfts-Hash (byte-genau, §III.13 (copilot-instructions.md))."""
    h = hashlib.sha256()
    with open(pfad, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def beleg_schreiben(ergebnis: dict[str, Any], zielpfad: str, quellen: dict[str, str]) -> dict[str, Any]:
    """Schreibt den Beleg mit sha256 der Eingangsdateien (§G5 (copilot-instructions.md))."""
    import json

    beleg = dict(ergebnis)
    beleg["quellen_sha256"] = {name: sha256_datei(pfad) for name, pfad in sorted(quellen.items())}
    with open(zielpfad, "w", encoding="utf-8") as f:
        json.dump(beleg, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return beleg


def lade_defekt_delta(beleg: dict[str, Any] | str) -> dict[str, Any]:
    """Fail-closed-Lader: fremdes Schema ⇒ verfuegbar=false + Grund.

    Akzeptiert einen Beleg-Dict oder einen Pfad (str). ``/1``-Belege tragen
    die veraltete RMS-Matching-Definition und werden als ``veraltet``
    markiert statt still interpretiert (Muster ``ohr_prior``/2-Ablehnung,
    D-K3-89).
    """
    import json

    daten = beleg
    if isinstance(beleg, str):
        with open(beleg, encoding="utf-8") as f:
            daten = json.load(f)
    schema = daten.get("schema") if isinstance(daten, dict) else "unbekannt"
    if schema == SCHEMA:
        return {"verfuegbar": True, "beleg": daten}
    if schema == _SCHEMA_VERALTET:
        return {
            "verfuegbar": False,
            "grund": f"veraltetes schema {schema!r} (RMS-Matching) — neu messen mit {SCHEMA!r}",
        }
    return {
        "verfuegbar": False,
        "grund": f"fremdes schema {schema!r} — erwartet {SCHEMA!r}",
    }


_REGION_SCHWELLE = 0.1  # Severity-Untergrenze der Regionen-Extraktion (dokumentierter Vertrag)
_KONTROLL_ANTEILE = (0.1, 0.25, 0.5, 0.75, 0.9)
_KONTROLL_FENSTER_S = 2.0
_KONTROLL_MAX = 3


def regionen_aus_defekt_scores(scores: dict, sample_rate: int) -> list[dict[str, Any]]:
    """Extrahiert Defektregionen aus DefectScanner-Scores (§G191 (GEBOTE.md)).

    Vertrag Spec 07 §8.6: Regionen-JSON {start, end, typ, staerke} in
    Samples. Aufgenommen werden Location-Intervalle von DefektTypen mit
    severity ≥ `_REGION_SCHWELLE`; staerke = severity (0..1). Determinismus
    §G5 (copilot-instructions.md): rein funktional, sortierte Ausgabe.
    """
    regionen: list[dict[str, Any]] = []
    for typ, score in sorted(scores.items(), key=lambda kv: str(kv[0])):
        sev = float(getattr(score, "severity", 0.0))
        if sev < _REGION_SCHWELLE:
            continue
        locs = getattr(score, "locations", None) or []
        for start_s, end_s in locs:
            start = int(round(float(start_s) * sample_rate))
            end = int(round(float(end_s) * sample_rate))
            if end <= start:
                continue
            regionen.append(
                {
                    "start": start,
                    "end": end,
                    "typ": str(getattr(typ, "value", typ)),
                    "staerke": round(sev, 4),
                }
            )
    regionen.sort(key=lambda r: (r["start"], r["end"]))
    return regionen


def kontrollregionen_aus_abdeckung(
    regionen: list[dict[str, Any]], dauer_samples: int, sample_rate: int
) -> list[dict[str, Any]]:
    """Deterministische Kontrollregionen (defektfrei, Δ ≈ 0 erwartet).

    Kandidaten-Fenster (`_KONTROLL_FENSTER_S` Sekunden) bei festen Anteilen
    `_KONTROLL_ANTEILE` der Dauer; Fenster, die eine Defektregion
    überlappen, werden übersprungen; maximal `_KONTROLL_MAX` Regionen.
    §G5 (copilot-instructions.md): rein deterministisch, kein Zufall.
    """
    fenster = int(_KONTROLL_FENSTER_S * sample_rate)
    kandidaten: list[tuple[int, int]] = []
    for anteil in _KONTROLL_ANTEILE:
        mitte = int(round(dauer_samples * anteil))
        start = max(0, mitte - fenster // 2)
        end = min(dauer_samples, start + fenster)
        if end - start >= _MIN_SAMPLES:
            kandidaten.append((start, end))

    def _ueberlappt(a: tuple[int, int], b: tuple[int, int]) -> bool:
        return a[0] < b[1] and b[0] < a[1]

    kontrollen: list[dict[str, Any]] = []
    for start, end in kandidaten:
        if any(_ueberlappt((start, end), (int(r["start"]), int(r["end"]))) for r in regionen):
            continue
        kontrollen.append({"start": start, "end": end, "typ": "kontrolle", "staerke": 0.0})
        if len(kontrollen) >= _KONTROLL_MAX:
            break
    return kontrollen
