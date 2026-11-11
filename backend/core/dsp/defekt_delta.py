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

SCHEMA = "aurik.defektdelta/1"
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


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-12)


def pegel_gain(signal: np.ndarray, referenz: np.ndarray) -> float:
    """RMS-Matching-Gain, um ``signal`` auf den Pegel der ``referenz`` zu heben.

    D-K3-95-Lehre: ungematcht misst man den Loudness-Regelkreis, nicht die
    Reparatur. Stille-Referenz (RMS unter 1e-6) ⇒ Gain 1,0 —
    §V6 (copilot-instructions.md): kein stiller Ausfall, Warnung wird geloggt.
    """
    rms_sig = _rms(signal)
    rms_ref = _rms(referenz)
    if rms_ref < 1e-6:
        logger.warning("§V6 (copilot-instructions.md) pegel_gain: Referenz ist Stille — Gain 1,0")
        return 1.0
    if rms_sig < 1e-6:
        logger.warning("§V6 (copilot-instructions.md) pegel_gain: Signal ist Stille — Gain 1,0")
        return 1.0
    return min(max(rms_ref / rms_sig, 0.25), 4.0)


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
    for e in ergebnisse:
        typen.setdefault(str(e["typ"]), []).append(float(e["delta"]))
        if e.get("reparaturquote_q") is not None:
            quoten.setdefault(str(e["typ"]), []).append(float(e["reparaturquote_q"]))
    aggregat: dict[str, Any] = {}
    for typ, deltas in typen.items():
        aggregat[typ] = {
            "n": len(deltas),
            "delta_median": round(float(np.median(deltas)), 4),
            "anteil_ueber_jnd": round(float(np.mean([d >= _JND for d in deltas])), 4),
            "anteil_verschlechtert": round(float(np.mean([d <= -_JND for d in deltas])), 4),
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

    Akzeptiert einen Beleg-Dict oder einen Pfad (str). Muster:
    ``ohr_prior``/2-Ablehnung (D-K3-89) — ein Beleg mit anderer Semantik
    wird nie still interpretiert.
    """
    import json

    daten = beleg
    if isinstance(beleg, str):
        with open(beleg, encoding="utf-8") as f:
            daten = json.load(f)
    if not isinstance(daten, dict) or daten.get("schema") != SCHEMA:
        fremd = daten.get("schema") if isinstance(daten, dict) else "unbekannt"
        return {
            "verfuegbar": False,
            "grund": f"fremdes schema {fremd!r} — erwartet {SCHEMA!r}",
        }
    return {"verfuegbar": True, "beleg": daten}
