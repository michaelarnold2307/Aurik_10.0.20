"""scripts/q0_witness_forschung.py — Metrik-Kanal-Suche für die Q≈0-Klassen (2026-11-14).

Die §G13-Q-Erstbelege zeigen: 10 Defektklassen erreichen Q ≈ 0 — teils, weil
die Pipeline sie nicht repariert, teils, weil die Gate-Metrik den Schaden
nicht sieht (Metrik-Blindheit, Handover Punkt 25/26/28). Der
Sättigungskanal (1/Crest) hat die Clipping-Klassen messbar gemacht; dieses
Skript sucht nach dem nächsten belegbaren, gain-invarianten Kanal je Klasse.

Gemessen wird pro Klasse auf einer deterministischen Regionen-Stichprobe
(Seed 42, data/klassen_paare/manifest.jsonl, clean/schaden/restauriert):
  * Richtung: Median(δ schaden−clean) über die JND hinaus UND >50 % Regionen
    über JND — sonst ehrlich „kein Kanal“.
  * Reparatur-Bewegung: Median(δ restauriert−schaden) Richtung clean.

Kandidaten (gain-invariant, physikalisch begründet):
  groove_echo        → Cepstral-Peak-Prominenz bei Echo-Lags 5–300 ms +
                       Spektral-Ripple-Tiefe (Kammfilter)
  jitter_artifacts   → Seitenband-Energie-Verhältnis um starke Spektrallinien
  sibilance          → Crest-Faktor 4–12 kHz
  sticky_shed_residue→ Impulsdichte (|x| > 6·RMS pro Sekunde)
  tape_head_level_dip→ langsame Hüllkurven-Modulationstiefe (0,1–5 Hz)
  transport_bump     → tieffrequenter Stoß-Energieanteil (1–30 Hz)

CPU-only (nice -n 15), kein Flag-Flip, kein Audio-Pfad-Eingriff — reine
Report-Forschung. Beleg: docs/reports/current/2026-11-14_q0_witness_forschung.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

SCHEMA = "aurik.q0_witness_forschung/1"
MANIFEST = _ROOT / "data" / "klassen_paare" / "manifest.jsonl"
REPORT_DIR = _ROOT / "docs" / "reports" / "current"
JND = 0.02
STICHPROBE = 300
MIN_SAMPLES = 4096


def _lade(pfad: str) -> np.ndarray | None:
    from scipy.io import wavfile

    p = Path(pfad)
    if not p.is_file():
        return None
    roh = wavfile.read(p)
    if not isinstance(roh, tuple) or len(roh) != 2:
        return None
    sr = int(roh[0])
    d = roh[1]
    x = np.asarray(d, np.float32)
    mono = x.mean(axis=1) if x.ndim == 2 else x
    return mono, sr


# ── Features ────────────────────────────────────────────────────────────────


def _spektrum(mono: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Log-Magnituden-Spektrum (Hanning, Mittelwert-frei für Ripple)."""
    win = np.hanning(len(mono))
    P = np.abs(np.fft.rfft(mono * win)) ** 2 + 1e-20
    return np.log(P), np.fft.rfftfreq(len(mono))


def groove_echo_cepstral(mono: np.ndarray, sr: int) -> float:
    """Cepstral-Peak-Prominenz bei Echo-Lags 5–300 ms (Echo ⇒ Peak)."""
    logP, _f = _spektrum(mono)
    ceps = np.fft.irfft(logP - np.mean(logP))
    n = len(ceps)
    lo = int(sr * 0.005)
    hi = int(sr * 0.300)
    if hi >= n or lo >= hi:
        return 0.0
    feld = np.abs(ceps[lo:hi])
    return float(np.max(feld) / (np.median(np.abs(ceps[1:])) + 1e-12))


def groove_echo_ripple(mono: np.ndarray, sr: int) -> float:
    """Spektral-Ripple-Tiefe 200–4000 Hz: Std des detrendeten Log-Spektrums."""
    logP, f = _spektrum(mono)
    m = (f >= 200) & (f <= 4000)
    if np.sum(m) < 32:
        return 0.0
    band = logP[m]
    trend = np.polyval(np.polyfit(np.arange(len(band)), band, 3), np.arange(len(band)))
    return float(np.std(band - trend))


def jitter_seitenband(mono: np.ndarray, sr: int) -> float:
    """Seitenband-Energie-Verhältnis um die stärksten Spektrallinien (Jitter ⇒
    verbreiterte Linien ⇒ mehr Energie in ±0,5–20 Hz-Nachbarschaft relativ
    zur Linie)."""
    win = np.hanning(len(mono))
    P = np.abs(np.fft.rfft(mono * win)) ** 2
    f = np.fft.rfftfreq(len(mono), 1.0 / sr)
    n = len(P)
    m = np.argpartition(P, -8)[-8:]
    linie = seite = 0.0
    for i in m:
        r = int(sr * 20.0 / (f[1] - f[0])) if (f[1] - f[0]) > 0 else 0
        r = max(r, 2)
        lo = max(0, int(i) - r)
        hi = min(n, int(i) + r)
        linie += float(P[i])
        seite += float(np.sum(P[lo:hi]) - P[i])
    return float(seite / (linie + 1e-20))


def sibilance_crest(mono: np.ndarray, sr: int) -> float:
    """Crest-Faktor 4–12 kHz (Sibilanten ⇒ spitze HF-Peaks)."""
    from scipy.signal import butter, sosfilt

    sos = butter(4, (4000, 12000), btype="band", fs=sr, output="sos")
    band = sosfilt(sos, mono)
    rms = float(np.sqrt(np.mean(band**2)) + 1e-12)
    return float(np.max(np.abs(band)) / rms)


def sticky_shed_impulse(mono: np.ndarray, sr: int) -> float:
    """Impulsdichte: Anteil Samples > 6·RMS (Dropout-/Oxid-Impulse)."""
    rms = float(np.sqrt(np.mean(mono**2)) + 1e-12)
    return float(np.mean(np.abs(mono) > 6.0 * rms))


def level_dip_modulation(mono: np.ndarray, sr: int) -> float:
    """Langsame Hüllkurven-Modulationstiefe (0,1–5 Hz; Pegel-Dip ⇒ Tiefe)."""
    hop = max(1, sr // 50)
    if len(mono) < 4 * hop:
        return 0.0
    env = np.array([float(np.sqrt(np.mean(mono[i : i + hop] ** 2)) + 1e-12) for i in range(0, len(mono) - hop, hop)])
    if len(env) < 8:
        return 0.0
    env_db = 20.0 * np.log10(env + 1e-12)
    env_db -= np.polyval(np.polyfit(np.arange(len(env_db)), env_db, 2), np.arange(len(env_db)))
    return float(np.std(env_db))


def transport_bump_lf(mono: np.ndarray, sr: int) -> float:
    """Tieffrequenter Stoß-Energieanteil 1–30 Hz (Transportstoß ⇒ LF-Burst)."""
    P = np.abs(np.fft.rfft(mono)) ** 2
    f = np.fft.rfftfreq(len(mono), 1.0 / sr)
    m = (f >= 1.0) & (f <= 30.0)
    return float(np.sum(P[m]) / (np.sum(P) + 1e-20))


FEATURES: dict[str, list[tuple[str, Any]]] = {
    "groove_echo": [("cepstral_echo", groove_echo_cepstral), ("ripple_tiefe", groove_echo_ripple)],
    "jitter_artifacts": [("seitenband_ratio", jitter_seitenband)],
    "sibilance": [("crest_4_12k", sibilance_crest)],
    "sticky_shed_residue": [("impulsdichte", sticky_shed_impulse)],
    "tape_head_level_dip": [("modulationstiefe", level_dip_modulation)],
    "transport_bump": [("lf_stoss", transport_bump_lf)],
}


def _messe_klasse(klasse: str, eintraege: list[dict[str, Any]], rng: random.Random) -> dict[str, Any]:
    gewaehlt = rng.sample(eintraege, min(STICHPROBE, len(eintraege)))
    out: dict[str, Any] = {"klasse": klasse, "regionen_gesamt": len(eintraege), "regionen_stichprobe": len(gewaehlt)}
    for feat_name, fn in FEATURES[klasse]:
        deltas_sc: list[float] = []
        deltas_rs: list[float] = []
        uebersprungen = 0
        for e in gewaehlt:
            schaden = _lade(e["schaden"])
            clean = _lade(e["clean"])
            if schaden is None or clean is None or len(schaden[0]) < MIN_SAMPLES or len(clean[0]) < MIN_SAMPLES:
                uebersprungen += 1
                continue
            fs = fn(schaden[0], schaden[1])
            fc = fn(clean[0], clean[1])
            deltas_sc.append(fs - fc)
            rest = _lade(e["restauriert"])
            if rest is not None and len(rest[0]) >= MIN_SAMPLES:
                deltas_rs.append(fn(rest[0], rest[1]) - fs)
        med_sc = float(np.median(deltas_sc)) if deltas_sc else 0.0
        anteil = float(np.mean([abs(d) > JND for d in deltas_sc])) if deltas_sc else 0.0
        med_rs = float(np.median(deltas_rs)) if deltas_rs else 0.0
        richtung_ok = abs(med_sc) > JND and anteil > 0.5
        reparatur_ok = bool(deltas_rs) and abs(med_rs) > 0 and np.sign(med_rs) == -np.sign(med_sc)
        out[feat_name] = {
            "delta_schaden_clean_median": round(med_sc, 5),
            "anteil_ueber_jnd": round(anteil, 4),
            "delta_restauriert_schaden_median": round(med_rs, 5),
            "richtung_ok": bool(richtung_ok),
            "reparatur_bewegung_ok": bool(reparatur_ok),
            "n_verwertbar": len(deltas_sc),
            "n_uebersprungen": uebersprungen,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Q≈0-Metrik-Kanal-Forschung (report-only)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stichprobe", type=int, default=STICHPROBE)
    ap.add_argument("--klassen", default=None, help="Komma-Liste (Default: alle)")
    args = ap.parse_args(argv)

    eintraege: dict[str, list[dict[str, Any]]] = {}
    with MANIFEST.open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            eintraege.setdefault(str(d["klasse"]), []).append(d)

    klassen = args.klassen.split(",") if args.klassen else sorted(FEATURES.keys())
    rng = random.Random(args.seed)
    ergebnisse = [_messe_klasse(k, eintraege.get(k, []), rng) for k in klassen]

    beleg = {
        "schema": SCHEMA,
        "datum": date.today().isoformat(),
        "seed": args.seed,
        "jnd": JND,
        "stichprobe_ziel": args.stichprobe,
        "ergebnisse": ergebnisse,
    }
    ziel = REPORT_DIR / f"{date.today().isoformat()}_q0_witness_forschung.json"
    ziel.write_text(json.dumps(beleg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(ergebnisse, indent=1, ensure_ascii=False))
    print(f"Beleg: {ziel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
