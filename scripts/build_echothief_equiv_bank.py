"""Kategorie-kalibrierter EchoThief-Äquivalenz-Synth-Generator (Pfeiler C).

**Problem.** EchoThief (195 RIRs, 12 exotische Raum-Kategorien: Tunnels,
Canyons, Domes, Sakralbauten …) ist für KI-Training lizenzlich ausgeschlossen
(SDSSUR/Chris Warren — „This restriction also applies to any training of
artificial intelligence models…"). Das Material schlicht wegzulassen ist jedoch
keine Lösung: die Fähigkeit — Lang-Raum-RIRs über exotische RT60/DRR-Bereiche —
fehlt sonst dem RT60-Regressor und dem Room-Fingerprinter.

**Lösung (dreiteilige Äquivalenz-Architektur, siehe
`docs/reports/current/2026-10-28_echothief_sota_aequivalenz.md`):**

* Pfeiler A — echte Beschaffung: SDSSURF-Lizenz anfragen (einziger Weg, die
  echten 195 Dateien zu nutzen). Menschliche Entscheidung/Vertrag.
* Pfeiler B — echte Äquivalenz-Bank: lizenzsaubere Mess-RIRs gleicher
  Kategorien (z. B. Nikolauskapelle Aachen, CC-BY-4.0, RT60 ≈ 4,6 s).
* **Pfeiler C (dieses Modul)** — kategorie-kalibrierte Synthese: reproduziert
  die *statistische Verteilung* (RT60, DRR, HF-Dämpfung) je Raum-Kategorie aus
  **publizierten Raumakustik-Fakten** (Fachliteratur), **ohne EchoThief-Audio
  zu berühren**. Zahlen/Physik sind nicht schutzfähig — es wird nie eine
  EchoThief-Waveform gelesen, gelernt oder eingebettet.

Jedes erzeugte RIR ist als **synthetisch** gekennzeichnet und deterministisch
(§G5 (copilot-instructions.md), §G5 (GEBOTE.md)): gleicher Seed ⇒ bit-identisch.
Kein ML-Fallback, kein stiller Modus (§V6 (copilot-instructions.md)).

Selbst-Verifikation: jedes RIR wird nach der Synthese per Schröder-Rückwärts-
integration nachgemessen; weicht RT60 > 8 % vom Ziel ab, wird hart abgebrochen
(§0a — kein NaN/fehlerhaftes Trainingsmaterial).

Nutzung (CLI):

    python3 -B scripts/build_echothief_equiv_bank.py --per-category 8 \\
        --out data/rir_synth_echothief --seed 20261028

Ausgabe: WAV-Dateien (32-bit float, 48 kHz) + `MANIFEST.json`
(Provenance je Datei: Kategorie, Ziel-RT60, gemessene RT60, Seed) +
`COVERAGE.json` (EchoThief-Kategorie → Zielbereich → Anzahl → Abdeckungsweg).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# Amplituden-Hüllkurve exp(−6.9078·t/RT60) ⇒ Energie −60 dB bei RT60.
_DECAY_CONST = 6.907755278982137  # ln(1000)

DEFAULT_SR = 48000


# ── EchoThief-Kategorien mit literaturgeerdeten Kalibrierungs-Zielen ─────────
# RT60/DRR-Bereiche aus Raumakustik-Fachliteratur (Beranek 2004; Cox &
# D'Antonio 2009; Engelmark 2018) für die jeweilige Raumklasse — NICHT aus
# EchoThief-Messwerten abgeleitet (Lizenz-Distanz, siehe Modul-Docstring).
# (rt60_min_s, rt60_max_s, drr_min_db, drr_max_db, hf_damp_0_1)


@dataclass(frozen=True)
class CategoryProfile:
    """Akustisches Zielprofil einer EchoThief-Raum-Kategorie."""

    name: str
    echothief_count: int  # Anzahl RIRs in EchoThief (nur Abdeckungs-Budget)
    rt60_min_s: float
    rt60_max_s: float
    drr_min_db: float
    drr_max_db: float
    hf_damp: float  # 0 = HF überlebt lang, 1 = HF dämpft schnell ab
    note: str = ""

    def sample_rt60(self, rng: np.random.Generator) -> float:
        """Log-uniforme Stichprobe im Zielbereich (Raumgrößen sind log-skaliert)."""
        lo, hi = math.log(self.rt60_min_s), math.log(self.rt60_max_s)
        return float(math.exp(rng.uniform(lo, hi)))

    def sample_drr(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.drr_min_db, self.drr_max_db))


CATEGORIES: tuple[CategoryProfile, ...] = (
    CategoryProfile("Underground", 29, 2.5, 7.0, -12.0, 2.0, 0.55, "U-Bahn-Schächte, Tunnel, Katakomben"),
    CategoryProfile("Underpasses", 23, 1.2, 3.5, -8.0, 4.0, 0.5, "Kurztunnel, Unterführungen, Brücken"),
    CategoryProfile("Nature", 20, 1.0, 4.0, -6.0, 6.0, 0.45, "Canyons, Schluchten, Felsformationen"),
    CategoryProfile("Brutalism", 10, 1.5, 3.5, -6.0, 5.0, 0.5, "Beton-Hallentruhen, Treppenhäuser großer Bauwerke"),
    CategoryProfile("Recreation", 8, 1.5, 3.5, -6.0, 5.0, 0.55, "Sport-/Schwimmhallen"),
    CategoryProfile("Stairwells", 7, 1.2, 3.0, -4.0, 6.0, 0.55, "Treppenhäuser"),
    CategoryProfile("Venues", 7, 1.2, 2.5, -2.0, 8.0, 0.5, "Säle, Auditorien"),
    CategoryProfile("Domes", 7, 3.0, 8.0, -10.0, 2.0, 0.4, "Kuppeln — langes Ringing"),
    CategoryProfile("Nautical", 6, 1.0, 3.0, -6.0, 5.0, 0.6, "Schiffsrümpfe, Laderäume"),
    CategoryProfile("Sanctuaries", 5, 2.5, 6.0, -8.0, 3.0, 0.45, "Kirchen, Kapellen, Hallenkirchen"),
    CategoryProfile("Miscellaneous", 26, 0.8, 5.0, -8.0, 6.0, 0.5, "Sonstige exotische Räume"),
    CategoryProfile("EchoThief", 47, 0.8, 5.0, -8.0, 6.0, 0.5, "Allgemeine/curated Sammlung (Wurzel-Ebene)"),
)


@dataclass
class SynthRIRRecord:
    """Provenance-Eintrag eines synthetischen RIR (Manifest)."""

    datei: str
    kategorie: str
    seed: int
    rt60_ziel_s: float
    rt60_gemessen_s: float
    drr_ziel_db: float
    sample_rate: int
    synthetisch: bool = True
    quelle: str = "kategorie-kalibrierte Synthese (Pfeiler C)"
    lizenz: str = "CC0-1.0 (synthetisch, kein EchoThief-Audio)"
    extras: dict = field(default_factory=dict)


# ── Physikalische RIR-Synthese ───────────────────────────────────────────────


def _schroeder_edc_db(x: np.ndarray) -> np.ndarray:
    """Schröder-Energiezerfallskurve in dB (Rückwärts-Integration)."""
    e = np.cumsum((x.astype(np.float64) ** 2)[::-1])[::-1]
    e = np.maximum(e, np.finfo(np.float64).tiny)
    edc = 10.0 * np.log10(e / e[0])
    return edc


def _measure_rt60_s(x: np.ndarray, sr: int) -> float:
    """RT60 aus dem linearen Abklingbereich (−5 dB … −35 dB, T30×2)."""
    edc = _schroeder_edc_db(x)
    # Start: erster Abtastpunkt ≤ −5 dB; Ende: ≤ −35 dB (oder Kurvenende).
    below = np.where(edc <= -5.0)[0]
    if below.size == 0:
        return 0.0
    i0 = int(below[0])
    seg = edc[i0:]
    below35 = np.where(seg <= -35.0)[0]
    i1 = i0 + int(below35[0]) if below35.size else len(edc) - 1
    if i1 <= i0:
        return 0.0
    t = np.arange(i1 - i0 + 1, dtype=np.float64) / sr
    slope, _intercept = np.polyfit(t, edc[i0 : i1 + 1], 1)
    if slope >= -1e-9:
        return 0.0
    return float(-60.0 / slope)


def _lowpass_onepole(x: np.ndarray, alpha: float) -> np.ndarray:
    """Einfacher Einpol-IIR-Tiefpass (deterministisch, kausal)."""
    if alpha <= 0.0:
        return x
    y = np.empty_like(x)
    acc = 0.0
    a = float(alpha)
    for i in range(x.size):
        acc = (1.0 - a) * acc + a * x[i]
        y[i] = acc
    return y


def synth_rir_category(
    profile: CategoryProfile,
    rt60_s: float,
    drr_db: float,
    sr: int = DEFAULT_SR,
    seed: int = 0,
    *,
    n_early: int = 12,
    tol: float = 0.08,
) -> tuple[np.ndarray, float]:
    """Erzeugt ein kategorie-kalibriertes RIR.

    Zusammensetzung (leichtgewichtige Image-Source + statistischer Ausklang):
    Direktschall-Impuls → sparse Frühe Reflexionen → frequenzabhängig
    abklingender diffuser Noise-Ausklang mit Ziel-RT60. DRR (Direkt-zu-Hall)
    wird über das Verhältnis Direkt-/Hallenergie eingestellt.

    Returns:
        (rir, rt60_gemessen): normalisiertes float32-RIR und die nach der
        Synthese nachgemessene RT60 (Selbst-Verifikation).
    """
    if rt60_s <= 0:
        raise ValueError(f"rt60_s muss > 0 sein, got {rt60_s}")
    rng = np.random.default_rng(seed)

    # Dauer: genug, dass der Ausklang ≥ 1.2·RT60 abklingt (sonst RT60-Messung
    # am Rand abgeschnitten ⇒ Fehlmessung).
    dur_s = max(1.0, rt60_s * 1.25)
    n = int(sr * dur_s)
    t = np.arange(n, dtype=np.float64) / sr

    # Diffuser Ausklang: exponentiell abklingender Noise.
    env = np.exp(-_DECAY_CONST * t / rt60_s)
    tail = rng.standard_normal(n) * env

    # Frequenzabhängige Dämpfung: hohe Frequenzen klingen in großen/absorbieren-
    # den Räumen schneller ab (hf_damp ↑ ⇒ stärkeres Tiefpass-Filtern des Ausklangs).
    if profile.hf_damp > 0:
        # Blockweise Tiefpass über den Ausklang, Stärke wächst mit hf_damp.
        alpha = 0.05 + 0.6 * profile.hf_damp
        tail = _lowpass_onepole(tail, alpha)

    # Frühe Reflexionen: sparse Taps in den ersten ~80 ms.
    early = np.zeros(n, dtype=np.float64)
    er_window = int(min(n, 0.08 * sr))
    n_early = max(0, min(n_early, er_window // 8))
    if n_early > 0:
        idx = np.sort(rng.integers(1, er_window, size=n_early))
        amps = rng.standard_normal(n_early) * np.exp(-3.0 * idx / er_window)
        np.add.at(early, idx, amps)

    # Direktschall-Impuls bei t=0.
    direct = np.zeros(n, dtype=np.float64)
    direct[0] = 1.0

    # Energie-Gewichtung für Ziel-DRR: drr_db = 10·log10(E_direkt / E_hall).
    hall = tail + 0.35 * early  # fruehe Reflexionen zaehlen zum fruehen Feld
    e_direct = float(np.sum(direct**2))
    e_hall = float(np.sum(hall**2))
    if e_hall <= 0:
        raise ValueError("Hall-Energie = 0 — Synthese degeneriert")
    target_ratio = 10.0 ** (drr_db / 10.0)
    # Skaliere Hall, bis E_direkt/E_hall = target_ratio.
    hall_scale = math.sqrt(e_direct / (target_ratio * e_hall))
    hall = hall * hall_scale

    x = direct + hall
    peak = float(np.max(np.abs(x)))
    if peak <= 0 or not np.isfinite(peak):
        raise ValueError("RIR-Synthese degeneriert (peak ≤ 0 oder NaN/Inf)")
    x = (x / peak).astype(np.float32)

    # Selbst-Verifikation: nachgemessene RT60 muss nahe Ziel liegen.
    rt60_meas = _measure_rt60_s(x, sr)
    if rt60_s > 0.25 and abs(rt60_meas - rt60_s) / rt60_s > tol:
        raise ValueError(
            f"RT60-Selbstverifikation fehlgeschlagen: Ziel={rt60_s:.3f}s "
            f"gemessen={rt60_meas:.3f}s (Kategorie {profile.name})"
        )
    return x, float(rt60_meas)


# ── Bank-Aufbau ──────────────────────────────────────────────────────────────


def build_bank(
    out_dir: Path,
    per_category: int,
    sr: int = DEFAULT_SR,
    seed: int = 20261028,
    *,
    categories: tuple[CategoryProfile, ...] = CATEGORIES,
) -> tuple[list[SynthRIRRecord], dict]:
    """Erzeugt die kategorie-kalibrierte Äquivalenz-Bank.

    Returns:
        (records, coverage): Provenance-Einträge + Abdeckungs-Matrix.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[SynthRIRRecord] = []
    coverage: dict = {
        "zweck": "EchoThief-Äquivalenz (Pfeiler C) — kategorie-kalibrierte Synthese",
        "lizenzhinweis": (
            "Synthetisch erzeugt aus literaturgeerdeten Raumakustik-Zielen. "
            "Kein EchoThief-Audio verwendet/gelernt (SDSURF-Lizenz bleibt "
            "ausschließlich für Pfeiler A / echte Beschaffung einschlägig)."
        ),
        "kategorien": {},
    }

    for cat_idx, cat in enumerate(categories):
        rng = np.random.default_rng(seed + 1000 * cat_idx)
        cat_recs: list[SynthRIRRecord] = []
        for i in range(per_category):
            rt60_t = cat.sample_rt60(rng)
            drr_t = cat.sample_drr(rng)
            file_seed = seed + 1000 * cat_idx + i
            x, rt60_m = synth_rir_category(cat, rt60_t, drr_t, sr=sr, seed=file_seed)
            fname = f"synth_{cat.name.lower()}_{i:03d}_rt60_{rt60_t:.3f}_seed_{file_seed}.wav"
            sf.write(str(out_dir / fname), x, sr, subtype="FLOAT")
            rec = SynthRIRRecord(
                datei=str(out_dir / fname),
                kategorie=cat.name,
                seed=file_seed,
                rt60_ziel_s=round(rt60_t, 4),
                rt60_gemessen_s=round(rt60_m, 4),
                drr_ziel_db=round(drr_t, 3),
                sample_rate=sr,
                extras={
                    "echothief_kategorie_count": cat.echothief_count,
                    "hf_damp": cat.hf_damp,
                },
            )
            records.append(rec)
            cat_recs.append(rec)

        meas = [r.rt60_gemessen_s for r in cat_recs]
        coverage["kategorien"][cat.name] = {
            "echothief_anzahl": cat.echothief_count,
            "rt60_zielbereich_s": [cat.rt60_min_s, cat.rt60_max_s],
            "drr_zielbereich_db": [cat.drr_min_db, cat.drr_max_db],
            "erzeugt": len(cat_recs),
            "rt60_gemessen_min_s": round(min(meas), 3) if meas else None,
            "rt60_gemessen_max_s": round(max(meas), 3) if meas else None,
            "abdeckungsweg": "synthetisch (Pfeiler C)",
            "beschreibung": cat.note,
        }
        logger.info(
            "Kategorie %s: %d RIRs erzeugt (Ziel-RT60 %.2f–%.2f s)",
            cat.name,
            len(cat_recs),
            cat.rt60_min_s,
            cat.rt60_max_s,
        )

    return records, coverage


def write_manifest(out_dir: Path, records: list[SynthRIRRecord], coverage: dict) -> None:
    """Schreibt MANIFEST.json + COVERAGE.json ins Ausgabeverzeichnis."""
    out_dir = Path(out_dir)
    manifest = {
        "schema": "aurik.rir_synth_echothief_equiv/1",
        "zweck": (
            "EchoThief-Äquivalenz-Bank (Pfeiler C) — kategorie-kalibrierte "
            "synthetische RIRs für RT60-Regressor & Room-Fingerprinter"
        ),
        "determinismus": "§G5 (copilot-instructions.md): Seed je Datei, bit-identisch",
        "anzahl": len(records),
        "dateien": [asdict(r) for r in records],
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "COVERAGE.json").write_text(json.dumps(coverage, indent=2, ensure_ascii=False), encoding="utf-8")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="data/rir_synth_echothief", help="Ausgabeverzeichnis")
    p.add_argument("--per-category", type=int, default=8, help="RIRs je Kategorie")
    p.add_argument("--sr", type=int, default=DEFAULT_SR, help="Abtastrate")
    p.add_argument("--seed", type=int, default=20261028, help="Basis-Seed §G5 (copilot-instructions.md)")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Nur Zielbereiche/Anzahlen ausgeben, nichts schreiben",
    )
    args = p.parse_args(argv)

    if args.per_category < 1:
        logger.error("--per-category muss ≥ 1 sein")
        return 2

    if args.dry_run:
        for cat in CATEGORIES:
            print(
                f"{cat.name:16s} echothief={cat.echothief_count:3d} "
                f"rt60=[{cat.rt60_min_s:.2f},{cat.rt60_max_s:.2f}]s "
                f"drr=[{cat.drr_min_db:.0f},{cat.drr_max_db:.0f}]dB "
                f"→ {args.per_category} synth"
            )
        return 0

    out_dir = Path(args.out)
    records, coverage = build_bank(out_dir, args.per_category, sr=args.sr, seed=args.seed)
    write_manifest(out_dir, records, coverage)
    logger.info(
        "EchoThief-Äquivalenz-Bank: %d synthetische RIRs → %s",
        len(records),
        out_dir,
    )
    print(f"OK: {len(records)} synthetische RIRs in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
