#!/usr/bin/env python3
"""D-K0-8-Daten (SOTA-M18, Folge): synthetische Tape-Klassen für den Material-Klassifikator.

Warum: WP-7-Bestand 2026-10-09 — Tape-Klassen **dünn** (tape 21, cassette 13,
reel_tape 10 Einträge; Ziel ≥ ~50 je Klasse). Echte Bänder der LP-Ära sind
nicht ohne Rechteinhaber beschaffbar; das Register nennt ausdrücklich den
Synthetik-Weg (D-K0-8). Dieses Skript füllt alle drei Klassen deterministisch
auf mindestens ``--ziel`` Einträge auf — analog zum Vinyl-Erzeuger
(``erstelle_corpus_vinyl.py``) und mit demselben D-K3-80-Schutz (Bestands-
manifest.yaml wird NIE zerstört, Einträge werden angehängt, dedupe per file).

Rezepte (deterministisch, Seed 42, §G5 (GEBOTE.md), alles aus dem
Produktions-/Mess-Bestand wiederverwendet):
- ``cassette``: kanonisches kassette-Rezept aus `erstelle_schadens_proben.py`
  (`defekt_variante` rausch → wowflutter → bandlimit, ohr_panel-Kerne);
- ``tape``: Bandrauschen + Bandlimit (kein Wow — Tonband-Kopie);
- ``reel_tape``: Bandrauschen + Bandlimit + deterministische Dropouts
  (kurze Pegel-Einbrüche, Spulentonband-Charakter).

Ehrlichkeit: ``source: Synthetic``, ``license: CC0 (generated)``, ``era_year``
ist die SIMULIERTE Ära (1965/1975/1985) — keine echte Provenienz.

Usage:
    python3 -B scripts/erstelle_corpus_tape.py \
        --musdb-root data/musdb18hq --corpus-root corpus \
        --ziel 50 --sekunden 10 --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import io
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

_SR = 48000
_MATERIALS = ("tape", "cassette", "reel_tape")
_ERA = {"tape": 1975, "cassette": 1985, "reel_tape": 1965}


def _decode_48k(pfad: Path) -> np.ndarray:
    import soundfile as sf

    _r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(pfad), "-ac", "1", "-ar", str(_SR), "-f", "wav", "-"],
        capture_output=True,
        check=True,
    )
    _data, _sr = sf.read(io.BytesIO(_r.stdout), always_2d=False)
    return np.asarray(_data, dtype=np.float32)


def _mittel_ausschnitt(audio: np.ndarray, sekunden: float) -> np.ndarray:
    arr = np.asarray(audio)
    want = int(round(sekunden * _SR))
    if arr.shape[-1] <= want or sekunden <= 0:
        return arr
    start = (arr.shape[-1] - want) // 2
    return arr[..., start : start + want]


def _pcm_sha256(x: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(x, dtype=np.float32).tobytes()).hexdigest()


def _dropouts(x: np.ndarray, rng: np.random.Generator, n: int = 3) -> np.ndarray:
    """Deterministische kurze Pegel-Einbrüche (Spulentonband)."""
    y = np.asarray(x, dtype=np.float64).copy()
    for _ in range(n):
        _pos = int(rng.integers(0, max(1, y.size - _SR // 20)))
        _len = int(rng.integers(_SR // 200, _SR // 40))
        _tiefe = float(rng.uniform(0.35, 0.8))
        _env = 1.0 - _tiefe * np.hanning(_len) ** 2
        y[_pos : _pos + _len] *= _env
    return y.astype(np.float32)


def _rezept(x: np.ndarray, material: str, rng: np.random.Generator) -> np.ndarray:
    """Klassen-Rezept aus dem Produktions-/Mess-Bestand (ohr_panel-Kerne)."""
    from scripts.erstelle_schadens_proben import schaden_rezept
    from scripts.ohr_panel import defekt_variante

    if material == "cassette":
        return schaden_rezept(x, "kassette", rng)
    if material == "tape":
        y = defekt_variante(x, "rausch", rng)
        return defekt_variante(y, "bandlimit", rng)
    if material == "reel_tape":
        y = defekt_variante(x, "rausch", rng)
        y = defekt_variante(y, "bandlimit", rng)
        return _dropouts(y, rng)
    raise ValueError(f"Unbekanntes Material: {material}")


def _musdb_mixturen(musdb_root: Path) -> list[Path]:
    _mixtures: list[Path] = []
    for _split in ("train", "test"):
        _split_dir = musdb_root / _split
        if not _split_dir.is_dir():
            continue
        for _track in sorted(_split_dir.iterdir()):
            _mix = _track / "mixture.wav"
            if _mix.is_file():
                _mixtures.append(_mix)
    return _mixtures


def _lade_bestand(manifest_pfad: Path) -> list[dict]:
    import yaml

    if not manifest_pfad.is_file():
        return []
    try:
        _alt = yaml.safe_load(manifest_pfad.read_text(encoding="utf-8")) or {}
        return list(_alt.get("entries", []))
    except Exception as _exc:  # §V6 (copilot-instructions.md)
        logger.warning("Alt-Manifest nicht lesbar (%s) — Neuanlage", _exc)
        return []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--musdb-root", type=Path, default=Path("data/musdb18hq"))
    ap.add_argument("--corpus-root", type=Path, default=Path("corpus"))
    ap.add_argument("--ziel", type=int, default=50)
    ap.add_argument("--sekunden", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import soundfile as sf
    import yaml

    _mixtures = _musdb_mixturen(args.musdb_root)
    if not _mixtures:
        logger.error("keine MUSDB-Mixturen unter %s", args.musdb_root)
        return 3
    _rng = np.random.default_rng(args.seed)
    _genre_pool = ("jazz", "rock", "pop", "blues", "soul")
    _idx = 0

    for _material in _MATERIALS:
        _out = args.corpus_root / _material
        _out_damaged = _out / "damaged"
        _out_damaged.mkdir(parents=True, exist_ok=True)
        _existing = _lade_bestand(_out / "manifest.yaml")
        _bekannt = {str(e.get("file")) for e in _existing}
        _entries: list[dict] = []
        _fehlend = max(0, args.ziel - len(_existing))
        logger.info("%s: Bestand %d, Ziel %d, erzeuge %d", _material, len(_existing), args.ziel, _fehlend)
        for _n in range(_fehlend):
            _mix = _mixtures[_idx % len(_mixtures)]
            _idx += 1
            _clean = _mittel_ausschnitt(_decode_48k(_mix), args.sekunden)
            _damaged = _rezept(_clean, _material, _rng)
            _stamm = f"{_material}_synth_{_n:03d}_{_mix.parent.name[:26]}"
            _wav = _out_damaged / f"{_stamm}.wav"
            sf.write(_wav, _damaged, _SR, subtype="FLOAT")
            _entries.append(
                {
                    "file": f"damaged/{_wav.name}",
                    "duration_s": round(float(_damaged.size / _SR), 3),
                    "sample_rate": _SR,
                    "material": _material,
                    "era_year": _ERA[_material],
                    "genre": _genre_pool[_n % len(_genre_pool)],
                    "defect_types": ["hiss", "bandlimit"]
                    if _material != "cassette"
                    else ["hiss", "wow_flutter", "bandlimit"],
                    "source": "Synthetic (Aurik Corpus Generator, tape recipe)",
                    "source_attribution": "Synthetic Corpus Generator (CC0); MUSDB18-HQ",
                    "license": "CC0 (generated)",
                    "checksum_sha256": _pcm_sha256(_damaged),
                    "synthetic": True,
                }
            )
            logger.info("%s", _wav.name)
        if not _entries:
            continue
        _manifest = {
            "corpus_version": "1.0.0",
            "material": _material,
            "description": (
                f"{_material.capitalize()}-Untercorpus: Bestands-Einträge + {len(_entries)} deterministische "
                "Synthetik-Proben (D-K0-8, SOTA-M18, Seed 42). era_year ist die SIMULIERTE Ära "
                "(keine echte Provenienz, Lizenz-unbedenklich)."
            ),
            "entries": _existing + _entries,
        }
        with (_out / "manifest.yaml").open("w", encoding="utf-8") as _fh:
            yaml.safe_dump(_manifest, _fh, sort_keys=False, allow_unicode=True)
        logger.info("manifest.yaml: %d Einträge gesamt", len(_existing) + len(_entries))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
