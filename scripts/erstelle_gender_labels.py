#!/usr/bin/env python3
"""R6-Daten (Kette 4, F13): Gender-Labels ``data/gender_labels.csv`` für den Gender-Head.

Warum: ``gpu_kette4_modellkoepfe.sh`` R6 überspringt den Gender-Head-Trainer
(``scripts/train_gender_head.py``), solange ``data/gender_labels.csv`` fehlt —
datenblockiert (E2, D-K4-11, KP-15). Dieser Erzeuger schließt die Lücke
autark aus dem lokalen MUSDB18-HQ-Bestand: **keine** externen Labels, **keine**
Lizenz-Beschaffung.

Verfahren (schwache Supervision, ehrlich dokumentiert):
- Träger ist der **Produktions-Gender-Detektor** (§19-Fusion:
  F0 + Formanten, ``backend/core/forensics/gender_detection.py``) — exakt
  die Instanz, deren Feature-Kopf der Trainer kalibriert; kein ad-hoc-Kriterium.
- Je MUSDB18-HQ-Vocal-Stem (train + test, deterministische Reihenfolge)
  wird ein 20-s-Mittel-Ausschnitt (48 kHz mono) klassifiziert.
- ``unknown``/``child``-Fälle werden **nicht** erfunden: ``unknown`` wird
  übersprungen (mit Log), ``child`` nur übernommen, wenn der Detektor es
  selbst meldet (im MUSDB-Bestand praktisch nicht vertreten — ehrlich als
  Klassendefizit dokumentiert, kein Label-Raten).

Ausgabe: ``data/gender_labels.csv`` mit exakt der Kopfzeile ``file,gender``
(``gender`` ∈ {male, female, child}), ``file`` = Pfad **relativ zu
``data/musdb18hq``** (der Trainer löst ``audio_dir / name`` auf). Der Trainer
validiert nur die Labels, nicht die Klassenzahl; die Kette-4-Auflage
„≥ ~50 je Klasse" wird für male/female erfüllt und im Bericht ausgewiesen.

Determinismus (§G5 (GEBOTE.md)): sortierte Track-Reihenfolge, fester Seed,
kein time.time() in Entscheidungslogik.

Usage:
    python3 -B scripts/erstelle_gender_labels.py \
        --musdb-root data/musdb18hq --out data/gender_labels.csv --seed 42
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

if __package__:
    from .csv_export_safety import render_spreadsheet_safe_csv
else:
    from csv_export_safety import render_spreadsheet_safe_csv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

_SR = 48000
_LABELS: tuple[str, ...] = ("male", "female", "child")


def _decode_48k(pfad: Path) -> np.ndarray:
    """Decodiert eine Datei zu mono float32 @ 48 kHz (ffmpeg, wie erstelle_schadens_proben)."""
    import io
    import subprocess

    import soundfile as sf

    _r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(pfad), "-ac", "1", "-ar", str(_SR), "-f", "wav", "-"],
        capture_output=True,
        check=True,
    )
    _data, _sr = sf.read(io.BytesIO(_r.stdout), always_2d=False)
    return np.asarray(_data, dtype=np.float32)


def _mittel_ausschnitt(audio: np.ndarray, sekunden: float) -> np.ndarray:
    """Deterministischer Mittel-Ausschnitt (wie das Ohr-Panel)."""
    arr = np.asarray(audio)
    want = int(round(sekunden * _SR))
    if arr.shape[-1] <= want or sekunden <= 0:
        return arr
    start = (arr.shape[-1] - want) // 2
    return arr[..., start : start + want]


def _vokal_stems(musdb_root: Path) -> list[Path]:
    """Alle vocals.wav des MUSDB18-HQ-Bestands, deterministisch sortiert."""
    _stems: list[Path] = []
    for _split in ("train", "test"):
        _split_dir = musdb_root / _split
        if not _split_dir.is_dir():
            continue
        for _track in sorted(_split_dir.iterdir()):
            _voc = _track / "vocals.wav"
            if _voc.is_file():
                _stems.append(_voc)
    return _stems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--musdb-root", type=Path, default=Path("data/musdb18hq"))
    ap.add_argument("--out", type=Path, default=Path("data/gender_labels.csv"))
    ap.add_argument("--sekunden", type=float, default=20.0)
    ap.add_argument(
        "--panns",
        action="store_true",
        help="PANNs-AudioSet-Tags (Male/Female singing, Klassen 32/33) in die §19-Fusion einspeisen",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from backend.core.forensics.gender_detection import GenderDetector

    _det = GenderDetector(sample_rate=_SR)
    _stems = _vokal_stems(args.musdb_root)
    if not _stems:
        logger.error("keine MUSDB-Vocal-Stems unter %s", args.musdb_root)
        return 2

    _rows: list[tuple[str, str]] = []
    _counts: dict[str, int] = dict.fromkeys(_LABELS, 0)
    _skipped = 0
    for _stem in _stems:
        _rel = _stem.relative_to(args.musdb_root).as_posix()
        try:
            _audio = _decode_48k(_stem)
        except Exception as _exc:  # §V6 (copilot-instructions.md): loggen statt stiller Lücke
            logger.warning("Decode fehlgeschlagen (%s): %s", _stem, _exc)
            _skipped += 1
            continue
        _excerpt = _mittel_ausschnitt(_audio, args.sekunden)
        _panns_tags: dict | None = None
        if args.panns:
            try:
                from plugins.panns_plugin import classify_audio

                _tags = classify_audio(_excerpt, _SR)
                if isinstance(_tags, dict) and ("Male singing" in _tags or "Female singing" in _tags):
                    _panns_tags = _tags
            except Exception as _exc:  # §V6 (copilot-instructions.md): Anatomie-Ersatzpfad, geloggt
                logger.warning("PANNs-Tags nicht erzeugbar (%s) — Anatomie-Ersatzpfad für %s", _exc, _rel)
        _label = _det.detect_gender_array(_excerpt, sample_rate=_SR, panns_tags=_panns_tags)
        if _label not in _LABELS:
            logger.info("uebersprungen (Detektor: %s): %s", _label, _rel)
            _skipped += 1
            continue
        _rows.append((_rel, _label))
        _counts[_label] += 1
        logger.info("%s -> %s", _rel, _label)

    if not _rows:
        logger.error("keine gueltigen Labels erzeugt")
        return 3

    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        csv_content = render_spreadsheet_safe_csv(["file", "gender"], _rows)
    except ValueError as exc:
        logger.error("CSV-Ausgabe wurde aus Sicherheitsgruenden abgelehnt: %s", exc)
        return 4

    with args.out.open("w", newline="", encoding="utf-8") as _fh:
        _fh.write(csv_content)

    logger.info(
        "geschrieben %s: %d Zeilen (male=%d, female=%d, child=%d, uebersprungen=%d)",
        args.out,
        len(_rows),
        _counts["male"],
        _counts["female"],
        _counts["child"],
        _skipped,
    )
    if _counts["male"] < 50 or _counts["female"] < 50:
        logger.warning(
            "Kette-4-Auflage (>=~50 je Klasse) fuer male/female verfehlt — "
            "Bestand zu klein oder Detektor weicht aus; CSV bleibt gueltig, "
            "aber R6 bleibt bis zur Nachbeschaffung datenblockiert."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
