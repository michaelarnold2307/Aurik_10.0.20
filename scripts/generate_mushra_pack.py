"""Create a reproducible subjective listening pack (MUSHRA-style) for human review."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _find_wav_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*.wav") if p.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate MUSHRA listening manifest")
    parser.add_argument("--reference-dir", required=True, help="Directory with reference WAV files")
    parser.add_argument("--candidate-dir", required=True, help="Directory with restored candidate WAV files")
    parser.add_argument("--output", default="reports/mushra_manifest.csv", help="Output CSV path")
    args = parser.parse_args()

    ref_dir = Path(args.reference_dir)
    cand_dir = Path(args.candidate_dir)
    out_path = Path(args.output)

    refs = _find_wav_files(ref_dir)
    cands = _find_wav_files(cand_dir)

    if not refs:
        print("Keine Referenz-WAV-Dateien gefunden.")
        return 2
    if not cands:
        print("Keine Kandidaten-WAV-Dateien gefunden.")
        return 2

    # Match by stem suffix policy: candidate starts with reference stem.
    rows: list[tuple[str, str, str]] = []
    for ref in refs:
        stem = ref.stem
        matched = [c for c in cands if c.stem.startswith(stem)]
        for cand in matched:
            rows.append((stem, str(ref), str(cand)))

    if not rows:
        print("Keine passenden Referenz/Kandidaten-Paare gefunden.")
        return 3

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "reference_wav", "candidate_wav"])
        writer.writerows(rows)

    print(f"MUSHRA-Paket erstellt: {out_path} ({len(rows)} Paare)")
    print("Hinweis: Bewertung mit randomisiertem Blind-Test und mindestens 10 Hoerern durchfuehren.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
