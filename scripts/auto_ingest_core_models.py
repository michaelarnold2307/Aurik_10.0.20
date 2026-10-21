"""Auto-ingest missing core model artifacts from local drop-in folders.

This script never downloads from the network. It only scans local directories,
selects best matching files, and copies them into canonical target paths.
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Artifact:
    name: str
    target_rel: str
    patterns: tuple[str, ...]
    required_source: bool = True


ARTIFACTS = [
    Artifact("rmvpe", "models/rmvpe/rmvpe.onnx", ("**/rmvpe.onnx", "**/*rmvpe*.onnx")),
    Artifact("sgmse_plus", "models/sgmse_plus/sgmse_plus.ts", ("**/sgmse_plus.ts", "**/*sgmse*.ts")),
    Artifact(
        "versa",
        "models/versa/hub_cache/checkpoints/ft_wav2vec2_large_ll60k_mdf_p1_200epochs_all_192epochs.pth",
        ("**/*ft_wav2vec2_large_ll60k*.pth", "**/*singmos*.pth", "**/*versa*.pth", "**/*versa*.pt"),
        required_source=False,
    ),
    Artifact(
        "flow_matching", "models/flow_matching/flow_matching.onnx", ("**/flow_matching.onnx", "**/*flow*matching*.onnx")
    ),
    Artifact("gacela", "models/gacela/model/01_400000.pt", ("**/01_400000.pt", "**/*gacela*.pt")),
]


def _default_dropins() -> list[Path]:
    rels = [
        "models/.dropin",
        "models/incoming",
        "models/_incoming",
        "imports/models",
        "imports",
    ]
    out = [ROOT / r for r in rels]

    env_val = os.environ.get("AURIK_MODEL_DROPINS", "").strip()
    if env_val:
        for part in env_val.split(":"):
            p = Path(part).expanduser()
            if not p.is_absolute():
                p = (ROOT / p).resolve()
            out.append(p)

    uniq: list[Path] = []
    seen = set()
    for p in out:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(rp)
    return uniq


def _find_candidates(dropins: list[Path], patterns: tuple[str, ...]) -> list[Path]:
    found: list[Path] = []
    for d in dropins:
        if not d.exists() or not d.is_dir():
            continue
        for pat in patterns:
            found.extend([p for p in d.glob(pat) if p.is_file()])
    uniq = sorted({p.resolve() for p in found}, key=lambda p: p.stat().st_size, reverse=True)
    return [Path(p) for p in uniq]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto-ingest core models from local drop-ins")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing target files")
    p.add_argument("--dry-run", action="store_true", help="Show actions without copying")
    p.add_argument(
        "--no-create-dropins",
        action="store_true",
        help="Do not create missing default drop-in directories.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    dropins = _default_dropins()

    created_dropins = 0
    if not args.no_create_dropins:
        for d in dropins:
            # Nur Verzeichnisse innerhalb des Projekts automatisch erstellen.
            if str(d).startswith(str(ROOT)) and not d.exists():
                d.mkdir(parents=True, exist_ok=True)
                created_dropins += 1

    print("Using drop-in dirs:")
    for d in dropins:
        state = "OK" if d.exists() else "MISSING"
        print(f"  [{state}] {d}")

    if created_dropins:
        print(f"\nHinweis: {created_dropins} Drop-In-Verzeichnis(se) wurden angelegt.")
        print("Lege die Modell-Dateien dort ab und starte Schritt 13 erneut.")

    copied = 0
    skipped = 0
    missing_source = 0
    optional_missing_source = 0

    for art in ARTIFACTS:
        target = ROOT / art.target_rel
        target.parent.mkdir(parents=True, exist_ok=True)

        if target.exists() and not args.overwrite:
            print(f"SKIP target exists: {art.target_rel}")
            skipped += 1
            continue

        candidates = _find_candidates(dropins, art.patterns)
        if not candidates:
            if art.required_source:
                print(f"MISSING source for {art.name}: {art.target_rel}")
                missing_source += 1
            else:
                print(f"OPTIONAL missing source for {art.name}: {art.target_rel} (fallback mode expected)")
                optional_missing_source += 1
            continue

        src = candidates[0]
        rel_src = src if not str(src).startswith(str(ROOT)) else src.relative_to(ROOT)
        if args.dry_run:
            print(f"DRYRUN copy {rel_src} -> {art.target_rel}")
            copied += 1
            continue

        shutil.copy2(src, target)
        print(f"COPIED {rel_src} -> {art.target_rel}")
        copied += 1

    print("\nSummary:")
    print(f"  copied={copied}")
    print(f"  skipped_existing={skipped}")
    print(f"  missing_source={missing_source}")
    print(f"  optional_missing_source={optional_missing_source}")

    if missing_source:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
