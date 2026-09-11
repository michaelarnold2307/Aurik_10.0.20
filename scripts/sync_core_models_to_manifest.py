#!/usr/bin/env python3
"""Sync core model artifacts into models/manifest.json when files exist.

Non-destructive behavior:
- Does not create model files.
- Adds or updates manifest entries only for artifacts that already exist.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "models" / "manifest.json"


@dataclass(frozen=True)
class ManifestItem:
    name: str
    rel_path: str
    fallback: str


ITEMS = [
    ManifestItem("rmvpe", "models/rmvpe/rmvpe.onnx", "crepe_full"),
    ManifestItem("sgmse_plus", "models/sgmse_plus/sgmse_plus_core.onnx", "wpe_dsp"),
    ManifestItem(
        "versa",
        "models/versa/hub_cache/checkpoints/ft_wav2vec2_large_ll60k_mdf_p1_200epochs_all_192epochs.pth",
        "pqs_dsp_gammatone",
    ),
    ManifestItem("flow_matching", "models/flow_matching/flow_matching.onnx", "cqtdiff_plus"),
    ManifestItem("gacela", "models/gacela/model/01_400000.pt", "dsp_harmonic_exciter"),
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _size_gb(size_bytes: int) -> float:
    return round(size_bytes / (1024**3), 3)


def main() -> int:
    if not MANIFEST.exists():
        print(f"Fehler: Manifest fehlt: {MANIFEST}")
        return 2

    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    models = data.get("models", [])
    if not isinstance(models, list):
        print("Fehler: manifest.models ist keine Liste")
        return 3

    by_name = {m.get("name"): i for i, m in enumerate(models) if isinstance(m, dict) and isinstance(m.get("name"), str)}

    added = 0
    updated = 0
    skipped = 0

    for item in ITEMS:
        abs_path = ROOT / item.rel_path
        if not abs_path.exists():
            print(f"SKIP missing file: {item.rel_path}")
            skipped += 1
            continue

        size_bytes = abs_path.stat().st_size
        entry = {
            "name": item.name,
            "bundled": True,
            "bundled_path": item.rel_path,
            "sha256": _sha256(abs_path),
            "size_bytes": size_bytes,
            "required": False,
            "fallback": item.fallback,
            "size_gb": _size_gb(size_bytes),
        }

        if item.name in by_name:
            models[by_name[item.name]] = entry
            updated += 1
            print(f"UPDATED {item.name}")
        else:
            models.append(entry)
            added += 1
            print(f"ADDED   {item.name}")

    data["models"] = models
    MANIFEST.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\nSummary:")
    print(f"  added={added}")
    print(f"  updated={updated}")
    print(f"  skipped_missing_files={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
