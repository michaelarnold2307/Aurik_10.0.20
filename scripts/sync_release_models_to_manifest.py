#!/usr/bin/env python3
"""scripts/sync_release_models_to_manifest.py — Release-Auslieferung ins Manifest.

Release-Strategie 2026-09-23 (§13.3):
  - Dateien > 40 MB: delivery="release", release_tag="models-10.2.0",
    Assets in deterministischer Namensgebung (Pfad → "__"), identisch zu
    scripts/release_upload_models.sh.
  - Dateien > 1.9 GB: Parts .part00/.part01/… mit je eigenem sha256 + Größe
    (GitHub-Limit 2 GB/Asset).
  - Dateien <= 40 MB: unverändert (LFS/bundled im Repo).

Deterministisch; berührt nur Manifest-Einträge existierender Dateien.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "models" / "manifest.json"
MODELS_DIR = ROOT / "models"
RELEASE_TAG = os.environ.get("AURIK_RELEASE_TAG", "models-10.2.0")
SIZE_THRESHOLD_BYTES = 40 * 1024 * 1024
CHUNK_BYTES = int(1900 * 1024 * 1024)

_MODEL_SUFFIXES = (".onnx", ".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".th")


def _sha256_bytes(buf: bytes) -> str:
    return hashlib.sha256(buf).hexdigest()


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _asset_base(rel: str) -> str:
    return rel.replace("/", "__")


def _plan_part_sizes(size: int) -> list[int]:
    parts: list[int] = []
    remaining = size
    while remaining > 0:
        part = min(CHUNK_BYTES, remaining)
        parts.append(part)
        remaining -= part
    return parts


def main() -> int:
    if not MANIFEST.exists():
        print(f"Fehler: Manifest fehlt: {MANIFEST}")
        return 2
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    models = data.get("models", [])
    if not isinstance(models, list):
        print("Fehler: manifest.models ist keine Liste")
        return 3

    by_path: dict[str, int] = {}
    for i, m in enumerate(models):
        p = m.get("bundled_path", "")
        if isinstance(p, str) and p:
            by_path[p] = i

    updated = 0
    missing = 0
    for abs_path in sorted(MODELS_DIR.rglob("*")):
        if not abs_path.is_file() or abs_path.suffix not in _MODEL_SUFFIXES:
            continue
        rel = str(abs_path.relative_to(ROOT)).replace("\\", "/")
        size = abs_path.stat().st_size
        if size <= SIZE_THRESHOLD_BYTES:
            continue
        if rel not in by_path:
            # Kein Manifest-Eintrag: anlegen (große Datei muss auslieferbar sein)
            _parts = rel.split("/")
            _name = f"{_parts[1]}_{abs_path.stem}" if len(_parts) > 2 else abs_path.stem
            used = {m.get("name") for m in models if isinstance(m, dict)}
            _n = _name
            _i = 2
            while _n in used:
                _n = f"{_name}_{_i}"
                _i += 1
            models.append(
                {
                    "name": _n,
                    "bundled": True,
                    "bundled_path": rel,
                    "sha256": _sha256_path(abs_path),
                    "size_bytes": size,
                    "required": False,
                    "fallback": "dsp_ersatzpfad",
                    "size_gb": round(size / (1024**3), 3),
                }
            )
            by_path[rel] = len(models) - 1
            print(f"NEW-ENTRY {rel} → {_n}")

        base = _asset_base(rel)
        part_sizes = _plan_part_sizes(size)
        if len(part_sizes) == 1:
            assets = [base]
        else:
            assets = [f"{base}.part{i:02d}" for i in range(len(part_sizes))]

        part_sha: dict[str, str] = {}
        part_size: dict[str, int] = {}
        offset = 0
        with abs_path.open("rb") as fh:
            for asset, part_len in zip(assets, part_sizes):
                fh.seek(offset)
                buf = fh.read(part_len)
                part_sha[asset] = _sha256_bytes(buf)
                part_size[asset] = len(buf)
                offset += len(buf)

        entry = models[by_path[rel]]
        entry.update(
            {
                "delivery": "release",
                "release_tag": RELEASE_TAG,
                "assets": assets,
                "part_sha256": part_sha,
                "part_size_bytes": part_size,
            }
        )
        updated += 1
        print(f"RELEASE {rel} ({len(assets)} Asset(s), {size} Bytes)")

    data["models"] = models
    MANIFEST.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nSummary: updated={updated} missing_manifest={missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
