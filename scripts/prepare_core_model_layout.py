"""Prepare directory layout for core missing model paths.

Creates only parent directories for required model artifacts.
It never creates fake model files.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORE_PATHS = [
    "models/rmvpe/rmvpe.onnx",
    "models/sgmse_plus/sgmse_plus.ts",
    "models/versa/hub_cache/checkpoints/ft_wav2vec2_large_ll60k_mdf_p1_200epochs_all_192epochs.pth",
    "models/flow_matching/flow_matching.onnx",
    "models/gacela/model/01_400000.pt",
]


def main() -> int:
    created = 0
    existing = 0

    for rel in CORE_PATHS:
        path = ROOT / rel
        parent = path.parent
        if parent.exists():
            print(f"DIR_OK {parent.relative_to(ROOT)}")
            existing += 1
        else:
            parent.mkdir(parents=True, exist_ok=True)
            print(f"DIR_CREATED {parent.relative_to(ROOT)}")
            created += 1

        if path.exists():
            print(f"FILE_OK {rel}")
        else:
            print(f"FILE_MISSING {rel}")

    print(f"\nSummary: dirs_created={created}, dirs_already_present={existing}")
    print("Hinweis: Es wurden keine Platzhalterdateien erzeugt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
