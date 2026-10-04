#!/usr/bin/env python3
"""Check missing core model artifacts and discover possible local source files.

This script does not download anything. It reports:
- target artifact presence
- potential local candidates found in models/
- concrete next action per artifact
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"


@dataclass(frozen=True)
class CoreArtifact:
    name: str
    target_rel: str
    search_globs: tuple[str, ...]
    next_action: str
    fallback_targets: tuple[str, ...] = ()


ARTIFACTS = [
    CoreArtifact(
        name="rmvpe",
        target_rel="models/rmvpe/rmvpe.onnx",
        search_globs=("**/*rmvpe*.onnx", "**/*rmvpe*.pt", "**/*rmvpe*.pth"),
        next_action="ONNX-Datei nach models/rmvpe/rmvpe.onnx bereitstellen oder Export-Skript integrieren.",
    ),
    CoreArtifact(
        name="sgmse_plus",
        target_rel="models/sgmse_plus/sgmse_plus.ts",
        search_globs=("**/*sgmse*.ts", "**/*sgmse*.onnx", "**/*sgmse*.pt", "**/*sgmse*.pth"),
        next_action="SGMSE+ TorchScript nach models/sgmse_plus/sgmse_plus.ts bereitstellen (ONNX optional).",
    ),
    CoreArtifact(
        name="versa",
        target_rel="models/versa/hub_cache/checkpoints/ft_wav2vec2_large_ll60k_mdf_p1_200epochs_all_192epochs.pth",
        search_globs=("**/*ft_wav2vec2_large_ll60k*.pth", "**/*singmos*.pth", "**/*versa*.pt", "**/*versa*.pth"),
        next_action="SingMOS-Pro-Checkpoint nach models/versa/hub_cache/checkpoints/ bereitstellen (ONNX optional).",
    ),
    CoreArtifact(
        name="flow_matching",
        target_rel="models/flow_matching/flow_matching.onnx",
        search_globs=("**/*flow*matching*.onnx", "**/*flow*matching*.pt", "**/*flow*matching*.pth"),
        next_action=(
            "Ersetzt 2026-10-04 (Hybrid-Release-Mode): Rolle haelt cqtdiff_plus bzw. diffwave "
            "als dokumentierter Fallback (validate_core_model_presence._runtime_ready_checks + "
            "tests/normative/test_hybrid_release_mode.py). Kein Download offen."
        ),
        fallback_targets=(
            "models/cqtdiff/score_network.onnx",
            "models/diffwave/diffwave_model.onnx",
        ),
    ),
    CoreArtifact(
        name="gacela",
        target_rel="models/gacela/model/01_400000.pt",
        search_globs=("**/01_400000.pt", "**/*gacela*.pt", "**/*gacela*.pth"),
        next_action="GACELA-Checkpoint 01_400000.pt nach models/gacela/model/ kopieren.",
    ),
]


def _find_candidates(globs: tuple[str, ...]) -> list[Path]:
    found: list[Path] = []
    for g in globs:
        found.extend(MODELS.glob(g))
    # unique + stable
    unique = sorted({p.resolve() for p in found})
    return [Path(p) for p in unique]


def main() -> int:
    missing = 0

    print("=" * 96)
    print("Core-Model Source Check")
    print("=" * 96)

    for a in ARTIFACTS:
        target = ROOT / a.target_rel
        if target.exists():
            print(f"OK      {a.target_rel}")
            continue

        active_fallbacks = [f for f in a.fallback_targets if (ROOT / f).exists()]
        if active_fallbacks:
            print(f"ERSETZT {a.target_rel} -> {active_fallbacks[0]}")
            continue

        missing += 1
        print(f"MISSING {a.target_rel}")
        candidates = _find_candidates(a.search_globs)

        if candidates:
            print("  candidates:")
            for c in candidates[:10]:
                rel = c.relative_to(ROOT) if str(c).startswith(str(ROOT)) else c
                print(f"    - {rel}")
            if len(candidates) > 10:
                print(f"    ... +{len(candidates) - 10} weitere")
        else:
            print("  candidates: none")

        print(f"  next_action: {a.next_action}")

    print("\nSummary:")
    if missing:
        print(f"  missing_artifacts={missing}")
        return 1

    print("  all core artifacts present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
