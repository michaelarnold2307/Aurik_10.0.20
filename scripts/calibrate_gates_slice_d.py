"""Slice D — Gate-Kalibrierung über N≥5 Songs (MUSDB18-HQ).

DECLIPPER_SOTA_PLAN.md Slice D + REKOMBINATION_ZEITPUNKT_ANALYSE.md §6.4:
No-Harm-Deltas messen (Gates auf unschädlichem Input), P90 als Schwellen-
Referenz, Evidenzblock (Seed, 95 %-CI, Maintainer Sign-off) schreiben.

Deterministisch (§G5 (copilot-instructions.md)): feste Song-Auswahl, fester
Seed, feste 60-s-Segmente.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # Repo-Root für backend/

logger = logging.getLogger(__name__)

MUSDB = Path("data/musdb18hq/test")
SONGS = [
    "AM Contra - Heart Peripheral",
    "BKS - Too Much",
    "Bobby Nobody - Stitch Up",
    "Motor Tapes - Shore",
    "Secretariat - Over The Top",
]
SEG_SEC = 60.0
SEG_OFFSET_SEC = 60.0
SEED = 20260912


def _seg(path: Path, sr: int) -> np.ndarray:
    x, fs = sf.read(str(path), dtype="float32", always_2d=True)
    n = min(len(x), int((SEG_OFFSET_SEC + SEG_SEC) * fs))
    x = x[:n]
    s0 = min(int(SEG_OFFSET_SEC * fs), max(0, len(x) - int(SEG_SEC * fs)))
    seg = x[s0 : s0 + int(SEG_SEC * fs)]
    if len(seg) < int(SEG_SEC * fs):
        seg = np.pad(seg, ((0, int(SEG_SEC * fs) - len(seg)), (0, 0)))
    return seg.T  # channels-first (C, N)


def _p90(values: list[float]) -> float:
    return float(np.percentile(np.asarray(values), 90))


def main() -> None:
    recomb_rows: list[dict] = []
    declip_rows: list[dict] = []

    for song in SONGS:
        song_dir = MUSDB / song
        if not (song_dir / "mixture.wav").exists():
            logger.warning("Song fehlt: %s", song)
            continue
        mixture = _seg(song_dir / "mixture.wav", 44100)
        vocals = _seg(song_dir / "vocals.wav", 44100)
        acc = mixture - vocals  # perfekte Summe → Residuum = 0 (No-Harm)

        # ── C1–C3-Rekombinations-Gates (No-Harm: perfekte Separation) ──
        from backend.core.dsp.stem_recombination_gates import recombine_stems_with_gates

        _, res = recombine_stems_with_gates(mixture, vocals, acc, vocals, acc, 44100)
        recomb_rows.append(
            {
                "song": song,
                "offset_samples": res.offset_samples,
                "alignment_corr": res.alignment_corr,
                "residue_bands_reused": res.residue_bands_reused,
                "residue_db_max_over": res.residue_db_max_over,
                "itd_drift_us": res.itd_drift_us,
                "ild_drift_db": res.ild_drift_db,
                "iacc_drop": res.iacc_drop,
                "stereo_ok": res.stereo_ok,
            }
        )

        # ── Declip-Gates (No-Harm: ungeclippte Musik) ──
        from backend.core.phases.phase_07_declipper import DeclipperPhase, _harmonic_distortion_proxy

        mono = mixture.mean(axis=0).astype(np.float32)
        result = DeclipperPhase().process(mono, sample_rate=44100)
        proxy_in = _harmonic_distortion_proxy(mono, 44100)
        proxy_out = _harmonic_distortion_proxy(np.asarray(result.audio, dtype=np.float32), 44100)
        declip_rows.append(
            {
                "song": song,
                "declip_applied": bool(result.metrics.get("declip_applied")),
                "sparse_used": bool(result.metrics.get("sparse_used")),
                "aspade_used": bool(result.metrics.get("aspade_used")),
                "cqtdiff_used": bool(result.metrics.get("cqtdiff_used")),
                "proxy_delta": round(proxy_out - proxy_in, 6),
                "clip_fraction": float(result.metrics.get("clip_fraction", 0.0)),
            }
        )

    report = {
        "slice": "D",
        "date": "2026-09-12",
        "seed": SEED,
        "songs": len(recomb_rows),
        "segment": f"{SEG_OFFSET_SEC}s..{SEG_OFFSET_SEC + SEG_SEC}s",
        "recombination_no_harm_p90": {
            "offset_samples": _p90([abs(r["offset_samples"]) for r in recomb_rows]),
            "residue_bands_reused": _p90([r["residue_bands_reused"] for r in recomb_rows]),
            "residue_db_max_over": _p90([r["residue_db_max_over"] for r in recomb_rows]),
            "itd_drift_us": _p90([r["itd_drift_us"] for r in recomb_rows]),
            "ild_drift_db": _p90([r["ild_drift_db"] for r in recomb_rows]),
            "iacc_drop": _p90([r["iacc_drop"] for r in recomb_rows]),
        },
        "declip_no_harm": declip_rows,
        "rows": recomb_rows,
    }
    out_dir = Path("docs/reports/calibration")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "2026-09-12_slice_d_gates.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["recombination_no_harm_p90"], indent=2))
    print("declip_no_harm:", json.dumps(declip_rows, indent=2))
    print("Report:", out_dir / "2026-09-12_slice_d_gates.json")


if __name__ == "__main__":
    main()
