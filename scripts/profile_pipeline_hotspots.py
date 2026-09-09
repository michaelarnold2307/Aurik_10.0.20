#!/usr/bin/env python3
"""cProfile-Hotspot-Analyse für die DSP-Phasen (P2-Perf, 2026-09-09).

Profilert einen 10-s-Slice des Elke-Best-Songs im Quality-Modus auf dem
Ganzsong-Pfad (AURIK_WHOLE_SONG=1 — der zukünftige Default) und schreibt
die Top-40-Cumtime-Funktionen nach /tmp/aurik_profile.txt.

Usage:
    .venv_aurik/bin/python scripts/profile_pipeline_hotspots.py

Läuft bewusst NACH dem Ganzsong-Verifikationslauf (CPU-Konkurrenz).
"""

from __future__ import annotations

import cProfile
import pstats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf


def main() -> int:
    from backend.core.performance_guard import QualityMode
    from backend.core.unified_restorer_v3 import RestorationConfig, UnifiedRestorerV3

    src = "test_audio/Elke Best - Du wolltest nur ein Abenteuer, aber ich suchte einen Freund.mp3"
    y, sr = sf.read(src, dtype="float32", always_2d=True)
    y = np.asarray(y[: sr * 10], dtype=np.float32)  # 10-s-Slice

    restorer = UnifiedRestorerV3(RestorationConfig(mode=QualityMode.QUALITY))
    prof = cProfile.Profile()
    prof.enable()
    try:
        restorer.restore(y, sr, whole_song=True)
    finally:
        prof.disable()

    out = Path("/tmp/aurik_profile.txt")
    with out.open("w") as fh:
        stats = pstats.Stats(prof, stream=fh).sort_stats("cumtime")
        stats.print_stats(40)
    print(f"Profil geschrieben: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
