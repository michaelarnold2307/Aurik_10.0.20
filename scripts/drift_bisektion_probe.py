#!/usr/bin/env python
"""Drift-Bisektions-Probe — lokalisiert den In-Prozess-Zustands-Leak je Phase.

Befund 2026-11-25: Zwei identische Pässe IM SELBEN PROZESS driften um
~0.17 Δ-MOS (1.172 vs. 0.951), zwei frische Einzel-Prozesse nur um ~0.001.
Diese Probe läuft ZWEI Pässe in einem Prozess, hasht den Audio-Ausgang
JEDER Phase (Hook: `UnifiedRestorerV3._normalize_phase_result` — gemeinsamer
Durchlaufpunkt des Parallel- und Sequenzial-Pfads) und meldet die Phasen,
deren Ausgang zwischen den Pässen divergiert. Die ERSTE divergierende
Phase ist der Leak-Kandidat (im Parallel-Design erhält jede Phase denselben
Eingang; eine Divergenz ist damit phasen-intern oder im gemeinsamen
Vorlauf begründet — letzteres erkennt man daran, dass VIELE Phasen
gleichzeitig divergieren).

Verwendung (CPU-only, nice 19 empfohlen):
  AURIK_FORCE_CPU=1 AURIK_NO_RT_LIMIT=1 .venv_aurik/bin/python -B \
      scripts/drift_bisektion_probe.py --fenster 10.0 --paare 2
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.environ.setdefault("AURIK_FORCE_CPU", "1")
os.environ.setdefault("AURIK_NO_RT_LIMIT", "1")

import numpy as np  # noqa: E402
from scipy.io import wavfile  # noqa: E402

_PROBE: dict[int, dict[str, str]] = {}  # pass_idx -> phase_id -> sha1


def _sha(audio: np.ndarray) -> str:
    x = np.ascontiguousarray(np.asarray(audio, dtype=np.float32))
    return hashlib.sha1(x.tobytes()).hexdigest()[:16]


def _installiere_hook(pass_idx: int) -> None:
    from backend.core.unified_restorer_v3 import UnifiedRestorerV3

    # ACHTUNG: _normalize_phase_result ist ein STATICMETHOD mit der
    # Signatur (result, phase_id, fallback_audio) — KEIN self!
    _orig = UnifiedRestorerV3.__dict__["_normalize_phase_result"].__func__
    _table: dict[str, str] = {}
    _PROBE[pass_idx] = _table

    def _wrap(result, phase_id, audio):
        r = _orig(result, phase_id, audio)
        try:
            if hasattr(r, "audio") and isinstance(r.audio, np.ndarray):
                _table[str(phase_id)] = _sha(r.audio)
        except Exception:  # noqa: BLE001 - Probe darf nie den Lauf stören
            pass
        return r

    UnifiedRestorerV3._normalize_phase_result = staticmethod(_wrap)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fenster", type=float, default=10.0)
    ap.add_argument("--paare", type=int, default=2, help="Zahl der In-Prozess-Pässe")
    ap.add_argument("--eingabe", default=None, help="Schadens-WAV (Default: kassette-Titel)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    from scripts.erstelle_schadens_proben import restauriere_schnipsel

    if args.eingabe:
        pfad = Path(args.eingabe)
    else:
        kandidaten = sorted(Path("data/schadens_proben").glob("* (kassette).wav"))
        pfad = kandidaten[0]
    roh = wavfile.read(str(pfad))
    sr = int(roh[0])
    d = np.asarray(roh[1], np.float32)
    if d.dtype != np.float32:
        d = d / 32768.0
    if d.ndim == 2:
        d = d.mean(axis=1)
    rng = np.random.default_rng(args.seed)
    n = int(args.fenster * sr)
    start = int(rng.integers(0, max(1, len(d) - n)))
    fenster = d[start : start + n]
    print(f"PROBE: {pfad.name} fenster={args.fenster}s start={start} sr={sr}", flush=True)

    import backend.core.music_model_flags as flags

    for name in set(vars(flags)) & {n for n in dir(flags) if n.startswith("use_witness_kopf_")}:
        setattr(flags, name, False)

    for i in range(args.paare):
        _installiere_hook(i)
        restauriere_schnipsel(fenster, sr, f"{pfad}#drift_probe_pass{i}")
        print(f"PROBE: Pass {i} fertig ({len(_PROBE[i])} Phasen gehasht)", flush=True)

    t0, t1 = _PROBE.get(0, {}), _PROBE.get(1, {})
    gemeinsame = sorted(set(t0) & set(t1))
    print("PROBE: Phase-Vergleich Pass 0 vs Pass 1:", flush=True)
    divergierend: list[str] = []
    for pid in gemeinsame:
        if t0[pid] != t1[pid]:
            divergierend.append(pid)
    if divergierend:
        print(f"  DIVERGIERENDE PHASEN ({len(divergierend)}): {', '.join(divergierend)}", flush=True)
    else:
        print("  KEINE Phase divergiert — Leak liegt im Merge/Vorlauf (nächster Schritt dokumentiert).", flush=True)
    nur0 = sorted(set(t0) - set(t1))
    nur1 = sorted(set(t1) - set(t0))
    if nur0 or nur1:
        print(f"  nur Pass0: {nur0}; nur Pass1: {nur1}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
