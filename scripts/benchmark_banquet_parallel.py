#!/usr/bin/env python3
"""§PERF-R P8/P10-Benchmark — BANQUET-Fenster-Parallelität + ORT-Thread-Tuning.

Misst die BANQUET-Inferenz (`_process_onnx`, phase_09-Pfad) unter verschiedenen
Konfigurationen. Ein Prozesslauf = eine Konfiguration (die Klasse-Attribute
werden beim Import aus der Umgebung gelesen):

  AURIK_BANQUET_INFER_PARALLEL   0..N  (0/1 = sequenziell, Default 0)
  AURIK_BANQUET_INTRA_OP_THREADS 1..N  (ORT-intra-op, Default 4)
  AURIK_BANQUET_BENCH_CPU        1     Provider-Auswahl auf CPU zwingen
  AURIK_BANQUET_BENCH_SECONDS    8.0   Probe-Länge in Sekunden

Ausgabe: eine JSON-Zeile auf stdout (wall_s, providers, sha des Outputs).
Die sha erlaubt den Bit-Identitäts-Nachweis CPU-sequenziell vs. CPU-parallel.

Hinweis §SOTA-ML-V5 (2026-09-18): Der ORT-ROCm-Pfad ist für BANQUET
abgeschaltet (numerisch defekte LSTM-Kernels); der GPU-Pfad ist der
Torch-ROCm-Kern (backend/core/dsp/banquet_torch_rocm.py). Dieses Skript
misst den ONNX-CPU-Pfad (P8/P10-Matrix); GPU-Zahlen liefert der
Produktionslauf mit AURIK_BANQUET_TORCH=1 + AURIK_BANQUET_TORCH_BATCH=N.

Aufrufbeispiel (16 Kerne):

  .venv_aurik/bin/python scripts/benchmark_banquet_parallel.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_PROBE_S = float(os.environ.get("AURIK_BANQUET_BENCH_SECONDS", "8.0"))
_FORCE_CPU = os.environ.get("AURIK_BANQUET_BENCH_CPU") == "1"

if _FORCE_CPU:
    import backend.core.ml_device_manager as _dm

    _dm.get_ort_providers = lambda name: ["CPUExecutionProvider"]
    import backend.core.gpu_model_registry as _gpr

    _gpr.apply_gpu_policy = lambda providers, model_path: providers


def main() -> None:
    sr = 48_000
    rng = np.random.default_rng(20260918)
    audio = (rng.standard_normal((1, int(sr * _PROBE_S))) * 0.05).astype(np.float32)

    from plugins.banquet_vinyl_plugin import get_banquet_plugin

    plugin = get_banquet_plugin()
    if not plugin.ensure_model_loaded():
        print(json.dumps({"error": "BANQUET-Modell nicht ladbar (models/banquet fehlt)"}))
        return

    providers = [str(p) for p in plugin._session.get_providers()] if plugin._session is not None else []

    # Warmup (ein Fenster) — danach saubere Messung über die ganze Probe.
    plugin._infer_window(audio[:, :sr].copy(), 1, sr, 1.0)

    t0 = time.perf_counter()
    out = plugin._process_onnx(audio, 1.0)
    wall_s = time.perf_counter() - t0

    sha = hashlib.sha256(np.ascontiguousarray(out).tobytes()).hexdigest()[:16]
    n_windows = int(out.shape[1] // (sr // 2)) + 1
    print(
        json.dumps(
            {
                "probe_s": _PROBE_S,
                "windows": n_windows,
                "infer_parallel": plugin.INFER_PARALLEL,
                "intra_op": plugin.INTRA_OP_THREADS,
                "providers": providers,
                "wall_s": round(wall_s, 3),
                "rt_factor": round(wall_s / _PROBE_S, 2),
                "sha": sha,
            }
        )
    )


if __name__ == "__main__":
    main()
