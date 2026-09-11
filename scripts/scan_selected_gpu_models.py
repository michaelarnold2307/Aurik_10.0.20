#!/usr/bin/env python3
"""§v10.748b (2026-09-09): Begrenzter GPU-Scan für ausgewählte Modelle.

Der Voll-Scan (onnx_gpu_compat_scan.py) lädt ALLE ONNX-Modelle nacheinander
und kann bei paralleler Pipeline-Ausführung OOM auslösen (Befund 2026-09-09).
Dieser Wrapper scannt nur die übergebenen Pfade (Standard: die vier
§v10.40c-Kernmodelle) und merged in die bestehende Registry.

Usage:
    .venv_aurik/bin/python scripts/scan_selected_gpu_models.py [pfad ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.onnx_gpu_compat_scan import _REGISTRY_OUT, _REPO_ROOT, _scan_model

_DEFAULTS = [
    "models/mert/mert_330m.onnx",
    "models/bs_roformer/bs_roformer_317_core.onnx",
    "models/clap/audio_encoder.onnx",
    "models/melbandroformer/melbandroformer_optimized.onnx",
]


def main() -> int:
    _targets = sys.argv[1:] or _DEFAULTS
    _registry: dict = {}
    for _rel in _targets:
        _p = _REPO_ROOT / _rel
        if not _p.is_file():
            print(f"FEHLT: {_rel}")
            continue
        _t0 = time.perf_counter()
        try:
            _entry = _scan_model(_p)
        except Exception as _e:  # pylint: disable=broad-except
            _entry = {
                "verdict": "unknown",
                "backend": None,
                "cpu_ms": None,
                "gpu_ms": None,
                "note": f"{type(_e).__name__}: {str(_e)[:100]}",
            }
        _entry["scan_s"] = round(time.perf_counter() - _t0, 1)
        _registry[_rel] = _entry
        print(f"{_rel} -> verdict={_entry['verdict']} cpu={_entry.get('cpu_ms')}ms gpu={_entry.get('gpu_ms')}ms")
    if _REGISTRY_OUT.is_file():
        _old = json.loads(_REGISTRY_OUT.read_text(encoding="utf-8"))
        _old.update(_registry)
        _registry = _old
    _REGISTRY_OUT.write_text(json.dumps(_registry, indent=2, sort_keys=True), encoding="utf-8")
    print(f"REGISTRY GESCHRIEBEN: {_REGISTRY_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
