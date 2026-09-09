"""§v10.40c (2026-09-09): ONNX-GPU-Kompatibilitäts-Scan + CPU-wenn-schneller-Regel.

Testet JEDES ONNX-Modell unter ``models/`` und ``plugins/*/``:
1. CPU-Baseline (Zeitmessung)
2. ROCMExecutionProvider (Kompatibilität + Zeitmessung)
3. MIGraphXExecutionProvider via MIGraphXSession (≤ 200 MB, §v10.40-Compile-Regel)

Verdict-Regeln:
- GPU inkompatibel (Session fällt zurück / wirft)          → ``cpu``
- GPU funktioniert, aber NICHT strikt schneller als CPU    → ``cpu``
  („Wenn Modelle auf CPU schneller sind, laufen sie auf CPU.")
- MIGraphX funktioniert und ist schneller                  → ``migraphx``
- ROCm funktioniert und ist schneller                      → ``rocm``

Ergebnis: ``backend/core/gpu_model_registry.json`` (konsumiert von
``backend/core/gpu_model_registry.py`` → ``ONNXInferenceSession``).

Aufruf:  .venv_aurik/bin/python scripts/onnx_gpu_compat_scan.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_REGISTRY_OUT = _REPO_ROOT / "backend" / "core" / "gpu_model_registry.json"
_MIGRAPHX_MAX_MB = 200  # §v10.40 Compile-Zeit-Regel
_DEFAULT_DIM = 256


def _collect_models(limit: int | None) -> list[Path]:
    _seen: set[str] = set()
    _out: list[Path] = []
    for _base in ("models", "plugins"):
        for _p in sorted((_REPO_ROOT / _base).rglob("*.onnx")):
            _rel = _p.relative_to(_REPO_ROOT).as_posix()
            if _rel in _seen:
                continue
            _seen.add(_rel)
            _out.append(_p)
            if limit and len(_out) >= limit:
                return _out
    return _out


def _dummy_inputs(session) -> dict:
    """Erzeugt realistische Dummy-Inputs aus den ORT-Eingabe-Signaturen."""
    _inputs: dict = {}
    for _inp in session.get_inputs():
        _shape: list[int] = []
        for _d in _inp.shape:
            if isinstance(_d, int):
                _shape.append(_d)
            elif isinstance(_d, str) and _d.lower() in ("batch", "n"):
                _shape.append(1)
            elif _d is None or isinstance(_d, str):
                _shape.append(_DEFAULT_DIM)
            else:
                _shape.append(1)
        _shape = [max(1, _d) for _d in _shape]
        for _dtype in (np.float32, np.int64):
            try:
                _arr = np.zeros(tuple(_shape), dtype=_dtype)
                session.run(None, {_inp.name: _arr})
                _inputs[_inp.name] = _arr
                break
            except Exception:
                continue
        else:
            _inputs[_inp.name] = np.zeros(tuple(_shape), dtype=np.float32)
    return _inputs


def _bench(session, inputs: dict, runs: int = 3) -> float:
    _times: list[float] = []
    for _ in range(runs):
        _t0 = time.perf_counter()
        session.run(None, inputs)
        _times.append((time.perf_counter() - _t0) * 1000.0)
    return float(np.median(_times))


def _with_timeout(fn, seconds: float):
    """Führt fn mit hartem Wanduhr-Timeout aus (SIGALRM, Hauptthread)."""
    import signal

    def _handler(signum, frame):
        raise TimeoutError(f"Schritt > {seconds}s")

    _old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(seconds))
    try:
        return fn()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, _old)


def _scan_model(path: Path) -> dict:
    import onnxruntime as ort

    _size_mb = path.stat().st_size / (1024**2)
    _entry: dict = {"size_mb": round(_size_mb, 1), "cpu_ms": None, "gpu_ms": None, "backend": None, "verdict": "cpu", "note": ""}

    # 1) CPU-Baseline
    try:
        def _cpu_step():
            _cpu = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            _inputs = _dummy_inputs(_cpu)
            return _cpu, _inputs

        _cpu, _inputs = _with_timeout(_cpu_step, 60.0)
        _entry["cpu_ms"] = round(_bench(_cpu, _inputs), 3)
    except Exception as exc:
        _entry["note"] = f"CPU-Load fehlgeschlagen: {type(exc).__name__}"
        return _entry

    # 2) ROCm EP
    try:
        def _rocm_step():
            return ort.InferenceSession(str(path), providers=["ROCMExecutionProvider", "CPUExecutionProvider"])

        _rocm = _with_timeout(_rocm_step, 90.0)
        _used = str(_rocm.get_providers()[0])
        if _used != "ROCMExecutionProvider":
            _entry["note"] = f"ROCm-EP nicht platziert ({_used})"
        else:
            _rocm_ms = _bench(_rocm, _inputs)
            _entry["gpu_ms"] = round(_rocm_ms, 3)
            _entry["backend"] = "rocm"
    except Exception as exc:
        _entry["note"] = f"ROCm-EP-Fehler: {type(exc).__name__}"

    # 3) MIGraphX (≤ 200 MB)
    if _size_mb <= _MIGRAPHX_MAX_MB:
        try:
            from backend.core.migraphx_adapter import MIGraphXSession, is_migraphx_available

            if is_migraphx_available():
                def _mgx_step():
                    return MIGraphXSession(path, providers=["MIGraphXExecutionProvider", "CPUExecutionProvider"])

                _mgx = _with_timeout(_mgx_step, 45.0)
                _mgx_ms = _bench(_mgx, _inputs)
                if _entry["gpu_ms"] is None or _mgx_ms < _entry["gpu_ms"]:
                    _entry["gpu_ms"] = round(_mgx_ms, 3)
                    _entry["backend"] = "migraphx"
                _entry["note"] = "MIGraphX OK"
        except Exception as exc:
            _entry["note"] = f"{_entry['note']}; MIGraphX: {type(exc).__name__}".strip("; ")

    # Verdict: GPU nur wenn strikt schneller als CPU
    if _entry["backend"] and _entry["gpu_ms"] is not None and _entry["cpu_ms"] is not None:
        if _entry["gpu_ms"] < _entry["cpu_ms"]:
            _entry["verdict"] = _entry["backend"]
        else:
            _entry["note"] = f"{_entry['note']}; CPU schneller/gleich".strip("; ")
            _entry["verdict"] = "cpu"
    return _entry


def main() -> int:
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--limit", type=int, default=None, help="Nur die ersten N Modelle scannen")
    _args = _ap.parse_args()

    _models = _collect_models(_args.limit)
    print(f"Scan: {len(_models)} ONNX-Modelle")
    _registry: dict = {}
    _counts: dict[str, int] = {"migraphx": 0, "rocm": 0, "cpu": 0}
    for _i, _p in enumerate(_models, 1):
        _rel = _p.relative_to(_REPO_ROOT).as_posix()
        print(f"[{_i:2d}/{len(_models)}] scanne {_rel} ...", flush=True)
        _t0 = time.perf_counter()
        _entry = _scan_model(_p)
        _entry["scan_s"] = round(time.perf_counter() - _t0, 1)
        _registry[_rel] = _entry
        _counts[_entry["verdict"]] = _counts.get(_entry["verdict"], 0) + 1
        print(
            f"[{_i:2d}/{len(_models)}] {_rel:<70} → {_entry['verdict']:<9} "
            f"(cpu={_entry['cpu_ms']}ms, gpu={_entry['gpu_ms']}ms, {_entry['note'][:40]})"
        )

    _registry["_summary"] = {
        "generated_by": "scripts/onnx_gpu_compat_scan.py",
        "rule": "GPU nur wenn strikt schneller als CPU; MIGraphX ≤ 200 MB",
        "counts": _counts,
    }
    _REGISTRY_OUT.parent.mkdir(parents=True, exist_ok=True)
    _REGISTRY_OUT.write_text(json.dumps(_registry, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nFertig: {_counts} → {_REGISTRY_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
