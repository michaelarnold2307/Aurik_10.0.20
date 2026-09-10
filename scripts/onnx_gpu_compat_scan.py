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
import gc
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

# Per-Modell-Input-Overrides: _DEFAULT_DIM=256 ist für zeitdynamische Modelle
# ungeeignet — wav2vec2-Conv degeneriert unterhalb ~320 Samples (Kernel 2 >
# Eingang → RuntimeError). Lange Zeitachse daher explizit vorgeben.
# §v10-SINGMOS-ONNX (2026-09-10)
_INPUT_OVERRIDES: dict[str, dict[str, tuple[tuple[int, ...], str]]] = {
    "models/singmos/singmos_pro.onnx": {
        "audio": ((1, 1, 160000), "float32"),
        "audio_length": ((1,), "int64"),
        "domain_id": ((1,), "int64"),
    },
    # vae_decoder (2026-09-10): _DEFAULT_DIM=256 erzeugt ein
    # (1,8,256,256)-Latent → 1024×1024-Mel, ~80 s/CPU-Lauf und VRAM-Spitzen
    # auf ROCm (2× System-Crash im Scan). Plugin-Realform (1,8,8,128) wie in
    # plugins/audioldm2_plugin.py (mel 64 Bins / 1024 Frames).
    "models/audioldm2/vae_decoder.onnx": {
        "latent": ((1, 8, 8, 128), "float32"),
    },
}


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


def _dummy_inputs(session, overrides: dict | None = None) -> dict:
    """Erzeugt realistische Dummy-Inputs aus den ORT-Eingabe-Signaturen."""
    _inputs: dict = {}
    for _inp in session.get_inputs():
        _ov = (overrides or {}).get(_inp.name)
        if _ov is not None:
            _shape, _dtype = _ov
            _inputs[_inp.name] = np.zeros(tuple(_shape), dtype=_dtype)
            continue
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
        # Dtype aus der deklarierten ORT-Signatur (z. B. "tensor(int64)") statt
        # Exception-getriebener Trial-Schleife — die lief bei Multi-Input-Modellen
        # ohnehin immer in den float32-Fallback und trieb einen Anti-Regression-
        # Verstoß (Bug 9: stummer except Exception: continue).
        _dtype = np.int64 if "int64" in str(_inp.type).lower() else np.float32
        _inputs[_inp.name] = np.zeros(tuple(_shape), dtype=_dtype)
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


_PARITY_TOL = 1e-3  # rel max|Δ| EP vs CPU; float32-Kernel-Normalabweichung liegt bei ~1e-5


def _random_feed(rng, inputs: dict, kind: str) -> dict:
    """Zufalls-Feed aus den Dummy-Inputs: 'normal' = Standardnormal; 'sane' =
    uniform(-1,1) mit 0.5 für Skalar-Inputs (Zeit-Konditionierung — t=0 erzeugt
    z. B. log(0)=NaN in der CPU-Referenz). Int64-Inputs bleiben Null-Dummies."""
    _feed: dict = dict(inputs)
    for _k, _v in _feed.items():
        if _v.dtype != np.float32 or _v.size == 0:
            continue
        if kind == "sane" and _v.size == 1:
            _feed[_k] = np.full(_v.shape, 0.5, dtype=np.float32)
        elif kind == "sane":
            _feed[_k] = rng.uniform(-1.0, 1.0, _v.shape).astype(np.float32)
        else:
            _feed[_k] = rng.standard_normal(_v.shape).astype(np.float32)
    return _feed


def _parity_note(cpu_sess, gpu_sess, inputs: dict) -> str:
    """Vergleicht EP-Ausgabe mit CPU-Ausgabe auf deterministischen Zufalls-Inputs.

    Geschwindigkeit allein reicht für ein GPU-Verdict nicht: ROCm/MIGraphX kann
    numerisch falsche Ergebnisse liefern (SGMSE+-Core 2026-09-10: rel ~3–5 %).
    Liefert "" bei Parität bzw. Begründung bei Abweichung/NaN. Ist die
    CPU-Referenz für beide Feed-Varianten nicht endlich, kein Urteil ("").
    """
    _rng = np.random.default_rng(0)
    if not any(v.dtype == np.float32 and v.size > 0 for v in inputs.values()):
        return ""
    for _kind in ("normal", "sane"):
        _par = _random_feed(_rng, inputs, _kind)
        try:
            _cpu_out = cpu_sess.run(None, _par)
        except Exception as exc:  # pylint: disable=broad-except
            return f"Paritaets-Check fehlgeschlagen: {type(exc).__name__}"
        if not _cpu_out or not all(np.all(np.isfinite(np.asarray(c))) for c in _cpu_out):
            continue  # CPU-Referenz nicht auswertbar — nächsten Feed versuchen
        try:
            _gpu_out = gpu_sess.run(None, _par)
        except Exception as exc:  # pylint: disable=broad-except
            return f"Paritaets-Check fehlgeschlagen: {type(exc).__name__}"
        for _c, _g in zip(_cpu_out, _gpu_out):
            _c_arr = np.asarray(_c, dtype=np.float32)
            _g_arr = np.asarray(_g, dtype=np.float32)
            if _c_arr.shape != _g_arr.shape:
                return "Paritaets-Check: Shape-Mismatch"
            if not np.all(np.isfinite(_g_arr)):
                return "EP liefert NaN/Inf (CPU endlich)"
            _err = float(np.max(np.abs(_c_arr - _g_arr)))
            _scale = max(float(np.max(np.abs(_c_arr))), 1e-9)
            if _err / _scale > _PARITY_TOL:
                return f"EP-Numerik weicht ab (rel={_err / _scale:.2e} vs CPU)"
        return ""
    return ""


def _join_note(*parts: str) -> str:
    return "; ".join(p for p in parts if p).strip()


def _scan_model(path: Path, overrides: dict | None = None) -> dict:
    import onnxruntime as ort

    _size_mb = path.stat().st_size / (1024**2)
    _entry: dict = {
        "size_mb": round(_size_mb, 1),
        "cpu_ms": None,
        "gpu_ms": None,
        "backend": None,
        "verdict": "cpu",
        "note": "",
    }
    _ok_backends: set[str] = set()
    # §Unload (2026-09-10): Referenzen vorab definieren, damit jeder Exit-Pfad
    # die Sessions freigeben kann — ROCm/CPU-Allokationen würden sonst über
    # 63 Modelle × 3 Sessions akkumulieren → OOM/System-Crash.
    _cpu = None
    _rocm = None
    _mgx = None

    # 1) CPU-Baseline
    try:

        def _cpu_step():
            _cpu = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            _inputs = _dummy_inputs(_cpu, overrides)
            return _cpu, _inputs

        _cpu, _inputs = _with_timeout(_cpu_step, 60.0)
        _entry["cpu_ms"] = round(_bench(_cpu, _inputs), 3)
    except Exception as exc:
        _entry["note"] = f"CPU-Load fehlgeschlagen: {type(exc).__name__}"
        _cpu = None
        gc.collect()
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
            # Numerik-Parität vs CPU — Geschwindigkeit allein genügt nicht.
            _pnote = _parity_note(_cpu, _rocm, _inputs)
            if _pnote:
                _entry["note"] = _pnote
            else:
                _ok_backends.add("rocm")
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
                _mgx_note = _parity_note(_cpu, _mgx, _inputs)
                if not _mgx_note and (_entry["gpu_ms"] is None or _mgx_ms < _entry["gpu_ms"]):
                    _entry["gpu_ms"] = round(_mgx_ms, 3)
                    _entry["backend"] = "migraphx"
                    _ok_backends.add("migraphx")
                _entry["note"] = _join_note(_entry["note"], _mgx_note or "MIGraphX OK")
        except Exception as exc:
            _entry["note"] = _join_note(_entry["note"], f"MIGraphX: {type(exc).__name__}")

    # Verdict: GPU nur wenn strikt schneller als CPU UND numerisch paritätisch.
    if _entry["backend"] and _entry["gpu_ms"] is not None and _entry["cpu_ms"] is not None:
        if _entry["backend"] in _ok_backends and _entry["gpu_ms"] < _entry["cpu_ms"]:
            _entry["verdict"] = _entry["backend"]
        else:
            if _entry["backend"] in _ok_backends:
                _entry["note"] = _join_note(_entry["note"], "CPU schneller/gleich")
            _entry["verdict"] = "cpu"
    # §Unload: Sessions dieses Modell-Tests freigeben, BEVOR das nächste Modell
    # geladen wird.
    _cpu = None
    _rocm = None
    _mgx = None
    gc.collect()
    return _entry


def _write_registry(_registry: dict) -> dict[str, int]:
    """Registry mit _summary persistieren (crash-resistent: nach jedem Modell)."""
    _counts: dict[str, int] = {"migraphx": 0, "rocm": 0, "cpu": 0}
    for _v in _registry.values():
        if isinstance(_v, dict) and _v.get("verdict") in _counts:
            _counts[_v["verdict"]] += 1
    _registry["_summary"] = {
        "generated_by": "scripts/onnx_gpu_compat_scan.py",
        "rule": "GPU nur wenn strikt schneller als CPU UND numerisch paritätisch (rel<=1e-3); MIGraphX <= 200 MB",
        "counts": _counts,
    }
    _REGISTRY_OUT.parent.mkdir(parents=True, exist_ok=True)
    _REGISTRY_OUT.write_text(json.dumps(_registry, indent=2, sort_keys=True), encoding="utf-8")
    return _counts


def main() -> int:
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--limit", type=int, default=None, help="Nur die ersten N Modelle scannen")
    _ap.add_argument(
        "--model",
        type=str,
        default=None,
        help="Nur dieses Modell scannen (relativer Pfad, z. B. models/sgmse_plus/sgmse_plus_core.onnx)",
    )
    _ap.add_argument(
        "--start",
        type=int,
        default=1,
        help="Erst ab diesem 1-basierten Index der VOLLEN Liste scannen (Resume nach Crash).",
    )
    _ap.add_argument(
        "--skip",
        type=str,
        default=None,
        help="Kommagetrennte Teilpfade; passende Modelle werden übersprungen (z. B. audioldm2).",
    )
    _args = _ap.parse_args()

    _models = _collect_models(None if _args.model else _args.limit)
    if _args.model:
        _models = [p for p in _models if p.relative_to(_REPO_ROOT).as_posix() == _args.model]
        if not _models:
            print(f"Modell nicht gefunden: {_args.model}")
            return 2
    if _args.start > 1 and not _args.model:
        _models = _models[_args.start - 1 :]
        print(f"Resume: starte bei Index {_args.start} der vollen Liste.")
    _skips = [s.strip() for s in (_args.skip or "").split(",") if s.strip()]
    if _skips and not _args.model:
        _before = len(_models)
        _models = [
            p
            for p in _models
            if not any(s in p.relative_to(_REPO_ROOT).as_posix() for s in _skips)
        ]
        if _before != len(_models):
            print(f"Übersprungen via --skip: {_before - len(_models)} Modell(e).")
    print(f"Scan: {len(_models)} ONNX-Modelle")
    # Bestehende Registry mergen, damit Teil-Scans (--model) keine Einträge löschen.
    _registry: dict = {}
    if _REGISTRY_OUT.exists():
        try:
            _registry = json.loads(_REGISTRY_OUT.read_text(encoding="utf-8"))
        except Exception:  # pylint: disable=broad-except
            _registry = {}
    _registry.pop("_summary", None)
    for _i, _p in enumerate(_models, 1):
        _rel = _p.relative_to(_REPO_ROOT).as_posix()
        print(f"[{_i:2d}/{len(_models)}] scanne {_rel} ...", flush=True)
        _t0 = time.perf_counter()
        _entry = _scan_model(_p, _INPUT_OVERRIDES.get(_rel))
        _entry["scan_s"] = round(time.perf_counter() - _t0, 1)
        _registry[_rel] = _entry
        print(
            f"[{_i:2d}/{len(_models)}] {_rel:<70} → {_entry['verdict']:<9} "
            f"(cpu={_entry['cpu_ms']}ms, gpu={_entry['gpu_ms']}ms, {_entry['note'][:40]})"
        )
        # §Crash-Resilienz: Ergebnis nach JEDEM Modell persistieren — stirbt der
        # Lauf mittendrin, bleiben alle bisherigen Verdicts erhalten.
        _write_registry(_registry)

    # Zählung aus der GEMERGTEN Registry (bei Teil-Scans sonst verfälscht).
    _counts = _write_registry(_registry)
    print(f"\nFertig: {_counts} → {_REGISTRY_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
