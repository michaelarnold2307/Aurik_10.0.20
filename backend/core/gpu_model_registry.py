"""§v10.40c (2026-09-09): Per-Modell-GPU-Policy-Registry.

Konsumiert das Ergebnis des Kompatibilitäts-Scans
(``scripts/onnx_gpu_compat_scan.py`` → ``backend/core/gpu_model_registry.json``).

Verdicts pro Modell:
- ``"migraphx"`` — GPU via MIGraphXExecutionProvider (gemessen schneller als CPU)
- ``"rocm"``      — GPU via ROCMExecutionProvider (gemessen schneller als CPU)
- ``"cpu"``       — CPU ERZWUNGEN: GPU inkompatibel ODER CPU gleich/schneller
                   (Regel: „Wenn Modelle auf CPU schneller sind, laufen sie auf CPU.")
- ``"unknown"``   — nicht gescannt → bisheriges Aufrufer-Verhalten bleibt

Schichtung: ``ml_device_manager`` entscheidet die Plugin-Ebene
(``_HEAVY_ML_PLUGINS``/``_gpu_disabled_plugins``, ``AURIK_FORCE_CPU``);
diese Registry verfeinert auf Datei-Ebene in ``ONNXInferenceSession``.
Sie fügt NIE einen GPU-Provider hinzu, wenn der Aufrufer CPU-only ist.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_REGISTRY_PATH = Path(__file__).resolve().parent / "gpu_model_registry.json"
_cache: dict | None = None
_cache_lock = threading.Lock()

_VALID_VERDICTS = ("migraphx", "rocm", "cpu", "unknown")


def load_registry(force: bool = False) -> dict:
    """Lädt die JSON-Registry (thread-sicher, gecacht). Fehlt sie → {}."""
    global _cache
    with _cache_lock:
        if _cache is None or force:
            _cache = {}
            try:
                if _REGISTRY_PATH.is_file():
                    with _REGISTRY_PATH.open(encoding="utf-8") as _f:
                        _data = json.load(_f)
                    if isinstance(_data, dict):
                        _cache = _data
            except Exception as exc:
                logger.debug("gpu_model_registry: Laden fehlgeschlagen (%s) — Default-Verhalten", exc)
        return _cache


def _match_key(model_path: str | Path) -> str | None:
    """Findet den Registry-Schlüssel über relative Pfad- oder Dateinamen-Übereinstimmung."""
    _p = Path(model_path)
    _candidates = [
        _p.as_posix(),
        _p.name,
    ]
    _reg = load_registry()
    for _c in _candidates:
        if _c in _reg:
            return _c
    # Rückwärts-Suffix-Match für verschobene Verzeichnisse (models/...)
    _suffix = f"/{_p.name}"
    for _key in _reg:
        if _key.endswith(_suffix) or _key == _p.name:
            return str(_key)
    return None


def verdict_for_model(model_path: str | Path) -> str:
    """Gibt den Verdict für ein Modell zurück (Standard: 'unknown')."""
    _key = _match_key(model_path)
    if _key is None:
        return "unknown"
    _entry = load_registry().get(_key, "unknown")
    # Registry-Format: {relpath: {"verdict": ...}} — Eintrag ist ein Dict
    # (Scan-Script), toleriert zusätzlich Legacy-String-Einträge.
    if isinstance(_entry, dict):
        _verdict = str(_entry.get("verdict", "unknown")).lower()
    elif isinstance(_entry, str):
        _verdict = _entry.lower()
    else:
        _verdict = "unknown"
    return _verdict if _verdict in _VALID_VERDICTS else "unknown"


def apply_gpu_policy(providers: list[str], model_path: str | Path) -> list[str]:
    """Wendet die Registry auf die Provider-Liste an (nie GPU-Hinzufügen bei CPU-only).

    - Verdict "cpu"      → CPU erzwungen (inkompatibel oder CPU schneller).
    - Verdict "migraphx" → MIGraphX zuerst, wenn GPU angefordert wurde.
    - Verdict "rocm"     → ROCMExecutionProvider zuerst, wenn GPU angefordert wurde.
    - "unknown"          → unverändert.
    """
    _verdict = verdict_for_model(model_path)
    _providers = [str(p) for p in (providers or [])]
    # GPU-Request-Erkennung: ROCm/CUDA/MIGraphX sind GPU-Provider — deren
    # Namen enthalten NICHT zwangsläufig "GPU" (ROCMExecutionProvider!).
    _gpu_provider_names = (
        "ROCMExecutionProvider",
        "CUDAExecutionProvider",
        "MIGraphXExecutionProvider",
        "TensorrtExecutionProvider",
    )
    _gpu_requested = any(_p in _gpu_provider_names or "GPU" in _p or "MIGraphX" in _p for _p in _providers)
    if not _gpu_requested:
        return _providers  # CPU-only-Aufrufer (AURIK_FORCE_CPU etc.) respektieren

    if _verdict == "cpu":
        logger.info(
            "§v10.40c GPU-Policy: %s → CPU erzwungen (Registry: cpu)",
            Path(model_path).name,
        )
        return ["CPUExecutionProvider"]
    if _verdict == "migraphx":
        return ["MIGraphXExecutionProvider", "CPUExecutionProvider"]
    if _verdict == "rocm":
        return ["ROCMExecutionProvider", "CPUExecutionProvider"]
    return _providers
