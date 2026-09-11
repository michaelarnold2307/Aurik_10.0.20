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
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# ORT-Provider-Eintrag: entweder Provider-Name oder (name, options)-Tupel (fp16-Pfad).
_Provider = str | tuple[Any, ...]

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


def apply_gpu_policy(providers: Sequence[_Provider], model_path: str | Path) -> list[_Provider]:
    """Wendet die Registry auf die Provider-Liste an (nie GPU-Hinzufügen bei CPU-only).

    - Verdict "cpu"      → CPU erzwungen (inkompatibel oder CPU schneller).
    - Verdict "migraphx" → MIGraphX zuerst, wenn GPU angefordert wurde.
    - Verdict "rocm"     → ROCMExecutionProvider zuerst, wenn GPU angefordert wurde.
    - "unknown"          → unverändert.

    Provider-Einträge können ORT-Tupel `(name, options)` sein (fp16-Pfad via
    get_ort_providers_fp16) — diese werden UNVERÄNDERT durchgereicht. Ein
    str()-Cast würde sie zu `"('ROCMExecutionProvider', {...})"` machen und ORT
    mit "Unknown Provider Type" scheitern lassen (Produktionsbefund).
    """
    _verdict = verdict_for_model(model_path)
    _providers: list[_Provider] = list(providers)

    def _pname(p) -> str:
        return str(p[0]) if isinstance(p, tuple) else str(p)

    # GPU-Request-Erkennung: ROCm/CUDA/MIGraphX sind GPU-Provider — deren
    # Namen enthalten NICHT zwangsläufig "GPU" (ROCMExecutionProvider!).
    _gpu_provider_names = (
        "ROCMExecutionProvider",
        "CUDAExecutionProvider",
        "MIGraphXExecutionProvider",
        "TensorrtExecutionProvider",
    )
    _gpu_requested = any(
        _pname(p) in _gpu_provider_names or "GPU" in _pname(p) or "MIGraphX" in _pname(p) for p in _providers
    )
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
        # ROCm bereits angefordert → Aufrufer-Optionen (fp16-Tupel) beibehalten;
        # sonst (z.B. MIGraphX-Request) auf ROCm downgraden (Registry-Verdict).
        if any(_pname(p) == "ROCMExecutionProvider" for p in _providers):
            return _providers
        return ["ROCMExecutionProvider", "CPUExecutionProvider"]
    return _providers


def get_onnx_providers(model_path: str | Path, prefer_gpu: bool = True) -> list[_Provider]:
    """Zentrale Provider-Wahl für ONNX-Sessions (Registry-konsultiert).

    Baut GPU-Kandidaten nur aus den auf DIESEM Host verfügbaren ORT-Providern
    (ROCm > MIGraphX > CUDA) und lässt die Registry entscheiden:
      - Verdict "rocm"/"migraphx" → GPU zuerst, CPU-Fallback.
      - Verdict "cpu"            → CPU erzwungen (Numerik-Paritäts-Gate
        §v10.762: EP rechnet nachweislich falsch oder langsamer).
      - "unknown"                 → GPU-Kandidaten + CPU-Fallback (unverändert
        durchgereicht; ORT ignoriert nicht verfügbare Provider selbst).
    """
    try:
        import onnxruntime as ort

        _avail = set(ort.get_available_providers())
    except Exception:
        return ["CPUExecutionProvider"]
    if not prefer_gpu:
        return ["CPUExecutionProvider"]
    candidates: list[_Provider] = []
    for _p in ("ROCMExecutionProvider", "MIGraphXExecutionProvider", "CUDAExecutionProvider"):
        if _p in _avail:
            candidates.append(_p)
            break
    candidates.append("CPUExecutionProvider")
    return apply_gpu_policy(candidates, model_path)


# Plugin-Name → Registry-Schlüssel-Hinweis: get_ort_providers("Plugin")-Aufrufer
# (ml_device_manager) erhalten damit ebenfalls das per-Modell-Numerik-Paritäts-
# Verdict (§v10.762) — GPU nur wo „rocm“ validiert, CPU wo „cpu“.
PLUGIN_MODEL_HINTS: dict[str, str] = {
    "SGMSE": "models/sgmse_plus/sgmse_plus_core.onnx",
    "CREPE": "models/crepe/crepe.onnx",
    "FCPE": "models/fcpe/fcpe.onnx",
    "RMVPE": "models/rmvpe/rmvpe.onnx",
    "PANNS": "models/panns/panns_wavegram_logmel_cnn14.onnx",
    "UTMOS": "models/utmosv2/utmosv2_ssl_encoder.onnx",
    "UTMOSv2": "models/utmosv2/utmosv2_ssl_encoder.onnx",
    "Vocos": "models/vocos_48khz/vocos_48khz.onnx",
    "HarmonicInpainting": "models/harmonic_inpainting/inpainting_best.onnx",
    "DemucsV4": "models/demucs/htdemucs_6s.onnx",
    "HiFiGAN": "models/hifi_gan/hifi_gan.onnx",
    "ApolloCore": "models/apollo/apollo_core.onnx",
    "BigVGAN": "models/bigvgan/bigvgan_v2.onnx",
    "BasicPitch": "models/basicpitch/basicpitch.onnx",
    "FlashSR": "models/flashsr/flashsr.onnx",
    "Aero": "models/aero/aero_12_48.onnx",
    "Gacela": "models/gacela/model/gacela_core.onnx",
    "DiffWave": "models/diffwave/diffwave_model.onnx",
    "MERT": "models/mert/mert_330m.onnx",
    "LaionCLAP": "models/clap/audio_encoder.onnx",
    "CLAP": "models/clap/audio_encoder.onnx",
    "Nvsr": "models/nvsr/nvsr.onnx",
    "MP_Senet": "models/mp_senet/mp_senet.onnx",
    "Miipher": "models/miipher_dit/flow_matching_dit.onnx",
    "Silero": "models/silero/silero_en_v5.onnx",
    "Whisper": "models/whisper/whisper_tiny.onnx",
    "BSRoFormer": "models/bs_roformer/bs_roformer_317_core.onnx",
}


def apply_gpu_policy_for_plugin(providers: Sequence[_Provider], plugin_name: str) -> list[_Provider]:
    """Brücke für get_ort_providers("Plugin")-Aufrufer: Registry-Verdict je Modell."""
    _hint = PLUGIN_MODEL_HINTS.get(plugin_name)
    if _hint is None:
        return list(providers)
    return apply_gpu_policy(providers, _hint)
