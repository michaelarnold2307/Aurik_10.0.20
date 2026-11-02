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
import os
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


def _force_cpu_requested() -> bool:
    """Globaler GPU-Stopp: AURIK_FORCE_CPU=1/true/yes (ml_device_manager-Vertrag)."""
    return os.environ.get("AURIK_FORCE_CPU", "").strip().lower() in ("1", "true", "yes")


_device_probe_cache: dict[str, bool] = {}


def _gpu_device_present() -> bool:
    """True wenn ein ECHTES GPU-Gerät existiert (nicht nur ORT-Build-Unterstützung).

    §III.9 / §V6 (copilot-instructions.md): ``ort.get_available_providers()``
    meldet ``ROCMExecutionProvider`` bereits build-seitig. Auf Hosts OHNE
    ROCm-Gerät (``hipGetDeviceProperties`` → ``HIP failure 100: no ROCm-capable
    device``) abortet ORT dann nativ (``Fatal Python error: Aborted``) —
    Produktionsbefund (phase_03-Denoise, Nativ-Absturz im Testlauf). GPU-EPs
    werden deshalb nur bei nachgewiesenem Gerät zugelassen.
    """
    if "present" in _device_probe_cache:
        return _device_probe_cache["present"]
    present = False
    try:
        import re

        # CUDA_VISIBLE_DEVICES/HIP_VISIBLE_DEVICES steuern die GPU-Sicht fuer ORT's
        # HIP. §III.9/§V6 (copilot-instructions.md): Produktionsbefund — eine
        # kaputte/leere Wertliste (z.B. " OMP_NUM_THREADS=1") laesst ORT's HIP
        # ``hipGetDeviceProperties`` → GPU=-1 liefern (HIP failure 100) und
        # abortet nativ. torch.cuda.is_available() sieht das NICHT (nutzt
        # HIP_VISIBLE_DEVICES) — deshalb hier explizit die Env-Liste pruefen:
        #   nicht gesetzt → None (alle Geraete), "" → versteckt, "0"/"0,1" →
        #   sichtbar, sonst (kaputt) → versteckt (sicherster Weg).
        def _visible(env: str) -> bool | None:
            v = os.environ.get(env)
            if v is None:
                return None
            v = v.strip()
            if v == "":
                return False
            return bool(re.fullmatch(r"\d+(,\d+)*", v))

        _cvd = _visible("CUDA_VISIBLE_DEVICES")
        _hvd = _visible("HIP_VISIBLE_DEVICES")
        if _cvd is False or _hvd is False:
            present = False
        elif _cvd is True or _hvd is True:
            present = True
        else:
            try:
                import torch

                present = bool(torch.cuda.is_available())
            except Exception:  # §V6 (copilot-instructions.md) — kein torch
                present = os.path.exists("/dev/kfd") or os.path.exists("/dev/nvidiactl")
    except Exception:  # §V6 (copilot-instructions.md)
        present = False
    _device_probe_cache["present"] = present
    return present


_ort_guard_installed = False


def install_onnx_gpu_guard() -> None:
    """Globaler ORT-GPU-Guard (§III.9/§V6 (copilot-instructions.md)).

    Patcht ``onnxruntime.InferenceSession`` so, dass GPU-EPs (ROCM/CUDA/MIGraphX/
    Tensorrt) entfernt werden, wenn keine echte sichtbare GPU existiert. Deckt ALLE
    Aufrufstellen ab (Plugins, ml_device_manager, onnx/runtime) — nicht nur die
    zentralen. Idempotent. Produktionsbefund: ORT's ROCm-EP abortet nativ
    (HIP failure 100) auf per Env versteckter GPU, wenn GPU-EP ohne Gerät
    angefordert wird.
    """
    global _ort_guard_installed
    if _ort_guard_installed:
        return
    try:
        import onnxruntime as ort

        _orig = ort.InferenceSession
        _GPU_EPS = {
            "ROCMExecutionProvider",
            "CUDAExecutionProvider",
            "MIGraphXExecutionProvider",
            "TensorrtExecutionProvider",
        }

        def _ep_name(p) -> str:
            return str(p[0]) if isinstance(p, tuple) else str(p)

        def _guarded(*args, **kwargs):
            providers = kwargs.get("providers")
            if providers is None:
                for _a in args[1:]:  # positional: (path, sess_options, providers, ...)
                    if isinstance(_a, (list, tuple)):
                        providers = _a
                        break
            if providers is not None and not _gpu_device_present():
                kept = [p for p in providers if _ep_name(p) not in _GPU_EPS] or ["CPUExecutionProvider"]
                kwargs["providers"] = kept
                args = tuple(a for a in args if a is not providers)  # Doppel-Arg vermeiden
            return _orig(*args, **kwargs)

        ort.InferenceSession = _guarded
        _ort_guard_installed = True
    except Exception as _guard_exc:  # §V6 (copilot-instructions.md)
        logger.warning(
            "§V6 (copilot-instructions.md): ORT-GPU-Guard nicht installierbar (%s: %s) — GPU-EPs bleiben ungeschützt",
            type(_guard_exc).__name__,
            _guard_exc,
        )


# Beim Import installieren — deckt alle späteren ort.InferenceSession-Aufrufe ab.
install_onnx_gpu_guard()


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
        return _providers  # CPU-only-Aufrufer respektieren
    # §III.9 / §V6 (copilot-instructions.md): kein echtes GPU-Gerät → CPU erzwungen
    # (kein nativer ORT-Abort auf CPU-Hosts wie HIP failure 100).
    if not _gpu_device_present():
        logger.info(
            "§v10.40c GPU-Policy: %s → CPU erzwungen (kein GPU-Gerät, nur ORT-Build-Support)",
            Path(model_path).name,
        )
        return ["CPUExecutionProvider"]

    if _force_cpu_requested():
        logger.info(
            "§v10.40c AURIK_FORCE_CPU=1 → %s CPU erzwungen (GPU-Forderung: %s)",
            Path(model_path).name,
            ", ".join(_pname(p) for p in _providers),
        )
        return ["CPUExecutionProvider"]

    if _verdict == "cpu":
        logger.info(
            "§v10.40c GPU-Policy: %s → CPU erzwungen (Registry: cpu)",
            Path(model_path).name,
        )
        return ["CPUExecutionProvider"]
    if _verdict == "migraphx":
        # NVIDIA-fähig: CUDA-EP in der Kette, damit dasselbe Register auch auf
        # CUDA-Hosts GPU nutzt (ORT ignoriert nicht verfügbare Provider).
        return ["MIGraphXExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    if _verdict == "rocm":
        # ROCm bereits angefordert → Aufrufer-Optionen (fp16-Tupel) beibehalten;
        # sonst (z.B. MIGraphX-Request) auf ROCm downgraden (Registry-Verdict).
        if any(_pname(p) == "ROCMExecutionProvider" for p in _providers):
            return _providers
        return ["ROCMExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
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
      - ``AURIK_FORCE_CPU=1``    → CPU erzwungen (globaler GPU-Stopp; Fix
        2026-10-05: gilt auch für direkte Registry-Aufrufer wie bigvgan/flashsr).
    """
    if _force_cpu_requested():
        logger.info("§v10.40c AURIK_FORCE_CPU=1 → %s CPU erzwungen (globaler GPU-Stopp)", Path(model_path).name)
        return ["CPUExecutionProvider"]
    try:
        import onnxruntime as ort

        _avail = set(ort.get_available_providers())
    except Exception as exc:
        logger.warning(
            "gpu_model_registry: onnxruntime-Provider nicht abfragbar (%s) -> CPU-Provider (§V6, copilot-instructions.md)",
            exc,
        )
        return ["CPUExecutionProvider"]
    if not prefer_gpu:
        return ["CPUExecutionProvider"]
    # §III.9 / §V6 (copilot-instructions.md): GPU-EP nur bei echtem Gerät — ORT
    # meldet ROCm build-seitig auch OHNE Gerät (Abort: HIP failure 100).
    if not _gpu_device_present():
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
    "BanquetVinyl": "models/banquet/banquet_vinyl_final.onnx",
}


def apply_gpu_policy_for_plugin(providers: Sequence[_Provider], plugin_name: str) -> list[_Provider]:
    """Brücke für get_ort_providers("Plugin")-Aufrufer: Registry-Verdict je Modell."""
    _hint = PLUGIN_MODEL_HINTS.get(plugin_name)
    if _hint is None:
        return list(providers)
    return apply_gpu_policy(providers, _hint)
