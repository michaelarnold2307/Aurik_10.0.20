"""MuQ-MuLan-Plugin — Musik-Text-Embedding-Witness (ONNX, GPU) für die Voranalyse.

MuQ-MuLan (Tencent AILab, arXiv:2501.01108, Gewichte CC-BY-NC-4.0) ist das
SOTA-Joint-Embedding der Musik-Text-Klasse (Nachfolger von LAION-CLAP für
Tagging/Ära/Genre). Dieses Plugin nutzt den **Audio-Turm** als ONNX:

    models/muq_mulan/muq_mulan.onnx   (Export: scripts/export_muq_mulan_onnx.py)

    Waveform (1, 240000) @ 24 kHz  →  Audio-Embedding (1, 768)

Laufzeit:

- ONNX Runtime mit ROCMExecutionProvider (GPU) → CPUExecutionProvider-Fallback.
- Lazy-Singleton, registriert in ML-Memory-Budget + PluginLifecycleManager.
- Determinismus (§G5 (GEBOTE.md)): eval-Graph, festes 10-s-Zentrumsfenster, keine Dropouts.
- Silent-Failure (§V6 (copilot-instructions.md)): fehlende .onnx/Fehler ⇒ None + logger.warning.

Wiring (CLI/GUI-synchron, AGENTS.md §3): Die Voranalyse
(backend/core/pre_analysis.py) läuft diesen Schritt asynchron — CLI und GUI
erhalten dieselben Felder (PreAnalysisResult.muq_mulan_embedding/-witness).

Hinweis: Der Text-Turm (x_clip/xlm-roberta, Zero-Shot-Tagging) ist bewusst
nicht Teil des ONNX-Exports — Audio-Embeddings decken Ähnlichkeits-Witness
und spätere Tag-Anker ab; der Text-Pfad ist ein eigener Folgeschritt.
"""

from __future__ import annotations

import gc
import hashlib
import logging
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover
    ort = None  # type: ignore[assignment]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ONNX_PATH = _PROJECT_ROOT / "models" / "muq_mulan" / "muq_mulan.onnx"
_REF_DIR = _PROJECT_ROOT / "corpus" / "vinyl" / "clean"
_REF_CACHE = _PROJECT_ROOT / "models" / "muq_mulan" / "muq_mulan_ref_embeddings.npz"
_SR = 24000
_CLIP_S = 10.0
_EMBED_DIM = 768
_MEMORY_GB = 1.5

_state_lock = threading.Lock()
_session: Any = None
_session_attempted: bool = False
_ref_embeddings: np.ndarray | None = None
_ref_attempted: bool = False


def is_available() -> bool:
    """True, wenn die exportierte ONNX-Datei vorhanden ist."""
    return _ONNX_PATH.is_file()


def _get_session() -> Any | None:
    """Lazy-Singleton: ORT-Session (CPU-first, deterministisch; ROCm nur Opt-in).

    Messung 2026-09-11 (RX 7900 XTX, ORT 1.22.2): Der ROCMExecutionProvider
    liefert für diesen Graphen NICHT-deterministische Ausgaben (max_diff
    0.15–1.66 zwischen identischen Läufen, auch mit session.deterministic=1) —
    das verletzt §G5 (GEBOTE.md) (Bit-Determinismus). CPUExecutionProvider ist
    bit-identisch (max_diff=0.0) und mit ~0.75 s pro 10-s-Clip schnell genug
    (Schritt läuft ohnehin asynchron mit 240-s-Budget). ROCm nur per
    AURIK_MULAN_GPU=1 (für spätere ORT-Versionen mit deterministischen
    ROCm-Kerneln).
    """
    global _session, _session_attempted
    if _session is not None:
        return _session
    if ort is None:  # pragma: no cover
        logger.warning("MuQ-MuLan: onnxruntime nicht verfügbar — Witness übersprungen (§V6 (copilot-instructions.md))")
        return None
    with _state_lock:
        if _session is not None:
            return _session
        if _session_attempted:
            return None
        _session_attempted = True
        if not _ONNX_PATH.is_file():
            logger.warning(
                "MuQ-MuLan-ONNX fehlt (%s) — Witness übersprungen (§V6 (copilot-instructions.md)). Ausgabe: scripts/Ausgabe_muq_mulan_onnx.py",
                _ONNX_PATH,
            )
            return None
        try:
            from backend.core.ml_memory_budget import try_allocate as _budget_allocate
            from backend.core.plugin_lifecycle_manager import register_plugin as _reg_plm

            if not _budget_allocate("MuQ-MuLan-ONNX", _MEMORY_GB):
                logger.warning(
                    "MuQ-MuLan: ML-Speicherbudget verweigert — Witness übersprungen (§V6 (copilot-instructions.md))"
                )
                return None
            _reg_plm("MuQ-MuLan", size_gb=_MEMORY_GB, unload_fn=_unload_muq_mulan)
            if os.environ.get("AURIK_MULAN_GPU", "0") == "1":
                logger.warning(
                    "MuQ-MuLan: AURIK_MULAN_GPU=1 — ROCm-Pfad aktiv (bekannt NICHT-deterministisch, "
                    "max_diff 0.15–1.66 gemessen; §G5-Warnung)"
                )
                _providers = ["ROCMExecutionProvider", "CPUExecutionProvider"]
            else:
                _providers = ["CPUExecutionProvider"]
            _sess = ort.InferenceSession(str(_ONNX_PATH), providers=_providers)
            _active = _sess.get_providers()
            _session = _sess
            logger.info(
                "MuQ-MuLan-Plugin: %s geladen (provider=%s)",
                _ONNX_PATH.name,
                _active[0] if _active else "unbekannt",
            )
        except Exception as _exc:
            logger.warning(
                "MuQ-MuLan-ONNX konnte nicht geladen werden (%s) — Witness übersprungen (§V6 (copilot-instructions.md))",
                _exc,
            )
            _session = None
        return _session


def _unload_muq_mulan() -> None:
    global _session
    with _state_lock:
        _session = None
    gc.collect()
    try:
        from backend.core.ml_memory_budget import release as _budget_release

        _budget_release("MuQ-MuLan-ONNX")
    except Exception as _exc:  # pragma: no cover
        logger.debug("MuQ-MuLan Grenze-Release fehlgeschlagen (unkritisch): %s", _exc)


def _to_mono_24k(audio: Any, sr: int) -> np.ndarray:
    """Deterministisch: Layout-sicher mono, 24 kHz, zentriertes 10-s-Fenster."""
    arr = np.asarray(audio, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim == 2:
        # Kanonisch (N, C) nach §11 Spec 02; (C, N) ebenfalls bedienen.
        if arr.shape[0] <= 2 and arr.shape[0] < arr.shape[1]:
            mono = arr.mean(axis=0)
        else:
            mono = arr.mean(axis=1)
    else:
        mono = arr
    if int(sr) != _SR:
        from scipy.signal import resample_poly

        gcd = int(np.gcd(int(sr), _SR))
        mono = resample_poly(mono.astype(np.float64), _SR // gcd, int(sr) // gcd).astype(np.float32)
    n_clip = int(_CLIP_S * _SR)
    if len(mono) > n_clip:
        start = (len(mono) - n_clip) // 2
        mono = mono[start : start + n_clip]
    elif len(mono) < n_clip:
        mono = np.pad(mono, (0, n_clip - len(mono)), mode="constant")
    _clip: np.ndarray = np.asarray(mono, dtype=np.float32)
    return _clip


def extract_muq_mulan_embedding(audio: Any, sr: int) -> np.ndarray | None:
    """Audio-Embedding (768-d float32) via ONNX-GPU; None bei fehlendem Modell."""
    sess = _get_session()
    if sess is None:
        return None
    try:
        clip = _to_mono_24k(audio, sr)
        out = sess.run(None, {"waveform": clip.reshape(1, -1)})[0]
        emb = np.asarray(out, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(emb))
        if norm > 1e-12:
            emb = emb / norm
        _emb_ret: np.ndarray = emb.astype(np.float32)
        return _emb_ret
    except Exception as _exc:
        logger.warning(
            "MuQ-MuLan-Extraktion fehlgeschlagen (%s) — Witness übersprungen (§V6 (copilot-instructions.md))", _exc
        )
        return None


def _ref_file_list_hash() -> str:
    if not _REF_DIR.is_dir():
        return "missing"
    files = sorted(p.name for p in _REF_DIR.glob("*.wav"))
    return hashlib.sha256(",".join(files).encode("utf-8")).hexdigest()[:16]


def _get_ref_embeddings() -> np.ndarray | None:
    """Clean-Referenz-Einbettungen (einmalig, deterministisch gecacht)."""
    global _ref_embeddings, _ref_attempted
    if _ref_embeddings is not None:
        return _ref_embeddings
    with _state_lock:
        if _ref_embeddings is not None:
            return _ref_embeddings
        if _ref_attempted:
            return None
        _ref_attempted = True
        tag = _ref_file_list_hash()
        if tag != "missing" and _REF_CACHE.is_file():
            try:
                cache = np.load(str(_REF_CACHE), allow_pickle=False)
                if str(cache["ref_hash"]) == tag:
                    _ref_embeddings = cache["embeddings"].astype(np.float32)
                    return _ref_embeddings
            except Exception as _exc:
                logger.warning("MuQ-MuLan-Referenz-Zwischenspeicher unlesbar (%s) — wird neu berechnet", _exc)
        if tag == "missing":
            logger.warning(
                "MuQ-MuLan-Referenzverzeichnis %s fehlt — Witness übersprungen (§V6 (copilot-instructions.md))",
                _REF_DIR,
            )
            return None
        try:
            from backend.file_import import load_audio_file

            emb_list: list[np.ndarray] = []
            for path in sorted(_REF_DIR.glob("*.wav")):
                _ld = load_audio_file(str(path), do_carrier_analysis=False)
                if not isinstance(_ld, dict) or _ld.get("audio") is None or _ld.get("sr") is None:
                    logger.warning("MuQ-MuLan-Referenz: nicht lesbar %s", path.name)
                    continue
                data = np.asarray(_ld["audio"], dtype=np.float32)
                if data.ndim == 2 and data.shape[0] < data.shape[1]:
                    data = data.T  # samples-first wie bisher sf.read
                file_sr = int(_ld["sr"])
                emb = extract_muq_mulan_embedding(data, file_sr)
                if emb is None:
                    logger.warning("MuQ-MuLan-Referenz-Einbettung fehlgeschlagen für %s", path.name)
                    continue
                emb_list.append(emb)
            if not emb_list:
                return None
            _ref_embeddings = np.stack(emb_list, axis=0).astype(np.float32)
            _REF_CACHE.parent.mkdir(parents=True, exist_ok=True)
            np.savez(str(_REF_CACHE), ref_hash=tag, embeddings=_ref_embeddings)
            logger.info("MuQ-MuLan-Referenz-Einbettungen berechnet: %d Clean-Referenzen", len(emb_list))
            return _ref_embeddings
        except Exception as _exc:
            logger.warning(
                "MuQ-MuLan-Referenz-Einbettungen fehlgeschlagen (%s) — Witness übersprungen (§V6 (copilot-instructions.md))",
                _exc,
            )
            return None


def _sim_to_witness(sim: float) -> float:
    """Deterministische Abbildung Kosinus-Ähnlichkeit → Witness 0–100 (linear, saturierend)."""
    return float(np.clip(50.0 + 100.0 * (sim - 0.5), 0.0, 100.0))


def estimate_muq_mulan_witness(audio: Any, sr: int) -> float | None:
    """Qualitäts-/Stil-Witness 0–100: Kosinus zum Clean-Referenz-Zentroid."""
    emb = extract_muq_mulan_embedding(audio, sr)
    refs = _get_ref_embeddings()
    if emb is None or refs is None:
        return None
    centroid = refs.mean(axis=0).astype(np.float64)
    denom = float(np.linalg.norm(centroid)) * float(np.linalg.norm(emb))
    if denom < 1e-12:
        return None
    sim = float(np.dot(centroid, emb.astype(np.float64)) / denom)
    return round(_sim_to_witness(sim), 1)
