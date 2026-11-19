"""
backend/core/model_warmup_pool.py — Plugin Warm-Up Pool (§v10.10)
=================================================================

Lädt die 5 häufigsten ML-Modelle im Hintergrund während der Pre-Analyse.
Nutzt ModelChainOrchestrator (v10.9) für Shared-Instance-Management.

Usage:
    from backend.core.model_warmup_pool import start_warmup, await_warmup
    start_warmup()  # Startet Hintergrund-Thread
    # ... Pre-Analyse läuft ...
    await_warmup(timeout=15.0)  # Wartet max 15s auf Warm-Up
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# Modelle die beim Warm-Up vorgeladen werden (nach Häufigkeit priorisiert)
_WARMUP_MODELS: list[str] = [
    "deepfilternet",  # Phase 03, MRN plugins — am häufigsten
    "bw_reconstructor",  # Phase 06, Shellac/Lacquer MRN
    "rmvpe",  # Phase 56 pitch tracking
    "beats",  # Phase 53 semantic audio
    "demucs",  # Stem separation
]

_WARMUP_TIMEOUT: float = 20.0  # Max 20s für Warm-Up


class ModelWarmUpPool:
    """Hintergrund-Thread für ML-Model Pre-Loading."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._loaded: set[str] = set()
        self._failed: set[str] = set()
        self._done = threading.Event()
        self._started = False

    def start(self) -> None:
        """Startet Warm-Up im Hintergrund."""
        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(target=self._warmup_worker, daemon=True)
        self._thread.start()
        logger.info("🔥 Model Warm-Up gestartet — %d Modelle im Hintergrund", len(_WARMUP_MODELS))

    def _warmup_worker(self) -> None:
        """Worker-Thread: lädt Modelle sequentiell."""
        try:
            from backend.core.model_chain_orchestrator import get_model_chain

            _mco = get_model_chain()

            for _model in _WARMUP_MODELS:
                try:
                    _instance = _mco.get(_model)
                    if _instance is not None:
                        self._loaded.add(_model)
                        logger.debug("🔥 Warm-Up: %s geladen", _model)
                    else:
                        self._failed.add(_model)
                except Exception as exc:
                    logger.debug("🔥 Warm-Up: %s fehlgeschlagen: %s", _model, exc)
                    self._failed.add(_model)
        except Exception as exc:
            logger.warning("⚠️ Model Warm-Up abgebrochen: %s", exc)
        finally:
            self._done.set()

    def await_ready(self, timeout: float = _WARMUP_TIMEOUT) -> dict[str, Any]:
        """Wartet auf Abschluss des Warm-Ups.

        Returns:
            Dict mit Status: {'loaded': [...], 'failed': [...], 'ready': bool}
        """
        if not self._started:
            return {"loaded": [], "failed": [], "ready": False}

        _ready = self._done.wait(timeout=timeout)
        _status = {
            "loaded": sorted(self._loaded),
            "failed": sorted(self._failed),
            "ready": _ready,
        }
        logger.info(
            "🔥 Warm-Up: %d/%d Modelle geladen (%.1fs)",
            len(self._loaded),
            len(_WARMUP_MODELS),
            timeout if not _ready else 0,
        )
        return _status

    @property
    def models_ready(self) -> set[str]:
        return self._loaded.copy()


# Singleton
_pool: ModelWarmUpPool | None = None
_pool_lock = threading.Lock()


def start_warmup() -> None:
    """Startet den Model-WarmUp-Pool (non-blocking)."""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ModelWarmUpPool()
    _pool.start()


def await_warmup(timeout: float = _WARMUP_TIMEOUT) -> dict[str, Any]:
    """Wartet auf Warm-Up Abschluss."""
    if _pool is None:
        return {"loaded": [], "failed": [], "ready": False}
    return _pool.await_ready(timeout=timeout)
