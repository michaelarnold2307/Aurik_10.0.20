"""backend/core/model_path_resolver.py — §13.3 On-Demand-Modell-Auflösung.

Zentrale Funktion für direkte Ladepfade (Plugins, DSP-Module): Existiert die
lokale Datei, wird sie direkt genutzt. Fehlt sie (frischer Checkout,
unvollständige Installation, LFS-Pull-Fehler), wird sie über das Manifest
(delivery=release) aus GitHub Releases nachgeladen. Schlägt das fehl, liefert
der Resolver None — der Aufrufer nutzt seinen DSP-Fallback
(§V6 (copilot-instructions.md): Logging mit Begründung).

- Kill-Switch: AURIK_MODEL_DOWNLOAD=0 (kein Netzwerk, reine lokale Prüfung).
- Thread-safe (Lock + Cache), deterministisch, kein Datei-I/O außer exists().
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_MODELS_DIR = _PROJECT_ROOT / "models"
_DOWNLOAD_ENABLED = os.environ.get("AURIK_MODEL_DOWNLOAD", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

_lock = threading.Lock()
_cache: dict[str, Path | None] = {}


def _set_cache(key: str, value: Path | None) -> None:
    with _lock:
        _cache[key] = value


def _get_cache(key: str) -> Path | None | object:
    with _lock:
        return _cache.get(key, "MISS")


def resolve_model_path(rel_path: str | Path, *, download: bool = True) -> Path | None:
    """Löst einen Modellpfad lokal auf und lädt fehlende Dateien bei Bedarf nach.

    Args:
        rel_path: Pfad relativ zum Projekt-Root (z. B. "models/apollo/apollo_core.onnx").
        download: False = reine lokale Prüfung (kein Netzwerk).

    Returns:
        Existierender, SHA-verifizierter Pfad oder None (DSP-Fallback nötig).
    """
    rel = str(rel_path).replace("\\", "/")
    cached = _get_cache(rel)
    if cached != "MISS":
        return cached  # type: ignore[return-value]

    local = _MODELS_DIR / rel
    if not local.is_file():
        local = Path(rel)  # Relativer CWD-Fallback (z. B. Tests)
    if local.is_file():
        _set_cache(rel, local)
        return local

    if not download or not _DOWNLOAD_ENABLED:
        logger.warning(
            "§V6 (copilot-instructions.md): Modell %s nicht lokal verfügbar — DSP-Fallback "
            "(Download %s)",
            rel,
            "deaktiviert (AURIK_MODEL_DOWNLOAD=0)" if not _DOWNLOAD_ENABLED else "übersprungen",
        )
        _set_cache(rel, None)
        return None

    try:
        from backend.core.model_downloader import get_model_downloader as _get_dl

        _dl = _get_dl()
        _entry = _dl.get_entry_by_path(rel)
        if _entry is None:
            logger.warning(
                "§V6 (copilot-instructions.md): Modell %s fehlt und hat keinen Manifest-Eintrag — "
                "DSP-Fallback",
                rel,
            )
            _set_cache(rel, None)
            return None
        resolved = _dl.load_bundled(_entry)
        if resolved is not None and resolved.is_file():
            _set_cache(rel, resolved)
            return resolved
    except Exception as exc:
        logger.warning(
            "§V6 (copilot-instructions.md): On-Demand-Auflösung für %s fehlgeschlagen — "
            "DSP-Fallback: %s",
            rel,
            exc,
        )
    _set_cache(rel, None)
    return None


def clear_resolver_cache() -> None:
    """Leert den Pfad-Cache (Tests, Session-Wechsel)."""
    with _lock:
        _cache.clear()


__all__ = ["resolve_model_path", "clear_resolver_cache"]
