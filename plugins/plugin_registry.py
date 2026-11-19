"""plugins/plugin_registry.py — Lazy Plugin-Registry für Aurik 10.0.0.

Alle Plugins werden NUR bei erstem Zugriff importiert (kein Eager-Loading
beim Start). Damit werden beim Programmstart 0 MB ML-Modell-RAM belegt.
Jeder Zugriff über PLUGIN_REGISTRY["..."] oder get_plugin("...") löst
genau einmal den Import aus; danach ist das Modul Python-intern gecacht.
"""

from __future__ import annotations

import importlib
import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

_PLUGIN_DIR = os.path.dirname(__file__)
_KNOWN: list[str] = [
    fname[:-3] for fname in os.listdir(_PLUGIN_DIR) if fname.endswith("_plugin.py") and not fname.startswith("_")
]
_cache: dict[str, Any] = {}
_lock = threading.Lock()


class _LazyPluginRegistry:
    """Proxy-Mapping das Plugin-Module lazy importiert (`importlib.import_module`)."""

    def __getitem__(self, key: str) -> Any:
        if key not in _cache:
            with _lock:
                if key not in _cache:
                    _cache[key] = importlib.import_module(f"plugins.{key}")
                    logger.debug("PluginRegistry: '%s' lazy geladen.", key)
        return _cache[key]

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in _KNOWN

    def keys(self) -> list[str]:
        return list(_KNOWN)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except (ImportError, ModuleNotFoundError):
            return default


PLUGIN_REGISTRY = _LazyPluginRegistry()


def get_plugin(name: str) -> Any:
    """Gibt das Plugin-Modul für `name` zurück (lazy import, thread-sicher).

    Args:
        name: Plugin-Modul-Name ohne '.py' (z. B. 'bs_roformer_plugin').

    Returns:
        Importiertes Modul-Objekt.

    Raises:
        ImportError: Falls das Modul nicht geladen werden kann.
        KeyError:    Falls `name` im Plugin-Verzeichnis nicht existiert.
    """
    if name not in PLUGIN_REGISTRY:
        raise KeyError(f"Plugin '{name}' nicht gefunden in {_PLUGIN_DIR}")
    return PLUGIN_REGISTRY[name]


# Beispiel-Nutzung (lazy — kein Import beim Modulstart):
# plugin_mod = get_plugin("bs_roformer_plugin")
# result = plugin_mod.separate(audio, sr)
