"""
PluginArchitecture: Offene, adaptive Architektur für neue Defekte und künstlerisch-experimentelle Anforderungen.
"""

from collections.abc import Callable
from typing import Any


class PluginManager:
    def __init__(self):
        self.plugins: dict[str, Callable[[Any], Any]] = {}
        self.plugin_meta: dict[str, dict] = {}

    def register_plugin(self, name: str, process_fn: Callable[[Any], Any], meta: dict | None = None):
        self.plugins[name] = process_fn
        self.plugin_meta[name] = meta or {}

    def process(self, name: str, data: Any) -> Any:
        if name in self.plugins:
            return self.plugins[name](data)
        raise ValueError(f"Plugin {name} not registered.")

    def list_plugins(self) -> list[str]:
        return list(self.plugins.keys())

    def get_plugin_meta(self, name: str) -> dict:
        return self.plugin_meta.get(name, {})
