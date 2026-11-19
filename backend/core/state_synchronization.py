"""
core/state_synchronization.py
Automatische State-Propagation und Synchronisation für alle AURIK-Module.
"""

import threading
from collections.abc import Callable
from typing import Any


class StateSynchronizationManager:
    _instance = None
    _lock = threading.RLock()

    def __new__(cls):
        if cls._instance is None:  # First check without lock (fast path)
            with cls._lock:
                if cls._instance is None:  # Second check with lock (DCL §3.x)
                    cls._instance = super().__new__(cls)
                    cls._instance._init()
        return cls._instance

    def _init(self):
        self.module_states: dict[str, dict[str, Any]] = {}
        self.listeners: list[Callable[[str, dict[str, Any]], None]] = []

    def register_module(self, module_name: str, initial_state: dict[str, Any]):
        with self._lock:
            self.module_states[module_name] = initial_state.copy()
            self._notify(module_name)

    def update_state(self, module_name: str, state_update: dict[str, Any]):
        with self._lock:
            if module_name not in self.module_states:
                self.module_states[module_name] = {}
            self.module_states[module_name].update(state_update)
            self._notify(module_name)

    def get_state(self, module_name: str) -> dict[str, Any]:
        with self._lock:
            return self.module_states.get(module_name, {}).copy()

    def synchronize(self):
        with self._lock:
            # Propagate all states to listeners
            for module, state in self.module_states.items():
                self._notify(module)

    def add_listener(self, callback: Callable[[str, dict[str, Any]], None]):
        with self._lock:
            self.listeners.append(callback)

    def _notify(self, module_name: str):
        state = self.module_states[module_name].copy()
        for listener in self.listeners:
            listener(module_name, state)

    def resolve_conflicts(self, resolution_strategy: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]):
        with self._lock:
            # Example: resolve conflicts between modules
            for module, state in self.module_states.items():
                # Custom conflict resolution logic can be applied here
                self.module_states[module] = resolution_strategy(state, state)


# Singleton instance
state_sync_manager = StateSynchronizationManager()
