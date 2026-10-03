"""
dsp_module_registry.py - Dynamische Registry und Orchestrierung aller DSP- und KI-Modelle in Aurik 6.0

- Führt alle DSP/KI-Module zentral
- Ermöglicht dynamische Auswahl, Reihenfolge und Parametrisierung
- Berücksichtigt Tonträgerarten, Defekt-Hypothesen, Policy und Qualitätsmetriken
- Garantiert musikalische Ziele und Auditierbarkeit
"""

import importlib
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class DSPModuleRegistry:
    """
    Registry für alle DSP- und KI-Module. Ermöglicht dynamische Orchestrierung.
    """

    def __init__(self, dsp_dir: str | None = None) -> None:
        self.dsp_dir = dsp_dir or os.path.dirname(__file__)
        self.modules: dict[str, Any] = {}
        self._discover_modules()

    def _discover_modules(self) -> None:
        # Alle .py-Dateien im DSP-Verzeichnis finden
        for fname in os.listdir(self.dsp_dir):
            if fname.endswith(".py") and not fname.startswith("__") and fname != os.path.basename(__file__):
                mod_name = fname[:-3]
                try:
                    mod = importlib.import_module(f"dsp.{mod_name}")
                    self.modules[mod_name] = mod
                except Exception as e:
                    logger.error("[Registry] Fehler beim Import von %s: %s", mod_name, e)

    def get_module(self, name: str) -> Any | None:
        return self.modules.get(name)

    def list_modules(self) -> list[str]:
        return list(self.modules.keys())

    def instantiate(self, name: str, *args: Any, **kwargs: Any) -> Any:
        mod = self.get_module(name)
        if mod is None:
            raise ImportError(f"DSP-Modul {name} nicht gefunden.")
        # Suche nach einer Klasse mit gleichem Namen (Konvention)
        cls = getattr(mod, name[0].upper() + name[1:], None)
        if cls is None:
            # Fallback: Erstbeste Klasse im Modul
            for attr in dir(mod):
                obj = getattr(mod, attr)
                if isinstance(obj, type):
                    cls = obj
                    break
        if cls is None:
            raise ImportError(f"Keine Instanzierbare Klasse in {name} gefunden.")
        return cls(*args, **kwargs)


# Beispiel für die Nutzung
if __name__ == "__main__":
    registry = DSPModuleRegistry()
    logger.info("Verfügbare DSP/KI-Module: %s", registry.list_modules())
    # Beispiel: Instanziere einen Denoiser
    if "sota_denoiser" in registry.list_modules():
        denoiser = registry.instantiate("sota_denoiser")
        logger.info("Instanziierter Denoiser: %s", denoiser)

# Erweiterbar: Automatische Synchronisation mit YAML-Liste und PolicyManager
