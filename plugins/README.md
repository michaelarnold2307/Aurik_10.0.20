# Aurik 6.0 – Plug-in- und Erweiterungsarchitektur

Dieses Verzeichnis dient als zentrale Schnittstelle für Drittanbieter-Module, neue DSPs und KI-Modelle.

## Integration neuer Plug-ins

- Jedes Plug-in wird als eigenes Python-Modul im Ordner `plugins/` abgelegt.
- Plug-ins müssen eine standardisierte Schnittstelle bereitstellen (z.B. `process(audio: np.ndarray, sr: int) -> np.ndarray`).
- Plug-ins können DSP-Algorithmen, KI-Modelle, Analyse-Tools oder Visualisierungen enthalten.
- Die Plug-in-Registrierung erfolgt über eine zentrale Plug-in-Registry (siehe Beispiel unten).

## Beispiel: Plug-in-Registry

```python
# plugins/plugin_registry.py
import importlib
import os

PLUGIN_REGISTRY = {}

for fname in os.listdir(os.path.dirname(__file__)):
    if fname.endswith("_plugin.py"):
        modulename = fname[:-3]
        module = importlib.import_module(f"plugins.{modulename}")
        PLUGIN_REGISTRY[modulename] = module

# Nutzung:
# plugin = PLUGIN_REGISTRY["waveunet_plugin"]
# result = plugin.process(audio, sr)
```

## Plug-in-Entwicklung

- Plug-ins sollten ausführlich dokumentiert und getestet sein.
- Für KI-Modelle: Modell- und Gewichtsdateien im Unterordner ablegen.
- Für DSPs: Parameter und Presets als YAML/JSON bereitstellen.

## Erweiterbarkeit

- Die Architektur erlaubt beliebig viele Plug-ins und Erweiterungen.
- Neue Plug-ins können ohne Änderung am Hauptsystem integriert werden.

---

Für Fragen und Beiträge: <aurik-community@aurik.com>
