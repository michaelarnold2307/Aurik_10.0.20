# Aurik Plugin SDK

> §15.6: Developer Guide für Aurik-Plugins.

## Übersicht

Das Aurik Plugin-SDK erlaubt Drittentwicklern, eigene Audio-Verarbeitungsmodule
zu schreiben und in die Aurik-Pipeline zu integrieren.

## Schnellstart

```bash
# 1. Beispiel-Plugin kopieren
cp -r plugins/sdk/example_plugin plugins/mein_plugin
cd plugins/mein_plugin

# 2. Dateien umbenennen
mv example_plugin.py mein_plugin.py

# 3. manifest.json bearbeiten
# → Name, Version, Beschreibung anpassen

# 4. Plugin implementieren
# → AurikPlugin.process_audio() überschreiben

# 5. Tests schreiben
# → test_example.py anpassen

# 6. Validieren
python scripts/validate_plugin.py plugins/mein_plugin

# 7. Testen
pytest plugins/mein_plugin/test_example.py -v
```

## Plugin-Struktur

```
mein_plugin/
├── __init__.py          ← Package-Initialisierung
├── manifest.json        ← Metadaten (Name, Version, ...)
├── mein_plugin.py       ← Plugin-Implementierung
├── test_mein_plugin.py  ← Tests
└── README.md            ← Dokumentation
```

## AurikPlugin-ABC

```python
from plugins.sdk.aurik_plugin_base import AurikPlugin, PluginManifest


class MeinPlugin(AurikPlugin):
    manifest = PluginManifest(
        name="mein-plugin",
        version="1.0.0",
        description="Meine Audio-Verarbeitung",
        author="Max Mustermann",
        min_aurik_version="10.0.0",
    )

    def process_audio(self, audio, sr=48000, **kwargs):
        # Audio-Verarbeitung hier
        return audio

    def validate(self):
        ok, msg = super().validate()
        if not ok:
            return ok, msg
        # Eigene Validierung hinzufügen
        return True, "OK"
```

## API-Stabilitätsgarantie

- **SemVer**: MAJOR.MINOR.PATCH
- **Deprecation-Window**: 2 Major-Versionen
- **Bridge-API**: Stabil seit Aurik 10.0.0.x
- **Breaking Changes**: Nur mit MAJOR-Bump, angekündigt im Changelog

## Testing

```python
from plugins.sdk.testing_fixtures import (
    VirtualAurikPipeline,
    make_test_audio,
    make_noisy_audio,
)


def test_mein_plugin():
    plugin = MeinPlugin()
    pipeline = VirtualAurikPipeline(material="vinyl")
    audio = make_noisy_audio(duration_s=3.0)
    result = pipeline.run_plugin(plugin, audio)
    assert result.success
```

## Distribution

Plugins können als:

1. **Lokales Verzeichnis**: `plugins/mein_plugin/`
2. **Python-Paket**: `pip install aurik-plugin-mein-plugin`
3. **Git-Repository**: `git clone ... plugins/mein_plugin`

## Referenz

- [AurikPlugin API](plugins/sdk/aurik_plugin_base.py)
- [Testing Fixtures](plugins/sdk/testing_fixtures.py)
- [Example Plugin](plugins/sdk/example_plugin/)
- [Plugin Validator](scripts/validate_plugin.py)
