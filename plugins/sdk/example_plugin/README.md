# Example Plugin

Minimalbeispiel für das Aurik Plugin-SDK (§15.6).

## Schnellstart

```bash
# Plugin kopieren und umbenennen
cp -r plugins/sdk/example_plugin plugins/mein_plugin
cd plugins/mein_plugin

# Dateien umbenennen
mv example_plugin.py mein_plugin.py

# manifest.json bearbeiten
# mein_plugin.py bearbeiten

# Tests ausführen
pytest test_example.py -v
```

## Struktur

```
example_plugin/
├── README.md           ← Diese Datei
├── manifest.json       ← Plugin-Metadaten
├── example_plugin.py   ← Plugin-Implementierung
└── test_example.py     ← Tests
```

## Plugin validieren

```bash
python scripts/validate_plugin.py plugins/sdk/example_plugin
```
