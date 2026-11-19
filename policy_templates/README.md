# Community-Policy-Templates für Aurik 6.0

Hier findest du Best-Practice-Policy-Blueprints für typische Restaurationsaufgaben. Diese YAML-Dateien können direkt per CLI-Option --policy oder --policy-template geladen werden.

## Beispiele

### standard.yaml

```yaml
use_inpainter: true
remove_hum: true
vocal_enhance: false
```

### minimal.yaml

```yaml
use_inpainter: false
remove_hum: false
vocal_enhance: false
```

### vocal_focus.yaml

```yaml
use_inpainter: true
remove_hum: false
vocal_enhance: true
```

## Nutzung

```bash
# python -m aurik6.orchestrator_and_cli input.wav output.wav --policy-template standard (sobald migriert)
# python -m aurik6.orchestrator_and_cli input.wav output.wav --policy policy_templates/vocal_focus.yaml (sobald migriert)
```

## Eigene Templates

Lege eigene YAML-Dateien im Verzeichnis `policy_templates/` ab. Jede Datei entspricht einer Policy-Vorlage und kann beliebig erweitert werden (z.B. für Genre, Instrument, User-Profile).

## Hinweise

- Alle Änderungen werden im Audit-Log dokumentiert.

## Automatisierter Review-Workflow & Qualitätsanforderungen

Jedes neue oder geänderte Policy-Template wird beim Testlauf automatisch geprüft:

- **Syntax-Check:** Die YAML-Datei muss fehlerfrei sein.
- **Struktur-Check:** Die Datei muss ein Dictionary (Mapping) sein.
- **Empfohlene Felder:** Die Felder `use_inpainter` und `remove_hum` sollten enthalten sein (siehe Beispiele oben).
- **Review-Report:** Bei Fehlern/Abweichungen wird automatisch eine Datei `review_report.txt` im Verzeichnis erzeugt, die alle Probleme auflistet.

### So prüfst du dein Template vor dem Pull Request

Führe folgenden Befehl aus, um alle Templates zu validieren und einen Review-Report zu erhalten:

```bash
# pytest -v aurik6/testing/test_policy_templates.py (sobald migriert)
```

**Nur fehlerfreie Templates werden akzeptiert!**

Weitere Hinweise und Beispiele findest du in den bestehenden YAML-Dateien.

- Templates können als Startpunkt für User-Feedback und Session-Memory dienen.
- Für Community-Beiträge: Pull-Request mit neuem Template und kurzer Beschreibung.
