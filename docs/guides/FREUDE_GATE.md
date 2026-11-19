# Freude-Gate (Psychoakustisches Release-Gate)

Das Freude-Gate prueft, ob eine Release-Version nicht nur technisch sauber, sondern
fuer anspruchsvolle Hoerer langfristig ueberzeugend ist.

Standardbetrieb ist bewusst **nicht-blockierend**: bei Unterschreitung liefert das Gate
konkrete Nachbesserungsempfehlungen statt Release-Abbruch.

## Ziel

Harte PASS/FAIL-Entscheidung auf Basis von Blindtest-Daten.

## Erwartetes Eingabeformat

JSON-Datei mit Feld `items`:

```json
{
  "items": [
    {
      "item_id": "song_001",
      "mushra": 84.0,
      "enjoyment": 4.5,
      "fatigue": 2.0,
      "artifact_flag": false
    }
  ]
}
```

- `mushra`: 0..100
- `enjoyment`: 1..5 (hoeher = besser)
- `fatigue`: 1..5 (niedriger = besser)
- `artifact_flag`: `true` falls wahrnehmbarer Artefakt auftrat

## Standard-Schwellen (v1)

- `n_items >= 20`
- `mean_mushra >= 80.0`
- `p10_mushra >= 70.0`
- `share_mushra_below_65 <= 0.10`
- `mean_enjoyment >= 4.20`
- `mean_fatigue <= 2.20`
- `artifact_rate <= 0.05`

## Ausfuehrung

```bash
.venv_aurik/bin/python scripts/freude_gate_check.py \
  --input docs/reports/studies/mushra_2026q2/results_panel.json \
  --output reports/freude_gate_report.json

# Optional: harter CI-Enforce-Modus (blocking)
.venv_aurik/bin/python scripts/freude_gate_check.py \
  --input docs/reports/studies/mushra_2026q2/results_panel.json \
  --output reports/freude_gate_report.json \
  --enforce
```

Rueckgabecodes:

- `0`: PASS
- `0`: PASS oder IMPROVE (non-blocking Nachbesserungsmodus)
- `1`: FAIL nur im `--enforce`-Modus
- `2`: Eingabefehler / ungueltiges Dataset

## Hinweise

- Das Gate ist absichtlich streng, aber standardmaessig nicht-blockierend: wenige Ausreisser
  fuehren zu gezielter Nachbesserung statt sofortigem Release-Abbruch.
- Schwellen duerfen nur datenbasiert und versioniert angepasst werden.
