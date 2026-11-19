# Aurik 10.0.0 — Kontinuierliches Qualitäts-Monitoring & Deep-Analysis

## Überblick

Das integrierte Monitoring-Ökosystem überprüft während der Restaurierung kontinuierlich auf:

1. **Musikalische Ziele** — alle 14 Goals pro Phase
2. **HPI-Score** — Holistic Perceptual Index
3. **Artefakt-Freiheit** — automatische Artifact-Detection
4. **Pegelexplosionen** — spezialisierte Detektion in kritischen Regionen (Fade-Out, Intro, Stille)
5. **Rauschboden** — LUFS-Monitoring
6. **Regelabweichungen** — Verstoß gegen §0 Klangwahrheit, Primum non nocere

## Komponenten

### 1. Frontend mit Analysis (`frontend_with_analysis.py`)

Startet GUI + Tiefenanalyze parallel.

```bash
# Standard: GUI + Analyzer
python scripts/frontend_with_analysis.py --audio test_audio/*.mp3

# Nur Analyzer (kein GUI)
python scripts/frontend_with_analysis.py --audio test_audio/*.mp3 --no-gui
```

### 2. Kontinuierliche Tiefenanalyse (`continuous_deep_analysis.py`)

Phase-weise Quality Checkpoints mit Musical-Goals-Tracking.

```bash
# Standalone-Lauf
python scripts/continuous_deep_analysis.py --audio test_audio/*.mp3 --mode restoration --realtime
```

**Output**: JSON-Datei in `analysis_results/` mit Checkpoints pro Phase.

### 3. Pegelexplosion-Detektor (`pegelexplosion_detector.py`)

Spezialisiert auf Level Spikes in:

- Fade-Out-Regionen (letzte 3s)
- Intro-Bereichen (erste 1s)
- Stille-Zonen (< -36 dBFS)

```bash
# Analysiere Audio auf Spikes
python scripts/pegelexplosion_detector.py --audio output_audio/restored.wav --verbose
```

**Automatische Diagnose** bekannter Bug-Muster:

- `emotional_arc_incorrect_gate` — `correct_arc()` mit -42 statt -36 dBFS
- `quiet_zone_makeup_gain` — MDEM ohne Stille-Gate
- `makeup_gain_after_hpf` — Energieverlust-Kompensation nach Hochpass

**Suggested Fixes**: Code-Snippets direkt zum Copy-Paste.

### 4. Echtzeit-Pegelexplosion-Monitor (`pegelexplosion_monitor.py`)

Läuft im Hintergrund, überwacht `output_audio/` Verzeichnis kontinuierlich.

```bash
# Starte Monitor
python scripts/pegelexplosion_monitor.py --watch-dir output_audio --interval 2
```

Bei neuen Export-Dateien:

- Automatische Analyse
- Warnung bei Problemen
- Suggested Fixes in Log

Severity-Level:

- `none` — ✓ OK
- `minor` — ⚠ Beachten
- `moderate` — ⚠⚠ Prüfen
- `critical` — 🚨 Sofort beheben

### 5. Master-Orchestrator (`orchestrate_quality_monitoring.py`) ⭐

**Empfohlen**: Starten Sie damit — koordiniert alle 4 Komponenten mit einheitlichem Dashboard.

```bash
# Vollständiges Ökosystem: GUI + Analyzer + Pegelexplosion-Monitor
python scripts/orchestrate_quality_monitoring.py --audio test_audio/*.mp3

# Headless-Modus (nur CLI, keine GUI)
python scripts/orchestrate_quality_monitoring.py --audio test_audio/*.mp3 --headless

# Verbose (für Debug)
python scripts/orchestrate_quality_monitoring.py --audio test_audio/*.mp3 --verbose
```

## Workflow

### Szenario 1: Interaktive Analyse mit GUI

```bash
python scripts/orchestrate_quality_monitoring.py --audio my_song.mp3
```

1. GUI öffnet sich
2. Analyzer lädt Audio und startet Restaurierung
3. Pegelexplosion-Monitor überwacht Exports
4. **Dashboard** zeigt gleichzeitig:
   - Phase-weise Fortschritt
   - Musical Goals Trend
   - Pegelexplosion-Alerts (Echtzeit)
   - Suggested Fixes bei Problemen

### Szenario 2: Batch-Analyse ohne GUI

```bash
python scripts/orchestrate_quality_monitoring.py --audio song1.mp3 --headless
python scripts/orchestrate_quality_monitoring.py --audio song2.mp3 --headless
```

Läuft vollständig in CLI. Ergebnisse in:

- `analysis_results/analysis_*.json`
- `temp_repro/monitoring_status.json`
- `orchestrator_runtime.log`
- `pegelexplosion_monitor.log`

### Szenario 3: Deep-Dive für einzelne Phase

```bash
# Analysiere nur eine Phase auf Anomalien
python scripts/pegelexplosion_detector.py \
  --audio output_audio/phase_42_vocal_export.wav \
  --phase phase_42_vocal_enhancement \
  --verbose
```

## Automatische Fehlerbehandlung

Wenn kritische Anomalien erkannt werden:

1. **Logger** gibt Warnung mit Details
2. **Suggested Fixes** werden angezeigt (Code-Snippets)
3. **Severity-Flag** in JSON gespeichert
4. Bei `critical`: Prozess kann auf "Recovery Mode" umschalten

## Output-Dateien

### `analysis_results/analysis_<song>_<mode>_<timestamp>.json`

```json
{
  "wall_time_s": 1245.3,
  "audio_path": "test_audio/Test_Track.mp3",
  "mode": "restoration",
  "checkpoints": [
    {
      "phase_id": "phase_01_click_removal",
      "musical_goals": {"natuerlichkeit": 0.82, "authentizitaet": 0.79, ...},
      "hpi_score": 0.621,
      "anomalies": []
    },
    ...
  ],
  "anomalies": ["HPI drop after phase_20", "Pegelexplosion in fade-out"],
  "summary": {
    "quality_status": "GOOD",
    "p1_avg_score": 0.851,
    "total_anomalies": 2
  }
}
```

### `temp_repro/monitoring_status.json`

```json
{
  "timestamp": 1714041234.5,
  "processes": {
    "gui": {"pid": 12345, "running": true},
    "analyzer": {"pid": 12346, "running": true},
    "pegelexplosion_monitor": {"pid": 12347, "running": true}
  }
}
```

### Logs

- `orchestrator_runtime.log` — Master-Prozess
- `analysis_runtime.log` — Tiefenanalyse
- `pegelexplosion_monitor.log` — Echtzeit-Überwachung
- `frontend_analysis_runtime.log` — GUI + Analyzer Integration

## Konfiguration

Alle Scripts sind konfigurierbar via CLI-Parameter:

```bash
# Analyzer mit Custom Output-Dir
python scripts/continuous_deep_analysis.py \
  --audio my_song.mp3 \
  --output-dir my_analysis_dir

# Pegelexplosion-Monitor mit schnellerem Intervall
python scripts/pegelexplosion_monitor.py \
  --watch-dir output_audio \
  --interval 1  # 1 statt 2 Sekunden
```

## Troubleshooting

### Monitor startet nicht

```bash
# Prüfe venv
source .venv_aurik/bin/activate
python -c "import scipy.signal; import numpy"
```

### Keine Pegelexplosionen erkannt?

- Prüfe LUFS-Schwelle in `pegelexplosion_detector.py` (`_QUIET_ZONE_LUFS = -36.0`)
- Erhöhe `_SPIKE_THRESHOLD_DB` für empfindlichere Detektion

### Loggers zu verbose?

```bash
# Nur INFO, keine DEBUG
python scripts/orchestrate_quality_monitoring.py --audio song.mp3
```

## Best Practices

1. **Starten Sie mit `orchestrate_quality_monitoring.py`** — alles andere ist für Advanced Use
2. **Beobachten Sie die Logs** — Anomalien werden mit Suggested Fixes angezeigt
3. **Beachten Sie Pegelexplosion-Alerts** — sind wichtige Indikatoren für Bugs
4. **Nutzen Sie `--headless` für Batch** — für automatisierte Test-Runs

---

**Version**: Aurik 10.0.0  
**Stand**: Mai 2026  
**Autor**: Aurik Engineering Team
