# AURIK v8.0 Persistence Layer

**Audit Logs & Monitoring**

---

## Zweck

Alle Verarbeitungsschritte werden **permanent** aufgezeichnet für:

- Nachvollziehbarkeit (Transparency)
- Compliance mit HIPS-6 (Auditierbarkeit)
- Continuous Learning Datengrundlage
- Forensische Analyse

---

## Module

### `audio_monitor.py`

**PermanentAudio Monitor:** Tracking aller Audio-Transformationen

**Capabilities:**

- **Pre/Post Metriken:** Bark Spectrum (24 Bands), F0, HNR
- **Spectral Features:** Centroid, Bandwidth, Contrast
- **Dynamic Features:** RMS, Peak, Crest Factor, Dynamic Range
- **Harmonic Features:** HNR Mean/Std
- **Quality Metrics:** SNR, Clipping Ratio, THD
- **Per-Module Logs:** Processing Time, Confidence, Quality Gate Status
- **Export Formats:** JSON, YAML, CSV

---

**Integration:**

```python
from backend.persistence.audit_logs.audio_monitor import PermanentAudioMonitor

monitor = PermanentAudioMonitor()

# 1. Capture Baseline
monitor.capture_baseline(audio, sr, file_path="input.wav")

# 2. Track each processing module
monitor.start_module("restoration")
# ... processing happens ...
monitor.end_module(audio_in, audio_out, sr, confidence=0.9, quality_passed=True)

# 3. Capture Final State
monitor.capture_final(audio_final, sr)

# 4. Export Audit Report
monitor.export_audit_report(output_dir="./audits", formats=["json", "yaml", "csv"])
```

---

### `logging_config.py`

**Centralized Logging Configuration**

**Features:**

- Rotating File Handler (5MB, 5 Backups)
- Standardized Format: `[timestamp] LEVEL module: message`
- Log Location: `../logs/aurik_backend.log`
- Error Notification Integration

**Usage:**

```python
from backend.persistence.audit_logs.logging_config import get_logger

logger = get_logger(__name__)
logger.info("Processing started")
logger.warning("Confidence below threshold")
logger.error("Quality gate failed")
```

---

### `error_notifier.py`

**Error Notification System**

Optional email notifications for critical errors.

---

## Audit Report Structure

**JSON Format:**

```json
{
  "file_path": "input.wav",
  "baseline_metrics": {
    "bark_spectrum": [0.12, 0.15, ...],
    "f0_mean": 220.5,
    "hnr_mean": 12.3,
    "rms": 0.08,
    "snr": 25.4
  },
  "module_logs": [
    {
      "module_name": "restoration",
      "pre_metrics": {...},
      "post_metrics": {...},
      "confidence": 0.92,
      "processing_time_ms": 156.3,
      "quality_gate_passed": true
    }
  ],
  "final_metrics": {...},
  "total_processing_time_ms": 487.2,
  "cas_improvement": 0.024,
  "timestamp": "2026-02-07T18:30:45.123456"
}
```

---

## Integration mit Continuous Learning

Die Audit Logs bilden die **Datengrundlage** für `continuous_learning.py`:

1. **Accumulation:** 1000+ Audit Reports werden gesammelt
2. **Analysis:** Success Patterns identifiziert
3. **Learning:** Gewichte optimiert basierend auf Historical Success
4. **Application:** Policy Engine Updates

**Workflow:**

```text
Audio Processing
  → Audio Monitor Tracks
  → Audit Report Exported
  → Accumulation (1000+ Files)
  → Continuous Learning Analyzes
  → Recommendations Generated
  → Policy Engine Updated
  → Next Cycle Improved
```

---

## HIPS-6 Compliance

**HIPS-6 Anforderung:** "Auditierbarkeit"

> Jeder Eingriff muss nachvollziehbar sein.
> Entscheidungen müssen erklärt werden können.

**Implementierung:**

- ✅ Jede Operation wird geloggt
- ✅ Pre/Post Metriken für alle Module
- ✅ Timestamps für Timing Analysis
- ✅ Confidence Scores für alle Entscheidungen
- ✅ Quality Gate Status dokumentiert
- ✅ Export in menschenlesbaren Formaten (YAML, CSV)
- ✅ Unveränderliche Logs (append-only)

---

## Best Practices

1. **Immer Baseline capturen:**

   ```python
   monitor.capture_baseline(audio, sr, file_path="...")
   ```

2. **Jeden Processing-Schritt wrappen:**

   ```python
   monitor.start_module("module_name")
   # ... processing ...
   monitor.end_module(audio_in, audio_out, sr, ...)
   ```

3. **Final State capturen:**

   ```python
   monitor.capture_final(audio_final, sr)
   ```

4. **Reports exportieren:**

   ```python
   monitor.export_audit_report(output_dir="./audits", formats=["json", "yaml"])
   ```

5. **Regelmäßig aggregieren für Learning:**

   ```python
   # Nach 1000+ Dateien
   learning = ContinuousLearningSystem(audit_dir="./audits")
   report = learning.run_learning_cycle(min_files=1000)
   ```

---

## Status: v8.0 (7. Februar 2026)

**Implementiert:**

- ✅ PermanentAudioMonitor (587 LOC)
- ✅ Multi-Format Export (JSON, YAML, CSV)
- ✅ 24-Band Bark Spectrum Analysis
- ✅ F0 & HNR Tracking
- ✅ Per-Module Logging
- ✅ Integration mit Pipeline (Phase 4.5)

**Metriken Tracked:**

- ✅ 14 Kategorieen (Spectral, Dynamic, Harmonic, Quality)
- ✅ Pre/Post für jeden Modul
- ✅ Total Processing Time
- ✅ CAS Improvement

---

## Output Location

**Default Pfade:**

- Logs: `logs/aurik_backend.log`
- Audits: `audits/*.json`, `audits/*.yaml`, `audits/*.csv`
- Learning Reports: `learning_reports/*.json`

**Customizable:**

```python
monitor.export_audit_report(output_dir="./custom_audits", formats=["json", "yaml", "csv"])
```
