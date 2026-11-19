# Aurik Betriebssicherheits-Spezifikation (§D3, §D4, §E3)

> Generiert aus der Fehleranalyse vom 12. Juli 2026.
> Normativ für alle Contributors. Ergänzt CLAUDE.md §Verboten.

---

## §D3 Log-Level-Richtlinie

Jede Log-Zeile MUSS dem passenden Level zugeordnet sein.
Falsches Level = false positives in Monitoring / Log-Analyse.

| Level | Definition | Beispiele |
|-------|-----------|-----------|
| **CRITICAL** | Datenverlust, korrupter Output, Sicherheitslücke | Startup-Selbsttest: Attribut-Fehler in Readiness-Check (A1) |
| **ERROR** | Phase komplett fehlgeschlagen, kein Fallback möglich | ONNX-Session-Inferenz crasht ohne Fallback |
| **WARNING** | DSP-Fallback aktiv, Modell nicht geladen, Ressource knapp | `ml_memory_budget` blockiert ML-Load, präventiver Guard schlägt zu, Circuit-Breaker aktiviert |
| **INFO** | Normaler Betriebsablauf, Phasen-Start/Ende, Modell erfolgreich geladen | `Phase X gestartet`, `PANNs loaded 0.7 GB`, `Systemprofil: RAM=31.2 GB` |
| **DEBUG** | Wiederholte Singleton-Nutzung, Pipeline-Zwischenschritte, Cache-Hits | `BasicPitch bereits geladen`, `Thrashing-Check: swap 45% ohne aktives Paging` |

### Spezifische Regeln

1. **Singleton-Lade-Logs**: Nur beim ERSTEN Laden `INFO`, alle weiteren Zugriffe `DEBUG`.
2. **Preemptive-Guard-Blocks**: `WARNING` (nicht `ERROR` – Fallback existiert via DSP).
3. **Readiness-Check-Fehler**: `WARNING` mit TTL-Cache (kein Spam).
4. **Startup-Selbsttest-Attribut-Fehler**: `CRITICAL` (Modell wird NIE als bereit erkannt).

---

## §D4 C-Level-Log-Deduplizierung

Problem: ONNX Runtime, CUDA, ROCm schreiben C-Level-Warnings direkt nach stderr.
Bei 40+ identischen Zeilen pro Modell-Ladung wird das Log unlesbar.

### Maßnahmen (umgesetzt)

1. **ONNX Runtime**: `ort.set_default_logger_severity(3)` (ERROR) – beim frühestmöglichen Import in `backend/core/onnx/runtime.py`.
2. **Umgebungsvariablen**: `ORT_DISABLE_MIOPEN_BN_EPSILON_WARNINGS=1` via `os.environ.setdefault` als Fallback (vor ONNX-Import).

### Maßnahmen (zukünftig)

3. **stderr-Wrapper**: Ein Startup-Hook, der stderr-Zeilen puffert und identische C-Level-Warnings dedupliziert:
   ```
   [W:onnxruntime:Default, miopen_common.h:111] ... (42 weitere identische Zeilen unterdrückt)
   ```
   Implementierung via `sys.stderr`-Wrapper oder `logging.StreamHandler` mit Dedup-Logik.

4. **CUDA/ROCm**: `CUDA_LAUNCH_BLOCKING=0`, `ROCR_VISIBLE_DEVICES` – dokumentieren, nicht hart setzen.

---

## §E3 Pre-Guard-Dokumentation

### Architektur

Der `ml_memory_budget.try_allocate()`-Flow hat DREI Schutzebenen:

```
try_allocate(model, size_gb)
  │
  ├─ 1. is_system_thrashing()         ← HART: Swap-Thrashing, RAM <8%
  │     └─ Block sofort, keine Ausnahme
  │
  ├─ 2. _should_block_heavy_ml_load() ← WEICH: Präventiver Vorfilter
  │     └─ Block nur bei Swap-Druck + geringem Headroom
  │     └─ Model-size-aware: kleine Modelle brauchen weniger freies RAM
  │     └─ KEIN finales Urteil — _preflight_system_memory hat letztes Wort
  │
  ├─ 3. _preflight_system_memory()    ← PRÄZISE: Load-Peak-Mathe, OOMD-Safety
  │     └─ Berechnet exakten RAM-Bedarf inkl. Deserialisierungs-Peak
  │     └─ Versucht Plugin-Eviction vor Block
  │
  └─ 4. Budget-Check (ML_MAX_GB)      ← LOGISCH: Budget-Limit
```

### Wichtige Invarianten

- **Schicht 2 darf NIE das letzte Wort sein.** Sie ist ein Performance-Vorfilter, der offensichtlich hoffnungslose Fälle früh abfängt. Wenn Schicht 2 durchlässt, entscheidet Schicht 3.
- **Schicht 2 nutzt 80% der Schicht-3-Margen** (`_preempt_factor = 0.80`). Dadurch werden Modelle, die Schicht 3 gerade noch durchlassen würde, nicht fälschlich von Schicht 2 blockiert.
- **Model-size-aware**: Kleine Modelle (1.1 GB) brauchen ~3.8 GB frei, große (7 GB) brauchen ~10.6 GB. Kein fixes 6-GB-Limit mehr.
- **Hard Floor**: Unter `_MIN_FREE_MB_HARD` (~2.4 GB auf 32 GB) blockiert Schicht 2 immer, unabhängig von Swap.

### Kalibrierung

Alle Schwellwerte werden beim Modul-Import automatisch an die System-RAM-Größe kalibriert:

| System | swap_early | swap_elevated | avail_ratio_max | min_free_mb |
|--------|:----------:|:-------------:|:---------------:|:-----------:|
| 16 GB  | 47% | 72% | 0.26 | 2458 MB |
| 32 GB  | 52% | 75% | 0.20 | 2400 MB |
| 64 GB  | 56% | 78% | 0.13 | 2400 MB |

Override via `set_budget(max_gb, guard_overrides={'heavy_swap_early_pct': 55.0})`.

---

## §E2 Load-Peak-Faktor aus Modell-Metadaten

### Problem

Aktuell nutzen `_should_block_heavy_ml_load` und `_preflight_system_memory` hartcodierte Load-Peak-Faktoren (1.30×–1.60× je nach System-RAM). Diese sind konservativ geschätzt, aber nicht modellspezifisch.

### Design (zukünftig)

Jedes Modell sollte seinen Deserialisierungs-Peak-Faktor deklarieren:

```python
# In der Plugin-Klasse:
_load_peak_factor: float = 1.60  # PyTorch torch.load() peak

# Oder über eine Registry:
register_model_metadata("AudioSR", load_peak_factor=1.45, format="onnx")
```

Der Faktor hängt ab vom:
- **Format**: ONNX (≈1.15×), PyTorch (≈1.60×), TensorFlow (≈1.40×)
- **Modellgröße**: Größere Modelle haben proportional kleinere Peaks
- **Architektur**: Transformer (höherer Peak) vs CNN (niedriger)

### Minimal-Implementierung (MVP)

Als MVP kann der Peak-Faktor aus der Dateigröße des Modells geschätzt werden:
- `<500 MB` → 1.50× (kleine Modelle: relativ mehr Overhead)
- `500 MB – 2 GB` → 1.35×
- `>2 GB` → 1.20× (große Modelle: kompakte Tensoren)

Dies würde den aktuellen hartcodierten 1.30–1.60-Range präzisieren und in ~70% der Fälle einen niedrigeren Wert liefern.
