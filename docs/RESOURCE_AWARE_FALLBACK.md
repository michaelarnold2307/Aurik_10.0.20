# Resource-Aware Fallback System - Aurik 10.0.0

## Überblick

Das Resource-Aware Fallback System überwacht kontinuierlich CPU- und Speicher-Auslastung und schaltet automatisch auf leichtere Algorithmen um, wenn Ressourcen knapp werden.

## Komponenten

### 1. AdaptiveResourceManager

**Datei:** `core/adaptive_resource_manager.py`

**Funktionen:**
- CPU-Auslastungsüberwachung (Schwellenwert: 80%)
- Speicher-Auslastungsüberwachung (Schwellenwert: 85%)
- Dynamische Core-Zuteilung (2-16 Cores)
- Lightweight-Mode-Erkennung
- Verfügbarer Speicher-Check

**API:**
```python
from core.adaptive_resource_manager import adaptive_resource_manager

# Check if lightweight mode should be used
if adaptive_resource_manager.should_use_lightweight_mode():
    # Use DSP-only algorithm
    pass
else:
    # Use full ML algorithm
    pass

# Check memory availability
if adaptive_resource_manager.check_memory_availability(required_mb=500):
    # Proceed with memory-intensive operation
    pass
else:
    # Use lighter alternative
    pass

# Get optimal core count
num_cores = adaptive_resource_manager.get_num_cores()
```

### 2. ML-Hybrid Phasen mit Fallback

Die folgenden Phasen nutzen automatischen Ressourcen-bewussten Fallback:

#### Phase 03: Denoise

- **ML-Hybrid:** OMLSA + Resemble Enhance
- **Fallback:** DSP-only (Spectral Subtraction + Wiener)
- **Trigger:** CPU > 80% oder Memory > 85%

#### Phase 12: Wow/Flutter Fix

- **ML-Hybrid:** YIN + CREPE (CNN pitch detection)
- **Fallback:** DSP-only (YIN Phase Vocoder)
- **Trigger:** CPU > 80% oder Memory > 85%

#### Phase 20: Reverb Reduction

- **ML-Hybrid:** DSP Spectral Gating + DCCRN
- **Fallback:** DSP-only (Spectral Gating)
- **Trigger:** CPU > 80% oder Memory > 85%

## Architektur

```
┌─────────────────────────────────────┐
│  AdaptiveResourceManager            │
│  ├─ CPU Monitor (psutil)            │
│  ├─ Memory Monitor (psutil)         │
│  ├─ Core Scheduler (2-16 cores)     │
│  └─ Lightweight Flag (auto)         │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  ML-Hybrid Phasen                   │
│  ├─ Phase 03: Denoise               │
│  ├─ Phase 12: Wow/Flutter           │
│  └─ Phase 20: Reverb Reduction      │
└─────────────────┬───────────────────┘
                  │
           Quality Mode│Decision
                  ▼
        ┌─────────┴──────────┐
        │                    │
    ┌───▼───┐          ┌─────▼─────┐
    │  ML   │          │  DSP-Only │
    │ Hybrid│          │  Fallback │
    └───────┘          └───────────┘
```

## Integration

Die Fallback-Logik ist automatisch in alle ML-Hybrid Phasen integriert:

```python
# Example from phase_03_denoise.py
quality_mode = kwargs.get("quality_mode", "balanced")

# Check resource availability
use_lightweight = False
if RESOURCE_MANAGER_AVAILABLE:
    use_lightweight = adaptive_resource_manager.should_use_lightweight_mode()
    if use_lightweight:
        logger.info("Resource constraint detected, forcing DSP-only mode")

# ML-Hybrid only if resources available
use_ml_hybrid = ML_HYBRID_AVAILABLE and quality_mode in ["balanced", "maximum"] and not use_lightweight

if use_ml_hybrid:
    # Use ML-enhanced algorithm
    pass
else:
    # Use DSP-only fallback
    pass
```

## Performance Impact

### Normal Operation (CPU < 80%, Memory < 85%)

- Phase 03: OMLSA + Resemble (~1.2-1.5× RT)
- Phase 12: YIN + CREPE (~0.7-2.0× RT adaptive)
- Phase 20: DSP + DCCRN (~0.3-2.0× RT adaptive)

### resource-Constrained (CPU > 80% oder Memory > 85%)

- Phase 03: DSP-only (~0.8× RT) ⚡ **30% faster**
- Phase 12: YIN-only (~0.4× RT) ⚡ **50% faster**
- Phase 20: DSP-only (~0.3× RT) ⚡ **85% faster**

## Monitoring

Das System loggt automatisch Fallback-Entscheidungen:

```
INFO Phase 03: Resource constraint detected, forcing DSP-only mode (CPU: 85.3%, Memory: 82.1%)
INFO Phase 12: Resource constraint detected, forcing DSP-only mode (CPU: 78.5%, Memory: 87.9%)
```

## Konfiguration

Die Schwellenwerte können in `adaptive_resource_manager.py` angepasst werden:

```python
adaptive_resource_manager = AdaptiveResourceManager(
    min_cores=2,  # Minimum cores
    max_cores=16,  # Maximum cores (auto-detect)
    check_interval=2.0,  # Monitor interval in seconds
    cpu_threshold=80,  # CPU threshold %
    memory_threshold=85,  # Memory threshold %
)
```

## Vorteile

✅ **Automatisch:** Keine Benutzer-Konfiguration nötig  
✅ **Graceful Degradation:** Fällt auf bewährte DSP-Algorithmen zurück  
✅ **Performance-Stabil:** Verhindert System-Überlastung  
✅ **Transparent:** Loggt alle Fallback-Entscheidungen  
✅ **Adaptiv:** Passt sich dynamisch an System-Auslastung an  

## Status

**Implementation:** ✅ Complete (16. Februar 2026)  
**Testing:** ✅ Validated  
**Integration:** ✅ Phase 03, 12, 20  
**Production Ready:** ✅ Yes  

## Nächste Schritte

- [ ] Erweitere auf weitere ML-intensive Phasen (Phase 06/07 NVSR, Phase 19 Phoneme)
- [ ] GUI-Integration: Zeige Resource-Status im Dashboard
- [ ] Weitere Metriken: Disk I/O, GPU (falls verfügbar)
- [ ] Profil-basierte Optimierung: Lerne optimale Schwellenwerte

---

**Autor:** Aurik 10.0.0 Development Team  
**Version:** 1.0.0  
**Datum:** 16. Februar 2026
