# Spec §20: Predictive Quality Guard (v10.53)

## Motivation

Phasen-Rollbacks (VQI, CIG/GDD, AFG) kosten Rechenzeit und führen zu
Restart-Overhead. §v10.53 ersetzt reaktive Rollbacks durch proaktive
Pre-Phase-Prädiktion: bevor eine Phase läuft, wird ihr Schadenspotenzial
abgeschätzt und die Strength präventiv gecappt — oder die Phase ganz
übersprungen.

## Architektur

```
_profiled_phase_call()
├── _predictive_quality_guard()   ← VOR phase.process()
│   ├── VQI-Guard (RMS-Stabilitäts-Proxy)
│   ├── CIG-STFT-Guard (Group-Delay-Estimation)
│   └── AFG-Guard (HF-Ratio-Check)
├── phase.process()               ← PHASE (ggf. mit gecappter Strength)
└── CIG-STFT-Tracking             ← NACH phase.process()
    ├── _cig_stft_phase_count++
    └── _cig_current_gdd_ms = count × 10ms
```

## Komponenten

### A. VQI-Guard (Voice Quality Index)

- **Proxy**: `_estimate_vqi_proxy()` — RMS-Stabilitäts-Ratio vs. Referenz
- **Regel**: Wenn VQI-Proxy < 0.70 bei vocal_enhancement → Strength × 0.5
- **Laufzeit**: O(n) — ein RMS-Pass, kein FFT

### B. CIG-STFT-Guard (Cumulative Interaction Guard)

- **Pre-Phase-Check**: Wenn ≥ 2 STFT-Phasen gelaufen und GDD > 8ms → SKIP (0.0)
- **Post-Phase-Tracking**: Nach jeder STFT-Phase:
  - `_cig_stft_phase_count` inkrementiert
  - `_cig_current_gdd_ms = count × 10ms` (konservative Schätzung)
- **Detektion**: Phase hat "stft" in `dsp_profile` oder `phase_id`

### C. AFG-Guard (Artifact Generation Feedback)

- **Proxy**: `_estimate_hf_ratio()` — HF/Total-FFT-Ratio
- **Regel**: Wenn `hf_ratio` um > 40% seit Referenz gestiegen → Strength × 0.60
- **Begründung**: Übermäßiger HF-Zuwachs deutet auf Artifakt-Generierung hin

## Integration in _profiled_phase_call

1. **Guard-Modulation (vor Phase)**:
   - `_predictive_quality_guard()` liefert optionalen Cap-Wert
   - Cap ≤ 0.0 → Phase wird komplett übersprungen
   - Cap > 0.0 → `kwargs["strength"]` wird auf `min(old, cap)` gecappt

2. **STFT-Tracking (nach Phase)**:
   - Nach `phase.process()` und GuardWisdom-Recording
   - Inkrementiert `_cig_stft_phase_count` für STFT-Phasen
   - Schätzt kumulative GDD für nächste CIG-Guard-Prüfung

## Abgrenzung zum bestehenden CIG

| Aspekt | CIG (cumulative_interaction_guard.py) | Predictive Guard |
|--------|--------------------------------------|-----------------|
| Zeitpunkt | Nach Phase (reaktiv) | Vor Phase (proaktiv) |
| GDD-Messung | Exakt (FFT-basiert) | Schätzung (count × 10ms) |
| Schwellwert | Adaptiv (Material/Ära) | Konservativ-fest (8ms) |
| Rollback | Ja | Nein (Skip/Cap) |
| Kosten | Hoch (STFT-Messung) | Niedrig (Counter + Arithmetik) |

## Tests

- `test_predictive_vqi_guard_rms_stability`: VQI-Proxy korreliert mit RMS-Stabilität
- `test_predictive_cig_stft_tracking`: Counter + GDD-Schätzung nach STFT-Phase
- `test_predictive_afg_guard_hf_ratio`: HF-Ratio-Check erkennt Artifakt-Tendenz
- `test_predictive_guard_integration`: End-to-End: Guard wird in _profiled_phase_call aufgerufen

## Dateien

- `backend/core/unified_restorer_v3.py`:
  - `_predictive_quality_guard()` (Zeile ~27980)
  - `_estimate_vqi_proxy()` (Zeile ~28044)
  - `_estimate_hf_ratio()` (Zeile ~28032)
  - CIG-STFT-Tracking (Zeile ~29803)
- `tests/unit/test_unified_restorer_v3.py`: Tests für alle Guard-Komponenten
