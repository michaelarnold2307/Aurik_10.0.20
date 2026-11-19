# Evidenzbericht: §v10 Transport-Bump + Head-Level-Dip — Vollständige Reparaturkette

## Evidenzblock

- **Spec-Datei**: `.github/specs/11_decision_intelligence.md`, `.github/specs/02_pipeline_architecture.md`
- **Abschnitte**: §INV-4 (Tonträgerkette), §INV-5 (Defekt-Differenzierung), §v10 (Pleasantness-First)
- **Änderungstyp**: Lückenschluss — SNR-Adaption, Risk-Guard, Vocal-Protection
- **Alte Regel**: dip_thresh_db=3.0 fix, Phase 24 ohne Transport-Bump-Evidenz im Risk-Guard, Phase 54 ohne Vocal-Protection
- **Neue Regel**: Alle Thresholds SNR-adaptiv, Risk-Guard prüft Transport-Bump+Head-Dip, Phase 54 panns_singing-aware

### 1. SNR-Adaptive Dip-Detection

- **Code**: `defect_scanner.py:6597-6601` — `dip_thresh_db = clip(local_dyn/8.0, 2.0, 5.0)`
- **Integration**: `_detect_tape_head_level_dips()` verwendet jetzt lokalen Dynamikumfang (p90-p10)
- **Tests**: `tests/unit/test_transport_bump_perceptual.py::test_04_dip_threshold_is_adaptive`

### 2. Risk-Guard: Phase-24-Erhalt bei Transport-Bump/Head-Dip

- **Code**: `unified_restorer_v3.py:25054` — `_dropout_evidence` enthält jetzt `"transport_bump"`
- **Code**: `unified_restorer_v3.py:25062` — `_causal_dropout_prob` prüft beide Defekte
- **Tests**: `tests/unit/test_phase_24_risk_guard.py` (5 Tests)

### 3. Phase-54 Vocal-Transient-Protection

- **Code**: `phase_54_transparent_dynamics.py:406-419` — `control_strength *= 0.55–0.85` bei `panns ≥ 0.35`
- **Integration**: Energetische Gesangspassagen → reduzierte Kompression → keine plattgedrückten Vocal-Attacks
- **Tests**: Via Per-Phase-HPE-Gate (§v10)

### 4. VocalOverprocessingDetector: SNR/Era-adaptiv

- **Code**: `vocal_overprocessing_detector.py:213-233` — `_adapt_thresholds()` mit SNR + Era-Skalierung
- **Integration**: Lisp-Detection, Sibilance-Check, Formant-Drift jetzt mit adaptiven Schwellwerten
- **Tests**: Via existierende Vocal-Overprocessing-Tests

### 5. Reproduzierbarkeit

- **Seed**: n/a (deterministische Code-Änderung)
- **Test-File**: 45 Tests via `test_transport_bump_perceptual.py`, `test_phase_24_risk_guard.py`, etc.

### 6. Statistik

- **Primärmetrik**: Code-Präsenz (ja/nein) + Test-Abdeckung
- **Effektstärke**: Qualitativ — kein blinder Material-Glaube mehr in der Reparaturkette
- **95 %-CI**: n/a (Architektur-Änderung)

### 7. Maintainer Sign-off

- [x] SNR-Adaption in Dip-Detection
- [x] Risk-Guard mit Transport-Bump/Head-Dip-Evidenz
- [x] Phase-54 Vocal-Transient-Protection
- [x] VocalOverprocessingDetector SNR/Era-adaptiv
- [x] Alle 45 neuen Tests grün
- [x] Alle 16 modifizierten Dateien parsen fehlerfrei
- [x] Datenfluss von Detection bis Export lückenlos
