# Spec-Evidence: §2.60–§3.0 Evolution + §2.70–§2.76 SOTA-Upgrades

Datum: 2026-07-09 | Spec: `.github/specs/12_evolution_260_30.md`

## Evidence Blocks

### §2.60 Fahrplan-Brücke ✅

**Spec**: Denker als Dirigent, UV3 folgt Phasen-Gruppen, Goal-Priorität, Risiko-Level

**Evidence**:
- `denker/aurik_denker.py`: _emit(pct, msg) mit 20 Stage-Checkpoints (L2, L4, L6…L98)
- `denker/phase_interaction_denker.py`: 44→44 Phasen Plan mit Konflikt-Resolution
- `backend/core/fahrplan.py`: PERCEPTUAL_BUDGET, PHASE_SUBSTITUTIONS
- Aktiv: `AURIK_EVOLUTION=1`

### §2.62 Per-Segment-Executor ✅

**Spec**: UV3 verarbeitet Audio in nicht-uniformen Segmenten basierend auf Defekt-Locations

**Evidence**:
- `backend/core/strength_envelope.py`: 561 Zeilen, 9-Stage SOTA Pipeline
  - Stage 1: Defect-type-specific Gaussian sigma
  - Stage 1.5: Psychoacoustic temporal masking (Moore 2003)
  - Stage 2: Asymmetric attack/release smoothing
  - Stage 3: Transient-aware gating
  - Stage 4: Psychoacoustic floor modulation
  - Stage 5: Per-frame vocal attenuation
  - Stage 6: Joint-Calibrator scaling
  - Stage 7: Floor + Ceiling
  - Stage 8: Crossfade windowing
- Aktiviert in 8 Phasen: 03, 04, 06, 07, 09, 20, 28, 29
- Integration: `backend/core/unified_restorer_v3.py` L32111-32131

### §2.63 Closed-Loop PID ✅

**Spec**: Goal-Error-getriebene Strength-Justierung

**Evidence**:
- `backend/core/unified_restorer_v3.py` L10272-10281
- `backend/core/closed_loop_pid.py`: ClosedLoopPIDController
- `§2.63 Closed-Loop PID: aktiviert mit 19 Goal-Targets`

### §CODEC Chain-Contamination ✅

**Spec**: MP3/AAC-Artefakte nicht als analoge Defekte werten

**Evidence**:
- `backend/core/defect_scanner.py` L2485-2564: `_apply_chain_contamination_discount`
- `backend/core/defect_scanner.py` L2541-2543: TRANSPORT_BUMP aus MP3-Guard entfernt (breitbandige Dips können nicht von MP3 stammen)
- `denker/phase_interaction_denker.py`: Codec-Aware-Phasen-Selektion
- `denker/aurik_denker.py` L1322: `__carrier_chain__` Live-Event

### §2.70 Joint-Calibrator ✅

**Spec**: Goal-Gap-getriebene Optimierung, keine hartcodierten Regeln

**Evidence**:
- `backend/core/joint_calibrator.py`: joint_calibrate() mit Goal-Impact-Scoring
- `backend/core/phase_effect_catalog.py`: 18 Phasen mit Goal-Impact-Profilen (+3 neu: 57, 63, 59)
- `backend/core/unified_restorer_v3.py` L10283-10332: Integration + conductor_strength_hints Injection
- Datenfluss: PhaseInteractionDenker → conflict_notes → Joint-Calibrator
- `denker/aurik_denker.py`: conflict_notes durchgereicht
- `denker/restaurier_denker.py`: conflict_notes an UV3

### §CODEC Phase 03 Guard ✅

**Spec**: BS-RoFormer nur wenn Denker es befürwortet; nie ohne Freigabe

**Evidence**:
- `backend/core/phases/phase_03_denoise.py` L856-884: 3-Verteidigungslinien
  - Guard primär: strength ≤ dsp_threshold → DSP-only
  - Guard fallback: strength ≥ 0.95 + panns ≥ 0.25 → DSP-only
  - BS-RoFormer gate: strength > 0.55 erforderlich
- `§DENKER Phase 03 FALLBACK: strength=1.00 → DSP-only` im Log bestätigt

### §2.71 Strength-Envelope v2 ✅

**Spec**: Zeitvariante Phasen-Stärke aus Defekt-Locations

**Evidence**:
- `backend/core/strength_envelope.py`: 561 Zeilen, 9-Stage Pipeline
- Integration in 8 Phasen via `apply_strength_envelope()`
- Stage 1.5: Psychoakustische Temporal Masking (Forward 200ms, Backward 20ms)

### §2.72 VitalityRestorer ✅

**Spec**: Stereo-Breite, Mikrodynamik, Transienten-Punch nach allen 42 Phasen

**Evidence**:
- `backend/core/dsp/vitality_restorer.py`: 285 Zeilen
- `restore_vitality()`: Stereo (M/S-Gain), Microdynamics (Crest-Expansion), Transients (Dry/Wet-Override)
- Integration: `_execute_pipeline()` vor Quiet-Edge-Clamp

### §2.73 FinalPolish ✅

**Spec**: Era-EQ + Noise-Texture + Dithering

**Evidence**:
- `backend/core/dsp/final_polish.py`: 310 Zeilen
- Era-EQ: 9 Epochen (1930–2000) mit Low-Shelf/Peak/High-Shelf
- CD-Noise-Texture: Per-Terzband Noise-Floor-Glättung
- Noise-Shaped Dither: A-gewichtet, TPDF, 16-bit

### §2.74 Tape-Transport-Defekte ✅

**Spec**: Transport Bumps vollständig erkennen und reparieren

**Evidence**:
- `backend/core/defect_scanner.py`: TRANSPORT_BUMP aus MP3-Guard entfernt
- `backend/core/phases/phase_12_wow_flutter_fix.py`: Alle 3 Code-Pfade checken transport_bump
- 359 Dips erreichen jetzt den Tape-Level-Stabilisator

### §2.75 Reverb-Detection ✅

**Spec**: Noise-Floor-kompensierte Reverb-Messung

**Evidence**:
- `backend/core/hybrid/hybrid_dereverb.py` L544-556: Stationarity-Check mit Noise-Floor-Subtraktion
- Schwelle von 0.20 auf 0.12 gesenkt (noise-kompensiert)

### §2.76 PhaseEffectCatalog-Vollständigkeit ✅

**Spec**: Alle Phasen im Catalog für Joint-Calibrator

**Evidence**:
- Phasen 57 (print_through), 63 (intermodulation), 59 (modulation_noise) hinzugefügt
- Alle 85 DefectType-Werte haben Reparatur-Pfad

### §2.9.x Gender-Detection ✅

**Spec**: Multi-Gender-Erkennung mit 4-Feature-Scoring

**Evidence**:
- `backend/core/phases/phase_19_de_esser.py`:
  - `_detect_gender_robust()`: Vibrato-basierter Contralto-Fallback
  - `_detect_gender_timeline()`: Per-Segment Multi-Gender
  - `_classify_gender_segment()`: 4-Feature gewichtetes Scoring
  - `_process_per_gender_segments()`: Echte Per-Gender-De-Essing

## Test-Status

| Test-Suite | Ergebnis |
|---|---|
| `test_authenticity_metrics_extended` | ✅ Gefixt |
| `test_unified_restorer_v3::TestPreventFirstQuietEdges` | ✅ 34 passed |
| `test_transport_bump` | ✅ Passed |
| Alle anderen | ✅ Keine neuen Failures |
