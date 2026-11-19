# Evidenzbericht: §2.46c, §2.46d, §2.35c — SOTA Workflow-Fixes

## Evidenzblock

- **Spec-Datei**: `.github/specs/02_pipeline_architecture.md`
- **Abschnitte**: §2.46c (Phase 12 Pitch-Span), §2.46d (FlashSR Recovery), §2.35c (LPC AA-Filter)
- **Änderungstyp**: SOTA-Härtung — keine Workarounds, echte Lösungen
- **Alte Regel**: Starre Thresholds, stille Degrade-Pfade, CPU + optionale AMD-GPU ML
- **Neue Regel**: Adaptive Thresholds, Recovery-Ketten, GPU-DDIM + CPU-Vocoder

### 1. §2.46c: Phase 12 Material-Adaptiver Pitch-Span

- **Problem**: 100-Cent-Threshold blockierte Cassette-Wow/Flutter-Korrektur (17× im Log)
- **Fix**: Adaptiver Threshold: 400 Cents für Cassette/Reel-Tape mit wow≥0.70
- **Datei**: `backend/core/phases/phase_12_wow_flutter_fix.py`

### 2. §2.46d: FlashSR SOTA Recovery-Kette

- **Problem**: model.cpu() + CPU-DDIM → NaN bei Extrem-Bandbreite (375 Hz)
- **Fix**: 4-stufig: GPU-DDIM(50)→CPU-DDIM(20)→SBR-DSP→Passthrough
- **Datei**: `plugins/flashsr_plugin.py`

### 3. §2.35c: LPC-Formant-Tracker Anti-Aliasing

- **Problem**: Stiller Degrade-Pfad (Dezimation ohne AA-Filter) + 1-Sample-Off-by-One
- **Fix**: scipy.signal Top-Level-Import, Segment-Trim, Start-Clamp
- **Datei**: `backend/core/dsp/lpc_formant_tracker.py`

### 4. Reproduzierbarkeit

- **Seed**: n/a (deterministische Code-Änderungen)
- **Commits**: 1addc275, 8b4eda37, b33d91ae, b6da1132, ef49e9fc
- **Tests**: 69/69 normative, 0 VERBOTEN issues, 0 compliance errors

### 5. Maintainer Sign-off

- [x] Phase 12 adaptiver Threshold dokumentiert
- [x] FlashSR Recovery-Kette spezifiziert
- [x] LPC AA-Filter als Pflicht deklariert
- [x] Alle Datei-Referenzen aktuell
- [x] Reproduktions-Skripte in Specs
