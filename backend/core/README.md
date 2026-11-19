# AURIK v8.0 Core Modules

**Normative Architektur gemäß:** `docs/00_normative/aurik_v_8_projektstruktur_ki_programmierregeln.md`

---

## Struktur

```text
backend/core/
├─ epistemic_gate/      # Zuständigkeitsprüfung (Ethics Engine)
├─ zone_engine/         # A/B/C Klassifizierung
├─ conduct_enforcer/    # Regeldurchsetzung
├─ regulator/           # Pipeline-Steuerung
└─ evaluation/          # CAS Score, Quality Gates, Learning
```

---

## Module

### 1. Epistemic Gate (`epistemic_gate/`)

**Verantwortlich für:** Zuständigkeitsprüfung vor jeder Verarbeitung

**Kern-Komponenten:**

- `ethics_engine.py` - Implementiert Epistemic Decision Logic
  - `EpistemicDecision` enum: PRESERVE, MODE_A, MODE_B, HARD_STOP
  - `epistemic_gate()` - 8-Branch Decision Tree
  - `conduct_regulator()` - 6 Constraint Checks

**Regel:** MUSS vor jeder Verarbeitung ausgeführt werden!

---

### 2. Zone Engine (`zone_engine/`)

**Verantwortlich für:** Klassifizierung in Zonen A, B, C

**Kern-Komponenten:**

- `region_analysis.py` - Region-basierte Analyse (non-invasive)
  - Detektiert: Silence, Speech, Music, Noise, Mixed  - Liefert Empfehlungen (nicht bindend!)
  - **WICHTIG:** Keine Verarbeitung, nur Analysis

- `context_analysis.py` - Kontext für Zonierung
  - Genre, Defect Type, Confidence
  - Cultural Significance

**Zonenmodell:**

- **Zone A:** Klar reparierbar (hohe Confidence)
- **Zone B:** Unsicher (mittlere Confidence)
- **Zone C:** Bedeutungstragend → Keine Automation!

---

### 3. Conduct Enforcer (`conduct_enforcer/`)

**Verantwortlich für:** Durchsetzung der 9 Conduct-Prinzipien

**Kern-Komponenten:**

- `adaptive_goal.py` - Goal Definition Engine
  - Prüft Ziele gegen Conduct Rules
  - Kann Goals ablehnen wenn zu aggressiv

**Conduct-Prinzipien:**

1. Primum non nocere
2. Integrität vor Optimierung
3. Stabilität vor Brillanz
4. Verbesserung ist optional
5. Unspektakulär ist erfolgreich
6. Funktionale Unvollkommenheit ist geschützt
7. Unsicherheit ist gültiger Endzustand
8. Lernen → Vorsicht (nicht Kühnheit)
9. Wo Interpretation beginnt, tritt AURIK zurück

---

### 4. Regulator (`regulator/`)

**Verantwortlich für:** Zentrale Pipeline-Steuerung

**Kern-Komponenten:**

- `adaptive_pipeline.py` - Haupt-Orchestrator
  - 12-Phasen Pipeline
  - Phase 4.5: Epistemic Gate Integration
  - Monitor-Wrapping aller Processing-Schritte
  - Quality Gate Enforcement

**Pipeline-Flow:**

```text
1. Context Analysis
2. Goal Definition
3. Feature Extraction
4. Detected Medium Analysis
★ 4.5. EPISTEMIC GATE (Ethics Check)
5. Restoration
6. Repair
7. Reconstruction
8. Remastering
9. Quality Control
10. Mastering
11. Audit Capture
12. Export
```

---

### 5. Evaluation (`evaluation/`)

**Verantwortlich für:** Qualitäts-Assessment & Continuous Learning

**Kern-Komponenten:**

- `quality_control.py`
  - **CAS Score Calculator:** Creative Authenticity Score (5 Dimensionen)
    - Brillanz (25%): Spectral Centroid, HF Content
    - Transparenz (20%): Spectral Contrast
    - Authentizität (20%): Harmonic/Percussive Balance
    - Emotionalität (20%): RMS Variation
    - Wärme (15%): LF Energy (60-500 Hz)
  - **Quality Gates:** 6 umfassende Checks
    1. SNR ≤ 2dB loss
    2. THD ≤ 50% increase
    3. No clipping (peak < 0.99)
    4. NISQA ≥ 4.0 (vocals)
    5. CAS ≥ 0.80 (minimum)
    6. Spectral Fidelity ≥ 90%

- `continuous_learning.py`
  - **SuccessPatternAnalyzer:** 1000+ Audit Report Analysis
  - **StrategyWeightOptimizer:** Bayesian Weight Optimization
  - **ConfidenceCalibrator:** Prediction Accuracy Analysis
  - **PerformanceMetricsAggregator:** Trend Detection
  - **Learning Cycle:** Offline, Conservative (learning_rate=0.1)

**Qualitäts-Ratings:**

- CAS ≥ 0.96: ⭐⭐⭐⭐⭐ World-Class
- CAS ≥ 0.92: ⭐⭐⭐⭐ Excellent
- CAS ≥ 0.80: ⭐⭐⭐ Good (Export-Minimum)
- CAS < 0.80: ❌ Rejected

---

## Architektur-Regeln

### Verbindliche Regeln (aus Norm):

1. **Keine Abkürzungen:** Jede Verarbeitung läuft durch:

   ```text
   Epistemic Gate → Zonen → Conduct → Regulator
   ```

2. **Epistemic Gate ist nicht umgehbar:**
   - MUSS vor jeder Verarbeitung laufen
   - Kann HARD_STOP oder PRESERVE erzwingen
   - Kein Override möglich

3. **DSP-Module nur in Sandbox:**
   - Kein persistenter Zustand
   - Parameter vom Regulator gesetzt
   - Siehe: `backend/dsp/sandbox/`

4. **ML nur Inference-Mode:**
   - Kein Online-Learning
   - Safety Wrappers required
   - Siehe: `backend/ml/safety_wrappers/`

5. **Quality Gates haben Vetorecht:**
   - Processing wird abgebrochen bei Gate-Failure
   - Original wird bewahrt bei Unsicherheit

---

## Verwendung

**Import aus Core-Modulen:**

```python
# Epistemic Gate
from backend.core.epistemic_gate.ethics_engine import EthicsEngine, EpistemicDecision

# Zone Engine
from backend.core.zone_engine.region_analysis import RegionAnalysisSystem
from backend.core.zone_engine.context_analysis import ContextAnalyzer

# Conduct Enforcer
from backend.core.conduct_enforcer.adaptive_goal import AdaptiveGoalEngine

# Regulator
from backend.core.regulator.adaptive_pipeline import AdaptiveProcessingPipeline

# Evaluation
from backend.core.evaluation.quality_control import CASScoreCalculator, QualityGates
from backend.core.evaluation.continuous_learning import ContinuousLearningSystem
```

**Typisches Flow:**

```python
# 1. Initialize Core Systems
ethics = EthicsEngine()
zone_engine = RegionAnalysisSystem()
goal_engine = AdaptiveGoalEngine()
pipeline = AdaptiveProcessingPipeline()

# 2. Analyze Context & Regions
context = context_analyzer.analyze(features, user_profile)
regions = zone_engine.analyze_audio_regions(audio, sr)

# 3. Define Goal (Conduct-aware)
goal = goal_engine.define_goal(context)

# 4. Epistemic Gate Check
ethics_report = ethics.epistemic_gate(ethics_context)
if ethics_report.decision == EpistemicDecision.HARD_STOP:
    return original_audio  # Keine Verarbeitung zulässig!

# 5. Run Regulated Pipeline
result = pipeline.run(audio, features, user_profile)

# 6. Quality Gates Enforcement
cas_calculator = CASScoreCalculator()
cas_score = cas_calculator.calculate_cas_score(result_audio, sr)
if cas_score < 0.80:
    return original_audio  # Quality threshold not met
```

---

## Status: v8.0 (7. Februar 2026)

**Implementiert:**

- ✅ Ethics Engine (Epistemic Gate)
- ✅ Region Analysis (Zone Engine Foundation)
- ✅ Quality Control (CAS + 6 Gates)
- ✅ Continuous Learning (1000+ File Analysis)
- ✅ Pipeline Integration (Phase 4.5)
- ✅ Audio Monitor (Permanent Tracking)

**In Progress:**

- 🔄 Conduct Enforcer (conduct_rules.yaml Integration)
- 🔄 Full Zone Classification (A/B/C Labeling)

**Geplant:**

- 📋 Regulator Sandbox Integration
- 📋 ML Safety Wrappers
- 📋 Audit-Visualisierung

---

## Kontakt

Für Fragen zur normativen Architektur:
→ Siehe: `docs/00_normative/`
