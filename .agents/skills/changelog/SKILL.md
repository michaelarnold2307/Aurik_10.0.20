# Changelog — Aurik 10.0.8

## 10.0.3 (2026-07-11) — BW Harmonic Exciter

### 🎵 Bandbreiten-Erweiterung via DSP (kein ML)

- **BW Harmonic Exciter**: DSP-basierte harmonische Obertongenerierung oberhalb der Cutoff-Frequenz
- Waveshaping (Soft-Clip + Rectification) für gerade/ungerade Harmonische
- Spektrale Hüllkurven-Extrapolation via Polynom-Fit für natürliche Klangbalance
- STFT-basierte Rekonstruktion mit Original-Phasen (keine Phasenartefakte)
- Garantiert: blend=0 = Passthrough, verschlechtert nie das Originalsignal
- <10ms Latenz pro 3s-Segment auf CPU, 0 MB Modellgröße
- Pipeline-Stage: `BWExciterStage` für `UnifiedRestorerV3`
- Plugin: `plugins/bw_harmonic_exciter.py`

## 10.0.2 (2026-07-10) — SOTA-Workflow & Architektur-Vereinheitlichung

### 🧠 SOTA-Workflow: HPE-gesteuerter Phase-Loop

- **PhaseSteeringGuard**: Jede UV3-Phase wird HPE-gemessen. CONTINUE | RETRY_LIGHTER | SKIP | ROLLBACK | STOP_GRACEFUL
- **Cross-Phase Naturalness Consensus**: 7-Band-Tracker verhindert Überbearbeitung (max ±8 dB, max 3 Phasen/Band)
- **HPE ist Chef**: PMGG-Regression wird akzeptiert wenn HPE steigt („klingt besser gewinnt")
- **Steering ist DEFAULT**: Kein opt-in mehr — läuft bei jedem `UnifiedRestorerV3()`-Aufruf

### 🎛️ NaturalnessOptimizer MAX — 12-Stage Post-Processing

- Multi-Band-Glue (3-Band SSL-Style Kompressor), Stereo-Feld-Optimierung, Transienten-Schutz
- De-Essing-Nachbearbeitung, Bass-Management, Sharpness-Korrektur (dynamisch)
- Wärmeband-Guard, Air-Band-Polish, Loudness-Feinschliff, Tonalness-Enhancement
- Alle Stages mit CrossPhaseTracker-gewrappt (Band-Sättigungs-Check)

### 🎯 Studio 2026 Re-Production Chain — 7-Stage Modern Mastering

- Dynamic EQ (6-Band proportional), Adaptive MB Compression (auto-threshold)
- Frequency-Dependent Stereo (wide highs, tight lows), Transient/Tonal Separation (HPSS)
- Dynamic Presence & Air, Sub-Bass Harmonic Synthesis
- True-Peak Limiter mit 4× Oversampling, ISP-Detection, Soft-Clip
- DNA-Guards: Voiceprint (MFCC-Cosine), Groove (Onset-DTW), Emotion (Contour-Pearson), Harmonics (Partial-Ratio)
- Album-Konsistenz: `album_ref`-Parameter für Cross-Track-LUFS-Angleichung (±2 dB)

### 🏗️ Architektur-Vereinheitlichung

- **Ein Steering-System**: NaturalnessOptimizer + StudioChain + UV3 nutzen alle `PhaseSteeringEngine.decide()`
- **Ein Lernsystem**: `MaterialAdaptiveLearner` wrappt `SelfLearningOptimizer` pro Material
- **Ein Entry-Point**: ARE-Pfad deprecated, UV3 immer direkt → NaturalnessOptimizer → StudioChain
- **Self-Learning aktiv**: `enable_self_learning=True` in beiden UV3-Pfaden

### 🖥️ GUI für Nicht-Techniker

- **Onboarding-Wizard**: 3-Schritt-Erststart-Assistent
- **A/B-Vorher/Nachher-Vorschau**: Erste 30s vor der vollen Restaurierung vergleichen
- **Ergebnis-Feedback**: "47 Knackser entfernt, Rauschen −60%" statt technischer Metriken
- **Export-Presets**: 7 Presets (WhatsApp, CD, E-Mail, Handy, Archiv, YouTube, Custom)
- **Kontextsensitive Hilfe**: ?-Buttons + `ErrorSimplifier` + F1-Hilfe-Dialog
- **6 Sprachen**: de/en/fr/es/ja/zh mit Fallback-Kette
- **Light Theme + Schriftgrößen**

### 🔧 Qualität

- **739 Silent-Except-Blöcke behoben**: Alle `except Exception:` → `except Exception as e:` mit `logger.warning`
- **Hard-Clip → Soft-Clip**: `np.clip()` → `np.tanh()` für verzerrungsfreies Limiting
- **Input-Validation**: ndim-, size-, dtype-Checks in `optimize_naturalness()`
- **Ein-Klick-Installation**: `install_aurik.sh` + `install_aurik.bat` mit Desktop/Startmenü-Eintrag
- **GPU-Dokumentation**: `GPU_SETUP.md` (ROCm Linux, DirectML Windows)

### 🧪 Tests

- **54 Unit-Tests grün** (14 Steering + 10 E2E + 10 StudioChain + 10 CrossPhase + 10 AdaptiveLearner)
- **Echte Musik validiert**: Aurik verarbeitet 80er-Schlager-MP3 — kein Crash, kein NaN, HPE +0.026

---

## 10.0.1 (2026-07-07) — Chirurgische Präzision

### 🎯 Zentralisierte Entscheidungsintelligenz

- **SongCalibration Multi-Faktor**: 8-Faktor global_scalar mit Bandwidth-Loss-Guard (−25%), Detektor-Dissens-Guard (−10%), Fragile-Material-Guard (Cap 0.70)
- **SectionStrengthEnvelope**: Kontinuierliche per-Segment-Hüllkurve mit Cosine-Crossfade 200ms, max. 1dB/100ms. Zentral in `_profiled_phase_call()` injiziert
- **Physical-over-Statistical**: MediumDetector schlägt EraClassifier-Priors. Era-Information bleibt als Precursor für Bandbreiten-Ziele erhalten

### 🎤 De-Essing Weltspitze

- **Spectral Dynamic EQ**: Pro-FFT-Bin Soft-Knee-Kompressor mit frequenzabhängigem Threshold (Soothe2/FabFilter-Niveau)
- **Phonem-adaptives De-Essing**: Dynamische Band-Mittenfrequenz basierend auf spektralem Schwerpunkt (/s/ schmal, /ʃ/ breit)
- **Librosa pYIN Gender**: Voicing-Confidence-basierte F0 + Contralto-Erkennung (F0 145–195Hz + weibliche Formanten → FEMALE)
- **Stages 2–6 aktiviert**: Breath Intelligence, Formant System, Vocal Presence, Spectral Inpainting, Vocal Dynamics vollständig geladen

### ⛓️ Tonträgerkette chirurgisch

- **Effective Chain**: `reel_tape → vinyl → cassette → mp3_low` aus physikalischer + statistischer Evidenz
- **Bayesian-Physical-Fusion**: Bayesian unknown > 0.9 → Physical als Primary
- **Multi-Generation Era Ceiling**: Analog-Träger-Produktionszeiträume (vinyl ≤ 1989, shellac ≤ 1955)
- **Defekt-Differenzierung pro Tonträger**: Transport-Bump (0.15/0.95), Print-Through (0.40/0.10), Tape-Head-Level-Dip (0.15/0.65)

### 👂 Fürs menschliche Ohr

- **GrooveMetric Onset-Guard**: ≥90% Onsets → Score ≥0.85 trotz DTW-Fehlschlag
- **Quality-Gate→Action**: PQS-MOS < 2.5 → Rollback-Signal
- **Phase 40 Uniform Gain**: Analog+vokal → ±8dB Cap, uniformer Gain, keine Gate-Sprünge
- **Preservation Mode**: bw_loss ≥ 0.90 ∧ SNR < 16dB → transparente Grenzakzeptanz

### 🏗️ Infrastruktur

- **Vocal Analysis Shared Memory**: VFA → restoration_context, von Phase 19 + SVM gelesen
- **SingerVoiceModel VFA-Integration**: Vibrato und Formanten aus VFA statt Eigenberechnung
- **4-Kern-Optimierung**: harter Default, keine 8-Kern-Überlastung

### 📋 Spezifikation

- **Spec 11**: Entscheidungsintelligenz — 10 INV + 7 ROADMAP
- **Spec 13**: Klangqualität fürs menschliche Ohr — 5 ROADMAP
- **Spec 14**: Vollständigkeit & Perfektion — Export, Fehlertoleranz, Deterministik, Metadaten

## 10.0.0 (2026-07-04) — Weltklasse-Intelligenz

### 🧠 Entscheidungsintelligenz

- **PIM** (Perceptual Intensity Mapper): 10 Frequenzbänder × N Song-Sektionen
- **RLP** (Reflective Listening Pass): Nachbesser-Schleife mit AB-Vergleich
- **Artistic Intent Modulator**: 12 Genres × 10 Epochen → Parameter-Strategie
- **Glue Stage**: Finale subtile Bus-Kompression (1.2:1 Ratio)
- **Stop-Regel**: PMGG-Δ < 0.01 über 3 Phasen → Pipeline stoppt
- **Cross-Phase Awareness**: Phase B kennt das Delta von Phase A

### 🔬 Psychoakustik

- **ATH** ISO 226:2023: Absolute Hörschwelle im Masking-Modell
- **Moore/Glasberg DLM**: 40 ERB-Bänder dynamisches Lautheitsmodell
- **BMLD**: Binaurales Masking via interaurale Kreuzkorrelation
- **PEAQ** ITU-R BS.1387: NMR→ODG im Perceptual Loss
- **Forward Masking**: Frequenzabhängig (logarithmisch 400ms@100Hz→50ms@8kHz)

### 🎤 Vokal-Supremacy

- **Speaker Identity Guard**: ECAPA-TDNN (192-dim) + MFCC (60-dim) Fallback
- **Vocal Overprocessing Detector**: Lisp, Formant-Drift, Sibilanz-Überreduktion
- **Vibrato-Guard**: Cross-Band-Coherence > 0.85 → kein Flutter

### 🐛 Kritische Bugfixes

- **Binäres Gate**: `apply_musical_gain_envelope()` hatte 3 Konstruktionsfehler:
  - Binäres Gate (0 oder 1) → Soft-Knee-Sigmoid mit 6dB Knee
  - 10ms Crossfade → 200ms Hanning-Window
  - §2.30b Hard-Clamp → Entfernt (Soft-Knee schützt inhärent)
- **Small-Gain-Bypass**: Gains ≤ 2dB jetzt uniform (kein Gate)
- **`_scale_audio_region()`**: 10ms Crossfade an Regionsgrenzen (keine Klicks)
- **`_multi_pass()`**: Von Dead-Code zu IAQS-Varianten-Evaluation reaktiviert

### 🆕 Neue Defekttypen (+8)

MPEG_FRAME_LOSS, STEREO_FIELD_COLLAPSE, PHASE_ROTATION,
DROPOUT_OXIDE, DROPOUT_HEAD_CONTACT, DROPOUT_SPLICE,
ASYMMETRIC_CLIPPING, TRANSIENT_IMD

### 🖥️ GUI/Laien

- `get_layman_summary()`: 5 Qualitätsstufen mit Icons (✨👍✅⚠️🔧)
- `get_pipeline_ab_snapshots()`: Base64-WAV für Vorher/Nachher-Player
- `--dry-run`, `--json`, `--abx`, `--progress`, `--resume` CLI-Flags
- ML-Modell-Status in GUI sichtbar
- Kontextbezogene CLI-Fehlermeldungen

### 📦 Export & Delivery

- `export_bitperfect()`: Integer-exakter Passthrough mit BWF-Metadaten
- 11 Playback-Profile (Car, SUV, Bluetooth, Club-PA)
- ISRC/UPC-Metadaten-Support
- `process_album()`: Batch mit Track-Reihenfolge-Intelligenz
- Checkpoint/Resume für abgebrochene Pipelines

### 🧪 ML-Verbesserungen

- 3 Silent-Fallbacks behoben (sota_universal_enhancer jetzt logged)
- Continuous Learning: UCB1 + State-Persistenz + Decay-Faktor 0.99
- GPU-Inferenz: CUDA/ROCm + fp16 für PANNs
- `speaker_identity_guard.py`: Komplettes Rewrite (robust, kein len()-Bug)

### 🔧 Infrastruktur

- Bridge-Compliance: 0 Bypasses in CLI und Batch
- 2 Bridge-Funktionen ergänzt (get_album_consistency_pass, RLP)
- 54 ML-Module inventarisiert und auditiert
- 38 Dateien modifiziert, 14 neue Dateien
- 358+ Tests bestehen

---

## Vorgängerversionen

Siehe Git-History für 9.20.3 und früher.
