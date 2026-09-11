# Aurik 10 — Weltklasse-Audio-Restaurierung

> **Universeller Einstiegspunkt für alle Agenten: [`AGENTS.md`](AGENTS.md) (Repo-Root).**
> Diese Datei liefert Projekt-Kontext und v10-Invarianten; bei Widersprüchen gilt die in `AGENTS.md` definierte normative Kette (copilot-instructions → VERBOTEN → Specs/Instructions).

**Ziel:** Intelligente Musikwiederherstellung mit psychoakustischer Präzision, deterministischer Reproduzierbarkeit und vollständiger Ausrichtung auf den natürlichen Wohlklang für das menschliche Ohr.

## 🎧 Hörordnung (normative Spitze für Hör-Entscheidungen)

Alle Qualitätsentscheidungen folgen `.github/instructions/hoerordnung.instructions.md` — sie steht ÜBER den Goal-Regeln:

1. **Hör-Invarianten** (unverletzlich, auch gegen Messwerte)
2. **Audibility**: Maskierungsschwelle statt Mess-Null — ein Defekt unter der Schwelle ist kein Defekt
3. **Lexikografische Wohlklang-Ordnung**: niemals niederrangige Goals (z.B. groove, brillanz) auf Kosten höherrangiger (authentizitaet, emotionalitaet, natuerlichkeit) verbessern
4. **Einladungs-Gate**: Export muss einladen, nicht ermüden (Fatigue)

Metriken sind Zeugen — die Hör-Instanz entscheidet (aber nie gegen Ebene 1).

## 🛡️ Produktions-Invarianten (aus überwachten Restaurierungsläufen)

Diese Invarianten entstanden aus realen Lauf-Befunden — Verstöße zerstören Exporte oder degradieren die Hörqualität:

- **§0c Export-Vertrag**: Bei fehlgeschlagenem Export-Gate MUSS das bestmögliche sichere Ergebnis mit Status „degraded“ exportiert werden. Hardstop ohne Ausgabedatei ist normativ unzulässig (Parität CLI ↔ Frontend/export_workflow).
- **Stereo-Layout-Invariante**: Die Pipeline arbeitet intern **channels-first (C, N)**. Jede Modul-Grenze (Optimizer, Plugins, Metriken) MUSS das Layout normalisieren oder beide Layouts bedienen. `audio[:, 0]` / `mean(axis=0)` auf einem (N, C)-Array kollabiert auf C Samples — Produktionsbefund: Export `(2,)` statt `(480000, 2)`.
- **ORT-Provider-Tupel**: `(name, options)`-Tupel aus dem fp16-Pfad dürfen NIE `str()`-t werden (Policy-Funktionen reichen sie unverändert durch). Sonst „EP Error Unknown Provider Type“ + stiller CPU-Fallback pro Session.
- **Guard-Kalibrierung**: Hard-Fail nur bei **Regression gegenüber dem Input** (delta-basiert). Absolute Produktions-Normalwerte (z.B. TP > −1 dBTP bei gemasterter Musik) sind KEIN Hard-Fail — sonst Rollback jeder Phase („0 Phasen“-No-Op).
- **Log-Zeilen sind Bug-Reports**: Bei überwachten Läufen sind CRITICAL/WARNING-Zeilen Pflichtbefunde. Bekannte Muster: „Signal degeneriert (2 Samples)“, „0 Phasen“, „Retry erzeugten keine Ausführung“, „unbekannte Action“, „self-comparison erkannt (no-op pipeline)“. Kein stilles Akzeptieren.
- **ONNX-Export-Regeln**: Artefakte MÜSSEN mit onnxruntime (CPU) ladbar sein und numerisch gegen TorchScript/Eager validiert werden (max-abs ≤ 1e-3). Komplexe FFTs vermeiden — ONNX `DFT(inverse=1, onesided=1)` lehnt ORT ab. Große Modelle mit externem `.onnx.data` (nur zusammen kopieren).
- **Facade-API-Kompatibilität**: `get_htdemucs_plugin()` liefert ein Stems-**Dict** (DemucsV4-Facade) — Aufrufer MÜSSEN Dict und `SeparationResult.reconstruct()` bedienen.
- **Retry-Semantik**: m1b-Nachbehandlung ist eine explizite Hörbarkeits-Gate-Entscheidung — ERB-Masken-Skips dürfen gequeuede Retries nicht vetoen.

## 🚀 v10 Invarianten

- **Bridge-Bypass-Verbot**: Kein UI-/Frontend-Code (Aurik10, CLI) importiert `backend/core/` direkt. Nur über `backend/api/bridge.py`. Die Denker-Schicht (`denker/`) ist Teil der Backend-Orchestrierung und von diesem Verbot ausgenommen.
- **Soft-Knee-Gate**: `apply_musical_gain_envelope()` arbeitet mit Sigmoid-Soft-Knee (6dB), 200ms Hanning-Crossfade. KEIN Hard-Clamp.
- **PIM-first**: Vor jedem Phasen-Loop wird die PIM-Intensitäts-Map berechnet und in `restoration_context` gespeichert.
- **RLP-last**: Nach jedem Phasen-Loop wird der RLP ausgeführt. Korrekturen werden nur bei objektiver Verbesserung übernommen.
- **ML-Fallback-Logging**: JEDER ML→DSP-Fallback MUSS mit `logger.warning()` protokolliert werden. Silent-Failures sind VERBOTEN.
- **ML-Device-Detection**: `next(model.parameters()).device` statt `model.device` — letzteres ist nach partiellen `.cpu()`/`.to()`-Aufrufen auf Sub-Modulen unzuverlässig (ROCm-NaN-Fix-Pattern).
- **ML-Recovery-API-Äquivalenz**: Recovery-Pfad nach GPU-Fehler MUSS dieselbe API wie der Hauptpfad verwenden (z. B. `model.generate_batch()`), nur mit reduzierten Steps. Niemals komplett andere Funktionssignatur im Retry.
- **Artistic Intent vor Defect-Scan**: `get_artistic_intent()` wird VOR dem Defect-Scan aufgerufen.
- **Glue Stage immer**: Die Glue-Stage läuft in ALLEN Modi als vorletzte Phase.
- **62 DefectTypes**: Keine willkürlichen neuen DefectTypes ohne Phase-Mapping und Material-Sensitivity.
- **NaN/Inf-Schutz für ALLE Phasen**: Jede Phase MUSS `np.nan_to_num()` oder `np.isfinite()` auf Ausgabe-Audio anwenden (§0a). PhaseInterface bietet Basis-Schutz; explizite Prüfung in jeder Phase als Defense-in-Depth.
- **Logger-Pflicht**: Jede Python-Datei, die `logger` verwendet, MUSS `import logging` und `logger = logging.getLogger(__name__)` auf Modulebene definieren. F821 (undefined name) ist Null-Toleranz.
- **Test-Assertion-Konvention**: `np.testing.assert_allclose` nimmt Toleranzen (`rtol`, `atol`). NIEMALS Toleranzen an numpy-Mathefunktionen übergeben (`np.abs(x, rtol=1e-5)` → `np.abs(x)`).
- **Guard-Counter-Lebendigkeit**: Jeder deklarierte Guard-Counter (`_max_measures`, `_timeout_s`) MUSS auch inkrementiert werden. Deklaration ohne `+= 1` ist toter Code — der Guard greift nie.
- **Messschleifen-Plateau**: Jede Messschleife mit ≥3 Kandidaten MUSS Plateau-Erkennung haben. 3 identische Violation-Sets in Folge → `break`. Verhindert ~30 s Latenz durch blindes Durchmessen aller Varianten.

### v10.0.0 Präzisions-Invarianten

- **Centralized Decision Intelligence (§2.16)**: Alle Stärke-Entscheidungen fließen zentral im Denker.
- **Section-Strength-Envelope (§2.17)**: Kontinuierliche 48kHz-Hüllkurve, Cosine-Crossfade 200ms.
- **Physical-over-Statistical (§6.8)**: Physikalische Evidenz schlägt statistische Priors.
- **Fragile-Material-Guard (§2.15)**: bw_loss ≥ 0.90 ∧ SNR < 16dB → global_scalar ≤ 0.70.
- **Bandwidth-Loss-Guard (§2.12)**: global_scalar −25% proportional zu bw_loss.
- **Uncertainty-from-Disagreement (§2.13)**: Detektor-Divergenz → global_scalar ×0.90.
- **Bayesian-Physical-Fusion (§6.8)**: Bayesian unknown > 0.9 → Physical als Primary.
- **Quality-Gate→Action (§2.14)**: PQS-MOS < 2.5 → Rollback-Signal.
- **Onset-Preservation-Guard (§2.14)**: ≥90% Onsets → Score-Override.
- **Chain-Aware Defect Differentiation (§6.7)**: Chirurgische Schwellwerte pro Tonträger.
- **Multi-Generation Era Ceiling (§2.13)**: Analog-Träger-Produktionszeiträume.
- **Vocal Analysis Shared Memory (§2.9)**: VFA → restoration_context.
- **Phonem-Adaptive De-Essing (§2.9)**: Dynamische Band-Mittenfrequenz.
- **Spectral Dynamic EQ (§2.10)**: Pro-FFT-Bin Soft-Knee, Soothe2-Niveau.
- **Librosa pYIN Gender (§2.11)**: Voicing-Confidence + Contralto-Erkennung.

### v10.303 Phase-0 ML-Pre-Processor (implementiert)

- **Carrier-Chain-Inversion**: Apollo (Codec) → DeepFilterNet v3 (Noise) → Resemble Enhance (Spektrum). ML VOR DSP.
- **Hallucination-Guard pro Stufe**: `spectral_novelty > threshold → Rollback`. Apollo=0.35, DFN=0.50, Resemble=0.40.
- **Breath-Preservation**: DeepFilterNet läuft NUR außerhalb von BreathDetector-Segmenten (§2.8).
- **Phase-0-Aware-Skips**: 12 redundante DSP-Phasen werden via `_should_skip_resolved_phase()` übersprungen.
- **Goal-Recalibration**: Nach Phase 0 wird Goal-Baseline gegen Phase-0-Output gemessen, nicht gegen degradiertes Original.
- **Precision-Phases**: Phase 40 (Loudness) + Phase 47 (True-Peak) umgehen Conductor/SongCal-Drossel.
- **Watchdog-Referenz**: DoNoHarmGuardian prüft gegen Phase-0-Output, nicht degradiertes Original (§v10.303.25).
- **Qualitäts-Baseline**: Nach Phase 0 wird `original_audio_for_goals` auf Phase-0-Referenz umgestellt (§v10.303.31).
  Alle Post-Pipeline-Checks (Goals, Goosebumps, EmotionalArc, IAD, HPI, MUSHRA) vergleichen gegen
  den ML-verbesserten Output — nicht gegen das physikalisch unerreichbare degradierte Original.
- **Cache**: Hash-basierte Persistenz in `~/.aurik/cache/phase0/` für Batch-Imports (§v10.303.18).
- **PLM-Lade-Reihenfolge Phase 0**: EAR_VAE → Apollo → DeepFilterNet v3 → Resemble Enhance (Stufen-Reihenfolge der Kette, Chain-Metadaten „ear_vae→apollo→deepfilternet→resemble_enhance“).
- **MP3-resistente Gender-Detection**: `bandwidth_loss` an `_detect_gender_robust()` übergeben.

### v10 Roadmap

> §3.1 (SectionStrengthEnvelope) und §3.4 (DAG-Reordering) sind inzwischen implementiert
> (aktiv im Produktionslauf: „§3.4 DAG reorder“ / „SectionGoalAdapter … sektionsweise
> Stärke-Skalierung aktiv“). Die übrigen Einträge bleiben Roadmap.

| § | Konzept | Beschreibung |
|---|---|---|
| §3.0 | **Cross-Phase Naturalness Consensus** | Phasen im gleichen Frequenzbereich stimmen sich ab. Naturalness-Guard prüft kumulative Wirkung |
| ~~§3.1~~ | ~~SectionStrengthEnvelope aktiv~~ | **Implementiert** — SectionGoalAdapter skaliert Stärke sektionsweise |
| §3.2 | **Artist/Track-Fingerprint** | BatchSessionLearner persistiert Stimm-Modell + Track-Modell für Transfer |
| §3.3 | **Blind Reference-Free Quality** | MERT-Embedding-basierte absolute Qualitätsschätzung ohne Vergleich zum degradierten Original |
| ~~§3.4~~ | ~~Dynamic Phase Ordering (DAG)~~ | **Implementiert** — PhaseInteractionDenker + DAG-Reorder aktiv |
| §3.5 | **Real-time Preview** | 10s in ~30s vorab restaurieren zur Validierung |
| §3.6 | **Human-Panel MUSHRA** | Ridge-Regression auf echten Hörtest-Daten → kalibrierter MUSHRA-Proxy |

## 🎯 Kernwerte

- **Präzision über Geschwindigkeit** — Akustische Wahrheit zuerst
- **Transparenz** — Jede Entscheidung ist nachvollziehbar
- **Konsistenz** — Derselbe Input → derselbe Output (immer)
- **Wissenschaftlichkeit** — SOTA-Modelle, verifiable Metriken
- **Natürlichkeit** — Vollständige Ausrichtung auf den Wohlklang für das menschliche Ohr
- **Chirurgie** — Jeder Defekt wird mit dem exakt richtigen Werkzeug in der exakt richtigen Intensität behandelt

## 📊 Architektur-Ebenen

```
CLI (cli/aurik_cli.py)
  ↓
Bridge API (backend/api/bridge.py) [Mode-Normalisierung]
  ↓
Denker-Schicht (denker/*.py) [ZENTRALE ENTSCHEIDUNGSINTELLIGENZ]
  ├─ AurikDenker         — Orchestrierung, GlobalPlan
  ├─ StrategieDenker     — Budget, Performance
  ├─ DefektDenker        — CausalDefectReasoner
  ├─ PhaseInteractionDenker — Phasen-Ordnung, Konflikte
  ├─ ReparaturDenker     — Pre-UV3 Reparatur
  ├─ RekonstruktionsDenker — Gap-Reparatur
  ├─ RestaurierDenker    — UV3-Instanz, Core-Skalierung
  └─ ExzellenzDenker     — Musical Goals, Goal-Repair
  ↓
UnifiedRestorerV3 (backend/core/unified_restorer_v3.py)
  ├─ **Phase 0: EAR_VAE→Apollo→DeepFilterNet v3→Resemble Enhance** (plugins/apollo_phase0_integration.py;
  │  EAR_VAE = neuraler Clean-Pass VOR den subtraktiven Stufen)
  ├─ SongCalibration     — global_scalar, family_scalars, ALLE Guards
  ├─ SectionStrengthEnvelope — kontinuierliche per-Segment-Hüllkurve
  ├─ Phase-Selektion     — Preservation Mode, Risk-Guard
  └─ _profiled_phase_call — zentrale Envelope-Injektion
  ↓
Kernmodule [Psychoakustik + DSP]
  ├─ Defekt-Scanner (backend/core/defect_scanner.py)
  ├─ Medium-Detector (forensics/medium_detector.py)
  ├─ Era-Classifier (backend/core/era_classifier.py)
  ├─ Phasen-Engine (backend/core/phases/)
  ├─ Musikalische Ziele (backend/core/musical_goals/)
  ├─ Vocal-Analyse (backend/core/vocal_focus_analyzer.py)
  ├─ DSP/Formanten (backend/core/dsp/)
  └─ ML-Bridges (backend/ml/inference_only/)
  ↓
Export (backend/exporter.py)
```

## 🔗 Chain-Architektur (v10.0.0)

Jeder Tonträger in der Kette treibt spezifische Entscheidungen:

```
reel_tape (Era-Precursor) → Bandbreiten-Ziel 18–20kHz, Studio-Dynamik
vinyl    (Physical)       → Primär-Material, RIAA-EQ, Knistern, Rotation
cassette (Physical)       → Transport-Bumps, Wow/Flutter, Bandsättigung
mp3_low  (Physical)       → IQR-Guard, Bandbreiten-Cap, Pre-Echo-Schutz
```

**Chain-Awareness über alle Detektoren hinweg:**

- MediumDetector → physikalische Chain + `physical_analog_sources`
- EraClassifier → `material_prior` als Precursor, §v10.303.42 Deep-Chain-Korrektur (≥3 Träger → +10y/Stufe)
- DefectScanner → kettenadaptive Schwellwerte für ALLE 20 Defekttypen + §v10.304 AST Defect-Music-Discriminator
- SourceFidelityReconstructor → Bandbreiten-Ziel vom ältesten Träger
- Phase-Selektion → `reel_tape`-Precursor aktiviert Phase 06 (Frequency Restoration)
- LyricsGuidedEnhancement → §v10.303.50 HF-Whisper-Decoder (echte Wort-Transkription) + §v10.303.52 Semantic-DSP
- CIG (CumulativeInteractionGuard) → §v10.304 GDD-Schwellen mit Material/Restorability/Chain-Depth-Context
- AST AudioSet Classifier → §v10.304 zentraler 527-Klassen-Classifier, Goal-Mappings korrigiert
- Strength-Floor-Gate → §v10.304.4 effective_strength < 0.15 → Phase skip („43→43" eliminiert)
- AST-Transient-Guard → §v10.304.5 NR-Strength sinkt bei Instrument-Präsenz (Piano/Snare)
- AST Pre-Filter → §v10.304.2 DefectScanner-Schwellen werden vor Scan angehoben
- P5-Elimination → §v10.304.3 STFT-Längen-Drift in phase_29 + universell in guard_phase_output gefixt
- Phase Contract Guard → §v10.304.3 Shape-Normalisierung für alle Phasen-Ausgaben
- Genre-Adaptive Goals → §v10.304.7 6 Genres mit spezifischen Referenz-Indizes

## ✅ Qualitäts-Gating

### 1. **Tests** (Ziel: 80%+ Abdeckung)

- Unit: schnell, isoliert, ≤1s
- Integration: echte Daten, ≤5s
- E2E: golden samples, ≤30s
- Contract: Bridge↔API Invarianten

### 2. **Typisierung** (mypy strict in Backend)

- Alle Backend-Funktionen: `def func(arg: Type) -> Type:`
- ML-Plugins: `ignore_errors = true` (extern)
- DSP-Core: Type-Hints für öffentliche APIs

### 3. **Linting** (ruff select=[E,W,F,I,N,UP,B,C4,SIM,RUF])

- **F821 Null-Toleranz**: Kein undefined name im gesamten Projekt. Jeder Build mit F821 ist FAIL.
- Automatische Fixes: `ruff check --fix`
- Per-File-Ignores nur für bewusste DSP-Konventionen
- Pre-Commit: ruff, black, isort
- Projektweit: `ruff check .` → "All checks passed!" als CI-Gate

### 4. **Konsistenz**

- **Imports:** isort (Black-kompatibel) → `from module import name`
- **Formatierung:** Black 120er Zeilenlänge
- **Namensgebung:** snake_case (PEP8), Math-Variablen (N, X, sr)

## 📝 Dateien-Struktur (v10.0.0)

```
backend/
  ├─ core/
  │  ├─ defect_scanner.py             # Audio-Defekt-Klassifizierung (62 Types)
  │  ├─ unified_restorer_v3.py        # Haupt-Orchestrator + SongCalibration
  │  ├─ singer_voice_model.py         # VFA-Daten-Integration (§2.9)
  │  ├─ vocal_focus_analyzer.py       # Register, Formanten, Vibrato, Style
  │  ├─ room_acoustics_fingerprinter.py
  │  ├─ era_classifier.py             # Multi-Gen-Era-Ceiling (§2.13)
  │  ├─ phases/
  │  │  ├─ phase_19_de_esser.py       # Phonem-adaptiv + Spectral Dynamic EQ
  │  │  ├─ phase_36_transient_shaper.py # Fragile-Skip (§2.15)
  │  │  ├─ phase_39_air_band_enhancement.py # Analog-Skip
  │  │  ├─ phase_40_loudness_normalization.py # Uniform Gain analog+vocal
  │  │  └─ phase_54_transparent_dynamics.py
  │  ├─ musical_goals/
  │  │  └─ musical_goals_metrics.py   # Onset-Preservation-Guard (§2.14)
  │  └─ dsp/
  │     └─ section_strength_envelope.py # Kontinuierliche Hüllkurve (§2.17)
  ├─ api/bridge.py
  └─ exporter.py

forensics/
  └─ medium_detector.py               # Bayesian-Physical-Fusion (§6.8)

denker/
  ├─ aurik_denker.py
  ├─ restaurier_denker.py             # Core-Skalierung (4 Kerne)
  ├─ exzellenz_denker.py
  └─ README.md
```

## 🚫 VERBOTEN

Siehe `.github/VERBOTEN.md` — nicht verhandelbarer Sicherheits- & Qualitäts-Katalog.

**v10.0.0 Ergänzung:** Workarounds sind VERBOTEN. Jede Lösung muss die Ursache beheben, nicht das Symptom umgehen. Phasen-Individuelle Schwellwerte sind VERBOTEN — alle Stärke-Entscheidungen fließen zentral über `global_scalar`.

## 🔗 Externe Ressourcen

- Spezifikationen: `.github/specs/`
  - `01_musical_goals.md` — 15 Musical Goals
  - `02_pipeline_architecture.md` — Pipeline-Ablauf, Modi
  - `11_decision_intelligence.md` — Denker, SongCalibration, SectionEnvelope, Roadmap
  - `13_human_ear_quality.md` — Klangqualität fürs menschliche Ohr, Roadmap
  - `14_completeness_and_perfection.md` — Fehlertoleranz, Deterministik, Export, Batch-Lernen, Roadmap
  - `v10.305_startup_integration_contract.md` — Startup-Sequenz, GPU-Detection, Unified Progress, Context-Aware Comm

## 🖥️ Startup & Kommunikation (§v10.305)

- **GPU-Detection im Hauptthread**: `get_ml_device_manager()` MUSS in `main.py` VOR `ModernMainWindow` aufgerufen werden. Kein `torch.zeros("cuda")` in der Erkennung.
- **Warmup ohne GPU-Touch**: `warmup_models_background()` darf `get_ml_device_manager()` NICHT aufrufen. Singleton ist bereits initialisiert.
- **Unified Progress**: `_sync_unified_progress()` ist die EINZIGE Methode, die `progress_bar.setValue()` und `phase_progress_bar.setValue()` aufruft. Fragmentierte Update-Pfade sind VERBOTEN.
- **Kontextbewusste Kommunikation**: Jeder Importsong bekommt individuelle Status-Botschaften via `_build_context_status()`. Medium, Ära, Genre, Score fließen ein. Generische „Aurik arbeitet an {file}" ist nur Fallback.
- **i18n-Pflicht**: Jeder benutzersichtbare String MUSS via `t()` internationalisiert sein. Hardcodierte Strings sind VERBOTEN (§G84).
- **Cache-Safety**: Launcher mit `python3 -B` starten. `.pyc`-Caches können Source-Änderungen verschleiern.
- **Lock-Disziplin**: `threading.Lock` DARF NICHT während `import`-Statements gehalten werden (§G174). Importe gehören VOR den Lock.
- **Event-Garantie**: Jedes `threading.Event` MUSS in `finally` oder garantiertem Exception-Handler gesetzt werden (§G173).
- **Plugin-Namen-Validierung**: Alle Zugriffsnamen in `warmup_models_background._plugins` MÜSSEN mit tatsächlichen Funktionen übereinstimmen (§G175).
- **Watchdog-Selbsttest**: Jeder Watchdog MUSS prüfen, dass seine Aktivierungsbedingung tatsächlich erreichbar ist (§G176).
- Instruktionen: `.github/instructions/`
- Copilot-Verhaltensrichtlinien: `.github/copilot-instructions.md`
