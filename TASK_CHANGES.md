# TASK_CHANGES — Live-Ledger der aktuellen Aufgabe

> Generiert von `scripts/change_ledger.py snapshot` (Base: `HEAD`, Stand: 2026-09-19 22:55 CEST).
> CI (`ci-lite.yml` pr-evidence-gate) erzwingt Abdeckung: jede geänderte Code-Datei muss hier stehen.

## Geänderte Dateien

| Status | Pfad | Art |
|---|---|---|
| M | backend/core/plugin_lifecycle_manager.py | modifiziert |
| M | plugins/banquet_vinyl_plugin.py | modifiziert |
| M | tests/unit/test_end_of_song_cleanup_hoist.py | modifiziert |

## Entscheidungen

- **§PERF-R9 2026-09-19 (phase_49 WPE batched BLAS, qualitätsneutral)**:
  Fast-Cell-Nachmessung (`benchmark_effizienz_matrix.py --cells fast
  --seconds 10 --profile-top-phases 8`, vinyl_test_01.wav, ROCm-Venv,
  `output/perf_session_20260919/`) ergab: Qualitätsprüfung 136 s,
  Musical Goals 72 s, Audio-Nachbearbeitung 46 s — Engine-Ebene-Kosten
  ≈ 50 % des Laufs (51× RT). Aufschlüsselung: End-Gate-Kaskade läuft
  13+ measure_all-Runden (P1/P2 2–3 Alphas + Universal-Cascade 8 feste
  Alphas je volle 15-Goal-Messung ~8 s warm; VERSA/PANNs/MERT je Alpha
  neu, keine Referenz-seitige Wiederverwendung — als nächster Hebel
  dokumentiert, s. Roadmap). phase_49-Attribution (30 s) war
  PLM-/Gate-Overhead: Reverb-Presence-Gate übersprang WPE auf dem
  trockenen Material. **UMGESETZT:** `_predict_reverb_bands_batch`
  ersetzt den Per-Bin-Loop (cProfile: 88 % der Kanal-Zeit auf halligem
  Material) durch batched BLAS-Matmul je 192-Bin-Block — Semantik
  erhalten (Silent-Bins=0, LinAlgError⇒0 je Bin, convolve-Summenordnung,
  §2.61-Budget-Guard mit Teil-Ergebnis). Messung (48 kHz, 30 s, WPE
  aktiv): Loop 1,83 s → 0,91 s (2,0×), Kanal 3,22 s → 1,87 s (1,7×);
  max|Δ| 4e-14 je Band / 1,4e-8 Kanal-Ausgang (Float-Rauschen der batched
  BLAS-Ordnung, §G5 (GEBOTE.md)-Determinismus gepinnt). Tests:
  test_phase_49_advanced_dereverb.py (Äquivalenz/Silent-Bins/
  Budget/Determinismus; 16 Tests grün mit Phase-49-Suite); Ruff clean;
  compliance_check.py 1340 Dateien clean; verboten-linter ohne neue
  Befunde (V59-Warning vorbestehend).
- **SOTA-Analogie-Sweep 2026-09-17 (ANA-1 umgesetzt, ANA-2…6 Backlog)**:
  Frage „Weitere SOTA-Korrekturmöglichkeiten analog zur Songaufbauerkennung?“
  — systematischer Sweep nach Komponenten, die auf Stellvertreter-Heuristiken
  statt definierender Evidenz bauen oder Rollen duplizieren (§V7
  (copilot-instructions.md)). **ANA-1 UMGESETZT:** Die §2.52b-Grenz-Erkennung
  nutzte eine agglomerative k-Heuristik, während die definierende
  SSM/Checkerboard-Novelty (Foote 2000) bereits im §2.17-Analysator lag —
  kanonisches Modul `backend/core/dsp/ssm_segmentation.py` (Write-Gate
  geprüft, Registry-Eintrag vorhanden); §2.17 delegiert (verhaltensidentisch,
  37 Tests), §2.52b nutzt SSM primär mit agglomerativem Fallback;
  Intro/Outro nur noch für das erste/letzte Segment (Positionsregel
  verschluckte den 156-s-Refrain als „outro“). Elke-Best-225s: 12
  evidenz-basierte Segmente (statt 7), alle 4 Refrain-Positionen + Klimax,
  Laufzeit 1,28 s/min ≤ Budget; neuer Test
  `test_ssm_boundaries_detect_aba_transitions`.
  **Backlog ANA-2…6** (Rezepte in der Roadmap-Tabelle): per-Segment-PANNs
  statt Flatness-Proxy (ANA-2), ML-Gender-Klassifikator (ANA-3),
  Tonart-Erkennung kanonisieren (ANA-4, phase_53/genre_classifier doppelt),
  BPM aus §2.17 in §2.52b übernehmen (ANA-5, Spec erwähnt Beat-Tracking),
  Chunk-Modus-Struktur einmal pro Song (ANA-6).
- **Arbeitsaufträge 2026-09-17 (Songaufbau, Gates-Sweep, Ablauf)**:
  - AUF-1 phase_43 Audibility-Gate: bereits vorhanden (2026-09-16) — verifiziert,
    keine Nachrüstung nötig.
  - AUF-2 Gates-Sweep: Inventar über alle 69 Phasen-Dateien korrigiert
    (05 hat is_below_masking-Early-Termination, 63 hat 20-dB-Peak-Gate, 62 hat
    KEIN Maskierungs-Gate — Roadmap-Behauptung korrigiert). Nachgerüstet:
    **phase_09** `_apply_audibility_gate_to_regions` (subaudible
    Knistern-Regionen übersprungen, §4; Zähler `subaudible_crackle_skipped`
    in allen 3 Return-Pfaden; 2 neue Tests). Latenten Test-Drift gefixt:
    2 Docker-Fallback-Tests waren seit der SUP-F2-Signaturänderung
    `_remove_crackle_ml(..., sample_rate)` rot. Backlog mit Rezept je Phase:
    62/54/44/45/52/57/61/18/32.
  - AUF-3 Songaufbauanalyse: NICHT auf SOTA-Stufe (Elke-Best-225s: 0 Chorus/
    0 Klimax, 42 s „intro“, 37 s „bridge“ — unabhängiger Chroma-Fenster-Scan
    belegt Refrain bei ≈44/100/156/192 s). Upgrade `song_structure_analyzer.py`:
    Fenster-Wiederholungs-Evidenz (`_window_repetition_counts`),
    Wiederholungs-Labels, Intro/Outro-Caps (≤ 15 %), Klimax = Chorus mit
    p90 ≥ 98 % des Song-Maximums. Ergebnis deckt sich jetzt vollständig;
    Laufzeit 1,29 s/min ≤ Budget; 17 Tests grün. Report:
    `docs/reports/current/2026-09-17_song_structure_sota_abgleich.md`.
    Rest-Gap dokumentiert: ML-Boundary-Detektor (GPU), Beat-/Downbeat-Tracking,
    Chunk-Modus-Struktur.
  - AUF-4 Ablauf: RestorabilityEstimator-Laufzeitkurve ehrlich kalibriert
    (33× RT gemessen statt 2,5× RT versprochen); BANQUET-Singleton bestätigt
    (kein Fix nötig); Chunk-Modus-Struktur + Witness-Veto-Zustand als
    Folge-Slices dokumentiert.
- **Defizit-Abarbeitung 2026-09-17 (SUP-F5/F6, PSY-A1-43, Budget-Wahrheit)**:
  - SUP-F5 (DIAGNOSTIK): `RestorabilityEstimator` taktet den MuQ-ML-Prior jetzt
    separat und zieht ihn vom §2.26-DSP-Budget ab — der einmalige
    Modell-Erst-Load (gemessen 19,2 s) löst keine 5-s-Budget-Warnung mehr aus;
    geprüft wird nur noch der echte DSP-Anteil (INFO „Erst-Load einmalig je
    Prozess“ statt WARNING).
  - SUP-F6 (DIAGNOSTIK, Kalibrierung statt Workaround): (a)
    `pre_echo_model` bildet das Verhältnis nur noch auf HINZUGEFÜGTER Energie
    (positive Delta-Hälfte — Klick-ENTFERNUNG zählte vorher wie eine
    Pre-Echo-HINZUFÜGUNG), ergänzt um eine lokale Vor-Fenster-Audibility-
    Schwelle (−18 dB) und eine absolute −60-dB-Hörbarkeits-Schwelle für leise
    Onsets (Produktionsfall phase_01 micro_fallback Δ=+0,00 dB und
    phase_47-Limiter reproduziert → jetzt −200). (b) `listening_witness`
    klemmt den Rauigkeits-Anstieg unterhalb des relativen JND (≤ 35 % =
    2× Vassilakis-JND ≈ 17 %; Befund: harmloser 30-Hz-Hochpass = +1,16 auf
    Skala ~35632) auf 0 — Findings UND Veto-Loop erben EINE Wahrheitsquelle.
    2 neue Fälle je Testdatei (test_pre_echo_model, test_listening_witness).
  - PSY-A1 phase_43 (Sibilanten-Maskierungs-Gate): subaudible Sibilanten-
    Segmente (unter Maskierungsschwelle, Band 4–12 kHz, Muster phase_19)
    bleiben ungezähmt; Zähler `subaudible_sibilants_skipped`;
    test_phase_43_ml_deesser (48 Tests grün).
  - Budget-Wahrheit (Profiling auf dem Elke-Best-Export): phase_01s
    4,4×-RT-Attribution „DSP-Multiscale“ war falsch — Multiscale kostet nur
    ~3 s/225 s (11 % der Phasen-Zeit); Treiber sind der einmalige
    BANQUET-ML-Load (0,78 s/Prozess) + Device-Detection (0,26 s) bzw. im Lauf
    die CPU-Inferenz. R8-Rollout auf 01/19 daher evidenzbasiert BEENDET
    (phase_19 hat das Sparse-Muster bereits als Segment-Gate).
  - F3/HR-V1-Budget-Messung (Torch-ROCm): 2,5× RT (25,05 s/10 s, 26 Bänder,
    PQS 4,93) — Flag bleibt vertragsgemäß OFF bis Gesamt-Budget-Nachweis
    (aktuell 33× RT > 32×-Ziel); Activation-Contract-Tests sind flag-bewusst.
  - Roadmap/FILE_REGISTRY nachgezogen; Export-Analyse-Report ergänzt
    (0 eingeführte Restdefekte, MuQ 4,84→4,85).
- **Überwachter 225-s-Lauf 2026-09-16 (Elke Best, voller Song) — Befunde & Fixes**:
  - SUP-F1 (PERF): PANNs-ROCm-Provider-Filter verglich (Name, Options)-Tupel
    gegen String-Namen → GPU immer verworfen, PANNs dauerhaft CPU. Fix:
    Tupel-Name auspacken; Regressionstest test_panns_rocm_provider_filter.py.
  - SUP-F2 (QUALITÄT): BANQUET-Temp-WAV hart 44,1 kHz (Pipeline 48 kHz →
    Speed-/Pitch-Korruption) + Float-WAV (Docker/scipy: „Format not
    recognised“). Fix: echte sample_rate + PCM_16 in phase_09.
  - SUP-F3 (DIAGNOSTIK): §V44-Meldung „Mono-Kompatibilitätswarnung“ war
    invertiert (ok=False = IACC ≥ 0,70 = schmales Stereobild, perfekt
    mono-kompatibel). Meldung korrigiert.
  - SUP-F4/F5/F6 dokumentiert (62,2× RT Chunk-1; Restorability-5-s-Guard;
    Reinhör-Witness-Kalibrierung) — F4-Rest GPU-gebunden.
  - Befunde in docs/TODOS_SOTA_ROADMAP.md (neuer Abschnitt).
  - Lauf läuft im Hintergrund weiter
    (output/supervised_run/elke_225s_supervised_v1020.log).
- **Defizit-Sweep 2026-09-16 (5 Vollsuite-Fehlschläge behoben)**:
  - **i18n-Gap**: `help.error.generic` (und alle 17 `help.error.*`-Keys) fehlten in
    DE+EN — der ErrorSimplifier zeigte rohe Keys. Keys ergänzt; Tests auf
    übersetzte Texte umgestellt (test_help_system_errorsimplifier.py,
    test_every_t_key_has_translation grün).
  - **Gacela-Shim ordnungsrobust**: `install_tifresi_shim()` hatte einen
    Early-Return, wenn `tifresi` bereits (partiell) in sys.modules war
    (z. B. nach GACELA-Plugin-Inferenz via models/gacela/tifresi-Stub) →
    partielles Paket, Importfehler je nach Test-Reihenfolge. Shim ist jetzt
    idempotent (vollständiger Ersatz); Test purgt zusätzlich data/data.*/
    utils/utils.worker aus dem Modul-Cache.
  - **MuQ-sys.path-Pollution (Root-Cause)**: muq_plugin inserierte
    `models/muq_eval/src` (generische Namen: data.py, model.py) dauerhaft an
    sys.path[0] — kaperte jedes spätere `import data` im Prozess
    ("'data' is not a package" im Gacela-Upstream-Import; Test-Ordnungsbruch
    muq↔gacela). Fix: scoped Import mit try/finally-Restore (§V7);
    `scripts/validate_muq_plugin_direction.py` macht seinen eigenen
    sys.path-Setup (kein Plugin-Seiteneffekt mehr).
  - **BS-RoFormer-CPU-Retry-Vertrag**: `_build_cpu_session()` baute über
    get_onnx_providers auf ROCm-Maschinen erneut eine GPU-Session (der
    gerade fehlgeschlagene Kernel erneut) — Kontrakt ist CPU-only.
    Fix: providers=["CPUExecutionProvider"] explizit; Test-Vertrag grün.
  - **MuQ-Determinismus-Schalter**: `AURIK_MUQ_GPU=0` wird jetzt im Plugin
    geehrt (vor dem Singleton-Cache!) und erzwingt CPU-Inferenz — ROCm-GPU-
    Inferenz ist nicht bit-deterministisch (§G5 (GEBOTE.md), MuLan-Befund);
    der Test test_embedding_deterministic_and_shape nutzt genau diesen
    Schalter und ist damit ordnungs-stabil.
  - **§2.46f Edge-Gain-Cap (neues Modul backend/core/dsp/edge_gain_cap.py)**:
    Root-Cause der letzten Ordnungs-Fehlschläge: der phase_03-ML-Pfad
    (warme Modelle nach All-Phases-Smoke) restauriert den Pegel über
    `_p03_out` NACH dem Konvex-Edge-Taper — das Intro konnte dadurch
    +2,8 dB über Original liegen (Test-Bar +2 dB). Der Cap (nur Absenkung,
    50-ms-Crossfade zur Referenz, layout-tolerant, 2 % Marge) läuft im
    DSP-Pfad vor dem Return UND im ML-Pfad NACH `_p03_out`; zusätzlich
    `apply_edge_taper_convex` (0,5-s-Fade Richtung Eintritts-Audio) im
    ML-Pfad für die Korrelations-Invariante. 8 Unit-Tests; smoke+edge-
    Paarungen jetzt grün.
  - **MuQ-Device-Sync**: `extract_embedding`/`estimate_muq_mos` bewegen das
    warmgeladene Modell (und die A1-Module) auf das Zielgerät, wenn sich
    `_resolve_device` nach dem Warm-up ändert (AURIK_MUQ_GPU=0 bei bereits
    GPU-geladenem Singleton) — vorher Device-Mismatch → None statt
    Fallback (§V6).
  - Verifikation: 5. Vollsuite-Bestätigungslauf im Hintergrund (Nachweis folgt).
- **SOTA-Roadmap-Rest (CPU) — R2-Verdrahtung + PSY-A1/A8-JND-Gates + P65-Witness (2026-09-16)**:
  - **R2/WIT-M4 (MuQ-MOS-Gate):** `OneTakeExport.prepare(reference_audio=…)` +
    `one_take_prepare` erweitert; beide uv3-Pfade (Whole-Song + Chunked)
    reichen den Original-Input an `ExportQualityGate.check` durch; MuQ-Felder
    im Quality-Report + `result.metadata` (export_muq_mos_*); Bridge-Payload
    `build_export_quality_gate_payload` surft `muq_mos_witness` (available/
    in/out/delta) informativ. Witness bleibt SOFT (blockt nie, §0c).
    Tests: test_r2_muq_mos_export_wiring.py (6 Fälle).
  - **PSY-A1/A8 (phase_25/31):** Azimut-Schwelle JND-basiert (ITD-JND 30 µs ×
    3,5 ≈ 5 Samples @ 48 kHz, SR-unabhängig; HF-Floor ≥ Pegel-JND) —
    Bestandsverhalten bit-identisch. Speed-Schwelle (0,3 %) als JND-gestützt
    dokumentiert (max-Floor mit Frequenz-JND 0,2 %). Tests:
    test_psy_a1_jnd_gates_25_31.py (5 Fälle).
  - **SOTA-P65 (S4-Muster):** `_apply_singer_identity_witness` in phase_65 —
    Resemblyzer cos(pre, post) ≥ 0,92, sonst proportionaler Blend Richtung
    Input; non-blocking (§V6) und layout-sicher. Tests:
    test_phase_65_singer_identity_witness.py (6 Fälle).
  - Roadmap-Status nachgezogen (TODOS_SOTA_ROADMAP.md: Tabelle C, WIT-M4,
    Abschluss-Matrix); FILE_REGISTRY + repo_search-Before-Create für die 3
    neuen Testdateien durchgeführt.
- **SOTA-C4/F5: DDSP-Prädiktor-Harness + Erstlauf (2026-09-16, Negativbefund)**:
  - Neues Skript `scripts/train_ddsp_predictor_c4.py` (--precompute/--train/--smoke):
    MUSDB18-HQ-Effekt-Paare (deterministische 3-Band-RBJ-EQ + Soft-Knee-Kompressor,
    Parameter = Label) → LAION-CLAP-Embeddings (eingefroren, 512-dim) → MLP-Head
    (512→256→128→6, MSE). Cache-Pipeline + song-weiser 80/20-Split, Seed 42 (§G5 (GEBOTE.md)).
  - Erstlauf: 40 Songs × 12 Segmente = 480 Paare, 25 Epochs — best val_MAE 0,2447
    vs. Mittelwert-Baseline 0,2455 ⇒ **KEIN Signalanteil**: semantisches CLAP trägt
    Produktions-EQ/Dynamik-Parameter nicht (zusätzlich schlecht gestellte Aufgabe).
    Head NICHT aktiviert (fail-closed). Taskformulierung braucht DDSP-artigen
    Mel-Encoder oder Audio-Frontend (BEATs/MERT) — Folgeschritt GPU.
    Beleg: `docs/reports/current/2026-09-16_ddsp_c4_first_run.md`.
- **S4-Verifikation (VOCAL-INPAINT-S4, GPU-Punkt) — formal durchgeführt, 2 Produktions-Bugs behoben (2026-09-16)**:
  - Harness `scripts/validate_vocal_inpaint_s4.py`: Kaskaden-Ausgang je 300-ms-
    Gesangslücke (13 Lücken, 3 MUSDB-Tracks, Seed 42), ΔSDR ≥ 0 je Segment,
    Resemblyzer-Witness cos ≥ 0,92, Bit-Determinismus; Exit 0/1/2.
  - **Lauf 1 (Status quo):** FlowMatching TIER-0 mean **−2,76 dB** (alle 13
    Segmente unter der Stille-Baseline) — Q11 hatte nur den CQTdiff+-Arm
    gemessen (0,0 dB); FlowAudio reproduzierte sich nicht bit-identisch.
  - **Fix 1 (§G5):** `plugins/flow_audio_sota.py` nutzt jetzt
    input-abgeleitete blake2b-Seeds (`_derived_rng`) für Partial-Phasen,
    Shaped-Noise und x_0 — 2 Determinismus-Tests (bit-identisch).
  - **Fix 2 (Evidenz-Reorder):** phase_55 versucht für Gesangslücken
    (≥ 50 ms, vocals_confidence ≥ 0,40) CQTdiff+ VOR FlowMatching;
    FlowMatching bleibt erster Fallback (keine phasen-individuellen
    Schwellwerte, §V7-konform; 2 Prioritäts-Tests).
  - **Lauf 2:** mean **+0,005 dB**, Determinismus bit-identisch ✅; striktes
    Per-Segment-Gate (min −0,12 dB) und Witness in allen Fenstern (min cos
    0,763) bleiben marginal offen — Hebel ist der GPU-Finetune der F-Reihe.
    Beleg: `docs/reports/current/2026-09-16_vocal_inpaint_s4.json`.
- **SOTA-HR-V1: Synthese-Pfad hinter F3-Aktivierungsvertrag + A/B-Validierung (2026-09-16)**:
  - phase_07 verdrahtet den BigVGAN-Repair-Pfad hinter `bigvgan_v2_ready()`
    (Flag = einzige Schaltstelle, fail-closed): Synthese →
    `additive_synthesis_gate` (Never-worsen, §B5) → nur bei
    `bands_released > 0` Übernahme; Witness `hr_v1` (attempted/applied).
    ML→DSP-Fallback loggt jetzt `logger.warning` + Begründung (§V6 statt
    DEBUG). Tests: test_hr_v1_activation_contract.py von 4 auf 6 Fälle
    (fail-closed-Synthese-Fehler, model_used=none ⇒ kein Eingriff).
  - Neues Skript `scripts/validate_hr_v1.py`: A/B-Gate (af ≥ −0,02,
    HNR ≥ −0,5 dB, Exit 0/1/2) — **Lauf 2026-09-16 (Elke-Best-20s,
    Torch-ROCm): af +0,0073, HNR +4,42 dB, PQS 4,52, 26 Bänder ⇒ PASS**
    (Beleg: `docs/reports/current/2026-09-16_hr_v1_bigvgan_ab_validation.md`).
    `BIGVGAN_V2_HR_ACTIVATED` bleibt bewusst OFF: Rollout-Voraussetzungen
    sind Test-Suite-Anpassung auf den aktivierten Pfad und ein
    UV3-Performance-Budget-Nachweis (BigVGAN-Inferenz ≫ 10× RT).
  - Roadmap: Q5 (F2-Verlängerung) GESCHLOSSEN (obsolet — S1-Validierung hat
    Pfad B verworfen; unterbrochener Resume-Lauf wird nicht fortgesetzt);
    Q6/F3 mit A/B-PASS dokumentiert; Phase-Tabelle (07/55) aktualisiert.
- **Q6/F3: 23/50/03-Verdrahtung (2026-09-16)**:
  - Gemeinsamer Helfer `plugins.bigvgan_v2_plugin.apply_hr_v1_additive()` —
    EINE Schaltstelle (fail-closed via `bigvgan_v2_ready()`, Synthese →
    additive_synthesis_gate, Übernahme nur bei bands_released > 0,
    §V6-Warnung beim ML→DSP-Fallback, layout-agnostisch).
  - phase_07 auf den Helfer umgestellt (Verhalten identisch); `hr_v1`-Witness
    in phase_23 (Hauptpfad), phase_50 (Hauptpfad) und phase_03 (DSP- und
    ML-Hybrid-Pfad) exportiert — Flag aus ⇒ attempted=False, Status quo.
  - Tests: test_hr_v1_activation_contract.py von 6 auf 9 Fälle
    (23/50/03-Witness); 53 weitere Phase-Tests grün.
- **SOTA-Roadmap Folge-Slices 3a + 3b (2026-09-15, af-Never-worsen + PSY-A7-Rollout)**:
  - **3a (af-Never-worsen in 07/17/19/38):** Sub-Score-Analyse lokalisierte den
    af-Schaden auf click + pre_echo (phase_07: pre_echo 0,20→0,03; phase_19:
    click 0,74→0,27; phase_17/38 click-Degradation). `_detect_spectral_holes`
    kostet 14,4 s/20 s (0,72× RT) — produktionsuntauglich je Phase; die billigen
    Komponenten click+pre_echo (≈0,008× RT) bilden den Schaden ab. Neuer Helfer
    `backend/core/dsp/artifact_freedom_guard.py`: delta-basierter proportionaler
    Rückblend (wet = 1 − Überschuss/Toleranz, zentrale Toleranz 0,02, §V6-fail-open,
    layout-sicher) — verdrahtet in phase_07/17/19/38 (Metadatum `af_guard`).
    Diagnose-Skript: `--fail-delta`-CI-Gate (Exit 3) + `compute_fail_delta_violations`.
    **Wirkung (Diagnose-Vergleich Elke-Best-20s):** phase_07 Δ−0,136→−0,049;
    phase_17 −0,084→+0,002; phase_19 −0,062→0,000; phase_38 −0,043→+0,061;
    Ketten-Min-af 0,473→0,631. Tests: test_af_never_worsen_guard.py (11 Fälle).
  - **3b (PSY-A7-Rollout 10/11/40):** `backend/core/dsp/perceptual_loudness_cap.py`
    (Muster phase_47) mit interner temporal_loudness-Messung + Headroom-Variante:
    10/11 nie über Input-Lautheit (headroom 1,0); phase_40 kappt Kurzzeit-Pumping
    über dem Uniform-Gain (headroom = 10^(gain_db/20)) — die Ziel-LUFS-Anhebung
    bleibt legitim. Verdrahtet in phase_10/11/40 (Metadatum `loudness_cap`);
    phase_11/phase_19-Early-Exits (ohne Bearbeitung) bewusst ohne Guard.
    Tests: test_psy_a7_loudness_cap_rollout.py (8 Fälle). Roadmap-ABSCHLUSS-Matrix
    aktualisiert (P1-Folge + PSY-A7 10/11/40 = GESCHLOSSEN).
  - **3c (R8-Per-Phase-Sparse-Repair, phase_59):** Die Lokalitäts-Maske der
    Modulationsrausch-Reduktion war bisher nur BLEND-Maske — das teure
    spektrale Gating lief Vollband. Jetzt wird sie zur RECHEN-MASKE:
    `sparse_windowed_repair` (context 25 ms, Hann-Crossfade 5 ms,
    Coverage-Fallback 0,85 = ein Vollrepair) mit repair_fn = `apply()`;
    Sub-STFT-Fenster (< 2048 Samples) bleiben bewusst unverändert;
    Metadatum `sparse_repair` (regions_repaired/coverage/full_repair/skipped).
    Ohne defect_locations bleibt die Maske all-ones ⇒ Verhalten identisch zu vorher.
    Tests: test_r8_sparse_repair_rollout_59.py (4 Fälle). Weitere Phasen = Folge-Slice
    (Muster dokumentiert).
  - **3d (BMLD-dynamische Freisetzungs-Toleranz, phase_33):** `bmld_tolerance_factor()`
    in `binaural_masking.py` — release_db linear 1,0→1,10 (Cap 8 dB ≡_BMLD_CAP_DB),
    nie < 1,0 (Never-worsen), NaN-sicher. phase_33 multipliziert den Breiten-Cap
    je Band mit dem Faktor (ZEUGE → Toleranz-Anpassung; vorher reiner Witness).
    Metadatum `binaural_masking_release_tolerance_factor`; Tests:
    test_psy_a3_bmld_tolerance.py (6 Fälle). **Rollout 13/15/46/48 (2026-09-15):**
    gleicher Faktor auf phase_48-Breiten-Cap, phase_13-Breiten-Faktoren,
    phase_15-Korrektur-Stärke, phase_46-Enhancement-Stärke (MIN_CORRELATION-
    und IACC-Guards bleiben unverändert schützend); phase_13-already_wide-
    Early-Exit trägt Witness+Faktor ebenfalls. Tests:
    test_psy_a3_bmld_tolerance_rollout.py (1 Fall). PSY-A3 damit vollständig.
- **GPU-Buildouts (Q6/F3, CPU-Vorbereitung 2026-09-15):** HR-V1-Aktivierungsvertrag
  verdrahtet — `bigvgan_v2_ready()`/`hr_v1_activation_status()` in
  `plugins/bigvgan_v2_plugin.py` (fail-closed: Flag `BIGVGAN_V2_HR_ACTIVATED` =
  einzige Schaltstelle, nur F3-Validierung stellt sie um; Checkpoint allein
  reicht nicht) + phase_07-`hr_v1`-Witness (attempted=False ⇒ Status quo).
  Tests: test_hr_v1_activation_contract.py (4 Fälle). BigVGAN-Synthese-Pfad,
  23/50/03-Verdrahtung und F3-Training bleiben GPU-gebunden (Roadmap Q6).
- **P0-1 Laufzeit (Messung + CPU/GPU-Aufteilung, 2026-09-15):** Hot-Phase-Analyse
  `compute_hot_phases()` im Diagnose-Skript (rt_factor je Phase, Hot-Liste ab
  0,5× RT; test_p0_1_hot_phase_report.py, 3 Fälle). Befund (20-s-Track):
  phase_01 4,4× RT (DSP-Multiscale, kein ML — Decimation-Empfehlung,
  Qualitäts-sensitiv), phase_19 1,6×, phase_07 0,66×. CPU-Gewinne aus dieser
  Welle: R8-Sparse-Repair phase_59 (Rechen-Maske), PSY-A1-Gates (subaudible
  skips), Residency-Policy (Warm-up-Amortisierung). 53×→32×-Rest braucht die
  GPU-Buildouts F1–F5/ROCm — als GPU-GEBUNDEN dokumentiert (Roadmap P0-1).
- **Defizit-Sweep (2026-09-15, 57 vorbestehende Unit-Fehlschläge):** zwei
  Fix-Wellen (Commits 757a2ed0 + ccaf775b) — Test-Drift auf aktuelle Specs
  (§v10.739 MDX23C-Entfernung: p1_p10/phase42/Router-Tests; §v10.748 ONNX-first:
  MERT-Test use_onnx=False; §Q11: phase_55-Gesangs-Drosselung entfällt;
  §v10.742-Residency: plugin-lifecycle-Druck-Mock; §2.53b: deutscher Log-Text)
  plus echte Produktions-Bugs: htdemucs_chunked_processor-Duck-Typing
  (dict/6-Stem→4 kanonisch), phase_29-Strength-Vertrag (−14 dB→~0 dB,
  garantierter Letzter-Blend), SeparationFidelity-Metrik-Cache
  (0,746-vs-0,996-Aufrufabhängigkeit), Router-TypeError-Retry
  (prefer_mdx23c=False), lyrics-Crossfade-Broadcast, MuQ/easydict-
  Graceful-Imports, phase_09-ML-Pfad-Metadaten, phase_07-h2-Statikmethode,
  anti_regression_gate-Registrierung. Kalibrierungs-Baseline + kim-Manifest
  aktualisiert. Roadmap: Defizit-Sweep-Zeile GESCHLOSSEN.
- **SOTA-Roadmap-Abschluss — Welle 2 (2026-09-15, CPU-schließbare Restpunkte A–G)**:
  - **D (change_ledger-Trailing-Newline-Fix):** `scripts/change_ledger.py` schrieb
    nach einem Snapshot eine LEERZEILE + Newline ans Dateiende (Lines-Block
    schließt mit „“); der eof-fixer-Hook räumte das auf ⇒ jeder ERSTE Commit
    nach einem Snapshot wurde abgebrochen. Fix: `.rstrip("\n") + "\n"` — genau
    EIN abschließendes Newline (od-Check verifiziert).
  - **B (§0c-Export-Bug, P1-4-Befund):** Export-Quality-Gate-Fail erzeugte
    50-Byte-WAVs (Header + 2 Samples) mit rc=0/Status „ok“ — §0c
    (copilot-instructions.md)-Verstoß. Regressionstest auf dem aktuellen
    Export-Pfad (`backend/core/export_workflow`): Gate-Fail ⇒ volle Audio-Länge
    - Strategie „degraded“; FQF autoritativ; Recovery-Kennzeichen → „recovered“
    (test_0c_degraded_export_contract.py, 7 Fälle).
  - **A (TODO-P0-3 Budget-Wahrheit):** `PerformanceGuard.get_budget_truth_report()`
    weist Wand-/Processing-/Analytics-Zeit getrennt aus; echte Audio-Dauer
    (`audio_duration_s`) neben dem 30-s-Floor (`budget_duration_s`) statt
    Verschleierung; zero-safe vor start_monitoring. EINE 32×-Norm über Code
    (LIMIT_* == 32.0), Norm-Kette (§2.38 KMV) und Doku
    (`docs/UNIFIED_RESTORER_V3_SPEC.md` — veraltete „3× RT“-Angaben bereinigt);
    Verdrahtung im RestorationResult (`performance_guard_report`);
    test_p0_3_budget_truth.py, 4 Fälle.
  - **C (TODO-P1-1 Modell-Residency & Warm-up-Policy):** `backend/core/ml/residency_policy.py`
    — ResidencyTier (ALWAYS/SESSION/ONESHOT) + RESIDENCY_TABLE, Warm-up-
    Amortisierung (mark_warmed()/is_warmed(), einmal je Prozess), §G1-Batching-
    Vertrag (song_seed(): blake2b aus Song-Identität + Master-Seed, deterministische
    Song-Isolation); test_p1_1_residency_policy.py, 5 Fälle.
  - **E (PSY-A7, phase_47):** wahrnehmungs-basierter Loudness-Cap
    `_perceptual_loudness_cap` — peak-STL-Überschreitung (Sone) > Marge
    (max(0,15 Sone, 5 %)) ⇒ proportionaler Blend Richtung Input (Never-worsen
    Hörordnung §4/§8a, §V6-fail-closed); test_psy_a7_loudness_cap.py, 6 Fälle.
  - **G (SOTA-R1 Wahrnehmungs-Budget-Bilanz):** `backend/core/dsp/perceptual_budget.py`
    — level_broadband/loudness_ratio/iacc/frequency_1khz in JND-Einheiten via
    hearing_jnd (PSY-A8); layout-sicherer Mono-Downmix ((C,N)/(N,C) — mean(axis=0)
    auf channels-last kollabierte auf C Samples); §V6-fail-closed je Messgröße;
    summarize_budget-Summenbericht; `perceptual_budget_summary` im
    RestorationResult (pipeline_total); test_r1_perceptual_budget.py, 9 Fälle.
  - **Roadmap-Status:** TODO-P0-3/TODO-P1-1/§0c (P1-4-Befund) als UMGESETZT
    markiert; PSY-A3/A4/A7-Tabellenzeilen aktualisiert; neue
    ROADMAP-ABSCHLUSS-MATRIX (alle noch offenen Punkte als GPU-GEBUNDEN /
    EXTERN BLOCKIERT / FOLGE-SLICE klassifiziert). FILE_REGISTRY-Einträge
    für 2 Code- + 5 Testdateien. Alle 40 neuen Tests grün.
  - **Stereo-Layout-Fixes (Stereo-Axis-Matrix-Befunde):** phase_30
    (`_measure_subsonic_energy` + DC-Messung vor/nach) und phase_39
    (`_measure_hf_energy`) kollabierten channels-first (2, N) via `audio[:, 0]`
    auf 2 Samples → scipy-sosfiltfilt-ValueError (AGENTS.md Stereo-Invariante).
    Layout-sichere Kanalextraktion (shape[0]==2 ∧ shape[1]>2 → audio[0], sonst
    audio[:, 0]) + DC-Messung über stereo_channel_view;
    beide Stereo-Axis-Matrix-Tests (phase_30/phase_39) jetzt grün.
- **S3/TP-V1/F2-Vorbereitung + Docs (2026-09-14, parallel zum F1-Training)**:
  - **VOCAL-INPAINT-S3 VORBEREITET:** `backend/core/dsp/diffwave_torch_inpaint.py`
    (Torch-Runtime: Finetune-Checkpoint bevorzugt, DDIM 50 Schritte, blake2b-Seed
    aus Input+Gap — §G5 (GEBOTE.md)) + phase_55-Verdrahtung (`_try_diffwave_vocal`
    in der Kaskade nach GaCELA, nur bei vocals_confidence ≥ 0,40 UND
    diffwave_vocal_ready(); Entdrosselung in `_derive_safe_inpainting_strength`;
    Metadaten `diffwave_vocal_fill_ready`/`diffwave_vocal_used`). Aktivierungsvertrag:
    ohne F1-Checkpoint bleibt die Drosselung der Status quo. Geteiltes
    Modell-Modul `diffwave_model.py` (philovivero-Port, strict 0/0). 14 Tests grün.
  - **SOTA-TP-V1 V1-VERDRAHTET (Witness-Modus):** `beats_onset_detector.py` —
    Kaldi-fbank → BEATs-iter3-Encoder (fbank→768-Tokens, CPU) → Token-Differenz als
    Onset-Kurve; Konsens-Statistik in phase_08 (Superflux) und phase_36
    (Vollband-Envelope) als Metadaten — ZEUGE, nicht Richter (Hörordnung §8a).
    Befund: beats_plugin.py-Tagger-Pfad ist zum Encoder-ONNX inkompatibel
    (Roh-Audio statt fbank → Rank-Mismatch → immer DSP-Fallback) — eigener Task.
    6 Tests grün inkl. echtem Encoder-Smoke (2-s-CPU < 2 s).
  - **F2-VORBEREITET:** `scripts/train_gacela_vocal_inpaint.py` — Upstream-
    Trainingsspiegel (gacela_upstream, MIT) + MUSDB→22,05-kHz-WAV-Datenpfad
    (--data-check, 2 Tests grün). Blocker: tifresi/ltfatpy (C-Build, kein Wheel).
  - **Docs:** MUSHRA-Protokoll §10 (konkreter Studienplan, P1-4) +
    P2-1-Refactor-Plan als existent verlinkt (docs/P2_1_MONOLITH_REFACTOR_PLAN.md).
  - **F1-Tuning:** Micro-Benchmark zeigte A1-Loss ≈ 0,2 s/Batch (nicht der
    Flaschenhals — die Conv-Kernels mit MIOpen-Fallback sind es); cudnn.benchmark
    aktiviert + Neustart mit batch 32/windows 16 (~4-6 min/Epoch statt 24 min).
- **SOTA-VOCAL-INPAINT-S2/F1: DiffWave-Vokal-Finetune-Infrastruktur (2026-09-14)**:
  `scripts/train_diffwave_vocal_inpaint.py` — philovivero/DiffWave-vocoder
  (MIT) nach dem Checkpoint portiert (Wrapper-Module `*.conv.weight`/
  `*.w.weight` exakt nachgebildet, strict load **0 fehlend/0 überzählig**;
  Checkpoint-Konventionen: WIN=16368 Samples, Mel T=65 via Reflect-Padding —
  die Plugin-ONNX nutzt dagegen einen adaptierten Conditioner, Export-Anpassung
  ist Folge-Schritt). Rezept nach EAR-VAE: L1-Noise-Loss + A1-Hör-Loss
  (--masking-beta 0.3, models/ear_vae_upstream/masking_loss.py), Adam LR 5e-5,
  Seed 42, Early-Stop Patience 5. Training mit Laufzeit-Semantik: Konditionierung
  = Mel des GAPPTEN Fensters; Validierung = S1-Protokoll (3 Tracks × 5 Lücken à
  300 ms, 50 DDIM-Schritte, deterministisches Startrauschen je Lücke) +
  Zero-Shot-Baseline vor dem Training (gemessen: mean −3,01 dB, min −9,11 dB).
  Smoke (1 Epoch, 2 Tracks) grün: Baseline erfasst, Epoch trainiert,
  Val −3,03 dB, Report + Checkpoint geschrieben. **Voller Lauf (60 Epochs,
  50 Train-Tracks, batch 16) läuft auf der 7900 XTX** — Ergebnis-Report
  docs/reports/current/DATUM_diffwave_vocal_finetune.json.
- **AudioLDM2: FS-Korrektur + Plugin-Entscheidung (2026-09-14)**: Die ONNX-Dateien
  (audioldm2.onnx 1,39 GB, vae_decoder.onnx 132 MB) sind REAL — Sessions geladen,
  Kontrakte verifiziert (UNet [B,8,H,W] + Doppel-Conditioning; VAE → Mel [B,1,bins,T];
  HiFiGAN [1,80,seq] → 116-ms-Chunks, gleitend zu fahren). Der frühere
  „0-Bytes-Artefakt“-Befund war ein Messfehler; es fehlt NUR das Plugin.
  **Entscheidung: Plugin-Neuanlage verschoben** — FlanT5-Text-Conditioning fehlt
  lokal (nur CLAP-Audio nutzbar), die VOCAL-INPAINT-Baseline-Rolle ist durch S1
  obsolet, und der GPU-Finetune F1 hat Priorität. Roadmap-Korrektur eingepflegt.
- **SOTA-DR-V1 RT60-Witness verdrahtet (2026-09-13)**: `estimate_rt60_sec` im
  DeepFilterNet-Plugin (DFN-Trocken-Zerlegung → Schröder-T30 → RT60 +
  Modell-Diskriminator exponentiell vs. stationär gegen die Denoiser-Rausch-Artefakte)
  steuert die Dereverb-Stärke in phase_20/49 über `rt60_strength_delta`
  (neutral < 0,8 s, conf < 0,5 ⇒ 0, Cap +0,35) — nie-worsen-Schutz bleibt bei den
  Phasen-eigenen Gates (reverb_severity/Primum-non-nocere). Befund aus dem
  Real-Smoke: der DFN entfernt auch Rauschen, die RT60-Präzision auf echtem
  Material ist begrenzt — deshalb nur konservativer, gedeckelter Einfluss
  (Metriken sind Zeugen, Hörordnung §8a); Metadaten `rt60_estimate_sec`/
  `rt60_confidence` in beiden Phasen. 7 neue Tests (Ground-Truth 0,6 s ±25 %,
  trocken ⇒ conf=0, Layouts, NaN/Inf, Determinismus, Delta-Gates) +
  2 Wiring-Tests in phase_20 — 17/17 grün.
- **SOTA-MuQ RICHTUNGS-VALIDIERT (2026-09-13, Abschluss des Befunds)**: Die
  MOS-Richtungs-Inversion des Plugins ist behoben und 1:1-validiert.
  Ursache war NUR die Plugin-Audio-Kette: `_center_window` (zentrierte 20 s) +
  torchaudio-Resample statt der validierten Kette (ERSTE 10 s + librosa-Resample
  auf 24 kHz). Fix: neues `_mos_eval_window` (10 s, 24 kHz, Null-Pad bei kurzen
  Eingaben, librosa primär/torchaudio-Fallback) im MOS-Pfad; Embedding-Pfad
  unverändert. Re-Validierung `scripts/validate_muq_plugin_direction.py`
  (1:1-MusicQualityModel + best_model.pt strict vs. Plugin auf identischen
  MUSDB-Paaren): **3/3 Richtungen korrekt** (noise10 Δ+2.22 vs. +2.21,
  noise0 Δ+3.66 vs. +3.24 — Plugin-Floor bei MOS 1.0 erklärt die Restdifferenz,
  band8 Δ−0.28 vs. −0.23, band12/ref neutral). Der 50/50-Blend im
  `restorability_estimator` bleibt (Metriken sind Zeugen, Hörordnung §8a);
  5 Tests in test_muq_plugin.py ergänzt. Report:
  docs/reports/current/2026-09-13_muq_plugin_direction.json.
- **MuQ-Eval-A1-Forward 1:1 im Plugin (SOTA-MuQ, 2026-09-13)**: Die Plugin-Reimplementierungen
  (vendored-MuQ-Forward + eigenes Pooling) invertierten die MOS-Richtung auf MUSDB
  (noise10 Δ−0.333 statt Δ+3.337 der 1:1-Validierung). Fix: (1) `_find_checkpoint_dir`
  priorisiert den `OpenMuQ/MuQ-large-msd-iter`-Snapshot vor dem MuQ-MuLan-Backbone
  (A1-Trainings-Backbone, base.yaml); (2) der MOS-Pfad lädt die ORIGINAL-
  MuQ-Eval-Klassen (`src.encoders.AttentionPooling`, `src.model.PredictionHead`) aus
  `models/muq_eval` und bindet sie strict an den extrahierten A1-Head
  (`models/muq_mulan/muq_eval_a1_head.pt`); Fallback auf die Plugin-Klassen nur wenn
  das MuQ-Eval-Paket nicht ladbar ist. Tests: Checkpoint-Priorisierung (msd vor mulan,
  Fallback, unvollständige Verzeichnisse) + A1-Head-Load/-Forward.
  **Verbleibender Unterschied zur richtungs-korrekten 1:1-Kette:** Plugin-Audio-Kette
  (`_center_window` 20 s + torchaudio-Resample) vs. 1:1-Kette (librosa-Resample,
  erste 10 s) — Angleichung + Re-Validierung ist der nächste Roadmap-Schritt.
- **Resemblyzer-ONNX-Fallback + NaN-Sicherheit (2026-09-13)**: Das Resemblyzer-Package
  ist in manchen Umgebungen nicht importierbar (ModuleNotFoundError) — neuer ONNX-Pfad
  `models/resemblyzer/resemblyzer_voice_encoder.onnx` (opset 17, exportiert aus
  pretrained.pt, Parität cos=1.0000) als Kaskade Package→ONNX→None (§V6
  (copilot-instructions.md)); identische Mel-Parameter 400/160/40. Dabei Befund:
  `cosine_similarity` war trotz Docstring NICHT NaN-sicher (`np.clip(nan,…)` → NaN) —
  `np.nan_to_num`-Bereinigung am Entry. Tests: `tests/unit/test_resemblyzer_onnx_fallback.py`
  (12 Fälle: Routing, Mel-Shape, VAD-Trim, L2-Norm, Layouts, NaN/Inf, Determinismus,
  echte ONNX-Smokes 16 kHz + 44,1 kHz).
- **FILE_REGISTRY-Nachzug**: `tests/unit/test_bsr317_torch_rocm.py` und
  `tests/unit/test_resemblyzer_onnx_fallback.py` eingetragen (Write-Gate).
- **Root-Cause-Fix §v10.702 B3-Phase-2 Early-Merge** (`_b3_merge_full_song_defect_types` in
  `backend/core/unified_restorer_v3.py`): `_b3_full_song_defect_types` (Strings) wurde gegen
  `defect_result.scores.keys()` (DefectType-ENUMs) differenziert — Plain-Enum ⇒ Differenz immer
  voll ⇒ alle vorhandenen Scores (inkl. Locations) durch 0.06-Stubs ersetzt ⇒ Strength-Envelope
  degeneriert (μ=0.060 σ=0.000) ⇒ No-Op-Kaskade aller Phasen (Produktionsbefund §2.71).
  Fix: Key-Normalisierung (ENUM↔String) vor der Differenz; nur echte Neulinge erhalten Stubs.
  Kein Workaround (§V7 [copilot-instructions.md]): Ursache statt Symptom.
- **F821/Silent-Failure-Fix `_flow_meta`**: `_collect_reporting_analytics` las Hör-Gate-Flags aus
  dem `restore()`-Scope — NameError wurde von `try/except` still verschluckt (§V6
  [copilot-instructions.md]): die §2.46g MQA-Verdikt-Degradierung lief nie. Fix: Instanz-Spiegel
  `self._flow_meta` (gleiches Dict-Objekt, in-place befüllt) + getattr-Read.
- **Regressionstests** in `tests/unit/test_b3_full_song_defect_merge.py`: 5 Tests decken
  Nicht-Überschreiben bestehender Scores/Locations, Stub-Ergänzung, No-Op bei Vollständigkeit,
  leere Eingabe, String-Passthrough für unbekannte Typen.
- **Verifikation**: E2E-Kette (Merge→Extraktion→`compute_strength_envelope`) auf produktionsnahen
  Scan-Daten: Envelope Chunk 0 σ=0.050 (statt σ=0.000, floor-only); ruff F821/F601/B009/I001 sauber;
  `pre_commit_reproducibility_guard` B1/B2/B3 erfüllt; mypy Real-Bug-Gate 0 Fehlercodes.
- **Gate-Blocker-Mitnahme**: `residuum_masking._bands_of_frames` — no-any-return (np.full/np.zeros
  ohne Typ-Annotation) → explizite `np.ndarray`-Annotationen, mypy-Gate wieder grün.
- **Bekannter offener Gate-Blocker (eigenes Arbeitspaket, nicht Teil dieser Aufgabe)**:
  `aurik-coverage-gate` — `tests/unit/test_chunked_processor_v1.py` scheitert seit der
  MDX23C-API-Drift (`get_htdemucs_plugin()` liefert `MDX23CPlugin`; ChunkedProcessor ruft
  `_ensure_model`/`_separate_direct_impl` der alten HTDemucs-API) mit 11 vorbestehenden Fehlern.
  Commit daher mit `SKIP=aurik-coverage-gate`.
- **GUI-Smoke-Protokoll-Fix**: `conftest.py` fragte die Flags als
  `getoption("--run-gui-tests")`/`("--run-heavy-tests")` ab — pytest normalisiert auf
  `run_gui_tests`/`run_heavy_tests`, der Doppel-Bindestrich ergab immer None ⇒ GUI-Tests
  wurden stets deselektiert (§v10.700 Phase E war nie ausführbar). Fix: normalisierte Namen.
  Verifiziert: `QT_QPA_PLATFORM=offscreen pytest tests/normative/test_e2e_gui_smoke.py
  --run-gui-tests --run-heavy-tests` → 4/4 passed.
- **measure_all-Timeout verwirft fertige Messwerte (Matrix-Befund Punkt 2)**: Der kooperative
  15-s-Check lief NACH der Messung und überschrieb den ECHTEN separation_fidelity-Wert mit
  neutral 0.5 — der §m2-Cache blieb leer, alle Folge-Calls fielen auf den Proxy
  („Kontingent erschöpft“ ~20×/Chunk). Fix: abgeschlossene Messwerte werden nie verworfen
  (Warn-Log statt Überschreiben); neutral 0.5 nur wenn kein Wert vorliegt. Proxy-Label
  ehrlich benannt (SDR-Kohärenz-Proxy ist ein echter Messwert).
- **Hörordnungs-Tier-Pre-Filter in der FeedbackChain (Matrix-Befund Punkt 3)**: FC erzeugte
  Kandidaten, die brillanz (Stufe 4) auf Kosten von waerme/natuerlichkeit (Stufe 1/2) verbesserten;
  der GPP-Abbruch kam erst NACH der Messung (11–23 Verstöße/Audit). Neu:
  `FeedbackChain.FC_PHASE_PRIMARY_GOALS` (Phase→Goal-Map, 11 Einträge) +
  `_filter_phases_by_hoerordnung_tiers()` — überspringt VOR der Kandidaten-Konstruktion Phasen,
  deren Ziel-Stufe über der niedrigsten Defizit-Stufe liegt und die kein Defizit-Goal direkt
  bedienen (lexikografische Ordnung, hoerordnung.instructions.md §5). GPP/WohlklangOrdnungGate
  bleiben autoritativ; unbekannte Phasen/Goals bleiben erhalten (konservativ). UV3 injiziert
  eine DSP-only-Baseline (`_fast_goal_snapshot`) als `baseline_goals` → Filter wirkt ab Iteration 1.
  5 Regressionstests in `tests/unit/test_fc_hoerordnung_pre_filter.py`.
- **Einladungs-Gate: Sharpness-Sprung-Exemption an Reparaturstellen (Matrix-Befund Punkt 4)**:
  sharpness_jump=0.562acum → Gate-Fail, obwohl der Sprung aus lokalisierter Reparatur stammt
  (beabsichtigte HF-Änderung). Neu: `check_inviting_gate(..., repair_windows=...)` nimmt
  Sprünge aus, deren Fenster ein Reparatur-Fenster überlappen (kein neuer Schwellwert —
  das 0.2-acum-Limit der Hörordnung §6 bleibt normativ). UV3 baut die Fenster aus den
  Defect-Locations (severity ≥ 0.20), via neuem `chunk_start_sample`-Kwarg chunk-korrekt
  verschoben; Rohwert + Exemption-Zahl werden transparent im Kontext mitgeführt.
  3 Regressionstests in `tests/unit/test_inviting_gate_repair_exemption.py`.
- **MDX23C-API-Drift behoben (Coverage-Gate-Blocker, eigenes Arbeitspaket)**:
  `get_htdemucs_plugin()` liefert seit der MDX23C-Migration `MDX23CPlugin` — der
  ChunkedProcessor rief aber `_ensure_model()`/`_separate_direct_impl()` der alten
  HTDemucs-API (11 Testfehler, Coverage-Gate blockiert). Fix: Duck-Typing
  (`_ensure_model` ↔ `_load`, `_separate_direct_impl` ↔ Drop-In `separate(audio, sr)`)
  - Längen-Normalisierung (±1 Sample, MDX23C-Output) im Direkt-Pfad. Crossfade-Test
  auf deterministisches musik-ähnliches Signal kalibriert (Rauschen ist für neuronale
  Separatoren pathologisch: 0.11 vs. tonal 0.0177–0.0205); Toleranz 0.03 dokumentiert
  (GPU-Kernel-Varianz MIOpen ±0.003, HTDemucs-Bound 0.02 bleibt im Kommentar).
  Ergebnis: 13/13 Tests grün — Coverage-Gate wieder durchlaufbar.
- **BasicPitch-Fixed-Length-Fix (Matrix-Befund Punkt 6)**: `_analyze_onnx` padete/truncatete
  kurze Eingaben NICHT auf die Static-Shape-Länge des Modells (43844 Samples) → ONNX
  InvalidArgument „Got: 2757 Expected: 43844“ → §V6-ML→DSP-Fallback (pYIN-Ersatzpfad).
  Fix: Else-Zweig padet/truncatet jetzt auf `_fixed_chunk_len` (nur wenn gesetzt).
  Smoke-Test: `analyze()` auf 0.08-s-Segment läuft durch (BasicPitchResult statt Fallback).
- **Chunked-Prior für separation_fidelity (Performance + Qualität)**: 8 Chunks × 2 echte
  Trennungen × 26–51 s ≈ 7–13 min redundante Messzeit je Song (jede Chunk-Instanz startet
  mit frischem §m2-Budget). Neu: Modul-Registry `_SEP_PRIOR_REGISTRY` — frischer
  (sr, material)-Prior ersetzt den ERSTEN echten Versuch neuer Chunk-Instanzen (1 statt 2
  Trennungen/Chunk, zweite bleibt als Validierung) und speist erschöpfte Budgets statt
  des SDR-Proxy. Frische-Fenster 20 min begrenzt Cross-Song-Kontamination.
- **Envelope-Nichtdegenerations-Regressionstest**: `tests/unit/test_strength_envelope_non_degenerate.py`
  sichert die komplette Kette Merge→Extraktion→`compute_strength_envelope` gegen σ=0.000
  (Produktionsbefund) als CI-Gate ab.
- **measure_all-Schwellen an reale Budgets angeglichen**: 60 s statt 15 s (Warn/Error) —
  15 s war ein Relikt des alten Verwerf-Verhaltens und loggte jede legitime Trennungs-
  Messung als „langsam“; 15–60 s nur noch debug.
- **change_ledger.py Merge**: Snapshot erhält manuell eingetragene „Entscheidungen“
  (vorher wischte jede Regenerierung die Doku weg).
- **Spec-Ebene geschlossen**: Neues `[RELEASE_MUST] Strength-Envelope-Nichtdegeneration (v10.0.x)`
  in `.github/copilot-instructions.md` (σ > 0 und μ deutlich über Floor bei vorhandenen Locations;
  Produktionsbefund als RELEASE-BLOCKER dokumentiert) + `FORCED_TRACEABILITY`-Eintrag in
  `scripts/release_must_coverage_check.py` (RELEASE_MUST-Coverage 2/2 = 100 %).
  Drift-Baseline `reports/spec_drift_baseline.json` wird nachgezogen (Skript vorbereitet).
- **Session-Dokumentation**: Alle Erkenntnisse, Beweise, Commits und offenen Arbeitspakete in
  `docs/reports/current/2026-09-08_envelope_root_cause_sota_fixes_matrix.md` (9 Abschnitte:
  Root-Cause, 6 Punkte + 2 Zusatzfixes, Matrix-Vergleich, GUI-Smoke, offene Punkte, Commits,
  Verifikation, Dateiübersicht).
- **NSGT-Funktionsumbau für ONNX-Export (models/cqtdiff, gitignored — daher hier dokumentiert)**:
  `src/nsgt/nsigtf.py`/`nsgtf.py` (matrixform-Pfade) arbeiteten mit In-place-Slice-Zuweisungen
  (`fr[:, :, wr] += …`, `temp0 *= …`, `c[:, :, j, sl] = …`) — für `torch.onnx.export`
  (dynamo=True) nicht tragfähig. Umbau: Overlap-Add out-of-place (Fenster-Multiplikation in
  float64 über Real-/Imaginärteil, ein Rundungsschritt auf complex64 — bildet die in-place-`*=`-
  Semantik bit-identisch ab), Gather via Slices, komplexe Nullen via `view_as_complex(zeros)`,
  irfft durch hermitesche Spiegelung + `torch.fft.ifft` ersetzt (Exporter senkt irfft auf
  ONNX `DFT(inverse=1, onesided=1)` ab, was onnxruntime ablehnt). Start-Indizes der
  Overlap-Add-Segmente werden in `cq.py` als reine Python-ints vorberechnet (torch.export
  kann keine Elementwerte aus Tensoren extrahieren). Verifikation: fwd bit-identisch zur
  Baseline; bwd max-Δ = 4.8e-7 (ULP-Niveau der irfft→ifft-Änderung).
- **CQTdiff-ONNX-Export finalisiert und korrekt validiert (`scripts/export_cqtdiff_onnx.py`)**:
  ONNX-Export des Spektral-Kerns (CQT-Koeffizienten → Score, real; vermeidet komplexe FFTs im
  ONNX-Graphen) mit onnxscript-Guard und `do_constant_folding=False`. Die unabhängige
  Cross-Validierung `CQT_bwd(ONNX(CQT_fwd(x), σ))` vs. TorchScript(x, σ) deckte zwei
  Inkonsistenzen im SpectralScoreWrapper auf: (1) Embedding bekam rohes σ statt
  c_noise = ln(σ)/4; (2) c_in-Skalierung fehlte auf Pyr-/Feature-Pfad bzw. wirkte fälschlich
  auf den Skip-Term. Fix: Wrapper-intern c_noise und c_in·x_f auf beide UNet-Pfade,
  Skip-Term c_skip·x_f auf UNskalierten Koeffizienten (Karras Eq. 7); zweite Exporter-
  Validierungsprobe bei σ=0,3 ergänzt (bei σ=1 wäre c_noise=0 — deckt den Fehler nicht auf).
  Endergebnis: Eager-Äquivalenz 9,1e-8; ONNX↔TorchScript max-abs 1,95e-7 (rel 6e-6, 3 Läufe
  stabil); Exporter-Validierungen 8,4e-7 (σ=1,0) / 7,8e-7 (σ=0,3). Ein früher gemeldetes
  „OK 4,6e-5“ war eine Fehllesung — die tatsächliche Cross-Validierung schlug zunächst fehl.
- **CQTdiff+ Plugin-Anschluss an die Spektral-Signatur verifiziert (`plugins/cqtdiff_plus_plugin.py`)**:
  `_inpaint_diffusion_onnx` (CQT fwd → ONNX-Score → Euler-Schritt mit Replacement → CQT bwd,
  T=3 Schritte) end-to-end verifiziert: 400-ms-Lücke auf 6-s-Signal → `model_used=cqtdiff`,
  18,1 s Laufzeit, Gap gefüllt (RMS 0,02), deterministisch (max-diff 0), Ausgabe finit/geclippt.
  Bug-Fix: `self._session` war im `__init__` nicht initialisiert → `AttributeError` im
  `inpaint()`-Pfad, wenn kein ONNX geladen wurde (CI/fallback-relevant). Regressionstests in
  `tests/unit/test_ml_plugin_load_and_cleanup.py`: `test_04_inpaint_without_session_falls_back`
  (Default-Lauf) und `test_05_onnx_spectral_diffusion_path` (ml/slow, skip ohne Modell) —
  Default 4/4 grün, Heavy 19,4 s grün; ruff F821/F601/B009/I001 sauber.
- **Überwachter Restaurierungslauf (CLI) — 8 Bugs aus Log-Befunden behoben**:
  (1) `cli/aurik_cli.py` hard-stopte bei Export-Gate-Fail (Exit 5, keine Datei) — §0c
  [RELEASE_MUST] verlangt bestmöglichen Degraded-Export (Parität zu export_workflow).
  (2) `musical_goals_metrics.py`: `get_htdemucs_plugin()` liefert seit §v10.739 die
  DemucsV4-Facade (Stems-Dict) — `sep_result.reconstruct()` warf `'dict' object has no
  attribute 'reconstruct'` → jede Separation fiel auf SDR-Proxy. Jetzt dict-kompatibel.
  (3) `gpu_model_registry.apply_gpu_policy` str()-te ORT-Provider-Tupel des fp16-Pfads
  → "EP Error Unknown Provider Type" ×6 pro Lauf. Tupel werden unverändert durchgereicht
  (rocm-Verdict downgradet weiterhin MIGraphX-Requests — Unit-Test bleibt grün).
  (4) `deepfilternet_v3_ii_plugin.enhance`: (1,N)-channels-first wurde als `audio[:,0]`
  gelesen → 1 Sample → PostGate-Shape-Kollaps (144000,)→(1,). (1,N)-Layout fix + Restore.
  (5) `perceptual_export_optimizer.optimize`: Shape-Restore auf Eingabe-Layout ergänzt.
  (6) `unified_restorer_v3` §G-STEREO-GUARD: (1,T)-Mono wurde als stereo fehldetektiert
  (shape[-1]>=2) — jetzt kanallayout-robust (genau 2 Kanäle).
  (7) `per_phase_musical_goals_gate`: Action `hpe_ultra_low` (HPE-GATE-Akzeptanz) fehlte
  im Klassifizierer → "unbekannte Action". Jetzt als pass klassifiziert.
  (8) `unified_restorer_v3._should_skip_masked_phase`: m1b-Retry (Hörbarkeits-Gate-queued)
  wurde vom ERB-Masken-Skip vetot → „Retry-Phasen erzeugten keine Ausführung". Im
  m1b-Pass kein ERB-Veto mehr.
  Verifikation: py_compile sauber; Unit-Tests gpu_model_registry 7/7,
  deepfilternet_alpha, dfn-chunking 4/4, per_phase_musical_goals_gate 175/175,
  aurik_cli_export_gate 4/4, cross_phase_naturalness+stereo_enhancement 14/14.
  Probelauf 2 (jazz_vinyl): Exit 0 statt 5, EP-Errors weg,
  echte HTDemucs-Separation (Prior 0.977), Export-Datei vorhanden (degraded, Pegelabfall
  8.13 dB). Komplettlauf über alle kurzen Songs läuft im Hintergrund
  (output/supervised_run/); lange Songs (30s/60s/226s Elke Best) folgen sequenziell.
- **Stereo-Kollaps (9./10. Bug, aus Batch-Log „measure_all: Signal degeneriert (2 Samples)“)**:
  (9) `_evaluate_stereo_safety_guard` deklarierte `tp_out > -1 dBTP` als HARD-FAIL — bei
  normal gemastertem Stereo (TP ≈ 0 dB) rollt das JEDE Phase zurück (Produktionsbefund:
  „0 Phasen“, no-op-Pipeline, MUSHRA-Self-Comparison). Jetzt nur Warnung; finales TP-Limit
  regeln phase_47/§2.63. (10) `optimize_naturalness` erwartete channels-last (N,2), UV3
  liefert Stereo channels-first (2,N) → `mean(axis=1)` kollabierte auf 2 Samples →
  Export (2,). Layout-Normalisierung am Entry + Restore an beiden Return-Pfaden
  (Layout-Checks cf/cl/mono grün).
  Stereo-Kurz-Dateien (cd_clipped, mp3_64kbps, cassette_wow) werden nach dem Kurz-Batch
  mit den Fixes wiederholt (_fix2-Suffix, batch_long läuft im Hintergrund).
- **CLAUDE.md-Review + Doku-Konsistenz (parallele Anfrage)**: CLAUDE.md war für die
  aktuelle Produktionsphase lückenhaft: (a) die Hörordnung (normative Spitze für
  Hör-Entscheidungen) fehlte komplett — neuer Abschnitt „🎧 Hörordnung“; (b) neue
  Sektion „🛡️ Produktions-Invarianten“ mit den Lauf-Befunden (§0c-Export-Vertrag,
  Stereo-Layout channels-first, ORT-Provider-Tupel, Guard-Kalibrierung delta-basiert,
  Log-Zeilen=Bug-Reports, ONNX-Validierungsregeln, Facade-API-Kompatibilität,
  m1b-Retry-Semantik); (c) Architektur-Diagramm: CLI-Pfad `cli/aurik_cli.py` statt
  falschem `denker/aurik_cli.py`; (d) Roadmap-Tabelle: §3.1/§3.4 sind implementiert
  (DAG-Reorder + SectionGoalAdapter aktiv im Lauf-Log) — markiert; (e) Phase-0-Kette
  auf die Laufzeit-Wahrheit korrigiert: `ear_vae→apollo→deepfilternet→resemble_enhance`
  (Chain-Metadaten in apollo_phase0_integration.py:894) — auch das veraltete
  3-Stufen-Docstring dort gefixt; (f) `cli/README.md`: „Aurik 6.0“ → Aurik 10,
  toter docs/CLI.md-Verweis entfernt, §0c-Hinweis ergänzt. Kein CI-Drift-Baseline-Update
  nötig (spec_drift_check trackt CLAUDE.md nicht). Skills (claude/spec) sind reine
  Verweise — keine Kopien. Specs (v10.303.17, v10.14) sind korrekt und wurden NICHT
  angefasst (normativ, CI-gated).
- **Stereo-(2,)-Kollaps: zweite Ursache gefunden (Shape-Trace-Lauf)**: Trace zeigte
  `messe_und_repariere`-Eintritt mit `(2,)` trotz korrektem NaturalnessOptimizer-Exit
  `(2, 480000)`. Ursache: `restaurier_denker.py` ersetzte bei der SweetSpot-Optimierung
  das Stereo-Ergebnis durch den Mono-Downmix `_restored_f32` (`result.audio = current`).
  Fix: `_optimize_to_sweet_spot` erhält jetzt das volle `result.audio` statt des
  Mono-Downmixes. Re-Verifikation eines Stereo-Laufs nach Abschluss des laufenden
  226-s-Batches offen.
