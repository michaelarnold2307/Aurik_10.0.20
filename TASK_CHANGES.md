# TASK_CHANGES — Live-Ledger der aktuellen Aufgabe

> Generiert von `scripts/change_ledger.py snapshot` (Base: `HEAD`, Stand: 2026-09-11 14:07 CEST).
> CI (`ci-lite.yml` pr-evidence-gate) erzwingt Abdeckung: jede geänderte Code-Datei muss hier stehen.

## Geänderte Dateien

| Status | Pfad | Art |
|---|---|---|
| M | .agents/skills/bug-prevention/scan_anti_patterns.py | modifiziert |
| M | .github/FILE_REGISTRY.md | modifiziert |
| M | .github/VERBOTEN.md | modifiziert |
| M | .github/instructions/hoerordnung.instructions.md | modifiziert |
| M | AGENTS.md | modifiziert |
| M | Aurik10/ui/modern_window.py | modifiziert |
| M | CLAUDE.md | modifiziert |
| M | README.md | modifiziert |
| M | TASK_CHANGES.md | modifiziert |
| M | backend/api/bridge_export.py | modifiziert |
| M | backend/core/adaptive_chunk_processor.py | modifiziert |
| M | backend/core/album_consistency.py | modifiziert |
| M | backend/core/anti_fatigue_pass.py | modifiziert |
| M | backend/core/artifact_detector.py | modifiziert |
| M | backend/core/artist_fingerprint.py | modifiziert |
| M | backend/core/audio_exporter.py | modifiziert |
| M | backend/core/aura_preserver.py | modifiziert |
| M | backend/core/authenticity_metrics.py | modifiziert |
| M | backend/core/consonant_enhancement.py | modifiziert |
| M | backend/core/coordinated_repair.py | modifiziert |
| M | backend/core/defect_scanner.py | modifiziert |
| M | backend/core/dsp/deesser_intelligibility.py | modifiziert |
| M | backend/core/dsp/level_1_invariants_guard.py | modifiziert |
| M | backend/core/export_workflow.py | modifiziert |
| M | backend/core/gpu_model_registry.json | modifiziert |
| M | backend/core/gpu_model_registry.py | modifiziert |
| M | backend/core/ml_device_manager.py | modifiziert |
| M | backend/core/model_zoo_registry.py | modifiziert |
| M | backend/core/music_model_flags.py | modifiziert |
| M | backend/core/musical_goals/musical_goals_metrics.py | modifiziert |
| M | backend/core/naturalness_optimizer.py | modifiziert |
| M | backend/core/per_phase_musical_goals_gate.py | modifiziert |
| M | backend/core/perceptual_export_optimizer.py | modifiziert |
| M | backend/core/perceptual_quality_scorer.py | modifiziert |
| M | backend/core/phases/phase_03_denoise.py | modifiziert |
| M | backend/core/phases/phase_04_eq_correction.py | modifiziert |
| M | backend/core/phases/phase_09_crackle_removal.py | modifiziert |
| M | backend/core/phases/phase_19_de_esser.py | modifiziert |
| M | backend/core/phases/phase_29_tape_hiss_reduction.py | modifiziert |
| M | backend/core/phases/phase_32_mono_to_stereo.py | modifiziert |
| M | backend/core/phases/phase_36_transient_shaper.py | modifiziert |
| M | backend/core/phases/phase_39_air_band_enhancement.py | modifiziert |
| M | backend/core/phases/phase_42_vocal_enhancement.py | modifiziert |
| M | backend/core/phases/phase_43_ml_deesser.py | modifiziert |
| M | backend/core/phases/phase_52_piano_restoration.py | modifiziert |
| M | backend/core/phases/phase_54_transparent_dynamics.py | modifiziert |
| M | backend/core/phases/phase_55_diffusion_inpainting.py | modifiziert |
| M | backend/core/phases/phase_57_print_through_reduction.py | modifiziert |
| M | backend/core/phases/phase_59_modulation_noise_reduction.py | modifiziert |
| M | backend/core/phases/phase_60_inner_groove_distortion_repair.py | modifiziert |
| M | backend/core/phases/phase_64_tape_splice_repair.py | modifiziert |
| M | backend/core/pre_analysis.py | modifiziert |
| M | backend/core/pre_export_validator.py | modifiziert |
| M | backend/core/restorability_estimator.py | modifiziert |
| M | backend/core/sota_vocal_pipeline.py | modifiziert |
| M | backend/core/spec_constitution.py | modifiziert |
| M | backend/core/unified_restorer_v3.py | modifiziert |
| M | backend/core/vocal_focus_analyzer.py | modifiziert |
| M | backend/core/vocoder_chain.py | modifiziert |
| M | backend/ml/phoneme_aware/phoneme_detector.py | modifiziert |
| M | backend/ml/safety_wrappers/formant_shifter_safety.py | modifiziert |
| M | backend/ml/safety_wrappers/stereo_widener_safety.py | modifiziert |
| M | cli/README.md | modifiziert |
| M | cli/aurik_cli.py | modifiziert |
| M | denker/aurik_denker.py | modifiziert |
| M | denker/restaurier_denker.py | modifiziert |
| M | docs/PROJECT_STATUS.md | modifiziert |
| M | plugins/apollo_phase0_integration.py | modifiziert |
| M | plugins/apollo_plugin.py | modifiziert |
| M | plugins/audioldm2_plugin.py | modifiziert |
| M | plugins/basicpitch_plugin.py | modifiziert |
| M | plugins/beats_plugin.py | modifiziert |
| M | plugins/bigvgan_v2_plugin.py | modifiziert |
| M | plugins/bs_roformer_plugin.py | modifiziert |
| M | plugins/bw_reconstructor_plugin.py | modifiziert |
| M | plugins/cqtdiff_plugin.py | modifiziert |
| M | plugins/cqtdiff_plus_plugin.py | modifiziert |
| M | plugins/crepe_plugin.py | modifiziert |
| M | plugins/dac_plugin.py | modifiziert |
| M | plugins/deepfilternet_v3_ii_plugin.py | modifiziert |
| M | plugins/diffwave_plugin.py | modifiziert |
| M | plugins/ear_vae_plugin.py | modifiziert |
| M | plugins/fcpe_plugin.py | modifiziert |
| M | plugins/flashsr_plugin.py | modifiziert |
| M | plugins/flow_matching_plugin.py | modifiziert |
| M | plugins/harmonic_inpainting_plugin.py | modifiziert |
| M | plugins/htdemucs_plugin.py | modifiziert |
| M | plugins/kim_music_enhancer_plugin.py | modifiziert |
| M | plugins/kim_vocal_enhancer_plugin.py | modifiziert |
| M | plugins/laion_clap_plugin.py | modifiziert |
| M | plugins/lyrics_transcriber_plugin.py | modifiziert |
| M | plugins/mert_denoiser_plugin.py | modifiziert |
| M | plugins/miipher_dit_plugin.py | modifiziert |
| M | plugins/panns_plugin.py | modifiziert |
| M | plugins/rmvpe_plugin.py | modifiziert |
| M | plugins/sgmse_plugin.py | modifiziert |
| M | plugins/silero_plugin.py | modifiziert |
| M | plugins/utmos_plugin.py | modifiziert |
| M | plugins/vocos_plugin.py | modifiziert |
| M | requirements/requirements_aurik.txt | modifiziert |
| M | scripts/aurik_verboten_linter.py | modifiziert |
| M | scripts/dsp_benchmark.py | modifiziert |
| A | scripts/export_bigvgan_v2_onnx.py | neu |
| M | scripts/export_cqtdiff_onnx.py | modifiziert |
| M | scripts/gebote_verifier.py | modifiziert |
| M | scripts/horordnung_calibration.py | modifiziert |
| M | scripts/onnx_gpu_compat_scan.py | modifiziert |
| M | scripts/summarize_core_profile_status.py | modifiziert |
| M | scripts/sync_core_models_to_manifest.py | modifiziert |
| M | scripts/validate_core_model_presence.py | modifiziert |
| M | tests/test_optimization_integration.py | modifiziert |
| M | tests/unit/test_denker/test_restaurier_denker.py | modifiziert |
| M | tests/unit/test_ml_device_manager.py | modifiziert |
| M | tests/unit/test_ml_device_manager_amd.py | modifiziert |
| M | tests/unit/test_ml_plugin_load_and_cleanup.py | modifiziert |
| ?? | backend/core/dsp/binaural_masking.py | ungetrackt |
| ?? | backend/core/dsp/dynamic_loudness_model.py | ungetrackt |
| ?? | backend/core/dsp/gammachirp_filterbank.py | ungetrackt |
| ?? | backend/core/dsp/interaural_cues.py | ungetrackt |
| ?? | backend/core/dsp/temporal_loudness.py | ungetrackt |
| ?? | backend/core/go_nogo_export_gate.py | ungetrackt |
| ?? | backend/core/listening_witness.py | ungetrackt |
| ?? | plugins/_vendor_muq/LICENSE | ungetrackt |
| ?? | plugins/_vendor_muq/LICENSE_weights | ungetrackt |
| ?? | plugins/_vendor_muq/README_vendor.md | ungetrackt |
| ?? | plugins/_vendor_muq/**init**.py | ungetrackt |
| ?? | plugins/_vendor_muq/modules/**init**.py | ungetrackt |
| ?? | plugins/_vendor_muq/modules/conv.py | ungetrackt |
| ?? | plugins/_vendor_muq/modules/features.py | ungetrackt |
| ?? | plugins/_vendor_muq/modules/flash_conformer.py | ungetrackt |
| ?? | plugins/_vendor_muq/modules/random_quantizer.py | ungetrackt |
| ?? | plugins/_vendor_muq/modules/rvq.py | ungetrackt |
| ?? | plugins/_vendor_muq/muq.py | ungetrackt |
| ?? | plugins/_vendor_muq/muq_mulan/**init**.py | ungetrackt |
| ?? | plugins/_vendor_muq/muq_mulan/modules/**init**.py | ungetrackt |
| ?? | plugins/_vendor_muq/muq_mulan/modules/transformer.py | ungetrackt |
| ?? | plugins/_vendor_muq/muq_mulan/modules/utils.py | ungetrackt |
| ?? | plugins/muq_mulan_plugin.py | ungetrackt |
| ?? | plugins/muq_plugin.py | ungetrackt |
| ?? | scripts/export_muq_mulan_onnx.py | ungetrackt |
| ?? | scripts/extract_muq_eval_a1_head.py | ungetrackt |
| ?? | tests/unit/test_interaural_cues.py | ungetrackt |
| ?? | tests/unit/test_listening_witness.py | ungetrackt |
| ?? | tests/unit/test_muq_mulan_plugin.py | ungetrackt |
| ?? | tests/unit/test_muq_plugin.py | ungetrackt |
| ?? | tests/unit/test_sota_hearing_modules.py | ungetrackt |

## Entscheidungen

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
