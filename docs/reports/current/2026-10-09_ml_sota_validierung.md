# ML-SOTA-Validierung A–F — Aussagen am Artefakt geprüft (Stand 2026-10-09)

> **Zweck:** Jede einzelne Aussage der SOTA-Lückengabe (Kategorien A–F) wurde gegen
> die **tatsächlichen Artefakte, Flags und Code-Pfade** validiert — nicht gegen die
> Dokumentation. Gemessen mit: `ls`/`stat`/`sha256sum`, `torch.load` am Checkpoint,
> ONNX↔Torch-Paritätsproben (fixe Inputs, §G5-Seeds), Code-Greps an den Produktionspfaden.
> Arbeitssprache Deutsch; §-Zitate mit Quelle.
>
> **Legende:** ✅ = Aussage bestätigt · ⚠️ = bestätigt, aber Doku-Korrektur nötig ·
> ❌ = widerlegt.

## Messumgebung

- GPU: Radeon RX 7900 XTX (ROCm), **23,9 GiB frei** gemessen (`torch.cuda.mem_get_info`,
  2-GiB-Allokationsprobe OK) — Trainingsläufe sofort ausführbar.
- Python: `.venv_aurik` (Python 3.10).

---

## A. Trainiert auf Musik, aber nicht im Wirkpfad ("stilles Kapital")

| # | Aussage | Befund (gemessen) | Urteil |
| --- | --- | --- | --- |
| A.1 | BigVGAN-F3 exportiert + Parität belegt, aktiver Pfad lädt Basis; C4-Hörstichprobe offen | `models/bigvgan/bigvgan_v2_f3.onnx` (2 856 056 B) + `.onnx.data` (491 323 392 B, mtime 2026-10-06 20:44), Manifest-Eintrag `bigvgan_bigvgan_v2_f3.onnx` SHA `30c829cf…`. Aktiver Pfad: `plugins/bigvgan_v2_plugin.py:262` hardkodiert `MODELS_DIR / "bigvgan_v2.onnx"` (Basis, mtime 2026-09-10). F3 ohne Loader-Pfad. `BIGVGAN_V2_HR_ACTIVATED = True` (Z. 108) mit §P1-3-Längen-Deckel | ✅ |
| A.2 | SGMSE+ Musik-Core trainiert + ONNX vorhanden; `use_sgmse_musik=False`; Metrik-A/B: Musik ~1 dB besser, Hörabnahme offen | **Korrektur am Artefakt:** `finetuned/sgmse_musik_best.ckpt` trägt **epoch = 80, val_loss = 6.15249** (gemessen via `torch.load`), NICHT "Epoch 24, Val 88,3" wie in Registry §2 / D-K0-3 / Roadmap-Zeile 35. Trainingslauf **abgeschlossen** am 2026-10-05: Log `output/train_sgmse_musik.log` (Ep 1–114, best 6.1525 bei Ep 80) + `output/train_sgmse_musik2.log` ("Done. Best val: 6.1525 \| Rollbacks: 3" — Rollback-Budget erschöpft, Ep 84). **Paritätsbeweis (neu):** ONNX ↔ best-ckpt auf fixen Inputs `[1,2,256,64]` Seed 42/99/5: max\|Δ\| = 6,7e-05 (σ=0.5), 9,3e-06 (σ=1.0), 2,9e-03 abs. (σ=0.01 — fp32-Grenze bei `scale_by_sigma`, relativ klein). ⇒ Deploytes ONNX == epoch-80-Finetune. Flag: `use_sgmse_musik=False` (`backend/core/music_model_flags.py`). A/B-Reports 2026-10-08 vorhanden (`docs/reports/current/2026-10-08_sgmse_ab_metrik*.json`) | ⚠️ Doku-Korrektur (Checkpoint-Identität + F7-Status) |
| A.3 | SCNet 4stems A/B gewonnen, verdrahtet hinter Never-worsen; `use_scnet_music=False` bis C4-Hörordnungs-Sign-off; RT ~7–9× langsamer | `models/scnet_4stems/huge_scnet_4stems_v1.2.ckpt` (395 599 229 B, Manifest SHA `807f470b…`) + vendored `zfturbo_scnet`. Flag `use_scnet_music=False`. RT-Budget: `docs/reports/current/2026-10-08_scnet_rt_budget.json` (3,43× RT Chunked / 2,20× Ganzsegment). C4 Runde 1 = ZU_WENIGE_TRIALS; Runde-2-Paket: `docs/reports/current/2026-10-08_hoerprobe_scnet_c4_r2.json` (12 Trials + 2 Fangfragen) | ✅ |
| A.4 | Whisper-Denoiser MUSDB-trainiert, deprecated (`use_whisper_denoiser=False`) — neu bewerten oder archivieren | `models/miipher_dit/whisper_denoiser_best.pt` vorhanden; Flag `False` mit Kommentar "DEPRECATED (Rev. 2026-08-16)" | ✅ |
| A.5 | EAR-VAE v2 Upstream beschafft, Rollout-/Ersatz-Entscheidung offen (D-K0-9) | `models/ear_vae2_upstream/` = 298 MB vorhanden; Produktionspfad bleibt v1-MUSDB-Finetune (`models/ear_vae/{encoder,decoder}.onnx`, SHA-belegt in Registry §2) | ✅ |

## B. Sprach-trainierte Modelle im Signalpfad (K2, §III.11 (copilot-instructions.md))

| # | Aussage | Befund (gemessen) | Urteil |
| --- | --- | --- | --- |
| B.1 | Vocos 48 kHz ist **primäre** Stufe des §1.4-Studio-Finishers (Maximum/Studio/Aggressive, PQS-MOS<4.3); Domäne Sprache belegt (Podcasts/VCTK/Expresso, kein Gesang) | Code: `backend/core/unified_restorer_v3.py` Z. 18790ff — `_is_vocos_studio = any(kw in mode for kw in ("maximum","studio","aggressive"))`, Schwellwert `_pqs_mos_norm < 0.825` (= MOS 4.3); Kaskade `plugins/vocos_plugin.py`: 1) `vocos_48khz.onnx` → 2) 44 kHz (**Datei fehlt, Stufe inaktiv**) → 3) `vocos_mel_spec_24khz.onnx` → 4) BigVGAN v2 → 5) HiFi-GAN → 6) Griffin-Lim. Lokale Karte `models/vocos_48khz/README.md`: `kittn/vocos-mel-48khz-alpha1`, **"Training details: TODO"** (Karte); Owner-Aussage (Podcasts/VCTK/Expresso, "reconstruction quality is quite poor") = externer Beleg HF-Discussion #2 — lokal nicht gegenprüfbar, aber mit Karte konsistent | ✅ (Primär-Rolle wirkt real im Studio-Pfad) |
| B.2 | Vocos 24 kHz = Fallback-Stufe 3; §V6-Gate-Check offen (WP-1) | Kaskadenposition 3 bestätigt (s. B.1); Artefakt `models/vocos/vocos_mel_spec_24khz.onnx` (54 MB) vorhanden | ✅ |
| B.3 | HiFi-GAN nur noch als Plugin-Fallback aktiv; Domäne unbelegt (`unbekannt`) | `plugins/bigvgan_v2_plugin.py` + `plugins/audioldm2_plugin.py:458` laden es als Fallback; `sota_speech_superres.py` / `vocoder_chain.py` ohne Aufrufer (Verifikation 2026-10-07 in D-K2-1). Artefakt `models/hifi_gan/hifi_gan.onnx` (3,7 MB) | ✅ |
| B.4 | NVSR/FlashSR-Namensdrift: `nvsr.onnx` = FlashSR-Produktionsexport; `nvsr.ts` archiviert (D-K0-10) | Byte-Identität belegt in `.github/ML_ARTIFACT_FINGERPRINTS.md`; `models/nvsr/` enthält jetzt **nur** `nvsr.onnx` (398 593 B); `nvsr.ts` liegt in `models/_archive_20260920/nvsr/` | ✅ |
| B.5 | AERO = BWE-Challenger, nicht verdrahtet; A/B verloren gegen FlashSR (ΔSDR min −2.4 / mean −1.6 dB) | Produktions-Aufrufstellen: nur `backend/core/gpu_model_registry.py:215` (Pfad-Registrierung für EP-Policy, kein Aufruf) + Plugin + Benchmark-Skripte. `models/aero/aero_12_48.onnx` (81 MB) vorhanden | ✅ |
| B.6 | DFN Musik-Finetune aktiv (`use_df_musik=True`) | Flag `True`; Plugin lädt bei Flag die `finetuned/`-ONNX (enc/dec/erb_dec, alle vorhanden, mtime 2026-09-09) — `plugins/deepfilternet_v3_ii_plugin.py:55–56` | ✅ |

## C. Sprach-trainierte "Richter" (Zeugen-Rolle)

| # | Aussage | Befund (gemessen) | Urteil |
| --- | --- | --- | --- |
| C.1 | UTMOSv2 = Sprach-MOS-Orakel, `use_utmos_music=False` → report-only; musik-native Substitution lokal verfügbar: MuQ + A1-Head + SingMOS Pro | `models/utmosv2/fold0_s42_best_model.pth` (818 MB) vorhanden; Flag `False`; Konsument `backend/core/dsp/utmos_delta_gate.py:37` liest das Flag. Substitution lokal: `plugins/muq_plugin.py` lädt `models/muq_mulan/muq_eval_a1_head.pt` (+ `muq_eval_bn_stats.pt`, ONNX 1,27 GB) und `models/singmos/singmos_pro.onnx` (2,8 MB + 1,26 GB `.data`) | ✅ — **kein neues Training nötig für die Substitution; F8 = Anbindung/Kalibrierung** |
| C.2 | Resemblyzer (LibriSpeech) als Vokal-Identitäts-Richter gesperrt (`use_resemblyzer_music=False`); musik-native Embeddings lokal vorhanden | `models/resemblyzer/resemblyzer_voice_encoder.onnx` + `pretrained.pt` vorhanden; Flag `False`; Substitution lokal: `models/mert/mert_330m.onnx` (355 MB) + MuQ-MuLan (`muq_mulan.onnx` 1,27 GB) | ✅ — F9 = Musik-Vokal-Finetune **oder** Ersetzung durch MERT/MuQ-Embeddings (Empfehlung §C Roadmap: Ersetzung) |
| C.3 | Silero VAD (English-Speech) unzuverlässig für Musik, `use_silero_vad_music=False`; Empfehlung PANNs/AST | Flag `False` (steuert Gate-Schwellen-Offset in phase_18); lokal: `models/panns/panns_wavegram_logmel_cnn14.onnx` (+ 329 MB `.data`) und `models/ast/ast_model.onnx` (294 KB + 346 MB `.data`). **Befund:** `models/panns/audioset_labels.txt` ist eine kuratierte **15-Klassen-Subset** (Speech, Female/Male/Child singing, … Silence) für Query-Zwecke — das volle 527er-Ausgaberaum des CNN14 bleibt erhalten | ✅ (mit Detail-Befund zum Label-File) |
| C.4 | SingMOS Pro: Kalibrierung auf verarbeitete Musik-Vocals nötig (F10) | `models/singmos/singmos_pro.onnx` (+`.data`) + Trainings-Toolkit vorhanden; deploytes ONNX vom 2026-09-10 | ✅ |
| C.5 | wav2vec2 CTC inkonsistent (`quality_predict.py`: Classification statt CTC) — Zeugenrolle, Upstream-Stray | `models/wav2vec2/quality_predict.py` ist **untracked** (`.gitignore` `models/*`) → D-K3-2 eingeordnet ("bewusst nicht angefasst") | ✅ |

## D. Musik-trainierte Modelle mit offenen GPU-Trainings-/Finetuning-Aufgaben

| # | Aussage | Befund (gemessen) | Urteil |
| --- | --- | --- | --- |
| D.1 | BW-Reconstructor v5: A1-HF-Gain-Gate nicht bestanden (0,73 < 1,02) → Re-Training mit masking-aware A1; `use_bw_v5=False` | Artefakte lokal: `best_model_v5.pt` + `bw_reconstructor_v5.onnx` (beide mtime 2026-08-14); produktiv aktiv ist das MUSDB-trainierte `bw_reconstructor.onnx` (2026-07-10) via Legacy-Pfad (`use_bw_v5=False` ⇒ `LEGACY_MODEL_PATHS["bw"]`). Gate-Zahl 0,73 belegt in `docs/reports/current/2026-10-04_offene_massnahmen.md` (B3); Generations-Provenienz v3/v4/v5-pre-retrain im Archiv | ✅ |
| D.2 | BANQUET Real-Vinyl-Finetune: Code + Tests fertig, GPU-Lauf + ΔSDR-Gate offen; **Präbedingung reale Vinyl↔Reissue-Paare** | `scripts/train_banquet_vinyl_finetune.py` vorhanden (Default-Korpus `corpus/vinyl`). **Gemessen:** `corpus/vinyl/damaged/` = 16 Dateien, `clean/` = 6 Dateien; Manifest: alle Einträge "Synthetic (Aurik Corpus Generator)" (CC0) ⇒ **keine echten Vinyl↔Reissue-Paare lokal** — Präbedingung noch nicht erfüllt | ✅ (Präbedingung offen = korrekt dokumentiert) |
| D.3 | DiffWave-Vokal-Finetune abgebrochen (−1.4/−1.7 dB-Deckel); Gate jetzt durch CQTdiff+ Cascade (+0,005 dB, marginal) | `models/diffwave/`: nur Basis `diffwave.ckpt` + `diffwave_model.onnx`; **kein** akzeptierter Vokal-Ft-Checkpoint — verworfene: `_archive_20260920/diffwave/diffwave_vocal_ft_rejected_epoch{2_gatefail,7_aborted}.ckpt`. CQTdiff-S4-Report (2026-09-16): `delta_vs_silence` ≈ 0 dB je Segment (−0.12 … +0.06) ⇒ Gate marginal erfüllt | ✅ |
| D.4 | GaCELA-Vokal-Finetune trainiert, in fairer Validierung verworfen (Halluzinationen; mean SDR −2.63 dB) | `models/gacela/model/`: `gacela_core_ft39.onnx` (57 MB, 2026-09-20) + Basis `gacela_core.onnx`; Validierungsreport `docs/reports/current/2026-09-15_gacela_vocal_val.json`: `gate_mean_sdr_ge_0: false`, `mean_sdr_db: −2.63` (Checkpoint 39) | ✅ — Ft-ONNX liegt lokal als Challenger-Artefakt vor |
| D.5 | DDSP/C4-CLAP-Negativbefund (val_MAE ≈ Baseline ⇒ kein Skill); Neu-Training auf DAC-Encoder oder BEATs offen | `models/ddsp_predictor/c4_head.pth` (662 KB) + `config.json`; Status "bewusst-akzeptiert" (D-K0-6). Ersatz-Basis lokal: `models/dac/encoder_model.onnx` (91 MB, dac_44khz) | ✅ |
| D.6 | Material-Classifier: CV 64.3 %, kein Konsument; WP-7-Datenaufbau läuft (shellac ≥50 erreicht) | `models/medium_shallow_v1.joblib` + Report vorhanden; **WP-7-Stand gemessen** (`docs/reports/current/2026-10-09_wp7_coverage.json`): 143 Labels gesamt; Echt-Träger: shellac 47, digital 10, cassette 4, reel_tape 1, tape 1, **vinyl 0** ⇒ `wp7_real_met: false` (Ziel 10 je Klasse). Konsument: `backend/core/medium_classifier.py` nutzt CLAP-Zweig — trainiertes Artefakt ungenutzt | ✅ |
| D.7 | MERT-Gender-Head (F13) + empirische Kalibrierung (F14): Skripte fertig, Label-Korpus fehlt | `scripts/train_gender_head.py` + `scripts/calibrate_gender_evidence.py` vorhanden; `data/gender_labels.csv` und `data/music/` **fehlen** ⇒ F13/F14 datenblockiert | ✅ |

## E. Fehlende Modelle / Fähigkeiten

| # | Aussage | Befund (gemessen) | Urteil |
| --- | --- | --- | --- |
| E.1 | Neuraler Warp-Schätzer (WF-V4): Quelle/Download unklar; §C: Warp-Head auf FCPE/CRePE-F0 trainieren | `find models/**` nach `*warp*`: nur VERSA-WarpedQ-Metrik-Dateien (funktionslos für WF). **Kein Warp-Modell in irgendeinem models/-Unterordner.** Trainings-Basen lokal: `models/fcpe/fcpe.onnx` + `fcpe.pt`, `models/crepe/crepe.onnx` | ✅ fehlt — aber Basen lokal vorhanden (kein Download nötig) |
| E.2 | Neuronale Phasen-Schätzung für Transienten (TP-V2): Modell + Quelle fehlen; §C: Vocoder als Phasen-Synthesizer / Training | `find models/**` nach `*phase*`: nur CQTdiff-Experiment (`exp_phase_retrieval.py`) + MP-SENet-Demos. Kein deployter Phasen-Schätzer. (Caveat: `cqtdiff/src/experimenters/exp_phase_retrieval.py` ist ein Experiment des Flow-Matching-Modells, kein TP-V2) | ✅ fehlt — lokale Vocoder (Vocos/BigVGAN/DAC) als Synthese-Basis vorhanden |
| E.3 | Präziser RT60-Regressor statt konservatives T60-Witness-Cap | `backend/core/room_acoustics_fingerprinter.py`: kein ONNX-/Modellbezug → T60-Witness ist rein DSP; kein lokaler Regressor-Checkpoint in models/** | ✅ fehlt — Training nötig (Daten: RIR-Paare, teils lokal via DFN-Assets `rir_sim_*.wav`) |
| E.4 | Referenzfreie Musik-Qualitätsmetrik (UTMOS richtungsblind/invertiert; MuQ 3/3) | = F8/C.1: UTMOS-Negativbefund + MuQ als einziger richtungsvalidierter Zeuge — lokal belegt (muq_eval_a1_head, SingMOS Pro) | ✅ (in C.1 abgedeckt) |
| E.5 | Genre-Modell (MERT-Head): GTZAN/MagnaTagATune-Trainingslauf fehlt | Keiner der Datensätze lokal (`data/`: musdb18hq, sgmse_musik, cantus_pairs, symphonia_*, phase_impacts, ab_tests — kein GTZAN/Magnatag); `models/mert/` = nur Encoder + `clean_music_centroid.npy` | ✅ fehlt (Daten-Beschaffung nötig ODER MUSDB-basiertes Genre-Labeling) |
| E.6 | BEATs-Tagger-Head (527): über panns/ast gedeckt (Encoder-only dokumentiert) | Lokal: PANNs CNN14 ONNX + AST ONNX; `models/beats` = Encoder-only (`[B,T,768]`, kein 527-Tagger — Registry §4). **Detail-Befund:** das lokale `panns/audioset_labels.txt` ist ein 15-Klassen-Subset (s. C.3) | ✅ gedeckt |
| E.7 | AudioLDM2-Plugin: "ONNX existiert lokal; Plugin-Neuanlage fehlt" | **Korrektur:** `plugins/audioldm2_plugin.py` **existiert** (vollständige ONNX-Pipeline: UNet 1,39 GB + VAE-Dec 126 MB + HiFi-GAN-Vocoder + CLAP-Text; git-verfolgt) und ist verdrahtet in `backend/core/phases/phase_24_dropout_repair.py` (Tier-3 für Dropouts > 3 s) + `ml_device_manager.py` + `ml_model_readiness.py`. Lücken-Matrix-B-Zeile "Plugin-Neuanlage fehlt" ist **veraltet**. Rest: Rolle "generative Baseline für destruktive Fälle" nach S1 aufgeschoben (Entscheidung offen, kein Mangel) | ⚠️ Doku-Korrektur |

## F. Menschliche Kalibrierung

| # | Aussage | Befund | Urteil |
| --- | --- | --- | --- |
| F.1 | MUSHRA-Studie (WP-8, n≥30) für MOS-Orakel-/Proxy-Kalibrierung + SCNet C4 etc. | Studie "Vorbereitung" (`docs/reports/studies/mushra_2026q2/` inkl. Preregistration); SCNet-C4-Runde-2-Paket bereits gebaut (12 Trials, `2026-10-08_hoerprobe_scnet_c4_r2.json`) | ✅ |

---

## Befund-Summe: falsche/veraltete Dokumentation zu korrigieren

| # | Ort | Falsch / veraltet | Korrektur (am Artefakt gemessen) |
| --- | --- | --- | --- |
| K1 | `.github/ML_MODEL_DOMAIN_REGISTRY.md` §2 (SGMSE+-Zeile) + `SOTA_DEFICIT_REGISTER.md` D-K0-3 (Evidenz-Zelle) + `TODOS_SOTA_ROADMAP.md` Zeile 35 (F7 "TEILWEISE") | Checkpoint als "Epoch 24, Val 88.3"; F7 als teilweise | **epoch 80 / val loss 6.1525**, Lauf abgeschlossen 2026-10-05 ("Done. Best val: 6.1525 \| Rollbacks: 3"); ONNX↔ckpt Parität max\|Δ\|=6,7e-05 (σ=0.5) neu belegt |
| K2 | `TODOS_SOTA_ROADMAP.md` Lücken-Matrix B (Zeile "AudioLDM2-Plugin") | "Plugin-Neuanlage fehlt" | Plugin existiert + in phase_24 verdrahtet; offen ist nur die Rollenentscheidung "generative Baseline für destruktive Fälle" (nach S1 aufgeschoben) |

## Konsequenzen fürs Konzept (kein Training, das durch Besser-Doku unnötig wird)

1. **F7/SGMSE+**: Kein Re-Training nötig — der epoch-80-Finetune ist besser als dokumentiert.
   Offene Entscheidungen bleiben: 3-Wege-A/B-Hörabnahme (C4-artiges Sign-off), dann Flag-Flip.
2. **F8/MOS-Orakel**: Substitution (MuQ+A1-Head, SingMOS Pro) ist lokal vollständig —
   Training nur für die **Kalibrierung auf MUSHRA-Musik-Bewertungen** nötig; alternativ
   direkter Musik-Einsatz der MuQ-SingMOS-Kombi mit A/B gegen UTMOS.
3. **F9/Vokal-Identität**: Lokale musik-native Embeddings (MERT/MuQ) als Ersetzungs-Pfad —
   nur dann Resemblyzer-Finetune, wenn das Ersetzungs-A/B nicht trägt.
4. **WF-V4/TP-V2/RT60/Genre**: wirklich fehlend; Trainings-Basen (FCPE/CRePE, Vocoder) lokal —
   Downloads nur für Genre-Labels (GTZAN o. ä.) nötig.
5. **BANQUET F12**: vor dem GPU-Lauf die echten Vinyl↔Reissue-Paare beschaffen
   (aktuell nur 6 synthetische Paare in `corpus/vinyl`).
