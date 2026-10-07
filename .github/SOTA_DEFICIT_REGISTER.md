# SOTA-DEFICIT_REGISTER — ID-getrackte Defizit-Matrix (WP-0)

> **Zweck (Roadmap WP-0):** Defizite des Modell-/Wohlklang-Bestands werden
> **einmal zentral** geführt statt verstreut in Roadmap-Prosa, Changelog und
> Lauf-Reports. Jede Zeile ist ID-getrackt (`D-<Klasse>-<n>`), trägt ihre
> **Domänen-Evidenz** mit und darf den Status `geschlossen` **nur mit
> prüfbarem Beleg-Pfad** annehmen.
>
> **Normative Anker:**
>
> - §III.13 (copilot-instructions.md) — Domänen-Registry + Evidenzpflicht;
>   kanonische Quelle `.github/ML_MODEL_DOMAIN_REGISTRY.md`.
> - §III.11 (copilot-instructions.md) — sprach-trainierte Kerne nur mit
>   Musik-Finetune + A/B als Signalpfad.
> - §G8 (copilot-instructions.md) — Transparenz (jede Entscheidung belegbar).
> - §G9 (copilot-instructions.md) — projektweite Konsistenz / eine Quelle.
> - §V6 (copilot-instructions.md) — ML→DSP-Fallback nur mit Warnung + Grund.
> - §V7 (copilot-instructions.md) — keine Workarounds; Ursache statt Symptom.
> - §v10.802 (copilot-instructions.md) — Version-/Release-Vertrag.
>
> **Maschinelle Prüfung (fail-closed):** `scripts/sota_deficit_gate.py`
> (Pre-Commit-Hook `aurik-sota-deficit-gate`) +
> `tests/normative/test_sota_deficit_gate.py`.
> Das Gate schlägt fehl bei
> **(a)** Statuswechsel ohne Beleg-Pfad,
> **(b)** Domäne `musik`/`sprache` ohne konkrete Evidenz,
> **(c)** Defizit-Klasse ohne Klassendeklaration + Eintrag.
> Zusätzlich wird jeder Artefaktpfad, der auch im kuratierten Manifest
> (`scripts/model_inventory.py`) steht, gegen die dort belegte Domäne
> abgeglichen (§G9 (copilot-instructions.md) — eine Quelle).

## Status-Enum

| Status | Bedeutung | Pflicht |
| --- | --- | --- |
| offen | Defizit erkannt, Abarbeitung ausstehend | Blocker benennen |
| in-arbeit | Welle läuft, Beleg noch unvollständig | Blocker benennen |
| geschlossen | behoben und belegt | **Beleg-Pfad muss existieren** |
| bewusst-akzeptiert | bewusster Verzicht (kein Defekt) | Begründung im Blocker-Feld |

## Domänen-Enum

`musik` · `sprache` · `gemischt` · `audio-allgemein` · `unueberwacht` ·
`unbekannt` · `—` (nicht anwendbar, z. B. Code-Defizite).

**Evidenzpflicht (§III.13 (copilot-instructions.md)):** `unbekannt` ist die
ehrliche Angabe, wenn keine Trainings-Evidenz vorliegt — **nie** ersatzweise
`sprache`. Domänen aus Ordnernamen sind verboten.

## Defizit-Klassen

| Klasse | Name | Definition | Belegquelle |
| --- | --- | --- | --- |
| K0 | Trainiert, nicht deployt (stilles Kapital) | belegtes Musik-Training/Finetune ohne Produktionskonsument | Roadmap „Neue Defizit-Klassen" |
| K1 | Synthetisch trainiert statt Musik | Modell hat nur Sinus-Harmonische + farbiges Rauschen gesehen, kein Musikmaterial | Roadmap „Neue Defizit-Klassen" |
| K2 | Sprach-Signalpfad (belegt **oder** unbelegt) | sprach-trainierter oder domänen-**unbelegter** Vocoder/SR, der Musik verändern kann (nur §V6 (copilot-instructions.md)-Fallback) | Roadmap WP-1 (K2) |
| K3 | Phantom-Referenz / Phantom-ML-Pfad | Code benennt ein Artefakt oder einen ML-Pfad, der nicht (mehr) existiert bzw. nie läuft | Roadmap WP-2 (K3) |

---

## K0 — Trainiert, aber nicht deployt (stilles Kapital)

| ID | Artefakt | Klasse | Domäne | Evidenz | Status | Blocker | Beleg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D-K0-1 | `output/_training_archive_20260920/f3_bigvgan/best.pt` (BigVGAN-F3 Musik-Finetune) | K0 | musik | Trainingskorpus MUSDB18-HQ über `scripts/train_bigvgan_f3.py`; A/B-Beleg `docs/reports/current/2026-09-16_hr_v1_bigvgan_ab_validation.md` (HNR +4,42 dB); **Export-Belag** `docs/reports/current/2026-10-06_p1_3_bigvgan_f3_rollout_entscheid.md` | offen | **Weg 2 umgesetzt (2026-10-06, §P1-3):** dokumentierte Budget-Ausnahme + **eine** Aufrufstelle + Längen-Deckel. Kosten 13,2× RT → **1,04× RT** je Passage (792 s → **62 s** je Audio-Minute = 26 % des Phasen-Budgets statt 330 %); Aufrufe in `phase_03` (×2), `23`, `50` entfernt, kanonisch allein `phase_07_harmonic_restoration` (§G9 copilot-instructions.md); außerhalb der Ausschnitte bit-identisch (gemessen). **F3-ONNX exportiert + paritätsbewiesen** (`models/bigvgan/bigvgan_v2_f3.onnx`, Checkpoint-cfg epoch 7, rel ≤ 6,5e-05 auf 3 strukturierten Feeds, §III.9 copilot-instructions.md) — **nicht verdrahtet**; F3 kostet +0,9 % ⇒ kein Rollout-Grund. **Offen:** Hörstichprobe **C4** zur Timbre-Konsistenz der gleichmäßig verteilten Ausschnitte (Hörordnung Ebene 4) | `docs/reports/current/2026-10-06_p1_3_bigvgan_f3_rollout_entscheid.md` |
| D-K0-2 | `models/bigvgan/bigvgan_v2.onnx` (aktiv: **Basis** statt Musik-F3, bytescharf bewiesen) | K0 | unbekannt | **Lokal am Artefakt gemessen** (2026-10-06): 128 Mel-Bänder (`conv_pre.weight_v` = `[1536, 128, 7]`), 6 Up-Stufen = 512×, 1536 Kanäle, SnakeBeta (109×`alpha`/109×`beta`) + Anti-Alias, 122,19 M Parameter = BigVGAN v2 **44,1-kHz-Konfiguration**; Gewichts-Vergleich: ONNX == `models/bigvgan/bigvgan_v2.pth` (max. Abweichung 0,0), F3-Finetune 469/783 Tensoren abweichend (max. Abweichung 0,10), ONNX 2026-09-10 < Finetune 2026-10-03. Die Zuordnung „LibriTTS/Sprache" ist architektur-**widerlegt** (LibriTTS-Config = 24 kHz/100 Bänder/256×/512 Kanäle) ⇒ §III.13 (copilot-instructions.md): `unbekannt` | offen | Rollout des MUSDB-Finetunes D-K0-1 offen (Kostenarchitektur, nicht das Artefakt); **das F3-Gegenstück liegt seit 2026-10-06 exportiert vor** (`models/bigvgan/bigvgan_v2_f3.onnx`, 2,86 MB + 491 MB `.onnx.data`), der **aktive Pfad bleibt die Basis**; **Beleg-Lücke**: Basis-Korpus ohne Modell-Karte/Hparams/SHA nicht belegbar (§III.13 (copilot-instructions.md)) | Mess-Block §4a `.github/ML_MODEL_DOMAIN_REGISTRY.md` |
| D-K0-3 | `models/sgmse_plus/finetuned/sgmse_musik_best.ckpt` (+ `models/sgmse_plus/sgmse_musik_core.onnx`) | K0 | musik | Trainings-/Exportbeleg `scripts/train_sgmse_musik.py` (Epoch 24, Val 88,3) | offen | `use_sgmse_musik=False`: 3-Wege-A/B (WSJ0 vs. Musik-Core vs. WPE) + Hörabnahme offen (§III.11 (copilot-instructions.md)) | — |
| D-K0-4 | `models/miipher_dit/whisper_denoiser_best.pt` | K0 | musik | Trainingsskript `scripts/train_whisper_v2.py` → MUSDB18-HQ | offen | `use_whisper_denoiser=False` (deprecated) — Nutzen neu bewerten oder Artefakt archivieren | — |
| D-K0-5 | `models/scnet_4stems/` | K0 | musik | A/B-Report `docs/reports/current/2026-10-06_p1_2_scnet_vs_demucs_fair_ab.md` (+1,96…+3,59 dB SI-SDR gegen die **fair** gemessene Demucs-v4-Stufe; der ältere Report `2026-10-04_p1_2_scnet_vs_mdx23c_ab.md` ist überholt); **verdrahtet 2026-10-06 (§P1-2)** — kanonische Implementierung `plugins/scnet_plugin.py`, Kandidat hinter dem Never-worsen-Vergleich in `backend/core/dsp/stem_separator.py`, Modell-Verdrahtungs-Audit: 0 undokumentierte Verwaiste | offen | **Aktivierung** offen: `use_scnet_music=False` bleibt gesperrt (§v10.802 copilot-instructions.md) bis C4-Hörstichprobe + Hörordnungs-Sign-off; CPU ~7× langsamer als Demucs v4 — RT-Beitrag vor Freigabe gegen das Budget messen (§9.5) | `docs/reports/current/2026-10-06_p1_2_scnet_vs_demucs_fair_ab.md` |
| D-K0-6 | `models/ddsp_predictor/c4_head.pth` | K0 | musik | Trainingsskript `scripts/train_ddsp_predictor_c4.py` → MUSDB (val_MAE ≈ Baseline) | bewusst-akzeptiert | Kein Skill-Nachweis gegenüber Baseline → bewusst kein Rollout ohne ΔSDR-Beleg | — |
| D-K0-7 | `models/_archive_20260920/` — 26 Artefakte, 9,7 G (BANQUET-Vinyl-Batch, melbandroformer-ROCm-Safe-Exporte, `inpainting_mask_best.pt`, BW-Reconstructor-Generationen v3/v4/v5-pre-retrain, `singmos_pro.pt`, `ear_vae/*_orig_20260730_backup.onnx`, `sgmse_wsj0_reverb.ckpt`) | K0 | — | **Lokal gemessen** (`.github/ML_ARTIFACT_FINGERPRINTS.md`, `scripts/model_artifact_probe.py`): 8 Artefakte **byte-identisch** mit deployten Gegenstücken, 6 **architektur-gleich** mit abweichenden Gewichten (Finetune-/Generationsbelege), 14 **ohne Gegenstück**. Sammelarchiv über Musik- und Sprach-Modelle ⇒ keine pauschale Domänen-Aussage (Einzelbelege im Fingerabdruck-Dokument) | offen | „Stilles Kapital“ mit offener Entscheidung: für die 14 Artefakte ohne Gegenstück ist weder Deploy noch Verwerfen dokumentiert (§G8 (copilot-instructions.md)); die Zuordnung der `ear_vae/*_orig_20260730_backup.onnx` (84,1/84,3 M Parameter gegen 73,8 M des deployten Finetunes) ist offen und wird **nicht** gleichgesetzt | `.github/ML_ARTIFACT_FINGERPRINTS.md` |
| D-K0-8 | `models/medium_shallow_v1.joblib` + `models/medium_shallow_v1_report.json` (flacher Material-/Depth-Klassifikator, 1,7 M) | K0 | unbekannt | **Trainingsbeleg** `scripts/train_medium_classifier.py` (Korpus-Manifest + kuratiertes Golden-Listening-Set, deterministisch §G5 (copilot-instructions.md)); **gemessener CV-Report** (n=56): Material-Accuracy **64,3 %**, bezogen auf den **heutigen** Erzeuger `forensics.medium_detector` (nur diese Zahl ist bewertbar). **Depth-Angabe zurückgezogen (2026-10-06):** die früher genannte Depth-Accuracy **85,7 %** ist **nicht bewertbar** — die Vergleichsgröße `detected_depth` ist `len(result.transfer_chain)` (die **Kettenlänge**, erzeugt von `scripts/golden_set_tool.py`), nicht die kuratierte Depth-Label-Größe (1/2/3/4+); zusätzlich liefert der heutige Konsument `backend/core/medium_classifier.py` **gar keine** Depth-Ausgabe (`ClassificationResult` ohne `depth`-Feld) ⇒ für Depth existiert **kein** applikationsgleicher Partner. Auch die Material-Baseline **10,7 %** ist **nicht applikationsgleich** (Träger-Space der gecachten `detected_material`-Felder inkl. `lacquer_disc`/`wax_cylinder` hat **kein** Pendant in den kuratierten 6 Klassen) und darf **nicht** als „+53,6 Pp“ gelesen werden. **Kein Konsument:** `backend/core/medium_classifier.py` nutzt im `use_ml`-Zweig CLAP (§6.8-Zeuge) — das trainierte Artefakt wird von keinem Produktionspfad geladen | offen | **DATEN:** Label-Korpus ≥ ~50 je Klasse (heute 6 Materialklassen, klein und unbalanciert — `reel_tape` durchgehend als `tape` fehlklassifiziert); danach **Wiring-Entscheidung** (CLAP-Zeuge vs. trainierter Kopf) + **Hörordnungs-Sign-off** (§v10.802 (copilot-instructions.md)). Material steuert Bandbreiten-Ceilings, Defekt-Schwellen und Era-Priors der ganzen Kette — der Hebel ist real, er hängt an **Daten**, nicht an einer Messung (§V7 (copilot-instructions.md): kein Rollout ohne A/B) | `models/medium_shallow_v1_report.json` |
| D-K0-9 | `models/ear_vae2_upstream/` (EAR-VAE-**v2**-Upstream-Klon, 298 M, inkl. `inference.py`/`configs`/`docs`) | K0 | unbekannt | **Beschafft 2026-10-06** (Bestandsabgleich Sicherung ↔ Repo); Artefakt-Fingerabdruck in `.github/ML_ARTIFACT_FINGERPRINTS.md` (Verzeichnis wird maschinell mitgeprobt). Kein Trainingskorpus lokal belegt ⇒ §III.13 (copilot-instructions.md): Domäne `unbekannt`, **keine** Musik-Annahme | offen | Rollout-/Ersatz-Entscheidung offen: Phase-0-Clean-Pass läuft heute mit dem v1-**MUSDB-Finetune** (§D-K0-4-Umfeld, SHA-belegt); ob v2 (upstream) den v1-Finetune schlägt, ist **ungebunden** und braucht Never-worsen-Benchmark + Hörordnungs-Sign-off; bis dahin kein Austausch (§V7 (copilot-instructions.md)) | `.github/ML_ARTIFACT_FINGERPRINTS.md` |

## K1 — Synthetisch trainiert statt Musik

| ID | Artefakt | Klasse | Domäne | Evidenz | Status | Blocker | Beleg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D-K1-1 | `scripts/train_bw_v2.py` + `scripts/train_bw_v3.py` + `scripts/train_bw_v5.py` + `scripts/train_bw_compact.py` (BW-Synthetik-Familie) | K1 | unbekannt | `make_synthetic()` in `scripts/train_bw_v5.py` erzeugt Harmonische + Rauschen, kein Musikkorpus | bewusst-akzeptiert | Nicht deployt (`use_bw_v5=False`); Produktion nutzt das MUSDB-trainierte `models/bw_reconstructor/bw_reconstructor.onnx` (`scripts/train_bw_reconstructor.py`). Die Synthetik-Varianten dürfen NICHT als musik-trainiert gelten | — |
| D-K1-2 | `scripts/train_clap_material_classifier.py` (Material-Klassifikator) | K1 | unbekannt | DatasetGenerator erzeugt synthetische DSP-Degradation (Skript-Kopf `scripts/train_clap_material_classifier.py`), kein Musikmaterial | offen | Material-Kalibrierung braucht echte gelabelte Träger (WP-7); synthetisches Training ist kein Domänen-Beleg | — |

**Gegenbelege (kein K1-Fall, zur Abgrenzung):** `scripts/train_bw_v4.py`
(GAN auf MUSDB18) und `scripts/train_bw_reconstructor.py` (MUSDB18-HQ) sind
musik-trainiert — das produktive `models/bw_reconstructor/bw_reconstructor.onnx`
ist damit korrekt ein Musik-Modell.

## K2 — Sprach-Signalpfad (Vocoder/SR auf Musik)

| ID | Artefakt | Klasse | Domäne | Evidenz | Status | Blocker | Beleg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D-K2-1 | `models/hifi_gan/hifi_gan.onnx` (HiFi-GAN-Topologie; „UNIVERSAL_V1" **widerlegt**) | K2 | unbekannt | **Lokal am Artefakt gemessen** (2026-10-06): 80 Mel-Bänder (`conv_pre.weight` = `[128, 80, 7]`), `[8,8,2,2]`/`[16,16,4,4]` = 256×, Kanäle 128/64 ⇒ 0,93 M Parameter, 22 050 Hz (`models/hifi_gan/hifigan_infer.py`) ⇒ V1-Zeitplan in ~¼ Kanalbreite = **kein offizieller Checkpoint** (V1: 512 Kanäle/13,9 M; V2: `[4,4,2,2]`/`[8,8,4,4]`) ⇒ §III.13 (copilot-instructions.md): `unbekannt`. Konsument `backend/core/dsp/sota_speech_superres.py` | offen | Domäne **unbelegt** statt belegt: Korpus ohne Karte/Hparams/SHA nicht nachweisbar; nur §V6 (copilot-instructions.md)-Fallback zulässig; Vollständigkeitstest gegen unflagged Vocoder-Synthese auf Musik offen (WP-1) | Mess-Block §4a `.github/ML_MODEL_DOMAIN_REGISTRY.md` |
| D-K2-2 | `models/vocos/vocos_mel_spec_24khz.onnx` | K2 | sprache | Vocos (LibriTTS) belegt in `.github/ML_MODEL_DOMAIN_REGISTRY.md`; Artefakt `models/vocos/vocos_mel_spec_24khz.onnx` | offen | Vocoder-Stufe 24 kHz — §V6 (copilot-instructions.md)-Gate prüfen (WP-1) | — |
| D-K2-3 | `models/vocos_48khz/vocos_48khz.onnx` | K2 | unbekannt | Modell-Karte `models/vocos_48khz/README.md` („Training details: TODO") — Evidenzpflicht (§III.13 (copilot-instructions.md)) nicht erfüllt | offen | Domäne undokumentiert: weder Musik noch Sprache belegbar; bis zum Beleg nur §V6 (copilot-instructions.md)-Fallback (WP-1) | — |
| D-K2-4 | `models/nvsr/` | K2 | sprache | **Lokal gemessen** (2026-10-06, `.github/ML_ARTIFACT_FINGERPRINTS.md`): `nvsr/nvsr.onnx` ist **byte-identisch** mit `_archive_20260920/flashsr/flashsr_onnx_prod_20260810.onnx` ⇒ das deployte Artefakt ist der **FlashSR**-Produktionsexport (0,09 M Parameter), nicht ein NVSR-Modell; Registry-Rolle `.github/ML_MODEL_DOMAIN_REGISTRY.md` (neuronales SR, Sprache) | offen | Namens-/Herkunfts-Drift: Verzeichnisname „nvsr“ vs. FlashSR-Artefakt — Rolle und Domäne am Artefakt neu bewerten; Signalpfad-Risiko Sprache-auf-Musik (§V6 (copilot-instructions.md)-Gate) | `.github/ML_ARTIFACT_FINGERPRINTS.md` |
| D-K2-5 | `models/aero/` | K2 | sprache | Registry-Beleg `.github/ML_MODEL_DOMAIN_REGISTRY.md` (slp-rl/aero, Sprache) | offen | BWE-Challenger nicht verdrahtet; ein Einsatz würde das Musik-F4 verdrängen (WP-1) | — |

## K3 — Phantom-Referenz / Phantom-ML-Pfad

| ID | Artefakt | Klasse | Domäne | Evidenz | Status | Blocker | Beleg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D-K3-1 | `scripts/ab_test_exports.py` → Phantomziel `models/bigvgan/bigvgan_v2_f3e29.onnx` | K3 | — | **Geschlossen 2026-10-06 (WP-2):** der Harness zielt jetzt auf die **realen** Artefakte (`models/bigvgan/bigvgan_v2.onnx` = Basis, `models/bigvgan/bigvgan_v2_f3.onnx` = MUSDB-Finetune) und bricht bei fehlendem Kandidaten **laut** ab statt still nur A zu messen (§V6 copilot-instructions.md); der Phantomname lebt nur noch als Prosa **ohne** Verzeichnispräfix | geschlossen | — (der Phantom-Schutz ist zusätzlich als fail-closed Regel `P1` des Wohlklang-Gates hinterlegt) | `scripts/ab_test_exports.py` |
| D-K3-2 | `models/wav2vec2/quality_predict.py` | K3 | — | **Eingeordnet 2026-10-06 (WP-2):** die Datei ist **untracked** — `models/*` ist per `.gitignore` ausgenommen (`git check-ignore` bestätigt) — also **Upstream-Stray** im Modellverzeichnis, von keinem Produktionspfad konsumiert; zudem **falsch beschriftet** (lädt `Wav2Vec2ForSequenceClassification` bei `config.json` = `Wav2Vec2ForCTC`) | bewusst-akzeptiert | Bewusst **nicht** angefasst: eine Änderung an untracked Fremdinhalt wäre Daten-Eingriff ohne Repo-Wirkung; entscheidend ist, dass die Datei **nicht** als Aurik-Pfad gilt — der Wohlklang-Vertrag prüft Pfade nur für getrackte Artefakte | `scripts/wohlklang_gate.py` |
| D-K3-3 | 17 Module mit TorchScript-Stub (14× `adaptive_*.py`, `automatic_declipper.py`, `masking_aware_dynamic_eq.py`, `psychoacoustic_enhancement.py`) | K3 | — | `grep -rn "TorchScript-Modell nicht implementiert" backend/core/dsp/` → Treffer u. a. in `backend/core/dsp/adaptive_vad.py` | offen | Deklarierter ML-Pfad läuft nie (DSP-Fallback) — ehrlich deklarieren oder implementieren; §V6 (copilot-instructions.md)-Warnung bleibt Pflicht (WP-2) | — |
| D-K3-4 | `backend/core/dsp/bandwidth_artifact_remover.py` (No-Op `audio_out = audio` im compression-Modus) | K3 | — | No-Op-Rückgabe in `backend/core/dsp/bandwidth_artifact_remover.py` mit Kommentar „Noch nicht implementiert" | bewusst-akzeptiert | No-Op ist never-worsen (Audibility-Prinzip der Hörordnung: kein Defekt → keine Änderung); Deklaration im Code vorhanden — Neubau nur mit A/B-Beleg | — |
| D-K3-5 | 53 nicht existente `models/…`-Referenzen im Produktionscode (1660 geprüfte Dateien) | K3 | — | **Bericht**, kein Fehlurteil: Menge aus dem `P2`-Lauf von `scripts/wohlklang_gate.py` — häufigste Ziele u. a. `models/flow_matching/flow_matching.onnx` (8×), `models/muq/muq_eval_a1_head.pt` (3×), `models/bw_reconstructor/best_model.pt` (3×), `models/mp_senet/mp_senet_musik.onnx` (1×, real ist der `finetuned`-Pfad) | offen | Kandidaten einzeln prüfen: legitime optionale/konditionale Pfade (mit §V6-Warnung) von echten Fehlverdrahtungen trennen — `P1` (fail-closed) deckt bisher nur die **Evidenz-Harneske** ab, weil ein harter Fail auf allen 53 sofort den Commit-Pfad blocken würde | `scripts/wohlklang_gate.py` |
| D-K3-6 | HR-V1-BigVGAN wirkt auf **Stereo** (Regelfall) **nicht** — Gate-Aufruf mit Mono-Kandidat gegen Stereo-Baseline | K3 | musik | **Gemessen 2026-10-07** am Produktionshelfer `plugins/bigvgan_v2_plugin.apply_hr_v1_additive` auf `test_audio/Elke Best …(1977)…wav` (48 kHz, 2ch, 60-s-Ausschnitt; Domänen-Beleg: BigVGAN v2 44,1 kHz mit Musik-Finetune F3, `.github/ML_MODEL_DOMAIN_REGISTRY.md`): **Stereo → `applied=False`**, zwei Ausgabedateien **bit-identisch** (max&#124;Δ&#124; = 0,000000); **Mono → `applied=True`, 66 Bänder** freigegeben. Ursache exakt lokalisiert: `backend/core/dsp/additive_synthesis_gate.py:181` `spec_c, _ = _stft(cand_cn[ch])` → `IndexError: index 1 is out of bounds for axis 0 with size 1`, weil das Plugin `synthesize_audio(mono[…])` (1-D) gegen `base[..., ps:pe]` (2, N) an das Gate gibt und das Gate über **2** Kanäle iteriert. Der Docstring des Plugins behauptet „Layout-agnostisch (wie das Gate)" — **beides ist unwahr**. Kanalkorrekter Aufruf funktioniert (nachgewiesen: 55/47/33/22 Bänder je Ausschnitt, außerhalb der Fenster bit-identisch). Folge: **W-1 (`.github/WOHLKLANG_CLAIMS.md`) ist seit dem Deckel-Commit (2026-10-06) nominal aktiv, aber wirkungslos** für reale Stereomusik; der Budget-Nachweis (1,04× RT) bleibt gültig, der **Wirkungsnachweis fehlte** | offen | Entscheidung des Maintainers: (a) Ursache beheben (Kanalangleichung im Plugin oder Broadcast im Gate) — **aktiviert hörbare Änderung**, daher erst nach C4-Hörprobe; (b) W-1 auf `gesperrt` setzen. Hörprobe liegt bereit (Harness-Weg, siehe Beleg) | `output_audio/mushra/thresholds_wohlklang_w1_hrv1/` (`fit.json` nach dem Hören), Manifest `docs/reports/current/2026-10-07_hoerprobe_w1_hrv1.json`, Anleitung `docs/reports/current/2026-10-07_hoerprobe_anleitung.md` |

---

## Drift-Korrekturen (Domänen-Abgleich Register ↔ Manifest)

Beim Aufbau dieses Registers wurde ein Widerspruch zwischen dem kuratierten
Manifest (`scripts/model_inventory.py`) und der kanonischen Registry
(`.github/ML_MODEL_DOMAIN_REGISTRY.md`) gefunden und an der Wurzel behoben
(§V7 (copilot-instructions.md) — Ursache statt Symptom; §G9
(copilot-instructions.md) — eine Quelle):

| Modellpfad | Manifest vorher | Registry (kanonisch) | Korrektur |
| --- | --- | --- | --- |
| `models/bigvgan/bigvgan_v2.onnx` | audio-allgemein | BigVGAN v2 (LibriTTS) = **Sprache** | 2026-10-06 **revidiert**: die Registry-Angabe war selbst Doku-Übernahme; am Artefakt gemessen ⇒ `unbekannt` (Mess-Block §4a) |
| `models/hifi_gan/hifi_gan.onnx` | audio-allgemein | HiFi-GAN UNIVERSAL_V1 = **sprach-trainiert** | 2026-10-06 **revidiert**: „UNIVERSAL_V1" ist architektur-widerlegt ⇒ `unbekannt` (Mess-Block §4a) |

**Lehre aus diesem Abgleich (§V7 (copilot-instructions.md) — Ursache statt Symptom):**
Die falsche Sprach-Zuordnung wurde nicht im Manifest, sondern im **Skript-Kopf**
von `scripts/train_bigvgan_f3.py` („BigVGAN-v2 ist sprach-trainiert
(LibriTTS/LJSpeech)") geboren und über die Registry in Manifest und Register
fortgeschrieben — drei Dokumente, eine unbelegte Annahme. Die Wurzel wurde
mitkorrigiert; die Architektur-Messung (§4a) ersetzt die Annahme. Für `hifi_gan`
gilt derselbe Befund (Name „UNIVERSAL_V1" ohne Artefakt-Deckung).

## Aufgelöste Prüfpunkte (vormals „nicht belegbar")

- **EAR-VAE MUSDB-Finetune — aufgelöst 2026-10-06 (SHA-belegt):** Die
  Roadmap-Angabe war **korrekt**. `models/ear_vae/encoder.onnx` (`8f97ef67…`)
  bzw. `decoder.onnx` (`ed64a1f1…`) sind **byte-identisch** mit
  `ear_vae_ft_encoder_inline.onnx`/`ear_vae_ft_decoder_inline.onnx` aus dem
  Backup (`/media/michael/Aurik_Backup/Aurik_Standalone 16.09.2026/models/ear_vae_upstream/`);
  der Checkpoint `models/ear_vae/ear_vae_music_finetuned.pyt` (SHA `993e69be…`)
  sowie die External-Data-Paarung `ear_vae_ft_encoder.onnx` (+ `.onnx.data`,
  SHA `4168db14…`) und `ear_vae_ft_decoder.onnx` (+ `.onnx.data`, SHA
  `48dad212…`) liegen seit 2026-10-06 **in-repo** (Rezept, Architektur und
  Config mit ihm).
  Reproduzierbarkeit geprüft: Re-Export aus dem Checkpoint ergibt
  **max|Δ| = 0,0** gegen die deployten ONNX-Artefakte (Encoder, Decoder,
  Ende-zu-Ende) — auch die kopierte External-Data-Paarung liefert max|Δ| = 0,0;
  der §III.9-Maßstab (rel ≤ 1e-3) ist damit weit unterschritten.

  **Lehre (§V7 (copilot-instructions.md): Ursache statt Symptom):** Die frühere
  Fassung dieser Sektion („kein Artefakt auffindbar") war ein Suchfehler — die
  Suche war auf den Arbeitsbaum begrenzt, der Beleg lag im Backup. **Kein**
  Registereintrag nötig: der Finetune **ist** deployt, also kein Defizit (§G8
  (copilot-instructions.md) Transparenz).

## Pflege

- Neues Defizit → neue Zeile `D-<Klasse>-<n>` **im selben Commit** wie der
  Befund; Belegpfad eintragen (§G9 (copilot-instructions.md)).
- Status auf `geschlossen` **nur** mit existierendem Beleg-Pfad.
- Nach jeder Welle: `python scripts/sota_deficit_gate.py` und
  `tests/normative/test_sota_deficit_gate.py` lokal laufen lassen.
- Roadmap-Verweis: `docs/TODOS_SOTA_ROADMAP.md` → Abschnitt
  „Arbeitspakete WP-0…WP-7" (WP-0).
