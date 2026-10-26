# SOTA-Implementierungs-/Trainings-/Fine-Tuning-Konzept — Aurik 10

> **Zweck:** Alle Modell-Arbeiten (Implementierung, Training, Fine-Tuning) in EINE
> Doku: was ist schon erledigt („stilles Kapital"), welche Entscheidungen ohne GPU
> möglich sind (P0), welche GPU-Läufe geplant sind (P1/P2), welche neuen Modelle
> fehlen (P3) — und an welchen menschlichen Gates (Hörproben/MUSHRA) alles hängt.
> **Normative Anker:** `.github/copilot-instructions.md` (§III.9, §III.11, §III.13,
> §G5, §V6, §V7, §v10.802), `.github/instructions/hoerordnung.instructions.md`
> (Ebene 4: das menschliche Ohr ist die letzte Instanz),
> `.github/SOTA_DEFICIT_REGISTER.md` (D-K-IDs), `.github/ML_MODEL_DOMAIN_REGISTRY.md`
> (Domänen am Artefakt).
> **Source of truth:** `docs/reports/current/2026-10-09_ml_sota_validierung.md` —
> jede Aussage dort wurde gegen die tatsächlichen Artefakte, Flags und Code-Pfade
> validiert (nicht gegen Dokumentation); diese Doku aktualisiert den Stand auf
> 2026-10-23 (v10.12.43). Arbeitssprache Deutsch; §-Zitate immer mit Quelle
> (AGENTS.md §5 ID-Zitierdisziplin).

---

## 0. Status und Abarbeitungs-Fortsetzung (Live-Abschnitt — bei jeder Änderung aktualisieren)

**Stand: 2026-10-23 (v10.12.43).** Alle drei Hörproben-Pakete der Welle A sind
gebaut; die menschlichen Gates stehen aus. Die GPU-Lanes (Wellen B/C) sind nur noch
durch Daten-Beschaffung blockiert (BANQUET, RT60, Material-Klassifikator) — KP-7 und
KP-8 wären sofort ausführbar.

| Welle | KPs | Zustand | Nächster Schritt |
| --- | --- | --- | --- |
| **A** (jetzt, kein Training) | KP-1…KP-5 | Pakete **3/3 bereit**, Flag-Flips 0/3 | Hörproben fahren (§10 + `docs/reports/current/2026-10-07_hoerprobe_anleitung.md`), `mushra_harness.py thresholds-fit`, Sign-off → bis zu drei Flag-Flips ohne GPU |
| **B** (GPU, parallel in VRAM-Grenzen) | KP-6…KP-8, KP-10 | BANQUET: Code fertig, Daten offen; BW-v5 + Cantus-Full sofort ausführbar; RT60: Daten teils lokal | Echte Vinyl-Paare + RIR-Paare beschaffen (§9/7); dann Runs in §8-Reihenfolge |
| **C** (abhängig von B) | KP-9, KP-11, KP-12 | bedingt / pending_training | erst nach Cantus-/SingMOS-Ergebnissen aus Welle B |
| P2/P3 (datenabhängig) | KP-13…KP-19 | Daten-Beschaffung ist der Flaschenhals (Label-Korpora, Genre-Labels, Leerrillen) | Lizenz-Politik-Entscheid (§9/7) löst Vinyl-/Tape-Klassen frei |

**Wenn hier unterbrochen:** Starte bei **§3/KP-1** — die drei fertigen Pakete mit
`python scripts/hoerpanel_player.py --study output_audio/mushra/thresholds_…` fahren,
mit `mushra_harness.py thresholds-fit` auswerten und das Sign-off in
`.github/SOTA_DEFICIT_REGISTER.md` dokumentieren. Flag-Flips erst danach, je einen
pro Entscheidung (§9).

### KP-Status-Tabelle (alle 19)

| KP | Name | Status | Nächster Schritt | Gate / Beleg |
| --- | --- | --- | --- | --- |
| KP-1 | Rollout des stillen Kapitals (Hörproben-Runde) | **Pakete bereit — Hörinstanz steht aus** | SCNet-C4-r2 + W-1/HR-V1 (gebaut 10-07/10-08) und SGMSE 3-Wege (gebaut 10-23, `scripts/build_sgmse_hoerprobe_3weg.py`) anhören + Sign-off | Hörordnung Ebene 4 / §v10.802 (copilot-instructions.md); Manifests §10 |
| KP-2 | Musik-MOS-Orakel für F8 (UTMOS-Ersatz: MuQ+A1-Head, SingMOS Pro) | offen — Substitution lokal komplett | Pfad A ohne Training: Zeugen-Paare + Schwellen-Kalibrierung auf MUSHRA-Subset; Anbindung in `utmos_delta_gate` (report-only → Veto nach Kalibrierung) | Richtungsstabilität über alle Degradations-Klassen + Non-Inferiority gegen UTMOS-Report-only-Zeitraum; Beleg Validierung C.1, Flag `use_utmos_music=False` |
| KP-3 | Vokal-Identität F9 (Resemblyzer-Ersatz) | offen — Substitution lokal komplett | Kandidat für Ebene-1-Guard + phase_65 mit MERT/MuQ-Embeddings bauen; A/B der Witness-Entscheidungen auf dem Golden Set | gleiche Entscheidungen → cos-invariant + Hörabnahme; Beleg Validierung C.2, Flag `use_resemblyzer_music=False` (Empfehlung: Ersetzung statt Finetune) |
| KP-4 | VAD: Silero → PANNs-Singing/AST | offen | Musikadaptierten Gate-Offset (Muster phase_19); A/B auf Korpus | Never-worsen der phase_18-Gate-Entscheidungen + Witness-neutral; Beleg Validierung C.3, Flag `use_silero_vad_music=False` (neutral §V7 copilot-instructions.md) |
| KP-5 | Studio-Finisher-Vocoder-Entscheidung (D-K2-3) | offen — wartet auf W-1-Hörprobe | BigVGAN-F3 als Stufe 1 des Studio-Pfads nach C4/W-1-Sign-off; bei RT-Budget-Verstoß: Vocos-Vokal-Finetune auf MUSDB18-HQ | W-1-Hörstichprobe + RT-Budget (D-K0-1); §V6-Domänen-Warnung sitzt bereits am Ladepfad (10.12.24) |
| KP-6 | BANQUET Real-Vinyl-Finetune F12 (+ Upstream-Optimalitäts-Bewertung) | in-arbeit — Daten der Flaschenhals | Echte Vinyl↔Reissue-Paare beschaffen (heute nur 16 damaged/6 clean, alle synthetische CC0), dann GPU-Run (wenige Stunden, 23.7 M Par.) | ΔSDR ≥ +2 dB vs. Zero-Shot auf destruktiven Fällen + never-worsen ≥ −0.1 dB auf sauberen Referenzen → voller Vertrag; Beleg Validierung D.2, Report `docs/reports/current/2026-10-09_ml_sota_validierung.md` |
| KP-7 | BW-Reconstructor v5 Re-Training F11 | offen — sofort ausführbar | Masking-aware A1-Loss (Muster SOTA-A1/EAR-VAE), GPU-Run (~Stunden) | HF-Gain-Gate ≥ 1.02 (aktuell 0.73) → `use_bw_v5=True` als Shellac/Lacquer-HF-Pfad; sonst bleibt DSP-SBR/FlashSR-Paar |
| KP-8 | Cantus = Vocal-Inpainting-DiT-Lane (§C) | offen — Full-Training ausstehend (tiny-Smoke CPU erledigt) | GPU: pretrain → fine_tune → domain_adapt, Re-Export, Paritätscheck, A/B gegen MIIPHER-DiT+HNR-Baseline, Hörabnahme, Zoo-Status training→active | voller Vertrag (§8-Kontrakt); Daten `data/cantus_pairs` (3600 Paare) lokal; Hygiene s. §4/KP-8 |
| KP-9 | GaCELA Re-Training F2 (optional) | bedingt auf KP-8 | nur bei Lücken 375–1500 ms, die KP-8 nicht trägt; Task-Neuformulierung (Silence-Gap + Energy-Gating statt Full-Waveform-Match — Lehre aus der Verwerfung: fair validiert mean SDR −2.63 dB) | A/B gegen Cantus-Lane auf Langlücken |
| KP-10 | RT60-Regressor | offen | GPU-Training auf RIR-Paaren (lokal DFN-Assets `rir_sim_*.wav` + echte Aufnahmen) | ersetzt konservatives T60-Witness-Cap in phase_20/49; Beleg Validierung E.3 (heute reines DSP, kein Modellbezug) |
| KP-11 | Symphonia = Instrumental-Kern (Cantus-Spiegelbild) | pending_training | Registry-Pfad `symphonia.onnx` fehlt (Hygiene: auf echten Export zeigen oder anlegen); dann Full-Lauf wie KP-8 | gleicher Vertrag wie KP-8; Daten `data/symphonia_pairs(+_full/_batches)` lokal |
| KP-12 | SingMOS Pro Kalibrierung F10 | offen | Leichtes Fine-Tuning/Kalibrierung auf verarbeitete Musik-Vocals (Golden Listening Set + MUSHRA/WP-8) | Vokal-MOS-Gates treffen bei Musik-Pegeln/-Spektren korrekt (heute Eichung auf trockene Gesangsaufnahmen); Beleg Validierung C.4 |
| KP-13 | Material-Klassifikator (D-K0-8) | in-arbeit (WP-7 Daten-Aufbau) | 6-Klassen-CLAP-/Shallow-Head erst bei ≥10 Echt-Trägern je Klasse trainieren; danach Wiring-Entscheidung (CLAP-Zeuge vs. trainierter Kopf) + Sign-off | Accuracy gegen Human-Labels + A/B — Material steuert Bandbreiten-Ceilings, Defekt-Schwellen, Era-Priors (§V7 copilot-instructions.md: kein Rollout ohne A/B); Stand `docs/reports/current/2026-10-09_wp7_coverage.json` |
| KP-14 | MERT Genre-Head | offen | GTZAN/MagnaTagATune ODER MUSDB-basierte Genre-Labels beschaffen (keiner lokal, Validierung E.5) | Accuracy gegen Human-Labels + A/B der Goal-Adaption |
| KP-15 | Gender-Head F13 + Kalibrierung F14 | datenblockiert | Label-Korpus ≥50 Aufnahmen je Klasse lizenzkonform (`data/gender_labels.csv` fehlt); Skripte fertig | empirische Konsens-Rate (F14) mit Mindest-Stichprobe + Mindest-Konsens 0.80; Beleg Validierung D.7 |
| KP-16 | DDSP C4 Re-Training auf DAC-Encoder | bedingt | nur wenn val-MAE klar unter der Mean-Baseline liegt, sonst D-K0-6 akzeptieren (heute: 0.2447 ≈ 0.2455 ⇒ kein Skill) | ΔSDR-Beleg vs. CLAP-Head; Ersatz-Basis lokal `models/dac/encoder_model.onnx` |
| KP-17 | Neuronaler Warp-Schätzer WF-V4 | offen | Warp-Head auf FCPE/CRePE-F0-Features trainieren (beide Basen lokal — kein Download nötig) | Rest-Wow/Flutter unter der Hörschwelle + never-worsen gegen phase_12-Warp; Beleg Validierung E.1 |
| KP-18 | Neuronale Phasen-Schätzung für Transienten TP-V2 | offen | Vocoder als Phasen-Synthesizer / kleines Netz; Daten = transient-reiche Musiksegmente mit Phasengroundtruth aus sauberen Referenzen | Kammfilter-Freiheit (Hörordnung Ebene 3) + never-worsen; Beleg Validierung E.2 |
| KP-19 | AudioLDM2 Latent-Inpainting F6 (optional) | offen — Rollenentscheidung nach S1 | nur wenn KP-8 extreme Fälle nicht deckt: RePaint-artige CLAP-Audio-Konditionierung | A/B auf Dropouts > 3 s; Plugin existiert und ist in phase_24 Tier 3 verdrahtet (Validierung E.7) |

---

## 1. Entscheidungsprinzipien (normative Kette)

1. **Aurik-Kriterium §III.11 (copilot-instructions.md):** Sprache muss Gesang/Musik
   sein — ein sprachtrainiertes Modell im Signalpfad ist ein Konzeptdefizit.
2. **Metriken sind Zeugen, die Hör-Instanz entscheidet**
   (`hoerordnung.instructions.md` §1): Nie-geschlechter an der IN-V1/V2-Naht;
   Flag-Flip erst nach ONNX → Parität → A/B → Witness → Hörabnahme.
3. **Determinismus §G5 (copilot-instructions.md):** Seeds pro Session, kein
   `time.time()` in Entscheidungslogik.
4. **Silent-Failure-Verbot §V6 (copilot-instructions.md):** jeder ML→DSP-Fallback mit
   `logger.warning()` + Begründung.
5. **Kein Überschreiben ohne Evidenz:** bestehende Modelle werden erst ersetzt, wenn
   das A/B die Überlegenheit belegt (User-Ziel).
6. **Echte Musik / echter Gesang vor Synthetik** — so weit der Use-Case es erlaubt;
   Hybrid-Regel: echtes Material + synthetisches Defekt-Overlay = OK für
   Denoise-Fine-Tuning; für die Material-/Ära-Klassifikation ist echte Trägerfläche
   erforderlich (KP-13, Register §7).

---

## 2. Inventar: was schon erledigt ist (kein Training nötig) — Rollout-Queue des stillen Kapitals

| Modell | Zustand | Rest-Aktion | Gate / Beleg |
| --- | --- | --- | --- |
| DeepFilterNet v3 Musik | ✅ aktiv (`use_df_musik=True`) | — | §v10.15, finetuned enc/dec/erb_dec ONNX |
| MIIPHER-DiT | ✅ aktiv (`use_miipher_dit=True`) | — | ersetzt proprietären MIIPHER (§v10.14) |
| Harmonic-Inpainting-DiT | ✅ aktiv (`use_harmonic_inpainting=True`) | — | §v10.300 DiT-Finetune, gedämpfte Obertöne |
| MP-SENet Musik | ✅ aktiv (`use_mp_senet_musik=True`) | — | A/B 2026-09-20 bestanden (seg-SNR +5.3…+11.3 dB), §v10.17/§v10.25 |
| FlashSR F4 | ✅ abgeschlossen | — | Musik-Finetune verifiziert, Gate +18…+19.5 dB bestanden, Export aktiv (WP-4-Delta) |
| APPLADE/A-SPADE | ✅ Produktion phase_07 De-Clicker | — | A/B 2026-09-12 (SOTA-ML-V3) |
| EAR-VAE v1-MUSDB | ✅ deployt (Phase-0-Clean-Pass) | v2-Upstream beschafft (298 M) → Vergleichs-Benchmark ausstehend | D-K0-9: Never-worsen + Hörordnungs-Sign-off vor Austausch (§V7 copilot-instructions.md) |
| BigVGAN-F3 (MUSDB-Finetune) | exportiert + paritätsbewiesen (`bigvgan_v2_f3.onnx`, rel ≤ 6.5e-05) | C4-Hörstichprobe (W-1-Paket bereit, §10) → Rollout-Entscheidung | D-K0-1/D-K0-2; Budget-Weg 2 umgesetzt (1.04× RT je Passage) |
| SGMSE+ Musik-Core | trainiert (ep 80 / val loss 6.1525), ONNX-Parität max\|Δ\|=6.7e-5, Metrik-A/B ~1 dB besser als WSJ0-Core | Hörabnahme (3-Wege-Paket seit 10-23 bereit, §10) → Flag-Flip | D-K0-3; SI-SDR straft generative Diffusion stärker als die Hörewahrnehmung — Entscheidung braucht das Ohr (§III.11 copilot-instructions.md) |
| SCNet 4stems | verdrahtet hinter Never-worsen (`plugins/scnet_plugin.py`, `use_scnet_music=False`) | C4-r2-Hörprobe (Paket bereit, §10) + Sign-off → Aktivierung | D-K0-5; RT gemessen 3.43× Chunked / 2.20× Ganzsegment (Performance-Schuld D-K3-37 wächst bei Aktivierung) |
| VinylNet De-Crackle v4 (phase_68) | **neu seit Draft:** deployt + verdrahtet (`vinylnet_base_best.pt` ep52, Val SI-SDR +0.58 dB; `detector_v4.pt`) | Testsplit-Re-Eval mit dem neuen Artefakt + Hörtest gegen iZotope RX 12 auf echten Platten | D-K1-4 (in-arbeit); Report `docs/reports/current/2026-10-12_vinylnet_decrackle.md` |
| Whisper-Denoiser | deprecated (`use_whisper_denoiser=False`) | neu bewerten oder archivieren | D-K0-4 |
| AudioLDM2 | Plugin existiert, in phase_24 Tier 3 verdrahtet (Dropouts > 3 s) | Rollenentscheidung „generative Baseline für destruktive Fälle" nach S1 | §9/8; Validierung E.7 (alte Lücken-Matrix-Zeile „Plugin-Neuanlage fehlt" war veraltet) |

---

## 3. P0 — maximale Wirkung mit minimalem Aufwand (Substitution + Kalibrierung, kein Training)

### KP-1 · Rollout des stillen Kapitals (Hörproben-Runde)

Die drei Pakete liegen fertig; es fehlt nur die menschliche Instanz:

| Paket | Studie | Flag-Folgen |
| --- | --- | --- |
| SCNet C4 Runde 2 (12 Trials + 2 Fangfragen) | `output_audio/mushra/thresholds_scnet_c4_fair_r2` | `use_scnet_music=True`, wenn Sign-off + RT-Budget-Akzeptanz (§9/1) |
| HR-V1 W-1 Timbre-Konsistenz (Defekt-Frage, 8+2 Trials) | `output_audio/mushra/thresholds_wohlklang_w1_hrv1` | BigVGAN-F3-Rollout / C4 (§9/2) + KP-5-Vocoder-Entscheidung |
| SGMSE+ 3-Wege (18 Präferenz-Trials + 2 Fangfragen, level-matched) | `output_audio/mushra/thresholds_sgmse_3weg` | `use_sgmse_musik=True`, wenn Musik-Core in beiden Kopf-an-Kopf-Vergleichen trägt (§9/3); sonst WPE-only-Entscheid |

**Bau-Beweis SGMSE 3-Wege (10-23):** `scripts/build_sgmse_hoerprobe_3weg.py` —
Arme über die kanonischen Plugin-Pfade (WSJ0-Sprach-Core, Musik-Finetune-Core,
Nara-WPE-DSP), Degradation identisch zum Metrik-Lauf 10-08 (`w04`/Seed 42 bit-exakt
der Evidenz, `w20`/Seed 43 = erweiterte Stichprobe gemäß D-K0-3 „mehr Stichproben"),
Triangelschaltung musik↔wsj0 / musik↔wpe / wsj0↔wpe + 2 Fangfragen (ITU-R BS.1534-
Anker), doppelblind (A/B-Zuordnung aus sha256(seed:trial_id)), Level-Matched
(Präferenz-Typ). Manifest: `docs/reports/current/2026-10-23_hoerprobe_sgmse_3weg.json`.

**Menschliches Gate:** MUSHRA/Hörpanel (`scripts/hoerpanel_player.py`), Auswertung
`mushra_harness.py thresholds-fit`; Sign-off in `.github/SOTA_DEFICIT_REGISTER.md` —
dann bis zu **drei Flag-Flips ohne GPU**.

### KP-2 · Musik-MOS-Orakel für F8 (UTMOS-Ersatz)

UTMOSv2 ist auf Sprache (BVCC) trainiert und heute reiner Zeuge (report-only,
`use_utmos_music=False`). Die musiknative Substitution liegt **komplett lokal**:
MuQ + A1-Head (`muq_eval_a1_head.pt`, einziger richtungsvalidierter Zeuge 3/3) und
SingMOS Pro. **Pfad A (empfohlen, kein Training):** Zeugen-Paare bilden, Schwellen auf
einem MUSHRA-Musik-Subset kalibrieren; Gate = Richtungsstabilität über alle
Degradations-Klassen + Non-Inferiority gegenüber dem UTMOS-report-only-Zeitraum.
**Pfad B (Training)** nur wenn A nicht trägt: ≥50 Paare MUSHRA-Musik-Bewertungen
(= WP-8-Daten).

### KP-3 · Vokal-Identität F9 (Resemblyzer-Ersatz)

Resemblyzer (LibriSpeech) wirkte in Ebene-1-Guard und phase_65 als Richter;
`use_resemblyzer_music=False`, domänenneutrale DSP-Proxies tragen heute die Messung.
**Empfehlung: Ersetzung statt Finetune** — MERT (`mert_330m.onnx`) und MuQ-MuLan-
Embeddings liegen lokal (musiktrainiert). Gate = A/B der Witness-Entscheidungen auf
dem Golden Set (gleiche Entscheidungen → cos-invariant) + Hörabnahme.

### KP-4 · VAD: Silero → PANNs/AST

Silero (English Speech) steuert in phase_18 den Gate-Schwellen-Offset bis −15 dB;
`use_silero_vad_music=False` ⇒ Gate ohne Offset (neutral, §V7 copilot-instructions.md).
Musikadaptierter Ersatz: PANNs-Singing-Klasse / AST als Gate-Offset (Muster phase_19).
Gate = A/B auf Korpus, nie-geschlechter der Phase-18-Entscheidungen.

### KP-5 · Studio-Finisher-Vocoder-Entscheidung (D-K2-3)

Vocos 48 kHz ist **primäre** Stufe des §1.4-Studio-Finishers (Maximum/Studio/
Aggressive, PQS-MOS < 4.3), Domäne = Sprache ohne Gesang (`kittn/vocos-mel-48khz-
alpha1`, Owner-Korpora Podcasts/VCTK/Expresso) — das wichtigste
Sprache-im-Signalpfad-Fix ohne neues Training. **Empfehlung:** BigVGAN-F3 (schon
musiktrainiert, exportiert + paritätsbewiesen) als Stufe 1 des Studio-Pfads nach der
W-1-Hörprobe; Vocos bleibt §V6-Fallback. Falls das RT-Budget scheitert: Vocos-Vokal-
Finetune auf MUSDB18-HQ (Hausroutine).

---

## 4. P1 — GPU-Trainingsläufe (7900 XTX, ROCm)

**Allgemeiner Vertrag jedes Laufs:** ONNX-Export → Torch-ROCm-Paritäts-Gate
(rel ≤ 1e-3 auf strukturierten Feeds, §III.9 copilot-instructions.md) → A/B-Gate →
Witness-Belege → Hörordnungs-Abnahme → erst dann Feature-Flag-Flip
(`backend/core/music_model_flags.py`). Der Skript-Lauf endet mit Evidenz-Report —
**nie** mit automatischem Flag-Flip (§V7 copilot-instructions.md).

### KP-6 · BANQUET Real-Vinyl-Finetune F12 (+ Upstream-Optimalitäts-Bewertung)

- **Upstream:** `kwatcharasupat/query-bandit` (ISMIR 2024, MoisesDB = Musik ✓ MIT;
  Zenodo `ev-pre-aug.ckpt`). **Optimalitäts-Urteil:** Architektur passt (Band-Split-
  RNN 24-Zellen, 23.69 M Par. — Low-Band-Rumble + High-Band-Crackle abdeckend; CPU
  parallel 1.58–1.61×); der Query-/PaSST-Anteil wird für Blind-Enhancement nicht
  gebraucht (ist in `vinyl_final` eingekocht).
- **Aurik-Use-Case:** phase_09-Vinyl-Crackle, blinder spektraler Masken-Enhancer
  `[1,128,128,128]` → gleiche Form; DSP-Fallback = Median-De-Clicker + HPF (§V6
  copilot-instructions.md); Gate `QUALITY_MODE` und `_is_vinyl` und
  `is_phase_ml_enabled(9)` (Root-Fix 2026-09-12). **Kontext seit Draft:** phase_68
  (VinylNet) deckt jetzt die chirurgische Klick-/Crackle-Lücke als Either-Or-Nachpass;
  BANQUET bleibt in der Blind-Masken-Enhancer-Lane von phase_09.
- **Erforderliche Anpassungen:**
  a) **Daten:** echte Vinyl↔Reissue-Paare (≥30, deckend Crackle/Hiss/Wow/
     Oberflächenrauschen); lokal nur 6 synthetische CC0-Paare (Validierung D.2).
     Hybrid-Ansatz akzeptabel (echtes PD-Material + Defekt-Overlay), für die
     Material-Klassifikation aber echte Trägerfläche (KP-13).
  b) **Provenienz-Hygiene:** `ev-pre-aug.ckpt` wird von `upload_models_to_hf.py`
     referenziert, fehlt in `models/banquet/`; kein SHA-Link zwischen Upstream-Zenodo-
     Gewichten und deploytem ONNX → Provenzkette dokumentieren oder aus
     `ev-pre-aug.ckpt` ableiten (§III.13 copilot-instructions.md).
  c) **Gate:** ΔSDR ≥ +2 dB vs. Zero-Shot auf destruktiven Fällen + never-worsen
     ≥ −0.1 dB auf sauberen Referenzen → voller Vertrag; Evidenz-Report am Ende, kein
     automatischer Flag-Flip.
- **Schätzung:** Daten-Beschaffung ist der Flaschenhals; der Run selbst dauert wenige
  Stunden (23.7 M Par.).

### KP-7 · BW-Reconstructor v5 Re-Training F11

Masking-aware A1-Loss (Muster SOTA-A1, EAR-VAE); Ziel HF-Gain-Gate ≥ 1.02 (aktuell
0.73). Bestanden → `use_bw_v5=True` als Shellac/Lacquer-HF-Pfad; sonst bleibt das
DSP-SBR/FlashSR-Paar der HF-Weg (Flag aus).

### KP-8 · Cantus = Vocal-Inpainting-DiT-Lane (§C)

Modell existiert (Full-Preset 204.27 M, ONNX-Export Parität rel ≤ 2.2e-5), in
StemLevelRestorer §SLR-1e2 hinter Zoo-Status-Gate + Halluzinations-Guard verdrahtet;
Pretrain-Tiny-Smoke erledigt (CPU); Daten `data/cantus_pairs` (3600 Paare, echte Musik
+ synthetische Degradation) lokal. **Rest:** Full-Preset-Training auf GPU (pretrain →
fine_tune → domain_adapt, 3 Phasen), Re-Export, Paritätscheck, A/B gegen die
MIIPHER-DiT+HNR-Baseline, Hörabnahme, Zoo-Status training→active.
**Hygiene:** README behauptet `checkpoint_best.pt` = „204.27 M Par., epoch 10" — ist
tatsächlich Tiny (0.37 M Par., ep1/step2); `parity.json` referenziert über schriebene
Checkpoints; Platzhalter-Dateien neben echten Artefakten → aktualisieren.

### KP-9 · GaCELA Re-Training F2 (optional)

Nur wenn KP-8 für Lücken 375–1500 ms nicht ausreicht; Task-Neuformulierung (Lehre
aus der GaCELA-Verwerfung: Silence-Gap + Energy-Gating statt Full-Waveform-Match).

### KP-10 · RT60-Regressor

GPU-Training auf RIR-Paaren (lokale DFN-Assets `rir_sim_*.wav` + echte Aufnahmen);
ersetzt das konservative T60-Witness-Cap in phase_20/49.

### KP-11 · Symphonia = Instrumental-Kern (Cantus-Spiegelbild)

Status pending_training, nur Tiny-ONNX; Registry-Pfad `symphonia.onnx` fehlt
(Hygiene: auf echten Export zeigen oder anlegen); Daten `data/symphonia_pairs(
+_full/_batches)` lokal. Nach dem Training → gleicher Vertrag wie KP-8.

### KP-12 · SingMOS Pro Kalibrierung F10

Leichtes Fine-Tuning/Kalibrierung des Singing-MOS auf verarbeitete Musik-Vocals —
Daten aus Golden Listening Set + MUSHRA (WP-8).

---

## 5. P2 — datenabhängig / bedingt

| KP | Aufgabe | Bedingung | Gate |
| --- | --- | --- | --- |
| KP-13 | Material-Klassifikator (D-K0-8) | WP-7-Stand: 143 Labels gesamt, Echt-Träger shellac 47 / digital 10 / cassette 4 / reel 1 / tape 1 / **vinyl 0** (`wp7_coverage.json`); Ziel ≥10 je Klasse nicht erreicht — „shellac 47 vs. 58"-Diskrepanz aufgelöst (zwei verschiedene Zähler desselben JSON) | Training des 6-Klassen-CLAP-/Shallow-Heads erst bei ≥10 Echt-Trägern je Klasse; Vinyl/Tape brauchen Lizenz-Politik (§9/7); danach Wiring + Sign-off |
| KP-14 | MERT Genre-Head | GTZAN/MagnaTagATune ODER MUSDB-basierte Genre-Labels beschaffen (keiner lokal) | Accuracy gegen Human-Labels + A/B der Goal-Adaption |
| KP-15 | Gender-Head F13 + Kalibrierung F14 | Label-Korpus ≥50 je Klasse (`data/gender_labels.csv` fehlt), lizenzkonform nach `corpus/README.md` | empirische Konsens-Rate (F14) mit Mindest-Stichprobe und Mindest-Konsens 0.80, als „empirisch, kein Ground Truth" markiert |
| KP-16 | DDSP C4 Re-Training auf DAC-Encoder | nur wenn val-MAE klar unter der Mean-Baseline (heute 0.2447 ≈ 0.2455 ⇒ kein Skill) | ΔSDR-Beleg; sonst D-K0-6 akzeptiert |

---

## 6. P3 — neue Modelle (keine lokale Basis)

| KP | Aufgabe | Daten/Basis | Gate |
| --- | --- | --- | --- |
| KP-17 | Neuronaler Warp-Schätzer WF-V4 | Warp-Head auf FCPE/CRePE-F0-Features (beide lokal — kein Download nötig) | Rest-Wow/Flutter unter der Hörschwelle + never-worsen gegen phase_12-Warp |
| KP-18 | Neuronale Phasen-Schätzung für Transienten TP-V2 | Vocoder als Phasen-Synthesizer / kleines Netz; Daten = transient-reiche Musiksegmente mit Phasengroundtruth aus sauberen Referenzen (lokale Vocodere: Vocos/BigVGAN/DAC) | Kammfilter-Freiheit + never-worsen |
| KP-19 | AudioLDM2 Latent-Inpainting F6 (optional) | RePaint-artige CLAP-Audio-Konditionierung — nur wenn KP-8 extreme Fälle nicht deckt; Plugin existiert, in phase_24 Tier 3 verdrahtet | A/B auf Dropouts > 3 s |

---

## 7. Synthetik → Echt-Musik-Ersatz-Register

| Synthese (Test/Training) | Ort | Ersatz mit echter Musik/Gesang | Status |
| --- | --- | --- | --- |
| BANQUET `corpus/vinyl` (16 damaged / 6 clean, alle synthetische CC0) | `scripts/train_banquet_vinyl_finetune.py` Default-Korpus | echte Vinyl↔Reissue-Paare | offen (KP-6; Lizenz-Politik §9/7) |
| VinylNet De-Crackle Noise-Bank = **0** in allen Läufen (Hybrid: echtes MUSDB-Material + synthetische Degradation) | `models/de_crackle/simulate.py` | echte Leerrillen-Aufnahmen (`--noise-bank`: Rillenrauschen vieler Platten/Tonabnehmer/Phono-Stufen) | offen (D-K1-4 — größter Hebel für Praxistransfer) |
| Material-Klassifikator Träger: 63 Echt-Träger statt zuvor 12 (WP-7 läuft; vinyl strukturell 0, Tape/Reel minimal) | `docs/reports/current/2026-10-09_wp7_realtraeger.md` | Lizenz-Politik-Entscheid für Vinyl/Tape | in-arbeit (D-K1-2, §9/7) |
| Gender-Labels: Daten fehlen | `data/gender_labels.csv` | ≥50 je Klasse, lizenzkonform | offen (KP-15) |
| SGMSE A/B n=3 @ 10 dB SNR + synthetischer Nachhall | `scripts/benchmark_sgmse_ab.py` | 3-Wege-Paket nutzt dieselbe deterministische Degradation wie die Metrik-Evidenz (Vergleichbarkeit); echter Raum-Nachhall als Erweiterung nach dem Sign-off | erledigt für die Abnahme (KP-1, seit 10-23) |
| DFN gepaarte Re-Training-Kandidaten (§v10.25 P2 3–4) | — | echtes PD-Musikmaterial + aufgesetzte Defekte = OK für Denoise-Fine-Tuning (Hybrid-Regel §1/6) | dokumentiert |
| Cantus-Paare: 3600 MUSDB-HQ-Vocals mit synthetischer Degradation | `data/cantus_pairs` | Hybrid akzeptabel fürs Flow-Matching-Pretraining; die domain_adapt-Phase soll echte degradierte Aufnahmen nutzen, wo verfügbar | dokumentiert (KP-8) |

---

## 8. Sequenzierung und GPU-Budget

- **Welle A (jetzt, kein Training):** Hörproben-Runde — alle drei Pakete sind gebaut
  (§3/KP-1); bis zu drei Flag-Flips ohne GPU nach Sign-off.
- **Welle B (GPU parallel in VRAM-Grenzen):** KP-6 BANQUET (erst Daten), KP-7 BW-v5
  (~Stunden), KP-10 RT60, KP-8 Cantus Full-Pretrain.
- **Welle C:** KP-8 fine_tune/domain_adapt + Export/Parität/A/B; danach bedingt
  KP-9 / KP-12 / KP-11 Symphonia.
- **MUSHRA-Studie WP-8 (n ≥ 30)** parallel als menschlicher Flaschenhalm — kalibriert
  F8 und stützt alle Hörabnahmen.
- **VRAM:** 23.9 GiB frei gemessen (`torch.cuda.mem_get_info`, Validierung); Cantus
  Full (204 M Par., fp32 ≈ ~1 GB Gewichte + Aktivierungen bei Chunk 4 s / Batch 8)
  passt; Runs sequenziell, außer VRAM erlaubt Parallelität.

---

## 9. Offene Entscheidungen (User-/Maintainer-Sign-off)

| # | Entscheidung | Empfehlung | Kontext |
| --- | --- | --- | --- |
| 1 | SCNet-Aktivierung nach C4-r2 (+ RT-Budget-Akzeptanz D-K3-37/D-K0-5: 3.43× Chunked / 2.20× Ganzsegment) | aktivieren, wenn Hörabnahme trägt und Performance-Schuld akzeptiert ist | §9/1 der Lücken-Matrix; Paket bereit (§10) |
| 2 | BigVGAN-F3-Rollout in die Studio-Finisher-Kaskade statt Vocos-48k-Primär (KP-5) | nach W-1-Sign-off: F3 als Stufe 1, Vocos bleibt §V6-Fallback | D-K0-1/D-K2-3; W-1-Paket bereit (§10) |
| 3 | SGMSE-Flag-Flip nach Hörabnahme (oder WPE-only-Entscheid) | Musik-Core nur bei gewonnenen Kopf-an-Kopf-Vergleichen in der Abnahme | D-K0-3; 3-Wege-Paket seit 10-23 bereit (§10) |
| 4 | EAR-VAE v1↔v2 Vergleichs-Benchmark | Never-worsen + Sign-off vor jedem Austausch (heute: v1-MUSDB-Finetune produktiv) | D-K0-9; v2-Upstream liegt lokal (298 M) |
| 5 | Whisper-Denoiser: archivieren oder neu bewerten | je nach Nutzen-Nachweis in der Spec-04-Kette (NR trägt heute) | D-K0-4; Flag False, deprecated |
| 6 | AERO: MUSHRA-Challenger-Runde vs. Task-Reformulierung vs. Archiv | bisheriger A/B-Verlust gegen FlashSR (ΔSDR min −2.4 / mean −1.6 dB) — erst menschliche Runde | D-K2-5; Challenger-Infrastruktur vollständig |
| 7 | Vinyl-Lizenz-Politik für den Echt-Träger-Korpus (Unlocked Recordings; D-K1-2) | Freigabe-Politik festlegen — blockiert KP-6a + KP-13-Vinyl-Klasse (+ Leerrillen-Nachschub für D-K1-4) | WP-7: vinyl strukturell 0 (LP-Ära unter Schutz, Upload-CC0 auf Kommerz-Releases verworfen) |
| 8 | AudioLDM2-Rolle als generative Baseline nach S1 | Rollenentscheidung erst nach S1-Ergebnis (Plugin liegt bereit in phase_24 Tier 3) | §9/8; Validierung E.7 |

---

## 10. Hörproben (gesammelt — je Paket die Entscheidung, die es ermöglicht)

| Paket | Pfad | Ermöglichte Entscheidung | Manifest / Anleitung | Status |
| --- | --- | --- | --- | --- |
| SCNet C4 Runde 2 | `output_audio/mushra/thresholds_scnet_c4_fair_r2` | `use_scnet_music`-Flag-Flip (D-K0-5) | `docs/reports/current/2026-10-08_hoerprobe_scnet_c4_r2.json`; Anleitung §3 (`…2026-10-07_hoerprobe_anleitung.md`) | **bereit** (12 Trials + 2 Fangfragen) |
| HR-V1 W-1 Timbre-Konsistenz | `output_audio/mushra/thresholds_wohlklang_w1_hrv1/` | BigVGAN-F3-Rollout / C4 (D-K0-1/D-K0-2) + KP-5 | `docs/reports/current/2026-10-07_hoerprobe_w1_hrv1.json`; Anleitung §4 | **bereit** (8+2 Trials, Produktionspfad seit D-K3-6-Fix) |
| SGMSE 3-Wege Hörabnahme | `output_audio/mushra/thresholds_sgmse_3weg/` | `use_sgmse_musik`-Flag-Flip ODER WPE-only (D-K0-3) | `docs/reports/current/2026-10-23_hoerprobe_sgmse_3weg.json`; Anleitung §5; Bau: `scripts/build_sgmse_hoerprobe_3weg.py` | **bereit seit 10-23** (18 Präferenz-Trials + 2 Fangfragen, level-matched) |
| MUSHRA-Smoke-Paket | `output_audio/mushra/smoke_player`, `smoke_thresholds` | Tooling-Verifikation | — | erledigt |
