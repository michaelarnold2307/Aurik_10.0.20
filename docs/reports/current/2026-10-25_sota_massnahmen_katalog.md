# SOTA-Maßnahmen-Katalog — nächste Maßnahmen bis „vollkommen SOTA“ (Stand 2026-10-25)

> **Zweck:** Arbeitsauftrag der nächsten Sessions — formuliert aus der
> Projektdokumentation, als Abarbeitungsplan für den Weg zu vollkommenem SOTA
> (weltbester Wohlklang). Jede Position trägt ihre Doku-Quelle; keine Position
> wird ohne Beleg abgehakt (Zero-Gap-Regel: GESCHLOSSEN / ERSETZT / GESTRICHEN).
> **Quellen (normativ):** `.github/SOTA_DEFICIT_REGISTER.md` (D-K-IDs, Klassen
> K0–K4), `docs/SOTA_TRAINING_KONZEPT.md` (Live-Stand 2026-10-23, v10.12.43,
> KP-1…KP-19, Wellen A–C), `docs/TODOS_SOTA_ROADMAP.md` (Lücken-Matrix
> 2026-10-06, WP-0…WP-8), `docs/reports/current/2026-10-04_offene_massnahmen.md`
> (Lanes M1–M3, Sign-offs B1–B4, Lanes E1–E3), `docs/TIEFENANALYSE_SOTA_ABGLEICH.md`
> (K4-Klassen-Lücken).
> **Arbeitssprache Deutsch; §-Zitate immer mit Quelle (AGENTS.md §5
> ID-Zitierdisziplin).**

## Grundregeln jeder Maßnahme (normativ, unverändert übernommen)

- Jeder Modell-Rollout durchläuft den vollen Vertrag: **ONNX-Export →
  Torch-ROCm-Parität (rel ≤ 1e-3 auf strukturierten Feeds, §III.9
  (copilot-instructions.md)) → A/B-Gate → Witness-Belege →
  Hörordnungs-Abnahme → erst dann Feature-Flag-Flip**
  (`backend/core/music_model_flags.py`). Nie automatischer Flag-Flip
  (§V7 (copilot-instructions.md)).
- **Aurik-Kriterium §III.11 (copilot-instructions.md):** Sprache muss Gesang
  sein — ein sprachtrainiertes Modell im Signalpfad oder als Richter ist ein
  Konzeptdefizit.
- **§V6 (copilot-instructions.md):** jeder ML→DSP-Fallback mit
  `logger.warning()` + Begründung; **§G5 (copilot-instructions.md):**
  Determinismus (Seeds pro Session).
- **Hörordnung Ebene 4** (`hoerordnung.instructions.md`): das menschliche Ohr
  ist die letzte Instanz; jede klangverändernde Stufe braucht die fünf Belege
  des Wohlklang-Vertrags (`.github/WOHLKLANG_CLAIMS.md`).
- Jede Position endet in GENAU einem Endzustand — GESCHLOSSEN (Artefakt +
  Beweis), ERSETZT (Substitut + Paritäts-/Never-worsen-Nachweis) oder
  GESTRICHEN (Sign-off + Begründung). „Extern blockiert“ ist kein Endzustand.

---

## Welle A — Sofort, ohne GPU: stilles Kapital rollen (größter Hebel)

Der Engpass ist ausschließlich die menschliche Hörinstanz; alle drei
Hörproben-Pakete sind gebaut und bereit (`docs/SOTA_TRAINING_KONZEPT.md` §0/§10).

### A1 · Hörproben-Runde fahren (KP-1) — drei Flag-Flips ohne GPU

- **Ausführung:** `python scripts/hoerpanel_player.py --study
  output_audio/mushra/thresholds_scnet_c4_fair_r2` (12 Trials + 2 Fangfragen),
  danach `output_audio/mushra/thresholds_wohlklang_w1_hrv1/` (8+2) und
  `output_audio/mushra/thresholds_sgmse_3weg/` (18+2, level-matched, bereit
  seit 10-23). Auswertung `mushra_harness.py thresholds-fit`; Sign-off in
  `.github/SOTA_DEFICIT_REGISTER.md` dokumentieren (Statuswechsel nur mit
  Beleg-Pfad — Gate `scripts/sota_deficit_gate.py`).
- **Ermöglichte Flag-Flips:**
  - SCNet C4-r2 → `use_scnet_music=True` (D-K0-5). Vorab: RT-Budget-Akzeptanz
    (3,43× RT Chunked / 2,20× Ganzsegment; Performance-Schuld D-K3-37).
  - HR-V1 W-1 Timbre-Konsistenz → BigVGAN-F3-Rollout (D-K0-1/D-K0-2) und
    Studio-Finisher-Vocoder-Entscheidung (D-K2-3, KP-5).
  - SGMSE+ 3-Wege → `use_sgmse_musik=True`, wenn der Musik-Core beide
    Kopf-an-Kopf-Vergleiche trägt; sonst WPE-only-Entscheid (D-K0-3).
- **Verifikation:** Manifeste `docs/reports/current/2026-10-08_hoerprobe_scnet_c4_r2.json`,
  `…2026-10-07_hoerprobe_w1_hrv1.json`, `…2026-10-23_hoerprobe_sgmse_3weg.json`;
  Anleitung `docs/reports/current/2026-10-07_hoerprobe_anleitung.md`.

### A2 · Offene Sign-off-Entscheidungen §9 treffen (User/Maintainer)

| # | Entscheidung | Empfehlung (Doku) | Quelle |
| --- | --- | --- | --- |
| 1 | SCNet-Aktivierung nach C4-r2 | aktivieren, wenn Hörabnahme trägt und RT-Schuld akzeptiert | D-K0-5, §9/1 |
| 2 | BigVGAN-F3 in Studio-Finisher statt Vocos-48k-Primär | nach W-1: F3 als Stufe 1, Vocos bleibt §V6-Fallback | D-K0-1/D-K2-3, KP-5 |
| 3 | SGMSE-Flag-Flip (oder WPE-only) | nur bei gewonnenen Kopf-an-Kopf-Vergleichen | D-K0-3, §9/3 |
| 4 | EAR-VAE v1↔v2 Vergleichs-Benchmark | Never-worsen + Sign-off vor Austausch | D-K0-9, §9/4 |
| 5 | Whisper-Denoiser | archivieren oder neu bewerten | D-K0-4, §9/5 |
| 6 | AERO | MUSHRA-Challenger-Runde (bisheriger A/B-Verlust gegen FlashSR) | D-K2-5, §9/6 |
| 7 | **Vinyl-Lizenz-Politik** (Unlocked Recordings) | Freigabe-Politik festlegen — blockiert KP-6a, KP-13-Vinyl, Leerrillen-Nachschub (D-K1-4) | D-K1-2, §9/7 |
| 8 | AudioLDM2-Rolle als generative Baseline | nach S1-Ergebnis | §9/8 |

### A3 · MUSHRA-Entscheidung (B2 / WP-8)

- Studie n≥30 terminieren **oder** formal beschließen, dass die Zwischenabnahme
  über Objektivmetriken läuft und die Studie nachgelagert ist — beides ist ein
  gültiger Endzustand, „extern blockiert (Hörer)“ nicht.
- Vorbedingung für KP-2-Pfad-B (UTMOS-Ersatz-Training) und MUSHRA-Proxy §3.6.

### A4 · Hängende Abnahmen schließen (aus der Abarbeitungsliste 2026-10-04)

- T6-1-Integration: D4 (Report + Ledger + Roadmap) → **T6-1 GESCHLOSSEN**;
  D5 Hörordnungs-Abnahme + Sign-off Optionsvertrag A; D6 optionaler Nachlauf
  mit den echten Referenz-Songs.
- SGMSE-Flag: A4 TorchScript-Export + Manifest-SHA, A5 Never-worsen vs.
  `sgmse_plus.ts` + Sign-off für `use_sgmse_musik=True` (deckungsgleich mit A1).
- B1: Hörstichproben (C4/D5) terminieren; B3: F11-Re-Training freigeben
  (= KP-7); B4: F8-Musik-Orakel priorisieren (= KP-2).

---

## Welle B — Autonome P0-Substitutionen (CPU, kein Training): Sprache aus Signalpfad & Richterbank

Substitution jeweils komplett lokal; Ziel: §III.11 (copilot-instructions.md)
durchgängig erfüllen.

### B1 · KP-2 Musik-MOS-Orakel (F8, höchster Oracle-Hebel)

- UTMOSv2 (Sprach-MOS, richtungsblind/invertiert — Negativbefund) ersetzen
  durch MuQ + A1-Head (`muq_eval_a1_head.pt`, einziger richtungsvalidierter
  Zeuge 3/3) + SingMOS Pro. **Pfad A (empfohlen, kein Training):**
  Zeugen-Paare + Schwellen-Kalibrierung auf MUSHRA-Subset; Anbindung in
  `utmos_delta_gate` (report-only → Veto nach Kalibrierung).
- **Gate:** Richtungsstabilität über alle Degradations-Klassen +
  Non-Inferiority gegen UTMOS-report-only-Zeitraum; Flag
  `use_utmos_music=False` bleibt bis dahin.

### B2 · KP-3 Vokal-Identität (F9): Resemblyzer-Ersatz

- MERT (`mert_330m.onnx`) + MuQ-MuLan-Embeddings (musiktrainiert, lokal) als
  Ebene-1-Guard- und phase_65-Kandidat. **Empfehlung Doku: Ersetzung statt
  Finetune.**
- **Gate:** A/B der Witness-Entscheidungen auf dem Golden Set (gleiche
  Entscheidungen → cos-invariant) + Hörabnahme.

### B3 · KP-4 VAD: Silero → PANNs-Singing/AST

- Musikadaptierter Gate-Offset (Muster phase_19) statt Silero (English Speech).
- **Gate:** A/B auf Korpus, Never-worsen der phase_18-Gate-Entscheidungen.

### B4 · KP-5 Studio-Finisher-Vocoder (D-K2-3)

- Vocos-48k ist primäre Stufe des §1.4-Studio-Finishers, Domäne Sprache ohne
  Gesang (Owner-Korpora Podcasts/VCTK/Expresso) — wichtigster
  Sprache-im-Signalpfad-Fix ohne neues Training.
- **Empfehlung:** BigVGAN-F3 (musiktrainiert, exportiert + paritätsbewiesen)
  als Stufe 1 nach W-1-Hörprobe; Vocos bleibt §V6-Fallback. Bei
  RT-Budget-Verstoß: Vocos-Vokal-Finetune auf MUSDB18-HQ.

---

## Welle C — GPU-Läufe (7900 XTX, ROCm; Vertrag §4 des SOTA-Trainingskonzepts)

### C1 · KP-7 BW-Reconstructor v5 Re-Training (F11) — sofort ausführbar

- Masking-aware A1-Loss (Muster SOTA-A1/EAR-VAE); Ziel HF-Gain-Gate ≥ 1,02
  (aktuell 0,73). Bestanden → `use_bw_v5=True` als Shellac/Lacquer-HF-Pfad;
  sonst bleibt das DSP-SBR/FlashSR-Paar.

### C2 · KP-8 Cantus = Vocal-Inpainting-DiT-Lane — sofort ausführbar

- GPU: pretrain → fine_tune → domain_adapt (Daten `data/cantus_pairs`, 3600
  Paare, lokal; VRAM reicht laut §8). Danach Re-Export, Paritätscheck,
  A/B gegen MIIPHER-DiT+HNR-Baseline, Hörabnahme, Zoo-Status
  training→active. Hygiene: README-/`parity.json`-Behauptungen korrigieren
  (Tiny vs. Full-Preset).

### C3 · KP-10 RT60-Regressor

- GPU-Training auf RIR-Paaren (lokale DFN-Assets `rir_sim_*.wav` + echte
  Aufnahmen); ersetzt das konservative T60-Witness-Cap in phase_20/49.

### C4 · KP-6 BANQUET Real-Vinyl-Finetune (F12) — nach Datenbeschaffung

- Echte Vinyl↔Reissue-Paare (≥30) beschaffen (heute nur 16 damaged/6 clean,
  alle synthetische CC0) → GPU-Run (wenige Stunden, 23,7 M Par.). Provenienz-
  Hygiene: SHA-Kette Upstream-Zenodo → deploytes ONNX dokumentieren.
- **Gate:** ΔSDR ≥ +2 dB vs. Zero-Shot auf destruktiven Fällen + never-worsen
  ≥ −0,1 dB auf sauberen Referenzen.

### C5 · Welle-C-Abschlüsse (bedingt / pending)

- KP-11 Symphonia (Instrumental-Kern, Cantus-Spiegelbild; Registry-Pfad
  `symphonia.onnx` Hygiene erledigt 2026-10-27: Resolver zeigt auf `symphonia_tiny.onnx`), KP-12 SingMOS-Pro-Kalibrierung
  F10, KP-9 GaCELA nur bei Cantus-Lücken 375–1500 ms (Task-Neuformulierung:
  Silence-Gap + Energy-Gating).

---

## Welle D — Daten-Beschaffung (der eigentliche Flaschenhals)

| # | Datum | Zweck | Blocker | Quelle |
| --- | --- | --- | --- | --- |
| D1 | Echte Vinyl↔Reissue-Paare (≥30) | KP-6 | Lizenz-Politik §9/7 | KP-6a |
| D2 | Leerrillen-Noise-Bank | VinylNet-Praxistransfer (Noise-Bank = 0 — größter Hebel) | echte Platten/Tonabnehmer/Phono-Stufen | D-K1-4 |
| D3 | Echt-Träger ≥10 je Material-Klasse | KP-13 (vinyl = 0, tape/reel minimal) | Lizenz-Politik §9/7 | WP-7/`wp7_coverage.json` |
| D4 | Gender-Labels ≥50 je Klasse | KP-15 (F13/F14) | `data/gender_labels.csv` fehlt | KP-15 |
| D5 | Genre-Labels (GTZAN/MagnaTagATune oder MUSDB-basiert) | KP-14 MERT-Head | keiner lokal | KP-14 |
| D6 | MUSHRA-Daten n≥30 | KP-2-Pfad-B, KP-12, Proxy-Kalibrierung | A3-Entscheidung | WP-8 |

---

## Welle E — Hartes SOTA-Gleis: Hygiene & Klassen-Lücken (K2–K4)

### E1 · WP-2 Phantom-ML + Phantom-Referenzen (risikoarm, sofort sichtbar)

- 17 TorchScript-Stubs (D-K3-3: 14× `adaptive_*.py`, `automatic_declipper.py`,
  `masking_aware_dynamic_eq.py`, `psychoacoustic_enhancement.py`) ehrlich
  deklarieren oder implementieren; §V6-Warnung bleibt Pflicht.
- Restliche Phantom-Referenzen aus D-K3-5-Prosa-Kandidaten einzeln prüfen.

### E2 · WP-1 §III.11-Lücken-Audit (Richter **und** Signalpfad, K2)

- Vocoder/SR-Pfade mit Domänen-Risiko: `hifi_gan` (D-K2-1), `vocos` (D-K2-2),
  `vocos_48khz` (D-K2-3), `nvsr` (D-K2-4, Namens-Drift), `aero` (D-K2-5).
- Drei Domänen unbelegt (`unbekannt` statt Annahme): `bigvgan_v2.onnx`,
  `hifi_gan.onnx`, `vocos_48khz.onnx` — §V6-Fallback dokumentieren bzw.
  Musik-Ersatz (BigVGAN-F3) verdrahten.

### E3 · WP-3 Wohlklang-Ordnung-Forensik

- Inkl. Vocoder-Hypothese: der sprachtrainierte Vocoder als Kandidat für
  Ordnungsverstöße — mitprüfen.

### E4 · K3-Restbestand (gemessen, offen)

- Denker-Baustellen: SNR-adaptive `CAUSE_PARAMS` inaktiv (D-K3-21),
  Budget-Divergenz 73,6× vs. 32× (D-K3-22, P0-3-Budget-Wahrheit), toter Code
  RestaurierDenker + doppelter PerceptualQualityCouncil (D-K3-23),
  `try/except: pass`-Testkonventionen (D-K3-20).
- 32×-End-to-End-Guard wirksam machen (D-K3-37: Durchsetzung als Default oder
  ehrliche Monitoring-Deklaration), Defect-Scanner 39× über Budget (D-K3-38/40),
  WCS-Gate-Metadaten im Chunked-Export (D-K3-60), Dither-Kaskade
  restore×Export (D-K3-64), P0-1-Abdeckungslücke EmotionalArc/Goosebumps/MDEM
  je Chunk (D-K3-65), Layout-Reduktionen (D-K3-8, nachrangig),
  AM-Kanal-Peak-Kalibrierung (D-K3-54).

### E5 · K4-Klassenmodelle (TIEFENANALYSE SOTA-ABGLEICH)

- D-K4-1: neuronaler A-SPADE-Klassen-Declipper als Phase-07-Challenger
  (PCHIP bleibt §V6-DSP-Fallback; Musik-Finetune + A/B vor Rollout).
- D-K4-2: nicht-kausaler Vollband-Denoiser (TF-GridNet-Klasse) als optionale
  Quality-Mode-Stufe; DFN bleibt kausaler Echtzeit-/Fallback-Pfad.
- D-K4-3: song-globaler Audio-Pfad (Ganzsong-Pipeline, Per-Stem-Restaurierung
  als Standard); Vorab-Entscheid D-K3-65 über Hörstichprobe.
- KP-17 WF-V4: Warp-Head auf FCPE/CRePE-F0-Features (beide Basen lokal).
- KP-18 TP-V2: neuronale Phasen-Schätzung für Transienten (Kammfilter-Freiheit).

### E6 · Doku-Konsistenz

- Das Defizit-Register verweist auf einen Roadmap-Abschnitt „SOTA-Maßnahmen
  2026-10-25 — Welle A–F“ (TODO-Blöcke `SOTA-M1`…`SOTA-M23`) — im Repo noch
  nicht vorhanden (gemessen 2026-10-25). Abschnitt in
  `docs/TODOS_SOTA_ROADMAP.md` anlegen (dieser Katalog liefert die Inhalte)
  und die SOTA-M-Blocker in den Blocker-Feldern des Registers nachtragen.
- Nach jeder Welle: `python scripts/sota_deficit_gate.py` +
  `tests/normative/test_sota_deficit_gate.py` lokal laufen lassen.

---

## Empfohlene Reihenfolge (Wirkung × Machbarkeit)

1. **Welle A sofort** (drei fertige Pakete → bis zu drei Flag-Flips ohne GPU;
   A2-Entscheidungen, insbesondere §9/7-Lizenzpolitik, parallel anstoßen).
2. **Welle B** (CPU-Substitutionen KP-2…KP-5) und die **freien GPU-Läufe**
   KP-7/KP-8/KP-10 (Welle C) parallel.
3. **Welle D** (Daten) als Dauerläufer — jeder Beschaffungsabschluss schaltet
   eine GPU- oder Trainings-Lane frei.
4. **Welle E** als kontinuierlicher Backlog (E1/E2 zuerst — risikoarm,
   sofort sichtbar), MUSHRA (A3/WP-8) parallel als menschlicher Flaschenhals.

## Beleg-Verweise (kompakt)

- Defizite: `.github/SOTA_DEFICIT_REGISTER.md` (D-K0-1…D-K4-9, Status +
  Blocker je Zeile)
- KP-Plan: `docs/SOTA_TRAINING_KONZEPT.md` (§0 Live-Status, §3 P0, §4 P1,
  §5/§6 P2/P3, §8 Sequenzierung, §9 Entscheidungen, §10 Hörproben)
- Roadmap: `docs/TODOS_SOTA_ROADMAP.md` (Lücken-Matrix 2026-10-06 Abschnitt
  „D. Priorisierte Reihenfolge“, WP-0…WP-8)
- Offene Lanes: `docs/reports/current/2026-10-04_offene_massnahmen.md`
- Validierung: `docs/reports/current/2026-10-09_ml_sota_validierung.md`
  (Source of truth der Modell-Aussagen)

---

## Umsetzungsstand (Live — bei jeder Welle aktualisieren)

### 2026-10-27 (Welle C/E — K4-Challenger + D-K3-54)
- **Roadmap neu gefasst (2026-10-27):** `docs/TODOS_SOTA_ROADMAP.md` ist jetzt
  um acht Pfeiler organisiert (Hör-Kapazität, Metrik-Identität, Flag-Queue,
  Provenienz, Fallback-KPI, Training, Daten, SOTA-Definition) — Anhang A
  enthält die unveränderte Bestandsplanung, alle Abschnitts-Anker bleiben
  gültig. `docs/AURIK_10_ROADMAP.md` (Planungshoheit B.H.) unberührt.

- **KP-7/B3 (F11-BWv5, HF-Gain-Gate): METRIK KALIBRIERT — GATE BESTANDEN.**
  Die historische 0,73-Metrik ist identifiziert (Kalibrierung gegen das
  eingefrorene Alt-Modell): HF-Gain = HF-Energie(out)/HF-Energie(Eingang)
  > 3 kHz auf milder HF-Dämpfung (6 dB/oct ab 2 kHz) — Alt reproduziert 0,73
  (gemessen 0,73–0,81 über Seeds). Die zuvor versuchte Recovery-Fraktion war
  die falsche Interpretation; die 11-kHz-Zeile degeneriert (Nenner≈0) und ist
  ausgeschlossen. **Ergebnis (Report `2026-10-27_kp7_hf_gate_v2.json`):**
  HF-Gain Breitband alt=0,81 / neu=51,0 ⇒ Gate ≥ 1,02 **BESTANDEN** (das
  Re-Training beseitigt die F11-HF-Dämpfung). Achtung: +17 dB HF-Boost über
  dem Eingangs-Rolloff ⇒ Artefakt-Risiko; Aktivierung `use_bw_v5=True` erst
  nach Hör-Sign-off (§v10.802), dann als Shellac/Lacquer-HF-Pfad. Bisher:
  Flag bleibt False.
- **KP-8 (Cantus): Pretrain ABGESCHLOSSEN (2026-10-27, best val 3,9013,
  Full-Preset 204,27 M, Seed 42; `checkpoint_best.pt` epoch=25/step=5000
  verifiziert).** `fine_tune` (Phase 2, SingMOS-Learned-Loss) läuft korrekt
  gestartet: Resume ab `checkpoint_best.pt`, Ziel-Epoche 55 (= 30
  Phasen-Epochen; `range(start_epoch, epochs)` ist ein absolutes Ziel —
  der erste Startversuch mit `--epochs=30` war deshalb ein No-Op und der
  Lauf mit `--resume=checkpoint_latest.pt` traf auf einen
  Pretrain-Endzustand. Belege: `train_report_fine_tune.noop_20261027.json`).
  **Phase-2-Pfad CPU-verifiziert:** offizieller Smoke `EXIT=0`
  (`--phase=fine_tune --singmos-loss --smoke`: SingMOS-Learned-Loss +
  Proxy-Kalibrierung + Checkpointing + Report, separater /tmp-Dir —
  Produktionsartefakte unangetastet; Log
  `output/train_cantus_finetune_smoke_20261027.log`).
  GPU nach Runtime-Restart der Umgebung nicht verfügbar ⇒ CPU-Lauf;
  Epoch-Checkpointing macht den Lauf über Runtime-Restarts resumierbar
  (`--resume=models/cantus/checkpoint_latest.pt --epochs=55`).
  **KP-8-Hygiene erledigt:** README-Artefakt-Tabelle korrigiert (alter
  „Seed 20261004, Epoche 10“-Fehlbeleg entfernt, `checkpoint_latest.pt`
  ergänzt, ONNX-Re-Export-Vermerk), `cantus_dit.parity.json` mit
  stale-Notiz versehen (Parität band an den überschriebenen Checkpoint),
  zwei irreführende Platzhalter entfernt (echte Artefakte vorhanden; nur
  `cantus_singmos_proxy.pt.placeholder` bleibt, solange kein Artefakt da ist).
- **GPU-Blocker (extern):** SOTA-M8/M9-Echt-Trainings und Cantus-Phasen 2/3
  bevorzugt GPU; die Trainings-Smokes sind CPU-verifiziert (M8/M9/Lanes).
  Sobald GPU verfügbar: `train_aspade_declipper.py`, `train_tf_gridnet_denoiser.py`
  (beide mit `--smoke` validiert) und ggf. fine_tune auf GPU fortsetzen.

- **SOTA-M12 (KP-17 + KP-18):** beide K4-Challenger implementiert, trainierbar
  und flag-gesperrt integriert (`use_wf4_warp_head` / `use_tp_v2_phase`,
  Default AUS ⇒ bit-identisch). Register D-K4-7/8 → in-arbeit. Offen:
  Echt-Training, Never-worsen/A-B-Gates, Hörordnungs-Sign-off (§v10.802).
- **SOTA-M15 (KP-10):** RT60-Regressor implementiert + flag-gesperrt in
  phase_20/49 (`use_rt60_regressor`, Default AUS); Register D-K4-10 neu
  (in-arbeit). Datenlücke: nur 1 lokale RIR-Simulation — Welle D offen.
- **SOTA-M22 (D-K3-54):** **geschlossen** — Extremwert-korrigierte
  Peak-Signifikanz (k=1,40, 48 Rausch-Realisierungen kalibriert),
  24-Seed-Regression ohne AM-FP, IEC-AM-Never-worsen belegt.

### 2026-10-25

- **Welle A:** drei Hörpakete verifiziert vorhanden (SCNet C4-r2, HR-V1 W-1,
  SGMSE 3-Wege, je mit Manifest). Hörinstanz/Sign-off ausstehend — menschliches
  Gate, nicht autonom schließbar (Hörordnung Ebene 4).
- **KP-2 (Welle B, Musik-MOS-Orakel):** Worker-Lieferung (deepseek-v4-pro;
  erster Versuch scheiterte an ungültiger Modell-Route `mimo-v2.5-pro`)
  geprüft und vom Manager nachgearbeitet:
  - Geliefert: `backend/core/dsp/music_mos_oracle.py` (report-only-Zeuge,
    `applied` konstant False), `scripts/benchmark_music_mos_oracles.py`,
    `tests/unit/test_music_mos_oracle.py` (17 grün), Evidenz-Report
    `2026-10-25_kp2_musik_mos_orakel.md`, FILE_REGISTRY (3 Einträge),
    D-K4-4-Blocker ergänzt.
  - Manager-Nacharbeit: Produktions-Verdrahtung nachgezogen (der Worker hatte
    den Witness NICHT in den Regelkreis gehängt) — `unified_restorer_v3.py`
    §C1-Consumer-Block misst jetzt zusätzlich `measure_and_log_music_mos`
    report-only; Float-Rundung in `_composite_delta` gefixt; nacktes
    `§V6`-Zitat qualifiziert; D-K4-9-Tabellenzeile für das naive Gate-Format
    korrigiert (`sota_deficit_gate.py` wieder 0 Verstöße, 96 Einträge K0–K4).
  - Harness-Lauf (A1-Evidenz, 2 Tracks × 10 s × 7 Klassen × 3 Orakel) läuft
    im Hintergrund → `2026-10-25_kp2_musik_mos_orakel.json`; SingMOS-Richtung
    wird erstmals gemessen.
  - **ABGESCHLOSSEN (Verdikt passed, 2026-10-25):** Manager-Lauf geliefert —
    UTMOS-Richtungsblindheit auf Rauschen reproduziert (Rate 0.0/0.0), MuQ+A1
    Rate 1.0/1.0 (noise) + 0.75 (codec), non-inferior in 5/7 Klassen;
    SingMOS-Pro 0 Messwerte (model_used-Gate, kein Vokal-Material) — Lücke
    §G8 (copilot-instructions.md)-dokumentiert, Rest hängt an KP-12/D-K4-6.
    Report §7 nachgetragen.
- **KP-3 (Welle B, Vokal-Identität):** **ABGESCHLOSSEN (Verdikt passed,
  2026-10-25).** `music_identity_oracle.py` (MuQ-MuLan, report-only) +
  Verdrahtung in `level_1_invariants_guard.py` (Feld
  `singer_identity_cosine_muq`) und phase_65; `use_resemblyzer_music` bleibt
  False, kein neues Flag. Manager-Verifikation: pytest 31 grün, Registry 0/0,
  SOTA-Gate 0, 3 nackte §-Zitate qualifiziert. A/B-Messung: Kosinus-Invarianz
  unter Verarbeitung belegt (ident: dsp 0.835 / muq 0.768); Fremd-Paar-
  Diskrimination inkonklusiv (beide Metriken ≈ 0.97 auf synthetischen
  Fremd-Sängern; Golden Set ohne echte Identitäts-Paare) — Kalibrierung
  wartet auf D-K4-5-Daten; Report §6 nachgetragen.
- **KP-4 (Welle B, Musik-VAD):** **ABGESCHLOSSEN (Verdikt passed, 2026-10-25).**
  `music_vad_oracle.py` (PANNs-Singing → Gate-Offset 0…−15 dB, report-only,
  `applied` konstant False) + Verdrahtung in `phase_18_noise_gate.py`
  (`panns_vad_offset_db` im meta, `gated_audio` unverändert);
  `use_silero_vad_music` bleibt False, W-11-Kommentar ergänzt. Worker starb an
  Provider-Stream-Fehler (44 Tool-Calls) — Rest (Harness, Report, Registry,
  Verifikation) vom Manager ausgeführt. Messung: PANNs-Singing auf 6-s-
  Ausschnitten nicht diskriminativ (max. 0,09 auf echten Vokal-Stems) →
  Offset überall 0, Never-worsen 9/9 per Konstruktion + gemessen; vokaler
  Nutzen unbelegt (§G8 (copilot-instructions.md)). Nächster Schritt:
  plugin-interne Multi-Window-Singing-Score-Logik/AST + Kalibrierung auf
  echten Stems. Report `2026-10-25_kp4_musik_vad.md`.
- **KP-5 (Welle B, Studio-Finisher-Vocoder):** **vorbereitet (2026-10-25),
  Aktivierung W-1-gesperrt.** `use_bigvgan_f3_studio` (Default AUS, W-13
  gesperrt) + Kandidat-Pfad im §1.4-Block von `unified_restorer_v3.py`:
  bei Flag True wird die BigVGAN-Synthese (48-kHz-Resampling, Stumm-Schutz)
  ZUERST versucht, bei Fehler §V6-Warnung + unveränderte Vocos-Kaskade
  (Flag AUS ⇒ bit-identisches Bestandsverhalten). Vertragstests
  `tests/unit/test_kp5_f3_studio_flag.py` (3 grün); `wohlklang_gate` 0.
  Aktivierung erst nach deiner W-1-Hörprobe (Timbre-Konsistenz,
  D-K0-1/D-K2-3) + Hörordnungs-Sign-off.
- **E1/WP-2 (Welle E, Phantom-ML-Stubs):** **ABGESCHLOSSEN (2026-10-25,
  D-K3-3 → geschlossen).** Alle 17 Stub-Module (`adaptive_*.py`,
  `automatic_declipper.py`, `masking_aware_dynamic_eq.py`,
  `psychoacoustic_enhancement.py`) tragen jetzt eine ehrliche
  WP-2-Deklaration im Modul-Kopf: „ML-/TorchScript-Pfad NICHT implementiert —
  DSP-Fallback ist das Verhalten“; §V6-Warnung unverändert. Beleg-Test
  `tests/unit/test_wp2_stub_declaration.py` (fail-closed bei neuem Stub ohne
  Marker); py_compile 17/17; sota_deficit_gate 0; Registry 0/0.
- **E2/WP-1 (Welle E, §III.11-Lücken-Audit Sprach-Signalpfad):**
  **ABGESCHLOSSEN für K2-Hygiene (2026-10-25, D-K2-1 → geschlossen).**
  Lücken geschlossen: §V6-Domänen-Warnung am hifi_gan-Ladepfad
  (`sota_speech_superres.py`) ergänzt; §WP-1-Domänen-Deklaration in
  `aero_plugin.py` (SPRACH-trainiert, Challenger-only, A/B-Verlust belegt);
  vocos-/hifigan-Plugins tragen die Warnungen bereits. Audit-Skript
  `scripts/audit_k2_signalpfad.py` (fail-closed über alle 5 K2-Pfade) +
  Vertragstests `tests/unit/test_k2_signalpfad_audit.py` (10 grün).
  Verbleibend offen: D-K2-2/-3 (Musik-Ersatz = KP-5/W-1), D-K2-4 (nvsr.ts-
  Ladepfad = D-K0-10), D-K2-5 (MUSHRA-Runde).
- **E4/K3 (Welle E, D-K3-20 Tests ohne Wirkung):** **ABGESCHLOSSEN (2026-10-25,
  D-K3-20 → geschlossen).** Die drei verbliebenen Denker-Testdateien
  (`test_defekt_denker.py`, `test_reparatur_denker.py`,
  `test_rekonstruktions_denker.py`) auf echte Verträge umgestellt: alle
  `try/except`-Hüllen und der Phantom-Mock `UnifiedRestorerV3` entfernt,
  Feld-Verträge an die echten Dataclasses angeglichen (dabei aufgedeckt:
  `ReparaturErgebnis`/`RekonstruktionsErgebnis` haben andere Pflichtfelder als
  die alten Tests annahmen); Tests rufen jetzt die reale Funktion aus —
  denker-Verzeichnis 293 grün (davon 63 in den drei Dateien).
  `sota_deficit_gate` 0.
- **E4/K3-8 (Welle E, Layout-Reduktionen Top-Module):** **Teilprüfung
  ABGESCHLOSSEN (2026-10-25).** Die 9 L2-Treffer in den drei Top-Modulen
  sind **False Positives**: `stereo_authenticity_invariant.py` (explizite
  Layout-Branches, (2,N)/(N,2)-invarianz gemessen), `verify_import_export_guards._to_mono`
  (Layout-Branches), `benchmark_stereo.py` (dokumentierte channels-first-
  Konvention via `_load_stereo`). Gepinnt durch
  `tests/unit/test_dk38_layout_reductions_pin.py` (5 grün). Rest-Backlog
  bleibt L2-Bericht (§G8-dokumentiert, kein Hör-Bezug).

- **E6 (Welle E, Doku-Konsistenz):** **ABGESCHLOSSEN (SOTA-M23).** Der
  Roadmap-Abschnitt „SOTA-Maßnahmen 2026-10-25 — Welle A–F" mit den
  TODO-Blöcken `SOTA-M1`…`SOTA-M23` liegt in `docs/TODOS_SOTA_ROADMAP.md`
  (Mappe A=M1–M4, B=M5/M6/M10/M11, C=M7/M14–M17, K4=M8–M13, D=M18,
  E=M19–M23). **§5-Korrektur dokumentiert:** das Register zitierte `SOTA-M11`
  dreifach — aufgelöst zu M11 = Vokal-Identität/SingMOS (F9/F10), KP-7 BW-v5 =
  **M7** (D-K1-5-Blocker im Register nachgezogen). 27 offene Register-Zeilen
  tragen jetzt ihren `SOTA-M*`-Blocker; `sota_deficit_gate` 0 Verstöße,
  normativer Test 27 grün. **Zugleich korrigiert:** D-K3-22 (Budget-Divergenz
  73,6× vs 32×) war im Register `offen`, im Code aber bereits durch den harten
  Deckel `min(_planned_s, _hard_rt × Dauer)` samt Wächter-Test behoben →
  `geschlossen` mit Beleg-Pfad (Zero-Gap).

### Compliance-Lauf 2026-10-25 (Vorgaben/Specs-Erzwingung)

- CI-Äquivalent-Läufe: `compliance_check.py` CI-Gate PASSED (0 Errors,
  1 nicht-blockierende R18-Warnung in `ab_test_manager.py:41` — vorbestehend,
  nicht von dieser Session); `release_must_coverage_check.py` 2/2 (100 %);
  `repo_graph.py --duplicates` 0 Klassen-Duplikate; `sota_deficit_gate` 0;
  `wohlklang_gate` „Vertrag erfüllt“; `file_registry_check` 0/0;
  `id_registry_check` 0 neue Warnungen; `layout_invariant_check` L1 0.
- `spec_drift_check.py`: Baseline war gegenüber HEAD veraltet (4 Dateien
  vorbestehend) + diese Session (FILE_REGISTRY) — Baseline als dokumentierter
  Akzeptanz-Mechanismus neu initialisiert, Verify-Lauf „No spec drift“.
- Volle Pre-Commit-Kette (`pre_commit run --all-files`) läuft als
  Generalprobe; Commit erst nach 0 Fehlschlägen.
- **E3/WP-3 (Welle E, Wohlklang-Ordnung-Forensik):** **ABGESCHLOSSEN (2026-10-25,
  Vocoder-Hypothese BESTÄTIGT).** Harness `scripts/forensik_vocoder_ordnung.py`
  (Seed 42, MUSDB-Stems, Goal-Deltas vor/nach dem §1.4-Vocoder + `would_violate_hearing_order`):
  **36 Hörordnungs-Verstöße in 4/4 Signalen** — tonal_center/brillanz
  (Tier 3/4) verbessert AUF KOSTEN von emotionalitaet/micro_dynamics/
  authentizitaet/natuerlichkeit (Tier 1, z. B. micro_dynamics −0.45 konstant).
  Beleg: Report `2026-10-25_wp3_wohlklang_forensik.md` + JSON, D-K2-3-Blocker
  aktualisiert. Gegenmaßnahme steht bereit (KP-5/W-13, W-1-Hörprobe als
  Aktivierungs-Gate); nach der Umstellung läuft derselbe Harness als
  Never-worsen-Gate.
- **KP-7 (Welle C, BW-v5 Re-Training):** vorbereitet und gestartet.
  - `scripts/train_bw_v5.py`: `--device`/`--seed` (§G5 (copilot-instructions.md))/
    `--a1-weight`/`--out-dir`/`--steps-per-epoch` ergänzt; ONNX-Export
    `dynamo=False` (torch-2.11-Dynamo-Pfad hing am LSTM-Graphen).
  - `scripts/train_banquet_vinyl_finetune.py`:
    `load_a1_masking_loss(sr=TARGET_SR)` parametrisiert (§G9
    (copilot-instructions.md) eine Quelle; BW-v5 nutzt `SR=22050`).
  - Smoke-Test bestanden (CPU, 1 Epoche × 2 Steps: Loss 9.2383, ONNX-Export
    14,4 MB, checker OK).
  - **GPU-Lauf läuft** (Session `wcwzi3yc`): 200 Epochen × 150 Steps,
    Batch 4, lr 1e-4, Seed 42, `--a1-weight 0.5`, `--pregen-samples 2000`
    (erster Lauf `ug9v2k2s` gestoppt: Schritt-Timing gemessen, Flaschenhals =
    Modell/LSTM auf ROCm ~2,25 s/Step; Batch 16 war 7× langsamer — Batch 4
    bleibt). ETA ≈ 19 h.
  - **ABGESCHLOSSEN (2026-10-25, ehrlich):** 200/200 Epochen, Best Loss
    2.176, Export `models/bw_reconstructor/retrain_a1_20261025/` (ONNX 16,4 MB,
    dynamo=False, checker OK); Bestands-Artefakte unangetastet. **A1-HF-Gain-
    Gate-Messung NICHT reproduzierbar** (historische 0.73-Metrik
    undokumentiert; zwei Kandidaten-Metriken ohne klare Überlegenheit —
    Report `2026-10-25_kp7_bwv5_ergebnis.md`). `use_bw_v5` bleibt False
    (§V7 (copilot-instructions.md)); offen: Gate-Definition oder
    Hörstichprobe Alt vs. Neu (deine Entscheidung).
    Ausgabe `models/bw_reconstructor/retrain_a1_20261025/` (aktuelle
    v5-Artefakte unangetastet), Log
    `output/train_bw_v5_a1_20261025.log`. Ende ohne Flag-Flip (§V7
    (copilot-instructions.md)); danach A1-HF-Gain-Gate ≥ 1,02 messen +
    Never-worsen vs. DSP-SBR/FlashSR-Paar + Hörordnungs-Sign-off.
