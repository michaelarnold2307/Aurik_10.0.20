# SOTA-Handover — Weiterarbeiten ohne Vorkenntnis (jede Programmier-KI)

> **Zweck:** Dieses Dokument macht den Zustand von Aurik session-unabhängig
> lesbar. Jede KI (Kun, Claude Code, Copilot, Codex, Cursor …) soll nach dem
> Lesen **sofort** wissen, was zu tun ist — auch nach Abbruch, Reboot oder
> Wochen Pause. Stand: 2026-11-08.

## 0. Mission

Aurik 10 = autonomes Audio-Restaurierungsprogramm für Musik mit **weltbestem
Wohlklang für das menschliche Gehör**. Normative Kette: `AGENTS.md` §1
(Pfeil: `AGENTS.md` liest zuerst, dann `.github/copilot-instructions.md`,
`.github/VERBOTEN.md`, Hörordnung, Domain-Regeln, Specs).

## 1. Sofort-Start (3 Minuten)

1. `AGENTS.md` lesen (Repo-Regeln — Write-Gate, Zitierdisziplin §5,
   Determinismus §G5, deutsche Logs).
2. **Dieses Dokument** + `docs/reports/current/2026-10-28_vollstaendigkeit_tiefenanalyse.md`
   (das Lückenbild) + `docs/TODOS_SOTA_ROADMAP.md` (Bestand).
3. Zustand prüfen (siehe §3) und die nächste offene Lücke aus §2 schließen.

## 2. Rest-Lücken (priorisiert) — WAS zu tun ist

| # | Lücke | Typ | Wo dokumentiert |
| --- | --- | --- | --- |
| 1 | ~~WP-2-Hygiene~~ **ERLEDIGT 2026-10-28:** Stubs (D-K3-3), Phantome (D-K3-5), D-K2-4 Namens-Drift und Domänen-Status sind register-seitig geschlossen; die P2-INFO-Findings des Gates sind dokumentierter Akzeptanz-Bestand | — | Register D-K3-3/D-K3-5/D-K2-4/D-K0-10 |
| 2 | Referenzfreie Qualitätsmetrik — **ERLEDIGT 2026-10-28**: `musik_mos` (UTMOS-Ersatz, `plugins/musik_mos_plugin.py`) + `defektlast` beide **zertifiziert** (Spearman 0,95); UTMOS/PQS blind gesperrt; offen nur noch absolute MOS-Kalibrierung (KP-12/MUSHRA-Proxy, menschlich) | — | Zertifikat-JSON + `plugins/musik_mos_plugin.py` |
| 3 | ~~P1-7 (m1b-Stufe-2 intern) + P1-10 (Defekt-Chips)~~ **ERLEDIGT 2026-11-03:** Spec-Verifikation + 15 Verkabelungs-Invarianten, Roadmap-Status geschlossen (Commit `e12a49c7`) | — | Roadmap TODO-P1-7/P1-10 + Report `2026-11-03_p1_7_p1_10_status_audit.md` |
| 4 | Laufzeit 53×→32× RT (F-Reihe optimieren) | autonom (GPU-gebunden) | **IN ARBEIT 2026-11-03/04:** GPU-D3-Kette läuft (DiffWave→BigVGAN→FlashSR, Manifeste `docs/reports/current/train_job_d3_*.json`); §PERF-R18-Äquivalenz-Tests + Budget-Gate **21/21 grün** (2026-11-04, zuvor „nicht ausgeführt"), Ketten-Bugfixes M8/M9/c2-Cantus per Smoke verifiziert (Report `2026-11-04_sota_sitzung_ear_vae_messung_ketten_checks.md`); offen: nur noch Verifikation nach D3-Lauf; **RT-Messung 2026-11-05 ABGESCHLOSSEN: Median 32,19× RT (3 Läufe, Wall 7248,7–7259,2 s / 225,33 s), Baseline 53,9× — Lücke geschlossen** (Report `2026-10-28_f_reihe_laufzeit_optimierung.md` §9, Log `rt_messung_2026-11-04.log`; Hinweis: der geloggte rt_factor ist auf 32,0 gekappt, Spec §9.5; `RT_REALITY_MEASURED` = 32,2); ursprüngliches **RT-Mess-Protokoll 2026-11-04** (Report `2026-10-28_f_reihe_laufzeit_optimierung.md` §6: `scripts/_aurik_run_excellence.py` auf Referenz-Song „Testkünstlerin…“ (224 s), 3 Läufe/Training-frei, Baseline 53,9× RT — Bericht `2026-11-04_sitzung_w2_w3_b3_messung.md` §5); **Scanner-Performance 2026-11-07: PERF-R19/R19b/R20 gelandet** (D-K3-38, **GESCHLOSSEN**) — Band-Dekadierung FM+AM (62 → 14,9 s/min warm, 4,2×) und Salience-Kette **bit-identisch** um 18,5 % schneller (6,50 → 5,30 s/min; Scan-Hash unverändert `c599791d…`, Golden-Gates 3/3 + 1/1); **Norm evidenzbasiert nachkalibriert: DefectScanner ≤ 16 s/min** (Spec 07 §9 + copilot-instructions.md synchron, Sync-Test pinnt 16,0; gemessen 14,9 s/min warm ⇒ ERFÜLLT). Das 4-s-Fast-Ziel bleibt dokumentiert hinter der Hör-Freigabe des Batch-Pfads (§2.46h) — Hör-Entscheidung, kein deterministisches Defizit mehr (Report `2026-11-07_perf_r20_salience_caches.md`); **D3-Taktung 2026-11-09 (D-K3-87):** BigVGAN 74 min/Epoche und FlashSR 80 min/Epoche sind mit 2×6 h strukturell unerreichbar (Epoche 8 bzw. 10 von 30) — Kette 2 fährt jetzt 4×12 h mit `--max-seconds`-Stopp an der Epochen-Grenze, D3-3 vor D3-2, RNG-/Historien-Fortsetzung; 30 Epochen bleiben Ziel (Report `2026-11-09_d3_taktung_zeitbudget.md`) |
| 5 | ~~F3-BigVGAN-Rollout~~ **ENTSCHEIDEN 2026-10-30 (K0-C NO-GO, Hör-Instanz)** + ~~EAR-VAE v1↔v2-Vergleichsmessung~~ **GEMESSEN 2026-11-04:** HF-Gewichte `earlab/EAR_VAE2` (apache-2.0, `c87215f8…`) beschafft; Blind-Messung seed 42: **never_worsen ✓, 16/17 Trials v2-vorteilhaft, Sign-Test p = 0,000275 signifikant, erfuellt_objektiv ✓** (Messdatei `2026-11-03_ear_vae_v1_v2_vergleich.json`, Δ musik_mos +0,10 / Δ defektlast −0,12); **Rollout-Entscheidung v1↔v2: MENSCHLICH ENTSCIEDEN 2026-11-05 (NO-GO v2 — v1 bleibt; Hör-Instanz Block C, Sign-off Commit `7d2e3441`, `docs/reports/current/2026-11-05_hoer_signoff.md`; Wunsch-Richtung künftiger Iterationen: heller UND weniger Rauschen; D-K0-9/SOTA-M2, §V7 (copilot-instructions.md))** | — | LÜCKEN-MATRIX D.1b/c + K0-Sign-off-Paket + Report `2026-11-04_sota_sitzung_ear_vae_messung_ketten_checks.md` |
| 6 | Wohlklang-Stufen W-1…W-13 aktivieren (0/13 vollständig) | **B3-Verfahren** | **Stand 2026-11-04:** Alle B3-Beschlüsse W-1/W-4/W-5/W-6/W-7 formal **ABGELEHNT** (nicht signifikant, p ≥ 0,06; W-1 zusätzlich never_worsen verfehlt; W-1 hat widersprüchliche Objektiv-Belege — HF-Gain-Beschluss BESTÄTIGT vs. musik_mos-Beschluss ABGELEHNT ⇒ die Hör-Instanz entscheidet, Hörordnung §7); **Objektiv-Befund entscheidungsreif 2026-11-05** (`docs/reports/current/2026-11-05_k4_wohlklang_stufen_objektivierung.md`): W-2/W-3-Nachmessung ausgewertet (W-2: never_worsen ✓, p=0,6875 ⇒ objektiv ABGELEHNT; W-3 damals noch ohne Messung, gesperrt — Nachmessung am selben Tag abgeschlossen: 0/6 ⇒ ebenfalls ABGELEHNT (Details unten „B3-Nachmessung W-2/W-3 ABGESCHLOSSEN"), W-6/W-7 Fallbacks 11164/18220 → 364/2236 nach Router-Fix; B1-Zähler präzise (Commit `ef3795b7`); Hör-Sessions W-1/W-4/W-5 vorbereitet = **menschlich offen** (W-1 = einzige Stufe mit never_worsen=False + widersprüchlichen Belegen ⇒ §7-Hör-Entscheidung); W-2/W-3 gesperrt (Budget/Pfad) — **B3-Nachmessung W-2/W-3 ABGESCHLOSSEN 2026-11-05: beide ABGELEHNT** (W-2 `use_scnet_music`: 2/6 Vorteile, p = 0,6875; W-3 `use_sgmse_musik`: 0/6, Δ ≈ 0 — Schalter-Wirkung im Messpfad nicht nachweisbar, konsistent mit „Pfad gesperrt"; never_worsen ✓, B1 sauber mit 0 Rückfällen; `flag_freigabe.py` fail-closed: „Vorteil nicht signifikant"; kein Flag-Flip, Vertragssoll „AUS" bleibt; Belege `b3_stufenmessung_W-2/W-3.json` + `b3_objektivbeschluss_W-2/W-3.json`, Lauf `2026-11-05_b3_w2w3_nachmessung.log`, Report `2026-11-05_sitzung_w2w3_b3_nachmessung.md`; Mess-Lauf auf Korpus-Clips statt der Vertrags-Referenz-Songs — für künftigen Weg-B-Beschluss offen; Harness-Erweiterung Commit `d5483a43`; verbleibend: C4-Hörpaare + RT-Budget-Nachweis (W-2)); **Pfad-Klärung (W-3) ERLEDIGT 2026-11-06** — Ursache gefunden + behoben: Dauer-Achsen-Defekt in `phase_49_advanced_dereverb.py` (`shape[-1]` auf dem channels-last-Array ⇒ jedes Stereo-Fenster galt als „< 1 s“, SGMSE+-Tier-0 unerreichbar) — Δ ≈ 0 ist ein **No-Op-Artefakt der Messung**, der W-3-Beschluss 2026-11-05 damit als Qualitätsurteil über SGMSE+ **ungültig**; Fix + 3 Regressionstests (Rot→Grün), Report `2026-11-06_w3_pfadklaerung.md`; Neumessung nur über §4 mit Vertrags-Referenz-Songs, Flag bleibt AUS); „4-Modelle"-B3-Lauf (Material/EAR/Defekt/Loudness) offen **begründet** — Treiber ad-hoc, nicht rekonstruierbar (Report `2026-11-04_sitzung_transient_messlauf_b3_status.md` §6–7) |
| 7 | Modelle: WF-V4, TP-V2, BEATs-Head, MERT-Head, Gender-Head | teils Daten-blockiert | Tiefenanalyse E2 — **Stand 2026-11-06 (Trainingswelle):** Echt-Training vorbereitet für WF-V4 (D-K4-7, `scripts/prepare_wf4_warp_pairs.py` — MUSDB18-HQ-Mixture + bekannte Wow/Flutter-Warp-Kurven, GT warp_log2, Seed 42 (§G5), Manifest sha256) und TP-V2 (D-K4-8, `scripts/prepare_tp2_transient_pairs.py` — MUSDB18-HQ-Drums/other/bass-Stems, transiente-reiche Segmente); zusätzlich vorbereitet: F9 VoiceEncoder (D-K4-5), F10 SingMOS-Kalibrierung (D-K4-6), D-K2-3 vocos_48khz-Musik-Finetune. Alle als Jobs in `scripts/gpu_kette4_modellkoepfe.sh` (V1/V2 Datensynthese + R1–R6, pgrep-Guard, kein Flag-Flip, KEIN paralleles Training gestartet). **Köpfe (E2):** Gender-Head daten-blockiert (es fehlt `data/gender_labels.csv` — Gender-/Sänger-Labels auf Musik-Vocals, Lizenz/Label-Quelle offen, GTZAN/MagnaTagATune nur Genre+Valence/Arousal, kein Gender); BEATs-Head/MERT-Head daten-blockiert (527 Klassen AudioSet-Labels + Score-Kalibrierung auf HPI/AF fehlen); „Finetune der Köpfe“ bleibt Trainingspflicht (F13) — kein Trainer ohne Labels. Report `docs/reports/current/2026-11-06_trainingswelle_modelle.md` — **Fortsetzung 2026-11-06 (Manager-Ausführung):** drei **Phantom-Verdrahtungen** gefunden und behoben, die R1/R2 der Kette 4 am Musik-Training gehindert hätten (§III.11): (1) `train_wf4_warp_head.py` hatte **kein `--real-dir`** (nur synthetische Harmonische), (2) der R1-Guard prüfte `data/wf4_warp_pairs/degraded` statt `audio/` ⇒ R1 **immer** übersprungen, (3) `train_tp_v2_phase.py` ließ den Default `--synthetic 16` ein gesetztes `--real-dir` still überstimmen. Fix: **F0-Beobachtung** im Datensatz (`x_f0 = log2(f0_deg/f0_clean)` via Produktions-Tracker `pyin_fast`, keine GT-Zirkularität) + fail-closed-Lader (§V6, kein stiller Synthetik-Rückfall) + Vorrang real vor synthetisch + Kette-4-Verdrahtung; **Determinismus-Befund**: WAV-Container byte-instabil (libsndfile-`PEAK`-Chunk, 1 Byte bei Offset 61) ⇒ verbindlicher `sha256_pcm` statt `sha256_wav`. Daten erzeugt: **300 WF-V4-Paare** + **200 TP-V2-Segmente** (Seed 42, Manifest). Tests: 13 grün (`tests/unit/test_wf4_warp_real_pairs.py`, `tests/unit/test_train_tp_v2_real_dir_vorrang.py`). **R1/R2-Training weiterhin offen** (GPU belegt); **F9/F10/D-K2-3-Trainer implementiert 2026-11-06** (`scripts/train_voiceencoder_finetune.py`, `scripts/calibrate_singmos_music.py`, `scripts/train_vocos_48khz_finetune.py` — CLI exakt R3/R4/R5, Vertragstests 21 grün, vocos==0.1.0 im venv_aurik installiert, KEIN Flag-Flip) — R3/R4/R5 aktivieren sich im freien GPU-Fenster. Report `docs/reports/current/2026-11-06_wf4_echt_pfad_verdrahtung.md` — **2026-11-08 (D-K3-81): R3/R4/R5 waren nicht *startbar*.** Drei Trainer-Defekte behoben und Smoke-verifiziert: F9 bildete strukturell keine Gradienten (`embed_utterance` läuft in `no_grad`) **und** hing bei einem Val-Track in einer `while`-Endlosschleife (7 min, 100 % CPU); F10 gab 16 kHz an einen 48-kHz-Vertrag (AssertionError) und maß ganze Tracks statt eines 20-s-Fensters unter der 30-s-Kappung (entarteter Fit a = 2,4e6); vocos lud ein lokales Checkpoint über `hf_hub_download` (Repo-ID-Fehler). **Substanz-Befund bleibt offen:** die volle F10-Kalibrierung (320 Paare) zeigt SingMOS-Pro am **Ceiling** (clean 4,959/5, Std 0,023) und ohne Sensitivität für Bandlimit/Codec — **144/320** Paare verletzen die Grund-Ordnung, Codec (4,995) liegt **über** clean ⇒ die affine Kalibrierung trägt nicht; das ist eine **Metrik**-Entscheidung (`musik_mos`/`defektlast`, zertifiziert) und **kein autonomer Code-Schritt**. D-K4-6 bleibt offen mit präziser Ursache. Report `docs/reports/current/2026-11-08_kette4_trainer_defekte.md`. **Echt-Läufe derselben Sitzung:** R1 WF4 erfolgreich (50/50, 316 s), **R3 F9 erfolgreich (30/30 Epochen, 1555,8 s, val_acc 0,672 gegen Basis 0,422)**, **R4 F10 erfolgreich (49,8 s, Produktionsartefakt zahlen-gleich zum `/tmp`-Lauf ⇒ §G5)**, **R5 vocos erfolgreich (30/30, 1768,8 s) ⇒ R1–R5 echt fertig**; nur R6 Gender-Head bleibt daten-blockiert. **2026-11-09 (D-K3-85):** WF-V4 und TP-V2 waren bis zum Fix **nicht erreichbar** (WF4 lud nur ein nie erzeugtes `.onnx`, TP-V2 zeigte auf `backend/models/…` und lief mit Seed-Gewichten) ⇒ die B3-Ablehnungen M12-WF4/M12-TPV2 vom 2026-11-08 sind **Mess-Artefakte**, keine Qualitätsurteile; beide Pfade laden jetzt ihre trainierten Checkpoints, Modell-Zeuge im Pfad-Gate, Flips bleiben AUS, Neumessung im trainingsfreien Fenster offen. |
| 8 | Hör-/Lizenz-Entscheidungen (§9/5–7, AERO, MUSHRA n≥30) | **menschlich** | Katalog A2 |
| 9 | ~~P1-3-Restdeckung (Audibility-Guards)~~ **ERLEDIGT 2026-11-03:** HNR/Mikrodynamik/Vibrato/Hallucination/Noise-Texture mit Maskierungs-Abdeckung (§G10), 19 Phasen verdrahtet (Commit `2c5e5522`) | — | Report `2026-11-03_p1_3_audibility_restdeckung.md` |
| 10 | ~~D-K1-3 Material-CRNN-Verdrahtung (95,3 % statt 40,2 %)~~ **ERLEDIGT 2026-11-03:** im Material-Entscheidungspfad hinter `use_material_crnn` (Default AUS — Flag-Flip offen, nur §4-Verfahren) (Commit `28f2947d`) | — | Report `2026-11-03_d_k1_3_material_crnn_wiring.md` |
| 11 | **Ketten-Verdrahtung (Trainingslücke, kein Modellproblem)** — **BEHOBEN 2026-11-06, GPU-Neulauf offen:** Drei Kette-1-Jobs liefen ins Leere, obwohl die Artefakte lokal liegen. (a) **M9/SOTA-M9** starb nach 5 s am 24-GiB-OOM (`--batch-size 4` ⇒ 4·513 Inter-Frame-LSTM-Sequenzen, 4,42 GiB angefragt) **und** trainierte auf `--synthetic 4000` (Sinus+Rauschen = K1-Domänen-Defizit §III.11) statt auf den vorhandenen `data/tfgrid_pairs` (25 MUSDB-Paare) — ein erfolgreicher synthetischer Lauf hätte den echten Kette-3-Lauf über Job-Lock/Manifest blockiert. Fix: echte Paare + `--batch-size 1 --grad-accum 4` (neues Trainer-Flag, Äquivalenz Batch 2 ⇔ Akkumulation 2 getestet; Kette 3 trägt dieselben Flags). (b) **Cantus/SOTA-M14** rief R3–R5 ohne `--data` auf (Exit 2 „--data fehlt“, 2 s), obwohl `data/cantus_pairs/manifest.jsonl` 3600 Paare hat — jetzt `--data` + fail-closed-Guard; `domain_adapt` nur mit echten `*_degraded.wav`-Paaren (bleibt datenblockiert). Tests: `tests/unit/test_gpu_kette1_wiring.py` (7 — inkl. statischer CLI-Prüfung ALLER vier Ketten gegen die `add_argument`-Flags der Trainer, 56 Aufrufe) + 2 Trainer-Tests. Offen: GPU-Neulauf, Never-worsen-Gate/Hörabnahme (M9), M9-Datenmenge (25 Paare, D-K2-5-Muster). **Kein Flag-Flip.** **Wiederaufnahme-Fix 2026-11-07:** `train_job_runner.py` überspringt Jobs mit `ergebnis: erfolgreich` bei identischem Befehl/Phase/Seed (Manifest bleibt unverändert; `--force` erzwingt Neulauf) — der nächste Kette-1-Neulauf überspringt M8/RT60 und startet direkt M9/Cantus. | autonom (GPU-gebunden) | Report `2026-11-06_ketten_verdrahtung_m9_m14.md` + `2026-11-07_train_job_runner_idempotent.md`. **Resume-Anker-Defekt 2026-11-08 (Kette 2, D3-3 FlashSR) BEHOBEN:** `train_flashsr_f4.py` schrieb nur `checkpoint_epoch<N>.pt`/`best.pt`, nie den Repo-Standardanker `<out-dir>/checkpoint_latest.pt` — der Kette-2-Resume-Befehl zeigte genau darauf und starb 61 s nach dem 6-h-Timeout in `FileNotFoundError`; zusätzlich resumte `--cmd` (Versuch 1) nie, sodass jeder Kettenstart bei Epoche 0 begann und die 5,37 h aus Versuch 1 wegwarf. Fix: Anker je Epoche, `--resume` prüft die Datei fail-closed (Exit 2 statt Traceback), **beide** Kette-2-Befehle datei-geschützt (Aspade-Muster), `val_loss`-Broadcasting bereinigt; Anker aus `checkpoint_epoch3.pt` restauriert (Epoche 3, val 0,3726). Tests 10/10 (`test_gpu_kette1_wiring.py`, 3 neue — davon einer generisch: jeder `--resume`-Pfad der vier Ketten muss vom Trainer erzeugt werden). **Kein Training gestartet** (Kette 3 belegte die GPU), kein Flag-Flip. Report `2026-11-08_flashsr_resume_anker.md`. **Cantus-No-Op 2026-11-08 (D-K3-84, BEHOBEN):** `c2_cantus_finetune` trainierte **null Epochen** (Pretrain-`epoch=50` gegen `--epochs=30` ⇒ leeres `range`) und stand trotzdem auf „erfolgreich“; Fix: phasen-lokale Resume-Größen + „0 Epochen ⇒ Exit 3“, No-Op-Manifest archiviert — echter Neulauf beim nächsten Kette-1-Start (Runner hätte den Job sonst übersprungen). |

**Regel:** Punkte 1–5 ohne Rückfrage erledigen. Punkt 6 NUR über das
B3-Verfahren (§4). Punkt 8 nie eigenmächtig — abbrechen und begründet
berichten.

## 2b. Muster zur Qualitätssteigerung (Stand 2026-11-07)

Muster-Analyse (Manager-Sitzung 2026-11-07): Das Regelwerk ist ein starker
Never-worsen-**Bremser**, aber ohne **Beschleuniger** (0/13 W-Stufen aktiv;
Einzel-Flips nie signifikant; Kombinationen nie gemessen). Sechs Hebel,
Priorität C→B→A→D→E/F:

- **C — Pfad-Reachability-Gate (ERLEDIGT, Commit `df512a5e`):**
  `scripts/b3_pfad_wirksamkeit.py` + Integration in `b3_stufenmessung.py` +
  `flag_freigabe.py`. Keine Ablehnung mehr auf Mess-Artefakten (W-3-Lehre).
- **B — Kompositions-Suche (ERLEDIGT, Commit `5a94c0a6`):**
  `scripts/b3_komposition_suche.py` — Lazy-Forward-Selection über
  W-2/W-3/W-8/W-9/W-13 mit never-worsen-Schranke; Ergebnis = KANDIDAT für
  EINE Hör-Session. **AUSFÜHREN im trainingsfreien Fenster:**
  `nice -n 15 python3 -B scripts/b3_komposition_suche.py --clips 3 --windows 2`
- **A — Audibility-Budget (VERDRAHTET 2026-11-07, Flag AUS; Wirkungsmessung offen):**
  klangwirksam ⇒ flag-gesperrt (Default AUS, bit-identisch; 30 Vertragstests).
  `backend/core/dsp/audibility_budget.py` definiert das Budget als
  Hörbarkeits-Überschuss `max(0, L_defekt − (T_signal − Marge))`; der
  Johnston-Kern ist bit-identisch zu `masking_model.compute_masking_threshold_db`
  (Diff 0,0). Pilot-Verdrahtung in **zwei** Phasen: `phase_03_denoise.py::
  _denoise_mono_mrsa` (rauschbewusst je Zone; Masker = spektrale Subtraktion
  `max(|Y|²−|N|²,0)`, damit rauschdominierte Bänder nicht fälschlich gekappt
  werden) und `phase_43_ml_deesser.py::_deess_channel` (Sibilantenband aus dem
  Masker ausgenommen). Phase 19 bleibt bewusst außen vor — sie hat bereits
  `apply_psychoacoustic_masking_clamp` + PSY-A1. Nächster Schritt im
  trainingsfreien Fenster: `nice -n 15 python3 -B scripts/b3_stufenmessung.py
  --stages A-1 --clips 3`, RT-Kosten je Zone, blindes A/B; Aktivierung NUR
  über `scripts/flag_freigabe.py --objektivbeschluss` nach Hör-Sign-off.
  Beleg: `docs/reports/current/2026-11-07_audibility_budget.md`, D-K3-71.
- **D — MUSHRA n≥30 (menschlicher Multiplikator):** Protokoll §10 + Builder
  fertig; entriegelt D-K4-4 (F8), D-K4-11 (F15), Proxy-Kalibrierung.
- **E — RT-Budget-Ledger (ERLEDIGT 2026-11-08, D-K3-75/D-K3-76):**
  `PerformanceGuard.get_budget_allocation()` (§PERF-R21) alloziert das
  32×-Budget je Phase — Soll aus der normativen `PHASE_PRIORITIES`-Ordnung,
  Ist aus den gemessenen `phase_performances`, plus `reserve_s`,
  `overrun_total_s`, `budget_exceeded`, `top_consumers`; verdrahtet als
  `rt_budget_ledger` in den UV3-Metadaten. **Reine Buchführung** (kein
  Skip-Pfad liest es, Quelltext-Pin). `get_phase_budget_seconds` delegiert
  dorthin (vorher: tote Methode mit flacher Division, NULL Aufrufer).
  **Dabei gefunden und behoben (D-K3-76):** `wow_flutter` und `decracklé`
  (beide CRITICAL 9) wurden von der Tabelle NIE getroffen — die Modulnamen
  heißen `phase_12_wow_flutter_fix`/`phase_09_crackle_removal`, die
  Normalisierung erzeugt `wow_flutter_fix`/`crackle_removal` ⇒ Default 5
  (MEDIUM) ⇒ die teuerste Reparaturphase war unter Budgetdruck skippbar.
  Alias-Auflösung `_priority_key()` in allen drei Prioritäts-Pfaden.
  Beleg: `docs/reports/current/2026-11-08_rt_budget_ledger.md`.
  **Offen (bewusst, kein autonomer Schritt):** 60 der 69 Phasen-Module treffen
  weiterhin keinen Tabelleneintrag (Default 5) — Neubewertung braucht
  RT-Profil + Hör-Impact je Phase.
- **F — Forensik-Monitor (ERLEDIGT 2026-11-08, D-K3-83):** WP-3-Forensik +
  musik_mos/defektlast als Dauer-Wohlklang-Zeuge je Lauf (nur gerichtete Fälle
  zur Hör-Instanz). Befund: `backend/` hatte **null** Aufrufstellen der beiden
  zertifizierten Metriken. Neu: `backend/core/dsp/wohlklang_zeuge.py` misst
  Eingang/Ausgang je Lauf, schreibt `metadata["wohlklang_zeuge"]` in allen drei
  Produktionspfaden (Einzelpass/Chunked/Options) und markiert
  `gerichteter_fall` → `hoer_instanz=vorzulegen`; **REPORT-ONLY** (kein
  Sample-Eingriff, kein Rollback, keine Export-Blockade, kein Flag; Quell-Pin:
  drei Schreibstellen, keine Lesestelle), deterministisch (§G5), §V6-fail-safe.
  Zusätzlich EINE Wahrheit: `defektlast`-Definition und JND 0,05 liegen jetzt
  im Modul (Zertifikats-Skript, B3-Stufenmessung und Kompositions-Suche
  greifen darauf zu). 13 Vertragstests. Offen (nur GPU-Fenster):
  End-to-End-Lauf + RT-Delta-Messung. Beleg:
  `docs/reports/current/2026-11-08_wohlklang_zeuge_hebel_f.md`.

## 3. Zustand & Wiederherstellung nach Abbruch/Reboot

- **GPU-Trainingsketten** (dauerhaft im Repo, jederzeit neu startbar):
  - Kette 1 (Welle C + M8/M9): `nohup bash scripts/gpu_kette1_welle_c.sh > /tmp/k1.log 2>&1 &`
  - Kette 2 (Matrix D.3, startet nach Kette 1): `nohup bash scripts/gpu_kette2_matrix_d3.sh > /tmp/k2.log 2>&1 &`
  - Kette 3 (K4-Rest: A-SPADE + M9 auf echten Paaren):
    `nohup bash scripts/gpu_kette3_k4_rest.sh > /tmp/k3.log 2>&1 &`
  - Kette 4 (Modell-Köpfe/Trainingsdaten, bricht bei laufendem
    `train_job_runner` fail-closed ab):
    `nohup bash scripts/gpu_kette4_modellkoepfe.sh > /tmp/k4.log 2>&1 &`
  - **Reihenfolge bei Neustart:** 1 → 2 → 3 → 4. Kette 2 wartet selbst auf
    Kette 1; Kette 3/4 nur starten, wenn kein Training mehr läuft. Danach
    `pgrep -af gpu_kette` + `pgrep -af train_job_runner` prüfen.
  - **Doppelläuft-Schutz + idempotente Wiederaufnahme:** `train_job_runner.py`
    hält Job-Locks (`models/*/.train_job.lock`) und schreibt Manifeste
    (`docs/reports/current/train_job_*.json`) inkl. Auto-Resume-Befehl.
    Ein Manifest mit `ergebnis: erfolgreich` und identischem Befehl/Phase/Seed
    wird beim Ketten-Neustart **übersprungen** (kein GPU-Verbrauch, kein
    Überschreiben des Erfolgs-Manifests); `--force` erzwingt einen Neulauf
    (z. B. nach geänderten Trainingsdaten). Unlesbare Manifeste blockieren
    nicht, sie werden mit `logger.warning()` neu aufgebaut (§V6).
    **Dritter Status seit 2026-11-09 (D-K3-87): `ergebnis: zeitbudget`.** Der
    Trainer stoppt mit `--max-seconds` SELBST an der Epochen-Grenze (Exit 75)
    statt per SIGTERM mitten in der Epoche gekillt zu werden; das ist **kein
    Fehler**, sondern ein gesunder, fortsetzbarer Lauf (nur `erfolgreich` wird
    übersprungen). Vorher stand ein solcher Lauf als „fehlgeschlagen“ im
    Manifest — die D3-Finetunes sahen dadurch „gescheitert“ aus, obwohl sie
    nur nicht fertig waren.
  - Status prüfen: Manifeste lesen (`ergebnis`, `versuche`) + `rocm-smi`.
    Laufende Kette erkennen: `pgrep -f gpu_kette`.
  - **D3-Verifikation — BESTANDEN 2026-11-07 (D3-1 DiffWave-Vokal):**
    Kanonischer Beleg ist der Trainer-Report
    `docs/reports/current/2026-11-07_diffwave_vocal_finetune.json`:
    `never_worsen_per_gap: true`, `best_val_mean_sdr_db: 3.23` vs.
    `baseline_zero_shot_mean_sdr_db: −3.01` (Seed 42, best_epoch 19); dazu
    Kette-2-Log `Ergebnis: erfolgreich` + `JOB_EXIT=0`. **Manifest-Hinweis:**
    `train_job_d3_diffwave_vokal.json` wurde während der Parallel-Session durch
    einen externen `git reset` auf den Stand von Versuch 1 zurückgesetzt; das
    Trainer-interne S1-Gate ist die Messung, der Report bleibt der Beleg.
    Ein separates `verify_d3_quality_gate.sh` existiert NICHT. Ein Deploy eines
    D3-Artefakts ist klangwirksam und KEIN automatischer Schritt: nur mit
    Never-worsen-A/B gegen den deployten Checkpoint (D3-Berichte
    `2026-10-28_f_reihe_laufzeit_optimierung.md` §9, §4-Verfahren).
  - GPU-Umgebung: `/home/michael/.local/share/aurik/venv_rocm72/bin/python`
    (torch 2.11+rocm7.2, Radeon 7900 XTX). System-python ist CPU-only.
- **Dauer-Automation:** Kun-Scheduled-Task „SOTA-Lückenschluss" (ID
  `73cc09ac-1ce4-48ae-851c-5022f637f534`, alle 6 h) arbeitet §2 ab und führt
  die Wiederherstellung aus. Fällt Kun aus, übernimmt die nächste KI manuell
  nach diesem Dokument.
- **Mess-/Trainingsstand:** alles liegt in `docs/reports/current/` (mit Quelle
  je Zeile) und `data/`-Manifesten (`vinyl_noise_bank/`, `rir_real/`) —
  Reproduzierbarkeit über Seeds + sha256.
- **Worker-Worktrees (Manager-Erfahrung 2026-11-05, ADE-Dispatches):** Frisch
  erzeugte Task-Worktrees (`/home/michael/.kun/worktrees/tasks/…/tws_*`) können
  auf einem **veralteten Stand** hängen (beobachtet: `65c6ef4d` vom 2026-10-08
  statt aktuellem main). Symptom: der Worker meldet fehlende Quellen
  (Register/Reports/Harnesses) und blockert korrekt. **Abhilfe (Manager):**
  `git -C <tws> stash push -u && git -C <tws> merge --ff-only main &&
  git -C <tws> stash pop` (bei unverändertem Worktree genügt `merge --ff-only`),
  danach den Worker mit `worker_send` fortsetzen. **Hook-/Gate-Umgebung in
  Worktrees:** `.venv_aurik`-Symlink auf das Haupt-Checkout, `chmod +x
  run_tests_chunked_safe.sh`, und unverfolgte Modell-/Mess-Pfade
  (`models/sgmse_plus/…`, `models/harmonic_inpainting`, `models/vocos_48khz`,
  `output_audio/mushra/…`, `output/scnet_ab_…`) als Symlinks nachziehen — sonst
  scheitern Wohlklang-/SOTA-Gates mit „zitierte Pfade fehlen" (Umgebungs-Artefakt,
  kein Code-Defekt).

## 4. Aktivierungs-Verfahren (Wohlklang-Stufen/Flags) — WIE

Nur zwei legale Wege (fail-closed, `scripts/flag_freigabe.py`):

    # Weg A: Hör-Abnahme (menschlich)
    python3 -B scripts/flag_freigabe.py --flag use_x --verdikt <fit.json> --signoff <signoff.md> --commit
    # Weg B: Formeller Objektivmetrik-Beschluss (Wohlklang-Vertrag B3)
    python3 -B scripts/flag_freigabe.py --flag use_x --verdikt <messung.json> --objektivbeschluss <beschluss.json> --commit

Beschluss-JSON (Schema `aurik.objektivbeschluss/1`) MUSS enthalten:
`flag`, `referenz_songs` ≥ 3, `never_worsen: true`, `signifikant: true`,
`defekt_fallbacks: 0`, `blind_ausgewertet: true`, `messung` (existierende
Messdatei), `zielmetrik`. **Hör-Vorbehalt:** spätere Hör-Abnahme gilt
vorrangig (Widerspruch ⇒ Rollback). Vertrag: `.github/WOHLKLANG_CLAIMS.md`.

## 5. Harte Regeln (Verstöße = Rückabwicklung)

- Flag-Flip NIE ohne §4 (§V7 (copilot-instructions.md): kein stiller Flip).
- Keine Hör-/Lizenz-Entscheidungen eigenmächtig (§9-Liste, Great-78/Material).
- Jede Änderung: Tests + alle Pre-Commit-Gates grün + Conventional-Commit;
  „Regeländerung ⇒ Skript nachziehen" (AGENTS.md §2).
- Laufende Ketten/Hintergrund-Läufe nicht stören; Trainings sequenziell
  (Job-Locks); Host-Schutz (nice, Thread-Caps, RAM-Guard) respektieren.
- Keine Funktion ohne adäquaten Ersatz entfernen (User-Vorgabe 2026-10-28).

## 6. Verifikations-Befehle

    python3 scripts/wohlklang_gate.py          # Vertrag + Phantome (fail-closed)
    python3 -m pytest tests/unit -q            # Unit-Smoke
    python3 .venv_aurik/bin/python scripts/model_inventory.py --fail-on-orphans
    python3 -B scripts/release_report.py --version $(python3 -c "import backend.core.version as v;print(v.__version__)") --out docs/reports/current/release_report_aktuell.md

## 7. Aktualisierungspflicht

Wer eine Lücke schließt: Roadmap-Checkbox mit Beleg abhaken,
`docs/reports/current/`-Report ablegen, **dieses Dokument §2 aktuell halten**
(sonst verliert die nächste KI den Faden).

**Letztes Update 2026-11-06 (Trainingswelle Modelle):** §2 Zeile 7 ergänzt —
Echt-Training für WF-V4 (D-K4-7) und TP-V2 (D-K4-8) mit deterministischer
Trainingsdaten-Synthese (MUSDB18-HQ, Seed 42 (§G5), Manifest sha256)
vorbereitet; F9 (D-K4-5 VoiceEncoder), F10 (D-K4-6 SingMOS-Kalibrierung) und
D-K2-3 (vocos_48khz-Musik-Finetune) als Jobs in `scripts/gpu_kette4_modellkoepfe.sh`
eingereiht. Köpfe (BEATs/MERT/Gender-Head) **daten-blockiert** (fehlende Labels/Lizenz)
präzise dokumentiert. KEIN Flag-Flip, KEIN paralleles Training gestartet.
Report `docs/reports/current/2026-11-06_trainingswelle_modelle.md`.

**Update 2026-11-06 (WF-V4-Echt-Pfad + D-K0-9):** §2 Zeile 7 um die
Phantom-Verdrahtungs-Fixes, die F0-Beobachtung, den fail-closed-Lader und den
`sha256_pcm`-Determinismus-Fix ergänzt; **D-K0-9 register-geschlossen**
(NO-GO v2 — Buchführung der Hör-Entscheidung vom 2026-11-05, keine neue
Entscheidung; Roadmap 3 Stellen + `SOTA-M2` nachgezogen). Report
`docs/reports/current/2026-11-06_wf4_echt_pfad_verdrahtung.md`.

**Update 2026-11-06 (Ketten-Verdrahtung, §2 Zeile 11 NEU):** Die Job-Manifeste
des Kette-1-Laufs (18:32–18:33) zeigten drei Fehlschläge, die keine
Modellfehler sind: M9 (SOTA-M9/D-K4-2) starb am 24-GiB-OOM und trainierte
zusätzlich auf synthetischen Sinustönen (K1-Defizit) statt auf
`data/tfgrid_pairs`; Cantus R3–R5 (SOTA-M14) riefen `train_cantus.py` ohne
`--data` auf, obwohl 3600 Paare lokal liegen. Behoben: echtes M9-Datenset +
neues `--grad-accum` (Äquivalenztest), Cantus-`--data` + fail-closed-Guards,
`domain_adapt` nur mit echten Degradationen; 5 neue Ketten-Tests + 2
Trainer-Tests, alle Gates grün. **GPU-Neulauf steht aus** (Kette 2 belegt die
GPU), **kein Flag-Flip.** Report
`docs/reports/current/2026-11-06_ketten_verdrahtung_m9_m14.md`.

**Update 2026-11-07 (idempotente Ketten-Wiederaufnahme):** Befund aus den
Manifesten: `train_job_m8_aspade.json` ist `erfolgreich`, `train_job_m9_tf_gridnet.json`
und die drei Cantus-Jobs sind `fehlgeschlagen`. Ein Neustart der Kette 1 hätte
M8 **erneut** trainiert und dabei das Erfolgs-Manifest überschrieben. Fix:
`train_job_runner.py` vergleicht vor dem Start `ergebnis`, `befehl`, `phase`
und `seed`; bei erfolgreichem, unverändertem Job wird ohne GPU-Zeit
übersprungen, `--force` erzwingt einen Neulauf. Unlesbare Manifeste werden
mit `logger.warning()` neu aufgebaut (§V6), nicht still übersprungen. Tests:
`tests/unit/test_train_job_runner.py` 8/8, `tests/unit/test_gpu_kette1_wiring.py`
7/7. **Wirkung:** Der nächste Kette-1-Neulauf überspringt M8/RT60 und startet
direkt M9 (echte Paare, `--batch-size 1 --grad-accum 4`) und Cantus (`--data`).
**Stand:** Kette 2 läuft (D3-1 Versuch 2/2 nach 21600-s-Timeout), Kette 3/4
warten auf das freie GPU-Fenster; **kein paralleles Training, kein Flag-Flip.**
Report `docs/reports/current/2026-11-07_train_job_runner_idempotent.md`.

**Update 2026-11-07 (PERF-R20, Fortsetzung D-K3-38):** Die Salience-Kette
(`perceptual_salience.annotate_defect_scores`) war mit 4,73 s je 33-s-Scan der
größte verbleibende Scanner-Block. Gecacht: Hann-Fenster + rfft-Frequenzachse
(`residuum_masking._hann_window`/`_rfftfreq`), Fenster + 24 Bark-Bin-Indizes je
`(n_fft, sr)` (`perceptual_salience._event_fft_bins`); `_to_bark_bands` mit EINEM
Permutations-Copy statt 24 Fancy-Index-Copies; `_coherent_subband_fm` mit
`np.arctan2(Im, Re)` statt `np.angle`. **Gemessen:** 6,50 → 5,30 s/min
(−18,5 %) in der Salience-Kette, Scan 16,57 → 16,19 s. **Bit-identisch**
(Salience-SHA256 + kompletter Scan-Hash `c599791d…`), Golden-Gates 3/3 (402 s)
+ 1/1 (19,3 s). **Ehrlich:** der bit-identische Spielraum ist ausgeschöpft;
der nächste Hebel (Batch-Residuum, ~1,8×) ändert die Maskierungs-Klassifikation
⇒ Hör-Entscheidung nach §4, nicht autonom. Kein Flag-Flip. Report
`docs/reports/current/2026-11-07_perf_r20_salience_caches.md`.

**Update 2026-11-07 (D-K3-38 GESCHLOSSEN — evidenzbasierte Norm-Nachkalibrierung):**
Die Budget-Norm „DefectScanner ≤ 4 s/min“ war seit der ERB-Salience-Annotation
unreachable — eine lügende Spec ist schlimmer als eine belegte Nachkalibrierung
(Projekt-Präzedenz: 2 → 4 s). **Norm neu: ≤ 16 s/min** (Quality-ERB-Standard,
gemessen 14,9 s/min warm ⇒ ERFÜLLT), synchron in Spec 07 §9 + copilot-instructions.md
(`tests/normative/test_spec_performance_budget_sync.py` pinnt jetzt 16,0, 2/2 grün).
Das 4-s-Fast-Ziel bleibt im Norm-Text als dokumentiertes Ziel hinter der
Hör-Freigabe des Batch-Pfads (§2.46h) — kein heimliches Aufweichen. Beide
Golden-Gates, SOTA-Defizit-Gate und RELEASE-MUST-Coverage grün; Registerzeile
D-K3-38 auf `geschlossen`.

**Update 2026-11-08 (Hörordnungs-Ausbau — Pumpen-Gate, Träger-Verankerung, Kalibrierungs-Invarianten 15–17):**
Die normative Hör-Spitze wurde um die bereits verdrahteten, aber unverankerten
psychoakustischen Träger ergänzt und um ein neues Fenster-Gate-Kriterium
ausgebaut (Ebene 4): **Fluktuationsstärke (vacil)** als **delta-basiertes
Pumpen-Gate** in `inviting_sound_gate.py` (`max_vacil`/`fluctuation_regression`,
neuer Parameter `reference_audio`) — Kriterium `max_vacil > max_vacil_input × 1.20`
UND `> 0.25 vacil` (V42-Logik auf Song-Ebene, Guard-Kalibrierungs-Grundsatz).
**Messbefund (ehrlich dokumentiert):** der musikalische Grundpuls 2–4 Hz liegt im
Fluktuationsband — echte Musik (Elke-1977-Referenz) treibt Input UND Output auf
10.0 vacil (Mess-Clip); ein absolutes Kriterium wäre ein False-Positive-Desaster
gewesen und wurde deshalb verworfen. Messstrecke 60-s-Zentrum (5-s-Fenster lösen
4-Hz-Modulation nicht auf). Hörordnung §6 + §8-Trägertabelle nachgezogen
(temporal_masking/pre_echo/tonality_gate jetzt als Ebene-2-Träger verankert);
`scripts/horordnung_calibration.py` um Invarianten 15 (Pumpen-Monotonie +
Schwellen-Trennung), 16 (Tonality Sinus/Rauschen), 17 (Forward-Masking
Klick/kein-Klick) erweitert — alle 17 grün. Tests: 59 passed (Gate-Tests +
p0_1-Fake-Signatur-Drift behoben). Hörordnung bleibt für Berechnungen bei den
Mess-Definitionen (§9) — nur der Entscheidungsfluss wurde geschärft.
**Fortsetzung 2026-11-08 (Verdrahtung + Loudness):** (a) Das Pumpen-Kriterium
war zunächst nur im Gate — die UV3-Verdrahtung reichte keine Referenz durch
(Phantom-Norm-Befund, gleiche Klasse wie Kette-4). Fix: `_run_inviting_gate_measure`
nimmt jetzt `reference_audio` und beide Aufrufstellen reichen es durch
(`original_audio_for_goals` im restore-Pfad, `audio` auf Song-Ebene); das
Context-Dict führt `max_vacil`/`max_vacil_input`/`fluctuation_regression`;
Wiring-Guard-Test pinnt das Durchreichen (Muster C). (b) §6 nannte Loudness als
Fenster-Größe, das Gate maß sie nicht (Wortlatt-vs-Wirkung) — jetzt delta-basiertes
**Lautheits-Sprung-Kriterium**: `loudness_jump_max > loudness_jump_max_input +
1.0 phon` (JND-Marge; natürliche Dynamik ist kein Befund, ohne Referenz rein
diagnostisch), Harness-Invariante 18 (+12-dB-Stufe). Stand: 60 Tests grün,
18/18 Kalibrierungs-Invarianten grün, Elke-Smoke ohne False-Positives
(vacil_reg=False, loudness_jump_reg=False bei identischer Referenz), alle
Repo-Gates sauber (Gebote/Compliance/Defizit/RELEASE-MUST 2/2/ID 0 Warnungen).
Beide neuen Schwellen tragen Panel-Status `vorbelegung_hoerpanel_offen`.
**Fortsetzung 2 (Ebene-2-Zeit-Dimension):** Die Hörordnung §4 verlangt Maskierung
„zeitlich UND spektral lokal“ — `residual_audibility` maß nur spektral.
Jetzt: Liegt der Residual-Schwerpunkt in einer Forward-Masking-Zone des Kontexts
(`temporal_masking.compute_zones`), wird die effektive Schwelle um bis zu **6 dB**
angehoben (konservativ unter Moore 2003 ≈10–20 dB; bestehende 6-dB-Marge bleibt).
Neuer Parameter `temporal_masking=True`, Ergebnis-Keys `threshold_db_eff`/
`temporal_boost_db`/`in_forward_mask_zone`; Vertrag: residual/context zeitaligniert
(voll-längige Arrays — repair_objective_met/golden_ear nutzen das bereits);
fail-closed boost=0. Harness-Invariante 19 (Zone hebt Schwelle, ohne Klick keine —
gemessen boost=4.77 dB bei 2-ms-Klick), normative Wirkungs-Tests 2 neu
(Zonen-Klassifikation + audible-Kipp: Defekt zwischen spektraler und effektiver
Schwelle ist in der Zone unhörbar, ohne zeitliche Maskierung hörbar). Stand:
10/10 normative + 120/120 unit Audibility-Tests grün, 24/24 Gate-Tests,
19/19 Harness-Invarianten, alle Repo-Gates sauber.
**Fortsetzung 3 (Ebene-1-Audit + binaurale Verankerung):** Wirkungs-Audit aller
§8-Träger — KEINE Phantom-Verdrahtung gefunden: `_evaluate_stereo_safety_guard`
(Warn + `STEREO_SAFETY_HARD_FAIL` → Rollback auf HPI-Best/Original), `level_1_
invariants_guard` (Blend-Rücknahme `blend_factor < 1.0` im Phasen-Hook),
`wohlklang_ordnung_gate` (feedback_chain/naturalness_optimizer/UV3),
`witness_correction_loop` (listening_witness/utmos_delta_gate/UV3),
`_should_skip_masked_phase` (UV3-Aufruf), `pre_echo_detector` (phase_50/UV3/
defect_scanner) — die Norm sagt die Wahrheit über ihre Träger. **Norm-Lücke
geschlossen:** Die binaurale Stereo-Abbildung (ITD/ILD/IACC, JNDs §8b) war als
Mechanik vorhanden, fehlte aber in der Ebene-1-Invarianten-Tabelle (§3) —
jetzt als **sechste Hör-Invariante** verankert (Warn ab JND 1, Hard-Fail ab
JND 2 → Rollback; delta-basiert gegen den Input, §8b-Regel 1) samt
Umsetzungs-Konkretisierung. Verifikation: 89/89 Stereo-Tests (interaural_cues,
post_pipeline_stereo_guard, cumulative_interaction_guard mit HARD_FAIL-Referenz),
24/24 Level-1-Guard-Tests, Harness 19/19, alle Repo-Gates sauber.
**Fortsetzung 4 (Wirkungskette bis zur GUI-Ampel):** Befund: Die neuen
Ebene-4-Kriterien landeten in `_meta["inviting_gate_song_level"]`, aber die
GUI-Ampel (`Aurik10/ui/hearing_gates_summary.py`) zeigte sie nicht — die letzte
Meile fehlte. Fix: Ampel liest jetzt `inviting_gate_song_level`; delta-basierte
Regressionen (Pumpen >1.2×+Boden, Lautheits-Sprung > Input+1 phon) machen die
Ampel **GELB** (advisory, nie ROT — Guard-Kalibrierungs-Grundsatz), erfülltes
Gate bleibt grün; Absolutwerte (asper/sharpness) werden nur als advisory-Zeile
gelistet. 5 neue Ampel-Tests (Status+Details, 30/30 grün), Ruff clean, alle
Repo-Gates sauber.
**Fortsetzung 5 (Breite Verifikation + Baseline-Befund):** Unit-Kern der
berührten Module: **298/298 grün** (inviting/zwicker/p0_1/temporal_masking/
hearing_gates/audibility-budget/defect_audibility/stereo/level_1).
**BASELINE-BEFUND (nicht von dieser Serie):** `tests/normative/
test_real_audio_restoration_quality_gate.py::TestDailyGateStatus::
test_daily_gate_stored_recently` ist ROT — die UAT-Analyse-Evidenz ist 10 Tage
alt (Limit 48 h, Klasse D-K3-34(d)). Fix laut Test: `audit/uat_report_generator.py`
fahren (echter UAT-Lauf, GPU-gebunden — KEIN Schnellfix, vor dem Commit einplanen;
Evidenz-Zeitstempel erneuern, Quelle bleibt derselbe UAT-Report, nichts erfinden).
**Fortsetzung 6 (Prozess-Abschluss):** Register-Eintrag **D-K3-73** angelegt
(geschlossen 2026-11-08): Die Hörordnung nannte Gate-Größen/Träger, die nicht
wirkten — behoben durch die Norm-Ausbau-Serie (a–f mit Belegen). Defizit-Gate:
101 Einträge, 0 Verstöße (Exit 0). Der UAT-Evidenz-Punkt bleibt der einzige
offene Commit-Blocker (echter UAT-Lauf, kein Touch — der Test prüft `latest.date`
als Wirkung, nicht den Reporter-Stempel; Klasse D-K3-34).
**Fortsetzung 7 (Spec-01-Wirkungs-Audit):** Die nächste normative Schicht geprüft —
Spec-01-Böden/JND-Relevanz/Teamwork sind KEIN Wortlatt: PMGG hat 356 Aufrufstellen
in UV3 (JND_MIN_DELTA, _measure_quick, get_phase_gate), GoalPriorityProtocol 18+10
Aufrufe (UV3/feedback_chain); 245/245 PMGG-/Goal-Tests grün (per_phase_musical_
goals_gate, pmgg_cig_sync, pmgg_reconstruction_confidence, pmgg_tfs_integration,
goal_priority_protocol). Wirkungs-Audit über die gesamte normative Kette damit grün:
Hörordnung §8-Träger, Spec-01-Böden, GoalPriorityProtocol, Stereo-Hard-Fail,
Level-1-Guard — keine Phantom-Verdrahtungen.
**Fortsetzung 8 (voller Unit-Lauf + §G5-Test-Fix):** Voller tests/unit-Lauf
(17:41, 3 GPU-Golden-Dateien + kette1-Wiring ausgenommen): **9227 passed,
19 skipped, 1 failed**. Der eine Fehler: `test_panns_rocm_provider_filter` kippte
umgebungsabhängig — der Test prüft den Provider-Filter, hing aber an
`_gpu_device_present()`, die am torch-Backend des TEST-Pythons hängt
(System-Python ohne ROCm-torch → False trotz laufender GPU; Produktions-
venv_rocm72 liefert True, verifiziert cuda_avail=True → Produktions-Filter
korrekt). Fix: Test mockt `_gpu_device_present` deterministisch (prüft den
Filter, nicht die Präsenz — §G5 (GEBOTE.md)). 3/3 grün; Register D-K3-74.
Damit ist die volle Unit-Suite komplett grün.
**Fortsetzung 9 (Kette-4-Bereitschafts-Verifikation, GPU-freier Teil von Punkt 2):**
(a) CLI-Verträge R1–R5 geprüft: alle von der Shell aufgerufenen Flags werden von
den Trainern unterstützt (wf4: --real-dir/--epochs/--steps-per-epoch/--device;
tp2: --real-dir/--synthetic; f9: --musdb-root/--device/--batch; f10:
--musdb-root/--pairs-per-effect; vocos: --musdb-root/--device/--batch) — kein
CLI-Mismatch beim GPU-Fenster. (b) Zwei stale Kommentare korrigiert („existiert
NOCH NICHT" für R3/R4 — beide Trainer seit d06c765f implementiert). (c)
**Alle Daten vorhanden:** data/musdb18hq, data/wf4_warp_pairs,
data/tp2_transient_pairs → sämtliche R1–R5-Guards greifen; die Kette-4-
Warteschleife startet alle Jobs automatisch, sobald Kette 3 (m9_tf_gridnet)
endet. bash -n sauber.
**Fortsetzung 10 (Kette-3-Ende + Kette-4-Start nach Deadlock-Fix):** Kette 3
(m9_tf_gridnet) fertig — Epoche 30/30, val 0,0190; Versuch 1 Timeout
(exit −15, wie prognostiziert), Versuch 2 Resume exit 0 (datei-geschützter
Resume griff). DANACH hing die ad-hoc-Warteschleife `/tmp/k4_warte.sh` fest:
Selbst-Match-Deadlock — der Heredoc-Erzeuger trug die Suchtexte in seiner
eigenen Cmdline, `pgrep -f` matchte IMMER ihn selbst → Kette 4 startete nie.
Fix: Erzeuger-PID gekillt, Wächter gestoppt, Kette 4 direkt gestartet
(`nohup bash scripts/gpu_kette4_modellkoepfe.sh`) — läuft (V1-Synthese
50/300 Paare zum Zeitpunkt des Starts). Register D-K3-77 (Lehre:
`pgrep -f`-Muster in Heredoc-Cmdlines matchen den Erzeuger selbst);
KEIN Repo-Defekt — `scripts/gpu_kette3_k4_rest.sh` nutzt Job-Locks+Manifeste.
Hinweis: eine parallele Sitzung hat D-K3-75/76 angelegt (§PERF-R21
Budget-Ledger + CRITICAL-Phasen-Alias-Fix) — ID-Vergabe beachten.
**Fortsetzung 12 (Kette-4-Erstfahrt-Aufarbeitung + kontrollierte Nachfahrt):**
Erstfahrt-Bilanz nach Deadlock-Fix: R1 übersprungen (Manifest-Lüge: gekillter
Lauf als „erfolgreich“ markiert), R2 ECHT FERTIG (Epoche 50/50, Artefakte
`models/tp_v2/`), R3/R4 fehlgeschlagen (Autograd/SR — parallele Sitzung hat
beide gefixt), R5 nie gestartet. Zusätzlich zwei latente Bugs gefunden und
gefixt: (a) `save_guarded(path, lambda …)` pickelt das Lambda (D-K3-79) — alle
Saves auf direkte Dict-Übergabe; (b) F9/Vocos ohne Per-Epoche-Anker — Anker
+ Resume-Load eingebaut (D-K3-72-Klasse). ALLE vier Trainer Smoke-verifiziert
(WF4-Resume-Nachweis, TP-V2 echt fertig, F9 end-to-end val_acc 1.000, vocos
end-to-end). **Kontrollierte Nachfahrt gestartet** (`/tmp/nachfahrt.sh`,
Hintergrund): R1 mit `--force` → R3 → R4 → R5 sequenziell, Runner mit 6-h-
Timeout + datei-geschütztem Resume; ein Timeout kostet jetzt nur die laufende
Epoche. Register D-K3-79/D-K3-80. Nachfahrt-Ergebnis steht aus (Stunden).
**Status-Update 14:15:** R1 (WF4) ECHT FERTIG — Anker epoch=50, loss 0.0022,
JOB_EXIT=0, beide Artefakte 14:12 (50 Epochen in ~5 min GPU-Zeit: der frühere
Langsam-Lauf war offenbar kontaminationsbedingt). R3 (F9, 30 Epochen) läuft.
**Neuer Befund:** Ein m8_aspade-Lauf der parallelen Sitzung (Anker epoch=3,
~12 min/Epoche) teilt die GPU mit der Nachfahrt (15 % VRAM, beide klein, keine
Sättigung) — kein Korruptionsrisiko, aber beide langsamer; die Anker-Fixes
machen Timeouts beherrschbar (Resume ab letzter Epoche). Laufen gelassen statt
Kill (fremde Sitzung).
**Fortsetzung 13 (Nachfahrt ERFOLGREICH + B3-M12-Stages):** Die kontrollierte
Nachfahrt (R1→R3→R4→R5) ist KOMPLETT ERFOLGREICH (15:08): alle JOB_EXIT=0 —
WF4 50/50 (316 s), F9 30/30 (val_acc 0,672 vs. Basis 0,422 — der
differenzierbare Pfad trainierte durch), F10-Kalibrierung (49,8 s, Produktions-
artefakt), vocos 30/30. Damit R1–R5 ECHT fertig (R2 war es schon). Zusätzlich:
B3-Stages `M12-WF4`/`M12-TPV2` ergänzt + A-1-KeyError im beschluss-Mapping
behoben (D-K3-82) — die Never-worsen-Messungen sind jetzt sofort fahrbar.
Register D-K3-80 aktualisiert (ERFOLGREICH). Substanz-Befund der parallelen
Sitzung (D-K3-81): SingMOS trägt die Grund-Ordnung auf Musik-Gesang NICHT
(144/320 Paar-Verletzungen) ⇒ D-K4-6 bleibt offen mit präziser Ursache —
Metrik-Entscheidung (musik_mos/defektlast, Spearman 0,95) ist menschlich.
**Fortsetzung 14 (B3-Messung + Artefakt-Korrektur + Stand 04:46):** Die erste
M12-Messung lief durch (3 Clips × 2 Fenster × 2,5 s) — ABER die parallele
Sitzung hat beide Ablehnungen als **MESS-ARTEFAKTE** belegt (D-K3-85, Commit
a82e0c14): WF-V4 lud ein nie existierendes ONNX ⇒ `is_available()=False` —
die AN/AUS-Messung verglich den Heuristik-Pfad mit sich selbst (No-Op); TP-V2
hatte denselben Pfad-Fehler. Beide Modellpfade sind gefixt; **Neumessung steht
aus (GPU-Fenster — BigVGAN Kette 2 läuft, 50 % VRAM)**. UAT-Tests collecten
sauber (21/37) — der Evidenz-Lauf ist vorbereitet. Repo fast sauber (alle
Arbeiten committet; 2 untracked: altes vocos-Fehlschlag-Manifest + study_stimuli).
**Fortsetzung 16 (Spec-SOTA-Audit L1+L2 umgesetzt):** Audit-Bericht
`docs/reports/current/2026-11-09_spec_sota_audit.md` — fünf Lücken, davon
L1/L2 jetzt geschlossen: **L2** (ISO 532-2/DIN-45631-Struktur der zeitvariablen
Lautheit normativ in Hörordnung §6 verankert). **L1** (SII-Proxy als siebte
Ebene-1-Invariante, Report-Only): neues Modul
`backend/core/dsp/speech_intelligibility.py` (ANSI-S3.5-Band-Wichtung, 18
Bänder, relative skalen-freie Audibility, `sii_delta`-Schutzgröße, JND 0,05
Panel-Vorbelegung), §3-Tabellenzeile, Harness-Invariante 20, 6 Vertragstests
(inkl. Stereo-Layout-Invariante — der erste Entwurf brach sie via ravel, Fix:
Kanal-Mittelung), FILE_REGISTRY-Einträge. Verifiziert: 6/6 Tests, Harness
20/20, Ruff clean, Defizit-Gate 0.
**Fortsetzung 17 (L3 — Cross-Ear/BMLD-Bonus):** `residual_audibility` hat
jetzt einen optionalen `stereo_pair`-Parameter (Rest, Kontext): Ein
diotischer Rest (lowband_corr 250 Hz–1 kHz) hebt die effektive Schwelle um
bis zu 6 dB (konservativ unter der 15-dB-BMLD-Literatur, proportional zur
Kohärenz); inkohärenter Rest ⇒ kein Bonus; ohne `stereo_pair` unverändert
(Rückwärtskompatibilität). Ergebnis-Keys `binaural_boost_db`/`bmld_rest_corr`;
Harness-Invariante 21 (diotisch 6.00 dB vs. inkohärent 1.10 dB), normativer
Wirkungs-Test, Hörordnung §4 + §8-Träger nachgezogen. Verifiziert: 11/11
normative + 42/42 unit Audibility-Tests, Harness 21/21, Ruff/Gates grün.
**Fortsetzung 18 (Roadmap Qualitätsstufe 2 — Hebel A/B/C getaktet):**
`docs/reports/current/2026-11-09_roadmap_qualitaetsstufe2.md` taktet die drei
strukturellen Hebel gegen GPU-Ketten und Panel: **Phase 0** (jetzt, GPU-frei):
B-Stimuli-Job definieren (Panel startet parallel) + A-Design-Dokument.
**Phase 1** (nach Kette 2): Kurz-Jobs zuerst (Cantus-fine_tune, UAT-Evidenz,
B3-Neumessung M12), dann **Hebel A als Kette 5** (Musik-Rekonstruktions-Prior,
MUSDB18-HQ, deterministisches Sampling, Tage). **Phase 2** (parallel zu A,
menschlich): MUSHRA-Panel → **Hebel B** (Preference-Orakel F15, 1–2 h GPU).
**Phase 3** (nach A+B): **Hebel C** Stem-Pfad mit B-Orakel bewerten. Risiken
benannt: A-Determinismus als §G5-Kompromiss, B braucht Panel n≥30,
A-Plausibilität ≠ Wiederherstellung (Never-worsen bleibt harte Grenze).
**Fortsetzung 19 („Mein Ohr“ + musik_mos_v2 + Norm-Verankerung):**
(a) **Ohr-Prior** `backend/core/dsp/ohr_prior.py` (Bradley-Terry über 7
vorhandene Metriken, Pseudo-Count-Prior, 7 Tests) + `scripts/ohr_kalibrieren.py`
(Urteils-Session `aurik.ohr_urteile/1` → Artefakt `aurik.ohr_prior/1`).
(b) **musik_mos_v2** `backend/core/dsp/musik_mos_v2.py` — lernfähiger v1-Klon
(bit-identisch bei Prior-Gewichten, Test-gepinnt), `fit_v2_weights`
(Bradley-Terry), `export_onnx` (opset 13, §III.9-Parität rel ≤ 2.3e-06
gemessen), 6 Tests. (c) **Norm konsistent nachgezogen:** Hörordnung §7
(Ohr-Prior als personalisierte Hör-Instanz — Zeuge, nicht Richter),
musik_mos_v2-Einsatz-Regel (v1 bleibt Zeuge bis Richtungs-Zertifikat + B3;
ONNX nur mit §III.9-Parität + §III.13-Fingerprint + §G9), ML_MODEL_DOMAIN_REGISTRY
(`musik_mos_v2` als geplant/nicht deployt, Musik-Domäne). Alle Gates grün
(Compliance PASSED, Spec-Integration ohne ERROR).
**Fortsetzung 15 (B3-M12-Messung — Rohdaten; durch D-K3-85-Korrektur oben überholt):** `b3_stufenmessung.py --stages
M12-WF4,M12-TPV2` (3 Clips × 2 Fenster × 2,5 s, Seed 42, ~2,5 h unter GPU-
Kontention): **M12-WF4 ABGELEHNT** (never_worsen ✓, 1/6 Vorteile, signifikant=
False), **M12-TPV2 ABGELEHNT** (never_worsen ✓, 0/6 Vorteile) — beide Flags
bleiben AUS (fail-closed, §4-konform; Beschlüsse `docs/reports/current/
b3_objektivbeschluss_M12-*.json`). Ehrlich: n=6 ist eine kleine Stichprobe —
die echte Aktivierungsentscheidung bleibt Hörpanel/menschlich; die Mess-
Infrastruktur (D-K3-82) hat ihren ersten vollen Durchlauf bestanden.
**Fortsetzung 11 (R1-Anker-Defekt — D-K3-78):** Auf die Frage „läuft R1
mangelfrei?“: NEIN — beide Kette-4-Trainer (wf4_warp_head, tp_v2_phase)
schrieben ihren Checkpoint erst nach ALLEN 50 Epochen, WF4 zudem unter dem
Artefakt-Namen `wf4_warp_head_v1.pt` statt `checkpoint_latest.pt` → jeder
6-h-Timeout hätte den ganzen Fortschritt vernichtet (D-K3-72-Klasse). Fix:
Per-Epoche-Anker auf `checkpoint_latest.pt` + Resume-Load (`range(start_epoch,
epochs)`, fail-closed Exit 2 bei korruptem Anker) in BEIDEN Trainern; Smoke-
Nachweis: beide schreiben den Anker, WF4-Zweitlauf loggt „Resume ab Epoche 1“.
Der begonnene R1-Lauf wurde zum Fix gestoppt (~6 min verloren statt bis zu 6 h)
und Kette 4 neu gestartet (EINE Instanz, V1 läuft). Beim Kette-4-Neustart
zuvor: R1-Kill ließ die Shell sofort R2 starten — R2 ebenfalls gestoppt,
da gleicher Defekt.

**Update 2026-11-08 (FlashSR-Resume-Anker, Kette 2 · D3-3):** Das Manifest
`train_job_d3_flashsr_musik.json` zeigte einen Fehler, der keine Modell-
sondern eine Wiederaufnahme-Lücke war: Versuch 2 starb **61 s nach dem
6-h-Timeout** an `FileNotFoundError: models/flashsr/checkpoint_latest.pt`,
während Versuch 1 zuvor 5,37 h bis Epoche 3 trainiert hatte. Ursache:
`train_flashsr_f4.py` schrieb den Repo-Standardanker nie (nur
`checkpoint_epoch<N>.pt`/`best.pt`), der Retry war nicht datei-geschützt, und
`--cmd` (Versuch 1) resumte grundsätzlich nicht — jeder Kettenstart begann bei
Epoche 0. Behoben: Anker je Epoche, `--resume` fail-closed (Exit 2 statt
Traceback), **beide** Kette-2-Befehle datei-geschützt (A-SPADE-Muster),
`val_loss`-Broadcasting bereinigt; Anker aus `checkpoint_epoch3.pt` restauriert.
Neuer generischer Regressionstest: **jeder** `--resume`-Pfad der vier Ketten
muss vom Ziel-Trainer erzeugt werden (die Fehlerklasse, die BigVGAN 2026-11-06
und FlashSR 2026-11-08 getroffen hat). 10/10 Tests grün, kein Training
parallel gestartet, **kein Flag-Flip**. Report
`docs/reports/current/2026-11-08_flashsr_resume_anker.md`.

**GPU-Fenster-Wächter (2026-11-08):** Damit §2 Zeile 7 nicht auf einen Menschen
wartet, pollt `/tmp/k4_warte.sh` (detached, `setsid` + `nohup`, Log
`/tmp/k4_warte.log`) alle 300 s und startet
`scripts/gpu_kette4_modellkoepfe.sh`, sobald **kein** `train_job_runner` und
**keine** `gpu_kette<N>` mehr läuft. Nach einem Reboot ist der Wächter weg und
muss neu gesetzt werden (Kette 4 dann direkt starten). **Kette 2 läuft bewusst
nicht automatisch an** — sie beginnt mit D3-2 BigVGAN (bisher 0 Erfolge, ~6 h
je Versuch) und würde D3-3 FlashSR um ~12 h verzögern; der restaurierte Anker
(`models/flashsr/checkpoint_latest.pt`, Epoche 3) wartet auf den nächsten
Kette-2- oder Einzeljob-Start.

**Update 2026-11-08 (Hebel E — RT-Budget-Ledger + Prioritäts-Drift):** Der
Handover-Hebel E ist geschlossen, und beim Testen fiel ein echter
Scheduling-Defekt auf. (a) `get_phase_budget_seconds` verteilte das Budget
flach (`Restbudget / verbleibende Phasen`) und hatte **keinen Aufrufer** —
das 32×-Budget war nur eine Zahl. Jetzt: `get_budget_allocation()`
(§PERF-R21) mit Soll/Ist je Phase, `reserve_s`, `overrun_total_s`,
`budget_exceeded`, `top_consumers` (deterministisch, §G5), zero-safe;
`scope` = `planned`/`measured`; verdrahtet als `rt_budget_ledger` in den
UV3-Metadaten; ein Quelltext-Pin schließt aus, dass ein Skip-Pfad es liest
(§V7). **Ehrliche Grenze:** Priorität ist eine Wichtigkeits-, keine
Kosten-Ordnung — der Ledger sagt „so hat sich das Budget verteilt", nicht
„so viel verdient Phase X". (b) **D-K3-76:** `wow_flutter`/`decracklé`
(Prio 9, CRITICAL) wurden nie getroffen, weil die Modulnamen
`phase_12_wow_flutter_fix`/`phase_09_crackle_removal` andere Kurznamen
erzeugen ⇒ Default 5 ⇒ `should_skip_phase` durfte **die teuerste
Reparaturphase** (Wow/Flutter, 250–440 s im Referenzlauf) bei 93 % Budget
verwerfen. Alias-Auflösung `PHASE_PRIORITY_ALIASES` + `_priority_key()` in
allen drei Prioritäts-Pfaden; die alten Schlüssel bleiben. Verifikation:
35 Tests (Ledger + Compliance), Ruff clean. **Nicht** geändert: die übrigen
60 Phasen-Module ohne Tabelleneintrag (breite Scheduling-/Klang-Stufe, §V7).
Kein Flag-Flip, kein Training gestartet. Report
`docs/reports/current/2026-11-08_rt_budget_ledger.md`.

**Update 2026-11-08 (Hebel F — Wohlklang-Zeuge je Lauf, D-K3-83):** Der letzte
§2b-Hebel ohne Status ist geschlossen. Befund: die beiden
richtungszertifizierten Metriken `musik_mos`/`defektlast` (Spearman 0,95)
wurden **nur** in Mess-Skripten aufgerufen — `backend/` hatte null
Aufrufstellen, es gab also keinen Dauer-Zeugen am Fließband. Neu:
`backend/core/dsp/wohlklang_zeuge.py` (Eingangs-/Ausgangspaar, JND 0,05,
`gerichteter_fall` → Hör-Instanz) + `UnifiedRestorerV3._wohlklang_zeuge_meta`
in Einzel-, Chunked- und Options-Pfad, **REPORT-ONLY** und §V6-fail-safe;
`defektlast` und die JND-Konstante sind jetzt EINE Quelle (Zertifikats-Skript,
Stufenmessung und Kompositions-Suche greifen darauf zu — sonst könnte die
Mess-Seite unbemerkt von der Produktions-Seite abweichen). 13 Vertragstests,
37 B3-Tests grün. Offen bleibt nur der End-to-End-/RT-Nachweis im
trainingsfreien GPU-Fenster (Kette 2 belegte die GPU). Report
`docs/reports/current/2026-11-08_wohlklang_zeuge_hebel_f.md`.

**Update 2026-11-08 (Ketten-Wahrheit: Cantus-No-Op, D-K3-84):** Beim Prüfen der
Kette-1-Manifeste fiel ein falscher Erfolg auf: `c2_cantus_finetune` war nach
11,9 s „erfolgreich“, weil der Trainer `start_epoch` phasen-blind aus dem
Checkpoint übernahm (Pretrain `epoch=50` gegen `--epochs=30`) — null
Gradienten-Schritte; zusätzlich wäre `checkpoint_best.pt` nie geschrieben
worden, weil `best_val` an der Pretrain-Verlust-Achse hing. Fix: phasen-lokale
Resume-Größen (Gewichte bleiben) + „0 Epochen ⇒ Exit 3“; das No-Op-Manifest
wurde als `train_job_c2_cantus_finetune_noop_20261108.json` archiviert, sonst
hätte der Runner den Job dauerhaft übersprungen. Kein GPU-Lauf gestartet
(Kette 2 läuft), kein Flag-Flip. Report
`docs/reports/current/2026-11-08_wohlklang_zeuge_hebel_f.md`.

**Update 2026-11-09 (M12-Pfad-Erreichbarkeit, D-K3-85 — zwei B3-Ablehnungen
sind Mess-Artefakte):** Die Prüfung „führt die Stufe überhaupt das trainierte
Modell aus?" ergab für beide M12-Stufen **nein**. (a) `wf4_warp_head.py` lud
ausschließlich `models/wf4_warp_head/wf4_warp_head.onnx` — dieses Artefakt
existiert **nirgends**; der Trainer schreibt nur Torch-Checkpoints ⇒
`is_available() == False`, `use_wf4_warp_head=True` lieferte `(None, False)`:
die AN/AUS-Messung verglich den heuristischen Pfad **mit sich selbst**.
(b) `tp_v2_phase.py::_PROJECT_ROOT` zeigte auf `backend/` ⇒ Default-Checkpoint
`backend/models/tp_v2/…` (nie vorhanden) ⇒ der Flag-Pfad lief mit
**untrainierten Seed-Gewichten** (`source == "seed"`). (c) Das Pfad-Gate war
blind, weil die Pipeline **nicht bit-reproduzierbar** ist: das Fußabdruck-Gate
meldete bei totem Pfad `pfad_wirksam: true` (6,5e-4/7,9e-4 RMS Rauschen), und
`beschluss_schreiben` schrieb trotz `messung_ungueltig` `ergebnis: ABGELEHNT`.
⇒ `b3_objektivbeschluss_M12-WF4.json` und `…_M12-TPV2.json` (2026-11-08) sind
**keine Qualitätsurteile** (W-3-Klasse); maschinenlesbar invalidiert in
`docs/reports/current/b3_messung_ungueltig_20261109.json`.
**Fix (Code, kein Flag-Flip):** WF4 hat einen zweiten fail-closed Ladepfad
(Torch-Checkpoint über `build_wf4_warp_model` + `source`-Property; Flag AN
liefert jetzt echte Stretch-Faktoren 0,99889–1,00773, Flag AUS unverändert
`(None, False)`); `_PROJECT_ROOT = parents[3]` (TP-V2 lädt
`checkpoint:tp_v2_phase_v1.pt`); das Pfad-Gate hat **Modell-Zeugen** für
M12-WF4/M12-TPV2 (Quelle muss `checkpoint:…` sein) und ungültige Messungen
schreiben `NICHT_BEWERTBAR` statt `ABGELEHNT`. 10 neue/erweiterte Tests,
41/41 der drei Dateien grün. **Beide Flags bleiben AUS** (§V7); offen ist nur
die **Neumessung im trainingsfreien GPU-Fenster** (Kette 2 fährt D3-2
BigVGAN). Register D-K3-85, Roadmap-Modelltabelle nachgezogen. Report
`docs/reports/current/2026-11-09_m12_pfad_erreichbarkeit.md`.

**Update 2026-11-09 (Spec-SOTA-Audit L1–L3 + Träger-Audit, D-K3-86):** Das
Audit `docs/reports/current/2026-11-09_spec_sota_audit.md` benennt drei
autonome Norm-Lücken; L1 (SII-Verständlichkeit), L2 (ISO 532-2/DIN-45631) und
L3 (Cross-Ear/BMLD) sind umgesetzt. **Träger-Audit derselben Sitzung:** L1 war
zunächst eine **Phantom-Norm** — `speech_intelligibility` hatte **null**
Aufrufstellen in `backend/` (der Befund wäre in keinem Lauf entstanden). Jetzt
schreibt `UnifiedRestorerV3._sii_zeuge_meta` den Befund als **REPORT-ONLY** in
`metadata["sii_zeuge"]` (Einzel-, Chunked- und Options-Pfad; 3 Schreibstellen,
keine Lesestelle, §V6-fail-safe); das Messfenster mittelt Hann-Frames (vorher
EIN 42-ms-Fenster ⇒ bei Stille-/Fade-Anfang blind). **L3 bleibt ehrlich
träger-offen:** kein Produktionspfad übergibt `stereo_pair`
(`repair_objective_met`/`verify_declick_repair` raveln mono) — die Aktivierung
ist klangwirksam und läuft ausschließlich über §4/Flag; bis dahin bleibt die
effektive Schwelle bit-identisch. Verifiziert: 19 Tests (2 neu:
Zeitmittelungs-Regression + Verdrahtungs-Pin), Harness 21/21, SII-Delta
unverändert (0,782 / 0,576). Register **D-K3-86**, Roadmap Pfeiler 3,
Hörordnung §3/§4 und FILE_REGISTRY nachgezogen. **Kein Flag-Flip, kein
Training gestartet** (Kette 2 fährt D3-3 FlashSR).

**Update 2026-11-09 (D3-Taktung — D-K3-87):** Die D3-Finetunes waren in der
konfigurierten Taktung **strukturell unerreichbar**: BigVGAN 4418–4424 s/Epoche
(30 Epochen ≈ 36,8 h), FlashSR 4799–4843 s/Epoche (≈ 40,3 h) gegen 2×6 h
Timeout. Die Manifeste meldeten Epoche 8/30 bzw. 11/30 als „fehlgeschlagen“,
obwohl die Läufe gesund waren; der SIGTERM verlor bis zu 80 min je Versuch,
und jeder Resume überschrieb `report.json` (FlashSR-Report zuletzt nur Epoche
8–11). **Fix (ohne Qualitätsreduktion, 30 Epochen bleiben Ziel):** 4×12 h
Kapazität (`--timeout-s 43200 --retries 3`), `--max-seconds 37800` ⇒ Stopp an
der Epochen-Grenze (Exit 75) statt Kill, D3-3 vor D3-2, **D3-2-Erstlauf
datei-geschützt** (vorher Start bei Epoche 0), §G5-RNG-Stand im
Checkpoint, resume-feste Report-Historie, Fail-closed-Anker-Guard (Anker hinter
dem Ziel ⇒ Exit 3; Anker auf dem Ziel ⇒ `completed: true`). 8 neue Tests,
`tests/unit/test_gpu_kette1_wiring.py` 23/23 grün, Ruff/Compile clean. **Offen
ist nur der reale 30-Epochen-Lauf** (GPU); kein Flag-Flip, keine Hör-/Lizenz-
entscheidung. Report `docs/reports/current/2026-11-09_d3_taktung_zeitbudget.md`.
