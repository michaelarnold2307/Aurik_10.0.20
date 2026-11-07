# SOTA-Handover — Weiterarbeiten ohne Vorkenntnis (jede Programmier-KI)

> **Zweck:** Dieses Dokument macht den Zustand von Aurik session-unabhängig
> lesbar. Jede KI (Kun, Claude Code, Copilot, Codex, Cursor …) soll nach dem
> Lesen **sofort** wissen, was zu tun ist — auch nach Abbruch, Reboot oder
> Wochen Pause. Stand: 2026-11-03.

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
| 4 | Laufzeit 53×→32× RT (F-Reihe optimieren) | autonom (GPU-gebunden) | **IN ARBEIT 2026-11-03/04:** GPU-D3-Kette läuft (DiffWave→BigVGAN→FlashSR, Manifeste `docs/reports/current/train_job_d3_*.json`); §PERF-R18-Äquivalenz-Tests + Budget-Gate **21/21 grün** (2026-11-04, zuvor „nicht ausgeführt"), Ketten-Bugfixes M8/M9/c2-Cantus per Smoke verifiziert (Report `2026-11-04_sota_sitzung_ear_vae_messung_ketten_checks.md`); offen: nur noch Verifikation nach D3-Lauf; **RT-Messung 2026-11-05 ABGESCHLOSSEN: Median 32,19× RT (3 Läufe, Wall 7248,7–7259,2 s / 225,33 s), Baseline 53,9× — Lücke geschlossen** (Report `2026-10-28_f_reihe_laufzeit_optimierung.md` §9, Log `rt_messung_2026-11-04.log`; Hinweis: der geloggte rt_factor ist auf 32,0 gekappt, Spec §9.5; `RT_REALITY_MEASURED` = 32,2); ursprüngliches **RT-Mess-Protokoll 2026-11-04** (Report `2026-10-28_f_reihe_laufzeit_optimierung.md` §6: `scripts/_aurik_run_excellence.py` auf Referenz-Song „Testkünstlerin…“ (224 s), 3 Läufe/Training-frei, Baseline 53,9× RT — Bericht `2026-11-04_sitzung_w2_w3_b3_messung.md` §5) |
| 5 | ~~F3-BigVGAN-Rollout~~ **ENTSCHEIDEN 2026-10-30 (K0-C NO-GO, Hör-Instanz)** + ~~EAR-VAE v1↔v2-Vergleichsmessung~~ **GEMESSEN 2026-11-04:** HF-Gewichte `earlab/EAR_VAE2` (apache-2.0, `c87215f8…`) beschafft; Blind-Messung seed 42: **never_worsen ✓, 16/17 Trials v2-vorteilhaft, Sign-Test p = 0,000275 signifikant, erfuellt_objektiv ✓** (Messdatei `2026-11-03_ear_vae_v1_v2_vergleich.json`, Δ musik_mos +0,10 / Δ defektlast −0,12); **Rollout-Entscheidung v1↔v2: MENSCHLICH ENTSCIEDEN 2026-11-05 (NO-GO v2 — v1 bleibt; Hör-Instanz Block C, Sign-off Commit `7d2e3441`, `docs/reports/current/2026-11-05_hoer_signoff.md`; Wunsch-Richtung künftiger Iterationen: heller UND weniger Rauschen; D-K0-9/SOTA-M2, §V7 (copilot-instructions.md))** | — | LÜCKEN-MATRIX D.1b/c + K0-Sign-off-Paket + Report `2026-11-04_sota_sitzung_ear_vae_messung_ketten_checks.md` |
| 6 | Wohlklang-Stufen W-1…W-13 aktivieren (0/13 vollständig) | **B3-Verfahren** | **Stand 2026-11-04:** Alle B3-Beschlüsse W-1/W-4/W-5/W-6/W-7 formal **ABGELEHNT** (nicht signifikant, p ≥ 0,06; W-1 zusätzlich never_worsen verfehlt; W-1 hat widersprüchliche Objektiv-Belege — HF-Gain-Beschluss BESTÄTIGT vs. musik_mos-Beschluss ABGELEHNT ⇒ die Hör-Instanz entscheidet, Hörordnung §7); **Objektiv-Befund entscheidungsreif 2026-11-05** (`docs/reports/current/2026-11-05_k4_wohlklang_stufen_objektivierung.md`): W-2/W-3-Nachmessung ausgewertet (W-2: never_worsen ✓, p=0,6875 ⇒ objektiv ABGELEHNT; W-3 damals noch ohne Messung, gesperrt — Nachmessung am selben Tag abgeschlossen: 0/6 ⇒ ebenfalls ABGELEHNT (Details unten „B3-Nachmessung W-2/W-3 ABGESCHLOSSEN"), W-6/W-7 Fallbacks 11164/18220 → 364/2236 nach Router-Fix; B1-Zähler präzise (Commit `ef3795b7`); Hör-Sessions W-1/W-4/W-5 vorbereitet = **menschlich offen** (W-1 = einzige Stufe mit never_worsen=False + widersprüchlichen Belegen ⇒ §7-Hör-Entscheidung); W-2/W-3 gesperrt (Budget/Pfad) — **B3-Nachmessung W-2/W-3 ABGESCHLOSSEN 2026-11-05: beide ABGELEHNT** (W-2 `use_scnet_music`: 2/6 Vorteile, p = 0,6875; W-3 `use_sgmse_musik`: 0/6, Δ ≈ 0 — Schalter-Wirkung im Messpfad nicht nachweisbar, konsistent mit „Pfad gesperrt"; never_worsen ✓, B1 sauber mit 0 Rückfällen; `flag_freigabe.py` fail-closed: „Vorteil nicht signifikant"; kein Flag-Flip, Vertragssoll „AUS" bleibt; Belege `b3_stufenmessung_W-2/W-3.json` + `b3_objektivbeschluss_W-2/W-3.json`, Lauf `2026-11-05_b3_w2w3_nachmessung.log`, Report `2026-11-05_sitzung_w2w3_b3_nachmessung.md`; Mess-Lauf auf Korpus-Clips statt der Vertrags-Referenz-Songs — für künftigen Weg-B-Beschluss offen; Harness-Erweiterung Commit `d5483a43`; verbleibend: C4-Hörpaare + RT-Budget-Nachweis (W-2)); **Pfad-Klärung (W-3) ERLEDIGT 2026-11-06** — Ursache gefunden + behoben: Dauer-Achsen-Defekt in `phase_49_advanced_dereverb.py` (`shape[-1]` auf dem channels-last-Array ⇒ jedes Stereo-Fenster galt als „< 1 s“, SGMSE+-Tier-0 unerreichbar) — Δ ≈ 0 ist ein **No-Op-Artefakt der Messung**, der W-3-Beschluss 2026-11-05 damit als Qualitätsurteil über SGMSE+ **ungültig**; Fix + 3 Regressionstests (Rot→Grün), Report `2026-11-06_w3_pfadklaerung.md`; Neumessung nur über §4 mit Vertrags-Referenz-Songs, Flag bleibt AUS); „4-Modelle"-B3-Lauf (Material/EAR/Defekt/Loudness) offen **begründet** — Treiber ad-hoc, nicht rekonstruierbar (Report `2026-11-04_sitzung_transient_messlauf_b3_status.md` §6–7) |
| 7 | Modelle: WF-V4, TP-V2, BEATs-Head, MERT-Head, Gender-Head | teils Daten-blockiert | Tiefenanalyse E2 — **Stand 2026-11-06 (Trainingswelle):** Echt-Training vorbereitet für WF-V4 (D-K4-7, `scripts/prepare_wf4_warp_pairs.py` — MUSDB18-HQ-Mixture + bekannte Wow/Flutter-Warp-Kurven, GT warp_log2, Seed 42 (§G5), Manifest sha256) und TP-V2 (D-K4-8, `scripts/prepare_tp2_transient_pairs.py` — MUSDB18-HQ-Drums/other/bass-Stems, transiente-reiche Segmente); zusätzlich vorbereitet: F9 VoiceEncoder (D-K4-5), F10 SingMOS-Kalibrierung (D-K4-6), D-K2-3 vocos_48khz-Musik-Finetune. Alle als Jobs in `scripts/gpu_kette4_modellkoepfe.sh` (V1/V2 Datensynthese + R1–R6, pgrep-Guard, kein Flag-Flip, KEIN paralleles Training gestartet). **Köpfe (E2):** Gender-Head daten-blockiert (es fehlt `data/gender_labels.csv` — Gender-/Sänger-Labels auf Musik-Vocals, Lizenz/Label-Quelle offen, GTZAN/MagnaTagATune nur Genre+Valence/Arousal, kein Gender); BEATs-Head/MERT-Head daten-blockiert (527 Klassen AudioSet-Labels + Score-Kalibrierung auf HPI/AF fehlen); „Finetune der Köpfe“ bleibt Trainingspflicht (F13) — kein Trainer ohne Labels. Report `docs/reports/current/2026-11-06_trainingswelle_modelle.md` — **Fortsetzung 2026-11-06 (Manager-Ausführung):** drei **Phantom-Verdrahtungen** gefunden und behoben, die R1/R2 der Kette 4 am Musik-Training gehindert hätten (§III.11): (1) `train_wf4_warp_head.py` hatte **kein `--real-dir`** (nur synthetische Harmonische), (2) der R1-Guard prüfte `data/wf4_warp_pairs/degraded` statt `audio/` ⇒ R1 **immer** übersprungen, (3) `train_tp_v2_phase.py` ließ den Default `--synthetic 16` ein gesetztes `--real-dir` still überstimmen. Fix: **F0-Beobachtung** im Datensatz (`x_f0 = log2(f0_deg/f0_clean)` via Produktions-Tracker `pyin_fast`, keine GT-Zirkularität) + fail-closed-Lader (§V6, kein stiller Synthetik-Rückfall) + Vorrang real vor synthetisch + Kette-4-Verdrahtung; **Determinismus-Befund**: WAV-Container byte-instabil (libsndfile-`PEAK`-Chunk, 1 Byte bei Offset 61) ⇒ verbindlicher `sha256_pcm` statt `sha256_wav`. Daten erzeugt: **300 WF-V4-Paare** + **200 TP-V2-Segmente** (Seed 42, Manifest). Tests: 13 grün (`tests/unit/test_wf4_warp_real_pairs.py`, `tests/unit/test_train_tp_v2_real_dir_vorrang.py`). **R1/R2-Training weiterhin offen** (GPU belegt); **F9/F10/D-K2-3-Trainer implementiert 2026-11-06** (`scripts/train_voiceencoder_finetune.py`, `scripts/calibrate_singmos_music.py`, `scripts/train_vocos_48khz_finetune.py` — CLI exakt R3/R4/R5, Vertragstests 21 grün, vocos==0.1.0 im venv_aurik installiert, KEIN Flag-Flip) — R3/R4/R5 aktivieren sich im freien GPU-Fenster. Report `docs/reports/current/2026-11-06_wf4_echt_pfad_verdrahtung.md` |
| 8 | Hör-/Lizenz-Entscheidungen (§9/5–7, AERO, MUSHRA n≥30) | **menschlich** | Katalog A2 |
| 9 | ~~P1-3-Restdeckung (Audibility-Guards)~~ **ERLEDIGT 2026-11-03:** HNR/Mikrodynamik/Vibrato/Hallucination/Noise-Texture mit Maskierungs-Abdeckung (§G10), 19 Phasen verdrahtet (Commit `2c5e5522`) | — | Report `2026-11-03_p1_3_audibility_restdeckung.md` |
| 10 | ~~D-K1-3 Material-CRNN-Verdrahtung (95,3 % statt 40,2 %)~~ **ERLEDIGT 2026-11-03:** im Material-Entscheidungspfad hinter `use_material_crnn` (Default AUS — Flag-Flip offen, nur §4-Verfahren) (Commit `28f2947d`) | — | Report `2026-11-03_d_k1_3_material_crnn_wiring.md` |
| 11 | **Ketten-Verdrahtung (Trainingslücke, kein Modellproblem)** — **BEHOBEN 2026-11-06, GPU-Neulauf offen:** Drei Kette-1-Jobs liefen ins Leere, obwohl die Artefakte lokal liegen. (a) **M9/SOTA-M9** starb nach 5 s am 24-GiB-OOM (`--batch-size 4` ⇒ 4·513 Inter-Frame-LSTM-Sequenzen, 4,42 GiB angefragt) **und** trainierte auf `--synthetic 4000` (Sinus+Rauschen = K1-Domänen-Defizit §III.11) statt auf den vorhandenen `data/tfgrid_pairs` (25 MUSDB-Paare) — ein erfolgreicher synthetischer Lauf hätte den echten Kette-3-Lauf über Job-Lock/Manifest blockiert. Fix: echte Paare + `--batch-size 1 --grad-accum 4` (neues Trainer-Flag, Äquivalenz Batch 2 ⇔ Akkumulation 2 getestet; Kette 3 trägt dieselben Flags). (b) **Cantus/SOTA-M14** rief R3–R5 ohne `--data` auf (Exit 2 „--data fehlt“, 2 s), obwohl `data/cantus_pairs/manifest.jsonl` 3600 Paare hat — jetzt `--data` + fail-closed-Guard; `domain_adapt` nur mit echten `*_degraded.wav`-Paaren (bleibt datenblockiert). Tests: `tests/unit/test_gpu_kette1_wiring.py` (7 — inkl. statischer CLI-Prüfung ALLER vier Ketten gegen die `add_argument`-Flags der Trainer, 56 Aufrufe) + 2 Trainer-Tests. Offen: GPU-Neulauf, Never-worsen-Gate/Hörabnahme (M9), M9-Datenmenge (25 Paare, D-K2-5-Muster). **Kein Flag-Flip.** **Wiederaufnahme-Fix 2026-11-07:** `train_job_runner.py` überspringt Jobs mit `ergebnis: erfolgreich` bei identischem Befehl/Phase/Seed (Manifest bleibt unverändert; `--force` erzwingt Neulauf) — der nächste Kette-1-Neulauf überspringt M8/RT60 und startet direkt M9/Cantus. | autonom (GPU-gebunden) | Report `2026-11-06_ketten_verdrahtung_m9_m14.md` + `2026-11-07_train_job_runner_idempotent.md` |

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
- **E — RT-Budget-Ledger:** 32×-Budget allozierbar machen
  (`PerformanceGuard.get_budget_truth_report` existiert; D-K3-37 ist seit
  2026-11-07 geschlossen — offen bleibt die Allokation im Ledger).
- **F — Forensik-Monitor:** WP-3-Forensik + musik_mos/defektlast als
  Dauer-Wohlklang-Zeuge je Lauf (nur gerichtete Fälle zur Hör-Instanz).

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
