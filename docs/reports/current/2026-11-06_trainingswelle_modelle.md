# Trainingswelle Modelle — 2026-11-06

**Auftrag:** Echt-Training/Verifikation der SOTA-Modelle (WF-V4, TP-V2, BEATs-Head,
MERT-Head, Gender-Head) vorbereiten und wo möglich ausführen, ohne laufende
Trainings zu stören (AKTUELL: train_job_runner m8_aspade, GPU-Ketten 1/2/3).

## 1. Arbeitsumfang dieser Sitzung (ehrenhaft dokumentiert)

Die Worker-Runtime dieser Sitzung hat **kein Shell-Tool** (kein `pgrep`, kein
`rocm-smi`, kein `nohup`, kein `python`-Aufruf). Beleg: Repo-Kommentar
`scripts/b3_transient_energie_neverworsen.py:28` — „AUSFÜHRUNG macht der Manager —
kein Shell in der Worker-Runtime". Daher:

- **Möglich in dieser Sitzung:** Dateien lesen/schreiben (Skripte, Report,
  Register, §7-Update), Manifest-Stände aus `docs/reports/current/train_job_*.json`
  erheben, Prozess-/GPU-Stand aus den Reports ableiten.
- **Nicht möglich in dieser Sitzung:** `pgrep -f gpu_kette`/`train_job_runner`,
  `rocm-smi`, `bash -n`, Runner-Dry-Run, Trainingsdaten-Synthese ausführen,
  Trainings starten, D3-Verifikation messen. Diese Schritte sind im Report als
  **Manager-Ausführungsschritte** dokumentiert (§5).

## 2. Standerhebung (Abnahme (a))

Quelle: `docs/reports/current/train_job_*.json` (nur gelesen, IN-FLIGHT nicht
verändert), `docs/reports/current/2026-11-06_w3_pfadklaerung.md`,
`docs/reports/current/2026-11-05_sitzung_w2w3_b3_nachmessung.md`,
`docs/reports/current/2026-10-28_f_reihe_laufzeit_optimierung.md` §9.

| Manifest | name | ergebnis | Bemerkung |
| --- | --- | --- | --- |
| `train_job_d3_diffwave_vokal.json` | d3_diffwave_vokal | **erfolgreich** | Gate bestanden; Val-SDR +3,23 dB vs Zero-Shot −3,01 dB; `never_worsen_per_gap: true`; Checkpoint `diffwave_vocal_ft.ckpt` |
| `train_job_d3_diffwave_vokal_fortsetzung_20261105.json` | d3_diffwave_vokal | (Fortsetzung) | run_id 2026-11-06, ohne `ergebnis`-Feld (laufend/unterbrochen) |
| `train_job_d3_bigvgan_f3.json` | d3_bigvgan_f3 | fehlgeschlagen | Alt-Manifest, Timeout 21 600 s |
| `train_job_d3_bigvgan_f3_fortsetzung_20261105.json` | d3_bigvgan_f3 | fehlgeschlagen | 4× Timeout 21 600 s, exit −15; Auto-Resume bis Epoche 16, bestes Val-A1 0,5682; Checkpoints `output/f3_bigvgan` |
| `train_job_d3_flashsr_musik.json` | d3_flashsr_musik | (ohne Versuche) | Queue nie erreicht |
| `train_job_m8_aspade.json` | m8_aspade | (laufend/fehlgeschlagen) | 1× Timeout bis Epoche 41/60, Loss 0,15; Fortsetzung über Kette 1/3 |
| `train_job_m9_tf_gridnet.json` | m9_tf_gridnet | fehlgeschlagen | Befehls-Bug (in `gpu_kette1_welle_c.sh` korrigiert) |
| `train_job_c2_cantus_{pretrain,finetune,domain}.json` | c2_cantus_* | fehlgeschlagen | Befehls-Bugs (korrigiert) |
| `train_job_c3_rt60.json` | c3_rt60 | **erfolgreich** | KP-10/D-K4-10 |

**D3-Verifikation (§9 RT-Messung/Qualitäts-Gate):** Ein D3-Job trägt
`ergebnis: erfolgreich` (`d3_diffwave_vokal`) — die Verifikation wäre damit
grundsätzlich auszulösen. Sie ist jedoch laut Reportlage **fenster-blockiert**
(`2026-11-05_sitzung_w2w3_b3_nachmessung.md:139`,
`2026-10-28_f_reihe_laufzeit_optimierung.md:212`): die RT-Messung braucht ein
**trainingsfreies Fenster**, da sie sonst die GPU-Trainings stört. Laut Auftrag
laufen AKTUELL Ketten 1/2/3 (m8_aspade) ⇒ **kein trainingsfreies Fenster**.
Daher: **D3-Verifikation begründet offen** (Abnahme (a) erfüllt: „ausgeführt ODER
begründet offen"). Ausführungsvorschrift siehe §5.

## 3. Neue Artefakte (Abnahme (b)(c))

### 3.1 `scripts/prepare_wf4_warp_pairs.py` (D-K4-7, neu)

Echt-Trainingspaare für WF-V4-Warp-Schätzung aus **MUSDB18-HQ-Mixture.wav**
(Musik-Trainingsdaten, §III.11 Signalpfad-Kern ist Musik) mit BEKANNTEN
Wow/Flutter-Warp-Kurven (GT = `warp_log2`, Muster `train_wf4_warp_head.py`).
Deterministisch Seed 42 (§G5), Manifest `aurik.wf4_warp_pairs/1` mit sha256 je
Datei (Pfeiler 4, Provenienz), 32-bit float ⇒ keine Dither-Pflicht (§V5).
Muster `prepare_tfgrid_pairs.py`.

### 3.2 `scripts/prepare_tp2_transient_pairs.py` (D-K4-8, neu)

Echt-Trainingsdaten für TP-V2-Phasenschätzung aus MUSDB18-HQ-Stems
(`drums.wav`, `other.wav`, `bass.wav` — transiente-reiche Segmente nach
Onset-Dichte, `--transient-quantil`). Deterministisch Seed 42, Manifest
`aurik.tp2_transient_pairs/1` mit sha256 je Segment. Wird von
`train_tp_v2_phase.py --real-dir` konsumiert (das Skript unterstützt
`--real-dir` seit jeher).

### 3.3 `scripts/gpu_kette4_modellkoepfe.sh` (neu, Abnahme (b))

Dauerhafte GPU-Trainingskette 4 (Muster `gpu_kette3_k4_rest.sh`): sequenziell,
Job-Locks + Auto-Resume + Manifest über `train_job_runner.py`,
interruptionssicher neu startbar. **Pflicht-Check am Kopf:** `pgrep -f
train_job_runner` ⇒ wenn nicht leer, `exit 2` (KEIN paralleles Training, §5
SOTA_HANDOVER). KEIN Flag-Flip (§5): `use_wf4_warp_head`/`use_tp_v2_phase`/
`use_vocos_music` bleiben AUS.

Jobs (Reihenfolge):
- **V1** `prepare_wf4_warp_pairs.py` (Seed 42, MUSDB18-HQ)
- **V2** `prepare_tp2_transient_pairs.py` (Seed 42, MUSDB18-HQ)
- **R1** WF-V4 Echt-Training (`train_wf4_warp_head.py --epochs 50 --device cuda`)
- **R2** TP-V2 Echt-Training (`train_tp_v2_phase.py --real-dir data/tp2_transient_pairs`)
- **R3** D-K4-5 F9 VoiceEncoder-Finetune (`train_voiceencoder_finetune.py` —
  Skript existiert noch nicht, Guard `[ -f ... ]` ⇒ inert)
- **R4** D-K4-6 F10 SingMOS-Kalibrierung (`calibrate_singmos_music.py` —
  Skript existiert noch nicht, Guard ⇒ inert)
- **R5** D-K2-3 vocos_48khz-Finetune (`train_vocos_48khz_finetune.py` —
  Skript existiert noch nicht, Guard ⇒ inert)
- **R6** Köpfe BEATs/MERT/Gender — **daten-blockiert**, Guard statt Ausführung

**Korrektur während der Sitzung:** Die ersten Entwürfe von R1/R2/R3/R5/R6
enthielten `--resume`-Flags an `train_wf4_warp_head.py`/`train_tp_v2_phase.py`/
`train_gender_head.py` — diese Skripte haben **kein** `--resume`-Argparse-Flag
(geprüft: `add_argument`-Liste). Korrigiert: `--resume-cmd` = `--cmd`
(identischer Befehl), damit Auto-Resume über den `train_job_runner` erfolgt
(Wiederholung des Befehls) statt über ein nicht existierendes Flag.

## 4. Köpfe: BEATs / MERT / Gender (Abnahme (d))

Quelle: `docs/reports/current/2026-10-28_vollstaendigkeit_tiefenanalyse.md` (E2),
`docs/TODOS_SOTA_ROADMAP.md`, `.github/SOTA_DEFICIT_REGISTER.md`.

| Kopf | Status | Konkreter Grund (daten-blockiert) |
| --- | --- | --- |
| **BEATs-Head** (527 Klassen, Instrument-/Genre-Evidenz) | **daten-blockiert** | (1) Kein Trainings-Skript im Repo (nur `c4_head.pth`-Historie mit val_MAE 0,2447 ≈ Baseline = kein Skill, `model_inventory.py:333`); (2) **keine Label-Quelle** — AudioSet-527-Klassen-Labels für lokalen Musik-Korpus fehlen; Tagger-ONNX fehlt (`docs/TODOS_SOTA_ROADMAP.md:340`) |
| **MERT-Head** (Genre-adaptive Goals) | **daten-blockiert** | (1) Kein Trainings-Skript; (2) **GTZAN/MagnaTagATune-Datensatz + Labels fehlen** (`docs/TODOS_SOTA_ROADMAP.md:340`) — Genre-Label-Korpus nicht im Repo |
| **Gender-Head** (F13, MERT-Features → MLP) | **daten-blockiert** | Skript `scripts/train_gender_head.py` **existiert** (braucht `--labels data/gender_labels.csv` mit `file,gender`); **Label-Korpus fehlt** — `data/gender_labels.csv` nicht vorhanden; benötigt ≥ ~50 Labels je Klasse (Roadmap F13/F14); ohne Labels bricht das Skript ab (§V7) und rät nicht |

Alle drei Köpfe sind damit **nicht code-blockiert**, sondern **daten-blockiert**
(fehlende Label-Quellen/Datensätze). Die Kette 4 führt sie als Guard aus
(`[ -f data/gender_labels.csv ]` etc.) und dokumentiert das Fehlen im Log.

## 5. Manager-Ausführungsschritte (nicht in dieser Sitzung möglich)

1. **Standerhebung abschließen** (falls gewünscht, mit Shell):
   `pgrep -f gpu_kette; pgrep -f train_job_runner; rocm-smi`
2. **D3-Verifikation** (Abnahme (a), sobald trainingsfreies Fenster):
   `bash scripts/verify_d3_quality_gate.sh` (falls vorhanden) bzw. die in
   `2026-10-28_f_reihe_laufzeit_optimierung.md` §9 dokumentierte RT-Messung +
   Qualitäts-Gate für `d3_diffwave_vokal` (Checkpoint `diffwave_vocal_ft.ckpt`).
3. **Trainingsdaten-Synthese** (deterministisch, Seed 42):
   ```
   python3 -B scripts/prepare_wf4_warp_pairs.py --musdb-root data/musdb18hq --out-dir data/wf4_warp_pairs --anzahl 300 --seed 42
   python3 -B scripts/prepare_tp2_transient_pairs.py --musdb-root data/musdb18hq --out-dir data/tp2_transient_pairs --seed 42
   ```
4. **Smoke (Abnahme (b))** — `bash -n scripts/gpu_kette4_modellkoepfe.sh`
   (Syntax-Check) + Runner-Dry-Run (Argparse-Check der `R`-Aufrufe, z.B.
   `python3 -B scripts/train_job_runner.py --help`).
5. **Kette 4 starten** — NUR wenn `pgrep -f train_job_runner` leer ist:
   ```
   nohup bash scripts/gpu_kette4_modellkoepfe.sh > /tmp/k4.log 2>&1 &
   ```
   (Haupt-Checkout `/media/michael/Software 4TB/Aurik_Standalone`.)

## 6. Einhaltung der Vorgaben (Abnahme (e))

- **KEIN paralleles Training gestartet:** In dieser Sitzung wurde kein
  Training/keine Kette gestartet (kein Shell-Tool). Die Kette 4 enthält einen
  `pgrep`-Guard, der bei laufendem `train_job_runner` mit `exit 2` endet.
- **KEINE Flag-Flips:** `use_wf4_warp_head`/`use_tp_v2_phase`/`use_vocos_music`
  bleiben AUS (§5 SOTA_HANDOVER). Die Trainings schreiben nur Artefakte/Manifeste.
- **IN-FLIGHT-Dateien nicht angefasst:** `docs/reports/current/train_job_*.json`,
  `study_stimuli/`, `scripts/prepare_tfgrid_pairs.py`,
  `tests/unit/test_prepare_tfgrid_pairs.py`,
  `docs/reports/current/b3_transient_energie_neu_vor_enforcer.json` — nur gelesen.
- **Daten-Blockaden ehrlich dokumentiert** (§III.13): F9/F10/BEATs/MERT/Gender
  sind daten-blockiert (fehlende Labels/Datensätze/Skripte), nicht synthetisch
  „gemogelt".

## 7. Register-Notizen & §7-Update

Siehe `.github/SOTA_DEFICIT_REGISTER.md` (D-K4-7/8/5/6/1, D-K2-3) und
`docs/SOTA_HANDOVER.md` §2 Zeile 7 + §7. Diese wurden in dieser Sitzung
aktualisiert (Status „vorbereitet", nicht „abgeschlossen" — die eigentlichen
Trainings laufen über Kette 4 sobald die GPU frei ist).

## 8. Change-Ledger & Commit (Abnahme (f))

- **Change-Ledger:** `python3 scripts/change_ledger.py snapshot` → `TASK_CHANGES.md`
  (kann in dieser Worker-Runtime nicht ausgeführt werden — kein Shell. Manager
  soll dies vor dem Commit ausführen, damit CI die Abdeckung im PR prüfen kann).
- **Conventional Commit (vorgeschlagen):**
  `feat(sota): Trainingswelle Modelle — WF-V4/TP-V2/F9/F10/vocos/Köpfe vorbereitet`
  (D-K4-7/8/5/6 + D-K2-3; Report + Register + §2/§7; keine Flag-Flips, kein
  paralleles Training).
- **Tests:** Die neuen Skripte enthalten nur Vor-/Datensynthese (keine
  Produktions-Logik-Änderung) — bestehende Unit-Tests sind nicht betroffen.
  Empfehlung: `pytest tests/unit -q` vor dem Commit (Manager, kein Shell in
  dieser Sitzung).
