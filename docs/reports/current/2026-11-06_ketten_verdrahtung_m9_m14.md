# Ketten-Verdrahtung GPU-Kette 1 — M9 (SOTA-M9) und Cantus (SOTA-M14)

**Datum:** 2026-11-06 · **Auslöser:** Pflicht-Check „open training rounds" nach
Session-Wechsel · **Status:** Fix umgesetzt, Tests grün, GPU-Neulauf steht aus
(GPU durch Kette 2 belegt) · **Kein Flag-Flip.**

## 1. Befund — drei Jobs liefen ins Leere

Die Job-Manifeste vom 2026-11-06 18:32–18:33 (Lauf der GPU-Kette 1) zeigen drei
Fehlschläge, die wie „Trainingsfehler" aussehen, aber Verdrahtungsfehler der
Kette sind:

| Job | Manifest | Dauer | Exit | Ursache |
| --- | --- | --- | --- | --- |
| `m9_tf_gridnet` (SOTA-M9/D-K4-2) | `train_job_m9_tf_gridnet.json` | 5,5 s / 4,2 s | 1 | `torch.OutOfMemoryError`: „Tried to allocate 4.42 GiB … 16.64 GiB is allocated" |
| `c2_cantus_pretrain` (KP-8/SOTA-M14) | `train_job_c2_cantus_pretrain.json` | 2,2 s / 2,2 s | 2 | `--data fehlt (Manifest-JSONL/Ordner)` |
| `c2_cantus_finetune` / `c2_cantus_domain` | analog | 2,3 s / 2,2 s | 2 | dito |

Belege zum Datenbestand (am Dateisystem gemessen):

- `data/cantus_pairs/manifest.jsonl` — **3600 Paare**, Audio unter
  `data/cantus_pairs/{train,test}`, `models/cantus/checkpoint_latest.pt`
  vorhanden (Pretrain-Stand 2026-10-27, best val 3,9013 laut Roadmap).
- `data/tfgrid_pairs/{clean,noisy}` — je **25** MUSDB18-HQ-Paare
  (`scripts/prepare_tfgrid_pairs.py`), dazu der Report
  `2026-11-06_m9_tfgrid_echt_daten.md`.

## 2. Ursachen (drei unabhängige Defekte)

**(a) M9-Speicherbudget.** Das Inter-Frame-BiLSTM verarbeitet `b·f` Sequenzen
der Länge `T = 752` (4-s-CHUNK, `CHUNK_SEC = 4.0`). Bei `--batch-size 4` sind
das 4 × 513 = 2052 Sequenzen; der LSTM-Workspace sprengt die 24 GiB der
Radeon 7900 XTX. Der Job hat also **nie** trainiert.

**(b) M9-Domäne.** Die Kette 1 trainierte M9 mit `--synthetic 4000`
(Sinus + Rauschen, laut Docstring „zum Test"), obwohl echte Musik-Paare
vorliegen und Kette 3 denselben Job bereits korrekt verdrahtet hat. Ein
erfolgreicher synthetischer Lauf hätte das Artefakt
`models/tf_gridnet/tf_gridnet_quality.onnx` erzeugt und den echten Lauf in
Kette 3 über den Job-Lock/Manifest-Status blockiert — ein K1-Domänen-Defizit
(§III.11 (copilot-instructions.md), `SOTA_DEFICIT_REGISTER`).

**(c) Cantus ohne `--data`.** `scripts/train_cantus.py` bricht ohne `--data`
fail-closed ab. Die Kette übergab den Parameter nie — dreimal hintereinander,
obwohl das Manifest lokal liegt (SOTA-M14). Zusätzlich hätte
`domain_adapt` synthetische Paare als „reale" Domänen-Adaptation verwendet,
wenn man `--data` einfach auf `cantus_pairs` gesetzt hätte.

## 3. Fix

- `scripts/train_tf_gridnet_denoiser.py`: neues `--grad-accum` (Default 1),
  Micro-Batch-Akkumulation im Trainingsloop, `--batch-size`-Default 4 → 1,
  `grad_accum` im Checkpoint dokumentiert, `0` wird mit `ValueError`
  abgelehnt (kein stiller „kein Optimizer-Schritt"-Pfad).
- `scripts/gpu_kette1_welle_c.sh`
  - R2: **echte** Musik-Paare (`data/tfgrid_pairs/{clean,noisy}`),
    `--batch-size 1 --grad-accum 4`, Guard „Daten fehlen ⇒ übersprungen";
    `--synthetic` ist raus.
  - R3/R4: `--data data/cantus_pairs/manifest.jsonl` (Auto-Erkennung), Guard
    `[ -n "$CANTUS_DATA" ]`.
  - R5 (`domain_adapt`): nur mit echten `*_degraded.wav`-Paaren
    (`data/cantus_real`, `data/real_degraded_vocals`), sonst begründetes
    Überspringen (§V7 (copilot-instructions.md)).
- `scripts/gpu_kette3_k4_rest.sh`: derselbe M9-Job trägt jetzt ebenfalls
  `--batch-size 1 --grad-accum 4` — sonst würde der neue Batch-Default 1 die
  effektive Batch-Größe des Kette-3-Laufs still von 4 auf 1 senken.
- Tests:
  - neu `tests/unit/test_gpu_kette1_wiring.py` (6 Tests): bash-Syntax,
    Speicherbudget in **jedem** M9-Aufruf inkl. Resume-Zweig, kein
    `--synthetic`, `--data` in R3/R4/R5 + fail-closed-Guards, Kette 3 mit
    denselben M9-Flags.
  - `tests/unit/test_tf_gridnet_denoiser.py` (+2): `grad_accum = 0` wird
    abgelehnt; **Äquivalenz** Batch 2 ⇔ Micro-Batch 1 × Akkumulation 2
    (Gewichte `allclose`, atol 1e-5) — die Akkumulation ist damit nicht nur
    „vorhanden", sondern nachweislich rechnerisch gleichwertig.

## 4. Verifikation

- `pytest tests/unit/test_gpu_kette1_wiring.py tests/unit/test_tf_gridnet_denoiser.py`
  → **32 passed** (27,1 s), inkl. der Äquivalenzprüfung.
- `bash -n scripts/gpu_kette1_welle_c.sh scripts/gpu_kette3_k4_rest.sh` → OK
  (im Test verankert).
- Alle Pre-Commit-Gates grün (u. a. Unit-Smoke, SOTA-Defizit-Register,
  Wohlklang-Vertrag, File-Lifecycle, Repo-Graph).

## 5. Offen (bewusst nicht überschritten)

- **GPU-Neulauf** von R2–R5: Kette 2 läuft (`d3_diffwave_vokal`, `train_job_runner`
  aktiv); kein paralleles Training (§5). Die Jobs starten beim nächsten
  Kettenlauf automatisch, sobald die GPU frei ist.
- **M9-Datenmenge:** 25 Paare sind statistisch dünn (D-K2-5-Muster); ein
  größerer `prepare_tfgrid_pairs.py`-Lauf bleibt eine Datenaufgabe, kein
  Verdrahtungsproblem.
- **Never-worsen-Gate + Hörabnahme** für M9 sind weiterhin offen (D-K4-2);
  ebenso `use_tf_gridnet_quality` bleibt **AUS** — kein Flag-Flip.
- **Cantus `domain_adapt`** bleibt datenblockiert (keine realen degradierten
  Vocal-Paare im Repo).
