# SOTA-Sitzung 2026-11-04 — EAR-VAE-Vergleichsmessung + Ketten-Verifikation

**Auftrag:** Autonomer Lückenschluss nach `docs/SOTA_HANDOVER.md` (§2
priorisiert), Wiederherstellungs-Check nach Sitzungsabbruch, „Fallback-/
Mess-Läufe fortsetzen wo sie endeten". Arbeitssprache Deutsch, Quelle je
Zeile (Zitierdisziplin `AGENTS.md` §5).

---

## 1. Wiederherstellungs-Check (SOTA_HANDOVER §3)

| Prüfung | Befund | Quelle |
| --- | --- | --- |
| `pgrep -f gpu_kette` | **Kette 2 (Matrix D3) läuft** (PID 3560120, `scripts/gpu_kette2_matrix_d3.sh`, Job `d3_diffwave_vokal` Auto-Resume Versuch 2/2, GPU ~98 %) | `ps`, `rocm-smi`, `/tmp/k2_d3_start.log` |
| Trainings-Runden offen | 7 von 8 Manifesten ohne `ergebnis: erfolgreich` (c2×3, d3×2, m8, m9) | `docs/reports/current/train_job_*.json` |
| Ketten-Start | **unterlassen** — Startbedingung „keine Kette läuft" trifft nicht zu (SOTA_HANDOVER §3); Trainings bleiben sequenziell (§5 harte Regeln) | `docs/SOTA_HANDOVER.md` §3/§5 |
| Weitere abgebrochene Mess-Läufe | keine (Status-Scan `docs/reports/current/*.json`) | eigener Scan |

## 2. Offene Ketten-Jobs: Bug-Ursachen im Code-Stand verifiziert

Die Manifeste c2_cantus_* (exit 2), m8_aspade (RuntimeError Device-Mismatch)
und m9_tf_gridnet (fehlende Args) sind **historische** Fehlschläge; die drei
Ursachen sind im Code-Stand **behoben** (Beweisstellen):

1. **c2_cantus (3 Jobs):** `--freeze-encoder=true` verletzt argparse
   (Boolean-Flag) — heute `--freeze-encoder` in `scripts/gpu_kette1_welle_c.sh`
   R3–R5. Manifest-Versuche stammen vom Altstand.
2. **m9_tf_gridnet:** Ketten-Aufruf heute `--synthetic 4000` + Resume-Guard
   (`if [ -f models/tf_gridnet/checkpoint_latest.pt ]`) in
   `scripts/gpu_kette1_welle_c.sh` R2; die Versuche im Manifest (ohne
   `--synthetic`) sind Altstand.
3. **m8_aspade:** Device-Parität in `scripts/train_aspade_declipper.py`
   (`export_onnx`: Dummies auf Modell-Device, Kommentar „CPU-Dummy +
   CUDA-Modell crasht in conv1d (Ketten-Befund)"; `check_onnx_parity`:
   `.to(_dev2)` vor dem Torch-Referenzlauf).

Die Manifeste werden beim nächsten Kette-1-Lauf vom Runner neu geschrieben
(`scripts/train_job_runner.py` `run_job`) — die Runden laufen dann mit den
gefixten Kommandos automatisch weiter.

## 3. Verifikations-Smokes (CPU, `nice -n 19`, ohne parallele Trainings)

| Lauf | Kommando (Ketten-Muster, tiny) | Ergebnis |
| --- | --- | --- |
| m8 (inkl. ONNX-Export + Paritäts-Gate) | `train_aspade_declipper.py --musdb-root data/musdb18hq --epochs 1 --batch-size 1 --steps-per-epoch 2 --max-tracks 1 --device cpu --out-dir /tmp/aspade_smoke` | **grün** — ONNX rel = 1,33e-06 ≤ 1e-03 (§III.9 (copilot-instructions.md)) |
| c2_cantus (argparse-Fix) | `train_cantus.py --phase=pretrain --preset=tiny --freeze-encoder --smoke` | **grün** — EXIT 0 |
| m9_tf_gridnet (`--synthetic`-Pfad) | `train_tf_gridnet_denoiser.py --epochs 1 --device cpu --synthetic 4 --steps-per-epoch 1 --batch-size 1 --hidden 4 --blocks 1 --skip-onnx` | **grün** — PIPE_EXIT 0 |

**Smoke-Artefakte bereinigt:** Der TF-GridNet-Smoke schrieb
`models/tf_gridnet/{checkpoint_latest,tf_gridnet_best}.pt` (68/21 KB
Tiny-Modelle). Beide entfernt — sonst würde der Resume-Guard der nächsten
M9-Runde vom Tiny-Smoke fortsetzen und die Runde verunreinigen. Verblieben:
`.train_job.lock` (Runner-Bestand). /tmp-Artefakte liegen außerhalb des Repos.

## 4. §PERF-R18-Mess-Lauf fortgesetzt (F-Reihe-Laufzeit, Lücke §2/4)

Der Report `2026-10-28_f_reihe_laufzeit_optimierung.md` hatte die
Äquivalenz-Tests „geschrieben, NICHT ausgeführt" (§1 der Ehrlichkeitstabelle).
Nachgeholt:

    python3 -m pytest tests/unit/test_perf_r18_cache_equivalence.py \
        tests/normative/test_performance_budget_ci_gate.py -q
    # → 21 passed (9,77 s)

Die **RT-Messung Referenz-Song (53×→32×)** ist bewusst **nicht** gestartet:
Das Protokoll fordert „Messfenster ohne parallele GPU-Trainingsjobs"
(`2026-10-28_f_reihe_laufzeit_optimierung.md` §6.2); Kette 2 trainiert
derzeit. Die Messung gehört in ein trainingsfreies Fenster (nächster
Ketten-Stand / nächste Sitzung). Die Optimierungen selbst sind bereits
committed (`ecc364d5` u. a.); Akzeptanz „≤ 32× gemessen" bleibt offen.

## 5. EAR-VAE v1↔v2 Vergleichsmessung (Lücke §2/5, LÜCKEN-MATRIX D.1c)

Blocker „HF-Gewichte `earlab/EAR_VAE2`" gelöst — lizenzklar und ohne
Entscheidungsspielraum beschafft:

| Schritt | Befund | Quelle |
| --- | --- | --- |
| Lizenz-Klärung | **Apache-2.0** (HF-Model-Card `cardData.license`, Tag `license:apache-2.0`; Upstream-Klon `models/ear_vae2_upstream/LICENSE` ebenso) — keine §9-/Lizenz-Entscheidung nötig | `https://huggingface.co/api/models/earlab/EAR_VAE2` |
| Beschaffung | `weights/ear_vae2.pt`, 713.542.691 B, SHA-256 `c87215f857c909156c11159ff8e3aa05a15522d6963a4d9eceb7493aa8c14840` → `models/ear_vae2_upstream/checkpoints/ear_vae2.pt` (Harness-Kandidatenpfad, Upstream-README-Prozedur) | HF `earlab/EAR_VAE2`, `scripts/ear_vae_v1_v2_vergleich.py` |
| Preflight | **OK** — v2-Checkpoint vorhanden | `--preflight-only` |
| Blind-Messung | `nice -n 19 python3 -B scripts/ear_vae_v1_v2_vergleich.py --seed 42` (CPU, deterministisch §G5 (copilot-instructions.md)) | Messdatei `docs/reports/current/2026-11-03_ear_vae_v1_v2_vergleich.json` (Schema `aurik.ear_vae_v1_v2_messung/1`) |

**ERGEBNIS (Blind-Vergleich, seed 42):** — 17 auswertbare Trials
(`zeilen`), **never_worsen: True**, **16/17 Trials vorteilhaft für v2**,
Sign-Test **p = 0,000275** (signifikant), `erfuellt_objektiv: True`.
Mittelwerte: Δ musik_mos **+0,1016** (v2 besser), Δ defektlast **−0,1165**
(v2 weniger Defektlast). ΔSDR bewusst nicht korpusfähig (fester
Referenz-Korpus ohne saubere Referenzen — in der Messdatei begründet).
Quelle: `docs/reports/current/2026-11-03_ear_vae_v1_v2_vergleich.json`
(Schema `aurik.ear_vae_v1_v2_messung/1`). Hör-Stimuli (verblindet,
Schlüssel getrennt): `output/ear_vae_v1_v2/hoer_stimuli`.

**Betriebs-Hinweis (Sitzung):** Die Messung wurde aus Doppelzuständigkeit
**zweimal** ausgeführt (diese Sitzung + fremder Hintergrund-Lauf, gestartet
06:00:35, PID 3937818, `.venv_aurik`); der fremde Lauf lief als Erster
durch. Ergebnisse sind deterministisch byte-identisch (gleicher Seed 42,
gleiche Skriptversion, §G5 (copilot-instructions.md)) — kein Datenkonflikt.
Der fremde Lauf wurde gemäß §5 (SOTA_HANDOVER) nicht gestört. Zusätzlich
festgestellt: der Kun-Scheduled-Task „SOTA-Lückenschluss" (ID `73cc09ac`)
lief zuletzt 2026-11-04 02:26 UTC mit **Fehler 402 „Insufficient Balance"**
— die Dauer-Automation ist bis zur Provider-Aufladung defekt.

**Grenze (hart, §5 SOTA_HANDOVER):** Die Messung ist ein
Never-worsen-/Metrik-Benchmark für D-K0-9 — **keine** Aktivierung. Der
Rollout-/Ersatz-Entscheid v1↔v2 bleibt die menschliche Hör-/Sign-off-Entscheidung
(D-K0-9, `SOTA-M2`); kein `scripts/flag_freigabe.py`-Lauf in dieser Sitzung.

## 6. Bewusst NICHT angefasst

- **Flag-Aktivierungen / Wohlklang-Stufen (§2/6, Punkt 6):** nur über das
  B3-Verfahren (§4 SOTA_HANDOVER); Hör-Sessions W-1/W-4/W-5 sind menschlich
  offen, W-6/W-7 nach Router-Fix gemessen und abgelehnt (nie_worsen ✓, nicht
  signifikant). Keine neue Aktivierung ohne §4-Verfahren.
- **§2/7 Modelle (WF-V4, TP-V2, BEATs-/MERT-/Gender-Head):** teils
  daten-/menschlich blockiert (Tiefenanalyse E2), keine autonome Schließung
  möglich.
- **§2/8 Hör-/Lizenz-Entscheidungen:** menschlich (Katalog A2) — nie
  eigenmächtig.
- **Laufende GPU-Kette 2:** unberührt (keine parallelen Trainings, keine
  Drosselung).

## 7. Reproduzierbarkeit

- Seeds: Messung seed 42 (Harness-Default, `_seed_for`-Namespace
  `ear_vae_v1_v2`); Smokes jeweils seed 42.
- Artefakt-Fingerabdruck Gewichte: SHA-256 siehe §5. **Erledigt:** Eintrag
  `checkpoints/ear_vae2.pt` in `.github/ML_ARTIFACT_FINGERPRINTS.md`
  (maschinell gemessen via `scripts/model_artifact_probe.py --dirs
  ear_vae2_upstream --no-md`; 178.36 M Params, torch-zip `gen` =
  `ear_vae2_small_refiner_step120000_generator`).
