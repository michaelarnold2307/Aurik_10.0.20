# D-K3-88 — GPU-Wirksamkeitsnachweis Cantus-`fine_tune` (24-GiB-OOM-Fix)

**Datum:** 2026-11-15
**Status:** geschlossen (Evidenz-Nachweis; die Code- und Testseite war bereits am 2026-11-09 fertig)
**Kern:** Der 2×4-Fix hat den Full-Preset-`fine_tune` auf der 24-GiB-Karte
getragen: **30/30 Epochen, Exit 0, kein OOM**. Damit ist die letzte offene
Frage der Registerzeile beantwortet.

## 1. Lücke

D-K3-88 war nach dem Code-Fix vom 2026-11-09 weiterhin `in-arbeit`. Der
Blocker lautete: „GPU-Wirksamkeitsnachweis offen (nächster Kette-1-Lauf:
bleibt 2×4 in 24 GiB und laufen 30 Epochen durch?)“. Die beiden vorherigen
Versuche waren nach 55,1 s bzw. 30,6 s mit `torch.OutOfMemoryError` gestorben
(22,77 GiB belegt, 750 MiB angefragt, 280 MiB frei). Die Frage war damit
ausschließlich: Hält die neue Mikro-Batch-/Akkumulations-Taktung auf der
realen Karte durch?

## 2. Beleg (am Manifest und Trainer-Report gemessen)

### 2.1 Job-Manifest (getrackt)

`docs/reports/current/train_job_c2_cantus_finetune.json`
(SHA-256 `93b313d6c7a43d0e880023d8f63f42a086058cdaa11a8007252bf675cf14717b`,
3644 B):

- `schema: aurik.train_job/1`, `name: c2_cantus_finetune`, `phase: fine_tune`
- `ergebnis: erfolgreich`
- `versuche[0].exit_code: 0`, `versuche[0].timed_out: false`,
  `versuche[0].sekunden: 20159.7` (rund 5,6 h)
- `versuche[0].befehl` und `befehl`/`resume_befehl` tragen exakt
  `--epochs=30 --batch-size 2 --grad-accum 4 --singmos-loss`
- `checkpoint_hashes`:
  - `checkpoint_best.pt`: `25981eef425fa7688e31cc059f11c192299081395a4967c50bca33af90f872e9`
  - `checkpoint_latest.pt`: `c6132ce8b231e80db7a28b15cd14918d70dd657ed473281fd2e589b468f8957c`
  - `cantus_dit.onnx`: `ea2556b23dd511bb8e73eb231efd06bfdfd3128f4bf941f3550bdf5b6d7820cc`

### 2.2 Trainer-Report (lokales Ausgabeartefakt)

`models/cantus/train_report_fine_tune.json`
(SHA-256 `a6dff1c2a6a92f397fe4d9a73a5e0a9e0a08189bb95a70be6af878ba1fc2884e`,
726 B):

- `phase: fine_tune`, `preset: full`, `device: cuda`,
  `singmos_loss: true`
- `epochen_trainiert: 30`, `epochs_target: 30`
- `final_val_loss: 3.5380738750100136`
- `run_id: fine_tune-20261110T230120Z-596634`

`models/` ist per `.gitignore` nicht versioniert; der getrackte Beleg ist das
Manifest, der Trainer-Report ist das lokale Ausgabeartefakt desselben Laufs.
Beide stimmen in Phase, Befehl, Epochenzahl und Checkpoint-Hashes überein.

### 2.3 Kette

`scripts/gpu_kette1_welle_c.sh`, R4 (`--name c2_cantus_finetune`), Zeilen
66–68: sowohl `--cmd` als auch `--resume-cmd` tragen
`--batch-size 2 --grad-accum 4`; `--singmos-loss` bleibt aktiv. Die
effektive Batch-Größe 8 und die Rezeptur sind unverändert (Peak-VRAM /4).

## 3. Tests

Grün in dieser Sitzung (`.venv_aurik`, CPU):

```
PYTHONPATH=. .venv_aurik/bin/python -B -m pytest -q \
  tests/unit/test_gpu_kette1_wiring.py::test_cantus_finetune_haelt_das_vram_budget_ein \
  tests/unit/test_gpu_kette1_wiring.py::test_cantus_pretrain_und_finetune_uebergeben_das_paar_manifest
```

Ergebnis: **2 passed**. Der erste Test pinnt 2×4 in Erst- und Resume-Befehl
sowie die Akkumulations-Logik im Trainer; der zweite pinnt das
Paar-Manifest `--data` (die frühere R3/R4-Verdrahtungslücke).

**Ehrliche Abgrenzung:** Der in der alten Registerzeile zusätzlich zitierte
generische Test `tests/unit/test_tf_gridnet_denoiser.py::test_grad_accum_entspricht_grossem_batch`
(TF-GridNet, nicht Cantus) ist in der `.venv_aurik`-Umgebung mit
`FileNotFoundError` auf dem erwarteten Checkpoint rot. Er ist kein
Cantus-Beleg und wird hier **nicht** als grün behauptet; die
Cantus-spezifischen Vertragstests oben sind die maßgebliche Testseite.

## 4. Grenzen

- **Kein Flag-Flip, kein Deploy, keine Klangänderung.** Der Lauf ist ein
  Trainingsartefakt; eine Aktivierung bleibt §4/Hör-Instanz.
- **Kein neues GPU-Training in dieser Sitzung.** Der Beleg ist der bereits
  am 2026-11-10 abgeschlossene Lauf; hier wurde nur Evidenz gelesen und
  geprüft.
- Der Fix beseitigt den OOM-Blocker, nicht die Qualitätsfrage des
  Cantus-Modells; die bleibt an den jeweiligen §4-/Hör-Vertrag gebunden.

## 5. Reproduktion

GPU/ROCm (nicht in dieser Sitzung ausgeführt):

```bash
env PYTHONPATH=. <rocm-venv>/bin/python -B scripts/train_cantus.py \
  --phase=fine_tune --preset=full --singmos-loss --epochs=30 \
  --batch-size 2 --grad-accum 4 \
  --data data/cantus_pairs/manifest.jsonl \
  --resume models/cantus/checkpoint_latest.pt
```

Der Fortschritt wird über `docker`-freie Job-Manifeste
(`docs/reports/current/train_job_c2_cantus_finetune.json`) und den
Trainer-Report (`models/cantus/train_report_fine_tune.json`) geprüft.
