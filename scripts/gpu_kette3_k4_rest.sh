#!/bin/bash
# GPU-Trainingskette 3 — K4-Rest (SOTA-M8 A-SPADE + SOTA-M9 TF-GridNet Echt-Daten).
# Dauerhaftes Repo-Artefakt: jederzeit neu startbar (bash scripts/gpu_kette3_k4_rest.sh)
# — Job-Locks + Manifeste verhindern Doppellaeufe (train_job_runner.py, siehe
# docs/SOTA_HANDOVER.md §3). NICHT parallel zur D3-Kette starten (§5: keine
# parallelen Trainings) — kleine Modelle zuerst, dann M9.
#
# WICHTIG (K1-Vermeidung): M9 trainiert NUR mit echten Musik-Paaren
# (data/tfgrid_pairs/{clean,noisy} via scripts/prepare_tfgrid_pairs.py).
# --synthetic (Sinus + Rauschen) ist laut Docstring "zum Test" und würde ein
# K1-Defizit erzeugen (SOTA_DEFICIT_REGISTER Klasse K1) — deshalb hier bewusst
# NICHT verwendet.
cd "$(dirname "$0")/.." || exit 1
PY=/home/michael/.local/share/aurik/venv_rocm72/bin/python
export PYTHONPATH=.
R() { python3 -B scripts/train_job_runner.py "$@"; echo "JOB_EXIT=$?"; }

echo "=== R1: SOTA-M8 A-SPADE Echt-Training (Resume ab Epoche 4, MUSDB18-HQ) ==="
# Bedingtes Resume: bei vorhandenem Checkpoint fortsetzen (sonst ging der
# Stand Epoche 3 verloren), sonst frisch starten (§G5).
R --name m8_aspade --ckpt-dir models/aspade --phase sota_m8 \
  --cmd "bash -c \"if [ -f models/aspade/checkpoint_latest.pt ]; then exec env PYTHONPATH=. $PY -B scripts/train_aspade_declipper.py --musdb-root data/musdb18hq --epochs 60 --device cuda --out-dir models/aspade --resume models/aspade/checkpoint_latest.pt; else exec env PYTHONPATH=. $PY -B scripts/train_aspade_declipper.py --musdb-root data/musdb18hq --epochs 30 --device cuda --out-dir models/aspade; fi\"" \
  --resume-cmd "bash -c \"if [ -f models/aspade/checkpoint_latest.pt ]; then exec env PYTHONPATH=. $PY -B scripts/train_aspade_declipper.py --musdb-root data/musdb18hq --epochs 60 --device cuda --out-dir models/aspade --resume models/aspade/checkpoint_latest.pt; else exec env PYTHONPATH=. $PY -B scripts/train_aspade_declipper.py --musdb-root data/musdb18hq --epochs 30 --device cuda --out-dir models/aspade; fi\"" \
  --timeout-s 21600 --retries 1

echo "=== R2: SOTA-M9 TF-GridNet Echt-Training (Musik-Paare, KEIN --synthetic) ==="
# Speicherbudget 24 GiB (Befund 2026-11-06: OOM bei batch_size 4) — Micro-Batch 1
# + Gradient-Akkumulation 4 haelt die effektive Batch-Groesse 4 bei kleinerem Peak.
if [ -d data/tfgrid_pairs/clean ] && [ -d data/tfgrid_pairs/noisy ]; then
  R --name m9_tf_gridnet --ckpt-dir models/tf_gridnet --phase sota_m9 \
    --cmd "env PYTHONPATH=. $PY -B scripts/train_tf_gridnet_denoiser.py --epochs 30 --device cuda --clean-dir data/tfgrid_pairs/clean --noisy-dir data/tfgrid_pairs/noisy --batch-size 1 --grad-accum 4" \
    --resume-cmd "bash -c \"if [ -f models/tf_gridnet/checkpoint_latest.pt ]; then exec env PYTHONPATH=. $PY -B scripts/train_tf_gridnet_denoiser.py --epochs 30 --device cuda --clean-dir data/tfgrid_pairs/clean --noisy-dir data/tfgrid_pairs/noisy --batch-size 1 --grad-accum 4 --resume models/tf_gridnet/checkpoint_latest.pt; else exec env PYTHONPATH=. $PY -B scripts/train_tf_gridnet_denoiser.py --epochs 30 --device cuda --clean-dir data/tfgrid_pairs/clean --noisy-dir data/tfgrid_pairs/noisy --batch-size 1 --grad-accum 4; fi\"" \
    --timeout-s 21600 --retries 1
else
  echo "M9-Daten fehlen (data/tfgrid_pairs/{clean,noisy}) — R2 uebersprungen."
  echo "Datenaufbereitung: scripts/prepare_tfgrid_pairs.py (Report 2026-11-06_m9_tfgrid_echt_daten.md)."
fi
echo "=== KETTE-3-ENDE ==="
