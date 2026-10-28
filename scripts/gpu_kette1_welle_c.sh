#!/bin/bash
# GPU-Trainingskette 1 — Welle C + SOTA-M8/M9 (host-sicher, nacheinander).
# Dauerhaftes Repo-Artefakt: Jede KI kann diesen Lauf jederzeit neu starten
# (bash scripts/gpu_kette1_welle_c.sh) — Job-Locks + Manifeste verhindern Doppel-
# laeufe; train_job_runner.py erlaubt Auto-Resume pro Job (siehe docs/SOTA_HANDOVER.md).
cd "$(dirname "$0")/.." || exit 1
PY=/home/michael/.local/share/aurik/venv_rocm72/bin/python
export PYTHONPATH=.
R() { python3 -B scripts/train_job_runner.py "$@"; echo "JOB_EXIT=$?"; }

echo "=== R1: SOTA-M8 A-SPADE Echt-Training (GPU) ==="
R --name m8_aspade --ckpt-dir models/aspade --phase sota_m8 \
  --cmd "env PYTHONPATH=. $PY -B scripts/train_aspade_declipper.py --musdb-root data/musdb18hq --epochs 30 --device cuda --out-dir models/aspade" \
  --resume-cmd "env PYTHONPATH=. $PY -B scripts/train_aspade_declipper.py --musdb-root data/musdb18hq --epochs 60 --device cuda --out-dir models/aspade --resume models/aspade/checkpoint_latest.pt" \
  --timeout-s 21600 --retries 1

echo "=== R2: SOTA-M9 TF-GridNet Echt-Training (GPU) ==="
R --name m9_tf_gridnet --ckpt-dir models/tf_gridnet --phase sota_m9 \
  --cmd "env PYTHONPATH=. $PY -B scripts/train_tf_gridnet_denoiser.py --epochs 30 --device cuda" \
  --resume-cmd "env PYTHONPATH=. $PY -B scripts/train_tf_gridnet_denoiser.py --epochs 30 --device cuda --resume models/tf_gridnet/checkpoint_latest.pt" \
  --timeout-s 21600 --retries 1

echo "=== R3: KP-8 Cantus pretrain (Full-Preset, GPU) ==="
R --name c2_cantus_pretrain --ckpt-dir models/cantus --phase pretrain \
  --cmd "env PYTHONPATH=. $PY -B scripts/train_cantus.py --phase=pretrain --preset=full --freeze-encoder=true --epochs=50" \
  --resume-cmd "env PYTHONPATH=. $PY -B scripts/train_cantus.py --phase=pretrain --preset=full --freeze-encoder=true --epochs=50 --resume models/cantus/checkpoint_latest.pt" \
  --timeout-s 43200 --retries 1

echo "=== R4: KP-8 Cantus fine_tune (Phase 2, GPU) ==="
R --name c2_cantus_finetune --ckpt-dir models/cantus --phase fine_tune \
  --cmd "env PYTHONPATH=. $PY -B scripts/train_cantus.py --phase=fine_tune --preset=full --singmos-loss=true --epochs=30 --resume models/cantus/checkpoint_latest.pt" \
  --resume-cmd "env PYTHONPATH=. $PY -B scripts/train_cantus.py --phase=fine_tune --preset=full --singmos-loss=true --epochs=30 --resume models/cantus/checkpoint_latest.pt" \
  --timeout-s 43200 --retries 1

echo "=== R5: KP-8 Cantus domain_adapt (Phase 3, GPU) ==="
R --name c2_cantus_domain --ckpt-dir models/cantus --phase domain_adapt \
  --cmd "env PYTHONPATH=. $PY -B scripts/train_cantus.py --phase=domain_adapt --preset=full --epochs=20 --resume models/cantus/checkpoint_latest.pt" \
  --resume-cmd "env PYTHONPATH=. $PY -B scripts/train_cantus.py --phase=domain_adapt --preset=full --epochs=20 --resume models/cantus/checkpoint_latest.pt" \
  --timeout-s 43200 --retries 1

echo "=== R6: KP-10 RT60-Regressor ==="
R --name c3_rt60 --ckpt-dir models/rt60_regressor --phase kp10 \
  --cmd "env PYTHONPATH=. $PY -B scripts/train_rt60_regressor.py --out models/rt60_regressor/rt60_regressor.joblib --manifest-out models/rt60_regressor/manifest.json" \
  --resume-cmd "env PYTHONPATH=. $PY -B scripts/train_rt60_regressor.py --out models/rt60_regressor/rt60_regressor.joblib --manifest-out models/rt60_regressor/manifest.json" \
  --timeout-s 14400 --retries 1

echo "=== R7: KP-11 Symphonia (GPU) ==="
SYMDATA=$(find data -name "*Instrumental-manifest.jsonl" 2>/dev/null | head -1)
if [ -n "$SYMDATA" ]; then
  R --name c4_symphonia --ckpt-dir models/symphonia --phase kp11 \
    --cmd "env PYTHONPATH=. $PY -B scripts/train_symphonia.py --data $SYMDATA --preset full --epochs 30" \
    --resume-cmd "env PYTHONPATH=. $PY -B scripts/train_symphonia.py --data $SYMDATA --preset full --epochs 30 --resume models/symphonia/checkpoint_latest.pt" \
    --timeout-s 43200 --retries 1
else
  echo "Symphonia-Manifest fehlt — R7 uebersprungen (datenblockiert)"
fi
echo "=== KETTE-1-ENDE ==="
