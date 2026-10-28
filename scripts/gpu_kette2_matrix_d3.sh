#!/bin/bash
# GPU-Trainingskette 2 — LUECKEN-MATRIX D.3 (Finetune-Queue).
# Startet automatisch, sobald Kette 1 nicht mehr laeuft; jederzeit manuell
# startbar (bash scripts/gpu_kette2_matrix_d3.sh) — Doppel-Laeufe verhindern
# die Job-Locks (models/*/.train_job.lock).
cd "$(dirname "$0")/.." || exit 1
while pgrep -f "gpu_kette1_welle_c\.sh" >/dev/null; do echo "warte auf Kette 1..."; sleep 600; done
PY=/home/michael/.local/share/aurik/venv_rocm72/bin/python
export PYTHONPATH=.
R() { python3 -B scripts/train_job_runner.py "$@"; echo "JOB_EXIT=$?"; }

echo "=== D3-1: DiffWave-Vokal-Inpaint (S1 -1,7 dB Beleg) ==="
R --name d3_diffwave_vokal --ckpt-dir models/diffwave_vocal_inpaint --phase matrix_d3 \
  --cmd "env PYTHONPATH=. $PY -B scripts/train_diffwave_vocal_inpaint.py --epochs 30" \
  --resume-cmd "env PYTHONPATH=. $PY -B scripts/train_diffwave_vocal_inpaint.py --epochs 30" \
  --timeout-s 21600 --retries 1

echo "=== D3-2: BigVGAN-v2 Musik+Vokal ==="
R --name d3_bigvgan_f3 --ckpt-dir models/bigvgan --phase matrix_d3 \
  --cmd "env PYTHONPATH=. $PY -B scripts/train_bigvgan_f3.py --epochs 30" \
  --resume-cmd "env PYTHONPATH=. $PY -B scripts/train_bigvgan_f3.py --epochs 30" \
  --timeout-s 21600 --retries 1

echo "=== D3-3: FlashSR-Musik-Finetune (out-of-domain -21..-24 dB) ==="
R --name d3_flashsr_musik --ckpt-dir models/flashsr --phase matrix_d3 \
  --cmd "env PYTHONPATH=. $PY -B scripts/train_flashsr_f4.py --epochs 30 --data-dir data/musdb18hq --out-dir models/flashsr" \
  --resume-cmd "env PYTHONPATH=. $PY -B scripts/train_flashsr_f4.py --epochs 30 --data-dir data/musdb18hq --out-dir models/flashsr --resume models/flashsr/checkpoint_latest.pt" \
  --timeout-s 21600 --retries 1

echo "=== KETTE-2-ENDE (DDSP bedingt kein Skill; BEATs-Head/Gender-Head datenblockiert) ==="
