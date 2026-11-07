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
  --resume-cmd "env PYTHONPATH=. $PY -B scripts/train_diffwave_vocal_inpaint.py --epochs 30 --resume models/diffwave/checkpoint_latest.pt" \
  --timeout-s 21600 --retries 1

echo "=== D3-2: BigVGAN-v2 Musik+Vokal ==="
# Resume-Pfad = output/f3_bigvgan (Default --out-dir von train_bigvgan_f3.py);
# models/bigvgan enthaelt nur die vortrainierten Artefakte, KEINE Trainings-
# checkpoints — der alte Pfad startete den Lauf bei jedem Start von vorn
# (Korrektur 2026-11-06, Beleg: Manifest
# docs/reports/current/train_job_d3_bigvgan_f3_fortsetzung_20261105.json:
# 4x Timeout-Lauf bis Epoche 16 mit Resume auf output/f3_bigvgan).
R --name d3_bigvgan_f3 --ckpt-dir output/f3_bigvgan --phase matrix_d3 \
  --cmd "env PYTHONPATH=. $PY -B scripts/train_bigvgan_f3.py --epochs 30" \
  --resume-cmd "env PYTHONPATH=. $PY -B scripts/train_bigvgan_f3.py --epochs 30 --resume output/f3_bigvgan/checkpoint_latest.pt" \
  --timeout-s 21600 --retries 1

echo "=== D3-3: FlashSR-Musik-Finetune (out-of-domain -21..-24 dB) ==="
# Resume-Anker = <out-dir>/checkpoint_latest.pt. Der Trainer schreibt ihn seit
# 2026-11-08 nach jeder Epoche; vorher existierte er NIE, und Versuch 2 starb
# 61 s nach dem 6-h-Timeout in einen harten FileNotFoundError (Manifest
# docs/reports/current/train_job_d3_flashsr_musik.json) — 6 h GPU-Zeit ohne
# Resume. Beide Befehle sind deshalb datei-geschuetzt (Muster wie A-SPADE in
# Kette 3): mit Anker wird fortgesetzt, ohne Anker frisch gestartet.
R --name d3_flashsr_musik --ckpt-dir models/flashsr --phase matrix_d3 \
  --cmd "bash -c \"if [ -f models/flashsr/checkpoint_latest.pt ]; then exec env PYTHONPATH=. $PY -B scripts/train_flashsr_f4.py --epochs 30 --data-dir data/musdb18hq --out-dir models/flashsr --resume models/flashsr/checkpoint_latest.pt; else exec env PYTHONPATH=. $PY -B scripts/train_flashsr_f4.py --epochs 30 --data-dir data/musdb18hq --out-dir models/flashsr; fi\"" \
  --resume-cmd "bash -c \"if [ -f models/flashsr/checkpoint_latest.pt ]; then exec env PYTHONPATH=. $PY -B scripts/train_flashsr_f4.py --epochs 30 --data-dir data/musdb18hq --out-dir models/flashsr --resume models/flashsr/checkpoint_latest.pt; else exec env PYTHONPATH=. $PY -B scripts/train_flashsr_f4.py --epochs 30 --data-dir data/musdb18hq --out-dir models/flashsr; fi\"" \
  --timeout-s 21600 --retries 1

echo "=== KETTE-2-ENDE (DDSP bedingt kein Skill; BEATs-Head/Gender-Head datenblockiert) ==="
