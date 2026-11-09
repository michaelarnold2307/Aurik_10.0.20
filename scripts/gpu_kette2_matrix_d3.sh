#!/bin/bash
# GPU-Trainingskette 2 — LUECKEN-MATRIX D.3 (Finetune-Queue).
# Startet automatisch, sobald Kette 1 nicht mehr laeuft; jederzeit manuell
# startbar (bash scripts/gpu_kette2_matrix_d3.sh) — Doppel-Laeufe verhindern
# die Job-Locks (models/*/.train_job.lock).
#
# TAKTUNG (D-K3-87, 2026-11-09): BigVGAN und FlashSR brauchen gemessen
# 4424 s bzw. 4843 s je Epoche => 30 Epochen sind 36,8 h bzw. 40,3 h. Mit
# 2x6 h war der Abschluss STRUKTURELL unerreichbar (Manifest "fehlgeschlagen"
# bei Epoche 8/30 bzw. 11/30, obwohl die Laeufe gesund waren). Deshalb:
#   * 4 Versuche x 12 h (`--timeout-s 43200 --retries 3`) = 48 h Kapazitaet
#   * `--max-seconds 37800` (10,5 h) je Prozess: der Trainer stoppt AN DER
#     EPOCHEN-GRENZE (Exit 75) statt per SIGTERM mitten in der Epoche — ein
#     Kill kostete bis zu 80 min GPU-Zeit (Beleg:
#     docs/reports/current/2026-11-09_d3_taktung_zeitbudget.md)
#   * D3-3 vor D3-2: FlashSR ist naeher am Ziel und darf nicht hinter dem
#     37-h-BigVGAN-Lauf warten
#   * Beide Erstlaeufe sind DATEI-GESCHUETZT (A-SPADE-Muster): ohne Guard
#     begann jeder Kettenstart bei Epoche 0 und ueberschrieb den Anker
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

echo "=== D3-3: FlashSR-Musik-Finetune (out-of-domain -21..-24 dB) ==="
# Resume-Anker = <out-dir>/checkpoint_latest.pt; der Trainer schreibt ihn seit
# 2026-11-08 nach jeder Epoche und merkt sich RNG-/Historien-Stand (D-K3-87).
R --name d3_flashsr_musik --ckpt-dir models/flashsr --phase matrix_d3 \
  --cmd "bash -c \"if [ -f models/flashsr/checkpoint_latest.pt ]; then exec env PYTHONPATH=. $PY -B scripts/train_flashsr_f4.py --epochs 30 --data-dir data/musdb18hq --out-dir models/flashsr --max-seconds 37800 --resume models/flashsr/checkpoint_latest.pt; else exec env PYTHONPATH=. $PY -B scripts/train_flashsr_f4.py --epochs 30 --data-dir data/musdb18hq --out-dir models/flashsr --max-seconds 37800; fi\"" \
  --resume-cmd "bash -c \"if [ -f models/flashsr/checkpoint_latest.pt ]; then exec env PYTHONPATH=. $PY -B scripts/train_flashsr_f4.py --epochs 30 --data-dir data/musdb18hq --out-dir models/flashsr --max-seconds 37800 --resume models/flashsr/checkpoint_latest.pt; else exec env PYTHONPATH=. $PY -B scripts/train_flashsr_f4.py --epochs 30 --data-dir data/musdb18hq --out-dir models/flashsr --max-seconds 37800; fi\"" \
  --timeout-s 43200 --retries 3

echo "=== D3-2: BigVGAN-v2 Musik+Vokal ==="
# Resume-Pfad = output/f3_bigvgan (Default --out-dir von train_bigvgan_f3.py);
# models/bigvgan enthaelt nur die vortrainierten Artefakte, KEINE
# Trainingscheckpoints. Befund 2026-11-09 (D-K3-87): der Erstlauf rief den
# Trainer OHNE --resume, startete damit bei Epoche 0 und ueberschrieb
# checkpoint_latest.pt — der Lauf kam ueber ~8 Epochen nie hinaus, obwohl
# checkpoint_epoch8.pt bereits existierte. Jetzt wie FlashSR datei-geschuetzt.
R --name d3_bigvgan_f3 --ckpt-dir output/f3_bigvgan --phase matrix_d3 \
  --cmd "bash -c \"if [ -f output/f3_bigvgan/checkpoint_latest.pt ]; then exec env PYTHONPATH=. $PY -B scripts/train_bigvgan_f3.py --epochs 30 --max-seconds 37800 --resume output/f3_bigvgan/checkpoint_latest.pt; else exec env PYTHONPATH=. $PY -B scripts/train_bigvgan_f3.py --epochs 30 --max-seconds 37800; fi\"" \
  --resume-cmd "bash -c \"if [ -f output/f3_bigvgan/checkpoint_latest.pt ]; then exec env PYTHONPATH=. $PY -B scripts/train_bigvgan_f3.py --epochs 30 --max-seconds 37800 --resume output/f3_bigvgan/checkpoint_latest.pt; else exec env PYTHONPATH=. $PY -B scripts/train_bigvgan_f3.py --epochs 30 --max-seconds 37800; fi\"" \
  --timeout-s 43200 --retries 3

echo "=== KETTE-2-ENDE (DDSP bedingt kein Skill; BEATs-Head/Gender-Head datenblockiert) ==="
