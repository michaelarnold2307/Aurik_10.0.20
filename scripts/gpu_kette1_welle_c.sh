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

echo "=== R2: SOTA-M9 TF-GridNet Echt-Training (Musik-Paare, KEIN --synthetic) ==="
# Speicherbudget 24 GiB: das Inter-Frame-BiLSTM laeuft ueber b*f Sequenzen der
# Laenge T=752. Mit --batch-size 4 waren das 4x513 Sequenzen und 4,42 GiB
# zusaetzlich -> OOM nach 5 s (Befund 2026-11-06, train_job_m9_tf_gridnet.json).
# Micro-Batch 1 + Gradient-Akkumulation 4: rechnerisch Batch 4, kleinerer Peak.
# Daten: NUR echte Musik-Paare (MUSDB18-HQ via scripts/prepare_tfgrid_pairs.py).
# Die frueher hier verdrahteten --synthetic 4000 (Sinus + Rauschen) sind laut
# Docstring "zum Test" und erzeugen ein K1-Domaenen-Defizit (§III.11 /
# SOTA_DEFICIT_REGISTER); Kette 3 traegt denselben Job bereits ebenfalls
# echte Paare — Job-Locks verhindern Doppel-Laeufe.
if [ -d data/tfgrid_pairs/clean ] && [ -d data/tfgrid_pairs/noisy ]; then
  R --name m9_tf_gridnet --ckpt-dir models/tf_gridnet --phase sota_m9 \
    --cmd "env PYTHONPATH=. $PY -B scripts/train_tf_gridnet_denoiser.py --epochs 30 --device cuda --clean-dir data/tfgrid_pairs/clean --noisy-dir data/tfgrid_pairs/noisy --batch-size 1 --grad-accum 4" \
    --resume-cmd "bash -c \"if [ -f models/tf_gridnet/checkpoint_latest.pt ]; then exec env PYTHONPATH=. $PY -B scripts/train_tf_gridnet_denoiser.py --epochs 30 --device cuda --clean-dir data/tfgrid_pairs/clean --noisy-dir data/tfgrid_pairs/noisy --batch-size 1 --grad-accum 4 --resume models/tf_gridnet/checkpoint_latest.pt; else exec env PYTHONPATH=. $PY -B scripts/train_tf_gridnet_denoiser.py --epochs 30 --device cuda --clean-dir data/tfgrid_pairs/clean --noisy-dir data/tfgrid_pairs/noisy --batch-size 1 --grad-accum 4; fi\"" \
    --timeout-s 21600 --retries 1
else
  echo "M9-Daten fehlen (data/tfgrid_pairs/{clean,noisy}) — R2 uebersprungen (datenblockiert)."
  echo "Datenaufbereitung: scripts/prepare_tfgrid_pairs.py (Report 2026-11-06_m9_tfgrid_echt_daten.md)."
fi

# Cantus (SOTA-M14) braucht --data EXPLIZIT: ohne Paar-Manifest bricht
# train_cantus.py fail-closed ab ("--data fehlt", Exit 2). Befund 2026-11-06:
# R3-R5 liefen ohne --data und scheiterten in 2 s, obwohl data/cantus_pairs mit
# 3600 Paaren lokal liegt. Auto-Erkennung statt Hardcode.
CANTUS_DATA=""
for cand in data/cantus_pairs/manifest.jsonl; do
  if [ -f "$cand" ]; then CANTUS_DATA="$cand"; break; fi
done

echo "=== R3: KP-8 Cantus pretrain (Full-Preset, GPU) ==="
if [ -n "$CANTUS_DATA" ]; then
  R --name c2_cantus_pretrain --ckpt-dir models/cantus --phase pretrain \
    --cmd "env PYTHONPATH=. $PY -B scripts/train_cantus.py --phase=pretrain --preset=full --freeze-encoder --epochs=50 --data $CANTUS_DATA" \
    --resume-cmd "env PYTHONPATH=. $PY -B scripts/train_cantus.py --phase=pretrain --preset=full --freeze-encoder --epochs=50 --data $CANTUS_DATA --resume models/cantus/checkpoint_latest.pt" \
    --timeout-s 43200 --retries 1
else
  echo "Cantus-Paar-Manifest fehlt (data/cantus_pairs/manifest.jsonl) — R3 uebersprungen (datenblockiert, SOTA-M14)."
fi

echo "=== R4: KP-8 Cantus fine_tune (Phase 2, GPU) ==="
if [ -n "$CANTUS_DATA" ]; then
  R --name c2_cantus_finetune --ckpt-dir models/cantus --phase fine_tune \
    --cmd "env PYTHONPATH=. $PY -B scripts/train_cantus.py --phase=fine_tune --preset=full --singmos-loss --epochs=30 --data $CANTUS_DATA --resume models/cantus/checkpoint_latest.pt" \
    --resume-cmd "env PYTHONPATH=. $PY -B scripts/train_cantus.py --phase=fine_tune --preset=full --singmos-loss --epochs=30 --data $CANTUS_DATA --resume models/cantus/checkpoint_latest.pt" \
    --timeout-s 43200 --retries 1
else
  echo "Cantus-Paar-Manifest fehlt (data/cantus_pairs/manifest.jsonl) — R4 uebersprungen (datenblockiert, SOTA-M14)."
fi

# domain_adapt braucht ECHTE degradierte Vocal-Paare; die synthetischen
# cantus_pairs sind dafuer die falsche Domaene (§V7 (copilot-instructions.md)).
CANTUS_REAL=""
for cand in data/cantus_real data/real_degraded_vocals; do
  if [ -d "$cand" ] && [ -n "$(find "$cand" -maxdepth 2 -name "*_degraded.wav" -print -quit 2>/dev/null)" ]; then
    CANTUS_REAL="$cand"; break
  fi
done

echo "=== R5: KP-8 Cantus domain_adapt (Phase 3, reale Degradationen) ==="
if [ -n "$CANTUS_REAL" ]; then
  R --name c2_cantus_domain --ckpt-dir models/cantus --phase domain_adapt \
    --cmd "env PYTHONPATH=. $PY -B scripts/train_cantus.py --phase=domain_adapt --preset=full --epochs=20 --data $CANTUS_REAL --resume models/cantus/checkpoint_latest.pt" \
    --resume-cmd "env PYTHONPATH=. $PY -B scripts/train_cantus.py --phase=domain_adapt --preset=full --epochs=20 --data $CANTUS_REAL --resume models/cantus/checkpoint_latest.pt" \
    --timeout-s 43200 --retries 1
else
  echo "Reale degradierte Vocal-Paare fehlen (data/cantus_real, data/real_degraded_vocals) — R5 uebersprungen (datenblockiert, SOTA-M14)."
fi

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
