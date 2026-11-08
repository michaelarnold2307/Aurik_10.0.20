#!/usr/bin/env bash
# GPU-Trainingskette 4 — Modell-Köpfe & Trainingsdaten-Pflicht (D-K4-7/8/5/6, D-K2-3, Köpfe).
# Dauerhaftes Repo-Artefakt: jederzeit neu startbar (bash scripts/gpu_kette4_modellkoepfe.sh).
# — Job-Locks + Manifeste verhindern Doppellaeufe (train_job_runner.py, siehe docs/SOTA_HANDOVER.md §3).
# — Sequenziell: KEIN paralleles Training zu laufenden Ketten (§5 SOTA_HANDOVER). Der Start-Befehl
#   wird NUR ausgeführt, wenn pgrep -f train_job_runner LEER ist (Pflicht-Check, siehe unten).
# — Trainingsdaten-Synthese deterministisch (Seed 42, §G5) — Muster prepare_tfgrid_pairs.py.
# — KEIN Flag-Flip (§5): use_wf4_warp_head/use_tp_v2_phase/use_vocos_music bleiben AUS; die Jobs
#   schreiben nur Artefakte/Manifeste, keine Aktivierung.
# — Host-Schutz: nice 15, Thread-Caps, RAM-Guard; GPU-Venv $PY (System-python ist CPU-only).
#
# Start (AUSFÜHRUNG durch den Manager, NACHDEM pgrep -f train_job_runner leer ist):
#   nohup bash scripts/gpu_kette4_modellkoepfe.sh > /tmp/k4.log 2>&1 &
#
# Reihenfolge: kleine/kopf-gebundene Jobs zuerst, dann daten-getriebene. Jeder Job über
# train_job_runner (Lock + Auto-Resume + Manifest docs/reports/current/train_job_<name>.json).
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/home/michael/.local/share/aurik/venv_rocm72/bin/python
export PYTHONPATH=.
# Host-Schutz (Pfeiler 6): CPU-Threads begrenzen, damit Trainings die GUI nicht ersticken.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

# ── Pflicht-Check: kein paralleles Training (§5 SOTA_HANDOVER) ─────────────────
# NUR starten, wenn kein train_job_runner laeuft. Sonst abbrechen (fail-closed),
# damit nie zwei Trainings gleichzeitig auf der GPU laufen.
if pgrep -f "train_job_runner" > /dev/null 2>&1; then
  echo "ABBRUCH: train_job_runner laeuft bereits (§5 keine parallelen Trainings)."
  echo "Erst alle laufenden Trainings abwarten, dann Kette 4 starten."
  exit 2
fi

R() { python3 -B scripts/train_job_runner.py "$@"; echo "JOB_EXIT=$?"; }

# ── Vorbereitung: deterministische Trainingsdaten-Synthese (Seed 42, §G5) ───────
# WF-V4 (D-K4-7): Warp-Trajektorien (Wow/Flutter) auf MUSDB18-HQ-Mischungen.
# TP-V2 (D-K4-8): transiente-reiche Segmente aus MUSDB18-HQ-Drums/Percussion-Stems.
echo "=== V1: WF-V4-Warp-Paare synthetisieren (MUSDB18-HQ, Seed 42) ==="
if [ -d data/musdb18hq ]; then
  env PYTHONPATH=. nice -n 15 python3 -B scripts/prepare_wf4_warp_pairs.py \
    --musdb-root data/musdb18hq --out-dir data/wf4_warp_pairs --anzahl 300 --seed 42
else
  echo "MUSDB18-HQ fehlt (data/musdb18hq) — WF-V4-Paare uebersprungen (daten-blockiert)."
fi

echo "=== V2: TP-V2-transiente Segmente synthetisieren (MUSDB18-HQ-Drums, Seed 42) ==="
if [ -d data/musdb18hq ]; then
  env PYTHONPATH=. nice -n 15 python3 -B scripts/prepare_tp2_transient_pairs.py \
    --musdb-root data/musdb18hq --out-dir data/tp2_transient_pairs --seed 42
else
  echo "MUSDB18-HQ fehlt (data/musdb18hq) — TP-V2-Segmente uebersprungen (daten-blockiert)."
fi

echo "=== R1: SOTA-M12 WF-V4 Warp-Schätzer Echt-Training (D-K4-7) ==="
if [ -f data/wf4_warp_pairs/manifest.json ] && [ -d data/wf4_warp_pairs/f0 ]; then
  R --name wf4_warp_head --ckpt-dir models/wf4_warp_head --phase sota_m12 \
    --cmd "env PYTHONPATH=. nice -n 15 $PY -B scripts/train_wf4_warp_head.py --epochs 50 --steps-per-epoch 200 --real-dir data/wf4_warp_pairs --device cuda --out-dir models/wf4_warp_head" \
    --resume-cmd "env PYTHONPATH=. nice -n 15 $PY -B scripts/train_wf4_warp_head.py --epochs 50 --steps-per-epoch 200 --real-dir data/wf4_warp_pairs --device cuda --out-dir models/wf4_warp_head" \
    --timeout-s 21600 --retries 1
else
  echo "WF-V4-Paare/Beobachtung fehlen (data/wf4_warp_pairs/{manifest.json,f0}) — R1 uebersprungen (fail-closed, kein synthetischer Ersatz)."
  echo "  Erst: env PYTHONPATH=. python3 -B scripts/prepare_wf4_warp_pairs.py --musdb-root data/musdb18hq --out-dir data/wf4_warp_pairs --anzahl 300 --seed 42"
fi

echo "=== R2: SOTA-M12 TP-V2 Phasenschätzer Echt-Training (D-K4-8) ==="
if [ -d data/tp2_transient_pairs ] && [ -n "$(ls -A data/tp2_transient_pairs/*.wav 2>/dev/null | head -1)" ]; then
  R --name tp_v2_phase --ckpt-dir models/tp_v2 --phase sota_m12 \
    --cmd "env PYTHONPATH=. nice -n 15 $PY -B scripts/train_tp_v2_phase.py --epochs 50 --steps-per-epoch 200 --real-dir data/tp2_transient_pairs --synthetic 0 --out-dir models/tp_v2 --device cuda" \
    --resume-cmd "env PYTHONPATH=. nice -n 15 $PY -B scripts/train_tp_v2_phase.py --epochs 50 --steps-per-epoch 200 --real-dir data/tp2_transient_pairs --synthetic 0 --out-dir models/tp_v2 --device cuda" \
    --timeout-s 21600 --retries 1
else
  echo "TP-V2-Segmente fehlen (data/tp2_transient_pairs/*.wav) — R2 uebersprungen."
fi

echo "=== R3: D-K4-5 F9 VoiceEncoder Musik-Vokal-Finetune (MUSDB18-HQ-Vocals) ==="
# F9: resemblyzer VoiceEncoder (sprach-trainiert) auf MUSDB18-HQ-Vocals finetunen.
# Trainer implementiert 2026-11-06 (Commit d06c765f, 21 Vertragstests) — Job
# eingereiht; Aktivierung sobald data/musdb18hq vorliegt (Guard unten).
if [ -f scripts/train_voiceencoder_finetune.py ] && [ -d data/musdb18hq ]; then
  R --name f9_voiceencoder_finetune --ckpt-dir models/resemblyzer/finetune --phase f9_voiceencoder \
    --cmd "env PYTHONPATH=. nice -n 15 $PY -B scripts/train_voiceencoder_finetune.py --musdb-root data/musdb18hq --epochs 30 --device cuda --out-dir models/resemblyzer/finetune" \
    --resume-cmd "env PYTHONPATH=. nice -n 15 $PY -B scripts/train_voiceencoder_finetune.py --musdb-root data/musdb18hq --epochs 30 --device cuda --out-dir models/resemblyzer/finetune" \
    --timeout-s 21600 --retries 1
else
  echo "F9 blockiert: scripts/train_voiceencoder_finetune.py fehlt ODER data/musdb18hq fehlt — R3 uebersprungen (daten-blockiert, siehe Report)."
fi

echo "=== R4: D-K4-6 F10 SingMOS-Kalibrierung auf verarbeitete Musik-Vocals ==="
# F10: SingMOS-Kalibrierung auf Effektketten/Misch-Material aus MUSDB-Vocals.
# Kalibrierer implementiert 2026-11-06 (Commit d06c765f, 21 Vertragstests) —
# Job eingereiht; Aktivierung sobald data/musdb18hq vorliegt (Guard unten).
if [ -f scripts/calibrate_singmos_music.py ] && [ -d data/musdb18hq ]; then
  R --name f10_singmos_calibrate --ckpt-dir models/singmos/calibration --phase f10_singmos \
    --cmd "env PYTHONPATH=. nice -n 15 $PY -B scripts/calibrate_singmos_music.py --musdb-root data/musdb18hq --out-dir models/singmos/calibration" \
    --resume-cmd "env PYTHONPATH=. $PY -B scripts/calibrate_singmos_music.py --musdb-root data/musdb18hq --out-dir models/singmos/calibration" \
    --timeout-s 21600 --retries 1
else
  echo "F10 blockiert: scripts/calibrate_singmos_music.py fehlt ODER data/musdb18hq fehlt — R4 uebersprungen (daten-blockiert, siehe Report)."
fi

echo "=== R5: D-K2-3 vocos_48khz Musik-Finetune (MUSDB18-HQ-Gesang) ==="
# D-K2-3: Musik-Finetune für models/vocos_48khz (Primär-Rollen-Entscheidung Studio-Finisher
# bleibt menschlich — dieser Job trainiert NUR, kein Flag-Flip, kein Rollout).
if [ -f scripts/train_vocos_48khz_finetune.py ] && [ -d data/musdb18hq ]; then
  R --name d_k2_3_vocos_finetune --ckpt-dir models/vocos_48khz/finetune --phase d_k2_3 \
    --cmd "env PYTHONPATH=. nice -n 15 $PY -B scripts/train_vocos_48khz_finetune.py --musdb-root data/musdb18hq --epochs 30 --device cuda --out-dir models/vocos_48khz/finetune" \
    --resume-cmd "env PYTHONPATH=. nice -n 15 $PY -B scripts/train_vocos_48khz_finetune.py --musdb-root data/musdb18hq --epochs 30 --device cuda --out-dir models/vocos_48khz/finetune" \
    --timeout-s 21600 --retries 1
else
  echo "D-K2-3 blockiert: scripts/train_vocos_48khz_finetune.py fehlt ODER data/musdb18hq fehlt — R5 uebersprungen (daten-blockiert, siehe Report)."
fi

echo "=== R6: Köpfe BEATs-Head / MERT-Head / Gender-Head (daten-blockiert) ==="
# BEATs-Head (527 Klassen, Instrument-/Genre-Evidenz): Trainingslauf fehlt + Label-Quelle fehlt.
# MERT-Head (Genre): GTZAN/MagnaTagATune-Trainingslauf fehlt (Datensatz + Labels).
# Gender-Head (F13): braucht data/gender_labels.csv (file,gender) mit ≥ ~50 Labels je Klasse —
#   ohne Labels bricht scripts/train_gender_head.py ab (§V7) und rät nicht.
# Alle drei sind daten-blockiert (nicht code-blockiert) — Jobs eingereiht, aber OHNE Ausfuehrung
# bis die Label-Quellen vorliegen (siehe Report §Köpfe).
if [ -f data/gender_labels.csv ] && [ -f scripts/train_gender_head.py ]; then
  R --name gender_head --ckpt-dir models/gender_head --phase f13_gender \
    --cmd "env PYTHONPATH=. nice -n 15 $PY -B scripts/train_gender_head.py --labels data/gender_labels.csv --audio-dir data/musdb18hq --out models/gender_head --epochs 200" \
    --resume-cmd "env PYTHONPATH=. nice -n 15 $PY -B scripts/train_gender_head.py --labels data/gender_labels.csv --audio-dir data/musdb18hq --out models/gender_head --epochs 200" \
    --timeout-s 21600 --retries 1
else
  echo "Gender-Head blockiert: data/gender_labels.csv fehlt (Label-Korpus ≥ ~50 je Klasse) — R6 uebersprungen (daten-blockiert)."
fi
echo "BEATs-Head / MERT-Head: keine Trainings-Scripts + keine Label-Quellen im Repo —"
echo "  daten-blockiert (BEATs: Head-Training/Tagger-ONNX fehlt; MERT-Head: GTZAN/MagnaTagATune fehlt)."

echo "=== KETTE-4-ENDE ==="
echo "Hinweis: Trainings NUR starten wenn pgrep -f train_job_runner leer (Pflicht-Check oben)."
echo "Start-Befehl (Manager, nohup): nohup bash scripts/gpu_kette4_modellkoepfe.sh > /tmp/k4.log 2>&1 &"
