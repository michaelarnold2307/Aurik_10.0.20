# KP-7 · BW-Reconstructor v5 Re-Training — Ergebnis-Report (ehrlich)

> Datum: 2026-10-25 · Arbeitssprache Deutsch · §-Zitate mit Quelle (AGENTS.md §5
> ID-Zitierdisziplin).

---

## 1. Training — ABGESCHLOSSEN

- Lauf: `scripts/train_bw_v5.py` — 200 Epochen × 150 Steps, Batch 4, lr 1e-4,
  Seed 42 (§G5 (copilot-instructions.md)), GPU (ROCm), **maskierungs-informierter
  A1-Hör-Loss aktiv (Gewicht 0.5, §G9 (copilot-instructions.md) eine Quelle:
  `train_banquet_vinyl_finetune.load_a1_masking_loss(sr=22050)`)**,
  Pre-Gen-Pool 2000 deterministische Samples (KP-7-Performance).
- Ergebnis: **Best Loss 2.1756** (Start 6.10); Checkpoint
  `models/bw_reconstructor/retrain_a1_20261025/best_model_v5.pt`; ONNX-Export
  `models/bw_reconstructor/retrain_a1_20261025/bw_reconstructor_v5.onnx`
  (16,4 MB, `dynamo=False`, checker OK).
- Bestands-Artefakte (`models/bw_reconstructor/bw_reconstructor_v5.onnx`,
  `best_model_v5.pt`) wurden **nicht überschrieben** (separates out-dir).
- Log: `output/train_bw_v5_a1_20261025.log`.

## 2. A1-HF-Gain-Gate — Messung NICHT reproduzierbar (§G8 (copilot-instructions.md))

- Die historische Gate-Definition (alt: 0.73 < 1.02, D-K1-5) ist im Repo
  **nicht dokumentiert** — kein Mess-Skript, keine Metrik-Formel auffindbar.
- Zwei Kandidaten-Metriken wurden implementiert und auf BEIDEN Exporten
  identisch gemessen (`scripts/validate_bw_v5_hf_gate.py`, Seed 42,
  deterministische bandbegrenzte Signale analog zum Trainingsdatenvertrag):
  1. HF-Gain (out/in) in dB: Alt +96.0 dB / Neu +92.9 dB (Cutoff-3000: Neu −5.0 dB,
     Cutoff-5000: Neu +3.5 dB, Cutoff-8000: Neu −2.3 dB, Cutoff-11000: Neu −8.4 dB).
  2. HF-Recovery (out/Referenz-Vollband): beide Modelle im Bereich 0.001–0.014
     (3/5/8 kHz) — am numerischen Boden der Metrik; der 11-kHz-Wert divergiert
     (Referenz-HF-Energie ≈ 0 über 11 kHz im Synthetik-Set).
- **Befund:** Auf dem deterministischen Synthetik-Set zeigt der neue Export
  **keine klare HF-Gain-Überlegenheit** (gemischt, teils schlechter) und die
  Recovery-Metrik ist numerisch nicht aussagefähig. Ein Gate-Verdikt
  (bestanden/nicht bestanden) kann daraus **nicht seriös abgeleitet** werden.
- Rohdaten: `docs/reports/current/2026-10-25_kp7_hf_gate.json`.

## 3. Konsequenz

- **`use_bw_v5` bleibt False** (§V7 (copilot-instructions.md): kein Rollout
  ohne Beleg; W-8 gesperrt). Das DSP-SBR/FlashSR-Paar bleibt der HF-Weg.
- **Offen (Nutzer/Author):** die ursprüngliche A1-HF-Gain-Gate-Metrik
  benennen (Formel + Mess-Set) ODER ein Hörbarkeits-basiertes Gate neu
  definieren (Hörordnung Ebene 2: Audibility statt Mess-Null) — dann
  Alt/Neu auf dieser Metrik messen und entscheiden. Alternativ:
  Hörstichprobe Alt vs. Neu (Hörordnung Ebene 4) als direkter Entscheid.
- Der neue Export bleibt im separaten Verzeichnis erhalten (kein Deployment,
  kein Flag-Flip).

## 4. Belege

- Trainings-Log `output/train_bw_v5_a1_20261025.log`
- Artefakte `models/bw_reconstructor/retrain_a1_20261025/`
- Gate-Messung `scripts/validate_bw_v5_hf_gate.py` +
  `docs/reports/current/2026-10-25_kp7_hf_gate.json`
