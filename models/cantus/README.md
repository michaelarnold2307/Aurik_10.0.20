# models/cantus — Cantus Vocal Restorer

Gesangsrestaurierung (Vocal-Stem-in → Vocal-Stem-out) mit music-aware
MERT-Features, Pitch-/Harmony-Konditionierung und Multi-Scale-Flow-Matching-DiT.

## Architektur (Abbildung → Code)

```
Input: Degradierter Vocal Stem (mono/stereo, 44.1/48 kHz)
│
├──► [Preprocessing] Vocal-Stem zuerst: Resample → 48 kHz, Normalize, DC-Remove   (plugins/cantus_plugin.py)
│
├──► [Feature Encoder - MERT-v1-330M]                          (models/mert/mert_330m.onnx → CantusConditionEncoder)
│     Music-aware SSL Features (1024-dim frame-level — gemessene Breite von
│     models/mert/mert.onnx; die 768-dim des Entwurfs betrafen die 95M-Variante)
│     + Pitch Track (FCPE/CREPE)                               (plugins/fcpe_plugin.py, plugins/crepe_plugin.py)
│     + Harmonic Context (MuQ-MuLan)                           (plugins/muq_mulan_plugin.py, 768-d)
│
├──► [Flow-Matching DiT Backbone]                              (models/cantus/cantus_model.py — CantusDiT)
│     - Conditional on MERT features + pitch track + harmony
│     - Multi-scale: Low-freq (harmonic, DiT-Tokens + Temporal Attention)
│                     / High-freq (transient/noise, LocalConvStack)
│     - Temporal attention for long-range musical context (Self-Attention über Tokens + MERT-Kontext-Tokens)
│     - Local convolutional layers for fine temporal detail (LocalConvStack)
│
├──► [Vocoder/Refinement] BigVGAN / HiFi-GAN (optional)         (plugins/bigvgan_v2_plugin.py, plugins/hifigan_plugin.py)
│
└──► Output: Restaurierter Vocal Stem
```

**Design-Entscheidung Waveform-Domäne:** Der Flow-Matching-Kern läuft wie
`models/miipher_dit/dit_model.py` direkt in der Waveform-Domäne
(Geschwindigkeitsfeld v̂ ≈ y − x). Damit ist die Rekonstruktion phasenkorrekt
per Konstruktion (kein Vocoder im kritischen Pfad) und die bewährte
miipher_dit-ONNX-/Plugin-Infrastruktur ist direkt wiederverwendet
(§„Flow-Matching DiT Backbone ← Projekt-Infrastruktur wiederverwenden").
Die Vocoder-Bühne (BigVGAN/HiFi-GAN) ist als optionale spektrale
Nachbearbeitung angebunden (Default: aus — Never-worsen hat Vorrang).

## Trainingsziel (Flow Matching)

```
t   ~ U(0,1);  x_t = (1-t)·x_degraded + t·y;  v = y − x_degraded
model(x_t, t, mert, pitch, harm, use_cond) → v̂
Inferenz (t=0.5):  ŷ = x + (1−t)·v̂
```

Multi-Objective-Loss (scripts/train_cantus.py):
`L = λ₁·L_flow + λ₂·L_mel + λ₃·L_stft_phase + λ₄·L_pitch + λ₅·L_SingMOS + λ₆·L_temporal`

## Gewichte und produktiver Inferenzpfad

| Datei | Inhalt | Status |
|---|---|---|
| `checkpoint_best.pt` | Torch-Training-Checkpoint | lokales Pretraining-Artefakt, 204,27 M Parameter, Seed `20261004`, Epoche 10 |
| `cantus_dit.onnx` | Flow-Matching-DiT Inferenzgraph (OpSet 14+, dynamische Achsen) | lokaler CPU-Ersatzpfad; Export-Parität zum Checkpoint bestanden |
| `cantus_singmos_proxy.pt` | Gewichte des SingMOS-Learned-Loss-Proxys | **Platzhalter** — via `scripts/train_cantus.py --phase=pretrain` |
| `cantus_config.json` | Hyperparameter (Modell + Training + Loss-Gewichte) | aktiv |

Die Modellartefakte sind lokal und werden nicht versioniert. Der vorgesehene
Produktionspfad in `plugins/cantus_plugin.py` lädt einen freigegebenen
Checkpoint über Torch auf ROCm und verwendet ONNX ausschließlich über
`CPUExecutionProvider` als Ersatzpfad. Solange der Modell-Zoo-Status nicht
`active` ist, bleibt der kanonische Checkpoint deaktiviert und Cantus nutzt den
protokollierten DSP-Fallback (§V6 (copilot-instructions.md)).
Die am 2026-10-04 auf der RX 7900 XTX gemessene Torch-ROCm↔ONNX-CPU-Parität
beträgt maximal `7.89e-06` relativ (Gate: `≤ 1e-3`); zwei GPU-Wiederholungen
waren bitidentisch. Scheitert Torch oder ONNX, läuft Cantus deterministisch auf
den DSP-Ersatzpfad (Wiener, §V6 (copilot-instructions.md), mit
`logger.warning`).

Der Checkpoint befindet sich noch in der Pretraining-Phase. Ein Release mit
Fine-Tuning oder Domain-Adaptation setzt die zugehörigen, reproduzierbaren
Trainings- und Hörqualitäts-Evidenzen voraus; der Artefaktstatus wird dabei
aktualisiert.

## Training (3 Phasen)

```bash
python3 -B scripts/generate_synthetic_degraded_vocals.py --musdb-root data/musdb18hq --out data/cantus_pairs
python3 -B scripts/train_cantus.py --phase=pretrain    --freeze-encoder=true  --epochs=50
python3 -B scripts/train_cantus.py --phase=fine_tune   --singmos-loss=true    --epochs=30
python3 -B scripts/train_cantus.py --phase=domain_adapt --data=data/real_degraded_vocals --epochs=10
python3 -B scripts/export_cantus_onnx.py   # ONNX + Paritäts-Gate rel ≤ 1e-3 (§III.9)
```

MERT ist in allen Phasen gefroren (ONNX-Features, §III.9 (copilot-instructions.md));
`--freeze-encoder` friert die trainierbaren Conditions-Adapter ein/aus.

## Lizenz

Eigenentwicklung im Aurik-Projekt (Copyright 2026 Michael Arnold, siehe
NOTICE-Skill). Eingebundene Fremdmodelle behalten ihre Lizenzen
(MERT: CC-BY-NC-SA-4.0-frei? → siehe models/mert-v1-330m, SingMOS: BSD-3).
