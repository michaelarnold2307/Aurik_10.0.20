# F5/C4 DDSP-Prädiktor — Erstlauf-Befund (2026-09-16)

> §SOTA-C4 (Roadmap, Phase-Tabelle 04/16/17) — EQ/Dynamik-Parameter-Prädiktion
> aus Audio-Embeddings; DSP-Phasen 04/16/17 führen aus.
> Skript: `scripts/train_ddsp_predictor_c4.py` (--precompute/--train/--smoke).

## Aufbau

| Feld | Wert |
|---|---|
| Encoder | LAION-CLAP (`plugins/laion_clap_plugin.py`, eingefroren, 512-dim, 48 kHz) |
| Trainingsdaten | MUSDB18-HQ train (40 Songs × 12 Segmente à 6 s = 480 Paare) + synthetische Effekt-Paare (3-Band-RBJ-EQ low/mid/high + Soft-Knee-Kompressor 2,5:1; Parameter = Label, normalisiert auf [0,1]) |
| Head | MLP 512→256→128→6, ReLU, MSE, Adam LR 1e-3, 25 Epochs, Seed 42 (§G5 (GEBOTE.md)) |
| Split | song-weise 80/20 (Val = 8 ungesehene Songs) |
| Cache | `output/ddsp_c4/cache/*.npz` (Embeddings + Targets, eingefrorener Encoder ⇒ einmal berechnet) |

## Ergebnis — NEGATIV (Taskformulierung reicht nicht)

| Prädiktor | val-MAE (normalisiert) |
|---|---|
| Modell (CLAP→MLP, best) | **0,24469** |
| Baseline: Train-Mittelwert je Parameter | 0,2455 |
| Baseline: Konstante 0,5 | 0,2453 |

Der eingefrorene CLAP-Head lernt **keinen nutzbaren Signalanteil** über der
trivialen Mittelwert-Baseline hinaus; ab Epoch ~21 überfittet er (val_MAE
steigt). Per-Parameter-MAE 0,219…0,283 (normalisiert).

## Interpretation

1. **CLAP ist semantisch trainiert** (Genre/Instrument/Stimmung), nicht für
   Produktions-EQ/Dynamik — die Embedding-Distanzen tragen die Effekt-Parameter
   nicht. Zusätzlich ist die Aufgabe schlecht gestellt (mehrere
   Effekt-Einstellungen klingen ähnlich; 6-s-Fenster).
2. **Task-Neuformulierung nötig (nächster Schritt, GPU):** DDSP-artiger
   Encoder — Mel-Spektrogramm-CNN-Encoder (Muster: DDSP-Autoencoder-Encoder)
   oder vortrainiertes Audio-Frontend (BEATs/MERT-Features) statt CLAP; ggf.
   größere Effekt-Paar-Korpora + Spektrum-Matching-Verlust statt reiner
   Parameter-Regression.
3. Der **Harness bleibt wiederverwendbar**: Effekt-Paar-Synthese
   (deterministisch, Seed 42), Cache-Pipeline und Trainings-Loop sind
   encoder-agnostisch; ein Encoder-Tausch ist eine lokale Änderung in
   `_embed()`.

## Dateien

- Harness: `scripts/train_ddsp_predictor_c4.py`
- Cache: `output/ddsp_c4/cache/` (40 Songs, gitignored)
- Erster Head (NICHT aktiviert, Negativbefund): `models/ddsp_predictor/c4_head.pth` + `config.json`
- Log: `output/train_ddsp_c4_2026-09-16.txt` (gitignored)
