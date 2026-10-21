# VinylNet v4 — chirurgische ML-Restaurierung für Vinyl (gelernter Detektor + InpaintNet + Band-Split-Transformer)

Stand: **2026-10-19** (v4, base-Preset auf MUSDB18-HQ trainiert; GAN-Resume 100 Epochen
abgeschlossen 2026-10-19, deployter Best-Checkpoint = ep 52 des Resume-Laufs). Beleg-Report:
`docs/reports/current/2026-10-12_vinylnet_decrackle.md`. Defizit-Tracking: **D-K1-4**
(`.github/SOTA_DEFICIT_REGISTER.md`).

## Pipeline (v4)
1. **ClickNet v4** (`model.py`, `train_detector.py`): gelernter Klickdetektor — dilatiertes TCN auf
   8 asinh-komprimierten Kanälen `[xL, xR, dxL, dxR, d2L, d2R, M, S]`; die gemeinsame Kanaleingabe
   macht links/rechts versetzte Klicks und Phasenlage (vertikale Rillenbewegung) direkt sichtbar.
   Fallback: klassischer AR-Detektor (`dsp.py`) — **als Fallback riskant** (gemessen: P 0,01–0,07 auf
   MUSDB-Testsplit), §V6-Warnung im Plugin.
2. **Hybrid-Reparatur**: Lücken ≥ `inpaint-min-gap` (96 Samples) per **InpaintNet** (gelerntes U-Net,
   residual, Zeitdomäne); kleinere bleiben roh/AR — das Restaurierungsnetz sieht beides und entscheidet
   selbst, wo die Reparatur taugt (v3-Befund: AR-Fehler übersteigt kleine Klicks selbst).
3. **VinylNet v4** (`model.py`, `train.py`): Stereo-**Band-Split-Transformer**: Zeit-Attention mit RoPE +
   Band-Attention; Eingabe = repariertes **UND rohes** Signal (8 STFT-Kanäle) + Klickmasken-Dichte →
   komplexe Maske je Kanal/Band (startet als Identität) → iSTFT → stereo TCN-**Refiner**; eigener Pfad:
   **ThumpNet** für tieffrequente Pops/Thumps (32× unterabgetastet). `strength` skaliert Thump-Abzug,
   Maske und Refiner (0 = Original … 1 = volle Restaurierung).

## Presets (`model.py`)
| Preset | dim/layers | Parameter | Einsatz |
| --- | --- | --- | --- |
| `tiny` | 32/1 | ~0,4 M | Smoke-Test CPU |
| `small` | 128/4 | 5,37 M | schnelles GPU-Training |
| **`base`** | 192/6 | **11,91 M** | **aktives Produktions-Preset (deployt)** |
| `large` | 256/8 | ~30 M | Ausbaustufe |
| `xl` | 320/10 | ~70 M | maximale SOTA-Ausbaustufe (GPU, 200+ Epochen) |

## Deployte Artefakte (kanonische Namen — Preset-Priorität im Plugin)
- **`vinylnet_base_best.pt`** (47,9 MB, SHA-256 `07f32782…`) — EMA-Gewichte + Config des base-Laufs auf
  MUSDB18-HQ (`runs/vinylnet/best_infer.pt`, GAN-Resume-Beste ep 52: Val SI-SDR 25,13 → 25,71 dB (+0,58),
  MERT sim 0,249 → 0,250; Komposit-Auswahl = SI-SDR + 10·MERT-Delta). Der Vor-Beste des ersten
  base-Laufs (ep 24, SI-SDR +0,25, SHA `2263cad8…`) ist archiviert unter
  `runs/vinylnet/best_ep24_pre_resume_20261017.pt`.
- **`detector_v4.pt`** (0,6 MB, SHA-256 `5d89c992…`) — ClickNet v4 (ch=64, layers=12, feat_ch=8),
  Val-F1 0,601 auf MUSDB-Split.
- Ladepfad-Priorität der Inferenz (`plugins/vinylnet_decrackle_plugin.py`):
  `vinylnet_base_best.pt` → `vinylnet_small_best.pt` → `runs/vinylnet/best_infer.pt`;
  Detektor: `detector_v4.pt` → `runs/detector/detector.pt`.

## Training (Reihenfolge)
```bash
python train_detector.py --data ../../data/musdb18hq/train --pattern mixture.wav --epochs 40
python train.py --data ../../data/musdb18hq/train --pattern mixture.wav --preset base \
    --epochs 100 --gan --gan-start <halbe Laufzeit> --mert ../../models/mert/mert_330m.onnx \
    --mem-fraction 0.45   # geteilte GPU (LM Studio): harter VRAM-Cap, OOM trifft nur diesen Prozess
```
- AMP bf16, EMA (decay 0,999), Schweregrad-Curriculum 0,4→1,0, Log-Mel-Loss + InpaintNet-Loss,
  optionale MR-STFT-GAN-Phase (Sub-Batch `--gan-sub` für VRAM-Effizienz).
- **MERT-Zeuge** in der Validierung: perzeptive Kosinus-Similarität zur sauberen Referenz; beste
  Checkpoints werden komposit (SI-SDR + MERT-Delta) ausgewählt — die Metrik ist Zeuge, nicht Entscheider
  im Loss (Hörordnung: Berechnung bleibt bei den Domain-Regeln).
- Das Netz wird mit **unvollkommenen Masken** trainiert (`perturb_mask`: verpasste Klicks, Fehlalarme,
  Dilatation) und sieht rohes + repariertes Signal → robust gegen Detektorfehler.
- Seed/Threads für Reproduzierbarkeit (§G5). Resume über `--resume runs/vinylnet/last.pt`.

## Simulator (`simulate.py`) — Hybrid-Degradation (K1!)
Stereo-Klicks mit Resonanz-Ring-down (Nadel-Sprung), Kratzer-Bursts, tieffrequente Pops/Thumps,
Poisson-Crackle (teilweise stereo-korreliert), farbiges Rillenrauschen + Rumble, Nadel-Impulsantwort +
zufällige Phono-Färbung, **Print-Through** (Vorecho benachbarter Rille), absolute Pegel. Optionale
Noise-Bank mit **echten Leerrillen-Aufnahmen** (`--noise-bank`) — im base-Lauf noch leer ⇒ D-K1-4:
Degradation synthetisch, Trägermaterial echt.

## Evaluation (gemessen 2026-10-17, MUSDB18-HQ-**Testsplit**, n=16 Clips à 8 s, MERT-Zeuge)
SI-SDR [dB] je Variante/Schweregrad — **VinylNet v4 ist bei sev ≥ 0,6 die beste Variante** (schlägt auch
die Oracle-Maske-DSP-Obergrenze):

| Schwere | unbearb. | AR-Reparatur | gelernter Det. | Oracle-Maske LSAR | Inpaint+Maske | **VinylNet v4** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0,3 | 35,22 | 19,58 | 30,76 | 34,02 | 35,22 | 34,35 |
| 0,6 | 27,51 | 17,95 | 25,64 | 26,59 | 27,58 | **27,75** |
| 1,0 | 23,18 | 16,41 | 22,32 | 22,25 | 23,22 | **24,18** |

- AR-Detektor auf echtem Musikmaterial: P 0,005–0,065 (F1 ≤ 0,11) → seine Reparaturen verschlechtern das
  Signal (Fehlalarme auf Transienten). Gelernter Detektor: F1 0,49–0,52 bei sev ≥ 0,6.
- MERT-Zeuge: sim ≈ 0,998–1,000 in allen Stufen → keine perzeptive Schädigung; bei diesem Task nicht
  diskriminierend (Crackle unterhalb der semantischen Empfindlichkeit).
- A/B-Audio-Export: `output_audio/vinylnet_decrackle_ab_musdb/` (3 Clips × 3 Schweregrade ×
  {noisy, vinylnet, clean}, PCM_24) → Hörabnahme ausstehend.

```bash
python eval.py --ckpt vinylnet_base_best.pt --detector detector_v4.pt \
    --data ../../data/musdb18hq/test --pattern mixture --mert ../../models/mert/mert_330m.onnx --export ab_out
python infer.py platte.wav out.wav --ckpt vinylnet_base_best.pt --detector detector_v4.pt --strength 0.9
```

## Grenzen / offene Punkte (D-K1-4)
- **Hybrid-Degradation**: trainiert auf MUSDB-Mixturen + synthetischem Defekt-Overlay; echte
  Leerrillen-Aufnahmen in der Noise-Bank fehlen noch → größter Hebel für den Praxistransfer.
- GAN-Phase des base-Laufs erst seit dem Resume aktiv (Start: OOM bei ep 25) — **abgeschlossen
  2026-10-19** (`logs/train_vinylnet_base_musdb_resume.log`, „Fertig.", ep 99/step 50 000; Finalwert
  Val SI-SDR +0,48 dB; Best = ep 52 mit +0,58 dB → neu deployt, s. o.). Testsplit-Re-Eval mit dem neuen
  Artefakt ausstehend (§3 des Reports gilt für den Vor-Besten).
- Nicht auf echten Platten verifiziert; SI-SDR auf Simulation ist nur Indikator — Abschlussmaßstab bleibt
  der Hörtest gegen iZotope RX 12 (Hörordnung Ebene 4).
- Erwartet 44,1 kHz (Inferenz resampelt automatisch, deterministisch polyphase).
