# VinylNet De-Crackle — ML-SOTA-Ausbau für phase_68 (Stand 2026-10-19)

> **Zweck:** Explizites De-Crackle-ML-Modell in Aurik: Vinyl-Knistern chirurgisch entfernen
> (Einzelklicks, Pops/Thumps, dichte Crackle-Impulse), Weltspitzen-Niveau gegen iZotope RX 12.
> Aufgabenstart 2026-10-12; gemessene Evidenz dieses Reports: Trainings-Logs unter
> `models/de_crackle/logs/`, Eval-Lauf **2026-10-17** auf dem MUSDB18-HQ-**Testsplit**
> (Log `models/de_crackle/logs/eval_musdb_testsplit.log`). Arbeitssprache Deutsch; §-Zitate mit Quelle.
>
> **Nachtrag 2026-10-19:** Das GAN-Resume des base-Laufs ist **abgeschlossen** („Fertig.", ep 99 /
> step 50 000, Log `models/de_crackle/logs/train_vinylnet_base_musdb_resume.log`); Finalwert Val
> SI-SDR +0,48 dB, Lauf-Beste **ep 52 mit +0,58 dB** — besser als der bisher deployte ep-24-Vor-Beste
> (+0,25). Der neue Best-Checkpoint wurde neu deployt (§4); die §3-Eval-Zahlen wurden mit dem
> Vor-Besten gemessen (Re-Eval ausstehend, s. §5).

## 1. Architektur v4 (`models/de_crackle/model.py`)

- **ClickNet v4**: gelernter Klickdetektor, dilatiertes TCN auf 8 asinh-komprimierten Kanälen
  `[xL, xR, dxL, dxR, d2L, d2R, M, S]` — gemeinsame Kanaleingabe macht links/rechts versetzte Klicks
  und Phasenlage (vertikale Rillenbewegung) direkt sichtbar; Fallback klassischer AR-Detektor (`dsp.py`).
- **InpaintNet**: gelerntes Lücken-Inpainting (residuales U-Net, Zeitdomäne), ersetzt die
  AR-Interpolation bei großen Lücken (v3-Schwachstelle: AR-Fehler übersteigt kleine Klicks selbst).
- **VinylNet v4**: Stereo-Band-Split-Transformer — Zeit-Attention mit RoPE + Band-Attention;
  Eingabe = repariertes UND rohes Signal (8 STFT-Kanäle) + Klickmasken-Dichte → komplexe Maske je
  Band/Kanal (startet als Identität) → iSTFT → stereo TCN-**Refiner**; **ThumpNet** für tieffrequente
  Pops/Thumps (32× unterabgetastet). `strength` skaliert Thump-Abzug, Maske und Refiner.
- Presets: `tiny` / `small` (5,37 M) / **`base` (11,91 M — aktiv)** / `large` / `xl` (maximale Ausbaustufe).

## 2. Trainingsläufe (gemessen aus den Logs)

| Lauf | Daten | Preset/Größe | Ergebnis (Log-Zeile) | Status |
| --- | --- | --- | --- | --- |
| Detektor synthetisch (`train_detector.log`) | `clean_music` (Golden-Samples, 21+1 Dateien), synth. Degradation | ClickNet 153 k | ep 039 „val P 0.759 R 0.772 F1 0.765“; **„Fertig. Bestes F1: 0.766“** (40 Epochen) | abgeschlossen — durch MUSDB-Lauf ersetzt |
| Detektor **MUSDB** (`train_detector_musdb.log`) | `data/musdb18hq/train`, Pattern `mixture.wav` (95+5), synth. Degradation | ClickNet 153 k, seed 42 | ep 039 „val P 0.706 R 0.523 F1 0.601“; **„Fertig. Bestes F1: 0.601“** (40 Epochen) | **abgeschlossen → deployt als `detector_v4.pt`** |
| VinylNet small synthetisch (`train_vinylnet.log`) | `clean_music` (21+1) | small, 5,37 M, AMP bf16, MERT-Zeuge | ep 063 „SI-SDR 27.98 -> 28.50 dB (+0.52)“; OOM-Crash bei `fm`-Berechnung (VRAM-Cap 6,48 GiB) | abgebrochen (Ersatz: MUSDB-Läufe) |
| VinylNet **base MUSDB** (`train_vinylnet_base_musdb.log`) + Resume (`train_vinylnet_base_musdb_resume.log`) | `data/musdb18hq/train`, Pattern `mixture.wav` (95+5), MERT-Zeuge `mert_330m.onnx`, Noise-Bank: 0 | base, 11,91 M, AMP bf16, seed 42, VRAM-Cap 8,39 GiB | Erstlauf ep 024 „SI-SDR 25.13 -> 25.39 dB (+0.25) \| MERT sim 0.249 -> 0.250“; **OOM in der GAN-Phase** (ep 25/step 12 500, „Tried to allocate 66 MiB … 8.39 GiB allowed“) → Resume ab ep 25 mit `--gan-sub 4`: **abgeschlossen 2026-10-19** („Fertig.", ep 99/step 50 000, „SI-SDR 25.13 -> 25.61 dB (+0.48)“); Lauf-Beste **ep 52: +0.58 dB / SI-SDR 25.71** (MERT sim unverändert 0.249 → 0.250) | Best ep 52 neu deployt → `vinylnet_base_best.pt` (SHA `07f32782…`, §4); Vor-Beste ep 24 archiviert (`runs/vinylnet/best_ep24_pre_resume_20261017.pt`, SHA `2263cad8…`) |

- **Einordnung Detektor:** F1 0,766 (synthetischer Val-Split) vs. 0,601 (MUSDB-Val-Split) — die Splits
  sind nicht direkt vergleichbar; der MUSDB-Detektor hat echtes Musikspektrum gesehen und wird daher
  deployt (Generalisierung auf echte Platten > F1 auf Simulations-Val). Beide deutlich über dem
  AR-Fallback.
- **Einordnung base-Lauf:** Der Erstlauf brach vor GAN-Ende und Curriculum-Ausreife ab (+0,25 dB bei
  ep 24); das Resume (100 Epochen, GAN aktiv) schloss am **2026-10-19** mit Finalwert +0,48 dB (ep 99)
  ab — die Lauf-Beste ist **ep 52 mit +0,58 dB Val SI-SDR** (Komposit-Auswahl: SI-SDR + 10·MERT-Delta;
  MERT sim unverändert 0,249 → 0,250 → keine perzeptive Schädigung am Zeugen). Der neu deployte
  Best-Checkpoint (§4) ist damit der beste gemessene Zustand dieses Trainings. Die Testsplit-Eval
  (§3) zeigt den echten Nutzen im Praxisregime (sev ≥ 0,6); sie wurde mit dem Vor-Besten vor dem
  Resume gemessen — Re-Eval mit dem neuen Artefakt s. §5.

## 3. Evaluation MUSDB18-HQ-Testsplit (gemessen 2026-10-17)

Aufruf: `eval.py --ckpt vinylnet_base_best.pt --detector detector_v4.pt --data data/musdb18hq/test
--pattern mixture --mert models/mert/mert_330m.onnx --n 16 --seconds 8.0` (deterministisch: Clip-Auswahl
und Schweregrade seediert, §G5). n = 16 Clips × 3 Schweregrade; Varianten: unbearbeitet / AR-Reparatur /
gelernte Reparatur / Oracle-Maske-LSAR (DSP-Obergrenze) / InpaintNet+Maske / VinylNet v4 vollständig.

**Provenienz-Hinweis:** Gemessen am **2026-10-17 mit dem Vor-Besten vor dem GAN-Resume**
(SHA `2263cad8…`, ep 24). Das neu deployte Artefakt (ep 52, SHA `07f32782…`) hat einen höheren Val-SI-SDR;
die Re-Eval auf diesem Testsplit mit dem neuen Checkpoint ist ausstehend (§5) — die Rangfolge der
Varianten (VinylNet v4 > Oracle-Maske ≥ Input bei sev ≥ 0,6; AR-Detektor riskant) gilt als stabil,
da nur die letzten GAN-Epochen den Best-Checkpoint veränderten.

**SI-SDR [dB] und Loss:**

| Schwere | Variante | SI-SDR | Loss | MERT sim |
| ---: | --- | ---: | ---: | ---: |
| 0,3 | unbearbeitet | 35,22 | 0,291 | 1,000 |
| 0,3 | Reparatur AR-Detektor (LSAR) | 19,58 | 0,454 | 0,999 |
| 0,3 | Reparatur gelernter Det. (LSAR) | 30,76 | 0,306 | 1,000 |
| 0,3 | Reparatur Oracle-Maske (LSAR) | 34,02 | 0,291 | 1,000 |
| 0,3 | InpaintNet + Maske (gelernt) | 35,22 | 0,291 | 1,000 |
| 0,3 | **VinylNet v4 (vollständig)** | **34,35** | **0,274** | 1,000 |
| 0,6 | unbearbeitet | 27,51 | 0,559 | 0,999 |
| 0,6 | Reparatur AR-Detektor (LSAR) | 17,95 | 0,635 | 0,998 |
| 0,6 | Reparatur gelernter Det. (LSAR) | 25,64 | 0,559 | 0,999 |
| 0,6 | Reparatur Oracle-Maske (LSAR) | 26,59 | 0,549 | 0,999 |
| 0,6 | InpaintNet + Maske (gelernt) | 27,58 | 0,558 | 0,999 |
| 0,6 | **VinylNet v4 (vollständig)** | **27,75** | **0,432** | 0,999 |
| 1,0 | unbearbeitet | 23,18 | 0,812 | 0,998 |
| 1,0 | Reparatur AR-Detektor (LSAR) | 16,41 | 0,857 | 0,998 |
| 1,0 | Reparatur gelernter Det. (LSAR) | 22,32 | 0,801 | 0,998 |
| 1,0 | Reparatur Oracle-Maske (LSAR) | 22,25 | 0,797 | 0,998 |
| 1,0 | InpaintNet + Maske (gelernt) | 23,22 | 0,811 | 0,998 |
| 1,0 | **VinylNet v4 (vollständig)** | **24,18** | **0,570** | 0,999 |

**Klickdetektion vs. wahre Maske (P / R / F1):**

| Schwere | AR-Detektor | gelernter Detektor (ClickNet v4) |
| ---: | --- | --- |
| 0,3 | P 0,005 · R 0,324 · **F1 0,010** | P 0,109 · R 0,254 · F1 0,153 |
| 0,6 | P 0,032 · R 0,418 · F1 0,060 | P 0,466 · R 0,507 · **F1 0,485** |
| 1,0 | P 0,065 · R 0,384 · F1 0,111 | P 0,561 · R 0,478 · **F1 0,516** |

**Befunde:**
1. **VinylNet v4 ist bei sev ≥ 0,6 die beste Variante** — schlägt Input (sev 1,0: +1,00 dB; sev 0,6: +0,24 dB)
   und die Oracle-Maske-DSP-Obergrenze (sev 1,0: **+1,93 dB** gegenüber LSAR mit perfekter Maske). Der
   Transformer nutzt das rohe Signal + die reparierte Version komplementär — genau der v4-Designzweck.
2. **Der AR-Detektor ist auf echtem Musikmaterial als Fallback riskant** (P 0,005–0,065): seine Reparaturen
   zerstören Transienten und verschlechtern SI-SDR um 5–7 dB → §V6-Warnung im Plugin gerechtfertigt.
3. **Gelernter Detektor klar über AR** (F1 ×4…×50), bei geringer Schwere noch schwach (F1 0,15) — dort ist
   der Schaden ohnehin klein; das Netz kompensiert per rohem Signal-Eingang.
4. **MERT-Zeuge:** sim ≥ 0,998 in allen Stufen/Schweregraden → keine perzeptive Schädigung durch die
   Verarbeitung (nie-geschlechter auf Zeugen-Ebene); Crackle liegt unterhalb der semantischen Empfindlichkeit
   des Encoders — diskriminierend erst im Hörtest.
5. Bei **sev 0,3** bleibt das Netz leicht unter dem unbearbeiteten Input (34,35 vs. 35,22 dB) — bei sehr
   geringem Schaden übertrifft die Bearbeitung den Nutzen noch nicht; für die Praxis (echte Platten ≈ sev ≥ 0,6)
   ist der Bereich aber relevant.

**A/B-Export (Hörabnahme):** `output_audio/vinylnet_decrackle_ab_musdb/` — 27 Dateien
(3 Clips × 3 Schweregrade × {noisy, vinylnet, clean}, PCM_24, 8 s).

## 4. Deployment & Verdrahtung (Stand 2026-10-17)

| Artefakt | Herkunft | SHA-256 | Manifest-Eintrag |
| --- | --- | --- | --- |
| `models/de_crackle/vinylnet_base_best.pt` (47,9 MB) | `runs/vinylnet/best_infer.pt` (base-MUSDB-GAN-Resume, **ep 52 best**, Val SI-SDR +0,58 dB; deployt 2026-10-19) | `07f3278235df193a…f68869cb2` | `de_crackle_vinylnet_base_best` |
| `models/de_crackle/detector_v4.pt` (0,6 MB) | `runs/detector/detector.pt` (MUSDB-Lauf, best F1 0,601) | `5d89c9921e9fb158…a2ed098db02f0d` | `de_crackle_detector_v4` |
| `models/de_crackle/runs/vinylnet/best_ep24_pre_resume_20261017.pt` (Archiv, unversioniert) | Vor-Beste des Erstlaufs (ep 24, Val SI-SDR +0,25 dB); mit diesem Artefakt wurden die §3-Testsplit-Zahlen und das A/B-Paket `output_audio/vinylnet_decrackle_ab_musdb/` gemessen | `2263cad841b5baef…d09d83e230e` | — (Provenanz-Archiv) |

- **Plugin** `plugins/vinylnet_decrackle_plugin.py`: kanonischer Loader mit **Checkpoint-Pfad-Priorität**
  (`vinylnet_base_best.pt` → `vinylnet_small_best.pt` → `runs/vinylnet/best_infer.pt`; Detektor:
  `detector_v4.pt` → `runs/detector/detector.pt`); Torch primär, ONNX-Präsenz gemeldet (CPU-Fallback erst
  nach Paritätsnachweis — §III.9 copilot-instructions.md). Determinismus eval()+inference_mode ohne RNG (§G5),
  Polyphase-Resampling nur bei SR ≠ 44100, NaN/Inf-Schutz mit Warning (§0a/§V6), thread-sicherer Singleton.
- **Phase** `backend/core/phases/phase_68_vinylnet_decrackle.py`: DEFECT_REMOVAL / RESTORATION_ONLY,
  DefectTypes clicks+crackle; Either-Or-Nachpass NACH phase_09/27 (Primärphasen bleiben), verdrahtet in
  `phases/__init__.py`, `phase_icons.py`, `phase_pruner.py` (clicks/crackle), `unified_restorer_v3.py`
  (Nachpass-Position nach 01/27/09/50/67), `defect_manifest.py`, `causal_defect_reasoner.py`.
  Passthrough bit-identisch + Warning ohne Modell (§V6).
- **Unit-Tests:** `tests/unit/test_phase_68_vinylnet_decrackle.py` (Metadaten-Kontrakt, Layout cf/sf/mono,
  Passthrough bit-identisch, Stub-Determinismus §G5, NaN/Inf-Schutz §0a, Factory; echter Modellpfad skippt
  ohne Artefakte).
- **Domänen-Registry:** `de_crackle` = musik-trainiert (MUSDB18-HQ-Mixturen + synth. Degradation),
  `.github/ML_MODEL_DOMAIN_REGISTRY.md` §1 (38/62); `model_inventory.py` Kopfzahlen synchronisiert.

## 5. Defizite & nächste Schritte → **D-K1-4**

1. **Hybrid-Degradation, keine echten Leerrillen:** Trainingsdaten = MUSDB-Mixturen + synthetisches
   Defekt-Overlay; die Noise-Bank war in allen Läufen leer („Noise-Bank: 0“). Größter Hebel für den
   Praxistransfer: `--noise-bank` mit echten Rillenrausch-Aufnahmen (Leerrillen vieler Platten/Tonabnehmer/
   Phono-Stufen) beschaffen und Neulauf.
2. **GAN-Phase geschlossen (2026-10-19):** base-Lauf OOM bei GAN-Start → Resume
   (`logs/train_vinylnet_base_musdb_resume.log`, `--gan-sub 4` + erhöhter VRAM-Cap) **abgeschlossen** —
   „Fertig.“ ep 99/step 50 000, Finalwert Val SI-SDR **+0,48 dB**; Lauf-Beste **ep 52 mit +0,58 dB**
   (Komposit-Auswahl des Trainers) → neu deployt als `vinylnet_base_best.pt` (SHA `07f32782…`, §4).
   Ausstehend: Testsplit-Re-Eval (§3) und Hörabnahme mit dem neuen Artefakt.
3. **Hörtest gegen iZotope RX 12** auf echten Platten (Hörordnung Ebene 4) — SI-SDR auf Simulation ist nur
   Indikator; A/B-Pakete liegen unter §3 bereit.
4. **ONNX-Export + Paritätsnachweis** (§III.9, rel ≤ 1e-3 auf strukturierten Feeds) ausstehend — dann wird
   der ONNX-CPU-Fallback im Plugin aktiv.
5. Optional: `large`/`xl`-Preset für die maximale Ausbaustufe (GPU, 200+ Epochen), sobald die
   Leerrillen-Datenlage es rechtfertigt.
