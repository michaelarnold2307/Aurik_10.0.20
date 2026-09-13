# Phasen-SOTA-Gap-Analyse — Ausbaustufen für alle Aurik-Domänen

Stand: 2026-09-13 · Verifiziert gegen den Ist-Zustand der 69 Phasen-Dateien
(`backend/core/phases/`, Algorithmus-Header) und die in dieser Session
umgesetzten Hybrid-Bausteine. Fortsetzung der Analyse-Linie
`docs/WITNESS_SOTA_GAP_ANALYSE.md`.

Normative Einordnung: Hörordnung (Maskierungsschwelle statt Mess-Null,
Metriken sind Zeugen), §V7 (z. B. „Stärke zentral über global_scalar"),
Never-worsen als Konstruktprinzip. Jede Ausbaustufe ist delta-basiert,
deterministisch und mit DSP-Fallback (§V6 (copilot-instructions.md)).

## Legende

| Kennung | Bedeutung |
|---|---|
| ✅ | SOTA erreicht (in dieser Session oder bereits vorher) |
| V1…V4 | Ausbaustufen nach Hör-Gewinn, V1 = größter Hebel |

## 1. Restaurations-Kern (Restoration-Modus)

### 1.1 Klick/Knistern — phase_01/09/27

- Ist: Multi-Scale-Detektion + **RBME** (AR(16)-Prior, iterative Sparse-Bayesian-
  Inpainting, Roux & Bimbot 2014) + linear/kubisch/spektral; phase_27
  AR-Residual-Detektion + Consistent Wiener; phase_09 Multi-Scale + BANQUET
  (coordinated_repair). → ✅ SOTA-Klasse erreicht.
- Lücke: Die **Detektion** in phase_01 ist rein DSP; BANQUET (neuronal) läuft
  nur im Knistern-Pfad, nicht als Klick-Detektor.
- **CR-V1 (B6-Rest):** BANQUET-Klick-Detektion als zusätzlicher Detektor im
  Multi-Scale-Konsens von phase_01 — ML detektiert, RBME rekonstruiert
  (voller Hybrid: neuronale Detektion + physikalisches Modell).

### 1.2 Denoise — phase_03 ✅

- Ist: OMLSA/IMCRA + DeepFilterNet + **H1-Masking-Fusion + H2-Musical-Noise-Gate
  + EAR-VAE (musik-finetuned, ΔSDR +4,83 dB)** + Witness-Veto.
  SOTA-Klasse erreicht. Keine offene Stufe (Multiresolution-Split D3 optional,
  s. §4).

### 1.3 Declipper — phase_07 ✅

- Ist: APPLADE + PnP-ADMM + Never-worsen-Router, GPU-finetuned
  (ΔSDR +1,2…+1,7 dB), CQT-Diff+ als zweiter neuronaler Zweig.
  SOTA-Klasse erreicht.

### 1.4 Hum — phase_02

- Ist: Multi-Fundamental-Detektion (RX-De-hum-Klasse). Solide.
- Lücke: klein (adaptive Harmonic-Comb mit Kalman-Tracking der Netzfrequenz-
  Drift). **HU-V1 (niedrige Priorität):** Kalman-getracktes Notch-Filter.

### 1.5 Wow/Flutter — phase_12 (Ausbaustufen WF-V1…V4, s. Todos)

- Ist: pYIN + FCPE/CREPE-Hybrid, np.interp-Resampling, Authentizitäts-Guard.
- **WF-V1** bandbegrenztes Spline/Sinc-Resampling statt np.interp.
- **WF-V2** harmonische Spektralkorrelations-Warp-Schätzung (Capstan-Prinzip,
  F0-unabhängig) als dritter Schätzer + Konsens-Gate; Cassette: breitere
  Warp-Bandbreite.
- **WF-V3** Kalman-Glättung der Warp-Trajektorie (Cassette: driftende
  Hub-Wow-Frequenz).
- **WF-V4** neuronaler Warp-Schätzer (2025/26-Checkpoint) mit Never-worsen-Gate.
- **WF-Cassette** Scrape-Flutter-Restpfad (Breitband-Modulations-Kompensation/
  Denoise) separat für Cassette.
- **SP-V1 (phase_31):** dasselbe bandbegrenzte Resampling wie WF-V1 auch im
  Speed/Pitch-Korrektur-Pfad verwenden (geteilte Komponente).

### 1.6 Frequenz-/Harmonik-Restaurierung — phase_06/07_harmonic

- Ist: SBR/LPC + FlashSR mit **B4-Synthesis-Gate** (masking-bewusst,
  onset-geschützt) ✅; phase_07_harmonic: DSP-Harmonik-Synthese.
- Lücke: fehlende Harmonische werden rein DSP-generiert.
- **HR-V1 (B5-Erweiterung):** BigVGAN-Repair-Pfad (mit demselben
  additive_synthesis_gate) für phase_07_harmonic_restoration — neuronaler
  Vocoder liefert, DSP-Gate begrenzt auf die Maskierungsschwelle.

### 1.7 Spektral-Reparatur/Inpainting — phase_23/50/55/24 (Ausbaustufe B8)

- Ist: phase_23 IMCRA-Floor + vektorisiertes Inpainting; phase_50
  STFT-Inpainting; phase_55 Diffusion-Inpainting (DDPM-inspiriert) mit
  optionalem DiffWave/AudioLDM2-Plugin-Pfad; phase_24 Multi-Modal-Detektion.
- Lücke: Die **Nähte** der generativen Inpainting-Ausgabe sind ungesichert —
  keine Masking-Freigabe, kein Hüllkurven-Alignment, kein Onset-Schutz.
- **IN-V1 (B8):** Generative-Inpainting-Naht-Gates: additive_synthesis_gate
  (masking-bewusste Freigabe, Onset-Schutz) + masked_denoise_fusion-Logik um
  die phase_55-Ausgabe (DiffWave/AudioLDM2/CQT-Diff+).
- **IN-V2:** C2-artiges Hüllkurven-Alignment an den Inpainting-Nähten
  (Komponente aus stem_recombination_gates wiederverwenden).

### 1.8 Dereverb — phase_20/49

- Ist: OMLSA/IMCRA-Dereverb mit Transienten-Erhaltung; phase_49 WPE/OMLSA
  consistent — klassische SOTA ✅.
- Lücke: Parameter (Nachhallzeit) werden blind geschätzt.
- **DR-V1:** Neurale RT60-Schätzung (kleines Modell auf BEATs/CLAP-Embedding
  oder direkte Waveform-Schätzung) steuert die Dereverb-Parameter — ML misst,
  DSP führt aus.

### 1.9 Transienten — phase_08/36 (D-Klasse)

- Ist: Spectral-Flux-Onset-Detektion, Multi-Band-Shaping, Log-Domain-Ballistik.
- **TP-V1 (D2):** Neurale Onset-Detektion (BEATs-getriggert) als Konsens mit
  Spectral Flux — weniger Fehl-Trigger, präzisere Anschläge.
- **TP-V2 (D1):** Neurale Phasen-Schätzung für Transienten-Frames (DSP-Phase
  im Rest) — schärfere Anschläge ohne Phasen-Verschmierung.

## 2. Mastering/Musical-Goals-Kette

### 2.1 EQ/Dynamik — phase_04/10/16/17/26/34/54

- Ist: adaptive EQ, Multi-Band-Parallel-Kompression, Transparent Dynamics mit
  psychoakustischer Maskierungs-Zonen-Erkennung — solide klassische SOTA.
- Lücke: Parameter werden heuristisch geschätzt statt prädiziert.
- **C4 DDSP:** Neuronales Netz prädiziert EQ-/Dynamik-Parameter
  (DDSP-artig, auf CLAP/BEATs-Embeddings), DSP führt aus — musikalisch
  konsistente Ziele ohne Wellenform-Generierung. Anbindung an phase_04/16/17.

### 2.2 Glue/Limiter/Loudness/Output — glue_stage/11/47/40/41 ✅

- Ist: Glue als vorletzte Phase, True-Peak-Limiter mit 4×-Oversampling und ISP,
  BS.1770-4/EBU R128, POW-r-Dither — SOTA-Standards erreicht. Keine offene Stufe.

### 2.3 Stereo — phase_13/14/15/25/32/33/34/48

- Ist: M/S-Verarbeitung, Gerzon-Phasenkorrektur, IACC-Guard, Azimuth —
  klassische SOTA ✅. Witness-Felder itd_drift/ild_drift/iacc_drop
  überwachen (P3). Keine offene Stufe.

## 3. Spezial-Defekte (phase_56–64) ✅

Band-Gap-Repair, Print-Through (LMS), Modulationsrauschen (Esquef & Biscainho),
IGD, Groove-Echo, Crosstalk (exakte α-Inverse, IEC 60098), IMD (Bispektrum),
Splice-Repair, Vocal-Naturalness (Korrektiv, §0a), Stem-Targeted-NR
(C1–C3 + KIM2) — klassische DSP-SOTA mit exakter Modellierung. Keine offene
Stufe; die ML-Seite ist über die Gates dieser Session abgedeckt.

## 4. Querschnitt (D-Klasse)

- **D3 Multiresolution-Split:** neuronale Verarbeitung nur im kritischen Band
  (2–5 kHz, höchste Ohr-Empfindlichkeit), DSP außerhalb — optional für die
  Denoise-Kette (phase_03), niedrige Priorität, da H1/H2 bereits bandweise
  fusionieren.
- **D4 Perceptual-Training (A1-Erweiterung):** Maskierungs-Loss (A1) künftig
  auch für APPLADE (DGT-Domain-Variante) und weitere Finetunes — die
  Waveform-Variante existiert und ist im EAR-VAE-Finetune aktiv.

## 5. ML-Trainingsdomänen-Matrix — welche Modelle brauchen Musik-/Gesangs-Finetuning?

Befund (2026-09-13, verifiziert gegen Plugin-Header/Modellherkunft):
Mehrere verdrahtete Modelle sind auf SPRACHE vortrainiert, werden aber auf
MUSIK eingesetzt — der MP-SENet-Benchmark (−5,9…−8,5 dB out-of-domain) ist
der Beweis, dass das Ohr das hört. Die A1-Maskierungs-Loss-Pipeline (torch)
steht für jeden Waveform-Finetune bereit; die EAR-VAE-/APPLADE-Runs liefern
das bewährte Vorgehen (Encoder-Frozen, Validierungs-Early-Stop, Never-worsen).

| Modell | Trainingsdomäne | Aurik-Einsatz | Musik-Fit | Maßnahme |
|---|---|---|---|---|
| FlashSR (HierSpeech++) | Sprache (16k→48k) | phase_06/07 Bandbreiten-Extension (B4-gegated) | ⚠️ out-of-domain | **ML-V1: Musik-Finetune** (MUSDB-HQ lowpass→original, A1-Loss) |
| BigVGAN-v2 | Sprache (LibriTTS-Klasse) | Vocoder-Repair (B5-gegated) | ⚠️ out-of-domain | **ML-V2: Musik-/Vokal-Finetune** für den Repair-Pfad |
| MP-SENet | Sprache (Interspeech 2023) | phase_43 ML-De-Esser (streng gegated) | ❌ gemessen −5,9…−8,5 dB | **ML-V3: Musik-Finetune ODER De-Wiring** aus Musikpfaden |
| UTMOS | Sprache (MOS) | C1-Zeuge (delta-basiert) | ⚠️ Bias möglich | **ML-V4: Musik-MOS-Validierung/Kalibrierung** (nur Delta-Urteil, bereits so verdrahtet) |
| DeepFilterNet v3.II | Sprache + Musik-Variante | Denoise-Kandidat | ✅ use_df_musik aktiv (verifiziert) | keine |
| CREPE/FCPE | Musik (Pitch) | Wow/Flutter-Schätzung | ✅ | keine |
| Demucs/HTDemucs | MUSDB18 (Musik) | Stem-Separation | ✅ | keine |
| CQT-Diff+ | Musik (Declip) | Declip-Zweig | ✅ | keine |
| KIM Music/Vocal | MDX23C (Musik) | Klarheit | ✅ | keine |
| BANQUET | Vinyl (Knistern) | Crackle/Klick | ✅ | keine |
| Whisper/Resemblyzer | Sprache | Lyrics/Stimm-Identität | ✅ zweckrichtig (Gesang) | keine |
| GaCELA | **Musik** (Maestro + FMA — verifiziert) | Lang-Lücken-Inpainting (375–1500 ms) | ✅ musik-nativ | **Kein Training nötig** — INTEGRATION: ltfatpy-Blocker lösen/Inverter portieren + IN-V1/V2-Gates (einziger musik-nativer Lang-Lücken-Inpainter) |
| AERO | 12k→48k SR (Rolle = FlashSR) | nicht in Produktion | ⚠️ Redundant zu FlashSR | **Kein Training** — nur Benchmark AERO vs. FlashSR auf Musik als Kandidatenwahl für ML-V1 (§V7: eine Lösung pro Rolle) |

Priorität: ML-V1 (FlashSR — direkt im HF-Hörbereich; AERO-Benchmark als Kandidatenwahl) →
ML-V2 (BigVGAN) → ML-V3 (MP-SENet entscheiden) → ML-V4 (UTMOS-Kalibrierung) →
GaCELA-Integration (musik-nativ, kein Training).

## Priorisierte Gesamt-Reihenfolge

1. WF-V1 → WF-V2 → WF-V3 (Wow/Flutter, größter Einzelhebel)
2. ML-V1 → ML-V2 → ML-V3 → ML-V4 (Musik-Finetunes der sprach-vortrainierten Modelle)
3. IN-V1 + IN-V2 (Inpainting-Naht-Gates, B8)
4. C4 DDSP (EQ/Dynamik-Parameter-Prädiktion)
5. TP-V1 + TP-V2 (D-Klasse Transienten)
6. HR-V1 (BigVGAN-Harmonik), CR-V1 (BANQUET-Klick-Detektion), DR-V1 (neurales RT60)
7. WF-V4 (neuraler Warp-Schätzer) + WF-Cassette (Scrape-Flutter) + D3/D4
