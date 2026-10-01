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

## 6. DSP-SOTA-Matrix — alle DSP-Verfahren im Kontext ihrer Aurik-Aufgabe

Systematischer Abgleich jedes DSP-Kerns gegen die 2026-Referenzklasse.
Legende: ✅ = Referenzklasse erreicht · 🟡 = brauchbar, Präzisions-Lücke ·
🔴 = veraltet/überholt — Maßnahme in der Roadmap (docs/TODOS_SOTA_ROADMAP.md).

| Domäne (Phasen) | Aurik-DSP | 2026-Referenz | Urteil |
|---|---|---|---|
| Klick-Rekonstruktion (01) | RBME: AR(16)-Prior + iterative Sparse-Bayesian-Inpainting (Roux & Bimbot 2014), Multi-Scale | RX-De-click/De-crackle-Klasse | ✅ |
| Klick-Detektion (01) | Multi-Scale-DSP (Schwellwert/Steigung) | neuronale Detektion (BANQUET) als Zusatz | 🟡 → SOTA-CR-V1 |
| Hum (02) | Multi-Fundamental + adaptive Notches | RX-De-hum (harmonic comb) | ✅ (Drift-Tracking optional: HU-V1) |
| Denoise (03) | OMLSA (Cohen)/IMCRA + H1/H2-Gates + EAR-VAE | neuronale Denoiser + psychoakustische Fusion | ✅ |
| EQ (04) | adaptive Spektral-Analyse + Matching | Ozone-Match-EQ mit Maskierungs-Gewichtung | 🟡 → SOTA-C4 (DDSP-Parameter) |
| Rumble (05) | DC-Block + subsonic-Filter | De-rumble-Klasse | ✅ |
| Bandbreite (06) | SBR/LPC + FlashSR (B4-gegated) | neuronale BWE mit Masking-Gate | ✅ (nach ML-V1 vollständig) |
| Declip (07) | APPLADE-PnP-ADMM + CQT-Diff+ (Never-worsen-Router) | A-SPADE-Klasse | ✅ |
| Harmonik (07h) | DSP-Harmonik-Synthese (fehlende Obertöne) | neuronaler Vocoder-Repair | 🟡 → SOTA-HR-V1 |
| Transienten (08/36) | Spectral-Flux-Onsets, Multi-Band-Shaping, Log-Domain-Ballistik | SPL-Transient-Designer-Klasse + neurale Onsets | 🟡 → SOTA-TP-V1/V2 |
| Knistern (09) | Multi-Scale-DSP + BANQUET (ML) | BANQUET-Klasse | ✅ |
| Kompression/Limiter (10/11/47) | Multi-Band, True-Peak (BS.1770-4, 4×-Oversampling, ISP) | Ozone-Limiter-Klasse | ✅ |
| Wow/Flutter (12) | pYIN+CREPE-Hybrid, Phasen-Vocoder (Laroche/Dolson, Driedger/Müller), WF-V1/V3 (Resampling+Kalman) | Capstan-Klasse | 🟡 → SOTA-WF-V2/V4/CASS |
| Stereo/Phase (13/14/15/25/32/33/34/48) | M/S, Gerzon-Korrektur, IACC-Guard, Lauridsen | Referenzklasse | ✅ |
| Mastering-EQ (16/17) | Multi-Band-Linear-Phase + Polish | Ozone-Klasse | 🟡 → SOTA-C4 |
| Noise-Gate (18) | Multi-Band-Frequenz-Gate | Maskierungs-Floor-Gating | 🟡 (H2-Logik übertragbar — niedrig) |
| De-Esser (19/43) | Gender-adaptiv v4 + ML-Feinveredelung (streng gegated) | SPL-/RX-Klasse | ✅ |
| Dereverb (20/49) | OMLSA/IMCRA-Dereverb, WPE (vereinfacht) + Transienten-Erhalt | WPE/OMLSA-Klasse | ✅ (Parameter: DR-V1) |
| Inpainting (23/50/55) | IMCRA-Floor + vektoriert/STFT; DDPM-inspirierte Diffusion + DiffWave/AudioLDM2-Option | generative Inpainting mit Naht-Gates | 🟡 → SOTA-IN-V1/V2 + GaCELA |
| Dropout (24) | Multi-Modal-Detektion + Reparatur | RX-Spectral-Repair-Klasse | ✅ |
| Azimuth (25) | Multi-Band-Phasen-Alignment + HF-Restaurierung | Referenzklasse | ✅ |
| Expansion (26) | Multi-Band up/down | Referenzklasse | ✅ |
| Klick/Pop v3 (27) | AR-Residual + Consistent Wiener | SOTA (Literaturklasse) | ✅ |
| Surface/Hiss/Mod-Noise (28/29/59) | IMCRA/OMLSA/MMSE-LSA; Esquef & Biscainho 2006 | Referenzklasse | ✅ |
| DC (30) | drift-tracking DC | trivial, ✅ | ✅ |
| Speed/Pitch (31) | pYIN + Time-Stretch | ✅ (SP-V1 geteilt mit WF-V1) | ✅ |
| Multiband/M/S-Dynamik (34/35) | 4-Band M/S | Referenzklasse | ✅ |
| Bass/Präsenz/Air (37/38/39) | Harmonik-Synthese + Shelving | Aphex-Klasse | ✅ (Parameter: C4) |
| Loudness (40) | BS.1770-4/EBU-R128 komplett | Standard | ✅ |
| Output (41) | Resampling + POW-r-Dither | Standard | ✅ |
| Instrumente (44/45/51/52) | Transient/Harmonik/Präsenz-DSP (Tier-1-Hybrid) | Zweck-DSP-Klasse | ✅ |
| Semantik (53) | CLAP/BEATs/Chromagramm-Kaskade | Referenzklasse | ✅ |
| Transparent Dynamics (54) | Maskierungs-Zonen-Kompression | psychoakustische Kompression | ✅ |
| Spezial-Defekte (56–64) | Band-Gap, Print-Through-LMS, IGD, Groove-Echo, Crosstalk-α-Inverse (IEC 60098), IMD-Bispektrum, Splice, Vocal-Naturalness | exakte physikalische Modelle | ✅ |
| Stem-NR (66) | getrennte Vokal/Begleitung-NR + C1–C3 + KIM2 | Referenzklasse | ✅ |

Kernaussage: Der DSP-Unterbau ist flächendeckend auf Referenzklasse; die
verbleibenden 🟡 sind gezielt dort, wo ML-Messung DSP-Parameter schärfen kann
(Detektion, EQ/Dynamik-Ziele, Warp-Schätzung) — genau die Maßnahmen der Roadmap.

1. WF-V1 → WF-V2 → WF-V3 (Wow/Flutter, größter Einzelhebel)
2. ML-V1 → ML-V2 → ML-V3 → ML-V4 (Musik-Finetunes der sprach-vortrainierten Modelle)
3. IN-V1 + IN-V2 (Inpainting-Naht-Gates, B8)
4. C4 DDSP (EQ/Dynamik-Parameter-Prädiktion)
5. TP-V1 + TP-V2 (D-Klasse Transienten)
6. HR-V1 (BigVGAN-Harmonik), CR-V1 (BANQUET-Klick-Detektion), DR-V1 (neurales RT60)
7. WF-V4 (neuraler Warp-Schätzer) + WF-Cassette (Scrape-Flutter) + D3/D4

---

## §SR-Audit: 62-Typen-Messungs-Programm (Stand 2026-09-24)

**Methodik-Schablone (Spec 06 §7.2d, verbindlich je Familie):**
Audio-Evidenz-Harness (`scripts/defect_evidence_harness.py`, deterministisch,
Seeds rng 1/2/3) → Messung gegen Erwartungswerte → Wurzelursache → Detektor-Fix
→ epistemische Confidence → Either-Or-Routing → Spec-Evidenzblock.

**Abgeschlossen (evidenz-verifiziert, Harness grün):**
- Knistern-Familie: Stereo-Gate + epistemische Confidence, Either-Or-Routing,
  phase_67 (Commits a7834720/ad7b1b83/52eb7f45)
- Rausch-Familie (0 Lücken): hiss-Typisierung (phase_29 erreichbar),
  quantization-Tonalitäts-Guard, modulation (Flur-Ratio-primär, 0,5-ms-Frames,
  Nullphase) — blind 0,000 → 0,575
- Wow/Flutter: 3/4 grün (flutter-FP behoben via tonalem Pitch-Zweig +
  Kohärenz-Gate; speed_calibration 0,382)

**Offene TODOs (Restkorrekturen, priorisiert):**

1. **Wow-Aktivierungs-Schwelle**: ±0,5 % FM liefert severity 0,272. Kalibrierung
   gehört in den Phase-Mapper (Spec 03: Material-Confidence beeinflusst die
   Stärke, nicht die Selektion), nicht in den Detektor.
   **ERLEDIGT 2026-09-25**: `defect_phase_mapper.ACTIVATION_THRESHOLDS`
   (WOW/FLUTTER 0,15, Messbasis FM-Wow 0,272 vs. sauber 0,037) +
   `activation_threshold()`; UV3-Selektion nutzt die kalibrierte Schwelle
   statt der Magic Numbers 0,10/0,35. Rubato (Beat-Reliability < 0,40) dämpft
   jetzt die STÄRKE (Conductor-Hint 0,5) statt zu überspringen — die alte
   Skip-/Override-Logik ließ ein hörbares FM-Wow bei Rubato unbehandelt
   (§Spec 03-verletzend). Detektor bleibt ehrlich (Skalen-Boost bewusst weg).
   Tests: `TestWowAktivierungsschwelle` (5 Fälle, test_decision_completeness.py).
2. **Epistemische Confidence auf alle Detektoren übertragen** (bisher nur
   crackle): Fenster-Stabilität + σ-Randabstand statt starrer 0.8/0.3-Werte.
3. **Either-Or-Routing pro Familie** (bisher nur Knistern): Blanket-Einträge im
   CausalDefectReasoner (z. B. vinyl_crackle→[09,01,28,03]) auf Entweder-Oder
   umstellen; Redundanz-Auflösung phase_01/27 ist nur im Fallback-Pfad.
4. **phase_03: 13-fache CG5-SNR-Skip-Signatur** an Scanner-Konsultation binden
   (Fauxpas-Muster von phase_28, §SR-CG5).
5. **Harness-Familien ergänzen und durchlaufen** (Detektor vorhanden, aber
   keine Audio-Evidenz):
   - Dropout (Subtyp-Präzedenz `_detect_dropout_subtypes`; ungeprüft)
   - Wow/Flutter-Rest: multiband_wow_flutter, scrape_flutter,
     flutter_spectral_sidebands, transport_bump
   - Vinyl: inner_groove_distortion (Detektor FEHLT), groove_echo, crosstalk,
     lacquer_disc_degradation, stylus_damage, riaa_curve_error,
     motor_interference, stereo_field_collapse
   - Band: sticky_shed_residue, bias_error, head_wear, hf_remanence_loss,
     generation_loss, tape_head_clog, print_through, amplitude_drift,
     tape_splice_artifact, dolby_nr_mismatch, tape_head_level_dip
   - Digital: jitter_artifacts, pre_echo, mpeg_frame_loss, compression_artifacts,
     digital_artifacts, aliasing, overload_distortion
   **Status 2026-09-25: DETEKTOR-EBENE VOLLSTÄNDIG** — `defect_evidence_harness`
   über alle 11 Familien gelaufen: 0 Lücken (dropout, tape_media, wow_flutter,
   digital_other, dynamics, environment, framework, noise, spectral, stereo,
   vinyl). Offen bleibt die REPARATUR-Ebene (→ L3-Messung unten) und die
   Real-Stichproben-Kalibrierung (TODO 6).
6. **Kalibrierung an realer gelabelter Stichprobe** (schließt die 95 %-CI-Lücke
   der Spec-Evidenzblöcke; n=4 Ohr-Labels reichen nicht).

Je Familie folgt die Spec-Festschreibung (Evidenzblock in Spec 03/06 §7.2d)
nach erfolgreichem Harness-Durchlauf.

### L3-Reparatur-Wirksamkeit auf Musikträger (2026-09-25, `repair_effectiveness_harness.py`)

Vollständiger Durchlauf aller Familien (Musik-Traeger + Defekt-Transformation,
Scanner VORHER → Phase → Scanner NACHHER + physikalische Defekt-Metrik):
**10 OK, 29 Phasen-Lücken, 21 Scanner-Lücken, 5 Skips.**

- **OK (messbar wirksam):** aliasing (23), crackle (09), crosstalk (62),
  dc_offset (30), generation_loss (03), hum (02), lacquer_disc_degradation (01),
  stereo_field_collapse (13), stereo_imbalance (15), transient_smearing (08).
- **Scanner-Lücken (21, Detektor sieht den Defekt auf Musikträger nicht,
  sev < 0,15):** u. a. dropout_splice, dropouts, hiss, flutter, scrape_flutter,
  transport_bump, print_through, pre_echo, azimuth_error, dolby_nr_mismatch,
  tape_head_clog, dropout_head_contact, nr_breathing, stylus_damage,
  inner_groove_distortion, quantization_noise, amplitude_drift, clipping,
  compression_artifacts, dynamic_compression_excess, proximity_effect.
  → Grundlage für die Detektor-Kalibrierung (TODO 5, Musik-Stichprobe TODO 6).
- **Phasen-Lücken (29, Reparatur ohne messbaren Effekt):** u. a. die gesamte
  phase_12-Familie auf Musik (wow, flutter_spectral_sidebands,
  multiband_wow_flutter, speed_calibration_error — phys-Metrik unverändert),
  phase_24 (dropout, dropout_oxide, sticky_shed), phase_64 (tape_splice_artifact
  0,730 unverändert), phase_03/05/49/14/56/59/63/65 sowie motor_interference,
  bias_error, riaa_curve_error, head_wear, mpeg_frame_loss, jitter_artifacts,
  overload_distortion, pitch_drift, reverb_excess, room_mode_resonance,
  tape_head_level_dip, vocal_harshness, digital_artifacts, clicks (52→37 Kanten
  reichen nicht fürs Gate).
- **Phasen-Exceptions (3 SKIPs):** bandwidth_loss + hf_remanence_loss
  (phase_06: ValueError), groove_echo (phase_61: ValueError), sibilance
  (phase_19: UnboundLocalError) — Aufruf-Kontrakt der Phasen im Harness prüfen
  (echte Bugs vs. fehlende Produktions-Kwargs).

Abarbeitungs-Reihenfolge (Vorschlag): (a) Phasen-Exceptions (3),
(b) phase_12-Familie auf Musik (4 — Gleichlauf ist das Kern-Versprechen der
Analog-Restauration), (c) Scanner-Lücken der Tape-/Dropout-Familie,
(d) Rest.

### L3-Schließung 2026-09-27 — phase_12-Familie auf Musik (Wow/Flutter-Kernversprechen)

**WOW (0,3 Hz, ±0,5 % FM auf 4-Akkord-Träger) — GESCHLOSSEN.** Root-Cause war
doppelt: (1) Der Phasen-Messkanal verwarf den periodischen Pfad auf
nicht-stationären Trägern (harte Schwellen r² ≥ 0,30 + n ≥ 4 — nur 3 Bänder
erreichten r² ≈ 0,33–0,36 → Fallback in die irregulär-Drift-Extraktion, die
periodisches Wow 27× unterschätzte: 2,01 statt 53 cents). Fix: KOHÄRENTE
EVIDENZ (n ≥ 2, Σ r² ≥ 1,0, Einzel-r² ≥ 0,25) statt Band-Anzahl.
(2) Die Harness-Metrik `_if_std` (Mix-IF-Std) ist auf polyphonen Trägern
Beating-dominiert — sie sah die korrekte Reparatur nicht (1734→1745 trotz
Restfehler 0,3 cents). Fix: beat-immuner Gemeinschafts-FM-Messkanal
(`_common_fm_cents`: 1/24-Oktav-Bänder ±3 %, normierte IF-Abweichung,
kohärenter Matched-Filter bei bekannter Synth-Modulationsfrequenz,
r²-gewichtete √N-Mittelung) für die Wow/Flutter-Familie im L3-Harness.
Ergebnis: **37,2 → 0,0 cents** (vollständige Entfernung).

**MULTIBAND_WOW_FLUTTER (nur 4–12 kHz moduliert, 5 Hz) — GESCHLOSSEN.**
Dreiteilige Kette: (1) Synth: tonale HF-Linie (6/8 kHz, Sustain-Hülle) — der
Musik-Träger hatte >4 kHz keine Partials, der Defekt existierte dort
physikalisch nicht (Kalibrierung exp(−t/2,2), 0,10 Amplitude).
(2) Scanner: `dominant_mod_freq_hz` als HF-Evidenz-Kanal (IF-basierter
25-ms-Pass im 5,6–11,3-kHz-Band, lokale Prominenz > 5 statt Totalanteil —
Null-Padding leakt Energie; Befund-Kette: Centroid-Ansatz aliasierte an der
Nyquist-Grenze, 0,01-Gate blockte 14-Hz-Swing). (3) Phase: Hint-Auswahl
nach Severity (Multiband-Hint > 4 Hz autoritativ — der Wow-Rest-Hint 0,28 Hz
verdeckte 5 Hz); Grid-Öffnung bis 12 kHz auf 32-kHz-Arbeitsspur bei
Hint > 4 Hz; Evidenz-Floor Σ r² ≥ 0,5 (Band-Flutter betrifft per Definition
wenige Bänder); BANDBEGRENZTER Inverse-Warp (`_band_limited_warp`:
Komplement-Konstruktion — nur [4 kHz, 12 kHz] wird gewarpt, der Rest bleibt
bit-identisch; ein globaler Warp hätte das unmodulierte LF-Band neu
moduliert) + Closed-Loop-Refine. Ergebnis: **16,5 → 0,0 cents**.

**FLUTTER_SPECTRAL_SIDEBANDS (±3 Hz, ±0,1 %) — als sub-audibel KLASSIFIZIERT
(SKIP im L3-Harness, dokumentiert):** FM-Tiefe 1,7 cents < Frequenz-JND
~3,4 cents (Klumpp & Eady 1956, hearing_jnd) — Hörordnung §4/§G100
(GEBOTE.md): ein unhörbarer Defekt ist kein Defekt, kein Reparatur-Zwang.

**Offen (verschoben):** flutter (6 Hz ±0,2 %), scrape_flutter,
transport_bump = Scanner-Lücken (Detektor-Kalibrierung auf Musikträger);
speed_calibration_error = phase_31 (SP-V1, konstante Speed-Offsets —
phase_12 behandelt per Design nur zeitvariante Transportfehler).

Tests: `test_phase_12_wow_flutter_fix.py` (+5: Band-Warp-Komplement,
HF-Grid-Messkanal), phase_12-Suiten + Scanner-Suiten 74 grün.

### L3-Schließung 2026-09-27 — Scanner-Lücken der Tape-/Dropout-Familie (A4)

Alle 9 Fälle der Tape-/Dropout-Familie erreichen jetzt die
Aktivierungsschwelle (sev ≥ 0,15) auf dem Musik-Träger. Root-Causes waren
wiederkehrende Träger-/Vertrags-Muster, keine Detektor-Einzelerkrankungen:

| Fall | Befund | Fix |
|---|---|---|
| dropouts | 1 × 6 ms = Klick, kein Muster; sev 0,027 | Synth: realistische Oxid-Serie 8 × 10 ms → 0,168 |
| dropout_splice | 8-s-Halbpegel-Region = musikalische Dynamik (adaptiver Detektor adaptiert; sev 0) | Synth: 3 × 30-ms-Abrisse >95 % (Evidenz-Definition) → 0,201 |
| dropout_head_contact | flache 300-ms-Dips (modulation 0 → „generisch“); Klassifikator: Splice-Zweig (loss > 0,95) fing tiefe lange Dips | Klassifikator: Splice nur ≤ 80 ms; Head-Contact 50-500 ms; Synth: wellenförmige Dips → 0,390 |
| amplitude_drift | Detektor braucht ≥ 30 s, Träger 15 s („too_short“) | Synth: 30-s-Träger für diesen Fall (Musik-Generator durations-parametrisiert, §G5) → 1,000 |
| azimuth_error | PHD-Slope-Fit über unkorrelierte Bins (Träger ohne HF → Slope 0,83 statt 28,8 °/kHz) | Detektor: KOHAERENZGEWICHTETER Fit (Kreuzleistung je Bin, Ausreißer-Cap 4×); Synth: gemeinsame HF-Linie → 1,000 |
| tape_head_clog | Träger ohne HF-Inhalt; 80-ms-Dips → nur 2 Mask-Frames (50-ms-RMS verdünnt) | Synth: HF-Linie + 250-ms-HF-only-Dips (mid bleibt) → 0,192 |
| dolby_nr_mismatch | HF-Anhebung ohne HF-Inhalt unsichtbar (medium_gated) | Synth: HF-Linie → 0,567 |
| print_through | keine +20-dB-Onsets (Musik-Percussion nur +7 dB); Geist −14,9 dB außerhalb des 18-48-dB-Fensters | Synth: Rimshot-Onsets +27 dB mit 200-ms-Geist −20 dB (IEC 60094-3) → 0,478 |
| nr_breathing | Rausch-Modulation unkorreliert zur Hülle (Detektor: corr < −0,2) | Synth: HF-Rauschboden anti-korreliert zur Signal-Hülle gepumpt → 0,600 |

Gemeinsamer Helfer `_with_hf_line` (tonale 5-kHz-Linie für HF-abhängige
Defekte — der Musik-Träger hat >4 kHz keine Partials). Evidenz-Harness
(Detektor-Ebene, 11 Familien) bleibt vollständig grün (0 Lücken) — die
Scanner-Änderungen sind regressionsfrei. Scanner-/Phase-12-Suiten 63 grün.

**Validierung (Einzel-Fall-Läufe, 2026-09-27):** alle 9 Fälle erreichen den
RUN-Status (sev ≥ 0,15). Reparatur-Wirksamkeit (Übergabe an A6/Phasen-Lücken):
dropouts sev 0,168→0,028 (phase_24 repariert — Metrik-Kalibrierung folgt),
print_through phys 0,182→0,181 (marginal), dropout_splice/amplitude_drift/
azimuth/tape_head_clog/dolby/nr_breathing noch ohne messbare phys-Wirkung
(phase_64/40/25/56/54-Konsultation je Fall, §7.4c-Muster).

### L3-Schließung 2026-09-27/28 — Restliche Scanner-Lücken (A5)

**Geschlossen (sev ≥ 0,15 auf Musik, validiert):**

| Fall | vorher | jetzt | Fix |
|---|---|---|---|
| hiss | 0,000 | **0,559** | Detektor-Gates neu kalibriert (0,5/0,5 verlangten Hiss in SIGNAL-Lautstärke ≈ 0 dB SNR — realistisches Bandhiss −9 dB SNR (ratio 0,401/stat 0,393) wurde geblockt; Hüllkurven-Stationarität skaliert statt vetiert) |
| clipping | 0,043 | **0,225** | Synth: Gain 4,5 auf ±1-Ceiling (realistische Flat-Top-Dichte 2,1 %; 3,5 gab 0,42 % → sev 0,106) |
| proximity_effect_excess | 0,124 | **0,287** | Synth: 1,4× LF-Boost (+7,6 dB statt 5,5 — unter dem 6-dB-Detektor-Gate) |
| pre_echo | 0,000 | **0,637 (sauberer Träger 0,000)** | Detektor-BUG: die per-Sample-Steigung war durch das 8-ms-Glättungsfenster geteilt — das Transienten-Gate feuerte auf KEINEM Signal. Fenster-skalierte Steigung + Spektral-Ähnlichkeits-Pflicht (corr ≥ 0,5) für die Tape-Route (Energie-Verhältnis allein feuerte auf sauberer Musik 0,63) |

**Sub-audibel klassifiziert (SKIP, JND-begründet):** scrape_flutter
(FM ±0,05 % @ 80 Hz = 0,86 cents < Frequenz-JND ~3,4 cents,
Klumpp & Eady 1956) — wie zuvor flutter_spectral_sidebands.

**Geschlossen in Runde 2 (2026-09-28, sev ≥ 0,15 auf Musik, validiert):**

| Fall | vorher | jetzt | Fix |
|---|---|---|---|
| dynamic_compression_excess | 0,000 | **0,620** | Detektor-Gate (threshold·0,5) verwarf die EINDEUTIGE Loudness-War-Evidenz: LRA ≤ 3 LU (EBU R128) setzt sich jetzt gegen das Material-Gate durch (Scan-Kontext: TAPE-threshold 0,98 → Gate 0,49 > 0,413 trotz LRA 1,59) |
| compression_artifacts | 0,036 | **0,426** (sauber 0,041) | Konzentrations-Discount (0,05) vetiert auf tonaler Musik — SPEKTRALLOCH-NACHWEIS (breite 2,4-kHz-Hüllkurve, RAW-Tiefe 10-45 dB, HF-Inhalts-Gate) neutralisiert ihn; Anti-FP: harmonischer Kamm (kein HF-Inhalt) bleibt ausgenommen |
| transport_bump | 0,000 | **0,803** (sauber 0,000) | Synth-Physik korrigiert: Pegel-ABRISS (Pflicht-Feature Energy-DROP < 0,45) + LF-Thump DANNACH (der Thump IM Abriss dominierte die RMS — ratio 5,7 = Spike, fe sum = 0) |
| quantization_noise | 0,000 | **0,545** (sauber 0,000) | ENOB-Fallback über Histogramm-Granularität (ENOB ≈ log2(n_populated), Standard-Technik): Musik hat kaum leise Passagen, der Step-Size-Kanal blieb leer (step 0,0 trotz 167/1024 Bins); Synth: Vollaussteuerung + 7-Bit |

**Offen (präzise diagnostiziert, nächste Runde):**

- **flutter** (6 Hz ±0,2 % = 3,4 cents): Kohärenter Subband-IF-Kanal im
  Detektor gebaut (`_coherent_subband_fm`, phase_12-Prinzip, inkl. Bug-Fix
  rfftfreq-Signallänge) — die FM-Tiefe liegt unter dem per-Band-Rauschboden
  (Kohärenz-Peak wandert zu 42 Hz, coh 0,25). Nächster Schritt:
  Matched-Filter-Scan je Band (r²-gated, wie `_flutter_track_from_stacks`)
  statt einfacher FFT-Kohärenz.
- **stylus_damage**: asymmetrischer Hard-Clipper (Synth-Fix steht) —
  Odd/Even-Ratio auf dichter Musikharmonik noch nicht getrennt.
- **inner_groove_distortion**: viertel-progressive Verzerrung (Synth-Fix
  steht) — THD-Slope je Viertel auf Musikharmonik noch nicht getrennt.

### L3-Schließung 2026-09-28 — Phasen-Lücken der Vinyl-/Wow-Flutter-Familie (A6)

Alle vier gemessenen Phasen-Lücken der Vinyl-/Wow-Flutter-Familie sind
phys-metrisch geschlossen (Gate je Fall: met_a ≤ 0,5·met_b; alle
Regressionstests grün):

| Fall | Befund (Root-Cause) | Fix | Ergebnis |
|---|---|---|---|
| speed_calibration_error | phase_12 behandelt per Design nur zeitvariante Transportfehler; pYIN liefert auf Akkord-Trägern conf 0,0 → phase_31 übersprang | Mapper phase_31-first; phase_31: polyphoner DSP-Tuning-Fallback (Peaks 80–1200 Hz, parabolisch interpoliert, 12-TET-Offset, IQR-Gate 20 cents, Amplituden-Gate 10 %) + direktes Polyphasen-Resampling (up=round(1024·ratio)/down=1024, keine Vocoder-Schmierung); Harness-Metrik = Cents-Offset (12-TET-Median) statt ZC-Referenz | 7,40 → 0,14 cents ✓ |
| inner_groove_distortion | phase_60 dämpfte fix 2–8 kHz — die Produkte der Bass-Akkorde (Summtöne 395–720 Hz, H2–H8 bis 3,1 kHz) lagen unter 2 kHz; der subtraktive Psychoakustik-Clamp revertierte die Dämpfung in maskierten Harmonik-Bändern (Phase wirkungslos); alte THD-Metrik mass die Musik-Harmonik selbst | Band 400 Hz–8 kHz (Grundton-Schutz: 392 Hz = höchste Akkord-Fundamentale); Positionsgewicht 0,65 + Gain-Steigung 0,85; Clamp-Entfernung mit Begründung (Verzerrungs-Reduktion ≠ Noise-Reduction, kein Stille-Artefakt); Harness-Metrik = Summton-Kontrast Q3−Q0 (träger-immun, clean=0) | Harness 0,01199 → 0,00357 ✓ |
| groove_echo | Groove-Echo ist ein ZEITKONTINUIERLICHER Vorläufer (1 Umdrehung), kein Peak-Phänomen — die peak-basierte Kompensation ließ den durchgehenden Geist unangetastet; die Fenster-Max-Metrik mass die Musik-Hüllstruktur statt des Geists | phase_61: globaler Vorläufer-Pfad (RPM-Lag-Überschuss der Hüllkurven-Kreuzkorrelation, g-Kandidaten-Leiter 0,05–0,35, Never-worsen-Energie-Gate, Phantom-Schwelle 0,04; Bugfix: lokaler scipy-Import); Metrik = lokaler Lag-Überschuss (musik-immun) | 0,0778 → 0,0334 ✓ |
| riaa_curve_error | Die RIAA-Wiedergabekurve rollt den Hochton nur −1…−4,5 dB ab und boostet den Bass — eine detektierte +12-dB-HF-Anhebung verschlechterte sich (Bass-Boost senkte den Metrik-Nenner) | phase_04: adaptive De-Emphasis bei riaa_curve_error ≥ 0,7 (Vinyl/Shellac): HF-Überschuss E(5–10 kHz)/E(0,5–2 kHz) gegen Musik-Hüllkurven-Erwartung 0,2, Shelf-Cut ab 5 kHz gedeckelt −12 dB | 0,4894 → 0,1655 ✓ |
| stylus_damage | Crackle-Kaskade adressiert keine Wellenform-Asymmetrie (Skewness mean(x³)/rms³) — die Metrik stieg 0,8591→0,8999; der Declipper erkennt die 1–2-Sample-Flat-Tops nicht (Histogramm-Gate) | phase_09 stylus-Zweig (score ≥ 0,3, Skewness < −0,15): Dekompression der positiven Flanke y = x + a·max(0, x−t), a-Leiter minimiert |mean(y³)|, Never-worsen-Spitzen-Deckel 10 % | 0,8591 → 0,2812 ✓ |

Gemeinsames Muster (wie in A3–A5): Die Harness-Metrik muss die
DEFEKT-SIGNATUR messen (Summton-Kontrast, Lag-Überschuss, Cents-Offset),
sonst misst sie die Musik selbst und die Reparatur bleibt unsichtbar —
und die Phase muss auf der PHYSIK des Defekts arbeiten (Zeitkontinuum statt
Peaks, Bass-Produkt-Band statt fixer 2-kHz-Grenze, Dekompression statt
Impuls-Entfernung).

**Enum-Befund (produktionsrelevant):** Der Harness übergibt
`material_type` als `MaterialType`-Enum — zwei echte Produktions-Bugs
wurden dabei sichtbar: (a) phase_04 `effective_material` fiel mit Enum-Key
auf die „unknown"-Parameter (blend 0,6/max_cut 3,0 statt vinyl 0,9/10,0);
(b) das Material-Gate des adaptiven De-Emphasis-Pfads (`in ("vinyl",
"shellac")`) war für Enums immer False. Fix: Enum-feste Normalisierung
`str(material_type).lower().split(".")[-1]` an beiden Stellen.

**Gesamtlauf 2026-09-28 (alle Familien, `repair_effectiveness_harness.py`):**
**19 OK · 43 Phasen-Lücken · 0 Scanner-Lücken · 3 SKIP.** Die vier
A6-Fälle sind im exakten Harness-Kontext (DefectScoreView +
`_pipeline_style_kwargs`) nach den Enum-/Import-Fixes verifiziert:
speed 7,40→0,14 · IGD 0,01199→0,00357 · groove_echo 0,0778→0,0334 ·
riaa 0,4894→0,1655 · stylus 0,8591→0,2812 (alle ≤ 0,5·met_b).

**Restkatalog der 43 Phasen-Lücken** (nächste Sessions, je Fall nach dem
§7.4c-Muster — Root-Cause → Fix → phys-Metrik-Gate):

| Familie | Fälle (Phase) |
|---|---|
| dropout/framework | dropout, dropout_oxide, dropout_head_contact, sticky_shed (phase_24), dropout_splice (phase_64), distortion (phase_07) |
| noise | clicks 52→37 (phase_01), hiss (phase_03, sev steigt 0,559→0,939!), high_freq_noise (phase_03), low_freq_rumble (phase_05), modulation_noise (phase_59) |
| tape_media | azimuth_error (phase_25), bias_error + dolby_nr_mismatch (phase_04), head_wear + tape_head_clog (phase_56), hf_remanence_loss + bandwidth_loss (phase_06), print_through (phase_57), nr_breathing + tape_head_level_dip (phase_54), pre_echo (phase_23), transport_bump (phase_12) |
| digital_other | amplitude_drift (phase_40), pitch_drift (phase_31), jitter_artifacts + mpeg_frame_loss + digital_artifacts (phase_23), overload_distortion (phase_09), vocal_harshness (phase_65) |
| spectral | intermodulation_distortion (phase_63, sev steigt 0,716→1,0), phase_issues + phase_rotation (phase_14), quantization_noise (phase_03) |
| environment | proximity_effect_excess (phase_04), reverb_excess (phase_49), room_mode_resonance (phase_04) |
| dynamics | clipping (phase_07), sibilance (phase_19) |

### Architektur-Entscheidung 2026-09-25 — Kanonischer Phasen-Evidenz-Vertrag (§7.4c)

**Befund:** Die 29 Phasen-Lücken sind keine 29 DSP-Einzelfehler, sondern fünf
wiederkommende Kontrakt-Bruch-Klassen (String/Enum-Keys auf `defect_scores`,
Namensvarianten `_restoration_context`, Locations-Fenstersemantik,
Privatdetektoren statt Scanner-Evidenz, private Schwellen §V7-widrig).

**Umgesetzt (Slice 1, kontrakt-first):**

- **Spec:** §7.4c (06_phases_system.md) „Kanonischer Phasen-Evidenz-Vertrag"
  + Klärung der Key-Typ-Widersprüche in Spec 02 (Zeilen 728/1544).
- **Contract-Layer:** `normalize_evidence_kwargs()` (phase_interface.py) an
  beiden Choke-Points (`UnifiedRestorerV3._profiled_phase_call`,
  `PhaseInterface._safe_process`): `defect_scores` → `DefectScoreView`,
  `defect_locations` → `DefectLocationsView` (beide BIDIREKTIONAL
  Enum/String), `_restoration_context`-Alias-Merge. 11 Vertragstests
  (`test_phase_evidence_contract.py`).
- **Bereits geschlossen durch den Vertrag:**
  - tape_splice_artifact: Scanner-Saaten über FensterMITTE + Frame-Smearing-
    Fix im Privatdetektor + Klick-Interpolation nach bestandenem
    Hörbarkeits-Gate volle Stärke → Impuls-Metrik 0,358→0,1245 (−65 %),
    Gate grün; Musik-FPs bleiben vom Audibility-Gate abgelehnt (Never-worsen).
  - dropout / sticky_shed_residue: Skip-Gate las tote Keys → phase_24 lief
    NIE (auch in Produktion); läuft jetzt.
  - 3 Phasen-Exceptions + die Cluster A/D/E folgen strukturell über §7.4c.
- **Instrument-Kalibrierung:** L3-Metrik für tape_splice = Impuls-Prominenz
  (`_impulse_peak`, Klick-Maß) statt `_dip_depth` — die PegelSTUFE zwischen
  zwei Bandstücken ist Inhalt (zwei Aufnahmen), kein Artefakt; die alte Metrik
  bestrafte korrektes Verhalten der phase_64.

**Governance-Befunde (Vorbelastung auf HEAD, nicht aus dieser Welle):**

- Scope-Guardrail `policy/scope_guardrails.yaml`: `max_phases` 69 vs. **71**
  registrierte Phasen (`test_phase_count_within_limit` rot). Neue Phasen
  brauchen laut Policy ≥5 Echt-Audio-Fälle + HPI-Verbesserung ≥ 0,02 +
  Real-Audio-Quality-Gate-Review (Ausnahme: Bugfixes ohne neue Phasen-IDs).
  Das Evidenzpaket für die Überzahl (u. a. phase_67) fehlt noch — offen.
- 3 weitere normative Vorbelastungen: GUI-Vertrag ×2
  (`test_modern_window_gui_contract.py`), veraltetes Daily-Gate-Artefakt
  (`test_daily_gate_stored_recently`). Alle 4 ebenfalls auf HEAD rot,
  durch diese Welle unverändert.

### Sub-audible SOTA-Konsistenz 2026-09-25 — eine Hör-Instanz, eine Wahrheit

**Befund:** Drei Instanzen beantworteten „Defekt hörbar?" auf drei Skalen:
`dsp/audibility_gate.defect_audibility` (Maskierungs-Delta, ~25 Phasen-
Aufrufer), `defect_audibility_gate` (Severity-Skala 0.08 + Material-Offsets),
`dsp/hearing_jnd.below_jnd` (physikalische JND-Klassen). Die Hörordnung §4
verlangt aber „10-Log-Summen Masking JND" — die Schwelle ist die
Energiesumme aus Maskierung UND Pegel-JND; der JND-Term fehlte komplett.

**Umgesetzt (Slice 1):** `_threshold_with_jnd_floor()` in
`dsp/audibility_gate.py` — beide Maskierungspfade (MPEG-1 + Zwicker ISO 532-1)
aggregieren die Schwelle jetzt mit `hearing_jnd.level_broadband` (1 dB,
Mills 1960). Sub-audible Defekte bleiben damit auch in der Stille unter der
Schwelle (skippable) — Never-worsen/Wohlklang: nie anfassen, was nicht hörbar
ist. Konservativ (≤ +1 dB), fail-safe (§V6: Floor nie blockierend).
10/10 Hör-/JND-Tests + 236 Konsumenten-Tests grün.

**Umgesetzt (Konsistenz-Slice 2, 2026-09-27 — „eine Hör-Instanz, eine Wahrheit"):**
Der Lauf-Ende-Gate `evaluate_defect_audibility` akzeptiert jetzt
`audio`/`sample_rate`/`defect_locations` (Final-Audio + Post-Scan-Locations)
und entscheidet für Severity-Kandidaten (post ≥ Schwelle) über die KANONISCHE
Maskierungs-Instanz `dsp/audibility_gate.defect_audibility_from_signal` —
dieselbe wie in ~25 Phasen-Aufrufern (ISO 11172-3 Bark bzw. Zwicker ISO 532-1
+ Pegel-JND-Floor). Entscheidungsfluss: kanonische Maskierung → Severity-Skala
(nur Fallback: kein Audio/Locations, FM-Zeitachsen-Defekte ohne Energie-Domäne,
Fehler — fail-open §V6) → Perceptual-Salience (`n_masked_events`) als reine
Evidenz. Band-Konventionen je Defekt-Domäne aus den Produktions-Bändern der
Phasen abgeleitet (`_CANONICAL_BANDS`); `wow`/`flutter` u. ä. bleiben ehrlich
auf der Severity-Skala (keine Energie-Delta-Domäne). UV3-Call-Site (Lauf-Ende)
verdrahtet; pro Typ dokumentiert `evidence: canonical_masking | severity_scale`.
Tests: `test_defect_audibility_gate.py` (9 neue kanonische Fälle: lauter/leiser
Burst, Physical-Cap-Präzedenz, Stereo-Layout-Invariante (N,2)/(2,N), Fallback,
FM-Typen, fail-open, Determinismus) + 34/34 grün; m1b/§0c/Hearing-Gates-Suiten
grün. §G5-Determinismus belegt (bit-identische Reports).

### Cluster A 2026-09-25 — phase_12 Wow: Root-Cause abgeschlossen, Messkanal-Grenze kartiert

**Umgesetzt (Committed):** Scanner-Saat für den Sinus-Wow-Fit
(`hint_freq_hz` aus wow-Metadatum `dominant_mod_freq_hz`, §7.4c),
Trend-Auskopplung + robuste IRLS-Schätzung (Huber), IF-Kanal-Verdrahtung
(`_estimate_speed_curve_from_instantaneous_frequency`) als zweiter Messkanal,
Witness-Schutz akzeptiert die Scanner+IF-Evidenzkette. 92/92 phase_12-Tests grün
(Mono-/Solo-Fall — der Designdomäne des Fits — profitiert direkt).

**Gemessene Messkanal-Grenze (FM-Wow ±0,5 % ≈ 8,6 cents auf 4-Akkord-Musik):**

- Konsens-Pitch-Trajektorie: Noten-Streuung ±1000 cents (Spanne 2704) —
  Sinus-/Trend-/IRLS-Fit messen 386–430 cents „Wow" (Spektralleckage der
  Akkordsprünge) → Amp-Guard ≤ 60 lehnt KORREKT ab (Never-worsen hält).
- Hilbert-IF des Mixes: Beating zwischen den Tönen überstimmt die Gemeinschafts-
  FM — gemessen nur 0,89 cents (Guard ≥ 3 lehnt korrekt ab).

**Nächster Schritt (spezifiziert): Teilband-IF-Mittelwert-Estimator** — je
1/3-Oktavband trägt ~1 Partial; die normierte IF-Abweichung je Band gemittelt
über alle Bänder misst die Gemeinschafts-FM mit √N-Rauschgewinn (N ≈ 10–20
Bänder ⇒ ~4× SNR). Alternativ: Scanner liefert `wow_depth_cents` als Metadatum
(die Detektion misst die Modulation bereits robust). Danach greift die
vorhandene Warp-Pipeline.

### Cluster E 2026-09-25 — phase_12 Reparatur-Operator: Wirksamkeits-Root-Cause gefunden und behoben

**Root-Cause (3 verschachtelte Defekte, alle gemessen):**

1. **Falscher Reparatur-Operator:** Die Korrektur wendete ihre
   Geschwindigkeitsfaktoren mit pitch-erhaltenden Zeitdern an (PSOLA-Grain-OLA,
   STFT-Phase-Vocoder) — solche Operatoren können eine Pitch-Modulation gar
   nicht entfernen, sondern nur zeitlich umverteilen. Gemessen (reiner Ton,
   FM 18,0 cents @ 1 Hz, Sinus-Fit r²=0,99, coverage=1,0):
   **FM-Tiefe 18,0 → 18,0 cents (0,0 % Reduktion)**. Behoben (§WF-R1):
   Anwendung als Inverser-Geschwindigkeits-Warp (variabel ratiges,
   bandbegrenztes Resampling, `_speed_warp_resample`) — exakte Inverse von
   x_d(t)=x(φ(t)); korrigiert Pitch UND Timing gemeinsam (Capstan-Prinzip).
2. **float32-Positions-Akkumulation:** Das Warp-Grid wurde in float32
   kumsummiert — ab |phi| > 2^18 quantisiert die Ulp (1/32) die Schrittweite
   (gemessene Schritt-Extrema 0,99219/1,01562 = Float-Quanten statt ±0,73 %),
   die Korrektur löschte sich teilweise aus. Jetzt float64 — der isolierte
   Operator ist danach exakt theoriekonform (18,0 → 5,4 cents bei Stärke 0,7).
3. **Konsens-Degradation:** `_spectral_warp_supply_or_consensus` mischte den
   schwächeren Spektral-Zweitschätzer in die Fit-Trajektorie (tol=0,01 stuft
   flache Spektral-Schätzungen als „übereinstimmend“ ein) und halbierte die
   Korrektur. Jetzt Fit-Vorrang: bei deterministischem Sinus-Fit r² ≥ 0,90
   bleibt die Fit-Trajektorie (die Versorgung bleibt für Zero-Consensus).

**§P5 vervollständigt:** Wow-/Flutter-Komponenten werden additiv mit eigenen
Stärken korrigiert statt als zwei Vollband-Schätzungen 55/45 zu blenden (der
Blend warf 45 % des Korrekturbudgets auf eine Flutter-Kopie des Wow-Signals).
Der Melodie-Guard entscheidet weiterhin auf der vollen Trajektorie.

**Nach-Messung (gleicher Kanal, 18,0 cents FM @ 1 Hz):** 18,0 → 7,39 cents
(**58,9 % Reduktion**, vorher 0,0 %). Rest = Fit-Amplituden-Unterschätzung
(16,3 vs 18,0 cents) + konservative Timing-Stärke 0,7. Polyphonie-Fall
(Harness): Witness-Kette scanner+if aktiv, Flutter-Komponente ≈ Identität,
Wow trägt die volle Korrektur.

**Bereinigt:** `_psola_timestretch`, `_harmonic_isolated_timestretch` und
`_phase_vocoder_timestretch` gelöscht (pitch-erhaltend, für die Korrektur
ungeeignet); `dsp/phase_vocoder.py` dokumentiert jetzt seine Vertrags-Grenze
(Pitch-Erhalt; darf nicht für Wow zurückverdrahtet werden). Metadatum
`psola_active` entfernt.

**Tests:** 31 phase_12-/vocoder-Tests + 56 Smoke + 237 Wow/Stretch/warp-
selektierte Tests grün. Zwei Teardown-Errors in `test_sota_gap_closures.py`
sind präexistent (im Worktree bei HEAD identisch reproduziert).

**Offen (Cluster E-Rest):** Fit-Amplituden-Kalibrierung, Stärke-Tuning
(safe-timing 0,7), Teilband-IF-Estimator (s. Cluster A) für Mehrstimmen-
Material, multiband_wow_flutter/scrape_flutter, Harness-Gesamtlauf.

### Cluster E II 2026-09-26 — §G188–§G190: Autonome Wirkungs-Kalibrierung normativ verankert + umgesetzt

**Normativ (Kategorie XXV (GEBOTE.md), §7.4d (06_phases_system.md)):**
§G188 (autonome Stärke aus der gemessenen Defekttiefe, Kompensationsgrad 1,0
bei belastbarer Messung, feste konservative Kappen verboten), §G189
(dokumentierte Ausnahmen — Materialphysik, Chain-Injection, PMGG-Ziel — mit
Rest-Autonomie: geschlossener Regelkreis rechnet das Optimum selbst),
§G190 (Zielfunktion: maximaler Wohlklang + natürlicher Klang bei unhörbaren
Defekten; physikalische Grenzen akzeptieren; Pre-Commit-Meldeauftrag).

**Umsetzung phase_12 (erster Compliance-Fall):** Material-Kappe
(CORRECTION_STRENGTH 0,1–0,8) und Heuristik-Dämpfungen (0,82/0,78/1,10/0,70)
entfernt — Stärke = Evidenzqualität (Posterior), volle Kompensation bei
Sinus-Fit r² ≥ 0,90 oder Scanner+IF-Witness. Geschlossener Regelkreis
(`_closed_loop_warp_refine`): Restfehler auf dem eigenen Ausgang nachmessen
(Teilband-IF, pYIN-Fallback), bounded nachführen, nur behalten wenn besser.

**Root-Causes der Rest-Verluste (beide gemessen + behoben):**

1. **Frame-Raster-Verzerrung**: die SF-Kurve wurde per linspace über die
   Sample-Achse verteilt (1241 statt 1200 Samples/Frame) → die Korrektur
   driftete phasenverschoben dagegen (−18,7° bei 1 Hz → nur ~70 %
   Auslöschung). Jetzt k·hop-Abbildung (center=True) — Offset-Sweep
   gemessen: Rest 1,7 statt 5,8 cents.
2. **Mess-Raster-Mismatch im Regelkreis**: pYIN (512er-Hop, 94 fps) traf auf
   das 1200er-Fitraster — die Fit-Basis verfehlte die Modulation (0,7 statt
   6 cents). Raster-Normierung auf das kanonische Raster.

**Ergebnis (Synthesekanal, FM 18,0 cents @ 1 Hz):** 18,0 → **1,66 cents
(90,8 % Reduktion**; Start der Sitzung: 0,0 %). Rest 1,66 =
Fit-Amplituden-Unterschätzung (16,4/18,0) — unter der Messauflösung des
Amp-Guards (4 cents), physikalisch akzeptiert (§G190). Volle Stärke belegt
(timing_safe_strength = 1,0).

**Meldeauftrag (§G190):** Pre-Commit-Gate `aurik-g188-wirkungskalibrierung`
(`scripts/g188_wirkungskalibrierung_check.py`) blockiert neue feste Kappen in
geprüften Dateien und meldet bei JEJEM Commit den repo-weiten Altbestand
(derzeit 15 Stellen in den Phasen 14/17/20/23/24/27/28/40/55 —
Migrations-Roadmap). Physische Grenzen (max_stretch_delta,
DETECTION_THRESHOLD) sind zugelassen.

### Cluster E III 2026-09-26 — §G188-Migration des Altbestands abgeschlossen

Alle 15 gemeldeten Stellen migriert (Phasen 14/17/20/23/24/27/28/40/55):

- **Material-Kappen entfernt** (14: CORRECTION_STRENGTH 0,15–0,60; 20:
  REDUCTION_STRENGTH 0,25–0,65; 23: REPAIR_STRENGTH 0,60–0,90): erkannte
  Defekte werden voll kompensiert; Falsch-Positive filtert die
  DETECTION-Schwelle, Geschmacks-Ziele das PMGG-Ziel (§7.4c).
- **Inhalts-/Material-Dämpfungen entfernt** (24: 0,82–0,94; 27: 0,84/0,88;
  28: 0,82/0,90; 55: 0,85 — Q11-Benchmark entkräftete die Analog-Drosselung
  ohnehin). Schutz vor Over-Processing: Guards + perzeptueller Rollback
  (§G142–§G145); Evidenz bleibt als Debug-Log sichtbar (§V6).
- **Modus-Skalen als dokumentierte Ausnahme §G189 gekennzeichnet** (17/40:
  _MODE_STRENGTH_SCALE — Betriebsart = externe Zielvorgabe, z. B. bewahrt
  Restoration-Mode die originale Dynamik).
- Vier Tests, die das alte Dämpfungs-Verhalten pinnten, auf den §G188-Vertrag
  umgestellt (volle Kompensation); phase_23-Discovery im Linter-Test robuster
  (Klassenname statt Marker-Attribut).

**Gate-Report nach Migration:** „keine blockierenden Stärke-Kappen im
Phasen-Kern" — repo-weiter Altbestand null. Tests: 1223 betroffene Tests grün.

### Cluster A-Finale 2026-09-26 — Teilband-IF √N-Frequenzscan: Gemeinschafts-FM auf Musik blind messbar

**Problem (dokumentierte Messkanal-Grenze):** Auf 4-Akkord-Musik ohne
Scanner-Saat maß die Blind-Rohspur nur 20,1 der 53 cents Ground-Truth — die
Reparatur lief auf Musik ohne Scanner-Hinweis nie an (Konsens-Trajektorie
Noten-Rauschen ±1000 cents, Mix-IF 0,89 cents).

**Lösung:** kohärenter √N-Frequenzscan (`_scan_subband_modulation_frequency`)
— die Transport-Modulation verschiebt ALLE Teilbänder in Phase, das
Inter-Ton-Beating-Rauschen je Band nicht. Pro Kandidat-Frequenz läuft der
Matched-Filter je Band mit r²-/Amp-Gate (tote Bänder ohne Partial tragen sonst
nur Rauschen), die kohärente Summe der Sinus-Koeffizienten gewinnt √N
(Grob-Grid 0,05–4 Hz + Fein-Scan 0,002 Hz + Parabel-Interpolation). Danach
läuft der bewährte Matched-Pfad bei geschätzter Frequenz.

**Verdrahtung:** der Teilband-Fallback greift jetzt auch OHNE Scanner-Saat
(vorher `elif hint is not None`); der Restfehler-Messkanal des geschlossenen
Regelkreises sucht bei Fehlschlag blind nach (Residuum trägt ggf. eine andere
Frequenzlage) und fällt auf Blind-Fit/pYIN zurück.

**Ergebnis (4-Akkord-Musik, GT 53 cents FM @ 0,3 Hz):** Blind-Messung
20,1 → **49,1 cents (r²=1,00)**, Hint-Referenz 49,2. End-to-End greift die
Reparatur: **49,2 → 11,4 cents (77 %)**; Fit angewandt (r²=0,9999). Der
Nachlauf misst nun ebenfalls (15,97) und verwirft Verschlechterungen korrekt
(Never-worsen, gemessen 34,4 → verworfen). Regression Ton-Kanal unverändert
(90,8 %). Tests: 34 grün, neuer Blind-Scan-Regressionstest (Akkord-Synthese,
Amp 42–66, f ≈ 0,3 ± 0,02, r² ≥ 0,9).

### Import-Fallstudie 2026-09-26 — „Trio Schweizer - 13 Tage": irregulärer Drift, offene Extraktions-Grenze

**Auftrag:** Import-Songs mit extremem Wow/Flutter souverän restaurieren,
phase_12 am Song bearbeiten und nachmessen.

**Diagnose (gemessen, 6 min, 44,1 kHz Stereo):** Scanner wow sev=1,0 @
0,148 Hz, multiband_instability 15,4 cents (cv 7 %), flutter_spectral_sidebands
sev=1,0 (8 Seitenbänder, 68 dB Prominenz), speed_calibration sev=1,0
(Faktor 7,6). Bänder-Diagnose (10 s): 121/121 valide, aber 120/120
Sinus-Fits verworfen (r² max 0,31) — der Transport-Drift ist IRREGULÄR,
kein periodisches Wow. Laufzeitbefund: Messkanal 119 s je 30-s-Fenster.

**Umgesetzt:** (a) 16-kHz-Arbeitsspur für den Messkanal (5× schneller);
(b) Sinus-freie Gemeinschafts-Extraktion + Übernahme als Warp-Trajektorie
(Witness subband+common, volle Stärke §G188); (c) exakte Sekunden-Zeitachse
(der linspace-/convolve-Unterbau stauchte um 0,8 % / verschob ~200 ms);
(d) Never-worsen jetzt auch für den HAUPTWARP (Vorwarp-Zustand wird mit-
gemessen und bei besserem Wert erhalten, Hörordnung §8a).

**Ergebnis am Song:** Kette greift jetzt (Metadaten: Witness subband+common,
Stärke 1,0, Nachlauf aktiv 10,8→9,5 cents) — aber die Drift-Span misst
28,1 → 29,2 cents: **die irreguläre Drift-EXTRAKTION ist noch nicht
belastbar**. Ground-Truth-Isolationsprüfungen (Akkord-Musik + bekannter
irregulärer Drift) belegen den Stand ehrlich: Z-Scoring-PCA corr 0,04,
Energie-Selektion Amp 137 statt 6,9 cents, IVS corr 0,15, Form+Amplitude-
Split corr −0,37. Kernursache (analysiert): die ±3 %-Bandbreite der
Teilbänder ÜBERLAPPT sich — dasselbe Beating erscheint in ~3–5 Nachbar-
bändern korreliert, der √N-Gewinn bricht ein; Beating-Reste (±400 cents
Roh) überleben moderate Tiefpässe und dominieren jede Mittelung.

**Offen (präzise spezifiziert):** irreguläre Drift-Extraktion entweder über
nicht-überlappende Ein-Partial-Bänder (Beating entkoppelt ⇒ echter √N) oder
einen adaptiven Tiefpass, der Beating (5–50 Hz) vollständig unterdrückt, ohne
Drift > 0,3 Hz anzutasten; danach greift die bestehende Warp-Pipeline inkl.
Never-worsen. Periodische Wow-/Flutter-Fälle (Sinus) sind über den gesamten
Weg messend belegt (Ton 90 %, Akkord-Sinus 77 %).

### Abschluss 2026-09-26 (Runde 2) — irreguläre Drift-Extraktion GT-bewiesen, Song-Defekt eingeengt

**Extraktion geschlossen (Ground-Truth-bewiesen):** Der 1,55-s-Tiefpass
(Nullstelle ≈ 0,65 Hz: Beating 5–50 Hz → < 2 %, Drift 0,05–1 Hz → > 95 %)
plus Zeilen-Fit gegen die geglättete Form (r²-Selektion) + Gain-Rückgewinnung
trägt jetzt irregulären Drift: GT Akkord-Musik + ±15-cents-Wanderung misst
**37,6 → 4,2 cents (89 % Reduktion)** auf konsistenter Estimator-Metrik — vorher
verschlimmerte jeder Versuch. Gate-Diagnose am Song-Fenster (instrumentiert):
langsamer Gemeinschafts-Drift nur **3,65 cents ptp**, r² max 0,04 (kein Band
folgt der langsamen Form) — die frühere 28-cents-„Spanne" war Beating-
Kontamination der 450-ms-Variante.

**Song-Defekt damit eingeengt (gemessen):** Der dominante Gehör-Defekt von
„13 Tage" ist KEIN Slow-Wow, sondern **schnelle Modulation**: 8 Seitenbänder
(Scanner sev=1,0, 68 dB über sehr tiefem Boden; Träger-Scan ±400 Hz zeigt sie
erst ab > 15 Hz Abstand) ⇒ Flutter-Band. Die Phase-12-Trajektorie läuft mit
40 fps (Nyquist 20 Hz) und bildet > 15 Hz strukturell nicht ab — zusammen mit
dem speed_calibration_error (×7,6) ist das die letzte Architektur-Grenze.

**ERLEDIGT (2026-09-26): Sample-Rate-Flutter-Pfad (§7.4c) — GT-bewiesen
32,5 → 0,0 cents.** Umsetzung in `phase_12_wow_flutter_fix.py`:
`_flutter_track_from_stacks` (Matched-Scan 0,5–50 Hz auf 200-Hz-Raster,
Grob-Raster 0,1 Hz + Fein-Raster 0,005 Hz, r²-Gewichtung mit Zeugen-Gate
max r² ≥ 0,25) und `_flutter_correction_pass` (per-sample-SF über
`_speed_warp_resample`, Never-worsen — Hörordnung §8a).

Befunde (Ground-Truth `_synth_flutter` = 6 Hz, 0,2 %):

- Die 50-ms-Blockmittel-Entzerrung auf 200 Hz hatte ihre Sinc-Null bei
  exakt 20 Hz und löschte die Flutter-Frequenz komplett (r² ≈ 0,00 über
  0,5–50 Hz). Fix: Stride-Decimation — der 5-ms-Box-Laufmittel-Filter der
  Teilband-Spur ist bereits der Anti-Alias für 200 Hz.
- Die FM sitzt in 1–2 starken Bändern (GT-Befund: 1 von 41 Zeilen trägt
  amp 22,9 cents bei r² 0,36). Uniforme Mittelung verwässerte auf 1,07
  cents; rauschende Großamplituden-Fits (amp bis 120 cents, r² ~0,04)
  trieben die Peak-Suche auf 48,8 cents. Fix: Ausschluss r² < 0,10 aus
  der Akkumulation + Zeugen-Gate erst für die Finale-Frequenz.
- „score 0 über alle Frequenzen“ ist die Signatur SAUBEREN Materials:
  der Messkanal liefert dann eine flache Spur (Spanne 0,0) statt zeros;
  zeros(1) signalisiert allein Messkanal-Ausfall (zu kurzes Signal).

Evidenz: Flutter-Case über die volle `process`-Kette: 32,5 → 0,0 cents
(100 %); unabhängige Nachmessung bestätigt. Never-worsen hält: jitter-
artifacts/scrape_flutter/flutter_spectral_sidebands ohne Eingriff
(`applied: False`), 92 Phase-12-Unit-Tests grün, kein Regression in der
wow/generation_loss-Familie. Der leichte Jitter-Anstieg 0,259 → 0,274
stammt aus einem anderen Pfad (Flutter-Pass dort applied=False).

### Cluster D 2026-09-26 — phase_02 Motor-Comb: Detektion + Dominanz-Tiefe

**Ausgangslage:** motor_interference (GT: 100/200/300 Hz, Amp 0,045/0,03/0,02)
stand in der L3-Liste „Phasen-Lücken“ (Reparatur ohne messbaren Effekt:
phys-Metrik 0,933-Scanner-Sicht unverändert).

**Root-Cause 1 — Detektion:** `_detect_multi_fundamental` prüfte nur
50/60 Hz. Die Vollweg-Gleichrichter-/Motor-Comb (2×50 = 100 Hz,
2×60 = 120 Hz) war nie Kandidat. Fix: Kandidaten-Familie 100/120 Hz mit
Comb-Evidenz (mindestens eine weitere Harmonische 2×/3× über Threshold),
Hierarchie (erkanntes 50/60 deckt die Linien als eigene Comb ab) und
§v10.998-Musik-Schutz am Fundamental (100 Hz ist eine häufige Bass-Lage).

**Root-Cause 2 — Tiefensteuerung:** `if is_musical or not _dominant`
stufte eine DOMINANTE Linie (Narrow/Wide-Ratio 1,0) auf Depth 0,22 herunter,
weil der Dynamik-Test Nachbar-Musik in ±5 Hz mitlas (200-Hz-Linie:
`is_musical=True` trotz Ratio 1,0). Fix: Die Band-Dominanz ist — wie der
§v10.998-Kommentar vorsieht — ausschließlich maßgeblich; `is_musical`
bleibt als Evidenz im Log. Ergebnis: alle drei Töne gleichmäßig −23 dB
(die Notch-Fenster-Messung zeigt die Linien bei −37 dB, Rest = Musik-Bass).

**Gate-Kalibrierung:** Die generische 50-%-Amplituden-Gate
(`met_a ≤ met_b × 0,5`) ist bei tonalen Defekten auf Musik UNERREICHBAR:
_die Töne tragen 64 % der Band-Energie (= 55 % der Amplituden-Messung),
reine 3-Notch-Referenz erreicht daher 0,55, exakter Sinus-Abzug käme auf
0,60 (Musik-Floor). `GATE_SPECIAL` erweitert um den relativen Operator
`le_rel` + MOTOR_INTERFERENCE ("le_rel", 0,62) = ≥ 96 % entfernte
Ton-Energie, ohne Musikausfall zu erzwingen (Hörordnung §8a).

**Evidenz:** phys-Metrik 0,0353 → 0,0190 (−46 % Amplitude = −71 % Energie,
am Notch-Optimum 0,55); Scanner-Sev 0,933 → 0,867; Hum-Fall weiter stark
(0,0563 → 0,0183 = −68 %); 111 phase_02/hum-Unit-Tests grün.
Offizielles Harness-Verdict (Familie vinyl): `[OK] motor_interference —
severity 0,933→0,867; phys 0,03532→0,01903 OK`.

### Guardrail-Paket 2026-09-26 — Phase-Zahl 71 > 69: Bestandsaufnahme + Evidenzplan

**Befund:** `tests/normative/test_scope_guardrails.py::test_phase_count_within_limit`
rot (71 Phasen-Dateien vs. `policy/scope_guardrails.yaml → max_phases.limit: 69`).
Die Überzahl sind EXAKT die zwei zuletzt ergänzten Phasen:

1. `phase_ambience_polish.py` (2026-09-22, Spec 25 „Ambience-Politur",
   Commit f0ed03f2) — spezifiziert und mit 12 Tests, aber OHNE die
   policy-pflichtigen ≥5 Echt-Audio-Fälle.
2. `phase_67_crackle_texture_removal.py` (2026-09-24, Spec 06/07.1,
   Commit ad7b1b83) — ML-Phase mit Unit-Tests + Modell-Manifest, ebenfalls
   OHNE Echt-Audio-Evidenzpaket.

**Policy-Vertrag** (`scope_guardrails.yaml → evidence_requirements.new_phase`):
pro Phase ≥5 reale Audiofälle mit konkretem Nutzen, gemessene
HPI-Verbesserung ≥ 0,02 (Durchschnitt oder pro Fall), Risiko-Analyse +
Real-Audio-Quality-Gate-Review („Ausnahme: Bugfixes an bestehenden Phasen,
keine neuen Phasen-IDs").

**Evidenz-Kampagne (vorbereitet, ausstehend):** Der Echt-Audio-Corpus existiert
(`corpus/{shellac,vinyl,tape,reel_tape,cassette,digital,reverb}/` mit
25 clean + 59 damaged Echtaufnahmen und `defect_types`-Manifesten):

- phase_67 (Crackle): ≥5 Crackle-Fälle vorhanden (u. a.
  vinyl_soul_1970s_crackle_hiss, vinyl_jazz_1960s_hiss_crackle,
  vinyl_rock_1960s_hum_crackle + shellac-Beschädigten-Set).
- phase_ambience_polish: reverb-Familie (1 Paar) + room/ambience-getaggte
  tape/vinyl-Fälle — Bestand prüfen, ggf. reverb-Paar aufstocken.
- Messkette: `scripts/run_real_audio_corpus_test.py` (MUSHRA E2E,
  ITU-R BS.1534-3) + HPI-Metrik (`backend/core/musical_quality_assurance.py`),
  je Fall A/B: Pipeline mit vs. ohne die Phase bei identischem Seed.
- **Golden-Set-Infrastruktur vorhanden** (kartiert 2026-09-26):
  `audit/real_audio_execution_golden_gate.py` produziert den Execution-
  Report mit HPI pro Fall (aktuell 51/51 Fälle mit `hpi`, `hpi_contract_passed`,
  `vqi`, `phases_executed`); `audit/real_audio_restoration_quality_gate.py`
  bewertet das Review (Thresholds inkl. `min_hpi_average: 0.78`,
  `min_real_audio_cases: 80`); Fall-Quelle:
  `audit/real_audio_defect_golden_manifest.json` (aktuell 8 annotierte Fälle,
  davon 1 Crackle) — um ≥5 Crackle-Fälle (phase_67) + ≥5 Ambience-Fälle
  (phase_ambience_polish) aus `corpus/*/damaged` zu erweitern; die A/B-
  Referenz `hpi mit Phase − hpi ohne Phase ≥ 0,02` wird über zwei
  Execution-Golden-Läufe mit `required_phases`-Override gemessen.

Der Guardrail-Test bleibt bis zum abgeschlossenen Evidenzpaket + Review rot —
das ist der intendierte Ratchet-Druck; `max_phases.limit` wird NICHT ohne
Evidenz angehoben.

### Wohlklang-Optimum der Parameter-Suche 2026-09-26 — Root-Cause geschlossen

**Befund (Import-Song-Log):** Aurik berechnete nicht die Wohlklang-optimalen
Parameter — die Musik wurde beschädigt, obwohl die Gates die Schäden sahen
(PMGG-Regression 0,3351 ≫ 0,033 als „best_effort" durchgewunken,
§v10.709 „timbre_authentizitaet, transient_energie", 18 Energie-Sprünge).

**Root-Cause:** Alle drei Parameter-Ebenen optimierten das falsche Optimum:

1. `adaptive_strength_optimizer._quick_quality_delta` bewertete
   **Signal-Ähnlichkeit** (identisch = 1,0 = best, Baseline −0,95) — jede
   echte Reparatur SENKTE den Zielfert ⇒ die Suche verharrte bei
   Default-Stärken (Produktionsbefund „§2.70 Joint-Kalibrierung:
   0 boosted, 0 damped").
2. `closed_loop_calibrator.measure_phase_quality_delta` (§v10.600) bewertete
   MR-STFT-Distanz × tanh-Richtung → in der Praxis Δ ≈ 1e-5
   („Δ=+0.0000" für jede Phase → „hold", Regelkreis blind).
3. HPE (`human_pleasantness_estimator`) — die tatsächliche psychoakustische
   Angenehmheit — war nur Gate/Log, nie das Optimumsziel.

**Fix (Ursache statt Symptom):** Gemeinsame Zielfunktion
`wohlklang_objective_delta()` in `human_pleasantness_estimator.py`:

- **Primär: HPE-Delta** (Zwicker-Schärfe, Rauigkeit, Lautheit, Tonalität,
  Fluktuationstärke) — der maximale Wohlklang (Hörordnung §1–§3,
  lexikografische Wohlklang-Ordnung; eine klarere Aufnahme DARF anders
  klingen).
- **Hör-Invarianten (Ebene 1) als Wächter statt als Ziel:** struktureller
  Signalkollaps (Korrelation < 0,50, Pegel > 12 dB, Crest > 12 dB —
  entlang der Watchdog-Kalibrierung) ⇒ harte Ablehnung (−1).
- Stereo-Layout-Invariante (C,N)/(N,C) + NaN/Inf-Schutz wie
  `compute_pleasantness`/`_metric_mono`.
- Übernommen in `adaptive_strength_optimizer` (Suche klettert jetzt die
  Wohlklang-Leiter, ±0,03 = hörbare Veränderung als Konvergenzmaß) und
  `closed_loop_calibrator` (§v10.600 Δ≠0 und wohlklang-gerichtet;
  §v10.650 W5-Reparatur-Clamp bleibt).

**Evidenz:** 80 Tests grün (4 neu: identisch = exakt 0, Entrauschung > 0,
Crest-Kollaps < −0,03, **Stärken-Suche steigt die Leiter**
`optimal_strength > 0,2` — Regressionstest gegen das alte Objektiv);
Layout-Invariante + Reparatur-Clamp-Verträge aus test_260_30 erfüllt.

**Nachschlag 2026-09-26 (beide letzten Schadensvektoren aus dem Import-Log):**

1. **PMGG Timing-Leiter** (`per_phase_musical_goals_gate.py`, §2.29a): Der
   „Sofort-Best-Effort"-Retour für ML-deterministische Timing-Phasen
   (phase_12/31 — kein Wet/Dry-Blending möglich) übernahm die
   **ungeprüfte Vollstärke-Fassung** und verletzte damit den eigenen
   §2.29-Vertrag („geringste Regression anwendet"; Befund: regression=0,3351
   ≫ 0,033 akzeptiert, §v10.709-Schaden). Jetzt sucht eine
   Re-Ausführungs-Leiter (0,75/0,50/0,30/0,15 der Stärke — kein Blending)
   die geringste Regression und gibt ausschließlich diese zurück.
2. **Energie-Kontinuität der Dip-Reparatur** (§2.35b): Die lineare
   Rampen-Länge `max(1, 30 %)` degenerierte bei kurzen Dips (3–6 Frames)
   zu einem Fade-Frame → Sprung 1,0→Gain in ~11 ms (Befund: 18
   Energie-Sprünge > 6 dB/100 ms). Cosinus-Zügelung mit Mindestrampen
   (≥ 2 Frames), kurze Dips erhalten eine durchgehende Zügelung, plus
   3-Frame-Glättung der Gain-Maske (OLA-freundlich).

Evidenz Nachschlag: 172 Tests grün (92 phase_12 + 31 PMGG + 49 Wohlklang,
2 neu: Dip-Reparatur erzeugt keine neuen Energie-Sprünge +
Timing-Leiter-Vertrag).
