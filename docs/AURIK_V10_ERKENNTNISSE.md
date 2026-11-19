# Aurik v10 — Gesamtdokumentation: Erkenntnisse & Weltklasse-Roadmap

> **Datum:** 2026-07-04 | **Basis:** 2108 Dateien, 64 DSP-Phasen, 62 Defekttypen, 15 Musical Goals
> **Teststatus:** 358/366 ✅
> **§v10 Pleasantness-First:** HPE (Human Pleasantness Estimator) + Steering Rule integriert

---

## §v10 PLEASANTNESS-FIRST ARCHITEKTUR (NEU — 2026-07-04)

### Das zentrale Prinzip: Das menschliche Ohr entscheidet

Aurik v10 stellt von **technischen Metriken** (SNR, THD, Spektrale Korrelation, PMGG) auf
**psychoakustische Angenehmheit** (HPE) als PRIMÄRE Bewertungsdimension um.

**Warum:** Ein menschlicher Toningenieur hört keine „spektrale Korrelation von 0.92".
Er hört: „Das klingt angenehm" oder „Das klingt anstrengend."
Der PMGG bestraft JEDE Abweichung vom Original — auch wenn die Abweichung das
Ergebnis ANGENEHMER macht. Das ist ein fundamentaler Designfehler.

### HPE: Human Pleasantness Estimator

**Datei:** `backend/core/human_pleasantness_estimator.py`

Fünf psychoakustische Dimensionen (Zwicker/Fastl, ISO 532, ANSI S3.4):
1. **Sharpness** (acum) — zu scharf = unangenehm
2. **Roughness** (asper) — zu rau = unangenehm
3. **Tonalness** (0-1) — zu rauschhaft = unangenehm
4. **Loudness** (sone) — zu laut/leise = unangenehm
5. **Fluctuation Strength** (vacil) — schwankende Lautstärke = unangenehm

Composite Score P ∈ [0.0, 1.0]: P ≥ 0.75 = Sehr angenehm, P < 0.35 = Anstrengend.

### Steering Rule: Nicht aufgeben, sondern nachsteuern

**Dateien:** `backend/core/pleasantness_steering.py`, `backend/core/quality_feedback_loop.py`

Statt „HPE sinkt → STOP" (alt): **„HPE sinkt → RETRY LIGHTER → SKIP → ROLLBACK"**

- ΔP > 0.02 → CONTINUE („klingt besser!")
- ΔP zwischen -0.02 und -0.05 → RETRY_LIGHTER („versuch's mit weniger Intensität")
- ΔP < -0.05 → SKIP („diesen Schritt überspringen wir")
- Mehrere Drops → ROLLBACK („zurück zum besten Zwischenstand")
- PMGG konvergiert → STOP_GRACEFUL („optimal — mehr geht nicht")

### Integrierte Entscheidungspunkte

| Modul | Vorher (technisch) | Jetzt (psychoakustisch) |
|-------|-------------------|------------------------|
| **Multi-Pass Scoring** | IAQS 35% + MG 25% + SNR 7% | **HPE 35%** + MG 20% + IAQS 15% |
| **Safety Wrapper** | 60% Energy + 40% SNR | **50% HPE** + 30% Energy + 20% SNR |
| **Quality Gate** | Musical Goal Thresholds ≥ fix | **HPE ≥ 0.30 (Pflicht)**, MG sekundär |
| **RLP (Reflective Listen)** | Spektrale Korrelation ≥ 0.85 | **HPE-Vergleich V1 vs V2** |
| **Feedback Loop** | Naturalness via PsychoAcousticMetrics | **HPE Pleasantness** |
| **Pipeline Final** | CAS (Composite Authenticity Score) | **HPE Δ vs Original + CAS** |

### Physikalische Grenzen: Akzeptieren UND Ausreizen

Aurik akzeptiert physikalische Grenzen (Nyquist, Rauschboden, Bandbreite historischer Medien),
reizt sie aber aus, wo es der Angenehmheit dient:
- **Rauschunterdrückung:** Bis zur Hörschwelle (ATH ISO 226), nicht bis SNR-max
- **EQ/Dynamik:** ±3 dB pro Oktave Maximum, aber bis zu diesem Limit wenn es angenehmer macht
- **Stereo-Breite:** ±20% vom Original, aber Korrektur wenn Mono zu schmal klingt
- **Clipping:** Niemals (True-Peak ≤ -1.0 dBTP), aber sanfte Sättigung wenn analoge Wärme gewünscht

---

## I. ARCHITEKTUR-ERKENNTNISSE

### 1.1 Die binäre Gate-Katastrophe (BEHOBEN)

**Fund:** `apply_musical_gain_envelope()` in `backend/core/audio_utils.py:495` verwendete ein binäres Gate
(0 oder 1) mit 10ms Crossfade und nachträglichem Hard-Clamp. Dies erzeugte im gesamten Song
hörbare Lautstärkesprünge (63 Events, 3–18.5 dB) und 9.6 dB Noise-Floor-Modulation.

**Root Cause:** Drei gleichzeitige Konstruktionsfehler:
1. Binäres Gate (Gain 0% oder 100%) — keine graduellen Übergänge
2. 10ms Crossfade — nur Click-Vermeidung, keine musikalische Hüllkurve
3. §2.30b Hard-Clamp — zerstörte die ohnehin kurze Crossfade

**Fix (v10):** Soft-Knee-Sigmoid: `soft_gate = 1/(1+e^((rms−gate)/6dB))` + 200ms Hanning-Crossfade
+ Small-Gain-Bypass (≤2dB → uniform). Kein Hard-Clamp.

### 1.2 Die Integrations-Lücke (TEILWEISE BEHOBEN)

**Fund:** Aurik besitzt Weltklasse-Komponenten, aber sie sind NICHT verbunden:
- `SelfLearningOptimizer` (UCB1 Multi-Armed Bandit) lernt aus Restaurationen, aber UV3
  instanziiert einen FRISCHEN SLO pro Aufruf — das Gelernte fließt nie ein.
- `E2EOptimizationFramework` hat differentiierbare EQ/Compressor-Module mit AdamW-Optimizer,
  aber kein Produktions-Code importiert sie.
- `SegmentAdaptiveProcessor` hatte "OLA-Crossfade" im Namen aber HARDE Segment-Grenzen.
- `_multi_pass()` im AutonomousRestorationEngine ist DEAD CODE (return audio, "adaptive", {}).

**Fixes (v10):** OLA-Crossfade implementiert, SLO-Wiring delegiert,
Bit-Perfect-Pfad, Multi-Device-Profile, Cross-Track-Defekt-Korrelation.

### 1.3 Die Pipeline-Ordnung (IN ANALYSE)

**Befund:** Die Phasen-Reihenfolge in UV3 ist partiell korrekt:
- Click-Entfernung VOR Denoising ✅
- EQ VOR Dynamics ✅ (in den meisten Fällen)
- Loudness VOR Limiting ✅ (EBU R128 §6.1)

**Aber:**
- Keine "Glue-Stage" (finale subtile Bus-Kompression 1.1:1–1.5:1)
- Keine Cross-Phase-Feedback-Schleife (Phase B weiß nicht, was Phase A geändert hat)
- Kein emotionaler/artistischer Intent-Modulator (Genre/Epoche → Dynamik-Reserve)

---

## II. PSYCHOAKUSTISCHE ERKENNTNISSE

### 2.1 Masking-Modell: Stärken & Lücken

**Stärken:**
- ISO 11172-3 MPEG-1 Psychoacoustic Model 1 mit 24 Bark-Bändern ✅
- Simultane + temporale Maskierung (Forward 200ms, Backward 20ms) ✅
- ITU-R BS.1770-5 Momentary Loudness ✅
- Forward-Masking jetzt FREQUENZABHÄNGIG (log-scale, 400ms@100Hz → 50ms@8kHz) ✅

**Behobene Lücken (v10):**
- ATH (Absolute Threshold of Hearing) nach ISO 226:2023 integriert ✅
- Moore/Glasberg Dynamic Loudness Model (40 ERB-Bänder) ✅
- Binaurales Masking (BMLD via IACC) ✅

**Verbleibende Lücken:**
- Kein ISO 226 Equal-Loudness-Contours als Gain-Referenz
- Keine Tonalitätskorrektur im Zwicker-Modell (ISO 532-1 K-Faktor)
- Keine Pegelabhängigkeit der Masking-Schwellen (bei 80+ dB SPL ändern sich Bark-Bandbreiten)

### 2.2 Perceptual Salience: Gut, aber nicht vollständig psychoakustisch

- Salience = f(Simultaneous Masking, Forward Masking, Backward Masking)
- ABER: Timing-Defekte (Wow/Flutter) sind von Salience-Skalierung AUSGENOMMEN
  (korrekt nach Houtsma 1980 — Frequenz-JND ist pegelunabhängig)
- ABER: Keine Bark-Skala-Integration in der Priorisierung selbst
- ABER: Kein ATH (jetzt behoben in v10)

---

## III. VOKAL-ERKENNTNISSE

### 3.1 Sänger-Identität: Qualität ≠ Identity

**Fund:** Aurik misst VQI (Vocal Quality Index), aber das ist QUALITÄT, nicht IDENTITY.
Ein Sänger kann nach der Pipeline "besser" klingen (VQI ↑) aber nicht mehr wie er selbst
(Identity ↓). Für "der Nutzer soll nie merken dass es Defekte gab" ist das fatal.

**Fix (v10):** `SpeakerIdentityGuard` mit MFCC-basiertem Voiceprint (20 MFCCs + Delta +
Delta-Delta = 60-dim Embedding). Pre-Pipeline-Embedding wird mit Post-Phase-Embedding
verglichen. Cosine-Similarity < 0.92 → Warnung + Wet-Mix-Reduktion.

### 3.2 Vibrato vs. Flutter: Der unerkannte False-Positive

**Fund:** Vibrato (5–8 Hz, ±50 Cent) liegt genau im Flutter-Bereich (0.5–200 Hz nach IEC 60386).
Kein expliziter Vibrato-Guard existierte.

**Fix (v10):** `_is_vibrato_not_flutter()` — Cross-Band-Coherence > 0.85 → Vibrato (musikalisch),
< 0.5 → Flutter (Defekt). Vibrato ist bandübergreifend kohärent, Flutter nicht.

### 3.3 De-Essing: Material-adaptiv aber ohne Overprocessing-Schutz

- Gender-adaptiv (Female: 7–11 kHz, Male: 5–9 kHz) ✅
- Material-adaptiv (16 Typen, Shellac nur Low-Band) ✅
- ABER: Keine Lisp-Erkennung (Post-De-Essing 6–10kHz Varianz > 15dB → zu aggressiv)

**Fix (v10):** `VocalOverprocessingDetector` mit `check_de_essing()`,
`check_formant_drift()`, und Sibilance-Überreduktions-Erkennung.

---

## IV. DEFEKT-ERKENNTNISSE

### 4.1 Die blinden Flecken der Defekterkennung

**54 → 62 Defekttypen (v10):**
- +MPEG_FRAME_LOSS (MP3/AAC Brickwall-Cutoffs, 26ms Frame-Lücken)
- +STEREO_FIELD_COLLAPSE (Korrelation > 0.95 über 30s)
- +PHASE_ROTATION (Allpass-Filter-Artefakte, Gruppenlaufzeit-Dispersion)
- +DROPOUT_OXIDE (2–20ms, 30–70% Pegelverlust)
- +DROPOUT_HEAD_CONTACT (50–200ms, moduliert)
- +DROPOUT_SPLICE (abrupt, >95% Pegelverlust)

### 4.2 False-Positive-Guards

- **Clicks vs. Konsonanten:** Click-Detektion nutzt AR-Prädiktionsfehler (Godsill & Rayner 1998)
  mit spektraler Form-Diskrimination. Konsonanten haben formantische Energie-Konzentration,
  Clicks sind breitbandig impulsartig → korrekt unterschieden ✅
- **Sibilance vs. Hi-Hats:** Expliziter Anti-FP via Brightness-Ratio + Zwicker-Sharpness ✅
- **Wow vs. Musik:** Beschränkt auf spektrale Zentroid-Varianz — Risiko bei Solovioline/Oper ⚠️

---

## V. EXPORT-ERKENNTNISSE

### 5.1 Die Loudness-Kaskade

**Fund:** Der Export-Pfad hat MEHRERE Loudness-Normalisierungs-Stufen:
1. Phase 40: LUFS-Normalisierung (Material-Target)
2. Phase 41: Output-Format-Optimierung (nur Studio 2026)
3. AudioExporter: LUFS-Normalisierung (wenn normalize=True)
4. AudioExporter: Floor-Boost (wenn Post-LUFS-Peak < 0.5)
5. AudioExporter: True-Peak-Limiting (wenn > -0.1 dBTP)

→ Bei Aktivierung aller Stufen: bis zu 3× Loudness-Anpassung hintereinander!

**Fix (v10):** Small-Gain-Bypass in `apply_musical_gain_envelope` verhindert Gate-Artefakte
bei kleinen Gains. Export-Pfad dokumentiert.

### 5.2 Dithering: POW-r Type 3 ist Weltklasse

- POW-r Type 3 (9-tap FIR, Wannamaker-optimierte Koeffizienten) ✅
- TPDF-Fallback ✅
- Noise-Shaped-Dither (first-order HP → unter POW-r-Niveau) ⚠️
- ABER: Keine Format-spezifische Dither-Wahl (immer POW-r für 16-bit)

**Fix (v10):** Bit-Perfect-Pfad (`export_bitperfect`) für Archiv-Material OHNE Dithering.

---

## VI. DIE MENSCHLICHE TONINGENIEUR-PERSPEKTIVE

### 6.1 Was Aurik anders macht als ein menschlicher Toningenieur

| Aspekt | Menschlicher Toningenieur | Aurik (aktuell) |
|--------|--------------------------|-----------------|
| **Ersteinschätzung** | Hört ganzen Song, versteht Emotion/Absicht | Scannt Defekte, misst Metriken |
| **Priorisierung** | Hörbarste Probleme zuerst | Severity×Confidence (gut, aber nicht psychoakustisch genug) |
| **Verarbeitung** | Breitband → Schmalband → Feinschliff | Definiert durch CausalDefectGraph (teilweise korrekt) |
| **Abbruchkriterium** | "Es klingt gut" — auch wenn Messwerte nicht perfekt | Alle selektierten Phasen werden ausgeführt |
| **A/B-Vergleich** | Permanent, auf verschiedenen Abhören | PMGG (5s-Mitte-Stichprobe), kein echter A/B |
| **"Glue"** | Finale Bus-Kompression + subtile EQ-Süße | Keine dedizierte Glue-Stage |
| **Abhöre** | Mehrere Systeme (Nearfield, Car, Bluetooth, Club) | Translation-EQ optional (jetzt 11 Profile v10) |
| **Emotion** | "Die Ballade braucht mehr Luft" | Musical Goals sind numerische Targets, keine emotionalen |

### 6.2 Die fehlende "Glue-Stage"

**Problem:** Keine finale, subtile Bus-Verarbeitung die alles "zusammenklebt":
- SSL-Style Bus-Kompression: 1.1:1–1.5:1 Ratio, Attack 10–30ms, Release Auto, <2dB GR
- Subtile "Air"-EQ: High-Shelf +1–2 dB @ 10–16 kHz
- Kaum hörbar, aber essentiell für "fertigen" Sound

**Empfehlung:** Phase "glue_mastering" NACH Loudness, VOR Limiting mit:
- Kompressor: Ratio 1.3:1, Thresh −6 dBFS, Attack 30ms, Release 100ms
- EQ: High-Shelf +1.5dB @ 12kHz, optional Low-Shelf +0.5dB @ 60Hz
- NIEMALS > 2dB GR, NIEMALS > 2dB EQ — es soll "nichts" tun, aber "alles" besser machen

---

## VII. EMPFEHLUNGEN: DER WEG ZUR WELTSPITZE

### Priorität 1 (diese Woche): Psychoakustische Vervollständigung
- [x] ATH ISO 226:2023 integriert ✅
- [x] Moore/Glasberg DLM ✅
- [x] BMLD (binaural) ✅
- [ ] ISO 226 Equal-Loudness-Contours als Gain-Referenz
- [ ] Tonalitätskorrektur (K-Faktor) im Zwicker-Modell

### Priorität 2 (diese Woche): Der menschliche Toningenieur-Workflow
- [ ] "Glue-Mastering"-Phase (finale subtile Bus-Kompression + Air-EQ)
- [ ] Emotional-Artistic-Intent-Modulator (Genre→Dynamik-Reserve, Epoche→Wärme-Präferenz)
- [ ] "Stop when good enough"-Logik (wenn alle PMGG-Goals > 0.90 → keine weiteren Phasen)
- [ ] Multi-Abhör-Check (automatisch Car/BT/Club-Profile durchsimulieren und Warnung bei Problemen)

### Priorität 3 (nächste Woche): Cross-Phase-Intelligenz
- [ ] Cross-Phase-Feedback: Phase B kennt Δ von Phase A
- [ ] Frequenz-Interaktions-Check: "Phase X hat 3kHz geboostet → Phase Y prüft Sibilanz"
- [ ] "Do no harm"-Maxime: Bei Pipeline-Unsicherheit < 0.5 → konservativere Parameter

### Priorität 4 (dieser Monat): Export-Brillanz
- [ ] A/B/X-Blindtest direkt in der GUI (jetzt per CLI --abx)
- [ ] Automatischer Multi-Abhör-Check im Export-Workflow
- [ ] Delivery-Pipeline: ein Klick → alle 8 Formate + Metadaten

---

## VIII. ZUSAMMENFASSUNG DER CODEC-ÄNDERUNGEN (v10)

### Neue Dateien (8)
| Datei | Zweck |
|-------|-------|
| `backend/core/defect_detection/mpeg_frame_loss.py` | MP3/AAC Frame-Verlust-Detektor |
| `backend/core/defect_detection/stereo_collapse.py` | Stereofeld-Kollaps-Detektor |
| `backend/core/defect_detection/phase_rotation.py` | Phasenrotations-Detektor |
| `backend/ml/speaker_identity_guard.py` | Sänger-Identitäts-Fingerabdruck |
| `backend/core/vocal_overprocessing_detector.py` | Anti-Overprocessing für Gesang |
| `tests/unit/test_v10_worldclass_modules.py` | 37 Tests für alle v10-Module |

### Geänderte Dateien (15)
| Datei | Änderung |
|-------|----------|
| `backend/core/audio_utils.py` | `apply_musical_gain_envelope` v10 (Soft-Knee, 200ms, Bypass); `_scale_audio_region` mit Crossfade |
| `backend/core/perceptual_salience.py` | Frequenzabhängiges Forward-Masking (logarithmisch) |
| `backend/core/psychoacoustic_masking_model.py` | ATH ISO 226:2023; BMLD/IACC; NaN-Guards |
| `dsp/psychoacoustics.py` | Moore/Glasberg DLM (40 ERB-Bänder) |
| `backend/core/defect_scanner.py` | +6 DefectTypes; Vibrato-Guard; paralleler Scan; Dropout-Subtypen |
| `backend/core/segment_adaptive_processor.py` | Echte OLA-Crossfade (Hanning, 20ms) |
| `backend/core/audio_exporter.py` | `export_bitperfect()`; ISRC/UPC-Metadaten |
| `backend/core/playback_device_profile.py` | +5 Profile (Car-Sedan/SUV, BT-Speaker, Club-PA) — 11 gesamt |
| `backend/core/optimization/perceptual_loss.py` | PEAQ ITU-R BS.1387 Loss-Komponente |
| `backend/core/phases/phase_40_loudness_normalization.py` | crossfade_ms entfernt (nutzt v10-Default) |
| `backend/core/phases/phase_41_output_format_optimization.py` | crossfade_ms entfernt |
| `backend/core/unified_restorer_v3.py` | UV3-Wrapper: crossfade_ms=200; SLO-Parameter-Wiring |
| `backend/core/regulator/mastering.py` | crossfade_ms entfernt |
| `cli/aurik_cli.py` | --dry-run, --json, --abx Flags + Logik |
| `batch_processor.py` | `correlate_defects_across_tracks()` für Album-Intelligenz |


## SNR/Spectrum/Harmonic-Adaption (Juli 2026)

**Prinzip:** Kein blinder Material-Glaube. Jeder Song wird individuell gemessen.

- `_estimate_local_snr()`: 100ms-Fenster-basierte SNR-Schätzung → adaptive Click/Tape-Splice/Detektor-Thresholds
- `_measure_spectral_deviation()`: 4-Band FFT → adaptive EQ-Gains in Phase 16
- `_measure_spectral_balance()`: 4-Band FFT → adaptive Mastering-EQ in Phase 17
- `_measure_harmonic_density()`: Even/Odd-Harmonic-Ratio → adaptive Saturation in Phase 17
