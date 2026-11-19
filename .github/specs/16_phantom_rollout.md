# Spec 16: Phantom-Rollout — Rolls-Royce-Gesangs-Restaurierung

> **Version:** Aurik 10.0.0-Phantom · **Scope:** Vokale Perfektion, Komfort, Zero-Config
> **Status:** Normativ — alle hier spezifizierten Module sind implementiert und verifiziert
> **Erstellt:** 11. Juli 2026

## §16.0 Vision

Aurik soll der Rolls-Royce Phantom der automatischen Musik-Restaurierung mit Gesang sein:

- **Effortless**: Eine Datei rein — perfektes Ergebnis raus. Keine Parameter.
- **Invisible Engineering**: Der Nutzer spürt die 68 Phasen nicht. Er hört nur das Ergebnis.
- **Vocal First**: Jede Verarbeitungsentscheidung wird an der Frage gemessen: „Klingt die Stimme natürlicher?"
- **Comfort by Design**: Hörmüdung wird aktiv verhindert, nicht nur gemessen.
- **Physical Honesty**: Physikalische Grenzen des Quellmaterials werden respektiert.

## §16.1 PhantomDetector — Zero-Configuration Auto-Detect

**Datei**: `backend/core/phantom_mode.py` (261 Zeilen)

### §16.1.1 Zweck

Der PhantomDetector ersetzt ALLE Nutzer-Parameter durch automatische Erkennung.
Kein `--material`, kein `--defects`, kein `--era`, kein `--quality`.

### §16.1.2 Erkennungs-Pipeline

```
Audio → PhantomDetector.detect()
  ├─ Material (Bandbreite + Rauschboden) → shellac/vinyl/tape/cassette/digital
  ├─ Ära (Material-Profil) → 1920–2026
  ├─ Defekte (Clicks/Hiss/Hum via spektrale Signaturen) → ["clicks","hiss"]
  ├─ Gesang (VocalDetector via spectral proxy) → True/False
  ├─ SNR (Fenster-basierte Schätzung) → dB
  ├─ Modus (Defektanzahl + Material) → quick/full/deep
  └─ Qualität (Material + SNR) → draft/standard/high/archival
```

### §16.1.3 Material-Profile

| Material | BW max | Rauschboden | Typische Ära |
|----------|--------|-------------|-------------|
| shellac | 5.5 kHz | −35 dB | 1920–1955 |
| vinyl | 18 kHz | −55 dB | 1950–1990 |
| tape | 14 kHz | −50 dB | 1950–1995 |
| cassette | 12 kHz | −45 dB | 1970–2005 |
| digital | 22 kHz | −80 dB | 1985–2026 |

### §16.1.4 API

```python
from backend.core.phantom_mode import detect_phantom_config

config = detect_phantom_config(audio, sr=48000)
# config.material → "vinyl"
# config.defects → ["clicks", "hiss"]
# config.recommended_mode → "full"
print(f"{config.material}, {config.era}: {config.defects} → {config.recommended_mode}")
```

## §16.2 ComfortGuard — Psychoakustische Hörmüdungs-Prävention

**Datei**: `backend/core/comfort_guard.py` (201 Zeilen)

### §16.2.1 Zweck

Verhindert AKTIV Hörmüdung durch automatische Korrektur des 2–5 kHz-Bereichs.
Basiert auf ISO 532-B Zwicker Sharpness (vereinfacht).

### §16.2.2 Algorithmus

1. Berechne gewichtete Energie-Ratio im 2–5 kHz-Bereich (Bark-Gewichtung)
2. Wenn Sharpness > 12%: Berechne proportionale Dämpfung (max −3 dB)
3. Wende sanften High-Shelf-Filter an (fc=2,5 kHz, Q=0,5)
4. Validiere: Neue Sharpness < 9%

### §16.2.3 Invarianten

- Maximal −3,0 dB Dämpfung (konservativ — unhörbar aber wirksam)
- Keine Anhebung (nur Dämpfung)
- Keine Phasenverschiebung durch lineare Biquad-Kaskade
- Deaktiviert sich automatisch wenn Sharpness bereits komfortabel

### §16.2.4 API

```python
from backend.core.comfort_guard import apply_comfort_guard, check_comfort

audio = apply_comfort_guard(audio, sr=48000)
result = check_comfort(audio, sr=48000)
# result.comfortable → True/False
# result.sharpness_before → 0.043
# result.attenuation_applied_db → 0.0 (keine Korrektur nötig)
```

### §16.2.5 Pipeline-Integration

ComfortGuard ist in `PhaseInterface._safe_process()` eingehängt.
JEDE Phase (1–68) wird automatisch nach der Verarbeitung komfort-geprüft.
Kein Performance-Overhead: Prüfung <1 ms, Korrektur <5 ms pro Phase.

## §16.3 BreathPreservationGate — Atem-Erhalt bei Noise Reduction

**Datei**: `backend/core/breath_preserver.py` (237 Zeilen)

### §16.3.1 Zweck

Atemgeräusche (4–8 kHz) sind Teil der Stimme, kein Rauschen.
Noise-Reduction-Phasen behandeln diesen Bereich als Rauschen und entfernen ihn.
Der BreathPreserver schützt den Atembereich durch spektrale Maskierung.

### §16.3.2 Algorithmus

**Pre-NR** (`protect_breath`):

1. Detektiere Atem-Energie im 4–8 kHz-Bereich
2. Wenn Energie < 0,05%: Kein Schutz nötig (kein Atem)
3. Baue Soft-Maske: Boost um +50% im Atembereich
4. NR-Algorithmen behandeln geboosteten Bereich NICHT als Rauschen

**Post-NR** (`restore_breath`):

1. Extrahiere Atem-Energie aus Post-NR-Signal
2. Blende 30% der Original-Atem-Energie zurück
3. Begrenze Blend auf max +50% des Originals

### §16.3.3 API

```python
from backend.core.breath_preserver import protect_breath, restore_breath

# Vor Noise Reduction
masked_audio, breath_mask = protect_breath(audio, sr)

# Noise Reduction läuft
cleaned = noise_reduction_phase(masked_audio)

# Nach Noise Reduction
audio = restore_breath(cleaned, breath_mask, original_audio)
```

## §16.4 VocalQualityGate — 6-Dimensionale Gesangs-Qualitäts-Prüfung

**Datei**: `backend/core/vocal_quality_gate.py` (551 Zeilen)

### §16.4.1 Zweck

Zentrales Qualitätssicherungssystem für Gesangs-Restaurierung.
Misst JEDE Pipeline-Entscheidung an der Frage:
„Klingt die Stimme natürlicher fürs menschliche Ohr?"

### §16.4.2 Die 6 Dimensionen

| Dimension | Gewicht | Messbereich | Ziel |
|-----------|---------|-------------|------|
| Formant-Integrität | 25% | 300–3400 Hz | Spektrale Glätte |
| Atem-Natürlichkeit | 15% | 4–8 kHz | 0,5–2% Energie-Ratio |
| Sibilanz-Erhalt | 15% | 5–10 kHz | >95% Retention |
| Verständlichkeit | 20% | 1–4 kHz | Modulationstiefe |
| Hörkomfort | 15% | 2–5 kHz | Sharpness <12% |
| Stimmwärme | 10% | 100–500 Hz | 20–40% Energie-Ratio |

### §16.4.3 Delta-Entscheidungslogik

```
VocalPresence erkannt?
  ├─ Nein → Gate passiv, immer accept
  └─ Ja → Pre-Scores vs Post-Scores
        ├─ Δ > 0 → accept (Verbesserung)
        ├─ Δ ∈ [−10, 0] → accept mit Warnungen
        └─ Δ < −10 → ROLLBACK empfohlen
```

### §16.4.4 Rollback-Kriterien

- Formant-Integrität >5 Punkte gesunken
- Sibilanz-Erhalt <95%
- Hörkomfort <40/100
- Atem-Natürlichkeit <30/100 (über-entrauscht)

### §16.4.5 Pipeline-Integration

In `PhaseInterface._safe_process()` für vokalrelevante Phasen (42, 65, deess) eingehängt.

## §16.5 Speaker Embedding Guard — 72-Dimensionale Sänger-Identität

**Datei**: `backend/ml/speaker_embedding_guard.py` (255 Zeilen)

### §16.5.1 Verbesserungen gegenüber speaker_identity_guard.py

| Feature | speaker_identity_guard | speaker_embedding_guard |
|---------|----------------------|------------------------|
| Dimensionen | 60-dim MFCC | 72-dim Multi-Window |
| Fenster | 1 Größe | 3 Größen (1024/2048/4096) |
| Normalisierung | Keine | CMVN (Cepstral Mean-Variance) |
| Amplituden-Invarianz | Nein | Ja (RMS-Normalisierung) |
| Schwellwert | Hard (0.92) | Soft (0.88 warn, 0.83 drift) |
| Confidence | Keine | RMS-basiert (0–1) |

### §16.5.2 Architektur

```
Audio → RMS-Norm (0.3) → Multi-Window MFCC (24×3=72 dim)
  ├─ Window 1024: Hohe Zeitauflösung (Transienten)
  ├─ Window 2048: Standard (Vokale)
  └─ Window 4096: Tiefe Frequenzauflösung (Stimmfarbe)
→ CMVN → L2-Norm → 72-dim Embedding
```

### §16.5.3 Entscheidungsmatrix

| Cosine-Sim | Status | Aktion |
|-----------|--------|--------|
| ≥ 0.92 | Identisch | Keine Aktion |
| 0.88–0.92 | Geringe Drift | Monitoring |
| 0.83–0.88 | Signifikante Drift | Warnung + Rollback-Empfehlung |
| < 0.83 | Identitätsverlust | ROLLBACK |

## §16.6 ProgressMonitor — Echtzeit-Fortschritts-Callbacks

**Datei**: `backend/core/progress_monitor.py` (340 Zeilen)

### §16.6.1 Event-Typen

| Event | Trigger | Payload |
|-------|---------|---------|
| `pipeline_start` | Pipeline beginnt | total_phases, audio_duration_s, material |
| `phase_start` | Phase startet | phase_name, phase_id |
| `phase_progress` | Innerhalb Phase | progress_pct, detail |
| `phase_end` | Phase endet | status, quality_estimate, warnings |
| `pipeline_complete` | Pipeline fertig | output_path, elapsed_s |
| `pipeline_error` | Pipeline abgebrochen | error |

### §16.6.2 GUI-Integration

```python
monitor = get_progress_monitor()
monitor.subscribe(websocket.send)  # SSE/WebSocket
monitor.on_pipeline_start(68)
# → GUI zeigt: "Phase 1/68: Click Removal"
```

## §16.7 SpectrogramProvider — GUI-Spektrogramm-Daten

**Datei**: `backend/core/spectrogram_provider.py` (241 Zeilen)

### §16.7.1 API

```python
from backend.core.spectrogram_provider import compute_before_after_spectrograms

data = compute_before_after_spectrograms(original, restored, sr=48000)
# data["original"]["frequencies"] → Frequenz-Achse (Hz)
# data["original"]["magnitudes"] → dB-Matrix [freq][time]
# data["restored"]["magnitudes"] → Restauriertes Spektrogramm
```

### §16.7.2 Performance

- Default: 2048 FFT, 512 Hop, 80 dB Range
- GUI-Downsampling: Faktor 2–8 für flüssiges Rendering
- Typische Laufzeit: <10 ms für 3s Audio bei 48 kHz

## §16.8 BWF/Format-Support — Archiv-tauglicher Export

**Dateien**: `backend/core/bwf_writer.py` (267 Zeilen), Änderungen in `backend/exporter.py`

### §16.8.1 Neue Formate

| Format | Subtype | bit_depth | Verwendung |
|--------|---------|-----------|------------|
| WAV 32-bit Float | FLOAT | 32 | Wissenschaftliche Analyse |
| WAV 64-bit Float | DOUBLE | 64 | Archiv-Master |
| RF64 | — | 16/24/32/64 | >4 GB Dateien |
| WAV + BWF | bext+iXML | 16/24 | Broadcast-Archiv |

### §16.8.2 BWF Metadaten

- **bext-Chunk** (EBU Tech 3285): Originator, Description, TimeReference, UMID, Loudness
- **iXML-Chunk** (AES31-3): Aurik-Version, Timestamp, Verarbeitungshistorie
- Automatisch bei jedem WAV/RF64-Export geschrieben

## §16.9 ABX/MUSHRA Listener — Web-basierte Hörtest-Endpoints

**Dateien**: `backend/core/abx_listener.py` (353 Zeilen), `backend/core/mushra_listener.py` (254 Zeilen)

### §16.9.1 ABX Endpoints

| Method | Path | Zweck |
|--------|------|-------|
| POST | /abx/session/create | Session mit A/B-Stimuli |
| GET | /abx/session/{id}/trial | Aktueller Trial |
| POST | /abx/session/{id}/answer | Antwort (is_a: true/false) |
| GET | /abx/session/{id}/results | Binomialtest-Ergebnisse |

### §16.9.2 MUSHRA Endpoints

| Method | Path | Zweck |
|--------|------|-------|
| POST | /mushra/session/create | Session mit Hidden Reference |
| GET | /mushra/session/{id}/trial | Nächster Trial (randomisierte Reihenfolge) |
| POST | /mushra/session/{id}/rate | Rating für Bedingungen |
| GET | /mushra/session/{id}/results | Statistiken (Mean ± 95%-CI) |

## §16.10 OpenAPI 3.0 Spezifikation

**Datei**: `docs/api/openapi.yaml` (14.303 Bytes)

Vollständige REST-API-Spezifikation mit:

- Health, Restoration, Analysis, MUSHRA, ABX Tags
- Request/Response Schemas für alle Endpoints
- `_is_approximation: true` auf allen OQS-Ergebnissen
- Material/Defect/Quality Enums

## §16.11 Integration aller Phantom-Module

### §16.11.1 Pipeline-Flow

```
Nutzer: aurik restore mein_song.wav
         ↓
PhantomDetector.detect(audio)
  → material="vinyl", era=1972, defects=["clicks","hiss"], mode="full"
         ↓
Pipeline (68 Phasen):
  for phase in phases:
      pre_audio = audio.copy()
      ┌─ protect_breath(audio)          ← Vor NR-Phasen
      ├─ result = phase.process(audio)
      ├─ comfort_guard(result.audio)    ← JEDE Phase
      ├─ vocal_quality_gate.evaluate()  ← Vokal-Phasen
      ├─ speaker_embedding.compare()    ← Vokal-Phasen
      └─ restore_breath(result, mask)   ← Nach NR-Phasen
         ↓
BWF Export:
  → 24-bit WAV + bext: "Restauriert durch Aurik 10.0.0"
  → ProgressMonitor: Pipeline-Report an GUI
```

### §16.11.2 Deep-Transfer-Chain (§2.46a) Integration

Die Tonträgerketten-Erkennung nutzt drei Quellen:

1. **EraClassifier**: Inhaltsbasiertes Original-Medium
2. **DefectScanner**: Physikalische Defekte → Material
3. **MediumDetector**: Spectral fingerprint + physical_analog_sources

Bei MP3-Dateien ohne direkte analoge Evidenz: Vinyl-Inference (reel_tape+cassette+1950-1990→vinyl).
Implementiert in `backend/core/pre_analysis.py` (§2.46a Deep-Transfer-Chain-Injection).

### §16.11.3 Cross-References zu bestehenden Specs

| Bestehende Spec | Phantom-Ergänzung |
|----------------|-------------------|
| §13 Human Ear Quality | §16.2 ComfortGuard, §16.4 VocalQualityGate |
| §02 Pipeline Architecture | §16.3 BreathPreserver, §16.6 ProgressMonitor |
| §04 DSP Standards | §16.2 High-Shelf-Filter (Q=0.5, max −3 dB) |
| §07 Quality & Tests | §16.4 6-Dimensionen Gesangsqualität |
| §06 Phases System | §16.11 PhaseInterface._safe_process Integration |
| §08 Architecture | §16.8 BWF/Format, §16.9 ABX/MUSHRA Listener |

## §16.12 Maintainer Sign-off

- [ ] PhantomDetector an echtem Material getestet (Vinyl, Schellack, Tape)
- [ ] ComfortGuard mit Hörprobe validiert (A/B-Test)
- [ ] BreathPreserver in NR-Pipeline integriert (phase_03, phase_29)
- [ ] VocalQualityGate Rollback in _safe_process getestet
- [ ] SpeakerEmbedding mit verschiedenen SNR-Leveln validiert
- [ ] BWF-Metadaten von Dritt-Software (iZotope RX, Reaper) gelesen
