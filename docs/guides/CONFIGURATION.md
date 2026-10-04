# Aurik 10.0.8 — Configuration Guide

**Version:** 10.0.8  
**Datum:** 19. Mai 2026  
**Status:** ✅ Production Ready

> **Normativer Hinweis (Release):** Verbindlicher Produktpfad ist Bridge -> `AurikDenker.denke(...)` -> `export_guard()`.
> Historische Beispiele mit direktem Legacy-Restore, alten Runtime-Pfaden oder erweiterten manuellen Parametern
> sind als `LEGACY_NON_RELEASE` zu behandeln.

---

## Inhaltsverzeichnis

- [Überblick](#überblick)
- [Processing Modes](#processing-modes)
- [Custom Configuration](#custom-configuration)
- [Parameter-Referenz](#parameter-referenz)
- [Anwendungsfälle](#anwendungsfälle)
- [Best Practices](#best-practices)

---

## Überblick

Aurik bietet **2 Processing Modes** plus **Custom Configuration** für maximale Kontrolle.

### Configuration-Hierarchie

```text
1. Processing Mode (Preset)
   ↓
2. Custom Config (Override)
   ↓
3. Auto-Detection (Adaptive)
```

**Beispiel (kanonischer Produktpfad):**

```python
from backend.api.bridge import (
    get_aurik_denker_instance,
    get_load_audio_fn,
    run_pre_analysis,
)

load_audio = get_load_audio_fn()
audio, sr = load_audio("input.wav")

# Voranalyse genau einmal pro Datei
pre_analysis = run_pre_analysis(audio, sr)

denker = get_aurik_denker_instance()

# 1. Standardpfad: Restoration
result_restoration = denker.denke(audio, sr, mode="restoration")

# 2. Alternative: Studio 2026
result_studio = denker.denke(audio, sr, mode="studio2026")

# Ergebnisobjekt traegt Metadaten fuer Export-/Qualitaetsgate
print(pre_analysis)
print(result_restoration.metadata.get("quality_gate_payload", {}))
print(result_studio.metadata.get("quality_gate_payload", {}))
```

---

## Processing Modes

### Übersicht

| Mode | Zweck | Aggressive | Denoise | Comp. | Phase 10 | Phase 11 |
| --- | --- | --- | --- | --- | --- | --- |
| **RESTORATION** | Authentisch | 0.5 | 0.3 | 3.0 | ❌ | ❌ |
| **STUDIO_2026** | Modern/3D | 0.8 | 0.5 | 3.5 | ✅ | ✅ |
| **FORENSIC** | Minimal | 0.2 | 0.1 | 1.5 | ❌ | ❌ |
| **VINTAGE_WARMTH** | Analog | 0.6 | 0.25 | 2.5 | ❌ | ❌ |
| **ARCHIVAL** | Konservativ | 0.3 | 0.2 | 2.0 | ❌ | ❌ |

### 1. RESTORATION (Default)

**Philosophie:** "Preserve the past, enhance the future"

**Charakteristika:**

- ✅ Vinyl/Tape-Charakter erhalten
- ✅ Moderate Noise Reduction
- ✅ Natural Dynamics
- ✅ No artificial "digital sound"

**Parameter:**

```python
ProcessingConfig(
    mode=ProcessingMode.RESTORATION,
    aggressive=0.5,
    denoise_strength=0.3,
    compression_ratio=3.0,
    enable_de_esser=True,
    enable_vocal_enhancement=True,  # Auto (nur wenn Vocals detected)
    enable_instrumental_enhancement=True,  # Auto
    enable_phase_10_soundstage=False,
    enable_phase_11_binaural=False,
)
```

**Verwendung:**

```python
restored = restorer.restore(audio, sr, mode=ProcessingMode.RESTORATION)
```

**Ideal für:**

- Vinyl → Digital Transfer
- Tape Digitalisierung
- 1920s-1990s Recordings
- Jazz, Blues, Classical, Folk

---

### 2. STUDIO_2026 (Modern)

**Philosophie:** "Immersive Audio for Modern Platforms"

**Charakteristika:**

- ✅ Phase 8: Air & Presence (12-20 kHz Sparkle)
- ✅ Phase 10: Soundstage Depth (3-Layer Spatial)
- ✅ Phase 11: Binaural (HRTF für Kopfhörer)
- ✅ Emotional Resonance Enhancement

**Parameter:**

```python
ProcessingConfig(
    mode=ProcessingMode.STUDIO_2026,
    aggressive=0.8,
    denoise_strength=0.5,
    compression_ratio=3.5,
    enable_de_esser=True,
    enable_vocal_enhancement=True,
    enable_instrumental_enhancement=True,
    enable_phase_10_soundstage=True,  # 3D Depth
    enable_phase_11_binaural=True,  # Binaural Enhancement
)
```

**Verwendung:**

```python
restored = restorer.restore(audio, sr, mode=ProcessingMode.STUDIO_2026)
```

**Ideal für:**

- Remastering für Streaming (Spotify, Tidal, Apple Music)
- Modern Pop, Rock, Electronic
- Podcast Enhancement
- Immersive Audio (Dolby Atmos Prep)

---

### 3. FORENSIC (Minimal)

**Philosophie:** "Just Fix What's Broken"

**Charakteristika:**

- ✅ Nur kritische Defekte (Clicks, Clipping)
- ✅ Keine künstliche Enhancement
- ✅ Chain of Custody (Logging empfohlen)
- ✅ Beweis-tauglich

**Parameter:**

```python
ProcessingConfig(
    mode=ProcessingMode.FORENSIC,
    aggressive=0.2,
    denoise_strength=0.1,  # Minimales Denoising
    compression_ratio=1.5,  # Fast keine Kompression
    enable_de_esser=False,
    enable_vocal_enhancement=False,
    enable_instrumental_enhancement=False,
    enable_phase_10_soundstage=False,
    enable_phase_11_binaural=False,
)
```

**Verwendung:**

```python
restored = restorer.restore(
    audio,
    sr,
    mode=ProcessingMode.FORENSIC,
    enable_logging=True,  # Dokumentiere alle Schritte
)
```

**Ideal für:**

- Gerichtsverfahren (Voice Authentication)
- Police/FBI Recordings
- Wiretap Analysis
- Scientific Analysis

---

### 4. VINTAGE_WARMTH (Analog)

**Philosophie:** "Embrace the Imperfection"

**Charakteristika:**

- ✅ Sanftes Denoising (Rauschen als "Ambience")
- ✅ Harmonic Exciter (Tube Saturation)
- ✅ Preserve Vinyl Crackle (wenn musikalisch)
- ✅ Analog-Aesthetic

**Parameter:**

```python
ProcessingConfig(
    mode=ProcessingMode.VINTAGE_WARMTH,
    aggressive=0.6,
    denoise_strength=0.25,  # Sehr sanft
    compression_ratio=2.5,
    enable_de_esser=True,
    enable_vocal_enhancement=True,
    enable_instrumental_enhancement=True,
    enable_phase_10_soundstage=False,
    enable_phase_11_binaural=False,
)
```

**Verwendung:**

```python
restored = restorer.restore(audio, sr, mode=ProcessingMode.VINTAGE_WARMTH)
```

**Ideal für:**

- Lo-Fi Hip-Hop
- Indie/Alternative
- Vintage Sound Preservation
- Analog-Inspired Modern Music

---

### 5. ARCHIVAL (Konservativ)

**Philosophie:** "Preserve for Eternity"

**Charakteristika:**

- ✅ Nur strukturell schädliche Defekte entfernen
- ✅ Original-Dynamik bewahren
- ✅ Minimale Veränderung
- ✅ Langzeitarchivierung (24-bit/48kHz)

**Parameter:**

```python
ProcessingConfig(
    mode=ProcessingMode.ARCHIVAL,
    aggressive=0.3,
    denoise_strength=0.2,
    compression_ratio=2.0,
    enable_de_esser=False,  # Keine künstliche Enhancement
    enable_vocal_enhancement=False,
    enable_instrumental_enhancement=False,
    enable_phase_10_soundstage=False,
    enable_phase_11_binaural=False,
)
```

**Verwendung:**

```python
restored = restorer.restore(audio, sr, mode=ProcessingMode.ARCHIVAL)
```

**Ideal für:**

- Museen & Bibliotheken
- Radio Archives (BBC, ORF, ARD)
- Historical Preservation
- Master Archives (Long-term storage)

---

## Custom Configuration

### Full Custom Config

Alle Parameter können individuell überschrieben werden:

```python
from core.processing_modes import ProcessingConfig, ProcessingMode

config = ProcessingConfig(
    # Basis-Mode
    mode=ProcessingMode.RESTORATION,
    # Globale Parameter
    aggressive=0.4,  # 0.0-1.0 (Restaurations-Aggressivität)
    denoise_strength=0.25,  # 0.0-1.0 (Noise Reduction Stärke)
    compression_ratio=2.5,  # 1.0-10.0 (Dynamik-Kompression)
    # Feature Toggles
    enable_de_esser=True,  # Sibilanz-Reduktion
    enable_vocal_enhancement=True,  # Vocal Clarity (Phase 2.2)
    enable_instrumental_enhancement=True,  # Instrument Enhancement (Phase 2.3)
    enable_phase_10_soundstage=False,  # 3D Soundstage Depth
    enable_phase_11_binaural=False,  # Binaural Processing
    # Advanced (optional)
    target_lufs=-16.0,  # Mastering Loudness (EBU R128)
    enable_true_peak_limiter=True,  # Broadcast-safe Limiting
    enable_multiband_compression=False,  # Studio Mode only
)

denker = get_aurik_denker_instance()
result = denker.denke(audio, sr, mode="restoration")
restored = result.audio
```

---

## Parameter-Referenz

### Globale Parameter

#### `aggressive` (float, 0.0-1.0)

**Beschreibung:** Kontrolliert die Restaurations-Aggressivität über alle Phasen hinweg.

**Effekt:**

- **0.0-0.3:** Sehr konservativ (minimal processing)
- **0.3-0.5:** Moderat (balanced)
- **0.5-0.7:** Aggressiv (strong restoration)
- **0.7-1.0:** Sehr aggressiv (maximum enhancement)

**Betroffene Phasen:**

- Phase 2: Click/Crackle Removal Sensitivity
- Phase 3: Noise Reduction Strength
- Phase 4: Transient Restoration Intensity
- Phase 5: Spectral Refinement
- Phase 6: Dynamic Processing

**Beispiel:**

```python
# Fragile 1920s Shellac Recording
config = ProcessingConfig(
    mode=ProcessingMode.ARCHIVAL,
    aggressive=0.15,  # Extra vorsichtig
)

# Stark beschädigte Kassette
config = ProcessingConfig(
    mode=ProcessingMode.RESTORATION,
    aggressive=0.85,  # Aggressiv restaurieren
)
```

---

#### `denoise_strength` (float, 0.0-1.0)

**Beschreibung:** Noise Reduction Stärke (Phase 3).

**Effekt:**

- **0.0-0.2:** Minimales Denoising (preserve noise floor)
- **0.2-0.4:** Moderates Denoising (natural result)
- **0.4-0.6:** Starkes Denoising (clean sound)
- **0.6-1.0:** Sehr starkes Denoising (broadcast-clean)

**Trade-Off:**

- **Zu niedrig:** Hörbare Hintergrundgeräusche bleiben
- **Zu hoch:** "Digital sound", Artifacts, Transient-Smearing

**Beispiel:**

```python
# Noisy room recording (Podcast)
config = ProcessingConfig(
    mode=ProcessingMode.STUDIO_2026,
    denoise_strength=0.6,  # Starkes Denoising für Podcast
)

# Vinyl with "musical" crackle
config = ProcessingConfig(
    mode=ProcessingMode.VINTAGE_WARMTH,
    denoise_strength=0.15,  # Preserve Vinyl-Charakter
)
```

---

#### `compression_ratio` (float, 1.0-10.0)

**Beschreibung:** Dynamik-Kompression (Phase 6).

**Effekt:**

- **1.0:** Keine Kompression (original dynamics)
- **1.5-2.5:** Sanfte Kompression (musical)
- **2.5-4.0:** Moderate Kompression (Radio/Streaming)
- **4.0-10.0:** Heavy Kompression (Broadcast/Mastering)

**Verwendung:**

- **1.0-2.0:** Classical, Jazz, Audiophile
- **2.0-3.5:** Pop, Rock, General Music
- **3.5-6.0:** Loudness War, Modern Mastering
- **6.0-10.0:** Podcast, Voice-Over (extreme loudness)

**Beispiel:**

```python
# Classical Music (preserve dynamics)
config = ProcessingConfig(
    mode=ProcessingMode.ARCHIVAL,
    compression_ratio=1.2,  # Minimal compression
)

# Modern Pop (loudness war)
config = ProcessingConfig(
    mode=ProcessingMode.STUDIO_2026,
    compression_ratio=4.5,  # Broadcast loudness
)
```

---

### Feature Toggles

#### `enable_de_esser` (bool)

**Beschreibung:** Sibilanz-Reduktion (Phase 5.1).

**Effekt:**

- **True:** Aggressive "s", "t", "f" Sounds werden gedämpft (5-8 kHz)
- **False:** Keine Sibilanz-Reduktion

**Verwendung:**

- **True:** Vocals, Speech, Broadcast
- **False:** Instrumental, Classical (keine Sibilanten)

**Beispiel:**

```python
# Podcast mit harschen Sibilanten
config = ProcessingConfig(
    mode=ProcessingMode.STUDIO_2026,
    enable_de_esser=True,  # Aktiviert
)

# Piano Solo (keine Vocals)
config = ProcessingConfig(
    mode=ProcessingMode.RESTORATION,
    enable_de_esser=False,  # Nicht benötigt
)
```

---

#### `enable_vocal_enhancement` (bool)

**Beschreibung:** Phase 2.2 Vocal Enhancement (5-Stage Pipeline).

**Was macht es:**

1. **Breath Intelligence:** Intelligente Atem-Reduktion
2. **Formant System:** Vokal-Klarheit (Formanten-Anhebung)
3. **Vocal Presence:** 2-5 kHz Präsenz-Boost
4. **Spectral Inpainting:** Reparatur beschädigter Vokal-Frequenzen
5. **Dynamics Intelligence:** Voice-spezifische Kompression

**Verwendung:**

```python
# Podcast / Voice-Over
config = ProcessingConfig(
    mode=ProcessingMode.STUDIO_2026,
    enable_vocal_enhancement=True,  # Maximale Vocal Clarity
)

# Instrumental Music (kein Gesang)
config = ProcessingConfig(
    mode=ProcessingMode.RESTORATION,
    enable_vocal_enhancement=False,  # Nicht benötigt
)
```

**Auto-Detection:**

```python
# Default: Auto (wird nur aktiviert wenn Vocals detected)
config = ProcessingConfig(
    mode=ProcessingMode.RESTORATION,
    enable_vocal_enhancement=None,  # Auto (empfohlen)
)
```

---

#### `enable_instrumental_enhancement` (bool)

**Beschreibung:** Phase 2.3 Instrumental Enhancement (6 System-Pipeline).

**Was macht es:**

1. **Bass Enhancement:** Sub-Bass Clarity (40-150 Hz)
2. **Drums Enhancement:** Attack Sharpness, Transient Preservation
3. **Guitar Enhancement:** String Clarity, Harmonic Richness
4. **Piano Restoration:** Note Definition, Decay Preservation
5. **Brass Enhancement:** Brilliance (2-5 kHz), Air (8-12 kHz)
6. **Spatial Enhancement:** Stereo Width, Depth Cues

**Verwendung:**

```python
# Rock Band
config = ProcessingConfig(
    mode=ProcessingMode.STUDIO_2026,
    enable_instrumental_enhancement=True,  # All Instruments
)

# A Cappella / Voice-Only
config = ProcessingConfig(
    mode=ProcessingMode.RESTORATION,
    enable_instrumental_enhancement=False,  # Nicht benötigt
)
```

---

#### `enable_phase_10_soundstage` (bool)

**Beschreibung:** Phase 10 Soundstage Depth (3D Immersion).

**Was macht es:**

- **Foreground (70%):** Direct Sound
- **Midground (20%):** Early Reflections (15ms, 22.5ms, 30ms)
- **Background (10%):** Diffuse Reverb (RT60=0.3s)
- **HF Damping:** Distance Cues (8 kHz Rolloff)

**CPU-Cost:** ~+15% Processing Time

**Verwendung:**

```python
# Remastering für Streaming (3D Audio)
config = ProcessingConfig(
    mode=ProcessingMode.STUDIO_2026,
    enable_phase_10_soundstage=True,  # Immersive Audio
)

# Archival (Original Soundstage)
config = ProcessingConfig(
    mode=ProcessingMode.ARCHIVAL,
    enable_phase_10_soundstage=False,  # Preserve Original
)
```

---

#### `enable_phase_11_binaural` (bool)

**Beschreibung:** Phase 11 Binaural & Emotional Enhancement.

**Was macht es:**

1. **Binaural Processing:**
   - ITD (Interaural Time Difference)
   - ILD (Interaural Level Difference)
   - Pinna Filtering (Elevation Cues)
   - Crossfeed (Bauer's Formula)

2. **Emotional Resonance:**
   - Warmth Enhancement (Low-Mid Boost)
   - Richness (Harmonic Saturation)
   - Air (High-Shelf @ 12 kHz)
   - Expression (Gentle Expansion)

**CPU-Cost:** ~+10% Processing Time

**Verwendung:**

```python
# Headphone Mastering
config = ProcessingConfig(
    mode=ProcessingMode.STUDIO_2026,
    enable_phase_11_binaural=True,  # Optimiert für Kopfhörer
)

# Speaker-only Playback
config = ProcessingConfig(
    mode=ProcessingMode.RESTORATION,
    enable_phase_11_binaural=False,  # Nicht benötigt für Lautsprecher
)
```

---

## Anwendungsfälle

### Use Case 1: Vinyl Transfer (Standard)

**Problem:** 1960s Vinyl → Digital (mit Clicks, Crackle, Rumble)

**Lösung:**

```python
from backend.api.bridge import get_aurik_denker_instance, get_load_audio_fn
import soundfile as sf

load_audio = get_load_audio_fn()
audio, sr = load_audio("vinyl_rip.wav")

denker = get_aurik_denker_instance()
result = denker.denke(audio, sr, mode="restoration")

sf.write("vinyl_restored.wav", result.audio, 48000, subtype="PCM_24")
```

**Ergebnis:**

- ✅ Clicks/Crackle entfernt (Phase 2A)
- ✅ Rumble gefiltert (Phase 0)
- ✅ Vinyl-Wärme erhalten (moderate denoising)

---

### Use Case 2: Podcast Enhancement (Modern)

**Problem:** Noisy room recording, breaths, sibilance

**Lösung:**

```python
from core.processing_modes import ProcessingConfig, ProcessingMode

config = ProcessingConfig(
    mode=ProcessingMode.STUDIO_2026,
    denoise_strength=0.6,  # Strong noise reduction
    enable_vocal_enhancement=True,  # Voice clarity
    enable_de_esser=True,  # Reduce sibilance
    enable_phase_11_binaural=True,  # Immersive listening
)

restored = restorer.restore(audio, sr, config=config)
sf.write("podcast_enhanced.wav", restored, 48000)
```

**Ergebnis:**

- ✅ Room noise entfernt (Phase 3)
- ✅ Sibilance gedämpft (Phase 5.1)
- ✅ Breath-Intelligenz (Phase 2.2)
- ✅ Binaural Enhancement (Phase 11)

---

### Use Case 3: Historical Archive (1920s Shellac)

**Problem:** 78 RPM Shellac, extreme clicks, narrow bandwidth

**Lösung:**

```python
from core.processing_modes import ProcessingConfig, ProcessingMode

config = ProcessingConfig(
    mode=ProcessingMode.ARCHIVAL,
    aggressive=0.15,  # Extra vorsichtig
    denoise_strength=0.1,  # Preserve historical noise
    compression_ratio=1.2,  # Minimal compression
    enable_de_esser=False,
    enable_vocal_enhancement=False,
    enable_instrumental_enhancement=False,
)

restored = restorer.restore(
    audio,
    sr,
    config=config,
    enable_logging=True,  # Dokumentiere alle Schritte für Archiv
)

sf.write("archive_master.wav", restored, 48000, subtype="PCM_24")
```

**Ergebnis:**

- ✅ Nur kritische Clicks entfernt (Phase 2A, sanft)
- ✅ Historical noise preserved
- ✅ No artificial enhancement
- ✅ Dokumentierter Processing Chain (Log)

---

### Use Case 4: Forensic Analysis

**Problem:** Police recording, voice authentication, beweis-tauglich

**Lösung:**

```python
config = ProcessingConfig(
    mode=ProcessingMode.FORENSIC,
    aggressive=0.1,  # Minimale Veränderung
    denoise_strength=0.05,  # Nur extreme Noise
    compression_ratio=1.0,  # Keine Kompression
)

restored = restorer.restore(
    audio,
    sr,
    config=config,
    input_file="evidence_2026_123.wav",
    enable_logging=True,  # Chain of Custody
)

# Save processing report für Gericht
if hasattr(restorer, "logger"):
    restorer.logger.save_trace("evidence_processing_report.json")
```

**Ergebnis:**

- ✅ Minimale Veränderung (nur kritische Defekte)
- ✅ Dokumentierte Processing Chain
- ✅ Beweis-taugliche Verarbeitung

---

## Best Practices

### 1. Start Conservative

```python
# Erste Restoration: Conservative
config = ProcessingConfig(
    mode=ProcessingMode.RESTORATION,
    aggressive=0.3,  # Niedrig starten
    denoise_strength=0.2,
)
restored_v1 = restorer.restore(audio, sr, config=config)

# Wenn zu wenig: Erhöhe schrittweise
config.aggressive = 0.5
config.denoise_strength = 0.3
restored_v2 = restorer.restore(audio, sr, config=config)

# A/B Compare
```

**Warum:** Over-Processing ist irreversibel, Under-Processing kann korrigiert werden.

---

### 2. Enable Logging für Experiments

```python
# Test verschiedene Configs
configs = [
    ProcessingConfig(mode=ProcessingMode.RESTORATION, denoise_strength=0.2),
    ProcessingConfig(mode=ProcessingMode.RESTORATION, denoise_strength=0.4),
    ProcessingConfig(mode=ProcessingMode.STUDIO_2026),
]

results = []
for i, config in enumerate(configs):
    restored = restorer.restore(audio, sr, config=config, output_file=f"test_{i}.wav", enable_logging=True)
    results.append(
        {
            "config": config,
            "snr": restorer.logger.trace.overall_snr_improvement if hasattr(restorer, "logger") else None,
        }
    )

# Compare SNR
for i, r in enumerate(results):
    print(f"Config {i}: SNR = {r['snr']:.1f} dB")
```

---

### 3. Mode als Basis, dann Fine-Tune

```python
# Starte mit Mode-Preset
config = ProcessingConfig(mode=ProcessingMode.RESTORATION)

# Teste mit Defaults
restored = restorer.restore(audio, sr, config=config)

# Wenn nicht optimal: Fine-Tune einzelne Parameter
config.denoise_strength = 0.4  # Erhöhe nur Denoising
restored_v2 = restorer.restore(audio, sr, config=config)
```

---

### 4. Use Auto-Detection (Default)

```python
# Lasse Aurik automatisch entscheiden
restored = restorer.restore(audio, sr)  # Auto: Mode=RESTORATION, adaptive parameters

# Nur überschreiben wenn nötig
restored = restorer.restore(audio, sr, mode=ProcessingMode.STUDIO_2026)
```

---

## Weitere Informationen

- **Python API:** [docs/api/PYTHON_API.md](../api/PYTHON_API.md)
- **User Guide:** [docs/guides/USER_GUIDE.md](USER_GUIDE.md)
- **Pipeline Flow:** [docs/architecture/PIPELINE_FLOW_ANALYSIS.md](../architecture/PIPELINE_FLOW_ANALYSIS.md)

---

**© 2026 Aurik Audio Restoration System**  
**Version:** 8.0.0 | **Configuration Guide** | **Status:** Complete
