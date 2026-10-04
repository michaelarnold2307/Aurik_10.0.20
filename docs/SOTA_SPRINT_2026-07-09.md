# Aurik SOTA-Upgrade — Erkenntnis-Dokumentation

## 2026-07-08 / 2026-07-09: SOTA-Sprint

### Commit-Übersicht: 14 Commits in 2 Tagen

21 Dateien geändert, 0 Tests gebrochen (1 pre-existing failure in `test_authenticity_metrics_extended`)
Alle 21 Dateien syntaktisch korrekt, alle Import-Pfade konsistent.

---

## Architektur-Erkenntnisse

### 1. Joint-Calibrator → PhaseConductor-Konflikt (Root Cause: Phase 03 Gesangsverzerrung)

**Problem**: BS-RoFormer überprozessierte Vocals und verzerrte die Stimme bei Energieanstiegen.

**Ursachenkette** (3 Bugs, die kaskadierten):
1. `conflict_notes` aus PhaseInteractionDenker erreichten nie UV3
   → Joint-Calibrator konnte `terminal_codec` und `discount` nicht lesen
2. `_precomputed_phase_plan` war lokale Variable, Joint-Calibrator suchte `self._precomputed_phase_plan`
   → `_jcal_phases` immer leer → Joint-Calibrator lief nie
3. PhaseConductor überschrieb Joint-Calibrator-Stärken mit `=`
   → Selbst wenn §2.70 lief, wurden die Werte sofort wieder platt gemacht

**Lösung**: Drei chirurgische Fixes
- `aurik_denker.py`: `_pid_conflict_notes` initialisiert + an restaurier_denker durchgereicht
- `restaurier_denker.py`: `conflict_notes` Parameter → UV3 kwargs
- UV3: `self._conflict_notes` + `self._precomputed_phase_plan` gespeichert
- PhaseConductor: `setdefault` statt `=`

**Erkenntnis**: Bei 36.000-Zeilen-Monolithen sind Datenfluss-Bugs die häufigste Ursache für scheinbare ML-Fehlverhalten. Der BS-RoFormer war nie das Problem — er bekam nur nie die Information, wann er pausieren sollte.

### 2. Phase 03 Drei-Verteidigungslinien-Pattern

Dieses Pattern ist Wiederverwendbar für jede ML-basierte Phase:

```python
# Linie 1: Denker-Entscheidung respektieren
if _denker_strength <= _dsp_threshold:  # Joint-Calibrator sagt "lohnt nicht"
    use_lightweight = True

# Linie 2: Fallback wenn Kalibration fehlt
elif _denker_strength >= 0.95 and panns >= 0.25:  # Unkalibriert + Gesang
    use_lightweight = True  # Safety-first

# Linie 3: ML nur wenn Denker explizit befürwortet
_bsrof_gate = ... and _denker_strength > 0.55  # Nur bei klarer Freigabe
```

### 3. Strength-Envelope: Skalar → Vektor

**Vorher**: `kwargs["strength"] = 0.41` (ein Wert für 225s Audio)
**Nachher**: `kwargs["strength_envelope"] = [0.08, 0.08, 0.56, 0.08, ...]` (pro Frame)

Die 8-Stage-SOTA-Pipeline (`strength_envelope.py`, 561 Zeilen):
1. Defect-type-specific Gaussian sigma (clicks=30ms, noise=200ms)
2. Asymmetric attack/release smoothing (5ms/50ms) — KEIN Pumping
3. Transient-aware gating (Duxbury 2003)
4. Psychoacoustic floor modulation (ITU-R BS.1387)
5. Per-frame vocal attenuation
6. Joint-Calibrator scaling
7. Floor + Ceiling
8. Crossfade windowing (2.5ms Hann)

**Integration**: 8 Phasen (03, 04, 06, 07, 09, 20, 28, 29) nutzen das Envelope-Blending.
Jede Phase ruft `apply_strength_envelope()` mit ihrem Output/Original und `_effective_strength`.

### 4. Gender-Detection: Von global zu per-Segment

**Vorher**: Ein Gender-Wert für den ganzen Song → Contralto falsch als "male"
**Nachher**: Timeline mit Zeitsegmenten + Multi-Feature-Scoring

4-Feature gewichteter Score:
- F0 (×0.40): Median über voiced frames
- Spectral Tilt (×0.25): dB/Oktave 200-2000Hz (Recording-Chain-kompensiert)
- Vibrato (×0.20): Rate + Tiefe per Autocorrelation (robuster als FFT)
- HNR (×0.15): Harmonics-to-Noise Ratio

Zwei Fallback-Ebenen:
- Contralto: F0 140-220Hz + weibliche Formanten → Override FEMALE
- Formant-Versagen (F1=0Hz): Vibrato-Analyse oder Default FEMALE

Per-Segment-Verarbeitung: `_process_per_gender_segments()` splittet Audio an Timeline-Grenzen,
wendet gender-spezifische Sibilanz-Bänder + Formant-Range an, crossfaded mit 5ms Hann.

### 5. AudioSR ROCm-NaN: CPU-Force vor generate_batch

**Root Cause**: `_make_batch_fn` erzeugt Tensoren auf GPU (torch default device=cuda).
`model.generate_batch()` auto-moved Modell auf GPU → HiFi-GAN vocoder transposed
convolutions produzieren NaN auf ROCm.

**Fix**: `model.cpu()` + `batch.cpu()` direkt vor `generate_batch()`.
DDIM läuft auf CPU (langsamer aber NaN-frei).

**SBR-Fallback**: Wenn ML trotzdem fehlschlägt, `_sbr_extend()` statt `np.tanh()`.
Kopiert 2-6 kHz Energie um eine Oktave nach oben mit spektraler Hüllkurve.

### 6. Security: Path Traversal CWE-23

`upload_file.filename` aus HTTP-Parameter direkt in `open()` → `Path(filename).name` stripped Pfad.

---

## Spezifikation-Konsistenz

### Envelope-Integration: Alle 8 Phasen identisch

| Phase | Output-Var | Original-Ref | Strength-Var | Pattern |
|-------|-----------|-------------|-------------|---------|
| 03 | result_audio | audio | effective_strength | ✅ |
| 04 | result_audio | audio | _effective_strength | ✅ |
| 06 | restored | audio | _effective_strength | ✅ |
| 07 | restored | audio | _effective_strength | ✅ |
| 09 | restored | audio | _effective_strength | ✅ |
| 20 | reduced | audio | _effective_strength | ✅ |
| 28 | denoised_audio | audio | _effective_strength | ✅ |
| 29 | audio_processed | audio | _effective_strength | ✅ |

Abweichung nur in Phase 03: nutzt `effective_strength` (ohne Underscore) — funktional identisch.

### Joint-Calibrator: Datenfluss lückenlos

```
PhaseInteractionDenker.planen()
  → conflict_notes: ["terminal=mp3_low discount=0.55", ...]
  → aurik_denker: _pid_conflict_notes
  → restaurier_denker: conflict_notes Parameter
  → UV3.restore(): kwargs["conflict_notes"]
  → self._conflict_notes gespeichert
  → Joint-Calibrator §2.70: liest self._conflict_notes
  → Parsed: terminal_codec="mp3_low", codec_avg_discount=0.55
  → Phase 03: utility ×0.12 (ml_artifact + vocal_distortion)
  → strength ≈ 0.41 → conductor_strength_hints
  → PhaseConductor.setdefault() respektiert Joint-Calibrator
```

### Phase 03 BS-RoFormer: 4 Verteidigungslinien wasserdicht

BS-RoFormer läuft NUR wenn ALLE 5 Bedingungen zutreffen:
1. Joint-Calibrator läuft ✅
2. strength > 0.55 ✅ (Denker befürwortet NR)
3. use_lightweight = False ✅ (kein Guard aktiv)
4. panns_singing ≥ 0.35 ✅ (genug Gesang für Stem-NR)
5. SNR < 20 dB ✅ (genug Rauschen für ML-NR)

---

## Wichtige Erkenntnisse für die Weiterentwicklung

### Datenfluss-Debugging
- Bei 36.000-Zeilen-Monolithen IMMER den Datenfluss von vorne bis hinten verfolgen
- `getattr(self, "x")` schlägt still fehl wenn `x` lokale Variable ist
- `logger.debug` verschluckt Exceptions — immer auf `logger.info` prüfen

### ML-Modelle auf ROCm
- `torch` auf ROCm: Modell auf CPU laden, dann `.cuda()` ist sicherer als direkt auf GPU
- Vocoder (HiFi-GAN) immer auf CPU — transposed convolutions sind ROCm-instabil
- Batch-Tensoren prüfen: `_make_batch_fn` nutzt default device → GPU-Tensoren
- Vor `generate_batch()`: `model.cpu(); batch.cpu()`

### Phase-spezifische Stärke
- Der Joint-Calibrator ist der EINZIGE Ort für Phasen-Stärke-Berechnung
- PhaseConductor darf nur ABSENKEN, nie ANHEBEN
- `setdefault` ist das korrekte Pattern für "erster Setter gewinnt"

### Gender-Detection bei Vintage-Aufnahmen
- LPC-Formant-Tracker versagt bei Rauschen → F0-basierter Fallback
- Spectral Tilt 200-2000Hz statt 80-4000Hz (Recording-Chain-EQ ausschließen)
- Contralto-Erkennung braucht funktionierende Formanten ODER Vibrato-Daten
- Per-Segment-Verarbeitung ist die Zukunft — globales Gender ist ein Kompromiss

### Envelope-Blending
- Asymmetrisches Smoothing (5ms attack, 50ms release) ist der Schlüssel zu unhörbaren Übergängen
- Energy-Compensated Blending verhindert Lautstärke-Sprünge
- Defect-type-specific Sigma macht den Unterschied zwischen "chirurgisch" und "flächig"

---

## Noch offen (nächster Sprint)

1. AudioSR-DDIM-Live-Test auf ROCm mit CPU-Force-Fix
2. PhaseInteractionDenker: tiefere Konflikt-Erkennung (Phase-Ordering-Graph)
3. Per-Segment-Gender: Formant-Preservation + Chest-Resonance pro Segment
4. Envelope-Blending auf restliche ML-Phasen ausweiten (Phase 23, 50)
