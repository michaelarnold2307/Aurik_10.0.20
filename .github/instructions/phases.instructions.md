---
applyTo: "backend/core/phases/phase_*.py"
---

# Phasen-Regeln (normativ, Aurik 10.2.0.x)

## Pflicht-Checkliste bei jeder neuen Phase

```
1. process()-Signatur: (audio, sample_rate=48000, material_type="unknown", **kwargs) → PhaseResult
2. assert sample_rate == 48000 am Eingang
3. audio = np.clip(audio, -1.0, 1.0) am Ausgang
4. result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0) vor Return
5. logger.info("phase=%s score=%.2f", phase_id, score) — kein print()
6. CAUSE_TO_PHASES + CAUSES bidirektional ergänzen (V12)
     → Die vollständige Defektoberfläche aus backend/core/defect_scanner.py MUSS
         in _PHASE_MAP und in jedem defektgebundenen Recommender vollständig
         abgebildet sein; Teilmengen oder lose Sonderfall-Tabellen sind
         Release-Blocker.
7. Neue HPF/Notch-Phase: 4-stufige Checkliste (s. unten)
8. Wenn panns_singing ≥ 0.25: apply_hnr_blend() nach jeder ML-NR-Phase (§0p)
9. Wenn panns_singing ≥ 0.25: Formant-Delta via lpc_formant_tracker prüfen (§0p)
    → ΔF1–F4 > resolve_formant_tolerance_db(era_decade, era_profile): sofortiger Rollback auf Phase-Input — fixer 2-dB-Wert nur als moderner Fallback
10. Vibrato-Zonen (4–7 Hz F0): strength = min(strength, 0.20) (§0p)
    → detect_performance_artifacts() VOR der Phase ausführen
11. Alle Instanzattribute MÜSSEN in __init__ deklariert werden (W0201)
    → Attribute die erst in process() gesetzt werden: in __init__ mit Sentinel initialisieren
    → Beispiel: self._omlsa_panns_singing: float = 0.0  # gesetzt in process()
12. [V33] Jede Phase mit material-indizierten Dicts (dict[MaterialType, ...]) MUSS bei Einführung
    eines neuen MaterialType-Members vollständig aktualisiert werden:
    → DETECTION_THRESHOLD, CORRECTION_STRENGTH und alle weiteren dict[MaterialType, ...]-Variablen
    → Wert aus IEC-/RIAA-Standard ableiten — kein Kopieren aus physikalisch anderem Träger
    → Fallback auf dict.get(material_type, default_from_other_carrier) ist VERBOTEN für physikalisch
      relevante Parameter (lautloser Defekt — kein RuntimeError, aber falsche Verarbeitung)
    → Beispiel: CASSETTE braucht eigene DETECTION_THRESHOLD-Schwelle basierend auf IEC 60094-1
      (≤ 0,2 % WRMS Capstan/Pinch-Roller); Vinyl-Default 0.5 % übersieht Kassetten-Transport-Bumps
    → Pflicht-Test: test_phase_XX_all_material_types_in_DICT() für jede Phase mit material-indizierten Dicts
13. [V38] Phasen mit Event-Schleifen (bump_locations, splice_points, Dropout-Segmente usw.):
    → `vfa_protected_zones` aus kwargs lesen: `vfa_result = kwargs.get("vfa_result", {})` →
      `protected_zones = [(s, e, cap) ...]` aus vfa_result (Vibrato 0.20, Frisson 0.30, Flüster 0.25, Passaggio 0.35)
    → Per-Event-Strength-Oracle MUSS implementiert sein:
      `_compute_<defect>_local_strength(mono_ref, start, end, sr, base_strength, protected_zones)`
      mit 250 ms Kontext-RMS-Proxy; `base_strength < 1e-6` → 0.0 (kein Event-Processing)
    → Einheitliche `strength` für alle Events in der Schleife ist VERBOTEN

14. [RELEASE_MUST] Defect-Coverage-Registry:
        → Jede komplette Defect-Registry muss alle DefectType-Werte aus
            backend/core/defect_scanner.py abdecken.
        → Vollständigkeits-Tests sind Pflicht; fehlende Defekte oder silent fallbacks
            gelten als unvollständig implementiert.

Hinweis zur Strength-Übergabe (kanonisch):
        - `strength` wird aus `kwargs` gelesen (z. B. `strength = float(kwargs.get("strength", 0.7))`).
        - `strength` darf NICHT als dritter Positionsparameter die `material_type`-Position ersetzen.
        - Contract-Referenz ist `backend/core/phases/phase_interface.py`.
```

## §2.46 Carrier-Chain-Inversion — Stufenreihenfolge (HARD)

```
Stufe 1: ADC-Artefakte     → phase_30 (DC), phase_31 (Quantisierung)
Stufe 2: Playback          → phase_04 (RIAA), phase_25 (Azimuth), phase_12 (Wow/Flutter)
         (VERBOTEN: phase_06 hier — phase_06 = BW-Erweiterung → gehört zu Stufe 5!)
         RIAA-Entzerrkurve: IEC 60098 / RIAA-Standard:
           Zeitkonstanten: 3180 μs (50 Hz Pol) | 318 μs (500 Hz Pol/Null) | 75 μs (2122 Hz Null)
           Boost LF: +20 dB bei 50 Hz; Cut HF: -20 dB @ 20 kHz (vor Entzerrung)
           Entzerrung: invers (LF anheben, HF bedämpfen)
           VERBOTEN: Näherungen ohne alle 3 Zeitkonstanten
Stufe 3: Alterungsschäden  → phase_09 (Knistern), phase_24 (Dropout)
Stufe 4: Carrier subtraktiv→ phase_29 (Bandrauschen), phase_03 (Surface Noise)
Stufe 4.5: Pegelkorrektur  → phase_40 (AMPLITUDE_DRIFT, nur wenn amplitude_drift_correction=True)
           REIHENFOLGE HARD: nach Stufe 4 (NR darf Gain-Detektion nicht verfälschen)
                             vor Stufe 5 (BW-Obertöne nicht durch Envelope-Verzerrung verzerren)
           Bedingung: DefectScanner.AMPLITUDE_DRIFT severity ≥ 0.30 UND Drift-Slope ≥ 1.5 dB/min
           VERBOTEN: phase_40 in Standard-Lautstärke-Normierung ohne amplitude_drift_correction=True
Stufe 5: Carrier additiv   → phase_06/23 (BW-Erweiterung), phase_07 (Harmonik)
                              ↑ IMMER nach Stufe 4 — sonst werden rekonstruierte
                                Obertöne sofort entrauscht
Stufe 6: Mixer/Preamp      → BEWAHREN (Recording-Chain-Signatur = Original)

Digitale Carrier-Kette (MP3/AAC/OGG) — Reihenfolge HARD:
  Phase A: phase_50 (Psycho-Masking-Residue, Pre-Echo) — VOR analoger Stufe 1-4
  Phase B: phase_23 (BW-Erweiterung auf Codec-Cutoff) — nach Stufe 4
  Phase C: wenn analog+digital kombiniert: analog Stufe 1-4 → digital Phase A-B
  MP3-Pre-Echo (HARD): siehe unten §MP3
```

## §2.46e Hallucination-Guard — ADDITIVE Phasen (phase_37/38/48/32)

```python
# PFLICHT nach jeder additiven Operation:
from backend.core.dsp.hallucination_guard import check_hallucination

guard = check_hallucination(pre_audio, post_audio, sr, mode)
if guard.requires_rollback:
    return pre_audio  # spectral_novelty > 0.15 in Restoration
if guard.score_penalty > 0:
    phase_score -= 0.3  # spectral_novelty > 0.08

# Drei verbotene Halluzinations-Kategorien in Restoration:
# 1. Harmonik über BW-Ceiling des Materials
# 2. Raumklang/Reverb der nicht im Signal nachweisbar ist
# 3. ML-generierte Spektral-Texturen ohne physikalisches Gegenstück
```

## §2.46f Natural-Performance-Artifacts-Guard

```python
# Diese drei Kategorien sind KEINE Defekte — niemals entfernen:
# 1. Atemgeräusche: -55 bis -40 dBFS, 50-500ms, spectral_flatness > 0.4
# 2. Vibrato/Portamento: F0-Modulation 4-7 Hz, Amplitude ≤ ±50 Cent
# 3. Early Reflections: 0-50ms nach Onset → Dereverb wet_mix cap = 0.35

from backend.core.dsp.natural_performance_detector import detect_performance_artifacts
protected_segments = detect_performance_artifacts(audio, sr)
# Phasen müssen protected_segments respektieren
```

## §0a — Crossfire-Modus-Invariante (absolut)

```python
# VERBOTEN in Restoration — diese Phasen NIE aktivieren:
_RESTORATION_FORBIDDEN = {
    "phase_21_exciter",           # §0a: kein künstlicher Harmonik-Zusatz
    "phase_35_multiband_compression",  # §0a: nur Studio 2026
    "phase_42_vocal_enhancement", # §0a: nur Studio 2026
}
# Diese Phasen dürfen auch nicht in CAUSE_TO_PHASES für Restoration-Causes stehen
```

## Material-Ceiling-Pflicht bei ADDITIVEN Phasen

```python
from backend.core.dsp.physical_ceiling import _MATERIAL_BW_CEILING_HZ, _MATERIAL_DR_CEILING_DB

# BW-Erweiterung (phase_06/07/23):
max_freq = _MATERIAL_BW_CEILING_HZ[material]  # Shellac ≤ 8kHz, Vinyl ≤ 16kHz, Cassette ≤ 12kHz, Tape ≤ 15kHz
# Keine Harmonik/Energie über max_freq hinzufügen

# DR-Expansion (phase_26):
max_dr = _MATERIAL_DR_CEILING_DB[material]  # Vinyl ≤ 70dB, Shellac ≤ 45dB, Cassette ≤ 62dB
# Expansion über Ceiling = Artefakt → sofortiger Rollback
```

## Phase 23 — BW-Ceiling-First-Invariante (v10.0.0)

Generative/Inpainting-Phasen (phase_23) MÜSSEN `_apply_material_bw_ceiling()` **VOR** dem HallucinationGuard aufrufen:

```python
# RICHTIG: Ceiling zuerst, dann Guard
audio_out, ceiling_applied, ceiling_hz = cls._apply_material_bw_ceiling(audio_out, sr, material_type, mode)
halluc_result = check_hallucination(pre_audio, audio_out, sr, mode,
                                    material_bw_ceiling_hz=ceiling_hz)
# VERBOTEN: Ceiling nur innerhalb des Guards — Output wurde bereits über Ceiling synthetisiert
```

Cassette-spezifisch: `_material_bw_ceiling_hz("cassette") == 12000.0` (IEC 60094-1 Type I).
Kanonischer Wert ist konsistent in `tonal_reference_profile.py`, `goal_applicability_filter.py` und `phase_23._material_bw_ceiling_hz()`.

## §2.63 Boundary-Mechanismus (STFT/ML-Phasen)

```python
# KANONISCH — Reflect-Padding VOR STFT:
_pad_len = hop_length * 4
audio_padded = np.pad(audio, _pad_len, mode="reflect")
# ... STFT-Verarbeitung ...
audio_out = audio_out[_pad_len: _pad_len + n_original]  # deterministischer Strip

# VERBOTEN: np.pad(..., mode="constant") NACH STFT als primäre Längenkorrektur
# Stereo-Lag-Invariante: L + R MÜSSEN identischen _pad_len und Strip-Offset haben

# Overlap-Pflicht (Konsistenz mit dsp.instructions.md):
# hop_length = n_fft // 4  (75 % Overlap) — für Analyse UND Synthese
# Ausnahme: ML-Phasen (DFN, SGMSE+) nutzen interne Fensterung — nicht überschreiben
```

## HPF/Notch-Phase — 4-stufige Checkliste

```
1. KEIN Loudness-Guard in der Phase-Datei selbst
2. enable_loudness=False in _phase_overrides setzen
3. Phase-ID in _HPF_NOTCH_CUM_RESET_PHASES eintragen
4. _update_positive_makeup_authority aufrufen
```

## §2.63 Stereo-Lag-Invariante

```python
# Wenn L/R separat verarbeitet:
# VERBOTEN: Per-Channel-Resampling als Längenkorrektur
# RICHTIG: identische Kontextlänge, identischer Strip-Offset, identische Zielsamplezahl
# VERBOTEN: assert (durch python -O deaktivierbar — see copilot-instructions.md V01)
if not (len(audio_L_out) == len(audio_R_out) == n_original):
    logger.error("stereo_lag_invariant phase=%s L=%d R=%d expected=%d — cropping",
                 phase_id, len(audio_L_out), len(audio_R_out), n_original)
    audio_L_out = audio_L_out[:n_original]
    audio_R_out = audio_R_out[:n_original]  # UV3 §2.61 fängt verbliebene Abweichungen ab
```

## §MP3/AAC-Artefakte — Spezialfälle (phase_50)

```python
# MP3-Artefakte (MDCT-basierter Codec, ISO/IEC 11172-3):
# 1. Pre-Echo: Schmiervorgabe vor Transienten durch lange MDCT-Fenster (576 Samples)
#    → Symptom: Vorgeräusch 5–20 ms vor Attack (Becken, Plänkselgeräusche, S-Laute)
#    → Korrektur: Transient-gestützte fensterwechsel-adaptive Unterdrückung
#    → VERBOTEN: allgemeine NR auf Pre-Echo (zerstört Original-Transient)
# 2. Psychoakustisches Masking-Residue: Artefakt-Energie in maskierten Banden
#    sichtbar sobald Maskierung abläuft (temporal unmasking)
#    → Korrektur: zeitabhängiger Masking-Floor in diesen Banden höher halten
# 3. HF-Rolloff: bei 128 kbps typisch bei 16 kHz, bei 64 kbps bei 11 kHz
#    → Korrektur: AudioSR/NVSR für BW-Erweiterung VOR RIAA (phase_06/23 nach phase_50)
# 4. Granule-Seams: Block-Grenzen alle 576 Samples (ca. 12 ms @ 48kHz)
#    → Symptom: subtile Amplitudensprünge an Block-Grenzen
#    → Korrektur: Überlapp-Glättung über Grenzen

# AAC-Artefakte (ISO/IEC 13818-7, MPEG-4 AAC):
# 1. Stereo-Downmix-Artefakte (Joint Stereo): ähnlich MP3, aber mit Intensity-Stereo
#    und Mid-Side-Coding → bei tiefen Bitraten Stereobreite kollabiert
# 2. TNS (Temporal Noise Shaping): kann bei Vokalen Präsenz-Region dämpfen
```

## §BW-Erweiterung — Algorithmuswahl (phase_06/07/23)

```python
# Entscheidung nach Frequenzlücke:
# 0–8 kHz fehlt (Shellac/Telefon): AudioSR (Liu et al. 2023, SpeechFlow-Diffusion)
#   → resampelt intern auf 16 kHz, generiert bis 48 kHz mittels Diffusion
#   → VERBOTEN bei Studio 2026 ohne MUSHRA ≥ 3.5 (Hallucination-Guard aktiv)
# 8–16 kHz fehlt (MP3 128kbps): SBR-Heuristik (Spectral Band Replication) oder NVSR
#   NVSR: nicht-DNN-basiert, schneller, weniger Halluzinationsrisiko
# 16–22 kHz fehlt (Vinyl, Tape): harmonische Extrapolation (phase_07) mit G_max-Ceiling
# Material-Ceiling IMMER einhalten (_MATERIAL_BW_CEILING_HZ)
```

## §2.36 LyricsGuidedEnhancement — Offline-Pfad (Lücke 3)

```python
# Aurik ist 100 % offline — Lyrics-Quelle: integriertes Whisper-Tiny ONNX (39 MB)
# Prioritätskaskade (vollständig, keine ad-hoc-Alternative erlaubt):
try:
    # Stufe 1: Whisper-Tiny ONNX (primär, offline, CPUExecutionProvider)
    lge = LyricsGuidedEnhancement()
    transcript = lge.transcribe(audio, sr)  # kein Netzwerk
    if transcript.confidence >= 0.60:
        phoneme_mask = lge.get_phoneme_mask(audio, sr, transcript=transcript)
    else:
        # Stufe 2: DSP-Fallback bei Whisper-Confidence < 0.60
        from backend.core.dsp.phoneme_boundary_detector import detect_phoneme_boundaries_dsp
        phoneme_mask = detect_phoneme_boundaries_dsp(audio, sr)
        # Algorithmus: Energie-Differenz (25 ms Fenster) + ZCR-Schwelle
        # voiced: ZCR < 0.10 | unvoiced: ZCR > 0.15 | plosive: Energie-Spike
        metadata["lyrics_fallback"] = "dsp_phoneme_boundary"
except Exception as e:
    # Stufe 3: Beide fehlgeschlagen — phoneme_mask deaktiviert, kein Absturz
    logger.warning("lge_offline: whisper+dsp failed — phoneme_mask=None: %s", e)
    phoneme_mask = None
    metadata["lyrics_fallback"] = "disabled"

# phoneme_mask[frame] = True → NR-Bypass (Konsonanten-Burst geschützt)
# VERBOTEN: Lyrics-Text loggen oder in metadata schreiben (§2.36 Datenschutz)
```

## Frisson-Schutz in Phasen mit Dynamik-Eingriff

```python
from backend.core.frisson_candidate_detector import get_frisson_detector

frisson_zones = get_frisson_detector().detect(original_audio, sr)
# Klimax-Segmente: NR-Strength × 0.85
# Strophen: NR-Strength × 1.15
# MDEM: Zwei-Stufen pre-SG + post-SG, Floor -1.0 LU
```

## VQI-Gate für Vokal-Phasen

```python
# PFLICHT nach vokal-beeinflussenden Phasen:
from backend.core.musical_goals.vocal_quality_index import compute_vqi

if panns_singing >= 0.35:  # kanonischer Name: panns_singing (≠ panns_singing_confidence)
    # audio_orig=audio_in: Phase-Eingang als Referenz (lokaler Degradierungs-Check)
    # Nur lokale Regression pruefen, keine globale Exportentscheidung in der Phase.
    result_after = compute_vqi(audio_orig=audio_in, audio_restored=audio_out, sr=sr)
    vqi_after = result_after["vqi"]
    if (1.0 - float(vqi_after)) > 0.05:
        return audio_in  # lokaler Rollback bei starker VQI-Regression

    metadata["vqi_phase_after"] = float(vqi_after)
    metadata["phase_decision_scope"] = "local_only"
    metadata["final_authority"] = "UV3"
```

Normative Abgrenzung:

- Die globale Einordnung als Hard-Veto, Recovery-Trigger oder Advisory erfolgt
  ausschliesslich in UV3 (`quality_gate_registry`, `recovery_state_machine`).
- Phasen duerfen keine globale A/B/C-Klassifikation setzen oder uebersteuern.

## Passaggio-Glättung

```python
# PFLICHT in Pitch/Register-Phasen:
from backend.core.dsp.vocal_register_detector import detect_vocal_register_temporal

register_sequence = detect_vocal_register_temporal(audio, sr)
# glättet Brust→Kopf-Übergänge ±5 Frames linear
# verhindert Timbre-Knick bei Passaggio-Sprüngen
```

## §PH-SSOT Keine lokalen Parallel-Gates [RELEASE_MUST]

Phasenmodule dürfen keine eigenen globalen Qualitäts-Gates etablieren.
Sie liefern Signal + Telemetrie; die finale Exportentscheidung liegt ausschließlich bei UV3.

Die normative songweite Steuerquelle ist `restoration_policy_profile`. Phasen dürfen
`song_goal_weights` nur noch als daraus abgeleiteten Kompatibilitätswert lesen;
eigene alternative Zielprofile oder parallel berechnete Song-Weights sind verboten.

```python
# VERBOTEN in Phase-Dateien:
# - Eigene globale Goal-Schwellen (z. B. natuerlichkeit >= 0.90 hardcoded)
# - Eigene Export-Veto-Logik außerhalb klarer Per-Phase-Rollback-Invarianten
# - Eigene alternative "final audio"-Pfade

# ERLAUBT:
# - Per-Phase-Safety-Rollback (z. B. Formant/Vibrato/Hallucination)
# - Telemetrie in metadata
# - Vorschlag für Recovery-Hinweis (non-binding)
```

Pflicht bei jeder Phase:

```python
metadata["phase_decision_scope"] = "local_only"
metadata["final_authority"] = "UV3"
metadata["policy_source"] = "restoration_policy_profile"
```

## §PH-NAT Natürlichkeits-Priorität ohne Übersteuerung

Für das Ziel "klingt wie ohne Eingriff" muss jede Phase im Zweifel konservativ werden,
statt Zielmetriken aggressiv zu erzwingen.

```python
# KANONISCH:
# Wenn Phase-Delta mehrere Goals gleichzeitig verschlechtert,
# dann strength dämpfen und lokalen Blend wählen statt weiterer Aggression.
if n_regressive_goals >= 2:
    strength = max(0.0, strength * 0.7)
    metadata["multi_goal_conservative_blend"] = True
```

**VERBOTEN**: Ein Einzelziel lokal maximieren, wenn dadurch Natuerlichkeit/Authentizitaet sinkt.

## §PH-UPG Lokale Qualitäts-Upgrades konservieren [RELEASE_MUST]

Wenn eine Phase lokal nachweisbar bessere Ergebnisse liefert (bei erfüllten Safety-Grenzen),
darf diese Verbesserung nicht auf ein älteres Spezifikationsniveau zurückgenommen werden.
Die Spezifikation ist stattdessen anzupassen.

```python
# KANONISCH pro Phase:
local_upgrade = (
    perceptual_delta > 0.0
    and artifact_freedom_proxy_not_worse
    and not violates_formant_vibrato_hallucination_guards
)

if local_upgrade:
    metadata["phase_upgrade_candidate"] = True
    metadata["phase_upgrade_reason"] = "better_local_quality_with_safety"

# VERBOTEN:
# - strength künstlich zu reduzieren, nur um historisches Verhalten zu reproduzieren
# - Upgrade ohne Telemetrie/Begründung stillschweigend einzubauen
```

## §2.8b De-Esser-Aktivierungsinvariante (phase_43)

```python
# RELEASE_MUST für phase_43_ml_deesser:
# Bei hoher Sibilance/Vocal-Harshness darf PMGG den zweiten De-Esser-Pass
# nicht auf nahezu wirkungslos reduzieren.

defect_scores = kwargs.get("defect_scores", {})
sib_pressure = max(
    severity(defect_scores, "sibilance"),
    severity(defect_scores, "sibilance_excess"),
    severity(defect_scores, "vocal_harshness"),
)

effective_strength = clip(strength * phase_locality_factor, 0.0, 1.0)

if sib_pressure >= 0.55:
    control_floor = clip(0.30 + 0.40 * ((sib_pressure - 0.55) / 0.45), 0.30, 0.70)
else:
    control_floor = 0.0

control_strength = max(effective_strength, control_floor)

# control_strength steuert nur De-Esser-Kontrollparameter (ratio/threshold),
# nicht den globalen Wet/Dry-Pfad (effective_strength bleibt PMGG-Wert).
ratio = 1.0 + (ratio - 1.0) * control_strength
if sib_pressure >= 0.55:
    threshold_db = clip(threshold_db - clip(4.0 + 8.0 * ((sib_pressure - 0.55) / 0.45), 4.0, 12.0), -40.0, -6.0)

# Pflicht-Metadata für Diagnose/Audit:
metadata["sibilance_pressure"] = sib_pressure
metadata["control_strength"] = control_strength
metadata["effective_strength"] = effective_strength
```

```python
# §2.36 Fallback-Gate in phase_43:
# Kein phoneme_timeline + geringe Vocal-Wahrscheinlichkeit => Phonem-Fallback deaktivieren.
# Sonst droht ungewolltes Voll-Restore (Pass-Through) in nicht-vokalen/synthetischen Fällen.

if phoneme_timeline is None and float(kwargs.get("vocal_probability", 0.0)) < 0.25:
    skip_phoneme_fallback = True
```

## §2.8c Kompressionsdefekt-Hardintervention (phase_54)

```python
# RELEASE_MUST für phase_54_transparent_dynamics:
# Bei hohen Kompressionsdefekten darf PMGG/Locality die Dynamik-Reparatur
# nicht praktisch deaktivieren.

defect_scores = kwargs.get("defect_scores", {})
compression_pressure = max(
    severity(defect_scores, "compression_artifacts"),
    0.85 * severity(defect_scores, "dynamic_compression_excess"),
    0.35 * severity(defect_scores, "digital_artifacts"),
)

effective_strength = clip(strength * phase_locality_factor, 0.0, 1.0)

if compression_pressure >= 0.25:
    control_floor = clip(0.25 + 0.65 * ((compression_pressure - 0.25) / 0.75), 0.25, 0.90)
else:
    control_floor = 0.0

control_strength = max(effective_strength, control_floor)

# Hard-Intervention aktivieren ab hoher Defektlast:
if compression_pressure >= 0.45:
    hard_norm = clip((compression_pressure - 0.45) / 0.55, 0.0, 1.0)
    threshold_db = clip(threshold_db - (2.0 + 6.0 * hard_norm), -36.0, -6.0)
    ratio = clip(ratio + (0.4 + 1.2 * hard_norm), 1.5, 5.0)
    attack_ms = clip(attack_ms * (1.05 + 0.30 * hard_norm), 3.0, 120.0)
    release_ms = clip(release_ms * (1.15 + 0.55 * hard_norm), 40.0, 1400.0)

# Pflicht: zusätzliche Gain-Envelope-Glättung gegen Pumpen,
# aber transient_mask + masking_curve müssen weiter gelten.

# Pflicht-Metadata für Diagnose/Audit:
metadata["compression_pressure"] = compression_pressure
metadata["control_floor"] = control_floor
metadata["control_strength"] = control_strength
metadata["hard_intervention_active"] = compression_pressure >= 0.45
```
