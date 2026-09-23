---
applyTo: "backend/core/musical_goals/*.py"
---

# Musical Goals — Regeln (normativ, Aurik 10.2.0.x)

## 15 Goals — Prioritäten und kanonische Böden

| Prio | Goal | Restoration-Boden | Studio-2026-Boden |
|---|---|---|---|
| **P0** ⚠️ | **vocal_quality** | ≥ 0.85 (wenn `panns_singing ≥ 0.35`); VQI-Recovery-Trigger < `material_vqi_floor` (Shellac 0.62 / Vinyl 0.72 / CD 0.82) | ≥ 0.90; VQI-Recovery-Trigger < `material_vqi_floor` |
| **P0** ⚠️ | **formant_fidelity** | ≥ 0.88 (F1–F4 LPC-Track); Überschreitung `resolve_formant_tolerance_db()` → Rollback | ≥ 0.92 |
| **P1** | natuerlichkeit | ≥ 0.90 | ≥ 0.92 |
| **P1** | authentizitaet | ≥ 0.88 | ≥ 0.90 |
| **P2** | tonal_center | ≥ 0.95 | ≥ 0.96 |
| **P2** | timbre_authentizitaet (`timbre`) | ≥ 0.87 | ≥ 0.89 |
| **P2** | artikulation | ≥ 0.88 | ≥ 0.90 |
| **P3** | emotionalitaet | ≥ 0.84 | ≥ 0.87 |
| **P3** | micro_dynamics (`mikrodynamik`) | ≥ 0.88 (Frame-Energie-Korrelation voiced-Zonen ≥ 0.97; V20 §2.75) | ≥ 0.90 |
| **P3** | transient_energie | ≥ 0.80 (§1.4.6 V26: HPSS-Onset-Fenster 0–20 ms nach Transient geschützt) | ≥ 0.83 |
| **P3** | groove | ≥ 0.83 | ≥ 0.85 |
| **P4** | transparenz | ≥ 0.82 | ≥ 0.85 |
| **P4** | waerme | ≥ 0.77 (war: 0.75; v10.0.0 Wärmeband-Guard V25: 200–800 Hz kumulativer Verlust ≤ 2.5 dB) | ≥ 0.78 |
| **P4** | bass_kraft | ≥ 0.78 | ≥ 0.80 |
| **P4** | separation_fidelity (`sep_fidelity`) | ≥ 0.80 | ≥ 0.83 |
| **P5** | brillanz | ≥ 0.78 | ≥ 0.82 |
| **P5** | spatial_depth (`raumtiefe`) | ≥ 0.70 | ≥ 0.74 |

> **P0-Goals sind Vokal-exklusiv**: Nur aktiv wenn `panns_singing ≥ 0.35`. P0-Unterschreitung → Recovery-Kaskade (**nicht** gleichwertig zu `artifact_freedom < 0.95`: VQI ist Recovery-Trigger, kein harter Export-Block).  
> **VERBOTEN**: Böden hardcoden. **RICHTIG**: `calibration_matrix.get_material_floor(material_type, goal)` — material-adaptive Böden: Shellac ~0.72, Vinyl ~0.82, CD ~0.90.

## `measure_all()` — Rückgabe-Invariante

```python
def measure_all(audio: np.ndarray, sr: int, **kwargs) -> dict[str, float]:
    """Misst alle 15 Goals. MUSS immer dict[str, float] zurückgeben — niemals None."""
    results: dict[str, float] = {}
    for goal_name, metric in self._metrics.items():
        try:
            results[goal_name] = float(metric.measure(audio, sr, **kwargs))
        except Exception:
            results[goal_name] = 0.0  # nie None, nie KeyError
    return results
```

## Kanonische Mess-Algorithmen pro Goal

### natuerlichkeit (P1)

```python
# Primär: DNSMOS P.835 (Microsoft, 2022) — automatisches MOS-äquivalent:
# Output: SIG (signal quality), BAK (background noise), OVR (overall) — Skala 1–5
# Normierung auf [0,1]: (dnsmos_ovr - 1.0) / 4.0
# Fallback: Spectral Flatness + THD + Cross-Correlation-Stationarität
# Für Gesangsmaterial: SingMOS (Singer-Quality MOS, 2023) bevorzugt statt DNSMOS
from backend.core.dsp.quality_predictors import get_dnsmos_predictor, get_singmos_predictor
if panns_singing >= 0.35:
    score = get_singmos_predictor().predict(audio, sr)  # [1,5] → normiert auf [0,1]
else:
    score = get_dnsmos_predictor().predict(audio, sr)["ovr"]
natuerlichkeit = float(np.clip((score - 1.0) / 4.0, 0.0, 1.0))  # Clip: Predictor kann theoretisch [1,5] verletzen
```

### tonal_center (P2)

```python
# Krumhansl-Schmuckler Tonarterkennung + NNLS Chroma (Mauch & Dixon, 2010):
# Chroma-Vektor über komplettes Stück + Korrelation mit Dur/Moll-Profilen
# Stabilität = Anteil der Frames mit Übereinstimmung zur dominanten Tonartstufe
from backend.core.dsp.tonal_analysis import compute_tonal_stability
tonal_center = compute_tonal_stability(audio, sr)  # [0, 1]
# VERBOTEN: reiner Pitch-Korrelations-Proxy ohne Tonarterkennung
```

### waerme (P4)

```python
# Warmth = Energie-Anteil im Bassbereich + Oberton-Balance:
# Warmth-Proxy: E(100-400 Hz) / E(100-8000 Hz) — psychoakustisch validiert
# Zusätzlich: Verhältnis H2/H3 zu H1 (zweite/dritte Harmonische zum Grundton)
# Normierung gegen Referenz-Signal (pre-Phase)
# V25 / §2.76: Kumulativer Wärmeband-Verlust (200–800 Hz) > 2.5 dB über alle Phasen
# → `measure_warmth_band_delta()` aus `backend/core/dsp/warmth_guard.py` in jeder Phase
# → `_restoration_context["warmth_band_loss_db"]` aktuell halten
# → bei Überschreitung: `warmth_blend = 1 - loss_db / 5.0` als Phase-Blend-Faktor
# VERBOTEN: Wärme-Score > 0.77 (Restoration) ohne aktiven Wärmeband-Guard in der Pipeline
from scipy.signal import butter, sosfilt
# audio_filtered_100_400: Bandpass 100–400 Hz (Butter 4. Ordnung, zero-phase via sosfiltfilt)
sos_low = butter(4, [100, 400], btype="bandpass", fs=sr, output="sos")
audio_filtered_100_400 = sosfiltfilt(sos_low, audio)  # VERBOTEN: sosfilt (nicht zero-phase)
# audio_filtered_100_8000: Bandpass 100–8000 Hz (gesamter relevanter Energiebereich)
sos_total = butter(4, [100, 8000], btype="bandpass", fs=sr, output="sos")
audio_filtered_100_8000 = sosfiltfilt(sos_total, audio)
low_energy = np.mean(audio_filtered_100_400 ** 2)
total_energy = np.mean(audio_filtered_100_8000 ** 2) + 1e-9
waerme = float(np.clip(low_energy / total_energy * 3.5, 0.0, 1.0))
# Faktor 3.5: empirisch für Normierung auf [0,1] bei typischer Musikproduktion
```

### groove (P3)

```python
# Beat-Tracking + Onset-Regularität:
# 1. MADMOM beat tracker (BeatNet oder DBNBeatTracker) → beat_times
# 2. Onset-Dichte und -Regularität (interonset-interval-Varianz)
# 3. Rhythmische Komplexität: Anteil off-beat Onsets (Synkopen)
# Groove-Score = beat_consistency × (1 + syncopation_weight) × tempo_stability
from backend.core.dsp.rhythm_analysis import compute_groove_score
groove = compute_groove_score(audio, sr)  # [0, 1]
# VERBOTEN: einfache Tempo-Konstanz als Groove (ignoriert Synkopen/Off-Beats)
# VERBOTEN: Groove mit Spektral-Merkmalen approximieren
```

### sep_fidelity (P4)

```python
# Stem-Separation-Fidelität: wie gut sind Vokal/Instrument nach Restaurierung trennbar?
# Proxy: Signal-zu-Artefakt-Ratio für Haupt-Stem (leichtgewichtig, kein volles Demix):
# HTDemucs 4-stem → vocal_stem SDR vs. pre-processing-vocal_stem SDR
# VERBOTEN: sep_fidelity ohne tatsächliche Stem-Trennung schätzen
from plugins.htdemucs_plugin import get_htdemucs_plugin  # lightweight separation
stems = get_htdemucs_plugin().separate_quick(audio, sr)
# _MATERIAL_MAX_SDR: theoretisch erreichbarer SDR-Deckel für Vocal-Stem pro Träger
_MATERIAL_MAX_SDR = {
    "shellac": 6.0,    # SNR ~15 dB — sehr begrenzte Trennbarkeit
    "vinyl": 12.0,     # SNR ~60 dB — gute Trennbarkeit
    "tape": 10.0,      # SNR ~60-70 dB — HF-Hiss reduziert SDR
    "cd": 16.0,        # SNR ~96 dB — volle Trennbarkeit
    "digital": 16.0,   # wie CD
    "mp3_high": 14.0,  # kompressionsbedingte SDR-Reduktion
    "mp3_low": 10.0,   # starke Codec-Artefakte begrenzen SDR
    "unknown_analog": 10.0,  # konservativer Universal-Fallback
}
sep_fidelity = float(np.clip(
    stems["vocal_sdr"] / _MATERIAL_MAX_SDR.get(material, 10.0),
    0.0, 1.0,  # SDR kann negativ sein (schlechte Trennung) → 0.0 statt negativer Score
))
```

## Material-adaptive Böden — Warum korrekt

```
Shellac (1920-1950): SNR ~15 dB, BW ~7 kHz, Mono
→ natuerlichkeit 0.90 ist physikalisch UNMÖGLICH auf diesem Medium
→ material_floor: natuerlichkeit ≈ 0.72

Vinyl (1950-1990): SNR ~60 dB, BW ~16 kHz
→ material_floor: natuerlichkeit ≈ 0.82

CD (1980+): SNR ~96 dB, BW ~22 kHz
→ material_floor: natuerlichkeit ≈ 0.90

VERBOTEN: Alle Böden auf CD-Wert anheben
→ Shellac-Restaurierungen wären permanenter Fail → Recovery-Kaskade sinnlos aktiv
```

## Per-Song Studio-Day-Target (§0k)

```python
# KANONISCH — VOR Pipeline:
from backend.core.studio_goal_targets import estimate_song_goal_targets

studio_targets = estimate_song_goal_targets(
    era_decade=era_decade,          # z.B. 1970
    genre_label=genre_label,        # z.B. "schlager"
    material_chain=material_chain,  # z.B. ["vinyl", "mp3_low"]
    restorability=restorability,    # 0-100
)
# Beispiele:
# 1920er Shellac: brillanz≈0.52, raumtiefe≈0.30 (Mono)
# 1970er Schlager: brillanz≈0.80, waerme≈0.85
# 1990er CD-Pop:   brillanz≈0.88, transparenz≈0.90

# PhaseConductor stoppt Enhancement sobald goal ≈ studio_targets[goal]
# VERBOTEN: Phasen über studio_targets[goal] optimieren ohne neue Signal-Evidenz
```

## §2.56 Per-Song-Gewichtung

```python
from backend.core.song_goal_importance import estimate_goal_importance

goal_weights = estimate_goal_importance(audio, sr, metadata)
# 5-stufige Kaskade: Label/Audio/Psychoakustik/Vokal-Harmonik/Interactions
# Bereich: [0.30, 2.00]
# P0/P1-Floor ≥ 0.70 (darf nie auf 0 gesetzt werden)
# Wenn panns_singing ≥ 0.35: vocal_quality + formant_fidelity Gewicht MINIMUM 1.80
```

## §2.56a Harmonik-Adaptation (advisory)

```python
# _compute_harmonic_adaptation_scalar() → Bereich [0.72, 1.18]
# advisory-only in UV3 _profiled_phase_call
# Explizite PMGG-Strength hat VORRANG vor diesem Scalar
```

## VQI — Vocal Quality Index (P0-Pflichtmäßig ab `panns_singing ≥ 0.35`)

```python
# KANONISCH (musical_goals-Ebene — nur compute_vqi aufrufen + Ergebnis zurückgeben):
from backend.core.musical_goals.vocal_quality_index import compute_vqi

result = compute_vqi(audio_orig=original_audio, audio_restored=restored_audio, sr=sr)
vqi_score = result["vqi"]  # float [0, 1]
# Rückgabe-Dict enthält außerdem: singer_identity_cosine, hnr_delta, formant_deviation

# Material-adaptive VQI-Böden:
# Shellac: 0.62  |  Vinyl: 0.72  |  CD/Digital: 0.82
# VQI < material_floor → Recovery-Kaskade, nie harter Export-Stopp
#
# WICHTIG: Das Export-Gate (Recovery-Kaskade-Logik) läuft in UV3 (unified_restorer_v3.py)!
# musical_goals/*.py ruft compute_vqi() auf und gibt result zurück — KEIN _recovery_cascade()-Aufruf
# hier. UV3 liest result["vqi"] und entscheidet über Recovery. Vollständige UV3-Gate-Logik:
# → pipeline.instructions.md, Abschnitt "VQI-Gate (Gesangsmaterial)"
```

## EraVocalProfile — Ära-adaptive VQI-Kalibrierung [RELEASE_MUST v10.0.0]

**Problem**: `compute_vqi()` verwendet feste Formant-Toleranz (±2 dB für F1–F4). Ein Falsett-Sänger von 1932 hat physikalisch andere Vibrato-Charakteristika und eine höhere nasale Resonanz als ein Popsänger von 1985. Die feste Toleranz erzeugt systematisch falsch-negative VQI-Scores für historisches Material und falsch-positive Recovery-Kaskaden, die gar nicht notwendig sind.

**Lösung**: `EraVocalProfile` übergibt ära-spezifische Toleranzwerte an `compute_vqi()`.

```python
# backend/core/musical_goals/era_vocal_profile.py
from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class EraVocalProfile:
    vibrato_hz_range: Tuple[float, float]  # typische Vibrato-Frequenz dieser Ära
    f1_tolerance_db: float                  # F1-Abweichungstoleranz in dB
    f2_f4_tolerance_db: float               # F2–F4-Abweichungstoleranz in dB (meist enger)
    nasality_expected: bool                 # Nasalität als Stilmittel dieser Ära
    dynamic_range_typical_lu: float         # typisches Dynamikfenster (für emotional_arc)

ERA_VOCAL_PROFILES: dict[str, EraVocalProfile] = {
    "1900_1925": EraVocalProfile(
        vibrato_hz_range=(5.0, 10.0), f1_tolerance_db=4.0, f2_f4_tolerance_db=3.5,
        nasality_expected=True,  dynamic_range_typical_lu=8.0,
    ),
    "1925_1945": EraVocalProfile(
        vibrato_hz_range=(5.5, 9.0),  f1_tolerance_db=3.5, f2_f4_tolerance_db=2.5,
        nasality_expected=True,  dynamic_range_typical_lu=10.0,
    ),
    "1945_1960": EraVocalProfile(
        vibrato_hz_range=(5.0, 7.5),  f1_tolerance_db=3.0, f2_f4_tolerance_db=2.0,
        nasality_expected=False, dynamic_range_typical_lu=12.0,
    ),
    "1960_1975": EraVocalProfile(
        vibrato_hz_range=(4.5, 7.0),  f1_tolerance_db=2.5, f2_f4_tolerance_db=2.0,
        nasality_expected=False, dynamic_range_typical_lu=14.0,
    ),
    "1975_plus": EraVocalProfile(
        vibrato_hz_range=(4.0, 7.0),  f1_tolerance_db=2.0, f2_f4_tolerance_db=2.0,
        nasality_expected=False, dynamic_range_typical_lu=16.0,
    ),
}

def get_era_vocal_profile(era_decade: int) -> EraVocalProfile:
    """Gibt das passende EraVocalProfile für ein Jahrzehnt zurück."""
    if era_decade < 1925:   return ERA_VOCAL_PROFILES["1900_1925"]
    if era_decade < 1945:   return ERA_VOCAL_PROFILES["1925_1945"]
    if era_decade < 1960:   return ERA_VOCAL_PROFILES["1945_1960"]
    if era_decade < 1975:   return ERA_VOCAL_PROFILES["1960_1975"]
    return ERA_VOCAL_PROFILES["1975_plus"]
```

**Übergabe an `compute_vqi()`** — Signatur-Update:

```python
def compute_vqi(
    audio_orig:     np.ndarray,
    audio_restored: np.ndarray,
    sr:             int,
    era_profile:    "EraVocalProfile | None" = None,  # NEU — era_profile=None → bisheriges Verhalten
) -> dict[str, float]:
    # Wenn era_profile gegeben:
    #   f1_tol         = era_profile.f1_tolerance_db        (statt fix 2.0 dB)
    #   f2_f4_tol      = era_profile.f2_f4_tolerance_db     (statt fix 2.0 dB)
    #   vibrato_range  = era_profile.vibrato_hz_range        (Vibrato-Gate-Anpassung)
    #   nasality_penalty_factor = 0.0 if era_profile.nasality_expected else 1.0
    # Wenn era_profile=None: bisheriges Verhalten (Toleranz 2.0 dB fix, Vibrato 4–7 Hz fix)
```

**Kanonischer Aufruf (UV3, VQI-Gate)**:

```python
from backend.core.musical_goals.era_vocal_profile import get_era_vocal_profile

era_profile = get_era_vocal_profile(era_decade)  # aus SongCalibration/EraClassifier
result = compute_vqi(
    audio_orig=original_audio,
    audio_restored=restored_audio,
    sr=sr,
    era_profile=era_profile,
)
```

**VERBOTEN**: `compute_vqi()` auf Material mit `era_decade < 1960` ohne `era_profile` aufrufen — festes ±2 dB erzeugt systematisch falsch-negative VQI-Scores für historische Vokalstile.

**Tests**: `tests/unit/test_era_vocal_profile.py` — mind. 5 Tests: `get_era_vocal_profile`-Grenzwerte (1900, 1924, 1925, 1975+), `compute_vqi`-Aufruf mit und ohne era_profile (Rückgabe-Struktur), Toleranz-Wirkung auf formant_fidelity (Mock-Formant-Abweichung 3.0 dB → je nach era_profile: fail/pass).

## Frisson-Schutz (§Frisson)

```python
# frisson_zones Format: List[FrissonZone] — FrissonZone ist ein Dataclass-Objekt
# mit den Attributen .start_s: float und .end_s: float (KEIN Tuple)
# start_s, end_s: Zeitstempel in Sekunden (float), relativ zum Beginn des Tracks
# Beispiel: [(45.2, 47.8), (118.5, 122.1)]  — zwei Klimax-Passagen
# VERBOTEN: MDEM ohne frisson_zones
# RICHTIG:
from backend.core.frisson_candidate_detector import get_frisson_detector
frisson_zones = get_frisson_detector().detect(audio_original, sr)
# frisson_zones: Liste von FrissonZone-Objekten (.start_s/.end_s) — Klimax-Passagen
# MDEM respektiert frisson_zones: wet_mix = 0.0 in diesen Zonen
# Exception → frisson_zones = [] (non-blocking)
```

```python
# 15% Nudge aus SongGoalFeedbackStore nach Stufe 7:
nudges = SongGoalFeedbackStore.get_nudges(song_fingerprint)
for goal, nudge in nudges.items():
    # studio_targets.get() statt [] — KeyError wenn goal nicht in estimate_song_goal_targets()-Rückgabe
    effective_target[goal] = 0.85 * studio_targets.get(goal, effective_target.get(goal, 0.0)) + 0.15 * nudge
```

## Regressions-Regime (P1/P2 vs. P3-P5)

```python
# P1/P2 (natuerlichkeit, authentizitaet, tonal_center, timbre, artikulation):
# → Pipeline-Ende-PFLICHT: am Ende MÜSSEN alle ≥ Schwellwert sein
# → Einzelphasen dürfen vorübergehend senken wenn Carrier-Repair Grund ist
# → §2.29c Baseline-Capping gilt für restorative Phasen

# P3-P5 (emotionalitaet, mikrodynamik, groove, transparenz, ...):
# → Pipeline-Netto-Budget: Einzelphasen dürfen vorübergehend sinken
# → PMGG loggt Zwischenregressionen, blockiert aber nicht
```

## Vocal Quality Index (VQI, §2.35c)

```python
from backend.core.musical_goals.vocal_quality_index import compute_vqi

# PFLICHT bei panns_singing >= 0.35:  (kanonischer Name, s. VQI-Gate oben)
result = compute_vqi(audio_orig=original_audio, audio_restored=restored_audio, sr=sr)
vqi = result["vqi"]  # float [0, 1]
metadata["vqi"] = vqi
metadata["singer_identity_cosine"] = result.get("singer_identity_cosine", 0.85)
# Schwellwert: 0.72 → darunter Recovery-Kaskade (kein harter Veto)

# singer_identity_cosine-Gate (KANONISCH — NACH compute_vqi):
# cos_sim < 0.92 → Rollback letzter Vokal-Phase; Gate deaktiviert bei multi_singer=True
# WICHTIG: Dieser Code läuft in UV3 (unified_restorer_v3.py) — NICHT in musical_goals-Modulen!
# musical_goals/*.py ruft nur compute_vqi() auf und liefert das result-Dict zurück.
# UV3 liest result["singer_identity_cosine"] und entscheidet über Rollback.
# → _rollback_last_vocal_phase() ist eine UV3-interne Funktion (s. pipeline.instructions.md)
#
# Referenz-Code (für UV3-Entwicklung, nicht für musical_goals-Implementierung):
# if not metadata.get("multi_singer", False):
#     sic = metadata["singer_identity_cosine"]
#     if sic < 0.92:
#         audio_restored = _rollback_last_vocal_phase(audio_restored)
# DSP-Fallback bei Resemblyzer-Ausfall: result.get("singer_identity_cosine", 0.85) → 0.85 ≥ 0.92 → kein Rollback
```

## Frisson-Schutz in Goal-Messungen

```python
from backend.core.frisson_candidate_detector import get_frisson_detector

# VOR MDEM-Aufruf:
try:
    frisson_zones = get_frisson_detector().detect(original_audio, sr)
except Exception:
    frisson_zones = []  # Non-blocking — Exception darf Goal-Messung nicht stoppen

# Zwei-Stufen-Invariante:
# Pre-SG + Post-SG: Frisson-Floor -1.0 LU
# SG verteilt sonst Dämpfung in Klimax-Passagen bis -8 LU zurück
```

## §MG-SSOT Einheitliche Ziel-Schwellenquelle [RELEASE_MUST]

Musical-Goal-Schwellen dürfen nur aus **einer** Quelle stammen, um Parallelwelten zu verhindern.

```python
# Einzige zulässige Schwellenquellen:
# 1) calibration_matrix.get_material_floor(...)
# 2) estimate_song_goal_targets(...)
# 3) FeasibilityController (max_achievable-Cap)

# VERBOTEN:
# - Goal-Schwellen in einzelnen Goal-Modulen hardcoden
# - Goal-Schwellen in Phasenmodulen überschreiben
# - Abweichende Schwellenpfade zwischen UV3 und MusicalGoalsChecker
```

Pflicht-Traceability:

```python
metadata["goal_threshold_source"] = {
    goal_name: {
        "material_floor": material_floor,
        "song_target": song_target,
        "feasibility_cap": feasibility_cap,
        "effective_threshold": effective_threshold,
    }
}
```

## §MG-LOCAL Locality-Closing für Zielerfüllung [RELEASE_MUST]

Zur robusten Zielerfüllung pro Song müssen Goals lokal (segmentweise) validiert werden,
nicht nur als globaler Mittelwert.

```python
# KANONISCH: mindestens 3 Segmente (25/50/75 Prozent) plus Frisson-Zonen
segments = build_goal_segments(audio, sr, include_frisson=True)
segment_scores = measure_goals_per_segment(audio, sr, segments)

# Goal gilt nur als erfüllt, wenn:
# - global >= threshold
# - und kein kritisches Segment deutlich darunter liegt
goal_pass = (
    global_score >= threshold
    and min(segment_scores[goal]) >= threshold - 0.05
)
```

**VERBOTEN**: Goal als "passed" markieren, wenn globale Mittelung lokale Ausreißer verdeckt.

## §MG-FEAS Erreichbarkeits-Invariante [RELEASE_MUST]

15/15 Zielschwellen sind ein **Optimierungsziel**, keine physikalisch absolute Zusage für jedes Signal.
Die finale Schwelle pro Goal MUSS den Feasibility-Cap respektieren (siehe §2.79 in pipeline.instructions.md).

```python
# KANONISCH:
effective_threshold = min(song_target_threshold, feasibility_max_achievable)

# VERBOTEN:
# - Zielschwellen über feasibility_max_achievable erzwingen
# - Failed markieren, wenn nur die physikalische Unerreichbarkeit Ursache ist
```

Quellenbasis: ITU-R BS.1116-3 und ITU-R BS.1534-3 (subjektive Qualitätsbewertung ist signal-/kontextabhängig).

## §MG-UPG Goal-Upgrade-Invariante [RELEASE_MUST]

Für die 15 musikalischen Ziele gilt: Nachweisbar bessere Zielerreichung innerhalb
der Safety-Grenzen führt zu Spezifikations-Upgrade, niemals zu Funktions-Downgrade.

```python
# KANONISCH:
candidate_better = (
    improved_goals_count >= 1
    and non_degraded_goals_count >= 14
    and artifact_freedom >= 0.95
    and (panns_singing < 0.35 or vqi_not_worse)
)

if candidate_better:
    decision = "promote_to_spec"
else:
    decision = "do_not_promote"

# VERBOTEN:
# - bessere Goal-Balance zurückdrehen, um alte statische Schwellen zu behalten
# - einzelne Goal-Maximierung akzeptieren, wenn Teamwork-Balance der 15 Ziele sinkt
```

Pflicht bei Promotion:

```python
metadata["goal_upgrade_decision"] = {
    "promoted": True,
    "improved_goals": improved_goals,
    "non_regression_goals": non_regression_goals,
    "safety_ok": safety_ok,
}
```
