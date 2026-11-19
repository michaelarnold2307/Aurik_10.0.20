# Kritischer Bug: PhasePruner Defekt-Namen-Mismatch — 2026-07-09

## Schweregrad: KRITISCH

Der `IntelligentPhasePruner` in `backend/core/phase_pruner.py` arbeitete mit
Defekt-Namen, die zu **76% nicht im `DefectType`-Enum des Scanners existierten**.
Dadurch wurden Phasen fälschlich geprunt — selbst wenn die entsprechenden
Defekte mit hoher Severity detektiert waren.

## Entdeckung

Ausgelöst durch die Beobachtung, dass ein MP3-File mit 25 detektierten
Defektarten nur 19 Phasen durchlief (34→19, 44% Reduktion). Der User fragte:
„Wenn 25 Defektarten erkannt werden, sollten doch mehr Phasen zum Einsatz kommen?"

## Root Cause

Der PhasePruner (`_PHASE_DEFECT_REQUIREMENTS`) verwendete **ad hoc erfundene**
Defekt-Namen, die nie mit den kanonischen `DefectType`-Enum-Werten abgeglichen
wurden:

```
Pruner:         "wow_flutter"      Scanner: WOW="wow", FLUTTER="flutter"
Pruner:         "click"            Scanner: CLICKS="clicks"
Pruner:         "speed_error"      Scanner: SPEED_CALIBRATION_ERROR="speed_calibration_error"
Pruner:         "bandwidth_limited" Scanner: BANDWIDTH_LOSS="bandwidth_loss"
Pruner:         "hiss"             Scanner: existiert nicht
Pruner:         "tape_hiss"        Scanner: existiert nicht
Pruner:         "buzz"             Scanner: existiert nicht
...
```

Die Match-Logik (`d in defect` — Substring) rettete einige Fälle:
- `"dropout" in "dropouts"` → True ✅ (Zufall)
- `"rumble" in "low_freq_rumble"` → True ✅ (Zufall)
- Aber: `"wow_flutter" in "wow"` → False ❌
- Und: `"bandwidth_limited" in "bandwidth_loss"` → False ❌

## Quantitative Analyse

```
Fehlende Pruner-Namen: 19/25 (76%)
  "bandwidth_limited" → Scanner hat: bandwidth_loss
  "buzz"              → Scanner hat: NICHTS
  "click"             → Scanner hat: clicks (Substring-Match rettet es)
  "distortion"        → Scanner hat: overload_distortion, intermodulation_distortion
  "dropout"           → Scanner hat: dropouts (Substring-Match rettet es)
  "ess"               → Scanner hat: sibilance
  "hiss"              → Scanner hat: NICHTS
  "noise"             → Scanner hat: high_freq_noise, modulation_noise, quantization_noise
  "phase_error"       → Scanner hat: phase_issues
  "pitch_error"       → Scanner hat: pitch_drift
  "pop"               → Scanner hat: NICHTS
  "rumble"            → Scanner hat: low_freq_rumble (Substring-Match rettet es)
  "spectral_gap"      → Scanner hat: NICHTS
  "speed_error"       → Scanner hat: speed_calibration_error
  "subsonic"          → Scanner hat: NICHTS
  "surface_noise"     → Scanner hat: NICHTS
  "tape_hiss"         → Scanner hat: NICHTS
  "transient_loss"    → Scanner hat: transient_smearing
  "wow_flutter"       → Scanner hat: wow, flutter, multiband_wow_flutter
```

## Betroffene Phasen

Phasen, die **trotz vorhandener Defekte** fälschlich geprunt wurden:

| Phase | Defekte vorhanden | Pruner-Name | Match? |
|---|---|---|---|
| phase_06 (Frequency Restoration) | bandwidth_loss=1.0 | "bandwidth_limited" | ❌ |
| phase_12 (Wow/Flutter Fix) | wow=1.0, flutter=1.0 | "wow_flutter" | ❌ |
| phase_31 (Speed/Pitch Correction) | pitch_drift | "pitch_error" | ❌ |

Nur Phasen mit Glücks-Matches (crackle, azimuth_error, sibilance, reverb_excess)
oder leeren Requirements (phase_04, phase_13, phase_16, phase_37, phase_38,
phase_54) überlebten den Pruner.

## Fix

`_PHASE_DEFECT_REQUIREMENTS` komplett auf echte `DefectType.values()` abgestimmt.

**Vorher:**
```python
"phase_12_wow_flutter_fix": ["wow_flutter", "speed_error"],
"phase_06_frequency_restoration": ["clipping", "bandwidth_limited"],
"phase_29_tape_hiss_reduction": ["hiss", "tape_hiss"],
```

**Nachher:**
```python
"phase_12_wow_flutter_fix": [
    "wow", "flutter", "multiband_wow_flutter", "scrape_flutter",
    "speed_calibration_error", "transport_bump", "pitch_drift",
],
"phase_06_frequency_restoration": ["clipping", "bandwidth_loss"],
"phase_29_tape_hiss_reduction": ["modulation_noise", "high_freq_noise"],
```

## Ergebnis nach Fix

- **0/27 Namen fehlen** im Scanner
- Bei reel_tape/cassette/vinyl mit echten Defekten: **deutlich mehr Phasen aktiv**
- Substring-Match bleibt erhalten: `"dropout" in "dropouts"` funktioniert weiter

## Impact auf verschiedene Medien

| Medium | Vor Fix (Phasen) | Nach Fix (erwartet) |
|---|---|---|
| mp3_low (unser Test) | 19 | ~22–24 |
| reel_tape mit Hiss+Azimuth+Dropout+Wow | ~18 | ~27–30 |
| vinyl mit Crackle+Rumble+Wow | ~16 | ~22–25 |
| cassette mit Hiss+Flutter+Azimuth+BW-Loss | ~18 | ~26–29 |

## Lehre für die Zukunft

**Defekt-Namen in Maps/Dicts MÜSSEN aus der kanonischen Quelle (`DefectType enum`)**
**abgeleitet werden, niemals ad hoc erfunden.** Ein Compliance-Check analog zum
`check_staticmethod_self.py`-Script kann dies zukünftig verhindern.

## Datum

2026-07-09 — Entdeckt & gefixt durch Kun
