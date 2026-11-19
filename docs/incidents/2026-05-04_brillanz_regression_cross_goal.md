# Incident Report: Brillanz-Regression nach Pegelexplosion-Fix

Datum: 2026-05-04
Status: Mitigation implementiert (Cross-Goal-Recovery), Verifikation ausstehend
Owner: UV3 / PMGG / Musical Goals

## Symptom

- Brillanz fällt auf `0.1196` (Schwelle ~`0.7455`), Transparenz auf `0.3764`, Waerme auf `0.6308`.
- GOAL_SCORECARD zeigt konsistent P4/P5-Verletzungen trotz sonst guter Pipeline-Werte.

Beleg (Run `deep_analysis_fix9_run.log`):

- `🎯 GOAL_SCORECARD ... brillanz:score=0.1196 ... transparenz:score=0.3764 ... waerme:score=0.6308`
- `phase_03_denoise ... action=best_effort_r1 ... strength=0.42`
- Danach additive HF-Phasen mit niedriger Stärke:
  - `phase_06_frequency_restoration ... strength=0.43`
  - `phase_07_harmonic_restoration ... strength=0.18`
  - `phase_39_air_band_enhancement ... strength=0.07`

## Root-Cause (exakt)

Dies ist eine Cross-Goal-Interaktion, kein einzelner Defekt:

1. `phase_03_denoise` landet in PMGG `best_effort*`.
2. Dadurch bleibt HF-Energie reduziert (subtraktive Dominanz).
3. Die folgenden HF-Recovery-Phasen (`phase_06/07/39`) laufen mit zu geringer effektiver Stärke.
4. End-of-pipeline MusicalGoalsChecker bewertet absolute HF-Klarheit (Brillanz/Transparenz) und fällt durch.

Wichtig: Der zuvor gelöste EAPC-Fehler (`_estimate_bandwidth`) ist hier **nicht** der aktuelle Primärtreiber; EAPC-HF-Synthese feuert im betroffenen Lauf nicht.

## Warum das nach einem anderen Fix wieder auftritt

Die Pegelexplosion-Fixes haben zurecht positive Gain-Pfade härter reguliert. Das verbessert Edge-Safety, reduziert aber implizit die Fähigkeit, nach aggressiven subtraktiven Schritten HF wiederaufzubauen, wenn `phase_03` bereits `best_effort` meldet. Ergebnis: Ein Problem gelöst, ein anderes verstärkt.

## Implementierte Mitigation (v10.0.0)

Datei: `backend/core/unified_restorer_v3.py`

### 1) Trigger

Wenn `phase_03_denoise` mit `best_effort*` endet (Restoration-Modus), wird ein Kontext-Flag gesetzt:

- `hf_recovery_boost_after_phase03 = { enabled, source_phase, source_action, source_strength }`

### 2) Wirkung

In `_profiled_phase_call` wird für die direkte HF-Recovery-Kette ein materialadaptiver Mindest-Strength-Floor erzwungen (nur implizite Strengths, keine Überschreibung expliziter Vorgaben):

- `phase_06_frequency_restoration`
- `phase_07_harmonic_restoration`
- `phase_39_air_band_enhancement`

Beispiel-Floors (analog):

- Vinyl: `phase_06 >= 0.56`, `phase_07 >= 0.28`, `phase_39 >= 0.16`

Ziel: HF-Recovery-Autorität nach `phase_03 best_effort` wiederherstellen, ohne global Over-Processing zu erzwingen.

## Tests

Ergänzt in `tests/unit/test_hebel_intelligence_levers.py` (Smoke-Wiring):

- Presence von `hf_recovery_boost_after_phase03`
- Presence der Cross-Goal-Recovery-Guard-Logik für `phase_06/07/39`

## Verifikationsplan

1. Re-Run des betroffenen Songs im gleichen Modus/Material.
2. Pflicht-Checks:
   - `phase_03` ggf. weiterhin `best_effort*` (zulässig)
   - neue Logs `§Cross-Goal-Recovery armed` und `§Cross-Goal-Recovery ... strength floor ...`
   - GOAL_SCORECARD: deutliche Erholung bei `brillanz` und `transparenz`
3. Guardrails:
   - Keine neuen Artefakte (`artifact_freedom >= 0.95`)
   - Keine Intro/Outro-Peaks (Pegelexplosion bleibt gelöst)

## Offene Punkte

- Falls Brillanz trotz Floor niedrig bleibt: gezielte Analyse der Brillanz-Metrik im Final-Checker (`BrillanzMetric`, crest-basierter Absolutscore) gegen reales HF-Spektrum des Outputs.
- Optional: End-Gate Recovery um eine explizite HF-Recovery-Variante erweitern (statt nur Blend-Kaskaden).
