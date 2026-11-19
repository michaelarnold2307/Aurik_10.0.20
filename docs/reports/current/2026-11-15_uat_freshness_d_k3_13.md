# D-K3-13 geschlossen — UAT-Evidenz 2026-11-15 (echter Produktionspfad)

**Datum:** 2026-11-15
**Lücke:** `D-K3-13` — `test_daily_gate_stored_recently` rot, weil die
UAT-Analyse vom 2026-11-11 bereits 4 Tage alt war; zusätzlich standen alle
15 Restoration-Kriterien im UAT-Report auf `ERROR`.
**Ergebnis:** geschlossen. Frische UAT-Evidenz liegt vor, der normative Test
ist grün. Kein Flag-Flip, keine Klangänderung, kein GPU-Training.

## 1. Ursache statt Symptom

Der UAT-Report vom 2026-11-11 (`audit/uat_results_2026-11-11.json`) führte
15/15 Restoration-Kriterien als `ERROR`. Der Traceback des einzelnen
Reproduktionslaufs zeigt keinen Qualitätsfehler, sondern:

```
RuntimeError: real-audio fixture timeout after 120.0s
```

Der Fixture-Restore läuft in einem eigenen Prozess mit einem harten
120-s-Wall-Clock-Limit. Der kalte Produktionspfad braucht auf demselben 8-s-
Eingang (`test_audio/tape/cassette_1980s_wow.wav`, 44,1 kHz → 48 kHz) jedoch
messbar deutlich länger:

- `PerformanceReport: Total: 714.1s for 30.0s audio`
- `RT Factor: 23.80× (exceeded)` — bezogen auf die interne 30-s-Analyse
  **innerhalb** des normativen 32×-Budgets
- 35 Phasen ausgeführt, 0 übersprungen; Qualität 100 %
- zusätzlich Modell-Ladezeit (u. a. MuQ, BEATs, Whisper/wav2vec2, BANQUET,
  CLAP) vor dem ersten Verarbeitungsschritt

Damit war das 120-s-Limit kein Guard gegen einen Hänger, sondern eine
Abschneidung eines legitimen Laufs. Die 15 `ERROR` waren die Folge — nicht
eine Aussage über die restaurierte Qualität.

## 2. Fix (Harness, fail-closed bleibt)

| Ort | Vorher | Nachher | Begründung |
| --- | --- | --- | --- |
| `tests/test_uat_acceptance_criteria.py` — `AURIK_R5_R12_RESTORE_TIMEOUT_S` | 120 s | **1800 s** | gemessene 714,1 s + Kaltstart-Reserve; echte Hänger fallen weiterhin auf |
| `tests/test_uat_acceptance_criteria.py` — `@pytest.mark.timeout` | 900 s | **1800 s** | der Modul-Fixture-Setup läuft im ersten Test |
| `audit/uat_report_generator.py` — Subprozess-Timeout | 1200 s | **3600 s** | der Generator muss den legitimen Fixture-Lauf abwarten können |

Die Umgebungsvariable `AURIK_R5_R12_RESTORE_TIMEOUT_S` bleibt als expliziter
Override erhalten; der Default ist jetzt realistisch. Es wurde **keine**
Qualitäts- oder Gate-Schwelle geändert.

## 3. Evidenz des Neulaufs

- `audit/uat_results_2026-11-15.json` — `generated_at=2026-11-15T09:10:05`,
  `restoration_errors=0`, `criteria_errors=0`.
- Restoration: **14/15 PASSED**, 0 ERROR, 1 echter FAIL:
  `R5` (Stereo-Kollaps-Guard) meldet für Segment 0
  `corr 0.595 → 0.759` (Grenze `before + 0.08`). Das ist ein realer
  Klang-/Stereo-Befund und wird hier nicht als Harness-Fehler verbucht.
- Studio 2026: **15/15 PASSED**.
- Release-Gates: **5/7 PASSED**, 1 SKIP (`G1`), 0 FAIL, **0 K.O.**;
  Empfehlung des Generators: `GO` (29/30 Kriterien).
- `audit/daily_real_audio_gate_status.json`: `latest.date=2026-11-15`,
  `recommendation=GO`, R5–R12 `7/8` (der R5-Befund).
- `tests/normative/test_real_audio_restoration_quality_gate.py`:
  **9 passed, 1 skipped** (vorher 8 passed, 1 skipped, 1 failed).

## 4. Tests / Abgrenzung

- `tests/unit/test_uat_report_generator.py`: 7/7 grün (ERROR-/ERROR-Zähler
  und fail-closed-Empfehlung bleiben unverändert).
- `tests/test_uat_acceptance_criteria.py` und
  `audit/uat_report_generator.py`: `py_compile` sauber.
- Der R5-Befund ist **nicht** durch diesen Fix entstanden; er war zuvor nur
  nicht sichtbar, weil das Fixture vor jeder Kriterienauswertung abbrach.
  Eine Stereo-Korrelations-Änderung ist klangwirksam und wird nicht
  eigenmächtig über einen Schwellenwert „grün“ gemacht
  (§V7 (copilot-instructions.md)/§4).

## 5. Verbleibende, ehrliche Grenzen

- Der UAT-Report ist **GO mit 29/30**, nicht „alle Kriterien grün“: R5 bleibt
  als echter Befund offen (Stereo-Korrelation in Segment 0).
- Der kalte UAT-Lauf dauert dadurch ~25 min; er ist ein bewusster
  Evidenz-Lauf, kein CI-Pre-Commit. Die 32×-RT-Buchführung bleibt Aufgabe des
  PerformanceGuard (23,8× gemessen, innerhalb des Budgets).
- Keine Hör-Instanz, kein Sign-off, kein Flag-Flip.
