# UAT-Frische D-K3-13 — Neulauf 2026-11-23

**Datum:** 2026-11-23 · **Kontext:** `test_daily_gate_stored_recently` (Evidenz-Alter > 48 h) ·
**Gerät:** CPU (nice 19, GPU frei) · **Scheduler-Regel:** eingehalten (keine GPU-Kette lief).

## Ablauf

1. `audit/uat_report_generator.py --output-dir docs --json-output audit/uat_results_2026-11-23.json`
   (echte Produktionspfad-Analyse, kalt, Timeouts D-K3-13-gehärtet 1800/3600 s).
2. `audit/daily_real_audio_gate.py` → `audit/daily_real_audio_gate_status.json`
   (`latest.date = 2026-11-23`).
3. `tests/normative/test_real_audio_restoration_quality_gate.py::TestDailyGateStatus`
   → **2/2 grün** (Frische + Konsistenz mit der Quellen-UAT).

## Ergebnis (29/30)

| Block | Ergebnis |
| --- | --- |
| Restoration-Kriterien (15) | **14 PASSED, 1 FAILED** |
| Studio-2026-Kriterien (15) | **15 PASSED** |
| Release-Gates (6 gefahrene) | **6 PASSED** |
| Empfehlung | **NO-GO** (MUST-FAIL R5, D-K3-104-Konvention) |

**Der eine FAIL ist R5 „Vocals in Stereo präserviert“** (MUST):
Stereo-Korrelation Segment 0: **0,595 → 0,758** — derselbe reproduzierbare
Befund wie im UAT 2026-11-15 (0,595 → 0,759). Kein Harness-Fehler, keine
neue Regression; der 2026-11-15-Lauf trug denselben R5-Befund bei
Empfehlung `GO` — die heute striktere fail-closed-Empfehlung folgt der
D-K3-104-Konvention (MUST-FAIL ⇒ NO-GO).

**Einordnung:** R5 ist der konkrete, maschinell reproduzierbare
Stereo-Qualitätsbefund der Parity-Zeile Stereo-Imaging (dort verlinkt) —
die Stereo-Kette (3/3 `balance_besser`, n=10: 8/10 besser) adressiert genau
diese Klasse; R5 bleibt bis zum §4-Hör-Vergleich offen. Kein Flag-Flip,
keine Klangwirkung, kein GPU-Training.
