# UAT-Report: ERROR-Kriterien verschwanden aus der Bilanz (D-K3-104)

**Datum:** 2026-11-15
**Klasse:** K3 — Prüfung existiert, sieht aber nichts
**Status:** geschlossen (Code + Tests + messbarer Beleg)
**Klangwirkung:** keine. Keine Audio-, Phasen- oder Flag-Änderung; kein Hör-Sign-off nötig.

## Befund

Der UAT-Beleg `audit/uat_results_2026-11-11.json` enthält 15 Restoration-Kriterien,
**alle mit Status `ERROR`** (die Scorecard zeigt in allen 15 R-Zeilen `ERROR`). Die
Summary desselben Belegs behauptete gleichzeitig:

| Feld | Wert im Beleg |
|---|---:|
| `restoration_passed` | 0 |
| `restoration_failed` | 0 |
| `restoration_skipped` | 0 |
| `criteria_passed` | 15 (nur Studio-2026) |
| `criteria_failed` | 0 |
| `criteria_skipped` | 0 |
| `criteria_executed` | 15 |
| `recommendation` | **CONDITIONAL** |
| `rationale` | „No executed criterion failed (15/15 passed), 0 criteria pending …" |

Damit meldete der Report **15/15 bestanden**, obwohl die gesamte
Restoration-Hälfte nicht auswertbar war. Ursache:

1. `compute_summary` zählte nur `PASSED`/`FAILED`/`SKIPPED`; `ERROR` fiel aus
   jedem Zähler.
2. Die Empfehlung prüfte nur `total_failed == 0` und `total_executed >= 8`;
   `total_executed` enthielt ausschließlich die 15 Studio-Kriterien.
3. Die Scorecard renderte `ERROR` im `else`-Zweig als `⊘ SKIP` — sichtbar,
   aber als nicht blockierend.

Das ist dieselbe Klasse wie D-K3-34 („die Prüfung existiert, sieht aber
nichts"): Ein Setup-Fehler der Restoration-Suite wurde zu einem stillen
Nicht-Ergebnis, und die Empfehlung blieb grün.

## Fix

`audit/uat_report_generator.py`:

- `UATSummary` führt `restoration_errors`, `studio_2026_errors`,
  `gates_errors` und `criteria_errors` als eigene Felder.
- `compute_summary` zählt `ERROR` separat. `total_accounted` ist
  `passed + failed + skipped + errors`.
- **Fail-closed:** `criteria_errors > 0` ⇒ `NO-GO` mit Begründung
  „nicht auswertbar". `total_accounted < 30` ⇒ `NO-GO` („ohne Ergebnis").
  `CONDITIONAL` bleibt nur für dokumentierte `SKIP`s zulässig.
- Eine Statusquelle `_status_symbol()` versorgt Scorecard und Final-Report;
  `ERROR` wird als `⚠ ERROR` gerendert, nie als `⊘ SKIP`.
- Statistik und Decision Matrix führen „Unexecutable Criteria (ERROR)".

## Messbarer Beleg

**Vorher** (`audit/uat_results_2026-11-11.json`, unverändert):

| Feld | Wert |
|---|---:|
| `restoration_errors` (aus den 15 Ergebnisobjekten gezählt) | 15 |
| `criteria_errors` in der Summary | **0** (Feld existierte nicht) |
| `criteria_passed` | 15 |
| `criteria_failed` | 0 |
| `recommendation` | **CONDITIONAL** |
| Scorecard R1 | `⊘ SKIP` |

**Nachher** (dieselben 30 Kriterien und 6 Gates durch den neuen
`compute_summary`):

```json
{
  "restoration_passed": 0,
  "restoration_failed": 0,
  "restoration_skipped": 0,
  "restoration_errors": 15,
  "criteria_passed": 15,
  "criteria_failed": 0,
  "criteria_skipped": 0,
  "criteria_errors": 15,
  "criteria_executed": 15,
  "recommendation": "NO-GO",
  "rationale": "15 Kriterien nicht auswertbar (ERROR) — UAT unvollständig; keine Qualitätsaussage möglich (15/30 bestanden)"
}
```

Scorecard R1: `⚠ ERROR` statt `⊘ SKIP`.

Reproduktion (ohne Audio, ohne GPU):

```bash
.venv_aurik/bin/python -B -m pytest tests/unit/test_uat_report_generator.py -q
```

## Tests

`tests/unit/test_uat_report_generator.py` — **7 passed**:

- `test_error_criteria_are_counted_and_block_conditional`
- `test_skipped_criteria_remain_conditional`
- `test_missing_criteria_block_conditional`
- `test_gate_error_blocks_conditional`
- `test_scorecard_and_final_report_render_error_not_skip`
- `test_scorecard_cell_combines_evidence_and_notes`
- `test_scorecard_groups_separate_tiers` (bestehend)

`tests/normative/test_real_audio_restoration_quality_gate.py` — **8 passed,
1 skipped, 1 failed**: Der einzige Rote ist weiterhin
`test_daily_gate_stored_recently` (UAT-Evidenz 4 Tage alt, D-K3-13). Der Fix
erzeugt keine frische Evidenz und täuscht keine vor.

## Abgrenzung / offen

- Die 15 Restoration-Kriterien sind **weiterhin ERROR**; sie sind jetzt
  sichtbar und blockieren die Empfehlung. Ein frischer UAT-Lauf bleibt nötig
  (D-K3-13) und gehört in ein trainingsfreies GPU-Fenster.
- Der historische Beleg `audit/uat_results_2026-11-11.json` wird **nicht
  rückwirkend umgeschrieben**; er bleibt Zeuge des alten Generator-Verhaltens.
- Keine Klangwirkung, kein Flag-Flip, kein §4-Hör-Sign-off, kein GPU-Training.
