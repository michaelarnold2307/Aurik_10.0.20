# D-K3-106 — §V8-Leck: die Budget-Lernhistorie entschied über den Song-Wechsel mit

**Datum:** 2026-11-16 · **Klasse:** K3 · **Status:** geschlossen ·
**Klangwirkung:** keine neue (stellt die normative Song-Isolation her) ·
**Flag-Flip:** keiner · **GPU-Training:** keines

## 1. Befund (gemessen, deterministisch, CPU-only)

`UnifiedRestorerV3` führt im Singleton `_phase_budget_gain_history`: je Material
und Phasen-Präfix ein laufender Mittelwert des akzeptierten Netto-Gewinns
(`_record_phase_budget_gain`). Zwei echte Budget-Entscheidungen lesen ihn:

- `_budget_pressure_skip_reason` → `reason="historically_weak_delta"` (Phase wird
  unter Budgetdruck übersprungen),
- `_pmgg_retry_budget_hint` → `retry_budget_scale=0,15`, `max_retries_cap=1`
  (Retries gedrosselt).

Die Historie wurde **nie** am Song-Übergang geleert. Damit entschied ein Vortitel
desselben Materials über die Budget-Skips und Retry-Drosselungen des Folgesongs
mit — ein §V8/§G1-Verstoß („alle Stateful-Module pro Song zurücksetzen").

Messung `2026-11-16_d_k3_106_song_budget_isolation.json` (echte Lernfunktion,
keine handgeschriebene Historie; Phase `phase_39_air_band_enhancement`,
Material `vinyl`):

| Zustand | Historie | `skip_reason` | `retry_hint` |
| --- | --- | --- | --- |
| Vortitel hat 2× 0,001 Netto-Gewinn akzeptiert | 2 Samples, mean 0,001 | `historically_weak_delta` | `historically_weak_delta`, scale 0,15 / cap 1 |
| **Folgesong ohne Reset** | unverändert | `historically_weak_delta` | `historically_weak_delta`, scale 0,15 / cap 1 |
| **Folgesong nach Reset** | `{}` | `null` | `null` |
| Innerhalb desselben Songs (2 neue Samples) | 2 Samples, mean 0,001 | `historically_weak_delta` | `historically_weak_delta`, scale 0,15 / cap 1 |

Die dritte Zeile ist der Fix, die vierte zeigt die bewahrte Wirkung: das Lernen
innerhalb eines Songs (auch über Chunks, weil der Chunked-Pfad denselben
Singleton nutzt) bleibt unangetastet — nur der Song-Übergang ist isoliert.

## 2. Fix (Ursache, nicht Symptom)

- `backend/core/unified_restorer_v3.py`: `reset_song_budget_state()` leert
  `_phase_budget_gain_history` am Singleton (fail-safe, §V6: ein Reset-Fehler
  bricht den Lauf nie ab).
- `denker/aurik_denker.py`: der bestehende §V8-Reset-Katalog (9. Eintrag) ruft
  die Funktion am Anfang von `AurikDenker.restauriere` — also **vor** jeder
  Verarbeitung des Songs und für alle Produktionspfade (CLI, GUI, Batch), die
  über `bridge → AurikDenker` laufen (`restorer.restore()` wird außerhalb des
  Denkers nur von Debug-Werkzeugen aufgerufen).

**Verworfen wurden zwei Behauptungen des Vorlaufs, weil nicht belegt:**

1. `_gdd_budget.phase_history` akkumuliere song-übergreifend — der
   `GddBudgetManager` wird in `_ausfuehren_pipeline` **je Lauf** neu erzeugt
   (`unified_restorer_v3.py:37586`), die Historie ist damit nicht song-übergreifend.
2. Das Zahlenpaar „−0,132/−0,014" für einen 2-Titel-Kontext — in keinem
   Beleg-Artefakt vorhanden. Es wurde ersatzlos aus dem Docstring entfernt
   (keine erfundenen Zahlen).

## 3. Tests

`tests/unit/test_wall_budget_gain_prioritization.py` (12 passed):

- `test_reset_song_budget_state_loescht_songuebergreifende_skip_historie` —
  Wirkung am Skip-Verbraucher (vorher `historically_weak_delta`, nachher `None`),
- `test_reset_song_budget_state_loescht_songuebergreifende_retry_drosselung` —
  Wirkung am Retry-Verbraucher,
- `test_budget_lernen_bleibt_innerhalb_eines_songs_aktiv` — Lernen bleibt aktiv,
  nur die Song-Grenze leert,
- `test_reset_song_budget_state_ohne_singleton_ist_failsafe` — §V6-Fail-safe.

`tests/unit/test_denker_sota_contracts.py::test_denker_reset_kette_deckt_verwaiste_stateful_module_ab`
pinnt zusätzlich den Katalog-Eintrag `reset_song_budget_state`.

## 4. Ehrliche Grenzen (offen, nicht Teil dieser Schließung)

- **Die Jets-vinyl-Kontextdifferenz ist mit diesem Fix NICHT erklärt.** Belegt
  ist nur der Kanal und seine Entscheidungswirkung. Ob er die gemessene
  Verschlechterung (3-Titel-Lauf −0,203/−0,551 vs. 1-Titel-Kontrolle
  −0,060/+0,101, `2026-11-16_kontext_beweis_kanal_entlastung.json`) trägt,
  bleibt offen.
- **Die vorhandenen Diagnose-Dumps sind kein A/B:** `dump_solo.json` und
  `dump_kontext.json` (2026-11-16, `study_stimuli/kontext_diagnose/`) zeigen
  unterschiedliche Eingänge (`rms_original_db` −19,25 vs. −14,47,
  `peak_original` 0,7685 vs. 0,995). Ein Kontext-Urteil daraus wäre ein
  Mess-Artefakt. Nötig ist eine Diagnose mit **identischem** Eingang (solo vs.
  nach Vorgänger im selben Prozess) plus Ganz-Titel-Metriken — das ist der
  nächste §V7-Schritt und bleibt außerhalb dieser Register-Zeile.
- **Keine Frame-Flip-Diskussion:** kein Flag, keine Schwelle, kein
  Klang-Design geändert; die Änderung stellt die normative Song-Isolation her.

## 5. Belege

- Messung: `docs/reports/current/2026-11-16_d_k3_106_song_budget_isolation.json`
- Code: `backend/core/unified_restorer_v3.py`, `denker/aurik_denker.py`
- Tests: `tests/unit/test_wall_budget_gain_prioritization.py`,
  `tests/unit/test_denker_sota_contracts.py`
- Kontext (Anlass, nicht Beweis): `docs/reports/current/2026-11-16_kontext_beweis_kanal_entlastung.json`
