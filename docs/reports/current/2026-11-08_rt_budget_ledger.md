# RT-Budget-Ledger + Prioritäts-Drift der CRITICAL-Phasen (D-K3-75, D-K3-76)

**Datum:** 2026-11-08
**Klasse:** K3 (Budget-Wahrheit / Scheduling-Norm)
**Hebel:** E aus `docs/SOTA_HANDOVER.md` §2b („32×-Budget allozierbar machen")
**Flag-Flips:** keine. §PERF-R21 ist **reine Buchführung**, §PERF-R21b stellt eine
bestehende Norm wieder her (Spec sagte 9, Code lieferte 5).

## 1. Befund A — das Budget war nicht allozierbar (D-K3-75)

`PerformanceGuard` kannte das 32×-Budget nur als **eine Zahl**:

- `get_phase_budget_seconds()` verteilte `Restbudget / verbleibende Phasen` —
  flach, ohne Hör-Impact-Bezug — und hatte **keinen Aufrufer** (gemessen:
  einziger Treffer außerhalb der Definition ist die `__main__`-Demo).
- `get_budget_truth_report()` (P0-3) weist Wand-/Processing-/Analytics-Zeit aus,
  aber keine Verteilung auf Phasen.
- Ergebnis: 32,19× RT ist belegt (Report
  `2026-10-28_f_reihe_laufzeit_optimierung.md` §9), **welche Phase es
  verbraucht hat, war nirgends ausgewiesen**.

Das ist exakt die Fehlerklasse „Prüfung existiert, sieht aber nichts"
(D-K3-9d/D-K3-34/D-K3-59).

## 2. Umsetzung — `get_budget_allocation()` (§PERF-R21)

    total_budget_s   = target_rt_factor × audio_duration      (Budget-Basis des Guards)
    je Phase:
      budget_s       = total_budget_s × (priority / Σ priority)   ← Soll
      measured_s     = Σ duration_seconds aus phase_performances  ← Ist
      overrun_s / headroom_s / utilisation / status
    in Summe:
      reserve_s, overrun_total_s, budget_exceeded, rt_measured, top_consumers

- **Soll** kommt aus der normativen `PHASE_PRIORITIES`-Tabelle, normalisiert über
  den Allokations-Scope (`scope = planned` mit `planned_phase_ids`, sonst
  `measured` = Nachschau über die gelaufenen Phasen).
- **Ist** kommt aus den gemessenen `phase_performances`; mehrfach gelaufene
  Phasen werden summiert (deckungsgleich mit `current_rt_factor`, das kumulativ
  rechnet).
- **Status** je Zeile: `im_budget` / `knapp` (≥ 85 %) / `ueberzogen` /
  `nicht_gelaufen` / `ohne_budget`.
- **Determinismus (§G5):** `top_consumers` sortiert nach `(-measured_s, phase_id)`;
  zwei Aufrufe auf identischem Zustand liefern identische Dicts (Test).
- **Zero-safe:** vor `start_monitoring()` sind alle Summen 0 und `phases` leer.
- **Verdrahtung:** UV3 legt das Ledger als `rt_budget_ledger` in die
  Restaurierungs-Metadaten (neben `performance_guard_report`). Ein Quelltext-Pin
  schließt aus, dass `get_budget_allocation` jemals in `should_skip_phase` oder
  `check_early_exit` auftaucht — **kein stiller Verhaltens-Flip** (§V7).
- `get_phase_budget_seconds()` delegiert jetzt an die Allokation (EINE
  Budget-Wahrheit statt einer flachen Division daneben); für unbekannte Phasen
  bleibt das historische Verhalten als Fallback erhalten.

**Ehrliche Grenze (keine Fantasiegewichte, D-K3-69-Lehre):** Die Priorität ist
eine **Wichtigkeits-**, keine **Kosten**-Ordnung. Der Ledger sagt deshalb nicht
„so viel Zeit hat Phase X verdient", sondern „so hat sich das Budget unter den
Phasen verteilt, für die er gilt" — und `scope` benennt, worüber er spricht.
Die erste echte Verteilung zeigt der nächste Referenzlauf (GPU derzeit durch
Kette 3 belegt).

## 3. Befund B — zwei CRITICAL-Phasen liefen als MEDIUM (D-K3-76)

Beim Testen des Ledgers fiel die Priorität von `phase_12_wow_flutter_fix` auf
den Default 5. Nachgemessen über **alle 69** Phasen-Module:

| Phase (Modul) | Kurzname nach `_normalize_phase_id` | Tabelleneintrag | Prio vorher | Prio jetzt |
| --- | --- | --- | --- | --- |
| `phase_12_wow_flutter_fix` | `wow_flutter_fix` | `wow_flutter` (9) | **5 (MEDIUM)** | 9 (CRITICAL) |
| `phase_09_crackle_removal` | `crackle_removal` | `decracklé` (9) | **5 (MEDIUM)** | 9 (CRITICAL) |

Nur **7 von 69** Modulen trafen vorher einen Tabelleneintrag (nach dem Fix: 9).

**Wirkung des Defekts:** `_is_phase_critical()` lieferte `False`, und
`should_skip_phase()` rechnete mit `skip_threshold = 0.93 × 32×` statt „never
skip". Damit durfte der Guard **die teuerste Reparaturphase des Systems**
(Wow/Flutter, im Referenzlauf 250–440 s bzw. 57 Heartbeats, D-K3-69) unter
Budgetdruck verwerfen — obwohl §2.29/§9.5 und der Tabellenkopf selbst sie als
CRITICAL führen („Phasen mit Priority ≥ 9 werden NIEMALS übersprungen").

**Fix:** `PHASE_PRIORITY_ALIASES` (`wow_flutter_fix → wow_flutter`,
`crackle_removal → decracklé`) + `_priority_key()`, benutzt in **allen drei**
Prioritäts-Pfaden (`should_skip_phase`, `_is_phase_critical`,
`_calculate_quality_degradation`) und im Ledger. Die alten Tabellenschlüssel
bleiben (keine Funktion ohne Ersatz). Ein Quelltext-Test pinnt, dass keiner der
drei Pfade wieder an der Alias-Auflösung vorbeiliest (Muster C).

**Ausdrücklich NICHT geändert:** Die übrigen **60** Phasen-Module treffen
weiterhin keinen Tabelleneintrag und laufen auf Default 5. Eine Neubewertung von
60 Phasen ist eine breite Scheduling-/Klang-Stufe (RT-Profil + Hör-Impact je
Phase nötig) und damit kein autonomer Schritt — sie bleibt als Beobachtung
dokumentiert (§V7, „kein stiller Flip").

## 4. Verifikation

    python3 -m pytest tests/unit/test_performance_guard_budget_ledger.py \
                      tests/unit/test_performance_guard_spec_compliance.py -q
    → 35 passed in 0,58 s

    python3 -m ruff check backend/core/performance_guard.py \
                           tests/unit/test_performance_guard_budget_ledger.py
    → All checks passed (inkl. F821/F601/B009/I001-Gate)

Abgedeckt: Ledger-Arithmetik (Summen, Anteile, Status, Reserve), Scope
`planned`/`measured`, Median-Priorität für unbekannte Phasen, Zero-Safe,
Determinismus, Delegation von `get_phase_budget_seconds`, Alias-Auflösung,
Skip-Verweigerung bei 95 % Budgetverbrauch, Phantom-Freiheit der Aliasse,
Quelltext-Pin gegen Rückfall, UV3-Verdrahtung.

## 5. Regelkonformität

- §V7 (copilot-instructions.md): kein stiller Verhaltens-Flip — der Ledger ist
  buchhalterisch (Quelltext-Pin), der Prioritäts-Fix stellt die bereits
  normative Aussage wieder her und ist als D-K3-76 registriert.
- §G5 (copilot-instructions.md): Determinismus-Test, keine Zeitabhängigkeit in
  der Sortierung.
- Keine neue Code-Datei außer der Testdatei (Write-Gate:
  `scripts/repo_search.py --before-create` → „keine kanonische Alternative",
  FILE_REGISTRY-Eintrag ergänzt).
- Laufende GPU-Ketten nicht gestört; kein Training gestartet; kein Flag-Flip.
