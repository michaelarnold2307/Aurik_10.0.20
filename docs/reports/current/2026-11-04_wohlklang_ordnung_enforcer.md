# Wohlklang-Ordnungs-Enforcer — Analyse, Design, Verifikation (2026-11-04)

**Auftrag:** Die lexikografische Wohlklang-Ordnung (Hörordnung §2/§5) wird
DURCHGESETZT statt nur gewarnt — die gemessenen 20 Ordnungs-Verletzungen je
Lauf (GOAL_SCORECARD/B3-Läufe 2026-11-03/04, Log-Zitat im Auftrag) werden im
erzwungenen Zustand auf 0 gebracht („Restverletzungen: 0"), ohne andere Ziele
unter ihre Basis zu drücken (Never-worsen).

**Branch:** `kun/wohlklang-ordnung-enforcer-g6hjqe`
**Wichtige Umgebungsgrenze:** Diese Worker-Env hat KEIN Shell-/Exec-Werkzeug.
Nichts wurde gemessen oder ausgeführt. Alle Befehle in Abschnitt 5/6 sind
dokumentiert und AUSSTEHEND — die Ausführung macht der Manager. Es werden
keine Grün-Stände behauptet.

---

## 1. Analyse (file:line, statisch verifiziert)

### 1.1 Normative Ordnung (Hörordnung)

- `.github/instructions/hoerordnung.instructions.md:46–53` (§2): vier Ebenen,
  jede dominiert die darunterliegende — 1. Hör-Invarianten, 2. Audibility,
  3. Wohlklang-Ordnung, 4. Einladungs-Gate; „Eine Entscheidung auf Ebene n
  darf niemals eine Invariante der Ebene < n verletzen."
- `.github/instructions/hoerordnung.instructions.md:133–160` (§5, Ebene 3):
  strikte lexikografische Reihenfolge **1. Natürlichkeit > 2. Wärme >
  3. Klarheit/Durchhörbarkeit > 4. Brillanz**; ausdrücklich verboten:
  „Ein Brillanz-/Transparenz-Boost, der Wärme oder Natürlichkeit senkt"
  (:155–157). Seit v10.1.0 maschinell auditiert via `WohlklangOrdnungGate`
  (:158–160).
- Konfliktregel §7 (:186–200): bei Ebene-1-Verletzung hat die Hör-Instanz
  keinen Vorrang („nie gegen Ebene 1").
- Tier-Quelle (recycelt, nicht neu erfunden): `backend/core/goal_priority_
  protocol.py:116–135` (`HEARING_TIER_MAP`) + `hearing_tier()` :137–143:
  Stufe 1 = natuerlichkeit, authentizitaet, timbre_authentizitaet,
  micro_dynamics, emotionalitaet, formant_fidelity, vocal_quality; Stufe 2 =
  waerme, bass_kraft; Stufe 3 = tonal_center, artikulation, transparenz,
  separation_fidelity; Stufe 4 = brillanz, spatial_depth, transient_energie,
  groove, raumtiefe.

### 1.2 Wo die Verletzungen entstehen

Die gemessenen Paare (niederrangige wie waerme/brillanz/groove/tonal_center/
separation_fidelity/transparenz verbessert auf Kosten von emotionalitaet/
authentizitaet/natuerlichkeit/artikulation) entstehen auf drei Wegen:

**(a) PMGG-Annahme-Pfade mit Toleranz-Bändern (Hauptursache).**
`backend/core/per_phase_musical_goals_gate.py`:

- :5810–5856 `_max_regression_priority_aware` bewertet Regressionen auf der
  **PRIORITY**-Achse (`PRIORITY_MAP`, `goal_priority_protocol.py:33–50`) mit
  Basis-Schwelle 0.020 × Faktor — nicht auf der Hörordnungs-Tier-Achse.
- :4926 `passed_p4p5_tolerated`: alle Regressionen innerhalb ihrer
  Prioritäts-Bänder ⇒ Phase wird ohne Retry angenommen. Eine kleine
  Stufe-1-Senkung (z. B. natuerlichkeit −0.005) PLUS ein Stufe-3/4-Gewinn
  (z. B. brillanz +0.03) ist damit ein akzeptierter Ordnungsverstoß.
- :4936–4954 `passed_team_balanced` („Teamwork schlägt Dominanz"): erlaubt
  Regressionen, solange `_worst_prio >= 3` (PRIORITY!) + Team-Netto positiv +
  P1/P2 über **absolutem** Boden (`_compute_team_net_delta` :5858–5937).
  **Achsen-Mismatch (Root Cause):** `goal_priority_protocol.py:75–78`
  dokumentiert, dass PRIORITY_MAP und HEARING_TIER_MAP „BEWUSST verschiedene
  Achsen" sind. `emotionalitaet` und `micro_dynamics` sind PRIORITY 3, aber
  Hörstufe 1 — ihr Verfall wird vom Teamwork-Gate also ausdrücklich toleriert,
  während niederrangige Ziele gewinnen. Genau das sind die gemessenen Paare.
- Best-Effort-Pfade (:5005, :5139, :5262, :5400) wenden trotz Rest-
  Regressionen den Versuch mit der geringsten Regression an.

**(b) Epsilon-Lücke im FC-Pfad.** `backend/core/feedback_chain.py:1122–1123`
prüft das Audit mit `threshold=REGRESSION_EPSILON` (0.012,
`goal_priority_protocol.py:105–107`); Senkungen ≤ 0.012 am höherrangigen Ziel
zählen dort nicht. Das Pipeline-Audit (`wohlklang_ordnung_gate.py`, `_EPSILON
= 1e-9`) meldet sie korrekt — die Mikro-Senkungen passieren die FC-Guards und
akkumulieren.

**(c) Kumulation.** Selbst ordnungskonforme Einzelschritte können in der
Komposition verletzen (Senkung ohne niederrangigen Gewinn in Schritt A +
Gewinn ohne Senkung in Schritt B ⇒ Paar im Endergebnis) — dokumentiert im
Kommentar des §P1-1-Blocks (`unified_restorer_v3.py:17434–17435`).

### 1.3 Warum bestehende Rollbacks die Verletzungen NICHT abfangen

| Mechanismus | Stelle | Warum nicht |
|---|---|---|
| ExcellenceOptimizer-Rollback | `unified_restorer_v3.py:16599` (Log), Rücknahme :16602 | Nur der Excellence-Schritt; Schwelle 0.02 (Senkungen ≤ 0.02 passieren); nicht tier-bewusst — ein Verstoß-Paar mit kleinen Senkungen ist kein „Goal-Regression"-Fall |
| EmotionalArc-Rollback | `unified_restorer_v3.py:17942` (Rücknahme auf `_pre_excellence_audio`) | Prüft arousal/valence-Korrelation, keine Goal-Tiers |
| Witness-Veto | `unified_restorer_v3.py:15704–15722` | HNR/Pitch/Flat-Top-Schwellen, keine Ordnungs-Paare |
| FC-Callback-Verwerfung | `unified_restorer_v3.py:15584–15615` | Wirft einzelne FC-Kandidaten korrekt bei VIOLATION weg — deckt aber nur FC-Iterationen ab; PMGG-Phasen laufen nicht durch diesen Callback |
| NaturalnessOptimizer | `backend/core/naturalness_optimizer.py:337–370` | Setzt die Ordnung für seinen Schritt bereits durch (Rücksprung auf Eingangs-Signal) — aber nur dort |

### 1.4 Zentraler Enforcement-Punkt

`unified_restorer_v3.py` §P1-1 (vor dem Patch :17486–17544, Befund-Log
:17524 „§P1-1 Pipeline-Wohlklang-Ordnungs-Violation: … N Verstoß/Verstöße"):
Das **Tail-Akzeptanz-Gate**, an dem das Pipeline-Ergebnis (Kandidat) gegen
den Input gemessen und angenommen wird. Vor dem Patch: Auswertung + Warnung +
Metadaten, **keine** Durchsetzung („nur Protokoll"). Genau hier werden die
20 Verletzungen je Lauf gezählt — und genau hier wird jetzt gekappt/
zurückgeworfen.

## 2. Enforcement-Design (zentral, §V7 (copilot-instructions.md))

**Neuer zentraler Kern — `backend/core/wohlklang_ordnung_gate.py`** (bestehende
Datei erweitert, keine neue Moduldatei):

- `WohlklangOrdnungEnforcer.enforce(goal_deltas, *, threshold=None,
  blend_probe=None)` (:332–457) — die Kandidaten-Entscheidung läuft NUR über
  das Gate + die zentrale Stärke-Logik:
  - **ACCEPT** (kein Verstoß): blend_factor 1.0 — der Kandidat geht
    bit-identisch unverändert durch (`blend_with_reference` gibt dasselbe
    Objekt zurück; §G5 (copilot-instructions.md)).
  - **CAP** (Verstoß + Messprobe): feste, deterministische Blend-Stufen
    `DEFAULT_BLEND_ALPHAS = (0.5, 0.25, 0.125, 0.0625)` (:68) — absteigend,
    die erste konforme Stufe gewinnt. Eine Stufe ist nur konform, wenn
    (1) das Gate auf ihren Deltas PASS/NOT_APPLICABLE meldet UND (2) kein
    Ziel unter seiner Basis liegt (Never-worsen, „höhere Ebenen verlieren nie
    für niedrigere"). Jede Stufe wird per `blend_probe` gemessen und gegen
    das Gate verifiziert.
  - **ROLLBACK** (keine konforme Stufe oder keine Probe): blend_factor 0.0 —
    der verletzende Anteil wird vollständig zurückgeworfen (Muster:
    ExcellenceOptimizer-Rollback), `enforced_deltas` = 0.
  - **Ebene-1-Schutz:** verletzte Stufe-1-Ziele werden im Ergebnis als
    `tier1_violated` markiert und im Event begründet („nie gegen Ebene 1",
    Hörordnung §2 (:48–49)/§7 (:190–192)); eine CAP-Stufe mit Restverlust an
    Stufe 1 scheidet über Prädikat (2) aus.
- `blend_with_reference(candidate, reference, blend_factor)` (:107–120):
  α≥1.0 → Kandidat unverändert (dasselbe Objekt); α≤0.0 → `reference.copy()`
  (Rollback-Muster); dazwischen lineare Überblendung. Duck-typed, das Modul
  bleibt numpy-frei.
- `compute_goal_deltas(before, after)` (:87–104): eine Delta-Definition
  (after − before, sortierte Keys, Finite-Filter) für Audit und Proben.
- `OrdnungEnforcementResult` (:286–329): `action`, `blend_factor`,
  `tier1_violated`, Befund (`violated_goals`/`violations`), `enforced_deltas`,
  `residual_violations`, deutsche `events`, `to_dict()`.

**Verdrahtung (Pipeline-Annahme, `backend/core/unified_restorer_v3.py`):**
Der §P1-1-Block ist bewusst VOR die abgeleiteten Werten gewandert
(:17432–17579; Zeiger-Kommentar am alten Ort :17635–17638), damit
`_measured_goals`, `_musical_goals_passed`, `_musical_excellence_score`,
`_mg_violations`, die GOAL_SCORECARD-Zeile und die End-Gate-Kaskade den
ERZWUNGENEN Zustand zeigen:

1. `compute_goal_deltas(input_scores, output_scores)` (:17469–17471) —
   identische Delta-Semantik wie vorher inline.
2. Gate-Audit (:17474) — unverändert; bei VIOLATION bleibt die bestehende
   Warnung „§P1-1 Pipeline-Wohlklang-Ordnungs-Violation: …" (:17477–17480)
   und die Forensik-Metadatum `wohlklang_ordnung_violation` (:17484–17489)
   erhalten (§G8 (copilot-instructions.md): Warnungen bleiben; §V6
   (copilot-instructions.md): keine stille Unterdrückung).
3. Zentrale Durchsetzung (:17490–17572): `WohlklangOrdnungEnforcer.enforce`
   mit `blend_probe` (deterministische `measure_all`-Messung je Blend-Stufe,
   :17501–17519), Anwendung via `blend_with_reference` (:17528–17532),
   Übernahme der erzwungenen Goal-Scores in `_musical_goal_scores`
   (:17536–17540). Shape-Mismatch → ehrliche Warnung, kein angewendetes
   Enforcement, `applied: false`, Restzählung = Befund (:17542–17557).
4. Transparenz: Enforcement-Events „Ordnung erzwungen: …" via
   `logger.warning` (:17565–17566) + Zusammenfassung
   „§P1-1 Ordnung erzwungen: action=… blend_factor=… — Restverletzungen: N"
   (:17567–17572). Metadaten `wohlklang_ordnung_enforcement` (inkl.
   `applied`, `residual_violations`) in `_phase_metadata_accumulator` und im
   `self._last_wohlklang_audit`-Dict unter `enforcement` (:17550–17563).

## 3. Konformität

- **§G5 (copilot-instructions.md) Determinismus:** feste Blend-Stufen
  (`DEFAULT_BLEND_ALPHAS`, numpy-freies Modul), kein `time.time()`, kein
  Sampling/RNG in der Entscheidung; die Messproben laufen über die
  deterministische `measure_all`-Kette. **Nicht verifiziert** (Worker ohne
  Shell): der Diff enthält in der Entscheidungslogik keinen Zeitbezug.
- **§G3 (copilot-instructions.md):** Never-worsen-Prädisposition verhindert,
  dass die Durchsetzung selbst Ziele unter die Basis drückt.
- **Hörordnung §2/§5/§7 (hoerordnung.instructions.md):** strikte Dominanz
  wird erzwungen statt protokolliert; Ebene-1-Schutz „nie gegen Ebene 1";
  Tier-Struktur ausschließlich aus `GoalPriorityProtocol` recycelt
  (`wohlklang_ordnung_gate.py` `_tier_from_protocol`) — keine neu erfundene
  Reihenfolge.
- **§V7 (copilot-instructions.md):** keine phasen-individuellen Schwellwerte;
  die Kandidaten-Entscheidung läuft NUR über Gate + zentrale Stärke-Logik
  (feste Blend-Stufen). Das `threshold`-Argument wird unverändert an das Gate
  durchgereicht — keine neue Schwelle.
- **§v10.802 (copilot-instructions.md):** keine Flag-Flips; keine
  Verhaltens-Flags geändert.
- **§V6/§G8 (copilot-instructions.md):** Befund-Warnungen bleiben, deutsche
  Enforcement-Events mit Begründung, ehrliche Restzählung bei
  Nicht-Anwendbarkeit.
- **Nicht angefasst (Konfliktgefahr, laufende Validierung):**
  `backend/core/musical_goals/transient_energy_metric.py`,
  `backend/core/musical_goals/musical_goals_metrics.py`,
  `scripts/b3_transient_energie_neverworsen.py`.

## 4. Erwartete Wirkung (ehrlich, ohne Messbehauptungen)

- **Garantiert (konstruktiv, per Unit-Tests belegt):** Der vom Enforcement
  angenommene Zustand hat Restverletzungen = 0 (CAP-Stufen sind gegen das
  Gate verifiziert, ROLLBACK hat Deltas 0) und verstößt nie gegen
  Never-worsen/Ebene-1. Der Zähler „Restverletzungen: N" in der
  §P1-1-Zusammenfassung ist die Zielmetrik 20 → 0.
- **Sichtbar bleibt:** die Befund-Warnung mit der Ursprungszahl (z. B.
  „20 Verstoß/Verstöße") — sie dokumentiert, was die Pipeline ohne
  Enforcement erzeugt hätte (§G8).
- **Trade-off (bewusst gewählt):** Reagieren die Goal-Scores nahezu linear
  auf die Blend-Stärke, bleibt die Verletzungs-Signatur (niederrangig ↑,
  höherrangig ↓) bei jedem α > 0 erhalten — dann ist α = 0 (ROLLBACK) die
  einzige ordnungskonforme Stufe und die Gewinne des Laufs (z. B. waerme/
  brillanz/groove) werden zugunsten der höherrangigen Ziele verworfen.
  Wie häufig CAP vs. ROLLBACK greift, ist eine MESSFRAGE und wird über den
  Scorecard-Vergleich (§6) durch den Manager erhoben — hier wird keine Zahl
  behauptet.
- **Restrisiko (offen, siehe §7):** Nach dem §P1-1-Annahmepunkt können
  End-Gate-Kaskade (`_run_song_level_end_gate`, :46321), §SR-CG8 und der
  EmotionalArc-Rollback das Audio noch ändern. Die Messung der
  B3-Scorecard erfolgt am selben Punkt wie die Baseline-Läufe
  (GOAL_SCORECARD/§P1-1), sodass der Vergleich vor/nach hier fair ist; für
  den Export-Endzustand gilt das als dokumentiertes Restrisiko.

## 5. Verifikations-Befehle (AUSSTEHEND — Worker-Env hat KEIN Shell)

Nichts wurde ausgeführt oder gemessen. Auszuführen durch den Manager:

```bash
# Unit-Tests (neu + Bestand des Gates):
nice -n 19 python3 -B -m pytest tests/unit/test_wohlklang_ordnung_enforcer.py \
    tests/unit/test_wohlklang_ordnung_gate.py -q

# Smoke über tests/unit (Pre-Commit-Parität, maxfail=3):
nice -n 19 python3 -B -m pytest tests/unit -q --maxfail=3

# Kritischer statischer Gate (wie aurik-ruff-critical-static-gate):
ruff check --select F821,F601,B009,I001 \
    backend/core/wohlklang_ordnung_gate.py \
    backend/core/unified_restorer_v3.py \
    tests/unit/test_wohlklang_ordnung_enforcer.py

# Write-Gate für die neue Testdatei (AGENTS.md §3; im Worker nicht möglich):
python scripts/repo_search.py --before-create tests/unit/test_wohlklang_ordnung_enforcer.py

# Task-Ledger (CI erzwingt TASK_CHANGES-Abdeckung):
python scripts/change_ledger.py snapshot
```

Statische Verifikation im Worker (ohne Shell) wurde gemacht: Signaturen der
Tests gegen die realen Implementierungen abgeglichen (`enforce(goal_deltas,
*, threshold=None, blend_probe=None)`, `blend_with_reference`,
`compute_goal_deltas`, `OrdnungEnforcementResult`-Felder), Float-Assertions
toleranzbasiert, bestehende Gate-Tests bleiben unverändert gültig
(`evaluate`-Semantik identisch — nur die interne Reinigung nutzt jetzt
`_clean_deltas`).

## 6. Never-worsen-/Scorecard-Vergleich vor/nach (B3-Muster)

Ziel der Messung: die 15 Goals + Verletzungszähler (20 → Restverletzungen 0)
auf Corpus-Fenstern. Auszuführen durch den Manager (AUSFÜHRUNG nicht Teil
dieses Dispatches). Muster analog `scripts/b3_transient_energie_neverworsen.py`
(Dokumentation dort, Zeilen 25–30: `run --label …` / `compare --alt … --neu …`;
der Harness misst je Fenster die 15-Goal-Scorecard mit
`pre_pipeline_reference` = Eingabe, Zeilen 112–122):

```bash
# 1) Lauf AUF DEM VORHERIGEN STAND (B3-Baseline-Verfahren):
nice -n 19 python3 -B scripts/b3_transient_energie_neverworsen.py run --label wo_vor
# 2) Lauf MIT Ordnungs-Enforcement (dieselben Corpus-Fenster/Seed):
nice -n 19 python3 -B scripts/b3_transient_energie_neverworsen.py run --label wo_nach
# 3) Blind-Vergleich inkl. Scorecard-Diff je Goal + Never-worsen (JND 0,05):
nice -n 19 python3 -B scripts/b3_transient_energie_neverworsen.py compare \
    --alt docs/reports/current/b3_transient_energie_alt.json \
    --neu docs/reports/current/b3_transient_energie_neu.json
```

Verletzungszähler und Enforcement-Events zusätzlich aus den Lauf-Logs
parsen (Ziel: keine „Restverletzungen: N" mit N > 0):

```bash
grep 'GOAL_SCORECARD ' <lauf>.log                                  # 15 Goals je Lauf
grep 'Pipeline-Wohlklang-Ordnungs-Violation' <lauf>.log             # Befund (bleibt sichtbar)
grep 'Ordnung erzwungen: ' <lauf>.log                              # deutsche Enforcement-Events
grep 'Restverletzungen: ' <lauf>.log                               # Ziel: 0
```

Akzeptanz des Vergleichs: (a) Restverletzungen = 0 in jedem Nach-Lauf,
(b) Scorecard-Diff je Goal ohne Verschlechterung über die Never-worsen-
Toleranz des B3-Verfahrens hinaus (JND 0,05 im Sinne der dortigen
`JND_TOLERANZ`, Zeile 53 des Harness). Auswertung und GO/NO-GO macht der
Manager.

## 7. Offene Punkte

1. **Messung CAP- vs. ROLLBACK-Anteil** auf Corpus-Fenstern (§4/§6) —
   entscheidet, ob die feste Blend-Stufen-Logik (0.5 … 0.0625) feiner
   getaktet werden muss.
2. **Restrisiko End-Gate-Kaskade/§SR-CG8/EmotionalArc:** spätere
   Audio-Änderungen nach dem §P1-1-Annahmepunkt könnten theoretisch wieder
   Verletzungen erzeugen; die Messung erfolgt am selben Punkt wie die
   Baseline, für den Export-Endzustand ist es unbelegt. Nachzieh-Option:
   dieselbe zentrale `enforce`-Instanz vor dem finalen Audio-Commit
   erneut aufrufen (eine zentrale Funktion, kein neuer Schwellwert).
3. **PMGG-Toleranzpfade (§1.2):** die Entstehungsursache bleibt aktiv —
   ob `passed_team_balanced`/`passed_p4p5_tolerated` zusätzlich über den
   zentralen Enforcer laufen sollen, ist eine Verhaltensänderung an der
   Phasen-Annahme und sollte nach der Messung entschieden werden (keine
   Flag-Flips, §v10.802).
4. **Write-Gate:** `scripts/repo_search.py --before-create` für
   `tests/unit/test_wohlklang_ordnung_enforcer.py` ist noch vom Manager
   nachzuholen (statischer Namensabgleich im Worker: keine Kollision;
   `tests/unit/test_wohlklang_ordnung_gate.py` bleibt die reine
   Audit-Gate-Testdatei).
5. **Shape-Mismatch-Pfad** (Chunked vs. Voll-Song) ist ehrlich ausgestaltet,
   aber nicht durch einen Test abgedeckt (Integrationsthema).

## 8. Geänderte Dateien + Conventional-Commit-Vorschlag

Geändert/neu (dieser Dispatch):

- `backend/core/wohlklang_ordnung_gate.py` — zentraler Enforcer
  (`WohlklangOrdnungEnforcer`, `OrdnungEnforcementResult`,
  `blend_with_reference`, `compute_goal_deltas`, `DEFAULT_BLEND_ALPHAS`).
- `backend/core/unified_restorer_v3.py` — §P1-1-Block versetzt +
  Enforcement-Verdrahtung (Warnungen/Events/Metadaten).
- `tests/unit/test_wohlklang_ordnung_enforcer.py` — NEU (27 Tests,
  synthetische Score-Vektoren).
- `.github/FILE_REGISTRY.md` — Eintrag für die neue Testdatei.
- `docs/reports/current/2026-11-04_wohlklang_ordnung_enforcer.md` — dieser
  Report.

Commit-Vorschlag (auszuführen durch den Manager):

```bash
git add backend/core/wohlklang_ordnung_gate.py \
        backend/core/unified_restorer_v3.py \
        tests/unit/test_wohlklang_ordnung_enforcer.py \
        .github/FILE_REGISTRY.md \
        docs/reports/current/2026-11-04_wohlklang_ordnung_enforcer.md
git commit -m "fix(wohlklang): enforce lexicographic hearing-order at pipeline acceptance" \
           -m "WohlklangOrdnungEnforcer kappt/zurueckwirft ordnungswidrige" \
           -m "Ergebnisanteile zentral (Hörordnung §5/§2) statt sie nur zu" \
           -m "protokollieren; Ebene-1-Schutz, Never-worsen, deterministische" \
           -m "Blend-Stufen (§G5), keine Flag-Flips (§v10.802)."
```
