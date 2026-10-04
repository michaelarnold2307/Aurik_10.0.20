# Hörordnung — offene Kanonisierung (Patch bereit, Ausführung wartet auf Read-Werkzeug)

**Datum:** 2026-10-03 · **Status:** inhaltlich final, maschinell blockiert
**Grund:** Die normative Spitze (`.github/instructions/hoerordnung.instructions.md`,
Hash-überwacht via `scripts/spec_drift_check.py`) nennt die seit 2026-10-03
existierenden **erzwingbaren Instanzen** noch nicht. Das ist genau die
Ausgangslücke der Wohlklang-Roadmap Punkt 1: „die Hörordnung schreibt es vor,
es fehlt nur als eine erzwingbare Instanz."

**Warum noch nicht angewandt:** Der Read-before-edit-Guard der Bearbeitungs-
werkzeuge verlangt einen `read`-Aufruf dieser Datei; das Read-Werkzeug ließ
sich in der Ausführungssitzung technisch nicht auslösen (Emissionsfehler).
Eine Umgehung über Shell ist ausdrücklich untersagt und wurde bewusst nicht
versucht. Die Datei ist daher **unverändert**.

## Anzuwendende Änderungen (exakt)

### A) Nach §3 „Verboten"-Abschnitt (Zeile ~70) einfügen

```markdown
**Umsetzung (kanonisch):** `backend/core/dsp/level_1_invariants_guard.py` misst
alle fünf Invarianten je Phase; `phase_retreat_required` (Stimm-Identität verletzt
oder ≥ 2 Verletzungen) löst die harte Phasen-Rücknahme aus — Rückfall auf die
schwächere Kette (Lauf ohne diese Phase, §L1-R im zentralen Phasen-Call).
```

### B) In §4 nach dem Absatz „Reparatur gilt als **abgeschlossen** …" zwei Bullets

```markdown
- **Erzwingbare Instanz (kanonisch):** `backend/core/dsp/audibility_targets.py`
  — `is_audible(residual, context) -> margin_db` steuert mit EINER Messung
  Detektion (`margin_db > 0` ⇒ hörbar), Dosierung (`dose_for_margin`,
  Stärke ∝ Hörbarkeit, zentral §V7 (copilot-instructions.md)-skaliert) und
  Ziel (`margin_db <= −MARGIN_DB_DEFAULT`, „Maske − Marge"; Margin-Band
  3–6 dB, Default 4 dB). Nachgangsprüfung jeder Reparatur: `witness_chain` —
  Frage A „kein hörbarer Restdefekt?", Frage B „kein hörbares Musikmaterial
  entfernt?" (**NIE OK**); hartes Gate vor Export (§0c (copilot-instructions.md):
  „degraded" statt stiller Verschlechterung).
- **Ohren-Kalibrierung statt Modell-Autorität:** Hörbarkeitsgrenzen werden pro
  Defektklasse am Hörpanel gemessen (Golden-Ear-Korpus
  `backend/core/golden_ear_corpus.py`; 2AFC-Kette
  `scripts/mushra_harness.py thresholds-build/-fit`; Hör-Player
  `scripts/hoerpanel_player.py`) und kalibrieren die Instanz — Literatur-
  Vorbelegungen tragen Status `vorbelegung_hoerpanel_offen`.
```

### C) §8-Tabelle, Zeilen „1" und „2" ergänzen

Zeile 1 anfügen:
`**§L1-R** (level_1_invariants_guard.phase_retreat_required → Rückfall auf schwächere Kette) und **§WG-R** (Familien-Nachgangsprüfung witness_chain, Veto auf family_scalars)`

Zeile 2 anfügen:
`**audibility_targets.is_audible/witness_chain** (erzwingbare Hörbarkeits-Instanz, Margin-Politik 3–6 dB), golden_ear_corpus.py (Ohren-Kalibrierung, Hörpanel-τ)`

## Nach dem Anwenden (verbindlich)

1. `python3 scripts/horordnung_calibration.py` — muss grün bleiben (14/14).
2. Drift-Baseline erneuern (Datei ist WATCHED):
   `python3 -c "from scripts import spec_drift_check as s; s._save_baseline(s._collect())"`
3. `python3 scripts/spec_drift_check.py` → „No spec drift detected".
4. `python3 scripts/id_registry_check.py` → 0 WARNUNG(en).
5. Hinweis für den PR-Evidenzblock (AGENTS.md §4): Dokumentänderung an der
   normativen Spitze — Evidenz sind die 14/14 Kalibrierungs-Checks, 73/73
   Gate-Tests und die beschriebenen Instanzen-Tests (112/112).
6. **Keine Skript-Nachziehung nötig** (hoerordnung §9): Die Ergänzungen
   beschreiben bestehende Enforcement-Implementierung („keine Regeländerung"),
   keine neuen Schwellwerte.
