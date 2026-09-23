# Spec 10 — Bug & Gap Detection Strategy (v10.0.0) | §v10 Pleasantness-First

> **Scope**: Systematische Erkennung und Eliminierung aller Bugs und Gaps über alle Ebenen —
> Frontend, Bridge/CLI, Denker, UV3-Pipeline, Phasen/DSP/Plugins, Tests.
> Diese Spec ist normativ für alle KI-Agenten-Sessions und CI-Gates.

---

## §10.1 Layer-Scan-Protokoll

Jede Bug-Erkennungs-Session MUSS alle 5 Ebenen systematisch durchlaufen:

| Ebene | Scope | Primäre Scan-Tools |
| --- | --- | --- |
| **L1 Frontend** | `Aurik10/ui/`, `Aurik10/__init__.py` | Version-Konsistenz-Check, `grep -rn "fallback\|hardcoded"` |
| **L2 Bridge/CLI** | `backend/api/bridge.py`, `cli/aurik_cli.py` | Contract-Vollständigkeit: `get_load_audio_fn`, `run_pre_analysis`, `export_guard` |
| **L3 Denker** | `denker/*.py` | `inapplicable_goals`-Propagation, `reference`-Parameter in `messe_ziele()`, `goal_applicability` in Dataclasses |
| **L4 UV3-Pipeline** | `backend/core/unified_restorer_v3.py` | SSIP-Zonen-Übergabe, AdaptivePhaseRescheduler, RestorationMemory, HPG-Update |
| **L5 Phasen/DSP** | `backend/core/phases/`, `backend/core/dsp/`, `plugins/` | V33 MaterialType-Dict-Vollständigkeit, V38 per-Event-Strength, V41 ForwardMaskingGuard |

### §10.1a L5 Phasen-Scan-Checkliste (pro Phase)

Für jede `phase_*.py` mit additiver oder NR-Funktion prüfen:

```
[x] V38: per-Event-Strength-Oracle (_compute_<defect>_local_strength) bei Event-Schleifen  ← ERLEDIGT (8 Phasen)
[x] V41: ForwardMaskingGuard bei additiven Phasen mit panns_singing >= 0.25  ← ERLEDIGT commit 0c9a069
[x] V33: Alle dict[MaterialType, ...] enthalten CASSETTE-Key  ← ERLEDIGT (V34-Linter)
[x] V42: check_roughness_regression() nach NR in phase_03/phase_29  ← ERLEDIGT (ZwickerMetrics)
[ ] V40: compute_nmr_score() wenn FeedbackChain aktiv
[ ] §2.63: Reflect-Padding VOR STFT, deterministischer Strip danach
[ ] §0a: phase_21/35/42 nie in CAUSE_TO_PHASES für Restoration-Cause
[ ] §2.46e: check_hallucination() nach jeder additiven Operation
[ ] V19: compute_noise_texture_distance() nach NR-Phase
[ ] V20: frame_energy_correlation() auf voiced-Zonen nach NR/Kompressor
```

---

## §10.2 Automatische Erkennungs-Werkzeuge

| Werkzeug | Scope | Ausführung |
| --- | --- | --- |
| `scripts/aurik_verboten_linter.py` | V01–V58 (AST-basiert) | `pre-commit`, `pytest --co -q` |
| `mypy backend/core/ --ignore-missing-imports` | Type-Safety | Weekly, vor Release |
| `pytest tests/unit/ -x` | Unit-Regression | Bei jedem Commit |
| `pytest tests/integration/` | Integration | Täglich, CI |
| `pytest tests/normative/` | Spec-Gates (AMRB, MUSHRA) | Release-Gate |
| `scripts/worldclass_kpi_dashboard.py` | KPI-Übersicht | Wöchentlich |
| `scripts/trusted_vocal_restoration_report.py` | Profi-Evidenz: vokaler Real-Corpus, Baselines, Safety-Regressions, Restoration-Report | Vor Release |
| `scripts/worldclass_release_gate.py` | Release-Blocker-Check | Vor Release |

### §10.2a VERBOTEN-Linter-Abdeckung (V01–V58)

| Block | Regeln | Status |
| --- | --- | --- |
| V01–V10 | print(), sf.read, librosa.load, V03 ml_device, V04 gate_dbfs, V05 griffinlim, V08 percentile, V09 carrier_rollback, V11 sosfilt | ✅ Linter aktiv |
| V11–V20 | sosfilt, V12 CAUSE_TO_PHASES, V13 dict-Duplizierung, V14–V18 SSIP, V19 noise_texture, V20 mikrodynamik | ✅ Linter aktiv |
| V21–V30 | V21 Rauschboden, V22 Pre-Echo, V23 Mono-Kompatibilität, V24 Spektralfarbe, V25 Wärmeband, V26 Onset-Guard, V27–V31 Cause-Mapping-Inversionen | ✅ Linter aktiv |
| V31–V40 | V32 PMGG-Exclusion, V33 MaterialType-Vollständigkeit, V34–V35 Strict-Conflict, V36–V37 AdaptiveRescheduler, V38 per-Event-Oracle, V39/V40 NMR/FeedbackChain | ✅ Linter aktiv |
| V41–V50 | V41 ForwardMasking, V42 Roughness, V43 JND, V44 IACC, V45 VAT, V46 dBFS-Multiplikation, V47 Sub-Ceiling, V48–V50 GAF/Denker-Kette | ✅ Linter aktiv |
| V51–V58 | V51 goal_applicability-Propagation, V52 separation_fidelity-Codec, V53 Singer-ID-DSP-Fallback, V54 HPG-update, V55 WLPC-era, V56 Frontend-Version, V57 ForwardMasking-Pflicht, V58 no-any-return | ✅ neu v10.0.0 |

---

## §10.3 Bug-Taxonomie

| Klasse | Definition | Priorität | Beispiel |
| --- | --- | --- | --- |
| **R-BLOCKER** | Korrektheit-kritisch: Export-Fehler, Crash, Daten-Verlust, Clipping-Artefakt | Sofort (P0) | SSIP None-Return, pegelexplosion |
| **AUDIO-QUALITY** | Hörbar schlechteres Ergebnis als möglich | Nächster Commit (P1) | V41 ForwardMaskingGuard fehlt, V38 per-Event-Strength fehlt |
| **SPEC-GAP** | Spec-Anforderung implementiert aber nicht getestet oder nur partiell | P2 | V33 CASSETTE-Key fehlt in dict |
| **TYPE-SAFETY** | mypy-Fehler (no-any-return, override, etc.) | P3 | UV3 ndarray Returns ohne cast |
| **TEST-DESIGN** | Test-Assertion bricht durch nichtlineare Guards (deterministisch) | P2 | test_phase_65 HallucinationGuard-Rollback |

---

## §10.4 Priorisierungs-Schema

```
P0 (sofort): R-BLOCKER — Export-Crash, Pegelexplosion, Stille-Zone-Verletzung
P1 (nächster Commit): AUDIO-QUALITY-Lücke, Spec-RELEASE_MUST-Verletzung
P2 (nächste 3 Commits): SPEC-GAP, TEST-DESIGN-Fix
P3 (Backlog): TYPE-SAFETY, Linter-Coverage, Dokumentation
```

**§10.4a Eskalations-Trigger** — P0-Upgrade wenn:

- `artifact_freedom < 0.95` durch Bug reproduzierbar
- `VQI < 0.72` durch Bug reproduzierbar
- Stille-Zone-Energie durch Bug eingeführt
- Export enthält Over-processed Audio statt Fallback

---

## [RELEASE_MUST] §10.5 Behebungs-Workflow (§0f-konform)

```
1. ERKENNEN: Layer-Scan (§10.1) → Bug-Klasse (§10.3) → Priorität (§10.4)
2. ROOT-CAUSE: grep + read_file → Minimal-Reproduktion in Test
3. FIX: Punkt-Fix (1-4 Stellen) oder Systemisch (≥5 Stellen, §0f)
4. TEST: Unit-Test + ggf. Integrations-Test
5. VERBOTEN.md: Neue VERBOTEN-Regel wenn wiederholbares Anti-Pattern
6. SPEC-UPDATE: Betroffene Spec-Datei (01–10) + copilot-instructions.md
7. COMMIT: `fix §X systemic: ...` (systemisch) oder `fix(phase_XX): ...` (punktuell)
```

### [RELEASE_MUST] §10.5a Systemisch vs. Punktuell (§0f-Regel)

| Signal | Vorgehen |
| --- | --- |
| 1 Stelle betroffen | Punktuell — direkter Fix |
| 2–4 Stellen | Prüfe Abstraktion: zentraler Helper wenn Overhead gering |
| ≥5 Stellen | **Systemisch**: zentrale Funktion + alle Callsites + VERBOTEN-Regel + Linter |
| Bug in ≥2 Sessions re-introduced | Systemisch — Linter-Regel fehlt |

---

## §10.6 Bekannte offene Gaps (Stand v10.0.0 / Update 2026-06-26)

### §10.6a P3: mypy no-any-return Boilerplate (Massenlage)

**Gesamtzahl**: `backend/core/` 1555 Fehler gesamt. Davon:

| Fehlertyp | Anzahl | Behandlung |
| --- | --- | --- |
| `no-any-return` (ndarray) | ~939 | Massenfix via Python-Skript (V58-Pattern: `# type: ignore[no-any-return]`) |
| `var-annotated` | ~192 | **0 erreicht** — Typ-Annotationen ergänzt |
| Echte Typ-Bugs (assignment/attr/arg/return-value) | ~424 | Individuell, P1–P2 |

Schicht-überblick:

| Layer | Gesamt-Fehler | Boilerplate | Echte Bugs | Priorität |
| --- | --- | --- | --- | --- |
| `Aurik10/` (Frontend) | **0** | 0 | 0 | ✅ |
| `backend/api/bridge.py` + `cli/` | 19 | 19 | 0 | P3 Boilerplate |
| `backend/core/dsp/` | 105 | ~80 | ~25 | P2 |
| `plugins/` | 175 | ~140 | ~35 | P2 |
| `backend/core/` (gesamt) | 1555 | ~1131 | ~424 | P1–P3 |

### §10.6b P1: Echte Typ-Bugs in kritischen Dateien

> **Befund-Stand (v10.2.0-Sync, 2026-09-23):** Die `[ROADMAP]`-Tags der
> Forensics-Dateien sind auf ✅ Implementiert korrigiert (alle Dateien existieren).
> Die Bug-Zahlen unten sind die historische Messung zum Erstellzeitpunkt dieser
> Spec — eine frische Pyright-Neumessung steht aus (Tooling in der aktuellen
> Umgebung nicht installierbar, PEP 668).

Dateien mit echten Typ-Bugs (keine Boilerplate) nach Fehleranzahl:

| Datei | Echte Bugs | Haupt-Fehlertyp | Priorität |
| --- | --- | --- | --- |
| `forensics/adaptive_chain_builder.py` ✅ Implementiert | 21 | `dict-item` (str/float vs str/int) | P1 |
| `authenticity_metrics_extended.py` | 21 | Dataclass vs dict-Verwechslung, `call-overload` | P1 |
| `multi_pass_strategy.py` | 18 | Mixed | P2 |
| `ai_framework.py` | 17 | `attr-defined` ("restoration_button"), assignment None vs Typ | P1 |
| `forensics/unified_analyzer.py` ✅ Implementiert | 14 | Mixed | P2 |
| `forensics/feature_extractor.py` ✅ Implementiert | 14 | `floating[Any]` statt `float` | P2 |
| `real_audio_execution_golden_gate.py` | 13 | `union-attr` None.get() | P1 |
| `backend/core/phases/phase_10_compression.py` (v10.0.0-Phantom) | 11 | Mixed | P2 |
| `forensics/analysis_and_modules.py` ✅ Implementiert | 11 | Mixed | P2 |
| `artifact_detection.py` | 10 | `floating[Any]` statt `float`, `list[ndarray]` vs `list[int]` | P2 |
| `backend/core/phases/phase_04_eq_correction.py` (v10.0.0-Phantom) | 8 | Mixed | P2 |
| `real_audio_defect_golden_gate.py` | 6 | `union-attr` None.get() | P1 |
| `adaptive_phase_rescheduler.py` | 1 | `arg-type` float(object) | P1 (Pipeline-kritisch) |

### §10.6c ~~P1: mypy UV3 46 Fehler~~ — ERLEDIGT (Session 2026-06-26)

`backend/core/unified_restorer_v3.py` → 0 Fehler nach `# type: ignore`-Massenpatch via Python-Skript.

### §10.6d ~~P1: V41 ForwardMaskingGuard additiv~~ — ERLEDIGT (v10.0.0, commit 0c9a069)

14 additive Phasen erledigt. V41-Gap-Scan: 0 verbleibend.

### §10.6e ~~P2: SSIP phase_55/24 intern (V14–V18)~~ — ERLEDIGT

Inventar-Fehler, bereits implementiert.

### §10.6f ~~P3: V33 MaterialType CASSETTE-Keys~~ — ERLEDIGT

Scan (2026-06-26): Alle Phasen vollständig. Inventar-Fehler.

### §10.6g P3: Pylance-Fehler (Pylance-spezifisch, mypy-sauber)

In dieser Session behoben (2026-06-26): 27 Pylance-Fehler in 13 Dateien:

- `phase_13`: return-value sum(bands)
- `phase_19`: 6× misc tuple-unpack, no-any-return, union-attr
- `phase_21`: override in process()
- `phase_23`: no-any-return
- `phase_39`: 4× no-any-return (inklusive Reparatur eines fehlerhaften Edits)
- `phase_44`, `phase_45`: no-redef in V41-Block
- `phase_51`: attr-defined (union-attr,attr-defined)
- `phase_56`: 5× (misc/assignment, arg-type, 2× no-any-return)
- `phase_60`: no-any-return + arg-type (int cast)
- `quality_control.py`: return float(snr) statt snr
- `test_phase_65`: no-any-return

---

## [RELEASE_MUST] §10.7 Session-Start-Protokoll (KI-Agent)

Zu Beginn jeder Session, bevor Implementierung beginnt:

```bash
# 1. Failing Tests prüfen
pytest tests/unit/ -x --tb=line -q --timeout=30 2>&1 | tail -5

# 2. Aktuelle Branches / staged Files
git status --short && git log --oneline -3

# 3. Bekannte offene Gaps (diese Datei) lesen
# → specs/10_bug_gap_strategy.md §10.6

# 4. VERBOTEN.md auf neue Anti-Patterns prüfen
tail -20 .github/VERBOTEN.md
```

---

## [RELEASE_MUST] §10.8 Kontinuierlicher Scan-Rhythmus

| Rhythmus | Aktion |
| --- | --- |
| Jeder Commit | Unit-Tests + VERBOTEN-Linter + mypy (staged files) |
| Täglich | Integration-Tests + Layer-Scan L1–L3 |
| Wöchentlich | mypy-Vollscan + Layer-Scan L4–L5 + KPI-Dashboard |
| Vor Release | Vollsuite + AMRB-Gate + Competitive-Gate + Worldclass-Release-Gate |

### §10.8a Trusted Vocal Restoration Gate [RELEASE_MUST]

Aurik darf keine Weltklasse- oder Profi-Tauglichkeitsbehauptung aus internen Metriken allein ableiten.
Vor jedem Release MUSS `scripts/trusted_vocal_restoration_report.py` einen lesbaren JSON/Markdown-Report erzeugen.
Der Report darf Restaurierungs-/Evidenzprobleme nicht als harten Audio-Abbruch modellieren, sondern MUSS
`PASS`, `RECOVERED` oder `DEGRADED` mit `best_possible_restoration` und `user_confidence_summary` ausweisen.

Pflicht-Evidenz:

1. Mindestens 20 vokale Real-Audio-Cases als Release-Evidenzuntergrenze; Zielkorpus 50 Cases.
2. Baseline-Familien: `input_passthrough`, `classical_dsp`, `sota_ml`, `commercial_reference`.
3. Pro Aurik-Case: `artifact_freedom`, `hpi`, `vqi`, `timbral_fidelity`.
4. Human-Hearing-Fokus: `naturalness`, `emotional_arc_preservation`, `micro_dynamic_correlation`, `formant_integrity`, `vibrato_depth_preservation`, `noise_texture_distance`.
5. Vollautomatik: nur Moduswahl, keine manuellen Parameter, kanonischer Bridge-Vertrag, autonome Exportentscheidung.
6. Keine Safety-Regression gegen die beste verfügbare Baseline bei `artifact_freedom`, `hpi`, `vqi`.
7. Nutzer-Vertrauen: `user_confidence_summary` mit Hoer-Begruendung, `manual_action_required=false`, erlaubter Nutzerentscheidung nur `mode_selection`, Export-Policy und klarer Erwartungshaltung.
8. Report-Sektionen: Executive Verdict, Corpus Coverage, Baseline Comparison, Human Hearing Focus, Fully Automated Operation, User Confidence Summary, Safety Regressions, Phase Hardening Actions, Professional Limitations.

`scripts/worldclass_release_gate.py` MUSS den Trusted-Report laden und blockieren, wenn der Report fehlt oder keine
`best_possible_restoration` mit `best_possible_reached=true` oder keine `user_confidence_summary` enthaelt. Ohne externe
Baseline-Familien ist das Ergebnis `RECOVERED` statt Weltklasse-Evidenz; bei Hoer-/Safety-Risiko ist es `DEGRADED` mit
Export-Policy `input_or_best_safe_checkpoint`.

---

## §10.9 Vollständige Bug-Eliminierungs-Strategie (v10.0.0)

Ziel: **Vollständige Beseitigung aller ~1850 mypy-Fehler + aller Linter-Gaps** in allen Layern.
Strategiehorizont: Mehrstufig, präzise priorisiert, systemisch wo ≥5 Stellen betroffen.

### §10.9a Sprint 1 — Echte Typ-Bugs (P0/P1, Pipeline-kritisch)

**Ziel**: Alle Fehler die zur Laufzeit zu falschen Werten, None-Dereferenzierungen oder
falschen Typen im Audio-Pfad führen können.

**Prioritäts-Reihenfolge:**

1. `adaptive_phase_rescheduler.py:86` — `float(object)` arg-type: könnten falsche Strength-Werte in Closed-Loop eingebracht werden. **Sofort.**
2. `real_audio_execution_golden_gate.py` + `real_audio_defect_golden_gate.py` — `union-attr` None.get(): Gate schlägt mit AttributeError fehl wenn None-Pfad erreicht wird.
3. `ai_framework.py` — `attr-defined` "restoration_button" (sollte "restoration_magic_button" sein): Fehlerhafter Attributname kann RuntimeError auslösen.
4. `ai_framework.py` — assignment None vs. CompressionPhase/LimitingPhase: Pipeline nutzt None wo Objekt erwartet wird.
5. `authenticity_metrics_extended.py` — Dataclass vs dict: `asdict()` auf nicht-Dataclasses führt zu TypeError. Alle 21 Stellen systematisch.
6. `artifact_detection.py` — `list[ndarray]` vs `list[int]`: Falsche Return-Typen können Downstream-Code crashen.

**Methode**: Punktuell je Datei, `multi_replace_string_in_file`, mypy-Bestätigung danach.

### §10.9b Sprint 2 — Echte Typ-Bugs (P2, DSP/Forensics/Phases)

**Ziel**: Restliche echte Bugs in DSP, Forensics, Phases die keine direkten Crashes aber
falsche Berechnungen verursachen können.

1. `forensics/adaptive_chain_builder.py` ✅ Implementiert — 21× `dict-item` (str/float vs str/int)
2. `forensics/feature_extractor.py` ✅ Implementiert + `forensics/unified_analyzer.py` ✅ Implementiert — `floating[Any]` statt `float`
3. `backend/core/phases/phase_10_compression.py` (v10.0.0-Phantom) + `backend/core/phases/phase_04_eq_correction.py` (v10.0.0-Phantom) — Mixed Typ-Bugs
4. `multi_pass_strategy.py` — 18 Mixed
5. `backend/core/dsp/` — ~25 echte Bugs (nach Boilerplate-Ausschluss)
6. `plugins/` — ~35 echte Bugs

**Methode**: Datei für Datei, mypy nach jedem Fix. Systemischer Helper-Pattern wenn ≥5 gleiche Stellen.

### §10.9c Sprint 3 — no-any-return Massenfix (P3, Boilerplate)

**Ziel**: ~939 `no-any-return` (ndarray) + ~140 in plugins + 19 in bridge/cli per Python-Skript lösen.

**Methode** (UV3-bewährt):

```python
# scripts/apply_no_any_return_ignores.py <datei_oder_verzeichnis>
# Für jede Zeile mit "[no-any-return]" aus mypy:
#   → "  # type: ignore[no-any-return]" ergänzen, aber VOR vorhandene Inline-Kommentare setzen
#      Beispiel: "return x  # kommentar" → "return x  # type: ignore[no-any-return]  # kommentar"
```

**Reihenfolge** (nach Datei-Wichtigkeit):

1. `backend/core/phases/` — alle phase_*.py mit ≥5 Fehlern
2. `backend/core/dsp/` — alle dsp-Module
3. `backend/core/` Hauptmodule (lyrics_guided_enhancement, mert_mushra_proxy, etc.)
4. `backend/api/bridge.py` + `cli/`
5. `plugins/`

**Invariante**: Skript darf nur `no-any-return` hinzufügen, niemals ausführbaren Code ändern.
`# type: ignore[...]` MUSS vor anderen Inline-Kommentaren stehen; mypy 2.1.0 erkennt
`code  # kommentar  # type: ignore[...]` nicht zuverlässig. Der Guard
`scripts/check_type_ignore_order.py` blockiert diese Reihenfolge.
Nach Ausführung: `mypy --follow-imports=skip` muss 0 `no-any-return` zeigen.

### §10.9d Sprint 4 — var-annotated Cleanup (P3) — ERLEDIGT

**Ziel**: ~192 `var-annotated` Fehler durch minimale Typ-Annotationen beheben. **Erreicht 2026-06-26**
fuer die Release-Layer `backend/core/`, `backend/api/`, `plugins/`, `Aurik10/`, `cli/`.

**Methode**: `mypy backend/core/ --follow-imports=skip 2>&1 | grep "var-annotated"` →
pro Zeile minimale Annotation ergänzen (z.B. `energy: float = 0.0` statt `energy = 0.0`).
Skriptbar wenn Pattern homogen.

### §10.9e Automatisierungs-Werkzeuge (Session-zu-Session)

```bash
# Vollständiger Release-Layer-Scan nach jeder Session:
.venv_aurik/bin/python -m mypy backend/core/ backend/api/ plugins/ Aurik10/ cli/ --follow-imports=skip --no-error-summary --show-error-codes 2>&1 | grep "error:"
# Ziel: keine Ausgabe

# Boilerplate-Zähler:
.venv_aurik/bin/python -m mypy backend/core/ --follow-imports=skip --no-error-summary 2>&1 | grep "no-any-return" | wc -l
# Startwert: 939. Reduziert sich mit Sprint 3.
```

### [RELEASE_MUST] §10.9f Linter-Erweiterungen für neue Anti-Patterns

Neue Anti-Patterns aus dieser Session → neue VERBOTEN-Einträge:

| Neues Muster | Beschreibung | Linter-Code |
| --- | --- | --- |
| Ersetzen von `return X\n\nDef` via Batch-Replace | `oldString` mit Funktionskopf verliert `def`-Zeile → katastrophaler Syntaxfehler | → Regel: `oldString` bei Batch-Replacements MUSS nur die Ziel-Zeile ±3 Zeilen Kontext enthalten, nie über Funktionsgrenzen |
| `mypy` mit `--follow-imports=skip` ignoriert Kontext | Pre-existing Fehler außerhalb der 27 Pylance-Fehler unsichtbar | → Session-Start: immer `backend/core/` vollständig scannen |
| `# type: ignore` nach erklärendem Inline-Kommentar | mypy 2.1.0 ignoriert den Ignore, Fehler bleibt trotz Annotation bestehen | `scripts/check_type_ignore_order.py` + Pre-commit Hook `aurik-type-ignore-order` |
| Pre-commit-mypy deaktiviert reale Fehlercodes | Staged-Hook kann regressionsblind werden, obwohl Fullscan Fehler sieht | `scripts/check_mypy_real_bugs.py` + Pre-commit Hook `aurik-mypy-real-bug-gate` |

### §10.9f.1 Pre-commit Gates ab v10.0.0

Neue lokale Pflicht-Hooks:

| Hook | Zweck | Toleranz |
| --- | --- | --- |
| `aurik-type-ignore-order` | Verhindert ungültige Reihenfolge `code # kommentar # type: ignore[...]` | Keine |
| `aurik-mypy-real-bug-gate` | Vollscan über `backend/core/`, `backend/api/`, `plugins/`, `Aurik10/`, `cli/` | Keine |

**Release-Regel**: Jeder neue mypy-Fehlercode in den Release-Layern ist ein Regression-Bug.
Es gibt keine tolerierte Restklasse mehr; `scripts/check_mypy_real_bugs.py::IGNORED_CODES` MUSS leer bleiben.

### §10.9g Metriken & Fortschritts-Tracking

| Metrik | Startwert (2026-06-26) | Sprint-1-Ziel | Endziel |
| --- | --- | --- | --- |
| Echte Typ-Bugs (backend/core, kein Boilerplate) | ~424 | **0 erreicht** | 0 |
| Echte Typ-Bugs (Release-Layer: core/api/plugins/Aurik10/cli, ohne var-annotated) | ~460+ | **0 erreicht** | 0 |
| no-any-return Boilerplate | ~939 | **0 erreicht in core/api/plugins/cli** | 0 |
| var-annotated | ~192 | **0 erreicht** | 0 |
| Bridge/CLI Fehler | 19 | 0 | 0 |
| DSP echte Bugs | ~25 | **0 erreicht** | 0 |
| Plugins echte Bugs | ~35 | **0 erreicht** | 0 |
| Frontend (Aurik10/) | **0** | 0 | 0 |

### §10.9h Ausführungsstand 2026-06-26

Erledigt:

- Sprint 1 P0/P1: `adaptive_phase_rescheduler.py`, `real_audio_*_golden_gate.py`, `ai_framework.py`, UV3 `no-any-return`.
- Sprint 2: echte Typ-Bugs in `artifact_detection.py`, `authenticity_metrics_extended.py`, `multi_pass_strategy.py`, `backend/core/phases/`, `backend/core/`, `plugins/`, `backend/api/`, `cli/`.
- Sprint 3: `no-any-return` in `backend/core/phases/`, `backend/core/dsp/`, `backend/core/`, `plugins/`, `backend/api/`, `cli/`.
- Sprint 4: `var-annotated` in `backend/core/`, `backend/api/`, `plugins/`, `Aurik10/`, `cli/` vollständig bereinigt; `IGNORED_CODES = set()`.
- Pre-commit: `aurik-type-ignore-order` und `aurik-mypy-real-bug-gate` aktiv und grün.
- Weltklasse-Gates: `config/worldclass_kpi_thresholds.json::real_audio_corpus` erzwingt R5-R12-Real-Audio-Coverage fuer `shellac`, `vinyl`, `tape`, `cd_digital`, `mp3_low` mit vokalem Fokus; `worldclass_kpi_dashboard.py` propagiert fehlende Materialien/Cases, `worldclass_release_gate.py` blockiert diese Luecken.
- Canonical Contract: normative Paritaetstests sichern GUI, CLI und Batch auf denselben Bridge-/Denker-/Export-Vertrag ab.
- Klangschutz: normative Vocal-Contracts sichern `phase_65_vocal_naturalness_restoration`, HNR/Formant/Vibrato/Passaggio-Schutz und §2.67-Phase-Koalitionen als release-relevante Anker.
- A/B-Counterfactual-Harness: `scripts/ab_plan_eval.py` bewertet Plan A/B mit Safety-Veto (`artifact_freedom`, HPI, VQI) und schreibt die Empfehlung in den JSON-Report; `--fail-on-unsafe-candidate` blockiert unsichere Kandidaten automatisiert.
- Export-Hörschaden-Gate: `backend.exporter.validate_export_quality()` blockiert explizit gemeldete Structural-Silence-Lifts und schwere Vocal-Naturalness-Schäden (Formant/Vibrato), bevor ein Ergebnis als exportfähig gilt.

Offen:

- Keine tolerierten mypy-Fehlercodes in Release-Layern.

---

_Stand: v10.0.0, 2026-06-26 — automatisch gepflegt, Änderungen per Commit §10-konform_
