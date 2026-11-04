# U4 — Phase 08 Transienten-Reparatur: Ursachen-Fix (2026-11-04)

**Status:** Code + Tests implementiert, statisch verifiziert; Messausführung liegt beim Manager
(Ausführung siehe §7/§8 — Worker-Env hat **keinen Shell-Zugang**; es wurde **nichts ausgeführt**,
es werden **keine Grün-Stände behauptet**).
**Ausgangspunkt:** `docs/reports/current/2026-11-04_transient_energie_ursache_fix.md` §3.4
(geprüft und vertieft, nicht blind übernommen).
**Scope-Hinweis:** Nicht angefasst: `scripts/b3_transient_energie_neverworsen.py`,
`backend/core/musical_goals/transient_energy_metric.py`, Wohlklang-Ordnungs-Enforcer, Ebene-3-Pfad.

---

## 1. Ergebnis

phase_08_transient_preservation war funktionslos (effektive Wirkung ≈ 1 %), weil der gemessene
Reparaturbedarf (transient_energie-Lücke) die Stärke-/Wet-Dry-Entscheidung **nie erreichte**.
Der Fix speist die Bedarfs-Kopplung zentral und deterministisch: Bei **nachweisbarer, hörbarer
Ziel-Lücke** dosiert die Phase nach der gemessenen Defekttiefe (gewichteter Ziel-Gap, §2.70
Joint-Calibration, §G188 (GEBOTE.md)) und wird weder von der Risiko-Kette stumm gedämpft noch
vom Strength-Floor-Gate still übersprungen. **Ohne hörbare Lücke bleibt das Verhalten
bit-identisch** (der neue Pfad liegt hinter `elif _goal_need_posterior > 0.0`,
`unified_restorer_v3.py:33298`).

---

## 2. Verifizierte Ursachenkette (file:line, am Code geprüft)

1. **Defekt-Severität ohne Modulation (U1).** `backend/core/defect_phase_mapper.py:2483`:
   `return 1.0  # Enhancement phase — no modulation`. Grund: `_build_reverse_phase_map`
   (`defect_phase_mapper.py:2416-2437`) mappt **nur** `primary_phases` („Only primary_phases
   are mapped … Secondary phases are support roles and should not be severity-scaled").
   phase_08 ist im `_PHASE_MAP` nur **secondary** ⇒ keine Reverse-Map ⇒
   `get_phase_defect_severity` (`defect_phase_mapper.py:2471`) liefert konstant 1.0 —
   der Reparaturbedarf moduliert die Phase nicht.
2. **Stärke-Kette nur dämpfend (U2).** In `_prepare_profiled_phase_runtime_context`
   (`unified_restorer_v3.py:32588` ff.) multiplizieren ausschließlich Risiko-/Unsicherheits-Skalare:
   UQ-Drive-Clip `_sev_wet_dry = float(np.clip(_sev_wet_dry * _uq_scalar, 0.12, 1.0))`
   (`unified_restorer_v3.py:32777`), analog HarmonicAdaptation, Frisson, RecoveryConservatism,
   FallbackTeamwork, StrengthOracle (jeweils Clip-Floor 0.12), und der §8.6g-II Psycho-Scalar
   `_compute_ppsychoacoustic_phase_strength_scalar` bzw. `_compute_psychoacoustic_phase_strength_scalar`
   (`unified_restorer_v3.py:6585` — `risk += 0.06` für Familie „transient“ bei vocals, Signale
   u. a. `vocal_supremacy_conservative`; gemessen 0.094 → 0.089). Kein Skalar liest Ziel-Feedback;
   alle wirken nur in eine Richtung (dämpfend).
3. **Wet/Dry auf Clip-Floor (U3).** Die Kette endet bei 0.12 (beobachtet `songcal=0.120` im
   PHASE_EXEC-Log) ⇒ nur 12 % eingemischt ⇒ effektiv ≈ 0.089 × 0.12 ≈ **1 %**.
4. **Strength-Floor-Gate mit stummem Skip (U4).** `unified_restorer_v3.py:33715` ff.:
   `_kw_strength < 0.15` und Familie `transient_shaping` (Klassifikation
   `unified_restorer_v3.py:7181`) **nicht** in `_LOW_CONFIDENCE_CORE_FAMILIES`
   (`unified_restorer_v3.py:7249`) ⇒ Phase wird übersprungen — der Log nennte keinen
   Reparaturkontext („stilles“ Überspringen, §V6/§G8 (copilot-instructions.md)).
5. **Steering (U5, zweitrangig).** `backend/core/phase_steering_guard.py:155-165`:
   `STOP_GRACEFUL` nur ab `current_phase_idx >= 12` (HPE 0.796 stabil) — phase_08 läuft bei
   6/42 und ist selbst nicht betroffen; der **primäre** transient-Recovery-Kandidat
   (phase_26_dynamic_range_expansion, `backend/core/calibration_matrix.py:1659`) wird im Betrieb
   ggf. abgeschnitten. Nicht Teil dieses Fixes.

## 3. Befund zur Bedarfs-Kopplung (die Leitfrage)

**Warum wird ein Reparaturbedarf (transient_energie-Lücke, Hörbarkeits-Gate) nicht als Bedarf in
die Stärke/Wet-Dry-Entscheidung gespeist?** Die Bedarfserkennung existiert, die Kopplung endet
jedoch vor der Stärke-Entscheidung — drei Brüche:

- **§GOAL_BASELINE_CHECK misst, koppelt aber nur die Phasen-AUSWAHL:**
  `unified_restorer_v3.py:13985` (`if _gbc_proxy < _gbc_target * 0.97:  # Proxy ±2-3% uncertainty`)
  erkennt die Ziel-Lücke auf dem Eingang und schreibt sie als Kontext
  (`unified_restorer_v3.py:14051` `self._restoration_context["goal_deficit_goals"] = ...`).
  Diesen Eintrag liest im Produktionscode **niemand** — einziger Leser ist
  `tests/unit/test_goal_baseline_check_optimizations.py`. Die Stärke-/Wet-Dry-Kette hat keinen
  Ziel-Feedback-Eingang.
- **Die zentrale goal-aware Skalierung kannte phase_08 nicht:** §2.70 Joint-Calibration
  (`backend/core/joint_calibrator.py:1-10`: „Enhancement-Phasen werden nur bei passenden
  Musical-Goal-Gaps aktiviert“) ist genau die verlangte Kopplung — aber `PHASE_EFFECT_CATALOG`
  (`backend/core/phase_effect_catalog.py:47`) hatte **keinen phase_08-Eintrag** ⇒
  `results[pid] = 0.0` (`joint_calibrator.py:158`): die Ziel-Lücke konnte die Stärke nie anheben.
  Zusätzlich geht deren Ergebnis nur als `conductor_strength_hints` ins Log („[advisory only]“,
  `unified_restorer_v3.py:33270`) und fließt nicht in `strength`/`_sev_wet_dry`.
- **Der einzige harte Stärken-Eingang ist die Defekt-Posterior-Schleife** (§G188-Override
  `unified_restorer_v3.py:33288-33297`), zu der phase_08 als Enhancement-Phase (U1) keinen
  Zugang hat.

Damit ist die Kette §3.4 des Ursachen-Reports vollständig bestätigt und um den
**Kopplungs-Befund** vertieft: Nicht die Stärke war falsch „justiert“, sondern der Bedarf kannte
keinen Weg in die Entscheidung.

## 4. Fix — Ursache statt Blind-Justieren (§V7 (copilot-instructions.md))

| # | Änderung | Stelle | Wirkung |
|---|----------|--------|---------|
| A | phase_08-Ziel-Wirkungs-Profil (transient_energie +0.05, artikulation +0.03, groove +0.02, micro_dynamics +0.02; risks harshness/over_punchiness; time_profile fast) | `backend/core/phase_effect_catalog.py:165-182` (§2.76-Muster) | Die §2.70 Joint-Calibration sieht die Phase wieder; Klassifikation nach Ziel-Wirkung statt Vergessenheit |
| B | `GOAL_NEED_AUDIBILITY_JND = 0.05` + `goal_need_strength()` (pure function, JND-Gate, Typ-Robustheit) | `backend/core/joint_calibrator.py:40-43, :76-100` | Zentrale, deterministische Bedarfs-Dosis = gewichteter Ziel-Gap (§2.70), nur bei hörbarer Lücke |
| C | Bedarfs-Posterior-Berechnung (Quellen: Phase-0-Baseline `goal_baseline_scores`, Fallback `_phase_pre_snapshot`; Ziele: kanonische Böden `CANONICAL_THRESHOLDS_RESTORATION`/`_STUDIO2026`) | `backend/core/unified_restorer_v3.py:32551-32579` | Ziel-Feedback gelangt zentral in die Stärke-Entscheidung |
| D | §G188-konformer Ziel-Bedarfs-Override: `strength = Dosis`, `_sev_wet_dry = 1.0`, Debug-Kwarg `phase_goal_need_strength`; nur als `elif` nach dem Defekt-Posterior-Pfad | `unified_restorer_v3.py:33298-33327` | Dosis = gemessene Defekttiefe; keine Doppel-Dämpfung; ohne Bedarf unberührt |
| E | §v10.0.5 überspringt bei hörbarem Bedarf nicht | `unified_restorer_v3.py:33237` | Kleine Planstärken blockieren die Reparatur nicht mehr |
| F | Strength-Floor-Gate: kein Skip bei hörbarem Bedarf (§G69); Skip begründet geloggt | `unified_restorer_v3.py:33715-33738` | „Phase darf bei nachweisbarem Reparaturbedarf wirken“; kein stilles Überspringen |

**Dosis-Definition (zentral, nicht Festwert):** `strength = gewichteter Ziel-Gap` aus
`goal_need_strength` (Normalisierung `utility / applicable_impact` in
`joint_calibrator.py:167` — bei nur einer Ziel-Lücke exakt der Gap; bei mehreren das
gewichtete Mittel, gewichtet mit `GOAL_WEIGHTS` (Hörordnung §2-Prioritäten)).
**Wet/Dry = 1.0 im Bedarfsfall:** Die Dosis steckt bereits in der Stärke (die Phase blendet
intern dosiert, `backend/core/phases/phase_08_transient_preservation.py:502`
`enhanced = audio + _effective_strength * (enhanced - audio)`); eine zweite Misch-Reduktion wäre
die beobachtete Doppel-Dämpfung und verstieße gegen §G188 (GEBOTE.md:536: „… nie einen gemessenen,
reparaturwürdigen Defekt pauschal nur teilweise entfernen“). Alternative `wet = Dosis` wurde
verworfen (quadratische Dämpfung Dosis² setzt die Wirkungslosigkeit fort).

**Warum kein Flag-Flip (§v10.802 — ID-Kollision beachtet, AGENTS.md §5):** Die naheliegende
statische Listen-Aufnahme von `transient_shaping` in `_LOW_CONFIDENCE_CORE_FAMILIES`
(`unified_restorer_v3.py:7249`) wurde **verworfen**: Sie ändert das Verhalten auch ohne Bedarf
(bricht die geforderte Bit-Identität) und widerspräche v10.303 (copilot-instructions.md:422 —
`transient_shaping` gehört zu den bei Low-Confidence gestrippten Familien). Stattdessen ist die
Klassifikation jetzt **bedarfsabhängig** und normativ begründet: §G69 (GEBOTE.md, Kategorie IX:
„Jede Phase, die Defekte füllt/ersetzt/repariert … MUSS als Repair-Phase klassifiziert sein; die
Klassifikation steuert SFT-Wet-Minimum und Strength-Floor“) — mit gemessener Ziel-Lücke **füllt**
phase_08 Transienten-Energie (HPSS-Rescue + Transient Re-Injection, `06_phases_system.md:170`)
⇒ Repair, ohne Lücke Enhancement wie bisher.

**§V7-Konformität:** keine phasen-individuellen Schwellwerte (einzige Schwelle: die repo-weite
JND 0,05 — identisch mit §v10.0.5 Min-Effective-Strength (`unified_restorer_v3.py:33236`),
Never-worsen-JND 0,05 (`scripts/b3_transient_energie_neverworsen.py`),
`_CORE_DROP_THRESHOLD` §v10.702 R4 (`backend/core/excellence_optimizer.py:363`)); keine
Stärke-Festzahlen (Dosis = gemessene Defekttiefe, §G188 (GEBOTE.md:536)); zentrale, rein
deterministische Entscheidung (§G5 (copilot-instructions.md): keine Zeit-/Zufallsquellen —
gleiche Proxies + Ziele ⇒ gleiche Dosis).

**Normkonflikt §G71/§G73 vs. §G188 (nicht stillschweigend entschieden, GEBOTE.md-Konfliktregel):**
Bei hörbaren, aber kleinen Gaps (z. B. 0,07 auf Fenster 2) liegt die evidenzdosierte Wirkung unter
der §G71-Wirksamkeitsschwelle 0,15 (GEBOTE.md:220) bzw. unter §G73 `min_strength` 0,20
(GEBOTE.md:224). Der Fix wählt §G188 (Kompensationsgrad folgt der Evidenzqualität, „nie über feste
Prozente“ — neuere, speziellere Regel 10.17.1), loggt die Abweichung zur Laufzeit als **WARNUNG**
(§V6) und dokumentiert sie hier für den PR-Evidenzblock (AGENTS.md §6). Nachkalibrierung nur
zentral (Formel in `goal_need_strength`), sobald die Scorecard-Messung vorliegt.

## 5. Schutz der Ebene-1/2-Ziele und des Enforcers

- **Hörordnung §2 (Ebene-1/2: emotionalitaet, authentizitaet, natuerlichkeit, waerme):**
  Die Dosisgewichte respektieren die Prioritäten (`GOAL_WEIGHTS`: natuerlichkeit 1,2 >
  authentizitaet 1,1 > waerme 1,0 > emotionalitaet 0,8 > transient_energie 0,7 —
  `joint_calibrator.py:26-38`; Test `test_ebene1_ziele_haben_hoehere_dosisgewichte_als_transient_energie`).
  Ein **rein** Ebene-1-Defizit dosiert die Transienten-Phase gar nicht
  (`test_ebene1_defizit_allein_dosiert_keine_transienten_phase` — die Phase bedient diese Ziele
  nicht). Die Dosis ist gap-proportional (kein pauschales Hochfahren); alle nachgelagerten
  Schutzebenen bleiben unverändert: Perceptual-JND-Gate, VQI/CIG/AFG-Rollbacks, §v10.0.5,
  Pre-Execution-Skip (greift bei `wet_dry = 1.0` nicht mehr — genau der behobene Dämpfungspfad).
- **Wohlklang-Ordnungs-Enforcer (`backend/core/wohlklang_ordnung_gate.py`) nicht unterlaufen:**
  Die Datei ist unverändert; die neuen Blöcke setzen ausschließlich `strength`/`_sev_wet_dry`/
  Debug-Kwargs **vor** der Phasenausführung. Ordnungs-Enforcer und Ziel-Delta-Prüfungen laufen
  danach wie bisher (Signatur-Regressionstest
  `test_wohlklang_ordnung_enforcer_bleibt_verdrahtet`).
- **§Ebene-3 Transparenz:** läuft separat, unberührt.

## 6. Transparenz (deutsche Logs, §G8/§V6 (copilot-instructions.md))

- `§U4 Ziel-Bedarfs-Kopplung <phase>: hörbare Ziel-Lücke → Dosis <x> (strength=… wet_dry=…, Kompensationsgrad 1,0 — §G188 (GEBOTE.md))` (INFO) — jede Bedarfs-Dosis begründet.
- `§U4/§G71-Grenzfall <phase>: Bedarfs-Dosis <x> unter der §G71-Wirksamkeitsschwelle 0.15 …` (WARNING) — sichtbarer Norm-Grenzfall.
- `🪫 Strength-Floor-Gate <phase>: … → uebersprungen (kein hörbarer Ziel-Bedarf — Familie … ohne Reparatur-Evidenz)` (INFO) — Skip **mit** Grund.
- `§U4 Strength-Floor-Gate <phase>: Dosis <x> < 0.15, aber hörbarer Ziel-Bedarf <y> → Reparatur wirkt` (INFO).
- §v10.0.5-Skip-Log unverändert („negligible effect“) — greift nur noch ohne hörbaren Bedarf.

## 7. Erwartete Wirkung (Ableitungen, keine Messzahlen — §V7)

Auf den harten Fenstern (transient_energie **0,63 / 0,73** gemessen, Ursachen-Report §5) ergibt
sich gegenüber dem Boden **≥ 0,80** (Restoration, `.github/instructions/musical_goals.instructions.md:20`;
kanonisch `backend/core/calibration_matrix.py:181`) eine Ziel-Lücke von **0,17 / 0,07** — beide ≥
JND 0,05 ⇒ beide erzeugen nun eine wirksame Dosis (0,17 bzw. 0,07, wet_dry 1,0) statt effektiv
≈ 1 % (0,089 × 0,12). Aus der Phase-Physik (`attack_gain_db` 3–6 dB,
`phase_08_transient_preservation.py:132`; dosierter Blend :300/:502) folgt bei Dosis 0,17 grob
+7…17 % Amplitude im Attack-Anteil; die tatsächliche Ziel-Erfüllung („Richtung 0,80“) wird über
die Scorecard gemessen und **nicht vorab zugesagt**. Ohne Ziel-Lücke: bit-identischer Altpfad.

## 8. Verifikations-Befehle (AUSSTEHEND — Manager führt aus; Worker ohne Shell)

```bash
# 1) Neue Tests
nice -n 19 python3 -B -m pytest tests/unit/test_phase_08_ziel_bedarfs_kopplung.py -q
# 2) Adjazenz (test_phase_08* / test_defect_phase_mapper* / Joint-Calibration)
nice -n 19 python3 -B -m pytest \
  tests/unit/test_phase_08_transient_preservation.py \
  tests/unit/test_phase_08_energy_directive_expansion.py \
  tests/unit/test_phase_08_repair_classification.py \
  tests/unit/test_phase_08_jdc_r2_truthful.py \
  tests/unit/test_defect_phase_mapper_tier_aware_vulnerability.py \
  tests/unit/test_defect_phase_mapper_token_variance_damping.py \
  tests/unit/test_defect_phase_mapper_v2_crackle.py \
  tests/unit/test_defect_phase_mapper_broadband_channel_centroid_gating.py \
  tests/unit/test_musical_harmonization.py -q
# 3) Kritischer Lint-Gate (wie Pre-Commit)
ruff check --select F821,F601,B009,I001 \
  backend/core/joint_calibrator.py backend/core/phase_effect_catalog.py \
  backend/core/unified_restorer_v3.py tests/unit/test_phase_08_ziel_bedarfs_kopplung.py
# 4) Unit-Smoke (wie Pre-Commit aurik-unit-smoke)
python3 -B -m pytest tests/unit -q --maxfail=3
# 5) Vor dem Anlegen/Commit (Write-Gate + Ledger, Worker-Env nicht verfügbar):
python scripts/repo_search.py --before-create tests/unit/test_phase_08_ziel_bedarfs_kopplung.py
python scripts/repo_search.py --before-create docs/reports/current/2026-11-04_u4_phase08_transient_reparatur.md
python scripts/change_ledger.py snapshot   # → TASK_CHANGES.md (CI-Abdeckung)
```

## 9. Never-worsen-Messbefehle vorbereitet (Ausführung: Manager)

Muster `scripts/b3_transient_energie_neverworsen.py` (unverändert; Flags am Skript-Header
prüfen, `--help`). Vor-/Nach-Läufe identisch bis auf den Code-Stand; Scorecard in einer
Dummy-Stufe „objektiv“ (kein Erzeugen menschlicher Labels):

```bash
python scripts/decision_round_preparation.py --mode scorecard --campaign b3 \
  --input-dir samples/corpus_music/B3_transient_energie_v2_wave_validation_input \
  --round-dir decision_rounds/b3_u4_phase08_vor     # Stand ohne diesen Fix
python scripts/b3_transient_energie_neverworsen.py --objektiv \
  --scorecards-dir decision_rounds/b3_u4_phase08_vor > neverworsen_vor.json
# … nach Checkout des Fix-Stands … analog mit decision_rounds/b3_u4_phase08_nach
python scripts/human_gate_round_trip.py --jdc --objektivbeschluss
```

**Akzeptanz (Ziel-Größen):** transient_energie auf den harten Fenstern (bisher 0,63 / 0,73) ↑
(Richtung Boden ≥ 0,80); **alle anderen Goals ≥ Basis** (Never-worsen/JND 0,05, Sign-Test);
Ordnungs-Verletzungen (WO_ENFORCER_DOMINANCE) = 0; SCHEMA_VERSION-Kompatibilität geprüft.
Vergleichslogik: `scripts/compare_scorecards.py` (Änderungs-Toleranz 0,02/JND 0,05 — Muster aus
`docs/reports/current/2026-11-04_transient_energie_ursache_fix.md` §6).

## 10. Conventional-Commit-Vorschlag (Manager führt aus)

```
fix(pipeline): U4 Ziel-Bedarfs-Kopplung für phase_08 (§G188)

- phase_08 in PHASE_EFFECT_CATALOG: gewichteter Ziel-Gap speist die Dosis (§2.70)
- goal_need_strength(): zentrale, deterministische Bedarfs-Dosis, JND-Gate 0,05
- §G188-Override: Dosis = Gap, wet/dry = 1,0 (keine Doppel-Dämpfung);
  ohne hörbaren Bedarf bit-identisch (§G5)
- Strength-Floor-Gate/§v10.0.5: kein Überspringen bei nachweisbarem Bedarf (§V6/§G8)
- 17 Tests; Report; Never-worsen-Messbefehle vorbereitet (Ausführung: Manager)
```

**Version:** 10.12.47 → 10.12.48 (§v10.802 (copilot-instructions.md:462): fix ⇒ Patch-Bump;
Sync erledigt in `backend/core/version.py`, `pyproject.toml`, `README.md`, `CHANGELOG.md`).
Hinweis zur ID-Kollision (AGENTS.md §5): „§v10.802“ existiert auch in der Hörordnung
(GO/NO-GO-Entscheidungsblatt) — hier ist die copilot-instructions.md-Variante gemeint; gemeintes
Verbot „keine Flag-Flips“: keine Verhaltens-/Freigabe-Flags wurden umgelegt (siehe §4).

## 11. Offene Punkte

1. **Normkonflikt §G71 (0,15)/§G73 (0,20) vs. §G188 (evidenzdosiert):** im PR-Evidenzblock
   dokumentieren (AGENTS.md §6); Laufzeit-WARNUNG ist gesetzt. Nach der Scorecard-Messung
   entscheiden, ob die Dosis-Formel zentral nachkalibriert wird.
2. **Ziel-Basis:** aktuell kanonische Böden (0,80/0,83) statt material-/ceiling-korrigierter
   Song-Ziele (§1.4.6b, §G190 (GEBOTE.md)). Bewusst so gewählt (Auftrags-Boden; die
   Bias-Korrektur ist für die Phasen-AUSWAHL gedacht). Zentral umstellbar an einer Stelle
   (`_u4_targets` in `unified_restorer_v3.py:32568`).
3. **Katalog-Ausbau:** phase_26_dynamic_range_expansion, phase_23_spectral_repair,
   phase_36_transient_shaper besitzen ebenfalls kein Profil — ohne Profil weiterhin keine
   Bedarfs-Dosis (Verhalten wie bisher). Nach dem Never-worsen-Befund entscheiden.
4. **`_joint_calibration`-Timing (§2.70):** Die Planungsinstanz liest `_phase_pre_snapshot`,
   das vor Phase 1 leer sein kann. Der Fix liest deshalb die Phase-0-Baseline
   (`goal_baseline_scores`) mit Fallback `_phase_pre_snapshot`; fehlen beide Daten, greift kein
   Bedarf (bit-identisch). Optionale Nachführung „resolved goals“ analog §v10.21
   (resolved-defects-capping) ist offen.
5. **Steering (U5):** STOP_GRACEFUL schneidet phase_26 ggf. ab — separat bewerten, sobald
   phase_08 wirkt.
6. **§1.4.6-Revision V27 (Onset-Set)** — unabhängig, aus dem Vorgänger-Report übernommen.

---

### Methodik / Verifikationsstand

Alle Ursachen-Behauptungen sind am Code mit file:line verifiziert (§2/§3). Die Tests sind
**statisch** gegen die echten Signaturen geprüft (Imports, Funktions-Signaturen,
Dosis-Berechnungen, DSP-Ableitungen aus `attack_gain_db`/Blend) und in
`tests/unit/test_phase_08_ziel_bedarfs_kopplung.py` abgelegt (17 Tests; 2 DSP-Tests mit
synthetischen Transient-Signalen). **Nichts wurde ausgeführt** (Worker-Env ohne Shell): alle
Laufzeit-Befehle stehen in §8/§9 als ausstehende Ausführung beim Manager. Die Bit-Identität
„ohne Bedarf“ ist konstruktiv belegt (Bedarfs-Funktion liefert 0,0; neuer Pfad hinter `elif`),
aber nicht laufzeit-messend verifiziert — vor dem Merge zu prüfen (§8, Test 1 + Unit-Smoke).

---

## 10. Manager-Nachtrag (2026-11-04): Ausführung, Review-Korrektur, Verifikation

Der Worker-Dispatch erreichte die 1-h-Wandzeit vor dem Abschluss-Report; die
Arbeit (10 Dateien) wurde übernommen, geprüft und vollendet. Das Review
ergab **einen Substanzfehler** (korrigiert) und **einen Testfehler** (korrigiert):

1. **Dosis-Funktion (Substanz):** `goal_need_strength()` delegierte an
   `joint_calibrate()` — für Defekt-Phasen (in `reverse_phase_map`, u. a.
   phase_08) liefert dessen §2.70-Defektzweig aber die **Defekt-Tiefe**, nie
   Ziel-Lücken: die Dosis war strukturell immer 0,0, die Bedarfs-Kopplung
   blieb tot. **Korrektur (Ursache statt Symptom, §V7):** die Funktion
   berechnet den gewichteten Ziel-Gap jetzt direkt (Katalog-Impact ×
   GOAL_WEIGHTS, normaliert über die anwendbaren Impacts — identische Formel
   wie der §2.70-Goalzweig; lazy Import wie `joint_calibrate`). `joint_calibrate`
   selbst bleibt unverändert (§2.70-Vertrag der Defekt-Phasen gewahrt).
2. **Test 4:** `test_joint_calibration_liefert_phase_08_aus_der_zielluecke`
   erwartete die Dosis von `joint_calibrate` — das widerspricht dem §2.70-
   Vertrag. Neu: die Dosis kommt aus `goal_need_strength`; `joint_calibrate`
   liefert ohne Defekt-Evidenz weiter 0,0 (Vertrags-Schutz).

**Verifikation (ausgeführt, nicht nur statisch):**
`tests/unit/test_phase_08_ziel_bedarfs_kopplung.py` **17/17 grün**;
Adjazenz (phase_08-Suiten, `test_calibration_matrix`, `test_decision_completeness`,
`test_all_phases_smoke`) **2.364 grün**; UV3 (`test_unified_restorer_v3`,
`test_unified_restorer_stability_guard`) **334 grün**; kritisches Lint-Gate
(F821/F601/B009/I001) **clean** (I001-Importblock in UV3 automatisch sortiert).
Version 10.12.49.

**Never-worsen-Messung** (§8) bleibt wie bei den Vorgänger-Dispatches durch den
Manager vorzubereiten/auszuführen; die Befehle stehen in §8/§9.
