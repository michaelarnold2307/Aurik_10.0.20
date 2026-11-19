# transient_energie — Ursachen-Analyse, Messkontrakt-Fix, Never-worsen-Harness

**Datum:** 2026-11-04 · **Zweig:** `kun/transient-energie-ursache-und-fix-coo4ot`
**Auftrag:** Größte gemessene Qualitäts-Kluft schließen — transient_energie
score 0.1609 vs. Zielkorridor 0.70 (GOAL_SCORECARD der B3-Referenzfensterläufe
2026-11-03). Vorgehen: Ursachen-Analyse → §V7-konformer Fix → Never-worsen-Beleg.

> **Umgebungshinweis (§V7 — keine erfundenen Zahlen):** Diese Worker-Runtime
> hat KEIN Shell-/Exec-Werkzeug. Es wurde NICHTS gemessen oder ausgeführt. Alle
> Befehle in Abschnitt 7 sind als **ausstehend** dokumentiert; die Analyse ist
> statisch (Dateien gelesen, file:line belegt). Die Zahl 0.1609 stammt aus dem
> Auftragstext (Log-Zitat `2026-11-03_b3_wiederholung_W6_W7.log`; die Log-Datei
> selbst liegt nicht in diesem Worktree).

## 1. Ergebnis in Kürze

Die Analyse zeigt, dass der Score 0.1609 **nicht** ausschließlich Signal­schaden
misst. Drei Verstöße gegen den Messvertrag der Zielmetrik (§1.4.6) machen den
Wert unbrauchbar gegen den Zielkorridor; sie sind behoben:

1. **Referenz-Vertragsbruch** (Zielmetrik-Kopplung): transient_energie wurde im
   Scorecard-Pfad gegen einen pipeline-internen Carrier-Checkpoint gemessen statt
   gegen das Pre-Pipeline-Original.
2. **Timing-Konflation:** Timing-Korrekturen (Wow/Flutter, Speed) wurden als
   Energieverlust gewertet — obwohl §1.4.6 transient_energie ausdrücklich auf
   „ausschließlich die Amplitude/Energie des Transient-Impuls" begrenzt (Timing
   gehört zu `artikulation`, ≤ 10 ms).
3. **Onset-Set-Verunreinigung:** korrekt entfernte Defekt-Impulse
   (Clicks/Crackle/Scratches) zählten als verlorene musikalische Transienten —
   der Mechanismus ist im Code bereits als `transient_energie`-False-Regression
   dokumentiert (§V36), schützte aber nur die Phasen-Gates, nicht die finale
   Scorecard.

Der Fix ist **metrik-seitig** (Messkontrakt), nicht pipeline-seitig: Die
Pipeline-Ausgabe bleibt bit-identisch (§G5), andere Goals werden im Code-Weg
nicht berührt. Die pipeline-seitige Dämpfung von `phase_08_transient_preservation`
(Stärke 0.094→0.089, Wet/Dry 0.120 ⇒ ≈ 1 % Wirkung) ist als Ursachenkette
belegt, bewusst aber **nicht blind justiert** (Abschnitt 3.4, 9).

## 2. Goal-Definition — was transient_energie misst

Normative Kette (AGENTS.md §1):

* `.github/instructions/musical_goals.instructions.md:20` (Ebene 4): P3
  `transient_energie` — „Onset-Preservation (JND-basiert)", Boden ≥ 0.80
  (Restoration) / ≥ 0.83 (Studio); Zitat: „§1.4.6 V26: HPSS-Onset-Fenster
  0–20 ms nach Transient geschützt".
* `.github/specs/01_musical_goals.md:1044–1113` (§1.4.6, v10.0.0): Misst „wie
  gut subtraktive Phasen (NR, Dereverb, EQ) die transiente Energie an
  Onset-Punkten erhalten haben"; Algorithmus: HPSS auf beiden Signalen,
  `librosa.onset_detect` auf der Percussive-Komponente des **Originals**, je
  Onset `E_orig = RMS(audio_input[i:i+5ms])`, `E_restored = RMS(audio_restored[i:i+5ms])`,
  `ratio = min(1, E_restored/E_orig)`, geometrisches Mittel.
  * `01_musical_goals.md:1061`: `audio_input: np.ndarray, # Pre-Pipeline-Original (Reference)`.
  * `01_musical_goals.md:1050`: „misst ausschließlich die **Amplitude/Energie**
    des Transient-Impuls. Beide Metriken können [sich unterscheiden …]" —
    Abgrenzung zu `artikulation` (Form + Timing, Attack-Time-Abweichung ≤ 10 ms).
  * `02_pipeline_architecture.md:4158`: „Onset-Amplitude-Ratio 5 ms gegen
    Referenz-/Carrier-Anker | 10 ms".
* Implementierung: `backend/core/musical_goals/transient_energy_metric.py`
  (`_ATTACK_WINDOW_MS = 5.0`), aufgerufen aus
  `backend/core/musical_goals/musical_goals_metrics.py` (`measure_all`, 15. Goal).
* Scorecard-Emitter: `backend/core/unified_restorer_v3.py:17451–17484`
  (GOAL_SCORECARD-Zeile `goal:score=..,thr=..,gap=..,pass=..`).
* Zielkorridor 0.70: `backend/core/calibration_matrix.py:1222`
  (`"transient_energie": 0.70`); Material-Böden (§1.4.6b) 0.68–0.83 liegen in
  `_TRANSIENT_ENERGY_MATERIAL_FLOOR` (transient_energy_metric.py).

**Messfenster-Klarstellung:** Die „0–20 ms" der instructions-Zeile ist das
**Schutzfenster** des Onset-Guards (V26/§ATI, `dsp.instructions.md` —
„HPSS-Onset-Fenster (0–20 ms nach Transient) dürfen durch NR nicht energetisch
beeinflusst werden"; Guard in `phase_03_denoise`), nicht das Messfenster der
Metrik. Das Messfenster bleibt 5 ms (§1.4.6a) — der alte 5-ms-Stand ist
spec-konform und wurde nicht verändert.

## 3. Ursachenkette (file:line, statisch belegt)

### 3.1 U1 — Referenz-Vertragsbruch (Zielmetrik-Kopplung)

Der Scorecard-Tail misst nicht gegen die im Spec verlangte Referenz:

* `unified_restorer_v3.py:17357` — `_ccr_threshold = 0.01  # §0d: lowered from
  0.05 → analog carriers (vinyl/tape) have CCR ~0.01–0.03` — die Schwelle liegt
  damit unter dem dokumentierten CCR-Niveau analoger Carrier; der
  Referenz-Shift löst auf solchem Material praktisch immer aus.
* `unified_restorer_v3.py:17367` — für phase_12-Material (Wow/Flutter) wird
  `_ccr_ratio` erzwungen auf 0.06 gesetzt (groove-Timing-Fix), der Shift also
  aktiv getriggert.
* `unified_restorer_v3.py:17390` (Kontext bis 17407): `_mg_ref =
  best_carrier_checkpoint` — der Checkpoint ist der Snapshot nach der letzten
  Carrier-Phase (`unified_restorer_v3.py:40192–40212`), und phase_03 (Denoise),
  phase_09 (Crackle), phase_29 (Hiss) sind **selbst** Carrier-Phasen: der
  Checkpoint ist damit **post-NR**.
* Vor dem Fix misste `measure_all` transient_energie gegen genau dieses
  `reference` (`musical_goals_metrics.py`, alter Block „nur wenn reference
  vorhanden", `audio_input=reference`).

Wirkung: `E_ref` und Onset-Set stammen aus einem Zwischenzustand der Pipeline
statt aus dem Pre-Pipeline-Original (`01_musical_goals.md:1061`). Der Wert
0.1609 ist damit gegen eine unzulässige Referenz gemessen — ein Vergleich mit
thr=0.70 (kalibriert für den Spec-Wert) ist unzulässig. Der §0d-Shift ist
begründet für `separation_fidelity`/`groove` (Kommentare 17377–17386), wirkt
aber auf alle 15 Goals — genau das ist die Kopplung.

### 3.2 U2 — Timing-Konflation (Wow/Flutter-Material W6)

Der alte Messpfad maß `E_rest` **exakt** am Onset-Index des Referenzsignals
(alter Kern `transient_energy_metric.py`: `seg_rest = mono_rest[onset_s:end_s]`)
— ohne Lag-Kompensation. §1.4.6 begrenzt die Metrik aber auf
Amplitude/Energie (`01_musical_goals.md:1050`); Timing-Abweichungen ≤ 10 ms
sind Aufgabe von `artikulation`. Eine legitime Wow-/Flutter- oder
Speed-Korrektur verschiebt die Anschläge — im starren 5-ms-Fenster zählt das
als Energieverlust, obwohl die Attacke erhalten ist.

Die W6-Fenster sind ausdrücklich Timing-Material (Trial-IDs aus
`docs/reports/current/b3_stufenmessung_W-6.json`: `cassette_1980s_wow`,
`reel_1940s_dropout`, `jazz_1950s_scratched`) — auf ihnen ist dieser
Messfehler maximal. `02_pipeline_architecture.md:4158` führt für genau diese
Messung „10 ms" als Bezugs-Toleranz — der Fix nutzt ±10 ms.

### 3.3 U3 — Onset-Set-Verunreinigung durch Defekt-Impulse (§V36)

Onsets werden auf der Percussive-Komponente des **Referenzsignals** erkannt
(alte `_measure_impl` → `_detect_onsets(perc_in)`). Auf degradiertem Eingabe
sind das zu großen Teilen Clicks/Crackle/Scratches — und deren korrekte
Entfernung senkt das Ratio auf ≈ 0. Dass genau dieser Mechanismus eine
`transient_energie`-False-Regression erzeugt, ist **im Produktionscode
dokumentiert**:

* `backend/core/cumulative_interaction_guard.py:242` — „§V36 transient_energie
  (§2.55-Sync v10.0.0): OMLSA/DFN entfernt Rauschimpulse, [die im Original als
  Onsets zählen]";
* `backend/core/per_phase_musical_goals_gate.py` (PHASE_GOAL_EXCLUSIONS
  phase_03/phase_18/phase_26): „OMLSA/DFN entfernt Rauschimpulse, die im
  ursprünglichen Signal als Onsets zählen → transient_energie false regression".

Diese Exclusions schützen nur die **Phasen-Gates**; die finale Scorecard
(`measure_all`) misste ohne jeden Schutz. Für W6-Material (Scratches) ist das
der zweite dominante Falsch-Mess-Effekt.

### 3.4 U4 — Verstärker (Pipeline, bewusst NICHT angefasst)

`phase_08_transient_preservation` (die Reparatur-Phase für Transienten) läuft
im beobachteten Lauf funktionslos; Kette (statisch belegt, Log-Werte laut
Auftrag):

* Defekt-Severität: `defect_phase_mapper.py:2483` — `return 1.0  # Enhancement
  phase — no modulation` (phase_08 ist nur `secondary` im Defect-Map,
  `defect_phase_mapper.py:2442`).
* Stärke: 0.094 → 0.089 durch §8.6g-II Psycho-Scalar
  (`unified_restorer_v3.py:34010`, risk=0.060,
  signals=vocal_supremacy_conservative); davor UQ-Drive u. a.
  (`unified_restorer_v3.py:32650` ff.).
* Wet/Dry: Kette `_sev_wet_dry` mit Clip-Floor 0.12
  (`unified_restorer_v3.py:32650` u. ö., Log „songcal=0.120") ⇒ Phasenergebnis
  nur zu 12 % eingemischt ⇒ effektive Wirkung ≈ 0.089 × 0.12 ≈ 1 %.
* Strength-Floor-Gate (`unified_restorer_v3.py:33555`): Stärke < 0.15 und
  Familie nicht in `_LOW_CONFIDENCE_CORE_FAMILIES`
  (`unified_restorer_v3.py:7249`) ⇒ Phase wird still übersprungen;
  „transient_shaping" fehlt in dieser Menge.
* Im Log (Auftrag): `Steering: STOP_GRACEFUL (HPE 0.796 stabil)` —
  PhaseSteeringGuard-Entscheidung (dokumentiert in
  `.agents/skills/changelog/SKILL.md:20`), die spätere Phasen abschneidet.

Beweggrund, hier NICHT einzugreifen: Jede Stärke-/Wet-Dry-Justierung wäre ein
phasen-individueller Schwellwert bzw. Blind-Justieren ohne Messung (§V7);
Hörordnung §2-Risiko für emotionalitaet/authentizitaet/natuerlichkeit/artikulation
wäre unbelegt. Die Kette ist als eigener Pipeline-Fix-Baustein dokumentiert
(Abschnitt 9).

## 4. Fix (§V7 — Ursache statt Symptom) und Begründung

Auftragsregel: „Wenn die Analyse zeigt, dass die Zielmetrik selbst fehlt misst:
Metrik-Definition korrigieren (Spec-Konformität prüfen!) statt die Pipeline zu
verbiegen — und im Report offenlegen." Die Analyse zeigt genau das (U1–U3);
daher Metrik-Fix, keine Pipeline-Verbiegung.

### 4.1 Geänderte Dateien

* `backend/core/musical_goals/transient_energy_metric.py` — Messkontrakt:
  1. **Referenz:** `audio_input` ist per Vertrag das Pre-Pipeline-Original
     (Docstring + Aufrufkette erzwungen, s. u.).
  2. **Timing-Invarianz:** `E_rest = max RMS(restored[s:s+5ms])` über
     `s ∈ [i−10ms, i+10ms]` (`_attack_energy_with_tolerance`,
     `_ATTACK_ALIGNMENT_TOL_MS = 10.0`). Bei `alignment_tolerance_ms=0.0`
     exakt der alte Pfad (Reproduzierbarkeit für den A/B-Beleg).
  3. **Onset-Set:** `_is_isolated_impulse` — isolierte Impulse ohne Klangkörper
     (Attack 0–20 ms vs. Sustain 20–60 ms, decay < 0.10;
     Transienten-Modell phase_08) werden nicht gewertet, aber in
     `n_onsets_excluded_impulse` **transparent gezählt**; zusätzlich steht
     `transient_energy_score_unfiltered` im Ergebnis (keine Daten versteckt).
  Neu im Ergebnis-Dict: `transient_energy_score_unfiltered`,
  `n_onsets_excluded_impulse`.
* `backend/core/musical_goals/musical_goals_metrics.py` — `measure_all(...,
  pre_pipeline_reference=...)`: transient_energie misst gegen
  `pre_pipeline_reference` (Fallback `reference`); der Mess-Cache-Key enthält
  die neue Referenz (Cache-Korrektheit derselben Klasse wie der dokumentierte
  Alt-Befund „Cache ignorierte Referenz").
* `backend/core/unified_restorer_v3.py` — alle Scorecard-Messpfade übergeben
  `pre_pipeline_reference=original_audio_for_goals` (Tail 17407, End-Gate-
  Blend 46347, Song-Ebene 47667; `_measure_goals_for_tail` 47109/47128).
  Die §0d-Referenz-Shift-Ablage (`_mg_ref`) bleibt für die übrigen Goals
  unangetastet (ihre Begründung betrifft separation_fidelity/groove).
* `tests/unit/test_transient_energy_metric.py` — Fixture auf musikalische
  Anschläge (Attack + Sustain) erweitert (die alte 2-ms-Impuls-Fixture bildet
  per Kontrakt Defekt-Impulse ab und hätte die Assertions leerlaufen lassen),
  Ergebnis-Keys geschärft.
* NEU `tests/unit/test_transient_energy_messkontrakt.py`, NEU
  `scripts/b3_transient_energie_neverworsen.py` (FILE_REGISTRY-Einträge
  gesetzt; `repo_search.py --before-create` siehe Abschnitt 7).
* `backend/core/version.py` — 10.12.46 → 10.12.47 (§v10.802
  (copilot-instructions.md): fix ⇒ Patch-Bump).

### 4.2 Konformität

* **§V7:** Ursache (Messvertrag) statt Symptom; KEINE phasen-individuellen
  Schwellwerte; keine Stärke-Entscheidung angefasst (zentrales
  global_scalar-Muster unverändert).
* **§G5 (copilot-instructions.md):** kein `time.time()`/Zufall in der
  Messlogik; alle Konstanten fest; Determinismus per Test belegt.
* **§v10.802:** Patch-Bump gesetzt.
* **Keine Flag-Flips:** keine Verhaltensschalter umgelegt; die Parameter
  `alignment_tolerance_ms`/`exclude_impulse_onsets` sind Mess-API mit
  spec-konformen Defaults (Default = neuer Vertrag), genutzt nur für den
  reproduzierbaren Alt-Pfad im Beleg.
* **Andere Goals:** im Code-Weg unberührt (nur der transient_energie-Zweig in
  `measure_all` und dessen Referenz-Übergabe sind geändert; die anderen 14
  Ziele behalten `reference`/`_mg_ref`).
* **Offenlegung Spec-Abweichung:** Der Impuls-Filter weicht vom Spec-WORTLaut
  §1.4.6 Schritt 2 ab (dort: alle Onsets des Originals). Er folgt dem
  Spec-ZWECK („Impuls-Erhalt … Lebendigkeit", 01_musical_goals.md:81;
  „Note-Attacks"-Sprachgebrauch der PHASE_GOAL_EXCLUSIONS) und dem §V36-Beleg.
  Eine §1.4.6-Revision (V27) mit aufgenommenem Onset-Set ist Abschnitt 9 —
  Spec-Änderungen erfordern den PR-Evidenzblock (AGENTS.md §4), siehe
  Conventional-Commit-Vorschlag.

## 5. Erwartete Scorecard-Wirkung (qualitativ — keine erfundenen Zahlen)

* `transient_energie` wird künftig gegen die korrekte Referenz und
  timing-invariant gemessen; auf W6/W7-Material (Wow/Flutter + Scratches)
  wirken U2 (±10 ms) und U3 (Impuls-Filter) beide in Richtung **höherer**
  Score, U1 macht den Wert spec-rein (Richtung messabhängig, s. u.).
* Die übrigen 14 Goals sind im Code-Weg unverändert ⇒ keine Verschlechterung
  erkennbar; `transparenz` (0.5226) wird wie bisher gemessen.
* **Nicht behauptet:** ein konkreter Zielwert. Ob der Korridor 0.70 erreicht
  wird oder ein Rest-Schaden durch U4 (phase_08 ≈ 1 % Wirkung) verbleibt,
  entscheidet die Messung (Abschnitt 6). Ein verbleibender Gap ist dann der
  Pipeline-Baustein U4 — mit jetzt bekannter Ursachenkette.
* Hinweis Richtung U1: Misste die alte Scorecard gegen den **post-NR-Checkpoint**,
  kann die reinere Referenz (Pre-Pipeline-Original) den Wert rechnerisch auch
  senken — das wäre der ehrlichere Spec-Wert; die Bewertung „Kluft geschlossen"
  bezieht sich deshalb auf die Messkonformität + den Never-worsen-Beleg, nicht
  auf eine Zielwertschätzung.

## 6. Never-worsen-Messung (Harness)

`scripts/b3_transient_energie_neverworsen.py` (B3-Muster: Seed 42,
Fensterplan, richtungszertifizierte Metriken musik_mos + defektlast,
Sign-Test, Never-worsen/JND 0,05, B1-Fallback-Ledger):

* **`run`** — je Corpus-Fenster einmal restaurieren, dann messen:
  musik_mos + defektlast, die 15-Goal-Scorecard (Vertrag NEU:
  `pre_pipeline_reference` = Eingabe) und `transient_energie` **doppelt auf
  demselben Audio** (Vertrag NEU vs. Vertrag ALT = `alignment_tolerance_ms=0.0,
  exclude_impulse_onsets=False`, blind verlost, Schlüssel getrennt).
* **`compare`** — blind Paarvergleich zweier `run`-JSONs (alt vs. neu) je
  Fenster inkl. Sign-Test + Never-worsen, dazu der Scorecard-Diff je Goal
  gegen den Lauf 2026-11-03 (`BASELINE_2026_11_03`, Quelle im Skriptkopf).
* Da der Fix metrik-seitig ist (Pipeline bit-identisch, §G5), genügt für den
  Vertrags-Beleg **ein** `run` — der alte Vertrag wird auf demselben Audio
  mitgemessen; Never-worsen der übrigen 14 Goals folgt konstruktiv (unberührter
  Code-Weg) und wird über den Scorecard-Diff bestätigt.

## 7. Verifikation — ausstehend (Worker-Runtime ohne Shell)

Vom Manager nachzuziehen bzw. nachgezogen (siehe Abschnitt 8):

```
nice -n 19 python3 -B -m pytest tests/unit/test_transient_energy_metric.py tests/unit/test_transient_energy_messkontrakt.py -q
python3 -m ruff check backend/core/musical_goals/transient_energy_metric.py backend/core/musical_goals/musical_goals_metrics.py backend/core/unified_restorer_v3.py tests/unit/test_transient_energy_messkontrakt.py scripts/b3_transient_energie_neverworsen.py --select F821,F601,B009,I001
nice -n 19 python3 -B scripts/b3_transient_energie_neverworsen.py run --label neu
```

## 8. Manager-Nachtrag (2026-11-04, Integration)

* **Test-Schranke korrigiert** (`test_referenzvertrag_misst_gegen_pre_pipeline_original`):
  die Festzahl ≤ 0.6 war auf den alten Messkern kalibriert; die ±10-ms-Toleranz
  hebt bei geclusterten Anschlägen einzelne Fenster (gemessen 0.6393). Ersetzt
  durch den prinzipienten Korridor-Bound (< 0.70 aus
  `backend/core/calibration_matrix.py:1222`) + Gap-Assertion (≥ 0.25).
  Vertrags-Kernaussagen (≥ 0.9 gegen Pre-Pipeline-Original) unverändert.
* **Versionssync §v10.802:** `version.py` 10.12.47 + `pyproject.toml`,
  `README.md`, `CHANGELOG.md` nachgezogen (Versions-Konsistenz-Gate).
* Tests nach Ausführung: 22/22 grün.

## 9. Offene Punkte

1. **§1.4.6-Revision V27 (Onset-Set):** der Impuls-Filter weicht vom
   Spec-WORTLAUT Schritt 2 ab (dort: alle Onsets des Originals) — er folgt dem
   Spec-ZWECK + §V36. Die Aufnahme des Onset-Set-Vertrags in
   `.github/specs/01_musical_goals.md` ist eine **Spec-Änderung** und erfordert
   den PR-Evidenzblock (AGENTS.md §4) + Maintainer-Sign-off — bewusst offen.
2. **Pipeline-Baustein U4:** `phase_08_transient_preservation` läuft mit ≈ 1 %
   effektiver Wirkung (Stärke 0.089 × Wet/Dry 0.12, Strength-Floor-Gate
   überspringt die Phase, STOP_GRACEFUL). Ursachenkette Abschnitt 3.4 —
   Behebung ist ein eigener Pipeline-Fix mit Never-worsen-Beleg (§V7: keine
   blinden Stärke-Pflaster).
3. **Zielwert-Frage:** ob der Korridor 0.70 mit dem korrigierten Vertrag
   erreicht wird, entscheidet die Messung (Abschnitt 6); ein verbleibender
   Gap ist dann echter Signalfaden (U4) statt Messartefakt.
