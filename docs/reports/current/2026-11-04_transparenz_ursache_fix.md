# transparenz — Ursachen-Analyse, Messkontrakt-Fix, Never-worsen-Harness

**Datum:** 2026-11-04 · **Zweig:** `kun/transparenz-kluft-ursache-und-fix-gswhht`
**Auftrag:** Ebene-3-Kluft schließen — transparenz Score 0.52 vs. Zielkorridor
0.55 (GOAL_SCORECARD der B3-Fensterläufe 2026-11-03/04; Fensterwerte
0.505–0.519). Vorgehen wie bei transient_energie: **Ursachen-Analyse →
§V7-konformer Fix → Never-worsen-Beleg** — kein Blind-Justieren.

> **Umgebungshinweis (§V7 — keine erfundenen Zahlen):** Diese Worker-Runtime
> hat KEIN Shell-/Exec-Werkzeug. Es wurde NICHTS ausgeführt oder selbst
> gemessen. Alle Zahlen sind **Zitate aus Worktree-Dateien oder dem
> Auftragstext** (Quelle jeweils am Ort); alle Befehle in Abschnitt 7 sind als
> **ausstehend** dokumentiert. Die im Auftrag genannte Datei
> `docs/reports/current/b3_transient_energie_neu_vor_enforcer.json` liegt
> NICHT in diesem Worktree (wie im Transient-Report der Log); die im Auftrag
> genannten Fensterwerte sind stattdessen wörtlich in
> `docs/reports/current/b3_transient_energie_neu.json` belegt (Zitate unten).

## 1. Ergebnis in Kürze

Die Kluft 0.52 vs. 0.55 ist **überwiegend ein Messvertragsbruch**, kein
Signalfaden: `TransparenzMetric.measure()` blendete Scores von Eingaben < 12 s
mit einem **wall-clock Short-Form-Prior** zu 95 % (2,5-s-Fenster) auf 0.50 —
der Score maß die **Excerpt-Dauer**, nicht die Spektralklarheit. Auf dem
B3-Fensterplan (2,5 s) war der Wert konstruktiv auf **[0.475, 0.525]**
gedeckelt; der Zielkorridor 0.55 (shellac-Material-Ceiling) war dort
**mathematisch unerreichbar** (Maximalwert 0.525). Die gemessenen Fensterwerte
0.505–0.519 sind exakt die Blend-Signatur (Roherwerte 0.60–0.88, um den
Faktor 20 gestaucht).

Behoben metrik-seitig (Messkontrakt, §V7 — Metrik korrigieren statt Pipeline
verbiegen):

1. **Keine Dauer-Konflation mehr:** Neutral-Shrinkage nur noch bei fehlender
   Estimator-Unterstützung (< 3 STFT-Frames ≈ 0,13 s); ab 3 Frames liefert die
   Metrik den rohen §9.7.25-Score — PMGG-identisch
   (02_pipeline_architecture.md:4152).
2. **PMGG-Identität auch für Near-Silence:** 0.5 (unmessbar, neutral) statt
   0.0 („katastrophal") — zweiter, kleinerer Bruch derselben Vertragsklausel.
3. **Reproduzierbarer Alt-Pfad** (`short_form_prior=True`) für den
   A/B-Never-worsen-Beleg (B3-Muster).

Der Wohlklang-Ordnungs-Enforcer (`wohlklang_ordnung_gate.py`) ist **nicht**
Ursache der Kluft (Abschnitt 3.5) und wurde **bewusst nicht angefasst** —
Ebene-1/2-Schutz bleibt intakt (Hörordnung §2 (hoerordnung.instructions.md):
„Eine Entscheidung auf Ebene n darf niemals eine Invariante der Ebene < n
verletzen").

## 2. Goal-Definition — was transparenz misst

Normative Kette (AGENTS.md §1):

* `.github/instructions/musical_goals.instructions.md:22` (P4):
  `transparenz` — Boden ≥ 0.82 (Restoration) / ≥ 0.85 (Studio 2026).
* `.github/specs/01_musical_goals.md:76`: „transparenz | Durchhörbarkeit |
  Maskierungs-SMR (Bark) | Details hörbar"; `:109`:
  „**Transparenz** (`TransparenzMetric`) | Klarheit, Trennung der
  Klangelemente | **4** | ≥ **0.82** | ≥ **0.85**".
* Algorithmus: `.github/specs/01_musical_goals.md:845` (§9.7.13):
  „Multi-Band Spectral Crest (5 Oktavbänder 250 Hz–8 kHz)" — implementiert ist
  die PMGG-gleiche **§9.7.25-Spektral-Flatness** der Oktavband-Energien
  (geom/arith, `score = 1 − 2·flat`), wie
  `.github/specs/02_pipeline_architecture.md:4152` festschreibt:
  „`transparenz` | §9.7.25 Spektral-Flatness der Oktavband-Energien
  (250 Hz–8 kHz), **identisch-semantisch zu PMGG/finaler `TransparenzMetric`**".
* Implementierung: `backend/core/musical_goals/musical_goals_metrics.py`
  (`TransparenzMetric`, Klasse ab :2167): frame-gemitteltes STFT
  (4096/Hop 2048, ≤ 100 Frames), 5 Oktavbänder 250 Hz–8 kHz
  (§6.2c BW-adaptiv, `_bw_limited`), `score = 1 − 2·geom/arith`. PMGG-Zwilling:
  `backend/core/per_phase_musical_goals_gate.py:2944–2999` (`_measure_quick`,
  `scores["transparenz"]` wird ausschließlich dort gesetzt (:2967, :2994,
  :2996, :2998)).
* Scorecard-Emitter: `backend/core/unified_restorer_v3.py:17605–17633`
  (GOAL_SCORECARD-Zeile `goal:score=..,thr=..,gap=..,pass=..`; `thr` =
  `_effective_goal_thresholds`).
* **Zielkorridor 0.55:**
  `backend/core/calibration_matrix.py:1123–1134` —
  `_CHAIN_END_GOAL_CEILINGS["shellac"]["transparenz"] = 0.55` (:1132),
  angewandt als Material-Ceiling-Clamp der Scorecard-Schwelle
  (`unified_restorer_v3.py:17193–17201`, §0a Material-Ceiling:
  `_effective_goal_thresholds[_mc_goal] = _mc_ceil`). Belegt als
  Scorecard-Schwelle durch die GOAL_SCORECARD-Baseline 2026-11-03
  (`scripts/b3_transient_energie_neverworsen.py:52–58`, Log-Zitat
  „b3_wiederholung_W6_W7": „transparenz score=0.5226, thr=0.5500 (pass=0)");
  die Log-Datei selbst liegt nicht im Worktree.
* **Ebene (Hörordnung §2 (hoerordnung.instructions.md:45–55)):** „Die vier
  Ebenen — Jede Ebene dominiert die darunterliegende": 1. Hör-Invarianten,
  2. Audibility-Schicht, **3. Wohlklang-Ordnung** (§5: lexikografische
  Ziel-Hierarchie; dort Position 3 „Klarheit / Durchhörbarkeit
  (Maskierungs-SMR — transparenz, sep_fidelity)", :142), 4. Einladungs-Gate.
  Über transparenz stehen nur Ebene 1/2 (Unverhandelbares) sowie in der
  Wohlklang-Ordnung Natürlichkeit und Wärme (§5, :131–145) — „Verboten: Ein
  Brillanz-/Transparenz-Boost, der Wärme oder Natürlichkeit senkt".

## 3. Ursachenkette (file:line, statisch belegt)

### 3.1 U1 — Dauer-Konflation: wall-clock Short-Form-Blend (Messvertragsbruch)

Der Messpfad der Zielmetrik endete (Altfassung,
`musical_goals_metrics.py:2284–2289`, ersetzt 2026-11-04) mit:

```python
# Short-form reliability blend: bei < 8 s wenig Kontext für Energieverteilung.
_tdur_s = float(len(audio)) / float(sr + 1e-9)
if _tdur_s < 8.0:
    _reliability = float(np.clip((_tdur_s - 2.0) / 10.0, 0.0, 1.0))
    _neutral_prior = 0.50
    score = float(np.clip(_reliability * score + (1.0 - _reliability) * _neutral_prior, 0.0, 1.0))
```

Der Blend war damit eine Funktion der **Excerpt-Wall-Clock-Dauer**, nicht des
Spektralinhalt: reliability = (dauer − 2 s)/10 s, Neutral-Prior 0.50. Der
B3-Fensterplan misst auf **2,5-s-Fenstern** (Fensterplan IDENTISCH in
`scripts/b3_stufenmessung.py:213` [`--seconds` default 2.5] und `:227`
[Starts (0.0, 2.5, 5.0, 7.5)] sowie `scripts/b3_transient_energie_neverworsen.py:344–347`):

* 2,5 s → reliability = 0.05 → `score = 0.05·raw + 0.475` →
  **Score ∈ [0.475, 0.525] für JEDES Signal**; Maximalwert 0.525 < 0.55.
* Der Zielkorridor 0.55 (`calibration_matrix.py:1132`) war auf dem
  B3-Fensterplan damit **mathematisch unerreichbar** — selbst ein perfekter
  Roherwert 1.0 lieferte nur 0.525. Die Kluft 0.52 ↔ 0.55 ist daher ein
  Messartefakt, kein Signalverlust.
* Auch 10-s-Fenster wären noch zu 80 % gepriort (reliability 0.8); der Blend
  verschwindet erst ab 12 s.

**Gemessene Evidenz (Worktree-Datei, keine erfundene Zahl):** die sechs
Fenster-Scorecards in `docs/reports/current/b3_transient_energie_neu.json`
(Extrakt der Fensterwerte des Auftrags) — bei gleichzeitig hohen übrigen
Goals (natuerlichkeit 0.883–0.972, authentizitaet 0.928–0.988,
emotionalitaet 0.904–0.972):

| Fenster (trial_id, Zitat-Zeile) | Material | transparenz gemeldet | Roherwert r = (x − 0.475)/0.05 |
|---|---|---|---|
| `cassette_1980s_wow__w0` (:27) | vinyl | 0.5050533970111816 | 0.601 |
| `cassette_1980s_wow__w1` (:62) | shellac | 0.5190634054457093 | 0.881 |
| `cassette_1980s_wow__w2` (:97) | shellac | 0.5190710599091442 | 0.881 |
| `cassette_1980s_wow__w3` (:132) | shellac | 0.5179427410331588 | 0.859 |
| `reel_1940s_dropout__w4` (:167) | tape | 0.5187862233469377 | 0.876 |
| `jazz_1950s_scratched__w5` (:202) | tape | 0.5160348435282238 | 0.821 |

Die Blend-Inversion (raw = 20·x − 9.5) ergibt für alle sechs Fenster plausible
Roherwerte 0.60–0.88 — der Streubereich 0.28 der Roherwerte erscheint als
Streubereich 0.014 der gemeldeten Werte (20×-Stauchung). Das ist exakt die
Vorhersage des Blends und die Signatur des Vertragsbruchs. (Die Inversion
gilt exakt für `dauer = 2.500 s`; ±0.05 s Ausgabelänge verschieben raw um
≤ ±0.02 — die Aussage bleibt.) Auch die GOAL_SCORECARD-Baseline 0.5226
(`scripts/b3_transient_energie_neverworsen.py:56`, thr=0.5500) liegt in der
Blend-Signatur (raw ≈ 0.952).

### 3.2 U2 — PMGG-Identitätsbruch (dieselbe Vertragsklausel)

`02_pipeline_architecture.md:4152` verlangt „identisch-semantisch zu
PMGG/finaler `TransparenzMetric`“; der Code der Metrik selbst beansprucht
„Gleicher Algorithmus in `TransparenzMetric.measure()` → PMGG-Konsistenz“
(`musical_goals_metrics.py`, Kommentar Block §9.7.25). Der PMGG-Proxy
`_measure_quick` (`per_phase_musical_goals_gate.py:2944–2999`) hat jedoch
**keinen** Short-Form-Blend und denselben Frame-gemittelten STFT-Kern
(`:2273–2293`, 4096/Hop 2048, ≤ 200 Frames). Auf Eingaben < 12 s divergierten
damit finale Metrik und PMGG-Proxy systematisch — genau in der Fensterklasse
des B3-Messplans. Für ≥ 3 Frames ist der Roherwert beider rechnerisch
identisch (gleiche Frames, gleiche Band-Energien, gleiche Formel) — belegt
und gepinnt durch `test_pmgg_identitaet_auf_b3_fenstern`.

### 3.3 U3 — Near-Silence → 0.0 statt 0.5 (sekundär, gleiche Klausel)

PMGG kennzeichnet Near-Silence als unmessbar-neutral:
`per_phase_musical_goals_gate.py:2967` (`_rms_trp < 1e-5 → 0.5`). Die finale
Metrik scorete Stille als Roherwert 0.0 (Flatness 1.0 → `clip(1 − 2)`), der
Blend stempelte daraus 0.475 — „0.0 = katastrophal gescheitert“-Semantik für
unmessbares Material (§v10.732-Analogie,
`unified_restorer_v3.py:17425–17432`). Nicht ursächlich für die B3-Kluft
(Musik-Fenster), aber derselbe Vertragsbruch — mit dem Fix behoben (0.5).

### 3.4 Checkliste geprüft und ausgeschlossen

* **Referenz-Kopplung (§0d-Shift):** ausgeschlossen — transparenz ist
  reference-free by design (`TransparenzMetric.measure()`: `_ = reference,
  material_type`; Docstring „crest-factor is reference-free by design —
  symmetric before/after“). Der §0d-Referenz-Shift
  (`unified_restorer_v3.py:17341–17407`, `_mg_ref = best_carrier_checkpoint`)
  kann diesen Score nicht beeinflussen; gepinnt durch
  `test_referenz_wird_laut_vertrag_ignoriert`.
* **Onset-/Segment-Sets:** nicht anwendbar (die Metrik hat keine Onsets).
* **Cache-Keys:** kein Bruch — der measure_all-Cache (§v10.98,
  `musical_goals_metrics.py` `measure_all`, Cache-Key-Block) enthält
  audio-/reference-/pre_pipeline_reference-Hashes, sr, material_type,
  panns_singing, global_scalar; alle Einflussgrößen von transparenz (audio,
  sr) sind abgebildet.
* **100-Frame-STFT-Cap (~4,3 s):** für 2,5-s-Fenster irrelevant (ganzes
  Fenster, ~56 Frames werden analysiert); bewusst unverändert (PMGG-
  Ausrichtung „analysiert den Anfang“, Kommentar in der Metrik).
* **Signalfaden (NR/De-Reverb/Compression):** für die Kluft **nicht mehr
  erforderlich** — die Blend-Arithmetik erklärt den Wertebereich vollständig
  und macht den Zielkorridor unerreichbar. Die Roherwerte 0.60–0.88 deuten
  auf vorhandene spektrale Klarheit im Restaurat; ob sie das physikalisch
  Erreichbare (Material-Ceilings) sind, entscheidet die Messung nach dem Fix
  + die Hör-Instanz (Hörordnung §1: „Metriken sind Zeugen, nicht Richter“).
  Bewusst KEINE Pipeline-Justierung (§V7).

### 3.5 Enforcer-Analyse — kappt `wohlklang_ordnung_gate.py` transparenz-Gewinne?

**Nein als Kluft-Ursache; Ja nur in dem vom Hörordnungs-Vertrag geforderten
Fall — und der bleibt unangetastet:**

* Die gemessenen Fensterwerte stammen aus einem Lauf **vor** dem Enforcer
  (Dateiname des Auftrags „vor_enforcer“; der Enforcer wurde 2026-11-04 aktiv,
  `unified_restorer_v3.py:17440–17448`, „Durchsetzung 2026-11-04“). An der
  0.52-Kluft war er nicht beteiligt.
* `WohlklangOrdnungGate.evaluate` (`wohlklang_ordnung_gate.py:186–273`)
  meldet nur **Verletzungen**: „Verbesserung eines NIEDERRANGIGEN Goals auf
  Kosten eines HÖHERRANGIGEN ist verboten“ (Ties erlaubt). Ein transparenz-
  Gewinn OHNE Kosten für Ebene 1/2 (Natürlichkeit/Wärme/…) ist per Konstruktion
  `PASS` → `enforce()` liefert `ACCEPT`, blend_factor 1.0, Kandidat
  bit-identisch (`:375–383`).
* `WohlklangOrdnungEnforcer.enforce` (`:333–452`) kappt nur bei VIOLATION —
  mit festen Blend-Stufen `DEFAULT_BLEND_ALPHAS = (0.5, 0.25, 0.125, 0.0625)`
  (`:70`) oder Rollback; jede Stufe wird gegen das Gate + Never-worsen
  verifiziert („nie gegen Ebene 1“). Das ist exakt die geforderte
  Hörordnungs-Semantik (§2/§5): ein transparenz-Boost darf Wärme/Natürlichkeit
  nicht senken — solche Gewinne **müssen** gekappt werden. Der Auftrag verbietet
  das Unterlaufen des Enforcers; er wurde daher nicht angefasst.
* **Nebenbefund (im Fix indirekt adressiert):** Auf < 8-s-Fenstern machte der
  Blend die Enforcer-Blend-Probes für transparenz blind (Deltas 20×
  komprimiert, `unified_restorer_v3.py:17486–17505` `_wo_blend_probe` misst
  via `measure_all`). Mit dem Messkontrakt-Fix werden die Probes wieder
  diskriminierend — der Enforcer sieht transparenz wieder, in beide
  Richtungen. Der Enforcement-Code selbst bleibt unverändert.

## 4. Fix (§V7 — Ursache statt Symptom) und Begründung

Auftragsregel: „Bei Vertragsbruch: Metrik korrigieren statt Pipeline
verbiegen — offenlegen im Report.“ Die Analyse zeigt genau das (U1–U3);
daher Metrik-Fix, keine Pipeline-Verbiegung.

### 4.1 Geänderte Dateien

* `backend/core/musical_goals/musical_goals_metrics.py` —
  `TransparenzMetric.measure()`:
  1. **Keine Dauer-Konflation mehr:** der wall-clock Short-Form-Blend ist
     ersetzt durch eine **support-basierte** Neutral-Shrinkage:
     `reliability = clip(n_support_frames / 3, 0, 1)` — nur Eingaben mit
     < 3 STFT-Frames (≈ < 0,13 s @ 48 kHz) shrinken zum Neutral-Prior 0.50;
     ab 3 Frames liefert die Metrik den rohen §9.7.25-Score (PMGG-identisch).
     Für den B3-Fensterplan (2,5 s ≈ 56 Frames) bedeutet das volle
     Auflösung — der Korridor 0.55 ist wieder erreichbar.
  2. **Near-Silence → 0.5** (PMGG-Identität, `_measure_quick`-Guard) statt
     Roherwert 0.0.
  3. **`short_form_prior: bool = False`** (Mess-API): True reproduziert
     bit-genau den Alt-Pfad inkl. Blend — ausschließlich für den
     reproduzierbaren A/B-Never-worsen-Beleg (Muster wie die Parameter
     `alignment_tolerance_ms`/`exclude_impulse_onsets` im transient-Fix).
* NEU `tests/unit/test_transparenz_messkontrakt.py` — Prove-It-Tests:
  Alt-Pfad zeigt den Defekt (Deckel 0.525, Blend-Identität
  `alt = 0.05·neu + 0.475`), Neuer Vertrag besteht (Korridor erreichbar,
  keine Dauer-Konflation, Diskriminierung, PMGG-Identität zu `_measure_quick`,
  Near-Silence 0.5, Referenz-Freiheit, Support-Guard, Determinismus §G5).
* NEU `scripts/b3_transparenz_neverworsen.py` — Never-worsen-Harness
  (B3-Muster; importiert Fensterplan/Restore/Bewertung/Baseline aus
  `scripts/b3_transient_energie_neverworsen.py`, das **unverändert** bleibt).
* `.github/FILE_REGISTRY.md` — Einträge für beide neuen Dateien (Write-Gate
  `aurik-file-lifecycle`; `repo_search.py --before-create` siehe Abschnitt 7).
* `backend/core/version.py` — 10.12.47 → 10.12.48 (§v10.802
  (copilot-instructions.md): fix ⇒ Patch-Bump; Versionssync
  `pyproject.toml`/`README.md`/`CHANGELOG.md` siehe Abschnitt 7).
* **Nicht angefasst:** `backend/core/musical_goals/transient_energy_metric.py`,
  `scripts/b3_transient_energie_neverworsen.py`, `docs/SOTA_HANDOVER.md`
  (Auftragsvorgabe); `backend/core/wohlklang_ordnung_gate.py`,
  `backend/core/unified_restorer_v3.py`, `backend/core/calibration_matrix.py`,
  `backend/core/per_phase_musical_goals_gate.py` (Ursache lag in der Metrik).

### 4.2 Konformität

* **§V7 (copilot-instructions.md):** Ursache (Messvertrag) statt Symptom;
  KEINE phasen-individuellen Schwellwerte; keine Pipeline-Justierung; die
  Shrinkage skaliert mit dem Estimator-Support (Sachverhalt), nicht mit
  Wall-Clock-Zeit (Symptom).
* **§G5 (copilot-instructions.md):** keine Zeit-/Zufallsabhängigkeit in der
  Messlogik; alle Konstanten fest; Determinismus per Test gepinnt.
* **§v10.802 (copilot-instructions.md):** Patch-Bump; **keine Flag-Flips** —
  `short_form_prior` ist Mess-API mit spec-konformem Default (Default =
  neuer Vertrag); kein bestehender Verhaltensschalter wurde umgelegt.
* **Ebene-1/2-Schutz (Hörordnung §2 (hoerordnung.instructions.md)):**
  unangetastet — der Wohlklang-Ordnungs-Enforcer bleibt unverändert und darf
  nicht unterlaufen werden (Abschnitt 3.5); die Metrik ändert keine
  Prioritäten, nur die Messgenauigkeit.
* **Spec-Offenlegung:** Der Alt-Blend hatte **keine** Spec-Grundlage —
  §9.7.13/§9.7.25 (01_musical_goals.md) spezifizieren keine Dauer-Skalierung;
  die PMGG-Identität (02_pipeline_architecture.md:4152) ist die normative
  Basis des Fixes. Vorrangregel „Spec > Code > Kommentar“ (AGENTS.md §1).
  **Keine Spec-Änderung** in diesem Fix (AGENTS.md §4: Specs lösen den
  PR-Evidenzblock-Zwang aus) — die Spec-Konsolidierung §9.7.13 „Crest“ vs.
  implementierte §9.7.25-Flatness ist offen (Abschnitt 8).

## 5. Erwartete Scorecard-Wirkung (qualitativ — keine erfundenen Zahlen)

* `transparenz` wird auf Fenstern/Excerpts jeder Länge ≥ 0,13 s künftig als
  roher §9.7.25-Score gemessen. Die Blend-Inversion der letzten Läufe
  (Roherwerte 0.60–0.88, Abschnitt 3.1) ist der ehrlichste verfügbare
  Schätzer der post-fix-Werte — aber **nicht behauptet**: ob und welche
  Fenster den Korridor 0.55 erreichen, entscheidet die Messung (Abschnitt 6).
* Für Song-Länge (≥ 12 s) ist der Messpfad unverändert (Alt-Blend dort mit
  reliability 1.0 inaktiv; Near-Silence trifft Musik nicht) — Song-Scorecards
  sind damit praktisch betragsgleich; die Pipeline-Ausgabe bleibt in diesem
  Regime bit-identisch (§G5).
* Für Excerpts < 12 s (B3-Fenster, Short-Excerpt-Feasibility,
  Enforcer-Probes) werden transparenz-Werte und -Deltas wieder
  diskriminierend (20× größere Auflösung). Das kann WIRKLICHKEIT sichtbar
  machen, die vorher im Prior verborgen war — beabsichtigt und über den
  Never-worsen-Beleg abzusichern.
* Die übrigen 14 Goals sind im Code-Weg unverändert (nur `TransparenzMetric`
  geändert; Dispatch/Cache in `measure_all` unberührt) ⇒ keine
  Verschlechterung erkennbar; der Scorecard-Diff des Harness' bestätigt das
  empirisch.

## 6. Never-worsen-Messung (Harness)

`scripts/b3_transparenz_neverworsen.py` (B3-Muster: Seed 42, Fensterplan,
richtungszertifizierte Metriken musik_mos + defektlast, exakter
Vorzeichen-Test, Never-worsen/JND 0,05; Fensterplan/Restore/Bewertung
**importiert** aus `scripts/b3_transient_energie_neverworsen.py`, damit der
Beleg exakt dieselben Fenster und denselben Produktionspfad misst):

* **`run`** — je Corpus-Fenster einmal restaurieren, dann messen:
  musik_mos + defektlast, die 15-Goal-Scorecard (Vertrag NEU) und
  `transparenz` **doppelt auf demselben Audio** (Vertrag NEU vs. Vertrag ALT =
  `short_form_prior=True`, blind verlost, Schlüssel getrennt).
* **`compare`** — blind Paarvergleich gegen einen Alt-Lauf (z. B.
  `b3_transient_energie_neu.json` = vor dem Fix) inkl. Vorzeichen-Test +
  Never-worsen (musik_mos/defektlast), Scorecard-Diff je Goal gegen Alt-Lauf
  **und** `BASELINE_2026_11_03` (transparenz 0.5226/thr 0.5500), dazu der
  Kontrakt-Beweis (Blend-Treue `ALT = 0.05·NEU + 0.475` je Fenster — der Fix
  beweist sich als Messkontrakt-Änderung auf demselben Audio) und die
  Prüfkriterien `korridor_0_55_erreichbar`, `transparenz_verbessert`,
  `never_worsen` (fail-closed; `ergebnis` = ERFUELLT/ABGELEHNT).
* Da der Fix metrik-seitig ist (Pipeline im Song-Regime bit-identisch, §G5),
  genügt **ein** `run` für den Kontrakt-Beweis; Never-worsen der übrigen
  14 Goals folgt konstruktiv (unberührter Code-Weg) und wird über den
  Scorecard-Diff gegen Alt-Lauf + Baseline bestätigt.

## 7. Verifikation — ausstehend (Worker-Runtime ohne Shell)

Vom Manager nachzuziehen (in dieser Reihenfolge):

```
# 1. Write-Gate für die neuen Dateien (AGENTS.md §3; nachzuholen, da Worker ohne Shell)
python3 scripts/repo_search.py --before-create tests/unit/test_transparenz_messkontrakt.py
python3 scripts/repo_search.py --before-create scripts/b3_transparenz_neverworsen.py

# 2. Tests (statisch vorbereitet, NICHT ausgeführt)
nice -n 19 python3 -B -m pytest tests/unit/test_transparenz_messkontrakt.py tests/unit/test_transient_energy_messkontrakt.py -q

# 3. Kritischer statischer Gate (identisch zum Pre-Commit-Gate)
python3 -m ruff check backend/core/musical_goals/musical_goals_metrics.py tests/unit/test_transparenz_messkontrakt.py scripts/b3_transparenz_neverworsen.py --select F821,F601,B009,I001

# 4. Versionssync §v10.802 (Versions-Konsistenz-Gate):
#    backend/core/version.py = 10.12.48 → pyproject.toml, README.md, CHANGELOG.md nachziehen

# 5. Never-worsen-Beleg (AUSFÜHRUNG durch den Manager)
nice -n 19 python3 -B scripts/b3_transparenz_neverworsen.py run --label nachher
nice -n 19 python3 -B scripts/b3_transparenz_neverworsen.py compare \
    --alt docs/reports/current/b3_transient_energie_neu.json \
    --neu docs/reports/current/b3_transparenz_nachher.json
```

Erwartete Test-Aussage (Prove-It, noch NICHT gemessen):
`test_alt_vertrag_deckelt_b3_fenster_auf_0_525` belegt den Alt-Defekt
(Alt-Wert ≤ 0.525); die Neu-Vertrags-Tests (Korridor ≥ 0.8 auf 2,5 s,
PMGG-Identität ≤ 1e-6, Dauer-Invarianz ≤ 0.05) bestehen nur mit dem Fix.

## 8. Offene Punkte

1. **Zahlen-Herkunft:** „Score 0.52“/„0.5226“ und „Fensterwerte 0.505–0.519“
   stammen aus Auftrag/GOAL_SCORECARD-Log bzw. sind wörtlich in
   `docs/reports/current/b3_transient_energie_neu.json` belegt (Zitat-Tabelle
   Abschnitt 3.1). Die im Auftrag genannte Datei
   `b3_transient_energie_neu_vor_enforcer.json` liegt nicht im Worktree.
2. **Weitere Short-Form-Blends:** andere Metriken in
   `musical_goals_metrics.py` nutzen ähnliche dauer-basierte Prior-Blends
   (z. B. emotionalitaet, Blend `(dur−2)/10` mit Priors 0.90/0.97). Auf
   < 8-s-Fenstern haben dieselbe Dauer-Konflation — nicht Teil dieser
   Ziel-Kluft und bewusst nicht mitgeändert (Kleinste-Kohärente-Änderung);
   separater Messkontrakt-Check empfohlen.
3. **Signalfaden-Rest:** ob die Roherwerte 0.60–0.88 das physikalisch
   Erreichbare für die Materialien sind (Material-Ceilings shellac 0.55,
   tape 0.74, vinyl 0.78 in `calibration_matrix.py:1123–1211`), entscheidet
   die Messung nach dem Fix + die Hör-Instanz (Hörordnung §1: „Metriken sind
   Zeugen, nicht Richter“). Ein verbleibender Gap wäre dann echter
   Signalfaden — mit bekannter Ursachenkette.
4. **Spec-Konsolidierung §9.7.13 vs. §9.7.25:** 01_musical_goals.md:845 nennt
   „Multi-Band Spectral Crest“, implementiert und in
   02_pipeline_architecture.md:4152 normiert ist die §9.7.25-Spektral-
   Flatness. Eine Klarstellung in 01_musical_goals.md ist **Spec-Arbeit**
   (PR-Evidenzblock + Maintainer-Sign-off, AGENTS.md §4) — bewusst offen.

## 9. Conventional-Commit-Vorschlag (Manager führt aus)

```
fix(musical_goals): transparenz-Messkontrakt — Dauer-Konflation entfernen (PMGG-Identität)

- TransparenzMetric.measure(): wall-clock Short-Form-Blend (score =
  ((dauer-2s)/10s)·raw + …·0.50 für < 8 s) ersetzt durch support-basierte
  Neutral-Shrinkage (< 3 STFT-Frames); B3-Fenster (2,5 s) waren auf
  [0.475, 0.525] gestaucht, Zielkorridor 0.55 (calibration_matrix.py:1132)
  unerreichbar (gemessen 0.505-0.519 bei Roherwerten 0.60-0.88)
- Near-Silence → 0.5 (PMGG-Identität _measure_quick) statt 0.0
- short_form_prior=True reproduziert den Alt-Pfad (A/B-Never-worsen)
- tests/unit/test_transparenz_messkontrakt.py (Prove-It inkl. Alt-Pfad)
- scripts/b3_transparenz_neverworsen.py (B3-Never-worsen-Harness)
- version 10.12.47 → 10.12.48
```

Keine Spec-Änderung ⇒ kein PR-Evidenzblock-Zwang (AGENTS.md §4); Pre-Commit-
Gates (FILE_REGISTRY, repo_graph --duplicates, unit-smoke) greifen über die
Einträge aus Abschnitt 4.1.
