# TODO-P0-1 · Slice B — Analytik + End-Gate von per-Chunk auf Song-Ebene (HPI/EmotionalArc/VQI)

**Datum:** 2026-11-05 · **Branch:** `kun/p0-1-song-ebene-analytik-xbr74g` (Basis `main`)
**Scope:** Ausführung von TODO-P0-1 laut Dispatch „Analytik + End-Gate von per-Chunk auf Song-Ebene heben".
**Status der Umsetzung:** implementiert (Code + Tests). **Ausführung von pytest/ruff/Messläufen ist NICHT erfolgt** — diese Runtime hat kein Shell-/Exec-Werkzeug. Alle Befehle in §5/§6 sind für den Manager vorbereitet und ausstehend.

> **Vorbemerkung zur Beweislage (§V6 (copilot-instructions.md), Silent-Failure-Verbot — hier: kein stiller Widerspruch):**
> Der Dispatch beschreibt einen Repository-Stand, der in diesem Worktree nicht vorliegt (siehe §1).
> Die Ziel-Blöcke GOAL_SCORECARD/End-Gate/Einladungs-Gate waren bereits song-global (Stand 2026-09-08,
> docs/TODOS_SOTA_ROADMAP.md:264 „Status 2026-09-08: VOLLSTÄNDIG UMGESETZT (Slice A + Blöcke a–d)").
> Die im dispatch-Ziel **zusätzlich** genannten song-globalen Größen **HPI, EmotionalArc, VQI**
> liefen dagegen weiterhin je Chunk — genau diese Lücke schließt Slice B.

---

## 1. Dispatch ↔ Worktree-Stand: belegte Abweichungen (nichts davon ist „erfunden")

| Dispatch-Angabe | Befund in diesem Worktree | Beleg |
| --- | --- | --- |
| Report `docs/reports/current/2026-10-28_f_reihe_laufzeit_optimierung.md` §9 (32,19× RT; Wall 7248,7–7259,2 s / 225,33 s) | **existiert nicht** (kein `docs/reports/current/*f_reihe*`, kein `*laufzeit*`) | `glob docs/reports/current/*.md` |
| `RT_REALITY_MEASURED = 32,2` in `denker/strategie_denker.py` | dort steht **`RT_REALITY_MEASURED: float = 53.0`** | denker/strategie_denker.py:50 |
| Muster `scripts/b3_transient_energie_neverworsen.py` | **existiert nicht** (kein Match `scripts/*neverworsen*`, `scripts/b3*`, `scripts/*transient*`) | glob |
| TODO-P0-1 „ab ~Zeile 464" | Abschnitt beginnt bei **docs/TODOS_SOTA_ROADMAP.md:261** und trägt bereits den Status „VOLLSTÄNDIG UMGESETZT (Slice A + Blöcke a–d)" (2026-09-08) | docs/TODOS_SOTA_ROADMAP.md:261–264 |
| „Projektion 15–25× RT" | normativ als **veraltet** gekennzeichnet (s. u.) | .github/copilot-instructions.md:355–365 |

**Normative Einordnung** (`.github/copilot-instructions.md:355–365`, wörtlich):
> „Die frühere Zuordnung ‚die Diskrepanz stammt aus per-Chunk statt per-Song laufender Analytik/End-Gate' ist **überholt**: TODO-P0-1 ist umgesetzt — die song-globalen Blöcke (GOAL_SCORECARD, End-Gate-Recovery-Kaskade, Einladungs-Gate, Reporting-Analytik, B2-Post-Scan) laufen nach der Chunk-Assembly genau einmal. … **die Kalibrierung zeigt, dass die Analytik nicht der Treiber war: die Zeit liegt in Defekt-Scanner, Phase-Pipeline und Einzelphasen.**"

**Gemessene Ausgangslage in DIESEM Worktree** (einzige hier belegbare Messung,
`docs/reports/current/2026-10-07_leistungskalibrierung.md` §Messung/§Zahlen/§Nachher-Messung):
30-s-Fenster der Quelle `test_audio/Elke Best - Hey Kleiner … .wav`, Zelle `balanced`:
73,3× RT → **60,1× RT Wall / 52,5× RT Engine** (nach Teilbehebung D-K3-39),
Kommando dort: `.venv_aurik/bin/python scripts/benchmark_effizienz_matrix.py --input "…" --seconds 30 --cells balanced --no-wav --enforce-budget --profile-top-phases 5`.
Die im Dispatch genannten **32,19× RT** werden hier **nicht als eigene Messung behauptet**; sie werden als
Manager-Stand notiert und für den RT-Vergleich in §6 als Baseline vorgemerkt, sobald die Messung in der
Ziel-Branch vorliegt.

Konsequenz für die Wirkung (§4): **keine RT-Zahl wird zugesagt** (§V7 (copilot-instructions.md) — Ursache
statt Symptom; erfundene Zahlen sind verboten). Slice B ist eine **Strukturaufgabe** (Qualitätskorrektur
der song-globalen Größen + Entfernen der letzten „measure_all je Chunk"-Reste); der Messlauf nach dem
Umbau entscheidet.

---

## 2. Analyse: was läuft wo (file:line, am Code belegt; Zeilen Stand nach Slice B)

### 2.1 Song-Assembly-Punkt

- Chunk-Loop + Assembly: `_restore_chunked`, **backend/core/unified_restorer_v3.py:46983** (def),
  Assembly-Abschluss mit `cp.verify_crossfade(results, sample_rate)` **:47415**.
- Per-Chunk-Restaurierung: `self.restore(chunk, …, _chunked_tail_skip=True, _chunked_last=…)` je Chunk
  (Kwargs-Muster, Vorbereitung `_chunk_kwargs`).
- Song-Level-Tail nach Assembly: GOAL_SCORECARD **:47417–47436** (Block a1), End-Gate-Kaskade
  `_run_song_level_end_gate` **:45994** (def) / Aufruf im Tail, m1b (§P1-7), OneTakeExport (§P1-8),
  Einladungs-Gate `_run_inviting_gate_measure` **:46779** (def) / Song-Aufruf **:47838**.

### 2.2 Song-global per Spec — Umsetzungsstand

| Größe | Spec-Quelle | lief vor Slice B | läuft nach Slice B |
| --- | --- | --- | --- |
| GOAL_SCORECARD (`measure_all`) | TODO-P0-1 Block a1; 01_musical_goals.md | song-global nach Assembly + Skip je Chunk (`_measure_goals_for_tail` **:46866**) | unverändert ✓ |
| End-Gate-Wiederherstellung (Recovery-Kaskade) | TODO-P0-1 Block a2; 02_pipeline_architecture.md | song-global `_run_song_level_end_gate` **:45994**, Safety-Netz letzter Chunk (Aufruf **:17477** via `_should_run_end_gate_cascade` **:210**) | unverändert ✓ |
| Einladungs-Gate | Hörordnung §6 (.github/instructions/hoerordnung.instructions.md) | song-global **:47838** + letzter Chunk | unverändert ✓ |
| **HPI (§2.44 HolisticPerceptualGate, „final export gate")** | 01_musical_goals.md; .github/instructions/pipeline.instructions.md §2.44 | **JE CHUNK** (Eval + finale Re-Eval + Rollback/Fail-Closed, restore()-Tail :19574–20430) | **einmal song-weit** (Song-Level-Block **:47528**, Teil 3) |
| **EmotionalArc (§8.2/§2.45b)** | pipeline.instructions.md §2.45b („song-globale Größe") | **JE CHUNK** (Messung+Rollback :17649ff., Post-MDEM-Messung+Makro-Korrektur :18339ff.) | **einmal song-weit** (Teil 1) |
| **VQI + Singer-ID (§2.35c/§0p)** | phases.instructions.md §2.35c; §0p Vocal-Supremacy | **JE CHUNK** (compute_vqi inkl. Resemblyzer + Singer-ID-/VQI-Rollbacks :19907–20147) | **einmal song-weit** (Teil 2, inkl. Ebene-1-Snapshot-Rollback) |
| `measure_all` Pre/Post-Excellence (§8.1 Regressions-Schutz) | — (lokal am ExcellenceOptimizer) | **2× JE CHUNK** (:16475–16527) | nur letzter Chunk + Ganzsong-Pfad (Messung deferriert, :16481) |

### 2.3 Bewusst pro Chunk (lokal-stationär) — unverändert

Stärke-Envelope, Phasen-Loop, FC-Iterationen, PMGG, MDEM, ExcellenceOptimizer,
Strength-Entscheidungen zentral über `global_scalar` (§V7 (copilot-instructions.md)).

### 2.4 Weitere song-global-semantische Reste (NICHT in Slice B — siehe §7)

PQS/§2.14 (läuft auf dem **letzten Chunk**, §v10.735 (copilot-instructions.md) :16914),
`_collect_reporting_analytics` (Block c „nur letzter Chunk", :18861), B2-Post-Scan (Block d, :18944),
§2.49-Export-Gate/„final export audio gate" (:22292ff.), SingMOS (§G4 (GEBOTE.md, Kategorie ...)),
PhonemeConsistency, MusicalCoherenceGuard (§2.78).

---

## 3. Umbau-Design (Slice B)

Prinzip: **Per-Chunk-Deferral im etablierten §v10.735/Block-a1-Muster** (Skip auf Nicht-Letzten-Chunks mit
begründetem Log-Event, deutsche Logs via `logging.getLogger(__name__)` (§G8/§V6 (copilot-instructions.md)))
**+ ein kompakter Song-Level-Block in `_restore_chunked` NACH der Assembly** (Z. 47528–47803), der die
drei Größen einmal song-weit bewertet. Keine Re-Indentierung ganzer Sektionen; chirurgische Guards.

1. **E1 — §8.1 (:16481):** `measure_all`-Snapshot + Re-Verifikation laufen nicht mehr auf Nicht-Letzten-Chunks
   (ExcellenceOptimizer selbst bleibt je Chunk — lokal-stationär). Die Re-Verifikation ist bereits durch
   `if _pre_excellence_mg_scores:` konsum-sicher geskippt.
2. **E3 — §8.2 (:17649):** EmotionalArcPreservationMetric-Messung auf Nicht-Letzten-Chunks deferriert
   (`_arc_result = None`, Kette mit `is not None`-Guards); der Chunk-Rollback war bereits durch §v10.735 gesperrt (:17789).
3. **E4 — §8.2 post-MDEM (:18339):** Messung + Makro-Korrektur deferriert (`_arc_post_mdem = None`,
   bestehende None-Guards überspringen Korrektur/Nachmessung).
4. **E5/E8 — §2.44 HPI-Eval + finale Re-Eval (:19574/:20380):** im Chunked-Pfad übersprungen
   („Song-Gate läuft einmal pro Song statt pro Chunk"); Ketten auf `is not None` umgestellt.
5. **E6 — fail-closed-Landmine (:19875):** `if _hpi_result is None and not _chunked_tail_skip:` — ohne diesen
   Guard hätte jeder Chunk wegen des deferrierten Gates auf das **Originalsignal** zurückgesetzt
   („Primum non nocere" darf hier nicht fehlzünden).
6. **E7 — §2.35c VQI (:19915):** Bedingung um `not _chunked_tail_skip` erweitert; VQI-Messung inkl.
   Singer-ID (Resemblyzer) + Rollbacks laufen nicht mehr je Chunk.
7. **Song-Level-Block (:47528):** (1) EmotionalArc song-weit mit **derselben zentralen Stärke-Policy** wie
   §8.2 (gain/damping aus `arc_preserved` + Restorability + Mode — keine neuen Schwellwerte, §V7);
   (2) VQI + Singer-ID song-weit mit den **zentralen Schwellen 0.92 bzw. `get_vqi_material_floor`**
   (identisch zum Per-Chunk-Pfad) und **Ebene-1-Snapshot-Rollback nur nach Verifikation**
   (Snapshot gewinnt nur bei `_rb_vqi > _vqi oder Singer-ID wieder ≥ 0.92`; Analogie §v10.14
   (copilot-instructions.md) Post-Processing-Checkpoint); (3) HPI-Export-Gate song-weit als
   Validierung + Metadaten (`metadata["p0_1_song_perceptual_gates"]`), Recovery bleibt zentral bei
   End-Gate (a2) + m1b + §0c-Vertrag (copilot-instructions.md, „bestmöglich exportieren");
   (4) **§P1-1 Wohlklang-Ordnungs-Audit song-weit** (`wohlklang_ordnung_gate.py` bewertet das finale
   Lied gegen den Input) — der Enforcer wird **ergänzt**, nicht ersetzt; §P1-1 im Tail (:17421) und der
   FC-GPP-Callback (:15579) bleiben unangetastet.
8. **Result-Verdrahtung:** `_meta["p0_1_song_perceptual_gates"]` + `emotional_arc`-Feld des
   `RestorationResult` aus dem Song-Level-Ergebnis (statt Chunk-0-Propagation).

### Konformität

- **§V7 (copilot-instructions.md), Workaround-Verbot:** keine phasen-individuellen Schwellwerte —
  0.92/Material-Floor/gain-damping-Policy sind unverändert aus dem zentralen Pfad übernommen.
- **§G5 (GEBOTE.md, Determinismus):** Song-Level-Block enthält keinen Zeit-/Zufallsbezug
  (per Test `test_song_level_block_is_deterministic_in_its_decisions` gepinnt).
- **§V8/§G1 (copilot-instructions.md, Song-Isolation):** keine neuen Stateful-Module; alles lokal im
  Aufruf; `compute_vqi`/`evaluate_*` sind zustandslos bezüglich Song-Wechsel.
- **§0a (copilot-instructions.md, verbotene Phasen + NaN/Inf):** keine neue Phase; EAP-Korrektur-Zeile
  spiegelt `np.nan_to_num`+Clip des Per-Chunk-Pfads.
- **§V6 (copilot-instructions.md):** jeder Skip/Deferral mit begründetem Log-Event; jeder Song-Level-Teil
  non-blocking mit `logger.warning` bei Nicht-Verfügbarkeit.
- **§v10.802 (copilot-instructions.md), Version-Sync:** `perf`-Commit ⇒ Patch-Bump in `version.py` im
  selben Merge (vom Manager auszuführen; hier nicht angefasst, da Messnachweis ausstehend).
- **Hörordnung §2 (.github/instructions/hoerordnung.instructions.md), Ebene 1/2:** Ebene-1-Schutz über
  Song-Level-VQI/Singer-ID-Gate + verifizierten Snapshot-Rollback; Wohlklang-Enforcer song-weit ergänzt;
  WO-Audit-Pipeline (§P1-1) und FC-GPP-Verwerfung intakt (per Test gepinnt).

---

## 4. Erwartete Wirkung (ohne erfundene Zahlen)

**Qualität (strukturell begründet, nicht geschätzt):**

- HPI, EmotionalArc, VQI/Singer-ID sind per Spec song-globale Größen (01_musical_goals.md;
  pipeline.instructions.md §2.44/§2.45b; phases.instructions.md §2.35c). Vor Slice B wurden sie auf
  30-s-Fenstern gemessen — der Dispatch nennt das zu Recht „semantisch verfälscht". Jetzt bewertet
  ein Satz Gates **den assemblierten Song**; der EmotionalArc-Bogen wird auf Song-Skala gemessen und
  (closed-loop, adaptive zentrale Dämpfung) korrigiert.
- **Entfernte Fehl-Entscheidungen:** 30-s-HPI-Fail-Closed/„HPI-Rollback" auf Chunks, 30-s-EAP-Rollback,
  30-s-VQI-/Singer-ID-Rollbacks — jeweils Entscheidungen auf Ausschnitts-Metriken.
- Das Result-Feld `emotional_arc` trägt jetzt den Song-Bogen statt Chunk-0-Propagation.

**Laufzeit (strukturell, keine Zusage):** pro Song entfallen im Chunked-Pfad rechnerisch
`(n−1)× 2 measure_all` (§8.1), `(n−1)× EAP-Messung (+ Korrektur)`, `n× HPI-Eval` inkl. `n× finale
Re-Eval`, `n× compute_vqi` (mit Resemblyzer/Singer-ID); hinzu kommt **einmal** die Song-Bewertung.
Für den 225-s-Referenzsong (n ≈ 8 Chunks bei 30 s) sind das strukturell ~25–30 entfernte
Analytik-/Gate-Auswertungen. **Wie viel RT das bringt, entscheidet ausschließlich die Messung (§6).**
Dokumentierte Gegenwirkung: pro Song kommen die Song-Level-Gates (1× EAP, 1–2× VQI bei Gate-Verfehlung,
1× HPI, 1× measure_all für das WO-Audit) hinzu. Normativ ist bekannt, dass Analytik **nicht** der
Haupttreiber der Laufzeit ist (Defekt-Scanner/Phasen-Pipeline/Einzelphasen,
.github/copilot-instructions.md:355–365) — die „15–25×"-Projektion aus dem TODO-Text wird hier
ausdrücklich **nicht** bestätigt oder zugesagt.

---

## 5. Verifikations-Befehle für den Manager (AUSSTEHEND — hier nicht ausgeführt)

```bash
# 1) Neue fokussierte Tests + Adjazenz (bestehende P0-1-Suiten)
nice -n 15 python3 -B -m pytest tests/unit/test_p0_1_song_level_perceptual_gates.py -q
nice -n 15 python3 -B -m pytest tests/unit/test_p0_1_end_gate_chunking.py -q
nice -n 15 python3 -B -m pytest tests/unit/test_determinism_p0_1.py tests/unit/test_end_gate_p0_1.py \
  tests/unit/test_process_end_gate_p0_1.py -q

# 2) Kritischer statischer Gate (wie Pre-Commit aurik-ruff-critical-static-gate)
ruff --select F821,F601,B009,I001 backend/core/unified_restorer_v3.py tests/unit/test_p0_1_song_level_perceptual_gates.py

# 3) Syntax-Gate (zusätzlich zur reinen Lese-Verifikation)
python3 -B -m py_compile backend/core/unified_restorer_v3.py tests/unit/test_p0_1_song_level_perceptual_gates.py

# 4) Write-Gate für die neue Testdatei (konnte hier nicht laufen)
python scripts/repo_search.py --before-create tests/unit/test_p0_1_song_level_perceptual_gates.py

# 5) Gesamt-Unit-Smoke + Spec-Drift (FILE_REGISTRY wurde erweitert)
nice -n 15 python3 -B -m pytest tests/unit -q --maxfail=3
python scripts/spec_drift_check.py   # ggf. FILE_REGISTRY-Hash nachziehen
```

Hinweis: `pytest`/`ruff` wurden in dieser Runtime **nicht** ausgeführt. Die Tests sind statisch gegen
die echten Modul-Signaturen und die Quell-Anker geprüft (`inspect.signature`-Checks sind Teil der
Suite); ein Grün-Stand wird hier **nicht** behauptet.

---

## 6. Never-worsen-Vorbereitung (Ausführung macht der Manager)

**A. Scorecard-Vergleich (B3-Muster, Corpus-Fenster):**
Das benannte Muster `scripts/b3_transient_energie_neverworsen.py` liegt in dieser Branch **nicht vor**
(§1). Sobald es in der Ziel-Branch existiert, unverändert als Muster nutzen (`git checkout <ziel> -- scripts/b3_transient_energie_neverworsen.py`).
Solange gilt das Ablaufmuster (analog B3, ohne das Skript anzufassen):

1. Corpus-Fenster (30 s) über die gleiche Material-Mix-Liste wie der B3-Lauf; je Fenster Basis-Lauf
   (vor Slice B) und Ist-Lauf (nach Slice B) mit identischer Zelle/Seed (§G5 (GEBOTE.md)).
2. GOAL_SCORECARD je Lauf: `MusicalGoalsChecker.measure_all(...)` (die gleiche Kennzahl wie
   Block a1/`_measure_goals_for_tail`) auf Ausgabe **und** Referenz.
3. **Kriterium:** alle 15 Goals ≥ Basis − 0,05 (Never-worsen/JND 0,05 laut Dispatch); Ausnahme dokumentieren.
4. Zusätzlich die neuen Song-Metadaten vergleichen: `metadata["p0_1_song_perceptual_gates"]`
   (hpi, vqi, singer_identity_cosine, emotional_arc) — Werte dürfen sich ändern (das ist der Zweck),
   aber das Gate-Verdict soll nicht häufiger „verfehlt" ausfallen als die Summe der früheren
   Chunk-Verdikte; begründete Ausnahmen protokollieren.

**B. RT-Vergleich (gleiches Protokoll wie die gemessene Kalibrierung):**
`.venv_aurik/bin/python scripts/benchmark_effizienz_matrix.py --input "test_audio/Elke Best - Hey Kleiner, mit Dir spielt wohl keiner (1977) HD audio 320 kbps german (original).wav" --seconds 30 --cells balanced --no-wav --enforce-budget --profile-top-phases 5`
→ Wall- und Engine-RT (3 Läufe, Median; Format wie docs/reports/current/2026-10-07_leistungskalibrierung.md §Zahlen).
Für den Dispatch-Abgleich zusätzlich ein 225,33-s-Lauf im dort genannten Protokoll (Report §9 der
F-Reihe — in der Manager-Branch), damit 32,19× ↔ neu vergleichbar sind. Berichtet wird
**nur gemessen**; die Projektion „15–25×" bleibt unbestätigt bis zum Messlauf.

---

## 7. Offene Punkte (bewusst NICHT in Slice B)

1. **`artifact_freedom` für das Song-HPI** stammt weiterhin vom letzten Chunk (`self._artifact_freedom_score`,
   gesetzt im per-Chunk AF-Gate :22292/22420) — AF song-weit wäre der saubere nächste Schnitt.
2. **PQS/§2.14** läuft auf dem letzten Chunk statt auf dem assemblierten Song (§v10.735-Status);
   analoger Hoist analog Slice B möglich.
3. **`_collect_reporting_analytics` (MQA/HPI-Reporting, Block c)** läuft auf dem letzten Chunk; die
   Extraktionsgrenzen des TODOs fordern „nach der Chunk-Assembly EINMAL" — bewusst offen gelassen
   (Aufwands-Diskussion, siehe Wirkung §4).
4. **§2.49-Export-Gate + „final export audio gate"** laufen weiterhin je Chunk (Export-Gate-Semantik).
5. **SingMOS (§G4 (GEBOTE.md)), PhonemeConsistency, MusicalCoherenceGuard (§2.78)** laufen je Chunk;
   MCG hat song-globale Semantik (Vers-1 ↔ Vers-2), prüfenswert als eigene Aufgabe.
6. **§v10.802 (copilot-instructions.md):** Patch-Bump `version.py` + CHANGELOG-Block im selben Merge
   (Manager); die Eignung „bit-identisch belegt" trifft nur den Ganzsong-Pfad zu — der Chunked-Pfad
   ändert sich by design (Qualität); Never-worsen-Lauf (§6) liefert die Evidenz.
7. Der dispatch-intern zitierte F-Reihe-Report (2026-10-28) ist dieser Branch fremd — Abgleich der
   Baselines (32,19× hier 60,1×/52,5×) ist Manager-seitig in der Ziel-Branch zu führen.

---

## 8. Conventional-Commit-Vorschlag (Manager führt aus)

```
perf(pipeline): §P0-1 Slice B — HPI/EmotionalArc/VQI von per-Chunk auf Song-Ebene

- §8.1 measure_all-Pre/Post-Excellence-Messung auf Nicht-Letzten-Chunks deferriert
- §8.2 EmotionalArc (Messung + Makro-Korrektur) song-weit nach Assembly statt je Chunk
- §2.44 HPI-Export-Gate + finale Re-Eval + §2.35c/§0p VQI/Singer-ID: Skip im Chunked-Pfad,
  einmal song-weit auf dem assemblierten Song (Song-Gate 1× pro Song statt pro Chunk)
- fail-closed-Landmine _hpi_result is None greift nicht mehr im Chunked-Pfad
- Ebene-1-Schutz: VQI/Singer-ID-Gate song-weit (zentrale Schwellen 0.92/Material-Floor),
  Snapshot-Rollback nur nach Verifikation
- §P1-1 Wohlklang-Ordnungs-Audit song-weit ergänzt (Enforcer bleibt intakt)
- Tests: tests/unit/test_p0_1_song_level_perceptual_gates.py (18 Fälle, statisch verifiziert)
```

Ergänzend im selben Merge: Patch-Bump `backend/core/version.py` (§v10.802 (copilot-instructions.md)),
`.github/FILE_REGISTRY.md`-Einträge (siehe Diff), CHANGELOG-Block.

---

## 9. Geänderte Dateien (Evidenz)

| Datei | Änderung |
| --- | --- |
| `backend/core/unified_restorer_v3.py` | E1/E3/E4/E5/E6/E7/E8 (Per-Chunk-Deferral) + Song-Level-Block :47528–47803 + Result-Verdrahtung |
| `tests/unit/test_p0_1_song_level_perceptual_gates.py` | **neu**, 18 Testfälle (Write-Gate-Befehl §5.4 ausstehend) |
| `.github/FILE_REGISTRY.md` | zwei Einträge (Testdatei + dieser Report) |
| `docs/reports/current/2026-11-05_p0_1_song_ebene_analytik.md` | dieser Report |
| `docs/TODOS_SOTA_ROADMAP.md` | Status-Nachtrag Slice B im TODO-P0-1-Abschnitt |

**Nicht angefasst (wie beauftragt):** `scripts/train_job_runner.py`, `scripts/train_*.py`, `models/`,
`output/f3_bigvgan`, `docs/reports/current/train_job_d3_*`, `docs/SOTA_HANDOVER.md`,
`backend/core/musical_goals/transient_energy_metric.py`, `scripts/b3_transient_energie_neverworsen.py`
(letztere existiert in dieser Branch ohnehin nicht).

---

## 10. Manager-Nachtrag (2026-11-05): Ausführung, Basis-Korrektur, Verifikation

Der Dispatch erreichte die 1-h-Wandzeit (Arbeit erhalten: 5 Dateien, +962/−31).
Übernahme, Review und Vollendung durch den Manager:

1. **Basis-Diskrepanz (Befund des Workers bestätigt):** Das Worktree hing auf
   `65c6ef4d` (2026-10-08, 10.12.15) — vier Wochen alt. Daher die beobachteten
   Abweichungen (RT_REALITY_MEASURED = 53,0 statt 32,2, kein
   `b3_transient_energie_neverworsen.py` usw.). Die Mess-Angaben des Dispatches
   waren korrekt für den Hauptbestand; der Worker hat sie ehrlich als
   „nicht belegbar in diesem Worktree" ausgewiesen. Merge-Basis auf `aa3e2383`
   nachgezogen.
2. **Konflikt `unified_restorer_v3.py` (gelöst):** Die Worker-Basis kannte den
   §G10-Song-Audibility-Nachzug (D-K3-68, `b57563e6`) nicht — beide Blöcke
   bleiben erhalten: §G10-Verdikt zuerst (Zustand nach Song-m1b), danach
   Slice B (Perceptual-Gates + Snapshot-Rollback, Song-Wohlklang-Audit als
   letztes Wort vor der Export-Finalisierung §P1-8).
3. **Test-Korrekturen (4 Fehler → 17/17 grün):** zwei Arc-Tests riefen
   `measure/correct_emotional_arc` mit 8000 Hz auf (Modul fordert 48000 Hz,
   `emotional_arc_preservation.py:179`); ein Test prüfte die Pipeline-Audit-
   Zeichenkette ohne das seit `a6a41cf4` bestehende „+ ENFORCEMENT"; zwei
   Quell-String-Asserts waren einrückungssensitiv.
4. **Verifikation (ausgeführt):** `test_p0_1_song_level_perceptual_gates.py`
   **17/17 grün**; Adjazenz (`test_unified_restorer_v3`,
   `test_unified_restorer_stability_guard`, `test_wohlklang_ordnung_gate`,
   `test_wohlklang_ordnung_enforcer`) **369 grün**; kritisches Lint-Gate
   (F821/F601/B009/I001/N811) **clean**. Version 10.12.51.
5. **Noch ausstehend (Manager):** Never-worsen-Scorecard-Vergleich vor/nach
   und RT-Nachmessung nach dem Muster der Reports §9 der F-Reihe — die
   Ausführung erfolgt nach dem D3-Training (GPU-freies Fenster).
