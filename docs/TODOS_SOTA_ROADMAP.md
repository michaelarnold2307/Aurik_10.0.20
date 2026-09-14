# TODOS — SOTA-Roadmap & Lücken-Schluss (nächste Sessions)

> **Stand:** 2026-09-08 · Quelle: Matrix-Endlauf (3 Zellen, Elke-Best-Vinyl) + unabhängige
> Spec-vs-SOTA-Tiefenanalyse (`docs/reports/current/2026-09-08_envelope_root_cause_sota_fixes_matrix.md`).
> **Wie erkannt wird:** Jede Aufgabe hat eine TODO-ID (`TODO-P0-1` …), Ziel, Wirkung,
> Beleg (Pfad:Zeile) und Akzeptanzkriterium. Agenten: Aufgabe mit ID greppen, Beleg lesen, umsetzen,
> Akzeptanzkriterium als Test/Gate nachziehen, `change_ledger.py snapshot` + Commit.
> **Reihenfolge:** P0-1 → P0-2 → P0-3 → P1-1 → P1-2 → P1-3 → P1-4 → P2-1.

---

## TODO-P0-1 · Analytik + End-Gate von per-Chunk auf Song-Ebene heben (größter Hebel)

- **Ziel:** Chunks restau­rieren → assemblieren → **einmal** song-weit validieren (GOAL_SCORECARD,
  End-Gate-Wiederherstellung, HPI, EmotionalArc, VQI, Einladungs-Gate). Nur lokal-stationäre
  DSP/ML-Phasen bleiben je Chunk.
- **Wirkung:** Performance ~53× → 15–25× RT (8–9 End-Gate-Runden × measure_all je Chunk entfallen);
  **Qualität steigt**, weil HPI/Sänger-Identität/EmotionalArc/VQI per Spec song-globale Größen sind
  und heute auf 30-s-Ausschnitten semantisch verfälscht laufen.
- **Beleg:** `backend/core/unified_restorer_v3.py` `_restore_chunked` (GOAL_SCORECARD/Recovery je
  `restore()`-Aufruf); `pipeline.instructions.md` §2.45b (EmotionalArc), §2.44 (HPI als Export-Gate);
  Session-Befund: 8–9 Runden je Chunk, 3 h 19–38 min/Lauf.
- **Extraktionsgrenzen (aus Session-Analyse 2026-09-08, restore()-Tail in unified_restorer_v3.py):**
  Song-globale Blöcke, die nach der Chunk-Assembly EINMAL laufen sollen:
  (a) GOAL_SCORECARD + End-Gate-Recovery-Kaskade (~Z. 16600–17200),
  (b) Einladungs-Gate (~Z. 17972), (c) MQA/_collect_reporting_analytics (~Z. 18667),
  (d) Audibility-Gate + m1b-Queue (~Z. 23260–23340). Vorschlag: neuer
  `_run_song_level_tail(assembled_audio, …)`-Aufruf am Ende von `_restore_chunked`;
  per-Chunk-restore() erhält einen Flag, der diese Blöcke überspringt. Chunk-lokal
  bleiben: Strength-Envelope, Phasen-Loop, FC-Iterationen, PMGG.
- **Status 2026-09-08: VOLLSTÄNDIG UMGESETZT (Slice A + Blöcke a–d).**
  `_should_run_end_gate_cascade()` + Kwargs `_chunked_tail_skip`/`_chunked_last`;
  im Chunked-Pfad läuft die End-Gate-Recovery-Kaskade nur noch auf dem letzten Chunk.
  Block a1: `_measure_goals_for_tail()` überspringt `measure_all()` auf
  Nicht-letzten-Chunks; `_restore_chunked` misst die GOAL_SCORECARD einmal auf dem
  assemblierten Song. Block a2: `_run_song_level_end_gate()` — die 654-Zeilen-Kaskade
  als Methode extrahiert und läuft nach der Assembly EINMAL song-global (kompensiert
  Slice A qualitativ; metadata `p0_1_song_end_gate_applied`). Block b:
  `_run_inviting_gate_measure()` — Einladungs-Gate-Messung nur letzter Chunk +
  einmal song-global nach Assembly (`inviting_gate_song_level`). Block c:
  `_collect_reporting_analytics` nur letzter Chunk (`analytics_last_chunk`).
  Block d: B2-Post-Scan (`defect_scanner.scan`) nur letzter Chunk — der
  Last-Chunk-Scan speist die song-globale m1b-Auswertung. Nur reine Analytik
  ist deferriert; der Audio-Pfad (Phasen, FC, MDEM, Goosebumps, Export) bleibt
  unverändert. Tests: 13 Fälle + Restorer-Suiten 298 passed.
- **Akzeptanz:** 224-s-Referenzlauf ≤ 40 min Gesamtlaufzeit; 3-Zellen-Output bleibt bit-identisch
  (Determinismus §G5); alle song-globalen Gates laufen nachweislich auf dem assemblierten Song.

## TODO-P0-2 · Tier-Map-Synchronisation + Pre-Commit-Gate

- **Ziel:** `PRIORITY_MAP` (brillanz=5, spatial_depth=5) und `HEARING_TIER_MAP` (brillanz=4,
  spatial_depth=4) in `goal_priority_protocol.py` vereinheitlichen; Goal-Namens-Aliase
  (`timbre`/`timbre_authentizitaet`, `raumtiefe`/`spatial_depth`, `sep_fidelity`/`separation_fidelity`)
  kanonisieren; deterministischen Sync-Test (analog `test_pmgg_cig_sync.py`) als Pre-Commit-Gate.
- **Wirkung:** Verhindert divergente Gate-Entscheidungen (WohlklangOrdnungGate nutzt `hearing_tier()`,
  FeedbackChain-Abort `priority_of()` — zwei Gewinner bei identischem Konflikt).
- **Beleg:** `backend/core/goal_priority_protocol.py:39–84`; `backend/core/wohlklang_ordnung_gate.py`;
  Tiefenanalyse Abschnitt A.2.
- **Akzeptanz:** Neuer Test schlägt bei jeder Divergenz fehl; CI grün; keine stillen Default-Tier-3-Fälle
  für bekannte Goals.
- **Status 2026-09-08: UMGESETZT** — `GOAL_ALIASES` + `canonical_goal()` + `verify_map_consistency()`
  in `goal_priority_protocol.py`; `hearing_tier()`/`priority_of()` nutzen die Kanonisierung;
  Sync-Test `tests/unit/test_goal_tier_map_sync.py` (5 Tests) läuft in unit-smoke/coverage.

## TODO-P0-3 · Budget-Wahrheit: drei Zahlen in eine konvergieren

- **Ziel:** Performance-Budget-Tabelle (240 s/min), PerformanceGuard-Limits (32× RT für alle Modi,
  obwohl Balanced-Doku „3× RT“ sagt) und Realität (53× RT) in **eine** normative Quelle bringen;
  `add_analytics_overhead()`-Verschleierung durch ehrliches Messzeit-Reporting ersetzen.
- **Wirkung:** Kein Schein-Soll mehr; Budget-Entscheidungen (Phase-Deferrals) werden korrekt kalibriert.
- **Beleg:** `.github/copilot-instructions.md` Performance-Budget; `backend/core/performance_guard.py:46,53,120–130,293`.
- **Akzeptanz:** Eine Budget-Norm mit Querverweisen; Benchmark-Matrix meldet Verletzungen ehrlich;
  nach TODO-P0-1 neu kalibriert (Ziel: 32× wieder erreichbar).

## TODO-P1-1 · Modell-Residency & Warm-up-Policy

- **Ziel:** Spezifizieren und umsetzen, welche Modelle warmgehalten werden (Residency/LRU je Session)
  und wie Warm-up einmalig je Modell amortisiert wird; deterministisches Multi-Song-Batching desselben
  Modells unter Wahrung von §G1 (Seed-Isolation pro Song).
- **Wirkung:** ~5 min Modell-Ladezeit je Lauf entfällt; batchfähige GPU-Nutzung senkt 53× weiter.
- **Beleg:** Session-Befund (jede Matrix-Zelle lädt Modelle neu); `spec 15 §15.9` (InferenceSessionManager
  Roadmap); `copilot-instructions.md` §G1.
- **Akzeptanz:** Zweiter Lauf in derselben Session ohne Modell-Nachladen; Determinismus-Nachweis je Song.

## TODO-P1-2 · Separation auf VS-1/GSEP + Demucs v5 heben

- **Ziel:** Vocal-Router um VS-1 (SongEval-2025-Gewinner) als Top-Stufe erweitern; Demucs v5 als
  zusätzliche Stufe prüfen. Hörordnungs-Invarianten (Sänger-Identität, Ebene 1) decken das Risiko ab.
- **Wirkung:** Größter Hörgewinn je Aufwand (Stem-Ersetzung, `separation_fidelity`, Ebene 3).
- **Beleg:** `backend/core/sota_vocal_model_router.py:4,57–150` (aktuelle Kette BS-RoFormer → Demucs v4 → MDX23C);
  Tiefenanalyse A.1.
- **Akzeptanz:** A/B-Metrik `separation_fidelity` + `singer_identity_cosine` ≥ MDX23C-Stand; Hörstichprobe.
- **Status 2026-09-08: GEWICHTE BESCHAFFT + KETTE REPARIERT** — BS-RoFormer-317-
  Checkpoint (609,7 MiB) vom kanonischen UVR-Mirror lokal beschafft und als
  state_dict verifiziert; tote Plugin-URLs ersetzt. Drei Silent-Failures in der
  Kette behoben (§V6): Demucs-Stufe war via gitignored Manifest stumm deaktiviert
  (→ produktiv, Opt-out-Env-Var), MelBandRoformer-Top-Stufe fiel durch
  UnboundLocalError immer in den Fallback, Router-Koerzition allozierte 215 GiB
  bei Mono-Stems. End-to-End-Smoke: model_used=melbandroformer, leere
  Fallback-Kette. VS-1/GSEP als öffentliche lizenzklare Gewichte nicht
  verifizierbar (SongEval-Leaderboard nicht erreichbar, SCNet ohne offizielle
  Weights) → ckpt→ONNX-Konversion + A/B-Metrik als Folge-Slice.

## TODO-P1-3 · Audibility (JND/Masking) auf alle Schwellwert-Guards

- **Ziel:** Formant-, Wärme-, Onset-, Spektralfarben-, Gain-Step-Toleranzen von fixen dB/Korrelations-Werten
  auf `max(fixed, lokale_Maskierungs-JND)` umstellen (wie Hörordnung Ebene 2 es als Prinzip fordert).
- **Wirkung:** Weniger falsche Rollbacks (Qualität) und weniger unnötige End-Gate-Recovery-Runden (Performance).
- **Beleg:** `dsp.instructions.md` §0p/§WBG/§SCK/§ATI; `hoerordnung.instructions.md` §4;
  `backend/core/residuum_masking.py` (bereits maskierungsbasiert — als Muster).
- **Akzeptanz:** Guard-Entscheidungen mit Maskierungskontext nachweisbar (Logs); Regressionstests für
  maskierte vs. unmaskierte Fälle.
- **Status 2026-09-08: SCK/WBG/ATI/Formant UMGESETZT** — `estimate_delta_masking_jnd_db()`
  + `delta_masking_margin_db_per_band()` in `backend/core/residuum_masking.py`
  (ISO-11172-3-Spread auf den Phasen-Delta, freq_range-Fenster, 6-dB-Cap,
  konservativ 0 bei Müll-Daten). §ATI: Toleranz = max(1,5 dB, JND) mit Log-Kontext.
  §WBG: maskierter Verlust-Anteil zählt nicht zum kumulativen Verlust.
  §SCK: maskierte Abweichung relaxiert die Korrelations-Schwelle begrenzt
  (max. −0,20 bei voller 6-dB-JND, linear). §0p Formant: Toleranz =
  max(fest/§V43-Frequenz-JND, lokale Bark-Marge) + Rollback-Warnung mit
  Maskierungskontext. DABEI §V6-Fix: `_burg_lpc` (Shape-Mismatch bei order ≥ 2)
  repariert — der Formant-Guard war ein stummer No-op; Exceptions laufen jetzt
  als warning. Tests: `tests/unit/test_p1_3_masking_jnd_guards.py` (12 Fälle:
  maskiert vs. unmaskiert je Guard, Burg-Regression, NaN-Schutz) + Bestandssuiten grün.
  §Gain-Step (2026-09-08): `temporal_continuity_guard` — effektive Schwelle =
  max(1,5 dB, JND) via `gain_step_threshold_db` im Result; uv3-Rescue-Trigger nutzt
  dieselbe Schwelle. **P1-3 damit VOLLSTÄNDIG UMGESETZT** (SCK/WBG/ATI/Formant/Gain-Step).

## TODO-P1-4 · Externe Blind-Hörstudie + GPU-A/B-Kalibration

- **Ziel:** MUSHRA-Blindstudie nach `docs/guides/MUSHRA_STUDIENPROTOKOLL.md` mit menschlichen Hörern;
  CPU-vs-ROCm-Gleichwertigkeitslauf erzwingen (löst „CPU ist Referenz“ vs. GPU-Produktion auf).
- **Wirkung:** Der zentrale „Ohr entscheidet“-Anspruch wird validiert; GPU-Betrieb bekommt Referenz-Status.
- **Beleg:** `spec 15_world_class_gap_closure.md §15.3/§15.10` („null menschliche Hörer“);
  `v10.900 §9.3` (CPU-Referenz).
- **Akzeptanz:** Studienbericht + statistische Auswertung im Repo; GPU-A/B-Bit-Identität oder dokumentierte
  tolerierte Abweichung.
- **Vorbereitung 2026-09-14:** Konkreter Studienplan erstellt —
  docs/guides/MUSHRA_STUDIENPROTOKOLL.md §10 (6 Pilot-Szenarien aus
  test_audio/output, Bedingungen, n=12-Pilot → n≥30, Latin-Square,
  Tooling-Hinweise; Auswertemetriken tests/test_blindtest_metrics.py vorhanden).
- **Stimuli-Builder 2026-09-14:** `scripts/build_mushra_stimuli.py` (Seed 42, BS.1770-Abgleich,
  Low-Anchor −6 LUFS, Manifest ohne Zeitstempel, 6 Tests grün) + Pilot-Set gebaut unter
  `output/mushra_study/pilot/` — 3/6 Szenarien mit Aurik-Stimulus.
  **Befund:** 3 Szenarien (cassette_1980s_wow, mp3_64kbps_artifacts, cd_clipped_2000s) haben
  leere 50-Byte-Exports in output/supervised_run (Quality-Gate-Fail ~0,48 < 0,55 → Datei
  ohne Audio, rc=0, Status „ok“) — §0c (copilot-instructions.md)-Verstoß: bestmögliches
  sicheres Ergebnis mit Status „degraded“ fehlt. Eigener Bug-Hunt-Task (nächste Empfehlung).

## TODO-P1-5 · §v10.709 authentizitaet-Erhalt nach phase_12_wow_flutter_fix

- **Befund (2026-09-08, Verifikationslauf P0-1):**
  `WARNING §v10.709 Quality-Degradation #1 nach phase_12_wow_flutter_fix: ['authentizitaet']` —
  die Flutter-Korrektur (4–100-Hz-Band, 45 % des Wow/Flutter-Blends) flacht auch
  Vibrato/Intonations-Bends der Performance ab (authentizitaet = versa_similarity fällt).
- **SOTA-Lösung (UMGESETZT 2026-09-08, §AUTH-P12):** `_preserve_musical_modulation()` in
  `backend/core/phases/phase_12_wow_flutter_fix.py` — Root-Cause statt Workaround (§V7):
  Wo die musikalische Modulationstiefe (Vibrato-Band, Vokal-Frames) die Flutter-Korrektur
  dominiert, wird die Korrektur proportional Richtung Identität zurückgenommen (max. 85 %);
  mechanischer Wow/Flutter bleibt voll korrigiert. Deterministisch, NaN/Inf-geschützt (§0a).
- **Beleg:** `tests/unit/test_phase_12_musical_modulation_preservation.py` (6 Fälle, grün):
  Vibrato → ≥60 % zurückgenommen, Wow-only → unverändert, Passthrough/NaN/Short-Guards.
- **Akzeptanz:** Nächster Referenzlauf ohne §v10.709-authentizitaet-Warnung nach phase_12;
  bit-identischer 3-Zellen-Output (Determinismus §G5); Wow/Flutter-Reduktion unverändert.

## TODO-P1-6 · DeepFilterNet ML→DSP-Fallback (dec.onnx ohne Alpha-Head) — UMGESETZT 2026-09-08

- **Befund (Live-Log 07:35:55, 2026-09-08):** `IndexError: list index out of range` in
  `_infer_spectral_chunk` (`alpha = dec_out[1]`) → stiller OMLSA-Fallback statt trainiertem DFN.
- **Root-Cause (gemessen):** Der DFN3-Export `models/deepfilternet_v3_ii/{,finetuned/}dec.onnx`
  hat NUR einen Output (`coefs`) — `df_fc_a` (Alpha-Head) ist im trainierten DeepFilterNet3-
  Forward unbenutzt (df/deepfilternet3.py:321 definiert, Forward wendet `df_op(coefs)` ohne
  Alpha-Blend an). `dec.onnx.orig` (DFN2-Ära, emb=512) ist nicht kompatibel. Das Export-Skript
  hatte einen toten `alpha`-Verweis in `dynamic_axes`.
- **Lösung (§P1-6):** (1) Plugin: alpha optional (`dec_out[1] if len>1 else None`),
  `alpha=None` → pure DF wie trainierter Forward (blend=1.0); Load-Time-Warnung bei fehlendem
  Alpha-Head. (2) Gepolsterte Rand-Chunks (T=100-Modell, l<100): nur die ersten l
  Output-Frames von Maske/Koeffizienten verwenden (Broadcast-Fix). (3) Kurze Signale
  (S<T): Chunk-Pfad mit Pad+Trim statt Ganzsignal (enc erwartet exakt T=100).
  (4) `export_df_musik_onnx.py`: toten Alpha-Verweis entfernt, DFN3-Realität dokumentiert.
- **Beleg:** `tests/unit/test_deepfilternet_plugin_alpha.py` (4 Fälle, grün);
  Echt-Modell-Probe: 1 s + 3 s Audio vollständig durch den ONNX-Pfad (0.04/0.19 s),
  kein Fallback mehr.
- **Akzeptanz:** Nächster Lauf ohne „ML→DSP-Fallback“-Traceback für DeepFilterNet;
  Rauschunterdrückung über trainiertes DFN statt OMLSA.

## TODO-P1-7 · Keine hörbaren Restdefekte: m1b intern ausführen (Stufe-2-Nachbehandlung)

- **Ziel (User-Anforderung 2026-09-08):** In allen Importfiles sollen keine hörbaren
  Restdefekte übrig bleiben — bei vollem Erhalt von Musikalität und Klang, soweit möglich.
- **Befund (§v10.703 Defekt-Countdown, Lauf 2026-09-08):** 46 gefunden → 42 über
  Hörbarkeits-Schwelle → 3 behoben → **42 über Schwelle verbleibend**. Die m1b-Queue
  (Hörbarkeits-Gate → `deferred_phases`) wurde bisher NUR in die GUI-KMV-Queue gestellt;
  im Headless-/CLI-Flow konsumierte niemand sie → Restdefekte blieben unangetastet.
- **Lösung (§P1-7):** `_run_m1b_targeted_retry()` in `unified_restorer_v3.py` führt die
  sicher zugeordneten Retry-Phasen (`DEFECT_RETRY_PHASE_MAP`: hum/clicks/crackle/wow-flutter/
  hiss/reverb/echo/compression) intern EINMAL aus: nur Phasen mit klarer Typ-Zuordnung
  (§V7 kein „mehr von allem“), verbotene Phasen (§0a) ausgeschlossen, Re-Entry-Guard
  `_m1b_pass_active`, deterministisch, bei Fehler/keiner Ausführung bleibt das Original
  (kein Audio-Ersatz). Verkabelt: (1) restore()-Tail nach dem Hörbarkeits-Gate
  (`m1b_retry_applied`/`m1b_retry_types` im Ergebnis-Metadata), (2) `_restore_chunked`
  nach der Song-Assembly (song-global, letzter Chunk-Scan bestimmt die Typen).
- **Beleg:** `tests/unit/test_m1b_targeted_retry.py` (6 Fälle, grün: nur gemappte Phasen,
  §0a-Ausschluss, Re-Entry-Guard, no-exec → None, Chunk-Shift-Restore);
  Restorer-Suiten 289 passed.
- **Akzeptanz:** Nächster Referenzlauf: `n_audible_unmasked` deutlich reduziert (Ziel: 0,
  soweit physisch möglich — `physical_cap`-Typen dokumentiert ausgenommen);
  `m1b_retry_applied=True` im Metadata; keine Verschlechterung von
  authentizitaet/natuerlichkeit (GOAL_SCORECARD ≥ Vorlauf).

## TODO-P1-8 · Export NACH dem 2. Durchgang — FinalPolish/OneTakeExport ans Tail-Ende — UMGESETZT 2026-09-08

- **Befund (User-Frage 2026-09-08):** Lief der Export nach dem 2. Durchgang? Nein —
  `apply_final_polish` (Era-EQ + Noise-Shaped Dither) und `OneTakeExport.prepare`
  (LUFS/True-Peak) liefen bei Z. 14059/14105 MITTEN im Tail (STUFE-8), die
  m1b-Nachbehandlung erst bei Z. 22730 — der m1b-Output wurde nie neu exportiert:
  Dither/LUFS/TP galten für einen Zwischenstand, Humanization/PEO/MDEM/Goosebumps/m1b
  liefen teils auf bereits gedithertem Audio.
- **Lösung (§P1-8):** Export-Finalisierung (FinalPolish → OneTakeExport) ans TAIL-ENDE
  verschoben — NACH m1b. Dither ist damit der letzte Quantisierungsschritt (§V5),
  LUFS/True-Peak gelten für das FINALE Audio, alle DSP/ML-Schritte laufen auf voller
  Float-Präzision. `result.audio` wird nach der Finalisierung aktualisiert.
  Chunked-Pfad: nach song-globaler m1b nur OneTakeExport (idempotente Zielkorrektur);
  FinalPolish lief je Chunk und wird nicht doppelt angewendet (kein Doppel-EQ).
- **Beleg:** Restorer-Suiten 296 passed (inkl. Alignment); Linter/GEBOTE clean.
- **Akzeptanz:** Referenzlauf: Log zeigt „§P1-8 FinalPolish (nach m1b)“ und
  „§P1-8 OneTakeExport (nach m1b)“ AM ENDE; LUFS/TP im Zielband; bit-identischer
  3-Zellen-Output (Determinismus §G5 innerhalb der Version).

## TODO-P1-9 · Fortschrittsanzeige-Stillstand bei 26,88 % — Chunk-Heartbeat + m1b-Adapter — UMGESETZT 2026-09-08

- **Befund (User 2026-09-08):** Fortschrittsbalken bleibt ab 26,88 % stehen (Chunked-Lauf);
  GUI-Watchdog meldete bereits „W-PROGRESS-STALE — Bar-Step-Abweichung >15%“.
  26,88 % = Chunk 2 (Fenster 25–37) bei Chunk-lokal 15,67 % — nach `_cb(15)`
  („Klangleitplanken“) bis zur nächsten Emission (`_cb(16)`/Pipeline-Start) liegen
  je Folge-Chunk Minuten ohne Fortschritts-Event (Pre-Analyse-Bypass + Tail-Lücken).
- **Lösung (§P1-9):** (1) Chunk-Progress-Heartbeat in `_make_chunk_pc`: Daemon-Thread
  emittiert alle 2 s monotones Mikro-Progress (asymptotisch zur Chunk-Decke, nie darüber);
  echte Events übernehmen per Monoton-Maximum; Stopp via `finally` nach jedem Chunk.
  (2) m1b-Progress-Adapter: `_run_m1b_targeted_retry` reicht den GUI-Callback jetzt über
  den 4-arg-Adapter (pct, msg, elapsed, metrics=None) durch — konsistent mit dem Haupt-Call.
- **Beleg:** `tests/unit/test_m1b_targeted_retry.py` (7 Fälle, neu: GUI-Signatur-Adapter);
  Restorer-Suiten 292 passed; Linter/GEBOTE clean.
- **Akzeptanz:** Referenzlauf: Balken bewegt sich kontinuierlich (kein Stillstand >30 s je
  Chunk; Watchdog ohne W-PROGRESS-STALE); Balken erreicht 100 % erst nach Assembly.

## TODO-P1-10 · Defekt-Rückwärtszählung (Chips) + weitere GUI-Bugs — UMGESETZT 2026-09-08

- **Befunde (Bug-Hunt 2026-09-08):** (1) `_defect_chip_counts`/`_defect_chip_total`
  wurden in der GUI NIE initialisiert → `apply_resolved_defects({}, …)` war ein No-op —
  die Chip-Rückwärtszählung lief ins Leere. (2) Die GUI verarbeitete nur den LETZTEN
  `resolved`-Diff (`_latest_live_metrics`-Snapshot) — frühere Behebungen gingen verloren
  und der aktive Defekt-Set wurde bei jedem Render aus der Scan-Liste neu aufgebaut
  (behobene Typen kehrten zurück). (3) Backend: `_resolved_sent_keys` wurde nie
  zurückgesetzt → behobene Typen nur im ersten Song/Chunk gesendet.
- **Lösung (§P1-10):** (1) GUI initialisiert die Chip-Zähler beim ersten Render aus dem
  aktiven Defekt-Set (ein Chip je Typ). (2) GUI kumuliert Behebungen in
  `_resolved_defects_accumulated`; Render wendet den kumulierten Satz an und filtert
  behobene Typen aus dem aktiven Set (`_done_filter`). (3) Backend setzt
  `_resolved_sent_keys` pro restore()-Aufruf zurück (§V8-Song-Isolation).
- **Beleg:** 27 GUI/m1b-Tests grün; Syntax/ruff clean; Restorer-Suiten grün.
- **Akzeptanz:** GUI-Lauf: Chip-Zähler „Verbleibende Schäden“ zählt bei jeder behobenen
  Defektkategorie herunter; behobene Chips bleiben grün über Re-Renders; zweiter Song
  zählt erneut korrekt.

## TODO-P1-11 · GUI-Funktionen debuggen (Abdeckungs-Lücken schließen) — UMGESETZT 2026-09-08

- **Befund (Abdeckungs-Audit):** 12/22 GUI-Module hatten KEINEN Test-Import; die
  Daten-/Entscheidungslogik (Radar-Farben, Fehlertexte, Tasten, Presets, Startup-Reihenfolge)
  war nur manuell verifiziert — genau dort lagen die P1-9/P1-10-Bugs.
- **Lösung (§GUI-T1…T4, pure Logik extrahiert + Tests):**
  T1 `musical_goals_radar`: `goal_bar_state()` + `build_radar_update_payload()` (5 Fälle).
  T2 `main.py`-Startup-Vertrag: GPU vor ModernMainWindow, __main__-Guard, -B-Launcher (4 Fälle).
  T3 `help_system.ErrorSimplifier`: Exception-Klassenname wird einbezogen — Befund dabei:
  `MemoryError("x")` wurde als Roh-Text „x“ angezeigt statt der freundlichen Meldung (7 Fälle).
  T4 `keyboard_shortcuts` (`key_action`/`seek_frac`) + `export_presets`-Presets-Vertrag (3 Fälle).
- **Beleg:** 39 GUI-Tests grün; ruff clean.
- **Akzeptanz:** Alle vier Entscheidungsketten sind headless-verifiziert; künftige
  Änderungen an Farben/Tasten/Presets/Fehlertexten brechen sofort einen Test.

## TODO-P1-12 · Nutzersichtbare Dezimalformate der Live-Anzeige vereinheitlicht (§GUI-T5) — UMGESETZT 2026-09-08

- **Befund:** Während der Restaurierung zeigt die GUI Prozentwerte an mehreren Stellen.
  Queue-Liste (`⏳ datei (26.88%)`), Smooth-Bar-Fallback (`26.88 %`) und ein
  Heartbeat-Prognose-Pfad (`55.0 %`) formatierten mit **Punkt**-Dezimaltrenner, während
  Hauptbalken und Status-Texte überall **Komma** („26,88 %“) nutzten — je nach Code-Pfad
  flackert/wechselt das Dezimaltrennzeichen in derselben Anzeige (deutsche Oberfläche).
  Zusätzlich Defekt-Chip-Schweregrade (`12.34%`/`0.00%`) mit Punkt.
- **Lösung (§GUI-T5):** Pure Modul-Funktion `_de_num(value, digits=2)` als **eine Quelle
  der Wahrheit** in `Aurik10/ui/modern_window.py`; umgestellt: `ModernProgressBar.
  _set_value_immediately`, `_tick_uv3_simple_progress`-setFormat, Queue-Listen-Eintrag
  in `_on_item_progress`, Defekt-Chip-Schweregrade (`sev_txt`); 5 verbliebene
  `.replace(".", ",")`-Duplikate konsolidiert (verhaltensneutral).
- **Beleg:** `tests/unit/test_gui_live_display_decimal_format.py` (8 Fälle: Verhalten der
  puren Funktion + Quelltext-Invarianten, dass kein Live-Pfad mehr Punkt formatiert und
  kein replace-Duplikat außerhalb `_de_num` existiert); GUI-Suiten 148 passed, 11 skipped.
- **Akzeptanz:** Kein nutzersichtbarer Live-Text nutzt mehr Punkt-Dezimaltrenner; künftige
  Punkt-Formatierung im Fortschritts-/Chip-Pfad bricht sofort einen Test.
## TODO-P2-1 · Hygiene: UTF-16-Bereinigung + Monolith-Hinweis — ERLEDIGT 2026-09-08 (Guard-Teil)

- **Befund (gemessen 2026-09-08):** Alle 3175 getrackten Textdateien sind valides UTF-8;
  0 UTF-16-Dateien, 0 BOMs, 0 invalide Sequenzen. Das früher beobachtete „UTF-16-Garble“
  war ein Anzeige-Artefakt des Tool-Kanals (UTF-8-Bytes werden in manchen Ausgaben als
  UTF-16LE fehlinterpretiert — per Hex-Analyse belegt), kein Repo-Zustand.
- **Umgesetzt:** `scripts/utf8_hygiene_check.py` (fail-closed: R1 UTF-16/32-BOM,
  R2 invalide UTF-8-Sequenzen, R3 NUL-Byte-Fenster ≥10 % je 2-KiB-Fenster, auch in
  Dateien < 2 KiB) + Pre-Commit-Hook `aurik-utf8-hygiene` + FILE_REGISTRY-Eintrag +
  Drift-Baseline nachgezogen. Negativtest: BOM-Datei und BOM-lose UTF-16LE-Datei → EXIT 1.
- **Offen (Folge-Session):** Refactor-Plan für das 45.309-Zeilen-God-Object
  `unified_restorer_v3.py` (Budget-, Gate-, Recovery-, Chunk-Logik getrennt) als Doc skizzieren.
- **ERLEDIGT (2026-09-14):** Der Refactor-Plan existiert bereits als
  `docs/P2_1_MONOLITH_REFACTOR_PLAN.md` (2026-09-09, Fassaden-Split in
  5 Phasen) — dieser Eintrag dient nur noch als Nachweis-Link.
- **Akzeptanz (erfüllt):** `file` meldet UTF-8 für alle .py; Guard verhindert Regressionen.

---

## Welle 2026-09-12/13 — Wohlklang-SOTA-Gesamtmaßnahmen (Maskierung, Hybrid-Gates, Finetunes)

> Quelle: `docs/PHASE_SOTA_GAP_ANALYSE.md` (alle 64 Phasen + ML-Trainingsdomänen-Matrix §5,
> verifiziert gegen die 69 Phasen-Dateien). Reihenfolge = Hör-Gewinn je Aufwand.
> IDs: `SOTA-*`; Akzeptanz = Never-worsen-Test + Determinismus + §V6 (copilot-instructions.md)-Fallback.

### ABGESCHLOSSEN (diese Welle, mit Commits)

| ID | Maßnahme | Commit | Beleg |
|---|---|---|---|
| SOTA-WIT-P1…P4 | Witness: Johnston-Maskierung, Rauigkeit, ITD/ILD, Pre-Echo | 56492ce8 | tests 27 grün |
| SOTA-C1…C3 | Rekombinations-Gates (Alignment, Bark-Residuum, Stereo-Check) | 19506b3e | tests 6 grün |
| SOTA-DECLIP | APPLADE + PnP-ADMM + GPU-Finetune (ΔSDR +1,2…+1,7 dB) | 07fcc8df | ONNX in Produktion |
| SOTA-DENOISE | EAR-VAE-Musik-Finetune v2 (ΔSDR +4,83 dB, Never-worsen ja) | 98276a32 | Final-Benchmark 3 Songs |
| SOTA-H1+H2 | Masking-Fusion + Musical-Noise-Gate um Denoiser | 6ab0dd2a | tests 7 grün |
| SOTA-H3 | Witness-Veto→global_scalar (delta-basiert, Clamping) | 6ab0dd2a | tests 7 grün |
| SOTA-B4+B5 | Synthesis-Gates um FlashSR/BigVGAN (masking, Onset-Schutz) | 4b783d18 | tests 21 grün |
| SOTA-C1-CON | UTMOS-Delta-Veto als Zeuge im FC-Regelkreis | 306d0a16 | tests 4 grün |
| SOTA-C3 | Preference-Lern-Schleife (Veto→sounds_artificial) | 4b783d18 | nicht-persistierend |
| SOTA-A1 | Maskierungs-Informierte Trainings-Loss (torch, differenzierbar) | 4b783d18 | EAR-VAE-Finetune nutzt sie |
| SOTA-WF-V1+V3 | Bandbegrenztes Resampling + Kalman-Glättung in phase_12 | ba151793 | tests 8 grün |
| SOTA-WF-V2 | Log-f-Zentroid-Warp-Schätzer + Zero-Consensus-Versorgung im Hauptfluss (Mono-Referenz, §2.51) | 86a10e10 | Akzeptanz 6,9 % < 10 %; 8 Tests grün |
| SOTA-IN-V1+V2 | Naht-Gates um phase_55-Kandidaten: Hüllkurven-Alignment + Band-weise Additive-Kappung (Kontext + 6 dB) | 09124e12 | 7 Tests grün; 253 phase_55/Inpainting-Tests grün |

### OFFEN (Reihenfolge = Ausführung)

1. **SOTA-WF-V2** · ✅ ERLEDIGT (86a10e10, 2026-09-13) — Versorgung läuft EINMAL im
   Hauptfluss vor dem M/S-Split (Mid/Side identische Faktoren, §2.51);
   Schätzer auf Log-f-Zentroid umgestellt (exakt unter uniformem Zeit-Warp).
2. **SOTA-ML-V1** · FlashSR-Musik-Finetune (MUSDB-HQ lowpass→original, A1-Loss,
   Encoder-Frozen, Validierungs-Early-Stop). VORAB: AERO-vs-FlashSR-Benchmark auf
   Musik als Kandidatenwahl (§V7: eine Lösung pro Rolle).
   Akzeptanz: Never-worsen-Benchmark wie EAR-VAE (ΔSDR/ΔSegSNR ≥ 0 je Segment).
   Status 2026-09-13: VORAB ✅ ERLEDIGT (`scripts/benchmark_bwe_candidates.py`,
   3 MUSDB18HQ-Tracks × 30 s, Seed 42, Report
   `docs/reports/current/2026-09-13_bwe_candidates.json`): **Beide Kandidaten
   liegen unter der Bandlimit-Baseline** — FlashSR nativ meanΔSDR −21…−24 dB
   (Sprach-Modell, auf Musik unbrauchbar), AERO nativ meanΔSDR −3,6…−5,5 dB
   (besser, aber Never-worsen verletzt); gleiche 12-kHz-Quelle: AERO +16…+18 dB
   über FlashSR. Kandidatenwahl: Kein Modell ohne Finetune einsetzbar;
   FlashSR bleibt Finetune-Basis (14× Echtzeit vs. AERO-BLSTM 0,8× Echtzeit,
   und FlashSR-Plugin ist bereits in der Pipeline verdrahtet).
   Hauptteil (Finetune): GPU-gebunden — ROCm (7900 XTX, 24 GB) am Nutzersystem via
   `.venv_aurik` (torch 2.11.0+rocm7.2) verfügbar; CI/Agent-Env CPU-only.
3. **SOTA-ML-V2** · BigVGAN-v2 Musik-/Vokal-Finetune für den Repair-Pfad (B5-gegated).
   Status 2026-09-13: GPU-gebunden (wie SOTA-ML-V1).
4. **SOTA-ML-V3** · MP-SENet: Musik-Finetune ODER De-Wiring (gemessen −5,9…−8,5 dB
   out-of-domain; Entscheidung anhand eines 3-Song-Vorher/Nachher-Benchmarks).
   Status 2026-09-13: VORAB ✅ ERLEDIGT — ENTSCHEIDUNG: DE-WIRING bestätigt
   (`scripts/benchmark_mpsenet_music_damage.py`, 3 MUSDB-Tracks × 30 s, Seed 42,
   Report `docs/reports/current/2026-09-13_mpsenet_music_damage.json`):
   Saubere Musik: SDR(out,in) +5,2 dB (moderate, aber unnötige Klangfärbung);
   verrauschte Musik (SNR 10 dB): ΔSDR **−5,9 dB** (BKS −5,1, Secretariat −7,4,
   Speak Softly −5,2) — der Sprach-Enhancer klassifiziert Musikanteile als Noise
   und verschlechtert damit genau das Ziel-Szenario aktiv. MP-SENet ist im
   unified_restorer nicht verdrahtet (grep 0) → De-Wiring = Status quo
   beibehalten; Musik-Finetune NICHT empfohlen (Denoise-Rolle durch die
   Pipeline abgedeckt).
5. **SOTA-ML-V4** · UTMOS-Musik-MOS-Validierung: Delta-Kalibrierung auf MUSDB-Paaren
   (kein Finetune möglich — nur Schwelle/-Richtung validieren).
   Status 2026-09-13: ✅ ERLEDIGT — NEGATIVBEFUND
   (`scripts/validate_utmos_music_calibration.py`, 3 MUSDB-Tracks × 30 s, Seed 42,
   Report `docs/reports/current/2026-09-13_utmos_music_calibration.json`):
   UTMOSv2-Fold-Ensemble ist auf Vollmix-Musik **richtungs-blind bis -invertiert**
   — alle Degradationsarten wurden HÖHER bewertet als das Original (ΔMOS(ref−deg)
   negativ: band12 −0,16, band16 +0,01, noise10 −0,21, noise0 −0,40; 0-dB-Rauschen
   gilt UTMOS als beste Variante). Voraussetzender Fix: UTMOS-Fold-Loader lief
   trotz vorhandener Weights in den PQS-DSP-Fallback (Geräte-Mismatch cuda/CPU
   beim SSL-Encoder-cat) — jetzt dokumentierte CPU-Policy forciert, ML-Pfad
   läuft (`utmosv2_fold_ensemble`). Konsequenz: UTMOS im Aurik-Audio-Modus
   nicht als Musik-MOS-Gate einsetzen; Delta-Gate-Richtung bei Vollmix prüfen
   bzw. PQS-DSP-Gate bevorzugen.
6. **SOTA-GACELA** · GaCELA-Integration (musik-nativ, KEIN Training): ltfatpy-Blocker
   lösen oder Inverter portieren + IN-V1/V2-Gates — einziger musik-nativer
   Lang-Lücken-Inpainter (375–1500 ms).
   Status 2026-09-13: KERN ✅ ERLEDIGT — ltfatpy-Blocker faktisch überholt:
   der Plugin läuft komplett auf eigener ONNX-Pipeline (GaussTruncTF → Mel →
   BorderEncoder → Generator → SpectrogramInverter; kein ltfat im Code).
   Verifiziert: End-to-End-Smoke (2,98-s-Lücke @48 kHz, ~1,3 s CPU), MUSDB-
   Realtest SDR −3,9 dB vs. Originalausschnitt bei 3-s-Lücke. Fix: Determinismus
   hergestellt — input-abgeleiteter blake2b-Seed für das GAN-Latent-Rauschen
   (vorher ungeseedet, maxdiff 1,06 bei identischem Input; jetzt bit-identisch, §G5).
   OFFEN: —
   ✅ ERLEDIGT (2026-09-13): Verdrahtung in phase_55 als Priorität 1.75 in der
   Kandidaten-Kaskade (`_try_gacela_plugin` in
   `backend/core/phases/phase_55_diffusion_inpainting.py`): 375–1500-ms-Lücken
   → 1-s-Kontext links/rechts → GaCELA → mittiger Zuschnitt auf Ziellänge →
   bestehender Damage-Guard + IN-V1/V2-Naht-Gates (generisch nach der Kaskade)
   → DSP/NMF-Fallback bei Fehlern (fail-closed, §V6). Verifiziert: 308 phase_55/
   Inpainting-Tests grün; Integrationstest 1-s-Lücke → GaCELA-Gap (48000,),
   200-ms-Lücke → None ohne Modell-Load.
7. **SOTA-IN-V1+V2** · ✅ ERLEDIGT (09124e12, 2026-09-13) — `inpainting_seam_gate.py`
   am Splice-Punkt jedes phase_55-Kandidaten (nach local_ratio-Blend, Fehler →
   unveränderter Kandidat + Debug-Log).
8. **SOTA-C4** · DDSP: neuronale EQ-/Dynamik-Parameter-Prädiktion (CLAP/BEATs-
   Embeddings) für phase_04/16/17 — DSP führt aus, nie Wellenform-Generierung.
9. **SOTA-TP-V1+V2** · Neurale Onset-Detektion (BEATs) als Konsens mit Spectral Flux +
   neurale Phasen-Schätzung für Transienten-Frames (phase_08/36).
   Status 2026-09-14: **TP-V1 V1-VERDRAHTET (Witness-Modus)** —
   `backend/core/dsp/beats_onset_detector.py`: Kaldi-fbank (25/10 ms, 128 Bins)
   → BEATs-iter3-Encoder-ONNX (fbank→768-Tokens, CPU) → zeitliche
   Token-Differenz als Onset-Kurve; Konsens-Statistik in phase_08 (Superflux)
   und phase_36 (Vollband-Envelope) als Metadaten (`onset_witness_beats`) —
   ZEUGE, nicht Richter (Hörordnung §8a); adaptive-Schwelle-Stufe als
   Folge-Schritt. 6 Tests grün inkl. echtem Encoder-Smoke.
   **Befund dabei:** plugins/beats_plugin.py füttert den ENCODER-ONNX mit
   Roh-Audio statt fbank (Rank-Mismatch) und interpretiert Token-Output als
   527-Scores — der Tagger-Pfad läuft nie (stiller DSP-Fallback).
   **Plugin-Seite ERLEDIGT (2026-09-14, Nachtrag):** Erkennung des
   Encoder-Exports (`beats_encoder_only`-Property, beim Laden), echte
   768-dim-Embeddings über den Encoder statt Nullen, ehrliche Cache-Herkunft
   (kein falsches „beats_onnx_cached“); 8 Tests. Offen bleibt nur der
   TAGGER-HEAD selbst (Head auf Tokens trainieren oder Tagger-ONNX
   beschaffen — GPU-Aufgabe). TP-V2 bleibt offen (Modell + Quelle fehlen).
10. **SOTA-HR-V1** · BigVGAN-Repair-Pfad mit additive_synthesis_gate in
    phase_07_harmonic_restoration.
11. **SOTA-CR-V1** · ✅ ERLEDIGT (36b452b4, 2026-09-13) — BANQUET-Klick-Detektion als
    zusätzlicher Detektor im Multi-Scale-Konsens von phase_01 (ML detektiert,
    RBME rekonstruiert): `_detect_clicks_banquet_ml` + `_merge_click_regions`
    (Never-worsen-Union, §V6-Fallback, 8 Tests grün; 111 hum/phase_01-Tests grün).
12. **SOTA-DR-V1** · Neurale RT60-Schätzung steuert Dereverb-Parameter (phase_20/49).
    Status 2026-09-13: **V1-VERDRAHTET** — `estimate_rt60_sec` im
    DeepFilterNet-Plugin (DFN-Trocken-Zerlegung → Schröder-T30 → RT60 +
    Modell-Diskriminator exponentiell vs. stationär) steuert die
    Dereverb-Stärke in phase_20/49 über `rt60_strength_delta` (neutral < 0,8 s,
    conf < 0,5 ⇒ 0, Cap +0,35); Metadaten `rt60_estimate_sec`/`rt60_confidence`.
    Befund Real-Smoke: der DFN entfernt auch Rauschen, die RT60-Präzision auf
    echtem Material ist begrenzt — deshalb konservativer, gedeckelter Einfluss;
    Never-worsen-Schutz bleibt bei den Phasen-eigenen Gates. 7 Tests
    (Ground-Truth 0,6 s ±25 %, trocken ⇒ conf=0, Layouts, NaN/Inf, Determinismus,
    Delta-Gates) + 2 Wiring-Tests phase_20 — 17/17 grün. Präziser neuraler
    RT60-Regressor bleibt Folge-Schritt (GPU-Training).
13. **SOTA-WF-V4** · Neuraler Warp-Schätzer (2025/26-Checkpoint) mit Never-worsen-Gate.
    Status 2026-09-13: BLOCKIERT — kein Warp-Schätzer-Checkpoint lokal
    (`plugins/flow_audio_sota.py` ist Conditional-Flow-Matching-Inpainting, kein
    Zeit-Warp-Schätzer; `models/` enthält keinen Warp-Estimator). Nächster Schritt:
    Checkpoint-Quelle klären (2025/26-Modell) + Download + Verdrahtung mit
    Never-worsen-Gate gegen phase_12-Warp — eigene Session.
14. **SOTA-WF-CASS** · Scrape-Flutter-Restpfad für Cassette (Breitband-Modulations-
    Kompensation/Denoise) — separat vom gemeinsamen Warp.
    Status 2026-09-13: KERN ✅ ERLEDIGT — `backend/core/dsp/scrape_flutter_rest.py`:
    Hochband-Einhüllende (>1,5 kHz) → Modulations-Peaks 5–120 Hz → adaptive
    Hüllkurven-Normalisierung (Trägerphase unberührt) mit Soft-Knee
    (conf 0,55–0,85) und Never-worsen-Energie-Gate (±10 %); (C,N)/(N,C)-sicher,
    deterministisch (§G5). 6 Tests grün (30-Hz-AM-Erkennung, Modulation −76 %,
    Passthrough, Energie-Gate, Determinismus, Stereo).
    OFFEN: —
    ✅ ERLEDIGT (2026-09-13): Verdrahtung als `_apply_scrape_flutter_rest`-Helper
    in `phase_12_wow_flutter_fix.process()` — an beiden Return-Pfaden nach dem
    Loudness-Preserve, Material-Gate tape/cassette/reel_tape, confidence ≥ 0,55,
    non-blocking §V6; Befund in result.metadata (`scrape_flutter_rest`).
    Verifiziert: 4 Phase-12-Tests grün, 6 WF-CASS-Tests grün, End-to-End
    (Cassette + 30-Hz-AM → conf 0,97/sev 0,28/f=[30,0]; CD → kein SFR).
15. **SOTA-HU-V1** · ✅ ERLEDIGT (a4428e50, 2026-09-13) — `backend/core/dsp/hum_drift_tracker.py`:
    Kalman-getrackter Netzfrequenz-Pfad [f, df] (Bandpass+Analytiksignal → Phasen-Differenz)
    + Fenster-LSQ-Subtraktion von Grundton+Harmonischen (Hann-Crossfade,
    Never-worsen-Energie-Gate je Fenster), verdrahtet hinter dem adaptiven Comb in
    phase_02 (Gate: Drift > 0,08 Hz + Reduktion > 3 dB, non-blocking §V6).
    Beleg: 6 Tests grün (Drift 49,8→50,2 Hz → >25 dB Hum-Reduktion, statisch >25 dB,
    Determinismus, Passthrough, Stereo); 111 hum/phase_02-Tests grün.
16. **SOTA-D3+D4** (niedrig) · Multiresolution-Split (neuronal 2–5 kHz, DSP außerhalb) +
    APPLADE-Maskierungs-Loss-Variante (DGT-Domäne).
17. **SOTA-VOCAL-INPAINT** · SOTA-Langlückenfüller FÜR GESANG (fehlt — Befund 2026-09-13:
    phase_55 drosselt bei vocal_confidence ≥ 0,40 bewusst die Füll-Stärke statt zu füllen;
    DiffWave ist Sprach-domänen-passend, aber ohne Vokal-Finetune und ohne Naht-Gates;
    GaCELA ist instrumental trainiert).
    Kandidaten: DiffWave-Vokal-Finetune (HAUPTWEG, ~2M Parameter, Plugin bereits in
    phase_55 verdrahtet; MUSDB18-HQ-Vocals lokal), AudioLDM2-Zero-Shot (Baseline),
    GaCELA-Vokal-Finetune (Pfad B, Upstream-Trainingscode vorhanden), RVC
    (2024/25-Singstimmen-SOTA, MIT — Konversions-Semantik, später).
    Umsetzung in 4 Schritten:
    - VOCAL-INPAINT-S1: ✅ ERLEDIGT (2026-09-13) — `scripts/benchmark_vocal_inpaint_baseline.py`
      (3 MUSDB-Vocal-Tracks × 30 s, 13 Lücken à 300 ms in aktiven Regionen, Seed 42,
      Report `docs/reports/current/2026-09-13_vocal_inpaint_baseline.json`):
      **DSP-Messlatte mean SDR −3,46 dB vs. Stille-Baseline (0 dB), DiffWave-Zero-Shot
      −1,69 dB — beide unter Never-worsen (min −7,2/−3,7 dB), DiffWave im Mittel +1,77 dB
      über DSP.** AudioLDM2-Zero-Shot entfällt weiterhin — das PLUGIN fehlt
      (`plugins/audiolm2_plugin.py` existiert nicht); **FS-Korrektur 2026-09-14:**
      die ONNX-Dateien `models/audioldm2/` sind real (1,39 GB + 132 MB, Kontrakte
      verifiziert) — der frühere „0-Bytes-Artefakt“-Befund war ein Messfehler.
      Plugin-Entscheidung: Neuanlage verschoben (FlanT5 fehlt lokal, Rolle durch
      S1 obsolet, GPU-Finetune F1 hat Priorität). Interpretation: Der Sprach-Checkpoint
      füllt Singstimmen bereits besser als der DSP-Pfad, aber noch unter der Messlatte
      „Nichts-Tun“ — der Vokal-Finetune (S2) hat eine quantifizierte Ziellücke (ΔSDR ≥ 0
      je Lücke, aktuell −1,7 dB).
    - VOCAL-INPAINT-S2: DiffWave-Vokal-Finetune (A1-Hör-Loss, Encoder-Frozen,
      Validierungs-Early-Stop — EAR-VAE-Rezept, GPU).
      **Status 2026-09-14: LÄUFT** — `scripts/train_diffwave_vocal_inpaint.py`
      (philovivero-Port, strict load 0/0; WIN=16368, Mel T=65 — Checkpoint-
      Konventionen; die Plugin-ONNX nutzt einen adaptierten Conditioner,
      Export-Anpassung ist Folge-Schritt). Smoke grün (Baseline mean −3,01 dB,
      1 Epoch Val −3,03 dB); voller Lauf (60 Epochs, 50 Train-Tracks, batch 16)
      auf der 7900 XTX. Gate: mean Val-SDR ≥ 0 dB (S1-Ziellücke: −1,7 dB) +
      Never-worsen vs. Zero-Shot je Lücke.
    - VOCAL-INPAINT-S3: IN-V1/V2-Naht-Gates + Verdrahtung in phase_55 (ersetzt
      die Drosselung bei vocal_confidence ≥ 0,40).
      **Status 2026-09-14: VORBEREITET** — `backend/core/dsp/diffwave_torch_inpaint.py`
      (Torch-Runtime: Finetune-Checkpoint, DDIM 50 Schritte, input-abgeleiteter
      Seed §G5 (GEBOTE.md)) + phase_55-Verdrahtung (`_try_diffwave_vocal` in der
      Kaskade nach GaCELA, Entdrosselung in `_derive_safe_inpainting_strength`,
      Metadaten `diffwave_vocal_fill_ready`/`diffwave_vocal_used`).
      **Aktivierungsvertrag:** erst bei vorhandenem F1-Checkpoint
      (diffwave_vocal_ready()) — bis dahin bleibt die Drosselung der Status quo.
      Naht-Gates IN-V1/V2 laufen generisch nach der Kaskade (bestehend).
      14 Tests grün (Modul + Wiring).
    - VOCAL-INPAINT-S4: Verifikation — ΔSDR ≥ 0 je Segment, VQI/Sänger-Identität
      per Resemblyzer-Witness unverändert, Determinismus.
18. **SOTA-BSR-GPU** · BSRoFormer/MelBandRoFormer auf ROCm-GPU freigeben (Stem-Trennung
    läuft aktuell auf CPU — funktional korrekt, aber langsam). Beleg 2026-09-13
    (`backend/core/gpu_model_registry.json`, `scripts/onnx_gpu_compat_scan.py`):
    (a) `bs_roformer_317_core.onnx`: ROCm-EP 5,4× schneller (189 ms vs. 1025 ms CPU),
    aber EP-Numerik weicht ab (rel=6.18 vs. CPU) → Verdict cpu;
    (b) `melbandroformer_optimized.onnx` (das tatsächlich geladene Modell):
    Scan scheiterte (CPU-Load NameError) → keine Messdaten → fail-closed cpu.
    Schritte:
    - BSR-GPU-S1: 🟡 DIAGNOSE ABGESCHLOSSEN (02af74003, 427089b4, 31ac7c00) —
      Softmax-Kernel als erster defekter Op identifiziert (rel 3,5 an Knoten 454) + Fix-Modul
      `onnx_softmax_rewrite.py` (6 Tests grün). ABER die ROCm-EP-Kernels sind grundsätzlich
      numerisch defekt für dieses Opset-17-Modell: Nach dem Softmax-Fix bricht die RoPE-Kette
      (vorher sauber! → EP-Knoten-Zuweisung ist graph-kontextabhängig); `tunable_op_enable=0`
      und Graph-Optimizer-aus ändern nichts (rel bleibt 6,11), als Provider-Option platziert der
      EP gar nicht mehr. Fazit: Fix auf ORT-Ebene nicht möglich — braucht einen Upstream-Fix im
      ONNX-Runtime-ROCm-EP. Bisektions-Werkzeug + Rewrite-Modul bleiben für später (ORT-Update).
    - BSR-GPU-S2: ✅ Scan läuft auf dem Original (NameError weg): CPU 4845 ms, ROCm 474 ms
      (10,2×), aber rel=6,11 → Verdict cpu bestätigt.
    - BSR-GPU-S3: ❌ NICHT ERREICHT — solange rel > 1e-3 bleibt CPU korrekt (fail-closed).
    Akzeptanz: Scan-Verdict `rocm` mit rel ≤ 1e-3; Determinis­mus auf GPU
    (gleicher Input + Device ⇒ bit-identisch, §G5); GPU-Lauf ≤ CPU-Laufzeit;
    SLR-VQI-Gate ≥ 0,72 weiterhin bestanden.

    **Update 2026-09-13 (Abschluss):** Statt ORT-ROCm-EP wurde ein eigener
    PyTorch-ROCm-Pfad umgesetzt — er ist **live und verifiziert**:
    - `backend/core/dsp/bsr317_torch_rocm.py` lädt den Original-Checkpoint
      `bs_roformer_317.ckpt` als reines PyTorch-Modul (`BSRCore`, kein
      onnxruntime) und separiert in 30-s-Chunks (STFT/ISTFT 2048/512,
      scipy-`even`-Paarung — exakt invertierbar, entspricht der
      torch-Reflect-Padding-Konvention des Original-Modells).
    - Maske ist (1,1,2050,T,2): je Kanal eine 1025-Bin-Hälfte; Kanal 0 wird
      für den Mono-Input genommen (Fix vom 2026-09-13 — vorher Broadcast-Fehler).
    - Plugin `bs_roformer_plugin.separate()` probiert zuerst den PyTorch-ROCm-
      Core (nur mit `torch.cuda.is_available()`), fällt bei jeder Exception
      warnend auf den ONNX-CPU-Pfad zurück (fail-closed, §V6).
    - **Paritäts-Beweis** (Benchmark, 7900 XTX): Torch-ROCm vs ONNX-CPU
      max_abs ≈ 1e-5 (Kern), End-to-End-Stems 4,9e-9; Summen-Invariante
      v+i=x auf 3,7e-10. Speedup **41,8×** (461 ms vs 19,3 s für t=512).
    - 6 Unit-Tests grün (`tests/unit/test_bsr317_torch_rocm.py`), GPU-Smoke
      `model_used=bs_roformer_317_torch_rocm` mit beiden Stems bestanden.
    - Offen bleibt nur S3 im ONNX-EP-Sinn (numerisch defekte ORT-ROCm-Kernels)
      — durch den PyTorch-Pfad funktional überholt, ONNX-CPU bleibt Fallback.

---

## Modell-Portfolio für destruktive Fälle (maximaler Wohlklang)

> Ziel: SOTA-Restaurierung des maximalen Wohlklangs — jede destruktive Klasse
> bekommt das beste musik-natives Modell mit Redundanz in der Kaskade und
> Witness-Absicherung (Hörordnung: Never-worsen, Naht-Gates, Sänger-Identität).

### Lokal verifiziert (stat-geprüft 2026-09-13)

- **AudioLDM2** — `models/audioldm2/audioldm2.onnx` (1390,9 MB) + `vae_decoder.onnx`
  (132,0 MB). Zero-Shot-Baseline für VOCAL-INPAINT. ONNX-Satz inspiziert
  (2026-09-13): `audioldm2.onnx` = UNet (IN: sample[B,8,H,W], timestep[1],
  encoder_hidden_states_0 [B,T,768] = FlanT5, encoder_hidden_states_1 [B,T,1024]
  = CLAP, attention_mask; OUT: out_sample[B,8,H,W]); `vae_decoder.onnx` = latent
  [B,8,H,W] → mel_spectrogram [B,1,bins,T].
  **FS-KORREKTUR (2026-09-14):** Die ONNX-Dateien sind REAL (stat-geprüft,
  Sessions geladen, Kontrakte verifiziert) — der frühere „0-Bytes-Artefakt“-
  Befund betraf nur das fehlende Plugin. **PLUGIN-ENTSCHEIDUNG (2026-09-14):
  Neuanlage verschoben** — FlanT5-Text-Conditioning ist nicht lokal
  (nur CLAP-Audio wäre nutzbar); die Rolle als VOCAL-INPAINT-Baseline ist
  durch S1 obsolet, und der GPU-Finetune F1 (DiffWave, HAUPTWEG) läuft mit
  Priorität. Reaktivierung, sobald ein Anwendungsfall mit CLAP-Audio-
  Conditioning steht. Voraussetzungen präzisiert:
  (1) Vocoder ✓ **vorhanden**: `models/hifi_gan/hifi_gan.onnx` (37,3 MB,
  validiert: IN [1,80,seq] → OUT [1,1,2560] = 116-ms-Chunks, gleitend zu fahren)
  + `plugins/hifigan_plugin.py` (PLM „HiFiGAN“, 22.05 kHz) — **Achtung
  Mel-Bin-Adapter:** HiFiGAN erwartet 80 Bins, AudioLDM2-VAE liefert 64 Bins
  → Bin-Mapping/Re-Projektion im Plugin nötig; (2) Conditioning-Embeddings
  (CLAP-Audio/FlanT5) müssen extern erzeugt werden — CLAP-Encoder lokal ✓ (C4),
  als Embedding-Quelle verdrahten; FlanT5 fehlt lokal;
  (3) VOCAL-INPAINT-S1-Benchmark um AudioLDM2-Arm ergänzen.
- **Resemblyzer** — `models/resemblyzer/resemblyzer/pretrained.pt` → **ONNX
  exportiert** `models/resemblyzer/resemblyzer_voice_encoder.onnx` (2026-09-13,
  opset 17, dynamische Zeitachse, Parität cos=1.0000). Sänger-Identitäts-
  Witness für VOCAL-INPAINT-S4. **ERLEDIGT (2026-09-13): Witness-Verdrahtung**
  in die Hörordnungs-Kette: `level_1_invariants_guard.py` maß die
  Stimm-Identität bislang über einen Import des nicht existierenden Pakets
  `Resemblyzer` (Großbuchstabe) — der ML-Pfad lief nie, stiller DSP-Ersatzpfad
  (§V74 (VERBOTEN.md)-Verstoß). Fix: Plugin-Kaskade Package→ONNX→None als
  primäre Methode (UV3-Post-Phase → Ebene-1-Guard → Witness); zusätzlich
  `cosine_similarity` NaN-sicher gemacht (np.nan_to_num am Entry).
  Tests: 12 Fälle test_resemblyzer_onnx_fallback.py + 7 Fälle
  test_level_1_guard_resemblyzer_wiring.py (inkl. echtem ONNX-Witness:
  identisch ⇒ cos=1.0); bestehende Guard-Suite (24 Tests) läuft weiter grün.
- **MuQ (Musik-MOS)** — bereits eingebunden: `plugins/muq_plugin.py` (lädt
  `OpenMuQ/MuQ-large-msd-iter` aus HF-Cache) + `models/muq_mulan/muq_mulan.onnx`
  + A1-Head. Musik-nativer MOS-Witness — Ersatz für UTMOS (Negativbefund:
  richtungs-invertiert auf Musik). **NEGATIVBEFUND 2026-09-13 (MUSDB-
  Richtungs-Validierung, Muster ML-V4):** Head-Extraktion repariert
  (`models/muq_mulan/muq_eval_a1_head.pt` aus HF-Cache `zhudi2825/MuQ-Eval-A1`
  extrahiert + Device-Fix cuda) — Inferenz läuft end-to-end, aber die
  MOS-Richtung ist **invertiert**: band12 Δ−0.007, band8 Δ−0.063,
  noise10 Δ−0.333, noise0 Δ+0.375 (6 Tracks × 10 s; Degradation wird teils
  BESSER bewertet als ref). Ursache: A1-Head wurde auf dem MuQ-Eval-
  Backbone trainiert, läuft hier aber auf MuQ-large-msd-iter. **Konsequenz:**
  MuQ-MOS ist bis zur Backbone-Korrektur KEIN Richtungs-Witness (§V6-
  DSP-MOS bleibt aktiv). **Root-Cause-Analyse (2026-09-13, abgeschlossen):**
  Head-Extraktion ist bit-identisch mit `best_model.pt` (max|diff|=0.0);
  `base.yaml` des A1-Snapshots bestätigt exakt die Plugin-Konstanten
  (encoder_id = `OpenMuQ/MuQ-large-msd-iter`, encoder_dim 1024, 24 kHz,
  10-s-Clips, attention-pooling, 2×256-MLP) — der A1-Encoder ist FROZEN,
  es gibt keinen Backbone-Gewichts-Unterschied. Verbleibender Unterschied:
  die Plugin-Reimplementierung (vendored `_vendor_muq`-Forward + eigenes
  Pooling) vs. die MuQ-Eval-Modellklasse. **Nächster Schritt:**
  `MusicQualityModel` aus `models/muq_eval/src/model.py` mit der A1-Config
  instanziieren, `best_model.pt['model_state']` strict laden und den
  MuQ-Eval-Evaluierungs-Forward 1:1 nutzen (encoder → pooling → heads['MI']);
  zeigt DAS die richtige Richtung, liegt der Unterschied im
  Feature-Fluss des Plugins; wenn nicht, ist der A1-Head für MUSDB-
  Degradationen nicht richtungs-stabil (Domain-Gap) und MuQ-mulan
  (SRCC 0.957) bleibt der Embedding-Witness bei DSP-MOS als Richtungs-Witness.
  **VALIDIERUNG 1:1 (2026-09-13):** `MusicQualityModel` (models/muq_eval/src)
  mit base+A1-Config (OmegaConf-Merge) instanziiert, `best_model.pt['model_state']`
  **strict geladen (0 fehlend/0 überzählig)**, GPU-Forward 1:1 → **Richtung KORREKT:**
  noise10 Δ+3.337 (6/6), noise0 Δ+3.270 (6/6), band12 Δ0.000 (neutral, bei 24 kHz
  unhörbar), band8 Δ−0.032 (uneinheitlich). Damit ist bewiesen: Der A1-Head ist
  auf MUSDB richtungs-stabil; die Plugin-Reimplementierung (vendored-MuQ-
  Forward-Details) verursacht die Inversion. **Nächster Schritt:** den
  MuQ-Eval-Forward in `muq_plugin.py` nachbilden (encoder-Aufruf + Pooling 1:1
  wie in models/muq_eval/src/encoders.py, ggf. MusicQualityModel direkt
  einbetten und den bereits geladenen MuQ-large-Encoder wiederverwenden),
  dann Plugin-Re-Validierung auf MUSDB.
  **UPDATE (Backbone-Fix getestet):** msd-iter-Snapshot-Priorisierung eingebaut
  (`_find_checkpoint_dir`, 2026-09-13) — Plugin-Werte unverändert (noise10
  Δ−0.333 identisch) → `models/muq_mulan/` enthält dieselben msd-iter-Gewichte
  (Kopie); der Unterschied zum richtungs-korrekten 1:1-Test liegt in der
  Plugin-Audio-Kette (`_center_window` + torchaudio-Resample 44.1k→24k vs.
  1:1-Kette librosa-Resample + erste 10 s).
  **ERLEDIGT + RICHTUNGS-VALIDIERT (2026-09-13):** Plugin-Audio-Kette auf die
  1:1-Kette angeglichen — neues `_mos_eval_window` (ERSTE 10 s, librosa-Resample
  24 kHz, Null-Pad bei kurzen Eingaben) ersetzt das zentrierte 20-s-Fenster im
  MOS-Pfad; Embedding-Pfad unverändert. Re-Validierung
  `scripts/validate_muq_plugin_direction.py` (1:1-MusicQualityModel +
  best_model.pt strict vs. Plugin auf identischen MUSDB-Paaren, 3 Tracks):
  **3/3 Richtungen korrekt** — noise10 Δ+2.22 vs. +2.21, noise0 Δ+3.66 vs. +3.24
  (Plugin-Floor bei MOS 1.0 erklärt die Restdifferenz), band8 Δ−0.28 vs. −0.23,
  band12/ref neutral. MuQ-MOS ist damit als Richtungs-Witness bestätigt;
  der 50/50-Blend im `restorability_estimator` bleibt (Metriken sind Zeugen,
  Hörordnung §8a). 5 Tests in test_muq_plugin.py ergänzt. Report:
  docs/reports/current/2026-09-13_muq_plugin_direction.json.
  Der MuQ-mulan-Embedding-Pfad ist vom Befund unberührt.
- **RVC-Basis** — `models/rvc/hubert_base.pt` + `rmvpe.pt` (HF-Beschaffung
  2026-09-13, MIT) + **bestehende ONNX**: `models/rmvpe/rmvpe.onnx` (128-mel,
  360 F0-Klassen) und `models/hubert/hubert_model.onnx` (768-dim, validiert).
- **BigVGAN-v2** — `models/bigvgan/bigvgan_v2.onnx` + `.data` + `bigvgan_v2.pth`.
  Finetune-Basis für Repair-Pfad (ML-V2/HR-V1).
- **Miipher-DiT (MIIPHER_DiT)** — `models/miipher_dit/flow_matching_dit.onnx`
  (806,6 MB, bestätigt 2026-09-13). Ersetzt das proprietäre Google-MIIPHER
  (Spec v10.14: ❌ Proprietär → ✅ Open-Source Flow-Matching-DiT).
  `plugins/miipher_dit_plugin.py` lädt das ONNX aktiv (18 Layer, 768-dim,
  12 Heads, unkonditioniert x[B,T,1]+t[B]); Halluzinations-Guard
  `spectral_novelty > 0.35 → Rollback`, §V6-DSP-Fallback (IMCRA/Wiener),
  PLM `MIIPHER_DiT`. Das alte `models/miipher/miipher.onnx` existiert nicht
  mehr — `plugins/miipher_plugin.py` ist nur noch der Router.
  **Spec-Warnung beachten:** `phase_42` überspringen, wenn
  `route_token == "miipher_dit"` (sonst Overprocessing, DiT läuft in Phase 03).

### Portfolio-Status (Rolle → Modell → Stand)

| Rolle | Modell | Stand |
|---|---|---|
| Vokal-Langlücken | DiffWave-Vokal-Finetune (HAUPTWEG) | Checkpoint lokal, **GPU-Finetune fehlt** (S1: −1,7 dB → Ziel ΔSDR ≥ 0) |
| Vokal-Langlücken 375–1500 ms | GaCELA-Vokal-Finetune (Pfad B) | Trainingscode lokal, **GPU-Finetune fehlt** |
| Vokal-Extremfälle/Identität | RVC (MIT) | rmvpe.pt/.onnx ✓, hubert_base.pt ✓, hubert_model.onnx ✓ |
| Vokal-Enhancement (Phase 03) | Flow-Matching-DiT (MIIPHER_DiT) | flow_matching_dit.onnx ✓, Plugin ✓, Guard ✓ — **bestätigt** |
| Generative Baseline | AudioLDM2 | ONNX lokal ✓, **Plugin fehlt** |
| Repair (Spektralregionen) | BigVGAN-v2 Musik- + Vokal-Finetune | Basis lokal ✓, **GPU-Finetune fehlt** |
| Hochband-Rekonstruktion | FlashSR-Musik-Finetune | Checkpoint lokal ✓, **GPU-Finetune fehlt** |
| Musik-MOS-Witness | MuQ | Plugin + mulan-ONNX ✓ — **RICHTUNGS-VALIDIERT (2026-09-13)**: Audio-Chain-Fix (10-s-Eval-Kette), 3/3 MUSDB-Richtungen korrekt, 50/50-Blend aktiv |
| Sänger-Identitäts-Witness | Resemblyzer | ONNX lokal ✓ — **VERDRAHTET (2026-09-13)** im Ebene-1-Guard (Plugin-Kaskade; toter „Resemblyzer“-Import ersetzt) |
| Dereverb-Steuerung (DR-V1) | RT60 via DeepFilterNet-v3.II | **V1-VERDRAHTET (2026-09-13)**: Schröder-T30-Witness + Delta-Gates in phase_20/49 (konservativ, Cap +0,35; präziser Regressor = GPU-Folgeschritt) |
| Zeit-Warp-Rest (WF-V4) | Neuraler Warp-Schätzer 2025/26 | **Quelle klären + Download fehlt** |
| EQ/Dynamik-Prädiktion (C4) | DDSP-Prädiktor auf CLAP/BEATs | Encoder lokal ✓, **Prädiktor-Training fehlt** |
| Transienten-Phasen (TP-V2) | Neurale Phasen-Schätzung | **Modell + Quelle fehlen** |

### Beschaffungs- und Einbindungs-Reihenfolge (maximaler Wohlklang)

1. **Sofort einbindbar (lokal):** Resemblyzer-Witness ✓ VERDRAHTET (Ebene-1-Guard) →
   RT60/DeepFilterNet-Verdrahtung ✓ V1 (phase_20/49, konservativ) →
   MuQ-Richtungs-Validierung ✓ 3/3 — **offen bleibt:** AudioLDM2-Plugin-Neuanlage.
2. **Beschaffung (Downloads):** RVC-Basis ✓ (rmvpe + hubert, ONNX vorhanden),
   WF-V4-Checkpoint (Quelle klären), TP-V2-Modell (Quelle klären).
3. **GPU-Finetunes (7900 XTX, MUSDB18HQ lokal):** DiffWave-Vokal → GaCELA-Vokal
   → BigVGAN-v2 (Musik+Vokal) → FlashSR-Musik → DDSP-Prädiktor.
4. **Governance:** Alle Einbindungen über Never-worsen-Gate + IN-V1/V2-Naht-Gates
   + Witness-Absicherung; Benchmarks nach dem Muster ML-V1-VORAB/VOCAL-INPAINT-S1.

### GPU-Finetune-Plan (7900 XTX, ROCm)

Daten: MUSDB18HQ lokal; Evaluations-Gate je Finetune = ΔSDR ≥ +2 dB gegenüber
jeweiliger Zero-Shot-Baseline auf den destruktiven Fällen + Never-worsen auf
sauberen Referenzen (Muster ML-V3-Negativbefund).

| # | Finetune | Basis | Ziel | Daten | Gate |
|---|---|---|---|---|---|
| F1 | DiffWave-Vokal (HAUPTWEG) | DiffWave-Checkpoint lokal | Langlücken-Reparatur ΔSDR ≥ 0 (S1: −1,7 dB) | MUSDB-Vocals | P1-Metrik |

**F1-Ergebnis 2026-09-14: ABGEBROCHEN — Ansatz ausgereizt** — zwei Läufe plafonierten
bei mean −1,44 dB bzw. −1,39 dB (Gate ≥ 0 dB): Der Wellenform-DDPM aus der
Sprach-Domäne ist für 300-ms-Gesangs-Inpainting am Limit (Retest oszillierte
−1,39 → −7,41 dB — instabile Konvergenz). Checkpoints archiviert
(`diffwave_vocal_ft_rejected_epoch{2,7}_*.ckpt`), S3 bleibt inaktiv.
Report: docs/reports/current/2026-09-14_diffwave_vocal_finetune.json.

**Erfolgreichere Alternativen (Entscheidung 2026-09-14, nach Evidenz):**
1. **F2-Verlängerung = HAUPTWEG** — GaCELA-GAN arbeitet spektral und musik-nativ;
   nach nur 10 Epochs from-scratch bereits mean −0,37 dB. 30–50 weitere Epochs
   kreuzen das Gate sehr wahrscheinlich (GPU frei).
2. **S1-Nachmessung der phase_55-Kaskade (Q11, billigster Beweis)** — CQTdiff+,
   Consistency-Modell und GaCELA sind für Gesangslücken bereits aktiv; S1 hat nur
   DiffWave + DSP gemessen. Erreicht die bestehende Kaskade ≥ 0 dB, ist das
   F1-Problem OHNE neues Training gelöst.
3. **F6: AudioLDM2-Latent-Inpainting (RePaint-Muster)** — UNet + VAE lokal
   (echte ONNX, verifiziert); maskierte Latent-Inpainting ohne Text-Konditionierung
   (CLAP-Audio oder unkonditioniert) + HiFiGAN (80-Bin-Adapter fehlt) = SOTA-Weg
   mit größerer Hebelwirkung als DiffWave-Wellenform-DDPM.
4. RVC bleibt „später“ (Konversions-Semantik, kein Inpainting).

**Q11-ERGEBNIS 2026-09-14 — GATE DURCH BESTEHENDE KASKADE ERFÜLLT:**
Die S1-Nachmessung der phase_55-Kaskade auf 300-ms-Vokallücken (3 Tracks ×
13 Lücken, Seed 42) zeigt: **CQTdiff+ mean SDR 0,0 dB** (je Lücke −0,12 … +0,1 dB;
Gate ≥ 0 dB ERFÜLLT) vs. DSP −3,46 dB und DiffWave-Zero-Shot −1,68 dB.
**Konsequenz:** Das Vocal-Inpainting-Gate ist OHNE jedes Training gelöst —
F1 bleibt abgebrochen, S3 (DiffWave-Pfad) bleibt inaktiv, F2-Verlängerung
wird zur QUALITÄTS-Verbesserung (Ziel: deutlich über 0 dB statt am Gate).
**Offene Folge-Frage (dokumentiert):** Die Drosselung bei vocal_confidence ≥ 0,40
kann mit dem Q11-Beleg neu bewertet werden (CQTdiff+-Füll statt Drosselung) —
konservativ erst nach einem Never-worsen-Test des schlechtesten Falls.
| F2 | GaCELA-Vokal (Pfad B) | **TRAINIERT 2026-09-14/15**: 30 Epochs (10→39) auf MUSDB-Vocals, Checkpoint 39_0499, S1-Val 9 Lücken | **S1-Validierung (2026-09-15): mean ΔSDR −2.46 dB vs. Ground-Truth (Gate ≥ 0 dB NICHT bestanden) ⇒ NICHT aktiviert** (Gate schließt korrekt fail-closed). Witness=None (Resemblyzer im Val-Lauf nicht verfügbar). Nächste Optionen: EAR-VAE-Rezept nachtrainieren, Gate auf ΔSDR vs. DSP-Baseline kalibrieren, oder Pfad B verwerfen | MUSDB-Vocals | P1-Metrik |

**F2-Vorbereitung 2026-09-14:** `scripts/train_gacela_vocal_inpaint.py`
(Upstream-Trainingsspiegel + MUSDB→22,05-kHz-WAV-Datenpfad, --data-check
2 Tests grün). **ENTBLOCKT (2026-09-14, Nachtrag):** Der tifresi/ltfatpy-
Blocker (C-Build, kein Binary-Wheel) ist über einen eigenen NumPy-Gabor-Shim
(`scripts/gacela_gabor_shim.py`, Truncated-Gaussian 1024/256, Forward-DGT +
log_spectrogram/preprocess_signal per sys.modules-Injektion) umgangen — der
Upstream-TrainDataset liefert damit end-to-end korrekte Spektrogramme
(Data-Check grün, 6 Shim-Tests). Kein Vendoring (tifresi-Lizenzlage
MIT/GPLv3 unklar); die Inversion (PGHI) wird vom Training nicht benötigt.
Nächster Schritt: F2-Smoke auf der GPU nach F1.

**F2-LAUF 2026-09-14: ABGESCHLOSSEN** (EXIT=0, 10 Epochs, 100 Tracks, ~2,5 h,
batch 64, Checkpoints `output/gacela_f2/gacela_vocal_ft_checkpoints/`; zusätzliche
Loop-Fixes: num_workers=0 [Fork-Deadlock], librosa-mel-API, CUDA-dtype im Shim,
essentia-Summaries aus, Epoch-Länge 10 Items/Datei, Save-Intervall 1000).
**Validierung** (`scripts/validate_gacela_vocal_inpaint.py`, 3 Test-Tracks × 3 native
743-ms-Lücken, Seed 42, Report `docs/reports/current/2026-09-14_gacela_vocal_val.json`):
**mean SDR −0,37 dB — Gate ≥ 0 dB KNAPP VERFEHLT** (nach 10 Epochs from-scratch
bereits nahe an der Never-worsen-Linie; Einzelwerte +0,19 … −2,74 dB).
Witness: 0,74-s-Segmente sind für den Resemblyzer-VAD zu kurz (cos=None, best-effort).
**Nächster Schritt:** F2-Verlängerung auf 30–50 Epochs (der Lauf war der erste
from-scratch-Wurf; das Gate wird bei weiterem Training erwartbar gekreuzt).
| F3 | BigVGAN-v2 Musik+Vokal | bigvgan_v2.pth lokal | Spektral-Repair ML-V2/HR-V1 | MUSDB-HQ | ΔSDR ≥ +2 dB |
| F4 | FlashSR-Musik | Checkpoint lokal | Hochband-Rekonstruktion (aus ML-V1-VORAB: Kandidatenwahl) | MUSDB-HQ | ΔSDR ≥ +2 dB |
| F5 | DDSP-Prädiktor (C4) | CLAP/BEATs-Encoder lokal | EQ/Dynamik-Prädiktion | MUSDB-HQ + Effekt-Paare | MOS-Witness (MuQ) |

**Reihenfolge:** F1 → F2 → F3 → F4 → F5 (je nach VRAM 1–2 parallel); jeder
Finetune committet nur mit Witness-Belegen (Resemblyzer cos ≥ 0.92 für
Sänger-Identität, MuQ-MOS nicht schlechter als Baseline).

---

## SOTA-EXPANSIONSMATRIX 2026-09-14 — Hörfähigkeits-Ausbaustufe (alle Phasen)

> Vollständige Lücken-Inventur (außer F1/F2): Was fehlt, um jede Hybrid- und
> jede pure DSP-Phase auf maximale SOTA-Ausbaustufe für das menschliche Gehör
> zu heben — modelliert nach Hörordnung §4 (Maskierungsschwelle als
> Reparaturziel) und §8b (interaurale Hörfähigkeiten). Quellen der Ist-Werte:
> Verdrahtungs-Inventur 2026-09-14 (grep über backend/core/phases).

### A. Psychoakustische Querschnitts-Maßnahmen („Gehör nachbilden“)

| ID | Maßnahme | Ist (2026-09-14) | Ziel/Wirkung |
|---|---|---|---|
| PSY-A1 | **Audibility-Gate-Rollout**: Maskierungsschwelle (masking_model, ISO 11172-3 Bark) als Reparatur-Entscheidung in ALLEN reparierenden Phasen (01, 03, 06, 07, 08, 19, 23, 27, 36, 50, 55, 56, 59, 64, 65, 66) | **TEIL-ROLLOUT 2026-09-14**: `audibility_gate.py` (defekt-zentrierte Messung, §V6 (copilot-instructions.md)-fail-open) + Verdrahtung **phase_01, 03, 06, 07, 08, 19, 23, 27, 36, 50, 55, 56, 59, 64, 65, 66** (subaudible Klicks/Pops/Splices/Lücken/Band-Lücken/Sibilanten/Spektral-Defekte/Modulations-Rauschen überspringen, Noise-Floor dämpfen, Vokal-/Stem-Delta gate-n); 7 Gate- + 45 phase_56- + 17 phase_64-Tests grün. **Gate-Fix dabei:** lange Defekt-Regionen werden ZENTRAL (Mitte) statt am Anfang gemessen. **06/07/08/36 erledigt 2026-09-14** — Rollout vollständig | §4-Vertrag: „Ist der Defekt über der Maskierungsschwelle hörbar?“ als Pflicht-Frage; Berichte weisen „hörbar“ als über-Schwelle aus |
| PSY-A2 | **Zwicker-Modell (ISO 532-1)** als Nachfolger des MPEG-1-Modells: stationäre + zeitvariante Loudness und Maskierung | stationäres Zwicker ISO 532-1 vorhanden + optional im Audibility-Gate; zeitvariantes Loudness bleibt offen | präzisere Schwelle bei tonalem/breitbandigem Material; Basis für PSY-A7 |
| PSY-A3 | **BMLD-Verdrahtung** (binaurale Maskierungs-Freisetzung) in die Stereo-Phasen-Gates (13, 15, 33, 34, 46, 48) | binaural_masking auf Guard-Ebene + phase_03; **2026-09-14: phase_33- UND phase_34-BMLD-Witness** (Metadaten release_db/nr_floor_release_db/ec_gain_db, ZEUGE-Modus — dynamische Freisetzungs-Toleranz bleibt Folge-Schritt); 13/15/46/48 offen | Stereo-Änderungen werden nach Hör-Freisetzung bewertet statt nach Mess-dB |
| PSY-A4 | **Equal-Loudness-Band-Gewichte (ISO 226) + JND-Gates** für EQ-/Enhancement-Phasen (04, 16, 17, 37, 38, 39) | fletcher_munson nur 1× verdrahtet; **2026-09-14: phase_37-Bass-Mix mit Equal-Loudness-Faktor temperiert (ISO 226, 60 phon, Deckel [0,5–1,0])**; 04/16/17/38/39 offen | Stärke-Entscheidungen in Phon-Hörbarkeit statt Roh-dB |
| PSY-A5 | **Temporal-Masking-Kompensation**: Forward/Backward-Masking-Zonen als Reparatur-Dämpfung nach Transienten (Rollout des §V41-Musters aus phase_55) | nur phase_55 + 22 Phasen nutzen temporal_masking (Guard); **2026-09-14: phase_01/27/64 dämpfen Klick-/Pop-/Splice-Reparatur in Forward-Masking-Zonen (×0,6)** — kein Nachschlag-Artefakt | Nachmaskierungs-Zonen werden weicher repariert — kein Nachschlag-Artefakt |
| PSY-A6 | **Personalisierte HRIR (CIPIC)** als Ausbaustufe der First-Order-HRTF (Ehrlichkeits-Klausel §8b.3 erfüllt) | First-Order-Modell in interaural_cues | bessere Bühnen-Bewertung bei Kopfhörer-Studien (P1-4) |
| PSY-A7 | **Loudness-Modell-getriebene Dynamik**: Zwicker-Kurzzeit-Loudness steuert 10/11/40/47 | BS.1770-integriert + Bark-LUFS vorhanden, Kurzzeit-Modell nicht in den Phasen | Punch/Lautheit nach Wahrnehmung statt Peak |
| PSY-A8 | **Generische JND-Gate-Tabelle** (Frequenz ±1 dB, Pegel ±1 dB, Zeit ±5 ms, Pan ±2°… nach Lit.) für alle Never-worsen-Gates | **ERLEDIGT (2026-09-14)**: `backend/core/dsp/hearing_jnd.py` — 9 JND-Klassen mit Quellen, `below_jnd()` (NaN-fail-safe), fail-closed bei unbekannter ID; 6 Tests | absolute dB-Gates weichen JND-basierten, hörbezogenen Grenzen |

### B. Hybrid-Phasen: offene SOTA-Maßnahmen

| Phase | Ist | Offene SOTA-Maßnahme |
|---|---|---|
| 03 denoise | BSR-Stem-NR (66) + DSP-Kaskade; MP-SENet de-wired (Negativbefund) | musik-nativer Denoiser-Finetune (F3 BigVGAN-Spektrallinie; DeepFilterNet-Musik-Finetune als Alternative prüfen) |
| 04/16/17 EQ/Dynamik | DSP + Zielkurven | **SOTA-C4**: DDSP-Prädiktor (CLAP/BEATs-Embeddings → EQ/Dynamik-Parameter, DSP führt aus) — GPU (F5); CLAP-Encoder lokal vorhanden |
| 07 harmonisch | DSP-Harmonic-Restoration | **SOTA-HR-V1**: BigVGAN-Repair-Pfad + additive_synthesis_gate (bigvgan_v2.pth lokal) — GPU (F3) |
| 08/36 Transienten | Superflux/Envelope + BEATs-Witness (TP-V1, V1) | **SOTA-TP-V2**: neurale Phasen-Schätzung für Transienten-Frames (Modell + Quelle klären); adaptive Schwelle aus TP-V1-Konsens |
| 19/43 De-Esser | DSP (19) + ML-Deesser (43) | Sibilanten-Maskierungs-Gate (PSY-A1) — sonst SOTA |
| 20/49 Dereverb | DR-V1-Verdrahtung (RT60-Witness, konservativ gedeckelt) | präziser neuraler RT60-Regressor (GPU-Training); PSY-A1-Gate für Dry/Wet |
| 23/50 Spektral-Repair | DSP/NMF | neuronale Spektral-Inpainting-Basis (F3/F4) |
| 55 Inpainting | Kaskade (FlowMatching, Consistency, CQTdiff+, DAC, GaCELA, DiffWave-S3-vorbereitet) | **F1/F2 laufen**; danach S4-Verifikation |
| 58 Lyrics-guided | vorhanden (DSP-geführt) | CLAP-Audio-Conditioning für Text-/Semantik-Führung (C4-Embedding-Quelle) |
| 66 Stem-NR | BSR-Stems + Per-Stem-Kette ✓ | PSY-A1-Gate pro Stem; BSR-GPU-Beschleunigung ✓ (Torch-ROCm) |

### C. Pure-DSP-Phasen: offene SOTA-Maßnahmen

| Phase | Ist | Offene SOTA-Maßnahme |
|---|---|---|
| 01 Klicks | ✓ CR-V1 (BANQUET-Konsens) | PSY-A1-Gate (subaudible Klicks nicht zählen) |
| 02 Hum | ✓ HU-V1 (Kalman + LSQ) | PSY-A1-Gate |
| 05 Rumpel / 25 Azimut / 31 Speed-Pitch / 62 Crosstalk / 63 Intermodulation / 64 Splice | DSP-Stand (funktional) | PSY-A1-Gates; 64: GaCELA/DiffWave-Fill-Reuse prüfen |
| 12 Wow-Flutter | ✓ WF-CASS (30-Hz-AM); DSP-Kompensation | **SOTA-WF-V4**: neuraler Warp-Schätzer (Checkpoint-Quelle klären + Download) |
| 13/15/33/34/46/48 Stereo/Bühne | DSP + interaural-Guard | PSY-A3 (BMLD-Gates); PSY-A6 (HRIR-Ausbaustufe) |
| 32 Mono→Stereo | DSP-Upmix | optional: BSR-Stem-basierte Verbreiterung (musik-nativ) |
| 37/38/39 Bass/Presence/Air | DSP | PSY-A4 (Equal-Loudness-JND-Gewichte) |
| 40/47 Loudness/TP | BS.1770 ✓ | PSY-A7 (Zwicker-Kurzzeit-Loudness-Steuerung) |
| 59 Modulationsrauschen | DSP | SOTA-D3+D4 (niedrig): Multiresolution-Split + APPLADE-Maskierungs-Loss |
| 65 Vocal-Natürlichkeit | DSP + Resemblyzer-Witness (Ebene-1) | Sänger-Identitäts-Witness als Per-Phase-Gate (S4-Muster) |

### D. Witness-/Modell-Lücken (Qualitäts-Urteile)

| ID | Lücke | Status |
|---|---|---|
| WIT-M1 | **MuQ-MOS-Richtung invertiert** (A1-Head auf falschem Backbone — muq_eval_a1_head.pt läuft auf MuQ-large-msd-iter statt MuQ-Eval-Backbone) | **ERLEDIGT 2026-09-14** — Ursache war der RESAMPLE-FILTER: das Plugin nutzte librosa, MuQ-Eval torchaudio.functional.resample (Kaiser-Sinc); nach dem Fix (Eval-exakt zuerst) ist die Richtung **3/3 korrekt** (noise0 Δ+3,66 ref / Δ+3,20 Plugin; Report 2026-09-14_muq_plugin_direction.json). MuQ ist damit als MOS-Richtungs-Witness für die F-Gates nutzbar (10-s-Clips, Vollmix) |
| WIT-M2 | BEATs-Tagger-Head fehlt (Encoder-Export ohne Head) | Head auf Tokens trainieren (GPU) oder Tagger-ONNX beschaffen |
| WIT-M3 | UTMOS für Musik unbrauchbar (Negativbefund) | dokumentiert — kein Einsatz |
| WIT-M4 | F-Gates hängen an SDR/Resemblyzer (objektiv) | nach WIT-M1: MOS-Witness (MuQ) als dritte Gate-Stimme |

### E. Prioritäten-Empfehlung (nach GPU-Verfügbarkeit)

1. Jetzt (CPU): PSY-A1-Gate-Rollout beginnend bei phase_01/03/08; PSY-A3 in phase_33/34; PSY-A8-JND-Tabelle als Modul.
2. Nach F1/F2: S4-Verifikation → F3 (BigVGAN, bedient HR-V1 + 23/50) → F5 (C4) → WIT-M1 (MuQ-Backbone).
3. Parallel: WF-V4-Quelle klären, TP-V2-Quelle klären, BEATs-Tagger-Head (GPU).
4. Ausbaustufen (optional, dokumentiert): PSY-A2 (Zwicker), PSY-A6 (CIPIC), PSY-A7.

### E2. Umsetzungs-Warteschlange 2026-09-14 (alle Empfehlungs-Punkte)

| ID | Punkt | Status | Rezept/Aktion |
|---|---|---|---|
| Q1 | PSY-A5-Rollout phase_27/64 (Forward-Masking ×0,6) | **ERLEDIGT 2026-09-14** (phase_01/27/64; 26+17+12 Tests grün) | Muster: Zonen einmal pro Kanal via get_forward_masking_guard, Dämpfung im Reparatur-Loop |
| Q2 | PSY-A1 in phase_55 (subaudible Lücken nicht füllen) | **ERLEDIGT 2026-09-14** (defect_audibility-Gate vor der Kaskade, Zähler `subaudible_gaps_skipped`) | defect_audibility auf die Gap-Region, skip → Kaskade überspringen |
| Q3 | PSY-A7: Kurzzeit-Loudness-Steuerung für 40/47 | **V1-VERDRAHTET 2026-09-14** — phase_47 führt STL/LTL-Witness (Sone, Metadaten `short_term_loudness`); wahrnehmungs-basierter Cap bleibt Folge-Schritt | temporal_loudness() nutzt ERB-Kurzzeit-Modell (vorhanden) |
| Q4 | WIT-M1: MuQ-Backbone-Fix (MOS-Richtung) | **NACH F1 (GPU)** | MuQ-Eval-Backbone beschaffen ODER A1-Head auf msd-iter neu trainieren; dann Richtungs-Validierung (Muster validate_muq_plugin_direction.py) — erst danach MuQ als Gate-Stimme |
| Q5 | F2-Verlängerung 30–50 Epochs | **NACH F1 (GPU)** | `python scripts/train_gacela_vocal_inpaint.py --train --data-folder data/gacela_vocals_train --epochs 40 --batch 64 --save-path output/gacela_f2_v2/ --experiment-name gacela_vocal_ft` (lädt/startet neu; Checkpoint-Warmstart aus output/gacela_f2 prüfen) |
| Q6 | F3: BigVGAN (HR-V1 + 23/50 + 03) | **NACH F1 (GPU)** | bigvgan_v2.pth lokal; Torch-Runtime nach S3-Muster + phase_07-Verdrahtung mit Aktivierungsvertrag |
| Q7 | F5/C4: DDSP-Prädiktor | **NACH F1 (GPU)** | CLAP-Encoder lokal; EQ/Dynamik-Parameter-Prädiktion für 04/16/17, DSP führt aus |
| Q8 | PSY-A1-Rest (masken-basierte Phasen 19/23/50/56/59/65/66) | **ERLEDIGT 2026-09-14** (Phasen 19/23/50/59/65/66 nach phase_56-Muster verdrahtet) | Profil-Fallback-Refactor (phase_56-Befund: leere Maske ⇒ „repariere überall“) dann Gate je Phase |
| Q9 | PSY-A2 (Zwicker ISO 532-1) | **STATIONÄR ERLEDIGT** | präzisere Maskierungsschwelle — verbessert alle PSY-A1-Gates |
| Q10 | PSY-A6 (CIPIC-HRIR) | AUSBAUSTUFE | personalisierte Bühnen-Bewertung |

---

## MODELL-LIZENZEN (Governance 2026-09-14)

> Lizenz-Register der genutzten Modelle — rechtlich ehrliche Einordnung.
> Kein Rechtsgutachten; bei Kommerzialisierung durch Fachanwalt prüfen lassen.

| Modell | Lizenz | Kommerziell ok? | Anmerkung |
|---|---|---|---|
| DiffWave (philovivero) | MIT | ✅ | Port dokumentiert (diffwave_model.py); Finetune-Checkpoints dürfen unter eigener Lizenz stehen (MIT erlaubt Sublizenzierung, Hinweis beibehalten) |
| GaCELA (Upstream-Code) | MIT | ✅ | F2-Checkpoint wurde FROM SCRATCH (Random-Init) trainiert ⇒ eigene Gewichte; Architektur folgt MIT-Code |
| BEATs (Microsoft) | MIT (Code+Checkpoints) | ✅ | iter3-ONNX lokal |
| BS-RoFormer | MIT | ✅ | |
| MuQ / MuQ-Eval-A1 | MIT | ✅ | |
| CQTdiff+ | **MIT** (verifiziert 2026-09-14, models/cqtdiff/LICENSE) | ✅ | |
| BigVGAN (NVIDIA) | MIT | ✅ | |
| HiFiGAN (jik876) | MIT | ✅ | |
| PANNs Cnn14 | MIT | ✅ | Ersatz-Tagger |
| Resemblyzer | MIT | ✅ | |
| DeepFilterNet | MIT | ✅ | |
| AudioLDM2 | **NC-restriktiv** (Model-Card, Forschung) | ❌ | Kommerzielle Nutzung der exportierten ONNX klären bzw. durch eigenes Training ersetzen (F6) |
| essentia | AGPL-3.0 | ⚠️ | NUR Trainings-Zeit (Tensorboard-Summaries), nicht im Produkt-Code — bei Distribution der Trainings-Umgebung prüfen |
| MUSDB18/HQ (Trainingsdaten F1/F2) | CC BY-NC-SA 4.0 | ❌ | **Daten-Lizenz:** auf NC-Daten trainierte Gewichte erben die Einschränkung — für kommerzielle Modelle auf eigenes/lizenziertes Vokalmaterial wechseln |
| RVC/hubert | MIT-Code / hubert NC | ❌ | daher „später“ |

**Leitlinien (rechtlich tragfähige Wege zu einem EIGENEN Modell):**
1. **From-Scratch + eigene Daten** = bulletproof: eigenes Training auf selbst
   lizenziertem Material, Architektur aus MIT-Code ⇒ Checkpoint trägt deine Lizenz.
2. **MIT-Basis + Finetune** = erlaubt: MIT gestattet Modifikation UND
   Sublizenzierung — du darfst den Finetune unter eigener Lizenz veröffentlichen,
   MIT-Text + Copyright-Hinweis der Basis beibehalten.
3. **NC-Basen/Daten lassen sich NICHT „wegverbessern“**: Eine drastische
   Verbesserung ist rechtlich kein lizenzbefreiender Transformationsakt — die
   NC-/SA-Klausel bleibt an den (eingebetteten) Gewichten bzw. Datenableitungen
   haften. Einziger Ausweg: Basis/Daten austauschen.
4. **Gewichts-Copyright ist ungeklärt**: Ob Modellgewichte überhaupt
   urheberrechtlich schutzfähig sind, ist gerichtlich unentschieden — darauf
   darf KEINE Lizenzentscheidung gebaut werden.

---

## AUSBAU-EMPFEHLUNGEN 2026-09-14 — Differenzierung zu anderen Restaurierungslösungen

> Zusätzliche Optimierungen, die Aurik qualitativ UND performance-seitig
> einzigartig positionieren — auf dem bereits Erreichten aufbauend
> (Hörordnungs-Gates, Witnesses, Determinismus, ROCm).

| ID | Hebel | Wirkung | Umsetzung |
|---|---|---|---|
| R1 | **Wahrnehmungs-Budget-Bilanz** (PSY-B): jede Phase verbraucht ein JND-Budget; die Kette bilanziert die Gesamt-Hörbarkeit (wie ein Wahrnehmungs-Wasserzeichen) | „Klangtreu“ wird messbar statt Absichtserklärung — kein anderes Werkzeug bilanziert Reparaturen in JND-Einheiten | `hearing_jnd.py` liegt vor; Budget-Metadaten je Phase + Summenbericht im Export |
| R2 | **MuQ-MOS-Export-Gate** (WIT-M4-Umsetzung): MOS-Delta (out vs. in) als Release-Gate — jetzt möglich, da WIT-M1 die Richtung repariert hat | Hör-Qualität entscheidet über den Export, nicht nur True-Peak/LUFS | MuQ-Plugin (10-s-Clips) als dritte Gate-Stimme im Export-Qualitäts-Gate verdrahten |
| R3 | **ROCm-Beschleunigung aller ML-Modelle** (PERF-A): CQTdiff+, MuQ, BEATs, DeepFilterNet auf Torch-ROCm (Muster bsr317_torch_rocm: 42×) | Echte GPU-Performance-Story auf AMD-Hardware; ORT-ROCm-Kernel-Bug bleibt umgangen | Ports nach dem BSR-Muster, Paritäts-Tests je Modell |
| R4 | **Audibility-First-Scheduling** (PERF-B): billige Detektion zuerst, teure Reparatur nur bei Hörbarkeit — als globales Prinzip formalisiert + als Benchmark gemessen | Rechenzeit sinkt dort, wo das Ohr nichts hört (PSY-A1 rollt das bereits aus) | Benchmark „PSY-A1-Einsparung“ je Phase + Scheduling-Formalisierung |
| R5 | **Determinismus-Zertifikat** (TRUST-A): bit-identische Läufe je Version als Studio-Feature + CI-Nachweis | Reproduzierbares Remastering ist ein Alleinstellungsmerkmal für Studios/Archive | CI-Test „gleicher Input ⇒ MD5-identischer Output“ je Release dokumentieren |
| R6 | **Per-Song-Zielklang** (C4/DDSP, = Q7): EQ/Dynamik-Parameter aus CLAP/BEATs-Embeddings vorhergesagt, DSP führt aus | Kein Mitbewerber lernt den Zielklang pro Song — Studio-Wohlklang statt fester Zielkurven | GPU nach F2 (F5) |
| R7 | **Adaptive Phase-Rescheduling nach Hör-Impact** (PERF-C): Reihenfolge/Budget der Phasen nach erwartetem Hör-Gewinn | Wall-Clock-Budget trifft die hörbarste Verbesserung zuerst | auf bestehendem wall_budget_s aufbauen; Impact-Schätzer aus PSY-A1 |
| R8 | **Sparse Repair** (PERF-D): Reparatur nur in Defekt-Nähe statt Vollband | Rechenzeit + Artefakt-Risiko sinken gemeinsam | Defekt-Masken (bereits vorhanden) als Rechen-Masken nutzen |

---

## Hintergrund (damit die nächste Session sofort einsteigt)

- **Session-Report:** `docs/reports/current/2026-09-08_envelope_root_cause_sota_fixes_matrix.md`
  (Abschnitt 10 = Tiefenanalyse; Abschnitte 1–9 = Root-Cause + Fixes + Matrix).
- **Matrix-Ergebnis (3 Zellen, gleicher Clip) — ABGESCHLOSSEN 2026-09-08:**
  3× EXIT=0, 3× bit-identischer Output (MD5 765c3f544c279f205d32288eef5db95c),
  Envelope μ=0.815 σ=0.081 (7 Chunks, nie mehr degeneriert), Einladungs-Gate:
  Zelle 1 (Baseline) 8× NICHT BESTANDEN → Zellen 2+3 (Fix-Stand) 8× BESTANDEN
  (8 Exemptions an Reparaturstellen, raw 0.562 → effektiv 0.193), Laufzeit
  3 h 19–38 min je Zelle (P0-1 zielt auf ≤40 min). bit-identischer Output (MD5 `765c3f54…`),
  Envelope μ=0.815 σ=0.081 (vor Fix μ=0.060 σ=0.000), Einladungs-Gate BESTANDEN nach
  Reparatur-Exemption (raw 0.562 → effektiv 0.193), Laufzeit 3 h 19–38 min je Zelle.
- **Bereits umgesetzt (nicht neu machen):** B3-Early-Merge-Fix, `_flow_meta`-Spiegel,
  measure_all-Verwerf-Fix, FC-Hörordnungs-Pre-Filter, Einladungs-Gate-Exemption,
  MDX23C-API-Drift, BasicPitch-Fixed-Length, Chunked-Prior (§m2), Envelope-Regressionstest,
  RELEASE_MUST Strength-Envelope-Nichtdegeneration, Ledger-Merge, GUI-Smoke-Flag-Fix.
