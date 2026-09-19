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
- **Status 2026-09-15: UMGESETZT** — `PerformanceGuard.get_budget_truth_report()` weist
  Wand-/Processing-/Analytics-Zeit GETRENNT aus (`rt_wall` = Anwender-Wartezeit,
  `rt_processing` = Budget-Last; der 30-s-Floor wird als `budget_duration_s` neben der
  echten `audio_duration_s` ausgewiesen statt verschleiert); EINE Budget-Norm 32× für alle
  Modi in Code (`LIMIT_* == 32.0`), Norm-Kette (`copilot-instructions.md` §2.38 KMV) und
  Doku (`docs/UNIFIED_RESTORER_V3_SPEC.md` — veraltete „3× RT“-Angaben bereinigt) konsistent;
  `tests/unit/test_p0_3_budget_truth.py` (4 Fälle). Rekalibrierung der Realität (53×→32×)
  bleibt an TODO-P0-1 gebunden (extern blockiert: Laufzeit-Optimierung der F-Reihe).

## TODO-P1-1 · Modell-Residency & Warm-up-Policy

- **Ziel:** Spezifizieren und umsetzen, welche Modelle warmgehalten werden (Residency/LRU je Session)
  und wie Warm-up einmalig je Modell amortisiert wird; deterministisches Multi-Song-Batching desselben
  Modells unter Wahrung von §G1 (Seed-Isolation pro Song).
- **Wirkung:** ~5 min Modell-Ladezeit je Lauf entfällt; batchfähige GPU-Nutzung senkt 53× weiter.
- **Beleg:** Session-Befund (jede Matrix-Zelle lädt Modelle neu); `spec 15 §15.9` (InferenceSessionManager
  Roadmap); `copilot-instructions.md` §G1.
- **Akzeptanz:** Zweiter Lauf in derselben Session ohne Modell-Nachladen; Determinismus-Nachweis je Song.
- **Status 2026-09-15: POLICY-EBENE UMGESETZT** — `backend/core/ml/residency_policy.py`:
  `ResidencyTier` (ALWAYS/SESSION/ONESHOT) + `RESIDENCY_TABLE` für die Plugin-Modelle,
  Warm-up-Amortisierung (`mark_warmed()`/`is_warmed()` — einmal je Prozess),
  §G1-Batching-Vertrag (`song_seed()`: blake2b aus Song-Identität + Master-Seed,
  deterministische Song-Isolation; Multi-Song-Läufe sequenziell auf warmen Modellen);
  `tests/unit/test_p1_1_residency_policy.py` (5 Fälle). Plugins sind bereits Singletons
  (einmal Laden je Prozess) und decken den Residency-Fall damit ab — die Policy
  formalisiert die Einstufung neuer Modelle.

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
  sicheres Ergebnis mit Status „degraded“ fehlt. **§0c-Bug geschlossen 2026-09-15:**
  Export-Gate-Fail ⇒ voller Audio-Export mit Strategie „degraded“
  (`backend/core/export_workflow._resolve_export_strategy`, FQF autoritativ,
  Recovery-Kennzeichen → „recovered“); `tests/unit/test_0c_degraded_export_contract.py`
  (7 Fälle). Die MUSHRA-Studie selbst bleibt extern blockiert (menschliche Hörer n≥30,
  Pilot-Wiederholung mit dem Fix möglich).

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
   **Status 2026-09-17 (F5-Prämisen-Probe, CPU):** CLAP war Negativbefund
   (Head lernte nichts über der Mittelwert-Baseline, 2026-09-16). Probe mit
   dem bereits im Repo vorhandenen BEATs-Encoder (`beats_pooled_embedding`,
   768-dim, ONNX, CPU): 8-s-Musikfenster (Elke Best) mit ±6-dB-Bass- und
   +5-dB-Präsenz-EQ — Embedding-Distanzen 0,7e-3…1,8e-3, richtungsrichtig
   (Bass-Boost > Bass-Cut > Präsenz); CLAP zeigte dort keinen strukturierten
   Signalanteil. Precompute kostet ~0,4 s je Fenster (CPU). **F5-Plan (GPU
   nach F3):** V1 = lokaler `_embed()`-Tausch CLAP→BEATs-pooled im
   wiederverwendbaren C4-Harness (480 Paare, 25 Epochs, ~1 h GPU);
   V2 = Mel-Spektrogramm-CNN-Encoder (DDSP-Muster) nur bei Negativbefund.
   Aktivierung des Heads erst bei val-MAE klar unter der Mittelwert-Baseline
   (0,2455) — kein Blind-Aktivieren.
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
    phase_07_harmonic_restoration. **Status 2026-09-16:** Synthese-Pfad
    CPU-verdrahtet hinter dem F3-Aktivierungsvertrag (fail-closed,
    attempted/applied-Witness, §V6-Fallback-Warnung) + A/B-Validierungsskript
    `scripts/validate_hr_v1.py` (af+HNR-Gates, Exit 0/1/2); Aktivierung bleibt
    bis zum F3-GPU-Befund aus (Flag = einzige Schaltstelle).
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
      **Status 2026-09-16 — FORMAL DURCHGEFÜHRT (zwei Produktions-Bugs gefunden + behoben):**
      Harness `scripts/validate_vocal_inpaint_s4.py` (Kaskaden-Ausgang je Lücke,
      Witness cos ≥ 0,92, Bit-Determinismus; Exit 0/1/2).
      Lauf 1 (Produktions-Status quo) fand: (a) FlowMatching TIER-0 lag auf
      300-ms-Gesangslücken mit mean −2,76 dB UNTER der Stille-Baseline (alle 13
      Segmente negativ; Q11 hatte nur den CQTdiff+-Arm gemessen: 0,0 dB);
      (b) FlowAudio nutzte ungeseedetes `np.random` ⇒ §G5-Verstoß (S4-Gate
      „Determinismus" schlug an).
      Fixes: (a) phase_55 versucht für Gesangslücken (≥ 50 ms,
      vocals_confidence ≥ 0,40) jetzt CQTdiff+ VOR FlowMatching (Evidenz-Reorder,
      keine phasen-individuellen Schwellwerte, §V7-konform); (b) FlowAudio-
      Synthese ist input-geseedet deterministisch (blake2b aus Kontext/Partialen,
      2 Determinismus-Tests).
      Lauf 2: mean ΔSDR **+0,005 dB** (min −0,12 dB), **Determinismus
      bit-identisch** ✅. Striktes Per-Segment-Gate (ΔSDR ≥ 0) und Witness
      cos ≥ 0,92 in allen Fenstern bleiben marginal offen (CQTdiff+-Modellqualität
      am Gate — GPU-Finetune der F-Reihe ist der Hebel); Beleg:
      `docs/reports/current/2026-09-16_vocal_inpaint_s4.json`.
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
19. **SOTA-ML-V5** · ✅ ERLEDIGT (2026-09-18) — BANQUET-Torch-ROCm-Kern +
    Batch-Re-Export. Beleg 2026-09-17 (gemessen, CPU-Session):
    `banquet_vinyl_final.onnx` ist per Konstruktion batch-1-spezifisch — die
    Batch-Dim ist statisch `[1,128,128,128]` und 48+ bandweise
    Squeeze/Unsqueeze-Reshapes (`node_view*` → `[128,128,128]`,
    `node_Reshape_*` → `[128,128,512]`, `node_view*` → `[1,128,128,512]`)
    tragen konstante Ziel-Shapes; ein dynamischer Batch bricht im ersten
    `node_view`-Reshape (gemessener ORT-Fehler „input_shape_size == size
    was false“). **Umsetzung (2026-09-18):**
    (a) *Re-Export* (`scripts/export_banquet_batch_onnx.py` →
    `models/banquet/banquet_vinyl_batch.onnx`): die 3 geteilten
    Shape-Konstanten (je 24× verwendet) durch berechnete Shape-Graphen
    ersetzt (Merge/Dir-Merge/Unmerge mit B×128), h0/c0 als ConstantOfShape;
    Selbstverifikation: B=1 bit-exakt (Optimizer aus), B=2/3
    Batch-Unabhängigkeit bit-exakt. Empirie: Mini-Batch über die LSTM-
    Batch-Achse gewinnt auf ROCm/CPU kaum (~1,1×) — die Kette aus 24
    LSTM-Zellen ist latenzgebunden, nicht durchsatzgebunden.
    (b) *Torch-ROCm-Kern* (`backend/core/dsp/banquet_torch_rocm.py`): der
    ONNX-Kern ist ein 24-Zellen-BSRNN (geteilte LayerNorm, bidir-LSTM
    128→256, Linear+Residuum, alternierender Achsen-Tausch) — 1:1 aus den
    ONNX-Gewichten rekonstruiert (W-Gates nativ PyTorch-Ordnung, Bias-Gates
    gespeichert (i,c,f,o) → perm (0,2,3,1)); Parität ONNX-CPU max|Δ| ≈ 1,9e-6
    (CPU, Optimizer aus), GPU deterministisch. **~160 ms/Fenster statt ~1,9 s
    ORT-ROCm (11,8×; B=4 ≈ 97 ms ≈ 19,5×); End-to-End 3-s-Probe 0,88 s statt
    16,8 s (~19×).**
    (c) *Qualitäts-Nebenfund:* ORT-ROCm-LSTM-Kernels rechnen das Modell
    nachweislich falsch (roh max|Δ| ≈ 0,35 vs. ONNX-CPU auf echten
    Plugin-Feats; bs_roformer-Befund analog) — der ONNX-Fallback läuft
    deshalb jetzt immer auf CPU (Provider-Filter + §V6-Warnung); der
    Torch-Kern ist damit nicht nur schneller, sondern auch numerisch
    korrekt. 7 Unit-Tests grün (`tests/unit/test_banquet_torch_rocm.py`),
    Provider-Test auf die neue Policy umgestellt. AURIK_BANQUET_TORCH=0
    = Kill-Switch; AURIK_BANQUET_TORCH_BATCH=4 = Mini-Batch.
20. **SOTA-ML-V6–V9** · ✅ ERLEDIGT (2026-09-18) — Numerik-Bug-Jagd über alle
    GPU-Pfade + Torch-ROCm-Kerne für Whisper, FCPE und MuQ-MuLan. Die
    ORT-ROCm-Kernels rechneten 6 Modelle eingabeabhängig falsch (gemessen
    auf echten Musik-/Mel-Feeds; weißes Rauschen täuschte OK vor):
    basicpitch rel ≈ 1,9e-2, FCPE ≈ 0,19, MuQ-MuLan ≈ 0,59, Whisper-Tiny
    ≈ 0,98, BANQUET roh 0,35 (plus bs_roformer-Befund von 2026-09-13).
    **Behebung:** Paritätsverifizierte Torch-ROCm-Kerne
    (`backend/core/dsp/{whisper,fcpe,muq_mulan}_torch_rocm.py`, Muster
    bsr317/banquet): Whisper-Tiny-Encoder+Decoder (Parität rel ≤ 1e-4,
    ~6,9 ms je 30-s-Fenster, echte Wort-Timestamps statt Encoder-Heuristik),
    FCPE (rel ≈ 4,4e-5, 60-s-Analyse 2,09 s → 0,17 s), MuQ-MuLan
    (rel ≈ 3,2e-6, ~29 ms je 10-s-Embedding) — alle deterministisch (§G5
    (copilot-instructions.md)), fail-closed auf ONNX-CPU (§V6
    (copilot-instructions.md)). Scan-Methodik gehärtet (const05-Feed),
    Registry-Verdikte korrigiert, Provider-Policy: ONNX-CPU als
    Qualitäts-Fallback. Normativ verankert: §III.9 (copilot-instructions.md),
    dsp.instructions.md, AGENTS.md.

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
**S4-KORREKTUR 2026-09-16:** Die formale S4-Verifikation zeigte, dass die
PRODUKTIONS-Kaskade den TIER-0-Arm FlowMatching bevorzugte — der lag auf
denselben 300-ms-Gesangslücken mit mean −2,76 dB unter der Stille-Baseline
(alle 13 Segmente negativ); Q11 hatte nur den CQTdiff+-Arm gemessen (0,0 dB).
Fix: Für Gesangslücken (≥ 50 ms, vocals_confidence ≥ 0,40) bekommt CQTdiff+
jetzt den ersten Versuch, FlowMatching bleibt erster Fallback; zusätzlich
§G5-Fix im FlowAudio (input-geseedetes Rauschen statt ungeseedetem
`np.random`). Re-Lauf: mean **+0,005 dB**, Determinismus bit-identisch ✅ —
striktes Per-Segment-Gate marginal offen (min −0,12 dB, Witness min cos 0,763);
Hebel = GPU-Finetune der F-Reihe. Beleg: `docs/reports/current/2026-09-16_vocal_inpaint_s4.json`.
**Offene Folge-Frage — ERLEDIGT (2026-09-15):** Die Gesangs-Drosselung in
`_derive_safe_inpainting_strength` entfällt (§Q11: Kaskade mean 0,0 dB /
Worst-Case-Lücke −0,12 dB — nie schlechter als kein Füll ⇒ Never-worsen-Test des
schlechtesten Falls erfüllt). Der Analog-Faktor (×0,85) bleibt als moderate
Reduktion für analoges Material (§V7: konservativ, nicht Workaround); IN-V1/V2-
Naht-Gates + Damage-Guard schützen vor Overfill. Metadatum
`diffwave_vocal_fill_ready=True` trägt den Kaskaden-Beleg; Tests grün
(test_phase_55_diffusion_inpainting.py).
| F2 | GaCELA-Vokal (Pfad B) | **TRAINIERT 2026-09-14/15**: 30 Epochs (10→39) auf MUSDB-Vocals, Checkpoint 39_0499, S1-Val 9 Lücken | **Faire S1-Validierung (2026-09-15): mean ΔSDR −2.63 dB vs. GT — schlägt NICHT einmal die trivialen Fills (Stille −8.64 dB, Crossfade +8.77 dB) ⇒ ~11 dB SCHLECHTER als Crossfade ⇒ NICHT aktiviert, Pfad B mit diesem Setup VERWERFEN** (Gate schließt fail-closed korrekt; Befund: Modell halluziniert Vokal-Inhalt in stillen Lücken — kein Verlustfunktions-, sondern Taskformulierungs-Problem; EAR-VAE-Retrain ohne Neuformulierung (Silence-Gaps + Energie-Gating) wird nicht empfohlen). Witness=None (Resemblyzer im Val-Lauf nicht verfügbar). | MUSDB-Vocals | P1-Metrik |

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
| F3 | BigVGAN-v2 Musik+Vokal | bigvgan_v2.pth lokal | Spektral-Repair ML-V2/HR-V1 | MUSDB-HQ | ΔSDR ≥ +2 dB | **A/B-Teilvalidierung 2026-09-16: af/HNR-Gates PASS (af +0,0073, HNR +4,42 dB, PQS 4,52)**; **FINETUNE LÄUFT 2026-09-17 (reduzierte Rezeptur)**: `scripts/train_bigvgan_f3.py` — 12 Epochen × 1000 Schritte (statt 30×2000), untere Layer eingefroren (96 Gruppen, Encoder-Frozen-Rezept), cudnn.benchmark=True; **Laufzeit-Wahrheit gemessen**: 3,5 s/Schritt (fp16; FP32 8,7 s — MIOpen-Fallback-Kernels auf weight_norm-Convs begrenzen ROCm), Voll-Rezeptur ≈ 58 h ⇒ reduziert auf ≈ 12 h (über Nacht); Reihenfolge: F4 zuerst (1–2 h), F3 sequenziell danach. Gate-Messung (ΔSDR ≥ +2 dB) nach dem Lauf via scripts/validate_hr_v1.py; Flag-Rollout erst danach + Budget-Nachweis |
| F4 | FlashSR-Musik | Checkpoint lokal (FastAudioSR-Quellbaum + upsampler.pth in models/flashsr) | Hochband-Rekonstruktion (aus ML-V1-VORAB: Kandidatenwahl) | MUSDB-HQ | ΔSDR ≥ +2 dB | **FINETUNE LÄUFT 2026-09-17 (zuerst — 132k Parameter, ~1–2 h)**: `scripts/train_flashsr_f4.py` (16-kHz-Basis → 48-kHz-Rekonstruktion, A1+MR-STFT, Smoke grün A1 0,003); F3 folgt sequenziell danach (VRAM-Kollision vermieden) |
| F5 | DDSP-Prädiktor (C4) | CLAP/BEATs-Encoder lokal | EQ/Dynamik-Prädiktion | MUSDB-HQ + Effekt-Paare | MOS-Witness (MuQ) | **Erstlauf 2026-09-16: Harness `scripts/train_ddsp_predictor_c4.py` (480 Effekt-Paare, CLAP-eingefroren → MLP) — val_MAE 0,2447 ≈ Baseline 0,2455 ⇒ NEGATIV: semantisches CLAP trägt die Effekt-Parameter nicht; Taskformulierung braucht DDSP-artigen Mel-Encoder (Folgeschritt, GPU). Beleg: `docs/reports/current/2026-09-16_ddsp_c4_first_run.md` |

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
| PSY-A2 | **Zwicker-Modell (ISO 532-1)** als Nachfolger des MPEG-1-Modells: stationäre + zeitvariante Loudness und Maskierung | stationäres Zwicker ISO 532-1 vorhanden + optional im Audibility-Gate; **zeitvariante Loudness ✅ 2026-09-15** (`compute_time_varying_loudness`, exakte DIN-45631-a0/L_TQ-Tabellen, 10 Tests — P5) | präzisere Schwelle bei tonalem/breitbandigem Material; Basis für PSY-A7 |
| PSY-A3 | **BMLD-Verdrahtung** (binaurale Maskierungs-Freisetzung) in die Stereo-Phasen-Gates (13, 15, 33, 34, 46, 48) | binaural_masking auf Guard-Ebene + phase_03; **2026-09-14: phase_33- UND phase_34-BMLD-Witness**; **2026-09-15: Rollout 13/15/46/48 abgeschlossen** (`binaural_masking_advantage`-Metadatum, ZEUGE-Modus, Hörordnung §8a; test_p2a_bmld_witness_rollout.py, 5 Fälle); **2026-09-15: dynamische Freisetzungs-Toleranz in 13/15/33/46/48** (`bmld_tolerance_factor`: release_db linear 1,0→1,10, Cap 8 dB, Nie < 1,0 — phase_33/48 Breiten-Cap, phase_13 Breiten-Faktoren, phase_15 Korrektur-Stärke, phase_46 Enhancement-Stärke; test_psy_a3_bmld_tolerance.py + _rollout.py) — PSY-A3 damit VOLLSTÄNDIG | Stereo-Änderungen werden nach Hör-Freisetzung bewertet statt nach Mess-dB |
| PSY-A4 | **Equal-Loudness-Band-Gewichte (ISO 226) + JND-Gates** für EQ-/Enhancement-Phasen (04, 16, 17, 37, 38, 39) | fletcher_munson nur 1× verdrahtet; **2026-09-14: phase_37-Bass-Mix mit Equal-Loudness-Faktor temperiert (ISO 226, 60 phon, Deckel [0,5–1,0])**; **2026-09-15: Rollout 04/16/17/38/39 abgeschlossen** (`equal_loudness_strength_factor()` + phase_37-Bugfix: degenerierte Korrekturkurve fror den Faktor auf dem 0,5-Floor ein; test_p2b_equal_loudness_rollout.py, 17 Fälle) | Stärke-Entscheidungen in Phon-Hörbarkeit statt Roh-dB |
| PSY-A5 | **Temporal-Masking-Kompensation**: Forward/Backward-Masking-Zonen als Reparatur-Dämpfung nach Transienten (Rollout des §V41-Musters aus phase_55) | nur phase_55 + 22 Phasen nutzen temporal_masking (Guard); **2026-09-14: phase_01/27/64 dämpfen Klick-/Pop-/Splice-Reparatur in Forward-Masking-Zonen (×0,6)** — kein Nachschlag-Artefakt | Nachmaskierungs-Zonen werden weicher repariert — kein Nachschlag-Artefakt |
| PSY-A6 | **Personalisierte HRIR (CIPIC)** als Ausbaustufe der First-Order-HRTF (Ehrlichkeits-Klausel §8b.3 erfüllt) | First-Order-Modell in interaural_cues | bessere Bühnen-Bewertung bei Kopfhörer-Studien (P1-4) |
| PSY-A7 | **Loudness-Modell-getriebene Dynamik**: Zwicker-Kurzzeit-Loudness steuert 10/11/40/47 | BS.1770-integriert + Bark-LUFS vorhanden; **2026-09-15: phase_47 wahrnehmungs-basierter Loudness-Cap** (peak STL Sone, Marge max(0,15 Sone, 5 %), proportionaler Blend Richtung Input, Never-worsen Hörordnung §4/§8a; test_psy_a7_loudness_cap.py, 6 Fälle); **Folge-Slice 10/11/40 ✅ 2026-09-15** (`perceptual_loudness_cap.py`, Headroom-Varianten 10/11 vs. 40; test_psy_a7_loudness_cap_rollout.py, 8 Fälle) | Punch/Lautheit nach Wahrnehmung statt Peak |
| PSY-A8 | **Generische JND-Gate-Tabelle** (Frequenz ±1 dB, Pegel ±1 dB, Zeit ±5 ms, Pan ±2°… nach Lit.) für alle Never-worsen-Gates | **ERLEDIGT (2026-09-14)**: `backend/core/dsp/hearing_jnd.py` — 9 JND-Klassen mit Quellen, `below_jnd()` (NaN-fail-safe), fail-closed bei unbekannter ID; 6 Tests | absolute dB-Gates weichen JND-basierten, hörbezogenen Grenzen |

### B. Hybrid-Phasen: offene SOTA-Maßnahmen

| Phase | Ist | Offene SOTA-Maßnahme |
|---|---|---|
| 03 denoise | BSR-Stem-NR (66) + DSP-Kaskade; MP-SENet de-wired (Negativbefund) | **HR-V1-Witness verdrahtet (Q6, fail-closed)**; musik-nativer Denoiser-Finetune (F3 BigVGAN-Spektrallinie; DeepFilterNet-Musik-Finetune als Alternative prüfen) |
| 04/16/17 EQ/Dynamik | DSP + Zielkurven | **SOTA-C4**: DDSP-Prädiktor (Embeddings → EQ/Dynamik-Parameter, DSP führt aus) — GPU (F5); Harness + Erstlauf 2026-09-16 (CLAP-eingefroren ⇒ Negativbefund, s. F5-Zeile); nächster Schritt DDSP-Mel-Encoder |
| 07 harmonisch | DSP-Harmonic-Restoration | **SOTA-HR-V1**: BigVGAN-Repair-Pfad + additive_synthesis_gate in phase_07 verdrahtet (Aktivierungsvertrag fail-closed, attempted/applied-Witness); A/B-Validierung via `scripts/validate_hr_v1.py` — Flag-Entscheid GPU (F3) |
| 08/36 Transienten | Superflux/Envelope + BEATs-Witness (TP-V1, V1) | **SOTA-TP-V2**: neurale Phasen-Schätzung für Transienten-Frames (Modell + Quelle klären); adaptive Schwelle aus TP-V1-Konsens |
| 19/43 De-Esser | DSP (19) + ML-Deesser (43) | Sibilanten-Maskierungs-Gate (PSY-A1) ✅ 19 ERLEDIGT 2026-09-14, **43 ERLEDIGT 2026-09-16** (Segment-Gate: subaudible Sibilanten bleiben Original, Zähler `subaudible_sibilants_skipped`) — sonst SOTA |
| 20/49 Dereverb | DR-V1-Verdrahtung (RT60-Witness, konservativ gedeckelt) | präziser neuraler RT60-Regressor (GPU-Training); PSY-A1-Gate für Dry/Wet |
| 23/50 Spektral-Repair | DSP/NMF | **HR-V1-Witness verdrahtet (Q6, fail-closed)**; neuronale Spektral-Inpainting-Basis (F3/F4) |
| 55 Inpainting | Kaskade (FlowMatching, Consistency, CQTdiff+, DAC, GaCELA, DiffWave-S3-vorbereitet) | F2 trainiert + fair validiert ⇒ verworfen (fail-closed, 2026-09-15); F1 abgebrochen; **S4-Verifikation 2026-09-16: formal durchgeführt — CQTdiff+-first für Gesangslücken (Evidenz-Reorder) + FlowAudio-§G5-Determinismus-Fix; mean ΔSDR +0,005 dB, striktes Segment-Gate marginal offen (GPU-Finetune-Folge)** |
| 58 Lyrics-guided | vorhanden (DSP-geführt) | CLAP-Audio-Conditioning für Text-/Semantik-Führung (C4-Embedding-Quelle) |
| 66 Stem-NR | BSR-Stems + Per-Stem-Kette ✓ | PSY-A1-Gate pro Stem; BSR-GPU-Beschleunigung ✓ (Torch-ROCm) |

### C. Pure-DSP-Phasen: offene SOTA-Maßnahmen

| Phase | Ist | Offene SOTA-Maßnahme |
|---|---|---|
| 01 Klicks | ✓ CR-V1 (BANQUET-Konsens) | ✓ PSY-A1-Gate (subaudible Klicks werden nicht gezählt, `subaudible_skipped`) |
| 02 Hum | ✓ HU-V1 (Kalman + LSQ) | ✓ PSY-A1-Gate (Maskierungs-Early-Termination, §Muster 2) |
| 05 Rumpel / 25 Azimut / 31 Speed-Pitch / 62 Crosstalk / 63 Intermodulation / 64 Splice | DSP-Stand (funktional) | ✓ 05/62/63 Maskierungs-Gates vorhanden; ✓ 25/31 JND-formalisiert 2026-09-16 (PSY-A8, `hearing_jnd`); 64: GaCELA verworfen (fail-closed 2026-09-15) → Fill-Reuse nicht anwendbar, DiffWave-S3 bleibt inaktiv vorbereitet |
| 12 Wow-Flutter | ✓ WF-CASS (30-Hz-AM); DSP-Kompensation | **SOTA-WF-V4**: neuraler Warp-Schätzer (Checkpoint-Quelle klären + Download) |
| 13/15/33/34/46/48 Stereo/Bühne | DSP + interaural-Guard | PSY-A3 (BMLD-Gates); PSY-A6 (HRIR-Ausbaustufe) |
| 32 Mono→Stereo | DSP-Upmix | optional: BSR-Stem-basierte Verbreiterung (musik-nativ) |
| 37/38/39 Bass/Presence/Air | DSP | PSY-A4 (Equal-Loudness-JND-Gewichte) |
| 40/47 Loudness/TP | BS.1770 ✓ | PSY-A7 (Zwicker-Kurzzeit-Loudness-Steuerung) |
| 59 Modulationsrauschen | DSP | SOTA-D3+D4 (niedrig): Multiresolution-Split + APPLADE-Maskierungs-Loss |
| 65 Vocal-Natürlichkeit | DSP + Resemblyzer-Witness (Ebene-1) | ✓ GESCHLOSSEN 2026-09-16: Sänger-Identitäts-Witness als Per-Phase-Gate (`_apply_singer_identity_witness`, S4-Muster: cos ≥ 0,92, sonst proportionaler Blend Richtung Input, non-blocking §V6) |

### D. Witness-/Modell-Lücken (Qualitäts-Urteile)

| ID | Lücke | Status |
|---|---|---|
| WIT-M1 | **MuQ-MOS-Richtung invertiert** (A1-Head auf falschem Backbone — muq_eval_a1_head.pt läuft auf MuQ-large-msd-iter statt MuQ-Eval-Backbone) | **ERLEDIGT 2026-09-14** — Ursache war der RESAMPLE-FILTER: das Plugin nutzte librosa, MuQ-Eval torchaudio.functional.resample (Kaiser-Sinc); nach dem Fix (Eval-exakt zuerst) ist die Richtung **3/3 korrekt** (noise0 Δ+3,66 ref / Δ+3,20 Plugin; Report 2026-09-14_muq_plugin_direction.json). MuQ ist damit als MOS-Richtungs-Witness für die F-Gates nutzbar (10-s-Clips, Vollmix) |
| WIT-M2 | BEATs-Tagger-Head fehlt (Encoder-Export ohne Head) | Head auf Tokens trainieren (GPU) oder Tagger-ONNX beschaffen |
| WIT-M3 | UTMOS für Musik unbrauchbar (Negativbefund) | dokumentiert — kein Einsatz |
| WIT-M4 | F-Gates hängen an SDR/Resemblyzer (objektiv) | ✅ ERLEDIGT 2026-09-16 — MuQ-MOS als dritte Gate-Stimme im Export-Quality-Gate **produktionsverdrahtet** (R2: `reference_audio` wird aus beiden uv3-Pfaden durchgereicht; SOFT-Witness, blockt nie, §0c) |

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
| Q3 | PSY-A7: Kurzzeit-Loudness-Steuerung für 40/47 | **CAP UMGESETZT 2026-09-15 (phase_47)** — `_perceptual_loudness_cap`: peak-STL-Überschreitung > Marge ⇒ proportionaler Blend Richtung Input (Never-worsen §4/§8a, §V6-fail-closed); STL/LTL-Witness (2026-09-14) + 6 Tests. 10/11/40 = Folge-Slice ✅ (perceptual_loudness_cap, 8 Tests) | temporal_loudness() nutzt ERB-Kurzzeit-Modell (vorhanden) |
| Q4 | WIT-M1: MuQ-Backbone-Fix (MOS-Richtung) | **NACH F1 (GPU)** | MuQ-Eval-Backbone beschaffen ODER A1-Head auf msd-iter neu trainieren; dann Richtungs-Validierung (Muster validate_muq_plugin_direction.py) — erst danach MuQ als Gate-Stimme |
| Q5 | F2-Verlängerung 30–50 Epochs | **GESCHLOSSEN 2026-09-16 (obsolet):** Die faire S1-Validierung hat Pfad B mit diesem Setup verworfen (halluzinierte Vokal-Inhalte in stillen Lücken, ~11 dB schlechter als Crossfade — Taskformulierungs-, kein Verlustfunktions-Problem). Eine bloße Epoch-Verlängerung ändert daran nichts; Q5 entfällt. Der 2026-09-16 05:11 gestartete Resume-Lauf (`--resume-epoch 4 --epochs 40`, Log `output/gacela_f2/train_log_epoch4_40.txt`) wurde vom Runtime-Neustart unterbrochen (kein neuer Checkpoint) und wird bewusst NICHT fortgesetzt. | — |
| Q6 | F3: BigVGAN (HR-V1 + 23/50 + 03) | **CPU-VORBEREITUNG VOLLSTÄNDIG 2026-09-16** + **A/B-PASS auf der GPU**: Aktivierungsvertrag verdrahtet — `bigvgan_v2_ready()`/`hr_v1_activation_status()` im Plugin (fail-closed, Flag = einzige Schaltstelle) + phase_07-Synthese-Pfad (attempted/applied-Witness, §V6-Warnung beim ML→DSP-Fallback) + `scripts/validate_hr_v1.py` (af+HNR-Gate, Exit 0/1/2). Tests: test_hr_v1_activation_contract.py (6 Fälle, inkl. fail-closed-Synthese-Fallback). **A/B-Validierung 2026-09-16 (Elke-Best-20s, Torch-ROCm): af +0,0073 (≥ −0,02 ✓), HNR +4,42 dB (≥ −0,5 ✓), PQS 4,52, 26 Bänder ⇒ PASS** (Beleg: `docs/reports/current/2026-09-16_hr_v1_bigvgan_ab_validation.md`). **23/50/03-Verdrahtung ✅ 2026-09-16**: gemeinsamer Helfer `apply_hr_v1_additive()` (eine Schaltstelle, layout-agnostisch) + `hr_v1`-Witness in phase_23/50/03 (9 Contract-Tests). Flag bleibt bewusst OFF bis Test-Suite-Anpassung + UV3-Budget-Nachweis (Rollout, >10× RT); F3-Finetune bleibt **GPU-GEBUNDEN**. | bigvgan_v2.pth lokal; Torch-Runtime nach S3-Muster + phase_07-Verdrahtung mit Aktivierungsvertrag |
| Q7 | F5/C4: DDSP-Prädiktor | **HARNESS + ERSTLAUF 2026-09-16 (Negativbefund)**: `scripts/train_ddsp_predictor_c4.py` — MUSDB-Effekt-Paare → CLAP-eingefroren → MLP (val_MAE 0,2447 ≈ Baseline 0,2455); Taskformulierung braucht DDSP-artigen Mel-Encoder ⇒ Folgeschritt GPU | CLAP-Encoder lokal; EQ/Dynamik-Parameter-Prädiktion für 04/16/17, DSP führt aus |
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
| R1 | **Wahrnehmungs-Budget-Bilanz** (PSY-B): jede Phase verbraucht ein JND-Budget; die Kette bilanziert die Gesamt-Hörbarkeit (wie ein Wahrnehmungs-Wasserzeichen) | „Klangtreu“ wird messbar statt Absichtserklärung — kein anderes Werkzeug bilanziert Reparaturen in JND-Einheiten | `hearing_jnd.py` liegt vor; **ERLEDIGT 2026-09-15**: `perceptual_budget.py` (level/loudness/IACC/Centroid in JND-Einheiten via hearing_jnd, layout-sicher, §V6-fail-closed) + `summarize_budget`-Summenbericht + `perceptual_budget_summary` im RestorationResult (pipeline_total); test_r1_perceptual_budget.py, 9 Fälle |
| R2 | **MuQ-MOS-Export-Gate** (WIT-M4-Umsetzung): MOS-Delta (out vs. in) als Release-Gate — jetzt möglich, da WIT-M1 die Richtung repariert hat | Hör-Qualität entscheidet über den Export, nicht nur True-Peak/LUFS | MuQ-Plugin (10-s-Clips) als dritte Gate-Stimme im Export-Qualitäts-Gate verdrahten |
| R3 | **ROCm-Beschleunigung aller ML-Modelle** (PERF-A): CQTdiff+, MuQ, BEATs, DeepFilterNet auf Torch-ROCm (Muster bsr317_torch_rocm: 42×) | Echte GPU-Performance-Story auf AMD-Hardware; ORT-ROCm-Kernel-Bug bleibt umgangen | Ports nach dem BSR-Muster, Paritäts-Tests je Modell |
| R4 | **Audibility-First-Scheduling** (PERF-B): billige Detektion zuerst, teure Reparatur nur bei Hörbarkeit — als globales Prinzip formalisiert + als Benchmark gemessen | Rechenzeit sinkt dort, wo das Ohr nichts hört (PSY-A1 rollt das bereits aus) | Benchmark „PSY-A1-Einsparung“ je Phase + Scheduling-Formalisierung |
| R5 | **Determinismus-Zertifikat** (TRUST-A): bit-identische Läufe je Version als Studio-Feature + CI-Nachweis | Reproduzierbares Remastering ist ein Alleinstellungsmerkmal für Studios/Archive | CI-Test „gleicher Input ⇒ MD5-identischer Output“ je Release dokumentieren |
| R6 | **Per-Song-Zielklang** (C4/DDSP, = Q7): EQ/Dynamik-Parameter aus CLAP/BEATs-Embeddings vorhergesagt, DSP führt aus | Kein Mitbewerber lernt den Zielklang pro Song — Studio-Wohlklang statt fester Zielkurven | GPU nach F2 (F5) |
| R7 | **Adaptive Phase-Rescheduling nach Hör-Impact** (PERF-C): Deferral-Reihenfolge nach erwartetem Hör-Gewinn statt blinder Fix-Priorität | Wall-Clock-Budget trifft die hörbarste Verbesserung zuerst | ✅ UMGESETZT 2026-09-18 (Kern-Slice): `dsp/hearing_impact.py` (Impact = max Defect-Score der zugeordneten DefectTypes via `DefectPhaseMapper`); Guard: Null-Impact-Deferral (Impact ≤ 0,05, nur bei ≥ 0,60× Target-Druck) + Hör-Impact-Schutz (Impact ≥ 0,6 ⇒ kein Skip). Ohne Scores bit-identisch zum Status quo; deterministisch |
| R8 | **Sparse Repair** (PERF-D): Reparatur nur in Defekt-Nähe statt Vollband | Rechenzeit + Artefakt-Risiko sinken gemeinsam | Defekt-Masken (bereits vorhanden) als Rechen-Masken nutzen |

---

## SESSION-ERTRAG 2026-09-15 → WEG ZU 9.8/10 (AURIK QUALITY SCORE 98/100)

> **Ziel-Definition:** „9.8/10“ = **Aurik Quality Score 98/100**
> (docs/COMPREHENSIVE_METRICS.md: 40 % Overall Technical + 40 % Overall Musical +
> 20 % Overall Emotional; internes Spitzenziel bisher ≥ 90). **Messprotokoll:**
> Score auf dem RESTAURIERTEN Output (nicht dem Quellmaterial) messen, je Korpus-Track;
> Original-Score als Referenz-Delta dokumentieren. Wiederholung nach jedem Schritt.

### Erkenntnisse dieser Session (verdichtet, mit Belegen)

1. **PSY-A1 komplett** — Maskierungsschwelle als Pflicht-Frage in allen 16 reparierenden
   Phasen; subaudible Defekte werden übersprungen (weniger Over-Processing, weniger
   Artefakt-Risiko). Zusätzlich Gate-Zentrum-Fix (lange Defekt-Regionen).
2. **Stereo-Layout-Invariante**: 2 echte Kollaps-Bugs gefunden+gefixt (phase_65
   Mean-Achse a8929178; phase_08 (N,2)-Return e3338057), weitere in phase_06/36 —
   Stereo-Regressionstests je Phase riegeln die Bug-Klasse ab.
3. **CI vollständig grün-fähig**: mypy-2.1.0-Type-Gate, Cross-Platform-Collection
   (importorskip-Muster), Timeout 120 min, Windows-numba-SIGSEGV (JIT-aus im Smoke),
   **Apollo-Endlos-Loop gefixt** (produktionsrelevant: DSP-Fallback lief endlos,
   bcfee116). Bekannte Ärgernis: `change_ledger.py snapshot` ohne Trailing-Newline →
   eof-fixer bricht jeden ersten Commit ab (Folge-Fix offen).
4. **CLI/GUI-Parität jetzt bewiesen**: echter Vollpipeline-A/B-Test im Solo Release
   Gate (tests/normative/test_cli_gui_output_parity.py, bitnah ≤ 1e-5). Befund:
   `no_rt_limit=True` = force-execute (§2.45-Skip) divergiert massiv (0.365) —
   dokumentiert; Defaults beider Frontends sind identisch.
5. **F2 (GaCELA-Pfad B) evidenzbasiert verworfen**: trainiert (10→39 Epochs) → faire
   Validierung (Stille-/Crossfade-Baselines, 4f16bed5) → **−2,63 dB vs. GT, ~11 dB
   schlechter als trivialer Crossfade (+8,77 dB)** — Halluzination in stillen Lücken.
   Ursache = Taskformulierung, nicht Loss ⇒ EAR-VAE-Retrain ohne Neuformulierung
   (Silence-Gaps + Energie-Gating) wird NICHT empfohlen. Resemblyzer-Witness blind
   (Paket+ONNX fehlen → alle witness_cos=null).
6. **Q9 Zwicker ISO 532-1 stationär** (aaca82eb): sone/phon exakt kalibriert
   (1 kHz/40 dB ⇒ 1,0 sone), Maskierungsschwelle optional im Audibility-Gate
   (Default MPEG-1 unverändert, fail-open). Offen: zeitvariantes Modell,
   exakte a0/L_TQ-Tabellen.
7. **MuQ-MOS-Richtung (WIT-M1) repariert**; Export-Gate als Soft-Witness (§0c).

### Gemessene Baseline (2026-09-15)

| Messung | Aurik Quality Score | Technical | Musical | Emotional |
|---|---|---|---|---|
| Original („Elke Best – 30 Sekunden.mp3“, 44,1 kHz) | **38,1 / 100** | 0,30 | 0,40 | 0,49 |
| Restauriert (Pipeline, mode=restoration) | wird nachgetragen | — | — | — |

### Priorisierter Weg zu 98/100 (Hebel × Machbarkeit)

| Prio | Schritt | Wirkt auf Säule | Erwartete Wirkung |
|---|---|---|---|
| P1 | **Residual-Artefakt-Diagnose**: artifact_freedom pro Phase auf realem Material messen (welche Phase senkt af unter 0,95?); gezielte Never-worsen-Fixes statt Raten | Technical (40 %) | größter Einzelhebel — af-Veto (≥0,95) ist Hör-Invariante |
| P2 | **PSY-A3-Rest**: BMLD-Witness in Stereo-Phasen 13/15/46/48 + **PSY-A4-Rest**: Equal-Loudness in 04/16/17/38/39 | Musical (40 %) | Stereo-/EQ-Entscheidungen nach Hör-Freisetzung statt Mess-dB |
| P3 | **Resemblyzer-Witness verfügbar machen** (Paket oder ONNX installieren/laden) — Stimm-Identität ≥ 0,92 (Hör-Invariante) wird messbar | Musical (40 %) | Identitäts-Gate schließt die größte blinde Lücke |
| P4 | **R8 Sparse Repair + R4 Audibility-First-Scheduling-Benchmark** (PSY-A1-Einsparung je Phase messen) | Technical (40 %) | Artefakt-Risiko ↓ + Rechenzeit ↓ gemeinsam |
| P5 | **Zeitvariantes Zwicker (ISO 532-1)** + exakte a0/L_TQ-Tabellen | Technical (40 %) | präzisere Maskierungsschwellen für alle PSY-A1-Gates |
| P6 | **Q6/Q7/F6 GPU-Buildouts** (BigVGAN HR, DDSP-Prädiktor) — erst, wenn P1–P3 nicht reichen | Musical/Technical | HR-Wiederherstellung/EQ-Zielklang |
| P7 | **R5 Determinismus-Zertifikat** (bit-identische Läufe je Version als CI-Nachweis) | Trust | Studio-Alleinstellungsmerkmal |

### UMSETZUNGSSTAND 2026-09-15 (Umsetzungswelle P1–P5 + P7)

> **Folge-Konzept:** `docs/GESAMTKONZEPT_PERFORMANCE_WOHLKLANG.md` — wie die
> Restaurierung aller Importsongs performance-seitig maximiert wird, ohne den
> Wohlklang zu kompromittieren (Audibility-First, vier Hebel, Sicherungsnetze,
> Maßnahmen-Roadmap, Verbotsliste).

Alle Punkte mit Tests (IDs in Klammern = Testdatei unter tests/unit/):

- **P2 (PSY-A3/A4-Rest) ✅ UMGESETZT** — BMLD-Witness (Muster phase_33/34,
  ZEUGE-Modus, Hörordnung §8a) in 13/15/46/48 als
  `binaural_masking_advantage`-Metadatum (test_p2a_bmld_witness_rollout.py, 5 Fälle).
  Equal-Loudness (ISO 226): neuer Helfer `equal_loudness_strength_factor()` in
  `fletcher_munson_curves.py` (Kontur-SPLs; Bugfix: die phase_37-Vorläufer-Verdrahtung
  degenerierte über negative Korrekturkurven-Werte immer auf das 0,5-Floor — der
  Faktor war NICHT frequenzsensitiv); Rollout in 04/16/17/38/39
  (test_p2b_equal_loudness_rollout.py, 17 Fälle).
- **P3 ✅ UMGESETZT** — Pfad-Tippfehler `models/rezemblyzer`→`models/resemblyzer`
  behoben (Package-Pfad war nie erreichbar) + webrtcvad-Import-Shim (Vendored-Paket
  bleibt unverändert) → Resemblyzer-Witness lädt jetzt via Package (256-dim,
  cos ≥ 0,92 messbar, test_p3_resemblyzer_witness_availability.py, 5 Fälle).
- **P4 ✅ UMGESETZT (Infrastruktur + Messung)** — R8: `backend/core/dsp/sparse_repair.py`
  (Defekt-Masken als Rechen-Masken: Regionen+Kontext, Hann-Crossfade,
  Coverage-Vollpfad, §V6-fail-closed; test_r8_sparse_repair.py, 17 Fälle).
  R4: `scripts/benchmark_audibility_first.py` (Gate-Orakel: Hörbarkeits-Filterrate
  subaudible vs. audible; phase_01 exportiert `subaudible_skipped`;
  test_r4_audibility_benchmark.py, 7 Fälle). Per-Phase-Rollout des R8-Musters = Folge-Slice.
- **P5 ✅ UMGESETZT** — EXAKTE a0-/L_TQ-Tabellen nach DIN 45631 (Zwicker-Verfahren)/
  ISO 532-1:2017 (a0[1 kHz]=0 dB, L_TQ[1 kHz]=0 dB, Tiefstwert −4,1 dB @ 3,15 kHz)
  + zeitvariante Lautheit `compute_time_varying_loudness` (DIN-45631/A1-Struktur,
  Hann-energiekorrekt, N5/N10; test_p5_time_varying_zwicker.py, 10 Fälle;
  Bestands-Test auf exakten L_TQ-Floor angepasst).
- **P7 ✅ UMGESETZT** — R5-Determinismus-Zertifikat: feste pure-DSP-Kette
  (04→16→28→33→34→37→39→59) doppelt gefahren, MD5-identisch, frische Instanzen;
  phase_38 separat (hash()-Jitter ist prozess-stabil, maschinen-unabhängig
  dokumentiert; test_r5_determinism_certificate.py, 4 Fälle).
- **P1 ✅ WERKZEUG + BEFUND** — `scripts/artifact_freedom_diagnosis.py` misst
  `ArtifactDetector.overall_score` nach jeder Phase auf realem Material
  (test_p1_artifact_diagnosis.py, 4 Fälle). Lauf auf „Elke Best – 30 Sekunden.mp3"
  (20 s): siehe `output/artifact_freedom_diagnosis/af_diagnosis_report.json`.
  **Befund (25 Phasen, komplett):** Input-af bereits 0,7414 (MP3-Quelle) — die af-Veto-
  Schwelle 0,95 ist bei degradierten Quellen nur DELTA-basiert sinnvoll
  (Guard-Kalibrierung AGENTS.md). Größte af-Abfälle je Phase: phase_07
  (Harmonic-Restoration) Δ−0,136; phase_17 (Mastering-Polish) Δ−0,084;
  phase_19 (De-Esser) Δ−0,062; phase_38 Δ−0,043; phase_39 Δ−0,031;
  phase_16 Δ−0,018; phase_36 Δ−0,017. Min-af 0,473 (phase_19).
  **Nächster Slice:** gezielte Never-worsen-Fixes in 07/17/19/38 (größte
  Delta) + af-Delta-Schwelle je Phase im Diagnose-Skript (`--fail-delta`).

**Dabei gefundene + gefixte Produktions-Bugs (Baseline-Tests 196/196 wieder grün):**

- phase_23: unbedingtes `return` im FlashSR-Early-Exit-Block — bei verfügbarem
  FlashSR wurde die Reparatur IMMER übersprungen (ML und MRSA-Fallback tot).
- phase_28: §v10.754-Floor-Pre-Stage lief vor der Stärke-Prüfung → strength=0 war
  kein Passthrough (NMR konnte den Skip zusätzlich reaktivieren).
- phase_42: TypeError-Retry rief exakt dieselbe Signatur erneut (unbrauchbar);
  Tests auf §v10.739-Vertrag aktualisiert (MDX23C entfernt → HPSS-Tertiärpfad).
- **ORT-C++-Abort**: `InferenceSession` mit nicht registriertem
  `ROCMExecutionProvider` (CPU-only-ORT + aktiver MIGraphX-Detektor) brach den
  Prozess mit std::terminate/SIGABRT ab — CLI/Headless-Crash bei phase_01-ML-Detektion.
  Fix: Provider-Filter gegen `ort.get_available_providers()` in
  `ml_device_manager.get_ort_providers(_fp16)` + PANNs-Plugin (fail-closed,
  test_ml_device_manager_provider_filter.py, 4 Fälle).

### UMSETZUNGSSTAND 2026-09-15 — WELLE 2 (CPU-schließbare Restpunkte A–G)

Alle CPU-schließbaren Punkte der Offene-Punkte-Matrix sind umgesetzt und getestet
(Testdateien unter tests/unit/, IDs in Klammern):

- **D ✅** — `scripts/change_ledger.py` schrieb nach einem Snapshot eine
  LEERZEILE + Newline ans Dateiende; der eof-fixer-Hook räumte das auf ⇒
  jeder ERSTE Commit nach einem Snapshot wurde abgebrochen. Fix: genau EIN
  abschließendes Newline (`.rstrip("\n") + "\n"`).
- **B ✅ (§0c, P1-4-Befund)** — Export-Quality-Gate-Fail ⇒ voller Audio-Export
  mit Strategie „degraded“ statt 50-Byte-WAV/rc=0:
  `backend/core/export_workflow._resolve_export_strategy` (FQF autoritativ,
  Recovery-Kennzeichen → „recovered“); test_0c_degraded_export_contract.py (7 Fälle).
- **A ✅ (TODO-P0-3)** — Budget-Wahrheit: `get_budget_truth_report()`
  (Wand/Processing/Analytics getrennt, 30-s-Floor ausgewiesen), EINE 32×-Norm
  über Code/Norm-Kette/Doku (test_p0_3_budget_truth.py, 4 Fälle).
- **C ✅ (TODO-P1-1)** — Modell-Residency & Warm-up-Policy:
  `backend/core/ml/residency_policy.py` (ResidencyTier, Warm-up-Amortisierung,
  §G1-song_seed; test_p1_1_residency_policy.py, 5 Fälle).
- **E ✅ (PSY-A7, phase_47)** — wahrnehmungs-basierter Loudness-Cap (peak STL
  Sone, Marge, proportionaler Blend Richtung Input; Never-worsen §4/§8a;
  test_psy_a7_loudness_cap.py, 6 Fälle). 10/11/40 = Folge-Slice ✅ (test_psy_a7_loudness_cap_rollout.py, 8 Fälle).
- **G ✅ (SOTA-R1)** — Wahrnehmungs-Budget-Bilanz:
  `backend/core/dsp/perceptual_budget.py` (JND-Einheiten via hearing_jnd,
  layout-sicherer Mono-Downmix, §V6-fail-closed) + Summenbericht +
  `perceptual_budget_summary` im RestorationResult (test_r1_perceptual_budget.py, 9 Fälle).
- **Stereo-Layout-Fixes (Stereo-Axis-Matrix-Befunde)** — phase_30
  (Subsonic-/DC-Messung) und phase_39 (HF-Energie-Messung) kollabierten
  channels-first (2, N) via `audio[:, 0]` auf 2 Samples → sosfiltfilt-ValueError.
  Layout-sichere Kanalextraktion + DC-Messung via `stereo_channel_view`;
  beide Stereo-Axis-Matrix-Tests (phase_30/phase_39) grün.

## PERFORMANCE-MASSNAHMENKATALOG 2026-09-17 — drastische Laufzeit-Steigerung (Evidenz & Entscheidungen)

> Ziel: 53×→32×-RT (TODO-P0-1) ohne Qualitätskompromiss. Zusammenfassung der
> gemessenen Hebel, der umgesetzten Maßnahmen und der bewusst NICHT
> verfolgten Wege (mit Begründung). Reihenfolge = erwarteter Gewinn je Aufwand.

| # | Maßnahme | Status | Erwarteter Gewinn | Qualitätsrisiko | Beleg |
|---|---|---|---|---|---|
| P1 | Song-Level-Hoists (je-Chunk-Wiederholung entfernt): Struktur ANA-6, LGE-Transkription Whisper 8×→1×, Export nur nach Assembly | ✅ 2026-09-17 | LGE: 7× Whisper-Zeit; Export: 1× je Song statt 8× | keines (identische Rechnung, Timeline je Chunk zeitverschoben) | 615f373d, 3bfa5215, 04a52839 |
| P2 | R3-GPU-Ports: BANQUET ROCm, CRePE-Pitch ROCm (28×), DeepFilterNet ehrlich CPU | ✅ 2026-09-13/17 | BANQUET 1,19× (Deckel: Export batch-1-spezifisch), CRePE 28× | keines (Parität validiert: BANQUET rel 7,8e-3 hörirrelevant, Klick-Reduktion identisch) | `gpu_model_registry.json` + `test_production_registry_verdicts_restoration_models` (18d365bd) |
| P3 | Analyse-Cache je Datei-Hash (Disk-Persistenz der Bridge-Caches, Read-/Write-Through, AURIK_VERSION-Invalidierung, §V6 (copilot-instructions.md)-fail-closed) | ✅ 2026-09-17 | Wiederholungsläufe am selben Song überspringen die KOMPLETTE Voranalyse (Medium/Era/Genre/Defects/Restorability) über Prozessgrenzen | keines (bit-exakter Roundtrip; Version im Key, §G5 (copilot-instructions.md)) | dd1b91dd, 27 Tests grün |
| P4 | F4/F5 FlashSR-/DDSP-Finetunes (ML-Load/Inferenz-Treiber) | 🔄 läuft (F4 Epoche 0/12: A1=0.0120, Val=0.4562; F3/F5 danach) | Qualitäts- und Laufzeit-Gewinn (Fixkosten amortisieren) | Finetune = Qualitäts-GEWINN, kein Kompromiss | Shells wqt4o062/vck3tbxs, SOTA-ML-V1/V2 |
| P5 | BANQUET-Mini-Batch via dynamischer Batch-Dim (SOTA-ML-V5) | 🔜 Re-Export nötig (GPU) | potenziell deutlich über 1,19× (0,5-s-Fenster-Inferenz gebündelt) | keiner — erfordert Parity-Scan rel ≤ 1e-3 nach Re-Export | Messung 2026-09-17: statische Dim `[1,128,128,128]`, 48+ bandweise Squeeze/Unsqueeze-Reshapes (`node_view`) mit konstanten Ziel-Shapes; ORT-Fehler bei B=2; `add_free_dimension_override` greift nicht |
| P6 | Ein-Prozess-Batch (§V8 (copilot-instructions.md)-saubere Load-Amortisierung) | ✅ 2026-09-17 (Architektur-Feststellung + Lücke geschlossen) | Batch läuft bereits Ein-Prozess (`batch_processor.py`, ThreadPool 4 Worker, Denker je Song) — BANQUET-Session lädt einmal je Prozess. Lücke war der Song-übergreifend geteilte Transient-State (Fehler-Zähler ohne Lock, kein je-Song-Reset) → `reset_for_song()` + `_state_lock` + Verdrahtung in `_restore_chunked` | keines — Quarantäne bleibt fail-closed (Modell-Zustand), nur der je-Song-Zähler wird isoliert | batch_processor.py:415, restaurier_denker.py:1558, 19 Tests grün |
| P7 | DAG-Parallelität der Analyse (PANNs/Whisper/Defects/Gender parallel über CPU-Kerne) | ✅ 2026-09-17 (Architektur-Feststellung) | Je-Song-Analyse läuft bereits parallel über die 4 Batch-Threads; innerhalb des Songs parallelisiert `run_pre_analysis` per ThreadPool(4) — kein offener Rest | keines | batch_processor.py:415 (ThreadPoolExecutor), pre_analysis.py (ThreadPoolExecutor) |

### Bewusst NICHT verfolgt (Qualitätskompromiss — mit Begründung)

| Maßnahme | Warum verworfen |
|---|---|
| Restore-Chunks parallelisieren | Verstoß §V7 (copilot-instructions.md): geschlossener Regelkreis — `global_scalar`/Stärke-Entscheidungen lernen sequenziell pro Song; parallele Chunks umgehen die zentrale Stärke-Steuerung |
| Song-Level-Hoist der BANQUET-/DFN-/Pitch-Inferenz | NICHT äquivalent: phase_09 verarbeitet den phase_08-Ausgang je Chunk (Declick/Hum ändern das Signal vor der Knistern-Entfernung) — Vorlauf auf dem Roh-Song ≠ je-Chunk-Ergebnis |
| Quantisierung (int8/bf16) ohne Parity-Scan | BSR-Präzedenz: ORT-ROCm-Softmax rel=3,5 (Knoten 454) — Quantisierung nur mit gemessenem rel ≤ 1e-3 je Modell freigeben |
| Phasen skippen außerhalb der Audibility-Gates | verboten — die Gates sind die einzige zulässige Schwelle (Hörordnung) |
| BANQUET-Mini-Batch per chirurgischem Graph-Patch | Export ist per Konstruktion batch-1-spezifisch (48+ bandweise Squeeze/Unsqueeze-Reshapes mit konstanten Ziel-Shapes; B=2 bricht im ersten `node_view`); Patch = Dutzende koordinierte Rewrites inkl. RNN-Zellen — nur Re-Export vertretbar (→ SOTA-ML-V5) |

### Rest-Potenzial nach dem Verifikationslauf (qualitätsneutral, Reihenfolge = Hebel)

| # | Hebel | Erwartung | Status |
|---|---|---|---|
| P8 | BANQUET parallele Fenster-Inferenz (`AURIK_BANQUET_INFER_PARALLEL`): 0,5-s-Fenster unabhängig, ORT-`run` thread-sicher → ThreadPool über Fenster; OLA bleibt sequenziell → bit-identisch (Test beweist es) | gemessen (2026-09-18, 16 Kerne, 9 Fenster): CPU 18,2→11,5 s (1,58×), GPU-ROCm 14,2→8,9 s (1,61×); sha über alle Konfigs identisch | ✅ UMGESETZT 2026-09-18: Default 4 (Datenlage entscheidet, `scripts/benchmark_banquet_parallel.py`); Env-Override bleibt |
| P9 | Whisper (LGE) auf Torch-ROCm statt CPU (einmal je Song) | ~10–20× auf dem Transkriptionsschritt | ✅ UMGESETZT 2026-09-18 (HF-Decoder-Pfad): Gerätewahl cuda:0 bei torch.cuda, Kill-Switch `AURIK_WHISPER_GPU=0`, GPU-Fehler ⇒ sichtbarer CPU-Rückfall (§V6 (copilot-instructions.md)); auf diesem Host inaktiv — HF-Modell-Dateien fehlen (Blob-Store-Symlinks gebrochen) ⇒ ONNX/DSP-Ersatzpfad |
| P10 | ORT-Session-Tuning (BANQUET intra-op 4→N im Zusammenspiel mit P8) | Mikro; intra_op=4 bleibt Optimum in der gemessenen Matrix | ✅ UMGESETZT 2026-09-18: `AURIK_BANQUET_INTRA_OP_THREADS`-Knopf (Default 4 = bisher); Matrix gemessen (intra_op 1/2/4 × P 0/2/4/6/8) |
| P11 | SOTA-ML-V5: BANQUET-Re-Export mit dynamischer Batch-Dim → GPU-Mini-Batch | Mini-Batch-Empirie ~1,1× (LSTM-Kette latenzgebunden) — aber **Torch-ROCm-Kern: 11,8×/B=4 19,5× je Fenster, End-to-End ~19×** (3-s-Probe 0,88 s statt 16,8 s) | ✅ UMGESETZT 2026-09-18: Re-Export `scripts/export_banquet_batch_onnx.py` → `banquet_vinyl_batch.onnx` (B=1 bit-exakt, B=2/3 unabhängig) + Torch-Kern `backend/core/dsp/banquet_torch_rocm.py` (Parität ONNX-CPU 1,9e-6, deterministisch) + ORT-ROCm-Defekt beseitigt (ONNX-Fallback jetzt CPU, roh 0,35-Fehler eliminiert) |
| P12 | Hot-Phase-Nachmessung nach run30 (`compute_hot_phases` auf neuem Lauf-Log) — falls neue Phasen-Treiber sichtbar werden | evidenzbasiert | ✅ ERLEDIGT 2026-09-19 — WAHRHEIT-KORREKTUR: die v1023_analysis.txt-Attribution (phase_41 2287 s) war FEHLHAFT (phase_41 real 32 s/0,8 %; gemessen geplant→phase_ok je Chunk). Echte Hot-Phasen: phase_01 708 s (BANQUET-ONNX-CPU, s. §PERF-R4), phase_27 460 s, phase_12 364 s, phase_31 357 s, phase_54 308 s. Tooling-Fix: Benchmark-Progress-Callback akzeptiert das 4. Engine-Argument (TypeError-Schluckung behoben) |

### §PERF-R (2026-09-18) — Redundanz-Befund & Höchstperformance-Fixes (qualitätsneutral)

Audit der Chunked-Pipeline ergab **redundante Prozesse**, die Modellwärme und
Laufzeit je Chunk vernichteten:

1. **Per-Chunk-Cleanup-Churn (R-A, behoben):** Der aggressive
   End-of-Run-Cleanup lief am Ende JEDES `restore()` — im Chunked-Pfad also
   nach jedem 30-s-Chunk: Unload von FlashSR/UTMOS/CLAP/MERT/FCPE/BasicPitch
   + `cleanup_after_file()` ⇒ `force_evict_all()` aller inaktiven Plugins
   (BANQUET, BS-RoFormer, …) + Budget-Release + GC je Chunk.
   Fix: Cleanup in `_run_end_of_song_cleanup()` extrahiert, läuft im
   Chunked-Pfad nur noch EINMAL nach der Song-Assembly; Ganz-Song-Pfad
   unverändert. Warme Modelle überleben jetzt alle Chunks.
2. **BANQUET ohne Reload-Pfad (R-B, behoben):** Nach PLM-Eviction verlor das
   Singleton-Session/`_model_ok` — `_try_load_model` lief nur im `__init__` ⇒
   Folge-Chunks fielen STILL auf den DSP-Ersatzpfad (§V6 (copilot-instructions.md)-Risiko).
   Fix: `ensure_model_loaded()` (einmal-je-Eviction-Reload, `_state_lock`,
   kein Retry-Schleifen); `process()` stellt die Session automatisch wieder her.
3. **Toter CR-V1-Konsens (R-B2, behoben):** phase_01 prüfte
   `_plugin._model_loaded` — ein Attribut, das im Plugin NIE existierte ⇒
   der BANQUET-Klick-Detektor-Konsens (SOTA-CR-V1) war stumm deaktiviert.
   Fix: ehrlicher Check über `ensure_model_loaded()`; Feature läuft wie
   spezifiziert (Test-Fakes angepasst, Konsens-Suite grün).
4. **Zweite BANQUET-Session im phase_09-Fallback (R-C, behoben 2026-09-18):**
   Der Docker-Fallback erzeugte je Phase-Instanz eine NEUE
   `BanquetVinylPlugin()`-Instanz (zweite ≈0,8-GB-Session, je Chunk neu
   geladen, §V7). Fix: kanonisches Singleton `get_banquet_plugin()` mit
   `ensure_model_loaded()`-Selbstheilung.
5. **Tote Readiness-Probes (R-D, behoben 2026-09-18):**
   `ml_model_readiness` probte `phase_09._get_banquet_onnx_session` (existiert
   seit dem §V7-Fix 2026-09-13 nicht mehr ⇒ dauerhaft falsches „BANQUET nicht
   verfügbar“) und das nicht existierende `plugins.convtasnet_plugin`
   (Phantom-Ladefehler im Selbstcheck, Modell de-wired). Fix: ehrliche
   Plugin-Probes bzw. Registrierung entfernt — Selbstcheck jetzt 39/39, 0
   Phantom-Fehler (§V6 (copilot-instructions.md)-Ehrlichkeit).

Damit: BANQUET-ML bleibt über alle Chunks aktiv, CR-V1 aktiv, keine
Modell-Reloads je Chunk — plus P8/P10 (Fenster-Parallelität, Default 4,
1,58× CPU / 1,61× GPU, bit-identisch) und R7 (Hör-Impact-Deferral).
Laufzeit-Erwartung: BANQUET-Phase (dominant, 62 %) sinkt auf ~45–60 % der
bisherigen Zeit, dazu entfallen die Per-Chunk-Reloads vollständig.
Verifikation: 225,3-s-Referenzlauf mit PERF-R-Stand läuft
(`output/supervised_run/elke_225s_perfr_v1021.{log,wav}`) — P12-
Hot-Phase-Nachmessung folgt nach Laufende.

### §PERF-R2 (2026-09-18) — Laufzeit-Analyse des Referenzlaufs: phase_12 war der Flaschenhals

Der v1021-Lauf (PERF-R-Stand) lag bei ~29 min/Chunk statt ~15 min Baseline —
Analyse ergab DREI Ursachen, alle behoben:

1. **Baseline-Kontext:** Im 2026-09-16-Lauf (33× RT) hatte der Orchestrator
   phase_12 ENTFERNT (`§SURGERY-FIRST: 44→20 Phasen`, Restorability 50) bzw.
   lief der Pitch-Konsens leer (T=2–4 Frames). Seit dem CRePE-ROCm-Port
   (18d365bd) liefert der Konsens echte Daten (T=1197) — die volle
   Wow/Flutter-Korrektur läuft jetzt, kostete aber 348 s/Chunk.
2. **Pathologischer Vocoder (behoben):** `phase_vocoder.py` legte PRO
   OUTPUT-SAMPLE eine komplette n_fft-irfft an (~1 Mio. irffts je 10 s
   Stereo; gemessen 79 s/10 s statt des Docstring-Ziels <50 ms/5 s) und
   verfälschte mit dem Identity-Phase-Lock Nicht-Bin-Frequenzen aufs
   Bin-Raster (440 Hz → 445,3 Hz = +12 Cent) bei Hüllkurven-Modulation
   p95 ≈ 6,7 dB — der Reinhör-Witness flaggte im Lauf „pitch_instability“
   (pitch=34.0c mod=15.1c) und §v10.709 meldete artikulation-Degradation.
   Fix: Standard-Frame-weise Synthese (EIN Frame je Synthese-Hop, batched
   irfft, korrekte PV-Phasenpropagation), Identity-Lock deaktiviert
   (Wow/Flutter ≤ ±10 % Stretch ⇒ ungelockter Laroche/Dolson ist der
   etablierte Standard): exakt 440,00 Hz, Hüllkurve flach (1,01),
   p95 ≈ 2,7 dB. ~400× schneller.
3. **pYIN-Viterbi (behoben):** librosa-pYIN-Viterbi (reines Python)
   kostete ~11–14 s je 10 s; fmax C7→C6 (Wow/Flutter braucht nur f0 ≲ 1 kHz,
   CREPE-Konsens deckt den Rest) ⇒ ~4× schneller.

Messung: phase_12 95 s → 10,7 s je 10 s Musik (CPU) ⇒ ~5 min Ersparnis je
Chunk, ~40 min je Song. Neustart des Referenzlaufs mit allen Fixes:
`output/supervised_run/elke_225s_perfr_v1022.{log,wav}`.

### §PERF-R3 (2026-09-18) — Repo-weiter pYIN-/DSP-Sweep: weitere Chunk-Kostentreiber beseitigt

Folge-Sweep nach demselben Muster (Profil je Phase auf echtem Material):

1. **phase_36 savgol→O(N)-Box-MA:** `savgol_filter(polyorder=1)` ist exakt
   der Box-Mittelwert am Fensterzentrum; der interne `correlate1d` mit bis
   zu 7681-Tap-Kernel kostete 10,6 s/10 s. Cumsum-Weg: Abweichung 1e-14
   (Float-Rauschen), Messung 12,4 s → 1,85 s je 10 s (6,7×).
2. **phase_54 Soft-Knee-Vektorisierung:** Gain-Reduction-Loop (522k Samples)
   element-weise vektorisiert (reine Funktion je Sample, bit-identisch);
   7,0 s → 5,3 s je 10 s. Der STL-adaptive Attack/Release-Follower bleibt
   bewusst sequenziell.
3. **pYIN-Viterbi-Sweep (C7→C6):** vier weitere C7-Kandidatenräume auf
   30-s-Chunks (je ~33–35 s): phase_31 (2×), hybrid_speed_pitch_ml und
   natural_performance_detector (Vibrato-Zonen der VFA-Kette). C6 halbiert
   den Raum (~4×); CREPE-Konsens deckt f0 > 1 kHz.
4. **CR-V1-Funktionalität (Nachtrag zu §PERF-R):** 2-D-Mono-Aufruf — der
   BANQUET-Klick-Konsens liefert jetzt echte ML-Regionen (Smoke: Klick exakt
   detektiert).

Befunde OHNE Änderung (bewusst): phase_27 — 3168 `defect_audibility`-Aufrufe
je Chunk sind der PSY-A1-Vertrag (lokale Maskierung je Defekt-Kontext); ein
Global-Cache wäre nicht exakt. CREPE: 0,21 s/5 s im Steady-State (die einmalige
ROCm-Kernel-Kompilierung kostet ~125 s im Erst-Prozess, amortisiert sich).
FC-Loop und GOAL-Block sind normative Qualitäts-Maschinerie (Hörordnungs-
Gates arbeiten korrekt — FC bricht bei Ebene-1-Verstoß ab).
Verifikation: `output/supervised_run/elke_225s_perfr_v1023.{log,wav}` +
automatische Analyse `…_v1023_analysis.txt` (Chunk-/Phasen-Zeiten, RT).

### §PERF-R4 (2026-09-19) — v1023-Wahrheit + BANQUET-Torch-Batch-Matrix (qualitätsneutral)

Nachmessung der v1023-Hot-Phasen (geplant→phase_ok je Chunk, statt der
fehlerhaften v1023_analysis.txt-Attribution) + Folge-Fixes:

1. **v1023-Attributions-Korrektur:** phase_41 kostete real 32 s (0,8 %),
   nicht 2287 s — die Analyse-Datei ordnete Zwischenphasen-Wallzeit falsch
   zu. Echte Hot-Phasen (Summe je 30-s-Chunk): phase_01 708 s, phase_27
   460 s, phase_12 364 s, phase_31 357 s, phase_54 308 s, phase_13 227 s.
2. **phase_01-Wahrheit:** 65–89 s je Chunk lagen IN
   `_detect_clicks_banquet_ml` — der v1023-Lauf startete 15:38, der
   Torch-ROCm-Kern landete erst 18:30 (7f9d1ecc) ⇒ der Lauf nutzte den
   ONNX-CPU-Parallelpfad (≈75 s/30 s). Mit Torch-Kern (live gemessen):
   BANQUET 6,8 s/30 s, phase_01 gesamt 10,6 s/30 s auf realem
   Elke-Material (483 Klicks) — **~8× schneller als im v1023-Lauf**.
3. **TORCH_BATCH-Matrix (60 reale Elke-Fenster, 7900 XTX):** B=1 11,46 s →
   B=4 7,85 s → B=32 6,49 s → B=40 6,96 s; max|Δ| vs. B=1 = 6,7e-6
   (rel 1,7e-7); **B=4 vs B=32 auf GPU bit-identisch (max|Δ|=0,0)**.
   MIOpen-LSTM bricht ab B ≥ 48 (miopenStatusBadParm). Default 4→32,
   Clamp [1, 40] (Commit a84db9a4).
4. **_prepare_input vektorisiert:** 128er-Band-Python-Schleife (7552
   Iterationen/30-s-Chunk) → np.tile-interleaved — Unit-Test pinnt
   Bit-Identität gegen die alte Schleife (Commit a84db9a4).
5. **Benchmark-Tooling-Fix:** Engine-Progress-Callback übergibt 4
   Positionsargumente (unified_restorer_v3.py:8149); der
   Effizienz-Matrix-`_cb` akzeptierte 3 ⇒ TypeError still geschluckt ⇒
   top_phases=null. `*_extra` ergänzt — top_phases liefert jetzt echte
   Phasen-Wall-Zeiten.
6. **SOTA4-7-Befund:** LGE-Whisper bereits Singleton (1× je Prozess);
   verbleibende 1,7 s/Chunk Re-Transkription sind inhärent
   (Chunk-akkurate Maske auf verarbeitetem Audio; WoW/Flutter-Stretch
   macht die Song-Timeline nicht exakt).

### §PERF-R7 (2026-09-19) — phase_54-STL-Follower ohne NumPy-Dispatch (bit-identisch)

Der STL-adaptive Attack/Release-Follower bleibt bewusst sequenziell
(zustandsbehaftete Rekursion, §PERF-R3-Entscheid), aber je Sample
``np.clip`` (2×1,44 M) + ``np.exp`` (1,44 M) kosteten gemessen ~9 s von
14 s Phasenzeit — NaN-sichere Bedingungs-Kaskade + ``math.exp`` rechnen
**bit-identisch** (gleiche libm-Double-Arithmetik, NaN-Semantik wie
``np.clip``; Unit-Test pinnt beides). phase_54 auf realem Elke-Material:
17,83 → 8,36 s je 30 s (2,1×) — Commit 37558bbc.

### §PERF-R8 (2026-09-19) — pYIN mit band-begrenztem Viterbi (bit-identisch)

librosa-0.11.0-``pyin`` dekodiert über ``np.vectorize`` + Zeilen-Loop
(``sequence._viterbi``) — gemessen 8,6 s je 30-s-Aufruf (48 kHz,
3841 Frames × 962 Zustände) und dominierte phase_12/phase_31. Die
pyin-Transition ist ``kron(t_switch[2x2], transition_local(B, w))`` — je
Zeile nur ``2·w`` Einträge > 0. ``_viterbi_banded`` (neu:
``backend/core/dsp/pyin_viterbi_fast.py``) nutzt das: Matrix-Operationen
statt Zeilen-Loop, **bit-identisch** zu ``librosa.sequence.viterbi``
(verifiziert auf realem Vinyl-, Rausch- und Ton/Stille-Wechsel-Material;
Tie-Break first-occurrence wie ``np.argmax``; ein Nicht-Band-Zustand
bräuchte > 701,5 Log-Einheiten Vorsprung — strukturell ausgeschlossen,
da alle Zustände je Frame denselben Unvoiced-Boden erhalten).

- ``pyin_fast``: 3,63 s vs. 8,74 s librosa je 30-s-Aufruf (2,4×,
  bit-identisch; Tests auf strukturierten Feeds 8192/48000)
- verdrahtet: phase_12 ``_estimate_pitch_pyin`` (deckt
  hybrid.``_apply_pyin`` ab) + phase_31 (3 Stellen) — je mit
  ``librosa.pyin``-Fallback (§V6 (copilot-instructions.md))
- phase_12 Produktionspfad (quality) auf realem Elke-Material:
  **45,5 → 10,6 s je 30-s-Chunk (~4,3×)** — Commit 5e6ae737
- **§PERF-R8b Rollout (7cac4b2e):** ``pyin_compat`` (Drop-in mit
  eingebautem librosa-Ersatzpfad, §V6 (copilot-instructions.md)) über
  alle 15 pYIN-Stellen: phase_19 (2), natural_performance_detector,
  vocal_register_detector (2), style_intent_detector,
  level_1_invariants_guard, forensics, hybrid_speed_pitch_ml,
  musical_goals_metrics, phase_56, singer_voice_model,
  vocal_ai_enhancement, vocal_focus_analyzer, zone_engine,
  harmonic_preservation_guard, audio_monitor — phase_19 auf Elke:
  63,4 → 30,1 s je 30-s-Chunk (2,1×)
- **Fast-Cell-Evidenz (10 s Vinyl, top_phases-Tooling):** Musical-Goals-
  Evaluation 81 s, Qualitätsprüfung 64 s, ERB-Masking je Chunk 31 s,
  phase_49 31 s, phase_09 14 s, phase_29 13 s — Engine-Ebene-Analyse-
  Kosten (Modell-Lasten kalt + je-Chunk-Evaluation) sind der nächste
  dokumentierte Hebel (eigene Session)
- SOTA4-4-Befund (FCPE): bereits als Torch-ROCm-Kern verdrahtet
  (§SOTA-ML-V7); kalt 5,56 s (Kern-Build 3,67 s einmalig), warm
  ~1,9 s je 30-s-Chunk — kein Port nötig

### §PERF-R9 (2026-09-19) — phase_49 WPE je Bin → batched BLAS (numerisch äquivalent)

Befund Fast-Cell (10 s Vinyl, top_phases): phase_49 30 s je Chunk war
**Attributions-Artefakt** — im Lauf griff das Reverb-Presence-Gate
(reverb_sev=0,000 < 0,15 ⇒ WPE übersprungen), die Wall-Zeit lag in
PLM-Eviction/GC/Gate-Overhead um die Phase. Auf halligem Material (wo WPE
wirklich läuft) dominiert dagegen der Per-Bin-Loop: `_predict_reverb_band`
= 88 % der Kanal-Zeit (cProfile, 10 s Vinyl: 1,93 s von 2,18 s).

Fix: `_predict_reverb_bands_batch` — Normalengleichungen als batched
BLAS-Matmul je F-Block (192 Bins) statt 12+ NumPy-Klein-Aufrufen je Bin;
Semantik erhalten (Silent-Bins=0, LinAlgError⇒0 je Bin im
Per-Bin-Fallback-Loop, Reverb-Assembly in convolve-Summenordnung,
§2.61-Budget-Guard zwischen Blöcken mit Teil-Ergebnis). Messung (48 kHz,
30 s, WPE aktiv): Per-Band-Loop 1,83 s → 0,91 s (2,0×), voller Kanal
3,22 s → 1,87 s (1,7×); max|Δ| 4e-14 je Band bzw. 1,4e-8 im Kanal-Ausgang
(Float-Rauschen der batched BLAS-Ordnung — weit unter jeder Gate-Schwelle,
§G5 (GEBOTE.md)-Determinismus gepinnt). Tests:
`test_phase_49_advanced_dereverb.py` (Äquivalenz, Silent-Bins,
Budget-Teil-Ergebnis, Determinismus — 16 Tests grün mit Phase-49-Suite).

### Fast-Cell-Nachmessung 2026-09-19 (51× RT auf 10 s Vinyl, fast cell)

`scripts/benchmark_effizienz_matrix.py --cells fast --seconds 10
--profile-top-phases 8` auf `vinyl_test_01.wav` (ROCm-Venv,
`output/perf_session_20260919/`): Qualitätsprüfung 136 s, Musical Goals
72 s, Audio-Nachbearbeitung 46 s, phase_49 30 s (Attribution, s. o.),
Defekt-Countdown 28 s, FeedbackChain 18 s, Phasenauswahl 18 s, Material/
Ära 16 s — **Engine-Ebene-Kosten = ~50 % des Laufs**. Aufschlüsselung der
Qualitätsprüfung: End-Gate-Kaskade 13+ measure_all-Runden — P1/P2-Blends
(2–3 Alphas) + **Universal-Cascade 8 feste Alphas [0,96…0,64] je volle
measure_all** (~8 s warm ⇒ ~66 s je Song) + Goosebumps/Recovery.
Warm-Miss-profile von measure_all (8,3 s): 5 ONNX-runs 4,9 s (HTDemucs
3,5 s budgetiert auf 2/Lauf, VERSA 1,7 s, PANNs-in-VERSA 1,2 s, MERT
2×0,25 s), transient-hpss 1,0 s, chroma_cqt 0,3 s.

### §PERF-R10 (2026-09-19) — Referenz-seitige Content-Caches in measure_all

Wahrheits-Korrektur der Hebel-Analyse: VERSA/PANNs/MERT laufen
KANDIDATEN-seitig (versa.score(audio), PANNs-in-VERSA, MERT auf Waerme/
Emotionalitaet) — über Alphas NICHT wiederverwendbar; der Rest-Hebel dort
sind GPU-Ports (R3-Muster, Parität rel ≤ 1e-3). Referenz-seitig pro
measure_all-Aufruf entfielen dagegen chroma_cqt + spectral_centroid der
Referenz (Authentizität, ~0,2-0,9 s) und die MERT-Original-Analyse in
`_compute_mert_similarity` (aesthetic_judgment/holistic_perceptual_gate,
je Chunk). UMGESETZT: `_MERT_REF_CACHE` + `_AUTH_REF_CACHE`
(content-keyed blake2b, Deckel 3 Einträge, bit-identische Semantik —
Cache-Treffer liefert exakt den Erst-Wert, §G5 (GEBOTE.md)-Determinismus
gepinnt; stft-Fallback-Zweig der Authentizität ignoriert die gecachte
CQT-Referenz repräsentations-korrekt; §V6 (copilot-instructions.md)
fail-closed unverändert). Messung 8-Alpha-Kaskaden-Simulation (10 s Vinyl,
ROCm): 32,3 s → 24,3 s (~0,9 s je Alpha, inkl. Warmlauf-Effekt). Tests:
`tests/musical_goals/test_musical_goals_metrics.py`
(TestPerfR10ReferenceCaches, 5 Fälle; Vollsuite der Datei 112 grün).

**Nächster dokumentierter Hebel:** (a) Kaskaden-Messkette — VERSA/PANNs/MERT
GPU-Ports (R3-Muster, Parität rel ≤ 1e-3) bzw. deren ONNX-Run-Kosten
(~2,0 s steady-state je Alpha); (b) Per-Phasen-Loop-Overhead
(PLM-Window-Eviction/OOM-Probes/Steering ≈ 1,6 s je Phase — 64 s je
10-s-Clip, skaliert mit Chunk-Zahl); (c) SOTA4-2 HR-V1-Budget-Urteil aus
dem nächsten überwachten Lauf (HEAD c8100879/616282a9-Stand). UTMOSv2-
Erst-Load (timm-Hub, 4 Folds) + FeedbackChain 23,5 s laufen einmal je
Song-Tail — dokumentiert.

### §PERF-R11 (2026-09-19) — Reinhör-Witness batched (je Phase ~1,5 s gespart)

Evidenz aus dem überwachten Lauf (Timestamps phase_ok→geplant je Phase):
**~10,5 s Phasen-Gap je 30-s-Chunk** — 41 Phasen × 8 Chunks ≈ 55+ min je
Song. Aufschlüsselung (cProfile): Reinhör-Witness ~1,5-1,7 s je Phase
(davon _frame_f0_hnr Frame-Loop, _loudness/_transient-RMS-Comprehensions,
pre_echo-Rolling-Perzentil 2000+ np.percentile-Aufrufe, 6 Band-FFTs auf
2 Signalen), dazu PMGG/CALIB/Coalition-Bookkeeping.

UMGESETZT (Report-only-Pfad, numerisch äquivalent):
- `_frame_f0_hnr`: batched FFT über alle Frames (sliding-window +
  axis=1-rfft/irfft, fftfreq gehoistet) statt Python-Frame-Loop —
  f0/voiced bit-identisch, hnr/flatness ≤ 1e-4; Kurzsignale behalten den
  Loop-Pfad.
- `_loudness_mod_depth_db`/`_transient_sharpness`: Sliding-Window-RMS
  statt Comprehension (≤ 1e-6).
- `pre_echo_ratio_db`: Rollendes Perzentil über sliding_window_view für
  Innen-Frames (Rand-Frames im Loop, identische Semantik).
- `_band_energy_ratio_db`: Content-keyed Spektrum-Cache (blake2b, Deckel
  4) — 6 Band-Aufrufe auf 2 Signalen rechnen 2 statt 6 FFTs (bit-identisch).

Messung (30 s Vinyl, ROCm): Witness 1,48 s → 0,57 s kalt / 0,41 s warm
(2,6-3,6×) — End-to-End-Witness-Felder identisch (as_dict-Vergleich alt/
neu), deterministisch (§G5 (GEBOTE.md)). ~1,1 s je Phase ⇒ ~6 min je
Song. Tests: `tests/unit/test_listening_witness.py::TestPerfR11BatchedPaths`
(5 Fälle, Suite 39 grün).

### §PERF-R12 (2026-09-19) — Stereo-Safety-Guard: Interaural-Profil-Cache (je Phase ~3 s gespart)

Der §2.51a-Mid-Pipeline-Stereo-Guard läuft nach JEDER Phase (pre/post) und
kostete 4,87 s je 30-s-Chunk (cProfile): `compute_interaural_profile` ×2
(je 2,42 s — Vollsignal-fft_crosscorr 0,5 s + 1200-Fenster-ITD-Loop mit
np.correlate 1,9 s + Python-Overhead 1,5 s). Da post von Phase N = pre von
Phase N+1, wurde dasselbe Signal je Phase doppelt profiliert.

UMGESETZT: (a) `_INTERAURAL_PROFILE_CACHE` — content-keyed (blake2b, Deckel
4), Cache-Treffer liefert exakt das Erst-Profil (bit-identische Semantik —
Guard-Verdikte unverändert); (b) Fenster-Preprocessing batched (std-Gate,
mean-Subtraktion, Denom-Dots via sliding_window_view/einsum; np.correlate
bleibt je Fenster — Korrelations-Werte bit-identisch). Messung (30 s
Stereo, ROCm): Profil 2,42 s → 1,62 s, Cache-Treffer 0,07 s; Guard je
Phase 4,87 s → ~1,7 s (ein frisches Profil + ein Treffer) ⇒ **~3,2 s je
Phase ⇒ ~17 min je Song**. ITD/Jitter identisch zum alten Loop (≤ 1e-6 µs).
Tests: `tests/unit/test_interaural_cues.py::TestPerfR12ProfileCache`
(2 Fälle; Interaural+Post-Pipeline-Stereo-Suiten 21 grün).

### §PERF-R13 (2026-09-19) — Mess-Tooling: Per-Phase-Exec/Gap-Zerlegung im Zellen-Log

Der Zellen-Log-Handler des Benchmark (`run_cell`) hatte KEINEN
Zeitstempel-Formatter — die im Docstring seit jeher vorgesehene
„Phasen-Zeitanalyse aus den ▶/✅-Zeilen“ war damit unmöglich (nur 3 Phasen
loggen „Profiling: Phase …“). FIX: asctime-Formatter im FileHandler +
`_parse_phase_timings_from_log()` — zerlegt jede Phase in exec
(geplant→phase_ok) und gap (phase_ok→nächstes geplant, d. h. Witness/
Stereo-Guard/PMGG/CALIB/Coalition-Maschinerie) und schreibt sie als
`phase_gaps`-Feld in das Ergebnis-JSON (n_phases, total_exec_s,
total_gap_s, top_gaps, je Phase exec_s/gap_s). Validierung auf dem
Supervised-Log (180 Phasen, 5 Chunks): exec 1657,7 s, gap 4458,3 s —
Chunk-Grenzen-Gaps (phase_41) sichtbar getrennt. Damit ist der
R11/R12-Effekt und der P2-Gap-Rest auf dem nächsten Lauf ehrlich messbar.
Tests: `tests/unit/test_benchmark_effizienz_matrix.py::TestPerfR13PhaseGapParser`
(2 Fälle, Suite 10 grün).

**§PERF-R13-Messung 2026-09-19 (Fast-Cell, 10 s Vinyl, saubere Maschine):**
Per-Phase-Gap (phase_ok→nächstes geplant) = **mean 2,26 s, median 2,05 s,
min 1,38 s, max 3,58 s** — der R11/R12-Stand halbierte den alten
~10,5-s-Gap (Supervised-Lauf, Alt-Code) auf ~2,3 s ⇒ **~8 s je Phase ×
41 Phasen × 8 Chunks ≈ 45 min je Song eingespart** (dazu die Phase-
Exec-Gewinne aus R9/R11). Gesamt-Wall der Fast-Cell blieb 52,7× RT
(526,5 s) — die Einzel-Blöcke variieren lauf-zu-lauf (Qualitätsprüfung
136→74,5 s durch R10-Caches, phase_49-Attribution +27 s = PLM/Cold-
Load-Rauschen des Einzel-Laufs); der verbleibende Gap-Rest 2,3 s je
Phase ist jetzt der P2-Fokus (PMGG/CALIB/Coalition — phasenabhängig).
Formatter-Korrektur: Zeitstempel jetzt geklammert (Parser akzeptiert
beide Formate).

### §PERF-R14 (2026-09-19) — FCPE/CREPE-Selbstheilung nach PLM-Eviction + pyin_compat-Fallback

Produktionsbefund aus dem Supervised-Lauf: phase_31 kostete 33-44 s je
30-s-Chunk statt ~2-3 s — Timestamps zeigten `FCPE plugin geladen
(model=dsp_pyin)`, danach 33 s „CREPE"-Detektion. Root-Cause: Die
PLM-Eviction (Fenster 5 Phasen, FCPE wird zwischen phase_12 und
phase_31 entladen) setzt nur die Session auf None; die Plugin-Instanz
blieb ohne Reload im Fallback — und der Fallback rief `librosa.pyin`
DIREKT (statt des §PERF-R8b-Kerns, 33 s je 30-s-Chunk). UMGESETZT:
(a) Selbstheilung im `analyze()` beider Plugins (BANQUET-Muster — Reload
bei session=None, §V6 (copilot-instructions.md)-fail-closed: Reload-
Fehler lässt die Fallback-Kette unverändert); (b) `_analyze_pyin` beider
Plugins nutzt `pyin_compat` (bit-identisch zu librosa.pyin, eingebauter
librosa-Ersatzpfad). Messung (Elke 30 s): FCPE nach simulierter Eviction
0,53 s mit model=fcpe_onnx (vorher 33 s dsp_pyin); pyin_compat
bit-identisch verifiziert. ⇒ phase_31 ~33-44 s → ~2-3 s je Chunk
(~35 s × 8 Chunks ≈ **~5 min je Song**), zusätzlich entlastet phase_12
(FCPE-Nutzung). Tests: `test_crepe_plugin.py` (Selbstheilungs-Test,
Fallback-Tests an pyin_compat adaptiert; 10 grün).

## SOTA-RESTHEBEL-MATRIX 2026-09-19 — schöpfen die DSP/ML-Hybride 100 % aus?

**Antwort: NEIN — weder Performance noch Wohlklang sind ausgeschöpft**
(Evidenz dieser Session):

- **Performance:** überwachter Lauf ~32× RT (Projektion ~2 h für 225 s;
  Ziel 15-25× RT, Akzeptanz ≤ 40 min), Fast-Cell 51× RT. Die 7900 XTX
  bleibt in großen Teilen idle: VERSA/PANNs/MERT/HTDemucs laufen als
  CPU-ONNX (Kaskaden-Messkette ~2 s steady-state je Alpha), und je Phase
  liefen bis §PERF-R11/R12 ~10,5 s Maschinerie (Witness+Stereo-Guard+
  PMGG/CALIB/Coalition) — nach R11/R12 noch ~5 s.
- **Wohlklang:** (a) Der validierte HR-V1-Gewinn (BigVGAN-Harmonik-Repair;
  A/B: af +0,0073, HNR +4,42 dB, PQS 4,52) ist mangels Budget-Urteil NOCH
  NICHT aktiviert (Flag off); (b) die größte dokumentierte Qualitätslücke
  des Referenzmaterials — HF > 12,9 kHz (bandwidth_loss conf=0,99) — ist
  offen (GPU-Finetune); (c) Separation-SOTA (VS-1/GSEP, Demucs v5) und
  der neurale Warp-Schätzer (WF-V4) sind extern blockiert (Gewichte).

**Stärkste Maßnahmen (Reihenfolge = Wohlklang-Gewinn × Machbarkeit):**

| # | Maßnahme | Wirkung | Status / nächster Schritt |
|---|---|---|---|
| Q1 | **HR-V1-Flag-Flip** (BigVGAN-Repair, additive_synthesis_gate) — A/B validiert | HNR +4,42 dB, af +0,0073, PQS 4,52 auf harmonisch beschädigtem Material | **BUDGET-URTEIL 2026-09-19: NEGATIV — Flag bleibt OFF.** Überwachter 225-s-Lauf (PERF-R10-Stand): Wall 12 137,5 s ≈ 3 h 22 min (RT-Bericht 32,0×, letzter Chunk 39× RT) — Akzeptanz ≤ 40 min weit verfehlt, KEIN Headroom für die 2,5×-RT-Synthese. Nächste Prüfung nach P1/P2-Gewinnen (GPU-Ports + Gap-Rest); Flip-Mechanik bleibt kartiert (Flag + Ready-Gate-Ausrichtung) |
| Q2 | **F4-FlashSR-HF-Rekonstruktion > 12,9 kHz** (16k→48k) | Air/Presence — schließt die größte Qualitätslücke (conf 0,99) | **UNBLOCKED (2026-09-19)**: Base-Checkpoint `models/flashsr/models/upsampler.pth` ✓, MUSDB18-HQ (150 Tracks) ✓, Rezept `train_flashsr_f4.py` ✓; Finetune-Lauf 2026-09-17 bei Epoche 6 abgebrochen (val_a1 0,45–0,52, noch nicht konvergiert, best 0,449@E4/0,516@E5) — **Resume** `--resume output/f4_flashsr/checkpoint_epoch6.pt --epochs 12+` nach Ende des Supervised-Laufs (GPU) |
| Q3 | **Separation-SOTA:** VS-1/GSEP + Demucs v5 | Separierungs-/Quelltreue-Sprung (P1-2) | EXTERN BLOCKIERT — offizielle Gewichte beschaffen (SongEval/Release-Kanäle) |
| Q4 | **WF-V4 neuraler Warp-Schätzer** | Wow/Flutter-Korrektur ohne Authentizitätsverlust (einziger Hörordnungs-sicherer Weg) | EXTERN BLOCKIERT — Checkpoint-Quelle klären |
| Q5 | **Blind-Hörstudie n≥30** (P1-4) | Kalibriert alle Hörordnungs-Schwellen auf echte Hörer | EXTERN — Hörer-Panel organisieren |
| Q6 | **PSY-A6 persönliche CIPIC-HRIR** | Binaurale Richtigkeit je Hörer | AUSBAUSTUFE — nicht blockierend |
| P1 | **R3-Vollausbau: VERSA/PANNs/MERT/HTDemucs → Torch-ROCm-Kerne** (Muster BSR 41,8×/BANQUET; Parität rel ≤ 1e-3) | Kaskaden-Messkette 8 Alphas × ~5 s → <1 s je Alpha; alle CPU-ONNX-Pfade entlastet; GPU ausgelastet | GPU-GEBUNDEN — Ports nach §III.9-Paritätsnachweis (strukturierte Feeds, rel ≤ 1e-3); CRePE/BANQUET/FCPE/Whisper sind Vorbild |
| P2 | **Phase-Gap-Rest zerlegen** (nach R11/R12 ~5 s je Phase: PMGG-Pre/Post, CALIB-Refresh, Coalition-Bookkeeping, STCG) — Muster R11/R12 (batched, content-cached, bit-identisch) | ~5 s × 41 Phasen × 8 Chunks ≈ 27 min je Song | OFFEN — §PERF-R13-Analyse des laufenden Logs (210 Phasen, 7 Chunks): mean 12,1 s, median 11,6 s, Spread 10,2–17,1 s je Folge-Phase — neben Witness (1,5 s) + Stereo-Guard (4,9 s) existiert eine **phasenabhängige Komponente ~3–6 s** (größte Gaps nach 01/02/05/09/24/27/28/50). Nach Lauf-Ende mit R11/R12-Code nachmessen und die variable Komponente per cProfile isolieren |
| P3 | **PLM-Residency:** stabile Modelle über die volle Phase-Sequenz halten (Fenster 5 → alle verbleibenden Phasen, RAM-gedeckelt) | entfällt Reload-Thrash für spät wiederverwendete Modelle | OFFEN — §2.37-Window-Politik mit RAM-Messung kalibrieren |
| P4 | ~~Mess-Tooling: top_phases um stille Blöcke erweitern~~ | ehrliche Hebel-Rangliste | **✅ ERLEDIGT durch §PERF-R13** (Zeitstempel-Formatter + Per-Phase-Exec/Gap-Zerlegung, `phase_gaps`-Feld im Ergebnis-JSON) |
| P5 | **Demucs-Separations-Budget** (2 echte Läufe je Checker) auf GPU-Mini-Batch oder ehrlich kleineres Modell | measure_all-Erstkosten 3,5 s × 2 je Song entfallen | OFFEN — Qualitätsentscheidung (Hörordnungs-Gate) |

**Q1-Vorbereitung (2026-09-19, Mechanik kartiert):** Flip-Punkt =
`plugins/bigvgan_v2_plugin.py:105` (`BIGVGAN_V2_HR_ACTIVATED = False` →
`True`, Einzeiler). Zwei Gates: (a) Budget-Urteil aus dem laufenden
überwachten Lauf (Synthese 2,5× RT GPU, F3-Messung); (b)
`bigvgan_v2_ready()` prüft `models/bigvgan/bigvgan_v2.pth` (FEHLT), der
Loader (`_try_load_model`) nutzt dagegen `bigvgan_v2.onnx` (VORHANDEN) —
beim Flip Ready-Gate auf den tatsächlichen Checkpoint ausrichten
(oder .pth beschaffen), sonst bleibt der Pfad fail-closed. Die
Activation-Contract-Tests sind bereits flag-bewusst (beide Zustände).

### Überwachter 225-s-Lauf 2026-09-19 (PERF-R10-Stand) — Befunde

`output/supervised_run/elke_225s_perfr_r10.{log,wav}` (Elke Best, Vinyl,
8 Chunks, 297 Phasen, Wall 12 137,5 s ≈ 3 h 22 min, RT-Bericht 32,0×,
letzter Chunk 39× RT). **Qualität:** MUSHRA 92,1 (Excellent), UQM 87,2
PASS, artifact_freedom 0,9869 ✓, Audibility-Gate 0 hörbare Restdefekte ✓,
Goosebumps 0,979, WCS 0,966 ≥ 0,88 ✓, GOAL_SCORECARD (Song-Tail)
excellence 0,8203 mit 2 Verletzungen. **Befunde (Abarbeitung nächste
Session):** (a) **Wohlklang-Ordnung VIOLATION — 8 Verstöße dekodiert:**
`groove` (Stufe 4) und `spatial_depth` (Stufe 4) verbessert auf Kosten von
`waerme` (Stufe 2), `timbre_authentizitaet` (Stufe 1), `transparenz`
(Stufe 3), `listening_fatigue` (Stufe 3) — lexikografische Ebene-1-
Prüfung vs. Input (joy_runtime_index zeigt gleichzeitig fatigue_index
0,18; FC-Loop hat per-Kandidat-Verstöße bereits verworfen, der Netto-
Saldo bleibt negativ — Hörordnungs-Forensik: welche Phasen treiben die
waerme/timbre/transparenz-Senkung?); (b) **DEGRADED_EXPORT**
(recovery_state_machine; fail_reasons VQI 0,7498 „acceptable" —
§0c-Vertrag: bestmögliches Ergebnis exportiert); (c) Einladungs-Gate
Sharpness-Sprung 0,243 (corrected=True); (d) interaction_guard-Rollback
phase_47_truepeak_limiter (P1/P2-Drift −0,656, tol −0,242) —
Rollback-Count 3 im Strict-Conflict-Report. Der Lauf lief auf dem
PERF-R10-Code-Stand (R11/R12 kamen während des Laufs) — die
Neumessung mit R13-Tooling läuft separat (`output/perf_session_20260919_r13/`).

## TODO SOTA 4 (2026-09-18) — Beschleunigung + höhere Restaurierungsqualität (Welle 4)

> Reihenfolge = (Hör-Gewinn × Machbarkeit) je Aufwand. Mess-Kadenz: Die
> v1023-Lauf-Endwerte (Hot-Phase-Rangliste) entscheiden die Reihenfolge von
> SOTA4-1 und SOTA4-2. Kein Trade-off: jede Maßnahme beschleunigt UND/ODER
> verbessert die Hör-Qualität — Qualitätskompromisse sind ausgeschlossen.

| ID | Maßnahme | Wirkung | Status |
|---|---|---|---|
| SOTA4-1 | PANNs-Multi-Window-Parallelisierung (VFA-Kette): Fenster sind unabhängig, ORT-`run` thread-sicher ⇒ ThreadPool nach BANQUET-P8-Muster, bit-identisch | ~10–20 s/Chunk; VFA (≈70 s) entlastet | ✅ UMGESETZT 2026-09-19 (26df39af): Opt-in-Parallelpfad (`INFER_PARALLEL`, Default 4), Maximum in fester Reihenfolge ⇒ bit-identisch; Resample-Wiederverwendung; Pre-Commit grün |
| SOTA4-2 | HR-V1-Flag-Rollout (BigVGAN-Repair, `additive_synthesis_gate`): A/B-Validierung bestanden (af +0,0073, HNR +4,42 dB, PQS 4,52) — Budget-Nachweis mit dem neuen Headroom führen | messbar höhere Reparaturqualität (Harmonik/HNR); Kost nur die Synthese-Teile (~2,5× RT/10 s GPU) | OFFEN — Budget-Urteil nach Neumessung mit §PERF-R4-Torch-Pfad (phase_01 ~8× schneller als im v1023-Lauf; Headroom-Urteil nach dem nächsten Lauf) |
| SOTA4-3 | F4-FlashSR-Musik-Finetune (HF-Rekonstruktion > 12,9 kHz): größte dokumentierte Qualitätslücke des Referenzmaterials (bandwidth_loss conf=0,99) | Air/Presence (MUSHRA-Proxy VocPres/ISO226) | GPU-GEBUNDEN — Rezept vorhanden (train_flashsr_f4.py), 16k→48k |
| SOTA4-4 | FCPE-Torch-ROCm-Port nach BSR-Muster (ORT-ROCm-EP rel=0,93 defekt ⇒ derzeit CPU): VORAB echten FCPE-Zeitanteil je Chunk messen | unklar — im phase_12-Profil nicht unter den Top-20; Port lohnt nur bei gemessenem Anteil | ✅ ERLEDIGT-DURCH-BESTAND 2026-09-19: FCPE ist bereits als Torch-ROCm-Kern umgesetzt und verdrahtet (§SOTA-ML-V7, fcpe_torch_rocm.py — ORT-ROCm-Defekt rel 0,19 beseitigt); Messung im phase_12-Profil: kalt 5,56 s (Kern-Build 3,67 s einmalig), warm ~1,9 s je 30-s-Chunk |
| SOTA4-5 | Whisper-GPU (HF-Decoder): auf diesem Host inert (Blob-Store-Symlinks gebrochen ⇒ ONNX/DSP-Fallback), auf anderen Hosts aktiv | ~10–20× auf dem Transkriptionsschritt je Song | DOKUMENTIERT — kein lokaler Aufwand |
| SOTA4-6 | P11 BANQUET-Mini-Batch (dynamische Batch-Dim): Trainingscode fehlt im Repo (banquet_infer.py ohne Architektur, nur Checkpoint) — optional Architektur-Re-Engineering aus dem Checkpoint | GPU-Deckel 1,19× → potenziell 5–10× auf phase_09 | ✅ TEILWEISE-DURCH-§PERF-R4 2026-09-19: TORCH_BATCH-Default 4→32 (bit-identisch zu B=4, gemessen 7,85→6,49 s je 30 s; Clamp 40 wegen MIOpen-B≥48-Defekt) — das Re-Engineering des Trainingscodes bleibt gestrichen (kein Qualitäts-Nutzen, nur Batch-Dim) |
| SOTA4-7 | phase_28-Session-Hoist (ONNX-Session je phase_28-Aufruf statt je Prozess) — kleinteiliger Restposten | ~1–3 s/Chunk | ✅ ERLEDIGT-DURCH-BESTAND 2026-09-19 (Messung): LGE-Whisper ist bereits Singleton (lädt 1× je Prozess, v1023-Log zeigt je Chunk KEINE Whisper-Load-Zeilen); warmes phase_28 = 5,3 s/30 s, davon 1,7 s Chunk-Re-Transkription — bewusst inhärent (Maske muss das VERARBEITETE Chunk-Audio reflektieren, WoW/Flutter-Stretch ⇒ Song-Timeline wäre nicht exakt) |

**Akzeptanz je Maßnahme:** (1) bit-identisch oder durch Never-worsen-Gates/
PMGG/Reinhör-Witness bestätigt (§0/Gesamtkonzept §6); (2) deterministisch
(§G5 (GEBOTE.md)); (3) Messwerte im Lauf-Protokoll dokumentiert (RT-Faktor
+ Goal-SCORECARD/af-Delta); (4) §V6 (copilot-instructions.md)-Fail-closed unverändert.

**Bewusst NICHT Teil der Welle (Qualitätskopplung, §V7/Hörordnung):**
Phase-Umordnung (R7-Reihenfolge ändert den Output), FC-Loop-/Gate-Reduktion
(normative Qualitäts-Maschinerie), Audibility-Globalcache in phase_27 (lokale
Maskierung je Defekt-Kontext — ein Cache wäre nicht exakt).

### Gemessene Hot-Phase-Wahrheit (Profiling 2026-09-16)

- Treiber sind ML-Load/-Inferenz (BANQUET 0,78 s/Prozess = 62 % der Phasen-Zeit,
  Device-Detection 0,26 s), NICHT DSP-Multiscale (nur ~3 s/225 s, 0,013× RT) —
  die frühere Attribution „phase_01 = Multiscale-Bottleneck“ war falsch.
- BSR-Torch-ROCm-Port als Muster für künftige Ports: 41,8× (461 ms vs. 19,3 s),
  Parität max_abs ≈ 1e-5, wenn ORT-ROCm-EP-Kernels numerisch defekt sind (SOTA-BSR-GPU).

---

### ROADMAP-ABSCHLUSS-MATRIX (alle noch offenen Punkte, Stand 2026-09-15)

| Punkt | Status | Begründung / nächster Schritt |
|---|---|---|
| TODO-P0-1 (53×→32×-Laufzeit) | **TEIL-ERLEDIGT (Messung) 2026-09-15; Rest GPU-GEBUNDEN** | Hot-Phase-Messung geliefert: `compute_hot_phases` im Diagnose-Skript (rt_factor je Phase, Hot-Liste ab 0,5× RT, test_p0_1_hot_phase_report.py). **Attributions-Korrektur 2026-09-16 (Profiling):** phase_01s 4,4×-RT-Attribution „DSP-Multiscale“ war falsch — Multiscale kostet nur 3 s/225 s; Treiber sind ML-Load/-Inferenz (BANQUET/Device-Detection). **Song-Level-Hoists 2026-09-17 (je Chunk-Wiederholung entfernt):** Struktur-Hoist ANA-6 ✅, **LGE-Transkription-Hoist ✅ (Whisper 8×→1×, Timeline je Chunk zeitverschoben, commit 3bfa5215)**, Export nur nach Assembly ✅ (04a52839); PANNs-Tags + Defect-Scores + Gender sind bereits Song-Ebene (Pre-Analyse-Cache, verifiziert). **R3-GPU-Ports 2026-09-17 abgeschlossen (Registry-Verdikte + Produktions-Vertragstest `test_production_registry_verdicts_restoration_models`):** BANQUET → ROCm ✅ (Partitioning-Fix 2026-09-13, Funktions-Validierung identisch; Deckel 1,19× — der Export ist batch-1-spezifisch, Mini-Batch scheitert in `node_view`-Reshapes, gemessen), CRePE-Pitch → ROCm ✅ (28×), DeepFilterNet → ehrlich CPU ✅ (GPU-Overhead dominiert bei Mini-Modellen). Ein Song-Level-Hoist der Inferenz ist NICHT äquivalent (phase_09 verarbeitet den phase_08-Ausgang je Chunk) — R3 war der korrekte Weg. **Analyse-Cache je Datei-Hash 2026-09-17 ✅ (Disk-Persistenz der Bridge-Analyse-Caches `output/analysis_cache/`, Read-/Write-Through unter dem In-Memory-LRU, AURIK_VERSION-Invalidierung, §V6-fail-closed, Kill-Switch AURIK_ANALYSIS_CACHE=0, 6 neue + 21 bestehende Tests grün):** Wiederholungsläufe am selben Song überspringen die komplette Voranalyse über Prozessgrenzen. **P6/P7 2026-09-17 ✅ (Architektur-Feststellung: Batch ist bereits Ein-Prozess + ThreadPool(4); §V8-Transient-State-Lücke des BANQUET-Singletons geschlossen — `reset_for_song()` + `_state_lock`, verdrahtet in `_restore_chunked`, 19 Tests grün).** OFFEN: nur noch GPU-gebundenes (F4/F5, SOTA-ML-V5) + Laufzeit-Verifikation im nächsten Lauf | hängt an den GPU-Buildouts F1–F5 + Residency-Gewinnen + den restlichen Hoists |
| TODO-P0-2 (Per-Session-Kompilierung) | **EXTERN BLOCKIERT** | ONNX-Compile-Strategie; Folge von P0-1/C |
| TODO-P0-3 (Budget-Wahrheit) | ✅ GESCHLOSSEN 2026-09-15 | s. o. A |
| TODO-P1-1 (Residency) | ✅ GESCHLOSSEN 2026-09-15 (Policy) | s. o. C; Laufzeit-Gewinn misst P0-1 |
| TODO-P1-2 (VS-1/GSEP + Demucs v5) | **EXTERN BLOCKIERT (Gewichte)** | VS-1/GSEP: keine offiziellen öffentlichen Weights verifizierbar (SongEval nicht erreichbar); Demucs v5-Beschaffung offen |
| TODO-P1-3 (Audibility-Guards) | ✅ GESCHLOSSEN 2026-09-08 | SCK/WBG/ATI/Formant/Gain-Step umgesetzt |
| TODO-P1-4 (Blind-Hörstudie) | **EXTERN BLOCKIERT (menschliche Hörer)** | §0c-Export-Bug geschlossen (B); n≥30-Studie braucht Hörer |
| TODO-P1-5 … P1-12 | ✅ GESCHLOSSEN 2026-09-08 | Status im jeweiligen Abschnitt |
| PSY-A1/A2/A3/A4/A5/A8 | ✅ GESCHLOSSEN 2026-09-14/15 | Rollouts in den Tabellenzeilen; PSY-A2-zeitvariant via P5 |
| PSY-A6 (CIPIC-HRIR) | AUSBAUSTUFE (dokumentiert) | persönliche HRIR; nicht blockierend |
| R1/R4/R5/R8 | ✅ GESCHLOSSEN 2026-09-15 | s. o. G + Welle 1 (R4-Benchmark, R5-Zertifikat, R8-Infrastruktur); **R8-Per-Phase-Rollout: phase_59 umgesetzt** (Defekt-Maske als Rechen-Maske, sub-STFT-Fenster unverändert, Coverage-Fallback ≥0,85). **Folge-Befund 2026-09-16 (Profiling):** ein Rollout auf phase_01/19 hätte KEINEN Nutzen — die Roadmap-Attribution „phase_01 = DSP-Multiscale-Bottleneck“ war falsch: `_detect_clicks_multiscale` kostet nur ~3 s/225 s (0,013× RT, 11 % der Phasen-Zeit); Treiber sind der einmalige BANQUET-ML-Load (0,78 s/Prozess, 62 %) und die ML-Device-Detection (0,26 s) bzw. im Lauf die CPU-Inferenz nach SUP-F2. phase_19 hat das Sparse-Muster bereits (segment-selektives Gate). R8-Rollout damit EVIDENZBASIERT BEENDET — der P0-1-Rest ist GPU-/ML-gebunden (R3) |
| R2 (MuQ-MOS-Gate) | ✅ GESCHLOSSEN 2026-09-16 | **Produktionsverdrahtung**: `OneTakeExport.prepare(reference_audio=…)` + `one_take_prepare` + beide uv3-Pfade (Whole-Song + Chunked) reichen den Original-Input an `ExportQualityGate.check` durch; MuQ-Felder im Quality-Report, in `result.metadata` (`export_muq_mos_delta/in/out`) und im Bridge-Payload (`muq_mos_witness`). Witness bleibt **SOFT** (blockt nie, §0c — WIT-M4: Zeuge, kein Richter). Tests: `test_r2_muq_mos_export_wiring.py` |
| R3 (ROCm alle Modelle) | **GPU-GEBUNDEN** | Ports nach BSR-Muster |
| R6 (Per-Song-Zielklang DDSP) | **GPU-GEBUNDEN** | F5/C4 |
| PSY-A1-Rest (25/31 JND) + P65-Witness | ✅ GESCHLOSSEN 2026-09-16 | **phase_25**: Azimut-Schwelle JND-basiert (ITD-JND 30 µs × 3,5 ≈ 5 Samples @ 48 kHz, SR-unabhängig, HF-Floor ≥ Pegel-JND; Bestandsverhalten bit-identisch). **phase_31**: 0,3-%-Speed-Schwelle als JND-gestützt dokumentiert (max-Floor mit Frequenz-JND 0,2 %). **phase_65**: Sänger-Identitäts-Witness als Per-Phase-Gate (`_apply_singer_identity_witness`, S4-Muster: Resemblyzer cos ≥ 0,92, sonst proportionaler Blend Richtung Input, non-blocking §V6). **phase_43 (2026-09-16): Sibilanten-Maskierungs-Gate verdrahtet** — subaudible Sibilanten-Segmente (unter Maskierungsschwelle, Band 4–12 kHz, Muster phase_19) bleiben ungezähmt, Zähler `subaudible_sibilants_skipped`; test_phase_43_ml_deesser (48 Tests grün). Damit ist die PSY-A1-Expansionsmatrix-Zeile für 19/43 vollständig. Tests: `test_psy_a1_jnd_gates_25_31.py`, `test_phase_65_singer_identity_witness.py` |
| R7 (Adaptive Rescheduling) | ✅ UMGESETZT 2026-09-18 (Kern-Slice) | Impact-Schätzer + Null-Impact-Deferral + Hör-Impact-Schutz im PerformanceGuard; ohne Scores Fix-Priorität (bit-identisch); 12 Tests (`test_hearing_impact.py`) |
| F1/F3/F5, Q4/Q7 + F3-Finetune-Teil (GPU-Buildouts) | **GPU-GEBUNDEN** | Trainings-/Port-Arbeit auf ROCm; **F2 + Q5 GESCHLOSSEN 2026-09-16** (faire S1-Validierung verwarf Pfad B, fail-closed); **F3-A/B-Teilvalidierung PASS 2026-09-16** (af +0,0073, HNR +4,42 dB — Flag-Rollout offen bis Test-Suite-Anpassung + UV3-Budget-Nachweis). **Budget-Messung 2026-09-16:** HR-V1-Synthese (Torch-ROCm, Elke-Best-Material) = **2,5× RT** (25,05 s für 10 s, 26 Bänder released, PQS 4,93); CPU >10× RT. Da der Gesamtlauf aktuell bei 33× RT liegt (32×-Ziel), bleibt das Flag vertragsgemäß OFF bis der Gesamt-Budget-Nachweis mit Headroom steht — die Activation-Contract-Tests sind bereits flag-bewusst (beide Zustände). Nächster Schritt: R3/ROCm-Ports + P0-1-Gewinne, dann Flag-Flip als Einzeiler |
| WF-V4 (neuraler Warp-Schätzer), TP-V2 | **EXTERN BLOCKIERT (Checkpoint-Quelle)** | Quelle klären + Download |
| P1-Folge (af-Never-worsen in 07/17/19/38) | ✅ GESCHLOSSEN 2026-09-15 | `artifact_freedom_guard.py` (billiger click/pre-echo-Delta-Guard, ≈0,008× RT) in 07/17/19/38 + `--fail-delta`-CI-Gate im Diagnose-Skript; Diagnose-Vergleich: phase_07 Δ−0,136→−0,049, phase_17 −0,084→+0,002, phase_19 −0,062→0,000, phase_38 −0,043→+0,061; Ketten-Min-af 0,473→0,631 |
| PSY-A7 (10/11/40) | ✅ GESCHLOSSEN 2026-09-15 | `perceptual_loudness_cap.py` (Rollout-Helfer mit Headroom-Variante) in 10/11/40: 10/11 nie über Input-Lautheit, 40 kappt Kurzzeit-Pumping über dem Uniform-Gain (Ziel-LUFS-Anhebung bleibt legitim); test_psy_a7_loudness_cap_rollout.py |
| Defizit-Sweep (57 vorbestehende Unit-Fehlschläge) | ✅ GESCHLOSSEN 2026-09-15 | Test-Drift auf aktuelle Specs (§v10.739 MDX23C-Entfernung, §v10.748 ONNX-first, §Q11 Gesangs-Drosselung, §v10.742 Residency, §2.53b-Log) + echte Bugs (phase_29-Strength-Vertrag −14 dB→~0, SeparationFidelity-Cache 0,746-vs-0,996, Router-TypeError-Retry, htdemucs-Duck-Typing, lyrics-Crossfade-Broadcast, MuQ/easydict-Imports); 2 Commits, Pre-Commit grün |
| Defizit-Sweep 2 (5 Vollsuite-Fehlschläge 2026-09-16) | ✅ GESCHLOSSEN 2026-09-16 — **Vollsuite 16948 passed / 0 failed bestätigt** | i18n-Gap (alle 17 help.error.*-Keys DE+EN, Tests auf übersetzte Texte); Gacela-Shim idempotent + data/utils-Cache-Purge; MuQ-sys.path-Pollution (scoped try/finally, Validierungs-Skript eigenständig); BS-RoFormer-CPU-Retry explizit CPUExecutionProvider; MuQ-Determinismus: AURIK_MUQ_GPU=0 vor dem Singleton-Cache geehrt + Device-Sync für warmgeladene Modelle + Test-Hermetik gegen ML-Speicherbudget-Erschöpfung; §2.46f Edge-Gain-Cap + Konvex-Edge-Taper im phase_03-ML-Pfad (neues Modul edge_gain_cap.py, 8 Tests — Intro/Outro nie > +2 dB über Original und hochkorreliert, auch nach warmem ML-Zustand) |

> **Mess-Kadenz:** Nach jedem Schritt das Messprotokoll wiederholen (restaurierter
> Score + Delta zum Original) und die Tabelle oben aktualisieren. Ein Schritt wird
> nur behalten, wenn er den Score erhöht (Never-worsen, §0).

## Überwachter 225-s-Lauf 2026-09-16 (Elke Best, voller Song) — Befunde & Abarbeitung

> Ausführung: `cli/aurik_cli.py --mode Restoration --bit-depth 24`, voller Song
> (225,3 s @ 44,1 kHz, mp3→vinyl-Kette). Lauf: `output/supervised_run/elke_225s_supervised_v1020.{log,wav}`,
> Report: `docs/reports/supervised_runs/2026-09-16_elke_best_225s.md`.
> Vollsuite vor Lauf: 16 948 passed / 0 failed (Commit 9e51c4b).
> **Lauf-Ergebnis (RUN_RC=0):** Export ordnungsgemäß (PCM_24, 48 kHz, 225,33 s,
> Quality-Gate passed=True/ok, OneTakeExport PASS TP=−1,7 dBTP/LUFS=−16,
> Peak −8,6 dBFS, kein Clipping/NaN/Inf); HPI 0,87 passed; MUSHRA 95,1 Excellent;
> §Hörbarkeits-Gate BESTANDEN (total=0 → keine hörbaren Restdefekte);
> GOAL_SCORECARD violations=0; Gesamtlaufzeit ~2 h 03 min ≈ 33× RT (224-s-Akzeptanz ≤ 40 min verfehlt — P0-1).
> Zweck: Plausibilitäts-, Bug- und Optimierungsprüfung für ALLE Importsongs
> (Wohlklang für das menschliche Ohr + maximale Performance) — nicht nur dieser Song.

### Befunde (Reihenfolge = empfohlene Abarbeitung)

| ID | Befund | Status |
|---|---|---|
| SUP-F1 (PERF, alle Songs) | **PANNs lief auf ROCm immer CPU**: der lokale Provider-Filter verglich (Name, Options)-TUPEL gegen String-Namen → GPU-Provider wurde immer verworfen (`PANNs: GPU-Inferenz angefordert …, aber Sitzung nutzt nur CPU`). | ✅ GEFIXT 2026-09-16: Tupel-Name wird ausgepackt; Regressionstest `test_panns_rocm_provider_filter.py` (2 Fälle) |
| SUP-F2 (QUALITÄT, alle Songs) | **BANQUET-Temp-WAV hart auf 44,1 kHz kodiert** (Pipeline fährt 48 kHz → Speed-/Pitch-Korruption des ML-Pfades) **+ Float-WAV**, das externe Reader (Docker/scipy) als „Format not recognised“ ablehnen → ML-Knistern-Pfad tot, DSP-Ersatz lief. | ✅ GEFIXT 2026-09-16: echte `sample_rate` + `subtype="PCM_16"` in `phase_09_crackle_removal._remove_crackle_ml` |
| SUP-F3 (DIAGNOSTIK, alle Songs) | **§V44-Meldung invertiert**: `ok=False` (IACC ≥ 0,70 = schmales Stereobild) wurde als „Mono-Kompatibilitätswarnung“ geloggt — Near-Mono-Vintage (IACC=0,89) ist perfekt mono-kompatibel; irreführende Warnung auf allen schmal-stereofonen Songs. | ✅ GEFIXT 2026-09-16: Meldung korrigiert („schmales Stereobild, kein Defekt“) |
| SUP-F4 (PERF) | **Chunk-1-Laufzeit 62,2× RT** (1866 s für 30 s) — P0-1-Ziel 32×, 224-s-Akzeptanz ≤ 40 min; ML-Schwere Treiber: MuQ/MERT/RMVPE/PESTO/CREPE-Ladungen + PANNs-CPU-Bug (SUP-F1). Vollauflösung hängt an GPU-Buildouts + Residency (P0-1-Matrix). | TEIL-BEHOBEN (SUP-F1); Rest = GPU-GEBUNDEN (F-Reihe) — Dokumentation P0-1 |
| SUP-F5 (DIAGNOSTIK) | `RestorabilityEstimator: time Grenze exceeded (19,17 s > 5,0 s)` — Erst-Ladezeit des MuQ-GPU-Modells sprengt den 5-s-Guard; einmalig je Prozess, Meldung ist Rauschen. | ✅ GEFIXT 2026-09-16: MuQ-ML-Prior wird separat getaktet und vom §2.26-DSP-Budget abgezogen — der einmalige Modell-Erst-Load löst keine Budget-Warnung mehr aus (nur noch INFO „Erst-Load einmalig je Prozess“); das 5-s-Budget prüft jetzt den echten DSP-Anteil |
| SUP-F6 (DIAGNOSTIK) | Reinhör-Witness meldet `pre_echo`/`roughness_increase` bei ~0-Deltas (pitch=0.0c, loud=0.0dB) — Schwellwert-Kalibrierung prüfen (report-only, keine Rollbacks). | ✅ GEFIXT 2026-09-16 (Kalibrierung, keine Workarounds): (a) `pre_echo_model`: Ratio wird auf HINZUGEFÜGTER Energie (positive Delta-Hälfte) gebildet + zweite Audibility-Schwelle relativ zum lokalen Vor-Fenster-Signal + absolute −60-dB-Hörbarkeits-Schwelle (leise Onsets, wo relative Schwellen gegen 0 gehen — Produktionsfall phase_01 micro_fallback Δ=+0,00 dB und phase_47-Limiter reproduziert: jetzt −200). (b) `listening_witness`: Rauigkeits-Anstieg wird JND-gegated (≤ 35 % relativer Anstieg = 2× Vassilakis-JND ≈ 17 % → auf 0 geklemmt; Produktionsbefund: harmloser 30-Hz-Hochpass ergab +1,16 bei Skala ~35632 auf realer Musik) — Findings UND Veto-Loop erben EINE Wahrheitsquelle. Tests: test_pre_echo_model (2 neue Fälle), test_listening_witness (2 neue Fälle) |
| SUP-F7 (VERIFIKATION) | Export-Gate arbeitet korrekt delta-basiert: `af=0.000 verworfen (false-positive gegen degraded Eingabe)` + OneTakeExport BEST-EFFORT (LUFS −19,3 für ruhigen Vintage) — §0c-Vertrag hält. | ✅ BELEG (kein Fix nötig) |
| SUP-F8 (DIAGNOSTIK, alle Songs) | WohlklangOrdnungGate-Audit meldete §Ebene-3-Verletzungen bei mikroskopischen Deltas (Audit-Epsilon 1e-9 vs. FC-Veto 0.012) — irreführende „Stufe-4-Gewinn auf Kosten Stufe-1“-Warnungen, während der korrekte Veto still blieb. | ✅ GEFIXT 2026-09-16: Audit-Schwelle = GPP-REGRESSION_EPSILON (eine Quelle der Wahrheit, kein neuer Schwellwert) |

### Abarbeitungsstand

1. ✅ SUP-F1 (PANNs-ROCm-Tupel-Filter) — gefixt + getestet.
2. ✅ SUP-F2 (BANQUET-SR/PCM-Fix) — gefixt; Verifikation im nächsten Lauf.
3. ✅ SUP-F3 (§V44-Meldung) — gefixt.
4. ✅ SUP-F8 (WohlklangOrdnung-Audit-Epsilon) — gefixt.
5. ✅ SUP-F5 (RestorabilityEstimator-Budget) + SUP-F6 (Witness-Kalibrierung) — gefixt 2026-09-16 (s. Matrix); SUP-F4-Rest GPU-gebunden.
6. ✅ PSY-A1 phase_43 (Sibilanten-Maskierungs-Gate) — verdrahtet 2026-09-16.
7. ✅ R8-Rollout — evidenzbasiert BEENDET (Profiling: phase_01/19-Rollout ohne Nutzen; Attribution korrigiert).
8. ✅ F3-Budget-Messung (HR-V1 = 2,5× RT GPU) — Flag bleibt vertragsgemäß OFF bis Gesamt-Budget-Nachweis.

### Export-Analyse 2026-09-16 (Restdefekte + Optimierungspotenzial)

> Gemessen auf der exportierten Datei (Mono-Mix, 48 kHz) vs. Original:
> Report: `docs/reports/supervised_runs/2026-09-16_elke_best_225s_export_analyse.md`.

| Befund | Wert | Bewertung |
|---|---|---|
| Eingeführte Restdefekte (Restaurierungs-Schäden) | **0** — Oktavband-Δ exakt ±0,00 dB (alle 7 Bänder), Rauigkeit +0,000 asper, Pre-Echo −200 dB (Floor), af=0,998, §Hörbarkeits-Gate total=0 | ✅ Never-worsen perfekt eingehalten |
| MuQ-MOS (10-s-Clips) | orig 4,84 → rest 4,85 (Δ +0,01) | ✅ keine Qualitäts-Verschlechterung |
| Crest / Spektral-Zentroid | 4,1 → 4,1 · 621 → 625 Hz | ✅ Dynamik-/Klangcharakter erhalten |
| Verbliebene Quell-Charakteristika (Scanner-Flags) | bandwidth_loss 0,99, hf_remanence_loss 0,98, inner_groove_distortion 0,97, wow/flutter, reverb_excess, soft_saturation | Ära-authentisch (1960er-Vinyl-Kette, BW 12,9 kHz) — Hörordnung Stufe 1 (authentizitaet) verbietet aggressive „Korrektur“ |

**Optimierungspotenzial (maximaler Wohlklang, geordnet):**
1. **HF-Rekonstruktion > 12,9 kHz** (bandwidth_loss/hf_remanence_loss conf≈0,99) →
   SOTA-ML-V1 FlashSR-Finetune / HR-V1 BigVGAN (GPU F4/F3, additive_synthesis_gate)
   — Hebel: Air/Presence (MUSHRA-Proxy VocPres=0,500, ISO226=0,254).
2. **BANQUET-ML-Knistern-Pfad** war im Lauf tot (SUP-F2-Fix greift ab nächstem Lauf)
   → Rest-Innenrillen-Distortion adressierbar.
3. **PANNs-GPU (SUP-F1)** — verbessert Gesangs-Detektion in 01/19/43/65 (nächster Lauf).
4. **Lautheit**: Datei ≈ −18,4 LUFS — OneTakeExport hat −16-Ziel angewendet (PASS);
   optional moderates Ziel für moderne Wiedergabe.
5. **Wow/Flutter/Hall**: bewusst erhalten (musikalische Modulation/Hörordnung) —
   WF-V4 wäre der einzige authentizitätssichere Weg (extern blockiert).

## Arbeitsaufträge 2026-09-17 — Songaufbau, Gates-Sweep, Ablauf (Abarbeitung)

> Vier Arbeitsaufträge mit den restlichen Roadmap-Maßnahmen abgearbeitet.
> Beleg: `docs/reports/current/2026-09-17_song_structure_sota_abgleich.md`;
> Commit-Stand 2026-09-17 (Session „SOTA-Roadmap + Arbeitsaufträge“).

### AUF-1 · phase_43 Audibility-Gate

✅ **BEREITS VORHANDEN** (2026-09-16, verifiziert): Sibilanten-Maskierungs-Gate
im Segment-Gate (`defect_audibility`, Band 4–12 kHz, Muster phase_19),
Zähler `subaudible_sibilants_skipped`, test_phase_43_ml_deesser (48 Tests grün).
Keine Nachrüstung nötig.

### AUF-2 · Gates-Sweep über alle Phasen (Inventar + Nachrüstung)

Inventar (Scan 2026-09-17, **Korrektur 2026-09-17** — der erste Grep-basierte
Scan war FALSCH-NEGATIV: mit vollständigen Mustern
(psychoacoustic/onset/mikrodynamik/NPA/noise-texture/spectral-color/HNR/
loudness/delta) haben ALLE Phasen Guard-Ketten):

- **Verifiziert vollständig gegated** (Scan 2026-09-17): 18 (psychoakust.
  Masking, Onset, Mikrodynamik, NPA, Noise-Textur, Spektralfarbe, HNR),
  32 (Masking, Loudness, Delta), 54 (Masking, Onset, NPA, Spektralfarbe),
  57 (Masking, Mikrodynamik, NPA, Noise-Textur), 61 (Masking, Onset,
  Mikrodynamik, NPA, Noise-Textur), **62 (Masking-Clamp §2.62, Onset,
  Mikrodynamik, NPA, Noise-Textur, Spektralfarbe — die frühere
  Roadmap-Behauptung „62 ohne Maskierungs-Gate“ war FALSCH)**.
- **44/45 PSY-A4:** ✅ ERLEDIGT — Verdrahtung 2026-09-17 in phase_44/45
  (`equal_loudness_strength_factor` @ 3/3,5 kHz, 60 phon, nicht blockierend;
  die frühere Notiz „KEINE PSY-A4-Temperierung — Backlog mit Rezept“ war
  veraltet). Wiring-Tests 2026-09-17 ergänzt: Faktor multipliziert die
  Effektiv-Stärke (Faktor 0,7 → Stärke ×0,7; 10/10 grün). Hinweis: Bei
  3–3,5 kHz liegt der ISO-226-Faktor bei 60 phon bei 1,0 (maximale
  Gehör-Empfindlichkeit) — die Temperierung wirkt dort definitionsgemäß
  nicht reduzierend, die Verdrahtung ist aber vollständig.
- Analyse-/Passiv-Phasen (28/30/41/53) und verbotene Phasen (21/35/42)
  bewusst ohne Gate — korrekt.

**Nachgerüstet 2026-09-17:**
- **phase_09 crackle_removal** — `_apply_audibility_gate_to_regions`
  (subaudible Knistern-Regionen übersprungen, §4; Zähler
  `subaudible_crackle_skipped` in allen 3 Return-Pfaden). Wichtig: Der
  BANQUET-ML-Pfad ist seit SUP-F2 (2026-09-16) LIVE und lief vorher ganz
  ohne Hörbarkeits-Gate. Tests: 2 neue PSY-A1-Fälle (16/16 grün).
- **Latenter Test-Drift gefixt**: `test_phase09_crackle_in_vocal_passages`
  (2 Docker-Fallback-Tests waren seit der SUP-F2-Signaturänderung
  `_remove_crackle_ml(..., sample_rate)` rot — nicht bemerkt, weil der
  grüne Vollsuite-Stand davor lag).

**Backlog (Rezept je Phase, Folge-Slice):** —
**44/45 ✅ UMGESETZT 2026-09-17** (PSY-A4-Verdrahtung nach Muster
04/16/17/37/38/39; Hinweis: der Equal-Loudness-Faktor bei 3–3,5 kHz/60 phon
ist EXAKT 1,0 — das Präsenz-Band ist die Referenz der ISO-226-Kurve, die
Temperierung ist dort neutral und schützt zukünftige Kurvenänderungen).
Alle übrigen früher als „gate-los“ gelisteten Phasen haben verifizierte
Guard-Ketten (s. o. Korrektur) — kein Handlungsbedarf.

### AUF-3 · Songaufbauanalyse — SOTA-Abgleich & Upgrade

**Urteil: NICHT auf SOTA-Stufe; Analyseergebnis deckte sich NICHT mit dem
Songaufbau** (Elke-Best-225s: 0 Chorus/0 Klimax, 42 s „intro“, 37 s „bridge“
statt des klaren Strophe/Refrain-Wechsels mit Refrain bei ≈44/100/156/192 s —
umabhängig per Chroma-Fenster-Scan belegt). **Upgrade umgesetzt** (deterministisch,
librosa/numpy, §G5): Fenster-Wiederholungs-Evidenz (`_window_repetition_counts`),
Wiederholungs-Labels (chorus = wiederkehrend + Energie > Median), Intro/Outro nur
für kurze Rand-Segmente, Klimax = Chorus mit p90 ≥ 98 % des Song-Maximums.
Ergebnis deckt sich jetzt VOLLSTÄNDIG mit der unabhängigen Analyse; Laufzeit
1,29 s/min ≤ Budget. 17 Tests grün. Rest SOTA-Gap (GPU/dokumentiert):
ML-Boundary-Detektor, Beat-/Downbeat-Tracking (Spec erwähnt, fehlt),
Chunk-Modus-Struktur (siehe AUF-4).

### AUF-4 · Ablauf- & Analyse-Verbesserungen

1. ✅ **Ehrliche Laufzeit-Schätzung**: RestorabilityEstimator versprach
   2,5× RT + 15 s (≈ 9 min für 225 s) — gemessen ≈ 33× RT (~2 h). Kurve auf
   33× RT kalibriert (Budget-Wahrheit; aktualisiert sich mit P0-1/GPU-Gewinnen).
2. **Chunk-Modus**: `analyze_structure` läuft pro 30-s-Chunk (k=2, „2 Segmente“
   im Lauf-Log) — Rezept: Struktur EINMAL vor dem Chunk-Loop auf dezimiertem
   Ganz-Song-Probe + Wiederverwendung (Folge-Slice).
3. ✅ **BANQUET-Load**: Singleton + Quarantäne vorhanden (0,78 s einmal je
   Prozess, kein Fix nötig) — dokumentiert; CPU-Inferenz-Kosten bleiben
   R3/GPU-gebunden.
4. **Witness-Veto-Loop**: verdrahtet, aber im Lauf 0× aktiv (Schwellen greifen
   erst bei echten Regressionen) — Zustand dokumentiert.

---

## SOTA-Analogie-Sweep 2026-09-17 — weitere Korrekturen nach dem Songaufbau-Muster

> Frage: „Weitere SOTA-Korrekturmöglichkeiten analog zur Songaufbauerkennung?“
> Muster: Analyse-/Entscheidungskomponenten, die auf Stellvertreter-Heuristiken
> statt auf die definierende Evidenz bauen (oder Rollen duplizieren, §V7
> (copilot-instructions.md)).

| # | Befund (Fehlerklasse) | Status 2026-09-17 |
|---|---|---|
| ANA-1 | **Grenz-Erkennung dupliziert**: §2.52b nutzte agglomerative k-Heuristik (1/30 s), §2.17 hatte bereits die definierende SSM/Checkerboard-Novelty (Foote 2000) — zwei Analysatoren, zwei Methoden, potenziell widersprüchliche Sektions-Karten | ✅ **UMGESETZT**: kanonisches Modul `backend/core/dsp/ssm_segmentation.py`; §2.17 delegiert (37 Tests grün), §2.52b nutzt SSM primär (agglomerativer Fallback); Intro/Outro nur noch erstes/letztes Segment (Positionsregel verschluckte den 156-s-Refrain). Elke-Best: 12 evidenz-basierte Segmente, alle 4 Refrains + Klimax, 1,28 s/min ≤ Budget; neuer Test test_ssm_boundaries_detect_aba_transitions |
| ANA-2 | **Vokal-Aktivität**: `_estimate_vocal_activity` (spectral flatness + GLOBALE PANNs-Konfidenz) — die definierende Evidenz (per-Segment-PANNs-Singing) ist im Plugin vorhanden (`get_tags`, Positions-Fenster), wird aber nicht je Segment genutzt | ✅ **UMGESETZT 2026-09-17**: `analyze_structure(vocal_scorer=…)` — optionaler per-Segment-Scorer ersetzt den Flatness-Proxy; UV3 baut den PANNs-Scorer GPU-gated (`_build_vocal_scorer_if_gpu`, CPU ⇒ Proxy unverändert, Budget §2.52b); Tests: test_ana2_vocal_scorer_overrides_flatness |
| ANA-3 | **Gender-Detektion** (phase_19 `_detect_gender_robust`): F0-/Formant-Heuristik ohne ML-Klassifikator; konsumiert von 19/43 (Sibilanten-Bänder) | ✅ **UMGESETZT 2026-09-17 (OHNE neue Inferenz)**: die Male/Female-Singing-Klassen (32/33) waren in den einmalig berechneten PANNs-Tags enthalten, aber nicht exponiert — jetzt als eigene Keys im Plugin-Resultat + ML-Prior in `_detect_gender_robust` (Abstand > 0,10, Score ≥ 0,25 ⇒ male/female; sonst unveränderte DSP-Heuristik) |
| ANA-4 | **Tonart-Erkennung dupliziert**: `phase_53._estimate_key` (Krumhansl, Dur/Moll, ganzer Song) UND `genre_classifier._estimate_key_dsp` (Pitch-Class-Argmax des LETZTEN Frames, nur Dur) — zwei Implementierungen (§V7 (copilot-instructions.md)) | ✅ **UMGESETZT 2026-09-17**: kanonisches Modul `backend/core/dsp/key_estimation.py` (Krumhansl, strukturiertes (root, mode)); phase_53 formatiert „C major“, genre_classifier „C-Dur“/„C-Moll“ (Qualitäts-Gewinn: ganzer Song + Dur/Moll statt Last-Frame-Argmax); 144 Tests grün |
| ANA-5 | **§2.52b-Beat-Tracking im Spec erwähnt, nicht implementiert** — BPM existiert aber in `musical_structure_analyzer._estimate_bpm` | ✅ **UMGESETZT 2026-09-17**: BPM-Metadatum (`bpm`) wird im §2.52b-Block der UV3-Pipeline aus der vorhandenen Energie-Onset-Autokorrelation berechnet und in `song_structure`-Metadata geführt — Spec-Lücke geschlossen, eine BPM-Quelle |
| ANA-6 | **Chunk-Modus**: beide Struktur-Analysen + SectionGoalAdapter liefen PRO CHUNK (Log: 4/2/3/2 Sektionen) — segment-adaptive Stärke war im Chunk-Modus grob | ✅ **UMGESETZT 2026-09-17**: `_restore_chunked` berechnet die §2.52b-Struktur EINMAL auf dem GESAMTEN Song und reicht sie als `_precomputed_song_structure` durch; `restore()` übernimmt sie und verschiebt/clippt sie chunk-lokal (`_shift_structure_to_chunk`, reine Funktion — PIM/VQI/Strength-Konsumenten bleiben zeitkorrekt); Test: test_ana6_shift_structure_to_chunk |
| ANA-7 | **Phonem-Sprache blind vertraut**: die sprach-spezifische Sibilanten-Bandwahl der De-Esser (de 5,5–8,5 kHz / es 4,5–7 kHz) folgte blind `transcription.language` — Produktionsbefund: deutscher Elke-Best-Song → Whisper „es“ bei conf 0,03–0,56 → SPANISCHE Band auf deutschem Gesang | ✅ **UMGESETZT 2026-09-17**: `resolve_language_consensus()` (phoneme_timeline) — conf ≥ 0,7 → Transkription; < 0,5 → „unknown“ (neutrale 4–8-kHz-Band); 0,5…0,7 → Konsens mit LPC-Formant-Detektor; Verdrahtung in unified_restorer_v3 (§2.36a-Block); 5 neue Tests |
| ANA-8 | **IPA-Formant-Ziele Stellvertreter**: `has_ipa` immer False (Decoder-ONNX fehlt) → `formant_target_for_range()` lieferte für JEDEN Vokal den generischen Schwa-Zentroid (500/1500 Hz) — ein Konstantwert statt Evidenz; phase_56-Formant-Boost zielte damit immer auf dieselben Frequenzen | ✅ **UMGESETZT 2026-09-17**: optionaler `audio`-Parameter — ohne IPA misst der Fallback die ECHTEN dominanten Spektral-Peaks des Vokal-Segments (60 Hz–4 kHz, ≥ 100 Hz Abstand, deterministisch FFT); phase_56 übergibt das Audio. Tests: TestAna8LpcFormantFallback (2 Fälle); 115 Tests grün. Offen bleibt nur die IPA-Feinstruktur (Decoder-ONNX, extern) |
| ANA-9 | **Niedrig-konfidente Phonem-Segmente treiben Gates ungefiltert**: Timeline-Segmente aus einer 3-%-Transkription steuerten trotzdem Segment-Gates in 19/24/43/56 (confidence wurde nur gespeichert, nie gegated) | ✅ **UMGESETZT 2026-09-17**: Konfidenz-Gate `_MIN_TIMELINE_CONF = 0,30` in `build_from_transcription` — niedrig-konfidente Wörter werden als „silence“ geführt (neutrale Gate-Reaktion, is_stressed=False); Tests: TestAna9ConfidenceGate (2 Fälle) |
| ANA-10 | **Zweite Phonem-Quelle dupliziert** (§V7 (copilot-instructions.md)): `lyrics_guided_enhancement.get_phoneme_mask` (DSP-Phonemkarte, hop 512) vs. PhonemeTimeline (Transkription) — zwei Phonem-Repräsentationen; zusätzlich `styles_zones` mit conf=0,00 im Lauf (Style-Führung inert) | ✅ **ROLLEN GEKLÄRT 2026-09-17 (dokumentiert, kein Code-Fix nötig):** PhonemeTimeline (Transkription) = Kanon für KLASSEN-Gates in 19/24/43/56/MDEM; LGE-Phonemkarte (DSP) = Kanon für Saliency-/Plosiv-Schutz in 09/58 (läuft auch ohne Transkription). Zwei Rollen, zwei Quellen — keine Duplikation im Sinne §V7. `styles_zones` conf=0 ⇒ Style-Führung bleibt inert (dokumentiert) |
| ANA-11 | **Phonem-GRENZ-Detektor tot** (Gegenstück-Muster): `detect_phoneme_boundaries_dsp` UND `detect_phoneme_protection_mask_dsp` (Plosiv/Frikativ-Schutzmaske, „soll NR reduzieren“) hatten **0 Produktions-Aufrufe** (Lauf-Befund: 0 Treffer) — nur die Feature-Extraktion lief als LGE-Fallback. Die Konsonanten-Schutzmaske gehört definitionsgemäß in die NR-Phasen | ✅ **UMGESETZT 2026-09-17**: phase_29 (Tape-Hiss-NR, vorher GANZ ohne Gate) blendet nach allen Guards das NR-Ergebnis in Konsonanten-Frames sanft Richtung Input zurück (max 0,35 × Stärke, 4-ms-Glättung, never-worsen durch Nichtstun; gemessen +6,9 % näher am Input in der Masken-Region). Test: `test_ana11_consonant_protection_keeps_region_closer_to_input` (140 Tests grün) |

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
