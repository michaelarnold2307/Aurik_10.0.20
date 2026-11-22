# Spec-Evidenz: SongContextMap (Spec 03 §SCM)

## Evidenzblock

- Spec-Datei: `.github/specs/03_cognitive_modules.md`, `.github/specs/07_quality_and_tests.md`
- Abschnitt: Spec 03 §SCM (neu, [RELEASE_MUST]); Spec 07 §8.7 (Ergänzung: offene Schulden Akkumulator-Schreibzugriffe und Registry-Lücke)
- Aenderungstyp: Sonstiges — neuer report-only Metadaten-Vertrag; kein Schwellwert, kein Algorithmus mit Audio-Wirkung, keine Entscheidungslogik
- Alte Regel: Musikverständnis lag verstreut in `_restoration_context` und im Phasen-Akkumulator (Ära/Genre, VocalFocusAnalyzer, HarmonicContextAnalyzer, SongStructureAnalyzer, PANNs-Tags). Es gab keine konsolidierte Sicht mit Konfidenz, Herkunft und sichtbaren Lücken; der Akkumulator-Schlüssel `song_structure` wurde nur geschrieben und erreichte das Ergebnis nie.
- Neue Regel: Pro `restore()` entsteht eine Kontextkarte (`metadata["song_context_map"]`, Schema `aurik.song_context_map/1`) über sieben Dimensionen. Sie misst nichts neu; Konfidenz stammt nur aus analysatoreigenen Feldern (sonst `None`), Fremdmaße werden nicht umetikettiert, fehlende Analysen heißen `unavailable`, der Scope (`song`/`chunk`) wird ausgewiesen, die Karte ist report-only und wird pro Song zurückgesetzt.
- Hypothese: Eine konsolidierte, ehrlich unsichere Sicht ist die Voraussetzung dafür, später hörkalibrierte Entscheidungen (Präferenz-Zeuge) auf belegte Eingaben zu stützen, ohne vorher klangwirksame Entscheidungen zu verändern.
- Referenzen/Standards: `.github/instructions/hoerordnung.instructions.md` §7 (Metriken sind Zeugen); §V6, §V7, §V8, §G5, §G8, §G10 (copilot-instructions.md); Spec 07 §8.7 (Metadaten-Wahrheit).
- Datensaetze/Szenarien: (1) Synthetische Test-Eingaben aus den echten Produzenten `VFAResult.to_dict()`, `HarmonicContextResult.to_dict()` und `SongSegment`. (2) `restore()`-Läufe im bestehenden Mock-Rahmen (Ganz-Song, Chunk, Song-Isolation). (3) Ein Realitätscheck mit den echten Analysatoren (Harmonik, Struktur, Tempo, VocalFocus) auf einem lokalen 20-s-Ausschnitt eines Schlager-Titels aus dem privaten Musikarchiv; das Audio wurde ausschließlich lokal verarbeitet und nicht abgelegt.
- Umfang (n): 20 Unit-Tests, 4 Verdrahtungstests; 1 Realitätscheck (n=1 Ausschnitt, nur Schema-/Wertebereichsprüfung, keine Qualitätsaussage).
- Primarmetrik: Wahrheitseigenschaften (bestanden/nicht bestanden): Konfidenz nur analysatoreigen, Lücken sichtbar, Determinismus, JSON-/NaN-Sicherheit, Eingaben unverändert, Song-Isolation, Ergebnis-Metadatum vorhanden. Im Realitätscheck: 5 von 7 Dimensionen gemessen (`provenance` und `instruments` ohne Pre-Analysis/PANNs `unavailable`), Analysedauer 1,2 s.
- Effektstaerke: N/A — Wahrheits-/Vertragsprüfung, keine Effektmessung; es wird kein Effekt auf Klangqualität behauptet.
- 95 %-CI: N/A — deterministische Vertragstests, keine Stichprobe; der Realitätscheck (n=1) ist kein statistischer Beleg.
- Signifikanztest + p-Wert: N/A — keine statistische Hypothese.
- Seed(s): N/A — der Builder ist deterministisch (keine Zufalls- oder Zeitabhängigkeit); die Tests verwenden keinen Zufall.
- Commit: Basis `2770cb49`; die Änderungen sind Teil des noch offenen Worktrees.
- Skript/Befehl: `AURIK_FORCE_CPU=1 python3 -B -m pytest -q tests/unit/test_song_context_map.py tests/unit/test_song_context_map_wiring.py tests/unit/test_unified_restorer_v3.py tests/normative/test_diagnostic_metadata_truth_gate.py tests/unit/test_v99_hybrid_modules.py tests/unit/test_spec_evidence_gate.py`
- Artefaktpfade: `backend/core/song_context_map.py`; `backend/core/unified_restorer_v3.py` (Aufbau nach dem SongStructureAnalyzer-Block, Ergebnis-Metadatum); `tests/unit/test_song_context_map.py`; `tests/unit/test_song_context_map_wiring.py`
- Risiko fuer P1/P2: Keines für Audio: report-only, keine Entscheidungslogik liest die Karte, kein Sample wird verändert. Ein Aufbaufehler wird mit `logger.warning` gemeldet und blockiert `restore()` nicht (`song_context_map` bleibt `None`).
- Risiko fuer Artefakte: Keines (kein Signaleingriff). Zusätzlicher Ergebnis-Schlüssel `song_context_map` (JSON-sicher, klein; die Segmentliste hat je Segment fünf Felder). Externe Leser, die eine feste Schlüsselmenge erwarten, sehen einen zusätzlichen Schlüssel.
- Bekannte Unsicherheiten: Analysatoreigene Konfidenzen sind nicht hörkalibriert (Realitätscheck: Harmonik-Konfidenz 0,997 für 20 s ohne Modulation). Das Tempo kann Oktavfehler enthalten (Realitätscheck: 200,9 BPM für einen Schlager-Ausschnitt, auffällig und unverifiziert). `instruments` sind AudioSet-Tags, keine Instrumentenerkennung; `dynamics_arc` ist gesangsbezogen. Im Chunked-Pfad stammt die Karte des Ergebnisses aus dem ersten Chunk (`scope=chunk`); eine Ganz-Song-Karte ist offene Schuld. Es wurde kein vollständiger `restore()`-Lauf mit echten Modellen auf einem ganzen Titel gemessen. Keine Aussage über Wohlklang oder Hörbarkeit.
- Rollback-Kriterium: Rücknahme, falls der Aufbau `restore()` verzögert oder blockiert, oder falls die Karte fälschlich Konfidenz oder Messung ausweist, wo ein Analysator keine geliefert hat; ein Übergang in Entscheidungslogik ohne §4-Hörnachweis ist nicht zulässig.
- Entscheidung: CONDITIONAL — Vertrag implementiert und getestet; keine Aussage zu Klangqualität, keine Aktivierung in Entscheidungen.
- Maintainer Sign-off: AUSSTEHEND — ein Maintainer-Sign-off wird nicht fingiert.
- Datum: 2026-11-22

## Validierung

- Gesamtlauf aller betroffenen Tests: 402 bestanden, 4 durch die Standardkonfiguration abgewählt (Heavy), 0 fehlgeschlagen. Darin: 20 Unit-Tests der Karte, 4 Verdrahtungstests, die komplette `test_unified_restorer_v3.py`, `test_v99_hybrid_modules.py`, das Metadaten-Wahrheits-Gate und `test_spec_evidence_gate.py`.
- Hostabhängigkeit gefunden und behoben: In einem früheren Gesamtlauf scheiterte ein Verdrahtungstest, weil `restore()` den Start bei Swap > 95 % verweigerte (§MEM-GATE; ein paralleles Training füllte den Swap). Die Tests stubben jetzt Swap-Messung und Thrashing-Erkennung deterministisch und bestehen auch bei real fast vollem Swap. Die Test-Anleitung (`.github/instructions/tests.instructions.md`) nannte zudem einen nicht existierenden Patch-Pfad und ist korrigiert.
- Realitätscheck mit echten Analysatoren: Schlüsselnamen und Wertebereiche passen zu den Produzenten; die Ausgabe ist NaN-frei JSON-serialisierbar. Die Stimm-Dimension wurde ohne PANNs-Eingabe (`panns_singing=0.0`) erzeugt und ist daher nicht musikalisch zu deuten.
- Ruff-Lint und Format der neuen und geänderten Python-Dateien: bestanden.
- Dies ist ein Prozess- und Wahrheitsnachweis für Metadaten, keine Hörabnahme und kein Qualitätsnachweis.
