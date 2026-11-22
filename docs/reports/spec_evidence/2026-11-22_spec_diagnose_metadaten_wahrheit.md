# Spec-Evidenz: Diagnose-Metadaten-Wahrheit (Spec 07 §8.7, Spec 02 §2.2.3a)

## Evidenzblock

- Spec-Datei: `.github/specs/02_pipeline_architecture.md`, `.github/specs/07_quality_and_tests.md`
- Abschnitt: Spec 07 §8.7 (neu, [RELEASE_MUST]); Spec 02 §2.2.3a (neu, Verweis und Kurzregel)
- Aenderungstyp: Gate (Metadaten-Vertrag und maschinelle Durchsetzung); kein Schwellwert, kein Algorithmus, kein Audio-Eingriff
- Alte Regel: Es gab keine Norm für Diagnose-Metadaten. 28 `_compute_*_result`-Helfer in `UnifiedRestorerV3._collect_reporting_analytics` lasen `sys.modules.get(...)`, werteten das Ergebnis nie aus und meldeten unbedingt `{"active": True}`; weitere sechs Meldungen lieferten `active: True` nach reinem Typ-Import oder Cache-Präsenz; der `QualityControl`-Block meldete `active: True` trotz gescheiterter Konstruktion.
- Neue Regel: Metadaten über Aktivität/Verfügbarkeit/Wirkung müssen aus einer ausgewerteten Prüfung oder realen Ausführung stammen. Begriffstrennung `loaded_in_process` ≠ `wired_into_restoration_pipeline` ≠ `active`/`executed`; `active` nur nach erfolgreicher Instanziierung oder Ausführung; Präsenz-Helfer ohne Frischimport (Deadlock-Schutz im `QThread`-Batch-Pfad); Diagnose-Metadaten sind report-only und kein Qualitäts-/Hörbeleg; Verdrahtungslücken sind offene Schuld.
- Hypothese: Eine maschinell durchgesetzte Begriffstrennung verhindert, dass Telemetrie (z. B. „perceptual_loss active“) als Wirkungs- oder Wohlklangnachweis gelesen wird, ohne das Audio-Verhalten zu verändern.
- Referenzen/Standards: `.github/instructions/hoerordnung.instructions.md` §7 (Metriken sind Zeugen); §V6, §G8, §G11 (copilot-instructions.md); `AGENTS.md` §3 (Silent-Failure-Verbot, Anti-Parallelwelten); Spec 07 §8.7.
- Datensaetze/Szenarien: Keine Audiodaten. Statische Quellcodeanalyse (AST über 1752 Python-Dateien in `backend`, `denker`, `plugins`, `Aurik10`, `cli`, `scripts`) und Aufrufstellen-Nachsuche je betroffenem Modul (grep über Produktionscode ohne Tests).
- Umfang (n): 28 Diagnose-Helfer mit totem oder strukturell leerem Zeugen (26 + 2), 6 Präsenz-Meldungen nach Typ-Import/Cache, 1 Falschmeldung im `QualityControl`-Block, 1 Enum-Auflistung ohne `active`; repo-weit 9 `sys.modules.get`-Zuweisungen, nach der Korrektur 0 tote Zeugen.
- Primarmetrik: Mit den Scannern des neuen Gates gemessen: tote `sys.modules`-Zeugen in `unified_restorer_v3.py` auf dem eingecheckten Stand 26, nachher 0 (repo-weit 0); `"active": True`-Ansprüche in `_compute_*` ohne Instanziierung vorher 36 (34 Helfer), nachher 0. Zusätzlich 2 strukturell leere Duck-Type-Suchen (`module_parallel`, `stereo_parallel`) ohne Scanner-Treffer, aber mit unbedingtem `active`.
- Effektstaerke: N/A — deterministische Zählung, keine Stichprobe; es wird kein Effekt auf Klangqualität behauptet.
- 95 %-CI: N/A — vollständige statische Analyse des Quellcodes, keine Stichprobe; kein Konfidenzintervall ableitbar oder behauptet.
- Signifikanztest + p-Wert: N/A — keine statistische Hypothese; die Regel wird per AST-Gate mit Selbsttest geprüft.
- Seed(s): N/A — keine stochastische Komponente; das Gate ist deterministisch.
- Commit: Basis `2770cb49`; die Änderungen sind Teil des noch offenen Worktrees.
- Skript/Befehl: `python3 -B -m pytest -q tests/unit/test_unified_restorer_v3.py tests/unit/test_v99_hybrid_modules.py tests/normative/test_diagnostic_metadata_truth_gate.py tests/unit/test_spec_evidence_gate.py`
- Artefaktpfade: `tests/normative/test_diagnostic_metadata_truth_gate.py`; `tests/unit/test_unified_restorer_v3.py::TestComputeModulePresenceResult`; `backend/core/unified_restorer_v3.py` (`_compute_module_presence_result` und umgestellte Helfer)
- Risiko fuer P1/P2: Keines für Audio: Die Änderung betrifft ausschließlich Reporting-Metadaten (report-only). Konsumenten der geänderten Schlüssel wurden gesucht (Tests, `Aurik10`, `cli`, `backend/api`, `denker`): keine gefunden.
- Risiko fuer Artefakte: Keines (kein Signaleingriff). Metadaten-Schlüssel ändern ihren Inhalt: `active` entfällt zugunsten von `loaded_in_process`/`wired_into_restoration_pipeline`; externe Leser, die `active` erwarten, würden `None`/fehlende Schlüssel sehen. Präsenz-Helfer liefern `None`, wenn ein Modul im Prozess nicht geladen ist (zuvor teils frischer Import).
- Bekannte Unsicherheiten: `wired_into_restoration_pipeline` ist eine per Quellcode-Nachsuche belegte Codebase-Tatsache, keine Laufzeitmessung; dynamische Importe über Strings können eine Aufrufstelle verbergen. Instanziierende Diagnose-Helfer (`hybrid_*`, `audit_log`, `regulator`, `quality_gate` u. a.) importieren weiterhin lokal im Analytik-Pfad — als offene Schuld in Spec 07 §8.7 Regel 3 ausgewiesen. Von 32 nicht verdrahteten Modulen sind 31 nicht im FILE_REGISTRY geführt (Spec 07 §8.7 Regel 5, Entscheidung je Modul offen). Keine Aussage über Wohlklang oder Hörbarkeit.
- Rollback-Kriterium: Rücknahme nur, wenn das Gate eine belegt korrekte, ausgewertete Prüfung fälschlich als toten Zeugen einstuft; eine Rückkehr zu unausgewerteten `active`-Meldungen ist nicht zulässig.
- Entscheidung: CONDITIONAL — Gate und Norm implementiert und getestet; keine Aussage zu Klangqualität.
- Maintainer Sign-off: AUSSTEHEND — ein Maintainer-Sign-off wird nicht fingiert.
- Datum: 2026-11-22

## Validierung

- Regressionslauf nach den letzten Code-Änderungen: 378 bestanden, 4 durch die Standardkonfiguration abgewählt (Heavy), 0 fehlgeschlagen. Darin: neues Normativ-Gate (4 Tests inkl. 2 Selbsttests), `TestComputeModulePresenceResult`, die komplette `test_unified_restorer_v3.py`, `test_v99_hybrid_modules.py`, `test_spec_evidence_gate.py`.
- Konsistenz-Gates in einem Lauf: grün — `id_registry_check`, `release_must_coverage_check`, `compliance_check --errors-only`, `gebote_verifier`, `spec_integration_scanner --fail-on error`, `repo_graph --duplicates`, `defect_coverage_check`, `horordnung_calibration`, `sota_deficit_gate`, `file_registry_check` (707 Dateien), `drt_contract_gate`, `test_canonical_contract_drift_gate`. Rot: `aurik_verboten_linter --ci --errors-only` mit zwei V01-Befunden (`print()`) in `rx_autopilot/http_share.py` und `rx_autopilot/sync.py` — vorbestehend (seit Commit `b9c17bfc`), nicht Teil dieser Änderung und hier nicht behoben.
- Ruff-Lint und Format der geänderten Python-Dateien: bestanden.
- Dies ist ein Prozess- und Wahrheitsnachweis für Metadaten, keine Hörabnahme und kein Qualitätsnachweis.
