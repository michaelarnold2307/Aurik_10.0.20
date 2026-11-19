# Canonical Contract Drift Gate — Evidenzbericht

## Evidenzblock

- Spec-Datei: .github/specs/08_architecture_and_distribution.md; .github/specs/07_quality_and_tests.md
- Abschnitt: §11.1b Canonical Contract Drift Gate; §8.3.2b Canonical Contract Drift Gate — Testpflicht
- Aenderungstyp: Gate
- Alte Regel: GUI/CLI/Batch-Pfade waren einzeln spezifiziert, aber Cross-Path-Drift war nicht als eigener Release-Blocker kodifiziert.
- Neue Regel: Release-faehige Oberflaechen muessen denselben Bridge-/Denker-/Exporter-Vertrag nutzen; Legacy-Serverpfade muessen als LEGACY_NON_RELEASE markiert sein oder migriert werden.

### 1. Wissenschaftliche Begruendung

- Fachliche Hypothese: Qualitaets-, Modus- und Exportdrift entsteht nicht nur in DSP-Phasen, sondern auch durch abweichende Einstiegspfade. Ein normatives Contract-Gate verhindert unterschiedliche Ergebnisse zwischen GUI, CLI und Batch.
- Referenzen (Paper/Standard): Aurik §0 Autonomes Entscheidungs-Doktrin; Aurik Spec 08 §11 Kanonischer Verarbeitungseinstieg; Aurik Spec 07 §8.3 Regressionstestpflicht.
- Warum ist die Aenderung kausal plausibel? Gleicher Import, gleiche Voranalyse, gleicher Denker-Einstieg und gleicher Exportvertrag reduzieren nicht-deterministische Pfadunterschiede und verhindern Quality-Gate-Bypaesse.

### 2. Datengrundlage

- Datensaetze/Szenarien: Statische Release-Pfad-Pruefung fuer CLI, Frontend und REST-Legacy-Markierungen; bestehende CLI-Paritaets- und Magic-Button-Normativtests.
- Umfang (n): 5 neue normative Contract-Drift-Assertions plus angrenzende CLI/Magic-Button/§0a-Gates.
- Material- und Modusabdeckung: Pfadvertrag ist materialunabhaengig; Modusoberflaeche beschraenkt auf Restoration und Studio 2026.
- Ausschlusskriterien: Historische REST-Serverpfade sind ausgeschlossen, wenn sie explizit LEGACY_NON_RELEASE markieren und nicht als Desktop-Releasepfad beworben werden.

### 3. Statistik

- Primarmetrik: Normativer Gate-Pass/Fail.
- Effektstaerke: n/a — statischer Architektur-Gate, keine Audio-Metrik.
- 95 %-CI: n/a — deterministischer statischer Gate.
- Signifikanztest + p-Wert: n/a.
- Multiple-Testing-Korrektur: n/a.

### 4. Reproduzierbarkeit

- Seed(s): n/a.
- Commit: v10.0.0-hotfix.2 Release-Kandidat.
- Skript/Befehl: `.venv_aurik/bin/python -m pytest tests/normative/test_canonical_contract_drift_gate.py -p no:xdist --override-ini="addopts=--strict-markers --import-mode=importlib" --timeout=30 --tb=short -q --disable-warnings --no-header`
- Artefaktpfade: tests/normative/test_canonical_contract_drift_gate.py; docs/reports/spec_evidence/2026-05-19_canonical_contract_drift_gate.md.

### 5. Risikoanalyse

- Risiko fuer P1/P2: Niedrig; keine DSP- oder Audio-Metrik-Schwelle wurde geaendert.
- Risiko fuer Artefakte: Niedrig; Gate blockiert Export-/Quality-Bypaesse, erzeugt aber keine Audioverarbeitung.
- Bekannte Unsicherheiten: REST-Legacy-Pfade bleiben historisch vorhanden und sind nicht Teil des Desktop-Releasepfads.
- Rollback-Kriterium: Falschpositive im statischen Gate, die einen korrekten Bridge-konformen Release-Pfad blockieren.

### 6. Entscheidung

- Entscheidung: APPROVED
- Maintainer Sign-off: Michael Arnold / Aurik Solo Maintainer
- Externer Reviewer (optional): n/a
- Datum: 2026-05-19
