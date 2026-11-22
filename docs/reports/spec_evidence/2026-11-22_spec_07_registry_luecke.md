# Spec-Evidenz: Registry-Lücke nicht verdrahteter Module (Spec 07 §8.7 Punkt 5)

## Evidenzblock

- Spec-Datei: `.github/specs/07_quality_and_tests.md`
- Abschnitt: Spec 07 §8.7 Punkt 5 — „Offene Schuld“ der Registry-Lücke wird zu „geschlossen, Entscheidung offen“
- Aenderungstyp: Sonstiges — Dokumentations-/Registry-Wahrheit; kein Schwellwert, kein Code mit Audio-Wirkung
- Alte Regel: 31 der 32 nicht in die Restaurierung verdrahteten Module standen nicht im FILE_REGISTRY (nur `regulator_v8.py`).
- Neue Regel: Alle sind geführt, ohne Löschung: `TEST_ONLY` bei null Produktions-Importeuren, sonst `ACTIVE` mit dem Vermerk „nicht in die Pipeline verdrahtet“. Die Integrationsentscheidung (Hörnachweis nach §4 oder Archivierung) bleibt ausdrücklich offen.
- Hypothese: Ehrliche Registrierung verhindert, dass nicht verdrahtete Module still als Pipeline-Bestandteil gelten (Anti-Parallelwelten-Prinzip).
- Referenzen/Standards: AGENTS.md §3 (Write-Gate/FILE_REGISTRY), Spec 07 §8.7, `scripts/file_registry_check.py`. Vorgabe des Maintainers (Nutzerentscheidung 2026-11-22): „Nur ehrlich registrieren … nichts löschen“.
- Datensaetze/Szenarien: Importer-Scan per Regex über Produktions- und Testverzeichnisse für 29 Module.
- Umfang (n): 29 neue Registry-Zeilen (13 TEST_ONLY, 16 ACTIVE).
- Primarmetrik: `scripts/file_registry_check.py`: 0 ERROR, 0 WARNUNG, 742 Dateien registriert.
- Effektstaerke: N/A — keine Effektmessung.
- 95 %-CI: N/A — deterministische Registry-Prüfung.
- Signifikanztest + p-Wert: N/A.
- Seed(s): N/A — kein Zufall.
- Commit: Basis `ce5e3af5`; Änderungen im folgenden Commit.
- Skript/Befehl: `python3 scripts/file_registry_check.py`
- Artefaktpfade: `.github/FILE_REGISTRY.md`; `.github/specs/07_quality_and_tests.md`
- Risiko fuer P1/P2: Keines für Audio. Kein Code geändert.
- Risiko fuer Artefakte: Keines.
- Bekannte Unsicherheiten: Regex-Importer-Scan ist statisch (dynamische Importe unsichtbar), nicht transitiv und zählt `__init__`-Re-Exports als Produktionsimport; „TEST_ONLY“ bedeutet hier „keine statisch erkennbaren Produktions-Importeure“, nicht „von Tests benutzt“ (einige haben 0 Test-Importeure).
- Rollback-Kriterium: Rücknahme, falls ein Modul mit dynamischem Produktionsimport fälschlich als TEST_ONLY geführt wird.
- Entscheidung: CONDITIONAL — Registry geschlossen, Integrationsentscheidung offen.
- Maintainer Sign-off: AUSSTEHEND — ein Maintainer-Sign-off wird nicht fingiert.
- Datum: 2026-11-22
