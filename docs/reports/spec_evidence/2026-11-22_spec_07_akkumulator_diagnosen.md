# Spec-Evidenz: Akkumulator-Schlüssel ohne Leser (Spec 07 §8.7 Punkt 5)

## Evidenzblock

- Spec-Datei: `.github/specs/07_quality_and_tests.md`
- Abschnitt: Spec 07 §8.7 Punkt 5 — „Offene Schuld (Schreibzugriffe ohne Leser)“ wird zu „geschlossen“
- Aenderungstyp: Sonstiges — report-only Metadaten-Vertrag; kein Schwellwert, keine Entscheidungslogik, keine Audio-Wirkung
- Alte Regel: Der Phasen-Akkumulator erreichte das Ergebnis nicht als Ganzes; 16 Schlüssel (u. a. `wohlklang_ordnung_violation`, `preflight_risk_guard`, `dgwcs`, `cumulative_budget`) wurden geschrieben und nie gelesen — Entscheidungen waren nicht nachvollziehbar (§G8 (copilot-instructions.md)).
- Neue Regel: Diese Schlüssel erscheinen report-only unter `metadata["phase_decision_diagnostics"]` (JSON-sicher, begrenzt). Ein neuer nur geschriebener Schlüssel ohne Eintrag in `_PHASE_DECISION_DIAGNOSTIC_KEYS` lässt das normative Gate fehlschlagen.
- Hypothese: Sichtbare Entscheidungs-Diagnosen sind Voraussetzung, um später zu prüfen, welche davon eine Entscheidung treiben sollen; ohne Sichtbarkeit bleibt es Blindflug.
- Referenzen/Standards: §G8, §V6 (copilot-instructions.md); Spec 07 §8.7; Spec 02 §2.2.3a.
- Datensaetze/Szenarien: Statische Literal-Suche über `unified_restorer_v3.py` (48 geschriebene Schlüssel, 16 ohne Lesezugriff); synthetische Akkumulator-Inhalte (NaN, große Arrays, numpy-Skalare) für die JSON-Sicherheit.
- Umfang (n): 3 neue Tests (Gate, Scanner-Selbsttest, JSON-Sicherheit); 16 Schlüssel.
- Primarmetrik: Wahrheitseigenschaften (bestanden/nicht bestanden).
- Effektstaerke: N/A — keine Effektmessung; keine Klangaussage.
- 95 %-CI: N/A — deterministische Vertragstests.
- Signifikanztest + p-Wert: N/A.
- Seed(s): N/A — kein Zufall.
- Commit: Basis `02393f68`; Änderungen im folgenden Commit.
- Skript/Befehl: `AURIK_FORCE_CPU=1 python3 -B -m pytest -q tests/normative/test_diagnostic_metadata_truth_gate.py tests/unit/test_song_context_map_wiring.py tests/unit/test_unified_restorer_v3.py`
- Artefaktpfade: `backend/core/unified_restorer_v3.py`; `tests/normative/test_diagnostic_metadata_truth_gate.py`
- Risiko fuer P1/P2: Keines für Audio: report-only, keine Entscheidung liest den neuen Schlüssel.
- Risiko fuer Artefakte: Ein zusätzlicher Ergebnis-Schlüssel (klein, begrenzt); externe Leser mit fester Schlüsselmenge sehen ihn.
- Bekannte Unsicherheiten: Die Suche ist literal (dynamische Schlüsselnamen unsichtbar); im Chunked-Pfad stammen die Werte vom ersten Chunk. Sichtbarkeit ist kein Wirkungsnachweis; ob ein Schlüssel Entscheidungen treiben soll, ist ungeklärt.
- Rollback-Kriterium: Rücknahme, falls der Schlüssel `restore()` verzögert, nicht JSON-serialisierbar ist oder eine Entscheidung beeinflusst.
- Entscheidung: CONDITIONAL — Sichtbarkeit hergestellt, Wirkungsentscheidung offen.
- Maintainer Sign-off: AUSSTEHEND — ein Maintainer-Sign-off wird nicht fingiert.
- Datum: 2026-11-22
