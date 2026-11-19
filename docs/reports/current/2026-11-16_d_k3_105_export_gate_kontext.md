# D-K3-105 — Export-Gate-Kontext: GUI/Batch exportierten ein Gate ohne Profil

**Datum:** 2026-11-16 · **Klasse:** K3 (Kette/Verdrahtung, keine Klangwirkung)
**Status:** geschlossen · **Flag-Flip:** keiner · **GPU-Training:** keines

## 1. Ausgangslage

Der Auftrag nannte sieben Kandidaten in fester Reihenfolge. Gemessen am
Register sind **alle sieben bereits `geschlossen`** (D-K3-13, D-K3-60,
D-K3-71, D-K3-75, D-K3-83, D-K3-90, D-K3-92 — Beleg:
`.github/SOTA_DEFICIT_REGISTER.md`, `scripts/sota_deficit_gate.py`:
127 Einträge, 0 Verstöße). Offen sind nur hör-, GPU- oder modellgebundene
Zeilen (D-K3-65 Hörstichprobe, D-K3-87/94/95/96 GPU/Modell,
D-K3-97 Pegel-/Hörkette, D-K4-* Daten/Mensch).

Die Prüfung des **Belegs** von D-K3-60 (Roadmap `SOTA-M22`: „geschlossen,
Ersatz Beleg prüfen“) hat eine echte, autonom schließbare Lücke freigelegt:
der Export-Payload trug in den GUI-/Batch-Pfaden ein Gate **ohne Profil,
ohne Material und ohne Schwellen** — und der Leerwert war selbst ein
Fehlsignal.

## 2. Befund (gemessen, vor dem Fix)

`build_export_quality_gate_payload` (kanonisch: `backend/api/bridge.py`)
las `export_gate_profile`, `export_gate_material`, `export_gate_thresholds`,
`export_gate_preserve_signal` **ausschließlich aus den Metadaten**. Diese
Metadaten schreibt aber nur `cli/aurik_cli.py` (`export_gate_profile`-Writer:
genau eine Stelle im Repo). GUI (`Aurik10/ui/modern_window.py:3764`), Batch
(`batch_processor.py:290`) und der Denker rufen den Builder ohne diesen
Kontext auf.

Gemessen an einem deterministischen Ergebnis-Objekt (`material_type=vinyl`,
`stage_notes.exzellenz_recovery_profile.preserve_signal=0.12`,
`degradation_status=ok`, WCS-Gate `wcs=0.912 / profile=vocal`):

| Feld | Experience-Payload (`get_experience_insights`) | Export-Payload (`build_export_quality_gate_payload`) |
| --- | --- | --- |
| `profile` | `modern_stable` | `""` |
| `material` | — | `""` |
| `thresholds` | — | `{quality_estimate: 0.0, level_drop_db: 0.0}` |
| `preserve_signal` | 0.12 | 0.0 |

Zwei Folgen, beide belegt:

1. **Zwei Wahrheiten für einen Lauf (§G9):** die GUI zeigte das abgeleitete
   Profil im Banner (`xp_quality_gate["profile"]`), schrieb aber
   `quality_gate_profile=""` in die Export-Metadaten
   (`modern_window.py:3972`). Ein „leeres Profil“ war damit **kein** Beweis
   für „kein Gate im Meta“ — genau diese Lesart hatte D-K3-60 benutzt.
2. **Fehlklassifikation (§V7-Klasse):** `failure_class` las
   `_qe_threshold <= 1e-4` als Signal für einen technischen Fehler. Fehlt die
   Schwelle strukturell (GUI/Batch), wurde **jede** Degradierung mit
   Begründung zu `technical_failure`; gemessen: `degradation_status=degraded`
   + `fail_reason=level_drop` ⇒ `technical_failure`, obwohl ein
   Qualitätsbefund vorlag.

## 3. Ursache

Der Gate-Kontext (Profil, Material, Signal-Erhalt, Schwellen) existierte nur
im CLI-Pfad. Der Bridge-Builder hatte keine Ableitung; die Regel
(`preserve_signal` ⇒ `fragile_or_transient_risk` / `modern_stable` /
`neutral`) stand **inline nur** im Experience-Payload. Damit war die
CLI/GUI-Funktionsgleichheit (AGENTS.md §3) für ein Export-Feld verletzt, und
ein fehlender Wert war von einem gemessenen Nullwert nicht unterscheidbar
(§G8).

## 4. Fix (Ursache, nicht Symptom)

- **Eine Ableitung:** `_export_gate_kontext()` in `backend/api/bridge.py`
  liefert Profil, Material, Signal-Erhalt **und deren Herkunft**. Reihenfolge:
  CLI-Kontext aus den Metadaten gewinnt (er trägt die Material-/Signatur-
  Risikorechnung), sonst gilt `_derive_export_gate_profile()` — exakt die
  bisherige Experience-Regel, jetzt an einer Stelle.
- **Beide Payloads nutzen sie:** `get_experience_insights` und
  `build_export_quality_gate_payload` (plus die Kopie
  `backend/api/bridge_export.py`, die an dieselbe Definition delegiert).
- **Herkunft statt stiller Nullen:** neue Felder `profile_source`
  (`export_gate_meta` | `derived_preserve_signal`), `material_source`
  (`export_gate_meta` | `result_material_type` | `absent`),
  `thresholds_source` (`export_gate_meta` | `absent`). Schwellen werden
  **nie erfunden** — fehlen sie, steht das da.
- **Fehlklassifikation behoben:** die `_qe_threshold <= 1e-4`-Heuristik gilt
  nur noch bei `thresholds_source == "export_gate_meta"`. Pipeline-Fehler
  (`pipeline_blocked:`, „pipeline-fehler“) bleiben `technical_failure`.
- **Export-Metadaten:** GUI (`modern_window.py`), Batch (`batch_processor.py`)
  und CLI (`aurik_cli.py`) führen `quality_gate_profile_source` und
  `quality_gate_thresholds_source` mit — die Herkunft ist damit im
  `export_metrics.json` lesbar.

## 5. Messung (nach dem Fix, gleiche Eingabe)

| Feld | GUI/Batch-Form (ohne CLI-Meta) | CLI-Form (mit Gate-Meta) |
| --- | --- | --- |
| `profile` | `modern_stable` | `fragile_or_transient_risk(risk=0.54)` (unverändert) |
| `profile_source` | `derived_preserve_signal` | `export_gate_meta` |
| `material` | `vinyl` (`result_material_type`) | `vinyl` (`export_gate_meta`) |
| `thresholds` | `{0.0, 0.0}` + `thresholds_source=absent` | `{0.42, 4.3}` + `export_gate_meta` |
| `preserve_signal` | `0.12` | `0.12` |
| `failure_class` (degraded + Grund) | `quality_failure` | `quality_failure` |

Der CLI-Pfad bleibt **bit-identisch** (Metadaten-Profil mit Risiko-Suffix,
Schwellen, `failure_class`), die WCS-Gate-Werte (`wcs`, `threshold`,
`profile`, `passed`) werden unverändert durchgereicht. Die
Experience-/Export-Parität ist jetzt Test-gepinnt.

## 6. Nachweise

- Neue Vertragstests: `tests/unit/test_dk3_105_export_gate_kontext.py`
  — **13 passed** (Ableitung, Parität Experience↔Export, CLI-Vorrang,
  `absent`-Herkunft, Klassifikationskorrektur, Builder-Parität
  bridge↔bridge_export, §G5-Determinismus, Quelltext-Pin gegen Regel-Drift).
- Bestands-Suiten: `test_experience_propagation.py`,
  `test_dk360_rest_metadata_degradation.py`, `test_weltklasse_kette.py`,
  `test_batch_processor_worldclass_contract.py` → **39 passed**;
  Bridge-/Paritäts-Suiten (`test_bridge*.py`, `test_253_experience_propagation`,
  `test_r2_muq_mos_export_wiring`, `test_cli_gui_output_parity`,
  `test_canonical_contract_drift_gate`) → **328 passed**.
- Ruff: `check` clean, `format` clean.
- Register-Gate: `scripts/sota_deficit_gate.py` → **133 Einträge, keine
  Verstöße**; `tests/normative/test_sota_deficit_gate.py` → **28 passed**
  (inkl. neuem Regressionstest).
- Bestehender CLI-Export als Gegenprobe:
  `output_audio/Trio Schweizer - 13 Tage (Restored 10.12.25).wav.export_metrics.json`
  trägt `quality_gate_profile="fragile_or_transient_risk(risk=0.54)"` — der
  CLI-Pfad schreibt das Profil, der GUI-/Batch-Pfad konnte es strukturell
  nicht.

## 8. Nebenbefunde (im selben Zug geschlossen)

Beim Einpflegen der Register-Zeile fiel eine **Gate-Blindstelle** auf:
`scripts/sota_deficit_gate.py` parste IDs mit `\d{1,2}`. Jede Zeile ab
D-…-100 war damit für das fail-closed Gate unsichtbar — sechs Zeilen
(D-K3-100…D-K3-105) wurden stillschweigend übersprungen, und das Gate meldete
„127 Einträge, keine Verstöße". Die Zeile D-K3-105 wäre also **dekorativ**
geblieben. Fix: `\d{1,3}`; Regressionstest
`test_dreistellige_ids_sind_fuer_das_gate_sichtbar` pinnt dreistellige IDs
**und** „keine Tabellenzeile unsichtbar" (Vergleich Zeilenzahl ↔ geparste
Einträge).

Nach dem Weiten kam ein echter Formatfehler zum Vorschein: D-K3-103 trug die
Manifest-Evidenz in der Domänen-Spalte (Zeilensprung). Spalte auf `—` gesetzt,
Evidenz in die Evidenz-Spalte übernommen — Inhalt unverändert, keine
Bewertung geändert.

## 9. Grenzen (ehrlich)

- Es gibt **kein** GUI-Export-Artefakt im Repo, an dem der Leerwert
  end-to-end nachweisbar wäre; der Beleg ist Quelltext-Wirkung plus
  deterministische Builder-Messung. Ein künftiger GUI-Lauf zeigt die
  Herkunft in `processing_metadata`.
- Die Schwellen bleiben im GUI-Pfad numerisch 0.0, weil dort keine
  material-/signaturabhängige Schwellenrechnung existiert. Sie werden jetzt
  als `absent` ausgewiesen statt als gemessene Null gelesen. Eine echte
  Schwellen-Setzung im GUI-Pfad wäre eine Export-Entscheidung (Status
  `degraded`) und damit nicht Teil dieses Fixes.
- Die Profil-*Regel* bleibt zwischen den Pfaden unterschiedlich reich: der CLI-Pfad
  rechnet das Profil material-/signaturbasiert (Risiko-Suffix), der GUI-Pfad hat
  diese Signatur nicht und nutzt die abgeleitete Regel. Das ist jetzt **benannt**
  (`profile_source`), nicht mehr still. Eine Angleichung hieße, die
  Signaturberechnung in den GUI-Pfad zu ziehen — eigener Schritt, nicht Teil dieses
  Fixes.
- Keine Klangwirkung, kein Flag-Flip, kein GPU-Training, kein Modellwechsel.
