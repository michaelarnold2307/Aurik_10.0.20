# WP-7 Label-Korpus — Datenaufbau-Stand (2026-10-09)

**Auftrag:** `docs/TODOS_SOTA_ROADMAP.md` WP-7 · Register D-K1-2 · Akzeptanz: Label-Korpus + Kalibrierreport.
**Zielvorgabe Datenaufbau:** ≥ 50 Provenienz-Labels je Material-Klasse (Roadmap WP-7: „echte gelabelte Träger").

## Ist-Stand (maschinell, `scripts/generate_corpus_from_public_domain.py --check`)

| Klasse | Ist (Labels) | Ziel | Lücke |
| --- | ---: | ---: | ---: |
| shellac | 23 | 50 | 27 |
| vinyl | 21 | 50 | 29 |
| tape | 20 | 50 | 30 |
| cassette | 9 | 50 | 41 |
| reel_tape | 9 | 50 | 41 |
| digital | 10 | 50 | 40 |
| **Gesamt** | **92** | 300 | **208** |

- `meets_minimum` (20 Dateien, 4 Klassen) ist erfüllt — das ist das alte Bestands-Mindestmaß, **nicht** das WP-7-Ziel (`wp7_met: false`).
- Validierung: `tests/corpus/` **42 Tests grün** (Struktur, Manifeste, Schema, Pipeline-Smoke).
- Neu (2026-10-09, additiv): Der Coverage-Check trägt `wp7_target_per_material`, `wp7_gap` und `wp7_met` — die Ziel-Lücke ist damit maschinenlesbar und CI-tauglich.
- Ausschnitt-Scheibe: `quality_gate_material` des Referenzlaufs 10.12.25 erkannte „vinyl" — die Klassifikation läuft, die Kalibrierung wartet auf die Trägerbasis (D-K1-2).

## Beschaffungswege

1. **Public Domain:** `scripts/generate_corpus_from_public_domain.py` (Internet Archive 78rpm, PD/CC0; verifizierte Einträge mit Lizenz + Jahr + Checksumme). Quellenpool derzeit: **shellac 12, vinyl 0, tape 0** — der Pool deckelt die automatisierbare Menge; Erweiterung braucht kuratierte PD-URLs (`--from-urls`, je Eintrag Lizenz + Jahr ≤ 1926 belegen).
2. **Echte Träger (Nutzer-Import):** wichtigster Pfad für vinyl/tape/cassette/reel_tape — Aufnahmen aus eigener Sammlung mit Provenienz, Manifest nach `corpus/MANIFEST_SCHEMA.yaml` (Pflichtfelder file, duration_s, sample_rate, material, era_year, genre).
3. **F13-Teil (`train_gender_head.py`):** verlangt `data/music` + `data/gender_labels.csv` — **beide fehlen** (vorhandenes Musikmaterial liegt unter `data/musdb18hq`); das Skript bricht ohne Labels bewusst ab (§V7 copilot-instructions.md — kein Blind-Training). Kalibrierreport erst nach Label-Satz.

## Abnahme-Checkliste (WP-7)

- [ ] ≥ 50 Labels je Klasse (`wp7_met: true`)
- [ ] Material-Kalibrierreport — `train_clap_material_classifier.py` auf ECHTEN Trägern statt synthetischer DSP-Degradation (D-K1-2)
- [ ] F13: Label-Satz + Gender-Head-Kalibrierreport

_Stand 2026-10-09 · Beleg: `--check`-JSON (dieser Commit) · `tests/corpus/` 42 grün_
