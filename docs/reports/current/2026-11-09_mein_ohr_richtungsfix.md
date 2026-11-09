# „Mein Ohr" — Richtungsfehler im Ohr-Prior gefunden und behoben (D-K3-89)

**Datum:** 2026-11-09 · **Klasse:** K3 (Verdrahtung/Messpfad) · **Kein Flag-Flip,
keine Hör-/Lizenzentscheidung, keine Schwellen-Änderung.**

## 1. Befund

Der in dieser Sitzung entstandene Ohr-Prior (`backend/core/dsp/ohr_prior.py`)
fütterte den Bradley-Terry-Head mit **Rohwerten**:

| Feature | Rohwert-Bedeutung | Prior-Gewicht | Wirkung im Ohr |
| --- | --- | --- | --- |
| `defektlast` | **höher = mehr Defekte** | **+0,30** | „mehr Defekte sind besser" |
| `roughness_asper` | höher = rauer | +0,05 | „rauer ist besser" |
| `sharpness_acum` | höher = schärfer | +0,04 | „schärfer ist besser" |
| `fluctuation_vacil` | höher = mehr Fluktuation | +0,03 | „Pumpen ist besser" |
| `musik_mos` | höher = besser | +0,40 | korrekt |
| `sii` | höher = besser | +0,10 | korrekt |
| `loudness_phon` | nicht monoton | +0,08 | linear statt Band |

Der `defektlast`-Term ist der schwerste Fall: Die zertifizierte Definition
(`backend/core/dsp/wohlklang_zeuge.py:69`) ist eine **Defekt-Last** — höher ist
schlechter. Mit positivem Prior hebelt das Ohr damit den stärksten
Musik-MOS-Term nahezu aus.

**Messung an echter Musik** (`data/musdb18hq/test/Motor Tapes - Shore/mixture.wav`,
10 s, 44,1 kHz, plus Weißrauschen bei −40 dBFS, Seed 7):

| Größe | vorher (Rohwerte) | nachher (Qualitäts-Vektor) |
| --- | --- | --- |
| `w · Δ(clean − noisy)` | **+0,045** | **+0,354** |
| P(clean gewinnt) | **0,511** (Münzwurf) | **0,588** |
| `defektlast`-Beitrag | **−0,149** (bestraft clean!) | **+0,149** |

Die Differenz der beiden Vektoren ist exakt der Vorzeichenwechsel des
`defektlast`-Terms; `musik_mos` (Δ −0,513 × 0,40) und `sharpness` (Δ −0,138 ×
0,04) wirken in beiden Fällen richtig.

## 2. Fix

1. **Qualitäts-Richtung zentral:** neuer reiner Transformator
   `qualitaets_vektor(rohwerte)` + `FEATURE_RICHTUNG`
   (`hoeher_besser` / `niedriger_besser` / `band`). `feature_vector()` misst
   nur noch Rohwerte und ruft den Transformator auf.
2. **Loudness als Band:** `band_qualitaet(phon, 70, 50)` — 70 phon Ziel,
   20/120 phon = 0. Begründet als **Prior** (musik-typisches Band), kein Gate;
   der gelernte Fit darf ihn überstimmen.
3. **Schema `/2`, fail-closed:** `OhrPriorFit.from_dict()` lehnt jedes andere
   Schema mit `ValueError` ab. Ein `/1`-Artefakt (Rohwert-Semantik) wird
   **nicht** interpretiert, sondern muss neu kalibriert werden (§V6).
   `lade_ohr_prior()` meldet `grund="ValueError"`, der Ohr-Brief bleibt
   `verfuegbar=False` (Default-Pfad unverändert).
4. **Determinismus:** unverändert — reine Funktionen, keine Zeit, kein
   Sampling, Seed 42 (§G5).

## 3. Tests

`tests/unit/test_ohr_prior.py` (+3):

- `test_qualitaets_vektor_richtungen` — pinnt die Drehung je Feature und die
  Band-Güte (70 → 1,0; 20 → 0,0; 95 → 0,5).
- `test_feature_qualitaets_richtung_auf_breitbandsignal` — End-to-End auf
  selbst erzeugtem Musik-Signal (rosa Rauschen + Hüllkurve + Transienten) vs.
  −30 dB Weißrauschen: `musik_mos`, `defektlast`, `sharpness` sinken, der
  Prior zieht das saubere Signal vor.
- `test_alte_schema_version_wird_abgelehnt` — `/1` ⇒ `None` + `ValueError`.

Verifikation: `tests/unit/test_ohr_prior.py`,
`test_sweet_spot_baseline_relative.py`, `test_denker/test_defekt_denker.py`,
`test_denker/test_strategie_denker.py` → **84 passed**.

## 4. Grenzen (ehrlich)

- **Kein Artefakt vorhanden** (`models/ohr/` ist leer): Die Wirkung ist bis zur
  menschlichen Kalibrierung **null** — der Default-Pfad bleibt bit-identisch.
- **Kalibrierung ist menschlich:** `python3 -B scripts/ohr_panel.py
  --erzeuge-paare --verzeichnis <Musikordner>` erzeugt A/B-Paare und fragt
  l/ r/ = ab; erst danach entsteht `models/ohr/ohr_prior.json`.
- Die Denker-Verdrahtung ist **report-only** (Ohr-Brief in den Ergebnissen,
  Priorisierung im Sweet-Spot-Loop); sie entscheidet nie gegen das
  Never-worsen-Gate oder die Hör-Instanz (§7 Hörordnung).
- `sii` bleibt als Feature mit kleinem Gewicht (0,10) erhalten, obwohl es ein
  Sprach-Intelligibilitäts-Maß ist; die Musik-Tauglichkeit ist nicht belegt.
  Das ist als offener Punkt dokumentiert, kein Qualitätsurteil.

## 5. Belege

- `backend/core/dsp/ohr_prior.py` (Fix, Schema `/2`)
- `backend/core/dsp/ohr_brief.py` (zentrale Quelle, fail-safe)
- `tests/unit/test_ohr_prior.py` (15 Tests)
- Messung: MUSDB18-HQ `Motor Tapes - Shore/mixture.wav`, 10 s, Seed 7
  (Zahlen oben; reproduzierbar über `feature_vector`/`OHR_PRIOR_WEIGHTS`)
