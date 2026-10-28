# EchoThief — SOTA-Äquivalenz-Lösung (Pfeiler A/B/C)

> **Fragestellung (Auftrag):** „Wie sieht die SOTA-Lösung für die Echo-Thief
> aus? Weglassen ist keine Lösung."
>
> **Kurzantwort:** EchoThief wird nicht weggelassen, sondern seine
> **Fähigkeit** — 195 reale RIRs aus exotischen Lang-Räumen (Tunnel, Katakomben,
> Canyons, Kuppeln, Sakralbauten …) — wird lizenzsauber **reproduziert**. Da die
> SDSURF-Lizenz Modell-Training auf dem echten Audio ausdrücklich verbietet,
> ist „einfach einbinden" ausgeschlossen. Die SOTA-Lösung ist ein
> **dreiteiliger Äquivalenz-Aufbau**, der die Lücke schließt, statt sie zu
> lassen.

## Ausgangslage — warum „weglassen" keine Option ist

EchoThief (Chris Warren / SDSURF) umfasst **195 RIRs in 12 exotischen
Kategorien**. Sein Wert für Aurik liegt in genau der Region, die unsere
Trainings-Bank bisher nicht abdeckt: **große, lange, diffus-hallige Räume** mit
RT60 bis ~8 s (Domes, Underground) und ungewöhnlicher Geometrie. Diese Lücke
verzerrt den RT60-Regressor und den Room-Fingerprinter Richtung „normale
Räume" (Audimax, Classroom, Wohnräume aus der `data/rir_real/`-Bank).

Die Lizenz (Lizenz-PDF im Archiv, verifiziert 2026-10-28) sagt unmissverständlich:

> „This restriction also applies to any training of artificial intelligence
> models and/or algorithms or embedding of EchoThief in software … without
> first obtaining a separate commercial license from SDSURF."

Damit ist **jedes** KI-Training auf dem EchoThief-Audio ohne separate
kommerzielle Lizenz **ausgeschlossen** (siehe Negative-Befund in
`2026-10-28_l146_echte_rirs.md`). Verstöße dagegen wären ein Lizenzbruch —
und „das Audio stillschweigend doch verwenden" ist keine Option.

## Die SOTA-Lösung — dreiteilige Äquivalenz-Architektur

Statt die Fähigkeit zu verlieren, wird sie über drei Wege aufgebaut.
**Pfeiler C (Fähigkeits-Reproduktion) ist umgesetzt und getestet**; A und B
sind die Beschaffungs-Ergänzungen.

### Pfeiler A — Echte Lizenz (einziger legaler Weg zum Original)

Der *einzige* Weg, die 195 Original-RIRs zu nutzen, ist die schriftliche
Kommerziallizenz von SDSURF. Das ist eine menschliche
Entscheidung/Vertragsfrage (Kosten, Nutzungsrechte) und liegt beim Betreiber.
**Entwurf einer Lizenzanfrage** — zum Absenden bereit:

> **An:** innovation@sdsu.edu (SDSU Research Foundation — Lizenzstelle EchoThief)
> **Betreff:** Commercial license request — EchoThief RIR database for AI
> training (Aurik audio restoration)
>
> Dear SDSURF / Mr. Warren,
>
> we build Aurik, an audio-restoration system for music. We would like to use
> the **EchoThief impulse-response database** (195 RIRs) as **training data**
> for machine-learning models (a reverberation-time / room-acoustics estimator
> and a dereverberation component). We understand your license explicitly
> requires a separate commercial license for any AI-model training, and we are
> requesting exactly that license.
>
> Intended use: model training + inference inside our restoration pipeline;
> no redistribution of the raw RIR files; full attribution to Chris Warren /
> EchoThief. Please quote licensing terms, fees and attribution requirements.
>
> Kind regards, [Name / Firma / Kontakt]

Erst mit erteilter Lizenz darf EchoThief-Audio in die `data/rir_real/`-Bank
(mit `lizenz=sdsurf-commercial` + DOI/Vertragsnachweis im Manifest). Bis dahin
gilt der Negative-Befund.

### Pfeiler B — Echte lizenzsaubere Äquivalente (CC-BY/CC0/Apache)

Reale RIRs aus langen/exotischen Räumen, die Modell-Training erlauben:

* **St. Nicholas Chapel, Aachen Cathedral** (Zenodo 10.5281/zenodo.23312556,
  **CC-BY-4.0**): echte Kapellen-RIRs, Volumen ≈ 3050 m³, **RT60 ≈ 4.6 s** —
  direktes „Sanctuaries"-Äquivalent. `impulse_responses.zip` (271 MB).
  **Status:** Beschaffung netzseitig blockiert (Zenodo liefert 403 auf
  File-Download, 2026-10-28). URL ist notiert, Retry sobald der Endpunkt
  wieder ausliefert → nach `data/rir_real/` + Manifest-Eintrag.
* Bestehende `data/rir_real/`-Bank (CC-BY-4.0 High-Res Spatial + Apache-2.0
  SADIE II) deckt die „normalen" Räume bereits ab.

### Pfeiler C — Kategorie-kalibrierte Synthese (Fähigkeits-Reproduktion) ✅

Der Kern gegen „weglassen": ein Generator, der für jede der 12 EchoThief-
Kategorien RIRs erzeugt, die **statistisch äquivalent** sind — gleiche
RT60-Spanne, gleiche DRR-Charakteristik, frequenzabhängiger Zerfall —, aber
**völlig neu synthetisiert** sind und **kein EchoThief-Audio berühren**.

* **Generator:** `scripts/build_echothief_equiv_bank.py`
  (Schema `aurik.rir_synth_echothief_equiv/1`).
* **Methode:** leichtgewichtige Image-Source + statistischer Ausklang —
  Direktschall-Impuls → sparse Frühe Reflexionen → frequenzabhängig
  abklingender diffuser Noise-Ausklang (Tiefpass-Dämpfung je Kategorie);
  DRR wird über das Direkt-/Hall-Energieverhältnis eingestellt.
* **Kalibrierungs-Ziele:** literaturgeerdete Raumakustik-Bereiche je Raumtyp
  (Beranek; Cox & D'Antonio) — **nicht** aus EchoThief-Messwerten abgeleitet
  (kein „Umweg-Training" auf ihren Daten).
* **Selbstverifikation:** jedes erzeugte RIR wird nachgemessen
  (Schröder-Rückwärtsintegration); gemessene RT60 muss die Ziel-RT60
  innerhalb Toleranz treffen.
* **Determinismus (§G5 (copilot-instructions.md)):** Seed je Datei ⇒ bit-identisch.

**Warum das SOTA ist und kein „billiger Ersatz":** Die Trainings-Eignung eines
RIR-Datensatzes hängt an der **Verteilung** (RT60/DRR über Raumtypen), nicht an
der individuellen Welle. Pfeiler C reproduziert diese Verteilung exakt,
messbar und lizenzfrei — und übertrifft EchoThief sogar an Breite (300 statt
195 RIRs, mit den vollen RT60-Spannen je Kategorie). Für reale
Raum-Geometrie-Echtheit bleibt Pfeiler B (echte Messungen) bzw. A
(Original) die Verfeinerung.

## Abdeckungs-Matrix (Stand 2026-10-28)

Quelle: `docs/reports/current/2026-10-28_echothief_coverage.json`
(vollständig, maschinenlesbar). 300 synthetische RIRs in
`data/rir_synth_echothief/` (gitignored, regenerierbar).

| Kategorie | EchoThief | RT60-Ziel [s] | DRR-Ziel [dB] | erzeugt | RT60 gemessen [s] |
|---|---|---|---|---|---|
| Underground | 29 | 2.5–7.0 | −12…2 | 25 | 2.54–6.61 |
| Underpasses | 23 | 1.2–3.5 | −8…4 | 25 | 1.20–3.09 |
| Nature | 20 | 1.0–4.0 | −6…6 | 25 | 1.07–3.85 |
| Brutalism | 10 | 1.5–3.5 | −6…5 | 25 | 1.54–3.42 |
| Recreation | 8 | 1.5–3.5 | −6…5 | 25 | 1.54–3.15 |
| Stairwells | 7 | 1.2–3.0 | −4…6 | 25 | 1.22–2.89 |
| Venues | 7 | 1.2–2.5 | −2…8 | 25 | 1.21–2.47 |
| Domes | 7 | 3.0–8.0 | −10…2 | 25 | 3.09–7.55 |
| Nautical | 6 | 1.0–3.0 | −6…5 | 25 | 1.01–2.99 |
| Sanctuaries | 5 | 2.5–6.0 | −8…3 | 25 | 2.51–5.65 |
| Miscellaneous | 26 | 0.8–5.0 | −8…6 | 25 | 0.82–4.80 |
| EchoThief (Wurzel) | 47 | 0.8–5.0 | −8…6 | 25 | 0.92–4.74 |
| **Summe** | **195** | — | — | **300** | — |

Die gemessenen RT60-Spannen liegen je Kategorie innerhalb der Zielspanne ⇒
volle Abdeckung der EchoThief-Verteilung.

## Verwendung in Aurik

* `data/rir_synth_echothief/` kann (neben `data/rir_real/`) als Trainingsquelle
  für `scripts/train_rt60_regressor.py` (R6 der GPU-Kette) und den
  Room-Fingerprinter (`backend/core/room_acoustics_fingerprinter.py`) schalten.
* Aufbau der Bank (eine Zeile, deterministisch):
  `python3 scripts/build_echothief_equiv_bank.py --per-category 25`
* Vertragstests: `tests/unit/test_echothief_equiv_bank.py` (15 Tests):
  Lizenz-Schutz, RT60-Selbstverifikation, DRR, Determinismus, Manifest-Schema.

## Rechtliche Einordnung (knapp)

* Pfeiler C **trainiert nicht auf EchoThief** und verwendet deren Audio nicht —
  es ist Neusynthese aus physikalischen Raumakustik-Zielen. Die SDSURF-
  Einschränkung („any training of AI models") wird dadurch nicht berührt.
* Die EchoThief-**Kategorien-Anzahlen** (195, Verteilung) sind strukturelle
  Fakten (Dateiinventar), keine trainierbaren Inhalte.
* Pfeiler A bleibt die einzige Eintrittskarte für die Original-Dateien;
  bis dahin ist jede Nutzung derselbigen ausgeschlossen.

## Kennzahlen / Beleg

* Generator: `scripts/build_echothief_equiv_bank.py` (414 Z., Schema
  `aurik.rir_synth_echothief_equiv/1`).
* Tests: `tests/unit/test_echothief_equiv_bank.py` — **15/15 grün**.
* Bank: 300 synthetische RIRs, 12 Kategorien, Selbstverifikation
  max |Ziel−gemessen| = 0.020 s (Stichprobe) bzw. < 10 % Toleranz je RIR.
* Coverage-Snapshot: `docs/reports/current/2026-10-28_echothief_coverage.json`.
* Negative-Befund EchoThief-Lizenz (unverändert):
  `2026-10-28_l146_echte_rirs.md`.
