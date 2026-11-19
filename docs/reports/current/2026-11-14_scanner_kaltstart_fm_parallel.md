# Scanner-Kaltstart-Teilbehebung: Subband-FM bit-identisch parallelisiert (2026-11-14)

**Anlass:** Der Handover-Befund 2026-11-13 dokumentierte den kalten
DefectScanner-Lauf deutlich über dem 16-s/min-Budget (28,87 s/min) und
benannte als offene Untersuchung: „Cache-/Kanal-Duplikation bleibt für
eine sichere Performance-Optimierung zu untersuchen". Das cProfile wies
`_coherent_subband_fm` als dominanten Einzelposten aus (2× 6,312 s).

## Diagnose

Instrumentierter Lauf (225-s-Stereo-Titel, frischer Prozess): Der
Subband-FM-Kern läuft pro Scan **zweimal** — einmal in der
Material-Erkennung auf dem vollen Audio (35,9 MB) und einmal im
Haupt-Scan auf dem 60-s-Center-Crop (11,5 MB). Der Content-Keyed-Cache
(`_wf_cached_features`) kann hier nicht greifen, weil die Eingangs-Arrays
verschieden sind — beide Läufe sind semantisch nötig. Die Kosten liegen
im seriellen Band-Loop: ~121 Butterworth-Bänder × `sosfiltfilt` +
Hilbert + `arctan2`/`unwrap` pro Band, je Band eine unabhängige, reine
Funktion aus `(fc, audio, sr)`.

## Fix (§PERF-R21b)

`_coherent_subband_fm`: Band-Loop in `_fm_band(fc)` extrahiert (reine
Funktion, identische Operationen, identische Abbruch-Gates) und über
`ThreadPoolExecutor` (≤ 6 Worker, Band-Reihenfolge erhalten) ausgewertet.
Keine RNG-Quellen, keine geteilten Zustände, Sammlung in Band-Reihenfolge
⇒ **bit-identisch zur seriellen Schleife (§G5 (copilot-instructions.md))**.

## Verifikation

| Prüfung | Ergebnis |
| --- | --- |
| FM-Kern 60 s (3 Läufe, gleiche Maschine) | **6,79 s → 3,06 s pro Aufruf (2,2×)**, Ergebnis-Floats identisch |
| Scan-Fingerprint Boogie (VINYL) | vor/nach identisch `095b5c00…` |
| Scan-Fingerprint tape33 (CASSETTE) | vor/nach identisch `5ce48454…` |
| Scan-Fingerprint Träumer (VINYL) | vor/nach identisch `c1a44dd5…` |
| Scanner-Test-Suiten (gold_standard, crop_rescue, anti_fp, crackle, gap, dk38) | 81 grün |
| py_compile + Ruff | clean |

Ehrliche Grenze: Die Kalt-Gesamtzeit wird auf dieser Maschine derzeit von
CPU-Konkurrenz (Kun-GUI, Load ≈ 3,5) dominiert — der FM-Kern-Gewinn ist
der verifizierte Anteil. Die zweite FM-Berechnung (Material-Erkennung auf
vollem Audio) ist semantisch nötig; eine weitere Beschleunigung müsste am
Material-Erkennungspfad ansetzen (eigene Untersuchung).

Kein Flag, keine Klangänderung — reine Performanz mit bit-identischem
Ausgang, damit sofort im Produktionspfad aktiv.
