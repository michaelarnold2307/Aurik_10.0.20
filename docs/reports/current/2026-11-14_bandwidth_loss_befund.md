# bandwidth_loss-Klassenkopf: Pre-Flight erstmals BESTANDEN — Amtsinhaber repariert die Klasse bereits (2026-11-14)

**Fahrplan-Referenz:** `docs/reports/current/2026-11-13_aktivierungs_fahrplan.md`,
Kette B (Per-Klassen-Köpfe). Erster Versuch nach dem groove_echo-Stopp
mit datengetriebener Klassenwahl.

## Warum bandwidth_loss?

Die Metrik-Sichtbarkeits-Messung (2026-11-14) über alle k398-Rezept-Familien
ergab: Die Gate-Metrik `defektlast_lang` sieht einen 8-kHz-Bandlimit-Schaden
am stärksten von allen Schadensfamilien — **63 % der Regionen > JND 0,05,
median Δ +0,17** (groove_echo: 0/24 in richtiger Richtung; rausch bei
24 dB SNR: 5 %; clipping ±0,90: 4 %). Die Schadens-Abbildung ist ein
linearer Tiefpass — die am einfachsten lernbare Abbildung der Q≈0-Klassen.

## Was gebaut und gemessen wurde

| Schritt | Ergebnis |
| --- | --- |
| Datensatz `scripts/erstelle_bandlimit_paare.py` (Produktions-Variante bandlimit, 8192er-Fenster, Seed 42, Kontrast-Filter > JND) | 1.482/3.640 Regionen, 5 Vertragstests |
| Training `bandwidth_loss` (40 Epoch × 300, GPU) | val_mae 0,122 |
| **Pre-Flight In-Sample-Q (Fahrplan B-0(3))** | **0,6604 ≥ 0,5 — BESTANDEN (erstmals überhaupt)** |
| Holdout-Q (13 Regionen, 1 Titel) | 0,093 — verfehlt |
| Variante Proxy-Gewicht 1,0 (60 Epoch) | In-Sample 0,45, Holdout 0,007 — schlechter |

## Der entscheidende Befund: Der Amtsinhaber repariert die Klasse bereits

Direkte Messung auf denselben Regionen:

- `phase_39_air_band_enhancement` (der Produktions-Amtsinhaber für
  BANDWIDTH_LOSS > 0,10): **Q-Median 5,52** auf den Test-Regionen.
- Voller UV3-Produktionspfad auf dem Schlimmstfenster: defektlast
  2,47 → 0,09 (Δ +2,37); k398-Kassette-Schlimmstfenster: Δ +0,41.

Der Produktionspfad repariert Bandlimit-Schäden also bereits **deutlich
besser als der neue Kopf** (Q 5,5 vs. 0,093). Ein Challenger, der dem
Amtsinhaber unterliegt, wird nicht verdrahtet — kein Flag-Flip, keine
Aktivierung. Der Kopf bleibt ungenutzt in `models/klassen_koepfe/`.

## Konsequenz für die Q-Mission

Der datengetriebene Auswahl-Mechanismus funktioniert (Pre-Flight erstmals
bestanden — die Metrik-Sichtbarkeit ist der richtige Selektions-Maßstab).
Für bandwidth_loss ist die Reparatur aber schon im Produktionspfad
vorhanden; die §G13-Q-Erstbelege hatten die Klasse mangels Clean-Kontrast
gar nicht bewertet. Nächster Kandidat: eine Q≈0-Klasse mit Metrik-Kontrast
**und ohne** produktiven Amtsinhaber — Kandidatenbestimmung nach derselben
Sichtbarkeits-Messung (z.B. dropout_oxide: 20/24 Kontrast, 19/24 richtige
Richtung; hum 7/7, crackle 5/5).

Kein Flag-Flip, keine Verdrahtung geändert. Der Musik-Declip-Kopf
(Kette A, objektiv bestanden, §4-Hörpaket liegt vor) ist nicht betroffen.
