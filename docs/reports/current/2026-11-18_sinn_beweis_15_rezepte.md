# Sinn-Beweis 15 Rezepte (2c-2) — 2026-11-18, CPU-only, Seed 42, pegel-gematcht

> Nach dem FORCE_CPU-Fix (miipher-Provider-Override entfernt + 13 weitere
> Device-Guards) lief der volle 15-Rezept-Lauf erstmals komplett durch:
> 45 Paare (3 Titel × 15 Rezepte), Produktions-Pipeline auf CPU bei voller
> GPU (Cantus-pretrain lief parallel) — kein Hip-/OOM-Crash mehr.

## Setup

- Quellen: `data/schadens_quellen/` (eindeutig benannte Symlinks auf MUSDB18-HQ-Train-Mixturen; Seed-42-Auswahl: „Music Delta - Rock“, „Alexander Ross - Goodbye Bolero“, „Snowmine - Curfews“).
- Bewertung: `data/schadens_proben/bewertung.json` (schema aurik.schadensbewertung/1, pegel_matching=True, jnd_defektlast 0,05).
- n_besser **6/45**, n_ueberrestauration **0**.

## Klassen-Befund (Mittel je Rezept, n=3)

| Rezept | Δ defektlast | Δ musik_mos | besser |
| --- | --- | --- | --- |
| **raum** | **−0,395** | **+0,039** | **3/3** |
| mp3_128 | −0,012 | −0,017 | 1/3 |
| vinyl | −0,005 | −0,012 | 0/3 |
| kassette | −0,001 | −0,025 | 0/3 |
| clipping | +0,004 | −0,029 | 1/3 |
| kammfilter | +0,000 | −0,003 | 1/3 |
| clicks/kompression/verzerrung/amp_drift/lautheit/pitch_drift/dropout/hum/wowflutter | ≈ 0,000 … +0,001 | −0,018 … −0,028 | 0/3 |

## Deutung (ehrlich)

- **raum (Dereverb) ist die einzige klar reparierte Klasse** — konsistenter
  defektlast-Gewinn auf allen 3 Titeln bei gleichzeitigem MOS-Gewinn
  (DRT-14/reparatur_nachweis-Evidenz für die Paritäts-Register-Zeile).
- Alle übrigen Klassen sind auf diesen 3 Titeln **neutral bis leicht
  negativ im MOS** (Beträge ≪ JND, 0 Überrestaurationen) — konsistent mit
  dem 12-Paar-Vergleich vom 16.11. (0/12) und den Q≈0-Befunden: die
  Bestands-Pipeline repariert diese Rezept-Klassen auf MUSDB-Quellen nicht
  hörbar-positiv. **Keine Aktivierung, kein Flag-Flip.**
- Folge-Evidenz gehört in `.github/WETTBEWERBS_PARITAET.md` (Raum-Zeile)
  und speist die nächste Sinn-Beweis-Runde (mehr Titel je Klasse).
