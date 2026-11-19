# §4-Hörpaket Musik-Declip-Kopf — Evidenz-Bundle (2026-11-14)

**Zweck:** Alle objektiven Gate-Belege gebündelt für die blinde A/B-Sitzung
(Hörordnung §4, menschlich). Kein Flag-Flip — `use_aspade_declipper` bleibt
AUS bis zum Hörordnungs-Sign-off.

## Proben

`study_stimuli/hoerproben/declip_produktionspfad/` — 3 blinde A/B-Paare
(Boogie/Träumer/Jets), je 10 s, 48 kHz, Schlimmstfenster (meiste ±0,85-Plateaus),
erzeugt über den **echten UV3-Produktionspfad** (Flag-AUS vs. Flag-AN,
Seed-42-Seiten, `schluessel.json` getrennt). **p25-Pegelmatching**: beide
Varianten je Paar auf den Eingangs-Pegel skaliert (rel. Delta 0,00 % verifiziert)
— die k398-Rezeptur dämpft 5× und der Musik-Declip-Kopf stellt das Clean-RMS
wieder her; ohne Matching wäre die Antwort durch Lautheit vorbestimmt.

## Objektive Gate-Belege (bereits erbracht)

| Beleg | Ergebnis | Quelle |
| --- | --- | --- |
| Leak-freier Produktions-Gate-Lauf, 10 MUSDB-Titel 61–90 (nie im Training) | **4/10 besser, 0/10 schlechter** über JND — Never-worsen ✓ | `2026-11-13_musik_declip_gate_leakfrei.md` |
| k398-Gegenmaterial leave-one-out | **3/3 besser** (Boogie +15,6 dB SDR) | `2026-11-13_musik_declip_durchbruch.md` |
| Produktionspfad-Erreichbarkeit | 6/6 Skip-Schichten gefixt; Probe v11: Phase läuft, Flag-Wirkung delta 1,86 | `2026-11-13_uv3_declip_erreichbarkeit.md` |
| Never-worsen-Witness | pro Kanal nur bei striktem Gewinn gegen Input UND Amtsinhaber | `backend/core/dsp/musik_declip_kopf.py` |
| A-SPADE-ONNX selbst | ΔSDR-Gate 0/3 Titel — NICHT aktivierungsreif; bleibt Witness-geschützt | `aspade_declipper_gate.json` |

## Befund je Paar (diagnostisch, vor dem Hören)

- **Boogie:** A(flag_aus) zeigt die echten ±1,0-Plateaus (Pipeline-Loudness
  ohne Declip = heutiger Produktionszustand); B(flag_an) repariert.
- **Träumer:** nahezu identisch — der Kopf greift nur am 6,3-s-Plateau, der
  Witness lehnt den Rest ab (konsistent mit „minimal schlechter" im Gate-Lauf).
- **Jets:** deutlich verschieden; größte Einzelwirkung am Schlimmstfenster.

## Ablauf

1. Blind hören: `scripts/hoer_entscheidung.py` (ffplay/aplay/paplay) oder
   Direktabspiel der WAVs + Schlüssel erst danach öffnen.
2. Urteil aufzeichnen (Beleg-Schema `aurik.ab_stats/1`).
3. Sign-off: `scripts/flag_freigabe.py --flag use_aspade_declipper
   --verdikt <beleg> --signoff <datei.md> --commit` — erst dieser Schritt
   flippt das Flag; ohne ihn bleibt es AUS (§4).
