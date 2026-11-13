# Musik-Declip-Kopf: leak-freier Produktionspfad-Gate-Lauf (2026-11-13)

**Zweck:** Objektiver Teil der D-K4-1-Aktivierungskette für den
Musik-Declip-Kopf (`use_aspade_declipper`-Kaskade, Flag AUS bis §4).

## Aufbau

- **Verdrahtung:** `backend/core/dsp/musik_declip_kopf.py` (neu) —
  Torch-Checkpoint `models/klassen_koepfe/musik_declip/best.pt` (345 KB),
  8192er-OLA mit Rand-Reflection-Padding, **Clip-Masken-Blend** (Kopf nur in
  geclippten Regionen + 5-ms-Überblendung) und **Harmonik-Proxy-Witness**
  (Rückgabe nur bei striktem Gewinn). In `DeclipperPhase` als zweiter
  Challenger hinter `use_aspade_declipper` verdrahtet — pro Kanal nur bei
  striktem Gewinn gegen Input UND Amtsinhaber; Flag AUS ⇒ bit-identisch.
- **Gate-Material:** MUSDB18-HQ-Titel 61–90 (30 Titel) mit dem
  Produktionsrezept `clipping` — **nie im Training** (leak-frei; die
  k398-Titel, deren Clean-Kulturen im Training lagen, sind hier bewusst
  NICHT das Gate-Kriterium).
- **Messung:** echter Produktionsschritt `DeclipperPhase` je Titel
  (Flag AUS = Basis, Flag AN = Challenger-Kaskade), 30-s-Schlimmstfenster,
  SDR best-lag-alignt gegen das Clean-Original.

## Befund (10 Titel, Seed-deterministische Auswahl)

| Titel | SDR Input | SDR Flag-AN | Δ |
| --- | ---: | ---: | ---: |
| ANiMAL — Clinic A | −8,82 | −8,82 | 0,00 |
| ANiMAL — Easy Tiger | −9,56 | −9,57 | −0,01 (sub-JND) |
| ANiMAL — Rockshow | −10,18 | −10,18 | 0,00 |
| A Classic Education — NightOwl | −8,49 | −8,49 | 0,00 |
| Actions — Devil's Words | −8,75 | **−7,92** | **+0,83** |
| Actions — One Minute Smile | −10,23 | **−9,04** | **+1,19** |
| Actions — South Of The Water | −9,23 | −9,23 | 0,00 |
| Aimee Norwich — Child | −9,03 | **−8,63** | **+0,40** |
| Alexander Ross — Goodbye Bolero | −10,32 | **−9,28** | **+1,04** |
| Alexander Ross — Velvet Curtain | −10,41 | −10,42 | −0,01 (sub-JND) |

**4/10 besser, 0/10 schlechter über JND (0,05) — Never-worsen ✓.**

## Einordnung

- Der Musik-Declip-Kopf verbessert den Produktionspfad auf **unabhängigem,
  nie trainiertem Material** in 4/10 Fällen messbar und verschlechtert nie.
  Auf dem k398-Gegenmaterial (LOO, Report `2026-11-13_musik_declip_
  durchbruch.md`): 3/3 verbessert, Boogie +15,6 dB.
- **Noch kein Flag-Flip.** Offen für die Aktivierung: blindes A/B +
  Hörordnungs-Sign-off (§4, menschlich). Der objektive Gate-Teil
  (nie-schlechter, echter Produktionspfad, leak-frei) ist damit erstmals
  für einen Kandidaten erfüllt.
