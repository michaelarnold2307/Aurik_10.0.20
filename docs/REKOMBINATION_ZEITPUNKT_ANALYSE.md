# Rekombinations-Zeitpunkt-Analyse: Wann muss die punktgenaue Zusammenführung erfolgen?

> Status: Analyse · Bezug: §v10.19-Paket (StemLevelRestorer §SLR-1, StemContext),
> Hörordnung (`hoerordnung.instructions.md`: Maskierungsschwelle statt Mess-Null)
> Datum: 2026-09-12

## 1. Fragestellung

Nach der getrennten Behandlung von **Musik** (DFN + KIM Inst) und **Gesang**
(MIIPHER-DiT + KIM2) muss die Zusammenführung so präzise erfolgen, dass das
menschliche Ohr **niemals wahrnimmt, dass separiert wurde**. Diese Analyse
bestimmt (a) den **exakten Zeitpunkt** der Zusammenführung, (b) die
**psychoakustischen Bedingungen** für Unhörbarkeit und (c) die daraus
abgeleiteten **neuen Witness-Kandidaten**.

## 2. Ist-Zustand (Code-Evidenz)

- `StemLevelRestorer._run` (§SLR-1f): **ein** Rekombinationspunkt — additive
  Summe `_vocal_out + _instr_out`, sample-exakt, **nach** allen Stem-Stufen,
  **vor** dem Mix-Phasen-Loop (UV3 Zeile ~13428, Pre-Phase).
- Guard: VQI-Gate (`vqi_after < 0.72` → Rollback) auf dem **remixten** Signal.
- `StemContext` hält die finalen Stems für Phasen 19/43/66 bereit.
- **Lücken (Befund):**
  1. ML-Separation (BS-RoFormer/Demucs) ist **nicht** perfekt-rekonstruierend —
     `vocal + instr ≠ input`. Das **Residuum wird bei der Summe still
     verworfen**; nur das grobe VQI-Gate fängt Schäden.
  2. **Kein Alignment-Check** vor der Summe: MIIPHER-DiT/KIM2/KIM-Inst/DFN
     verarbeiten die Stems **unabhängig** → latenz-/kantenbedingter
     Sample-Versatz zwischen den Stems → Kammfilter am Nahtpunkt.
  3. Stereo-Kohärenz (IACC/ILD/ITD) wird bei der Rekombination **nicht**
     geprüft (interaural_cues existiert, läuft aber erst im
     Stereo-Safety-Guard, nicht am Nahtpunkt).

## 3. Psychoakustische Bedingungen für Unhörbarkeit

| # | Bedingung | Kriterium (Hörordnung: Maskierung, nicht Mess-Null) |
|---|---|---|
| C1 | Perzeptuelles Residuum | `\|mix − vocal − instr\|` muss unter der **Bark-Maskierungsschwelle** des lokalen Inhalts liegen |
| C2 | Zeit-Alignment | Gruppenlaufzeit beider Stems **sample-exakt** gleich (Kammfilter-Präkursor) |
| C3 | Stereo-Kohärenz | IACC/ILD/ITD-Drift des Remixes vs. Original unter den Hör-JNDs (30/60 µs ITD, 1/2 dB ILD, ΔIACC 0.08/0.15) |
| C4 | Keine Doppelverarbeitung | überlappende Spektralanteile beider Stems dürfen nicht beide verändert worden sein (sonst Kamm-/Phantom-Artefakte) |
| C5 | Pegel-Kontinuität | kein Gain-Sprung am Nahtpunkt (Soft-Knee/Crossfade nur falls nötig) |

## 4. Zeitpunkt-Analyse (WANN zusammenführen?)

**Option A — aktuell: früh, ein Punkt, vor dem Mix-Loop.**
Pro: Mix-Phasen (03/29/54/Glue) sehen ein kohärentes Ganzes; N=1 Nahtstelle;
Stem-Domänen-Verarbeitung endet dort, wo Mix-Ziele beginnen.
Kontra: Residuum früh verworfen; Stem-Latenzen kumulieren ungeprüft bis zur Summe.

**Option B — Stems bis zum Ende getrennt halten.**
Pro: punktgenaue Domain-Kontrolle bis zum Export.
Kontra: Mix-Ziele (Loudness/Glue/Stereo-Weite) brauchen den Mix; Phasen müssten
N-mal rekombinieren → **N Nahtstellen, kumulative Artefakte**. Verworfen (§V7:
Ursache statt Symptom, kein Bypass durch Dauer-Separation).

**Option C — EMPFOHLEN: ein Rekombinationspunkt an der jetzigen Stelle,**
**aber mit Vor-Gates direkt VOR der Summe (§SLR-1f):**
1. **C2-Alignment-Korrektur**: Kreuzkorrelations-Peak-Offset zwischen
   Vokal-/Instrumental-Stem-Hüllkurven → Sample-Korrektur vor der Summe.
2. **C1-Residuum-Gate**: Bark-gewichtete Residuum-Energie vs. Maskierungs-
   schwelle; überschreitet sie die Schwelle, wird das Residuum (nicht
   verworfen, sondern) **spektral bedarfsweise zurückgemischt** — die
   Separation wird dadurch perzeptuell perfekt.
3. **C3-Stereo-Check**: interaural_cues am Nahtpunkt (nicht erst beim Export).
4. Danach wie bisher: VQI-Gate als letzte Instanz auf dem Remix.

**Begründung des Zeitpunkts:** Die Zusammenführung muss **nach** der letzten
Stem-Stufe (KIM-Inst §SLR-1e3) und **vor** der ersten kohärenzbedürftigen
Mix-Phase (phase_03 Breitband-NR) liegen — genau die jetzige Position. Später
(im Loop) wäre jede Phase eine neue Nahtstelle; früher (vor KIM2/KIM-Inst)
würde die getrennte Brillianz-Behandlung unvollständig. Der Punkt ist also
**richtig gewählt**; was fehlt, sind die C1–C3-Gates unmittelbar davor.

## 5. Neue Witness-Kandidaten (indirekt aus dieser Analyse)

| ID | Witness | Erkennt was | Status |
|---|---|---|---|
| W1 | **Residuum-Witness** | verworfenes `mix − vocal − instr` oberhalb der Maskierungsschwelle (C1) | **neu** |
| W2 | **Alignment-Witness** | Sample-Versatz der Stems zueinander (C2, Kammfilter-Präkursor) | **neu** |
| W3 | **Stereo-Kohärenz-Witness** | IACC/ILD/ITD-Drift Remix vs. Original am Nahtpunkt (C3) | interaural_cues vorhanden → **am Nahtpunkt verdrahten** |
| W4 | **Kammfilter-Witness** | spektrale Ripple-Tiefe nach der Rekombination (Cepstral-Peak) | **neu** |
| W5 | **Stem-Leakage-Witness** | Vokal-Energie im Instrumental-Stem & umgekehrt (Geister-Anteile) | **neu** |
| W6 | **Seam-Kontinuitäts-Witness** | Gruppenlaufzeit-Sprung über die Nahtregion (C2/C5) | **neu** |

Alle sind **deterministisch, report-only** (Hörordnung §8a) — als Gates dienen
sie in der Reihenfolge C2 → C1 → C3, mit den Hör-JNDs als Schwellen (keine
neuen magischen Zahlen, Hörordnung Ebene 1/2).

## 6. Umsetzungsreihenfolge (nächste Slices)

1. ✅ W2 Alignment-Korrektur + W1 Residuum-Gate **vor** §SLR-1f (C1/C2) —
   `backend/core/dsp/stem_recombination_gates.py` (2026-09-12): Hüllkurven-
   xcorr (±50 ms, normierter Koeffizient ≥ 0.25), Bark-Residuum-Gate gegen
   die Maskierungsschwelle des Gehörten, STFT-Rückmischung.
2. ✅ W3 interaural_cues am Nahtpunkt (C3) — im selben Modul, report-only.
3. W4/W5/W6 als Witness-Reports in `StemContext.witness_reports` (Messung,
   Kalibrierung gegen echte Songs im überwachten Run)
4. Kalibrierung: Schwellen auf P90 der No-Harm-Deltas über N≥3 Songs
   (Evidenzblock, 95 %-CI, Maintainer Sign-off — PR-Vertrag §4 AGENTS.md)
