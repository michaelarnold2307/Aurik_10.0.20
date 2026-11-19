# Analyse-Plan: Punktgenaue SOTA-Bewertung von Gesang & Musik nach der Restauration

> Status: Plan · Bezug: §v10.19-Paket (KIM2/KIM-Inst/StemContext), Hörordnung
> (`hoerordnung.instructions.md`), Listening-Witness (REPORT-ONLY, §8a)
> Datum: 2026-09-12 · Ergänzung: docs/REKOMBINATION_ZEITPUNKT_ANALYSE.md (Zeitpunkt der Zusammenführung + neue Witness-Kandidaten W1-W6)

## 1. Ziel

Nach der sauberen Restauration von **Musik** (DFN + KIM Inst, §SLR-1d/1e3) und
**Gesang** (MIIPHER-DiT + KIM2, §SLR-1b/1e2) muss punktgenau nachgewiesen werden:

1. dass jede Stufe **hörbar besser oder neutral** ist (nie schlechter, §0),
2. dass die **Witness-Gate-Schwellen** gegen echtes Material korrekt kalibriert
   sind (weder zu lax → Artefakte durch, noch zu streng → Nutzen verschenkt),
3. ob **Anpassungsbedarf** an den Stufen, Schwellen oder der Slot-Reihenfolge
   besteht — als messbare Evidenz, nicht als Bauchgefühl.

## 2. Mess-Ebenen (pro Bereich)

| Bereich | Witness-Metriken (bereits vorhanden) | Zusatz-Metrik (zu ergänzen) |
|---|---|---|
| Gesang | `hnr_drop_db`, `pitch_drift_cents`, `flat_top_rise` | Air-Band-Energie 10–20 kHz (Brillanz) |
| Musik | `bass_drop_db`, `transient_smear_ratio`, `flat_top_rise` | Instrumental-HNR bzw. Warmt/Sharness-Delta |
| Mix | `loud_mod_rise_db` (Pumpen), VQI, SNR-Gain | Stereo-Kollaps, Interaural (ITD/ILD) |

Referenz: `listening_witness.evaluate_listening_witness()` ist deterministisch und
report-only — die Gates in §SLR-1e2/1e3 nutzen sie als Entscheider.

## 3. Ablauf des überwachten Runs

1. **Vorlauf:** ein repräsentatives Importfile (Gesang + Instrumental) über die
   Bridge/UV3 restaurieren; Log mit `§SLR-1*`, `§KIM2`, `§KIM-Inst`,
   Witness-Report pro Phase sammeln.
2. **Per-Phase-Reinhören:** nach jeder Phase den Witness-Report der Phase
   auswerten (Pitch-Drift, HNR, Bass, Transienten, Pumpen) — Abweichungen
   gegenüber dem Input sind die "Übeltäter"-Kandidaten.
3. **Stem-Fokus:** nach §SLR-1 die `stem_context`-Stufen einzeln beurteilen:
   - MIIPHER-DiT akzeptiert? (HNR-Gewinn, kein Pitch-Drift)
   - KIM2 akzeptiert? (Brillanz-Gewinn, Witness-Gate hält sonst)
   - DFN + KIM Inst akzeptiert? (Bass/Transienten stabil)
4. **Gate-Statistik:** Zählung `applied/rejected/reason` pro Stufe über mehrere
   Songs → Kalibrierung der Schwellen (siehe §5).

## 4. Anpassungsbedarf — Entscheidungsbaum

- **Gate lehnt fast immer ab** (`witness_gate` dominant) → Schwelle zu streng
  ODER Stufe erzeugt echte Regression → Ursache am Stem prüfen (KIM-Inst auf
  transientenreichem Material z. B. STFT-Verschmierung).
- **Gate akzeptiert, Reinhören zeigt Artefakt** → Witness-Metrik blind für
  diesen Artefakttyp → Metrik ergänzen (z. B. Air-Band-Roughness).
- **Slot-Reihenfolge:** KIM2/KIM-Inst nach Hallucination-Guard? Wenn der Guard
  KIM-Ausgaben zurückrollt, Reihenfolge tauschen (Guard zuletzt).
- **Mix-Ebene:** SOTA-4-Layer-Denoiser (phase_03) + Stem-Block interagieren —
  prüfen, ob die Stem-Stufen auf bereits geraubtem Input unnötig greifen.

## 5. Kalibrierungsschleife (Schwellen)

Aktuelle Werte (konservativ): KIM2 `hnr_drop<1.0, pitch<8 ct, flat_top<0.02`;
KIM Inst `bass_drop<1.0, smear>0.85, flat_top<0.02`.

Nach N≥3 Songs: Perzentil der akzeptierten/rejected Deltas → Schwellen auf
P90 der gemessenen No-Harm-Deltas legen; jede Änderung nur mit Evidenzblock
(Seed, 95 %-CI, Maintainer Sign-off — PR-Vertrag §4 AGENTS.md).

## 6. Deliverables

1. Lauf-Report: per-Phase-Witness-Tabelle + Stem-Stufen-Statistik
2. Kalibrierungs-Evidenz (falls Schwellen angepasst werden)
3. Liste der "Übeltäter" mit Root-Cause-Fix (kein Bypass, §V7)
