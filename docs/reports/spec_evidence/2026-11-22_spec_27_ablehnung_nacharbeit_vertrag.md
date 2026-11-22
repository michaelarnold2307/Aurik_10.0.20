# Spec-27-Evidenzbericht — Ablehnungs-Nacharbeit-Vertrag (ANT)

Datum: 2026-11-22 · Schema: Spec-Evidenz (Spec 26 DRT-16-Form, reduziert auf Nicht-Statistik)
Status: Eingecheckt mit Spec 27 im selben Push (spec_evidence_gate-Pflicht).

## Evidenzblock

- **Warum Spec 27:** In der Sitzung 2026-11-21/22 wurden mehrere Gate-Ablehnungen
  als **Mess-Artefakte** statt als Qualitätsurteile entlarvt (Witness-Mel-Drift:
  max-normalisierte Mel löscht die Pegel-Dimension, in der dropout_oxide
  +17,4 dB / dropouts +13,5 dB / tape_splice +4,6 dB Kontrast haben; gemessen
  in der Mel: 0,01 — Belege `/tmp/pool_exp4.txt`, `/tmp/abs_mel.txt`,
  `/tmp/paar_dist.txt`). Dasselbe Muster war zuvor bei W-3 (Dauer-Achsen-Bug),
  W-2 (Δ ≤ 1e-4) und WF-V4/TP-V2 (D-K3-85) aufgetreten. Ohne normative Regel
  wiederholt sich der Fehlschluss „abgelehnt = überarbeitungspflichtiges Modell".
- **Was Spec 27 festlegt:** Klassen-Entscheidung ANT-A (Substanz) / ANT-B
  (Mess-Artefakt) vor jeder Überarbeitung, das priorisierte Umkehr-Register
  (Cantus → W-1 → W-2/W-3 → Dropout-Witnesse → M4 → BigVGAN/FlashSR), und die
  Bindungsregeln ANT-1..ANT-9 (keine Schwellwert-Senkung, Paritätspflicht,
  Re-Gate über den Produktions-Pfad, Aktivierung nur §4, ehrliches Scheitern).
- **Konsistenz:** Spec 27 zitiert nur Spec 26 (DRT-4/5/6/7/8/12), die
  SOTA-HANDOVER-Befunde und die Session-Belege; keine neuen Schwellwerte,
  keine neuen Flag-Mechanismen — reiner Prozess-Vertrag.
- **Maschinelle Prüfung:** `audit/spec_integration_scanner.py` (Index-Konsistenz,
  96 Specs) und `scripts/spec_evidence_gate.py` (Bericht im selben Push).

## Seed / 95 %-CI

Nicht anwendbar — Spec 27 definiert keinen statistischen Vertrag, sondern einen
Prozess; die zitierten Messzahlen stammen aus den jeweiligen Gate-Belegen
(Seed 42, deterministisch, dort dokumentiert).

## Maintainer Sign-off

Autonom vorbereitet; die Spec tritt mit Merge auf `main` in Kraft. Die in §3
priorisierten Umkehr-Arbeiten laufen in den bestehenden Ketten bzw. brauchen
GPU-Fenster — keine davon ändert den bit-identischen Bestand (alle Flags AUS).

## Nachtrag 2026-11-22 (gleiche Nacht): Register-Priorität 4, Runde 1

- **dropouts — GATE BESTANDEN:** Detektor-Neutraining (abs-Mel + rms_skip +
  Clean-Gewichtung 2.0) senkt p_clean_holdout 0,76→0,30 (hoAUC 0,9474);
  Re-Gate über den Produktions-Pfad: q_median 0,71, q_anteil 1,0,
  never_worsen 1,0 (vorher 0,9286), musik_mos +1,35 — Beleg
  `2026-11-22_witness_kopf_gate_dropouts.json`. Kein Kopf-Retraining nötig,
  kein Flag-Flip (ANT-5/ANT-7).
- **dropout_oxide / tape_splice — ehrliche Umkehr:** Die Pegel-Instrumente
  verschlechtern diese Detektoren (hoAUC 0,79→0,21 bzw. 0,76→0,53) — die
  ANT-B-Hypothese gilt für dropouts, nicht für diese beiden Klassen (Pegel-
  Kontrast im Musik-Dynamikbereich bzw. durch das 1-s-Pad verdünnt).
  Deterministische Wiederherstellung der alten Instrumenten-Checkpoints
  (Seed 42, Default-Flags, hoAUC wieder 0,7853/0,7645); Ursache ist
  ANT-A-Datenarmut (tape_splice 51 Regionen/32 Trainingspaare, dropout_oxide
  58 trainierbare Trainingsregionen). Nächste Runde nach der
  Daten-Expansions-Kette (ANT-8: ehrliches Scheitern dokumentiert).
- **Messpfad-Reparatur (ANT-B, Commit `a55fa13e`):** rms_skip-Parität über
  Trainer/Gate/Produktion (DRT-4) + CLI-Flags `--lambda-kontrast`/
  `--clean-gewicht` (waren unerreichbar, AttributeError); 75/75 Tests grün.
  Gate-Schwellen unangetastet (ANT-4).
