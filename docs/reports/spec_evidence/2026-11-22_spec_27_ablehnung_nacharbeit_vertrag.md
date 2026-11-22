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
