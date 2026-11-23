# Spec 27: Ablehnungs-Nacharbeit-Vertrag (ANT)

> **Status: Aktiv — kanonisch über `00_SPEC_INDEX.md`, maschinell geprüft durch
> `audit/spec_integration_scanner.py`.**
> Dieser Vertrag macht die **Ablehnung eines Gate-Verdikts zum Startpunkt einer
> belegten SOTA-Überarbeitung** statt zum Endzustand. Er ergänzt Spec 26
> (DRT-1..DRT-16): Spec 26 definiert die Messung, Spec 27 den Umgang mit dem
> gemessenen „nein".

## 1. Zweck

Jede Ablehnung (B3-Beschluss, Witness-Kopf-Gate, Detektor-Gate, D3-Gate,
Cantus-Gate, …) trägt eine der beiden folgenden Ursachenklassen. Die Klasse
bestimmt den Überarbeitungsweg. **Eine Ablehnung wird nie durch Schwellwert-
Senkung, Mess-Anpassung oder phasen-individuelle Ausnahme umgekehrt** (DRT-5);
sie wird durch Ursachen-Arbeit am Modell, an den Daten oder am Messpfad
umgekehrt und anschließend über denselben Produktions-Pfad neu gemessen.

## 2. Die zwei Ablehnungs-Klassen

### ANT-A — Substanz-Ablehnung (das Modell ist wirklich nicht gut genug)

Erkennung: Der Messpfad ist verifiziert wirksam (D-K3-85-Klasse „Schalter hat
Lesestelle", an/aus-Deltas über Lauf-Rauschen), und der Kandidat verfehlt die
Gate-Bedingung auf Substanz.

Beispiele (belegt):

- **Cantus v3** (2026-11-19): never_worsen verfehlt auf sauberem Gesang
  (mos 4,71→4,43) — Substanz, weil der Pfad wirksam gemessen wurde. Zugleich
  belegter Nutzen auf degradiertem Gesang (AvaLuna +0,74 MOS, 3,51→4,25).
- **W-3-Neumessung** (2026-11-11): 2/6 Vorteile trotz pfad_wirksam ✓ und
  never_worsen ✓ — die 9 ML→DSP-Fallbacks sind ein Produktions-Defekt der
  Tier-0-Erreichbarkeit, kein Mess-Artefakt.
- **dropouts-Kopf** (2026-11-21): q −0,26 (Kopf regressiert den Holdout),
  später q 2,0 (Clip = Überrestauration) — Substanz des Kopfs.

### ANT-B — Mess-Artefakt-Ablehnung (der Messpfad hat nie richtig gemessen)

Erkennung: an/aus-Deltas ≤ Lauf-Rauschen (≤ 1e-4) ODER widersprüchliche
Objektiv-Belege ODER der Schalter hat keine wirksame Lesestelle
(D-K3-85-Klasse) ODER die Metrik ist für die Defekt-Dimension blind.

Beispiele (belegt):

- **W-2** (2026-11-11): an/aus-Deltas ≤ 1e-4 = Lauf-Rauschen — es wurde nie
  das Modell gemessen; die Ablehnung ist ein Qualitätsurteil über den
  Messpfad, nicht über SCNet.
- **W-3-Erstbeschluss** (2026-11-05): No-Op-Artefakt durch Dauer-Achsen-Bug —
  nachweislich ungültig (2026-11-06 korrigiert).
- **W-1**: never_worsen verfehlt UND widersprüchliche Belege (HF-Gain
  BESTÄTIGT vs. musik_mos ABGELEHNT) — der mono-basierte musik_mos-Teacher
  honoriert Stereo-/HF-Verbesserungen nicht (Snowmine-Muster 2026-11-20:
  mono-blinde Metrik verwarf einen robusten Stereo-Gewinn).
- **WF-V4 / TP-V2** (2026-11-08/09): B3-Ablehnungen waren Mess-Artefakte
  (nie erzeugtes `.onnx` geladen bzw. Seed-Gewichte statt Checkpoint) —
  D-K3-85-Klasse, nachweislich keine Qualitätsurteile.
- **Witness-Blindheit dropout_oxide/tape_splice** (2026-11-22): die
  max-normalisierte Mel löscht die Pegel-Dimension, in der die Defekte leben
  (Kontrast +17,4/+4,6 dB im Audio, 0,01 in der Mel) — die Ablehnung ist ein
  Repräsentations-Artefakt, kein Urteil über die Reparierbarkeit.

## 3. Umkehr-Register (priorisiert nach Wohlklang-Wert fürs menschliche Ohr)

| Priorität | Kandidat | Klasse | Belegter Nutzen | Überarbeitungs-Hebel |
|---|---|---|---|---|
| 1 | Cantus (Gesang) | A | +0,74 MOS auf degradiertem Gesang | Clean-seitige Never-worsen-Regularisierung im Domain-Adapt |
| 2 | W-1 (Brillanz/Air) | B | HF-Gain-Beschluss bestätigt; entspricht §4-Wunschrichtung „heller UND weniger Rauschen" | Stereo-bewusste Messung + §7-Hör-Entscheidung |
| 3 | W-2/W-3 (Musik-Denoise) | B/A | „Decrackle ist nach Denoise der größte Register-Hebel" (SOTA-HANDOVER) | W-2: Schalter-Lesestelle verifizieren, dann echt messen; W-3: 9 Fallbacks beheben |
| 4 | Witness dropouts / dropout_oxide / tape_splice | B | 8–17 dB Pegel-Kontrast der Defekte; Dropout-Reparatur = unmittelbarer Hörgewinn auf Bandmaterial | **Runde 1 (2026-11-22): dropouts GATE BESTANDEN** (abs-Mel + rms_skip + Clean-Gewichtung 2.0: p_clean 0,76→0,30; q_median 0,71, never_worsen 1,0, mos +1,35; Beleg `2026-11-22_witness_kopf_gate_dropouts.json`); dropout_oxide/tape_splice: Pegel-Instrumente negativ (hoAUC 0,79→0,21 bzw. 0,76→0,53) ⇒ deterministische Umkehr auf alte Instrumente, ANT-A-Datenarmut (51 bzw. 38 verwertbare Regionen) — nächste Runde nach Daten-Expansions-Kette (Befund `docs/reports/current/2026-11-22_witness_nacharbeit_tiefenanalyse.md`) |
| 5 | M4-Router | A | indirekt (Routing-Präzision) | nur nach belegtem Router-Fehlerbild |
| 6 | BigVGAN-F3 / FlashSR-Finetunes | A | inkrementeller HF-Gewinn über funktionierender Basis | Produktions-Mel-Training (BigVGAN) bzw. HF-Residual (FlashSR) |
| 7 | Witness dropout_head_contact | A | q_med 2,0 = Clip-Obergrenze (Überrestauration, dropouts-Muster) bei gleichzeitig schwachem Witness (hoAUC 0,72 < 0,80); mos +1,30 | **GELÖST 2026-11-23 (Maximalausbau):** Runde 1 negativ (Pegel-Instrumente: hoAUC 0,40, deterministische Umkehr). Runde 2 = Transfer-Warmstart (`--init-von dropouts`, gleiche Defekt-Familie) + abs-Mel + rms_skip + Clean-Gewichtung 2.0 + Kontrast-Hinge 0,5 ⇒ **Witness hoAUC 0,9387, Kontrast 0,66** (Detektor-Gate bestanden); Kopf-Retraining mit Q-Ziel-1.0-Auswahl (der NW-Filter blockierte fälschlich — er modelliert den Produktions-Trigger nicht; Produktions-NW war 1,0) ⇒ **Re-Gate BESTANDEN: q_median 0,58, q≥0,5 = 0,93, never_worsen 1,0, musik_mos +1,30** — Beleg `2026-11-23_witness_kopf_gate_dropout_head_contact.json`. Überrestauration behoben, Flag bleibt Default-AUS (ANT-5), §4-Freigabe menschlich |
| 8 | Witness transport_bump / jitter_artifacts | A | Q-Anteil 0,63/0,70 — Gewinn-Arm auf einem Drittel der Regionen unwirksam (q < 0,5); nw 1,0 | **GELÖST 2026-11-23:** transport_bump-Witness-Kalibrierung (Clean-Gewichtung 2.0, p_clean 0,76→0,63) + Kopf-Retraining mit Q-Ziel 1.0 ⇒ q≥0,5 = 1,0, nw 1,0; der erste Retrain landete trotzdem auf q 2,0 (mean-Q-Proxy 1,20 vs. Gate-Median 2,0) ⇒ **Auswahl-Metrik auf Regionen-Median gestellt** (ANT-3/DRT-4, Commit) ⇒ **q_median 0,82, q≥0,5 = 1,0, nw 1,0, mos +1,29** (Beleg `2026-11-23_witness_kopf_gate_transport_bump.json`). jitter_artifacts-Kopf mit Q-Ziel 1.0: **q_median 0,57→1,03, q≥0,5 = 0,70→1,0, nw 1,0, mos +1,12** (Beleg `2026-11-23_witness_kopf_gate_jitter_artifacts.json`) |
| 9 | Witness sibilance | A | never_worsen 0,9598 — Schutz-Arm nur ~0,01 über der 0,95-Schwelle | **GELÖST 2026-11-23:** Kopf-Retraining mit verstärktem Identitäts-Anker (lambda_id 0,9, NW-Filter 0,97) ⇒ **never_worsen 0,9598→1,0, q 1,09, mos +0,89** (Beleg `2026-11-23_witness_kopf_gate_sibilance.json`) |
| 10 | Witness wow / phase_rotation / dynamic_compression_excess | A | never_worsen 0,984/0,979/0,9885 — Schutz-Arm knapp unter 1,0, aber über der 0,95-Schwelle | Beobachtung ohne Handlungs-Pflicht (Gate bestanden); Runde erst bei erneuter Verfehlung |

## 4. Bindende Regeln (ANT-1..ANT-9)

1. **Keine Umkehr ohne Ursachen-Beleg.** Jede Überarbeitung beginnt mit einer
   Klassen-Entscheidung ANT-A/ANT-B und einem dokumentierten Befund
   (Messzahlen, Log-Verweis, Beleg-JSON).
2. **Mess-Artefakt zuerst.** ANT-B-Verdikte werden vor jeder Modell-Änderung
   durch Messpfad-Reparatur aufgelöst und neu gemessen — nie durch
   Modell-Tuning „um den Fehler herum" trainiert.
3. **Repräsentations-Parität ist Messpflicht** (DRT-4, D-K3-96-Lehre): Training,
   Gate und Produktions-DSP müssen dieselbe Merkmals-Repräsentation nutzen.
   Jede Mel-/Fenster-/Normalisierungs-Änderung ist ein Paritäts-Ereignis mit
   Beleg über alle drei Seiten.
4. **Keine Schwellwert-Senkung** (DRT-5): Gate-Schwellen sind unantastbar;
   überarbeitet wird das Modell, die Daten oder der Messpfad.
5. **Default-AUS während der Überarbeitung:** jeder Kandidat bleibt hinter
   einem bit-identischen Flag (Default AUS); Verdrahtung erzeugt keine
   klangliche Änderung (§G1 (copilot-instructions.md)).
6. **Re-Gate über den Produktions-Pfad** (DRT-7): nach jeder Überarbeitung
   läuft das vollständige Gate erneut — Regionen-Beweis allein ist nie
   hinreichend.
7. **Aktivierung nur über §4:** auch ein bestandenes Gate aktiviert nichts;
   der Flag-Flip bleibt der Hör-Instanz vorbehalten (DRT-6, DRT-8/12).
8. **Ehrliches Scheitern bleibt erlaubt** (DRT-6): führt auch die Überarbeitung
   nicht zum Gate-Erfolg, bleibt der Kanal report-only mit präziser
   Rest-Ursache — nie ein stiller No-Op, nie ein Schein-Erfolg.
9. **Umkehr-Register pflegen:** jede neue Ablehnung wird nach Klassen-Entscheid
   in das Register (§3) aufgenommen; jede gelungene Umkehr wird mit Datum,
   Gate-Zahlen und Beleg dokumentiert.

## 5. Prozess

```
Gate-Verdikt „abgelehnt"
   → Klassen-Entscheidung (ANT-A Substanz / ANT-B Mess-Artefakt) mit Beleg
   → ANT-B: Messpfad reparieren → erneut messen
   → ANT-A: Ursache (Modell/Rezeptur/Daten) beheben
   → Re-Gate über den Produktions-Pfad (DRT-7, DRT-4-Parität)
   → bestanden: Flag bleibt AUS, §4-Hör-Sign-off entscheidet über Aktivierung
   → verfehlt: ehrlicher Report-only-Status mit Rest-Ursache (DRT-6)
```
