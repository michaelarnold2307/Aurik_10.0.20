# Witness-Nacharbeit Tiefenanalyse — drei verfehlte Gates (ANT-1-Ursachen-Beleg)

Datum: 2026-11-22 · Schema: aurik.befund/1 · Spec 27 (Ablehnungs-Nacharbeit-Vertrag)

## 1. Die drei Fehlschläge (2026-11-21, `verify_witness_kopf_gate.py`, Produktions-Pfad)

| Klasse | n_q_bewertbar | q_median | never_worsen | gate | musik_mos-Δ |
| --- | --- | --- | --- | --- | --- |
| dropouts | 98/112 | 2,0 (Clip) | 0,9286 (8/112 false) | ❌ | +1,35 |
| tape_splice_artifact | 0/19 | None | 0,8947 (2/19 false) | ❌ | +1,54 |
| dropout_oxide | 0/38 | None | 1,0 | ❌ | +1,15 |

Belege: `2026-11-21_witness_kopf_gate_{dropouts,tape_splice_artifact,dropout_oxide}.json`.

Wichtig: Der musik_mos-Zeuge (zertifizierter Wohlklang-Teacher, Spearman 0,95) ist in
allen drei Klassen **stark positiv** — der Mensch hört die Reparatur vermutlich als
Gewinn. Die Gate-Verfehlungen sind **Mess-/Trainings-Artefakte**, kein Klang-Urteil.
Das ist die ANT-B-Ausgangslage: erst den Messpfad reparieren, dann Substanz.

## 2. Ursachen (Region-Ebene)

**U1 — dropouts, Schutz-Arm (never_worsen 0,9286):** Der Witness-Detektor ist
unkalibriert: `p_clean_holdout = 0,76` (report.json) liegt **über der
Trigger-Schwelle 0,5** des Produktions-DSP. Der Trigger feuert damit auf
clean-Regionen, der Kopf überrestauriert dort, und der Never-worsen-Guard
(P_out ≤ P_in − 0,02) fängt 8/112 Regionen nicht mehr ab. Der Detektor wurde
pegel-blind trainiert (`compute_mel_produktion`, `power_to_db(ref=np.max)` löscht
die absolute Pegel-Dimension) und ohne Clean-Gewichtung — p_clean driftete nach oben.

**U2 — tape_splice_artifact + dropout_oxide, Gewinn-Arm (n_q_bewertbar = 0):**
Der Detektor sieht die Klasse nicht: Kontrast p_schaden − p_clean ≈ **0,01**
(tape_splice: 0,5254/0,5131; dropout_oxide: 0,4514/0,4398) — unter der
Q-Schwelle 0,05. Beide Klassen leben in der **Pegel-Dimension**
(tape_splice +4,6 dB, dropout_oxide +17,4 dB Kontrast laut Instrumenten-Doku),
die `ref=np.max` exakt entfernt. Der Witness wurde auf einer Repräsentation
trainiert, die das diskriminierende Merkmal nicht enthält.
Datenmangel verschärft: tape_splice hat nur 51 Regionen (19 Holdout);
dropout_oxide-Holdout besteht zu 62/100 aus <512-Sample-Mini-Regionen.

**D1 — Wiring-Defekt (Qualitätshebel unerreichbar):**
`scripts/train_witness_detektor.py::main()` referenzierte `args.lambda_kontrast`
und `args.clean_gewicht`, argparse definierte beide **nicht** ⇒ `AttributeError`
bei jedem CLI-Lauf (verifiziert 2026-11-22). Die 2026-11-22 implementierten Hebel
waren toter Code.

**D2 — rms_skip-Paritätsbruch (ANT-3/DRT-4):** Vier Drift-Stellen zwischen
Trainer/Gate und der Produktions-Inferenz `_witness_p`:
1. Detektor-Trainer bildete die abs-Mel auf dem **RAW**-Audio — die Produktion
   misst auf dem RMS-normalisierten 1-s-Fenster.
2. Gate `_p` speiste `rms_global` aus dem **normalisierten** Tensor
   (konstant ≈ −14 dB) statt aus dem Roh-Pegel.
3. Gate-Schutzarm maß p_out_c auf dem RAW-Modul-Ausgang **ohne** eigene-RMS-
   Renormierung gegen eine schaden-RMS-normalisierte Referenz.
4. Kopf-Trainer rief den Witness im Loss ohne `rms_global` auf — mit einem
   rms_skip-Witness ein Dimensions-Crash (192 vs. 193, Head-Linear).

## 3. ANT-Klassifikation

- **ANT-B (Mess-Artefakt, zuerst):** D1, D2 — behoben 2026-11-22, dieser Bericht.
- **ANT-A (Substanz, danach):** U1/U2 — Detektor-Neutraining mit den neuen
  pegel-bewussten Instrumenten (abs-Mel + rms_skip), dropouts zusätzlich
  Clean-Gewichtung, tape_splice/dropout_oxide zusätzlich Kontrast-Hinge (DRT-3);
  danach Kopf-Retraining nur wo nötig und Re-Gate über den Produktions-Pfad (ANT-6).
- Gate-Schwellen (Q_MIN 0,5 / NW 0,95 / NW_TOL 0,02) bleiben **unangetastet** (ANT-4).
- Alle Flags bleiben Default-AUS (ANT-5).

## 4. Code-Fixes (ANT-B) — verifiziert

- `scripts/train_witness_detektor.py`: `--lambda-kontrast`, `--clean-gewicht` CLI-Flags;
  `main(argv)`; `_fenster_1s` + `_mel_fest_abs_normiert` (RMS-normalisierte abs-Mel,
  Produktions-Konvention); `_RMS_ZIEL`-Spiegel (Paritäts-Pin, Import wäre zirkulär).
- `scripts/verify_witness_kopf_gate.py`: `_p(witness, x, raw_db)` — wahrer RAW-Pegel;
  Q-Arm und Schutz-Arm messen im rms_skip-Pfad exakt `_witness_p`-konform
  (clean-eigene RMS, Denormalisierung/Renormierung des Kopf-Ausgangs).
- `scripts/train_witness_kopf.py`: `_lade_split` liefert die RAW-RMS je Region;
  `_db_roh_fenster`/`_db_roh_ausgabe` (20·log10-Konventionen); Witness-Aufrufe in
  Loss/Holdout-Q/Never-worsen/Report führen den wahren Pegel; Alt-Pfad (ohne
  rms_skip) bleibt bit-identisch (kein rms_global-Keyword).
- Tests: 75/75 grün (8 Witness-Testdateien inkl. 8 neuer Pin-Tests:
  rms_skip-Parität Gate/Trainer, CLI-Weiterleitung, Fenster-Konvention,
  Denormalisierungs-Formeln).

## 5. Ausstehend (ANT-A) — läuft

1. Detektor-Neutraining (CPU, nice 19, Seed 42, §G5): dropouts
   (`--abs-mel --rms-skip --clean-gewicht 2.0`), tape_splice_artifact +
   dropout_oxide (`--abs-mel --rms-skip --lambda-kontrast 0.5`).
2. Re-Gate der drei Klassen über den Produktions-Pfad (`--mos`).
3. Kopf-Retraining nur bei weiterem Gate-Verfehlen (mit `--q-ziel 1.0` gegen
   Überrestauration).
4. Umkehr-Register (Spec 27 §9) + SOTA-HANDOVER-Nachtrag mit den Gate-Ergebnissen.

Die Daten-Expansions-Ketten (`kette_dropout_drt8.sh`, `kette_dropouts_followup.sh`)
laufen parallel weiter; ihre Folgeläufe auf dem erweiterten Manifest sind die
kanonischen nächsten Runden (dropouts mit den bewährten neuen Instrumenten,
tape_splice/dropout_oxide mit den alten — siehe §6).

## 6. Zwischenergebnis Runde 1 (2026-11-22, gleiche Nacht)

**dropouts — GATE BESTANDEN** (Beleg `2026-11-22_witness_kopf_gate_dropouts.json`):
Detektor-Neutraining mit abs-Mel + rms_skip + Clean-Gewichtung 2.0 senkt
`p_clean_holdout` von 0,76 auf **0,30** (hoAUC 0,9474) — der Trigger feuert nicht
mehr auf clean. Gate: **q_median 0,71, q_anteil 1,0, never_worsen 1,0** (vorher
0,9286), musik_mos +1,35, alle über JND. Kein Kopf-Retraining nötig — der
bestehende Kopf besteht mit dem kalibrierten Witness. Flag bleibt Default-AUS
(ANT-5); die §4-Freigabe bleibt menschlich.

**tape_splice_artifact / dropout_oxide — weiter verfehlt, ehrliche Umkehr (ANT-A):**
Die Pegel-Instrumente **verschlechtern** diese beiden Detektoren (hoAUC
0,76→0,53 bzw. 0,79→0,21; Richtung verkehrt) — der Pegel-Kontrast trägt in der
1-s-Fenster-Domäne bei diesen Klassen nicht (tape_splice +4,6 dB liegt im
Musik-Dynamikbereich; dropout_oxide-Holdout besteht zu 62/100 aus
<512-Sample-Mini-Regionen, deren Pegel durch das 1-s-Pad verdünnt wird).
Umkehr gemäß ANT: die alten Instrumenten-Checkpoints werden deterministisch
wiederhergestellt (Seed 42, Default-Flags) — sie sind der bessere Zwischenstand.
Ursache ist Datenarmut: tape_splice 51 Regionen (32 Trainingspaare),
dropout_oxide 58 trainierbare Trainingsregionen. Die Expansions-Kette läuft;
mit dem erweiterten Manifest ist die nächste ANT-A-Runde daten-gedeckt.
Gate-Schwellen unangetastet (ANT-4), kein Flag-Flip (ANT-5).

## 7. Zeile-7-Maximalausbau dropout_head_contact (2026-11-22, Nacht)

Ausgang: q_med 2,0 (Clip = Überrestauration), Witness schwach (hoAUC 0,72,
Kontrast 0,03), nur 15 trainierbare Trainingsregionen. Runde 1 (Pegel-Instrumente)
negativ (hoAUC 0,40) — deterministische Umkehr. **Runde 2 — Transfer-Warmstart**
(neues Flag `--init-von`, Commit: warm-start aus dem dropouts-Witness der
verwandten Defekt-Familie): **Durchbruch** — der Transfer-Witness erreicht
**hoAUC 0,9387, p_schaden 0,92 / p_clean 0,26 (Kontrast 0,66)** und besteht das
Detektor-Gate. Der Kopf brauchte ein Retraining gegen den neuen Witness: der
`--nw-min`-Filter blockierte (Rohkopf-NW 0,36 — der Filter modelliert den
Produktions-Trigger nicht; Produktions-NW war 1,0), deshalb R2b mit
Q-Ziel-1.0-Auswahl ohne NW-Filter; das Gate misst den Produktions-Schutzarm
unverändert. **Ergebnis (2026-11-23): GATE BESTANDEN — q_median 0,58,
q≥0,5 = 0,93, never_worsen 1,0, musik_mos +1,30** — Beleg
`2026-11-23_witness_kopf_gate_dropout_head_contact.json`. Die Überrestauration
(q 2,0 → 0,58) ist behoben, der Witness-Kontrast von 0,03 auf 0,66 gehoben.
Transfer-Warmstart ist damit das dokumentierte Muster für daten-arme
Witness-Klassen derselben Defekt-Familie.
