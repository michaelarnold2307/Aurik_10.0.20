# Kanalsatz-A/B vollständig (45/45) — Analyse und DRT-12-Einordnung

**Datum:** 2026-11-23 · **Beleg:** `docs/reports/current/2026-11-19_kanalsatz_ab_voll.json`
(schema `aurik.kanalsatz_ab/1`, datum 2026-11-21, 45 Zeilen `ok`, seed 42,
subprozess-isoliert, Bestand-Beleg `data/schadens_proben/bewertung.json`).

## Kanal-Satz des Laufs

12 Witness-Flags: groove_echo, tape_head_level_dip, clicks,
compression_artifacts, jitter_artifacts, tape_head_clog, phase_rotation,
sibilance, sticky_shed_residue, transport_bump, dynamic_compression_excess,
dropout_head_contact (alle Default-AUS im Produktionspfad, nur Messprozess AN).
**Nicht enthalten:** hum, wow, dropouts, dropout_oxide, tape_splice_artifact —
deren §4-Pakete (2026-11-23) stammen aus einem Kanal-Satz-Stand VOR der
Spec-27-Nacharbeit 2026-11-21/23.

## Aggregat (45 identische Paare, Kanalsatz AN vs. Bestand AUS)

| Metrik | Mittel | Median | Min | Max | besser (>JND) | schlechter (>JND) | neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| musik_mos (Δ zu Bestand) | +0,1114 | +0,0035 | −0,1613 | +2,4537 | 9 | 17 | 19 |
| defektlast (Δ zu Bestand) | −0,0363 | −0,0004 | −0,9866 | +0,4046 | 6 | 4 | 35 |

15 Rezepte × 3 Titel (wowflutter/hum/lautheit/raum/kassette/verzerrung/
clipping/mp3_128/kammfilter/kompression/vinyl/dropout/pitch_drift/clicks/
amp_drift).

## Einordnung (Regel 6/DRT-12, ehrlich)

1. **Der Kanal-Satz gewinnt den Ganz-Titel-A/B auf diesem Stand nicht:**
   musik_mos 17/45 schlechter über JND vs. 9/45 besser; defektlast weitgehend
   neutral (35/45). Median ≈ 0 ⇒ im Mittel sub-JND, mit einzelnen großen
   Gewinnen (max +2,45 mos) und moderaten Verlusten (min −0,16).
2. **§4-Batch bleibt blockiert** (Regel 6: „Nur Kanäle, die den A/B gewinnen,
   kommen in den §4-Batch“) — und zwar für DIESEN Satz auf DIESEM Messstand
   (2026-11-19/21, vor der Nacharbeit 2026-11-21/23, die u. a. die
   Schutz-Arm-Konvention für ALLE Witnesses repariert hat, Commit `406eb527`).
3. **Nächster Schritt:** Kanalsatz-A/B-Wiederholung auf dem aktualisierten
   Kanal-Satz (inkl. hum/wow/dropouts-Familie, transport_bump-NOR-Kopf) mit
   identischen Paaren; erst bei A/B-Gewinn §4-Batch. Kein Status-/Flag-Flip
   aus dieser Analyse.
