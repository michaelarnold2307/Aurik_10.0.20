# Hör-Batch — fünf Entscheidungsblöcke in möglichst wenigen Sitzungen (§4, menschlich)

> **Zweck:** Die wartenden Hör-Entscheidungen (Declip-Sign-off,
> MUSHRA n≥30, W-1, Witness-Kanal-Batch, Stereo) in möglichst **wenigen**
> Sitzungen abarbeiten. Alle Vorbereitungen sind erledigt und belegt; die
> Infrastruktur ist sitzungsfest (D-K3-93). **Update 2026-11-18:** Declip-Hörpaket
> NEU GERENDERT (CPU-only via AURIK_FORCE_CPU=1, 3 Paare, p25-rel +0,00 % je Paar
> erneut gemessen, Schlüssel getrennt) — die Pfade in diesem Brief sind aktuell.
> **Update 2026-11-19:** + 3 neue Witness-Kanäle (groove_echo-mos,
> dynamic_compression_excess, dropout_head_contact) — §4-Pakete frisch
> (`2026-11-19_hoerproben_witness_kopf.json`), Belege im Abschnitt „Entscheidung 4“.

## Entscheidung 1 — Musik-Declip-Kopf (größter Wohlklang-Hebel)

**Belege (objektiv):** MUSDB leak-frei 4/10 besser / 0/10 schlechter;
k398-LOO 3/3 (Boogie +15,6 dB SDR). Hörpaare in
`study_stimuli/hoerproben/declip_produktionspfad/` (3 blinde A/B, p25-gematcht).

```bash
cd "/media/michael/Software 4TB/Aurik_Standalone"
# 1. Paare anhören (Schlüssel erst nach dem Urteil öffnen)
# 2. Urteil als aurik.ab_stats/1 festhalten
# 3. Nur bei signifikantem Vorteil:
.venv_aurik/bin/python -B scripts/flag_freigabe.py \
  --flag use_aspade_declipper \
  --verdikt <beleg.json> --signoff <signoff.md> --commit
```

Detail-Ablauf inkl. Rollback und Ablehnung: `docs/guides/DECLIP_SIGNOFF_RUNBOOK.md`.

## Entscheidung 2 — MUSHRA n≥30 (Norm-Kalibrierung)

Protokoll §10 + Builder fertig (`scripts/erstelle_hoerproben.py`,
`scripts/ohr_panel.py`). Entriegelt D-K4-4 (F8 UTMOSv2-Musik-MOS),
D-K4-11 (F15 Preference-Orakel) und die absolute MOS-Kalibrierung
(KP-12) — der „menschliche Multiplikator“ für alle weiteren Gates.

## Entscheidung 3 — W-1 (§7-Fall, Hörordnung)

W-1 ist die einzige Wohlklang-Stufe mit widersprüchlichen Objektiv-Belegen
(HF-Gain-Beschluss BESTÄTIGT vs. musik_mos-Beschluss ABGELEHNT,
never_worsen=False) ⇒ die Hör-Instanz entscheidet (§7). B3-Akte:
`docs/reports/current/b3_objektivbeschluss_W-1.json`.

## Entscheidung 4 — Witness-Kanal-Batch (12 Kanäle, defekt-spezifisch)

**Belege (objektiv):** alle 12 Kanäle Gate-bestanden (`verify_witness_kopf_gate.py`);
A/B-Paare in `study_stimuli/hoerproben/witness_kopf/<klasse>/probeNN_*.wav`
(nur wirksame Regionen, A≠B verifiziert).

- **9 Alt-Kanäle (2026-11-15):** groove_echo, tape_head_level_dip, clicks,
  compression_artifacts, tape_head_clog, phase_rotation, sibilance,
  sticky_shed_residue, jitter_artifacts, transport_bump.
- **3 Neu-Kanäle (2026-11-19, `--lambda-mos`):** groove_echo (mos-Variante,
  q 0,997, musik_mos-Anteil 1,0), dynamic_compression_excess (q 1,186),
  dropout_head_contact (q 2,0 = Clip-Obergrenze + Detektor-AUC 0,72 —
  ehrlich markiert; Urteil bleibt menschlich).

```bash
cd "/media/michael/Software 4TB/Aurik_Standalone"
# Je Kanal: Paare anhören → Verdikt als aurik.ab_stats/1 festhalten
# (hoer_entscheidung.py); nur bei signifikantem Vorteil:
.venv_aurik/bin/python -B scripts/flag_freigabe.py \
  --flag use_witness_kopf_<klasse> \
  --verdikt <beleg.json> --signoff <signoff.md> --commit
```

## Entscheidung 5 — Stereo-Sinn-Beweis (Balance-Richtungs-Fix, §III.15)

**Belege (objektiv):** 2/3 `balance_besser` (ANiMAL −3,50→−0,15 dB,
Alexander Ross +4,55→0,00 dB); Snowmine-Befund offen (phase_15 dort nie
geplant — Plan/Defekt-Ranking). Hörpaare:
`study_stimuli/hoerproben/stereo_sinnbeweis/` (automatisch gerendert).
Ziel: Bestätigung, dass der Richtungs-Fix hörbar korrekt ist — Voraussetzung
für die GESICHERT-Einstufung der Zeile „Stereo-Imaging/Azimuth“.

## Empfohlene Reihenfolge

1. **Declip** zuerst — fertig gemessen, sofort aktivierbar.
2. **Witness-Batch** (12 Kanäle) als zweiter Block — größter einzelner
   Reifesprung (12 freischaltbare, defekt-spezifische Reparaturen).
3. **Stereo** als dritter Block (1 Urteil, schließt die §III.15-Zeile).
4. **MUSHRA** als vierter Block (Kalibrierung für alles Folgende).
5. **W-1** als Abschluss (Entscheidung ohne Eile; Flag bleibt bis dahin AUS).

## Was danach automatisch weitergeht

Mit Declip-Sign-off: erster klangwirksamer Modell-Flip im Produktionspfad.
Mit MUSHRA n≥30: D-K4-4/D-K4-11 entblockt. Danach: M9-Neulauf auf den
2.000 expandierten Paaren (erzeugt, Gate-Werkzeug vorhanden) und die
Witness-gestützte Q-Neubuchung der Q≈0-Klassen (6/6 Detektoren
Gate-bestanden, report-only verdrahtet).
