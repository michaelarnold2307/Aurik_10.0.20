# Hör-Batch — drei Entscheidungen in einer Sitzung (§4, menschlich)

> **Zweck:** Die drei wartenden Hör-Entscheidungen (Declip-Sign-off,
> MUSHRA n≥30, W-1) in **einer** Sitzung abarbeiten. Alle Vorbereitungen
> sind erledigt und belegt; die Infrastruktur ist sitzungsfest
> (D-K3-93). Stand: 2026-11-14.

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

## Empfohlene Reihenfolge

1. **Declip** zuerst — fertig gemessen, sofort aktivierbar.
2. **MUSHRA** als zweiter Block (Kalibrierung für alles Folgende).
3. **W-1** als Abschluss (Entscheidung ohne Eile; Flag bleibt bis dahin AUS).

## Was danach automatisch weitergeht

Mit Declip-Sign-off: erster klangwirksamer Modell-Flip im Produktionspfad.
Mit MUSHRA n≥30: D-K4-4/D-K4-11 entblockt. Danach: M9-Neulauf auf den
2.000 expandierten Paaren (erzeugt, Gate-Werkzeug vorhanden) und die
Witness-gestützte Q-Neubuchung der Q≈0-Klassen (6/6 Detektoren
Gate-bestanden, report-only verdrahtet).
