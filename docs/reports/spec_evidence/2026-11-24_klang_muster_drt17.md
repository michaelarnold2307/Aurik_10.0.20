# Spec-Evidenzbericht — DRT-17 Klang-Muster + DRT-8 mos-Arm-Präzisierung

- **Datum:** 2026-11-24
- **Geänderte Spec:** `.github/specs/26_defekt_reparatur_trainingsvertrag.md`
  (DRT-8: Produktions-Semantik-Klausel im musik_mos-Arm; DRT-17 neu:
  Sechs Klang-Muster).
- **Mitgeänderte Normen:** `.github/instructions/hoerordnung.instructions.md`
  (neuer Abschnitt §8c), `.github/GEBOTE.md` (Kategorie XXVII, §G192–§G197).

## 1. Zweck der Änderung

Die sechs in der Kalibrierungs-Nacht 2026-11-24 bewiesenen Klang-Muster
(Clean-Referenz-Kalibrierung, Detektor-Zurückhaltung, Wahrnehmungs-Lehrer,
Audibility, Determinismus, Daten-Authentizität) werden als operationale
Verdichtung der Hörordnung-Ebenen 1–4 normativ verankert. Sie ändern keine
Ebene, keine Schwelle und kein Gate — sie legen fest, WIE die bestehenden
Ziele erreicht werden.

## 2. Beleg-Basis (gemessen, nicht behauptet)

| Muster | Mess-Beleg |
| --- | --- |
| Clean-Referenz-Kalibrierung | ts_artifact q 2,0→1,0893 (λ-nor 1,0); jitter 0,78→1,0063 (Q-Ziel 1,0) — `2026-11-24_witness_kopf_gate_{tape_splice_artifact,jitter_artifacts}.json` |
| Detektor-Zurückhaltung | tdhl-Region 4: Witness P(schaden)=0,0015 ⇒ Produktionsmodul Pass-Through; roher Kopf −0,091 mos (Diagnose `/tmp/tdhl_region4_diag.log`); dropouts-Witness hoAUC 1,000 |
| Wahrnehmungs-Lehrer | dropouts mos_anteil 0,965→1,0, mos_min −0,306→−0,018 bei q 1,0027 (λ-mos 0,5) — `2026-11-24_witness_kopf_gate_dropouts.json` |
| Audibility | normativ verankert in §G13 (copilot-instructions.md) und Hörordnung Ebene 2 — unverändert übernommen |
| Determinismus | D-K3-GoNoGo-Befund `2026-11-23_determinismus_go_nogo_befund.json` (Lauf 1: dl +0,3646, Lauf 2: bit-identischer Pass-Through) |
| Daten-Authentizität | dhc: synthetische vinyl/dropout-Rezepte ⇒ 0 Scanner-Treffer (Probe `/tmp/dhc_regionen_probe.txt`); 112 reale Regionen + clean-gewicht-Witness ⇒ q 0,5849→1,0732 — `2026-11-24_witness_kopf_gate_dropout_head_contact.json` |

## 3. Gate-/Schwellen-Auswirkungen

Keine. Es wurden keine Gate-Kriterien, keine §0h-Werte und keine
Trennung-Schwellen geändert (DRT-5). Die einzige Mess-Korrektur betrifft den
Report-only-mos-Arm des Witness-Kopf-Gates: er misst jetzt die
Produktions-Ausgabe (Trigger + Never-worsen-Guard) statt des rohen Kopfs —
die Produktionssemantik, die der Schutz-Arm seit 2026-11-15 verwendet.
29/29 zugehörige Unit-Tests grün (`test_witness_kopf_gate.py`,
`test_witness_mos_loss.py`).

## 4. Nicht anwendbare Messungen

- Statistik/Konfidenzintervalle: nicht anwendbar — es wurden keine neuen
  Verteilungsvergleiche eingeführt, sondern bestehende, einzeln belegte
  Gate-Messungen normativ verankert.
- Panel/Hörtest: nicht anwendbar — kein Aktivierungs-/Flag-Flip; §4 bleibt
  unverändert menschlich (Hörordnung §7).

## 5. Maintainer Sign-off

Autonom vorbereitet; die Spec tritt mit Merge auf `main` in Kraft. Die
Norm-Erweiterung ändert kein Produktionsverhalten — alle 17 Witness-Kanäle
bleiben Default-AUS.
