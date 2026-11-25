# Spec-Evidenzbericht: Wohlklang-Maximierungs-Pflicht + Verlust-Negation (2026-11-25)

## 1. Anlass und Beleg

**Muster-Befund** (`docs/reports/current/2026-11-25_muster_wohlklang_ordnung_rollback.md`):
die Standard-Verarbeitung verstößt auf bestimmtem Material gegen die
lexikografische Wohlklang-Ordnung (brillanz/waerme/transparenz gewinnen auf
Kosten der Stufe-1-Ziele authentizitaet/emotionalitaet/timbre_authentizitaet)
und korrigiert per **Voll-Rollback** (blend_factor 0.0 = Eingang = Ausgang).
Gemessen in einem Beleg-Log (`2026-11-05_b3_w2w3_nachmessung.log`):
**1.797 §Ebene-3-Verletzungen, 24 Pipeline-Ende-Enforcements, alle ROLLBACK
mit blend_factor 0.0** — die gesamte Restaurations-Arbeit wird verworfen.

**Auftrag des Auftraggebers (2026-11-25):** „Der Wohlklang-Verlust muss
negiert werden. Es muss normativ verankert werden, dass jede Programmier-KI
stets das Maximum des Wohlklangs für das menschliche Gehör aus allen Modulen
von Aurik herausholt."

## 2. Normative Änderungen

| Datei | Änderung |
| --- | --- |
| `.github/copilot-instructions.md` | **§G14 (neu, Kategorie I): Wohlklang-Maximierungs-Pflicht** — jede Programmier-KI holt bei jeder Änderung das Maximum des Wohlklangs aus ALLEN Modulen heraus; Wohlklang-Verlust wird negiert (Voll-Rollback ≠ Normalzustand), Reserven heben, Leitplanken (§G5, Never-worsen, Hörordnung Ebene 1–3, §4). **§G5 (erweitert):** Last-Unabhängigkeit — Phasen-Teilnahme-/Speicher-Entscheidungen nur aus Maschinen-Konstanten + Input; Live-Notbremsen nur als Notfall. |
| `.github/instructions/hoerordnung.instructions.md` | **§5a (neu): Konstruktive Ordnungs-Konformität** — konstruktiv statt korrektiv; Voll-Rollback nur als Notfall; Verletzungs-/Rollback-Zähler als Restauration-Verlust; maschinelles Erfolgskriterium Zähler → 0 bei unangetasteten Stufe-1-Deltas; Blend-Hygiene (kleinster ausreichender blend_factor, nie 0.0 als Default). |
| `.github/GEBOTE.md` | **§G198 (neu, Kategorie XXVII):** Wohlklang-Maximierungs-Pflicht (entspricht §G14 (copilot-instructions.md)); **§G196 erweitert:** last-unabhängige Phasen-RAM-Planung als Implementierung der Determinismus-Regel. Kategorie-Header §G192–§G198. |
| `.github/ID_REGISTRY.md` | `§G1[0-3]` → `§G1[0-4]` (copilot-instructions.md), Ambiguity-Hinweis §G1–§G14. |
| `AGENTS.md` | §3 Schnell-Referenz: neue Spitzen-Bullet „Wohlklang-Maximierungs-Pflicht (§G14)". |
| `.github/SOTA_DEFICIT_REGISTER.md` | **D-K3-99 (neu):** §P1-1-Ordnungs-Rollback als geführtes Defizit (in-arbeit, Fix-Plan mit Sweep → Rekalibrierung → §4). |
| `.github/GEBOTE_INTEGRATION_MATRIX.md` | Regeneriert (`scripts/gen_integration_matrix.py`, 244 IDs). |
| `reports/spec_drift_baseline.json` | Neu gebaselined (bewusste Änderung, kein Drift). |

## 3. Maschinelle Erfolgskriterien (neu normativ)

1. Verletzungs- und Voll-Rollback-Zähler gehen auf dem Schadens-Korpus
   gegen **0** (`scripts/wohlklang_ordnungs_sweep.py`, Default-Pfad),
   ohne dass Stufe-1-Deltas sinken (Never-worsen).
2. Phasen-Teilnahme ist last-unabhängig (Determinismus-Probe:
   Drift-Bisektion `scripts/drift_bisektion_probe.py`).
3. Kein korrektiver Voll-Rollback als Default-Lösung (Hörordnung §5a
   Blend-Hygiene).

## 4. Verifikation (Status zum Berichtszeitpunkt)

- `scripts/spec_drift_check.py` → exit 0 (Baseline nachgeführt).
- `scripts/gen_integration_matrix.py` → exit 0, 244 IDs.
- Nachfolgend in diesem Push: `id_registry_check`, `sota_deficit_gate`,
  `compliance_check` — Ergebnisse im Commit-Body dokumentiert.

## 5. Statistik/Begründung nicht anwendbar

Klang-/Paritäts-Statistik ist hier nicht anwendbar: Dieser Bericht
verankert Normen und Messpflichten; die Klangmessung folgt im
Ordnungs-Konformitäts-Sweep nach Abschluss der laufenden DRT-12-Kampagne
(kein Parallel-Betrieb, um die Kampagnen-Kinder nicht über die
Live-Notbremse zu stören).
