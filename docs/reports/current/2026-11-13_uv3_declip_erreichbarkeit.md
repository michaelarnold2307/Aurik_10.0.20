# UV3-Produktionspfad: Declipper-Erreichbarkeit — 6 Schichten gelöst (2026-11-13)

**Rätsel:** `DeclipperPhase` (mit dem Musik-Declip-Kopf hinter
`use_aspade_declipper`) war im UV3-Produktionspfad unerreichbar — auf dem
k398-Clipping-Material wurde die Phase nie ausgeführt, das Flag änderte
nichts am Ergebnis.

## Die sechs Schichten (alle gefixt, Commits feb7c886 + Folge-Commit)

1. **Leichtpfad-Keep-Liste** (Zeile 13246): veraltete Id
   `phase_01_declipping` → `phase_07_declipper`.
2. **Scanner-Blindheit:** Sättigungs-Clipping (±0,90-Plateaus) erhält
   CLIPPING-Severity 0,0 ⇒ Selektionsschwelle `sev > 0,10` griff nie.
   Signal-Trigger (Plateau-Fraktion ≥ 0,85×Peak ≥ 0,005) in `_select_phases`.
3. **PID-Plan-Pfad:** der PhaseInteractionDenker-Plan überspringt die
   UV3-Selektion und enthielt den Declipper nicht → Trigger am Plan-Merge.
4. **Pilot-Rollout-Whitelist** (`_PHASE_STRENGTH_ORACLE_PILOT_PHASES`):
   `phase_07_declipper` aufgenommen.
5. **Resolved-Defekt-Skip** (`_should_skip_resolved_phase`): Signal-Guard
   über `self._clip_fraction_raw` — verifiziert (Probe v10: resolved → False).
6. **§v10.707-Defekt-Absenz-Gate** (`_should_skip_absent_defect_phase`):
   Der eigentliche Skip (Log: `phase_skip_defect_absent`) — Signal-Override:
   bei harter Plateau-Fraktion läuft die Phase trotz Scanner-Absenz.

## Beleg (Probe v11, Boogie-Schlimmstfenster 10 s, MONO (N,), echter `UnifiedRestorerV3`)

- Flag AUS: `phase_07_declipper` in `phases_executed` ✓ (PCHIP-Amtsinhaber)
- Flag AN: Phase läuft mit Challenger-Kaskade ✓
- `delta_off_on_max = 1,86` — das Flag wirkt substanziell auf die
  Produktions-Ausgabe (Never-worsen-Witness akzeptiert auf harten Plateaus)

## Einordnung

- Technische Aktivierungskette geschlossen: Auswahl → Plan → Whitelist →
  Skip-Guards → Ausführung → Flag-Wirkung, alles im Produktionspfad belegt.
- **Kein Flag-Flip.** `use_aspade_declipper` bleibt AUS; der Flip läuft über
  das §4-Verfahren (blindes A/B + Hörordnungs-Sign-off) — die objektiven
  Gate-Belege (leak-freier MUSDB-Lauf 4/10 besser, 0/10 schlechter;
  k398-LOO 3/3 besser) liegen vor.
- Trigger/Override sind bit-identisch für Material ohne harte Plateaus.
