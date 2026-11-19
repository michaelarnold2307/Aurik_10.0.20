# Strict Vocal Genre Floor (2026-06-04)

- `_compute_vocal_presence_confidence()` brauchte neben Choir und `is_schlager=True` auch einen harten 0.35-Floor fuer exakte definitionsgemaess vokale Genre-Labels.
- Sicherer Fix: nur exakte Labels wie `opera`, `oper`, `aria`, `chanson`, `lied`, `art song`, `crooner`, `vocal`, `vocals only`, `vocal jazz`, `singer songwriter` hart boosten.
- Breite Keyword-Genres wie `folk`/`gospel` absichtlich NICHT in den strikten Floor aufnehmen, sonst entstehen False Positives bei reinem Genre-Keyword ohne PANNs-Hinweis.
- Regressionen in `tests/unit/test_unified_restorer_v3.py`: `TestStrictVocalGenreZeroPanns` plus bestehende `folk`-Negativabsicherung beibehalten.
- Zweite Restluecke sass in `backend/core/vocal_focus_analyzer.py`: `_resolve_vocal_presence_confidence()` hatte noch die aeltere schwache Logik und konnte den UV3-Fix im Runtime-Pfad wieder verduennen.
- VFA muss denselben strikten Genre-Floor nutzen und `vocal_material_prior=True` auch ohne zusatzliche PANNs-Tags direkt auf 0.35 heben; sonst bleiben `vocal_present`/`vqi_gate_active` trotz bekannter Vokalspur deaktiviert.
- Regressionen in `tests/unit/test_vocal_focus_analyzer.py`: Opera-ohne-PANNs und `vocal_material_prior=True` ohne Tags muessen `vqi_gate_active=True` liefern.
- Dritte Restluecke sass in `UnifiedRestorerV3._classify_quality_gate_events()`: die finale Gate-Registry schaute nur auf `panns_singing >= 0.35` und ignorierte `vocal_material_prior`/bereits aktiven VQI-Pfad.
- Fix: `_classify_quality_gate_events(..., vocal_gate_active=...)` nutzt `vocal_material_prior` oder `vfa_result.vqi_gate_active` als gleichwertigen Aktivierungsweg; sonst fehlen trotz berechnetem VQI rote `vocal_contract`-Metadaten im Export-Payload.
- Regression in `tests/unit/test_unified_restorer_v3.py`: prioraktivierter Vocal-Gate-Fall mit `panns_singing=0.0` muss `vqi_below_material_floor`, `singer_identity_below_threshold` und `vqi_below_mode_target` emittieren.
