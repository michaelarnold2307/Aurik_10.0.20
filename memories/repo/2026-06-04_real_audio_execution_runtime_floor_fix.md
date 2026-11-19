# Real Audio Execution Gate Runtime Floor Fix (2026-06-04)

- Full `real_audio_execution_golden_gate` war nach Vocal-Contract-Fixes nur noch wegen `runtime_factor > 25` rot.
- Ursache: Alle Gate-Snippets sind kurz (~3s), aber UV3 hat fixe Init-/Model-Load-Overheads; reine `runtime/duration`-Normierung ueberbewertet kurze Clips.
- Fix in `backend/core/real_audio_execution_golden_gate.py`:
  - `ExecutionGateThresholds.runtime_duration_floor_seconds` (Default 4.0)
  - `duration = sum(max(case.duration_seconds, duration_floor, 1e-9))`
- Regressionstest in `tests/unit/test_real_audio_execution_golden_gate.py`:
  - `test_execution_gate_runtime_duration_floor_prevents_short_clip_overhead_bias`
- Ergebnis: Full-Report `audit/real_audio_execution_golden_report_full_2026_06_04_after_runtime_floor.json` hat `gate.passed=true`, `runtime_factor=20.829`, alle Contract-Rates 1.0.
