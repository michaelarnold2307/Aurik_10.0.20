# Chain-Aware Oracle Worldclass Audit (2026-05-20)

## Ziel

Verifiziert die Umstellung auf transfer-chain-aware Strength-Oracles als direkte Weltklasse-Invariante.

## Scope

- `backend/core/dsp/phase_strength_oracle.py`
- `backend/core/unified_restorer_v3.py`
- `tests/unit/test_phase_strength_oracle.py`
- Regelwerk: `policy/dsp_policy_contracts_overview.yaml`

## Normative Kriterien

1. Oracle berechnet `chain_factor` deterministisch aus `material_key`, `transfer_chain`, `chain_confidence`.
2. `chain_factor` beeinflusst sowohl Driver als auch Hard-Caps.
3. UV3 uebergibt `transfer_chain` und Confidence direkt an den Oracle-Resolver.
4. Oracle-Profil exportiert `hard_caps.chain_factor`, `chain_depth`, `chain_confidence`.
5. Mehrstufige Ketten sind konservativer als Einzeltraeger bei gleicher Defektlast.

## Testnachweis

- `tests/unit/test_phase_strength_oracle.py::test_oracle_chain_factor_reduces_strength_for_multistage_chain`
- `tests/unit/test_phase_strength_oracle.py::test_uv3_runtime_context_applies_o8_cap_for_spectral_family`
- `tests/unit/test_phase_strength_oracle.py::test_uv3_runtime_context_applies_o10_cap_for_output_family`
- `tests/unit/test_phase_strength_oracle.py::test_uv3_runtime_context_keeps_explicit_strength_for_o10_output_phase`

## Ergebnis

- Status: PASS
- Befund: Direkte Kettenkonditionierung ist produktiv verdrahtet und testgesichert.
- Rest-Risiko: Niedrig (non-blocking Fallback bleibt aktiv).
