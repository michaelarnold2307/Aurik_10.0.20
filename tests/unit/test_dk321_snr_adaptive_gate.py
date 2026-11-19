"""D-K3-21: SNR-adaptive CAUSE_PARAMS hinter explizitem Gate (Default AUS).

Die SNR-Skalierung (``clip(25/max(5,SNR), 0.5, 1.5)`` auf ``strength``/``boost``)
war bis 10.12.9 ein toter Zweig: ``_last_snr_estimate`` wurde nirgends gesetzt.
Gepinnt wird der neue Vertrag:

  - Default: Gate ``_SNR_ADAPTIVE_ENABLED`` ist AUS → bit-identisches Verhalten,
    auch wenn eine SNR-Schätzung gesetzt ist.
  - Gate AN: Skalierung greift exakt mit dem dokumentierten Faktor.
  - ``set_snr_estimate()`` normalisiert deterministisch (§G5 (copilot-instructions.md))
    und ist die explizite Datenquelle.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core.causal_defect_reasoner import CausalDefectReasoner


def _strength_keys(plan) -> dict[str, float]:  # type: ignore[no-untyped-def]
    return {
        k: float(v)
        for k, v in plan.phase_parameters.items()
        if ("strength" in k or "boost" in k) and isinstance(v, (int, float))
    }


def test_snr_scaling_disabled_by_default() -> None:
    r = CausalDefectReasoner()
    assert r._SNR_ADAPTIVE_ENABLED is False
    assert r._last_snr_estimate is None

    # Ohne SNR-Schätzung: Referenzplan.
    plan_before = r.reason({"tape_hiss": 0.9}, material="tape")
    base = _strength_keys(plan_before)
    assert base, "Fixture-Vertrag: tape_hiss muss noise_reduction_strength liefern"

    # Mit gesetzter SNR-Schätzung: Gate AUS ⇒ unverändert.
    r.set_snr_estimate(10.0)
    plan_after = r.reason({"tape_hiss": 0.9}, material="tape")
    assert _strength_keys(plan_after) == base


def test_snr_scaling_applies_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    r_off = CausalDefectReasoner()
    r_off.set_snr_estimate(10.0)
    off = _strength_keys(r_off.reason({"tape_hiss": 0.9}, material="tape"))

    monkeypatch.setattr(CausalDefectReasoner, "_SNR_ADAPTIVE_ENABLED", True)
    r_on = CausalDefectReasoner()
    r_on.set_snr_estimate(10.0)
    on = _strength_keys(r_on.reason({"tape_hiss": 0.9}, material="tape"))

    assert off and on
    scale = float(np.clip(25.0 / 10.0, 0.5, 1.5))
    assert scale == 1.5
    for key, value in off.items():
        assert on[key] == pytest.approx(value * scale, rel=1e-9), f"{key} nicht mit ×{scale} skaliert"


def test_set_snr_estimate_normalizes_deterministically() -> None:
    r = CausalDefectReasoner()
    assert r.set_snr_estimate(None) is None
    assert r.set_snr_estimate(float("nan")) is None
    assert r.set_snr_estimate(-3.0) is None
    assert r.set_snr_estimate(0.0) is None
    assert r.set_snr_estimate(12.5) == 12.5
    assert r.set_snr_estimate(500.0) == 80.0
    assert r._last_snr_estimate == 80.0
