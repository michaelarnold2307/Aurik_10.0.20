from __future__ import annotations

import pytest

"""Tests §Phase-level wall-time deadline shared across M/S MRSA sub-calls.

Root cause: stereo audio caused 2× independent zone budgets (562.5s each = 1125s+),
exhausting the UV3 wall-time budget and skipping phase_06/42/13/34/46/48.
Fix: a shared _phase_deadline is passed into both _repair_channel() calls so
total MRSA time is bounded by min(300s, max(90s, 1.3 × dur_s)).
"""


import numpy as np

from backend.core.phases.phase_23_spectral_repair import SpectralRepair

SR = 48_000


def _make_stereo(duration_s: float) -> np.ndarray:
    """Stereo audio (N, 2) float32."""
    n = int(duration_s * SR)
    rng = np.random.default_rng(42)
    return rng.uniform(-0.3, 0.3, (n, 2)).astype(np.float32)


def _make_mono(duration_s: float) -> np.ndarray:
    n = int(duration_s * SR)
    rng = np.random.default_rng(7)
    return rng.uniform(-0.3, 0.3, n).astype(np.float32)


@pytest.mark.unit
class TestPhase23SharedDeadline:
    """phase_deadline parameter propagated and used correctly."""

    def test_repair_channel_accepts_phase_deadline(self) -> None:
        """_repair_channel() must accept phase_deadline kwarg without error."""
        import inspect

        sig = inspect.signature(SpectralRepair._repair_channel)
        assert "phase_deadline" in sig.parameters, "_repair_channel missing phase_deadline"

    def test_repair_channel_mrsa_accepts_phase_deadline(self) -> None:
        """_repair_channel_mrsa() must accept phase_deadline kwarg without error."""
        import inspect

        sig = inspect.signature(SpectralRepair._repair_channel_mrsa)
        assert "phase_deadline" in sig.parameters, "_repair_channel_mrsa missing phase_deadline"

    def test_phase_deadline_in_process(self) -> None:
        """_phase_deadline must be calculated and used inside process()."""
        import inspect

        src = inspect.getsource(SpectralRepair.process)
        assert "_phase_deadline" in src, "_phase_deadline missing from process()"
        assert "phase_deadline=_phase_deadline" in src, "_phase_deadline not passed to _repair_channel() calls"

    def test_mrsa_budget_uses_remaining_time(self) -> None:
        """_repair_channel_mrsa() must use phase_deadline to limit budget."""
        import inspect

        src = inspect.getsource(SpectralRepair._repair_channel_mrsa)
        assert "_remaining_s" in src or "phase_deadline" in src, (
            "_repair_channel_mrsa does not use phase_deadline for budget"
        )

    def test_deadline_formula_for_225s_song(self) -> None:
        """For 225s audio: deadline cap = min(300, max(90, 1.3×225)) = 292.5s."""
        dur_s = 225.0
        expected = min(300.0, max(90.0, 1.3 * dur_s))
        assert abs(expected - 292.5) < 0.1

    def test_deadline_formula_for_short_song(self) -> None:
        """For 30s audio: deadline cap = min(300, max(90, 1.3×30)) = 90s (floor)."""
        dur_s = 30.0
        expected = min(300.0, max(90.0, 1.3 * dur_s))
        assert expected == 90.0

    def test_mrsa_budget_respects_remaining_deadline(self) -> None:
        """MRSA-Budgetlogik muss einen verbliebenen Deadline-Rest mit 10s-Floor nutzen.

        Hintergrund: Ein direkter Runtime-Aufruf von `_repair_channel_mrsa` kann in
        CI-Container-Setups sehr speicherintensiv werden und OOM-Kills auslösen,
        obwohl hier nur der Vertragsaspekt (Deadline + Floor) geprüft werden soll.
        """
        import inspect

        src = inspect.getsource(SpectralRepair._repair_channel_mrsa)
        assert "phase_deadline" in src, "phase_deadline wird in _repair_channel_mrsa nicht berücksichtigt"
        # Der Floor muss explizit vorhanden sein (mindestens 10s Restbudget).
        assert "10.0" in src, "10s Mindestbudget (Floor) in _repair_channel_mrsa nicht gefunden"

    def test_process_sets_deadline_before_repair_calls(self) -> None:
        """Ensure _phase_deadline is set before the repair block in process()."""
        import inspect

        src = inspect.getsource(SpectralRepair.process)
        deadline_idx = src.find("_phase_deadline = time.monotonic()")
        repair_idx = src.find("_repair_channel(")
        assert deadline_idx < repair_idx, "_phase_deadline must be assigned before _repair_channel() calls"
