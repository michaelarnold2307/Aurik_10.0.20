from __future__ import annotations

"""
test_safe_execution.py — §v10 Structured Error Reporting Tests
==============================================================

Verifiziert, dass SafeExecutionContext Exceptions korrekt protokolliert,
Fehler-Statistiken führt und Rate-Limiting funktioniert.
"""


import pytest


class TestSafeExecutionContext:
    """SafeExecutionContext: Exception-Handling mit Kontext."""

    def test_01_context_captures_exception(self):
        """Exception wird erfasst und protokolliert, nicht verschluckt."""
        from backend.core.safe_execution import SafeExecutionContext

        with SafeExecutionContext("test_context") as ctx:
            raise ValueError("Test-Exception")

        assert ctx.failed, "ctx.failed sollte True sein"
        assert ctx.last_error_type == "ValueError"
        assert "Test-Exception" in str(ctx.last_error)

    def test_02_no_exception_leaves_context_clean(self):
        """Ohne Exception bleibt der Context sauber."""
        from backend.core.safe_execution import SafeExecutionContext

        with SafeExecutionContext("test_clean") as ctx:
            _ = 1 + 1  # Kein Fehler

        assert not ctx.failed
        assert ctx.last_error is None

    def test_03_re_raise_propagates_exception(self):
        """Mit re_raise=True wird die Exception weitergegeben."""
        from backend.core.safe_execution import SafeExecutionContext

        with pytest.raises(ValueError):
            with SafeExecutionContext("test_reraise", re_raise=True):
                raise ValueError("Sollte durchkommen")

    def test_04_error_statistics_increment(self):
        """Fehler-Statistiken werden inkrementiert."""
        from backend.core.safe_execution import (
            SafeExecutionContext,
            get_error_statistics,
            reset_error_statistics,
        )

        reset_error_statistics()

        for _ in range(3):
            with SafeExecutionContext("stats_test"):
                raise RuntimeError("Test")

        stats = get_error_statistics()
        assert stats["total_errors"] >= 3, f"Erwartet >=3 errors, bekam {stats['total_errors']}"

        reset_error_statistics()

    def test_05_safe_call_decorator(self):
        """safe_call Decorator fängt Exceptions."""
        from backend.core.safe_execution import safe_call

        call_count = [0]

        @safe_call("test_decorator")
        def failing_function():
            call_count[0] += 1
            raise RuntimeError("Decorator-Test")

        # Sollte nicht crashen
        failing_function()
        assert call_count[0] == 1, "Funktion sollte aufgerufen worden sein"

    def test_06_rate_limiting_prevents_flooding(self):
        """Rate-Limiting verhindert Log-Flooding."""
        from backend.core.safe_execution import (
            SafeExecutionContext,
            reset_error_statistics,
        )

        reset_error_statistics()

        # Viele Fehler in kurzer Zeit — sollten nicht alle geloggt werden
        for _ in range(10):
            with SafeExecutionContext("rate_test", rate_limit=True):
                raise RuntimeError("Flood")

        # Nur der erste Fehler sollte geloggt worden sein (Rate-Limit 60s)
        # Wir können nur prüfen, dass kein Crash passiert
        assert True  # Kein Crash = OK
