from __future__ import annotations

"""Normative fault-injection tests for primary/fallback/blocked release modes."""


import pytest

from backend.core.fallback_guard import execute_with_fallback


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_primary_mode_when_no_fault() -> None:
    result = execute_with_fallback(lambda: "primary_ok", lambda: "fallback_ok")
    assert result.release_mode == "primary"
    assert result.value == "primary_ok"
    assert result.fail_reason is None
    assert result.degradation_status == "ok"
    assert result.fail_reasons == []


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_fallback_mode_on_primary_failure() -> None:
    def primary() -> str:
        raise RuntimeError("simulierte Modellfehler")

    result = execute_with_fallback(primary, lambda: "fallback_ok")
    assert result.release_mode == "fallback"
    assert result.value == "fallback_ok"
    assert "primary_failed" in (result.fail_reason or "")
    assert result.degradation_status == "degraded"
    assert isinstance(result.fail_reasons, list)
    assert result.fail_reasons
    assert result.fail_reasons[0]["error_code"] == "PRIMARY_FAILED"
    assert result.fail_reasons[0]["severity"] == "degraded"


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_blocked_mode_when_both_paths_fail() -> None:
    def fail_a() -> str:
        raise MemoryError("simulierte OOM")

    def fail_b() -> str:
        raise RuntimeError("DSP fallback unavailable")

    result = execute_with_fallback(fail_a, fail_b)
    assert result.release_mode == "blocked"
    assert result.value is None
    assert "fallback_failed" in (result.fail_reason or "")
    assert result.degradation_status == "blocked"
    assert isinstance(result.fail_reasons, list)
    assert len(result.fail_reasons) == 2
    assert result.fail_reasons[0]["error_code"] == "PRIMARY_FAILED"
    assert result.fail_reasons[1]["error_code"] == "FALLBACK_FAILED"
    assert result.fail_reasons[1]["severity"] == "blocked"
