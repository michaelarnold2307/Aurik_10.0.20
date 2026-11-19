from __future__ import annotations

import pytest

"""Unit tests for shared pipeline health state helpers."""


from backend.core.pipeline_health_state import (
    PipelineHealthState,
    normalize_pipeline_health_state,
    pipeline_health_from_fail_reasons,
    primary_fail_reason_from_fail_reasons,
    resolve_fail_reason,
)


@pytest.mark.unit
def test_normalize_pipeline_health_state_unknown_defaults_ok():
    state = normalize_pipeline_health_state("unexpected")
    assert state == PipelineHealthState.OK


def test_pipeline_health_from_fail_reasons_empty_is_ok():
    state = pipeline_health_from_fail_reasons([])
    assert state == PipelineHealthState.OK


def test_pipeline_health_from_fail_reasons_detects_critical():
    state = pipeline_health_from_fail_reasons([{"component": "arc", "error_code": "ARC_REGRESSION_ROLLBACK"}])
    assert state == PipelineHealthState.CRITICAL_DEGRADED


def test_pipeline_health_from_fail_reasons_detects_blocked():
    state = pipeline_health_from_fail_reasons([{"component": "pipeline", "error_code": "PIPELINE_BLOCKED"}])
    assert state == PipelineHealthState.BLOCKED


def test_pipeline_health_from_fail_reasons_prefers_severity_over_error_code():
    state = pipeline_health_from_fail_reasons(
        [{"component": "pqs", "error_code": "PQS_UNAVAILABLE", "severity": "critical"}]
    )
    assert state == PipelineHealthState.CRITICAL_DEGRADED


def test_pipeline_health_from_fail_reasons_uses_degraded_severity_without_code():
    state = pipeline_health_from_fail_reasons([{"component": "adapter", "severity": "degraded"}])
    assert state == PipelineHealthState.DEGRADED


def test_resolve_fail_reason_prefers_typed_field_then_fallbacks():
    assert (
        resolve_fail_reason(
            typed_fail_reason="typed",
            metadata={"fail_reason": "meta"},
            stage_notes={"fail_reason": "notes"},
        )
        == "typed"
    )
    assert (
        resolve_fail_reason(
            typed_fail_reason=None,
            metadata={"fail_reason": "meta"},
            stage_notes={"fail_reason": "notes"},
        )
        == "meta"
    )


def test_primary_fail_reason_from_fail_reasons_prefers_error_code_then_message_fields():
    assert (
        primary_fail_reason_from_fail_reasons([{"error_code": "PQS_GATE_FAILED", "exc_msg": "ignored"}])
        == "PQS_GATE_FAILED"
    )
    assert primary_fail_reason_from_fail_reasons([{"exc_msg": "Nur Nachricht"}]) == "Nur Nachricht"


def test_resolve_fail_reason_uses_structured_fail_reasons_when_text_fields_empty():
    assert (
        resolve_fail_reason(
            typed_fail_reason=None,
            metadata={"fail_reason": "", "fail_reasons": [{"error_code": "QUALITY_GATE_FAILED"}]},
            stage_notes={"fail_reason": ""},
        )
        == "QUALITY_GATE_FAILED"
    )


def test_resolve_fail_reason_prefers_explicit_fail_reasons_argument():
    assert (
        resolve_fail_reason(
            typed_fail_reason=None,
            metadata={"fail_reason": ""},
            stage_notes={"fail_reason": ""},
            fail_reasons=[{"exc_msg": "Direkte strukturierte Fehlermeldung"}],
        )
        == "Direkte strukturierte Fehlermeldung"
    )
