from __future__ import annotations

"""[RELEASE_MUST] §2.49b Stereo-No-Regress Contract.

Sichert den post-pipeline kumulativen Stereo-Collapse-Guard gegen Regression ab.
"""


from pathlib import Path

import pytest

_UV3 = Path("backend/core/unified_restorer_v3.py")


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_post_pipeline_collapse_guard_thresholds_present() -> None:
    assert _UV3.exists(), f"Fehlt: {_UV3}"
    text = _UV3.read_text(encoding="utf-8")

    assert "§2.49b POST-PIPELINE cumulative stereo collapse" in text, "§2.49b Guard-Block fehlt in UV3."
    assert "if _cu2_imb > 20.0 and _pp2_imb < 6.0" in text, (
        "Kritische Schwellen (20 dB Kollaps / 6 dB Referenz) fehlen im Guard."
    )


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_collapse_guard_recovery_cascade_present() -> None:
    text = _UV3.read_text(encoding="utf-8")

    assert "_afg_best_clean_checkpoint" in text, (
        "Stereo-Collapse-Guard muss best_clean_checkpoint als erste Recovery-Stufe nutzen."
    )
    assert "pre_pipeline_audio" in text, "Stereo-Collapse-Guard braucht Fallback auf pre_pipeline_audio."
    assert "best_clean_checkpoint selbst kollabiert" in text, (
        "Guard muss den Fall eines bereits kollabierten Checkpoints explizit behandeln."
    )


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_hpi_rollback_checkpoint_stereo_health_validation_present() -> None:
    text = _UV3.read_text(encoding="utf-8")

    assert "HPI-Rollback-Checkpoint hat silent R-Kanal" in text or "HPI-Rollback-Checkpoint" in text, (
        "HPI-Rollback-Checkpoint-Studio-Health-Validierung fehlt."
    )
    assert "_hpi_best_rollback_audio" in text, "HPI-Rollback-Ziel muss als dediziertes Checkpoint-Feld vorhanden sein."
