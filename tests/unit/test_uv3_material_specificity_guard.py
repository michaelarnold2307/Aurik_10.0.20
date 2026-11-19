from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.core.defect_scanner import MaterialType
from backend.core.unified_restorer_v3 import UnifiedRestorerV3


@pytest.mark.unit
def test_preserve_specific_transport_material_for_cassette_chain() -> None:
    mc_result = SimpleNamespace(
        confidence=0.40,
        is_multi_generation=True,
        transfer_chain=["cassette", "mp3_low"],
    )
    assert UnifiedRestorerV3._should_preserve_specific_transport_material(
        MaterialType.CASSETTE,
        "tape",
        mc_result,
    )


def test_no_preserve_when_not_cassette_to_tape_case() -> None:
    mc_result = SimpleNamespace(
        confidence=0.25,
        is_multi_generation=False,
        transfer_chain=["vinyl", "mp3_low"],
    )
    assert not UnifiedRestorerV3._should_preserve_specific_transport_material(
        MaterialType.VINYL,
        "tape",
        mc_result,
    )
