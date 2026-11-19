from __future__ import annotations

import pytest

from backend.core.defect_scanner import MaterialType
from backend.core.performance_guard import QualityMode
from backend.core.unified_restorer_v3 import UnifiedRestorerV3


@pytest.mark.unit
def test_is_mp3_material_true_for_low_high() -> None:
    assert UnifiedRestorerV3._is_mp3_material(MaterialType.MP3_LOW) is True
    assert UnifiedRestorerV3._is_mp3_material(MaterialType.MP3_HIGH) is True


def test_is_mp3_material_false_for_non_mp3() -> None:
    assert UnifiedRestorerV3._is_mp3_material(MaterialType.CD_DIGITAL) is False
    assert UnifiedRestorerV3._is_mp3_material(MaterialType.VINYL) is False


def test_is_lossy_codec_material_includes_non_mp3_lossy() -> None:
    assert UnifiedRestorerV3._is_lossy_codec_material(MaterialType.MP3_LOW) is True
    assert UnifiedRestorerV3._is_lossy_codec_material(MaterialType.MP3_HIGH) is True
    assert UnifiedRestorerV3._is_lossy_codec_material(MaterialType.AAC) is True
    assert UnifiedRestorerV3._is_lossy_codec_material(MaterialType.MINIDISC) is True
    assert UnifiedRestorerV3._is_lossy_codec_material(MaterialType.STREAMING) is True


def test_is_lossy_codec_material_excludes_lossless_digital() -> None:
    assert UnifiedRestorerV3._is_lossy_codec_material(MaterialType.CD_DIGITAL) is False
    assert UnifiedRestorerV3._is_lossy_codec_material(MaterialType.DAT) is False


def test_mp3_guard_disabled_for_non_maximum_mode() -> None:
    assert (
        UnifiedRestorerV3._should_force_mp3_maximum_guard(
            mode=QualityMode.QUALITY,
            material_type=MaterialType.MP3_HIGH,
            input_snr_db=48.0,
            max_defect_severity=0.05,
            clean_digital_mode=True,
        )
        is False
    )


def test_mp3_guard_enabled_for_clean_digital_mp3_in_maximum() -> None:
    assert (
        UnifiedRestorerV3._should_force_mp3_maximum_guard(
            mode=QualityMode.MAXIMUM,
            material_type=MaterialType.MP3_HIGH,
            input_snr_db=42.0,
            max_defect_severity=0.10,
            clean_digital_mode=True,
        )
        is True
    )


def test_mp3_guard_enabled_for_high_snr_low_defect_even_without_clean_flag() -> None:
    assert (
        UnifiedRestorerV3._should_force_mp3_maximum_guard(
            mode=QualityMode.MAXIMUM,
            material_type=MaterialType.MP3_LOW,
            input_snr_db=35.0,
            max_defect_severity=0.18,
            clean_digital_mode=False,
        )
        is True
    )


def test_codec_guard_enabled_for_aac_in_maximum() -> None:
    assert (
        UnifiedRestorerV3._should_force_mp3_maximum_guard(
            mode=QualityMode.MAXIMUM,
            material_type=MaterialType.AAC,
            input_snr_db=36.0,
            max_defect_severity=0.15,
            clean_digital_mode=False,
        )
        is True
    )


def test_codec_guard_disabled_for_cd_digital_even_in_maximum() -> None:
    assert (
        UnifiedRestorerV3._should_force_mp3_maximum_guard(
            mode=QualityMode.MAXIMUM,
            material_type=MaterialType.CD_DIGITAL,
            input_snr_db=45.0,
            max_defect_severity=0.05,
            clean_digital_mode=True,
        )
        is False
    )


def test_codec_guard_enabled_for_minidisc_in_maximum() -> None:
    assert (
        UnifiedRestorerV3._should_force_mp3_maximum_guard(
            mode=QualityMode.MAXIMUM,
            material_type=MaterialType.MINIDISC,
            input_snr_db=37.0,
            max_defect_severity=0.12,
            clean_digital_mode=False,
        )
        is True
    )


def test_codec_guard_enabled_for_streaming_when_clean_digital() -> None:
    assert (
        UnifiedRestorerV3._should_force_mp3_maximum_guard(
            mode=QualityMode.MAXIMUM,
            material_type=MaterialType.STREAMING,
            input_snr_db=31.0,
            max_defect_severity=0.22,
            clean_digital_mode=True,
        )
        is True
    )


def test_mp3_guard_disabled_for_noisy_or_strongly_defective_mp3() -> None:
    assert (
        UnifiedRestorerV3._should_force_mp3_maximum_guard(
            mode=QualityMode.MAXIMUM,
            material_type=MaterialType.MP3_LOW,
            input_snr_db=28.0,
            max_defect_severity=0.30,
            clean_digital_mode=False,
        )
        is False
    )
