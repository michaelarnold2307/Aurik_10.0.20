import pytest

"""
Policy-Engine Decision Logic Test
==================================

Tests whether Policy-Engine makes correct model selections
based on different audio contexts (without Docker execution).

This validates the BRAIN of the system - the decision logic.
"""

import logging
import sys
from typing import Any

from policy.ml_policy_engine import (
    CANONICAL_INSTRUMENTAL_NR_ROUTE,
    CANONICAL_REPAIR_ROUTE,
    CANONICAL_SEPARATION_ROUTE,
    CANONICAL_VOCAL_NR_ROUTE,
    MLModelPolicyEngine,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@pytest.mark.unit
def test_denoise_selections():
    """Test denoise model selection logic (🔬 semantic-aware)."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST: Denoise Model Selection (Semantic-Aware)")
    logger.info("=" * 60)

    engine = MLModelPolicyEngine()

    # 🎯 SEMANTIC-AWARE TEST CASES (replaces genre-based logic)
    test_cases: list[dict[str, Any]] = [
        {
            "name": "Classical Piano (Sustained)",
            "context": {
                "content_character": "HIGHLY_SUSTAINED",
                "dominant_instrument": "KEYS",
                "has_vocals": False,
                "has_ambient": False,
                "noise_type": "broadband",
                "sample_rate": 48000,
            },
            "goal": {"target_quality": 0.85},
            "expected": CANONICAL_INSTRUMENTAL_NR_ROUTE,
        },
        {
            "name": "Speech/Podcast (Vocals)",
            "context": {
                "content_character": "BALANCED",
                "dominant_instrument": "SPEECH",
                "has_vocals": True,
                "noise_type": "broadband",
                "sample_rate": 16000,
            },
            "goal": {"target_quality": 0.85},
            "expected": CANONICAL_VOCAL_NR_ROUTE,
        },
        {
            "name": "Vinyl Jazz",
            "context": {
                "content_character": "BALANCED",
                "detected_medium": "vinyl",
                "has_vocals": False,
                "sample_rate": 44100,
            },
            "goal": {"target_quality": 0.85},
            "expected": CANONICAL_INSTRUMENTAL_NR_ROUTE,
        },
        {
            "name": "Pop Music (Vocals)",
            "context": {
                "content_character": "BALANCED",
                "dominant_instrument": "VOCALS",
                "has_vocals": True,
                "noise_type": "unknown",
                "sample_rate": 48000,
            },
            "goal": {"target_quality": 0.85},
            "expected": CANONICAL_VOCAL_NR_ROUTE,
        },
        {
            "name": "Drum Kit (Transient-Rich)",
            "context": {
                "content_character": "HIGHLY_TRANSIENT",
                "dominant_instrument": "DRUMS",
                "has_drums": True,
                "has_vocals": False,
                "preserve_transients": True,
                "sample_rate": 48000,
            },
            "goal": {"target_quality": 0.85},
            "expected": CANONICAL_INSTRUMENTAL_NR_ROUTE,
        },
        {
            "name": "Ambient/Drone (Sustained)",
            "context": {
                "content_character": "HIGHLY_SUSTAINED",
                "dominant_instrument": "AMBIENT",
                "has_ambient": True,
                "has_vocals": False,
                "sample_rate": 48000,
            },
            "goal": {"target_quality": 0.85},
            "expected": CANONICAL_INSTRUMENTAL_NR_ROUTE,
        },
    ]

    results = {}
    for test in test_cases:
        selected = engine.select_denoise_model(test["context"], test["goal"])
        expected = test["expected"]
        match = selected == expected

        status = "✅" if match else "❌"
        logger.info(f"{status} {test['name']}: {selected} (expected: {expected})")

        results[test["name"]] = match

    failed = [name for name, ok in results.items() if not ok]
    assert not failed, f"Policy test(s) failed: {failed}"


def test_repair_selections():
    """Test repair model selection logic."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST: Repair Model Selection")
    logger.info("=" * 60)

    engine = MLModelPolicyEngine()

    # Note: repair_model selection uses has_vocals, no semantic changes needed
    test_cases: list[dict[str, Any]] = [
        {
            "name": "Speech with Clipping",
            "context": {"has_vocals": True},
            "goal": {"target_quality": 0.85},
            "expected": CANONICAL_REPAIR_ROUTE,
        },
        {
            "name": "Music with Clipping",
            "context": {"has_vocals": False},
            "goal": {"target_quality": 0.85},
            "expected": CANONICAL_REPAIR_ROUTE,
        },
    ]

    results = {}
    for test in test_cases:
        selected = engine.select_repair_model(test["context"], test["goal"])
        expected = test["expected"]
        match = selected == expected

        status = "✅" if match else "❌"
        logger.info(f"{status} {test['name']}: {selected} (expected: {expected})")

        results[test["name"]] = match

    failed = [name for name, ok in results.items() if not ok]
    assert not failed, f"Policy test(s) failed: {failed}"


def test_separation_selections():
    """Test separation model selection logic."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST: Separation Model Selection")
    logger.info("=" * 60)

    engine = MLModelPolicyEngine()

    test_cases: list[dict[str, Any]] = [
        {
            "name": "High Quality (6-stem)",
            "context": {"sample_rate": 44100},
            "goal": {"target_quality": 0.9, "stems": 6},  # FIXED: added stems parameter
            "expected": CANONICAL_SEPARATION_ROUTE,
        },
        {
            "name": "Fast Processing",
            "context": {"sample_rate": 44100},
            "goal": {"quality_level": "maximal", "priority": "speed"},  # FIXED: added quality_level
            "expected": CANONICAL_SEPARATION_ROUTE,
        },
    ]

    results = {}
    for test in test_cases:
        selected = engine.select_separation_model(test["context"], test["goal"])
        expected = test["expected"]
        match = selected == expected

        status = "✅" if match else "❌"
        logger.info(f"{status} {test['name']}: {selected} (expected: {expected})")

        results[test["name"]] = match

    failed = [name for name, ok in results.items() if not ok]
    assert not failed, f"Policy test(s) failed: {failed}"


def main():
    """Run all policy decision tests."""
    logger.info("=" * 60)
    logger.info("POLICY-ENGINE DECISION LOGIC TEST SUITE")
    logger.info("=" * 60)

    failed_count = 0
    total = 3
    for test_fn in (test_denoise_selections, test_repair_selections, test_separation_selections):
        try:
            test_fn()
            logger.info("✅ PASS: %s", test_fn.__name__)
        except AssertionError as e:
            logger.warning("❌ FAIL: %s — %s", test_fn.__name__, e)
            failed_count += 1

    logger.info("\n" + "=" * 60)
    logger.info(f"TOTAL: {total - failed_count}/{total} tests passed")
    logger.info("=" * 60)

    if failed_count == 0:
        logger.info("\n🎉 ALL POLICY TESTS PASSED!")
        return 0
    else:
        logger.warning("\n⚠️  %d test(s) failed.", failed_count)
        return 1


if __name__ == "__main__":
    sys.exit(main())
