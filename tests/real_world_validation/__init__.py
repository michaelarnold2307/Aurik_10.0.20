"""
Real-World Validation Suite for AURIK v8

This package provides tools for comprehensive real-world validation of audio restoration:
- Objective metrics (SNR, THD, Spectral analysis)
- Subjective evaluation (Blind tests)
- Statistical analysis

Components:
- test_dataset_creator.py: Create test audio datasets
- validation_suite.py: Compute objective quality metrics
- blind_test_generator.py: Generate A/B/X blind tests
- results_analyzer.py: Analyze blind test results
"""

__version__ = "1.0.0"
__author__ = "AURIK Development Team"
__date__ = "2026-02-09"

from pathlib import Path

# Package root
PACKAGE_ROOT = Path(__file__).parent

# Test library paths
TEST_LIBRARY = PACKAGE_ROOT / "test_library"
VINYL_DIR = TEST_LIBRARY / "vinyl"
TAPE_DIR = TEST_LIBRARY / "tape"
DIGITAL_DIR = TEST_LIBRARY / "digital"
VOCALS_DIR = TEST_LIBRARY / "vocals"

__all__ = ["DIGITAL_DIR", "TAPE_DIR", "TEST_LIBRARY", "VINYL_DIR", "VOCALS_DIR"]
