from pathlib import Path

from scripts.g188_wirkungskalibrierung_check import ROOT, scan_file, scan_source


def test_g188_scanner_reports_only_subunit_strength_factors() -> None:
    source = """
strength *= 0.7
boost_strength *= 1.15
strength = strength * 1.0
strength = strength * 0.0
"""

    issues = scan_source(Path("backend/core/sample.py"), source)

    assert len(issues) == 2
    assert "0.70" in issues[0]
    assert "0.00" in issues[1]


def test_g188_scanner_ignores_test_fixtures() -> None:
    path = ROOT / "tests" / "unit" / "fixture.py"

    assert scan_file(path) == []
