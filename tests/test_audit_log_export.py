import pytest

from audit.audit_report_generator import generate_audit_report


@pytest.mark.unit
def test_audit_log_dummy():
    """Audit report generator is available and writes all required formats."""
    import tempfile
    from pathlib import Path

    result_data = {
        "medium": "cd_digital",
        "chain": "restoration",
        "policy_name": "default",
        "audit": True,
        "quality": {"pqs_mos": 4.2},
        "passed": True,
        "details": "ok",
        "benchmarks": {"visqol": 0.82, "custom": 0.80},
    }

    tmp_path = Path(tempfile.mkdtemp(prefix="aurik_audit_"))
    generate_audit_report(result_data, out_dir=str(tmp_path))

    assert list(tmp_path.glob("*.yaml")), "YAML-Auditreport wurde nicht erzeugt"
    assert list(tmp_path.glob("*.json")), "JSON-Auditreport wurde nicht erzeugt"
    assert list(tmp_path.glob("*.csv")), "CSV-Auditreport wurde nicht erzeugt"


def test_audit_log_export_creates_json_and_csv():
    """Audit export writes parseable JSON and CSV files."""
    import csv
    import json
    import tempfile
    from pathlib import Path

    result_data = {
        "medium": "tape",
        "chain": "studio_2026",
        "policy_name": "strict",
        "passed": False,
        "details": "demo",
        "benchmarks": {"visqol": 0.74, "custom": 0.69},
    }

    out_dir = Path(tempfile.mkdtemp(prefix="aurik_audit_export_"))
    generate_audit_report(result_data, out_dir=str(out_dir))

    json_files = sorted(out_dir.glob("*.json"))
    csv_files = sorted(out_dir.glob("*.csv"))
    assert json_files, "Kein JSON-Auditreport gefunden"
    assert csv_files, "Kein CSV-Auditreport gefunden"

    with open(json_files[-1], encoding="utf-8") as f:
        payload = json.load(f)
    assert payload.get("medium") == "tape"
    assert payload.get("chain") == "studio_2026"

    with open(csv_files[-1], newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows, "CSV-Auditreport enthält keine Datensätze"
    assert rows[0].get("medium") == "tape"
