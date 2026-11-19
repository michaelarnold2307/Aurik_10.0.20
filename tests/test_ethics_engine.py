import json
import os

from backend.core.dsp.ethics_engine import check_ethics_and_originality


def test_ethics_engine(tmp_path):
    # Simuliere Audit-Log mit Policy- und Quality-Gate-Verstößen
    log = [
        {"step": "policy_check", "policy": "bias detected"},
        {"step": "quality_gate", "result": "fail"},
        {"step": "ethics_engine", "result": "pass"},
    ]
    log_path = tmp_path / "audit_trail.json"
    os.makedirs(tmp_path, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log, f)
    # Sollte False zurückgeben wegen Policy- und Quality-Gate-Verstoß
    result = check_ethics_and_originality(str(log_path))
    assert result is False
    print("Ethik-Engine Test bestanden.")
