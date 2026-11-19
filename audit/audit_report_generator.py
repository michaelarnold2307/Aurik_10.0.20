"""
audit_report_generator.py – Automatische Audit-Report-Generierung nach jedem Batch-Run

- Liest Ergebnisdaten (z.B. aus einer Pipeline-JSON oder YAML)
- Erstellt Audit-Report nach AUDIT_REPORT_SCHEMA.yaml
- Versioniert Report mit Zeitstempel und eindeutigem Namen
"""

import json
import os
from datetime import datetime

import yaml


def generate_audit_report(result_data, out_dir="."):
    # Alle Felder aus AUDIT_REPORT_SCHEMA.yaml werden unterstützt
    now = datetime.utcnow().isoformat()
    report = {
        "medium": result_data.get("medium", "unknown"),
        "defects": result_data.get("defects", []),
        "chain": result_data.get("chain", "default"),
        "policy": {
            "name": result_data.get("policy_name", "unknown"),
            "audit": result_data.get("audit", True),
            "quality": result_data.get("quality", {}),
        },
        "result": {
            "passed": result_data.get("passed", False),
            "details": result_data.get("details", ""),
            "timestamp": now,
        },
        "user_feedback": result_data.get("user_feedback", {"rating": None, "comment": ""}),
        "benchmarks": result_data.get("benchmarks", {}),
    }
    # Versionierung: medium_chain_audit_YYYY-MM-DDTHH-MM-SS
    base_name = f"{report['medium']}_{report['chain']}_audit_{now.replace(':', '-')}"
    # YAML
    yaml_path = os.path.join(out_dir, base_name + ".yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(report, f, sort_keys=False, allow_unicode=True)
    # JSON
    json_path = os.path.join(out_dir, base_name + ".json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    # CSV (flach, nur Hauptfelder)
    import csv

    csv_path = os.path.join(out_dir, base_name + ".csv")
    flat = {
        "medium": report["medium"],
        "chain": report["chain"],
        "policy_name": report["policy"]["name"],
        "passed": report["result"]["passed"],
        "timestamp": report["result"]["timestamp"],
        "rating": report["user_feedback"].get("rating"),
        "visqol": report["benchmarks"].get("visqol"),
        # "dnsmos" entfernt — verboten §4.4+§10.2 (Sprach-Metrik)
        "custom": report["benchmarks"].get("custom"),
        "details": report["result"]["details"],
    }
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)
    print(f"Audit-Report gespeichert: {yaml_path}, {json_path}, {csv_path}")


if __name__ == "__main__":
    # Beispielaufruf: Daten aus JSON laden
    import sys

    if len(sys.argv) < 2:
        print("Nutzung: python audit_report_generator.py <result.json>")
        exit(1)
    with open(sys.argv[1]) as f:
        result_data = json.load(f)
    generate_audit_report(result_data)
