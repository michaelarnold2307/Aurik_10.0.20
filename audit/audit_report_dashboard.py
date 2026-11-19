"""
Audit-Report-Visualisierung und Dashboard-Export
Aggregiert und visualisiert alle Audit-Reports (z.B. als HTML, CSV, Plots).
"""

import glob
import json
import os

import matplotlib.pyplot as plt
import pandas as pd

AUDIT_DIR = "./"
REPORT_CSV = "audit_report_summary.csv"
REPORT_HTML = "audit_report_dashboard.html"

# Sammle alle Audit-Reports
audit_files = glob.glob(os.path.join(AUDIT_DIR, "*_audit.json"))

# Daten sammeln
records = []
for afile in audit_files:
    with open(afile) as f:
        report = json.load(f)
    rec = {
        k: report.get(k)
        for k in [
            "output_file",
            "passed",
            "hf_ratio",
            "loudness_before",
            "loudness_after",
            "dynamic_range_before",
            "dynamic_range_after",
            "sib_ratio",
            "artefact_energy",
            "correlation",
        ]
    }
    # Quality-Gate-Details ergänzen, falls vorhanden
    if "policy" in report and "_quality_passed" in report["policy"]:
        rec["quality_log"] = report["policy"]["_quality_passed"]
    records.append(rec)

df = pd.DataFrame(records)
df.to_csv(REPORT_CSV, index=False)

# Plots
plt.figure(figsize=(10, 6))
df[["hf_ratio", "correlation", "artefact_energy"]].plot(kind="box")
plt.title("Verteilung wichtiger Qualitätsmetriken")
plt.savefig("audit_metrics_boxplot.png")
plt.close()

# HTML-Dashboard
with open(REPORT_HTML, "w") as f:
    f.write("<h1>SOTA Audit Dashboard</h1>")
    f.write(df.to_html())
    f.write('<img src="audit_metrics_boxplot.png" width="600">')

print(f"[Audit-Report] Zusammenfassung gespeichert als {REPORT_CSV} und {REPORT_HTML}")
