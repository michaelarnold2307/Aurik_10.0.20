"""audit_dashboard.py – Minimaler Dashboard-Server für Policy-/Audit-Übersicht

- Liest alle Audit-Reports (YAML) im Workspace
- Zeigt Policy-Status, Qualitätsziele, User-Feedback und Benchmarks als Web-UI
- Nutzt Flask und Bootstrap für einfache Darstellung
"""

import logging

logger = logging.getLogger(__name__)

import os

import yaml
from flask import Flask, render_template_string

template = """
<!DOCTYPE html>
<html lang="de">
<head>
  <meta charset="UTF-8">
  <title>Audit Dashboard</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
</head>
<body>
<div class="container mt-4">
  <h1>Policy-/Audit-Übersicht</h1>
  {% for report in reports %}
    <div class="card mb-3">
      <div class="card-header">
        <b>{{ report.medium|capitalize }}</b> | Chain: {{ report.chain }} | Policy: {{ report.policy.name }}
      </div>
      <div class="card-body">
        <ul>
          <li>Defekte: {{ report.defects|join(', ') }}</li>
          <li>Qualitätsziele: HF-Ratio {{ report.policy.quality.hf_ratio }}, Transparenz {{ report.policy.quality.min_transparency }}, Artefaktenergie {{ report.policy.quality.artefact_energy }}, Loudness {{ report.policy.quality.max_loudness }}</li>
          <li>Status: <b>{{ 'OK' if report.result.passed else 'Fehler' }}</b></li>
          <li>Details: {{ report.result.details }}</li>
          <li>Quality Gates: <pre>{{ report.policy._quality_passed if report.policy and report.policy._quality_passed else '-' }}</pre></li>
          <li>User-Feedback: {{ report.user_feedback.rating if report.user_feedback else '-' }} / Kommentar: {{ report.user_feedback.comment if report.user_feedback else '-' }}</li>
          <li>Benchmarks: ViSQOL {{ report.benchmarks.visqol if report.benchmarks else '-' }}, DNSMOS {{ report.benchmarks.dnsmos if report.benchmarks else '-' }}, Custom {{ report.benchmarks.custom if report.benchmarks else '-' }}</li>
          <li>Timestamp: {{ report.result.timestamp }}</li>
        </ul>
      </div>
    </div>
  {% endfor %}
</div>
</body>
</html>
"""

app = Flask(__name__)


@app.route("/")
def dashboard():
    reports = []
    for fname in os.listdir("."):
        if fname.endswith("_audit.yaml") or fname.endswith("_audit_feedback.yaml"):
            with open(fname) as f:
                try:
                    rep = yaml.safe_load(f)
                    reports.append(rep)
                except Exception:
                    logger.debug("Stiller Ersatzpfad dokumentiert (Bug 9/V74)", exc_info=True)
                    continue
    return render_template_string(template, reports=reports)


if __name__ == "__main__":
    # Security: NEVER run with debug=True in production!
    # Debug mode exposes Werkzeug debugger and allows arbitrary code execution (CWE-94)
    app.run(host="0.0.0.0", port=8080, debug=False)
