"""audit_api.py – REST-API für Audit-/Policy-Abfragen und Compliance-Status

- Endpunkte für Audit-Reports, Policy-Status, User-Feedback und Benchmarks
- Nutzt FastAPI für moderne API
"""

import logging

logger = logging.getLogger(__name__)

import os
from typing import Any

import yaml
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="Audit/Policy API", description="REST-API für Audit- und Policy-Compliance")


@app.get("/audits")
def get_audits() -> JSONResponse:
    audits: list[dict[str, Any]] = []
    for fname in os.listdir("."):
        if fname.endswith("_audit.yaml") or fname.endswith("_audit_feedback.yaml"):
            with open(fname) as f:
                try:
                    audits.append(yaml.safe_load(f))
                except Exception:
                    logger.debug("Stiller Ersatzpfad dokumentiert (Bug 9/V74)", exc_info=True)
                    continue
    return JSONResponse(audits)


@app.get("/policies")
def get_policies() -> JSONResponse:
    with open("dsp_policy_contracts_overview.yaml") as f:
        policies = yaml.safe_load(f).get("policies", [])
    return JSONResponse(policies)


@app.get("/compliance")
def get_compliance() -> JSONResponse:
    # Liefert Quality-Gate-Details und Policy-Logik aus Audit-Reports
    import glob

    reports: list[dict[str, Any]] = []
    for fname in glob.glob("*_audit.json"):
        with open(fname) as f:
            try:
                data = yaml.safe_load(f)
                # Quality-Gate-Details extrahieren
                qlog = data.get("policy", {}).get("_quality_passed")
                if qlog:
                    data["quality_log"] = qlog
                reports.append(data)
            except Exception:
                logger.debug("Audit-Report-Extraktion fehlgeschlagen", exc_info=True)
                continue
    return JSONResponse(reports)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8081)
