"""
Template für Schellack-spezifische Quality-Reports (auditierbar)
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def create_shellac_quality_report(
    anchor_stability_before: float,
    anchor_stability_after: float,
    click_density_before: float,
    click_density_after: float,
    crackle_density_before: str,
    crackle_density_after: str,
    rumble_energy_before_db: float,
    rumble_energy_after_db: float,
    noise_floor_before_dbfs: float,
    noise_floor_after_dbfs: float,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """
    Erstellt einen Schellack-Quality-Report. Quality-Gate, Audit-Logging, robuste Fehlerbehandlung integriert.
    :return: Report-Dict
    """
    try:
        # Quality-Gate: Input-Check
        for val in [
            anchor_stability_before,
            anchor_stability_after,
            click_density_before,
            click_density_after,
            rumble_energy_before_db,
            rumble_energy_after_db,
            noise_floor_before_dbfs,
            noise_floor_after_dbfs,
        ]:
            if not isinstance(val, (float, int)):
                _audit_log("error", f"Input {val} ist kein float/int")
                raise ValueError("Alle numerischen Inputs müssen float/int sein")
        report: dict[str, Any] = {
            "anchor_stability": {
                "before": anchor_stability_before,
                "after": anchor_stability_after,
                "delta": round(anchor_stability_after - anchor_stability_before, 3),
            },
            "click_density_per_sec": {
                "before": click_density_before,
                "after": click_density_after,
            },
            "crackle_density": {
                "before": crackle_density_before,
                "after": crackle_density_after,
            },
            "rumble_energy_db": {
                "before": rumble_energy_before_db,
                "after": rumble_energy_after_db,
            },
            "noise_floor_dbfs": {
                "before": noise_floor_before_dbfs,
                "after": noise_floor_after_dbfs,
            },
            "warnings": warnings or [],
        }
        _audit_log("success", "Quality-Report erfolgreich erstellt")
        return report
    except Exception as e:
        _audit_log("error", f"Fehler beim Erstellen des Reports: {e}")
        return {"error": str(e)}


def _audit_log(level: str, message: str) -> None:
    _fn = {"error": logger.error, "warn": logger.warning, "warning": logger.warning}.get(level.lower(), logger.info)
    _fn("[shellac_quality_report_template] %s", message)


# Interpretationsregeln (automatisch)
def interpret_shellac_report(report: dict[str, Any]) -> dict[str, str]:
    """
    Interpretiert einen Schellack-Quality-Report. Quality-Gate, Audit-Logging, robuste Fehlerbehandlung integriert.
    :return: Dict mit Interpretationen
    """
    try:
        interpretations: dict[str, str] = {}
        delta = report["anchor_stability"]["delta"]
        if delta >= 0.20:
            interpretations["anchor"] = "klarer Erfolg"
        if (report["click_density_per_sec"]["before"] - report["click_density_per_sec"]["after"]) / max(
            1, report["click_density_per_sec"]["before"]
        ) > 0.8:
            interpretations["clicks"] = "Immersion hergestellt (>80% Reduktion)"
        if (report["noise_floor_dbfs"]["before"] - report["noise_floor_dbfs"]["after"]) > 6 and "musical-noise" not in (
            report["warnings"] or []
        ):
            interpretations["noise"] = "Noise-Reduktion ok"
        _audit_log("success", "Interpretation erfolgreich")
        return interpretations
    except Exception as e:
        _audit_log("error", f"Fehler bei Interpretation: {e}")
        return {"error": str(e)}
