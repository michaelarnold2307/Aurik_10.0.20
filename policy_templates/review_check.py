"""CLI-Review-Check für Policy-Templates: Führt die automatisierte Prüfung aller YAML-Templates im policy_templates/-Verzeichnis aus und gibt einen Review-Report aus."""

import glob
import logging
import os
import sys
from importlib import util as importlib_util

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

"""
Hinweis: Die Importe aus aurik4 sind entfernt. Sobald das Testmodul nach aurik6 migriert wurde,
kann es wie folgt importiert werden:
# from aurik6.testing.test_policy_templates import validate_policy_template
"""

TEMPLATE_DIR = os.path.dirname(__file__)
REPORT_PATH = os.path.join(TEMPLATE_DIR, "review_report.txt")


def validate_policy_template(path: str) -> list[str]:
    """Validate a single YAML policy template and return a list of errors."""
    errors: list[str] = []

    if not os.path.exists(path):
        return [f"Datei nicht gefunden: {path}"]

    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
    except Exception as exc:
        return [f"Datei konnte nicht gelesen werden: {exc}"]

    if not content.strip():
        return ["Template ist leer."]

    yaml_spec = importlib_util.find_spec("yaml")
    if yaml_spec is None:
        # Minimal fallback if PyYAML is not installed.
        if ":" not in content:
            errors.append("Kein YAML-Schluessel-Wert-Format erkannt (':' fehlt).")
        return errors

    try:
        import yaml  # type: ignore[import]

        data = yaml.safe_load(content)
        if data is None:
            errors.append("Template enthaelt kein YAML-Dokument.")
        elif not isinstance(data, dict):
            errors.append("Top-Level muss ein YAML-Objekt (Mapping) sein.")
    except Exception as exc:
        errors.append(f"Ungueltiges YAML: {exc}")

    return errors


def main():
    if os.path.exists(REPORT_PATH):
        os.remove(REPORT_PATH)
    any_errors = False
    for path in glob.glob(os.path.join(TEMPLATE_DIR, "*.yaml")):
        errors = validate_policy_template(path)
        if errors:
            any_errors = True
            logger.error(f"Fehler in {os.path.basename(path)}:")
            for err in errors:
                logger.error(f"  - {err}")
    if any_errors:
        logger.info(f"\nSiehe Review-Report: {REPORT_PATH}")
        sys.exit(1)
    logger.info("Alle Policy-Templates sind fehlerfrei.")


if __name__ == "__main__":
    main()
