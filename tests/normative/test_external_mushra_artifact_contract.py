from __future__ import annotations

"""Externes Mini-MUSHRA-Artefakt-Kontrakt — [RELEASE_MUST] (§8.4 / §5.7 spec 07, v10.0.0)

Spec §8.4 / §5.7 (copilot-instructions.md, spec 07 §5.7):
    Bei Änderungen an Kernphasen, PMGG, DefectScanner oder heavy ML-Fallbacks
    ist ein externer Mini-MUSHRA-Bericht Pflicht.

    Pflichtanforderungen:
        1. Mindestens 6 Szenarien, davon mindestens 2 Vocal-Szenarien
        2. Mindestens 8 Hörer
        3. Szenario-Score, Konfidenzintervall, Delta zur Vorversion
        4. Bericht als Release-Artefakt versioniert abgelegt

    Kein Release ohne gültiges Bericht-Artefakt.

Dieser Test prüft:
    a) Den Artefakt-Schema-Kontrakt (normative Pflichtfelder)
    b) Mini-MUSHRA-Validierungs-API (mock-basiert)
    c) Wenn ein Artefakt in reports/ vorliegt, wird es gegen das Schema validiert

Gate-Tabelle (copilot-instructions.md):
    "Externer Mini-MUSHRA bei Kern-aenderungen [RELEASE_MUST]"
    → tests/normative/test_external_mushra_artifact_contract.py (diese Datei)

Ausführung: pytest tests/normative/test_external_mushra_artifact_contract.py --timeout=30 -v
"""


import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Normative Kontrakt-Definitionen (§8.4)
# ---------------------------------------------------------------------------

# Pflichtfelder auf oberster Ebene des MUSHRA-Artefakts
REQUIRED_TOP_LEVEL_FIELDS: frozenset[str] = frozenset(
    {
        "aurik_version",
        "report_date",
        "n_listeners",
        "scenarios",
        "overall_delta",
    }
)

# Pflichtfelder pro Szenario
REQUIRED_SCENARIO_FIELDS: frozenset[str] = frozenset(
    {
        "scenario_id",
        "description",
        "score",
        "confidence_interval",
        "delta_to_previous",
        "is_vocal",
    }
)

# Mindestwerte (§8.4)
MIN_SCENARIOS: int = 6
MIN_VOCAL_SCENARIOS: int = 2
MIN_LISTENERS: int = 8


# ---------------------------------------------------------------------------
# Pfade
# ---------------------------------------------------------------------------

_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
_REPORTS_DIR = _WORKSPACE_ROOT / "reports"
_MUSHRA_ARTIFACT_PATH = _REPORTS_DIR / "mushra_artifact.json"


def _strict_artifact_mode_enabled() -> bool:
    """Strict mode makes missing artifact a hard failure instead of skip."""
    return os.environ.get("AURIK_STRICT_MUSHRA_ARTIFACT", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


# ---------------------------------------------------------------------------
# Validierungs-API
# ---------------------------------------------------------------------------


@dataclass
class MushraArtifactValidationResult:
    """Ergebnis der Artefakt-Validierung."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return len(self.errors)


def validate_mushra_artifact(artifact: dict) -> MushraArtifactValidationResult:
    """Validiert ein MUSHRA-Artefakt gegen das normative Schema (§8.4).

    Args:
        artifact: Geparster JSON-Inhalt des MUSHRA-Berichts.

    Returns:
        MushraArtifactValidationResult mit Fehlern und Warnungen.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # 1. Pflicht-Top-Level-Felder prüfen
    for required_field in REQUIRED_TOP_LEVEL_FIELDS:
        if required_field not in artifact:
            errors.append(f"Fehlendes Pflichtfeld: '{required_field}'")

    # 2. n_listeners prüfen
    n_listeners = artifact.get("n_listeners", 0)
    if not isinstance(n_listeners, int) or n_listeners < MIN_LISTENERS:
        errors.append(f"n_listeners={n_listeners} < Minimum {MIN_LISTENERS} (§8.4 Pflicht: ≥ 8 Hörer)")

    # 3. Szenarien prüfen
    scenarios = artifact.get("scenarios", [])
    if not isinstance(scenarios, list):
        errors.append("'scenarios' muss eine Liste sein")
        return MushraArtifactValidationResult(valid=False, errors=errors)

    if len(scenarios) < MIN_SCENARIOS:
        errors.append(f"Nur {len(scenarios)} Szenarien — mindestens {MIN_SCENARIOS} erforderlich (§8.4)")

    # 4. Vocal-Szenarien prüfen
    vocal_scenarios = [s for s in scenarios if s.get("is_vocal") is True]
    if len(vocal_scenarios) < MIN_VOCAL_SCENARIOS:
        errors.append(
            f"Nur {len(vocal_scenarios)} Vocal-Szenarien — mindestens {MIN_VOCAL_SCENARIOS} erforderlich (§8.4)"
        )

    # 5. Pflichtfelder pro Szenario
    for i, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict):
            errors.append(f"Szenario [{i}] ist kein dict")
            continue
        for req_field in REQUIRED_SCENARIO_FIELDS:
            if req_field not in scenario:
                errors.append(
                    f"Szenario [{i}] (id={scenario.get('scenario_id', '?')}): fehlendes Pflichtfeld '{req_field}'"
                )

        # 6. Score-Wert validieren
        score = scenario.get("score")
        if score is not None:
            if not isinstance(score, (int, float)) or not math.isfinite(score):
                errors.append(f"Szenario [{i}]: score={score!r} muss endliche Zahl sein")
            elif not (0 <= score <= 100):
                errors.append(f"Szenario [{i}]: score={score} außerhalb [0, 100]")

        # 7. Confidence-Interval validieren
        ci = scenario.get("confidence_interval")
        if ci is not None:
            if not isinstance(ci, (int, float)) or ci < 0:
                errors.append(f"Szenario [{i}]: confidence_interval={ci!r} muss ≥ 0 sein")

        # 8. delta_to_previous muss endliche Zahl sein
        delta = scenario.get("delta_to_previous")
        if delta is not None:
            if not isinstance(delta, (int, float)) or not math.isfinite(delta):
                errors.append(f"Szenario [{i}]: delta_to_previous={delta!r} muss endliche Zahl sein")

    # 9. aurik_version muss vorhanden und nicht leer sein
    version = artifact.get("aurik_version", "")
    if not isinstance(version, str) or not version.strip():
        errors.append("'aurik_version' muss ein nicht-leerer String sein")

    # 10. overall_delta Warnhinweis
    overall_delta = artifact.get("overall_delta")
    if overall_delta is not None and isinstance(overall_delta, (int, float)):
        if overall_delta < 0:
            warnings.append(
                f"overall_delta={overall_delta:.1f} negativ — Release überprüfen (Regression gegenüber Vorversion)"
            )

    valid = len(errors) == 0
    return MushraArtifactValidationResult(valid=valid, errors=errors, warnings=warnings)


# ---------------------------------------------------------------------------
# Mock-Artefakte für Tests
# ---------------------------------------------------------------------------


def _make_valid_scenario(
    scenario_id: str = "scene_01",
    is_vocal: bool = False,
    score: float = 82.0,
    delta: float = 3.0,
    ci: float = 2.5,
) -> dict:
    return {
        "scenario_id": scenario_id,
        "description": f"Test scenario {scenario_id}",
        "score": score,
        "confidence_interval": ci,
        "delta_to_previous": delta,
        "is_vocal": is_vocal,
    }


def _make_valid_artifact(
    n_listeners: int = 10,
    n_scenarios: int = 7,
    n_vocal: int = 2,
) -> dict:
    scenarios = []
    for i in range(n_scenarios):
        is_vocal = i < n_vocal
        scenarios.append(
            _make_valid_scenario(
                scenario_id=f"scene_{i + 1:02d}",
                is_vocal=is_vocal,
            )
        )
    return {
        "aurik_version": "9.10.79",
        "report_date": "2026-03-28",
        "n_listeners": n_listeners,
        "scenarios": scenarios,
        "overall_delta": 4.2,
    }


# ===========================================================================
# Klasse 1: Schema-Kontrakt-Definitionen
# ===========================================================================


@pytest.mark.unit
class TestMushraSchemaContractDefinitions:
    """Tests: Die normativen Kontrakt-Konstanten sind korrekt definiert."""

    def test_min_scenarios_is_6(self):
        """§8.4: Mindestens 6 Szenarien."""
        assert MIN_SCENARIOS == 6

    def test_min_vocal_scenarios_is_2(self):
        """§8.4: Mindestens 2 Vocal-Szenarien."""
        assert MIN_VOCAL_SCENARIOS == 2

    def test_min_listeners_is_8(self):
        """§8.4: Mindestens 8 Hörer."""
        assert MIN_LISTENERS == 8

    def test_required_top_level_fields_includes_aurik_version(self):
        assert "aurik_version" in REQUIRED_TOP_LEVEL_FIELDS

    def test_required_top_level_fields_includes_n_listeners(self):
        assert "n_listeners" in REQUIRED_TOP_LEVEL_FIELDS

    def test_required_top_level_fields_includes_scenarios(self):
        assert "scenarios" in REQUIRED_TOP_LEVEL_FIELDS

    def test_required_top_level_fields_includes_overall_delta(self):
        assert "overall_delta" in REQUIRED_TOP_LEVEL_FIELDS

    def test_required_scenario_fields_includes_scenario_id(self):
        assert "scenario_id" in REQUIRED_SCENARIO_FIELDS

    def test_required_scenario_fields_includes_score(self):
        assert "score" in REQUIRED_SCENARIO_FIELDS

    def test_required_scenario_fields_includes_confidence_interval(self):
        assert "confidence_interval" in REQUIRED_SCENARIO_FIELDS

    def test_required_scenario_fields_includes_delta_to_previous(self):
        assert "delta_to_previous" in REQUIRED_SCENARIO_FIELDS

    def test_required_scenario_fields_includes_is_vocal(self):
        assert "is_vocal" in REQUIRED_SCENARIO_FIELDS


# ===========================================================================
# Klasse 2: Validierung gültiger Artefakte
# ===========================================================================


class TestValidArtifactValidation:
    """Tests: Gültige Artefakte bestehen die Schema-Validierung."""

    def test_valid_artifact_passes_validation(self):
        artifact = _make_valid_artifact()
        result = validate_mushra_artifact(artifact)
        assert result.valid, f"Gültiges Artefakt sollte validieren, Fehler: {result.errors}"

    def test_minimal_valid_artifact_passes(self):
        """Genau 6 Szenarien, 2 vocal, 8 Hörer — Minimum nach §8.4."""
        artifact = _make_valid_artifact(n_listeners=8, n_scenarios=6, n_vocal=2)
        result = validate_mushra_artifact(artifact)
        assert result.valid, f"Minimal gültiges Artefakt soll passen, Fehler: {result.errors}"

    def test_more_than_minimum_listeners_passes(self):
        artifact = _make_valid_artifact(n_listeners=15)
        result = validate_mushra_artifact(artifact)
        assert result.valid

    def test_more_than_minimum_scenarios_passes(self):
        artifact = _make_valid_artifact(n_scenarios=10)
        result = validate_mushra_artifact(artifact)
        assert result.valid

    def test_no_validation_errors_for_valid_artifact(self):
        artifact = _make_valid_artifact()
        result = validate_mushra_artifact(artifact)
        assert result.error_count == 0

    def test_valid_artifact_is_json_serializable(self):
        artifact = _make_valid_artifact()
        # must not raise
        json.dumps(artifact)


# ===========================================================================
# Klasse 3: Validierung ungültiger Artefakte
# ===========================================================================


class TestInvalidArtifactDetection:
    """Tests: Schema-Fehler werden korrekt erkannt."""

    def test_missing_n_listeners_fails(self):
        artifact = _make_valid_artifact()
        del artifact["n_listeners"]
        result = validate_mushra_artifact(artifact)
        assert not result.valid
        assert any("n_listeners" in e for e in result.errors)

    def test_too_few_listeners_fails(self):
        """7 Hörer < Minimum 8 → Fehler."""
        artifact = _make_valid_artifact(n_listeners=7)
        result = validate_mushra_artifact(artifact)
        assert not result.valid
        assert any("n_listeners" in e for e in result.errors)

    def test_too_few_scenarios_fails(self):
        """5 Szenarien < Minimum 6 → Fehler."""
        artifact = _make_valid_artifact(n_scenarios=5)
        result = validate_mushra_artifact(artifact)
        assert not result.valid
        assert any("Szenarien" in e or "scenarios" in e.lower() for e in result.errors)

    def test_too_few_vocal_scenarios_fails(self):
        """1 Vocal-Szenario < Minimum 2 → Fehler."""
        artifact = _make_valid_artifact(n_vocal=1)
        result = validate_mushra_artifact(artifact)
        assert not result.valid
        assert any("Vocal" in e or "vocal" in e.lower() for e in result.errors)

    def test_zero_vocal_scenarios_fails(self):
        artifact = _make_valid_artifact(n_vocal=0)
        result = validate_mushra_artifact(artifact)
        assert not result.valid

    def test_missing_aurik_version_fails(self):
        artifact = _make_valid_artifact()
        del artifact["aurik_version"]
        result = validate_mushra_artifact(artifact)
        assert not result.valid
        assert any("aurik_version" in e for e in result.errors)

    def test_empty_aurik_version_fails(self):
        artifact = _make_valid_artifact()
        artifact["aurik_version"] = ""
        result = validate_mushra_artifact(artifact)
        assert not result.valid

    def test_missing_scenarios_field_fails(self):
        artifact = _make_valid_artifact()
        del artifact["scenarios"]
        result = validate_mushra_artifact(artifact)
        assert not result.valid

    def test_scenario_with_missing_score_fails(self):
        artifact = _make_valid_artifact()
        del artifact["scenarios"][0]["score"]
        result = validate_mushra_artifact(artifact)
        assert not result.valid

    def test_scenario_with_out_of_range_score_fails(self):
        """Score > 100 ist ungültig."""
        artifact = _make_valid_artifact()
        artifact["scenarios"][0]["score"] = 101.0
        result = validate_mushra_artifact(artifact)
        assert not result.valid

    def test_scenario_with_negative_score_fails(self):
        artifact = _make_valid_artifact()
        artifact["scenarios"][0]["score"] = -5.0
        result = validate_mushra_artifact(artifact)
        assert not result.valid

    def test_scenario_nan_score_fails(self):
        """NaN-Score ist ungültig."""
        artifact = _make_valid_artifact()
        artifact["scenarios"][0]["score"] = float("nan")
        result = validate_mushra_artifact(artifact)
        assert not result.valid

    def test_scenario_missing_ci_fails(self):
        artifact = _make_valid_artifact()
        del artifact["scenarios"][0]["confidence_interval"]
        result = validate_mushra_artifact(artifact)
        assert not result.valid

    def test_scenario_missing_delta_fails(self):
        artifact = _make_valid_artifact()
        del artifact["scenarios"][0]["delta_to_previous"]
        result = validate_mushra_artifact(artifact)
        assert not result.valid

    def test_scenario_missing_is_vocal_fails(self):
        artifact = _make_valid_artifact()
        del artifact["scenarios"][0]["is_vocal"]
        result = validate_mushra_artifact(artifact)
        assert not result.valid

    def test_missing_overall_delta_fails(self):
        artifact = _make_valid_artifact()
        del artifact["overall_delta"]
        result = validate_mushra_artifact(artifact)
        assert not result.valid


# ===========================================================================
# Klasse 4: Negative-Regression-Warnung
# ===========================================================================


class TestRegressionWarning:
    """Tests: Negative overall_delta erzeugt eine Warnung."""

    def test_negative_overall_delta_generates_warning(self):
        artifact = _make_valid_artifact()
        artifact["overall_delta"] = -2.5
        result = validate_mushra_artifact(artifact)
        assert result.valid  # Warnung, kein Fehler
        assert any("Regression" in w or "delta" in w.lower() for w in result.warnings)

    def test_positive_overall_delta_no_regression_warning(self):
        artifact = _make_valid_artifact()
        artifact["overall_delta"] = 3.0
        result = validate_mushra_artifact(artifact)
        # Keine Regression-Warnung
        regression_warnings = [w for w in result.warnings if "Regression" in w]
        assert len(regression_warnings) == 0


# ===========================================================================
# Klasse 5: Artefakt-Pfad-Kontrakt
# ===========================================================================


class TestArtifactPathContract:
    """Tests: Der Ablageort des MUSHRA-Artefakts folgt dem Kontrakt."""

    def test_reports_directory_exists(self):
        """Das reports/-Verzeichnis muss existieren."""
        assert _REPORTS_DIR.is_dir(), (
            f"reports/-Verzeichnis nicht gefunden: {_REPORTS_DIR}\n"
            "§8.4: MUSHRA-Berichte müssen in reports/ abgelegt werden."
        )

    def test_mushra_artifact_path_is_under_reports(self):
        """Artefakt-Pfad muss unter reports/ liegen."""
        assert _REPORTS_DIR in _MUSHRA_ARTIFACT_PATH.parents or _MUSHRA_ARTIFACT_PATH.parent == _REPORTS_DIR

    def test_existing_mushra_artifact_is_valid_json(self):
        """Wenn mushra_artifact.json vorhanden, muss es valides JSON sein."""
        if not _MUSHRA_ARTIFACT_PATH.exists():
            if _strict_artifact_mode_enabled():
                pytest.fail(
                    "mushra_artifact.json nicht vorhanden — STRICT MODE aktiv. "
                    "Für Release-Audits muss reports/mushra_artifact.json vorhanden sein (§8.4)."
                )
            pytest.skip(
                "mushra_artifact.json nicht vorhanden — "
                "SKIP: Artefakt nur bei Kern-Änderungen erforderlich (§8.4). "
                "Vor Release muss reports/mushra_artifact.json angelegt werden."
            )

        content = _MUSHRA_ARTIFACT_PATH.read_text(encoding="utf-8")
        artifact = json.loads(content)  # raises on invalid JSON
        assert isinstance(artifact, dict)

    def test_existing_mushra_artifact_passes_schema_validation(self):
        """Wenn mushra_artifact.json vorhanden, muss es das Schema erfüllen (§8.4)."""
        if not _MUSHRA_ARTIFACT_PATH.exists():
            if _strict_artifact_mode_enabled():
                pytest.fail(
                    "mushra_artifact.json nicht vorhanden — STRICT MODE aktiv. "
                    "Für Release-Audits muss reports/mushra_artifact.json vorhanden sein (§8.4)."
                )
            pytest.skip(
                "mushra_artifact.json nicht vorhanden — SKIP: Artefakt nur bei Kern-Änderungen erforderlich (§8.4)."
            )

        content = _MUSHRA_ARTIFACT_PATH.read_text(encoding="utf-8")
        artifact = json.loads(content)
        result = validate_mushra_artifact(artifact)
        assert result.valid, "reports/mushra_artifact.json verletzt §8.4-Schema:\n" + "\n".join(
            f"  - {e}" for e in result.errors
        )


# ===========================================================================
# Klasse 6: Vollständiger Release-Check-Vertrag
# ===========================================================================


class TestReleaseCheckContract:
    """Tests: Der Release-Check-Vertrag: valides Artefakt = Release-Freigabe."""

    def test_valid_artifact_clears_release_blocker(self):
        """Ein gültiges Artefakt muss den Release-Blocker aufheben."""
        artifact = _make_valid_artifact()
        result = validate_mushra_artifact(artifact)
        assert result.valid is True
        assert result.error_count == 0

    def test_invalid_artifact_blocks_release(self):
        """Ein ungültiges Artefakt muss den Release blockieren."""
        artifact = _make_valid_artifact(n_listeners=3)  # zu wenig
        result = validate_mushra_artifact(artifact)
        assert result.valid is False

    def test_validate_fn_returns_correct_type(self):
        """validate_mushra_artifact muss MushraArtifactValidationResult zurückgeben."""
        artifact = _make_valid_artifact()
        result = validate_mushra_artifact(artifact)
        assert isinstance(result, MushraArtifactValidationResult)

    def test_validation_result_has_errors_field(self):
        artifact = _make_valid_artifact()
        result = validate_mushra_artifact(artifact)
        assert isinstance(result.errors, list)

    def test_validation_result_has_warnings_field(self):
        artifact = _make_valid_artifact()
        result = validate_mushra_artifact(artifact)
        assert isinstance(result.warnings, list)
