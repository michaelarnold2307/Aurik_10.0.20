"""Real-Audio Restoration Quality Gate — CI Enforcement (§15.2, §19).

Prüft den Report aus audit/real_audio_restoration_quality_gate.py und den
separaten Daily-UAT-Status. Fails CI bei verfehlten Manifest-Schwellen oder
veralteter bzw. inkonsistenter UAT-Evidenz.

Nutzung:
  pytest tests/normative/test_real_audio_restoration_quality_gate.py -v
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

REPORT_PATH = REPO_ROOT / "audit" / "real_audio_restoration_quality_report.json"
MANIFEST_PATH = REPO_ROOT / "audit" / "real_audio_strategy_golden_manifest.json"
EXECUTION_PATH = REPO_ROOT / "audit" / "real_audio_execution_golden_report.json"
DAILY_STATUS = REPO_ROOT / "audit" / "daily_real_audio_gate_status.json"


def _gate_report_exists() -> bool:
    return REPORT_PATH.exists()


def _load_gate_report() -> dict | None:
    if not REPORT_PATH.exists():
        return None
    try:
        return json.loads(REPORT_PATH.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
    except (json.JSONDecodeError, OSError):
        return None


def _load_daily_status() -> dict | None:
    if not DAILY_STATUS.exists():
        return None
    try:
        return json.loads(DAILY_STATUS.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
    except (json.JSONDecodeError, OSError):
        return None


def _quality_thresholds() -> dict:
    if not MANIFEST_PATH.exists():
        pytest.fail(f"Golden-Set-Manifest fehlt: {MANIFEST_PATH}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    thresholds = manifest.get("restoration_quality_thresholds") if isinstance(manifest, dict) else None
    if not isinstance(thresholds, dict):
        pytest.fail("restoration_quality_thresholds fehlt im Golden-Set-Manifest")
    return thresholds


# ── Core Quality Gate Tests ────────────────────────────────────────────────


@pytest.mark.quality_gate
@pytest.mark.real_audio
class TestRealAudioQualityGate:
    """Prüft den Real-Audio Quality Gate Report."""

    @pytest.fixture(autouse=True)
    def report(self) -> dict:
        data = _load_gate_report()
        if data is None:
            pytest.skip(
                "Kein Quality-Gate-Report gefunden. Führe 'python audit/real_audio_restoration_quality_gate.py' aus."
            )
        return data

    @pytest.fixture
    def gate(self, report: dict) -> dict:
        return report.get("gate", {})  # type: ignore[return-value]

    def test_gate_exists(self, gate: dict):
        """Quality Gate Report muss existieren und ein 'gate'-Feld enthalten."""
        assert gate, "Gate-Feld fehlt im Report"

    def test_gate_passed(self, gate: dict):
        """A high average HPI cannot waive failures in other mandatory dimensions."""
        fail_reasons = gate.get("fail_reasons", [])
        reason_text = "\n  ".join(fail_reasons) if fail_reasons else "keine Gründe angegeben"
        assert gate.get("passed") is True, (
            f"Real-Audio Quality Gate NICHT bestanden:\n"
            f"  HPI-Avg: {gate.get('hpi_average', '?')}\n"
            f"  Musical-Goal-Rate: {gate.get('musical_goal_case_pass_rate', 0):.1%}\n"
            f"  Noise-Texture-Rate: {gate.get('noise_texture_case_pass_rate', 0):.1%}\n"
            f"  Goosebumps-Rate: {gate.get('goosebumps_case_pass_rate', 0):.1%}\n"
            f"  Gründe: {reason_text}\n"
            f"  Actions: {gate.get('prioritized_actions', [])}"
        )

    def test_minimum_case_count(self, gate: dict):
        """Der Case-Umfang folgt dem Golden-Set-Manifest."""
        cases = gate.get("real_audio_cases", 0)
        minimum = int(_quality_thresholds()["min_real_audio_cases"])
        assert cases >= minimum, (
            f"Nur {cases} Real-Audio-Cases (mindestens {minimum} erforderlich). "
            f"Nutze scripts/generate_corpus_from_public_domain.py."
        )

    def test_minimum_vocal_cases(self, gate: dict):
        """Der Vokal-Case-Umfang folgt dem Golden-Set-Manifest."""
        vocal_cases = gate.get("vocal_cases", 0)
        minimum = int(_quality_thresholds()["min_vocal_cases"])
        assert vocal_cases >= minimum, f"Nur {vocal_cases} Vokal-Cases (mindestens {minimum} erforderlich)."

    def test_exact_golden_set_case_coverage(self, gate: dict):
        """Kein Golden-Set-Fall darf fehlen, dupliziert oder ersetzt werden."""
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        expected_cases = manifest.get("cases")
        assert isinstance(expected_cases, list) and expected_cases
        assert gate.get("case_id_coverage_complete") is True
        assert gate.get("case_id_coverage_rate") == 1.0
        assert gate.get("expected_case_count") == len(expected_cases)

    def test_hpi_above_minimum(self, gate: dict):
        """HPI muss vorliegen und die im Golden-Set definierte Schwelle erreichen."""
        hpi = gate.get("hpi_average")
        assert isinstance(hpi, (int, float)) and math.isfinite(float(hpi)), "HPI-Average fehlt oder ist nicht endlich"
        minimum = float(_quality_thresholds()["min_hpi_average"])
        assert hpi >= minimum, f"HPI-Average {hpi:.3f} < {minimum:.3f}. Siehe diagnose_gate_failures.py."

    def test_quality_estimate_is_report_only(self, gate: dict):
        """Quality-Estimate-Average wird nur als endlicher Diagnosewert berichtet."""
        assert "min_quality_estimate_average" not in _quality_thresholds()
        qe = gate.get("quality_estimate_average")
        if qe is None:
            return
        assert math.isfinite(qe) and 0.0 <= qe <= 1.0, f"Ungültiger Quality-Estimate-Wert: {qe!r}"

    def test_noise_texture_above_minimum(self, gate: dict):
        """Noise-Texture-Pass-Rate folgt der Golden-Set-Schwelle."""
        rate = gate.get("noise_texture_case_pass_rate", 0)
        minimum = float(_quality_thresholds()["min_noise_texture_case_pass_rate"])
        assert rate >= minimum, f"Noise-Texture-Pass-Rate {rate:.1%} < {minimum:.1%}."

    def test_goosebumps_above_minimum(self, gate: dict):
        """Goosebumps-Pass-Rate folgt der Golden-Set-Schwelle."""
        rate = gate.get("goosebumps_case_pass_rate", 0)
        minimum = float(_quality_thresholds()["min_goosebumps_case_pass_rate"])
        assert rate >= minimum, f"Goosebumps-Pass-Rate {rate:.1%} < {minimum:.1%}."

    def test_external_benchmark_case_count(self, gate: dict):
        """Externe Vergleichsevidenz folgt der Golden-Set-Vorgabe."""
        cases = gate.get("external_benchmark_cases", 0)
        minimum = int(_quality_thresholds()["min_external_benchmark_cases"])
        assert cases >= minimum, f"Nur {cases} externe Benchmark-Cases (mindestens {minimum} erforderlich)."


# ── Daily Status Tests ─────────────────────────────────────────────────────


@pytest.mark.quality_gate
@pytest.mark.daily
class TestDailyGateStatus:
    """Prüft den täglichen Gate-Status (via audit/daily_real_audio_gate.py)."""

    @pytest.fixture(autouse=True)
    def status(self) -> dict:
        data = _load_daily_status()
        if data is None:
            pytest.skip("Kein täglicher Gate-Status gefunden.")
        return data  # type: ignore[return-value]

    def test_daily_gate_stored_recently(self, status: dict):
        """Evidenz-Frische: die UAT-ANALYSE muss aktuell sein (D-K3-34).

        Der Reporter `audit/daily_real_audio_gate.py` aggregiert vorhandene
        `uat_results_*.json` und stempelt `generated_at` neu — das ist die
        Reporter-Laufzeit, nicht frische Qualitäts-Evidenz. Maßgeblich für
        echte Evidenz ist die UAT-Analyse selbst (`latest.date`):
        Wirkung statt Formulierung (D-K3-34).
        """
        from datetime import date, datetime, timedelta, timezone

        latest = status.get("latest") or {}
        ds = latest.get("date")
        if not ds:
            pytest.fail("Daily-Status hat keine UAT-Analyse (latest.date fehlt)")
        analysis_age = datetime.now(timezone.utc).date() - date.fromisoformat(str(ds))
        assert analysis_age < timedelta(days=2), (
            f"UAT-Analyse ist {analysis_age.days} Tage alt (maximal 48h). "
            f"Führe audit/uat_report_generator.py aus (echte Evidenz erneuern)."
        )

    def test_daily_status_consistent_with_source_uat(self, status: dict):
        """Daily-Zusammenfassung muss exakt zur referenzierten UAT-Quelle passen."""
        from audit.daily_real_audio_gate import _extract_r5_r12_pass_counts, _safe_ts

        latest = status.get("latest")
        assert isinstance(latest, dict), "Daily-Status hat keinen latest-Datensatz"
        source_name = Path(str(latest.get("source_file", ""))).name
        assert source_name.startswith("uat_results_"), f"Ungültige UAT-Quelle: {source_name!r}"
        source_path = REPO_ROOT / "audit" / source_name
        assert source_path.is_file(), f"Referenzierte UAT-Quelle fehlt: {source_path}"
        source = json.loads(source_path.read_text(encoding="utf-8"))
        summary = source.get("summary") if isinstance(source, dict) else None
        assert isinstance(summary, dict), "UAT-Quelle hat kein summary-Feld"

        source_date = _safe_ts(source.get("generated_at")).date().isoformat()
        assert latest.get("date") == source_date
        assert latest.get("recommendation") == summary.get("recommendation", "UNKNOWN")
        assert latest.get("gates") == {
            "passed": int(summary.get("gates_passed", 0)),
            "total": int(summary.get("gates_total", 0)),
        }
        failed_results = int(summary.get("criteria_failed", 0)) + int(summary.get("gates_failed", 0))
        errored_results = int(summary.get("criteria_errors", 0)) + int(summary.get("gates_errors", 0))
        if failed_results or errored_results:
            assert latest.get("recommendation") == "NO-GO", (
                "UAT-Quelle meldet fehlgeschlagene oder nicht auswertbare Kriterien, "
                "darf aber nicht als GO/CONDITIONAL zusammengefasst werden"
            )
        r5_r12_passed, r5_r12_total = _extract_r5_r12_pass_counts(source)
        assert latest.get("r5_r12", {}).get("passed") == r5_r12_passed
        assert latest.get("r5_r12", {}).get("total") == r5_r12_total
