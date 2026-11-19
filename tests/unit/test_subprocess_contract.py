"""
§v10.50 Regression-Tests: Subprocess-Vertrag (§V34), WAV-Retry (§V35), Scipy-Unpack (§V36).

Diese Tests sichern die Bugfixes 7-10 aus dem Orchestrator-Runtime-Log 2026-04-25 ab.
"""

import pathlib
import textwrap


class TestSubprocessContract:
    """§V34: sys.executable in Orchestrator-Subprozessen."""

    def test_orchestrate_quality_monitoring_uses_sys_executable(self):
        """Bug 7/9: orchestrator startet Analyzer + Monitor mit sys.executable."""
        path = pathlib.Path(__file__).parent.parent.parent / "scripts" / "orchestrate_quality_monitoring.py"
        assert path.exists(), f"orchestrate_quality_monitoring.py nicht gefunden: {path}"
        source = path.read_text(encoding="utf-8")

        # Darf KEIN hartcodiertes .venv_aurik in subprocess.Popen-Kontext haben
        assert '".venv_aurik"' not in source, (
            "§V34 verletzt: orchestrator enthält hartcodiertes .venv_aurik — muss sys.executable sein"
        )
        # Muss sys.executable referenzieren
        assert "sys.executable" in source, "§V34 verletzt: orchestrator referenziert sys.executable nicht"

    def test_frontend_with_analysis_uses_sys_executable(self):
        """Bug 7/9: frontend_with_analysis startet Analyzer mit sys.executable."""
        path = pathlib.Path(__file__).parent.parent.parent / "scripts" / "frontend_with_analysis.py"
        assert path.exists(), f"frontend_with_analysis.py nicht gefunden: {path}"
        source = path.read_text(encoding="utf-8")

        assert '".venv_aurik"' not in source, "§V34 verletzt: frontend_with_analysis enthält hartcodiertes .venv_aurik"
        assert "sys.executable" in source, "§V34 verletzt: frontend_with_analysis referenziert sys.executable nicht"

    def test_simple_restoration_monitor_uses_sys_executable(self):
        """Bug 7/9: simple_restoration_monitor startet Monitor mit sys.executable."""
        path = pathlib.Path(__file__).parent.parent.parent / "scripts" / "simple_restoration_monitor.py"
        assert path.exists(), f"simple_restoration_monitor.py nicht gefunden: {path}"
        source = path.read_text(encoding="utf-8")

        assert '".venv_aurik"' not in source, (
            "§V34 verletzt: simple_restoration_monitor enthält hartcodiertes .venv_aurik"
        )


class TestWavRetryContract:
    """§V35: Retry-Logik in WAV-Loadern."""

    def test_pegelexplosion_monitor_has_retry(self):
        """Bug 7: Pegelexplosion-Monitor hat 3× Retry bei WAV-Load-Fehlern."""
        path = pathlib.Path(__file__).parent.parent.parent / "scripts" / "pegelexplosion_monitor.py"
        assert path.exists(), f"pegelexplosion_monitor.py nicht gefunden: {path}"
        source = path.read_text(encoding="utf-8")

        assert "_max_retries = 3" in source or "_max_retries=3" in source, (
            "§V35 verletzt: pegelexplosion_monitor hat keinen 3× Retry-Loop"
        )
        assert "unpack" in source, "§V35 verletzt: Retry-Loop prüft nicht auf 'unpack'-Fehler"

    def test_continuous_deep_analysis_has_retry(self):
        """Bug 7: continuous_deep_analysis hat 3× Retry bei Audio-Import."""
        path = pathlib.Path(__file__).parent.parent.parent / "scripts" / "continuous_deep_analysis.py"
        assert path.exists(), f"continuous_deep_analysis.py nicht gefunden: {path}"
        source = path.read_text(encoding="utf-8")

        assert "_max_load_retries" in source, (
            "§V35 verletzt: continuous_deep_analysis hat keinen Retry-Loop für Audio-Load"
        )
        assert "unpack" in source, "§V35 verletzt: Retry-Loop prüft nicht auf 'unpack'-Fehler"


class TestScipyUnpackContract:
    """§V36: Robustes scipy.io.wavfile.read()-Unpack."""

    def test_meta_router_no_tuple_destructure_wavfile(self):
        """Bug 7: meta_router._load_audio() verwendet kein Tuple-Destructuring für wavfile.read."""
        path = pathlib.Path(__file__).parent.parent.parent / "backend" / "meta_router.py"
        assert path.exists(), f"meta_router.py nicht gefunden: {path}"
        source = path.read_text(encoding="utf-8")

        # Darf NICHT sr, data = wavfile.read(...) verwenden
        assert "sr, data = wavfile.read" not in source, (
            "§V36 verletzt: meta_router verwendet unsicheres Tuple-Unpack für wavfile.read()"
        )
        # Muss isinstance-Prüfung haben
        assert "isinstance(_wf_result, tuple)" in source or "isinstance(_wf_result,tuple)" in source, (
            "§V36 verletzt: meta_router hat keine isinstance-Prüfung für wavfile.read()-Rückgabe"
        )


class TestAntiRegressionGateCoverage:
    """Anti-Regression-Gate deckt Bugs 10-12 ab."""

    def test_anti_regression_gate_has_bug10_check(self):
        """Bug 10 Check existiert in anti_regression_gate.py."""
        path = pathlib.Path(__file__).parent.parent.parent / "scripts" / "compliance" / "anti_regression_gate.py"
        assert path.exists(), f"anti_regression_gate.py nicht gefunden: {path}"
        source = path.read_text(encoding="utf-8")

        assert "check_hardcoded_venv_in_subprocess" in source, (
            "Bug 10 fehlt: check_hardcoded_venv_in_subprocess nicht in anti_regression_gate.py"
        )
        assert "check_hardcoded_venv_in_subprocess(fp)" in source, (
            "Bug 10 nicht registriert: check_hardcoded_venv_in_subprocess fehlt in main()"
        )

    def test_anti_regression_gate_has_bug11_check(self):
        """Bug 11 Check existiert in anti_regression_gate.py."""
        path = pathlib.Path(__file__).parent.parent.parent / "scripts" / "compliance" / "anti_regression_gate.py"
        source = path.read_text(encoding="utf-8")

        assert "check_missing_wav_retry" in source, (
            "Bug 11 fehlt: check_missing_wav_retry nicht in anti_regression_gate.py"
        )
        assert "check_missing_wav_retry(fp)" in source, (
            "Bug 11 nicht registriert: check_missing_wav_retry fehlt in main()"
        )

    def test_anti_regression_gate_has_bug12_check(self):
        """Bug 12 Check existiert in anti_regression_gate.py."""
        path = pathlib.Path(__file__).parent.parent.parent / "scripts" / "compliance" / "anti_regression_gate.py"
        source = path.read_text(encoding="utf-8")

        assert "check_unsafe_wavfile_unpack" in source, (
            "Bug 12 fehlt: check_unsafe_wavfile_unpack nicht in anti_regression_gate.py"
        )
        assert "check_unsafe_wavfile_unpack(fp)" in source, (
            "Bug 12 nicht registriert: check_unsafe_wavfile_unpack fehlt in main()"
        )

    def test_sys_executable_hook_exists(self):
        """Der check_sys_executable Pre-Commit-Hook existiert."""
        path = pathlib.Path(__file__).parent.parent.parent / "scripts" / "compliance" / "check_sys_executable.py"
        assert path.exists(), "§V34 Pre-Commit-Hook fehlt: check_sys_executable.py nicht vorhanden"
        source = path.read_text(encoding="utf-8")

        assert "sys.executable" in source, "check_sys_executable.py erwähnt sys.executable nicht"
        assert "find_hardcoded_venv" in source, "check_sys_executable.py hat keine find_hardcoded_venv-Funktion"

    def test_pre_commit_config_has_sys_executable_hook(self):
        """Der check_sys_executable Hook ist in .pre-commit-config.yaml registriert."""
        path = pathlib.Path(__file__).parent.parent.parent / ".pre-commit-config.yaml"
        assert path.exists(), f".pre-commit-config.yaml nicht gefunden: {path}"
        source = path.read_text(encoding="utf-8")

        assert "aurik-sys-executable-guard" in source, "aurik-sys-executable-guard fehlt in .pre-commit-config.yaml"
        assert "check_sys_executable.py" in source, (
            "check_sys_executable.py nicht in .pre-commit-config.yaml referenziert"
        )

    def test_spec_18_exists(self):
        """Spec §18 existiert und enthält alle Vorgaben."""
        path = (
            pathlib.Path(__file__).parent.parent.parent
            / ".github"
            / "specs"
            / "18_subprocess_contract_and_wav_retry.md"
        )
        assert path.exists(), f"Spec 18 nicht gefunden: {path}"
        source = path.read_text(encoding="utf-8")

        for required in ["§V34", "§V35", "§V36", "sys.executable", "wavfile.read", "Retry"]:
            assert required in source, f"Spec 18 unvollständig: '{required}' fehlt"
