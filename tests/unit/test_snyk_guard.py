import sys
from pathlib import Path

import pytest

import scripts.snyk_guard as snyk_guard


def _test_command(manifest):
    return [
        "snyk",
        "test",
        f"--file={manifest}",
        "--package-manager=pip",
        f"--command={sys.executable}",
        "--severity-threshold=high",
    ]


def _install_runner(monkeypatch, results):
    calls = []
    responses = iter(results)

    def fake_run(command, cwd=None):
        calls.append((list(command), cwd))
        return next(responses)

    monkeypatch.setattr(snyk_guard, "run", fake_run)
    return calls


def test_clean_scans_pass_for_both_supported_manifests(monkeypatch):
    calls = _install_runner(monkeypatch, [(0, "clean"), (0, "clean")])

    assert snyk_guard.main([]) == 0
    assert calls == [(list(_test_command(manifest)), str(Path.cwd())) for manifest in snyk_guard.SUPPORTED_MANIFESTS]


def test_findings_block_and_other_manifests_are_still_scanned(monkeypatch):
    calls = _install_runner(monkeypatch, [(1, "findings"), (0, "clean")])

    assert snyk_guard.main([]) == 1
    assert [command for command, _ in calls] == [
        list(_test_command(manifest)) for manifest in snyk_guard.SUPPORTED_MANIFESTS
    ]


def test_reports_are_written_to_configured_directory(monkeypatch, tmp_path):
    report_dir = tmp_path / "snyk-reports"
    monkeypatch.setenv("SNYK_REPORT_DIR", str(report_dir))
    calls = _install_runner(monkeypatch, [(0, "clean")])

    assert snyk_guard.main(["--manifest", snyk_guard.SUPPORTED_MANIFESTS[0]]) == 0
    assert report_dir.is_dir()
    assert calls[0][0][-1] == f"--json-file-output={report_dir / 'requirements_aurik.json'}"


@pytest.mark.parametrize("exit_code", [2, 3, 127])
def test_scan_errors_fail_closed_and_continue_scanning(monkeypatch, exit_code):
    calls = _install_runner(monkeypatch, [(exit_code, "scan error"), (0, "clean")])

    assert snyk_guard.main([]) == 1
    assert len(calls) == len(snyk_guard.SUPPORTED_MANIFESTS)


def test_missing_snyk_executable_fails_closed(monkeypatch):
    def missing_executable(*args, **kwargs):
        raise FileNotFoundError("snyk not found")

    monkeypatch.setattr(snyk_guard.subprocess, "run", missing_executable)

    assert snyk_guard.run(["snyk", "test"]) == (127, "snyk not found")
