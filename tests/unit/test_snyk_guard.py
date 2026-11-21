from pathlib import Path

import pytest

import scripts.snyk_guard as snyk_guard


def _install_runner(monkeypatch, results):
    calls = []
    responses = iter(results)

    def fake_run(command, cwd=None):
        calls.append((list(command), cwd))
        return next(responses)

    monkeypatch.setattr(snyk_guard, "run", fake_run)
    return calls


def test_clean_scan_passes_without_running_fix(monkeypatch):
    calls = _install_runner(monkeypatch, [(0, "clean")])

    assert snyk_guard.main() == 0
    assert calls == [(["snyk", "test", "--severity-threshold=high"], str(Path.cwd()))]


def test_findings_are_fixed_and_retested(monkeypatch):
    calls = _install_runner(monkeypatch, [(1, "findings"), (0, "fixed"), (0, "clean")])

    assert snyk_guard.main() == 0
    assert [command for command, _ in calls] == [
        ["snyk", "test", "--severity-threshold=high"],
        ["snyk", "fix"],
        ["snyk", "test", "--severity-threshold=high"],
    ]


def test_remaining_findings_block_commit(monkeypatch):
    _install_runner(monkeypatch, [(1, "findings"), (0, "fixed"), (1, "still found")])

    assert snyk_guard.main() == 1


@pytest.mark.parametrize("exit_code", [2, 3, 127])
def test_scan_errors_fail_closed_without_fix(monkeypatch, exit_code):
    calls = _install_runner(monkeypatch, [(exit_code, "scan error")])

    assert snyk_guard.main() == 1
    assert len(calls) == 1


def test_missing_snyk_executable_fails_closed(monkeypatch):
    def missing_executable(*args, **kwargs):
        raise FileNotFoundError("snyk not found")

    monkeypatch.setattr(snyk_guard.subprocess, "run", missing_executable)

    assert snyk_guard.run(["snyk", "test"]) == (127, "snyk not found")
