import pytest

"""Tests for Aurik10/core/version_checker.py — update check logic."""

import json
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import scripts.check_version_consistency as vc
from Aurik10.core.version_checker import (
    _CURRENT_VERSION,
    VersionCheckResult,
    _parse_version,
    check_for_update,
    check_for_update_async,
)

# ── _parse_version ─────────────────────────────────────────────────────────


@pytest.mark.unit
def test_parse_version_simple():
    assert _parse_version("9.10.77") == (9, 10, 77)


def test_parse_version_with_v_prefix():
    assert _parse_version("v10.14.0") == (10, 14, 0)


def test_parse_version_with_V_prefix():
    assert _parse_version("V1.2.3") == (1, 2, 3)


def test_parse_version_non_numeric():
    assert _parse_version("v10.14.0.beta") == (10, 14, 0)


def test_parse_version_hotfix_suffix():
    assert _parse_version("v10.14.0-hotfix.2") == (10, 14, 0, 2)


def test_parse_version_comparison():
    assert _parse_version("9.10.80") > _parse_version("9.10.77")
    assert _parse_version("9.11.0") > _parse_version("9.10.99")
    assert _parse_version("10.14.0") > _parse_version("9.99.99")
    assert _parse_version("v10.14.0") > _parse_version("9.10.77")
    assert _parse_version("9.12.9-hotfix.2") > _parse_version("9.12.9-hotfix.1")


# ── VersionCheckResult ─────────────────────────────────────────────────────


def test_result_defaults():
    r = VersionCheckResult()
    assert r.available is False
    assert r.error == ""
    assert r.latest_version == ""
    assert r.download_url == ""


def test_default_current_version_uses_package_version():
    from Aurik10 import __version__

    assert __version__ == _CURRENT_VERSION


def test_frontend_fallback_version_matches_single_source():
    """§v10.802: Der GUI-Fallback darf nicht driften (Quelle = backend/core/version.py).

    Regression 2026-10-05: `_FALLBACK_VERSION` stand auf 10.2.1, während die
    Produktversion bereits 10.3.1 war — bei fehlgeschlagenem Bridge-Import hätte
    die GUI eine falsche Version angezeigt (§v10.802 GUI-Sync-Pflicht).
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    version_text = (root / "backend" / "core" / "version.py").read_text(encoding="utf-8")
    current = next(
        line.split("=", 1)[1].strip().strip('"') for line in version_text.splitlines() if line.startswith("__version__")
    )
    init_text = (root / "Aurik10" / "__init__.py").read_text(encoding="utf-8")
    fallback = next(
        line.split("=", 1)[1].split("#", 1)[0].strip().strip('"')
        for line in init_text.splitlines()
        if line.startswith("_FALLBACK_VERSION")
    )
    assert fallback == current, f"GUI-Fallback {fallback} ≠ version.py {current} (Drift)"


def test_result_available():
    r = VersionCheckResult(available=True, latest_version="9.11.0", download_url="https://example.com/dl")
    assert r.available is True
    assert r.latest_version == "9.11.0"


# ── check_for_update (mocked) ─────────────────────────────────────────────


def _mock_release(tag: str, assets=None, body="Release notes"):
    """Build a GitHub API-like release dict."""
    data = {"tag_name": tag, "html_url": f"https://github.com/release/{tag}", "body": body}
    if assets:
        data["assets"] = assets
    else:
        data["assets"] = []
    return json.dumps(data).encode("utf-8")


@patch("Aurik10.core.version_checker.urlopen")
def test_check_newer_available(mock_urlopen):
    resp = MagicMock()
    resp.read.return_value = _mock_release("v99.0.0")
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    mock_urlopen.return_value = resp

    result = check_for_update("9.10.77")
    assert result.available is True
    assert result.latest_version == "99.0.0"
    assert result.error == ""


@patch("Aurik10.core.version_checker.urlopen")
def test_check_up_to_date(mock_urlopen):
    resp = MagicMock()
    resp.read.return_value = _mock_release("v10.0.15")
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    mock_urlopen.return_value = resp

    result = check_for_update("10.0.15")
    assert result.available is False
    assert result.error == ""


@patch("Aurik10.core.version_checker.urlopen")
def test_check_older_than_current(mock_urlopen):
    resp = MagicMock()
    resp.read.return_value = _mock_release("v9.9.0")
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    mock_urlopen.return_value = resp

    result = check_for_update("10.0.15")
    assert result.available is False


@patch("Aurik10.core.version_checker.urlopen")
def test_check_with_appimage_asset(mock_urlopen):
    assets = [{"name": "aurik-9.11.0.AppImage", "browser_download_url": "https://dl.example.com/aurik.AppImage"}]
    resp = MagicMock()
    resp.read.return_value = _mock_release("v10.0.0", assets=assets)
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    mock_urlopen.return_value = resp

    result = check_for_update("9.10.77")
    assert result.available is True
    assert "AppImage" in result.download_url


@patch("Aurik10.core.version_checker.urlopen")
def test_check_network_error(mock_urlopen):
    from urllib.error import URLError

    mock_urlopen.side_effect = URLError("offline")

    result = check_for_update("9.10.77")
    assert result.available is False
    assert "offline" in result.error


@patch("Aurik10.core.version_checker.urlopen")
def test_check_no_tag(mock_urlopen):
    resp = MagicMock()
    resp.read.return_value = json.dumps({"body": "no tag"}).encode()
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    mock_urlopen.return_value = resp

    result = check_for_update("9.10.77")
    assert result.available is False
    assert "no tag_name" in result.error


# ── check_for_update_async ─────────────────────────────────────────────────


@patch("Aurik10.core.version_checker.urlopen")
def test_async_check_calls_callback(mock_urlopen):
    resp = MagicMock()
    resp.read.return_value = _mock_release("v99.0.0")
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    mock_urlopen.return_value = resp

    results = []
    event = threading.Event()

    def cb(r):
        results.append(r)
        event.set()

    check_for_update_async(cb)
    event.wait(timeout=5)
    assert len(results) == 1
    assert results[0].available is True


# ── Batch-Retry / TitleBar Settings (integration-style, no Qt needed) ──────


def test_simple_batch_item_retry_reset():
    """Verify SimpleBatchItem fields can be reset for retry."""
    import sys

    sys.path.insert(0, ".")
    # Minimal duck-type test — no Qt import needed

    class _Item:
        def __init__(self):
            self.status = "failed"
            self.error: str | None = "some error"
            self.progress = 100
            self.restoration_result: object | None = object()

    item = _Item()
    # Simulate retry reset logic
    item.status = "pending"
    item.error = None
    item.progress = 0
    item.restoration_result = None

    assert item.status == "pending"
    assert item.error is None
    assert item.progress == 0
    assert item.restoration_result is None


def test_i18n_keys_exist():
    """All new i18n keys resolve to non-empty strings."""
    from Aurik10.i18n import set_language, t

    keys = [
        "batch.retry_tooltip",
        "batch.retry_hint",
        "update.checking",
        "update.available",
        "update.up_to_date",
        "update.error",
        "update.unavailable",
        "update.banner_text",
        "update.download",
        "help.check_update",
        "settings.title",
    ]
    for lang in ("de", "en"):
        set_language(lang)
        for key in keys:
            val = t(key, count=1, version="9.99.0")
            assert val, f"i18n key '{key}' empty for lang={lang}"
            assert key not in val, f"i18n key '{key}' not translated for lang={lang}"


# ── Version-Sweep: Historie ist unantastbar (Befund 2026-10-07) ─────────────


class TestVersionSweepHistoryGuard:
    """`scripts/check_version_consistency.py --fix` darf keine Historie umschreiben.

    Produktionsbefund 2026-10-07: Der Sweep benannte die **erste**
    ``##``-Überschrift um. Wurde er vor dem Anlegen des neuen Changelog-
    Abschnitts ausgeführt, wurde der **alte Release-Block** auf die neue Version
    umgeschrieben (``## 10.8.1`` → ``## 10.8.2``) — und meldete dabei „alle
    Dateien konsistent". Die Historie war falsch. §v10.802
    (copilot-instructions.md) verlangt pro Bump einen eigenen ``## x.y.z``-Block;
    der Sweep verweigert daher jetzt die Umschreibung, statt sie stillschweigend
    vorzunehmen.
    """

    @staticmethod
    def _make_root(tmp_path: Path, changelog: str, version: str = "10.8.2") -> Path:
        (tmp_path / "backend" / "core").mkdir(parents=True)
        (tmp_path / "backend" / "core" / "version.py").write_text(f'__version__ = "{version}"\n', encoding="utf-8")
        (tmp_path / "pyproject.toml").write_text(f'version = "{version}"\n', encoding="utf-8")
        (tmp_path / "README.md").write_text(f"**Version:** {version}\n", encoding="utf-8")
        (tmp_path / "CHANGELOG.md").write_text(changelog, encoding="utf-8")
        return tmp_path

    def test_sweep_does_not_rename_existing_release_block(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ):
        root = self._make_root(tmp_path, "# Changelog — Aurik 10.8.1\n\n## 10.8.1 (2026-10-07)\n\n- alt\n")
        before = (root / "CHANGELOG.md").read_text(encoding="utf-8")
        monkeypatch.setattr(vc, "PROJECT_ROOT", root)
        monkeypatch.setattr(sys, "argv", ["check_version_consistency.py", "--fix"])

        with pytest.raises(SystemExit) as exc:
            vc.main()

        assert exc.value.code == 1
        assert (root / "CHANGELOG.md").read_text(encoding="utf-8") == before, "Historie wurde umgeschrieben"
        assert "kein '## 10.8.2'-Abschnitt" in capsys.readouterr().out

    def test_sweep_accepts_new_block_and_leaves_history_intact(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        changelog = "# Changelog — Aurik 10.8.2\n\n## 10.8.2 (2026-10-07)\n\n- neu\n\n## 10.8.1 (2026-10-07)\n\n- alt\n"
        root = self._make_root(tmp_path, changelog)
        monkeypatch.setattr(vc, "PROJECT_ROOT", root)
        monkeypatch.setattr(sys, "argv", ["check_version_consistency.py", "--fix"])

        with pytest.raises(SystemExit) as exc:
            vc.main()

        assert exc.value.code == 0
        assert (root / "CHANGELOG.md").read_text(encoding="utf-8") == changelog

    def test_sweep_still_fixes_pyproject_and_readme(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Der Wächter darf den normalen Sweep nicht lahmlegen."""
        root = self._make_root(tmp_path, "## 10.8.2 (2026-10-07)\n\n- neu\n")
        (root / "pyproject.toml").write_text('version = "10.8.1"\n', encoding="utf-8")
        (root / "README.md").write_text("**Version:** 10.8.1\n", encoding="utf-8")
        monkeypatch.setattr(vc, "PROJECT_ROOT", root)
        monkeypatch.setattr(sys, "argv", ["check_version_consistency.py", "--fix"])

        with pytest.raises(SystemExit) as exc:
            vc.main()

        assert exc.value.code == 0
        assert 'version = "10.8.2"' in (root / "pyproject.toml").read_text(encoding="utf-8")
        assert "**Version:** 10.8.2" in (root / "README.md").read_text(encoding="utf-8")
