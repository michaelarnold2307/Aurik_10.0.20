"""tests/unit/test_model_downloader_release.py — §13.3 Release-Auslieferung.

Testet die GitHub-Release-Auslieferung großer Modelle: Manifest-Parsing der
neuen Felder, Part-Download + Reassembly (gemockter Download), Part-/Gesamt-
SHA256-Verifikation, Fehlerfälle. Kein Netzwerk, kein Datei-I/O außer tmp_path.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from backend.core import model_downloader as md
from backend.core.model_downloader import ModelDownloader, ModelEntry


def _entry(tmp_path: Path, parts: list[bytes]) -> tuple[ModelEntry, Path]:
    """Baut einen Release-Eintrag mit synthetischen Parts + Zielpfad."""
    assets = [f"m.part{i:02d}" for i in range(len(parts))]
    whole = b"".join(parts)
    entry = ModelEntry(
        name="test_model",
        bundled=False,
        bundled_path="models/test/m.bin",
        sha256=hashlib.sha256(whole).hexdigest(),
        size_bytes=len(whole),
        required=False,
        fallback="dsp",
        delivery="release",
        release_tag="models-10.2.0",
        assets=assets,
        part_sha256={a: hashlib.sha256(p).hexdigest() for a, p in zip(assets, parts)},
        part_size_bytes={a: len(p) for a, p in zip(assets, parts)},
    )
    return entry, tmp_path / "models" / "test" / "m.bin"


@pytest.mark.unit
class TestReleaseDelivery:
    def test_manifest_parses_release_fields(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.json"
        manifest.write_text(
            '{"models": [{"name": "x", "bundled": true, "bundled_path": "models/x.bin", '
            '"sha256": "aa", "size_bytes": 1, "delivery": "release", '
            '"release_tag": "models-10.2.0", "assets": ["x.bin"], '
            '"part_sha256": {"x.bin": "bb"}, "part_size_bytes": {"x.bin": 1}}]}',
            encoding="utf-8",
        )
        orig = ModelDownloader.MANIFEST
        ModelDownloader.MANIFEST = manifest
        try:
            dl = ModelDownloader()
            dl._manifest_entries = dl._parse_manifest()
            entry = dl.get_entry("x")
            assert entry is not None
            assert entry.delivery == "release"
            assert entry.release_tag == "models-10.2.0"
            assert entry.assets == ["x.bin"]
            assert entry.part_sha256 == {"x.bin": "bb"}
        finally:
            ModelDownloader.MANIFEST = orig

    def test_download_and_reassembly(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        entry, target = _entry(tmp_path, [b"A" * 1000, b"B" * 1000, b"C" * 500])

        def fake_download(url: str, target: Path, **kwargs) -> bool:  # noqa: ANN003
            part = next((a for a in entry.assets if url.endswith(a)), None)
            assert part is not None, url
            target.write_bytes(entry.part_sha256 and b"" or b"")
            # Inhalt über die Part-Größe schreiben (wie der echte Download)
            idx = entry.assets.index(part)
            sizes = [entry.part_size_bytes[a] for a in entry.assets]
            prefix_len = sum(sizes[:idx])
            whole = b"".join([b"A" * 1000, b"B" * 1000, b"C" * 500])
            target.write_bytes(whole[prefix_len : prefix_len + sizes[idx]])
            return True

        monkeypatch.setattr(md, "_download_with_retry", fake_download)
        dl = ModelDownloader()
        assert dl.download_release_assets(entry, target) is True
        assert target.read_bytes() == b"A" * 1000 + b"B" * 1000 + b"C" * 500

    def test_part_hash_mismatch_removes_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        entry, target = _entry(tmp_path, [b"A" * 100, b"B" * 100])

        def fake_download(url: str, target: Path, **kwargs) -> bool:  # noqa: ANN003
            target.write_bytes(b"X" * 100)  # falscher Inhalt
            return True

        monkeypatch.setattr(md, "_download_with_retry", fake_download)
        dl = ModelDownloader()
        assert dl.download_release_assets(entry, target) is False
        assert not target.exists()

    def test_missing_assets_field_fails_gracefully(self, tmp_path: Path) -> None:
        entry, target = _entry(tmp_path, [b"A" * 10])
        entry.assets = []
        dl = ModelDownloader()
        assert dl.download_release_assets(entry, target) is False
        assert not target.exists()

    def test_load_bundled_release_fallback(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # bundled_path absolut in tmp_path — beide Auflösungszweige landen dort
        abs_bundled = tmp_path / "models" / "test" / "m.bin"
        abs_bundled.parent.mkdir(parents=True, exist_ok=True)
        entry, _ = _entry(tmp_path, [b"X" * 64])
        entry.bundled_path = str(abs_bundled)
        dl = ModelDownloader()
        # bundled-Datei existiert nicht → Release-Fallback muss greifen
        monkeypatch.setattr(md, "OFFLINE_MODE", False)
        monkeypatch.setattr(dl, "download_release_assets", lambda e, t: t.write_bytes(b"X" * 64) or True)
        monkeypatch.setattr(ModelDownloader, "PROJECT_MODELS_DIR", tmp_path)
        result = dl.load_bundled(entry)
        assert result == abs_bundled
        assert abs_bundled.read_bytes() == b"X" * 64


@pytest.mark.unit
class TestModelPathResolver:
    def test_local_hit_no_download(self) -> None:
        from backend.core.model_path_resolver import clear_resolver_cache, resolve_model_path

        clear_resolver_cache()
        p = resolve_model_path("models/manifest.json", download=False)
        assert p is not None and p.is_file()

    def test_missing_download_disabled_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import backend.core.model_path_resolver as mpr

        mpr.clear_resolver_cache()
        monkeypatch.setattr(mpr, "_DOWNLOAD_ENABLED", False)
        assert mpr.resolve_model_path("models/gibt_es_nicht.onnx") is None

    def test_missing_without_manifest_entry_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import backend.core.model_path_resolver as mpr

        mpr.clear_resolver_cache()
        monkeypatch.setattr(mpr, "_DOWNLOAD_ENABLED", True)
        assert mpr.resolve_model_path("models/unbekannt/x.onnx") is None

    def test_download_fallback_via_load_bundled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import backend.core.model_downloader as md
        import backend.core.model_path_resolver as mpr

        mpr.clear_resolver_cache()
        monkeypatch.setattr(mpr, "_DOWNLOAD_ENABLED", True)
        fake_path = tmp_path / "fake.onnx"
        fake_path.write_bytes(b"x")
        entry = ModelEntry(
            name="fake",
            bundled=False,
            bundled_path="models/fake/fake.onnx",
            sha256="",
            size_bytes=1,
            required=False,
            fallback="dsp",
            delivery="release",
            release_tag="t",
            assets=["fake.onnx"],
        )

        class _FakeDL:
            def get_entry_by_path(self, bundled_path: str):
                return entry if bundled_path == "models/fake/fake.onnx" else None

            def load_bundled(self, entry: ModelEntry):
                return fake_path

        monkeypatch.setattr(md, "get_model_downloader", lambda: _FakeDL())
        assert mpr.resolve_model_path("models/fake/fake.onnx") == fake_path
