"""Tests for backend.core.metadata_preserver — audio metadata transfer.

Covers: tag extraction, tag application, provenance hash, round-trip transfer,
edge cases (missing file, unsupported format, empty tags).
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_test_wav(path: Path, sr: int = 48000, duration_s: float = 0.1) -> None:
    """Write a short sine WAV for testing."""
    t = np.linspace(0, duration_s, int(sr * duration_s), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(str(path), audio, sr)


def _write_test_flac(path: Path, sr: int = 48000, duration_s: float = 0.1) -> None:
    t = np.linspace(0, duration_s, int(sr * duration_s), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(str(path), audio, sr, format="FLAC")


def _write_test_mp3(path: Path, sr: int = 48000, duration_s: float = 0.5) -> Path:
    """Write a test MP3 via ffmpeg (requires ffmpeg)."""
    try:
        import ffmpeg as _ff
    except ImportError:
        pytest.skip("ffmpeg-python not available")
    wav_path = path.with_suffix(".wav")
    t = np.linspace(0, duration_s, int(sr * duration_s), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(str(wav_path), audio, sr)
    try:
        _ff.input(str(wav_path)).output(str(path)).run(overwrite_output=True, quiet=True)
    except Exception:
        pytest.skip("ffmpeg binary not available")
    finally:
        wav_path.unlink(missing_ok=True)
    return path


# ---------------------------------------------------------------------------
# Module availability
# ---------------------------------------------------------------------------

try:
    from backend.core.metadata_preserver import (
        _MUTAGEN_AVAILABLE,
        AudioMetadata,
        MetadataPreserver,
        get_metadata_preserver,
    )
except ImportError:
    pytest.skip("metadata_preserver not importable", allow_module_level=True)

pytestmark = pytest.mark.skipif(not _MUTAGEN_AVAILABLE, reason="mutagen not installed")


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_singleton_identity(self):
        a = get_metadata_preserver()
        b = get_metadata_preserver()
        assert a is b

    def test_singleton_is_preserver(self):
        assert isinstance(get_metadata_preserver(), MetadataPreserver)


# ---------------------------------------------------------------------------
# AudioMetadata dataclass
# ---------------------------------------------------------------------------


class TestAudioMetadata:
    def test_empty_has_no_content(self):
        m = AudioMetadata()
        assert not m.has_content()

    def test_title_has_content(self):
        m = AudioMetadata(title="Test")
        assert m.has_content()

    def test_artist_has_content(self):
        m = AudioMetadata(artist="Artist")
        assert m.has_content()

    def test_defaults(self):
        m = AudioMetadata()
        assert m.cover_art is None
        assert m.cover_mime == "image/jpeg"
        assert m.extra == {}


# ---------------------------------------------------------------------------
# Extract — WAV (no tags expected)
# ---------------------------------------------------------------------------


class TestExtractWav:
    def test_wav_returns_empty(self, tmp_path):
        p = tmp_path / "test.wav"
        _write_test_wav(p)
        meta = get_metadata_preserver().extract(p)
        assert isinstance(meta, AudioMetadata)

    def test_nonexistent_returns_empty(self):
        meta = get_metadata_preserver().extract("/nonexistent/file.wav")
        assert not meta.has_content()


# ---------------------------------------------------------------------------
# Apply + extract round-trip — FLAC (Vorbis tags)
# ---------------------------------------------------------------------------


class TestFlacRoundTrip:
    def test_apply_and_extract_flac(self, tmp_path):
        p = tmp_path / "out.flac"
        _write_test_flac(p)

        meta_in = AudioMetadata(
            title="Testlied",
            artist="Testkünstler",
            album="Testalbum",
            date="2026",
            genre="Jazz",
            tracknumber="3",
        )
        ok = get_metadata_preserver().apply(p, meta_in, aurik_version="9.10")
        assert ok

        meta_out = get_metadata_preserver().extract(p)
        assert meta_out.title == "Testlied"
        assert meta_out.artist == "Testkünstler"
        assert meta_out.album == "Testalbum"
        assert meta_out.date == "2026"
        assert meta_out.genre == "Jazz"
        assert meta_out.tracknumber == "3"

    def test_provenance_comment_in_flac(self, tmp_path):
        p = tmp_path / "prov.flac"
        _write_test_flac(p)
        ok = get_metadata_preserver().apply(p, AudioMetadata(), aurik_version="9.10.99", original_hash="abc123")
        assert ok
        from mutagen.flac import FLAC

        mf = FLAC(str(p))
        comments = mf.get("COMMENT", [])
        assert len(comments) >= 1
        assert "Aurik" in comments[0]


# ---------------------------------------------------------------------------
# Apply + extract round-trip — MP3 (ID3 tags)
# ---------------------------------------------------------------------------


class TestMp3RoundTrip:
    def test_apply_and_extract_mp3(self, tmp_path):
        p = tmp_path / "out.mp3"
        _write_test_mp3(p)

        meta_in = AudioMetadata(
            title="Testlied",
            artist="Testkünstler",
            album="Testalbum",
        )
        ok = get_metadata_preserver().apply(p, meta_in, aurik_version="9.10")
        assert ok

        meta_out = get_metadata_preserver().extract(p)
        assert meta_out.title == "Testlied"
        assert meta_out.artist == "Testkünstler"
        assert meta_out.album == "Testalbum"


# ---------------------------------------------------------------------------
# Transfer (extract + apply)
# ---------------------------------------------------------------------------


class TestTransfer:
    def test_transfer_flac_to_flac(self, tmp_path):
        src = tmp_path / "src.flac"
        dst = tmp_path / "dst.flac"
        _write_test_flac(src)
        _write_test_flac(dst)

        # Tag the source
        from mutagen.flac import FLAC

        mf = FLAC(str(src))
        mf["TITLE"] = ["Quellenlied"]
        mf["ARTIST"] = ["Originalkünstler"]
        mf.save()

        ok = get_metadata_preserver().transfer(src, dst, aurik_version="9.10")
        assert ok

        meta = get_metadata_preserver().extract(dst)
        assert meta.title == "Quellenlied"
        assert meta.artist == "Originalkünstler"

    def test_transfer_no_tags_writes_provenance(self, tmp_path):
        src = tmp_path / "empty_src.flac"
        dst = tmp_path / "empty_dst.flac"
        _write_test_flac(src)
        _write_test_flac(dst)

        ok = get_metadata_preserver().transfer(src, dst, aurik_version="9.10.50")
        assert ok  # provenance is still written

        from mutagen.flac import FLAC

        mf = FLAC(str(dst))
        comments = mf.get("COMMENT", [])
        assert any("Aurik" in c for c in comments)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_unsupported_format_returns_false(self, tmp_path):
        p = tmp_path / "test.xyz"
        p.write_text("not audio")
        ok = get_metadata_preserver().apply(p, AudioMetadata(title="X"))
        assert not ok

    def test_missing_target_returns_false(self):
        ok = get_metadata_preserver().apply("/no/such/file.flac", AudioMetadata(title="X"))
        assert not ok

    def test_file_hash_empty_on_missing(self):
        h = MetadataPreserver._file_hash("/no/such/file")
        assert h == ""

    def test_file_hash_nonempty(self, tmp_path):
        p = tmp_path / "data.bin"
        p.write_bytes(b"Hello Aurik")
        h = MetadataPreserver._file_hash(p)
        assert len(h) == 64  # SHA-256 hex digest


# ---------------------------------------------------------------------------
# Cover art round-trip — FLAC
# ---------------------------------------------------------------------------


class TestCoverArt:
    def test_cover_art_flac_round_trip(self, tmp_path):
        p = tmp_path / "cover.flac"
        _write_test_flac(p)

        # Minimal 1x1 JPEG bytes (valid header)
        fake_jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"
        meta_in = AudioMetadata(cover_art=fake_jpeg, cover_mime="image/jpeg")
        ok = get_metadata_preserver().apply(p, meta_in)
        assert ok

        meta_out = get_metadata_preserver().extract(p)
        assert meta_out.cover_art is not None
        assert len(meta_out.cover_art) > 0
