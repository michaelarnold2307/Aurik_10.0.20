from __future__ import annotations

"""
Tests für core/audio_file_validator.py — AudioFileValidator (§10.5)

Prüft:
- Sicherheits-Checks (Pfad-Traversal, Dateigröße, Magic-Bytes)
- Plausibilitäts-Checks (SR, Kanäle, Dauer)
- Singleton-Thread-Safety
- Deutsch-sprachige Fehlermeldungen
- Korrekte Exception-Typen (AudioLoadError)
"""


import pathlib
import struct
import threading
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Fixture-Helper
# ---------------------------------------------------------------------------


def _write_wav_header(f, n_samples: int = 48000, sr: int = 48000, n_ch: int = 1) -> None:
    """Schreibt einen minimalen WAV-Header in eine Datei."""
    data_size = n_samples * n_ch * 2  # 16-bit PCM
    chunk_size = 36 + data_size
    f.write(b"RIFF")
    f.write(struct.pack("<I", chunk_size))
    f.write(b"WAVE")
    f.write(b"fmt ")
    f.write(struct.pack("<I", 16))  # fmt chunk size
    f.write(struct.pack("<H", 1))  # PCM
    f.write(struct.pack("<H", n_ch))
    f.write(struct.pack("<I", sr))
    f.write(struct.pack("<I", sr * n_ch * 2))  # byte rate
    f.write(struct.pack("<H", n_ch * 2))  # block align
    f.write(struct.pack("<H", 16))  # bits per sample
    f.write(b"data")
    f.write(struct.pack("<I", data_size))
    f.write(b"\x00" * data_size)


def _make_wav_file(tmp_path: pathlib.Path, suffix: str = ".wav") -> pathlib.Path:
    p = tmp_path / f"test{suffix}"
    with open(p, "wb") as f:
        _write_wav_header(f)
    return p


def _make_flac_stub(tmp_path: pathlib.Path) -> pathlib.Path:
    """FLAC-Stub (Magic-Bytes korrekt, Rest minimal)."""
    p = tmp_path / "test.flac"
    p.write_bytes(b"fLaC" + b"\x00" * 128)
    return p


def _make_mp3_stub(tmp_path: pathlib.Path) -> pathlib.Path:
    """MP3-Stub mit ID3-Header."""
    p = tmp_path / "test.mp3"
    p.write_bytes(b"ID3" + b"\x03\x00\x00\x00\x00\x00\x00" + b"\x00" * 128)
    return p


@pytest.mark.unit
class TestAudioFileValidatorImport:
    """Import und Singleton."""

    def test_01_import(self):
        """AudioFileValidator ist importierbar."""
        from backend.core.audio_file_validator import AudioFileValidator, AudioLoadError

        assert AudioFileValidator is not None
        assert AudioLoadError is not None

    def test_02_singleton_identity(self):
        """get_audio_file_validator() gibt identisches Objekt zurück."""
        from backend.core.audio_file_validator import get_audio_file_validator

        assert get_audio_file_validator() is get_audio_file_validator()

    def test_03_singleton_thread_safe(self):
        """Parallele Zugriffe liefern identisches Singleton-Objekt."""
        from backend.core.audio_file_validator import get_audio_file_validator

        instances = []
        errors = []

        def fetch():
            try:
                instances.append(get_audio_file_validator())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=fetch) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert all(inst is instances[0] for inst in instances)

    def test_04_audio_load_error_has_message_user(self):
        """AudioLoadError speichert message_user."""
        from backend.core.audio_file_validator import AudioLoadError

        err = AudioLoadError("Nutzermeldung", cause="Technisch")
        assert err.message_user == "Nutzermeldung"
        assert "Technisch" in err.cause


class TestAudioFileValidatorValidate:
    """Tests für validate()-Methode."""

    def test_05_valid_wav_file(self, tmp_path):
        """Gültige WAV-Datei wird akzeptiert."""
        from backend.core.audio_file_validator import get_audio_file_validator

        wav = _make_wav_file(tmp_path)
        result = get_audio_file_validator().validate(wav)
        assert result.is_valid is True
        assert result.file_size_bytes > 0

    def test_06_valid_flac_stub(self, tmp_path):
        """FLAC-Magic-Bytes werden erkannt."""
        from backend.core.audio_file_validator import get_audio_file_validator

        flac = _make_flac_stub(tmp_path)
        result = get_audio_file_validator().validate(flac)
        assert result.is_valid is True
        assert result.detected_format == "flac"

    def test_07_valid_mp3_stub(self, tmp_path):
        """MP3-ID3-Header wird erkannt."""
        from backend.core.audio_file_validator import get_audio_file_validator

        mp3 = _make_mp3_stub(tmp_path)
        result = get_audio_file_validator().validate(mp3)
        assert result.is_valid is True

    def test_08_nonexistent_file_raises(self, tmp_path):
        """Nicht-existente Datei → AudioLoadError."""
        from backend.core.audio_file_validator import AudioLoadError, get_audio_file_validator

        with pytest.raises(AudioLoadError) as exc_info:
            get_audio_file_validator().validate(tmp_path / "ghost.wav")
        assert (
            "nicht gefunden" in exc_info.value.message_user.lower() or "geöffnet" in exc_info.value.message_user.lower()
        )

    def test_09_empty_file_raises(self, tmp_path):
        """Leere Datei → AudioLoadError."""
        from backend.core.audio_file_validator import AudioLoadError, get_audio_file_validator

        empty = tmp_path / "empty.wav"
        empty.write_bytes(b"")
        with pytest.raises(AudioLoadError) as exc_info:
            get_audio_file_validator().validate(empty)
        # Fehlermeldung ist deutschsprachig
        msg = exc_info.value.message_user
        assert len(msg) > 10

    def test_10_file_too_large_raises(self, tmp_path):
        """Datei > 10 GB simuliert → AudioLoadError."""
        from backend.core.audio_file_validator import AudioLoadError, get_audio_file_validator

        wav = _make_wav_file(tmp_path)
        validator = get_audio_file_validator()
        original_max = validator.MAX_FILE_SIZE_BYTES
        try:
            validator.MAX_FILE_SIZE_BYTES = 10  # 10 Bytes — mini limit
            with pytest.raises(AudioLoadError) as exc_info:
                validator.validate(wav)
            assert "groß" in exc_info.value.message_user.lower() or "GB" in exc_info.value.message_user
        finally:
            validator.MAX_FILE_SIZE_BYTES = original_max

    def test_11_unknown_extension_warning(self, tmp_path):
        """Unbekannte Extension erzeugt Warnung, kein Fehler."""
        from backend.core.audio_file_validator import get_audio_file_validator

        p = tmp_path / "audio.xyz"
        p.write_bytes(b"RIFF" + b"\x00" * 64)  # WAV-Magic, falsche Extension
        result = get_audio_file_validator().validate(p)
        assert result.is_valid is True
        assert any("Erweiterung" in w or "Extension" in w or "Format" in w for w in result.warnings)

    def test_12_wav_magic_vs_mp3_extension_warning(self, tmp_path):
        """WAV-Magic in .mp3-Datei → Warnung geloggt, Datei trotzdem akzeptiert."""
        from backend.core.audio_file_validator import get_audio_file_validator

        p = tmp_path / "wrong.mp3"
        # Schreibe WAV-Magic in .mp3
        with open(p, "wb") as f:
            _write_wav_header(f)
        # Soll nicht hart scheitern:
        result = get_audio_file_validator().validate(p)
        assert result.is_valid is True  # FFmpeg-Fallback

    def test_13_directory_raises(self, tmp_path):
        """Verzeichnis-Pfad → AudioLoadError."""
        from backend.core.audio_file_validator import AudioLoadError, get_audio_file_validator

        with pytest.raises(AudioLoadError):
            get_audio_file_validator().validate(tmp_path)

    def test_14_validation_result_dataclass(self, tmp_path):
        """ValidationResult hat alle Pflichtfelder."""
        from backend.core.audio_file_validator import ValidationResult, get_audio_file_validator

        wav = _make_wav_file(tmp_path)
        result = get_audio_file_validator().validate(wav)
        assert isinstance(result, ValidationResult)
        assert isinstance(result.path, pathlib.Path)
        assert isinstance(result.is_valid, bool)
        assert isinstance(result.file_size_bytes, int)
        assert isinstance(result.warnings, list)
        assert isinstance(result.detected_format, str)


class TestAudioFileValidatorSampleRate:
    """Tests für validate_sample_rate()."""

    def test_15_valid_sr_48000(self):
        """48000 Hz ist gültig."""
        from backend.core.audio_file_validator import get_audio_file_validator

        get_audio_file_validator().validate_sample_rate(48000)  # kein Fehler

    def test_16_valid_sr_44100(self):
        """44100 Hz ist gültig."""
        from backend.core.audio_file_validator import get_audio_file_validator

        get_audio_file_validator().validate_sample_rate(44100)

    def test_17_valid_sr_8000(self):
        """8000 Hz (Minimum) ist gültig."""
        from backend.core.audio_file_validator import get_audio_file_validator

        get_audio_file_validator().validate_sample_rate(8000)

    def test_18_valid_sr_384000(self):
        """384000 Hz (Maximum) ist gültig."""
        from backend.core.audio_file_validator import get_audio_file_validator

        get_audio_file_validator().validate_sample_rate(384000)

    def test_19_invalid_sr_too_low(self):
        """SR 4000 Hz → AudioLoadError."""
        from backend.core.audio_file_validator import AudioLoadError, get_audio_file_validator

        with pytest.raises(AudioLoadError) as exc_info:
            get_audio_file_validator().validate_sample_rate(4000)
        assert "Hz" in exc_info.value.message_user

    def test_20_invalid_sr_too_high(self):
        """SR 500000 Hz → AudioLoadError."""
        from backend.core.audio_file_validator import AudioLoadError, get_audio_file_validator

        with pytest.raises(AudioLoadError):
            get_audio_file_validator().validate_sample_rate(500000)

    def test_21_invalid_sr_zero(self):
        """SR 0 → AudioLoadError."""
        from backend.core.audio_file_validator import AudioLoadError, get_audio_file_validator

        with pytest.raises(AudioLoadError):
            get_audio_file_validator().validate_sample_rate(0)


class TestAudioFileValidatorChannels:
    """Tests für validate_channels()."""

    def test_22_mono_no_warning(self):
        """1 Kanal → keine Warnung."""
        from backend.core.audio_file_validator import get_audio_file_validator

        warnings = get_audio_file_validator().validate_channels(1)
        assert warnings == []

    def test_23_stereo_no_warning(self):
        """2 Kanäle → keine Warnung."""
        from backend.core.audio_file_validator import get_audio_file_validator

        warnings = get_audio_file_validator().validate_channels(2)
        assert warnings == []

    def test_24_multichannel_warning(self):
        """6 Kanäle → Warnung (kein Fehler)."""
        from backend.core.audio_file_validator import get_audio_file_validator

        warnings = get_audio_file_validator().validate_channels(6)
        assert len(warnings) > 0
        assert any("Stereo" in w or "Kanal" in w or "Mono" in w for w in warnings)

    def test_25_zero_channels_raises(self):
        """0 Kanäle → AudioLoadError."""
        from backend.core.audio_file_validator import AudioLoadError, get_audio_file_validator

        with pytest.raises(AudioLoadError):
            get_audio_file_validator().validate_channels(0)

    def test_26_negative_channels_raises(self):
        """Negative Kanal-Anzahl → AudioLoadError."""
        from backend.core.audio_file_validator import AudioLoadError, get_audio_file_validator

        with pytest.raises(AudioLoadError):
            get_audio_file_validator().validate_channels(-1)


class TestAudioFileValidatorDuration:
    """Tests für validate_duration()."""

    def test_27_short_duration_ok(self):
        """10 s ist gültig."""
        from backend.core.audio_file_validator import get_audio_file_validator

        get_audio_file_validator().validate_duration(10.0)

    def test_28_max_duration_ok(self):
        """8 Stunden sind gültig."""
        from backend.core.audio_file_validator import get_audio_file_validator

        get_audio_file_validator().validate_duration(8 * 3600)

    def test_29_too_long_raises(self):
        """9 Stunden → AudioLoadError."""
        from backend.core.audio_file_validator import AudioLoadError, get_audio_file_validator

        with pytest.raises(AudioLoadError) as exc_info:
            get_audio_file_validator().validate_duration(9 * 3600)
        assert "Stunden" in exc_info.value.message_user or "lang" in exc_info.value.message_user

    def test_30_zero_duration_ok(self):
        """0 s ist technisch gültig (leere Datei wird woanders abgefangen)."""
        from backend.core.audio_file_validator import get_audio_file_validator

        get_audio_file_validator().validate_duration(0.0)


class TestAudioFileValidatorConvenience:
    """Tests für Convenience-Funktion validate_audio_file()."""

    def test_31_convenience_function(self, tmp_path):
        """validate_audio_file() Wrapper funktioniert."""
        from backend.core.audio_file_validator import validate_audio_file

        wav = _make_wav_file(tmp_path)
        result = validate_audio_file(wav)
        assert result.is_valid is True

    def test_32_convenience_accepts_str(self, tmp_path):
        """validate_audio_file() akzeptiert String-Pfade."""
        from backend.core.audio_file_validator import validate_audio_file

        wav = _make_wav_file(tmp_path)
        result = validate_audio_file(str(wav))
        assert result.is_valid is True

    def test_33_error_message_is_german(self, tmp_path):
        """Fehlermeldungen bei ungültiger Datei sind deutschsprachig."""
        from backend.core.audio_file_validator import AudioLoadError, validate_audio_file

        with pytest.raises(AudioLoadError) as exc_info:
            validate_audio_file(tmp_path / "does_not_exist.wav")
        msg = exc_info.value.message_user
        # Mindestens ein deutsches Wort
        german_markers = ["nicht", "kann", "diese", "die", "der", "das", "wird", "wurde", "bitte", "Datei"]
        assert any(m.lower() in msg.lower() for m in german_markers), f"Fehlermeldung ist nicht deutsch: {msg!r}"

    def test_34_path_return_is_absolute(self, tmp_path):
        """ValidationResult.path ist immer absolut."""
        from backend.core.audio_file_validator import validate_audio_file

        wav = _make_wav_file(tmp_path)
        result = validate_audio_file(wav)
        assert result.path.is_absolute()

    def test_35_no_shell_in_method(self, tmp_path):
        """subprocess wird NICHT mit shell=True aufgerufen."""
        import subprocess

        from backend.core.audio_file_validator import get_audio_file_validator

        wav = _make_wav_file(tmp_path)
        original_run = subprocess.run
        shell_called = []

        def mock_run(*args, **kwargs):
            if kwargs.get("shell") is True:
                shell_called.append(True)
            return original_run(*args, **kwargs)

        with patch("subprocess.run", side_effect=mock_run):
            get_audio_file_validator().validate(wav)

        assert not shell_called, "subprocess.run mit shell=True aufgerufen!"
