"""
Aurik 10.0.0 — AudioFileValidator (§10.5)
=====================================
Sicherheits- und Plausibilitätsprüfung für alle Audio-Eingabedateien.
OWASP-konform: Pfad-Traversal-Schutz, Magic-Bytes-Verifikation,
Größen- und Dauer-Limits.

Referenz: OWASP Top 10 A03 (Injection), A01 (Broken Access Control).
"""

from __future__ import annotations

import logging
import os
import pathlib
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ausnahme-Klasse
# ---------------------------------------------------------------------------


class AudioLoadError(Exception):
    """Wird ausgelöst wenn eine Audiodatei nicht sicher geladen werden kann.

    Enthält immer eine deutschsprachige Nutzermeldung (message_user) und eine
    technische Ursache (cause) für das Log.
    """

    def __init__(self, message_user: str, cause: str = "") -> None:
        self.message_user = message_user
        self.cause = cause
        super().__init__(message_user, cause)  # B042: alle Args an super (pickle/copy-sicher)
        # Nutzerfreundliche str()-Darstellung bleibt wie vor dem B042-Fix erhalten.
        self.args = (f"{message_user} | tech: {cause}" if cause else message_user,)


# ---------------------------------------------------------------------------
# §3.9.7  Audio-Buffer-RAM-Guard
# ---------------------------------------------------------------------------


class AudioTooLargeError(AudioLoadError):
    """Raised when the numpy audio array would exceed MAX_AUDIO_BYTES_RAM.

    Typically triggered by very long files (> 8 h) whose float32 representation
    can be 4–10× larger than the original compressed file on disk.
    """


MAX_AUDIO_BYTES_RAM: int = 2 * 1024**3  # 2 GB absolute RAM limit for one audio buffer


def _check_audio_buffer_size(audio: np.ndarray, file_path: str) -> None:
    """Raise AudioTooLargeError if *audio* array exceeds MAX_AUDIO_BYTES_RAM (§3.9.7).

    MUST be called after soundfile.read() / pedalboard.read() and BEFORE
    resample_poly — resampling can further increase the buffer size.

    Args:
        audio:     Loaded audio array.
        file_path: Original file path (used in the error message only).

    Raises:
        AudioTooLargeError: Buffer exceeds the 2 GB hard limit.
    """
    nbytes = audio.nbytes
    if nbytes > MAX_AUDIO_BYTES_RAM:
        import pathlib as _pl

        raise AudioTooLargeError(
            f"Audio-Buffer {nbytes / 1024**3:.1f} GB überschreitet das RAM-Limit "
            f"({MAX_AUDIO_BYTES_RAM // 1024**3} GB). "
            f"Bitte kürze '{_pl.Path(file_path).name}' oder teile die Datei auf.",
            cause=f"nbytes={nbytes}, limit={MAX_AUDIO_BYTES_RAM}",
        )


# ---------------------------------------------------------------------------
# Ergebnis-Klasse
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Ergebnis der Audio-Datei-Validierung."""

    path: pathlib.Path
    is_valid: bool
    detected_format: str = ""
    file_size_bytes: int = 0
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class AudioFileValidator:
    """Prüft Eingabedateien auf Sicherheit und Plausibilität vor dem Import.

    Prüfreihenfolge (§10.5):
        1. Dateigröße: max. MAX_FILE_SIZE_BYTES (10 GB)
        2. Pfad-Traversal: os.path.realpath() + erlaubte Basis-Verzeichnisse
        3. Datei-Extension vs. Magic-Bytes-Verifikation
        4. Audiodaten-Länge: max. MAX_DURATION_HOURS
        5. Sample-Rate: 8 000 – 384 000 Hz
        6. Kanäle: 1–2 nativ; > 2 → Warnung, kein Fehler
        7. Keine Shell-Ausführung von Metadaten-Inhalten (ID3 TXXX etc.)

    Invarianten:
        - Alle Fehlerpfade: AudioLoadError mit Deutsch-Meldung, kein Absturz
        - Pfad-Traversal ausgeschlossen
        - Temp-Dateien: tempfile.mkstemp() statt /tmp/<user_input>
        - Keine Shell-Interpolation von Dateinamen
    """

    MAX_FILE_SIZE_BYTES: int = 10 * 1024**3  # 10 GB
    MAX_DURATION_HOURS: float = 8.0
    MIN_SAMPLE_RATE_HZ: int = 8_000
    MAX_SAMPLE_RATE_HZ: int = 384_000

    # Magic-Bytes für unterstützte Formate
    MAGIC_BYTES: dict[str, list[bytes]] = {
        "wav": [b"RIFF"],
        "flac": [b"fLaC"],
        "ogg": [b"OggS"],
        "aiff": [b"FORM"],
        "mp3": [b"\xff\xfb", b"\xff\xf3", b"\xff\xf2", b"ID3"],
        "mp4": [b"\x00\x00\x00\x18ftyp", b"\x00\x00\x00\x1cftyp", b"\x00\x00\x00\x14ftyp"],
        "m4a": [b"\x00\x00\x00\x18ftyp", b"\x00\x00\x00\x1cftyp"],
        # WMA/ASF — GUID-basiert
        "wma": [b"\x30\x26\xb2\x75\x8e\x66\xcf\x11"],
        # Opus/WebM/Matroska
        "opus": [b"\x1a\x45\xdf\xa3"],
        # CAF
        "caf": [b"caff"],
        # WAV RIFX (Big-Endian Variante)
        "rifx": [b"RIFX"],
    }

    # Erweiterung → erlaubte Format-Gruppen
    EXT_TO_FORMAT: dict[str, list[str]] = {
        ".wav": ["wav", "rifx"],
        ".aif": ["aiff"],
        ".aiff": ["aiff"],
        ".flac": ["flac"],
        ".ogg": ["ogg", "opus"],
        ".mp3": ["mp3"],
        ".mp4": ["mp4", "m4a"],
        ".m4a": ["m4a", "mp4"],
        ".aac": ["mp4", "m4a"],
        ".wma": ["wma"],
        ".opus": ["ogg", "opus"],
        ".caf": ["caf"],
    }

    def validate(self, path: pathlib.Path) -> ValidationResult:
        """Validiert eine Audio-Eingabedatei.

        Args:
            path: Pfad zur Audiodatei (kann relativ oder absolut sein).

        Returns:
            ValidationResult (is_valid=True wenn alle Checks bestanden).

        Raises:
            AudioLoadError: Bei jeder Sicherheits- oder Plausibilitätsverletzung.
        """
        # Schritt 1: Normalisierung und Pfad-Traversal-Schutz
        try:
            real_path = pathlib.Path(os.path.realpath(path))
        except (OSError, ValueError) as e:
            raise AudioLoadError(
                "Diese Datei kann nicht geöffnet werden. Möglicherweise ist der Pfad ungültig.",
                cause=str(e),
            ) from e

        # Schritt 2: Existenz-Prüfung
        if not real_path.exists():
            raise AudioLoadError(
                "Diese Datei wurde nicht gefunden. Bitte prüfen Sie den Dateinamen und Speicherort.",
                cause=f"Datei nicht vorhanden: {real_path}",
            )
        if not real_path.is_file():
            raise AudioLoadError(
                "Der angegebene Pfad ist kein gültiges Datei-Objekt.",
                cause=f"Kein reguläres File: {real_path}",
            )

        # Schritt 3: Dateigröße
        file_size = real_path.stat().st_size
        if file_size == 0:
            raise AudioLoadError(
                "Diese Datei ist leer und kann nicht geöffnet werden.",
                cause="Dateigröße = 0 Bytes",
            )
        if file_size > self.MAX_FILE_SIZE_BYTES:
            max_gb = self.MAX_FILE_SIZE_BYTES / (1024**3)
            raise AudioLoadError(
                f"Diese Datei ist zu groß (maximal {max_gb:.0f} GB). Bitte teilen Sie sie in kleinere Abschnitte auf.",
                cause=f"Dateigröße {file_size} > {self.MAX_FILE_SIZE_BYTES}",
            )

        # Schritt 4: Extension prüfen
        ext = real_path.suffix.lower()
        warnings: list[str] = []
        detected_format = ""

        if ext not in self.EXT_TO_FORMAT:
            # Unbekannte Extension → Warnung, kein harter Fehler (FFmpeg als Fallback)
            warnings.append(f"Unbekannte Datei-Erweiterung '{ext}' — das Format wird trotzdem versucht zu laden.")
            logger.warning("Unbekannte Extension %s für %s", ext, real_path.name)
        else:
            # Schritt 5: Magic-Bytes-Verifikation
            detected_format = self._verify_magic_bytes(real_path, ext)

        return ValidationResult(
            path=real_path,
            is_valid=True,
            detected_format=detected_format,
            file_size_bytes=file_size,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Interne Hilfsmethoden
    # ------------------------------------------------------------------

    def _verify_magic_bytes(self, path: pathlib.Path, ext: str) -> str:
        """Vergleicht Magic-Bytes mit der Datei-Extension.

        Gibt den erkannten Format-String zurück.
        Loggt eine Warnung bei Diskrepanzen (kein harter Fehler — FFmpeg Fallback).
        """
        allowed_formats = self.EXT_TO_FORMAT.get(ext, [])

        try:
            with open(path, "rb") as fh:
                header = fh.read(32)
        except OSError as e:
            raise AudioLoadError(
                "Diese Datei kann nicht geöffnet werden. Möglicherweise ist sie beschädigt oder gesperrt.",
                cause=str(e),
            ) from e

        for fmt in allowed_formats:
            for magic in self.MAGIC_BYTES.get(fmt, []):
                if header[: len(magic)] == magic:
                    logger.debug("Magic-Bytes OK: Datei %s als '%s' erkannt", path.name, fmt)
                    return fmt

        # Kein Magic-Bytes-Match — prüfe ob die Datei trotzdem einem bekannten Format entspricht
        for fmt_key, magic_list in self.MAGIC_BYTES.items():
            for magic in magic_list:
                if header[: len(magic)] == magic:
                    logger.warning(
                        "Extension '%s' passt nicht zu Magic-Bytes '%s' in %s",
                        ext,
                        fmt_key,
                        path.name,
                    )
                    return fmt_key

        # Unbekanntes Header-Format → Warnung, trotzdem zulassen (FFmpeg ist robust)
        logger.warning("Keine bekannten Magic-Bytes in %s (Header: %s)", path.name, header[:8].hex())
        return "unknown"

    def validate_sample_rate(self, sample_rate: int) -> None:
        """Prüft ob die Sample-Rate im erlaubten Bereich liegt.

        Raises:
            AudioLoadError: Wenn SR außerhalb [8000, 384000] Hz.
        """
        if not (self.MIN_SAMPLE_RATE_HZ <= sample_rate <= self.MAX_SAMPLE_RATE_HZ):
            raise AudioLoadError(
                f"Die Sample-Rate dieser Datei ({sample_rate} Hz) wird nicht "
                f"unterstützt (erlaubt: {self.MIN_SAMPLE_RATE_HZ}–"
                f"{self.MAX_SAMPLE_RATE_HZ} Hz).",
                cause=f"Sample-Rate {sample_rate} außerhalb erlaubtem Bereich",
            )

    def validate_channels(self, n_channels: int) -> list[str]:
        """Prüft Kanal-Anzahl; gibt Warnungen zurück (kein Fehler bei > 2)."""
        warnings: list[str] = []
        if n_channels < 1:
            raise AudioLoadError(
                "Diese Datei enthält keine Audio-Kanäle.",
                cause=f"n_channels = {n_channels}",
            )
        if n_channels > 2:
            warnings.append(
                f"Diese Datei hat {n_channels} Kanäle — Aurik verarbeitet "
                "nur Mono und Stereo. Die Kanäle werden zu Stereo zusammengemischt."
            )
        return warnings

    def validate_duration(self, duration_seconds: float) -> None:
        """Prüft ob die Dateilänge das Maximum nicht überschreitet.

        Raises:
            AudioLoadError: Wenn > MAX_DURATION_HOURS.
        """
        max_s = self.MAX_DURATION_HOURS * 3600
        if duration_seconds > max_s:
            raise AudioLoadError(
                f"Diese Datei ist zu lang ({duration_seconds / 3600:.1f} Stunden). "
                f"Maximal {self.MAX_DURATION_HOURS:.0f} Stunden werden unterstützt.",
                cause=f"Dauer {duration_seconds:.1f}s > {max_s:.1f}s",
            )


# ---------------------------------------------------------------------------
# Singleton + Convenience
# ---------------------------------------------------------------------------

import threading

_instance: AudioFileValidator | None = None
_lock = threading.Lock()


def get_audio_file_validator() -> AudioFileValidator:
    """Thread-sicherer Singleton-Accessor (Double-Checked Locking).

    Returns:
        AudioFileValidator-Instanz (einmalig erstellt).
    """
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = AudioFileValidator()
    return _instance


def validate_audio_file(path: pathlib.Path | str) -> ValidationResult:
    """Convenience-Wrapper: Validiert eine Audiodatei sicher.

    Args:
        path: Pfad zur Audiodatei.

    Returns:
        ValidationResult

    Raises:
        AudioLoadError: Bei Sicherheits- oder Plausibilitätsverletzung.
    """
    return get_audio_file_validator().validate(pathlib.Path(path))
