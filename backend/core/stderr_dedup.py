"""C-level stderr log deduplication (§D4).

ONNX Runtime, CUDA, ROCm and other C libraries write warnings directly to
stderr, bypassing Python's logging and warnings frameworks entirely.  When a
model load triggers 40+ identical C-level warnings, the console becomes
unreadable.

This module wraps sys.stderr with a rate-limiting deduplicator that:
- Tracks the last N unique C-level warning lines.
- When the same line appears ≥3 times within a window, suppresses repeats
  and emits a single summary line.
- Preserves all non-duplicate output passthrough unmodified.

Usage (early in startup, before any ONNX import)::

    from backend.core.stderr_dedup import install_stderr_dedup
    install_stderr_dedup()
"""

from __future__ import annotations

import sys
import threading
from typing import TextIO


class _DedupStderr:
    """Rate-limiting stderr wrapper for C-level log deduplication."""

    def __init__(self, real_stderr: TextIO, window: int = 50, threshold: int = 3) -> None:
        self._real = real_stderr
        self._lock = threading.Lock()
        self._window: int = window
        self._threshold: int = threshold
        # Circular buffer: (line_hash, count, last_full_line)
        self._buf: list[tuple[int, int, str]] = []
        self._pos: int = 0
        self._line_buf: str = ""

    def write(self, s: str) -> int:
        """Write to stderr with deduplication."""
        if not s:
            return 0

        with self._lock:
            self._line_buf += s
            written = 0

            while "\n" in self._line_buf:
                line, self._line_buf = self._line_buf.split("\n", 1)
                full_line = line + "\n"

                # Only deduplicate lines that look like C-level warnings
                if self._is_c_level_warning(line):
                    h = hash(line)
                    existing = self._find_in_buf(h)
                    if existing is not None:
                        idx, count, _ = existing
                        self._buf[idx] = (h, count + 1, line)
                        if count + 1 == self._threshold:
                            # First suppression: emit a summary
                            written += self._real_write(
                                f"[stderr-dedup] {self._threshold - 1} frühere identische Zeilen unterdrückt, letzte:\n"
                            )
                            written += self._real_write(full_line)
                        # Beyond threshold: silent suppression
                        continue
                    else:
                        # New unique line: add to circular buffer
                        self._add_to_buf(h, line)
                        # If we just had suppressed lines, note the count
                        written += self._real_write(full_line)
                else:
                    written += self._real_write(full_line)

            return written

    def _is_c_level_warning(self, line: str) -> bool:
        """Check if a line looks like a C-level or ML-framework warning."""
        prefixes = (
            "[W:",  # ONNX Runtime warning
            "[E:",  # ONNX Runtime error
            "Warning:",  # Generic C warning
        )
        if line.startswith(prefixes):
            return True
        # §F6: PyTorch/HuggingFace warnings printed to stderr
        # e.g. "Some weights of the model checkpoint at ... were not used..."
        stderr_patterns = (
            "Some weights of",  # HuggingFace model loading
            "You should probably TRAIN",  # HuggingFace training advice
            "This IS expected if",  # HuggingFace initialization note
        )
        return any(p in line for p in stderr_patterns)

    def _find_in_buf(self, h: int) -> tuple[int, int, str] | None:
        """Find a hash in the circular buffer. Returns (index, count, line) or None."""
        for i, (bh, count, line) in enumerate(self._buf):
            if bh == h:
                return (i, count, line)
        return None

    def _add_to_buf(self, h: int, line: str) -> None:
        """Add a new entry to the circular buffer."""
        entry = (h, 1, line)
        if len(self._buf) < self._window:
            self._buf.append(entry)
        else:
            self._buf[self._pos % self._window] = entry
            self._pos += 1

    def _real_write(self, s: str) -> int:
        """Write directly to the real stderr."""
        return self._real.write(s)

    def flush(self) -> None:
        """Flush any buffered output."""
        with self._lock:
            if self._line_buf:
                self._real.write(self._line_buf)
                self._line_buf = ""
        self._real.flush()

    def __getattr__(self, name: str):
        """Delegate unknown attributes to the real stderr."""
        return getattr(self._real, name)


_installed: bool = False


def install_stderr_dedup(window: int = 50, threshold: int = 3) -> bool:
    """Install the stderr deduplication wrapper.

    Safe to call multiple times — only installs once.

    Args:
        window: Number of unique lines to track in the buffer.
        threshold: Number of identical lines before suppression begins.

    Returns:
        True if installed now, False if already installed.
    """
    global _installed
    if _installed:
        return False

    sys.stderr = _DedupStderr(sys.stderr, window=window, threshold=threshold)  # type: ignore[assignment]
    _installed = True
    return True


def uninstall_stderr_dedup() -> bool:
    """Remove the stderr wrapper and restore original stderr."""
    global _installed
    if not _installed:
        return False
    if isinstance(sys.stderr, _DedupStderr):
        sys.stderr = sys.stderr._real  # type: ignore[assignment]
    _installed = False
    return True
