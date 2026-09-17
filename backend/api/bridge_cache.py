"""Aurik Bridge — Cache Infrastructure (§11 Spec 08)
===================================================
Thread-safe LRU caches for analysis results (defect, era/genre, medium,
restorability). Content-addressed keys prevent redundant re-analysis when
files are renamed or moved.

Public API:
    _AnalysisLruCache (class)
    content_cache_key
    cache_defect_result, get_cached_defect_result, clear_defect_cache
    cache_era_genre_result, get_cached_era_genre_result, clear_era_genre_cache
    cache_medium_result, get_cached_medium_result, clear_medium_cache
    cache_restorability_result, get_cached_restorability_result

Referenz: Spec 08 §11 Softwareschichten-Architektur.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import shutil
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ANALYSIS_CACHE_MAX = 64
_CONTENT_CHUNK = 4096  # Bytes vom Anfang + Ende für SHA-256 Content-Key
_CONTENT_KEY_CACHE_MAX = 512

# Disk-Persistenz der Analyse-Caches (Prozess-übergreifend, §G5 (copilot-instructions.md)-deterministisch).
# Kill-Switch für Tests/Nutzer: AURIK_ANALYSIS_CACHE=0 deaktiviert die Platte;
# AURIK_ANALYSIS_CACHE_DIR überschreibt das Verzeichnis.
_DISK_CACHE_SCHEMA = 1
_DISK_CACHE_DISABLED = os.environ.get("AURIK_ANALYSIS_CACHE", "1").strip().lower() not in ("", "1", "true", "yes")
_disk_write_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Fast path for repeated cache lookups: (path, size, mtime_ns) -> content-key
# ---------------------------------------------------------------------------

_content_key_cache: OrderedDict[tuple[str, int, int], str] = OrderedDict()
_content_key_lock = threading.Lock()


# ---------------------------------------------------------------------------
# LRU Cache Class
# ---------------------------------------------------------------------------


class _AnalysisLruCache:
    """Thread-safe LRU cache keyed by content-hash (or arbitrary string).

    Stores analysis results under a content-addressed key so that the same
    audio file is not re-analysed when its path changes (e.g. rename before
    OOM-checkpoint resume).  Path→key aliases are maintained for fast
    backward-compatible path lookups.

    Args:
        maxsize: Maximum number of entries before LRU eviction.
    """

    def __init__(self, maxsize: int = _ANALYSIS_CACHE_MAX) -> None:
        self._maxsize = maxsize
        self._data: OrderedDict[str, Any] = OrderedDict()
        self._path_to_key: dict[str, str] = {}  # path → content_key
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def put(self, key: str, value: Any, path_alias: str | None = None) -> None:
        """Insert *value* under *key*, evicting LRU entry when full."""
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = value
            if path_alias:
                self._path_to_key[path_alias] = key
            while len(self._data) > self._maxsize:
                evicted_key, _ = self._data.popitem(last=False)
                # Clean up alias mapping for evicted key
                self._path_to_key = {p: k for p, k in self._path_to_key.items() if k != evicted_key}

    def get(self, key: str) -> Any | None:
        """Gibt cached value for *key* and promote to MRU, or ``None`` zurück."""
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def get_by_path(self, path: str) -> Any | None:
        """Gibt cached value using a path alias, or ``None`` zurück."""
        with self._lock:
            key = self._path_to_key.get(path)
            if key is None or key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def remove(self, key_or_path: str) -> None:
        """Entfernt entry by content-key or path alias."""
        with self._lock:
            # Try as path alias first
            key = self._path_to_key.pop(key_or_path, key_or_path)
            self._data.pop(key, None)
            # Also remove any alias pointing to same key
            self._path_to_key = {p: k for p, k in self._path_to_key.items() if k != key}

    def clear(self) -> None:
        """Entfernt all entries."""
        with self._lock:
            self._data.clear()
            self._path_to_key.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


# ---------------------------------------------------------------------------
# Content-Addressed Keying
# ---------------------------------------------------------------------------


def content_cache_key(file_path: str) -> str:
    """Berechnet a content-addressed cache key for *file_path*.

    Uses SHA-256 over the first and last ``_CONTENT_CHUNK`` bytes of the
    file (fast, file-size independent).  Falls back to the path itself when
    the file is not readable (e.g. missing/locked).

    Args:
        file_path: Absolute path to an audio file.

    Returns:
        A 64-character hex string suitable as a cache key, or the path
        itself on I/O error.
    """
    normalized_path = os.path.normpath(os.path.realpath(file_path))
    try:
        stat_result = os.stat(normalized_path)
    except OSError as exc:
        logger.debug(
            "§V6 (copilot-instructions.md) os.stat fehlgeschlagen — Dateipfad als Schlüssel verwendet: %s", exc
        )
        return file_path

    size = int(stat_result.st_size)
    mtime_ns = int(getattr(stat_result, "st_mtime_ns", int(stat_result.st_mtime * 1_000_000_000)))
    meta_key = (normalized_path, size, mtime_ns)

    with _content_key_lock:
        cached = _content_key_cache.get(meta_key)
        if cached is not None:
            _content_key_cache.move_to_end(meta_key)
            return cached

    try:
        with open(normalized_path, "rb") as fh:
            head = fh.read(_CONTENT_CHUNK)
            if size > _CONTENT_CHUNK * 2:
                fh.seek(-_CONTENT_CHUNK, 2)
                tail = fh.read(_CONTENT_CHUNK)
            else:
                tail = b""
        digest = hashlib.sha256(head + tail + str(size).encode()).hexdigest()
    except OSError as exc:
        logger.debug(
            "§V6 (copilot-instructions.md) Datei-Lesen fehlgeschlagen — Dateipfad als Zwischenspeicher-Key verwendet: %s",
            exc,
        )
        return file_path

    with _content_key_lock:
        _content_key_cache[meta_key] = digest
        _content_key_cache.move_to_end(meta_key)
        while len(_content_key_cache) > _CONTENT_KEY_CACHE_MAX:
            _content_key_cache.popitem(last=False)

    return digest


# ---------------------------------------------------------------------------
# Disk-Persistenz (Read-/Write-Through unter dem In-Memory-LRU)
# ---------------------------------------------------------------------------


def _disk_cache_root() -> Path:
    """Verzeichnis der Platten-Caches (Env-overridable, z. B. für Tests)."""
    _env = os.environ.get("AURIK_ANALYSIS_CACHE_DIR")
    if _env:
        return Path(_env)
    return Path(__file__).resolve().parent.parent.parent / "output" / "analysis_cache"


def _analysis_cache_version() -> str:
    """Analyse-Code-Version für die Cache-Invalidierung (§G5 (copilot-instructions.md): Version im Key)."""
    try:
        from backend.core.version import AURIK_VERSION  # lazy: leaf-Modul, kein Zyklus

        return str(AURIK_VERSION)
    except Exception as _exc:
        logger.debug("bridge: AURIK_VERSION nicht lesbar (%s) — Zwischenspeicher-Version 'unbekannt'", _exc)
        return "unknown"


def _disk_path(subdir: str, key: str) -> Path:
    """Plattenpfad eines Eintrags; Dateiname = SHA-256(key) (pfad-/zeichen-sicher)."""
    return _disk_cache_root() / subdir / (hashlib.sha256(key.encode()).hexdigest() + ".pkl")


def _disk_write(subdir: str, key: str, payload: object) -> None:
    """Schreibt ein Analyse-Ergebnis atomar auf Platte (tmp + os.replace).

    Fehler sind nie blockierend (§V6 (copilot-instructions.md)): Analyse wird
    dann beim nächsten Prozess schlicht neu berechnet.
    """
    if _DISK_CACHE_DISABLED:
        return
    try:
        _target = _disk_path(subdir, key)
        _target.parent.mkdir(parents=True, exist_ok=True)
        _envelope = {
            "schema": _DISK_CACHE_SCHEMA,
            "version": _analysis_cache_version(),
            "key": key,
            "payload": payload,
        }
        _tmp = _target.with_name(_target.name + f".tmp{os.getpid()}")
        with _disk_write_lock:
            with open(_tmp, "wb") as _fh:
                pickle.dump(_envelope, _fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(_tmp, _target)
        logger.debug("bridge: Analyse-Zwischenspeicher (Platte) geschrieben %s/%.8s…", subdir, key)
    except Exception as exc:
        logger.warning(
            "bridge: Analyse-Zwischenspeicher (Platte) schreiben fehlgeschlagen (%s) — Analyse wird neu berechnet (§V6 (copilot-instructions.md))",
            exc,
        )


def _disk_read(subdir: str, key: str) -> object | None:
    """Lädt ein Analyse-Ergebnis von Platte; ``None`` bei Miss/Versionswechsel/Fehler.

    Versions- oder Schema-Mismatch sowie nicht lesbare Einträge werden
    sichtbar (§V6 (copilot-instructions.md)) verworfen und gelöscht — nie still verwendet.
    """
    if _DISK_CACHE_DISABLED:
        return None
    _target = _disk_path(subdir, key)
    if not _target.is_file():
        return None
    try:
        with open(_target, "rb") as _fh:
            _envelope = pickle.load(_fh)  # nosec B301 — lokaler Nutzer-Cache im Projekt-Ausgabeverzeichnis (gleiche Vertrauensdomäne wie Modelldateien); fremde/manipulierte Einträge werden bei Fehler verworfen (§V6 (copilot-instructions.md))
        if not isinstance(_envelope, dict) or _envelope.get("schema") != _DISK_CACHE_SCHEMA:
            logger.warning("bridge: Analyse-Zwischenspeicher (Platte) Schema-Abweichung — Eintrag verworfen")
            _target.unlink(missing_ok=True)
            return None
        if _envelope.get("version") != _analysis_cache_version():
            logger.info(
                "bridge: Analyse-Zwischenspeicher (Platte) Versionswechsel (%s → %s) — Eintrag verworfen",
                _envelope.get("version"),
                _analysis_cache_version(),
            )
            _target.unlink(missing_ok=True)
            return None
        return _envelope.get("payload")
    except Exception as exc:
        logger.warning(
            "bridge: Analyse-Zwischenspeicher (Platte) laden fehlgeschlagen (%s) — Analyse wird neu berechnet (§V6 (copilot-instructions.md))",
            exc,
        )
        try:
            _target.unlink(missing_ok=True)
        except OSError as _del_exc:
            logger.debug(
                "bridge: Analyse-Zwischenspeicher (Platte) löschen nach Ladefehler nicht möglich (%s)", _del_exc
            )
        return None


def _disk_clear(subdir: str, key: str | None) -> None:
    """Löscht einen Platten-Eintrag (key) oder den ganzen Teil-Cache (key=None)."""
    try:
        if key is not None:
            _disk_path(subdir, key).unlink(missing_ok=True)
        else:
            shutil.rmtree(_disk_cache_root() / subdir, ignore_errors=True)
    except Exception as exc:
        logger.debug("bridge: Analyse-Zwischenspeicher (Platte) löschen fehlgeschlagen (%s)", exc)


# ---------------------------------------------------------------------------
# Singleton caches — one per analysis type for independent eviction
# ---------------------------------------------------------------------------

_defect_lru: _AnalysisLruCache = _AnalysisLruCache()
_era_genre_lru: _AnalysisLruCache = _AnalysisLruCache()
_medium_lru: _AnalysisLruCache = _AnalysisLruCache()
_restorability_lru: _AnalysisLruCache = _AnalysisLruCache()


# ---------------------------------------------------------------------------
# Defect-Scan-Cache  (Thread-sicher, LRU, content-addressed)
# ---------------------------------------------------------------------------


def cache_defect_result(file_path: str, result: object) -> None:
    """Cache a DefectScanner result under a content-addressed key.

    Thread-safe.  Uses LRU eviction (max 64 entries).  Identical audio
    stored under a different path will hit the same cache slot.
    """
    key = content_cache_key(file_path)
    _defect_lru.put(key, result, path_alias=file_path)
    _disk_write("defect", key, result)
    logger.debug("bridge: DefectScan zwischengespeichert for '%s' (key=%.8s…)", file_path, key)


def get_cached_defect_result(file_path: str) -> object | None:
    """Gibt a cached DefectScanner result or ``None`` zurück."""
    key = content_cache_key(file_path)
    result = _defect_lru.get(key)
    if result is None:
        result = _defect_lru.get_by_path(file_path)
    if result is None:
        result = _disk_read("defect", key)
        if result is not None:
            _defect_lru.put(key, result, path_alias=file_path)
    return result


def clear_defect_cache(file_path: str | None = None) -> None:
    """Entfernt one entry (by path) or all entries from the defect cache."""
    if file_path is not None:
        key = content_cache_key(file_path)
        _defect_lru.remove(key)
        _disk_clear("defect", key)
    else:
        _defect_lru.clear()
        _disk_clear("defect", None)


# ---------------------------------------------------------------------------
# Era/Genre-Cache  (Thread-sicher, LRU, content-addressed)
# ---------------------------------------------------------------------------


def cache_era_genre_result(
    file_path: str,
    era_result: object | None = None,
    genre_result: object | None = None,
) -> None:
    """Cache Era/Genre classification results for *file_path*.

    Thread-safe, LRU-evicting, content-addressed.
    """
    key = content_cache_key(file_path)
    _era_genre_lru.put(
        key,
        {"era_result": era_result, "genre_result": genre_result},
        path_alias=file_path,
    )
    _disk_write("era_genre", key, {"era_result": era_result, "genre_result": genre_result})
    logger.debug("bridge: Era/Genre zwischengespeichert for '%s' (key=%.8s…)", file_path, key)


def get_cached_era_genre_result(file_path: str) -> dict[str, object] | None:
    """Gibt cached Era/Genre results or ``None`` zurück.

    Returns:
        dict with keys ``era_result`` and ``genre_result``, or ``None``.
    """
    key = content_cache_key(file_path)
    result = _era_genre_lru.get(key)
    if result is None:
        result = _era_genre_lru.get_by_path(file_path)
    if result is None:
        _disk = _disk_read("era_genre", key)
        if isinstance(_disk, dict):
            result = _disk
            _era_genre_lru.put(key, result, path_alias=file_path)
    return result


def clear_era_genre_cache(file_path: str | None = None) -> None:
    """Entfernt one entry (by path) or all entries from the Era/Genre cache."""
    if file_path is not None:
        key = content_cache_key(file_path)
        _era_genre_lru.remove(key)
        _disk_clear("era_genre", key)
    else:
        _era_genre_lru.clear()
        _disk_clear("era_genre", None)


# ---------------------------------------------------------------------------
# Medium-Cache  (Thread-sicher, LRU, content-addressed)
# ---------------------------------------------------------------------------


def cache_medium_result(file_path: str, result: object) -> None:
    """Cache a MediumClassifier result for *file_path*."""
    key = content_cache_key(file_path)
    _medium_lru.put(key, result, path_alias=file_path)
    _disk_write("medium", key, result)
    logger.debug("bridge: Medium zwischengespeichert for '%s' (key=%.8s…)", file_path, key)


def get_cached_medium_result(file_path: str) -> object | None:
    """Gibt a cached MediumClassifier result or ``None`` zurück."""
    key = content_cache_key(file_path)
    result = _medium_lru.get(key)
    if result is None:
        result = _medium_lru.get_by_path(file_path)
    if result is None:
        result = _disk_read("medium", key)
        if result is not None:
            _medium_lru.put(key, result, path_alias=file_path)
    return result


def clear_medium_cache(file_path: str | None = None) -> None:
    """Invalidate medium cache entry for *file_path*, or entire cache when ``None``."""
    if file_path is None:
        _medium_lru.clear()
        _disk_clear("medium", None)
        logger.debug("bridge: Medium-Zwischenspeicher vollständig geleert.")
    else:
        key = content_cache_key(file_path)
        _medium_lru.remove(key)
        _medium_lru.remove(file_path)  # remove() handles path-alias too
        _disk_clear("medium", key)
        logger.debug("bridge: Medium-Zwischenspeicher für '%s' geleert.", file_path)


# ---------------------------------------------------------------------------
# Restorability-Cache  (Thread-sicher, LRU, content-addressed)
# ---------------------------------------------------------------------------


def cache_restorability_result(file_path: str, result: object) -> None:
    """Cache a RestorabilityEstimator result for *file_path*."""
    key = content_cache_key(file_path)
    _restorability_lru.put(key, result, path_alias=file_path)
    _disk_write("restorability", key, result)
    logger.debug("bridge: Restorability zwischengespeichert for '%s' (key=%.8s…)", file_path, key)


def get_cached_restorability_result(file_path: str) -> object | None:
    """Gibt a cached RestorabilityEstimator result or ``None`` zurück."""
    key = content_cache_key(file_path)
    result = _restorability_lru.get(key)
    if result is None:
        result = _restorability_lru.get_by_path(file_path)
    if result is None:
        result = _disk_read("restorability", key)
        if result is not None:
            _restorability_lru.put(key, result, path_alias=file_path)
    return result
