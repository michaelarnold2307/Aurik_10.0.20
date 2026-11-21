"""Aurik Bridge — Cache Infrastructure (§11 Spec 08)
===================================================
Thread-safe LRU caches for analysis results (defect, era/genre, medium,
restorability). Content-addressed keys prevent redundant re-analysis when
files are renamed or moved.
Cross-process disk entries use a size-bounded JSON codec and a fixed type
allowlist; cache data is never deserialized as executable Python objects.

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

import base64
import dataclasses
import enum
import hashlib
import json
import logging
import math
import os
import shutil
import stat
import sys
import tempfile
import threading
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

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
_DISK_CACHE_SCHEMA = 2
_DISK_CACHE_MAX_BYTES = 64 * 1024 * 1024
_DISK_CACHE_DISABLED = os.environ.get("AURIK_ANALYSIS_CACHE", "1").strip().lower() not in ("", "1", "true", "yes")
_disk_write_lock = threading.Lock()
_SAFE_CACHE_DICT_TYPES = {
    "backend.core.defect_scanner.DefectLocationsView",
    "backend.core.defect_scanner.DefectScoreView",
}


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
        # Optionaler Callback bei Eviction — damit disk-gestützte Caches den
        # Disk-Eintrag mitlöschen können (sonst liest get() den verdrängten
        # Eintrag vom Platten-Fallback zurück → LRU wirkungslos, Befund 2026-09-21).
        self.evict_callback: Any | None = None

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
                if self.evict_callback is not None:
                    try:
                        self.evict_callback(evicted_key)
                    except Exception as _ev_exc:  # §V6 (copilot-instructions.md): nie blockierend
                        logger.debug("LRU-Verdrängung: Platten-Aufräumung fehlgeschlagen: %s", _ev_exc)

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
    """Return a per-user cache root; an explicit environment override is supported."""
    _env = os.environ.get("AURIK_ANALYSIS_CACHE_DIR")
    if _env:
        return Path(_env)
    if os.name == "nt":
        _base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return _base / "Aurik" / "Cache" / "analysis_cache"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "Aurik" / "analysis_cache"
    _xdg = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    if not _xdg.is_absolute():
        _xdg = Path.home() / ".cache"
    return _xdg / "aurik" / "analysis_cache"


def _prepare_disk_cache_dir(subdir: str, *, create: bool) -> Path | None:
    """Prepare an app-owned cache directory and reject symlink or writable roots."""
    _root = _disk_cache_root()
    _cache_dir = _root / subdir
    for _directory in (_root, _cache_dir):
        if create:
            _directory.mkdir(parents=True, mode=0o700, exist_ok=True)
        elif not _directory.exists():
            return None
        if _directory.is_symlink() or not _directory.is_dir():
            raise PermissionError(f"Analyse-Cache-Verzeichnis ist kein sicheres Verzeichnis: {_directory}")
        if os.name == "posix":
            _directory_stat = _directory.stat(follow_symlinks=False)
            if _directory_stat.st_uid != os.getuid() or _directory_stat.st_mode & 0o022:
                raise PermissionError(
                    f"Analyse-Cache-Verzeichnis ist fremd oder gruppen-/weltbeschreibbar: {_directory}"
                )
            _directory.chmod(0o700)
    return _cache_dir


def _analysis_cache_version() -> str:
    """Analyse-Code-Version für die Cache-Invalidierung (§G5 (copilot-instructions.md): Version im Key)."""
    try:
        from backend.core.version import AURIK_VERSION  # lazy: leaf-Modul, kein Zyklus

        # NumPy-Major mit in die Cache-Version aufnehmen: Cache-Dateien älterer
        # NumPy-Generationen (numpy._core-Umbau) sind nicht mehr ladbar — sie
        # werden dann über den regulären Versionswechsel-Pfad (INFO) verworfen
        # statt beim Laden zu crashen.
        try:
            import numpy as _np_ver

            _np_major = int(str(_np_ver.__version__).split(".")[0])
        except Exception:
            _np_major = 0
        return f"{AURIK_VERSION}-np{_np_major}"
    except Exception as _exc:
        logger.debug("bridge: AURIK_VERSION nicht lesbar (%s) — Zwischenspeicher-Version 'unbekannt'", _exc)
        return "unknown"


@lru_cache(maxsize=1)
def _disk_cache_types() -> dict[str, type]:
    """Fixed type allowlist for typed analysis results; disk data cannot import classes."""
    from backend.core.defect_scanner import (
        DefectAnalysisResult,
        DefectLocationsView,
        DefectScore,
        DefectScoreView,
        DefectType,
        MaterialType,
    )
    from backend.core.era_classifier import EraResult
    from backend.core.genre_classifier import SchlagerClassificationResult
    from backend.core.medium_classifier import ClassificationResult, MaterialEvidence
    from backend.core.restorability_estimator import RestorabilityResult
    from forensics.medium_detector import MediumDetectionResult, SpectralFingerprint, TransferChain

    safe_types = (
        ClassificationResult,
        DefectAnalysisResult,
        DefectLocationsView,
        DefectScore,
        DefectScoreView,
        DefectType,
        EraResult,
        MaterialEvidence,
        MediumDetectionResult,
        MaterialType,
        RestorabilityResult,
        SchlagerClassificationResult,
        SpectralFingerprint,
        TransferChain,
    )
    return {f"{safe_type.__module__}.{safe_type.__qualname__}": safe_type for safe_type in safe_types}


def _encode_disk_value(value: object) -> object:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Nicht-endliche Cache-Zahl")
        return value

    import numpy as np

    if isinstance(value, np.ndarray):
        _array = np.ascontiguousarray(value)
        if _array.dtype.hasobject or _array.dtype.fields is not None or _array.dtype.kind not in "biufc":
            raise TypeError(f"Nicht unterstützter Array-Datentyp: {_array.dtype}")
        if _array.nbytes > _DISK_CACHE_MAX_BYTES:
            raise ValueError("Cache-Array überschreitet das Größenlimit")
        return {
            "$aurik": "ndarray",
            "dtype": _array.dtype.str,
            "shape": list(_array.shape),
            "data": base64.b64encode(_array.tobytes()).decode("ascii"),
        }
    if isinstance(value, np.generic):
        _scalar = np.asarray(value)
        if _scalar.dtype.hasobject or _scalar.dtype.fields is not None or _scalar.dtype.kind not in "biufc":
            raise TypeError(f"Nicht unterstützter Array-Datentyp: {_scalar.dtype}")
        return {
            "$aurik": "numpy_scalar",
            "dtype": _scalar.dtype.str,
            "data": base64.b64encode(_scalar.tobytes()).decode("ascii"),
        }

    _registry = _disk_cache_types()
    _type_name = f"{type(value).__module__}.{type(value).__qualname__}"
    if isinstance(value, enum.Enum):
        if _registry.get(_type_name) is not type(value):
            raise TypeError(f"Nicht freigegebener Enum-Typ: {_type_name}")
        return {"$aurik": "enum", "type": _type_name, "value": _encode_disk_value(value.value)}
    if isinstance(value, type):
        raise TypeError(f"Cache-Klassenobjekte sind nicht unterstützt: {_type_name}")
    if dataclasses.is_dataclass(value):
        if _registry.get(_type_name) is not type(value):
            raise TypeError(f"Nicht freigegebener Datentyp: {_type_name}")
        return {
            "$aurik": "dataclass",
            "type": _type_name,
            "fields": {
                field.name: _encode_disk_value(getattr(value, field.name)) for field in dataclasses.fields(value)
            },
        }
    if type(value) is dict or (_type_name in _SAFE_CACHE_DICT_TYPES and _registry.get(_type_name) is type(value)):
        if not isinstance(value, dict):
            raise TypeError(f"Nicht unterstützter Cache-Dictionary-Typ: {_type_name}")
        _mapping = value
        _items = [(_encode_disk_value(key), _encode_disk_value(item)) for key, item in _mapping.items()]
        _items.sort(key=lambda pair: json.dumps(pair[0], sort_keys=True, separators=(",", ":"), ensure_ascii=False))
        return {
            "$aurik": "dict",
            "type": _type_name if type(value) is not dict else None,
            "items": _items,
        }
    if type(value) is list:
        return {"$aurik": "list", "items": [_encode_disk_value(item) for item in value]}
    if type(value) is tuple:
        return {"$aurik": "tuple", "items": [_encode_disk_value(item) for item in value]}
    raise TypeError(f"Nicht unterstützter Cache-Datentyp: {_type_name}")


def _decode_disk_value(value: object) -> object:
    if value is None or type(value) in (bool, int, float, str):
        return value
    if not isinstance(value, dict):
        raise ValueError("Ungültiger Cache-Wert")

    _kind = value.get("$aurik")
    if _kind in {"ndarray", "numpy_scalar"}:
        import numpy as np

        _dtype = np.dtype(value["dtype"])
        if _dtype.hasobject or _dtype.fields is not None or _dtype.kind not in "biufc":
            raise ValueError("Nicht unterstützter Cache-Array-Datentyp")
        _raw = base64.b64decode(value["data"], validate=True)
        if len(_raw) > _DISK_CACHE_MAX_BYTES:
            raise ValueError("Cache-Array überschreitet das Größenlimit")
        if _kind == "numpy_scalar":
            if len(_raw) != _dtype.itemsize:
                raise ValueError("Ungültige Cache-Skalargröße")
            return np.frombuffer(_raw, dtype=_dtype).copy()[0]

        _shape = value["shape"]
        if (
            not isinstance(_shape, list)
            or len(_shape) > 8
            or any(type(dimension) is not int or dimension < 0 for dimension in _shape)
        ):
            raise ValueError("Ungültige Cache-Arrayform")
        _expected_size = math.prod(_shape) * _dtype.itemsize
        if _expected_size != len(_raw) or _expected_size > _DISK_CACHE_MAX_BYTES:
            raise ValueError("Ungültige Cache-Arraygröße")
        return np.frombuffer(_raw, dtype=_dtype).reshape(_shape).copy()

    if _kind == "enum":
        _enum_type_name = value.get("type")
        if not isinstance(_enum_type_name, str):
            raise ValueError("Nicht freigegebener Cache-Enum")
        _enum_candidate: object = _disk_cache_types().get(_enum_type_name)
        if not isinstance(_enum_candidate, type) or not issubclass(_enum_candidate, enum.Enum):
            raise ValueError("Nicht freigegebener Cache-Enum")
        return _enum_candidate(_decode_disk_value(value["value"]))
    if _kind in {"list", "tuple"}:
        _decoded_items: list[object] = [_decode_disk_value(item) for item in value.get("items", [])]
        return _decoded_items if _kind == "list" else tuple(_decoded_items)
    if _kind == "dict":
        _dict_items: list[tuple[object, object]] = [
            (_decode_disk_value(key), _decode_disk_value(item)) for key, item in value.get("items", [])
        ]
        _dict_type_name: object = value.get("type")
        if _dict_type_name is None:
            return dict(_dict_items)
        if not isinstance(_dict_type_name, str) or _dict_type_name not in _SAFE_CACHE_DICT_TYPES:
            raise ValueError("Nicht freigegebener Cache-Dictionary-Typ")
        _dict_type = _disk_cache_types().get(str(_dict_type_name))
        if _dict_type is None:
            raise ValueError("Nicht freigegebener Cache-Dictionary-Typ")
        _result = _dict_type()
        _result.update(_dict_items)
        return _result
    if _kind == "dataclass":
        _class_type_name = value.get("type")
        if not isinstance(_class_type_name, str):
            raise ValueError("Nicht freigegebener Cache-Datentyp")
        _class = _disk_cache_types().get(_class_type_name)
        if _class is None or not dataclasses.is_dataclass(_class):
            raise ValueError("Nicht freigegebener Cache-Datentyp")
        _fields = value.get("fields")
        _field_defs = {field.name: field for field in dataclasses.fields(_class)}
        if not isinstance(_fields, dict) or set(_fields) != set(_field_defs):
            raise ValueError("Ungültige Cache-Dataclass-Felder")
        _decoded = {name: _decode_disk_value(field_value) for name, field_value in _fields.items()}
        _instance = _class(**{name: field_value for name, field_value in _decoded.items() if _field_defs[name].init})
        for _name, _field_value in _decoded.items():
            if not _field_defs[_name].init:
                setattr(_instance, _name, _field_value)
        return _instance
    raise ValueError("Unbekannter Cache-Werttyp")


def _disk_path(subdir: str, key: str) -> Path:
    """JSON-Plattenpfad eines Eintrags; Dateiname = SHA-256(key)."""
    return _disk_cache_root() / subdir / (hashlib.sha256(key.encode()).hexdigest() + ".json")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Ungueltige JSON-Zahl: {value}")


def _disk_write(subdir: str, key: str, payload: object) -> None:
    """Schreibt ein Analyse-Ergebnis atomar auf Platte (tmp + os.replace).

    Fehler sind nie blockierend (§V6 (copilot-instructions.md)): Analyse wird
    dann beim nächsten Prozess schlicht neu berechnet.
    """
    if _DISK_CACHE_DISABLED:
        return
    _tmp: Path | None = None
    try:
        _target = _disk_path(subdir, key)
        _prepare_disk_cache_dir(subdir, create=True)
        _envelope = {
            "schema": _DISK_CACHE_SCHEMA,
            "version": _analysis_cache_version(),
            "key": key,
            "payload": _encode_disk_value(payload),
        }
        _serialized = json.dumps(_envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        if len(_serialized) > _DISK_CACHE_MAX_BYTES:
            raise ValueError("Analyse-Cache-Eintrag überschreitet das Größenlimit")
        with _disk_write_lock:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=_target.parent,
                prefix=f".{_target.name}.",
                suffix=".tmp",
                delete=False,
            ) as _fh:
                _tmp = Path(_fh.name)
                _fh.write(_serialized)
            if os.name == "posix":
                _tmp.chmod(0o600)
            os.replace(_tmp, _target)
            _tmp = None
        logger.debug("bridge: Analyse-Zwischenspeicher (Platte) geschrieben %s/%.8s…", subdir, key)
    except Exception as exc:
        logger.warning(
            "bridge: Analyse-Zwischenspeicher (Platte) schreiben fehlgeschlagen (%s) — Analyse wird neu berechnet (§V6 (copilot-instructions.md))",
            exc,
        )
    finally:
        if _tmp is not None:
            try:
                _tmp.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("bridge: temporärer Analyse-Zwischenspeicher konnte nicht entfernt werden (%s)", exc)


def _disk_read(subdir: str, key: str) -> object | None:
    """Lädt ein Analyse-Ergebnis von Platte; ``None`` bei Miss/Versionswechsel/Fehler.

    Versions- oder Schema-Mismatch sowie nicht lesbare Einträge werden
    sichtbar (§V6 (copilot-instructions.md)) verworfen und gelöscht — nie still verwendet.
    """
    if _DISK_CACHE_DISABLED:
        return None
    _target = _disk_path(subdir, key)
    try:
        _prepare_disk_cache_dir(subdir, create=False)
        try:
            _target.with_suffix(".pkl").unlink(missing_ok=True)
        except OSError as exc:
            logger.info("bridge: veralteter Pickle-Zwischenspeicher konnte nicht entfernt werden (%s)", exc)
        _flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        _fd = os.open(_target, _flags)
        with os.fdopen(_fd, "rb") as _fh:
            _target_stat = os.fstat(_fh.fileno())
            if not stat.S_ISREG(_target_stat.st_mode):
                raise PermissionError(f"Analyse-Cache-Eintrag ist keine reguläre Datei: {_target}")
            if os.name == "posix" and (_target_stat.st_uid != os.getuid() or _target_stat.st_mode & 0o022):
                logger.warning(
                    "bridge: nicht vertrauenswürdiger Analyse-Zwischenspeicher-Eintrag — Cache-Miss (%s)", _target
                )
                return None
            if _target_stat.st_size > _DISK_CACHE_MAX_BYTES:
                raise ValueError("Analyse-Cache-Eintrag überschreitet das Größenlimit")
            _raw = _fh.read(_DISK_CACHE_MAX_BYTES + 1)
        if len(_raw) > _DISK_CACHE_MAX_BYTES:
            raise ValueError("Analyse-Cache-Eintrag überschreitet das Größenlimit")
        _envelope = json.loads(_raw.decode("utf-8"), parse_constant=_reject_json_constant)
        if not isinstance(_envelope, dict) or _envelope.get("schema") != _DISK_CACHE_SCHEMA:
            logger.warning("bridge: Analyse-Zwischenspeicher (Platte) Schema-Abweichung — Eintrag verworfen")
            _target.unlink(missing_ok=True)
            return None
        if _envelope.get("key") != key:
            logger.warning("bridge: Analyse-Zwischenspeicher (Platte) Key-Abweichung — Eintrag verworfen")
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
        return _decode_disk_value(_envelope["payload"])
    except FileNotFoundError:
        return None
    except PermissionError as exc:
        logger.warning(
            "bridge: unsicheres Zwischenspeicher-Verzeichnis — persistenter Zwischenspeicher wird nicht genutzt (%s)",
            exc,
        )
        return None
    except Exception as exc:
        # Cache-Load-Fehler kosten keine Qualität (Analyse wird neu berechnet) —
        # INFO statt WARNING; §V6 (copilot-instructions.md) bezieht sich auf
        # Qualitäts-Fallbacks.
        logger.info(
            "bridge: Analyse-Zwischenspeicher (Platte) laden fehlgeschlagen (%s) — Analyse wird neu berechnet",
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
            _target = _disk_path(subdir, key)
            _target.unlink(missing_ok=True)
            _target.with_suffix(".pkl").unlink(missing_ok=True)
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

# §LRU-Disk-Konsistenz (2026-09-21): Eviction muss den Platten-Eintrag
# mitlöschen — sonst liest get_cached_*() den verdrängten Eintrag über den
# Platten-Fallback zurück und die LRU-Semantik ist wirkungslos
# (Voll-Suite-Befund: test_lru_keeps_recently_accessed_entry).
_defect_lru.evict_callback = lambda key: _disk_clear("defect", key)
_era_genre_lru.evict_callback = lambda key: _disk_clear("era_genre", key)


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
