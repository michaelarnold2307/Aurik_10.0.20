"""
backend/core/gp_memory_migration.py — GP-Memory-Format-Migration (§6.4).

Migriert ~/.aurik/gp_memory/<material>.json auf das aktuelle Schema-Format
wenn die Datei eine ältere Version enthält. Läuft automatisch beim Laden.

Schema-Versionen:
    v1 (< 9.10): {"observations": [...]}  — kein "version"-Feld
    v2 (≥ 9.10): {"version": 2, "observations": [...], "best_score": float, ...}
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import pathlib
import threading
from typing import Any

logger = logging.getLogger(__name__)

# Aktuelles Schema-Version
GP_MEMORY_SCHEMA_VERSION: int = 2

# Maximale Beobachtungen pro Datei (LRU-Trim)
MAX_OBSERVATIONS: int = 500

_migration_lock = threading.Lock()


def migrate_gp_memory_file(path: pathlib.Path) -> dict[str, Any]:
    """Lädt eine GP-Memory-Datei und migriert sie ggf. auf die aktuelle Version.

    Gibt immer ein gültiges Dict zurück — niemals raise.
    Beschädigte Dateien werden zu <name>.corrupted.json umbenannt.

    Args:
        path: Absoluter Pfad zur <material>.json-Datei.

    Returns:
        Gültiges GP-Memory-Dict mit "version", "observations" und optionalen Feldern.
    """
    EMPTY: dict[str, Any] = {"version": GP_MEMORY_SCHEMA_VERSION, "observations": []}

    if not path.exists():
        return dict(EMPTY)

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        logger.warning("GP-Memory '%s' nicht lesbar (%s) — starte neu", path.name, exc)
        _backup_corrupted(path)
        return dict(EMPTY)

    if not isinstance(raw, dict):
        logger.warning("GP-Memory '%s' hat unbekanntes Format — starte neu", path.name)
        _backup_corrupted(path)
        return dict(EMPTY)

    version = raw.get("version", 1)

    # --- Migration v1 → v2 ---
    if version == 1:
        logger.info("GP-Memory '%s': migriere v1 → v2", path.name)
        raw = _migrate_v1_to_v2(raw)

    # --- Validierung ---
    observations = raw.get("observations", [])
    valid_obs = [
        o
        for o in observations
        if (isinstance(o, dict) and math.isfinite(o.get("score", float("nan"))) and isinstance(o.get("params"), dict))
    ]

    if len(valid_obs) < len(observations):
        logger.debug(
            "GP-Memory '%s': %d von %d Beobachtungen gültig",
            path.name,
            len(valid_obs),
            len(observations),
        )

    # LRU-Trim: neueste 500 behalten  (sortiert nach Einfügereihenfolge, neueste zuletzt)
    if len(valid_obs) > MAX_OBSERVATIONS:
        valid_obs = valid_obs[-MAX_OBSERVATIONS:]

    raw["observations"] = valid_obs
    raw["version"] = GP_MEMORY_SCHEMA_VERSION
    return raw  # type: ignore[no-any-return]


def _migrate_v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
    """Migriert v1-Format (kein version-Feld) auf v2."""
    observations = data.get("observations", [])
    best_score = max(
        (o.get("score", -1.0) for o in observations if math.isfinite(o.get("score", float("nan")))), default=0.0
    )
    return {
        "version": 2,
        "observations": observations,
        "best_score": best_score,
        "n_updates": len(observations),
    }


def save_gp_memory_file(path: pathlib.Path, data: dict[str, Any]) -> None:
    """Speichert GP-Memory-Dict atomar (temp-Datei + os.replace).

    Verhindert teilweise geschriebene JSON-Dateien.

    Args:
        path: Ziel-Pfad (wird erstellt/überschrieben).
        data: Zu speicherndes Dict (wird vor dem Schreiben validiert).
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Beobachtungen bereinigen
    observations = data.get("observations", [])
    valid_obs = [
        o
        for o in observations
        if isinstance(o, dict) and math.isfinite(o.get("score", float("nan"))) and isinstance(o.get("params"), dict)
    ]
    if len(valid_obs) > MAX_OBSERVATIONS:
        valid_obs = valid_obs[-MAX_OBSERVATIONS:]

    data["observations"] = valid_obs
    data["version"] = GP_MEMORY_SCHEMA_VERSION

    tmp_path = path.with_suffix(".tmp.json")
    try:
        tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_path, path)
        logger.debug("GP-Memory gespeichert: %s (%d Obs.)", path.name, len(valid_obs))
    except OSError as exc:
        logger.error("GP-Memory '%s' konnte nicht geschrieben werden: %s", path.name, exc)
        with contextlib.suppress(OSError):
            tmp_path.unlink(missing_ok=True)


def _backup_corrupted(path: pathlib.Path) -> None:
    """Benennt eine beschädigte Datei in <name>.corrupted.json um."""
    backup = path.with_suffix(".corrupted.json")
    try:
        with _migration_lock:
            if path.exists():
                path.rename(backup)
                logger.info("Beschädigte GP-Memory-Datei gesichert als: %s", backup.name)
    except OSError as exc:
        logger.debug("Backup von '%s' fehlgeschlagen: %s", path.name, exc)


def ensure_memory_dir(material: str, base_dir: pathlib.Path | None = None) -> pathlib.Path:
    """Gibt den Pfad zur GP-Memory-Datei zurück und stellt das Verzeichnis sicher.

    Args:
        material: Materialname (z. B. "tape", "vinyl", "schlager").
        base_dir: Optionaler alternativer Basisordner (Standard: ~/.aurik/gp_memory/).

    Returns:
        Pfad zur <material>.json-Datei.
    """
    if base_dir is None:
        base_dir = pathlib.Path.home() / ".aurik" / "gp_memory"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / f"{material}.json"
