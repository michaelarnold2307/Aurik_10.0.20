"""scripts/backup_gp_memory.py — Sicherung des GP-Gedächtnisses (§6.4).

Erstellt ein datiertes Backup aller ~/.aurik/gp_memory/*.json-Dateien
und optional der artist_signatures/ und batch_sessions/.

Ausführen:
    python scripts/backup_gp_memory.py [--dest ORDNER]
    python scripts/backup_gp_memory.py --restore backup_2026-03-09.tar.gz
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
from datetime import datetime

AURIK_DIR = pathlib.Path.home() / ".aurik"

BACKUP_DIRS = [
    "gp_memory",
    "artist_signatures",
    "batch_sessions",
    "era_cache",
    "presets",
]


def backup(dest_dir: pathlib.Path) -> pathlib.Path:
    """Sichert alle Aurik-Lernkurven als tar.gz-Archiv.

    Args:
        dest_dir: Zielordner für das Archiv.

    Returns:
        Pfad zum erstellten Archiv.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    archive_name = dest_dir / f"aurik_gp_backup_{timestamp}"

    # Temporärer Sammelordner
    tmp_dir = dest_dir / f"_aurik_backup_tmp_{timestamp}"
    tmp_dir.mkdir()

    try:
        total_files = 0
        for subdir_name in BACKUP_DIRS:
            src = AURIK_DIR / subdir_name
            if src.exists():
                dst = tmp_dir / subdir_name
                shutil.copytree(src, dst)
                n = sum(1 for _ in dst.rglob("*") if _.is_file())
                total_files += n
                print(f"  {subdir_name}/: {n} Datei(en)")
            else:
                print(f"  {subdir_name}/: nicht vorhanden — übersprungen")

        # tar.gz erstellen
        archive_path = shutil.make_archive(str(archive_name), "gztar", root_dir=str(tmp_dir), base_dir=".")
        print(f"\nBackup erstellt: {archive_path}")
        print(f"Gesicherte Dateien: {total_files}")
        return pathlib.Path(archive_path)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def restore(archive_path: pathlib.Path) -> None:
    """Stellt ein Backup aus einem Archiv wieder her.

    Args:
        archive_path: Pfad zum .tar.gz-Archiv.
    """
    if not archive_path.exists():
        print(f"Fehler: Archiv nicht gefunden: {archive_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Stelle Backup wieder her: {archive_path}")
    print(f"Zielordner: {AURIK_DIR}")

    confirm = input("Bestehende Lernkurven werden überschrieben. Fortfahren? [j/N] ")
    if confirm.lower() not in ("j", "ja", "y", "yes"):
        print("Abgebrochen.")
        return

    AURIK_DIR.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(str(archive_path), extract_dir=str(AURIK_DIR))
    print("Wiederherstellung abgeschlossen.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aurik 10.0.0 GP-Memory Backup/Restore")
    parser.add_argument(
        "--dest",
        default=str(pathlib.Path.home() / "aurik_backups"),
        help="Zielordner für das Backup (Standard: ~/aurik_backups)",
    )
    parser.add_argument(
        "--restore",
        metavar="ARCHIV",
        help="Backup aus angegebenem .tar.gz-Archiv wiederherstellen",
    )
    args = parser.parse_args()

    if args.restore:
        restore(pathlib.Path(args.restore))
    else:
        backup(pathlib.Path(args.dest))


if __name__ == "__main__":
    main()
