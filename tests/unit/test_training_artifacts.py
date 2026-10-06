"""Tests für den Schutz der Trainingsartefakte (Backup + atomares Schreiben).

Deckt den Befund 2026-10-06 ab: Trainingsskripte dürfen vorhandene Checkpoints
nie still überschreiben — vorher wird ein rotierendes Backup angelegt, und eine
anomale Verkleinerung (Smoke auf Vollmodell / Presetwechsel) wird gewarnt.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from backend.core.training_artifacts import backup_existing, save_guarded, write_text_guarded


def test_save_guarded_backs_up_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "checkpoint_latest.pt"
    assert save_guarded(target, {"v": 1}) is None  # keine Vorversion
    save_guarded(target, {"v": 2})

    backups = list((tmp_path / ".checkpoint_backups").glob("checkpoint_latest.pt.*.bak"))
    assert len(backups) == 1, "Vorversion muss gesichert sein"
    assert torch.load(backups[0], weights_only=True)["v"] == 1
    assert torch.load(target, weights_only=True)["v"] == 2


def test_save_guarded_no_backup_without_previous(tmp_path: Path) -> None:
    result = save_guarded(tmp_path / "new.pt", {"v": 1})
    assert result is None
    assert not (tmp_path / ".checkpoint_backups").exists()


def test_backup_rotation_keeps_only_n(tmp_path: Path) -> None:
    target = tmp_path / "c.pt"
    for i in range(5):
        save_guarded(target, {"v": i}, keep=2)
    backups = list((tmp_path / ".checkpoint_backups").glob("c.pt.*.bak"))
    assert len(backups) == 2, "Rotation muss auf `keep` begrenzen"


def test_save_guarded_warns_on_shrink(tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
    target = tmp_path / "m.pt"
    save_guarded(target, {"w": torch.zeros(200_000)})  # ~0,8 MB
    with caplog.at_level(logging.WARNING):
        save_guarded(target, {"w": torch.zeros(10)})  # ~40 B
    assert "schrumpft stark" in caplog.text


def test_save_guarded_is_atomic_no_tmp_left(tmp_path: Path) -> None:
    target = tmp_path / "a.pt"
    save_guarded(target, {"v": 1})
    save_guarded(target, {"v": 2})
    assert not list(tmp_path.glob("*.tmp")), "keine tmp-Reste"


def test_write_text_guarded_backs_up(tmp_path: Path) -> None:
    target = tmp_path / "report.json"
    write_text_guarded(target, '{"a": 1}')
    write_text_guarded(target, '{"a": 2}')

    backups = list((tmp_path / ".checkpoint_backups").glob("report.json.*.bak"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == '{"a": 1}'
    assert target.read_text(encoding="utf-8") == '{"a": 2}'


def test_backup_existing_returns_none_for_missing(tmp_path: Path) -> None:
    assert backup_existing(tmp_path / "absent.pt") is None
