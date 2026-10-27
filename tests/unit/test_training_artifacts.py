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


# ── Pfeiler 4 — Run-Lock & Provenienz (Single-Writer, SOTA-Roadmap) ─────────
# Eigentümer-Datei ist scripts/train_cantus.py; die Tests sind bewusst leicht
# (kein ML) und laufen damit in jedem Unit-Smoke.


def _load_train_cantus():
    import importlib.util

    path = tmp_root = __file__
    from pathlib import Path as _Path

    path = _Path(tmp_root).resolve().parents[2] / "scripts" / "train_cantus.py"
    spec = importlib.util.spec_from_file_location("train_cantus_runlock", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_lock_blockiert_zweiten_lauf(tmp_path) -> None:
    mod = _load_train_cantus()
    run_a = mod._acquire_run_lock(tmp_path, "run-a")
    assert run_a is not None
    try:
        assert mod._acquire_run_lock(tmp_path, "run-b") is None
    finally:
        run_a.close()
    run_c = mod._acquire_run_lock(tmp_path, "run-c")
    assert run_c is not None
    run_c.close()


def test_run_lock_traegt_run_id_und_kommando_ein(tmp_path) -> None:
    import json

    mod = _load_train_cantus()
    fh = mod._acquire_run_lock(tmp_path, "run-x")
    assert fh is not None
    try:
        data = json.loads((tmp_path / ".train.lock").read_text(encoding="utf-8"))
        assert data["run_id"] == "run-x"
        assert data["command"]
    finally:
        fh.close()


def test_sha256_file_ist_stabil(tmp_path) -> None:
    mod = _load_train_cantus()
    f = tmp_path / "a.bin"
    f.write_bytes(b"aurik")
    assert mod._sha256_file(f) == mod._sha256_file(f)
    assert len(mod._sha256_file(f)) == 64
