"""Tests für den Trainings-Job-Runner (Pfeiler 4: Run-ID + Lock + Auto-Resume)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.training_artifacts import acquire_run_lock
from scripts.train_job_runner import JOB_LOCK_NAME, run_job

PY_OK = [sys.executable, "-c", "print('ok')"]
PY_FAIL = [sys.executable, "-c", "raise SystemExit(3)"]
PY_SLEEP = [sys.executable, "-c", "import time; time.sleep(60)"]


def test_manifest_traegt_befehl_seed_phase_checkpoint_hash(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model_a.pt").write_bytes(b"gewichte")
    man = tmp_path / "man.json"
    code = run_job(
        name="t1",
        ckpt_dir=ckpt,
        cmd=PY_OK,
        resume_cmd=PY_OK,
        phase="v2",
        seed=42,
        timeout_s=60,
        retries=0,
        manifest_path=man,
    )
    assert code == 0
    m = json.loads(man.read_text(encoding="utf-8"))
    assert m["run_id"].startswith("v2-"), "Run-ID trägt die Phase"
    assert m["befehl"] and m["resume_befehl"], "Resume-Befehl steht im Manifest (Pfeiler 4)"
    assert m["phase"] == "v2" and m["seed"] == 42
    assert "model_a.pt" in m["checkpoint_hashes"], "Checkpoint-Hash im Manifest"
    assert m["ergebnis"] == "erfolgreich" and len(m["versuche"]) == 1


def test_timeout_und_auto_resume(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    man = tmp_path / "man.json"
    code = run_job(
        name="t2", ckpt_dir=ckpt, cmd=PY_SLEEP, resume_cmd=PY_OK, phase="x", timeout_s=2, retries=1, manifest_path=man
    )
    assert code == 0, "Resume-Versuch läuft durch"
    m = json.loads(man.read_text(encoding="utf-8"))
    assert len(m["versuche"]) == 2
    assert m["versuche"][0]["timed_out"] is True, "erster Versuch lief in den Timeout"
    assert m["versuche"][1]["exit_code"] == 0
    assert m["versuche"][1]["befehl"] == " ".join(PY_OK), "zweiter Versuch nutzt Resume-Befehl"
    assert m["ergebnis"] == "erfolgreich"


def test_lock_verhindert_doppellauf(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    held = acquire_run_lock(ckpt, "run-fremd", lock_name=JOB_LOCK_NAME)
    assert held is not None
    try:
        man = tmp_path / "man.json"
        code = run_job(
            name="t3", ckpt_dir=ckpt, cmd=PY_OK, resume_cmd=PY_OK, phase="x", timeout_s=30, retries=0, manifest_path=man
        )
        assert code == 2, "kein Doppel-Lauf (Pfeiler 4)"
        assert not man.exists(), "gesperrter Lauf schreibt kein Manifest"
    finally:
        held.close()


def test_alle_versuche_fehlgeschlagen(tmp_path: Path) -> None:
    man = tmp_path / "man.json"
    code = run_job(
        name="t4",
        ckpt_dir=tmp_path / "ckpt",
        cmd=PY_FAIL,
        resume_cmd=PY_FAIL,
        phase="x",
        timeout_s=30,
        retries=1,
        manifest_path=man,
    )
    assert code == 3
    m = json.loads(man.read_text(encoding="utf-8"))
    assert m["ergebnis"] == "fehlgeschlagen"
    assert [v["exit_code"] for v in m["versuche"]] == [3, 3], "Resume wurde versucht"
