#!/usr/bin/env python3
"""Regressionstests GPU-Kette 1 — Verdrahtung der Jobs R2–R5 (Befund 2026-11-06).

Belege aus ``docs/reports/current/``:

* ``train_job_m9_tf_gridnet.json`` — R2 (SOTA-M9) starb nach 5,5 s an
  ``torch.OutOfMemoryError`` (Batch 4 ⇒ 4·513 LSTM-Sequenzen, 4,42 GiB
  zusätzlich angefragt). Der Job hat folglich **nie** trainiert.
* ``train_job_c2_cantus_{pretrain,finetune,domain}.json`` — R3–R5 riefen
  ``scripts/train_cantus.py`` **ohne** ``--data`` auf (Exit 2, „--data fehlt"),
  obwohl ``data/cantus_pairs`` mit 3600 Paaren lokal liegt (SOTA-M14).

Die Tests pinnen die Ketten-Verdrahtung, damit ein späterer Refactor die Jobs
nicht wieder still ins Leere laufen lässt (§V6/§V7 (copilot-instructions.md):
kein stiller Fehlpfad, Ursache statt Symptom).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "gpu_kette1_welle_c.sh"


def _text() -> str:
    return _SCRIPT.read_text(encoding="utf-8")


def _block(name: str) -> str:
    """Textblock eines Jobs: ab ``=== <name>`` bis zum nächsten Job-Echo."""
    text = _text()
    start = text.find(f"=== {name}")
    assert start != -1, f"Job {name} fehlt in {_SCRIPT.name}"
    rest = text[start:]
    nxt = rest.find('echo "=== ', 10)
    return rest if nxt == -1 else rest[:nxt]


def test_kette1_shell_syntax_gueltig() -> None:
    """Die Kette muss syntaktisch ladbar sein, bevor sie Jobs startet."""
    res = subprocess.run(["bash", "-n", str(_SCRIPT)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


def test_m9_haelt_das_speicherbudget_ein() -> None:
    """Jeder M9-Aufruf trägt Micro-Batch 1 + Gradient-Akkumulation 4."""
    block = _block("R2:")
    assert "--batch-size 1 --grad-accum 4" in block, "R2 ohne Speicherbudget"
    for zeile in block.splitlines():
        if "train_tf_gridnet_denoiser.py" in zeile:
            assert "--grad-accum 4" in zeile, f"M9-Aufruf ohne Akkumulation: {zeile.strip()}"


def test_m9_trainiert_nur_auf_echten_musik_paaren() -> None:
    """Sinus+Rauschen (``--synthetic``) wäre ein K1-Domänen-Defizit (§III.11)."""
    block = _block("R2:")
    assert "data/tfgrid_pairs/clean" in block
    assert "data/tfgrid_pairs/noisy" in block
    for zeile in block.splitlines():
        if zeile.lstrip().startswith("#"):
            continue
        if "train_tf_gridnet_denoiser.py" in zeile:
            assert "--synthetic" not in zeile, f"M9 auf synthetischen Sinustönen: {zeile.strip()}"


def test_cantus_pretrain_und_finetune_uebergeben_das_paar_manifest() -> None:
    """R3/R4 müssen ``--data`` mit dem Paar-Manifest übergeben (sonst Exit 2)."""
    text = _text()
    assert "data/cantus_pairs/manifest.jsonl" in text, "Manifest-Erkennung fehlt"
    assert 'if [ -n "$CANTUS_DATA" ]' in text, "fail-closed-Guard für R3/R4 fehlt"
    assert text.count("--data $CANTUS_DATA") == 4, "R3/R4 (cmd + resume-cmd) ohne --data"
    for job in ("R3:", "R4:"):
        assert "--data $CANTUS_DATA" in _block(job), f"{job} ohne --data"


def test_cantus_domain_adapt_nur_mit_realen_degradationen() -> None:
    """R5 darf nicht mit synthetischen Paaren als Domänen-Adaptation laufen."""
    text = _text()
    assert 'if [ -n "$CANTUS_REAL" ]' in text, "Guard für R5 fehlt"
    assert text.count("--data $CANTUS_REAL") == 2, "R5 (cmd + resume-cmd) ohne --data"
    assert "*_degraded.wav" in text, "Erkennung realer Paare fehlt"
    block = _block("R5:")
    assert "--data $CANTUS_REAL" in block
