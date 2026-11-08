#!/usr/bin/env python3
"""Regressionstests der GPU-Ketten-Verdrahtung (Befund 2026-11-06).

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

import importlib.util
import re
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "gpu_kette1_welle_c.sh"
_KETTEN = (
    _ROOT / "scripts" / "gpu_kette1_welle_c.sh",
    _ROOT / "scripts" / "gpu_kette2_matrix_d3.sh",
    _ROOT / "scripts" / "gpu_kette3_k4_rest.sh",
    _ROOT / "scripts" / "gpu_kette4_modellkoepfe.sh",
)
_FLAG_RE = re.compile(r"--[a-z0-9][a-z0-9-]*")
_AUFTRAG_RE = re.compile(r"([a-z0-9_]+)\.py(?P<rest>.*)")


def _trainer_flags(script: Path) -> set[str]:
    """Bekannte Flags eines Trainers (``add_argument("--x" …)``)."""
    quelle = script.read_text(encoding="utf-8")
    return set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', quelle))


def _text(script: Path = _SCRIPT) -> str:
    return script.read_text(encoding="utf-8")


def _block(name: str, script: Path = _SCRIPT) -> str:
    """Textblock eines Jobs: ab ``=== <name>`` bis zum nächsten Job-Echo."""
    text = _text(script)
    start = text.find(f"=== {name}")
    assert start != -1, f"Job {name} fehlt in {script.name}"
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


def test_kette3_traegt_dasselbe_m9_speicherbudget() -> None:
    """Kette 3 trainiert denselben Job — ohne die Flags fiele sie auf Batch 1 zurück."""
    kette3 = _SCRIPT.with_name("gpu_kette3_k4_rest.sh")
    res = subprocess.run(["bash", "-n", str(kette3)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    block = _block("R2:", kette3)
    assert "data/tfgrid_pairs/clean" in block
    for zeile in block.splitlines():
        if zeile.lstrip().startswith("#"):
            continue
        if "train_tf_gridnet_denoiser.py" in zeile:
            assert "--batch-size 1 --grad-accum 4" in zeile, f"Kette-3-M9 ohne Speicherbudget: {zeile.strip()}"
            assert "--synthetic" not in zeile


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


def test_ketten_flags_existieren_in_den_trainern() -> None:
    """Kein Ketten-Aufruf darf ein Flag übergeben, das der Trainer nicht kennt.

    Ein unbekanntes Flag beendet argparse mit Exit 2 — der Job stirbt wie die
    Cantus-Läufe nach 2 s, ohne dass es wie ein Verdrahtungsfehler aussieht.
    Geprüft werden alle Ketten (1–4) gegen die ``add_argument``-Flags des
    jeweiligen Zielskripts.
    """
    geprueft = 0
    for kette in _KETTEN:
        for zeile in kette.read_text(encoding="utf-8").splitlines():
            if zeile.lstrip().startswith("#"):
                continue
            # Jede Zeile kann mehrere Aufrufe tragen (bash -c mit if/else).
            for teil in re.split(r"scripts/", zeile)[1:]:
                treffer = _AUFTRAG_RE.match(teil)
                if not treffer:
                    continue
                ziel = _ROOT / "scripts" / f"{treffer.group(1)}.py"
                if not ziel.is_file():
                    continue  # z. B. train_gender_head.py (daten-blockiert)
                bekannt = _trainer_flags(ziel)
                if not bekannt:
                    continue  # Skript ohne argparse (Helfer)
                genutzt = set(_FLAG_RE.findall(treffer.group("rest")))
                unbekannt = genutzt - bekannt
                assert not unbekannt, (
                    f"{kette.name}: {ziel.name} kennt {sorted(unbekannt)} nicht (argparse würde mit Exit 2 abbrechen)"
                )
                geprueft += 1
    assert geprueft >= 10, f"zu wenige Aufrufe geprüft: {geprueft}"


_RESUME_RE = re.compile(r"--resume\s+([^\s\"';]+)")
_SKIPPT_RE = re.compile(r"scripts/([a-z0-9_]+)\.py")


def test_jeder_ketten_resume_anker_wird_vom_trainer_geschrieben() -> None:
    """Kein ``--resume <pfad>`` darf auf einen Checkpoint zeigen, den der
    Trainer nie erzeugt.

    Befund 2026-11-08: ``train_flashsr_f4.py`` schrieb nur
    ``checkpoint_epoch<N>.pt`` und ``best.pt``; der Kette-2-Resume-Befehl
    zeigte auf ``models/flashsr/checkpoint_latest.pt``. Versuch 2 starb
    deshalb **61 s nach dem 6-h-Timeout** in einen harten
    ``FileNotFoundError`` (Manifest ``train_job_d3_flashsr_musik.json``) —
    sechs GPU-Stunden ohne jede Wiederaufnahme.
    """
    geprueft = 0
    for kette in _KETTEN:
        for zeile in kette.read_text(encoding="utf-8").splitlines():
            if zeile.lstrip().startswith("#"):
                continue
            treffer = _RESUME_RE.search(zeile)
            if not treffer:
                continue
            ziel_treffer = _SKIPPT_RE.search(zeile)
            assert ziel_treffer, f"{kette.name}: --resume ohne Trainer: {zeile.strip()}"
            ziel = _ROOT / "scripts" / f"{ziel_treffer.group(1)}.py"
            if not ziel.is_file():
                continue
            anker = Path(treffer.group(1)).name
            assert anker in ziel.read_text(encoding="utf-8"), (
                f"{kette.name}: {ziel.name} erzeugt den Resume-Anker '{anker}' nie "
                "(Resume würde mit FileNotFoundError sterben)"
            )
            geprueft += 1
    assert geprueft >= 6, f"zu wenige Resume-Pfade geprüft: {geprueft}"


def test_flashsr_trainer_schreibt_anker_und_faellt_fail_closed() -> None:
    """Der Trainer hält die Repo-Konvention ``<out-dir>/checkpoint_latest.pt``
    und startet ein Resume ohne Datei gar nicht erst (Exit 2 statt Traceback)."""
    quelle = (_ROOT / "scripts" / "train_flashsr_f4.py").read_text(encoding="utf-8")
    assert 'save_guarded(out_dir / "checkpoint_latest.pt", ckpt)' in quelle
    assert "is_file()" in quelle, "Resume-Guard fehlt (FileNotFoundError statt Exit 2)"


def test_flashsr_resume_ist_datei_geschuetzt() -> None:
    """Beide Befehle (Erstlauf + Retry) prüfen den Anker vor dem Resume.

    ``train_job_runner`` nutzt bei Versuch 1 ``--cmd`` und erst danach
    ``--resume-cmd``; ohne Guard im ``--cmd`` startet jeder Kettenlauf von
    Epoche 0 und wirft die bereits trainierten Epochen weg.
    """
    kette2 = _SCRIPT.with_name("gpu_kette2_matrix_d3.sh")
    block = _block("D3-3:", kette2)
    assert block.count("if [ -f models/flashsr/checkpoint_latest.pt ]") == 2, (
        "Erstlauf und Retry müssen den Anker prüfen"
    )
    assert block.count("--resume models/flashsr/checkpoint_latest.pt") == 2
    for zeile in block.splitlines():
        if "train_flashsr_f4.py" in zeile:
            assert "--out-dir models/flashsr" in zeile, f"Anker läge neben dem Out-Dir: {zeile.strip()}"


# ── D-K3-84: Cantus-Phasenwechsel trainierte NULL Epochen ────────────────────
# Befund 2026-11-08 aus `train_job_c2_cantus_finetune.json`: der Lauf stand als
# „erfolgreich" im Manifest, war aber nach 11,9 s fertig — `start_epoch` kam aus
# dem Pretrain (50) und `--epochs=30`, also lief `range(50, 30)` leer (kein
# einziger Gradienten-Schritt). Zusätzlich konnte `checkpoint_best.pt` in der
# neuen Phase nie geschrieben werden, weil `best_val` (4,5922) an einer fremden
# Verlust-Definition hing. Dieselbe Klasse wie D-K3-80 (Manifest-Lüge).


def _cantus_trainer():
    """Lädt ``scripts/train_cantus.py`` als Modul (braucht torch → skip ohne torch)."""
    pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("train_cantus_dk384", _ROOT / "scripts" / "train_cantus.py")
    assert spec and spec.loader
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    return modul


def test_phasenwechsel_setzt_epoche_und_bestval_zurueck() -> None:
    tc = _cantus_trainer()
    start, step, best, wechsel = tc._phasen_lokale_resume_groessen("pretrain", "fine_tune", 50, 10000, 4.5922)

    assert (start, step, best) == (0, 0, float("inf")), "Pretrain-Epoche darf fine_tune nicht blockieren"
    assert wechsel is True


def test_gleiche_phase_resumt_weiter() -> None:
    tc = _cantus_trainer()
    start, step, best, wechsel = tc._phasen_lokale_resume_groessen("fine_tune", "fine_tune", 7, 900, 1.23)

    assert (start, step, best) == (7, 900, 1.23)
    assert wechsel is False


def test_ohne_phasenangabe_gilt_als_wechsel() -> None:
    """Alte Checkpoints ohne ``phase``-Feld dürfen keinen Fremd-Zähler erben."""
    tc = _cantus_trainer()
    start, step, best, wechsel = tc._phasen_lokale_resume_groessen(None, "fine_tune", 12, 340, 0.9)

    assert (start, step, best) == (0, 0, float("inf"))
    assert wechsel is True


def test_no_op_lauf_meldet_keinen_erfolg() -> None:
    """0 trainierte Epochen ⇒ ``stopped_reason`` ⇒ Exit 3, nie Exit 0."""
    quelle = (_ROOT / "scripts" / "train_cantus.py").read_text(encoding="utf-8")

    assert 'report["stopped_reason"] = "no_epochs"' in quelle
    assert 'report["epochen_trainiert"] = _epochen_trainiert' in quelle
    # Der bestehende stopped_reason-Pfad schreibt den Report und liefert 3.
    assert 'logger.error("Vorzeitig beendet (%s) — Report: %s", report["stopped_reason"], report_path)' in quelle
    assert "return 3" in quelle
    # Der Zähler muss IM Epochen-Loop erhöht werden (nicht aus start_epoch geraten).
    assert "_epochen_trainiert += 1" in quelle
