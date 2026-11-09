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
import json
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


# ── D-K3-87: D3-Finetunes waren in 2×6 h strukturell unerreichbar ──────────
# Befund 2026-11-09 (Manifeste train_job_d3_bigvgan_f3.json /
# train_job_d3_flashsr_musik.json): BigVGAN 4418–4424 s/Epoche, FlashSR
# 4799–4843 s/Epoche ⇒ 30 Epochen = 36,8 h bzw. 40,3 h. Die Kette gab 2×6 h
# und markierte den gesunden Teilfortschritt als „fehlgeschlagen“ (Epoche 8
# bzw. 10 von 30). Die Tests pinnen die Taktung gegen die Messwerte, damit ein
# späterer Refactor die Lücke nicht wieder einbaut.


def test_d3_finetunes_haben_realistisches_zeitbudget() -> None:
    kette2 = _SCRIPT.with_name("gpu_kette2_matrix_d3.sh")
    gemessene_epoche_s = {"D3-2:": 4424.0, "D3-3:": 4843.0}

    for job, epoche_s in gemessene_epoche_s.items():
        block = _block(job, kette2)
        timeout = float(re.search(r"--timeout-s\s+(\d+)", block).group(1))
        retries = int(re.search(r"--retries\s+(\d+)", block).group(1))
        budget = float(re.search(r"--max-seconds\s+(\d+)", block).group(1))

        assert budget < timeout, f"{job}: Zeitbudget {budget} muss unter dem Timeout {timeout} liegen"
        kapazitaet = (retries + 1) * budget
        assert kapazitaet >= 30 * epoche_s, (
            f"{job}: {kapazitaet:.0f} s Kapazität < 30 Epochen à {epoche_s:.0f} s "
            "— der Job wäre wieder strukturell unerreichbar"
        )


def test_d3_jobs_tragen_das_zeitbudget_in_allen_befehlen() -> None:
    """Erstlauf UND Resume müssen an der Epochen-Grenze stoppen (kein SIGTERM)."""
    kette2 = _SCRIPT.with_name("gpu_kette2_matrix_d3.sh")
    for job in ("D3-2:", "D3-3:"):
        block = _block(job, kette2)
        assert block.count("--max-seconds") >= 2, f"{job}: Zeitbudget fehlt in cmd/resume-cmd"


def test_flashsr_laeuft_vor_bigvgan() -> None:
    """Der kurze FlashSR-Lauf darf nicht hinter dem 37-h-BigVGAN-Lauf warten."""
    kette2 = _SCRIPT.with_name("gpu_kette2_matrix_d3.sh").read_text(encoding="utf-8")
    assert kette2.find("=== D3-3:") < kette2.find("=== D3-2:")


def test_bigvgan_erstlauf_resumt_vom_anker() -> None:
    """D3-2: Auch der Erstlauf muss den Anker prüfen (D-K3-72-Klasse).

    Befund 2026-11-09: der `--cmd` rief `train_bigvgan_f3.py --epochs 30`
    **ohne** `--resume`; jeder Kettenstart begann bei Epoche 0 und überschrieb
    `checkpoint_latest.pt`. Der Job kam dadurch über ~8 Epochen nie hinaus,
    obwohl `checkpoint_epoch8.pt` bereits existierte. FlashSR hatte den Guard
    seit 2026-11-08, BigVGAN nicht.
    """
    kette2 = _SCRIPT.with_name("gpu_kette2_matrix_d3.sh")
    block = _block("D3-2:", kette2)

    assert block.count("if [ -f output/f3_bigvgan/checkpoint_latest.pt ]") == 2, (
        "Erstlauf und Retry müssen den BigVGAN-Anker prüfen"
    )
    assert block.count("--resume output/f3_bigvgan/checkpoint_latest.pt") == 2


def test_zeitbudget_stoppt_vor_der_naechsten_epoche() -> None:
    from backend.core.training_artifacts import zeitbudget_erreicht

    assert not zeitbudget_erreicht(verstrichen_s=0.0, letzte_epoche_s=0.0, max_seconds=100.0)
    assert not zeitbudget_erreicht(verstrichen_s=10.0, letzte_epoche_s=50.0, max_seconds=100.0)
    assert zeitbudget_erreicht(verstrichen_s=60.0, letzte_epoche_s=50.0, max_seconds=100.0)
    assert not zeitbudget_erreicht(verstrichen_s=1e9, letzte_epoche_s=50.0, max_seconds=0.0)


def test_historie_zusammenfuehren_ist_resume_fest() -> None:
    from backend.core.training_artifacts import historie_zusammenfuehren

    alt = [{"epoch": 0, "val_a1": 1.0}, {"epoch": 1, "val_a1": 0.9}]
    neu = [{"epoch": 1, "val_a1": 0.85}, {"epoch": 2, "val_a1": 0.8}]

    zusammen = historie_zusammenfuehren(alt, neu)

    assert [eintrag["epoch"] for eintrag in zusammen] == [0, 1, 2]
    assert zusammen[1]["val_a1"] == 0.85, "der jüngste Versuch muss gewinnen"


def test_report_historie_verwirft_fremden_seed(tmp_path: Path) -> None:
    from backend.core.training_artifacts import report_historie_laden

    (tmp_path / "report.json").write_text(json.dumps({"seed": 7, "history": [{"epoch": 0}]}), encoding="utf-8")
    assert report_historie_laden(tmp_path, seed=42) == []

    (tmp_path / "report.json").write_text(
        json.dumps({"seed": 42, "history": [{"epoch": 0}, "kaputt"]}), encoding="utf-8"
    )
    assert report_historie_laden(tmp_path, seed=42) == [{"epoch": 0}]


def test_beide_d3_trainer_stoppen_an_der_epochengrenze() -> None:
    for name in ("train_bigvgan_f3.py", "train_flashsr_f4.py"):
        quelle = (_ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "zeitbudget_erreicht(" in quelle, f"{name}: Zeitbudget fehlt"
        assert "return 75" in quelle, f"{name}: Exit 75 (Zeitbudget) fehlt"
        assert "start_epoch > args.epochs" in quelle, f"{name}: Fail-closed-Guard (Anker hinter Ziel) fehlt"
        assert "start_epoch == args.epochs" in quelle, f"{name}: Ziel-bereits-erreicht-Zweig fehlt"
        assert "ziel_epochen_bereits_erreicht" in quelle, f"{name}: Beleg-Grund fehlt"
        assert '"best_val": best_val' in quelle, f"{name}: best_val nicht resume-fest"
        assert '"rng_state": rng.bit_generator.state' in quelle, f"{name}: RNG-Fortsetzung fehlt"


def _m8_optionen(script: Path) -> tuple[str, str]:
    """Liefert die `--cmd`- und `--resume-cmd`-Zeile des M8-Jobs einer Kette."""
    zeilen = script.read_text(encoding="utf-8").splitlines()
    start = next(i for i, z in enumerate(zeilen) if "--name m8_aspade" in z)
    cmd = next(z for z in zeilen[start:] if "--cmd " in z)
    resume = next(z for z in zeilen[start:] if "--resume-cmd " in z)
    return cmd.strip(), resume.strip()


def test_m8_ziel_ist_30_epochen_und_beide_ketten_identisch() -> None:
    """M8 (A-SPADE) ist mit 30 Epochen ECHT FERTIG (D-K4-1, 3,8 h, 2026-11-08).

    `train_job_runner` vergleicht die Befehlszeile LITERAL gegen das Manifest.
    Eine abweichende 60-Epochen-Variante wurde deshalb als „Befehl weicht ab —
    Neulauf" gewertet und startete den fertig belegten Job erneut (Befund
    2026-11-09, rund 6 h GPU-Zeit). Der Erstlauf trägt daher exakt den
    dokumentierten Zielumfang; nur der Resume-Pfad darf den Datei-Guard tragen.
    """
    cmd1, resume1 = _m8_optionen(_ROOT / "scripts" / "gpu_kette1_welle_c.sh")
    cmd3, resume3 = _m8_optionen(_ROOT / "scripts" / "gpu_kette3_k4_rest.sh")

    assert cmd1 == cmd3, "Kette 1 und Kette 3 müssen denselben M8-Erstlaufbefehl fahren"
    assert "--epochs 30" in cmd1, "M8-Ziel sind 30 Epochen (D-K4-1)"
    assert "--epochs 60" not in cmd1 and "--epochs 60" not in resume1
    assert "--resume" not in cmd1, "der Erstlauf darf nicht resumen (Manifest-Vergleich)"
    assert "--resume models/aspade/checkpoint_latest.pt" in resume1
    assert "--resume models/aspade/checkpoint_latest.pt" in resume3


def test_cantus_finetune_haelt_das_vram_budget_ein() -> None:
    """D-K3-88: Der Full-Preset-fine_tune starb reproduzierbar am 24-GiB-OOM
    (22,77 GiB belegt, Versuch 1 nach 55 s, Versuch 2 nach 31 s). Fix: Mikro-Batch
    2 × Akkumulation 4 = effektive Batch 8 (Rezeptur unverändert, Peak /4).
    Beide Befehle (Erstlauf + Resume) müssen die Flags tragen — sonst startet der
    Runner den OOM-Lauf erneut.
    """
    _quelle = (_ROOT / "scripts" / "gpu_kette1_welle_c.sh").read_text(encoding="utf-8")
    _zeilen = _quelle.splitlines()
    _start = next(i for i, z in enumerate(_zeilen) if "--name c2_cantus_finetune" in z)
    _cmd = next(z for z in _zeilen[_start:] if "--cmd " in z)
    _resume = next(z for z in _zeilen[_start:] if "--resume-cmd " in z)
    for _zeile in (_cmd, _resume):
        assert "--batch-size 2" in _zeile, f"Cantus-fine_tune ohne Mikro-Batch 2: {_zeile}"
        assert "--grad-accum 4" in _zeile, f"Cantus-fine_tune ohne Akkumulation 4: {_zeile}"
        assert "--singmos-loss" in _zeile, (
            "SingMOS-Lehrer darf nicht still abgeschaltet werden (§V7 (copilot-instructions.md))"
        )
    _trainer = (_ROOT / "scripts" / "train_cantus.py").read_text(encoding="utf-8")
    assert '"--grad-accum"' in _trainer, "train_cantus.py: --grad-accum fehlt"
    assert "akkumulations_schritt_faellig" in _trainer, "train_cantus.py: Akkumulations-Logik fehlt"
