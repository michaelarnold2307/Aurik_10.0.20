"""Tests für scripts/mushra_harness.py — Auswertung (ITU-R BS.1534)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scripts.mushra_harness import (
    TASK_PREFERENCE,
    StimulusScore,
    analyze_answers,
    cmd_thresholds_build,
    cmd_thresholds_fit,
)


def _write_wav(path: Path, seconds: float = 1.0, freq: float = 220.0, amp: float = 0.3) -> None:
    """Kurze, deterministische Testdatei (48 kHz, mono, PCM_16)."""
    import numpy as np
    import soundfile as sf

    n = int(48000 * seconds)
    t = np.arange(n, dtype=np.float64) / 48000.0
    sig = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), sig, 48000, format="WAV", subtype="PCM_16")


def _manifest(tmp_path: Path, trials: list[dict], task: str = "defect", seed: int = 123) -> tuple[Path, Path]:
    ref = tmp_path / "ref.wav"
    cand = tmp_path / "cand.wav"
    if not ref.exists():
        _write_wav(ref, freq=220.0)
    if not cand.exists():
        _write_wav(cand, freq=222.0, amp=0.31)
    full: list[dict] = []
    for t in trials:
        e = dict(t)
        e.setdefault("reference", str(ref))
        if not e.get("catch"):
            e.setdefault("candidate", str(cand))
        full.append(e)
    man = tmp_path / "manifest.json"
    man.write_text(json.dumps({"build": "t", "task": task, "seed": seed, "trials": full}), encoding="utf-8")
    return man, tmp_path / "out"


def _build_args(man: Path, out: Path, **kw) -> argparse.Namespace:
    base = {"build": "t", "manifest": str(man), "out": str(out), "seed": None, "level_match": False}
    base.update(kw)
    return argparse.Namespace(**base)


def _fit_args(study: Path, **kw) -> argparse.Namespace:
    base = {"study": str(study), "min_hit_rate": 0.75, "min_trials": 12}
    base.update(kw)
    return argparse.Namespace(**base)


def _answer(study: Path, rows: list[tuple[str, str, str]]) -> None:
    p = study / "answers.csv"
    with open(p, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["listener", "trial_id", "answer_interval"])
        for r in rows:
            w.writerow(list(r))


class TestThresholdsBuild:
    def test_erzeugt_spieler_vertrag(self, tmp_path: Path) -> None:
        """Die gebaute Studie MUSS vom Hör-Player ladbar sein (Vertragstreue)."""
        from scripts.hoerpanel_player import load_study

        man, out = _manifest(tmp_path, [{"trial_id": "a", "class": "c"}])
        assert cmd_thresholds_build(_build_args(man, out)) == 0
        assert (out / "meta.json").is_file() and (out / "trial_key.json").is_file()
        assert (out / "answers_template.csv").is_file()
        assert (out / "trials" / "a__A.wav").is_file() and (out / "trials" / "a__B.wav").is_file()
        study = load_study(out)  # fail-closed, wenn der Vertrag verletzt wäre
        assert "a" in study.key

    def test_ab_zuordnung_deterministisch(self, tmp_path: Path) -> None:
        """§G5 (copilot-instructions.md): gleicher Seed ⇒ identische A/B-Zuordnung (unabhängig vom Prozess)."""
        man, out1 = _manifest(tmp_path, [{"trial_id": f"t{i}", "class": "c"} for i in range(6)])
        out2 = tmp_path / "out2"
        assert cmd_thresholds_build(_build_args(man, out1)) == 0
        assert cmd_thresholds_build(_build_args(man, out2)) == 0
        k1 = json.loads((out1 / "trial_key.json").read_text(encoding="utf-8"))
        k2 = json.loads((out2 / "trial_key.json").read_text(encoding="utf-8"))
        assert k1 == k2

    def test_praeferenz_ohne_defektseite(self, tmp_path: Path) -> None:
        """Präferenz-Paare haben KEINEN Defekt — das Feld bleibt ehrlich None."""
        man, out = _manifest(tmp_path, [{"trial_id": "a", "class": "c"}], task=TASK_PREFERENCE)
        assert cmd_thresholds_build(_build_args(man, out)) == 0
        key = json.loads((out / "trial_key.json").read_text(encoding="utf-8"))
        assert key[0]["defective_interval"] is None
        assert key[0]["expected_answer"] in ("A", "B")
        assert key[0]["task"] == TASK_PREFERENCE

    def test_fangfrage_erwartet_je_aufgabe_andere_seite(self, tmp_path: Path) -> None:
        """Fangfrage: der Anker ist der auffällige Reiz — bei Präferenz ist die
        richtige Antwort die GEGENseite, bei Defekt die Ankerseite."""
        man_d, out_d = _manifest(tmp_path, [{"trial_id": "c1", "class": "catch", "catch": True}])
        assert cmd_thresholds_build(_build_args(man_d, out_d)) == 0
        kd = json.loads((out_d / "trial_key.json").read_text(encoding="utf-8"))[0]
        assert kd["is_catch"] is True
        assert kd["expected_answer"] == kd["defective_interval"]

        man_p, out_p = _manifest(tmp_path, [{"trial_id": "c1", "class": "catch", "catch": True}], task=TASK_PREFERENCE)
        assert cmd_thresholds_build(_build_args(man_p, out_p)) == 0
        kp = json.loads((out_p / "trial_key.json").read_text(encoding="utf-8"))[0]
        assert kp["expected_answer"] != kp["defective_interval"]

    def test_fehlerhafte_trials_werden_uebersprungen(self, tmp_path: Path) -> None:
        """Fail-closed (§V6 copilot-instructions.md): fehlende Datei und Längen-Mismatch ⇒ kein Trial."""
        _write_wav(tmp_path / "short.wav", seconds=0.2)
        man, out = _manifest(
            tmp_path,
            [
                {"trial_id": "gut", "class": "c"},
                {"trial_id": "laenge", "class": "c", "candidate": str(tmp_path / "short.wav"), "seconds": 1.0},
                {"trial_id": "fehlt", "class": "c", "candidate": str(tmp_path / "gibtsnicht.wav")},
            ],
        )
        assert cmd_thresholds_build(_build_args(man, out)) == 0
        ids = {e["trial_id"] for e in json.loads((out / "trial_key.json").read_text(encoding="utf-8"))}
        assert ids == {"gut"}

    def test_ohne_gueltiges_trial_abbruch(self, tmp_path: Path) -> None:
        man, out = _manifest(tmp_path, [{"trial_id": "fehlt", "class": "c", "candidate": str(tmp_path / "nix.wav")}])
        assert cmd_thresholds_build(_build_args(man, out)) == 2

    def test_level_match_gleicht_pegel_an(self, tmp_path: Path) -> None:
        man, out = _manifest(tmp_path, [{"trial_id": "a", "class": "c"}])
        assert cmd_thresholds_build(_build_args(man, out, level_match=True)) == 0
        e = json.loads((out / "trial_key.json").read_text(encoding="utf-8"))[0]
        assert abs(e["rms_db_ref_after"] - e["rms_db_cand_after"]) < 0.1


class TestThresholdsFit:
    def test_verdikt_verlangt_mindestzahl(self, tmp_path: Path) -> None:
        """3/3 ist p = 0,125 — ohne Mindestzahl darf KEIN Beleg entstehen."""
        man, out = _manifest(tmp_path, [{"trial_id": "a", "class": "c"}], task=TASK_PREFERENCE)
        assert cmd_thresholds_build(_build_args(man, out)) == 0
        key = json.loads((out / "trial_key.json").read_text(encoding="utf-8"))
        _answer(out, [("L01", e["trial_id"], e["expected_answer"]) for e in key])
        assert cmd_thresholds_fit(_fit_args(out, min_trials=12)) == 0
        rep = json.loads((out / "fit.json").read_text(encoding="utf-8"))
        assert rep["verdict"] == "ZU_WENIGE_TRIALS"
        assert rep["by_class"]["c"]["above_chance"] is False

    def test_fangfrage_schliesst_hoerer_aus(self, tmp_path: Path) -> None:
        trials: list[dict] = [{"trial_id": f"t{i}", "class": "c"} for i in range(4)]
        trials.append({"trial_id": "c1", "class": "catch", "catch": True})
        man, out = _manifest(tmp_path, trials, task=TASK_PREFERENCE)
        assert cmd_thresholds_build(_build_args(man, out)) == 0
        key = json.loads((out / "trial_key.json").read_text(encoding="utf-8"))
        rows: list[tuple[str, str, str]] = []
        for e in key:
            if e["is_catch"]:
                wrong = "B" if e["expected_answer"] == "A" else "A"
                rows.append(("L01", e["trial_id"], wrong))  # Fangfrage verfehlt
            else:
                rows.append(("L01", e["trial_id"], e["expected_answer"]))
        _answer(out, rows)
        assert cmd_thresholds_fit(_fit_args(out, min_trials=4)) == 0
        rep = json.loads((out / "fit.json").read_text(encoding="utf-8"))
        assert rep["listeners"]["L01"]["valid"] is False
        assert rep["pooled"]["listeners_excluded_by_catch"] == ["L01"]

    def test_praeferenz_verdikt_braucht_ueberzufall(self, tmp_path: Path) -> None:
        trials: list[dict] = [{"trial_id": f"t{i:02d}", "class": "c"} for i in range(16)]
        man, out = _manifest(tmp_path, trials, task=TASK_PREFERENCE)
        assert cmd_thresholds_build(_build_args(man, out)) == 0
        key = json.loads((out / "trial_key.json").read_text(encoding="utf-8"))

        _answer(out, [("L01", e["trial_id"], e["expected_answer"]) for e in key])
        assert cmd_thresholds_fit(_fit_args(out, min_trials=12)) == 0
        assert json.loads((out / "fit.json").read_text(encoding="utf-8"))["verdict"] == "PRAEFERENZ_BELEGT"

        # Gegenprobe: immer die Gegenseite ⇒ klare Anti-Präferenz, kein Beleg
        _answer(out, [("L01", e["trial_id"], "B" if e["expected_answer"] == "A" else "A") for e in key])
        assert cmd_thresholds_fit(_fit_args(out, min_trials=12)) == 0
        assert json.loads((out / "fit.json").read_text(encoding="utf-8"))["verdict"] == "KEINE_PRAEFERENZ"

    def test_fehlende_antworten_klarer_abbruch(self, tmp_path: Path) -> None:
        man, out = _manifest(tmp_path, [{"trial_id": "a", "class": "c"}])
        assert cmd_thresholds_build(_build_args(man, out)) == 0
        assert cmd_thresholds_fit(_fit_args(out)) == 2  # answers.csv fehlt


def _write(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "answers.csv"
    with open(p, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["listener", "kind", "rating", "trial_label"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return p


def _args(**kw) -> argparse.Namespace:
    base = {"gap": 8.0, "margin": 15.0}
    base.update(kw)
    return argparse.Namespace(**base)


class TestAnalyzeAnswers:
    def test_go_case(self, tmp_path: Path) -> None:
        rows = [
            {"listener": "L1", "kind": "hidden_ref", "rating": "95", "trial_label": "p1"},
            {"listener": "L1", "kind": "aurik", "rating": "88", "trial_label": "p1"},
            {"listener": "L1", "kind": "anchor", "rating": "40", "trial_label": "p1"},
        ]
        p = _write(tmp_path, rows)
        rc = analyze_answers(p, tmp_path, _args())
        assert rc == 0
        data = json.loads((tmp_path / "analysis_answers.json").read_text(encoding="utf-8"))
        assert data["go"] is True
        assert data["scores"]["p1"]["aurik"]["mean"] == 88.0
        assert data["listeners_valid"] == 1

    def test_no_go_and_listener_exclusion(self, tmp_path: Path) -> None:
        rows = [
            {"listener": "L1", "kind": "hidden_ref", "rating": "40", "trial_label": "p1"},
            {"listener": "L1", "kind": "aurik", "rating": "90", "trial_label": "p1"},
            {"listener": "L2", "kind": "hidden_ref", "rating": "92", "trial_label": "p1"},
            {"listener": "L2", "kind": "aurik", "rating": "75", "trial_label": "p1"},
            {"listener": "L2", "kind": "anchor", "rating": "35", "trial_label": "p1"},
        ]
        p = _write(tmp_path, rows)
        rc = analyze_answers(p, tmp_path, _args())
        assert rc == 1  # Aurik 75 < hidden_ref 92 - gap 8
        data = json.loads((tmp_path / "analysis_answers.json").read_text(encoding="utf-8"))
        assert "L1" in data["excluded"]
        assert data["scores"]["p1"]["aurik"]["mean"] == 75.0
        assert data["go"] is False

    def test_empty_input(self, tmp_path: Path) -> None:
        p = _write(tmp_path, [])
        assert analyze_answers(p, tmp_path, _args()) == 2


class TestStimulusScore:
    def test_fields(self) -> None:
        s = StimulusScore(kind="aurik", mean=25.0, ci95=3.1, n=4)
        assert s.kind == "aurik" and s.mean == 25.0 and s.n == 4
