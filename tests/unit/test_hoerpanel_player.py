"""Tests für den Hör-Player (scripts/hoerpanel_player.py) — 2AFC, doppelblind.

Kernverträge: die Lösung verlässt den Server nie, die Reihenfolge ist je Hörer
deterministisch, Antworten landen CSV-kompatibel zu ``thresholds-fit`` in
``answers.csv`` (atomar, wiederaufnahmefähig), Pfadaufstieg ist gesperrt.
"""

from __future__ import annotations

import csv
import json
import threading
import wave
from pathlib import Path

from scripts.hoerpanel_player import (
    AnswerStore,
    listener_order,
    load_study,
    make_handler,
    serve,
    session_payload,
)


def _write_wav(path: Path, n: int = 200) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(48000)
        wf.writeframes(b"\x00\x00" * n)


def _make_study(tmp_path: Path) -> Path:
    study = tmp_path / "thresholds_test"
    trials = study / "trials"
    trials.mkdir(parents=True)
    key = [
        {
            "trial_id": "t_a_m3r1",
            "defect_class": "shellac_sprung",
            "margin_db": 3.0,
            "defective_interval": "A",
            "measured_margin_db": 3.0,
        },
        {
            "trial_id": "t_b_m0r1",
            "defect_class": "bandrauschen",
            "margin_db": 0.0,
            "defective_interval": "B",
            "measured_margin_db": 0.1,
        },
        {
            "trial_id": "t_catch_r1",
            "defect_class": "catch",
            "margin_db": 15.0,
            "defective_interval": "B",
            "measured_margin_db": 15.0,
        },
    ]
    (study / "trial_key.json").write_text(json.dumps(key), encoding="utf-8")
    (study / "meta.json").write_text(
        json.dumps({"build": "t", "seed": 7, "classes": [], "margins_db": [3.0]}), encoding="utf-8"
    )
    for tid in ("t_a_m3r1", "t_b_m0r1", "t_catch_r1"):
        for iv in ("A", "B"):
            _write_wav(trials / f"{tid}__{iv}.wav")
    return study


def test_session_doppelblind_und_deterministisch(tmp_path: Path) -> None:
    study = load_study(_make_study(tmp_path))
    store = AnswerStore.load(study.study_dir)
    o1 = listener_order(study, "L01")
    o2 = listener_order(study, "L01")
    assert o1 == o2
    assert sorted(o1) == sorted(study.key)
    payload = session_payload(study, "L01", store)
    assert payload["order"] == o1
    # Doppelblind: die Lösung darf dem Client nie begegnen
    assert "defective_interval" not in json.dumps(payload)


def test_answers_csv_kompatibel_und_resume(tmp_path: Path) -> None:
    study = load_study(_make_study(tmp_path))
    store = AnswerStore.load(study.study_dir)
    assert store.record("L01", "t_a_m3r1", "a") is True  # Normalisierung A/B
    assert store.record("L01", "t_a_m3r1", "B") is False  # Duplikat gesperrt
    assert store.record("L01", "unbekannt", "A") is True  # Trial-Prüfung liegt beim Server
    rows = list(csv.DictReader(open(store.csv_path, encoding="utf-8")))
    assert rows[0]["listener"] == "L01"
    assert rows[0]["answer_interval"] == "A"
    # Wiederaufnahme: frisch geladen zählt die beantworteten Trials
    store2 = AnswerStore.load(study.study_dir)
    assert "t_a_m3r1" in store2.answered_for("L01")


def test_http_ende_zu_ende_und_pfadsperre(tmp_path: Path) -> None:
    study = load_study(_make_study(tmp_path))
    server = serve(study.study_dir, 0)
    host, port = server.server_address[:2]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        import urllib.request

        base = f"http://{host}:{port}"
        page = urllib.request.urlopen(f"{base}/", timeout=5).read().decode("utf-8")
        assert "Hör-Panel" in page

        req = urllib.request.Request(
            f"{base}/api/answer",
            data=json.dumps({"listener": "L01", "trial_id": "t_catch_r1", "answer_interval": "B"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        assert json.loads(urllib.request.urlopen(req, timeout=5).read())["ok"] is True

        sess = json.loads(urllib.request.urlopen(f"{base}/api/session?listener=L01", timeout=5).read())
        assert "t_catch_r1" not in sess["order"]  # beantwortet = raus (Resume)

        wav = urllib.request.urlopen(f"{base}/trials/t_a_m3r1__A.wav", timeout=5)
        assert wav.status == 200 and len(wav.read()) > 44

        # Pfadaufstieg ist gesperrt
        try:
            urllib.request.urlopen(f"{base}/trials/..%2F..%2Ftrial_key.json", timeout=5)
            raise AssertionError("Pfadaufstieg nicht blockiert")
        except Exception as exc:
            assert "404" in str(exc) or "trial_key" not in str(exc)
    finally:
        server.shutdown()
        server.server_close()
