#!/usr/bin/env python3
"""Vertragstests fuer scripts/train_klassen_kopf.py (modell-frei wo möglich).

Trainingsstrategie Stufe 3 (Report
`docs/reports/current/2026-11-13_hoerzentrierte_trainingsstrategie.md`):
kleiner Per-Klassen-Reparatur-Kopf auf den Rezept-Regionen. Gepinnt werden
die fail-closed-Datenprüfung (§V6 (copilot-instructions.md)), die
Fenster-Normierung und die Modell-Geometrie. Zitierdisziplin AGENTS.md §5.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import scripts.train_klassen_kopf as kk


def test_fenster_pad_und_trunc() -> None:
    kurz = np.ones(100, dtype=np.float32)
    lang = np.ones(kk.WINDOW + 500, dtype=np.float32)
    assert kk._fenster(kurz).shape == (kk.WINDOW,)
    assert kk._fenster(lang).shape == (kk.WINDOW,)
    assert kk._fenster(kurz)[:100].sum() == pytest.approx(100.0)
    assert kk._fenster(kurz)[100:].sum() == pytest.approx(0.0)


def test_eintraege_laden_fail_closed(tmp_path) -> None:
    with pytest.raises(SystemExit) as e1:
        kk._eintraege_laden(tmp_path / "fehlt.jsonl", "groove_echo")
    assert e1.value.code == 2
    leer = tmp_path / "leer.jsonl"
    leer.write_text('{"klasse": "clicks", "clean": "x.wav"}\n', encoding="utf-8")
    with pytest.raises(SystemExit) as e2:
        kk._eintraege_laden(leer, "groove_echo")
    assert e2.value.code == 2
    gut = tmp_path / "gut.jsonl"
    gut.write_text(
        '{"klasse": "groove_echo", "clean": "c.wav", "schaden": "s.wav", "sha256_pcm_schaden": "a"}\n',
        encoding="utf-8",
    )
    eintraege = kk._eintraege_laden(gut, "groove_echo")
    assert len(eintraege) == 1
    assert eintraege[0]["klasse"] == "groove_echo"


def test_build_model_geometrie() -> None:
    import torch

    model = kk.build_model(torch.device("cpu"))
    x = torch.zeros(2, 1, kk.WINDOW)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 1, kk.WINDOW)
    assert torch.equal(y, x), "Untrainierter Reparaturkopf muss identity-initialisiert sein"
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 200_000, f"Per-Klassen-Kopf soll klein bleiben, hat {n_params} Parameter"


def test_report_schema_felder() -> None:
    report = {
        "schema": "aurik.klassen_kopf/1",
        "klasse": "groove_echo",
        "ergebnis": "erfolgreich",
        "val_mae": 0.01,
        "anker": "checkpoint_latest.pt",
    }
    assert json.loads(json.dumps(report, ensure_ascii=False)) == report


def test_titel_holdout_ist_deterministisch_und_disjunkt() -> None:
    entries = [
        {"titel": title, "sha256_pcm_schaden": f"{title}-{index}"}
        for title in ("song-a", "song-b", "song-c")
        for index in range(3)
    ]

    train_a, holdout_a, provenance_a = kk._split_nach_titel(entries, 42)
    train_b, holdout_b, provenance_b = kk._split_nach_titel(entries, 42)

    assert provenance_a == provenance_b
    assert holdout_a == holdout_b
    assert {entry["titel"] for entry in holdout_a}.isdisjoint({entry["titel"] for entry in train_a})
    assert len({entry["titel"] for entry in train_a}) == 2
    assert len({entry["titel"] for entry in holdout_a}) == 1


def test_titel_holdout_scheitert_bei_nur_einem_titel() -> None:
    with pytest.raises(ValueError, match="mindestens zwei"):
        kk._split_nach_titel([{"titel": "song-a"}], 42)


def test_resume_verlangt_exakt_passende_split_provenienz() -> None:
    provenance = {"schema": "aurik.klassen_kopf_split/1", "manifest_sha256": "abc"}
    checkpoint = {"split_provenance": provenance, "model_schema": kk._MODEL_SCHEMA}

    assert kk._resume_provenance_gueltig(checkpoint, provenance)
    assert not kk._resume_provenance_gueltig(
        {"split_provenance": provenance, "model_schema": "aurik.klassen_kopf_model/1"}, provenance
    )
    assert not kk._resume_provenance_gueltig({"split_provenance": {}}, provenance)
    assert not kk._resume_provenance_gueltig({"modell": {}}, provenance)
    assert not kk._resume_provenance_gueltig(None, provenance)
