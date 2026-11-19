#!/usr/bin/env python3
"""Vertragstests fuer scripts/bewerte_klassen_kopf.py (fail-closed-Pfade).

Trainingsstrategie Stufe 3 (Report
`docs/reports/current/2026-11-13_hoerzentrierte_trainingsstrategie.md`):
Der §G13-Q-Erstbeleg eines Per-Klassen-Kopfes ist der EINZIGE Erfolgsmaßstab —
nicht der Trainings-Verlust. Gepinnt werden die fail-closed-Pfade (§V6 (copilot-instructions.md),
(copilot-instructions.md)) und die Q-Semantik. Zitierdisziplin AGENTS.md §5.
"""

from __future__ import annotations

import pytest

import scripts.bewerte_klassen_kopf as bk


def test_fail_closed_manifest_fehlt(tmp_path) -> None:
    assert bk.main(["--klasse", "x", "--manifest", str(tmp_path / "fehlt.jsonl"), "--checkpoint", "c.pt"]) == 2


def test_fail_closed_checkpoint_fehlt(tmp_path) -> None:
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        '{"klasse": "groove_echo", "clean": "c.wav", "schaden": "s.wav", "sha256_pcm_schaden": "a"}\n',
        encoding="utf-8",
    )
    assert (
        bk.main(["--klasse", "groove_echo", "--manifest", str(manifest), "--checkpoint", str(tmp_path / "fehlt.pt")])
        == 2
    )


def test_q_semantik_zielkonstante() -> None:
    from backend.core.dsp.defekt_delta import REPARATURQUOTE_ZIEL

    assert REPARATURQUOTE_ZIEL == 0.5
    assert bk.REPARATURQUOTE_ZIEL == 0.5
    assert bk.JND_TOLERANZ == 0.05


def test_q_verwirft_clean_kontrast_unterhalb_jnd() -> None:
    assert bk._q_berechnen(0.5, 0.4, 0.46) is None
    assert bk._q_berechnen(0.5, 0.4, 0.45) is None


def test_q_nutzt_nur_gerichteten_clean_kontrast_oberhalb_jnd() -> None:
    assert bk._q_berechnen(0.5, 0.4, 0.44) == pytest.approx(0.1 / 0.06)
    assert bk._q_berechnen(0.5, 0.4, 0.6) is None


def test_q_erfolg_allein_ueberstimmt_nie_schlechter_nicht() -> None:
    assert (
        bk._evidence_status(
            q_met=True,
            never_worsen=False,
            control_preserved=False,
            evaluation_complete=True,
            holdout_title_count=1,
        )
        == "nicht_bestanden"
    )


def test_holdout_entfernt_identische_trainings_pcm_hashes() -> None:
    train_title = bk._titel_hash("train-title")
    holdout_title = bk._titel_hash("holdout-title")
    provenance = {
        "schema": "aurik.klassen_kopf_split/1",
        "train_titel_sha256": [train_title],
        "holdout_titel_sha256": holdout_title,
    }
    entries = [
        {
            "titel": "train-title",
            "sha256_pcm_schaden": "train-damage",
            "sha256_pcm_clean": "shared-clean",
        },
        {
            "titel": "holdout-title",
            "sha256_pcm_schaden": "holdout-damage",
            "sha256_pcm_clean": "shared-clean",
        },
        {
            "titel": "holdout-title",
            "sha256_pcm_schaden": "independent-damage",
            "sha256_pcm_clean": "independent-clean",
        },
    ]

    holdout, excluded = bk._holdout_zeilen(entries, provenance)

    assert excluded == 1
    assert [entry["sha256_pcm_schaden"] for entry in holdout] == ["independent-damage"]


def test_alter_checkpoint_ohne_holdout_provenienz_wird_abgelehnt(monkeypatch, tmp_path) -> None:
    import json

    import torch

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "klasse": "groove_echo",
                "titel": "song",
                "clean": "clean.wav",
                "schaden": "damage.wav",
                "sha256_pcm_schaden": "damage",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"modell": {}}, checkpoint)
    monkeypatch.setattr(bk, "ROOT", tmp_path)

    assert (
        bk.main(
            [
                "--manifest",
                str(manifest),
                "--checkpoint",
                str(checkpoint),
                "--out",
                str(tmp_path / "report.json"),
            ]
        )
        == 2
    )
