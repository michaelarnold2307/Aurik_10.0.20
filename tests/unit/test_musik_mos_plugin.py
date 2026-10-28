"""Tests für MusikMOS — den zertifizierten UTMOS-Ersatz (E2)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plugins.musik_mos_plugin import get_musik_mos


def _musik_basis(n: int = 48000 * 4, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    zeit = np.arange(n) / 48000
    return (
        0.25 * np.sin(2 * np.pi * 220 * zeit)
        + 0.12 * np.sin(2 * np.pi * 440 * zeit)
        + 0.06 * np.sin(2 * np.pi * 880 * zeit)
        + rng.standard_normal(n).astype(np.float32) * 0.008
    ).astype(np.float32)


def test_mos_bereich_und_determinismus() -> None:
    p = get_musik_mos()
    x = _musik_basis()
    a = p.estimate_mos(x, 48000)
    b = p.estimate_mos(x, 48000)
    assert 1.0 <= a.mos <= 5.0
    assert a.mos == b.mos, "Determinismus §G5 (copilot-instructions.md)"
    assert a.model_used == "musik_mos_v1"


def test_richtung_auf_defektleiter() -> None:
    p = get_musik_mos()
    x = _musik_basis()
    rng = np.random.default_rng(11)
    rauschen = rng.standard_normal(len(x)).astype(np.float32) * 0.3
    sauber = p.estimate_mos(x, 48000).mos
    verrauscht = p.estimate_mos((x + rauschen).astype(np.float32), 48000).mos
    geklippt = p.estimate_mos(np.tanh(3.0 * x).astype(np.float32), 48000).mos
    assert sauber > verrauscht, f"Rauschen muss MOS senken: {sauber} !> {verrauscht}"
    assert sauber > geklippt, f"Clipping muss MOS senken: {sauber} !> {geklippt}"


def test_details_vollstaendig() -> None:
    res = get_musik_mos().estimate_mos(_musik_basis(), 48000)
    for schluessel in ("z", "flatness", "q_rausch", "q_tonal", "q_clip", "q_hf", "q_dyn"):
        assert schluessel in res.details, schluessel
