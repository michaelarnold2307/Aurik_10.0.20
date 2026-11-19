"""Tests fuer den T6-1 A/B-Harness (scripts/validate_t61_ab.py).

Sichert die PRAE-REGISTRIERTE Entscheidungsregel gegen nachtraegliches
Drehen (TODO-T6-1 A/B-Abnahme, `docs/TODOS_SOTA_ROADMAP.md`) sowie
§G5-Determinismus des Bootstrap-CIs (Seed-fixiert) und die harte
Hoerordnung-Ebene-1-Schranke.

Beweisschluss: `pytest tests/unit/test_validate_t61_ab.py`.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.validate_t61_ab import (
    SEED,
    bootstrap_ci95,
    decide_ab,
    load_matrix,
)


def test_bootstrap_determinism_and_direction() -> None:
    deltas = [0.5, 0.7, 0.4, 0.6, 0.8]
    ci1 = bootstrap_ci95(deltas)
    ci2 = bootstrap_ci95(deltas, seed=SEED)
    assert ci1 == ci2  # §G5 (GEBOTE.md): Seed-fixiert, reproduzierbar
    lo, hi = ci1
    assert 0.0 < lo <= hi  # klare positive Deltas schliessen 0 aus


def test_decide_variant_wins_when_all_criteria_met() -> None:
    d = decide_ab([5.0] * 24, [5.6] * 24)  # +0.6 >= min_delta 0.25, CI > 0
    assert d.variant_wins is True
    assert d.n_configs == 24 and d.ci95_low > 0.0
    assert "Variante gewinnt" in d.reason


def test_decide_rejects_small_gain() -> None:
    d = decide_ab([5.0] * 24, [5.1] * 24)  # +0.1 < min_delta
    assert d.variant_wins is False
    assert "min_delta" in d.reason


def test_decide_rejects_ci_including_zero() -> None:
    ref = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
    var = [6.0, 4.0, 6.0, 4.0, 6.0, 4.0, 6.4, 5.2]  # Mittel ~ +0.3, aber breit
    d = decide_ab(ref, var, min_delta=0.2)
    assert d.variant_wins is False
    assert d.ci95_low <= 0.0 and "CI" in d.reason


def test_decide_ebene1_violation_is_hard_veto() -> None:
    d = decide_ab([5.0] * 24, [9.0] * 24, ebene1_violation=True)  # sonst klarer Sieg
    assert d.variant_wins is False
    assert "Ebene-1" in d.reason


def test_decide_input_validation() -> None:
    with pytest.raises(ValueError):
        decide_ab([], [])
    with pytest.raises(ValueError):
        decide_ab([1.0, 2.0], [1.0])


def test_load_matrix_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "matrix.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "score_ref", "score_var"])
        w.writerow(["c1", "5.0", "5.5"])
        w.writerow(["c2", "4.0", "4.25"])
    ref, var = load_matrix(p)
    assert ref == [5.0, 4.0] and var == [5.5, 4.25]
    d = decide_ab(ref, var, min_delta=0.3)
    assert d.n_configs == 2  # Matrix wird vollstaendig gelesen
