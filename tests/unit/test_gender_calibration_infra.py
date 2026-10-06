"""Tests der F13/F14-Infrastruktur (Gender-Head-Training + Evidenz-Kalibrierung).

Abgedeckt:
  - Determinismus (§G5 GEBOTE.md): stratifizierter Split ist reproduzierbar.
  - Label-Pflicht (§V7 copilot-instructions.md): ohne Labels kein Training —
    geprüft über die strenge CSV-Validierung.
  - Konsens-/Gate-Logik der Kalibrierung: eine Schwellenempfehlung wird NUR bei
    ausreichender Stichprobe UND ausreichendem Konsens erlaubt (kein
    „kalibriert" ohne Referenz).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import calibrate_gender_evidence as cge
import train_gender_head as tgh


class TestStratifiedSplit:
    def test_deterministic(self) -> None:
        """Gleiche Eingabe ⇒ identische Indizes (§G5 GEBOTE.md)."""
        _y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)
        _a = tgh._stratified_split(_y)
        _b = tgh._stratified_split(_y)
        assert np.array_equal(_a[0], _b[0])
        assert np.array_equal(_a[1], _b[1])

    def test_covers_all_indices_without_overlap(self) -> None:
        _y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
        _tr, _va = tgh._stratified_split(_y)
        assert set(_tr.tolist()).isdisjoint(set(_va.tolist()))
        assert sorted(np.concatenate([_tr, _va]).tolist()) == list(range(_y.size))

    def test_single_example_class_stays_in_train(self) -> None:
        """Eine Klasse mit einem einzigen Beispiel darf nicht komplett in Val landen."""
        _y = np.array([0, 0, 0, 0, 1], dtype=np.int64)
        _tr, _va = tgh._stratified_split(_y)
        assert 4 in _tr.tolist()


class TestMertPooling:
    def test_shape_and_finiteness(self) -> None:
        _feat = np.random.RandomState(0).randn(37, 1024).astype(np.float32)
        _vec = tgh._pool_mert(_feat)
        assert _vec.shape == (tgh._FEATURE_DIM,)
        assert np.isfinite(_vec).all()

    def test_empty_input_is_zeros(self) -> None:
        assert np.allclose(tgh._pool_mert(np.zeros((0, 1024), dtype=np.float32)), 0.0)

    def test_nonfinite_input_is_sanitized(self) -> None:
        _feat = np.full((5, 1024), np.nan, dtype=np.float32)
        _vec = tgh._pool_mert(_feat)
        assert np.isfinite(_vec).all()


class TestLabelContract:
    def test_missing_labels_raise(self, tmp_path: Path) -> None:
        """§V7 copilot-instructions.md: ohne Labels wird nicht trainiert."""
        with pytest.raises(FileNotFoundError):
            tgh._read_labels(tmp_path / "fehlt.csv")

    def test_invalid_label_raises(self, tmp_path: Path) -> None:
        _csv = tmp_path / "labels.csv"
        _csv.write_text("file,gender\na.wav,male\nb.wav,bariton\n", encoding="utf-8")
        with pytest.raises(ValueError):
            tgh._read_labels(_csv)

    def test_valid_labels_parse(self, tmp_path: Path) -> None:
        _csv = tmp_path / "labels.csv"
        _csv.write_text("file,gender\na.wav,MALE\nb.wav,female\n", encoding="utf-8")
        assert tgh._read_labels(_csv) == [("a.wav", "male"), ("b.wav", "female")]


class TestCalibrationReport:
    def _rec(self, fusion: str, anatomy: str, panns: bool) -> dict:
        return {
            "gender_fusion": fusion,
            "gender_anatomy": anatomy,
            "confidence": 0.8,
            "f0_hz": 200.0,
            "aperiodicity": 0.2,
            "panns_available": panns,
            "file": "x.wav",
        }

    def test_quantiles_ignores_nonfinite(self) -> None:
        _q = cge._quantiles(np.array([1.0, np.nan, np.inf, 3.0]))
        assert _q["n"] == 2
        assert _q["p50"] == pytest.approx(2.0)

    def test_no_panns_blocks_recommendation(self) -> None:
        """Ohne musiktaugliche ML-Evidenz gibt es keinen Konsens — und keine Empfehlung."""
        _records = [self._rec("male", "male", panns=False) for _ in range(20)]
        _report = cge._build_report(_records, min_samples=12)
        assert _report["consensus_rate"] is None
        assert _report["threshold_recommendation_allowed"] is False

    def test_high_consensus_allows_recommendation(self) -> None:
        _records = [self._rec("female", "female", panns=True) for _ in range(15)]
        _report = cge._build_report(_records, min_samples=12)
        assert _report["consensus_rate"] == pytest.approx(1.0)
        assert _report["threshold_recommendation_allowed"] is True

    def test_low_consensus_blocks_recommendation(self) -> None:
        _records = [self._rec("male", "female", panns=True) for _ in range(15)]
        _report = cge._build_report(_records, min_samples=12)
        assert _report["consensus_rate"] == pytest.approx(0.0)
        assert _report["threshold_recommendation_allowed"] is False

    def test_small_sample_blocks_recommendation(self) -> None:
        _records = [self._rec("male", "male", panns=True) for _ in range(5)]
        _report = cge._build_report(_records, min_samples=12)
        assert _report["threshold_recommendation_allowed"] is False


class TestMacroF1:
    def test_perfect_prediction(self) -> None:
        _y = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
        assert tgh._macro_f1(_y, _y.copy(), 3) == pytest.approx(1.0)

    def test_no_overlap_is_zero(self) -> None:
        _y = np.array([0, 0, 1, 1], dtype=np.int64)
        _pred = np.array([1, 1, 0, 0], dtype=np.int64)
        assert tgh._macro_f1(_y, _pred, 3) == pytest.approx(0.0)
