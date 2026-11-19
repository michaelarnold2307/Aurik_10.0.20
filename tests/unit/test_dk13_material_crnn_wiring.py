"""tests/unit/test_dk13_material_crnn_wiring.py — D-K1-3 Material-CRNN-Verdrahtung.

Vertrag: Die gelernte Material-Repraesentation (Material-CRNN,
scripts/train_material_crnn.py) entscheidet im Material-Entscheidungspfad
(MediumDetector.detect, Roh-Aufruf, Register-Artefakt D-K1-3) — flag-bewusst
hinter use_material_crnn (Default AUS = bisheriges Verhalten, kein Flag-Flip
ohne §4-Verfahren scripts/flag_freigabe.py).

Kontrakte:
  * Flag AUS: CRNN wird NICHT konsultiert (bit-identisches Bestandsverhalten).
  * Flag AN: CRNN-Vorhersage steuert primary_material/confidence inkl.
    ClassificationResult-Passthrough und Spec-05-§6.1-Key-Normierung.
  * §V6 (copilot-instructions.md): ohne Modell/Bei Fehlern Fallback auf den
    Gauss/Mixture-Pfad mit logger.warning + Begründung.
  * Stereo-Layout-Invariante (AGENTS.md §3): (C, N) und (N, C) werden beide
    bedient — Kanal 0 wie im Training, kein interleaved-ravel.
  * Determinismus §G5 (copilot-instructions.md): gleicher Input ⇒ gleiches
    Ergebnis.
"""

from __future__ import annotations

import logging
import math
from unittest.mock import Mock

import numpy as np
import pytest

from backend.core import music_model_flags
from forensics import medium_detector as md
from forensics.medium_detector import MediumDetector

SR = 48_000
_det = MediumDetector()


def _make_audio(duration_s: float = 3.0) -> np.ndarray:
    """Synthetisches Rauschen (Seed 42, §G5 (copilot-instructions.md)) als deterministischer Detektor-Input."""
    rng = np.random.default_rng(42)
    return (rng.standard_normal(int(duration_s * SR)) * 0.05).astype(np.float32)


@pytest.fixture(autouse=True)
def _reset_crnn_cache(monkeypatch):
    """Ladecache je Test frisch setzen — kein Zustand zwischen Tests (§G8-Analog)."""
    monkeypatch.setattr(md, "_CRNN_LOADED", False)
    monkeypatch.setattr(md, "_CRNN_MODEL", None)


class _StubModel:
    """Mock-Inferenz: deterministische Logits statt trainierter Gewichte."""

    def __init__(self, logits) -> None:
        self._logits = logits
        self.inputs: list = []

    def __call__(self, x):
        self.inputs.append(x)
        return self._logits


# ---------------------------------------------------------------------------
# Flag-Vertrag (Default AUS = bisheriges Verhalten)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMaterialCrnnFlagVertrag:
    """use_material_crnn bleibt Default AUS — Flag-Flip nur nach §4-Verfahren."""

    def test_flag_default_aus_kein_flag_flip(self) -> None:
        assert music_model_flags.use_material_crnn is False, (
            "use_material_crnn muss Default False bleiben — Flag-Flip nur nach §4-Verfahren (scripts/flag_freigabe.py)"
        )

    def test_flag_aus_crnn_wird_nicht_konsultiert(self, monkeypatch) -> None:
        """Flag AUS: der Gauss/Mixture-Pfad entscheidet, der CRNN-Pfad wird nie berührt."""
        mock_predict = Mock(return_value=("shellac", 0.99))
        monkeypatch.setattr(md, "_crnn_predict", mock_predict)

        result = _det.detect(_make_audio(), SR)

        mock_predict.assert_not_called()
        assert result.classification_result is not None
        assert result.classification_result.classifier_source == "bayesian_fusion"


# ---------------------------------------------------------------------------
# Material-Entscheidungspfad (Register-Artefakt: MediumDetector.detect)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMaterialCrnnDecisionPath:
    """Flag AN: die gelernte Repräsentation steuert die Material-Entscheidung."""

    def test_flag_an_crnn_material_entscheidet(self, monkeypatch) -> None:
        monkeypatch.setattr(music_model_flags, "use_material_crnn", True)
        monkeypatch.setattr(md, "_crnn_predict", Mock(return_value=("shellac", 0.87)))

        result = _det.detect(_make_audio(), SR)

        assert result.primary_material == "shellac"
        assert result.confidence == pytest.approx(0.87)
        assert any(e.startswith("Material-CRNN:") for e in result.evidence), (
            "CRNN-Entscheidung muss im evidence-Trail sichtbar sein"
        )
        assert result.classification_result is not None
        assert result.classification_result.material == "shellac"
        assert result.classification_result.classifier_source == "material_crnn"

    def test_crnn_klasse_digital_wird_zu_mp3_high_normiert(self, monkeypatch) -> None:
        """Spec 05 §6.1 [RELEASE_MUST]: alle Keys vor Rückgabe auf SUPPORTED_MATERIALS."""
        monkeypatch.setattr(music_model_flags, "use_material_crnn", True)
        monkeypatch.setattr(md, "_crnn_predict", Mock(return_value=("digital", 0.9)))

        result = _det.detect(_make_audio(), SR)

        assert result.primary_material == "mp3_high"


# ---------------------------------------------------------------------------
# §V6-Fallback (copilot-instructions.md): logger.warning + Begründung
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMaterialCrnnFallback:
    """Ohne Modell/Bei Fehlern bleibt der bisherige Gauss/Mixture-Pfad aktiv."""

    def test_flag_an_ohne_modell_fallback_mit_warnung(self, monkeypatch, tmp_path, caplog) -> None:
        monkeypatch.chdir(tmp_path)  # models/material_crnn.pt existiert hier nicht
        baseline = _det.detect(_make_audio(), SR)
        monkeypatch.setattr(music_model_flags, "use_material_crnn", True)

        with caplog.at_level(logging.WARNING, logger="forensics.medium_detector"):
            result = _det.detect(_make_audio(), SR)

        assert "models/material_crnn.pt fehlt" in caplog.text, (
            "§V6 (copilot-instructions.md): Ersatzpfad braucht logger.warning"
        )
        assert "§V6" in caplog.text, "§V6 (copilot-instructions.md) muss im Log-Text stehen"
        assert result.primary_material == baseline.primary_material
        assert result.confidence == pytest.approx(baseline.confidence)

    def test_loader_ohne_modell_liefert_none_mit_warnung(self, monkeypatch, tmp_path, caplog) -> None:
        monkeypatch.chdir(tmp_path)
        with caplog.at_level(logging.WARNING, logger="forensics.medium_detector"):
            model = md._load_material_crnn()
        assert model is None
        assert "models/material_crnn.pt fehlt" in caplog.text

    def test_flag_an_predict_fehler_faellt_mit_warnung_zurueck(self, monkeypatch, caplog) -> None:
        baseline = _det.detect(_make_audio(), SR)
        monkeypatch.setattr(music_model_flags, "use_material_crnn", True)
        monkeypatch.setattr(md, "_crnn_predict", Mock(side_effect=RuntimeError("Defekt")))

        with caplog.at_level(logging.WARNING, logger="forensics.medium_detector"):
            result = _det.detect(_make_audio(), SR)

        assert "Material-CRNN-Entscheidung nicht aktiv" in caplog.text
        assert result.primary_material == baseline.primary_material
        assert result.confidence == pytest.approx(baseline.confidence)


# ---------------------------------------------------------------------------
# Stereo-Layout-Invariante (AGENTS.md §3) + Determinismus (§G5 (copilot-instructions.md))
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMaterialCrnnLayoutUndDeterminismus:
    """Kanal 0 in BEIDEN Stereo-Layouts — Training/Inferenz fenstergleich."""

    def test_crnn_mono_kanal0_in_beiden_stereo_layouts(self) -> None:
        n = 1000
        ch0 = np.full(n, 0.5, dtype=np.float32)
        ch1 = np.zeros(n, dtype=np.float32)

        mono_1d = md._crnn_mono(ch0)
        mono_cn = md._crnn_mono(np.stack([ch0, ch1]))  # (C, N)
        mono_nc = md._crnn_mono(np.stack([ch0, ch1], axis=1))  # (N, C)

        assert np.array_equal(mono_cn, mono_1d)
        assert np.array_equal(mono_nc, mono_1d)
        # Kein interleaved-ravel (Alter Bug: mischt 0.5/0.0 abwechselnd).
        assert np.allclose(mono_cn, 0.5)
        assert np.allclose(mono_nc, 0.5)

    def test_crnn_predict_layout_gleich_softmax_deterministisch(self, monkeypatch) -> None:
        torch = pytest.importorskip("torch")
        train_mod = pytest.importorskip("scripts.train_material_crnn")

        logits = torch.zeros(1, 10)
        logits[0, 1] = 5.0  # _CRNN_MATERIALS[1] == "vinyl"
        stub = _StubModel(logits)
        seen: list[np.ndarray] = []

        def fake_logmel(mono, sr):
            seen.append(np.asarray(mono).copy())
            return np.zeros((64, 96), dtype=np.float32)  # synthetisches Spektrogramm

        monkeypatch.setattr(train_mod, "_logmel", fake_logmel)
        monkeypatch.setattr(md, "_load_material_crnn", lambda: stub)

        n = SR
        ch0 = np.full(n, 0.5, dtype=np.float32)
        ch1 = np.zeros(n, dtype=np.float32)

        res_cn = md._crnn_predict(np.stack([ch0, ch1]), SR)
        res_nc = md._crnn_predict(np.stack([ch0, ch1], axis=1), SR)

        assert res_cn is not None and res_nc is not None
        material, conf = res_cn
        assert res_nc[0] == material
        assert res_nc[1] == pytest.approx(conf)
        assert material == "vinyl"
        expected_conf = math.exp(5.0) / (math.exp(5.0) + 9.0)
        assert conf == pytest.approx(expected_conf, abs=1e-4)

        assert len(seen) == 2
        for mono in seen:
            assert mono.shape == (n,)
            assert np.allclose(mono, 0.5), "Stereo-Layout: es muss Kanal 0 ankommen (kein Interleave)"
        # Mock-Inferenz-Schnittstelle: Log-Mel [64, 96] → Tensor [1, 1, 64, 96].
        assert tuple(stub.inputs[0].shape) == (1, 1, 64, 96)

    def test_crnn_predict_deterministisch(self, monkeypatch) -> None:
        """§G5 (copilot-instructions.md): gleicher Input ⇒ gleiches Ergebnis."""
        torch = pytest.importorskip("torch")
        pytest.importorskip("scripts.train_material_crnn")
        pytest.importorskip("librosa")  # echte _logmel-Kette im Determinismus-Pin

        logits = torch.zeros(1, 10)
        logits[0, 2] = 3.0  # _CRNN_MATERIALS[2] == "tape"
        stub = _StubModel(logits)
        monkeypatch.setattr(md, "_load_material_crnn", lambda: stub)

        mono = _make_audio(1.0)
        first = md._crnn_predict(mono, SR)
        second = md._crnn_predict(mono, SR)

        assert first is not None and second is not None
        assert first[0] == second[0] == "tape"
        assert first[1] == pytest.approx(second[1], abs=0.0)
