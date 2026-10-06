"""tests/unit/test_scnet_wiring.py — §P1-2: SCNet-Tier hinter Never-worsen.

Prüft die Verdrahtung des SCNet-4-Stems-Kandidaten (§P1-2 / TODO-P1-2) am
Verhalten, nicht an der Existenz von Zeilen:

1. **Sperr-Pin**: Ohne Freigabe (``music_model_flags.use_scnet_music=False``)
   wird das Plugin NIE angefasst und der Bestand unverändert zurückgegeben
   (§V7 copilot-instructions.md — kein Blind-Aktivieren).
2. **Auflösung**: ``resolve_model_path("scnet")`` liefert ohne Freigabe ``None``
   (kein Legacy-Gegenstück) und mit Freigabe den Kandidaten-Checkpoint.
3. **Never-worsen**: SCNet ersetzt den Bestand nur, wenn der referenzfreie
   Vergleich ihn trägt — Rekonstruktions-Treue UND Vokal-Erhalt (§v10.26).
4. **Kanonische Formel**: ``reconstruction_fidelity`` ist exakt 1,0 bei
perfekter Rekonstruktion (§G9 (copilot-instructions.md) — eine Quelle für A/B und Produktion).

Kein Torch-/ONNX-Load: alle Kandidaten werden injiziert (schnell, deterministisch).
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from backend.core import music_model_flags
from backend.core.dsp.stem_separator import (
    MLStemSeparator,
    reconstruction_fidelity,
    stem_set_fidelity,
)

_N = 4096
_STEM_NAMES = ("vocals", "drums", "bass", "other")


def _mix() -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.standard_normal((_N, 2)) * 0.05).astype(np.float32)


def _perfect_stems(mix: np.ndarray) -> dict[str, np.ndarray]:
    """Stem-Set, dessen Summe exakt die Mixtur ist (Treue 1,0)."""
    part = (mix * 0.25).astype(np.float32)
    return {name: part.copy() for name in _STEM_NAMES}


def _silent_stems(mix: np.ndarray) -> dict[str, np.ndarray]:
    """Kollabiertes Stem-Set (Summe 0 ⇒ Treue 0, Vokal-Energie 0)."""
    return {name: np.zeros_like(mix) for name in _STEM_NAMES}


class _FakePlugin:
    """Injizierter SCNet-Kandidat — kein Torch, kein Checkpoint."""

    def __init__(self, stems: dict[str, np.ndarray]) -> None:
        self._stems = stems
        self.calls = 0

    def separate(self, audio: np.ndarray, sr: int, **kwargs: object) -> dict[str, np.ndarray]:
        self.calls += 1
        return {name: value.copy() for name, value in self._stems.items()}


@pytest.fixture()
def separator() -> MLStemSeparator:
    return MLStemSeparator()


# ── 1. Sperre und Auflösung ─────────────────────────────────────────────────


def test_flag_is_locked_by_default() -> None:
    """Die Sperre ist eine dokumentierte Entscheidung, keine Lücke (§v10.802)."""
    assert music_model_flags.use_scnet_music is False


def test_resolve_model_path_without_flag_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ohne Freigabe kein Pfad — SCNet hat bewusst kein Legacy-Gegenstück."""
    monkeypatch.setattr(music_model_flags, "use_scnet_music", False)
    assert music_model_flags.resolve_model_path("scnet") is None


def test_resolve_model_path_with_flag_points_at_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(music_model_flags, "use_scnet_music", True)
    resolved = music_model_flags.resolve_model_path("scnet")
    assert resolved is not None
    assert resolved.name == "huge_scnet_4stems_v1.2.ckpt"
    assert resolved.exists(), f"SCNet-Artefakt fehlt: {resolved}"


def test_separator_never_touches_plugin_when_locked(
    monkeypatch: pytest.MonkeyPatch, separator: MLStemSeparator
) -> None:
    """Sperr-Pin: Flag aus ⇒ kein Plugin-Zugriff, Bestand bit-identisch zurück."""
    import plugins.scnet_plugin as scnet_module

    def _forbidden() -> object:  # pragma: no cover — darf nie laufen
        raise AssertionError("SCNet-Plugin ohne Freigabe angefasst (§V7 copilot-instructions.md)")

    monkeypatch.setattr(music_model_flags, "use_scnet_music", False)
    monkeypatch.setattr(scnet_module, "get_scnet_plugin", _forbidden)

    mix = _mix()
    incumbent = _perfect_stems(mix)
    monkeypatch.setattr(MLStemSeparator, "_separate_incumbent", lambda self, a, sr: incumbent)

    out = separator.separate(mix, 48000)

    assert set(out) == set(_STEM_NAMES)
    for name, arr in incumbent.items():
        np.testing.assert_array_equal(out[name], arr)
    assert "never_worsen" not in separator.metrics


def test_plugin_separate_fails_closed_without_approval(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """§V6 (copilot-instructions.md): Ohne Freigabe liefert das Plugin ``None`` mit Begründung — nicht still."""
    from plugins.scnet_plugin import SCNetPlugin

    monkeypatch.setattr(music_model_flags, "use_scnet_music", False)
    with caplog.at_level(logging.INFO):
        result = SCNetPlugin().separate(np.zeros((1024, 2), dtype=np.float32), 48000)

    assert result is None
    assert any("SCNet" in record.message for record in caplog.records), (
        "Sperre nicht protokolliert (§V6 (copilot-instructions.md))"
    )


# ── 2. Never-worsen Entscheidung ────────────────────────────────────────────


def test_never_worsen_accepts_better_candidate(monkeypatch: pytest.MonkeyPatch, separator: MLStemSeparator) -> None:
    import plugins.scnet_plugin as scnet_module

    mix = _mix()
    incumbent = _silent_stems(mix)
    candidate = _perfect_stems(mix)

    monkeypatch.setattr(music_model_flags, "use_scnet_music", True)
    monkeypatch.setattr(scnet_module, "get_scnet_plugin", lambda: _FakePlugin(candidate))
    monkeypatch.setattr(MLStemSeparator, "_separate_incumbent", lambda self, a, sr: incumbent)

    out = separator.separate(mix, 48000)

    np.testing.assert_array_equal(out["vocals"], candidate["vocals"])
    witness = separator.metrics["never_worsen"]
    assert witness["accepted"] is True
    assert witness["fidelity_candidate"] > witness["fidelity_incumbent"]
    assert witness["reasons"] == []
    assert separator.metrics["backend"] == "SCNet"


def test_never_worsen_rejects_worse_candidate(
    monkeypatch: pytest.MonkeyPatch, separator: MLStemSeparator, caplog: pytest.LogCaptureFixture
) -> None:
    """Treue und Vokal-Energie schlechter ⇒ Bestand bleibt, Grund wird berichtet."""
    import plugins.scnet_plugin as scnet_module

    mix = _mix()
    incumbent = _perfect_stems(mix)
    candidate = _silent_stems(mix)

    monkeypatch.setattr(music_model_flags, "use_scnet_music", True)
    monkeypatch.setattr(scnet_module, "get_scnet_plugin", lambda: _FakePlugin(candidate))
    monkeypatch.setattr(MLStemSeparator, "_separate_incumbent", lambda self, a, sr: incumbent)

    with caplog.at_level(logging.WARNING):
        out = separator.separate(mix, 48000)

    np.testing.assert_array_equal(out["vocals"], incumbent["vocals"])
    witness = separator.metrics["never_worsen"]
    assert witness["accepted"] is False
    assert witness["reasons"], "Verwerfungsgrund fehlt (§G8 copilot-instructions.md)"
    assert any("VERWORFEN" in record.message for record in caplog.records)


def test_never_worsen_rejects_vocal_collapse(monkeypatch: pytest.MonkeyPatch, separator: MLStemSeparator) -> None:
    """Gleiche Treue, aber kollabierter Vokal-Stem ⇒ Bestand bleibt (§V1-Geist).

    Die Summe der Stems bleibt exakt die Mixtur (Treue ≈ 1,0) — nur die
    Vokal-Energie wird zwischen den Stems verschoben. Damit prüft der Test
    ausschließlich den zweiten Zeugen (Vokal-Erhalt).
    """
    import plugins.scnet_plugin as scnet_module

    mix = _mix()
    incumbent = _perfect_stems(mix)
    candidate = _perfect_stems(mix)
    removed = candidate["vocals"].copy()
    candidate["vocals"] = (mix * 0.0025).astype(np.float32)  # 1 % der Bestands-Energie
    candidate["other"] = (candidate["other"] + (removed - candidate["vocals"])).astype(np.float32)

    monkeypatch.setattr(music_model_flags, "use_scnet_music", True)
    monkeypatch.setattr(scnet_module, "get_scnet_plugin", lambda: _FakePlugin(candidate))
    monkeypatch.setattr(MLStemSeparator, "_separate_incumbent", lambda self, a, sr: incumbent)

    out = separator.separate(mix, 48000)

    np.testing.assert_array_equal(out["vocals"], incumbent["vocals"])
    witness = separator.metrics["never_worsen"]
    assert witness["accepted"] is False
    assert any("Vokal-Energie" in reason for reason in witness["reasons"])
    assert witness["fidelity_candidate"] > witness["fidelity_incumbent"] - 0.01, (
        "Testdesign: nur der Vokal-Zeuge darf verletzt sein"
    )


def test_never_worsen_survives_broken_candidate(monkeypatch: pytest.MonkeyPatch, separator: MLStemSeparator) -> None:
    """Ein werfender Kandidat darf den Bestand nicht gefährden (§V6 (copilot-instructions.md))."""
    import plugins.scnet_plugin as scnet_module

    class _Broken:
        def separate(self, audio: np.ndarray, sr: int, **kwargs: object) -> None:
            raise RuntimeError("SCNet kaputt")

    mix = _mix()
    incumbent = _perfect_stems(mix)

    monkeypatch.setattr(music_model_flags, "use_scnet_music", True)
    monkeypatch.setattr(scnet_module, "get_scnet_plugin", _Broken)
    monkeypatch.setattr(MLStemSeparator, "_separate_incumbent", lambda self, a, sr: incumbent)

    out = separator.separate(mix, 48000)

    np.testing.assert_array_equal(out["vocals"], incumbent["vocals"])
    assert "never_worsen" not in separator.metrics


# ── 3. Kanonische Formel (§G9 (copilot-instructions.md) — eine Quelle) ─────────


def test_reconstruction_fidelity_exact_and_degenerate() -> None:
    mix = _mix()
    assert reconstruction_fidelity(mix, mix) == pytest.approx(1.0, abs=1e-6)
    assert reconstruction_fidelity(mix, np.zeros_like(mix)) == 0.0
    assert stem_set_fidelity(mix, _perfect_stems(mix)) == pytest.approx(1.0, abs=1e-5)
    assert stem_set_fidelity(mix, _silent_stems(mix)) == 0.0
