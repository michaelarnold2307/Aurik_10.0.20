from __future__ import annotations

"""Unit-Tests für musiclover-Exportoptimierung im AudioExporter.

Testet _apply_musiclover_export_optimizations direkt — keine sf.write-Mocks nötig.
"""


import numpy as np
import pytest

from backend.core.audio_exporter import _apply_musiclover_export_optimizations


def _anti_phase_stereo(n: int = 128) -> np.ndarray:
    return np.column_stack([np.ones(n, dtype=np.float32), -np.ones(n, dtype=np.float32)])


@pytest.mark.unit
def test_mono_guard_applies_side_softening() -> None:
    """Anti-Phasen-Stereo: Side-Pegel wird reduziert (kein harter Mono-Downmix)."""
    audio = _anti_phase_stereo(96)
    metadata = {
        "quality_gate_musiclover_mono_warning": "True",
        "quality_gate_musiclover_mono_softened": "False",
        "quality_gate_musiclover_vqi": "1.0",
        "quality_gate_musiclover_temporal_hotspots": "0",
        "quality_gate_musiclover_remaining_goals": "0",
    }

    result = _apply_musiclover_export_optimizations(audio, metadata, None)

    assert result.shape == audio.shape
    # Side softening: Links > 0.92 (original 1.0 minus side_scale), Rechts < -0.92
    assert float(result[:, 0].mean()) == pytest.approx(0.92, abs=1e-6)
    assert float(result[:, 1].mean()) == pytest.approx(-0.92, abs=1e-6)


def test_mono_guard_skipped_when_already_softened() -> None:
    """Bereits gemildertes Audio wird nicht erneut modifiziert (Passthrough)."""
    audio = _anti_phase_stereo(96)
    metadata = {
        "quality_gate_musiclover_mono_warning": "True",
        "quality_gate_musiclover_mono_softened": "True",
        "quality_gate_musiclover_vqi": "1.0",
    }

    result = _apply_musiclover_export_optimizations(audio, metadata, None)

    assert result.shape == audio.shape
    # Keine Änderung — already_softened=True → Passthrough
    assert float(result[:, 0].mean()) == pytest.approx(1.0, abs=1e-6)
    assert float(result[:, 1].mean()) == pytest.approx(-1.0, abs=1e-6)


def test_mono_input_unchanged() -> None:
    """Mono-Audio wird nicht verändert."""
    audio = np.ones((96, 1), dtype=np.float32) * 0.5
    metadata = {"quality_gate_musiclover_mono_warning": "True"}

    result = _apply_musiclover_export_optimizations(audio, metadata, None)

    np.testing.assert_array_almost_equal(result, audio)


def test_no_warning_passthrough() -> None:
    """Ohne Mono-Warnung: Passthrough."""
    audio = _anti_phase_stereo(96)
    metadata: dict[str, str] = {}

    result = _apply_musiclover_export_optimizations(audio, metadata, None)

    np.testing.assert_array_almost_equal(result, audio)


# ---------------------------------------------------------------------------
# Regression (§V5 (copilot-instructions.md) / §IV / §G5 (GEBOTE.md)): Dither-Kette des Produktions-Exporters
# ---------------------------------------------------------------------------
# Produktionsbefund (empirisch gemessen):
#   * bit_depth=24 (CLI-Default UND GUI-Default "FLAC 24-bit") erhielt GAR
#     KEINEN Dither → unditherte Quantisierung, §V5-Verstoß.
#   * bit_depth=16 erhielt ZWEI Dither-Durchläufe auf demselben Pfad
#     (_apply_dither_16bit + apply_powr_dither) → +3 dB Rauschboden und
#     Reihenfolgeverstoß gegen §IV (4 → 3 → 4).
#   * Beide Pfade nutzten ungesäte Zufallsgeneratoren → §G5 (GEBOTE.md) gebrochen.


def _fade_stereo(n: int = 48000, sr: int = 48000) -> np.ndarray:
    """Leises, tonales Ausklang-Material (der Fall, in dem Dither hörbar wird)."""
    t = np.arange(n) / sr
    env = np.linspace(1.0, 5e-4, n)
    return np.column_stack(
        [
            0.02 * np.sin(2 * np.pi * 440 * t) * env,
            0.02 * np.sin(2 * np.pi * 445 * t) * env,
        ]
    ).astype(np.float32)


@pytest.mark.unit
@pytest.mark.parametrize("bit_depth", [16, 24])
def test_export_applies_exactly_one_dither(bit_depth: int, monkeypatch, tmp_path) -> None:
    """Genau EIN Dither-Schritt pro Export, auch bei 24 bit (§V5 (copilot-instructions.md), §IV)."""
    import backend.core.dsp.powr_dither as _pd
    from backend.core.audio_exporter import AudioExporter

    calls: list[object] = []
    real = _pd.apply_powr_dither

    def _spy(audio, sample_rate, bit_depth=16, seed=None):
        calls.append(seed)
        return real(audio, sample_rate, bit_depth=bit_depth, seed=seed)

    monkeypatch.setattr(_pd, "apply_powr_dither", _spy)

    AudioExporter().export(_fade_stereo(), 48000, tmp_path / f"d{bit_depth}.flac", bit_depth=bit_depth)

    assert len(calls) == 1, f"bit_depth={bit_depth}: {len(calls)} Dither-Durchläufe statt 1"
    assert calls[0] is not None, "Dither lief ohne Seed (§G5 (GEBOTE.md))"


@pytest.mark.unit
@pytest.mark.parametrize("bit_depth", [16, 24])
def test_export_is_bit_identical_for_identical_input(bit_depth: int, tmp_path) -> None:
    """§G5 (GEBOTE.md): gleicher Input ⇒ bit-identischer Output (auch mit Dither)."""
    import hashlib

    from backend.core.audio_exporter import AudioExporter

    audio = _fade_stereo()
    exp = AudioExporter()
    first = exp.export(audio, 48000, tmp_path / f"a{bit_depth}.flac", bit_depth=bit_depth)
    second = exp.export(audio, 48000, tmp_path / f"b{bit_depth}.flac", bit_depth=bit_depth)

    assert hashlib.sha256(first.read_bytes()).hexdigest() == hashlib.sha256(second.read_bytes()).hexdigest()


@pytest.mark.unit
def test_dither_seed_depends_on_content() -> None:
    """Verschiedene Signale müssen verschiedene Dither-Seeds erhalten."""
    from backend.core.audio_exporter import _deterministic_dither_seed

    a = _fade_stereo(4800)
    b = a.copy()
    b[1234] += 1e-3
    assert _deterministic_dither_seed(a) != _deterministic_dither_seed(b)
    assert _deterministic_dither_seed(a) == _deterministic_dither_seed(a.copy())
