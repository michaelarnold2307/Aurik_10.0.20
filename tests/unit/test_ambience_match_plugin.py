"""Tests für das Ambience-Match-Plugin (Spec 25, Hörordnung Ebene 2).

Testbasis: echtes Audio (tests/real_world_validation/test_library), keine
Synthese. Abgedeckte Invarianten:
- H4/Passthrough: blend=0 => bit-identischer Passthrough (dtype erhalten)
- Determinismus §G5 (copilot-instructions.md): gleicher Seed => bit-identisches Ambiente; zustandslos
  (zweiter Aufruf identisch)
- H1 Audibility: Ambiente-Pegel je Bark-Band <= Maskierungsschwelle − σ
- H2 Timbre-Neutralität: rein additiv — Signal-Komponente unverändert
- H3 Defekt-Maskierung: >= 12 dB Zusatzdämpfung in Defekt-Bändern
- Stereo: Shape erhalten, Kanäle unabhängig

Spec: .github/specs/25_ambience_match_plugin.md
"""

from pathlib import Path
from typing import cast

import numpy as np
import pytest
import soundfile as sf

from backend.core.dsp.masking_model import (
    _frame_band_energies,
    compute_masking_threshold_db,
)
from plugins.ambience_match_plugin import (
    DEFAULT_SESSION_SEED,
    GLOBAL_AMBIENCE_SCALAR,
    AmbienceMatchPlugin,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REAL_AUDIO = _REPO_ROOT / "tests/real_world_validation/test_library/digital/digital_test_01.wav"
SR = 44100
N_SECONDS = 9  # 3 Chunks: Ring-In (Chunk 0) + stationäre Bereiche


@pytest.fixture(scope="module")
def signal() -> np.ndarray:
    """Echtes Audio: digital_test_01.wav (erste 9 s, 44,1 kHz, mono)."""
    x_raw, sr = sf.read(str(_REAL_AUDIO), dtype="float32")
    x = cast(np.ndarray, x_raw)
    assert sr == SR
    assert x.ndim == 1
    return x[: N_SECONDS * SR]


def _band_levels_db(x: np.ndarray, sr: int) -> np.ndarray:
    e_db, _ = _frame_band_energies(np.asarray(x, dtype=np.float32), sr)
    return cast(np.ndarray, np.mean(e_db, axis=0))


def _loudest_band(x: np.ndarray, sr: int) -> int:
    thr, _ = compute_masking_threshold_db(x, sr)
    return int(np.argmax(np.mean(thr, axis=0)))


def test_blend_zero_passthrough(signal: np.ndarray) -> None:
    p = AmbienceMatchPlugin(blend=0.0)
    out = p.process(signal, SR)
    assert out is signal  # unverändertes Objekt, bit-identisch (H4)
    assert out.dtype == np.float32


def test_blend_zero_passthrough_stereo(signal: np.ndarray) -> None:
    p = AmbienceMatchPlugin(blend=0.0)
    stereo = np.stack([signal, signal * 0.5]).astype(np.float32)
    out = p.process(stereo, SR)
    assert out is stereo


def test_determinism_same_seed(signal: np.ndarray) -> None:
    p1 = AmbienceMatchPlugin(blend=0.3, seed=DEFAULT_SESSION_SEED)
    p2 = AmbienceMatchPlugin(blend=0.3, seed=DEFAULT_SESSION_SEED)
    a1 = p1.ambience(signal, SR)
    a2 = p2.ambience(signal, SR)
    assert np.array_equal(a1, a2)  # bit-identisch (§G5 (copilot-instructions.md))


def test_stateless_repeat_call(signal: np.ndarray) -> None:
    p = AmbienceMatchPlugin(seed=7)
    a1 = p.ambience(signal, SR)
    a2 = p.ambience(signal, SR)
    assert np.array_equal(a1, a2)  # zustandslos: kein per-Song-State


def test_different_seed_differs(signal: np.ndarray) -> None:
    a1 = AmbienceMatchPlugin(seed=1).ambience(signal, SR)
    a2 = AmbienceMatchPlugin(seed=2).ambience(signal, SR)
    assert not np.array_equal(a1, a2)
    assert float(np.max(np.abs(a1 - a2))) > 1e-4


def test_h1_audibility_below_masking_threshold(signal: np.ndarray) -> None:
    sigma = 6.0
    p = AmbienceMatchPlugin(safety_margin_db=sigma, seed=11)
    amb = p.ambience(signal, SR)
    thr, _ = compute_masking_threshold_db(signal, SR)
    thr_mean = np.mean(thr, axis=0)
    amb_db = _band_levels_db(amb, SR)
    # H1: Ambiente bleibt je Band unter der Maskierungsschwelle − σ
    assert np.all(amb_db <= thr_mean - sigma + 1.0)
    # Sanity: das profil-dominante Band erreicht die Schwelle (kein Null-Ambiente)
    assert np.any(amb_db >= thr_mean - sigma - 3.0)


def test_h2_signal_component_untouched(signal: np.ndarray) -> None:
    blend = 0.35
    p = AmbienceMatchPlugin(blend=blend, seed=13)
    out = p.process(signal, SR)
    amb = p.ambience(signal, SR)
    expected = signal + np.float32(blend * GLOBAL_AMBIENCE_SCALAR) * amb
    assert np.array_equal(out, expected)  # rein additiv (H2)


def test_h3_defect_band_attenuation(signal: np.ndarray) -> None:
    band = _loudest_band(signal, SR)
    amb_ref = AmbienceMatchPlugin(seed=17).ambience(signal, SR)
    amb_def = AmbienceMatchPlugin(seed=17).ambience(signal, SR, defect_bands=[band])
    ref_db = _band_levels_db(amb_ref, SR)
    def_db = _band_levels_db(amb_def, SR)
    # H3: >= 12 dB Zusatzdämpfung im Defekt-Band
    assert def_db[band] <= ref_db[band] - 12.0 + 0.5


def test_stereo_shape_and_independent_channels(signal: np.ndarray) -> None:
    p = AmbienceMatchPlugin(blend=0.25, seed=19)
    stereo = np.stack([signal, signal * 0.5]).astype(np.float32)
    out = p.process(stereo, SR)
    assert out.shape == stereo.shape
    assert out.dtype == np.float32
    amb = p.ambience(stereo, SR)
    assert amb.shape == stereo.shape
    assert not np.array_equal(amb[0], amb[1])  # unabhängige Kanäle


def test_short_real_audio_below_chunk_size(signal: np.ndarray) -> None:
    short = signal[:8000]  # echtes Audio, kürzer als ein 3-s-Chunk
    p = AmbienceMatchPlugin(blend=0.5, seed=23)
    out = p.process(short, SR)
    assert out.shape == short.shape
    assert np.all(np.isfinite(out))


def test_blend_validation() -> None:
    with pytest.raises(ValueError):
        AmbienceMatchPlugin(blend=1.5)
    with pytest.raises(ValueError):
        AmbienceMatchPlugin(safety_margin_db=2.0)  # unter Mindestmarge
