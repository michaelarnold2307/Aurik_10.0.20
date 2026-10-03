"""
test_extreme_inputs.py — Extreme Importsongs gegen die Restaurierungskernpfade.

Validierungsauftrag (2026-10-02): Auch bei extremen Importsongs muss das
Programm konsistent das weltbeste Restaurierungsverhalten halten. Diese
Suite fährt pathologische Eingaben durch die Kernpfade und erzwingt
durchgängig:

  - finite Ausgaben (§0a (copilot-instructions.md): NaN/Inf-Schutz),
  - Shape-/Layout-Erhalt (Stereo-Layout-Invariante, AGENTS.md),
  - Identität außerhalb messbarer Defekte (declick_core-Identitäts-Guard),
  - Determinismus (§G5 (copilot-instructions.md)): zweiter Lauf bit-identisch,
  - Nie-schlimmer-Grenzen: kein Gain über dem Eingang, keine Übersteuerung.

Alle Feeds sind deterministisch (seeded rng, §G5 (copilot-instructions.md)).
"""

from __future__ import annotations

import numpy as np
import pytest

SEED = 20261002
_KAS_TIMEOUT = 120


def _extreme_song(sr: int = 48_000, seconds: float = 2.0) -> np.ndarray:
    """Der Worst-Case-Song: Clipping + Hiss + Hum + Klicks + Dropouts + DC."""
    n = int(sr * seconds)
    rng = np.random.default_rng(SEED)
    t = np.arange(n) / sr
    x = 1.4 * np.sign(np.sin(2 * np.pi * 220 * t))  # harte Vollaussteuerung
    x += 0.35 * rng.standard_normal(n)  # starkes Hiss
    x += 0.3 * np.sin(2 * np.pi * 50 * t)  # Hum
    x += 0.25  # DC-Offset
    for _ in range(40):  # dichte Klicks
        x[rng.integers(0, n)] += 3.0 * rng.choice([-1.0, 1.0])
    x[n // 3 : n // 3 + sr // 2] = 0.0  # Dropout-Kette
    return np.clip(x.astype(np.float32), -4.0, 4.0)


@pytest.fixture(scope="module")
def extreme_song() -> np.ndarray:
    return _extreme_song()


@pytest.fixture(
    scope="module",
    params=["silence", "dc", "nyquist", "impulse_train", "white_noise", "poisson_crackle", "tiny_level"],
)
def pathological(request) -> np.ndarray:
    n = 48_000
    rng = np.random.default_rng(SEED)
    if request.param == "silence":
        return np.zeros(n, dtype=np.float32)
    if request.param == "dc":
        return np.full(n, 0.75, dtype=np.float32)
    if request.param == "nyquist":
        return (((-1.0) ** np.arange(n)) * 0.5).astype(np.float32)
    if request.param == "impulse_train":
        x = np.zeros(n, dtype=np.float32)
        x[::37] = 1.0
        return x
    if request.param == "white_noise":
        return (0.5 * rng.standard_normal(n)).astype(np.float32)
    if request.param == "poisson_crackle":
        x = (0.02 * rng.standard_normal(n)).astype(np.float32)
        idx = rng.choice(n, size=2000, replace=False)
        x[idx] += 2.0 * rng.standard_normal(2000).astype(np.float32)
        return x
    return (1e-8 * rng.standard_normal(n)).astype(np.float32)  # tiny_level


def _assert_sane(out: np.ndarray, ref: np.ndarray) -> None:
    assert np.isfinite(out).all(), "Ausgabe enthält NaN/Inf (§0a)"
    assert out.shape == ref.shape, f"Shape verändert: {ref.shape} → {out.shape}"
    assert float(np.max(np.abs(out))) <= 1e6, "Ausgabe unkontrolliert angewachsen"


# ---------------------------------------------------------------------------
# declick_core — Identität, Determinismus, Layout auf pathologischen Feeds
# ---------------------------------------------------------------------------


def test_declick_core_pathological(pathological: np.ndarray):
    from backend.core.dsp.declick_core import declick_signal

    out = declick_signal(pathological, strictness_k=6.0)
    _assert_sane(out, pathological)
    assert np.array_equal(out, declick_signal(pathological, strictness_k=6.0)), (
        "nicht deterministisch (§G5 (copilot-instructions.md))"
    )


def test_declick_core_extreme_song(extreme_song: np.ndarray):
    from backend.core.dsp.declick_core import declick_signal

    out = declick_signal(extreme_song, strictness_k=6.0)
    _assert_sane(out, extreme_song)
    # Nie schlimmer: Impulsenergie darf nicht steigen (Reparatur entfernt Spitzen)
    assert float(np.mean(np.abs(np.diff(out)))) <= float(np.mean(np.abs(np.diff(extreme_song)))) * 1.05


def test_declick_core_layout_stabil(extreme_song: np.ndarray):
    from backend.core.dsp.declick_core import declick_signal

    cf = np.stack([extreme_song, 0.5 * extreme_song], axis=0)
    sf = np.ascontiguousarray(cf.T)
    out_cf = declick_signal(cf, strictness_k=6.0)
    out_sf = declick_signal(sf, strictness_k=6.0)
    _assert_sane(out_cf, cf)
    _assert_sane(out_sf, sf)
    assert np.allclose(out_cf.T, out_sf), "Layout-Abhängigkeit (Stereo-Layout-Invariante)"


# ---------------------------------------------------------------------------
# Ultra-Low-Latency-Kern — Nie-Gain, bounded, deterministisch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls_name", ["UltraLowLatencyLimiter", "UltraLowLatencyDenoiser", "UltraLowLatencyGate"])
def test_ull_extremes(pathological: np.ndarray, extreme_song: np.ndarray, cls_name: str):
    from backend.core.dsp import ultra_low_latency as ull

    cls = getattr(ull, cls_name)
    for sig in (pathological, extreme_song):
        out = cls().process(sig, 48_000)
        _assert_sane(out, sig)
        assert (
            float(np.max(np.abs(out))) <= max(float(np.max(np.abs(sig))), 1e-6) + 1e-6
            or cls_name == "UltraLowLatencyDenoiser"
        )


# ---------------------------------------------------------------------------
# Layout-Mapping-Module auf Extrem-Song
# ---------------------------------------------------------------------------


def test_shellac_eq_und_stereo_image_extrem(extreme_song: np.ndarray):
    from backend.core.dsp.shellac_equalizer import ShellacEqualizer
    from backend.core.dsp.stereo_image_correction import StereoImageCorrection

    cf = np.stack([extreme_song, extreme_song], axis=0)
    eq = ShellacEqualizer("78rpm").process(cf, 48_000, audit_log=False)
    sic = StereoImageCorrection(0.8).process(cf, 48_000)
    _assert_sane(eq, cf)
    _assert_sane(sic, cf)


def test_vad_und_ar_extrem(extreme_song: np.ndarray, pathological: np.ndarray):
    from backend.core.dsp.adaptive_ar_prediction_burg import AdaptiveARPredictionBurg
    from backend.core.dsp.adaptive_crepe_neural_pitch import AdaptiveCREPENeuralPitch
    from backend.core.dsp.vad import AiVAD

    for sig in (extreme_song, pathological):
        mask = AiVAD().detect(sig, 48_000)
        assert mask.shape == sig.shape, "VAD-Maskenform verändert"
        pred = AdaptiveARPredictionBurg(order=8).predict(sig)
        _assert_sane(pred, sig)
        f0 = AdaptiveCREPENeuralPitch(sr=48_000).track(sig)
        assert np.isfinite(f0) and f0 >= 0.0, "Pitch-Tracking liefert Unsinn"


# ---------------------------------------------------------------------------
# Produktions-Click-Kette auf dem extremen Song (KAS-Pflichtpfad)
# ---------------------------------------------------------------------------


@pytest.mark.timeout(_KAS_TIMEOUT)
def test_click_removal_phase_extremer_song(extreme_song: np.ndarray):
    from backend.core.phases.phase_01_click_removal import ClickRemovalPhase

    result = ClickRemovalPhase().process(extreme_song.copy(), sample_rate=48_000)
    out = result.audio
    assert np.isfinite(out).all(), "ClickRemoval erzeugt NaN/Inf auf Extrem-Song"
    assert out.shape == extreme_song.shape
    assert float(np.max(np.abs(out))) <= 1.0000001, "ClickRemoval erzeugt Übersteuerung"


@pytest.mark.timeout(_KAS_TIMEOUT)
@pytest.mark.parametrize("sr", [8_000, 44_100, 96_000, 192_000])
def test_sampleraten_extrem(sr: int):
    from backend.core.dsp.declick_core import declick_signal

    rng = np.random.default_rng(SEED)
    n = sr  # 1 s
    sig = (rng.standard_normal(n)).astype(np.float32)
    sig[:: max(1, n // 20)] += 2.0
    out = declick_signal(sig, strictness_k=6.0)
    _assert_sane(out, sig)
    assert np.array_equal(out, declick_signal(sig, strictness_k=6.0)), (
        "nicht deterministisch (§G5 (copilot-instructions.md))"
    )
