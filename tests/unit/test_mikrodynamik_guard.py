"""§MKK-V20: Mikrodynamik-Guard — Delta-Korrelations-Metrik (2026-09-11).

Der Guard schützt die Mikrodynamik-FORM (Frame-zu-Frame-Übergänge) statt der
absoluten Frame-Energien: Absichtliches Vocal-Presence-Shaping (lyrics-guided)
passiert (Delta-Korr ≥ 0.93), echte Dynamik-Zerstörung (Limiter/Kompressor)
fällt durch. Produktionsbefund: absolute Korrelation verwarf 61 % der
Restaurierung (wet=0.39) ohne hörbaren Dynamikschaden.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core.dsp.mikrodynamik_guard import frame_energy_correlation, recommend_mikrodynamik_wet

SR = 48_000


def _music_like(seconds: float = 6.0, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    x = (0.4 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 440 * t) + 0.25 * rng.normal(0, 1, len(t))) * (
        1.0 + 0.4 * np.sin(2 * np.pi * 0.5 * t) + 0.2 * np.sin(2 * np.pi * 1.7 * t)
    )
    return np.clip(x, -1, 1).astype(np.float32)


def test_identity_is_one() -> None:
    x = _music_like()
    assert frame_energy_correlation(x, x.copy(), SR) == pytest.approx(1.0, abs=1e-4)


def test_smooth_word_boosts_pass() -> None:
    """Glatte Wort-Regionen-Boosts (Vocal-Presence-Shaping) dürfen nicht als
    Dynamik-Zerstörung gewertet werden — sie erhalten die Übergangs-Form."""
    x = _music_like()
    y = x.copy()
    sr = SR
    for t0 in (0.5, 1.6, 2.8, 3.9, 5.0):
        i0, i1 = int(t0 * sr), int((t0 + 1.0) * sr)
        n = i1 - i0
        fade = 0.5 - 0.5 * np.cos(np.pi * np.arange(n) / n)
        y[i0:i1] = y[i0:i1] * (1.0 + 0.10 * fade)
    corr = frame_energy_correlation(x, y.astype(np.float32), SR)
    assert corr >= 0.93, f"Wort-Boosts dürfen nicht blocken: corr={corr:.3f}"


def test_hard_limiter_fails() -> None:
    """Harter Limiter zerstört die Übergangs-Form — der Guard muss greifen."""
    x = _music_like()
    thr_db = -28.0
    lim = np.clip(x / (10 ** (thr_db / 20)), -1, 1) * (10 ** (thr_db / 20))
    corr = frame_energy_correlation(x, lim.astype(np.float32), SR)
    assert corr < 0.93, f"Limiter muss durchfallen: corr={corr:.3f}"


def test_recommend_wet_full_for_shaped_chain() -> None:
    """LGE-typische Kette (Delta-Korr ~0.95) → voller Wet-Anteil."""
    wet = recommend_mikrodynamik_wet(0.954, panns_singing=0.35, material="mp3_low")
    assert wet >= 0.95, f"Intentionales Shaping darf nicht verworfen werden: wet={wet:.3f}"


def test_recommend_wet_low_for_limiter() -> None:
    """Limiter-Korrelation → stark reduzierter Wet-Anteil."""
    wet = recommend_mikrodynamik_wet(0.73, panns_singing=0.35, material="mp3_low")
    assert wet < 0.35, f"Echter Schaden muss Wet reduzieren: wet={wet:.3f}"
