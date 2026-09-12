"""Pre-Echo-Proxy — deterministische Forward-Masking-Verletzungserkennung.

§Witness-SOTA P4 (2026-09-12, docs/WITNESS_SOTA_GAP_ANALYSE.md §5):
Zeitbereichs-Prozessoren (Codec/ML-Klassen) können Energie VOR einem
Transienten „vorladen" — hörbar als Pre-Echo (Forward-Masking-Verletzung).
Proxy: an starken Onsets des Nach-Signals wird die Delta-Energie im
Vor-Fenster [t−20 ms, t−2 ms] gegen die Onset-Energie [t−2 ms, t+10 ms]
gesetzt — positiv dB = Pre-Echo-Verdacht. Rein deterministisch (5-ms-Hüllkurven).

Referenz: Zwicker & Fastl (2007) §7.2 (Pre-/Post-Masking); Godsill & Rayner
(1998) (Impuls-Interpolation als Gegenstück).
"""

from __future__ import annotations

import numpy as np

_PRE_WIN_S = 0.020  # 20 ms Vor-Fenster
_GAP_S = 0.002  # 2 ms Schutzabstand zum Onset
_POST_WIN_S = 0.010  # 10 ms Onset-Fenster
_ENV_WIN_S = 0.005  # 5 ms Hüllkurven-Raster
_ONSET_RISE = 4.0  # Hüllkurven-Anstieg × Median = Onset-Kandidat


def pre_echo_ratio_db(x_before: np.ndarray, x_after: np.ndarray, sr: int) -> float:
    """Maximales Pre-Echo-Verhältnis (dB) des Deltas (after − before).

    Returns:
        float — positiv = Pre-Echo-Verdacht; −200.0, wenn kein hörrelevantes
        Delta existiert oder keine Onsets gefunden werden. Deterministisch.
    """
    before = np.asarray(x_before, dtype=np.float32)
    after = np.asarray(x_after, dtype=np.float32)
    n = min(len(before), len(after))
    before, after = before[:n], after[:n]
    if n < int(sr * 0.05):
        return -200.0

    delta = (after - before).astype(np.float32)
    delta_rms = float(np.sqrt(np.mean(delta**2))) + 1e-12
    after_rms = float(np.sqrt(np.mean(after**2))) + 1e-12
    if delta_rms < 1e-4 * after_rms:
        return -200.0  # kein hörrelevantes Delta — kein Pre-Echo möglich
    hop = max(1, int(sr * _ENV_WIN_S))
    n_frames = max(1, (n - hop) // hop + 1)
    idx = np.arange(hop)[None, :] + hop * np.arange(n_frames)[:, None]

    env_a = np.sqrt(np.mean(after[idx] ** 2, axis=1)) + 1e-12
    env_d = np.sqrt(np.mean(delta[idx] ** 2, axis=1)) + 1e-12

    # Onset-Kandidaten: Nach-Hüllkurve steigt stark relativ zum lokalen Median.
    med = float(np.median(env_a)) + 1e-12
    rise = env_a / med
    onset_frames = np.where(rise > _ONSET_RISE)[0]
    if onset_frames.size == 0:
        return -200.0

    pre_frames = max(1, int(_PRE_WIN_S / _ENV_WIN_S))
    gap_frames = max(1, int(_GAP_S / _ENV_WIN_S))
    post_frames = max(1, int(_POST_WIN_S / _ENV_WIN_S))

    worst_db = -200.0
    for _of in onset_frames:
        _pre_lo = max(0, _of - pre_frames)
        _pre_hi = max(0, _of - gap_frames)
        _post_hi = min(n_frames, _of + post_frames)
        if _pre_hi <= _pre_lo or _post_hi <= _of:
            continue
        _pre_e = float(np.mean(env_d[_pre_lo:_pre_hi] ** 2)) + 1e-18
        _post_e = float(np.mean(env_d[_of:_post_hi] ** 2)) + 1e-18
        ratio_db = 10.0 * np.log10(_pre_e / _post_e)
        worst_db = max(worst_db, ratio_db)
    return float(round(worst_db, 2))
