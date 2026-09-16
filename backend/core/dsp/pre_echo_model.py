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
_ONSET_RISE = 4.0  # Hüllkurven-Anstieg × lokaler Baseline = Onset-Kandidat
_LOCAL_WIN_S = 1.0  # Lokales Referenz-Fenster (±0,5 s) für die Onset-Baseline
_LOCAL_PERC = 10.0  # Baseline = 10. Perzentil des lokalen Fensters (robust ggü. Lautheit)
_FWD_MASK_DB = 18.0  # Forward-Masking (Zwicker & Fastl §7.2): Pre-Energie unter
# Onset-Pegel − 18 dB ist nicht hörbar → kein Befund


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
    # Hinzugefügte (positive) Delta-Energie je Frame — für das Audibility-Gate.
    # |delta|² allein ist vorzeichenblind: Klick-ENTFERNUNG im Vor-Fenster zählt
    # sonst wie Pre-Echo-HINZUFÜGUNG (False-Positive bei Declickern).
    env_d_pos = np.sqrt(np.mean(np.maximum(delta[idx], 0.0) ** 2, axis=1)) + 1e-18

    # Onset-Kandidaten: Nach-Hüllkurve steigt stark relativ zur LOKALEN Baseline
    # (10. Perzentil über ±0,5 s) — der globale Median verschluckt Anstiege in
    # lauten Passagen (Produktionsbefund: 6,3×-Attack als 1,85× gemessen).
    _loc_win = max(4, int(_LOCAL_WIN_S / _ENV_WIN_S))
    _floor = float(np.max(env_a)) * 10.0 ** (-60.0 / 20.0)  # −60 dB unter Song-Peak:
    # Stille erzeugt keine Onsets (Baseline ≈ 0 würde jedes Rauschen als Anstieg
    # werten — Produktionsbefund auf realem Song-Anfang).
    _loc_base = np.zeros(n_frames, dtype=np.float64)
    for f in range(n_frames):
        _lo = max(0, f - _loc_win)
        _hi = min(n_frames, f + _loc_win + 1)
        _loc_base[f] = max(float(np.percentile(env_a[_lo:_hi], _LOCAL_PERC)), _floor)
    rise = env_a / (_loc_base + 1e-12)
    onset_frames = np.where(rise > _ONSET_RISE)[0]
    if onset_frames.size == 0:
        return -200.0

    pre_frames = max(1, int(_PRE_WIN_S / _ENV_WIN_S))
    gap_frames = max(1, int(_GAP_S / _ENV_WIN_S))
    post_frames = max(1, int(_POST_WIN_S / _ENV_WIN_S))

    worst_db = -200.0
    _mask_floor_db = 10.0 ** (-_FWD_MASK_DB / 10.0)
    # SUP-F6 (2026-09-16): Absolute Hörbarkeits-Schwelle — hinzugefügte
    # Pre-Energie unter −60 dB des Song-Peaks (gleiche Konstante wie die
    # Onset-Baseline) ist unter jeder Maskierungsbedingung unhörbar. Bei
    # leisen Onsets gehen die relativen Schwellen gegen 0 und Delta-Rauschen
    # ≈0-Delta-Phasen (phase_01 micro_fallback Δ=+0,00 dB, phase_47-Limiter)
    # passierte sie — Produktionsbefund 2026-09-16.
    _abs_floor_e = (float(np.max(env_a)) * 10.0 ** (-60.0 / 20.0)) ** 2
    for _of in onset_frames:
        _pre_lo = max(0, _of - pre_frames)
        _pre_hi = max(0, _of - gap_frames)
        _post_hi = min(n_frames, _of + post_frames)
        if _pre_hi <= _pre_lo or _post_hi <= _of:
            continue
        # Forward-Masking-Gate: Nur HINZUGEFÜGTE Vor-Fenster-Energie, die über
        # Onset-Pegel − 18 dB liegt, kann als Pre-Echo hörbar sein.
        _added_pre_e = float(np.mean(env_d_pos[_pre_lo:_pre_hi] ** 2))
        if _added_pre_e < float(env_a[_of] ** 2) * _mask_floor_db:
            continue
        if _added_pre_e < _abs_floor_e:
            continue
        # SUP-F6 (2026-09-16): Zweite Audibility-Schwelle — hinzugefügte
        # Pre-Energie unter dem lokalen Vor-Fenster-Signalpegel − 18 dB ist
        # maskiert und löst KEINEN Befund aus. Bisher erzeugte Delta-Rauschen
        # (≈0-Delta-Phasen, z. B. Limiter bei pitch=0.0c/loud=0.0dB) einen
        # Pre-Echo-Befund, weil das Verhältnis zweier Rausch-Energien beliebig
        # groß werden kann.
        _pre_signal_e = float(np.mean(env_a[_pre_lo:_pre_hi] ** 2)) + 1e-18
        if _added_pre_e < _pre_signal_e * _mask_floor_db:
            continue
        # SUP-F6 (2026-09-16): Das Verhältnis wird auf HINZUGEFÜGTER Energie
        # (positive Delta-Hälfte) gebildet — die signierte Delta-Energie zählte
        # Klick-ENTFERNUNG im Vor-Fenster wie eine Pre-Echo-HINZUFÜGUNG
        # (False-Positive bei phase_01/23, Produktionsbefund 2026-09-16).
        _pre_e = float(np.mean(env_d_pos[_pre_lo:_pre_hi] ** 2)) + 1e-18
        _post_e = float(np.mean(env_d_pos[_of:_post_hi] ** 2)) + 1e-18
        ratio_db = 10.0 * np.log10(_pre_e / _post_e)
        worst_db = max(worst_db, ratio_db)
    return float(round(worst_db, 2))
