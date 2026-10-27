"""D-K3-62: Worst-Onset-Lokalisierung + Frame-Energie-Profil für pre_echo-Befunde.

Zeigt für den/die schlimmsten Onsets: liegt die hinzugefügte Vor-Energie
im 5-15ms-Fenster vor der Attacke (echtes Pre-Echo/Ringing) oder direkt an
der Flanke (Attacken-Blutung/FP)?
"""

import sys

sys.path.insert(0, "/media/michael/Software 4TB/Aurik_Standalone")

import numpy as np
import soundfile as sf

from backend.core.dsp import pre_echo_model as pem

BASE = "/media/michael/Software 4TB/Aurik_Standalone/tmp_repro/witness_debug"


def analyze(phase: str) -> None:
    b_raw, sr = sf.read(f"{BASE}/{phase}_before.wav", dtype="float32", always_2d=True)
    a_raw, _ = sf.read(f"{BASE}/{phase}_after.wav", dtype="float32", always_2d=True)
    b = b_raw.mean(axis=1)
    a = a_raw.mean(axis=1)
    n = len(a)
    hop = max(1, int(sr * pem._ENV_WIN_S))
    n_frames = max(1, (n - hop) // hop + 1)
    idx = np.arange(hop)[None, :] + hop * np.arange(n_frames)[:, None]
    env_a = np.sqrt(np.mean(a[idx] ** 2, axis=1)) + 1e-12
    env_b = np.sqrt(np.mean(b[idx] ** 2, axis=1)) + 1e-12
    delta = (a - b).astype(np.float32)
    env_d_pos = np.sqrt(np.mean(np.maximum(delta[idx], 0.0) ** 2, axis=1)) + 1e-18

    _loc_win = max(4, int(pem._LOCAL_WIN_S / pem._ENV_WIN_S))
    _floor = float(np.max(env_a)) * 10.0 ** (-60.0 / 20.0)
    _loc_base = np.zeros(n_frames, dtype=np.float64)
    _w_len = 2 * _loc_win + 1
    _w = np.lib.stride_tricks.sliding_window_view(env_a, _w_len)
    _loc_base[_loc_win : n_frames - _loc_win] = np.percentile(_w, pem._LOCAL_PERC, axis=1)
    for f in list(range(0, _loc_win)) + list(range(n_frames - _loc_win, n_frames)):
        _lf = max(0, f - _loc_win)
        _hf = min(n_frames, f + _loc_win + 1)
        _loc_base[f] = max(float(np.percentile(env_a[_lf:_hf], pem._LOCAL_PERC)), _floor)
    np.maximum(_loc_base, _floor, out=_loc_base)
    rise = env_a / (_loc_base + 1e-12)
    onset_frames = np.where(rise > pem._ONSET_RISE)[0]

    pre_frames = max(1, int(pem._PRE_WIN_S / pem._ENV_WIN_S))
    gap_frames = max(1, int(pem._GAP_S / pem._ENV_WIN_S))
    _mask_e = 10.0 ** (-pem._FWD_MASK_DB / 10.0)
    _abs_floor_e = (float(np.max(env_a)) * 10.0 ** (-60.0 / 20.0)) ** 2

    rows = []
    for _of in onset_frames:
        _pre_lo = max(0, _of - pre_frames)
        _pre_hi = max(0, _of - gap_frames)
        if _pre_hi <= _pre_lo or _of + 1 >= n_frames:
            continue
        _shift_s = pem._local_shift_samples(b, a, sr, _of * hop + hop // 2)
        if _shift_s:
            _added = pem._aligned_delta_pre_energy(b, a, _pre_lo * hop, _pre_hi * hop, int(_shift_s))
        else:
            _added = float(np.mean(env_d_pos[_pre_lo:_pre_hi] ** 2))
        if _added < float(env_a[_of] ** 2) * _mask_e or _added < _abs_floor_e:
            continue
        _pre_sig_e = float(np.mean(env_a[_pre_lo:_pre_hi] ** 2)) + 1e-18
        if _added < _pre_sig_e * _mask_e:
            continue
        ratio_db = 10.0 * np.log10(_added / max(float(env_a[_of] ** 2), 1e-18))
        rows.append((ratio_db, int(_of), _added, float(env_a[_of] ** 2), int(_shift_s or 0)))

    rows.sort(key=lambda r: -r[0])
    print(f"== {phase} — {len(rows)} Befund-Onsets, Top 3: ==")
    for ratio_db, _of, added, onset_e, shift in rows[:3]:
        c_ms = _of * hop / sr * 1000.0
        print(f"  ratio={ratio_db:.2f} dB  onset_frame={_of} ({c_ms:.0f} ms)  shift={shift}")
        lo, hi = _of - 6, _of + 4
        lo = max(0, lo)
        hi = min(n_frames, hi)
        print(f"    Frames {lo}..{hi} (je 5ms) — a=envelope, b=vorher, d+=pos. Delta:")
        for f in range(lo, hi):
            mark = " <<<" if f == _of else (" [PRE]" if lo <= f <= _of - gap_frames and f >= _of - pre_frames else "")
            print(
                f"      f{f:>6}: a={env_a[f]:.3e} b={env_b[f]:.3e} d+={env_d_pos[f]:.3e}"
                f" a/b={20 * np.log10(env_a[f] / max(env_b[f], 1e-12)):+.2f}dB{mark}"
            )
        # Peak-Positionen ±50ms um Onset
        c = _of * hop + hop // 2
        lo_s = max(0, c - int(0.05 * sr))
        hi_s = min(n, c + int(0.05 * sr))
        for name, sig in (("a", a), ("b", b), ("delta", a - b)):
            pk = int(np.argmax(np.abs(sig[lo_s:hi_s])))
            print(f"    abs-peak {name}: {pk - int(0.05 * sr)} samples vs onset-center")


for ph in ("phase_01_click_removal", "phase_24_dropout_repair"):
    analyze(ph)
    print()
