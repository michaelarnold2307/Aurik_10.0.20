"""DSP-Stufe (v3): AR-Klickdetektion (Fallback) + lokale LSAR-Interpolation.

v3-Aenderungen:
  * LSAR loest nur lokal (Luecke +- Ordnung); LPC wird aus grossem Kontext geschaetzt -> schnell und genau
  * Regularisierung der LPC-Schaetzung (stabiler bei tonalem Material)
  * repair(): Interpolation mit extern gelieferter Maske (z.B. vom gelernten Detektor)
"""
import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.ndimage import binary_closing
from scipy.signal import correlate, lfilter

ABS_FLOOR = 2e-4  # keine Klickdetektion in digitaler Stille


def lpc(x, order, reg=1e-4):
    """AR-Koeffizienten a (a[0]=1), so dass e[n] = sum_k a[k] x[n-k] minimal ist."""
    x = x - x.mean()
    r = correlate(x, x, mode="full", method="fft")[len(x) - 1: len(x) + order].copy()
    if r[0] < 1e-12:
        return np.concatenate([[1.0], np.zeros(order)])
    r[0] *= 1 + reg
    phi = solve_toeplitz(r[:order], r[1:order + 1])
    return np.concatenate([[1.0], -phi])


# ---------------------------------------------------------------- Detektion (AR, klassisch)
def _grow(hit, p):
    L = p // 2 + 4
    ext = np.convolve(hit.astype(np.uint8), np.ones(L, np.uint8))[: len(hit)] > 0
    ext[:-2] |= ext[2:]
    return ext


def _residual_hits(seg, a, thr, p):
    err = lfilter(a, [1.0], seg)
    err[:p] = 0
    sig = 1.4826 * np.median(np.abs(err[p:])) + 1e-9
    return (np.abs(err) > thr * sig) & (np.abs(err) > ABS_FLOOR)


def _block(seg, thr, p):
    w = np.hanning(len(seg))
    hit = _residual_hits(seg, lpc(seg * w, p), thr, p)
    if hit.any():  # Pass 2: LPC ohne Klickbereiche neu schaetzen
        g = _grow(hit, p)
        idx = np.arange(len(seg))
        seg2 = seg.copy()
        if (~g).sum() > 2:
            seg2[g] = np.interp(idx[g], idx[~g], seg[~g])
        hit = _residual_hits(seg, lpc(seg2 * w, p), thr, p)
    return _grow(hit, p)


def detect_clicks(x, thr=6.0, order=24, win=1024):
    n = len(x)
    mask = np.zeros(n, bool)
    if n < win:
        return mask
    starts = list(range(0, n - win + 1, win // 2))
    if starts[-1] + win < n:
        starts.append(n - win)
    for s in starts:
        mask[s:s + win] |= _block(x[s:s + win], thr, order)
    return mask


# ---------------------------------------------------------------- Interpolation
def _lsar_local(y, m, i, j, p, ctx, reg):
    """LSAR fuer Luecke [i,j). LPC aus Kontext +-ctx, Least-Squares nur lokal (+-p). None bei zu wenig Kontext."""
    n = len(y)
    lo, hi = max(0, i - ctx), min(n, j + ctx)
    cw, cm = y[lo:hi], m[lo:hi]
    if (~cm).sum() < 2 * p + 4:
        return None
    idx = np.arange(len(cw))
    base = cw.copy()
    base[cm] = np.interp(idx[cm], idx[~cm], cw[~cm])
    a = lpc(base * np.hanning(len(base)), p, reg)

    lo2, hi2 = max(0, i - p), min(n, j + p)
    w, wm = y[lo2:hi2], m[lo2:hi2]
    N = len(w)
    if N - p < wm.sum() + 1 or (~wm).sum() < p:
        return None
    rows = np.arange(p, N)
    A = np.zeros((N - p, N))
    A[np.arange(N - p)[:, None], rows[:, None] - np.arange(p + 1)[None, :]] = a[None, :]
    u, k = np.where(wm)[0], np.where(~wm)[0]
    # Start: lineare Interpolation; Korrektur delta per Ridge-Regression (verhindert instabile Loesungen)
    w0 = w.copy()
    wi = np.arange(N)
    w0[u] = np.interp(wi[u], wi[k], w[k])
    Au = A[:, u]
    G = Au.T @ Au
    lam = 1e-2 * np.trace(G) / max(len(u), 1) + 1e-12
    delta = np.linalg.solve(G + lam * np.eye(len(u)), -Au.T @ (A @ w0))
    lim = 1.5 * np.abs(w[k]).max() + 1e-3
    out = w.copy()
    out[u] = np.clip(w0[u] + delta, -lim, lim)
    return out[i - lo2: j - lo2]


def interpolate(x, mask, order=24, ctx=512, max_gap=1500, reg=1e-3):
    y = x.astype(np.float64).copy()
    n = len(y)
    pad = 16
    m = binary_closing(np.pad(mask.astype(bool), pad), structure=np.ones(9, bool))[pad:-pad]
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    for i, j in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        fixed = _lsar_local(y, m, i, j, order, ctx, reg) if (j - i) <= max_gap else None
        if fixed is None:
            y[i:j] = np.interp(np.arange(i, j), [i - 1, j], [y[max(i - 1, 0)], y[min(j, n - 1)]])
        else:
            y[i:j] = fixed
    return y.astype(x.dtype)


def repair(x, mask):
    """x (C,T), mask bool (C,T) -> repariertes Signal."""
    return np.stack([interpolate(x[c], mask[c]) for c in range(x.shape[0])]).astype(np.float32)


def process_stereo(x, thr=6.0, link=False):
    """Klassische AR-Pipeline (Detektion + Interpolation)."""
    masks = np.stack([detect_clicks(x[c].astype(np.float64), thr) for c in range(x.shape[0])])
    if link:
        masks[:] = masks.any(0)
    return repair(x, masks)
