"""Inferenz v4:  python infer.py platte.wav sauber.wav --ckpt runs/vinylnet/best_infer.pt \
                    --detector runs/detector/detector.pt --strength 0.9

Pipeline: Klickdetektion (gelernt/AR/beides) -> Hybrid-Reparatur (InpaintNet fuer grosse Luecken,
rohes Signal bleibt bei kleinen — v3-Befund: AR-Fehler uebersteigt kleine Klicks selbst)
-> VinylNet (sieht repariert + roh + Maske)."""
import argparse
import math

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from dsp import detect_clicks, repair
from model import ClickNet, VinylNet

SR = 44100
CH, OV = SR * 6, SR  # Chunk 6 s, Ueberlappung 1 s (lineare Kreuzblende)


def load_model(path, dev):
    ck = torch.load(path, map_location=dev)
    m = VinylNet(**ck["config"]).to(dev)
    m.load_state_dict(ck.get("ema", ck.get("model")))
    return m.eval()


def load_detector(path, dev):
    ck = torch.load(path, map_location=dev)
    cfg = dict(ck["config"])
    # v3-Checkpointe kennen feat_ch nicht — Default 8 passt zu v4; bei Abweichung strict-Fail sichtbar
    m = ClickNet(**cfg).to(dev)
    m.load_state_dict(ck["model"], strict=True)
    return m.eval()


@torch.inference_mode()
def detect_learned(x, det, dev):
    """Klickwahrscheinlichkeit (2,T); chunkweise mit Kontext."""
    T = x.shape[1]
    out = np.zeros((2, T), np.float32)
    CHD, CTX = SR * 20, 4096
    for s in range(0, T, CHD):
        e = min(T, s + CHD)
        lo, hi = max(0, s - CTX), min(T, e + CTX)
        p = torch.sigmoid(det(torch.from_numpy(x[:, lo:hi])[None].to(dev)))[0].cpu().numpy()
        out[:, s:e] = p[:, s - lo: s - lo + (e - s)]
    return out


def get_mask(x, detector=None, dev="cpu", mode="auto", det_thr=0.5, ar_thr=6.0, grow=2, link=False):
    """Klickmaske (2,T) bool. mode: auto (gelernt falls Detektor, sonst AR) | learned | ar | both."""
    if mode == "auto":
        mode = "learned" if detector is not None else "ar"
    if mode in ("learned", "both") and detector is None:
        raise ValueError("--det-mode verlangt --detector")
    m = None
    if mode in ("learned", "both"):
        m = detect_learned(x, detector, dev) > det_thr
    if mode in ("ar", "both"):
        am = np.stack([detect_clicks(x[c].astype(np.float64), ar_thr) for c in range(x.shape[0])])
        m = am if m is None else (m | am)
    if grow > 0:
        k = np.ones(2 * grow + 1, np.uint8)
        m = np.stack([np.convolve(m[c].astype(np.uint8), k, "same") > 0 for c in range(m.shape[0])])
    if link:
        m = np.repeat(m.any(0, keepdims=True), m.shape[0], 0)
    return m


@torch.inference_mode()
def inpaint_gaps(x, mask, model, dev, min_gap=96, ctx=4096):
    """Hybrid-Reparatur (v4): Luecken >= min_gap per InpaintNet (mit Kontext-Padding),
    kleinere bleiben roh — das Restaurierungsnetz sieht beides und entscheidet."""
    if not mask.any():
        return x.copy()
    y = x.astype(np.float32).copy()
    T = x.shape[1]
    for c in range(x.shape[0]):
        d = np.diff(np.concatenate([[0], mask[c].astype(np.int8), [0]]))
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
        big = [(i, j) for i, j in zip(starts, ends) if (j - i) >= min_gap]
        if not big:
            continue
        lo = max(0, min(i for i, _ in big) - ctx)
        hi = min(T, max(j for _, j in big) + ctx)
        seg_m = mask[:, lo:hi].astype(np.float32)[None]
        seg_x = x[:, lo:hi][None]
        filled = model.inpaint_fill(torch.from_numpy(seg_x).to(dev), torch.from_numpy(seg_m).to(dev))[0].cpu().numpy()
        for i, j in big:
            y[c, i:j] = filled[c, i - lo: j - lo]
    return y


@torch.inference_mode()
def restore(x, model, dev, strength=1.0, mask=None, repair_first=True):
    """x: (2,T) float32 bei 44.1 kHz; mask: bool (2,T) -> (2,T)."""
    x = x.astype(np.float32)
    T0 = x.shape[1]
    if mask is None:
        mask = np.zeros(x.shape, bool)
    raw = x
    if repair_first and mask.any():
        x = inpaint_gaps(x, mask, model, dev)
    pad = max(0, 8192 - T0)
    x, raw = np.pad(x, ((0, 0), (0, pad))), np.pad(raw, ((0, 0), (0, pad)))
    mk = np.pad(mask.astype(np.float32), ((0, 0), (0, pad)))
    T = x.shape[1]
    out = np.zeros_like(x)
    s = 0
    while True:
        e = min(T, s + CH)
        n = e - s
        t = lambda a: torch.from_numpy(a[:, s:e])[None].to(dev)
        y = model(t(x), strength, click_mask=t(mk), x_raw=t(raw))[0].cpu().numpy()
        w = np.ones(n, np.float32)
        r = min(OV, n)
        if s > 0:
            w[:r] = np.linspace(0, 1, r, dtype=np.float32)
        if e < T:
            w[-r:] *= np.linspace(1, 0, r, dtype=np.float32)
        out[:, s:e] += y * w
        if e >= T:
            break
        s = e - OV
    return out[:, :T0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--detector", default=None, help="Pfad zu detector.pt (gelernter Klickdetektor)")
    ap.add_argument("--det-mode", default="auto", choices=["auto", "learned", "ar", "both"])
    ap.add_argument("--det-threshold", type=float, default=0.5)
    ap.add_argument("--click-threshold", type=float, default=6.0, help="AR-Detektor-Schwelle")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--inpaint-min-gap", type=int, default=96, help="Mindest-Lueckenlaenge fuer InpaintNet (v4)")
    ap.add_argument("--no-inpaint", action="store_true", help="InpaintNet aus (nur VinylNet auf Rohsignal + Maske)")
    ap.add_argument("--link-channels", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    model = load_model(a.ckpt, a.device)
    det = load_detector(a.detector, a.device) if a.detector else None
    x, fs = sf.read(a.inp, dtype="float32", always_2d=True)
    x = x.T
    mono = x.shape[0] == 1
    if mono:
        x = np.repeat(x, 2, 0)
    x = x[:2]
    if fs != SR:
        g = math.gcd(SR, fs)
        x = resample_poly(x, SR // g, fs // g, axis=1).astype(np.float32)
    mask = get_mask(x, det, a.device, a.det_mode, a.det_threshold, a.click_threshold, link=a.link_channels)
    if not a.no_inpaint:
        x_rep = inpaint_gaps(x, mask, model, a.device, min_gap=a.inpaint_min_gap)
    else:
        x_rep = x
    y = restore(x_rep, model, a.device, a.strength, mask, repair_first=False)
    if fs != SR:
        g = math.gcd(SR, fs)
        y = resample_poly(y, fs // g, SR // g, axis=1).astype(np.float32)
    if mono:
        y = y[:1]
    sf.write(a.out, np.clip(y.T, -1, 1), fs, subtype="PCM_24")


if __name__ == "__main__":
    main()
