"""Air-Presence-Enhancer — Brillianz im Luftband (8–20 kHz) ohne Phasenartefakte.

§v10.19-Paket (2026-09-12): Gesangs-Klarheit/Brillianz — die vierte Stufe
neben MIIPHER-DiT (NR), KIM2 (Klarheit) und dem Witness-Gate. Reine DSP-Lösung
(STFT + Original-Phasen-Rekonstruktion), deterministisch, kein Modell.

Garantien (§0 Primum non nocere):
- ``strength=0.0`` ⇒ bit-identischer Passthrough (kein Gain, keine Filterung).
- Gain wird NUR angewendet, wo das Luftband Energie oberhalb des Noise-Floors
  hat — Rauschen wird nie angehoben.
- Raised-Cosine-Bandkanten statt harter Bandgrenzen (Soft-Knee-Prinzip, §III
  (copilot-instructions.md)); Phasen bleiben unverändert (Original-Phase-STFT).

Layout: channels-first ``(C, N)`` für Stereo bzw. ``(N,)`` für Mono — Ein- und
Ausgabe identisch.
"""

from __future__ import annotations

import numpy as np


def _stft_frames(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """STFT-Frames (rfft) mit Hann-Fenster, deterministisch."""
    win = np.hanning(n_fft).astype(np.float32)
    n_frames = max(0, (len(x) - n_fft) // hop + 1)
    x = np.asarray(x, dtype=np.float32)
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    _spec: np.ndarray = np.fft.rfft(x[idx] * win, n=n_fft, axis=1).astype(np.complex64)
    return _spec


def _istft_frames(spec: np.ndarray, n_fft: int, hop: int, orig_len: int) -> np.ndarray:
    """Overlap-Add-Rekonstruktion (scipy.signal.istft, Hann) — deterministisch."""
    from scipy.signal import istft as _istft

    # scipy-Konvention: Frames sind mit win/Σwin gefenstert — unsere STFT
    # nutzt win allein, daher hier um Σwin dividieren (Einheits-Gain-Rekonstruktion).
    _win_sum = float(np.hanning(n_fft).sum())
    _spec = (np.asarray(spec, dtype=np.complex64) / max(_win_sum, 1e-12)).T  # (n_freq, n_frames)
    _, _rec = _istft(
        _spec,
        fs=1.0,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop,
        nfft=n_fft,
        boundary=False,
        input_onesided=True,
    )
    _out: np.ndarray = np.asarray(_rec, dtype=np.float32)
    if len(_out) < orig_len:
        _out = np.pad(_out, (0, orig_len - len(_out)))
    return _out[:orig_len]


def enhance_air_presence(
    audio: np.ndarray,
    sr: int = 48000,
    strength: float = 0.15,
    air_lo_hz: float = 8000.0,
    air_hi_hz: float = 18000.0,
    noise_floor_db: float = -80.0,
) -> np.ndarray:
    """Hebt das Luftband (Brillianz) an — harmlos, deterministisch.

    Args:
        audio:          float32, channels-first (C, N) oder mono (N,).
        sr:             Sample-Rate (Hz).
        strength:       Gain-Stärke im Luftband [0, 1]; 0.0 = Passthrough.
        air_lo_hz:      Untere Luftband-Grenze (Raised-Cosine-Kante).
        air_hi_hz:      Obere Luftband-Grenze (auf Nyquist begrenzt).
        noise_floor_db: Schwelle, unterhalb derer kein Gain erfolgt.

    Returns:
        Audio im identischen Layout/Dtype wie der Input.
    """
    _in = np.asarray(audio, dtype=np.float32)
    if strength <= 0.0:
        return audio  # bit-identischer Passthrough (§0)

    _strength = float(np.clip(strength, 0.0, 1.0))
    _max_gain_db = 6.0 * _strength  # Soft-Knee-Deckel (§III): max 6 dB bei s=1
    n_fft, hop = 2048, 512
    nyq = sr / 2.0
    _lo = float(np.clip(air_lo_hz, 1000.0, nyq * 0.9))
    _hi = float(np.clip(air_hi_hz, _lo, nyq))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr).astype(np.float32)

    # Raised-Cosine-Bandkanten (Soft-Knee, keine harten Grenzen)
    edge_hz = min(1500.0, (_hi - _lo) * 0.5)
    lo_edge = np.clip((freqs - (_lo - edge_hz)) / max(edge_hz, 1e-3), 0.0, 1.0)
    hi_edge = np.clip(((_hi + edge_hz) - freqs) / max(edge_hz, 1e-3), 0.0, 1.0)
    band_ramp = (0.5 - 0.5 * np.cos(np.pi * lo_edge)) * (0.5 - 0.5 * np.cos(np.pi * hi_edge))
    gain_lin = 10.0 ** ((_max_gain_db * band_ramp) / 20.0)

    # Noise-Floor in der Spektraldomäne: |X[k]| ~ RMS * sqrt(Σ win²) für weißes Rauschen.
    _win = np.hanning(n_fft).astype(np.float32)
    noise_ref = 10.0 ** (float(noise_floor_db) / 20.0) * float(np.sqrt(np.sum(_win**2)))

    _channels = [_in] if _in.ndim == 1 else [_in[c] for c in range(_in.shape[0])]
    _out_ch: list[np.ndarray] = []
    for _ch in _channels:
        _spec = _stft_frames(_ch, n_fft, hop)
        _mag = np.abs(_spec)
        # Maske: nur wo Luftband-Energie über dem Noise-Floor liegt
        _mask = (_mag * band_ramp[None, :]) > noise_ref
        _mask = _mask.astype(np.float32)
        # weiche Maske über Frames glätten (kein Zittern)
        if _mask.shape[0] > 2:
            _mask[1:-1] = 0.5 * _mask[:-2] + 0.5 * _mask[2:]
        _spec_out = _spec * (1.0 + (gain_lin[None, :] - 1.0) * _mask)
        _out_ch.append(_istft_frames(_spec_out, n_fft, hop, len(_ch)))

    _out = np.stack(_out_ch, axis=0) if _in.ndim == 2 else _out_ch[0]
    _out = np.nan_to_num(_out, nan=0.0, posinf=0.0, neginf=0.0)
    _out = np.clip(_out, -1.0, 1.0).astype(np.float32)
    _result: np.ndarray = np.asarray(_out, dtype=np.float32)
    return _result
