"""Stereo-Vinyl-Degradationssimulator (v4).

Realismus-Features:
  * Stereo: Klicks/Crackle mit zufaelliger Panorama-/Phasenlage (laterale vs. vertikale Rillenbewegung)
  * Absolute Pegel: Rauschen ist unabhaengig vom Musikpegel (wie bei echten Platten)
  * Nadel-Impulsantwort + zufaellige spektrale Faerbung (Tonabnehmer/Phono-Stufe)
  * Kratzer-Bursts, tieffrequente Pops (Thumps), farbiges Rillenrauschen, Rumble
  * Optionale Noise-Bank: echte Leerrillen-Aufnahmen werden zugemischt
v4-Zusaetze (SOTA):
  * Print-Through: schwaches Vorecho der benachbarten Rille (typisches Vinyl-Artefakt)
  * Resonante Klick-Variante mit Ring-down (Nadel-Sprung auf defekter Stelle)
Gibt (noisy, click_mask) zurueck; beide (2, T).
"""
import glob
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter, resample_poly, sosfilt


def load_noise_bank(folder, sr=44100):
    """Laedt echte Rillenrausch-Aufnahmen (ohne Musik) als Liste von (2, N)-Arrays."""
    bank = []
    files = glob.glob(f"{folder}/**/*.wav", recursive=True) + glob.glob(f"{folder}/**/*.flac", recursive=True)
    for f in files:
        x, fs = sf.read(f, dtype="float32", always_2d=True)
        if x.shape[1] == 1:
            x = np.repeat(x, 2, 1)
        x = x[:, :2]
        if fs != sr:
            g = gcd(sr, fs)
            x = resample_poly(x, sr // g, fs // g, axis=0).astype(np.float32)
        if len(x) > sr:
            bank.append(x.T.copy())
    return bank


def _colored(T, beta, rng):
    """Rauschen mit 1/f^beta-Spektrum, Standardabweichung 1."""
    spec = np.fft.rfft(rng.standard_normal(T))
    f = np.maximum(np.arange(len(spec)), 1).astype(np.float64)
    y = np.fft.irfft(spec * f ** (-beta / 2), T)
    return (y / (y.std() + 1e-12)).astype(np.float32)


def _needle_kernel(rng):
    n = int(rng.integers(6, 28))
    t = np.arange(n)
    k = np.exp(-t / rng.uniform(1.5, 6.0)) * np.cos(2 * np.pi * t * rng.uniform(0.02, 0.25) + rng.uniform(0, 6.28))
    k[0] += rng.uniform(0.5, 1.5)
    return (k / np.abs(k).max()).astype(np.float32)


def _click_kernel(sr, rng):
    """Klick-Kern. v4: mit gewisser Wahrscheinlichkeit resonante Variante mit Ring-down."""
    length = int(sr * rng.uniform(0.0003, 0.004))
    t = np.arange(length) / sr
    if rng.random() < 0.35:
        # Resonant (Nadel-Sprung auf defekter Stelle): hoher Q, langer Ring-down
        f0 = rng.uniform(800, 6000)
        q = rng.uniform(4, 20)
        tau = q / (np.pi * f0)
        k = np.exp(-t / max(tau, 1e-5)) * np.sin(2 * np.pi * f0 * t + rng.uniform(0, 6.28))
    else:
        # Breitbandiger Klick mit schneller Decay
        k = np.exp(-rng.uniform(800, 4000) * t) * np.sin(2 * np.pi * rng.uniform(1500, 9000) * t + rng.uniform(0, 6.28))
    if rng.random() < 0.5:
        k[0] += rng.choice([-1, 1]) * rng.uniform(0.5, 1.5)
    return (k / (np.abs(k).max() + 1e-9)).astype(np.float32)


def _print_through(clean, rng, sr=44100):
    """Print-Through: schwaches Vorecho der benachbarten Rille (typisches Vinyl-Artefakt)."""
    T = clean.shape[1]
    delay_ms = float(rng.uniform(80, 350))
    d = int(delay_ms / 1000.0 * sr)
    gain = rng.uniform(0.02, 0.12)
    out = np.zeros_like(clean)
    for c in range(clean.shape[0]):
        x = clean[c]
        # verzögerte, gedämpfte Kopie addieren
        y = np.zeros(T, np.float32)
        if d < T:
            y[d:] = gain * x[:T - d]
        out[c] = y
    return out


def _pan(rng, n):
    """Kanalgewichte; ~35 % antiphasig (vertikale Rillenbewegung)."""
    th = rng.uniform(0, np.pi / 2, n)
    sign = np.where(rng.random(n) < 0.35, -1.0, 1.0)
    return np.cos(th) * 1.41, np.sin(th) * sign * 1.41


def _add_click(noise, mask, pos, k, gl, gr):
    """Addiert einen Klick; die Maske markiert nur den signifikanten Teil (nicht den abklingenden Schwanz)."""
    T = noise.shape[1]
    end = min(T, pos + len(k))
    L = end - pos
    for c, g in enumerate((gl, gr)):
        seg = g * k[:L]
        noise[c, pos:end] += seg
        a = np.abs(seg)
        if a.max() > 0.02:  # nur signifikante Klicks: kleinere ueberlaesst man dem Netz (Reparatur lohnt nicht)
            idx = np.where(a > max(0.2 * a.max(), 0.01))[0]
            mask[c, pos + idx[0]: pos + idx[-1] + 1] = 1.0


def perturb_mask(mask, rng, sr=44100):
    """Emuliert einen unvollkommenen Detektor: Dilatation, verpasste Klicks, Fehlalarme. (2,T) -> bool (2,T)."""
    out = np.zeros(mask.shape, bool)
    T = mask.shape[1]
    for c in range(mask.shape[0]):
        d = np.diff(np.concatenate([[0], (mask[c] > 0).astype(np.int8), [0]]))
        for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            if rng.random() < 0.05:
                continue
            g = int(rng.integers(0, 7))
            out[c, max(0, s - g // 2): min(T, e + g)] = True
        for _ in range(rng.poisson(0.4 * T / sr)):
            s = int(rng.integers(0, T))
            out[c, s:s + int(rng.integers(5, 40))] = True
    return out


def _color(noise, rng):
    """Zufaellige Faerbung der Impulsanteile (Tonabnehmer-/Phono-Frequenzgang)."""
    a = rng.uniform(0.0, 0.7)
    wet = rng.uniform(0.0, 1.0)
    for c in range(noise.shape[0]):
        lp = lfilter([1 - a], [1, -a], noise[c])
        noise[c] = ((1 - wet) * noise[c] + wet * lp).astype(np.float32)
    return noise


def add_vinyl(clean, sr=44100, rng=None, severity=None, noise_bank=None):
    rng = rng or np.random.default_rng()
    assert clean.ndim == 2 and clean.shape[0] == 2, "erwartet (2, T)"
    T = clean.shape[1]
    secs = T / sr
    sev = severity if severity is not None else rng.uniform(0.15, 1.0)
    level = sev * 10 ** rng.uniform(-2.0, -0.9)  # absolut (Vollaussteuerung = 1), unabhaengig von der Musik
    noise = np.zeros((2, T), np.float32)
    mask = np.zeros((2, T), np.float32)

    # 1. Einzelklicks
    n_clicks = rng.poisson(rng.uniform(2, 30) * sev * secs)
    if n_clicks:
        gl, gr = _pan(rng, n_clicks)
        for n, pos in enumerate(rng.integers(0, T, n_clicks)):
            k = _click_kernel(sr, rng) * level * float(np.clip(rng.lognormal(0, 0.8), 0.1, 6)) * rng.choice([-1, 1])
            _add_click(noise, mask, int(pos), k, gl[n], gr[n])

    # 2. Kratzer-Burst
    if rng.random() < 0.15:
        p0 = int(rng.integers(0, T))
        m = int(rng.integers(5, 40))
        gl, gr = _pan(rng, m)
        for n, pos in enumerate(p0 + rng.integers(0, int(sr * 0.05), m)):
            if pos < T:
                k = _click_kernel(sr, rng) * level * rng.uniform(0.3, 3.0) * rng.choice([-1, 1])
                _add_click(noise, mask, int(pos), k, gl[n], gr[n])

    # 3. Pop / Thump (tieffrequent, nicht per AR detektierbar -> Aufgabe des Netzes)
    if rng.random() < 0.3:
        dur = int(sr * rng.uniform(0.05, 0.2))
        t = np.arange(dur) / sr
        th = np.exp(-t * rng.uniform(15, 50)) * np.sin(2 * np.pi * rng.uniform(15, 70) * t)
        th = (th * level * rng.uniform(1, 5)).astype(np.float32)
        pos = int(rng.integers(0, max(T - dur, 1)))
        gl, gr = _pan(rng, 1)
        end = min(T, pos + dur)
        noise[0, pos:end] += gl[0] * th[: end - pos]
        noise[1, pos:end] += gr[0] * th[: end - pos]

    # 4. Dauerknistern (teilweise stereo-korreliert)
    n_cr = rng.poisson(rng.uniform(300, 5000) * sev * secs)
    if n_cr:
        idx = rng.integers(0, T, n_cr)
        amp = rng.laplace(0, level * 0.12, n_cr)
        common = rng.random(n_cr) < rng.uniform(0.1, 0.7)
        gl, gr = _pan(rng, n_cr)
        ch = rng.integers(0, 2, n_cr)
        l = np.where(common, gl, (ch == 0) * 1.0)
        r = np.where(common, gr, (ch == 1) * 1.0)
        cr = np.zeros((2, T), np.float32)
        np.add.at(cr[0], idx, (amp * l).astype(np.float32))
        np.add.at(cr[1], idx, (amp * r).astype(np.float32))
        for c in range(2):
            cr[c] = np.convolve(cr[c], _needle_kernel(rng), "same")
        noise += cr
    noise = _color(noise, rng)

    # 5. Rillenrauschen (farbig, teilkorreliert) + Rumble
    hiss = 10 ** rng.uniform(-3.8, -2.4) * (0.3 + sev)
    common_n = _colored(T, rng.uniform(0.3, 1.2), rng)
    for c in range(2):
        g = rng.uniform(0.3, 0.8)
        ind = _colored(T, rng.uniform(0.3, 1.2), rng)
        noise[c] += hiss * (g * common_n + np.sqrt(1 - g * g) * ind)
    sos = butter(2, 40, "low", fs=sr, output="sos")
    rum = sosfilt(sos, rng.standard_normal(T))
    rum = (rum / (rum.std() + 1e-12) * 10 ** rng.uniform(-3.5, -2.0) * sev).astype(np.float32)
    noise += rum[None] * np.array([[1.0], [rng.uniform(0.5, 1.0)]], np.float32)

    # 5b. Print-Through (v4): schwaches Vorecho der benachbarten Rille
    if rng.random() < 0.3:
        noise += _print_through(clean, rng).astype(np.float32)

    # 6. Echte Rillenrausch-Aufnahmen zumischen
    if noise_bank and rng.random() < 0.7:
        nb = noise_bank[int(rng.integers(len(noise_bank)))]
        N = nb.shape[1]
        seg = np.zeros((2, T), np.float32)
        pos = 0
        while pos < T:
            s = int(rng.integers(0, max(N - 1, 1)))
            take = min(T - pos, N - s)
            seg[:, pos:pos + take] = nb[:, s:s + take]
            pos += take
        noise += seg * rng.uniform(0.5, 2.0)

    noisy = np.clip(clean + noise, -1.0, 1.0).astype(np.float32)
    return noisy, mask
