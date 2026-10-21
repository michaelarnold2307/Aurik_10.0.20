"""Datenpipeline (v4): zufaellige Stereo-Ausschnitte -> Vinyl-Degradation -> (optional) DSP-Reparatur.

v4-Zusaetze:
  * `pattern`-Filter fuer die Dateiauswahl (z.B. "mixture.wav" bei MUSDB18-HQ, Default: alles)
"""
import glob
import math
import random

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from dsp import repair
from simulate import add_vinyl, perturb_mask

SR = 44100


def find_audio(folder, pattern=None):
    fs = []
    for e in ("wav", "flac", "mp3", "ogg"):
        fs += glob.glob(f"{folder}/**/*.{e}", recursive=True)
    if pattern:
        fs = [f for f in fs if pattern in f]
    return sorted(fs)


def load_clip(f, seg, rnd):
    """Zufaelliger Stereo-Ausschnitt (2, seg) bei 44.1 kHz."""
    info = sf.info(f)
    need = int(math.ceil(seg * info.samplerate / SR))
    for _ in range(3):
        start = rnd.randint(0, max(0, info.frames - need))
        x, fs = sf.read(f, start=start, frames=need, dtype="float32", always_2d=True)
        if np.sqrt((x ** 2).mean()) > 2e-3:
            break
    if x.shape[1] == 1:
        x = np.repeat(x, 2, 1)
    x = x[:, :2]
    if fs != SR:
        g = math.gcd(SR, fs)
        x = resample_poly(x, SR // g, fs // g, axis=0).astype(np.float32)
    x = x[:seg]
    if len(x) < seg:
        x = np.pad(x, ((0, seg - len(x)), (0, 0)))
    return x.T.copy()


class ClipDataset(Dataset):
    """mode='restore': (rohes bzw. repariertes Signal, sauber, Klickmaske, rohes Signal)
       mode='detect' : (rohes Signal, wahre Klickmaske)"""

    def __init__(self, files, n, seg, bank, dsp_prob=0.5, train=True, mode="restore", sev_max=1.0):
        self.files, self.n, self.seg, self.bank = files, n, seg, bank
        self.dsp_prob, self.train, self.mode, self.sev_max = dsp_prob, train, mode, sev_max

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self.train:
            rnd, rng = random.Random(), np.random.default_rng()
        else:
            rnd, rng = random.Random(1000 + i), np.random.default_rng(1000 + i)
        x = load_clip(self.files[rnd.randrange(len(self.files))], self.seg, rnd)
        x = x * (10 ** rnd.uniform(-1.0, 0.0) / (np.abs(x).max() + 1e-6))  # Pegel-Jitter
        sev = rng.uniform(0.15, max(0.16, self.sev_max))
        noisy, mask = add_vinyl(x, SR, rng, severity=sev, noise_bank=self.bank)
        if self.mode == "detect":
            return torch.from_numpy(noisy), torch.from_numpy(mask)
        pm = perturb_mask(mask, rng)  # unvollkommener Detektor
        fixed = repair(noisy, pm) if rnd.random() < self.dsp_prob else noisy
        return (torch.from_numpy(fixed), torch.from_numpy(x), torch.from_numpy(pm.astype(np.float32)),
                torch.from_numpy(noisy))
