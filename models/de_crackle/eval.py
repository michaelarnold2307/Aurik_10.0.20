"""Evaluation v4: SI-SDR/Loss je Variante und Schweregrad, Detektor-P/R/F1, Oracle-Obergrenzen
(LSAR + Inpaint), optionaler MERT-Zeuge (perzeptuelle Similaritaet), A/B-Export.

  python eval.py --ckpt vinylnet_base_best.pt --detector detector_v4.pt \
                 --data ./clean_val --mert ../../models/mert/mert_330m.onnx --export ab_out

 MUSDB18-HQ-Testsplit (Realkorpus, nur Mixturen als Referenz):
  python eval.py --ckpt vinylnet_base_best.pt --detector detector_v4.pt \
                 --data ../../data/musdb18hq/test --pattern mixture \
                 --mert ../../models/mert/mert_330m.onnx --export ab_out
"""
import argparse
import os
import random
from collections import defaultdict

import numpy as np
import soundfile as sf
import torch

from data import SR, find_audio, load_clip
from dsp import repair
from infer import get_mask, inpaint_gaps, load_detector, load_model, restore
from losses import f1_from_counts, recon_loss, si_sdr
from simulate import add_vinyl, load_noise_bank
from train import MertWitness

ORDER = [
    "unbearbeitet",
    "Reparatur AR-Detektor (LSAR)",
    "Reparatur gelernter Det. (LSAR)",
    "Reparatur Oracle-Maske (LSAR)",
    "InpaintNet + Maske (gelernt)",
    "VinylNet v4 (vollstaendig)",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--detector", default=None)
    ap.add_argument("--data", required=True)
    ap.add_argument("--pattern", default=None, help="Datei-Filter (z.B. mixture bei MUSDB18-HQ)")
    ap.add_argument("--noise-bank", default=None)
    ap.add_argument("--mert", default=None, help="MERT-ONNX als perzeptueller Zeuge")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--severities", type=float, nargs="+", default=[0.3, 0.6, 1.0])
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--export", default=None, help="Ordner fuer A/B-Wavs (erste 3 Clips je Schweregrad)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    dev = a.device
    model = load_model(a.ckpt, dev)
    det = load_detector(a.detector, dev) if a.detector else None
    files = find_audio(a.data, a.pattern)
    assert files, "Keine Audiodateien gefunden"
    bank = load_noise_bank(a.noise_bank) if a.noise_bank else []
    mert = MertWitness(a.mert) if a.mert else None
    if a.export:
        os.makedirs(a.export, exist_ok=True)
    res, cnt = defaultdict(list), defaultdict(lambda: np.zeros(3))
    seg = int(a.seconds * SR)
    for i in range(a.n):
        rnd = random.Random(i)
        x = load_clip(files[rnd.randrange(len(files))], seg, rnd)
        x = x * (0.5 / (np.abs(x).max() + 1e-6))
        ref = torch.from_numpy(x)[None]
        for sev in a.severities:
            noisy, tm = add_vinyl(x, SR, np.random.default_rng(i * 97 + int(sev * 100)), severity=sev, noise_bank=bank)
            tm = tm > 0
            m_ar = get_mask(noisy, None, dev, "ar")
            m_ln = get_mask(noisy, det, dev, "learned") if det is not None else None
            m_use = m_ln if m_ln is not None else m_ar
            outs = {
                "unbearbeitet": noisy,
                "Reparatur AR-Detektor (LSAR)": repair(noisy, m_ar),
                "Reparatur Oracle-Maske (LSAR)": repair(noisy, tm),
                "InpaintNet + Maske (gelernt)": inpaint_gaps(noisy, m_use, model, dev) if m_ln is not None else None,
                "VinylNet v4 (vollstaendig)": restore(noisy, model, dev, a.strength, m_use),
            }
            if m_ln is not None:
                outs["Reparatur gelernter Det. (LSAR)"] = repair(noisy, m_ln)
            for name, y in outs.items():
                if y is None:
                    continue
                yt = torch.from_numpy(y)[None]
                row = [si_sdr(yt, ref).item(), recon_loss(yt, ref).item()]
                if mert is not None:
                    row.append(mert.sim(yt, ref))
                res[(sev, name)].append(tuple(row))
            for name, m in (("AR", m_ar), ("gelernt", m_ln)):
                if m is not None:
                    cnt[(sev, name)] += [(m & tm).sum(), (m & ~tm).sum(), (~m & tm).sum()]
            if a.export and i < 3:
                for name, y in (("noisy", noisy), ("vinylnet", outs["VinylNet v4 (vollstaendig)"]), ("clean", x)):
                    sf.write(os.path.join(a.export, f"clip{i}_sev{sev}_{name}.wav"), np.clip(y.T, -1, 1), SR, subtype="PCM_24")
    mert_col = "MERT sim" if mert is not None else ""
    print(f"{'Schweregrad':<12}{'Variante':<30}{'SI-SDR [dB]':>12}{'Loss':>10}{mert_col:>12}")
    for sev in a.severities:
        for name in ORDER:
            if (sev, name) in res:
                arr = np.array(res[(sev, name)])
                line = f"{sev:<12}{name:<30}{arr[:, 0].mean():>12.2f}{arr[:, 1].mean():>10.3f}"
                if mert is not None:
                    line += f"{arr[:, 2].mean():>12.3f}"
                print(line)
    print("\nKlickdetektion vs. wahre Maske (Precision / Recall / F1):")
    for (sev, name), c in sorted(cnt.items()):
        p, r, f1 = f1_from_counts(*c)
        print(f"  sev {sev:<4} {name:<8} P {p:.3f}  R {r:.3f}  F1 {f1:.3f}")


if __name__ == "__main__":
    main()
