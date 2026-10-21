"""Training VinylNet v4 (Restaurierungsnetz + InpaintNet): AMP bf16, EMA, Curriculum, GAN-Phase,
Validierung (SI-SDR + optional MERT-Zeuge), Resume.

Reihenfolge: 1) train_detector.py (gelernter Klickdetektor)  2) dieses Skript.
Das Netz wird mit unvollkommenen Masken (perturb_mask) trainiert und sieht rohes + repariertes Signal.
InpaintNet lernt im selben Lauf, grosse Luecken zu fuellen (ersetzt AR-Interpolation bei grossen Klicks).

  python train.py --data ./clean_music --noise-bank ./groove_noise --preset base --epochs 200 --gan

v4-Zusaetze:
  * --seed / --threads fuer reproduzierbare Laeufe (§G5)
  * --pattern Dateifilter (z.B. mixture.wav bei MUSDB18-HQ)
  * Log-Mel-Loss (perzeptuelle Gewichtung im Gradienten, --mel-weight)
  * InpaintNet-Training (--inpaint-weight, Default an)
  * MERT-Zeuge in der Validierung (--mert <onnx>): perzeptuelle Kosinus-Similaritaet zur sauberen
    Referenz; beste Checkpoints werden dann komposit (SI-SDR + MERT-Delta) ausgewaehlt —
    die Metrik ist Zeuge, nicht Entscheider im Loss (Hörordnung: Berechnung bleibt bei Domain-Regeln)
"""
import argparse
import copy
import math
import os
import random
import time

import numpy as np
import torch
from scipy.signal import resample_poly
from torch.utils.data import DataLoader

from data import SR, ClipDataset, find_audio
from losses import inpaint_loss, recon_loss, si_sdr
from model import MRDisc, PRESETS, VinylNet
from simulate import load_noise_bank


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for s, b in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(b)


class MertWitness:
    """MERT-ONNX als perzeptueller Zeuge (Validierung/Best-Auswahl), kein Gradient."""

    def __init__(self, path, sr_in=44100, sr_model=24000):
        import onnxruntime as ort

        self.sr_in, self.sr_model = sr_in, sr_model
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def _embed(self, x):  # (T,) float -> globales Embedding (D,): Mean-Pooling ueber Batch UND Zeitachse.
        # Das ONNX liefert last_hidden_state (1, frames, D) — ohne Mean-Pooling ueber
        # die Zeitachse bleibt eine (frames, D)-Matrix statt eines Embedding-Vektors.
        v = self.sess.run(None, {self.inp: (x * 0.5).astype(np.float32)[None]})[0]
        return np.asarray(v, dtype=np.float64).mean(axis=(0, 1))

    @torch.no_grad()
    def sim(self, est, ref):  # (B,C,T) float -> mittlere Kosinus-Similaritaet Embeddings
        out = []
        for i in range(est.shape[0]):
            e = est[i].mean(0).cpu().numpy()
            r = ref[i].mean(0).cpu().numpy()
            if self.sr_model != self.sr_in:
                e = resample_poly(e, self.sr_model, self.sr_in)
                r = resample_poly(r, self.sr_model, self.sr_in)
            fe = self._embed(e)
            fr = self._embed(r)
            out.append(float(fe @ fr / ((np.linalg.norm(fe) * np.linalg.norm(fr)) + 1e-9)))
        return sum(out) / max(len(out), 1)


@torch.no_grad()
def validate(model, dl, dev, use_amp, mert=None):
    model.eval()
    tot = dict(sdr_in=0.0, sdr_out=0.0, loss=0.0, m_in=0.0, m_out=0.0)
    n = 0
    for fixed, clean, mask, raw in dl:
        fixed, clean, mask, raw = fixed.to(dev), clean.to(dev), mask.to(dev), raw.to(dev)
        with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=use_amp):
            est = model(fixed, click_mask=mask, x_raw=raw)
        est = est.float()
        b = fixed.shape[0]
        tot["sdr_in"] += si_sdr(raw, clean).sum().item()
        tot["sdr_out"] += si_sdr(est, clean).sum().item()
        tot["loss"] += recon_loss(est, clean, mask).item() * b
        if mert is not None:
            tot["m_in"] += mert.sim(raw.float(), clean)
            tot["m_out"] += mert.sim(est, clean)
        n += b
    return {k: v / max(n, 1) for k, v in tot.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--noise-bank", default=None, help="Ordner mit echten Leerrillen-Aufnahmen")
    ap.add_argument("--pattern", default=None, help="Datei-Filter (z.B. mixture.wav)")
    ap.add_argument("--preset", default="small", choices=list(PRESETS))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--epoch-size", type=int, default=2000)
    ap.add_argument("--val-size", type=int, default=64)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seg-seconds", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--dsp-prob", type=float, default=0.7, help="Anteil der Samples mit DSP-Reparatur")
    ap.add_argument("--curriculum-epochs", type=int, default=10, help="Schweregrad steigt von 0.4 auf 1.0")
    ap.add_argument("--mel-weight", type=float, default=0.15, help="Log-Mel-Loss-Gewicht (v4)")
    ap.add_argument("--inpaint-weight", type=float, default=1.0, help="InpaintNet-Training (v4); 0 = aus")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=0, help="torch-Threads (0 = Auto)")
    ap.add_argument("--mert", default=None, help="Pfad zu MERT-ONNX als perzeptueller Zeuge (v4)")
    ap.add_argument("--gan", action="store_true")
    ap.add_argument("--gan-start", type=int, default=20000)
    ap.add_argument("--lambda-adv", type=float, default=0.1)
    ap.add_argument("--lambda-fm", type=float, default=0.5)
    ap.add_argument("--d-ch", type=int, default=32)
    ap.add_argument("--gan-sub", type=int, default=2, help="Diskriminator-Sub-Batch-Faktor (VRAM-Effizienz in der GAN-Phase)")
    ap.add_argument("--mem-fraction", type=float, default=0.0,
                    help="Torch-VRAM-Cap als Anteil der GPU (geteilte GPU mit LM Studio); 0 = aus")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default="runs/vinylnet")
    ap.add_argument("--resume", default=None)
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    seed_everything(a.seed)
    if a.threads > 0:
        torch.set_num_threads(a.threads)
    dev = a.device
    use_amp = (not a.no_amp) and dev == "cuda" and torch.cuda.is_bf16_supported()
    if a.mem_fraction > 0.0 and dev == "cuda":
        # Geteilte GPU (z.B. mit LM Studio): harter VRAM-Cap, damit ein OOM nur diesen Prozess trifft
        torch.cuda.set_per_process_memory_fraction(a.mem_fraction)
    seg = int(a.seg_seconds * SR)

    files = find_audio(a.data, a.pattern)
    assert files, "Keine Audiodateien gefunden"
    random.Random(0).shuffle(files)
    nval = max(1, int(0.05 * len(files)))
    val_files, train_files = files[:nval], (files[nval:] or files)
    bank = load_noise_bank(a.noise_bank) if a.noise_bank else []
    mert = MertWitness(a.mert) if a.mert else None
    print(f"{len(train_files)} Train-, {len(val_files)} Val-Dateien, Noise-Bank: {len(bank)}, MERT-Zeuge: {bool(mert)}")

    cfg = dict(PRESETS[a.preset])
    model = VinylNet(**cfg).to(dev)
    print(f"Parameter: {model.n_params / 1e6:.2f} M | AMP bf16: {use_amp} | seed {a.seed}")
    ema = EMA(model, a.ema_decay)
    opt_g = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01)
    spe = max(1, a.epoch_size // a.bs)
    total = a.epochs * spe
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt_g, lambda s: min(1.0, (s + 1) / a.warmup) * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    )
    D = opt_d = None
    if a.gan:
        D = MRDisc(a.d_ch).to(dev)
        opt_d = torch.optim.AdamW(D.parameters(), lr=2e-4, betas=(0.8, 0.99))

    step, start_ep, best = 0, 0, -1e9
    if a.resume:
        ck = torch.load(a.resume, map_location=dev)
        model.load_state_dict(ck["model"])
        ema.shadow.load_state_dict(ck["ema"])
        opt_g.load_state_dict(ck["opt_g"])
        sched.load_state_dict(ck["sched"])
        if D is not None and "D" in ck:
            D.load_state_dict(ck["D"])
            opt_d.load_state_dict(ck["opt_d"])
        step, start_ep, best = ck["step"], ck["epoch"] + 1, ck.get("best", best)
        print(f"Resume ab Epoche {start_ep}, Schritt {step}")

    train_ds = ClipDataset(train_files, a.epoch_size, seg, bank, a.dsp_prob, True)
    val_ds = ClipDataset(val_files, a.val_size, seg, bank, a.dsp_prob, False)
    train_dl = DataLoader(train_ds, batch_size=a.bs, num_workers=a.workers, drop_last=True)  # nicht persistent: Curriculum
    val_dl = DataLoader(val_ds, batch_size=a.bs, num_workers=a.workers)

    for ep in range(start_ep, a.epochs):
        train_ds.sev_max = min(1.0, 0.4 + 0.6 * ep / max(1, a.curriculum_epochs))
        model.train()
        t0, acc, na = time.time(), 0.0, 0
        for fixed, clean, mask, raw in train_dl:
            fixed, clean, mask, raw = fixed.to(dev), clean.to(dev), mask.to(dev), raw.to(dev)
            with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=use_amp):
                est = model(fixed, click_mask=mask, x_raw=raw)
                if a.inpaint_weight > 0:  # InpaintNet-Aufgabe (v4): grosse Luecken generativ fuellen
                    ip = model.inpaint_fill(raw, mask).float()
            est = est.float()
            loss = rec = recon_loss(est, clean, mask, mel_weight=a.mel_weight)
            if a.inpaint_weight > 0:
                flat_m = mask.float().reshape(mask.shape[0] * mask.shape[1], -1)
                loss = rec + a.inpaint_weight * inpaint_loss(ip.reshape(-1), clean.reshape(-1), flat_m)
            gan_on = D is not None and step >= a.gan_start
            gh = max(1, est.shape[0] // max(1, a.gan_sub))  # Diskriminator-Sub-Batch (VRAM-Effizienz)
            if gan_on:
                with torch.autocast(device_type=dev, enabled=False):
                    fake = D(est[:gh])
                    with torch.no_grad():
                        real = D(clean[:gh])
                    adv = sum(((lg - 1) ** 2).mean() for lg, _ in fake) / len(fake)
                    fm = sum(sum((ff - fr).abs().mean() for ff, fr in zip(fk[1], rl[1])) / len(fk[1])
                              for fk, rl in zip(fake, real)) / len(fake)
                loss = rec + a.lambda_adv * adv + a.lambda_fm * fm
            opt_g.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_g.step()
            sched.step()
            ema.update(model)
            if gan_on:
                with torch.autocast(device_type=dev, enabled=False):
                    r_out, f_out = D(clean[:gh]), D(est.detach()[:gh])
                    d_loss = sum(((r[0] - 1) ** 2).mean() + (f[0] ** 2).mean() for r, f in zip(r_out, f_out)) / len(r_out)
                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_d.step()
            acc += rec.item()
            na += 1
            step += 1
        v = validate(ema.shadow, val_dl, dev, use_amp, mert=mert)
        extra = ""
        if mert is not None:
            extra = f" | MERT sim {v['m_in']:.3f} -> {v['m_out']:.3f}"
        print(f"ep {ep:03d} step {step} sev<= {train_ds.sev_max:.2f} train {acc / max(na, 1):.4f} | val loss {v['loss']:.4f} "
              f"SI-SDR {v['sdr_in']:.2f} -> {v['sdr_out']:.2f} dB (+{v['sdr_out'] - v['sdr_in']:.2f}){extra} | {time.time() - t0:.0f}s"
              + (" | GAN" if gan_on else ""))
        score = v["sdr_out"] + (10.0 * (v["m_out"] - v["m_in"]) if mert is not None else 0.0)
        ck = dict(config=cfg, model=model.state_dict(), ema=ema.shadow.state_dict(), opt_g=opt_g.state_dict(),
                  sched=sched.state_dict(), step=step, epoch=ep, best=best)
        if D is not None:
            ck.update(D=D.state_dict(), opt_d=opt_d.state_dict())
        if score > best:
            best = ck["best"] = score
            torch.save(dict(config=cfg, ema=ema.shadow.state_dict(), val=v), os.path.join(a.out_dir, "best_infer.pt"))
        torch.save(ck, os.path.join(a.out_dir, "last.pt"))
    torch.save(dict(config=cfg, ema=ema.shadow.state_dict()), os.path.join(a.out_dir, "final_infer.pt"))
    print("Fertig.")


if __name__ == "__main__":
    main()
