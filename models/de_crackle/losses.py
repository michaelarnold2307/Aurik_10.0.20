"""Verluste und Metriken (v4).

v4-Zusaetze:
  * log_mel_loss — mehrkanaliges Log-Mel-L1 (perzeptuelle Gewichtung, billiger Ersatz
    fuer vortrainierte Encoder im Trainingsloop; der MERT-Encoder laeuft als Zeuge in
    der Validierung statt im Gradienten)
  * inpaint_loss — relative L1 nur auf maskierten Samples (InpaintNet-Aufgabe)
"""
import torch
import torch.nn.functional as F

RES = (256, 512, 1024, 2048)
MEL_RES = (512, 2048)


def _mag(x, n):
    w = torch.hann_window(n, device=x.device)
    return torch.stft(x, n, n // 4, window=w, return_complex=True).abs()


def _mel_bands(sr: int, n_mels: int):
    """Mel-Bandgrenzen (Hz), von 40 Hz bis sr/2."""

    def hz2mel(f):
        return 2595.0 * torch.log10(1.0 + f / 700.0)

    mels = torch.linspace(hz2mel(torch.tensor(40.0)), hz2mel(torch.tensor(float(sr / 2))), n_mels + 2)
    return (700.0 * (torch.pow(10.0, mels / 2595.0) - 1.0))[1:-1]


def _band_matrix(bands: torch.Tensor, sr: int, nfreq: int, device) -> torch.Tensor:
    """Gewichtsmatrix W (nfreq, n_bands+1): Zeile f enthält 1/|Band| im zugeordneten Mel-Band."""
    bin_freq = torch.linspace(0.0, sr / 2, nfreq, device=device)
    idx = torch.bucketize(bin_freq, bands.to(device))  # (nfreq,) in [0, len(bands)]
    counts = torch.bincount(idx, minlength=len(bands) + 1).clamp_min(1)
    W = torch.zeros(nfreq, len(bands) + 1, device=device)
    W[torch.arange(nfreq), idx] = 1.0 / counts[idx]
    return W


def log_mel_loss(y: torch.Tensor, t: torch.Tensor, sr: int = 44100, n_mels: int = 64) -> torch.Tensor:
    """Multi-Resolution Log-Mel L1 (perzeptuell gewichtet). y,t: (B,C,T)."""
    B, C, T = y.shape
    yy, tt = y.reshape(B * C, T), t.reshape(B * C, T)
    loss = 0.0
    bands = _mel_bands(sr, n_mels)
    for n in MEL_RES:
        w = torch.hann_window(n, device=y.device)
        Ym = torch.stft(yy.float(), n, n // 4, window=w, return_complex=True).abs()  # (BC,nfreq,frames)
        Tm = torch.stft(tt.float(), n, n // 4, window=w, return_complex=True).abs()
        W = _band_matrix(bands, sr, Ym.shape[1], y.device)  # (nfreq, n_bands+1)
        ly = torch.stack([W.T @ Ym[b] for b in range(Ym.shape[0])]).clamp_min(1e-5).log()
        lt = torch.stack([W.T @ Tm[b] for b in range(Tm.shape[0])]).clamp_min(1e-5).log()
        loss = loss + F.l1_loss(ly, lt)
    return loss / len(MEL_RES)


def recon_loss(y: torch.Tensor, t: torch.Tensor, mask=None, mel_weight: float = 0.2):
    """Multi-Res-STFT + relative Waveform-L1 (an Klickstellen hoeher gewichtet) + Side-Kanal-L1
    (+ optionales Log-Mel, v4). y,t: (B,2,T) float32; mask: optionale (B,2,T) Klickmaske."""
    B, C, T = y.shape
    yy, tt = y.reshape(B * C, T), t.reshape(B * C, T)
    mr = 0.0
    for n in RES:
        Y, Tm = _mag(yy, n), _mag(tt, n)
        mr = mr + torch.norm(Tm - Y) / (torch.norm(Tm) + 1e-8) + F.l1_loss(torch.log(Y + 1e-5), torch.log(Tm + 1e-5))
    mr = mr / len(RES)
    ref = t.abs().mean() + 1e-4
    err = (y - t).abs()
    if mask is not None:
        w = 1 + 4 * F.max_pool1d(mask.float(), 33, 1, 16)
        err = err * w
    l1 = err.mean() / ref
    side = F.l1_loss(y[:, 0] - y[:, 1], t[:, 0] - t[:, 1]) / ref
    out = mr + 0.2 * l1 + 0.2 * side
    if mel_weight > 0:
        out = out + mel_weight * log_mel_loss(y, t)
    return out


def inpaint_loss(est_gap: torch.Tensor, clean_gap: torch.Tensor, mask_flat: torch.Tensor) -> torch.Tensor:
    """Relative L1 auf den maskierten Samples. est/clean/mask werden intern geplatet (N,)."""
    est_gap = est_gap.reshape(-1)
    clean_gap = clean_gap.reshape(-1)
    m = mask_flat.reshape(-1) > 0.5
    if not m.any():
        return torch.zeros((), device=est.device)
    ref = clean_gap[m].abs().mean() + 1e-4
    return ((est_gap - clean_gap)[m].abs().mean()) / ref


def si_sdr(est, ref):
    """SI-SDR in dB, gemittelt ueber Kanaele. est, ref: (B,C,T) -> (B,)"""
    est = est - est.mean(-1, keepdim=True)
    ref = ref - ref.mean(-1, keepdim=True)
    alpha = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + 1e-8)
    proj = alpha * ref
    noise = est - proj
    v = 10 * torch.log10(proj.pow(2).sum(-1) / (noise.pow(2).sum(-1) + 1e-10) + 1e-10)
    return v.mean(-1)


def detector_loss(logits, mask):
    """BCE (positiv gewichtet) + Dice."""
    bce = F.binary_cross_entropy_with_logits(logits, mask, pos_weight=torch.tensor(8.0, device=logits.device))
    p = torch.sigmoid(logits)
    dice = 1 - (2 * (p * mask).sum() + 1) / (p.sum() + mask.sum() + 1)
    return bce + dice


def detector_counts(logits, mask, thr=0.5):
    pred = torch.sigmoid(logits) > thr
    m = mask > 0.5
    return (pred & m).sum().item(), (pred & ~m).sum().item(), (~pred & m).sum().item()


def f1_from_counts(tp, fp, fn):
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)
