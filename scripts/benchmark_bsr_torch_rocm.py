#!/usr/bin/env python3
"""§SOTA-BSR-GPU — PyTorch-ROCm vs. ONNX-CPU Paritäts- und Speed-Validierung.

Lädt den 317-Checkpoint (Rezept aus scripts/export_bs_roformer_onnx.py),
führt den Core auf der ROCm-GPU aus und vergleicht numerisch + zeitlich
mit der ONNX-CPU-Referenz (bs_roformer_317_core.onnx).

Usage:
  .venv_aurik/bin/python scripts/benchmark_bsr_torch_rocm.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CKPT = ROOT / "models" / "bs_roformer" / "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
ONNX = ROOT / "models" / "bs_roformer" / "bs_roformer_317_core.onnx"

import torch
import torch.nn as nn
from einops import pack, rearrange, unpack


class BSRCore(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.band_split = model.band_split
        self.layers = model.layers
        self.mask_estimators = model.mask_estimators

    def forward(self, x):
        x = rearrange(x, "b s f t c -> b (f s) t c")
        x = rearrange(x, "b f t c -> b t (f c)")
        x = self.band_split(x)
        for time_transformer, freq_transformer in self.layers:
            x = rearrange(x, "b t f d -> b f t d")
            x, ps = pack([x], "* t d")
            x = time_transformer(x)
            x = unpack(x, ps, "* t d")[0]
            x = rearrange(x, "b f t d -> b t f d")
            x, ps = pack([x], "* f d")
            x = freq_transformer(x)
            x = unpack(x, ps, "* f d")[0]
        mask = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
        mask = rearrange(mask, "b n t (f c) -> b n f t c", c=2)
        return mask


def build_core():
    from bs_roformer import BSRoformer

    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    freqs: list[int] = []
    i = 0
    while f"band_split.to_features.{i}.1.weight" in sd:
        freqs.append(int(sd[f"band_split.to_features.{i}.1.weight"].shape[1] // 4))
        i += 1
    full = BSRoformer(
        512,
        depth=12,
        stereo=True,
        num_stems=1,
        freqs_per_bands=tuple(freqs),
        dim_freqs_in=sum(freqs),
        time_transformer_depth=1,
        freq_transformer_depth=1,
        stft_n_fft=2048,
        stft_hop_length=512,
        stft_win_length=2048,
    )
    res = full.load_state_dict(sd, strict=False)
    assert not res.missing_keys and not res.unexpected_keys
    full.eval()
    return BSRCore(full).eval()


def main() -> int:
    import onnxruntime as ort  # pylint: disable=import-outside-toplevel

    dev = torch.device("cuda")
    print(f"Device: {dev} ({torch.cuda.get_device_name(0)}), hip={torch.version.hip}")

    core = build_core().to(dev)
    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(7)

    def run_parity(t_frames: int):
        z = rng.standard_normal((1, 2, 1025, t_frames, 2)).astype(np.float32)
        with torch.no_grad():
            mask_t = core(torch.from_numpy(z).to(dev)).cpu().numpy()
        mask_o = sess.run(None, {"stft_view_real": z})[0]
        max_abs = float(np.abs(mask_t - mask_o).max())
        scale = max(float(np.abs(mask_o).max()), 1e-9)
        rel = float(np.max(np.abs(mask_t - mask_o)) / scale)
        return max_abs, rel

    for t_frames in (64, 512):
        max_abs, rel = run_parity(t_frames)
        status = "PARITY_OK" if max_abs < 1e-3 else "PARITY_FAIL"
        print(f"t={t_frames}: {status} max_abs={max_abs:.3e} rel={rel:.3e}")

    # ── Benchmark (warmup + 5 Läufe) ──
    z512 = rng.standard_normal((1, 2, 1025, 512, 2)).astype(np.float32)
    t_t = torch.from_numpy(z512).to(dev)
    with torch.no_grad():
        core(t_t)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            core(t_t)
        torch.cuda.synchronize()
        gpu_ms = (time.perf_counter() - t0) / 5 * 1000
    t0 = time.perf_counter()
    for _ in range(3):
        sess.run(None, {"stft_view_real": z512})
    cpu_ms = (time.perf_counter() - t0) / 3 * 1000
    print(
        f"Bench t=512: torch-ROCm {gpu_ms:.1f} ms | ONNX-CPU {cpu_ms:.1f} ms | Speedup {cpu_ms / max(gpu_ms, 1e-9):.1f}x"
    )

    # ── End-to-End-Stem-Parität: STFT(2048/512) → Core → Maske → ISTFT ──
    from scipy.signal import istft, stft  # pylint: disable=import-outside-toplevel

    audio = (0.05 * rng.standard_normal((2, 48000 * 4))).astype(np.float32)
    win = np.hanning(2048)

    def stems_via(core_fn):
        z_list = []
        for ch in range(2):
            _f, _t, zc = stft(
                audio[ch], fs=48000, nperseg=2048, noverlap=1536, window=win, boundary="zeros", padded=False
            )
            z_list.append(np.stack([zc.real, zc.imag], axis=-1))  # (1025, T, 2)
        zri = np.stack(z_list, axis=0)[None].astype(np.float32)  # (1, s=2, 1025, T, 2)
        mask = core_fn(zri)  # (1, 1, 1025, T, 2)
        mc = mask[0, 0, ..., 0] + 1j * mask[0, 0, ..., 1]  # (1025, T) komplexe Maske
        outs = []
        for ch in range(2):
            _f, _t, zc = stft(
                audio[ch], fs=48000, nperseg=2048, noverlap=1536, window=win, boundary="zeros", padded=False
            )
            _t2, v = istft(zc * mc, fs=48000, nperseg=2048, noverlap=1536, window=win, boundary=False)
            outs.append(v[: len(audio[ch])])
        return np.stack(outs)

    def via_onnx(zri):
        return sess.run(None, {"stft_view_real": zri})[0]

    def via_torch(zri):
        with torch.no_grad():
            return core(torch.from_numpy(zri).to(dev)).cpu().numpy()

    v_onnx = stems_via(via_onnx)
    v_torch = stems_via(via_torch)
    e2e_max = float(np.abs(v_onnx - v_torch).max())
    e2e_rel = float(np.max(np.abs(v_onnx - v_torch)) / max(float(np.max(np.abs(v_onnx))), 1e-9))
    print(
        f"End-to-End-Stems: {'PARITY_OK' if e2e_max < 1e-3 else 'PARITY_FAIL'} max_abs={e2e_max:.3e} rel={e2e_rel:.3e}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
