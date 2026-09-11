#!/usr/bin/env python3
"""§P1-4b (2026-09-09): BS-RoFormer-317 → Core-only-ONNX-Export + End-to-End-Parität.

Architektur-Match verifiziert: bs-roformer==0.3.10 lädt den 317-Checkpoint
strict mit missing=0/unexpected=0 (dim=512, depth=12, stereo, 62 Bänder,
Summe 1025 Bins, time/freq-Tiefe 1, stft n_fft=2048/hop=512).

Export-Strategie (wie MelBandRoformer-ONNX): STFT und ISTFT bleiben
AUSSERHALB des Graphen (NumPy/scipy, deterministisch, §G5 (GEBOTE.md)) — exportiert
wird der Core: view_as_real-STFT (b,2,f,t,2) → Band-Split → Axial-
Transformer → Masken (b,1,2050,t,2) real/imag. Der Legacy-Exporter
(dynamo=False) verarbeitet einops korrekt; der einzige bekannte Blocker
(STFT-komplex) wird durch die Core-only-Grenze umgangen.

Usage:
    .venv_aurik/bin/python scripts/export_bs_roformer_onnx.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CKPT = ROOT / "models" / "bs_roformer" / "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
OUT = ROOT / "models" / "bs_roformer" / "bs_roformer_317_core.onnx"

import torch
import torch.nn as nn
from einops import pack, rearrange, unpack


class BSRCore(nn.Module):
    """Core: (b, s=2, f=1025, t, c=2) → Masken (b, 1, f*2, t, c=2) real/imag."""

    def __init__(self, model):
        super().__init__()
        self.band_split = model.band_split
        self.layers = model.layers
        self.mask_estimators = model.mask_estimators

    def forward(self, x):
        # entspricht BSRoformer.forward ab view_as_real (Schritte 2–6, inkl.
        # pack/unpack-Achsen-Flattening der axialen Transformer)
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


def _build_models():
    from bs_roformer import BSRoformer  # pylint: disable=import-outside-toplevel

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
    assert not res.missing_keys and not res.unexpected_keys, (
        f"MISMATCH missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}"
    )
    full.eval()
    return full, BSRCore(full).eval()


def main() -> int:
    import numpy as np  # pylint: disable=import-outside-toplevel
    import onnxruntime as ort  # pylint: disable=import-outside-toplevel

    if OUT.exists():
        print(f"existiert bereits: {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
        return 0

    full, core = _build_models()
    dummy = torch.zeros((1, 2, 1025, 64, 2), dtype=torch.float32)  # b s f t c
    with torch.no_grad():
        torch.onnx.export(
            core,
            dummy,
            str(OUT),
            input_names=["stft_view_real"],
            output_names=["masks"],
            dynamic_axes={"stft_view_real": {0: "batch", 3: "time"}, "masks": {0: "batch", 3: "time"}},
            opset_version=17,
            dynamo=False,
            do_constant_folding=True,
        )
    print(f"Export OK: {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")

    # ── Core-Parität: BSRCore(torch) vs. ONNX auf identischem STFT-Eingang ──
    rng = np.random.default_rng(7)
    z_ri = rng.standard_normal((1, 2, 1025, 64, 2)).astype(np.float32)
    with torch.no_grad():
        mask_torch = core(torch.from_numpy(z_ri)).numpy()
    sess = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
    mask_onnx = sess.run(None, {"stft_view_real": z_ri})[0]
    diff = float(np.abs(mask_torch - mask_onnx).max())
    rel = float(np.abs(mask_torch - mask_onnx).mean() / (np.abs(mask_torch).mean() + 1e-9))
    print(f"Core-Parität: max_abs={diff:.3e} rel_mean={rel:.3e} (Toleranz 1e-3)")
    return 0 if diff < 1e-3 else 1


if __name__ == "__main__":
    sys.exit(main())
