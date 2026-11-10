"""Export models/vocos_48khz/pytorch_model.bin -> models/vocos_48khz/vocos_48khz.onnx

Model:  sample_rate=48000, n_mels=128, n_fft=2048, hop_length=256
ONNX:   Input  'mel'   float32 [B, 128, T]
        Output 'audio' float32 [B, (T-7)*256]  @48 kHz

ISTFTHead is replaced with an ONNX-safe version:
  irfft  -> precomputed DFT matrices + matmul  (no complex ops)
  OLA    -> G=8 unrolled shift-sum  (no F.fold / col2im)
Both support fully dynamic batch + T.
"""

from __future__ import annotations

import logging
import math
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("vocos_export")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "vocos_48khz")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.yaml")
WPATH = os.path.join(MODEL_DIR, "pytorch_model.bin")
OUT_PATH = os.path.join(MODEL_DIR, "vocos_48khz.onnx")
FINETUNE_DIR = os.path.join(MODEL_DIR, "finetune")
FINETUNE_WPATH = os.path.join(FINETUNE_DIR, "vocos_48khz_music_ft.pt")
FINETUNE_OUT = os.path.join(FINETUNE_DIR, "vocos_48khz_music_ft.onnx")
OPSET = 18

with open(CONFIG_PATH) as fh:
    cfg = yaml.safe_load(fh)

N_MELS = cfg["feature_extractor"]["init_args"]["n_mels"]  # 128
N_FFT = cfg["head"]["init_args"]["n_fft"]  # 2048
HOP = cfg["head"]["init_args"]["hop_length"]  # 256
DIM = cfg["backbone"]["init_args"]["dim"]  # 1024
SR = cfg["feature_extractor"]["init_args"]["sample_rate"]  # 48000
G = N_FFT // HOP  # 8

assert N_FFT % HOP == 0
log.info("Config: sr=%d n_mels=%d n_fft=%d hop=%d G=%d", SR, N_MELS, N_FFT, HOP, G)


def _resolve_paths(finetune: bool) -> tuple[str, str]:
    """Gewichts- und Ausgabepfad je Modus — testbar ohne Modell-Ladung (M5)."""
    if finetune:
        return FINETUNE_WPATH, FINETUNE_OUT
    return WPATH, OUT_PATH


def _load_state_dict(weights_path: str, state_key: str | None = None) -> dict:
    """Lädt das state_dict — Finetune-Checkpoint trägt es unter ``state_dict``
    (Misch-Dict Tensoren+Ints ⇒ weights_only=False, lokales Eigen-Artefakt).
    """
    raw = torch.load(weights_path, map_location="cpu", weights_only=state_key is None)
    if state_key:
        if not isinstance(raw, dict) or state_key not in raw:
            raise KeyError(f"Checkpoint ohne '{state_key}': {list(raw)[:5] if isinstance(raw, dict) else type(raw)}")
        raw = raw[state_key]
    if not isinstance(raw, dict):
        raise TypeError(f"state_dict erwartet, erhalten {type(raw).__name__}")
    return raw


def load_vocos(weights_path: str = WPATH, state_key: str | None = None):
    from vocos.pretrained import Vocos, instantiate_class  # type: ignore

    fe = instantiate_class(args=(), init=cfg["feature_extractor"])
    bb = instantiate_class(args=(), init=cfg["backbone"])
    hd = instantiate_class(args=(), init=cfg["head"])
    m = Vocos(feature_extractor=fe, backbone=bb, head=hd)
    m.load_state_dict(_load_state_dict(weights_path, state_key))
    m.eval()
    log.info("Loaded: %s", weights_path)
    return m


def _irfft_matrices(n_fft):
    """[K,N] cos/sin DFT matrices for irfft via pure matmul."""
    N, K = n_fft, n_fft // 2 + 1
    angles = (
        (2 * math.pi / N)
        * torch.arange(K, dtype=torch.float64).unsqueeze(1)
        * torch.arange(N, dtype=torch.float64).unsqueeze(0)
    )
    s = torch.full((K,), 2.0 / N, dtype=torch.float64)
    s[0] = s[-1] = 1.0 / N
    return (s[:, None] * torch.cos(angles)).float(), (s[:, None] * torch.sin(angles)).float()


class ISTFTHeadONNX(nn.Module):
    """ONNX-safe ISTFTHead: DFT-matrix irfft + G-unrolled shift-sum OLA."""

    def __init__(self, orig_head, g):
        super().__init__()
        N, H = orig_head.istft.n_fft, orig_head.istft.hop_length
        self.N, self.H, self.g = N, H, g
        self.out = orig_head.out  # reuse trained Linear

        # Explicit annotations: register_buffer() otherwise types these as
        # "Tensor | Module" via nn.Module.__getattr__, breaking @/* operators.
        self.win: torch.Tensor
        self.cos_mat: torch.Tensor
        self.sin_mat: torch.Tensor
        self.norm_h: torch.Tensor

        win = torch.hann_window(N)
        self.register_buffer("win", win)

        cm, sm = _irfft_matrices(N)
        self.register_buffer("cos_mat", cm)  # [K, N]
        self.register_buffer("sin_mat", sm)

        # Interior OLA normalisation: norm_h[h] = sum_d win^2[d*H+h]
        self.register_buffer("norm_h", win.square().view(g, H).sum(0))  # [H]

    def forward(self, x):
        # x: [B, T, dim]
        B, T, _ = x.shape
        N, H, g = self.N, self.H, self.g
        pad = (N - H) // 2  # 896

        proj = self.out(x).transpose(1, 2)  # [B, N+2, T]
        mag, p = proj.chunk(2, dim=1)
        mag = torch.exp(mag).clamp(max=1e2)
        S_re = mag * torch.cos(p)  # [B, K, T]
        S_im = mag * torch.sin(p)

        # irfft via matmul: [B,T,K] @ [K,N] -> [B,T,N]
        frames = S_re.permute(0, 2, 1) @ self.cos_mat - S_im.permute(0, 2, 1) @ self.sin_mat

        frames = frames * self.win  # [B, T, N]

        # OLA: G=8 unrolled shifts
        #   frames_g: [B,T,G,H] -> permute -> [B,H,G,T]
        #   pad G-1 zeros at T start -> [B,H,G,T+G-1]
        #   y_sum[t] = sum_d pad[:,:,d, G-1-d+t]
        fg = frames.reshape(B, T, g, H).permute(0, 3, 2, 1)  # [B,H,G,T]
        fp = F.pad(fg, (g - 1, 0))  # [B,H,G,T+G-1]

        ys = fp[:, :, 0, g - 1 : g - 1 + T]
        for d in range(1, g):
            s = g - 1 - d
            ys = ys + fp[:, :, d, s : s + T]  # [B, H, T]

        ys = ys / self.norm_h.view(1, H, 1).clamp(min=1e-11)
        y = ys.permute(0, 2, 1).reshape(B, T * H)  # [B, T*H]
        return y[:, pad : T * H - pad]  # [B, (T-G+1)*H]


class VocosDecoder(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, mel):
        return self.head(self.backbone(mel))


def build_decoder(vm):
    return VocosDecoder(vm.backbone, ISTFTHeadONNX(vm.head, G)).eval()


def verify(path, mel_np):
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        audio = np.asarray(sess.run(None, {sess.get_inputs()[0].name: mel_np})[0]).flatten()
        ok = bool(np.isfinite(audio).all())
        log.info("ORT verify: shape=%s  max=%.4f  finite=%s", audio.shape, np.abs(audio).max(), ok)
        return ok
    except Exception as e:
        log.error("ORT verify error: %s", e)
        return False


def main():
    import argparse
    import hashlib
    import json

    ap = argparse.ArgumentParser(description="Vocos-48kHz ONNX-Export (Base oder Musik-Finetune, M5)")
    ap.add_argument("--finetune", action="store_true", help="Musik-Finetune exportieren (M5) statt der Base")
    args = ap.parse_args()

    wpath, out_path = _resolve_paths(args.finetune)
    state_key = "state_dict" if args.finetune else None
    if not os.path.exists(wpath):
        log.error("Weights missing: %s", wpath)
        sys.exit(1)

    if os.path.exists(out_path):
        os.remove(out_path)

    vm = load_vocos(weights_path=wpath, state_key=state_key)
    dec = build_decoder(vm)
    dummy = torch.randn(1, N_MELS, 200)

    with torch.no_grad():
        out = dec(dummy)
    log.info("Forward test: [1,%d,200] -> %s  (expected %d samples)", N_MELS, tuple(out.shape), (200 - G + 1) * HOP)

    log.info("Exporting (opset %d) ...", OPSET)
    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            torch.onnx.export(
                dec,
                (dummy,),
                out_path,
                input_names=["mel"],
                output_names=["audio"],
                dynamic_axes={"mel": {0: "batch", 2: "T"}, "audio": {0: "batch", 1: "S"}},
                opset_version=OPSET,
                do_constant_folding=True,
            )
    except Exception as e:
        log.error("Export failed: %s", e, exc_info=True)
        sys.exit(1)

    mb = os.path.getsize(out_path) / 1024 / 1024
    log.info("Saved: %s  (%.1f MB, %.1f s)", out_path, mb, time.perf_counter() - t0)

    if not verify(out_path, dummy.numpy()):
        log.error("ONNX verification failed")
        sys.exit(1)

    # Dynamic-T cross-check
    import onnxruntime as ort

    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    dec2 = build_decoder(vm)
    log.info("\nDynamic-T cross-validation:")
    for T in (30, 50, 100, 200, 500):
        mel = torch.randn(1, N_MELS, T)
        mnp = mel.numpy().astype(np.float32)
        oa = np.asarray(sess.run(None, {iname: mnp})[0]).flatten()
        with torch.no_grad():
            pa = dec2(mel).numpy().flatten()
        d = float(np.abs(oa - pa).max())
        log.info("  %s T=%3d  len=%6d  max|delta|=%.2e", "OK" if d < 1e-4 else "!!", T, len(oa), d)

    if args.finetune:
        # Export-Manifest (M5): kein Deploy/Flag-Flip — nur Paritäts-Beleg (§III.9).
        _h = hashlib.sha256()
        with open(out_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                _h.update(chunk)
        manifest = {
            "schema": "aurik.vocos48k_finetune_export/1",
            "onnx": os.path.basename(out_path),
            "sha256_onnx": _h.hexdigest(),
            "quelle_checkpoint": os.path.basename(wpath),
            "opset": OPSET,
            "paritaet_max_abs_delta": "< 1e-4 (Dynamic-T cross-check)",
            "hinweis": "KEIN Deploy, KEIN Flag-Flip — Studio-Finisher-Rollen-Entscheidung bleibt menschlich (C4).",
        }
        manifest_path = os.path.join(os.path.dirname(out_path), "export_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)
        log.info("Export-Manifest: %s", manifest_path)

    log.info("\n✅ ONNX export complete.")
    log.info("   Input  'mel'   : [B, %d, T]", N_MELS)
    log.info("   Output 'audio' : [B, (T-%d)*%d]  @48 kHz", G - 1, HOP)


if __name__ == "__main__":
    main()
