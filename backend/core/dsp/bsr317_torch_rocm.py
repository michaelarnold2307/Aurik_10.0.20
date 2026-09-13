"""§SOTA-BSR-GPU — BS-RoFormer-317 PyTorch-ROCm-Separation (GPU-Pfad).

Befund 2026-09-13 (scripts/benchmark_bsr_torch_rocm.py): ONNX-Runtime-ROCm-
Kernels rechnen das Modell falsch (rel 6,2 — Upstream-Bug); der PyTorch-ROCm-
Pfad ist numerisch paritätisch zur ONNX-CPU-Referenz (rel ~1e-5) und 46×
schneller (7900 XTX). Dieser Baustein lädt den 317-Checkpoint, führt den Core
auf ROCm aus und liefert vocal/instrumental-Stems (STFT/ISTFT 2048/512,
Hanning, Chunking 30 s wegen O(T²)-Attention).

Fail-closed (§V6 (copilot-instructions.md)): Ohne torch/ROCm/Checkpoint gibt
get_bsr317_torch_core() None zurück — der Aufrufer bleibt auf dem ONNX-CPU-Pfad.
Deterministisch (§G5 (copilot-instructions.md)): eval + no_grad, feste Parameter.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_NFFT = 2048
_HOP = 512
_WIN = 2048
_CHUNK_SAMPLES = 48000 * 30  # 30 s — Zeit-Attention ist O(T²)

_lock = threading.Lock()
_core = None
_core_resolved = False


def _build_core():
    import torch  # pylint: disable=import-outside-toplevel
    from bs_roformer import BSRoformer  # pylint: disable=import-outside-toplevel
    from einops import pack, rearrange, unpack  # pylint: disable=import-outside-toplevel

    class BSRCore(torch.nn.Module):
        """Core: (b, s=2, f=1025, t, c=2) → Maske (b, 1, 1025, t, 2)."""

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

    ckpt = Path(__file__).resolve().parents[3] / "models" / "bs_roformer" / "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
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
        stft_n_fft=_NFFT,
        stft_hop_length=_HOP,
        stft_win_length=_WIN,
    )
    res = full.load_state_dict(sd, strict=False)
    assert not res.missing_keys and not res.unexpected_keys, "317-Checkpoint-Mismatch"
    full.eval()
    return BSRCore(full).eval()


def get_bsr317_torch_core():
    """Lazy-Singleton: ROCm-Core oder None (fail-closed, §V6 (copilot-instructions.md))."""
    global _core, _core_resolved
    with _lock:
        if _core_resolved:
            return _core
        _core_resolved = True
        try:
            import torch  # pylint: disable=import-outside-toplevel

            if not torch.cuda.is_available():
                logger.debug("§BSR-GPU torch-ROCm nicht verfügbar — ONNX-Pfad bleibt")
                return None
            _core = _build_core().to("cuda")
            logger.info("§BSR-GPU 317-Core auf ROCm geladen (%s)", torch.cuda.get_device_name(0))
            return _core
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("§BSR-GPU torch-ROCm-Core nicht ladbar: %s — ONNX-Pfad bleibt", exc)
            return None


def separate_bsr317_torch(
    audio: np.ndarray,
    sr: int,
    core,
    stems: tuple[str, ...] = ("vocals", "instruments"),
) -> dict[str, np.ndarray]:
    """Trennt Mix in vocal/instrumental-Stems über den ROCm-Core.

    vocal = Maske × STFT; instrumental = Mix − vocal (num_stems=1-Modell).
    Ausgabe je Stem in der Kanal-Layout-Konvention der Eingabe.
    """
    import torch  # pylint: disable=import-outside-toplevel
    from scipy.signal import istft, stft  # pylint: disable=import-outside-toplevel

    x = np.asarray(audio, dtype=np.float32)
    device = next(core.parameters()).device
    win = np.hanning(_WIN)
    noverlap = _NFFT - _HOP

    was_nx2 = x.ndim == 2 and x.shape[1] == 2
    if was_nx2:  # (N, 2) → (2, N)
        x = x.T
    if x.ndim != 2:
        x = x[None]  # mono → (1, N)

    vocal_channels = []
    instr_channels = []
    for ch in range(x.shape[0]):
        channel = x[ch]
        total = channel.size
        # Chunking (30 s, ohne Überlapp — deterministisch, Naht-gefahr dokumentiert)
        chunk = _CHUNK_SAMPLES if sr == 48000 else int(_CHUNK_SAMPLES * sr / 48000)
        chunks = [channel[i : i + chunk] for i in range(0, total, chunk)]
        voc_parts = []
        for cchunk in chunks:
            _f, _t, zc = stft(cchunk, fs=sr, nperseg=_NFFT, noverlap=noverlap, window=win, boundary="even", padded=True)
            zri = np.stack([zc.real, zc.imag], axis=-1).astype(np.float32)  # (1025, T, 2)
            zri = np.stack([zri, zri], axis=0)[None]  # (1, s=2, 1025, T, 2)
            with torch.no_grad():
                mask = core(torch.from_numpy(zri).to(device)).cpu().numpy()
            mc_full = mask[0, 0, ..., 0] + 1j * mask[0, 0, ..., 1]  # (2050, T)
            mc = mc_full[: _NFFT // 2 + 1]  # Kanal 0 (1025 Bins) — Mono-Input liegt auf s=0/1 identisch
            _t2, v = istft(zc * mc, fs=sr, nperseg=_NFFT, noverlap=noverlap, window=win, boundary="even")
            voc_parts.append(v[: cchunk.size])
        vocal = np.concatenate(voc_parts) if voc_parts else np.zeros_like(channel)
        vocal = np.concatenate([vocal, np.zeros(max(0, total - vocal.size), dtype=np.float32)])
        vocal_channels.append(vocal[:total])
        instr_channels.append(channel - vocal[:total])

    out_v = np.stack(vocal_channels).astype(np.float32)
    out_i = np.stack(instr_channels).astype(np.float32)
    if was_nx2:
        out_v, out_i = out_v.T, out_i.T  # zurück ins (N, 2)-Layout
    if out_v.shape[0] == 1:
        out_v, out_i = out_v[0], out_i[0]
    result = {}
    if "vocals" in stems:
        result["vocals"] = np.clip(out_v, -1.0, 1.0)
    if "instruments" in stems:
        result["instruments"] = np.clip(out_i, -1.0, 1.0)
    return result
