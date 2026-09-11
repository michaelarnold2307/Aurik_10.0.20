#!/usr/bin/env python3
"""Exportiert AERO (12 kHz → 48 kHz Super-Resolution) nach ONNX.

§v10-AERO-Challenger (2026-09-10): Der AERO-Backbone (HDemucs-Abkömmling,
plugins/_vendor_aero) rechnet im Spektralbereich mit komplexen STFT-Tensoren
(torch.stft → view_as_real → UNet → view_as_complex → torch.istft). ONNX kennt
keinen Complex-Typ — der dynamo-Exporter (torch ≥ 2.6, Default) deckt
stft/istft/fft ab. Legacy-Exporter (dynamo=False) ist für dieses Modell NICHT
geeignet (Complex-Senke).

Shape-Konstanz (wichtig): BLSTM(max_steps=200)-Framing, LocalState-T×T-
Deltamatrix und das STFT-Padding werden vom Tracer als Konstanten eingebacken.
Das Artefakt ist deshalb NUR am Export-Shape korrekt. Produktions-Shape:
1 Kanal, 120000 Samples (10 s @ 12 kHz, AeroPlugin._SEGMENT_S). Die
Plugin-Integration MUSS kürzere Endsegmente auf 120000 padden und das
4-fach verlängerte Ergebnis trimmen.

Parität: ONNX vs PyTorch auf Zufallsdaten am Export-Shape (rel < 1e-3).

Nutzung:
    .venv_aurik/bin/python scripts/export_aero_onnx.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_VENDOR = _ROOT / "plugins" / "_vendor_aero"
_CKPT = _ROOT / "models" / "aero" / "checkpoint_12-48_hl256.th"
_OUT = _ROOT / "models" / "aero" / "aero_12_48.onnx"
_SEGMENT = 120000  # 10 s @ 12 kHz (AeroPlugin._SEGMENT_S * _LR_SR)


def _fix_int32_scatter_indices(path: Path) -> None:
    """Post-Processing: Scatter-Indizes auf int64 casten.

    Der dynamo-Exporter emittiert für Slice-Zuweisungen im BLSTM-Framing
    ScatterND-Knoten mit int32-Indizes — laut ONNX-Spec ungültig (ort:
    INVALID_GRAPH, Befund 2026-09-10). Unkonditionaler Cast (int64→int64 ist
    Identität) macht den Graphen ort-ladbar; Constant-Folding entfernt No-Ops.
    """
    import onnx  # pylint: disable=import-outside-toplevel
    from onnx import helper  # pylint: disable=import-outside-toplevel

    model = onnx.load(str(path), load_external_data=True)
    new_nodes: list = []
    seen: dict[str, str] = {}
    n_casts = 0
    for node in model.graph.node:
        if node.op_type in ("ScatterND", "ScatterElements", "Scatter"):
            idx_pos = 2 if node.op_type == "Scatter" else 1
            iname = node.input[idx_pos]
            if iname in seen:
                node.input[idx_pos] = seen[iname]
            else:
                cast_name = f"{iname}_cast_i64"
                cast = helper.make_node(
                    "Cast", [iname], [cast_name], to=onnx.TensorProto.INT64, name=f"{node.name}_idx_cast"
                )
                node.input[idx_pos] = cast_name
                seen[iname] = cast_name
                n_casts += 1
                new_nodes.append(cast)  # direkt VOR dem Konsumenten einfügen
        new_nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    onnx.checker.check_model(model)
    # Tensoren liegen nach load_external_data im Speicher → Inline-Save
    onnx.save(model, str(path))
    _data = Path(str(path) + ".data")
    if _data.is_file():
        _data.unlink()
    print(f"[aero-export] {n_casts} Scatter-Indizes auf int64 gecastet")


def _build_model():
    import math  # pylint: disable=import-outside-toplevel

    import torch  # pylint: disable=import-outside-toplevel
    from torch.nn import functional as F

    if str(_VENDOR) not in sys.path:
        sys.path.insert(0, str(_VENDOR))
    from src.models.aero import Aero  # type: ignore

    class AeroOnnxModel(Aero):
        """Export-Variante: Frequenz-Pad auf reellen Paaren statt auf complex.

        Der Original-Schwanz (_convert_to_complex → _ispec mit F.pad auf dem
        komplexen Tensor) ist nicht exportierbar („No decompositions for
        complex-valued input", Befund 2026-09-10). Hier wird das +1-Frequenz-
        Bin VOR view_as_complex auf die reellen Paare gepaddet; istft bleibt
        dem dynamo-Exporter überlassen (stft/istft/fft-Abdeckung).
        """

        def forward(self, mix, return_spec=False, return_lr_spec=False):
            x = mix
            length = x.shape[-1]
            z = self._spec(x)
            x = self._move_complex_to_channels_dim(z)
            B, C, Fq, T = x.shape
            mean = x.mean(dim=(1, 2, 3), keepdim=True)
            std = x.std(dim=(1, 2, 3), keepdim=True)
            x = (x - mean) / (1e-5 + std)
            saved = []
            lengths = []
            for idx, encode in enumerate(self.encoder):
                lengths.append(x.shape[-1])
                x = encode(x, None)
                if idx == 0 and self.freq_emb is not None:
                    frs = torch.arange(x.shape[-2], device=x.device)
                    emb = self.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                    x = x + self.freq_emb_scale * emb
                saved.append(x)
            x = torch.zeros_like(x)
            for idx, decode in enumerate(self.decoder):
                skip = saved.pop(-1)
                x = decode(x, skip, lengths.pop(-1))
            assert len(saved) == 0
            x = x.view(B, self.out_channels, -1, Fq, T)
            x = x * std[:, None] + mean[:, None]
            # ── ONNX-Schwanz: manuelle reelle ISTFT (keine komplexen Tensoren) ──
            # Befund 2026-09-10: view_as_complex+istft emittiert DFT mit
            # inverse+onesided — von ort 1.27 abgelehnt (INVALID_GRAPH).
            # Stattdessen: irfft als feste Synthese-Matrizen (N/K konstant),
            # Overlap-Add über F.fold + Envelope-Division. Empirisch verifiziert
            # gegen torch.istft(normalized=True, center=True): rel ≈ 2e-5.
            x = x.permute(0, 1, 3, 4, 2).contiguous()  # [B,1,Fq,T,2]
            x = F.pad(x, (0, 0, 0, 0, 0, 1))  # +1 Frequenz-Bin (reelle Paare)
            x = x.squeeze(1)  # [B,Fq+1,T,2]
            re = x[..., 0].transpose(1, 2)  # [B,T,K]
            im = x[..., 1].transpose(1, 2)  # [B,T,K]
            N = int(self.nfft)
            K = int(x.shape[-3])  # Fq+1 = N//2 + 1 (Achse -3 nach squeeze)
            n_idx = torch.arange(N, dtype=torch.float32)
            k_idx = torch.arange(K, dtype=torch.float32)
            ang = 2.0 * math.pi * torch.outer(k_idx, n_idx) / N  # [K,N]
            scale = torch.full((K,), 2.0 / N)
            scale[0] = 1.0 / N
            scale[K - 1] = 1.0 / N
            a_mat = (scale[:, None] * torch.cos(ang)).unsqueeze(0)  # [1,K,N]
            b_mat = (scale[:, None] * torch.sin(ang)).unsqueeze(0)  # [1,K,N]
            frames = torch.matmul(re, a_mat) - torch.matmul(im, b_mat)  # [B,T,N]
            frames = frames.transpose(1, 2)  # [B,N,T]
            hl = int(self.hop_length * self.scale)
            win_length = int(self.win_length * self.scale)
            win = torch.hann_window(win_length)
            frames = frames * win[None, :, None]
            out_len = (frames.shape[-1] - 1) * hl + N
            fold = F.fold(
                frames,
                output_size=(1, out_len),
                kernel_size=(1, N),
                stride=(1, hl),
            )[:, 0, 0]  # [B,L]
            env = F.fold(
                win.pow(2)[None, :, None].expand(1, N, frames.shape[-1]),
                output_size=(1, out_len),
                kernel_size=(1, N),
                stride=(1, hl),
            )[:, 0, 0]  # [B,L]
            out = fold / env.clamp(min=1e-10) * math.sqrt(N)
            out = out[..., N // 2 : N // 2 + (frames.shape[-1] - 1) * hl]  # center-Slice
            return out[..., : int(length * self.scale)].unsqueeze(1)

    package = torch.load(str(_CKPT), map_location="cpu", weights_only=False)
    kwargs = dict(package["models"]["generator"].get("kwargs") or {})
    import inspect  # pylint: disable=import-outside-toplevel

    sig = inspect.signature(Aero.__init__).parameters
    kwargs = {k: v for k, v in kwargs.items() if k in sig and k != "self"}
    model = AeroOnnxModel(**kwargs)
    model.load_state_dict(package["models"]["generator"]["state"])
    model.eval()

    # Original-Modell für die Schwanz-Patch-Parität (gleiche Gewichte)
    orig = Aero(**kwargs)
    orig.load_state_dict(package["models"]["generator"]["state"])
    orig.eval()
    return model, orig


def main() -> int:
    import numpy as np  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel

    model, orig = _build_model()
    print(f"[aero-export] Modell geladen ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M Parameter)")

    rng = torch.Generator().manual_seed(0)
    x = torch.randn(1, 1, _SEGMENT, generator=rng) * 0.05

    with torch.no_grad():
        ref = model(x)
        try:
            ref_orig = orig(x)
        except RuntimeError as exc:
            # torch ≥ 2.7: Original-Aero-istft (4D-komplex) crasht eager —
            # Patch-Parität wurde separat unter torch 2.5.1 verifiziert.
            print(
                f"[aero-export] Original-Referenz nicht lauffähig ({type(exc).__name__}) — Patch-Parität übersprungen"
            )
            ref_orig = None
    print(f"[aero-export] PyTorch-Referenz: {tuple(ref.shape)}")
    if ref_orig is not None:
        _patch_err = float((ref - ref_orig).abs().max())
        _patch_rel = _patch_err / float(ref_orig.abs().max())
        print(f"[aero-export] Schwanz-Patch vs Original: max|Δ|={_patch_err:.3e} rel={_patch_rel:.3e}")
        assert _patch_rel < 1e-4, "Schwanz-Patch weicht vom Original ab"

    torch.onnx.export(
        model,
        (x,),
        str(_OUT),
        opset_version=17,
        input_names=["mix"],
        output_names=["output"],
        do_constant_folding=True,
    )
    _fix_int32_scatter_indices(_OUT)
    print(f"[aero-export] ONNX geschrieben: {_OUT} ({_OUT.stat().st_size / 1e6:.1f} MB)")

    import onnxruntime as ort  # pylint: disable=import-outside-toplevel

    sess = ort.InferenceSession(str(_OUT), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"mix": x.numpy()})[0]
    err = float(np.abs(out - ref.numpy()).max())
    rel = err / float(np.abs(ref.numpy()).max())
    print(f"[aero-verify] ONNX vs PyTorch: max|Δ|={err:.3e} rel={rel:.3e}")
    assert rel < 1e-3, "Parität verletzt"
    print("[aero-export] ERFOLG")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
