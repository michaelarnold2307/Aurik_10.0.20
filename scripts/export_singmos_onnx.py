#!/usr/bin/env python3
"""Exportiert SingMOS Pro (MOS_Predictor, wav2vec2-large) nach ONNX.

§v10-SINGMOS-ONNX (2026-09-10): singmos_pro.pt ist ein TorchScript-Artefakt,
dessen s3prl-wav2vec2-Experte die Layer-Liste dynamisch baut (prim::ListConstruct
aus dynamischem Gather, expert.py:68) — der Legacy-Exporter scheitert daran
(SymbolicValueError, reproduziert). Neuaufbau-Pfad stattdessen:

  1. MOS_Predictor eager aus dem Original-state_dict bauen
     (models/versa/hub_cache/checkpoints/ft_wav2vec2_large_ll60k_mdf_p1_200
     epochs_all_192epochs.pth — identische Gewichte wie singmos_pro.pt),
  2. Wrapper mit festen Eingängen (audio, audio_length, domain_id),
  3. dynamo-Exporter (venv_rocm72: torch 2.11 + s3prl 0.4.18) — dynamische
     Python-Listen werden statisch entrollt, keine Complex-Ops.

I/O-Vertrag: audio [B,1,T] (16 kHz), audio_length [B] int64, domain_id [B]
int64 → mos [B] (Skalar). Zeitachse dynamisch (faltungsbasiertes wav2vec2).

Nutzung:
    /home/michael/.local/share/aurik/venv_rocm72/bin/python scripts/export_singmos_onnx.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SINGMOS_ROOT = _ROOT / "models" / "singmos"
_PTH = (
    _ROOT
    / "models"
    / "versa"
    / "hub_cache"
    / "checkpoints"
    / "ft_wav2vec2_large_ll60k_mdf_p1_200epochs_all_192epochs.pth"
)
_OUT = _SINGMOS_ROOT / "singmos_pro.onnx"


def _build_wrapper():
    import sys as _sys

    # Shim: aurik_bridge.pth leitet s3prl auf .venv_aurik (torchaudio neu —
    # set_audio_backend/sox_effects entfernt). s3prl/hub.py importiert ALLE
    # Upstream-Hubkonfs beim Import und kippt dadurch um. Für wav2vec2 reicht
    # das direkte Laden der wav2vec2-Hubconf — wir ersetzen s3prl.hub durch
    # einen Minimal-Shim, bevor s3prl.nn ihn importiert (nur Importpfad).
    import types  # pylint: disable=import-outside-toplevel

    import torch  # pylint: disable=import-outside-toplevel
    from torch import nn

    _hub_shim = types.ModuleType("s3prl.hub")
    _hub_shim.options = lambda _only_registered_ckpt=False: []  # ungenutzt in unserem Pfad

    def _load_hubconf():
        from s3prl.upstream.wav2vec2.hubconf import wav2vec2_large_ll60k  # type: ignore

        _hub_shim.wav2vec2_large_ll60k = wav2vec2_large_ll60k

    _load_hubconf()
    _sys.modules["s3prl.hub"] = _hub_shim

    if str(_SINGMOS_ROOT) not in sys.path:
        sys.path.insert(0, str(_SINGMOS_ROOT))

    from singmos.ssl_mos.singmos_pro import MOS_Predictor  # type: ignore

    model = MOS_Predictor(
        ssl_model_type="wav2vec2_large_ll60k",
        use_domain_id=True,
        domain_num=6,
    )
    ckpt = torch.load(str(_PTH), map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()

    # Direkter Zugriff auf die Fairseq-wav2vec2-Modelle: MOS_Predictor →
    # SSL_Model → S3PRLUpstream → UpstreamExpert.model. Damit umgehen wir die
    # s3prl-Hook-Sammlung und Python-builtin max/min über Tensoren, an denen
    # torch.export scheitert (maximum()/minimum()-TypeErrors, Befund 2026-09-10).
    _expert = model.ssl_model.ssl_model.upstream
    _fairseq = _expert.model
    _fairseq.feature_grad_mult = 0.0
    _fairseq.encoder.layerdrop = 0.0

    import torch.nn.functional as F  # pylint: disable=import-outside-toplevel

    class SingMosOnnx(nn.Module):
        """Fester Eingangsvertrag: audio, audio_length, domain_id → MOS.

        Repliziert singmos_pro.forward für use_pitch=False, use_judge_id=False,
        use_domain_id=True, is_train=False — mit direktem extract_features-
        Aufruf (Encoder-Endoutput res["x"]), _match_length-Semantik und
        Decoder-Kopf. Rein tensor-basierte Ops (kein int()/range/pad_sequence),
        damit torch.export den Graphen mit dynamischer Zeitachse bauen kann.
        """

        def __init__(self, mos_model: nn.Module, fairseq_model: nn.Module):
            super().__init__()
            self.fairseq = fairseq_model
            self.domain_emb = mos_model.domain_emb
            self.decoder = mos_model.decoder
            self.wav_normalize = bool(getattr(_expert, "wav_normalize", False))
            self.apply_padding_mask = bool(getattr(_expert, "apply_padding_mask", True))

        def forward(self, audio: torch.Tensor, audio_length: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
            wav = audio.squeeze(1)  # [B, T]
            if self.wav_normalize:
                wav = F.layer_norm(wav, wav.shape)
            wav_len = audio_length
            # Tensor-only Padding-Maske (exportierbar, Zeitachse dynamisch):
            # cumsum statt arange(int(max_len)) — int() auf Tensoren ist im
            # torch.export-Graphen verboten (Befund 2026-09-10).
            seq = torch.ones_like(wav).cumsum(dim=1)  # [B, T], Werte 1..T
            mask = torch.ge(seq, wav_len.unsqueeze(1) + 1)  # True ab Position len (0-basiert)
            padded = wav.masked_fill(mask, 0.0)
            # padding_mask=None statt all-False-Maske: Das Modell wandelt eine
            # all-False-Maske intern selbst in None um (wav2vec2_model.py:
            # 2652 ff., else-Zweig) — für den Full-Length-Fall identisch,
            # eliminiert aber den datenabhängigen padding_mask.any()-Branch,
            # an dem torch.export scheitert (Befund 2026-09-10).
            results = self.fairseq.extract_features(padded, None)
            # res["x"] ist der Encoder-Endoutput (B, T', 1024) inkl. finaler
            # LayerNorm (layer_norm_first-Pfad, wav2vec2_model.py:3049).
            # layer_results[-1][0] wäre der rohe letzte Layer-Output ohne diese
            # Norm (Paritätsabweichung max|Δ|≈2200, Befund 2026-09-10).
            h = results["x"]  # [B, T', 1024]
            # S3PRLUpstream._match_length (stride 320): Die Conv-Kaskade
            # (k10s5, 4×k3s2, 2×k2s2) liefert für alle T ≥ 320 genau
            # T' = ceil(T/320) - 1 — der letzte Frame wird also IMMER genau
            # einmal wiederholt (Crop-Branch ist toter Code). Als bedingungs-
            # loser Tensor-Op exportierbar, ohne datenabhängige Shape-Logik.
            h = torch.cat((h, h[:, -1:, :]), dim=1)  # [B, ceil(T/320), 1024]
            dom = self.domain_emb(domain_id.long())  # [B, 128]
            x = torch.cat((h, dom.unsqueeze(1).expand(-1, h.size(1), -1)), dim=-1)
            pred_frame = self.decoder(x).squeeze(-1)
            return torch.mean(pred_frame, dim=1)  # [B]

    return SingMosOnnx(model, _fairseq).eval(), model


def main() -> int:
    import numpy as np  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel

    wrapper, full_model = _build_wrapper()
    print(f"[singmos-export] Modell gebaut ({sum(p.numel() for p in wrapper.parameters()) / 1e6:.1f}M Parameter)")

    T = 160000  # 10 s @ 16 kHz
    rng = torch.Generator().manual_seed(0)
    audio = torch.randn(1, 1, T, generator=rng) * 0.05
    length = torch.tensor([T], dtype=torch.long)
    domain = torch.tensor([1], dtype=torch.long)

    with torch.no_grad():
        ref = wrapper(audio, length, domain)
        ref_full = full_model.forward(audio, length, domain_id=domain, is_train=False)
    print(f"[singmos-export] PyTorch-Referenz: {tuple(ref.shape)} = {float(ref[0]):.4f}")
    _diff = float((ref - ref_full).abs().max())
    print(f"[singmos-export] Wrapper vs volles Modell: max|Δ|={_diff:.3e}")
    assert _diff < 1e-3, "Wrapper weicht vom vollen Modell ab"

    torch.onnx.export(
        wrapper,
        (audio, length, domain),
        str(_OUT),
        opset_version=17,
        input_names=["audio", "audio_length", "domain_id"],
        output_names=["mos"],
        dynamic_axes={"audio": {0: "batch", 2: "time"}, "mos": {0: "batch"}},
        do_constant_folding=True,
    )
    print(f"[singmos-export] ONNX geschrieben: {_OUT} ({_OUT.stat().st_size / 1e6:.1f} MB)")

    import onnxruntime as ort  # pylint: disable=import-outside-toplevel

    sess = ort.InferenceSession(str(_OUT), providers=["CPUExecutionProvider"])
    for t_val in (T, T // 2):
        audio_t = audio[:, :, :t_val]
        length_t = np.array([t_val], dtype=np.int64)
        out = sess.run(
            None,
            {"audio": audio_t.numpy(), "audio_length": length_t, "domain_id": np.array([1], dtype=np.int64)},
        )[0]
        with torch.no_grad():
            ref_t = wrapper(audio_t, torch.tensor([t_val], dtype=torch.long), domain)
        err = float(np.abs(out[0] - ref_t[0].numpy()))
        print(f"[singmos-verify] T={t_val}: ONNX={float(out[0]):.4f} PyTorch={float(ref_t[0]):.4f} |Δ|={err:.3e}")
        assert err < 1e-3, "Parität verletzt"
    print("[singmos-export] ERFOLG")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
