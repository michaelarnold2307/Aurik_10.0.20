"""Exportiert den SGMSE+-Score-Core (NCSNpp) als ONNX via Real-IO-Modus.

§v10.762 (2026-09-10): Der Backbone rechnet intern reell; die einzigen
Complex-Operationen sitzen an der I/O-Grenze (torch.complex/.real/.imag am
Eingang, torch.view_as_complex am Ausgang). ONNX kennt keinen Complex-Typ —
stattdessen: _real_io=True (gepatcht in ncsnpp.py), Eingang [B,4,F,T] reell
(cat([x_t, y], dim=1)), Ausgang [B,2,F,T] (re, im). Der TorchScript-Pfad bleibt
unverändert (Default _real_io=False).

Vorzeichen-Konvention: Das TS-Artefakt sgmse_plus.ts negiert den Score
(SgmseWrapper: out = -dnn(x, t)). Der ONNX-Wrapper backt dieselbe Negation
ein, damit sgmse_plus_core.onnx drop-in mit dem TS-Artefakt ist.

Verifikation: Parität Real-IO vs. komplexer Pfad DESSELBEN dnn (gleiche
Gewichte, gleiche Daten) — validiert den _real_io-Patch selbst.
T = 64 (durch 2^6=64 teilbar: 6 Downsample-Stufen brauchen gerade Dims,
sonst Skip-Connection-Mismatch im UNet — empirisch mit T=96 reproduziert).

Hinweis TS-Artefakt: sgmse_plus.ts trägt ANDERE Gewichte (gebaut aus
HF train_vb_29nqe0uh_epoch=115.ckpt, nicht aus sgmse_plus_src_1.ckpt;
alle 647 Parameter weichen ab, beobachtete Ausgabeabweichung rel ~3e-1).
Der TS-Vergleich ist daher informativ und nicht assertbar.

Nutzung:
    .venv_aurik/bin/python scripts/export_sgmse_onnx.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
_SGMSE_PKG = _ROOT / "models" / "sgmse_plus"  # Parent des 'sgmse'-Packages
# Per argv übersteuerbar: später z. B. sgmse_musik_best.ckpt → sgmse_musik_core.onnx
_CKPT = Path(sys.argv[1]) if len(sys.argv) > 1 else _ROOT / "models" / "sgmse_plus" / "sgmse_plus_src_1.ckpt"
_TS = _ROOT / "models" / "sgmse_plus" / "sgmse_plus.ts"
_OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else _ROOT / "models" / "sgmse_plus" / "sgmse_plus_core.onnx"


def _load_dnn():
    import torch  # pylint: disable=import-outside-toplevel

    sys.path.insert(0, str(_SGMSE_PKG))
    from sgmse.backbones import BackboneRegistry  # type: ignore

    ckpt = torch.load(str(_CKPT), map_location="cpu", weights_only=False, mmap=True)
    hyper = dict(ckpt.get("hyper_parameters") or {})
    backbone_name = str(hyper.get("backbone") or "ncsnpp").strip()
    dnn_cls = BackboneRegistry.get_by_name(backbone_name)
    dnn = dnn_cls(**hyper)
    state = ckpt.get("state_dict", {})
    dnn_state = {k[4:]: v for k, v in state.items() if k.startswith("dnn.")}
    assert dnn_state, "keine dnn.*-Gewichte im Checkpoint"
    dnn.load_state_dict(dnn_state, strict=False)
    dnn._real_io = True  # Aurik-ONNX-Patch aktivieren
    dnn.eval()
    return dnn, backbone_name


def main() -> int:
    import numpy as np  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel

    dnn, backbone_name = _load_dnn()
    print(f"[export] Backbone: {backbone_name}")

    class RealIOWrapper(torch.nn.Module):
        def forward(self, x_t, y, t):
            # x_t/y liegen reell [B,2,F,T] vor (re, im als Kanalachse).
            x = torch.cat([x_t, y], dim=1)  # [B,4,F,T] = x_t.re, x_t.im, y.re, y.im
            # Negation wie SgmseWrapper im TS-Artefakt (out = -dnn(x, t))
            return -self.dnn(x, t)  # [B,2,F,T] (re, im)

    wrapper = RealIOWrapper()
    wrapper.dnn = dnn

    F, T = 256, 64  # T: Vielfaches von 2^6=64 (6 Downsample-Stufen, gerade Dims)
    rng = torch.Generator().manual_seed(0)
    x_t = torch.randn(1, 2, F, T, generator=rng)
    y = torch.randn(1, 2, F, T, generator=rng) * 0.1
    t = torch.tensor([0.5])

    with torch.no_grad():
        real_out = wrapper(x_t, y, t)
        print(f"[parity] Real-IO-Ausgabe: {tuple(real_out.shape)}")

        # Referenz: komplexer Pfad desselben dnn (gleiche Gewichte) —
        # validiert den _real_io-Patch. Beide Pfade negieren den Score.
        dnn._real_io = False
        x_c = torch.complex(x_t[:, 0], x_t[:, 1])
        y_c = torch.complex(y[:, 0], y[:, 1])
        cplx_out = -dnn(torch.stack([x_c, y_c], dim=1), t)
        cplx_ref = torch.stack([cplx_out.real[:, 0], cplx_out.imag[:, 0]], dim=1)
        dnn._real_io = True
        err = float((cplx_ref - real_out).abs().max())
        scale = float(cplx_ref.abs().max())
        print(f"[parity] Real-IO vs komplexer Pfad: max|Δ|={err:.3e} rel={err / max(scale, 1e-9):.3e}")
        assert err / max(scale, 1e-9) < 1e-4, "Parität verletzt"

        # Informativ: TS-Artefakt trägt andere Gewichte (kein Assert).
        try:
            ts = torch.jit.load(str(_TS), map_location="cpu")
            ts.eval()
            ts_out = ts.forward(x_t, y, t)
            if ts_out.is_complex():
                ts_ref = torch.stack([ts_out.real[:, 0], ts_out.imag[:, 0]], dim=1)
            else:
                ts_ref = ts_out.squeeze(2)
            ts_err = float((ts_ref - real_out).abs().max())
            print(
                f"[info] TS-Artefakt vs Real-IO: max|Δ|={ts_err:.3e} (erwartet groß — andere Gewichte, siehe Docstring)"
            )
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[info] TS-Artefakt-Check übersprungen: {type(exc).__name__}: {exc}")

    torch.onnx.export(
        wrapper,
        (x_t, y, t),
        str(_OUT),
        opset_version=17,
        input_names=["x_t", "y", "t"],
        output_names=["score"],
        dynamic_axes={
            "x_t": {0: "batch", 2: "freq", 3: "frames"},
            "y": {0: "batch", 2: "freq", 3: "frames"},
            "t": {0: "batch"},
            "score": {0: "batch", 2: "freq", 3: "frames"},
        },
        dynamo=False,  # Legacy-Exporter (Muster: export_bs_roformer_onnx.py);
        # torch.export spezialisiert die Frames-Dim statisch → dynamo-Konflikt.
        do_constant_folding=True,
    )
    print(f"[export] ONNX geschrieben: {_OUT} ({_OUT.stat().st_size / 1e6:.1f} MB)")

    # ONNX-Verifikation: Ausgabe == PyTorch-Real-IO (CPU)
    import onnxruntime as ort  # pylint: disable=import-outside-toplevel

    sess = ort.InferenceSession(str(_OUT), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(
        None,
        {"x_t": x_t.numpy(), "y": y.numpy(), "t": np.array([0.5], dtype=np.float32)},
    )[0]
    err = float(np.abs(onnx_out - real_out.numpy()).max())
    rel = err / float(np.abs(real_out.numpy()).max())
    print(f"[verify] ONNX vs PyTorch max|Δ|={err:.3e} rel={rel:.3e}")
    assert rel < 1e-4, "ONNX-Verifikation fehlgeschlagen"
    print("[export] ERFOLG")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
