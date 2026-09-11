#!/usr/bin/env python3
"""§P1-4d (2026-09-09): Apollo TorchScript → Core-only-ONNX-Export + Parität.

Apollo enthält torch.stft im Graphen (→ §v10.736 CPU-Force, ROCm-Crash).
Der Core-Export schneidet vor der STFT: 80 Band-Eingänge
(real/imag/log-power concat, je 2·BW+1 Kanäle) → BN-Convs → Roformer-Netz
→ Output-GLUs → RI-Masken (1, 2, 442, T). STFT/Band-Split und ISTFT bleiben
in NumPy (deterministisch, §G5 (GEBOTE.md)).

Voraussetzung: vendored apollo.py nutzt .transpose(-2,-1) statt .mT
(§v10.750 — der Legacy-Exporter kennt aten::mT nicht).

Usage:
    .venv_aurik/bin/python scripts/export_apollo_onnx.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models" / "apollo"))

TS = ROOT / "models" / "apollo" / "apollo_model.pt"
OUT = ROOT / "models" / "apollo" / "apollo_core.onnx"


def main() -> int:
    import numpy as np  # pylint: disable=import-outside-toplevel
    import onnxruntime as ort  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel
    import torch.nn as nn  # pylint: disable=import-outside-toplevel
    from look2hear.models.apollo import Apollo  # pylint: disable=import-outside-toplevel

    if OUT.exists():
        print(f"existiert bereits: {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
        return 0

    ts = torch.jit.load(str(TS), map_location="cpu")
    sd = ts.state_dict()
    feature_dim = int(sd["BN.0.1.weight"].shape[0])
    n_layer = len({k.split(".")[1] for k in sd if k.startswith("net.") and k.split(".")[1].isdigit()})
    model = Apollo(sr=44100, win=20, feature_dim=feature_dim, layer=n_layer)
    res = model.load_state_dict(sd, strict=False)
    assert not res.missing_keys and not res.unexpected_keys, (
        f"MISMATCH missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}"
    )
    model.eval()

    class ApolloCoreBN(nn.Module):
        """80 Band-Eingänge → BN → net → output → RI-Masken."""

        def __init__(self, m):
            super().__init__()
            self.BN = m.BN
            self.net = m.net
            self.output = m.output
            self.nband = m.nband
            self.band_width = m.band_width

        def forward(self, *bands):
            subband_feature = torch.stack([self.BN[i](b) for i, b in enumerate(bands)], 1)
            feature = self.net(subband_feature)
            outs = []
            for i in range(self.nband):
                ri = self.output[i](feature[:, i])
                outs.append(ri.view(ri.shape[0], 2, self.band_width[i], -1))
            return torch.cat(outs, 2)

    core = ApolloCoreBN(model).eval()
    dummy = tuple(torch.zeros((1, 2 * bw + 1, 32), dtype=torch.float32) for bw in model.band_width)
    torch.onnx.export(
        core,
        dummy,
        str(OUT),
        input_names=[f"band_{i}" for i in range(model.nband)],
        output_names=["ri_masks"],
        dynamic_axes={f"band_{i}": {2: "time"} for i in range(model.nband)} | {"ri_masks": {2: "time"}},
        opset_version=17,
        dynamo=False,
        do_constant_folding=True,
    )
    print(f"Export OK: {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")

    rng = np.random.default_rng(7)
    np_bands = tuple((rng.standard_normal((1, 2 * bw + 1, 32)) * 0.1).astype(np.float32) for bw in model.band_width)
    with torch.no_grad():
        y_torch = core(*(torch.from_numpy(b) for b in np_bands)).numpy()
    sess = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
    y_onnx = sess.run(None, {f"band_{i}": np_bands[i] for i in range(model.nband)})[0]
    diff = float(np.abs(y_torch - y_onnx).max())
    print(f"Parität: max_abs={diff:.3e} (Toleranz 1e-3)")
    return 0 if diff < 1e-3 else 1


if __name__ == "__main__":
    sys.exit(main())
