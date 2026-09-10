"""Liest Backbone-Namen und state_dict-Präfixe aus sgmse_plus_src_1.ckpt (mmap).

§v10-SGMSE-ONNX (2026-09-10): Bestimmt, welche NCSNpp-Variante der Checkpoint
enthält (ncsnpp / ncsnpp_48k / ncsnpp_v2), damit der Real-IO-Patch an der
richtigen Stelle sitzt, bevor scripts/export_sgmse_onnx.py läuft.
"""
import re
import sys
from pathlib import Path

# Lightning-Checkpoint pickled sgmse-Klassen — Package muss importierbar sein.
_SGMSE_PKG = Path(__file__).resolve().parent.parent / "models" / "sgmse_plus"
if str(_SGMSE_PKG) not in sys.path:
    sys.path.insert(0, str(_SGMSE_PKG))

import torch

CKPT = "/media/michael/Software 4TB/Aurik_Standalone/models/sgmse_plus/sgmse_plus_src_1.ckpt"


def main() -> int:
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
    print("[ckpt] top-level keys:", list(ckpt.keys()))

    hp = ckpt.get("hyper_parameters") or {}
    if isinstance(hp, dict):
        print("[ckpt] hyper_parameters:")
        for k, v in hp.items():
            s = str(v)
            if len(s) > 200:
                s = s[:200] + "…"
            print("    ", k, "=", s)

    sd = ckpt.get("state_dict") or {}
    print("[ckpt] state_dict tensors:", len(sd))
    print("[ckpt] backbone (hyper_parameters):", hp.get("backbone") if isinstance(hp, dict) else None)
    prefixes = sorted({k.split(".")[0] for k in sd})
    print("[ckpt] top-level Praefixe:", prefixes)

    refs = sorted({k for k in sd if re.search(r"ncsnpp|backbone|score|dnn", k, re.I)})
    print("[ckpt] ncsnpp/backbone-Keys (erste 25):")
    for k in refs[:25]:
        print("    ", k, tuple(sd[k].shape))

    blob = str(hp) + "\n" + "\n".join(sorted(sd)[:400])
    for pat in ("ncsnpp_48k", "ncsnpp_wide", "ncsnpp_v2", "ncsnpp", "NCSNpp"):
        print(f"[ckpt] '{pat}' referenziert:", pat in blob)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
