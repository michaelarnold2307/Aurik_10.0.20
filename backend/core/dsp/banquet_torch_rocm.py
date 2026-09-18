"""§SOTA-ML-V5-GPU — BANQUET-Vinyl PyTorch-ROCm-Kern (GPU-Pfad).

Befund 2026-09-18 (Re-Export-Analyse SOTA-ML-V5): `banquet_vinyl_final.onnx`
ist ein BSRNN-artiger Kern aus 24 sequenziellen Zellen
(LayerNorm(128, geteilt) → bidir-LSTM(128→256) → Linear(512→128) + Residuum),
mit alternierendem Achsen-Tausch (0,2,1,3) zwischen den Zellen. Der ORT-ROCm-
Pfad liegt bei ~1,9 s je 1-s-Fenster (Kette aus 24 LSTM-Knoten, latenz-
gebunden); der PyTorch-ROCm-Kern braucht ~160 ms je Fenster (7900 XTX,
**~11,8×**) und mit Mini-Batch B=4 ~97 ms/Fenster (~19,5×).

Dieser Baustein rekonstruiert den Kern 1:1 aus den ONNX-Gewichten
(W-Gates native PyTorch-Ordnung; die Bias-Gates tragen die gespeicherte
Ordnung (i, c, f, o) und werden mit (0,2,3,1) auf PyTorch-Ordnung (i,f,g,o)
abgebildet — der ONNX-Graph reordnet nur W/R per Slice/Concat, B wird direkt
durchgereicht). Paritäts-Beweis (CPU, Optimizer aus): max|Δ| ≈ 1,9e-6 vs.
ONNX; GPU deterministisch (bit-identische Wiederholung,
§G5 (copilot-instructions.md)).

Fail-closed (§V6 (copilot-instructions.md)): Ohne torch/CUDA/ONNX gibt
get_banquet_torch_core() None zurück — der Aufrufer bleibt auf dem
ONNX/DSP-Pfad.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_N_CELLS = 24
_HIDDEN = 256
_FEAT = 128
_GATE_W = (0, 1, 2, 3)  # W/R: native PyTorch-Ordnung (i, f, g, o)
_GATE_B = (0, 2, 3, 1)  # Bias: gespeicherte Ordnung (i, c, f, o) → (i, f, g, o)

_lock = threading.Lock()
_core = None
_core_resolved = False


def _first_consumer(graph, tensor_name: str):
    for node in graph.node:
        if tensor_name in node.input:
            return node
    return None


def _perm_gates(t: np.ndarray, perm: tuple[int, ...]) -> np.ndarray:
    """Ordnet die 4 Gate-Blöcke à 256 Zeilen von `t` gemäß `perm` um."""
    return np.concatenate([t[256 * i : 256 * (i + 1)] for i in perm], axis=0)  # type: ignore[no-any-return]


def _init(inits: dict[str, np.ndarray | None], name: str) -> np.ndarray:
    """Initializer-Zugriff mit None-Guard (mytype-sauber)."""
    value = inits.get(name)
    if value is None:
        raise ValueError(f"BANQUET-ONNX: Initializer fehlt oder leer: {name}")
    return value


def _build_core():
    import onnx  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel
    import torch.nn as nn  # pylint: disable=import-outside-toplevel

    model_path = Path(__file__).resolve().parents[3] / "models" / "banquet" / "banquet_vinyl_final.onnx"
    graph = onnx.load(str(model_path)).graph
    inits: dict[str, np.ndarray | None] = {
        i.name: np.frombuffer(i.raw_data, dtype=onnx.helper.tensor_dtype_to_np_dtype(i.data_type)).reshape(i.dims)
        if i.raw_data
        else None
        for i in graph.initializer
    }

    lstms = [n for n in graph.node if n.op_type == "LSTM"]
    if len(lstms) != _N_CELLS:
        raise ValueError(f"BANQUET-ONNX: erwartet {_N_CELLS} LSTM-Zellen, gefunden {len(lstms)}")

    cell_data: list[dict[str, np.ndarray]] = []
    for k, lstm_node in enumerate(lstms):
        prefix = f"seqband.{k}."
        cur = lstm_node.output[0]
        # LSTM → Transpose → Reshape → Transpose → Reshape → MatMul(fc)
        for _want in ("Transpose", "Reshape", "Transpose", "Reshape"):
            cur = _first_consumer(graph, cur).output[0]
        matmul = _first_consumer(graph, cur)
        fc_weight = _init(inits, matmul.input[1]).T.copy()  # ONNX [512,128] → torch [128,512]
        add_bias = _first_consumer(graph, matmul.output[0])
        fc_bias = _init(inits, add_bias.input[1]).copy()
        lstm_bias = _init(inits, lstm_node.input[3]).copy()  # [2, 2048]
        cell_data.append(
            {
                "wih": _init(inits, prefix + "rnn.weight_ih_l0").copy(),
                "whh": _init(inits, prefix + "rnn.weight_hh_l0").copy(),
                "wih_r": _init(inits, prefix + "rnn.weight_ih_l0_reverse").copy(),
                "whh_r": _init(inits, prefix + "rnn.weight_hh_l0_reverse").copy(),
                "lstm_b": lstm_bias,
                "fc_w": fc_weight,
                "fc_b": fc_bias,
            }
        )

    norm_w = _init(inits, "seqband.0.norm.weight").copy()
    norm_b = _init(inits, "seqband.0.norm.bias").copy()

    class BanquetVinylCore(nn.Module):
        """24-Zellen-BSRNN-Kern: [B, 128, 128, 128] → [B, 128, 128, 128].

        Zelle k: LayerNorm(shared) → Merge B×D1 → LSTM(seq=D2, batch=B·D1)
        → fc + Residuum → Achsen-Tausch (0,2,1,3). Nach der letzten Zelle
        liefert der Achsen-Tausch exakt das ONNX-Ausgabe-Layout.
        """

        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(_FEAT)  # geteilt über alle Zellen
            self.rnns = nn.ModuleList(
                [nn.LSTM(_FEAT, _HIDDEN, bidirectional=True, batch_first=False) for _ in range(_N_CELLS)]
            )
            self.fcs = nn.ModuleList([nn.Linear(2 * _HIDDEN, _FEAT) for _ in range(_N_CELLS)])

        def forward(self, x):
            y = x
            for k in range(_N_CELLS):
                batch, d1, d2, feat = y.shape
                z = self.norm(y)
                z = z.reshape(batch * d1, d2, feat).permute(1, 0, 2)
                z, _ = self.rnns[k](z)
                z = z.permute(1, 0, 2).reshape(batch, d1, d2, 2 * _HIDDEN)
                y = self.fcs[k](z) + y
                y = y.permute(0, 2, 1, 3)
            return y

    core = BanquetVinylCore()
    with torch.no_grad():
        core.norm.weight.copy_(torch.from_numpy(norm_w))
        core.norm.bias.copy_(torch.from_numpy(norm_b))
        for k in range(_N_CELLS):
            data = cell_data[k]
            rnn = core.rnns[k]
            rnn.weight_ih_l0.copy_(torch.from_numpy(_perm_gates(data["wih"], _GATE_W)))
            rnn.weight_hh_l0.copy_(torch.from_numpy(_perm_gates(data["whh"], _GATE_W)))
            rnn.weight_ih_l0_reverse.copy_(torch.from_numpy(_perm_gates(data["wih_r"], _GATE_W)))
            rnn.weight_hh_l0_reverse.copy_(torch.from_numpy(_perm_gates(data["whh_r"], _GATE_W)))
            bias = data["lstm_b"]
            rnn.bias_ih_l0.copy_(torch.from_numpy(_perm_gates(bias[0, :1024], _GATE_B)))
            rnn.bias_hh_l0.copy_(torch.from_numpy(_perm_gates(bias[0, 1024:], _GATE_B)))
            rnn.bias_ih_l0_reverse.copy_(torch.from_numpy(_perm_gates(bias[1, :1024], _GATE_B)))
            rnn.bias_hh_l0_reverse.copy_(torch.from_numpy(_perm_gates(bias[1, 1024:], _GATE_B)))
            core.fcs[k].weight.copy_(torch.from_numpy(data["fc_w"]))
            core.fcs[k].bias.copy_(torch.from_numpy(data["fc_b"]))
    return core.eval()


def get_banquet_torch_core():
    """Lazy-Singleton: ROCm-Kern oder None (fail-closed, §V6 (copilot-instructions.md))."""
    global _core, _core_resolved
    with _lock:
        if _core_resolved:
            return _core
        _core_resolved = True
        try:
            import torch  # pylint: disable=import-outside-toplevel

            if not torch.cuda.is_available():
                logger.debug("§SOTA-ML-V5 BANQUET-torch-ROCm nicht verfügbar — ONNX-Pfad bleibt")
                return None
            _core = _build_core().to("cuda")
            logger.info("§SOTA-ML-V5 BANQUET-Kern auf ROCm geladen (%s)", torch.cuda.get_device_name(0))
            return _core
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("§SOTA-ML-V5 BANQUET-torch-Kern nicht ladbar: %s — ONNX-Pfad bleibt", exc)
            return None


def restore_banquet_torch(core, feat: np.ndarray) -> np.ndarray:
    """Führt den ROCm-Kern auf [B, 128, 128, 128]-Features aus.

    Args:
        core: Modul aus get_banquet_torch_core() (eval, auf cuda).
        feat: float32 [B, 128, 128, 128] (Mini-Batch von Fenstern).

    Returns: float32 [B, 128, 128, 128], NaN/Inf-geschützt (§0a
    (copilot-instructions.md)).
    """
    import torch  # pylint: disable=import-outside-toplevel

    device = next(core.parameters()).device
    x = np.asarray(feat, dtype=np.float32)
    if x.ndim != 4 or x.shape[1:] != (128, 128, 128):
        raise ValueError(f"BANQUET-torch: erwartet [B,128,128,128], bekam {x.shape}")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    with torch.no_grad():
        out = core(torch.from_numpy(x).to(device)).cpu().numpy()
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32)  # type: ignore[no-any-return]
