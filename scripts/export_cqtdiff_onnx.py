"""
CQTdiff Score-Network → TorchScript Exporter für Aurik 10.0.0
=========================================================
Exportiert das Score-Netzwerk (UNet-CQT) des CQTdiff-Modells als TorchScript (.pt)
und als einzelne ONNX-Datei mit eingebetteten Parametern.

Checkpoint: models/cqtdiff/src/models/cqt_weights.pt  (119 MB, EMA step 319999)
Ausgabe:    models/cqtdiff/score_network.pt             (~62 MB, TorchScript)
            models/cqtdiff/score_network.onnx           (~62 MB, ONNX)

Das exportierte Modell erwartet:
    Input  "x_noisy"   shape [1, 65536]   float32  (konditioniertes Audio @ 22050 Hz)
    Input  "sigma"     shape [1, 1]       float32  (Rauschpegel σ für EDM-Preconditioning)
    Output              shape [1, 65536]  float32  (Schätzung des clean signal D(x_noisy, σ))

EDM-Preconditioning (Karras et al. 2022, Gleichungen 7):
    c_skip = σ_data² / (σ² + σ_data²)
    c_out  = σ · σ_data / √(σ² + σ_data²)
    c_in   = 1 / √(σ² + σ_data²)
    c_noise= ln(σ) / 4
    D(x,σ) = c_skip·x + c_out · UNet(c_in·x, c_noise)

EMA-Weight-Mapping:
    Das Checkpoint-Format speichert 171 EMA-Tensoren (= Parameter-Count) als Liste.
    10 Resample-Kernel sind registered buffers und werden aus dem Original-Checkpoint übernommen.
Aurik-Spec: §4.4 — CQTdiff (IEEE TASLP 2022) als Primär-Inpainting ≥ 50 ms.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

# ---------------------------------------------------------------------------
# Workspace-Root und Pfade
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).parent.parent
CQTDIFF_DIR = WORKSPACE / "models" / "cqtdiff"
CHECKPOINT = CQTDIFF_DIR / "src" / "models" / "cqt_weights.pt"
OUTPUT_PT = CQTDIFF_DIR / "score_network.pt"
OUTPUT_ONNX = CQTDIFF_DIR / "score_network.onnx"

# CQTdiff-Quellcode zum Suchpfad hinzufügen
sys.path.insert(0, str(CQTDIFF_DIR))

# plotly ist in cqtdiff nur für Logging-Visualisierung — hier nicht benötigt
for _mod in ["plotly", "plotly.express", "plotly.graph_objects", "plotly.subplots"]:
    _m = types.ModuleType(_mod)
    _m.__spec__ = None  # type: ignore[assignment]
    sys.modules[_mod] = _m


def _make_args() -> types.SimpleNamespace:
    """Erstellt Konfigurations-Namespace, der das hydra-args-Objekt simuliert."""

    def make_ns(d: dict) -> types.SimpleNamespace:
        ns = types.SimpleNamespace()
        for k, v in d.items():
            setattr(ns, k, make_ns(v) if isinstance(v, dict) else v)
        return ns

    return make_ns(
        {
            "sample_rate": 22050,
            "audio_len": 65536,
            "cqt": {"binsoct": 64, "numocts": 7, "use_norm": False},
            "unet_STFT": {"depth": 5},
        }
    )


def _load_ema_weights(model: torch.nn.Module, checkpoint_path: str) -> torch.nn.Module:
    """Lädt EMA-Gewichte aus dem Checkpoint.

    Der Checkpoint speichert 171 EMA-Tensoren (in model.parameters()-Reihenfolge)
    sowie 10 Resample-Puffer (registered buffers). Die Puffer werden aus dem
    Original-Checkpoint übernommen.

    Args:
        model:           Frisch initialisiertes Unet_CQT-Modell
        checkpoint_path: Pfad zur .pt-Checkpoint-Datei

    Returns:
        Modell mit geladenen EMA-Gewichten
    """
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    ema_weights = ckpt["ema_weights"]  # list[Tensor], len=171
    orig_model_sd = ckpt["model"]  # OrderedDict, len=181

    # Baue State-Dict: Parameter aus EMA, Puffer aus Original
    param_names = [name for name, _ in model.named_parameters()]
    buffer_names = {name for name, _ in model.named_buffers()}

    assert len(param_names) == len(ema_weights), (
        f"EMA-Zähler ({len(ema_weights)}) passt nicht zur Parameteranzahl ({len(param_names)})"
    )

    state_dict: dict = {}
    for name, ema_tensor in zip(param_names, ema_weights):
        state_dict[name] = ema_tensor

    for name in buffer_names:
        state_dict[name] = orig_model_sd[name]

    model.load_state_dict(state_dict, strict=True)
    return model


def main() -> None:
    # ------------------------------------------------------------------
    # Checkpoint-Validierung
    # ------------------------------------------------------------------
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint nicht gefunden: {CHECKPOINT}\nErwartet in: models/cqtdiff/src/models/cqt_weights.pt"
        )

    try:
        import torch
    except ImportError as e:
        raise ImportError(f"torch nicht verfügbar: {e}") from e

    # ------------------------------------------------------------------
    # Modell laden
    # ------------------------------------------------------------------
    print(f"Lade Checkpoint: {CHECKPOINT} (EMA step 319999)")
    args = _make_args()

    from src.models.unet_cqt import Unet_CQT  # type: ignore[import]

    model = Unet_CQT(args, "cpu")
    model.eval()
    model = _load_ema_weights(model, str(CHECKPOINT))
    model.eval()
    print(f"  Modell: {sum(p.numel() for p in model.parameters()):,} Parameter")

    # ------------------------------------------------------------------
    # EDM-Preconditioning-Wrapper
    # ------------------------------------------------------------------
    SIGMA_DATA = 0.057  # Maestro-Trainingsparameter

    class ScoreNetWrapper(torch.nn.Module):
        """Wraps UNet-CQT with EDM preconditioning (Karras et al. 2022, Eq. 7).

        Forward: D(x_noisy, σ) = c_skip·x_noisy + c_out · UNet(c_in·x_noisy, c_noise)
        """

        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.inner: Any = inner
            self.sigma_data = SIGMA_DATA

        def forward(self, x_noisy: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            sigma = sigma.view(-1, 1)  # [B, 1]
            sd2 = self.sigma_data**2
            c_skip = sd2 / (sigma**2 + sd2)
            c_out = sigma * self.sigma_data / (sigma**2 + sd2).sqrt()
            c_in = 1.0 / (sigma**2 + sd2).sqrt()
            c_noise = sigma.log() / 4.0  # [B, 1]

            x_in = c_in * x_noisy
            raw_out = self.inner(x_in, c_noise)  # [B, 65536]
            return c_skip * x_noisy + c_out * raw_out  # type: ignore[no-any-return]

    wrapper = ScoreNetWrapper(model)
    wrapper.eval()

    class SpectralScoreWrapper(torch.nn.Module):
        """ONNX-fähiger UNet-Kern ohne komplexe CQT/iCQT-Operationen."""

        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, x_f, sigma):
            inner: Any = self.inner
            # EDM (Karras et al. 2022, Eq. 7) im Spektralbereich:
            # D(c, σ) = c_skip·c + c_out·U(c_in·c, c_noise)
            # — c_skip wirkt auf UNskalierte Koeffizienten c, der UNet-Kern bekommt c_in·c
            # (wie Unet_CQT.forward, das c_in·x im Zeitbereich erhält).
            sigma_value = sigma.view(-1, 1)
            sd2 = SIGMA_DATA**2
            c_in = 1.0 / (sigma_value**2 + sd2).sqrt()
            # c_noise = ln(σ)/4 wie im ScoreNetWrapper — das Modell wurde mit
            # c_noise-Konditionierung trainiert.
            sigma_emb = inner.embedding(sigma.log() / 4.0)
            pyr = c_in * x_f
            x = inner.freq_encoding(c_in * x_f) if inner.use_fencoding else c_in * x_f
            x = inner.init_conv(x)
            hidden = []
            for index, modules in enumerate(inner.downs):
                if index < inner.depth - 1:
                    resnet, downsample, combiner = modules
                    x = resnet(x, sigma_emb)
                    hidden.append(x)
                    x = downsample(x)
                    pyr = downsample(pyr)
                    x = combiner(pyr, x)
                else:
                    (resnet,) = modules
                    x = resnet(x, sigma_emb)
                    hidden.append(x)
            for (resnet,) in inner.middle:
                x = resnet(x, sigma_emb)
            pyr = None
            for index, modules in enumerate(inner.ups):
                depth_index = inner.depth - index - 1
                if depth_index > 0:
                    resnet, upsample, combiner = modules
                    skip = hidden.pop()
                    x = inner.cropconcat(x, skip)
                    x = resnet(x, sigma_emb)
                    pyr = combiner(pyr, x)
                    x = upsample(x)
                    pyr = upsample(pyr)
                else:
                    (resnet,) = modules
                    skip = hidden.pop()
                    x = inner.cropconcat(x, skip)
                    x = resnet(x, sigma_emb)
                    pyr = combiner(pyr, x)
            c_skip = sd2 / (sigma_value**2 + sd2)
            c_out = sigma_value * SIGMA_DATA / (sigma_value**2 + sd2).sqrt()
            return c_skip.unsqueeze(-1).unsqueeze(-1) * x_f + c_out.unsqueeze(-1).unsqueeze(-1) * pyr

    # ------------------------------------------------------------------
    # Forward-Probe
    # ------------------------------------------------------------------
    audio_len = args.audio_len  # 65536
    dummy_x = torch.zeros(1, audio_len)
    dummy_sigma = torch.ones(1, 1) * 1.0
    spectral_wrapper = SpectralScoreWrapper(model).eval()
    with torch.no_grad():
        spectral_input = model.CQTransform.fwd(dummy_x).permute(0, 3, 2, 1).contiguous()
        spectral_reference = spectral_wrapper(spectral_input, dummy_sigma)
    print(f"  Spectral-Core bereit — Ausgabe: {list(spectral_reference.shape)}")

    with torch.no_grad():
        out = wrapper(dummy_x, dummy_sigma)
    assert out.shape == (1, audio_len), f"Forward-Shape fehlerhaft: {out.shape}"
    print(f"  Forward-Test OK — Ausgabe: {list(out.shape)}, Bereich: [{out.min():.4f}, {out.max():.4f}]")

    # ------------------------------------------------------------------
    # TorchScript-Export via torch.jit.trace
    # ------------------------------------------------------------------
    print(f"\nExportiere nach: {OUTPUT_PT}")
    OUTPUT_PT.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (dummy_x, dummy_sigma), strict=False)

    # Zweite Probe mit anderen Werten
    dummy_x2 = torch.randn(1, audio_len) * 0.1
    dummy_sigma2 = torch.ones(1, 1) * 0.3
    with torch.no_grad():
        out_trace = traced(dummy_x2, dummy_sigma2)
    assert out_trace.shape == (1, audio_len), f"Trace-Shape fehlerhaft: {out_trace.shape}"

    traced.save(str(OUTPUT_PT))
    size_mb = OUTPUT_PT.stat().st_size / 1e6
    print(f"✓ TorchScript gespeichert — Größe: {size_mb:.1f} MB")

    # ------------------------------------------------------------------
    # ONNX-Export — ein eigenständiges Modell inklusive aller Parameter
    # ------------------------------------------------------------------
    # Hinweis: Unet_CQT.forward enthält die komplette NSGT (CQT fwd + bwd),
    # d. h. der ONNX-Graph deckt den vollen Audio-Roundtrip ab. Die NSGT ist
    # dafür funktional (ohne In-place-Schreibzugriffe) umgebaut.
    print(f"\nExportiere ONNX nach: {OUTPUT_ONNX}")
    try:
        import onnxscript  # Voraussetzung für dynamo=True
    except ImportError as e:
        raise ImportError(
            "onnxscript fehlt — für dynamo=True erforderlich. Installation: .venv_aurik/bin/pip install onnxscript"
        ) from e
    torch.onnx.export(
        spectral_wrapper,
        (spectral_input, dummy_sigma),
        str(OUTPUT_ONNX),
        input_names=["cqt", "sigma"],
        output_names=["score_cqt"],
        opset_version=18,
        # Constant-Folding deaktiviert: hält die Exportzeit deterministisch
        # (der Optimizer skaliert am großen, entrollten UNet-Graphen überlinear).
        # Validierung ist davon unberührt — die Parameter sind bereits eingebettet.
        do_constant_folding=False,
        dynamo=True,
    )
    onnx_size_mb = OUTPUT_ONNX.stat().st_size / 1e6
    print(f"✓ ONNX gespeichert — Größe: {onnx_size_mb:.1f} MB")

    # ------------------------------------------------------------------
    # Lade-Validierung
    # ------------------------------------------------------------------
    loaded = torch.jit.load(str(OUTPUT_PT), map_location="cpu")
    loaded.eval()
    with torch.no_grad():
        out_val = loaded(dummy_x2, dummy_sigma2)

    import numpy as np

    assert out_val.shape == (1, audio_len), f"Validierungs-Shape fehlerhaft: {out_val.shape}"
    assert np.isfinite(out_val.numpy()).all(), "NaN/Inf in TorchScript-Ausgabe!"

    diff = (out_val - out_trace).abs().max().item()
    assert diff < 1e-4, f"Ausgabedifferenz zu groß: {diff}"

    print(f"✓ Lade-Validierung OK — Max-Diff: {diff:.2e}")

    # ONNX-Runtime-Validierung gegen denselben PyTorch-Forward.
    import onnxruntime as ort

    session = ort.InferenceSession(str(OUTPUT_ONNX), providers=["CPUExecutionProvider"])
    onnx_out = session.run(
        ["score_cqt"],
        {"cqt": spectral_input.numpy(), "sigma": dummy_sigma.numpy()},
    )[0]
    assert onnx_out.shape == tuple(spectral_reference.shape), f"ONNX-Shape fehlerhaft: {onnx_out.shape}"
    assert np.isfinite(onnx_out).all(), "NaN/Inf in ONNX-Ausgabe!"
    onnx_diff = np.max(np.abs(onnx_out - spectral_reference.numpy()))
    assert onnx_diff < 1e-3, f"ONNX-Ausgabedifferenz zu groß: {onnx_diff}"
    print(f"✓ ONNX-Validierung OK — Max-Diff zu PyTorch: {onnx_diff:.2e}")

    # Zweite Probe bei σ=0.3 — deckt die c_noise-Konditionierung ab (ln σ/4).
    # (Bei σ=1.0 ist c_noise=0; ein fehlerhaftes Embedding würde dort nicht auffallen.)
    with torch.no_grad():
        spectral_ref2 = spectral_wrapper(spectral_input, dummy_sigma2)
    onnx_out2 = session.run(
        ["score_cqt"],
        {"cqt": spectral_input.numpy(), "sigma": dummy_sigma2.numpy()},
    )[0]
    assert onnx_out2.shape == tuple(spectral_ref2.shape), f"ONNX-Shape fehlerhaft: {onnx_out2.shape}"
    assert np.isfinite(onnx_out2).all(), "NaN/Inf in ONNX-Ausgabe (σ=0.3)!"
    onnx_diff2 = np.max(np.abs(onnx_out2 - spectral_ref2.numpy()))
    assert onnx_diff2 < 1e-3, f"ONNX-Ausgabedifferenz zu groß (σ=0.3): {onnx_diff2}"
    print(f"✓ ONNX-Validierung σ=0.3 OK — Max-Diff zu PyTorch: {onnx_diff2:.2e}")

    print(f"\n✓ Exportiert: {OUTPUT_ONNX}")
    print("  Nächster Schritt: Aurik starten — CQTdiff wird automatisch geladen.")


if __name__ == "__main__":
    main()
