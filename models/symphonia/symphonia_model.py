"""Symphonia — Flow-Matching-Kern für Instrumentalstems.

Der Kern verwendet die getestete Cantus-DiT-Topologie, aber die zweite
frameweise Bedingung steht für Rhythmus/Transienten statt Stimmtonhöhe.
So bleiben Gewichtsformat, Multi-Scale-FIR und ONNX-Vertrag identisch, während
das Trainingsziel ausschließlich auf drums+bass+other liegt.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

from models.cantus.cantus_model import CantusConfig, CantusModel


class SymphoniaModel(CantusModel):
    """Instrumental-DiT: ``rhythm`` enthält normierte Energie- und Onset-Features.

    ``pitch_dim`` bleibt als internes Cantus-Konfigurationsfeld erhalten; für
    Symphonia bezeichnet es ausschließlich die Breite dieses Rhythmus-Streams.
    """

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mert: torch.Tensor,
        rhythm: torch.Tensor,
        harm: torch.Tensor,
        use_cond: torch.Tensor,
    ) -> torch.Tensor:
        return super().forward(x, t, mert, rhythm, harm, use_cond)


class SymphoniaExportWrapper(nn.Module):
    """Explizite ONNX-Grenze mit semantischem Rhythmus-Eingang."""

    def __init__(self, model: SymphoniaModel) -> None:
        super().__init__()
        self.model: SymphoniaModel = model

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mert: torch.Tensor,
        rhythm: torch.Tensor,
        harm: torch.Tensor,
        use_cond: torch.Tensor,
    ) -> torch.Tensor:
        return cast(torch.Tensor, self.model(x, t, mert, rhythm, harm, use_cond))


def create_symphonia(config: CantusConfig | None = None, **overrides: object) -> SymphoniaModel:
    """Erzeugt den konfigurierbaren Symphonia-Kern deterministisch (§G5)."""
    return SymphoniaModel(config=config, **overrides)
