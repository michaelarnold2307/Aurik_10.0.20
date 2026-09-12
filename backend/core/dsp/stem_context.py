"""StemContext — First-Class-Stem-Objekt für den Vokal-/Musik-Block.

§v10.19-Paket (2026-09-11): StemContext trägt die separierten Stems
(Vokal + Instrumental) samt Bookkeeping (angewandte Stufen, Witness-Befunde)
durch die Pipeline. Phasen wie 19/43/66 können den Kontext aus dem
Restoration-Kontext lesen, statt selbst zu separieren — ein
Rekombinationspunkt (§SLR-1f) statt N Rekombinationen.

Layout-Vertrag (§G5 (copilot-instructions.md), Stereo-Layout-Invariante):
Stems sind channels-first ``(C, N)`` für Stereo bzw. ``(N,)`` für Mono.
:meth:`StemContext.recombine` erzwingt das Referenz-Layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class StemContext:
    """Separierte Stems + Stage-Bookkeeping für den Stem-Block."""

    vocal_stem: np.ndarray
    """Vokal-Stem, channels-first (C, N) bzw. (N,) mono, float32."""

    instrumental_stem: np.ndarray
    """Instrumental-Stem (Summe aller Nicht-Vokal-Stems), gleiches Layout."""

    sample_rate: int
    """Sample-Rate der Stems (Hz)."""

    panns_singing: float = 0.0
    """PANNs-Singing-Confidence [0, 1] zum Separationszeitpunkt."""

    separation_model: str = ""
    """Modellroute der Separation (bs_roformer / demucs_v4_htdemucs / dsp)."""

    applied_stages: list[str] = field(default_factory=list)
    """Angewandte Stem-Stufen in Reihenfolge (miipher, dfn, kim_vocal_2, kim_inst)."""

    witness_reports: dict[str, Any] = field(default_factory=dict)
    """Listening-Witness-Befunde pro Stufe (phase_id → as_dict())."""

    def recombine(self, reference: np.ndarray | None = None) -> np.ndarray:
        """Rekombiniert Vokal + Instrumental zum Mix (ein Rekombinationspunkt).

        Args:
            reference: Optionales Referenz-Audio, dessen Layout/Dtype erzwungen wird
                       (Default: Layout der Stems).

        Returns:
            Mix als float32; bei Stereo channels-first (C, N), sonst (N,).
        """
        _voc = np.asarray(self.vocal_stem, dtype=np.float32)
        _ins = np.asarray(self.instrumental_stem, dtype=np.float32)
        # Shapes angleichen (Bordline: ein Stem kann 1-D sein, der andere 2-D)
        if _voc.ndim != _ins.ndim:
            if _voc.ndim == 1:
                _voc = np.stack([_voc, _voc], axis=0)
            else:
                _ins = np.stack([_ins, _ins], axis=0)
        n = min(_voc.shape[-1], _ins.shape[-1])
        _mix = _voc[..., :n] + _ins[..., :n]
        _mix = np.nan_to_num(_mix, nan=0.0, posinf=0.0, neginf=0.0)
        _mix = np.clip(_mix, -1.0, 1.0).astype(np.float32)
        if reference is not None:
            _ref = np.asarray(reference, dtype=np.float32)
            if _ref.ndim == 2 and _mix.ndim == 1:
                _mix = np.stack([_mix, _mix], axis=0)
            elif _ref.ndim == 1 and _mix.ndim == 2:
                _mix = np.mean(_mix, axis=0, dtype=np.float32)
            _mix = np.asarray(_mix, dtype=_ref.dtype)
        _result: np.ndarray = np.asarray(_mix, dtype=np.float32)
        return _result

    def as_dict(self) -> dict[str, Any]:
        """Leichtgewichtige Zusammenfassung (ohne Audio-Arrays)."""
        return {
            "sample_rate": self.sample_rate,
            "panns_singing": float(self.panns_singing),
            "separation_model": self.separation_model,
            "applied_stages": list(self.applied_stages),
            "witness_reports": {k: dict(v) if isinstance(v, dict) else v for k, v in self.witness_reports.items()},
            "vocal_shape": tuple(int(x) for x in self.vocal_stem.shape),
            "instrumental_shape": tuple(int(x) for x in self.instrumental_stem.shape),
        }
