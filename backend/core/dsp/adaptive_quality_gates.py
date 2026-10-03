"""
Aurik 10.0.0.x — Adaptive Quality Gates
Material- und genre-adaptive Prüfgrenzen für Pipeline-Qualitätskontrolle.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Material-adaptive HF thresholds (§6.2 Material-MOS)
_MATERIAL_HF_THRESHOLDS: dict[str, tuple[float, float]] = {
    "shellac": (0.05, 0.20),  # sehr wenig HF erwartet
    "vinyl": (0.08, 0.28),
    "tape": (0.10, 0.32),
    "cassette": (0.10, 0.30),
    "cd_digital": (0.15, 0.40),
    "dat": (0.15, 0.40),
    "mp3_high": (0.12, 0.35),
    "aac": (0.12, 0.35),
}

# Genre-adaptive HF thresholds
_GENRE_HF_THRESHOLDS: dict[str, tuple[float, float]] = {
    "default": (0.15, 0.35),
    "klassik": (0.10, 0.25),
    "pop": (0.18, 0.40),
    "jazz": (0.12, 0.30),
    "electronic": (0.20, 0.45),
    "rock": (0.18, 0.40),
    "schlager": (0.14, 0.35),
}


def adaptive_hf_gate(
    hf_ratio: float,
    style: str = "default",
    material: str | None = None,
) -> bool:
    """HF-ratio quality gate, adaptiv nach Material und Genre.

    Args:
        hf_ratio: HF energy ratio (0.0-1.0)
        style: Genre/style identifier
        material: Material type (shellac, vinyl, tape, cd_digital, etc.)

    Returns:
        True if HF ratio is within acceptable range
    """
    # Material takes priority over genre if available
    if material and material in _MATERIAL_HF_THRESHOLDS:
        low, high = _MATERIAL_HF_THRESHOLDS[material]
    else:
        low, high = _GENRE_HF_THRESHOLDS.get(style, _GENRE_HF_THRESHOLDS["default"])
    return low <= hf_ratio <= high


def adaptive_corr_gate(
    corr: float,
    min_corr: float = 0.98,
    material: str | None = None,
) -> bool:
    """Correlation quality gate with material adaptation.

    For analog materials (shellac, vinyl, tape), the minimum correlation
    threshold is relaxed since analog processing introduces more variation.
    """
    corr = float(np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=0.0))
    # Analog materials: relax threshold
    if material in ("shellac", "vinyl", "tape", "cassette"):
        min_corr = min(min_corr, 0.95)
    return corr >= min_corr


def adaptive_snr_gate(
    snr_db: float,
    material: str | None = None,
) -> bool:
    """SNR quality gate with material-adaptive thresholds.

    Returns True if SNR is above the material-specific minimum.
    """
    # Material-specific SNR expectations
    thresholds = {
        "shellac": 20.0,
        "vinyl": 30.0,
        "tape": 35.0,
        "cassette": 30.0,
        "cd_digital": 60.0,
        "dat": 60.0,
        "mp3_high": 50.0,
    }
    min_snr = thresholds.get(material or "", 25.0)
    return snr_db >= min_snr


def adaptive_mos_gate(
    mos: float,
    material: str | None = None,
) -> bool:
    """MOS quality gate per §8.1 Material-MOS thresholds.

    MOS >= 4.5 only for cd_digital/dat/mp3_high/aac.
    Shellac >= 3.8, Vinyl >= 4.0, Tape >= 4.2.
    """
    thresholds = {
        "shellac": 3.8,
        "vinyl": 4.0,
        "tape": 4.2,
        "cassette": 4.0,
        "cd_digital": 4.5,
        "dat": 4.5,
        "mp3_high": 4.5,
        "aac": 4.5,
    }
    min_mos = thresholds.get(material or "", 3.8)
    return mos >= min_mos
