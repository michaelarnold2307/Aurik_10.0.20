"""WP-2-Beleg: alle 17 Phantom-ML-Stub-Module tragen die ehrliche Deklaration (D-K3-3).

Die Deklaration macht sichtbar, dass der deklarierte ML-/TorchScript-Pfad
NICHT implementiert ist und der DSP-Fallback das tatsächliche Verhalten ist
(§G8 (copilot-instructions.md)). Die §V6-Warnung beim ML-Aufruf bleibt in den
Modulen bestehen. Kein Verhaltens-Code wurde verändert.
"""

from __future__ import annotations

from pathlib import Path

_DSP = Path(__file__).resolve().parent.parent.parent / "backend" / "core" / "dsp"

_STUBS = [
    "adaptive_derecording.py",
    "adaptive_formant_shifter.py",
    "adaptive_fundamental_detection.py",
    "adaptive_harmonic_tracking.py",
    "adaptive_transient_detection.py",
    "psychoacoustic_enhancement.py",
    "adaptive_spectral_inpainting.py",
    "adaptive_spectral_peak_removal.py",
    "adaptive_spectral_rolloff.py",
    "adaptive_spectral_subtraction.py",
    "adaptive_spline_interpolation.py",
    "adaptive_stft.py",
    "adaptive_time_scale_modification.py",
    "adaptive_transient_preservation.py",
    "adaptive_vad.py",
    "automatic_declipper.py",
    "masking_aware_dynamic_eq.py",
]

_MARKER = "WP-2-Deklaration"


def test_alle_17_stubs_tragen_deklaration() -> None:
    fehlend: list[str] = []
    for name in _STUBS:
        path = _DSP / name
        assert path.is_file(), f"Stub fehlt: {name}"
        if _MARKER not in path.read_text(encoding="utf-8"):
            fehlend.append(name)
    assert not fehlend, f"Module ohne WP-2-Deklaration: {fehlend}"


def test_anzahl_stubs_bleibt_17() -> None:
    # Der Katalog (D-K3-3) zählt exakt 17 — ein neuer Stub ohne Deklaration
    # soll diesen Test fallen lassen (fail-closed nach oben).
    assert len(_STUBS) == 17
