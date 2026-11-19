from __future__ import annotations

"""
Tests für §0 BW-Cap-Invariante in phase_55_diffusion_inpainting.

Normative Anforderungen (§0 Primum non nocere, §2.55 VERBOTEN: Inpainting HF-Halluzination):
- wax_cylinder: BW-Cap ≤ 5 kHz — rekonstruierte Inhalte dürfen keine HF über 5 kHz enthalten.
- wire_recording: BW-Cap ≤ 6 kHz.
- shellac: BW-Cap ≤ 7 kHz (§0 Vintage Aesthetics).
- lacquer_disc: BW-Cap ≤ 8 kHz.
- VERBOTEN: AR/Diffusion ohne BW-Begrenzung für historische Träger.
"""


import numpy as np
import pytest


@pytest.fixture
def sr():
    return 48000


def _spectral_energy_above_hz(audio: np.ndarray, sr: int, cutoff_hz: float) -> float:
    """Gibt Energie-Anteil des Signals oberhalb cutoff_hz zurück."""
    mono = audio if audio.ndim == 1 else audio[0]
    n_fft = 4096
    if len(mono) < 64:
        return 0.0
    n = min(n_fft, len(mono))
    win = np.hanning(n).astype(np.float32)
    spec = np.abs(np.fft.rfft(mono[:n] * win, n=n_fft))
    total = float(np.sum(spec**2)) + 1e-12
    freq_bin = int(cutoff_hz / (sr / n_fft))
    above = float(np.sum(spec[freq_bin:] ** 2))
    return above / total


# ── _MATERIAL_BW_CAP_HZ Tabelleninhalt ────────────────────────────────────


@pytest.mark.unit
def test_bw_cap_table_has_wax_cylinder():
    """_MATERIAL_BW_CAP_HZ enthält wax_cylinder ≤ 5 kHz (§0 VERBOTEN-Invariante)."""
    from backend.core.phases.phase_55_diffusion_inpainting import _MATERIAL_BW_CAP_HZ

    assert "wax_cylinder" in _MATERIAL_BW_CAP_HZ
    assert _MATERIAL_BW_CAP_HZ["wax_cylinder"] <= 5000.0, (
        f"wax_cylinder BW-Cap {_MATERIAL_BW_CAP_HZ['wax_cylinder']} Hz überschreitet §0-Maximum 5000 Hz"
    )


def test_bw_cap_table_has_wire_recording():
    """_MATERIAL_BW_CAP_HZ enthält wire_recording ≤ 6 kHz."""
    from backend.core.phases.phase_55_diffusion_inpainting import _MATERIAL_BW_CAP_HZ

    assert "wire_recording" in _MATERIAL_BW_CAP_HZ
    assert _MATERIAL_BW_CAP_HZ["wire_recording"] <= 6000.0, (
        f"wire_recording BW-Cap {_MATERIAL_BW_CAP_HZ['wire_recording']} Hz überschreitet §0-Maximum 6000 Hz"
    )


def test_bw_cap_table_has_shellac():
    """_MATERIAL_BW_CAP_HZ enthält shellac ≤ 7 kHz (§0 Vintage Aesthetics)."""
    from backend.core.phases.phase_55_diffusion_inpainting import _MATERIAL_BW_CAP_HZ

    assert "shellac" in _MATERIAL_BW_CAP_HZ
    assert _MATERIAL_BW_CAP_HZ["shellac"] <= 7000.0


# ── _apply_bw_cap funktioniert ─────────────────────────────────────────────


def test_apply_bw_cap_removes_hf_above_5khz(sr):
    """_apply_bw_cap(seg, 48000, 5000) entfernt Energie über 5 kHz."""
    from backend.core.phases.phase_55_diffusion_inpainting import _apply_bw_cap

    # White noise enthält Energie im gesamten Spektrum
    rng = np.random.default_rng(42)
    white = rng.standard_normal(sr).astype(np.float32) * 0.3
    capped = _apply_bw_cap(white, sr, 5000.0)

    # Nach Cap sollte Energie über 6 kHz signifikant reduziert sein
    energy_above_6k_before = _spectral_energy_above_hz(white, sr, 6000.0)
    energy_above_6k_after = _spectral_energy_above_hz(capped, sr, 6000.0)
    assert energy_above_6k_after < energy_above_6k_before * 0.20, (
        f"BW-Cap 5 kHz hat HF-Energie nicht ausreichend reduziert: "
        f"vorher={energy_above_6k_before:.3f}, nachher={energy_above_6k_after:.3f}"
    )


def test_apply_bw_cap_passes_low_frequencies(sr):
    """_apply_bw_cap behält Energie unter dem Cap-Punkt bei."""
    from backend.core.phases.phase_55_diffusion_inpainting import _apply_bw_cap

    # 440 Hz Ton — sollte von 5 kHz Cap nicht betroffen werden
    t = np.arange(sr) / sr
    tone = (np.sin(2 * np.pi * 440 * t)).astype(np.float32) * 0.8
    capped = _apply_bw_cap(tone, sr, 5000.0)

    rms_before = float(np.sqrt(np.mean(tone**2)))
    rms_after = float(np.sqrt(np.mean(capped**2)))
    # Max. 5 % Verlust durch Filter-Passband-Unschärfe
    assert rms_after > rms_before * 0.95, (
        f"BW-Cap 5 kHz hat 440 Hz-Ton zu stark bedämpft: rms_before={rms_before:.3f} rms_after={rms_after:.3f}"
    )


def test_apply_bw_cap_no_crash_short_segment(sr):
    """_apply_bw_cap verarbeitet sehr kurze Segmente (< 20 Samples) ohne Absturz."""
    from backend.core.phases.phase_55_diffusion_inpainting import _apply_bw_cap

    tiny = np.random.randn(10).astype(np.float32) * 0.1
    result = _apply_bw_cap(tiny, sr, 5000.0)
    assert len(result) == len(tiny)
    assert not np.any(np.isnan(result))


def test_apply_bw_cap_above_nyquist_passthrough(sr):
    """cap_hz ≥ Nyquist → Signal wird nicht verändert (kein unnötiger Filter)."""
    from backend.core.phases.phase_55_diffusion_inpainting import _apply_bw_cap

    white = np.random.randn(sr).astype(np.float32) * 0.1
    result = _apply_bw_cap(white, sr, sr / 2 - 50)  # nahe Nyquist
    # Passthrough: praktisch identisch
    np.testing.assert_array_almost_equal(result, white, decimal=4)


# ── Integration: Phase-Process wendet BW-Cap auf wax_cylinder an ──────────


def test_phase55_applies_bw_cap_for_wax_cylinder(sr):
    """
    §0 VERBOTEN: phase_55 muss BW-Cap für wax_cylinder anwenden.
    Ausgabe darf keine HF-Halluzinationen über 5 kHz enthalten.
    Verwendet ein Signal mit Gap → prüft BW des reparierten Bereichs.
    """
    from backend.core.phases.phase_55_diffusion_inpainting import DiffusionInpaintingPhase

    phase = DiffusionInpaintingPhase()

    # Signal mit absichtlichem Gap und viel HF-Rauschen
    n = sr * 2  # 2 Sekunden
    rng = np.random.default_rng(123)
    audio = rng.standard_normal(n).astype(np.float32) * 0.2

    # Dropout/Silence-Gap einführen (Phase_55 repariert diese)
    gap_start = sr // 2
    gap_end = sr // 2 + int(sr * 0.05)  # 50 ms Gap
    audio[gap_start:gap_end] = 0.0

    result = phase.process(audio, sr, material_type="wax_cylinder")

    # Das reparierte Audio darf keine starke Energie über 5 kHz haben
    if result is not None and result.audio is not None:
        repaired_region = result.audio[gap_start:gap_end]
        if len(repaired_region) > 64:
            energy_above = _spectral_energy_above_hz(repaired_region, sr, 5500.0)
            # Energieanteil über 5.5 kHz sollte minimal sein nach BW-Cap
            assert energy_above < 0.30, (
                f"§0 VERBOTEN: wax_cylinder-Inpainting hat HF-Halluzinationen "
                f"über 5 kHz erzeugt (Energie-Anteil über 5.5 kHz: {energy_above:.3f})"
            )


def test_phase55_no_bw_cap_for_digital_material(sr):
    """
    Digitales Material (cd_digital) hat keinen BW-Cap —
    volle Bandbreite wird rekonstruiert.
    """
    from backend.core.phases.phase_55_diffusion_inpainting import _MATERIAL_BW_CAP_HZ

    assert "cd_digital" not in _MATERIAL_BW_CAP_HZ, (
        "cd_digital darf keinen BW-Cap haben — digitale Quellen haben volle Bandbreite"
    )


def test_bw_cap_hierarchy_preserved():
    """
    Die BW-Caps folgen der Träger-Alters-Hierarchie:
    wax_cylinder < wire_recording < shellac ≤ lacquer_disc.
    """
    from backend.core.phases.phase_55_diffusion_inpainting import _MATERIAL_BW_CAP_HZ

    caps = _MATERIAL_BW_CAP_HZ
    if "wax_cylinder" in caps and "wire_recording" in caps:
        assert caps["wax_cylinder"] <= caps["wire_recording"], (
            "wax_cylinder BW-Cap muss ≤ wire_recording BW-Cap sein (älterer Träger → engere Bandbreite)"
        )
    if "wire_recording" in caps and "shellac" in caps:
        assert caps["wire_recording"] <= caps["shellac"], "wire_recording BW-Cap muss ≤ shellac BW-Cap sein"
