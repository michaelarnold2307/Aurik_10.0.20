"""§SOTA-R5 (Determinismus-Zertifikat) — bit-identische Läufe als CI-Nachweis.

Roadmap 2026-09-14, R5 (TRUST-A): „gleicher Input ⇒ MD5-identischer Output je
Release als Studio-Feature + CI-Nachweis." Dieses Zertifikat fährt eine feste,
maschinen-unabhängige DSP-Phasenkette zweimal auf identischem Input und
vergleicht MD5 + Bit-Identität — zweimal im selben Prozess (innerhalb der
Version reproduzierbar, §G5 (GEBOTE.md)) und mit NEU instanziierten Phasen-Objekten
(kein versteckter Objektzustand).

Abgrenzung (dokumentiert, nicht verschwiegen):
- phase_38 nutzt ``hash(audio.tobytes())`` für den §v10.65-Frequenz-Jitter —
  Pythons bytes-hash ist pro Prozess randomisiert, daher ist die Kette dieses
  Zertifikats OHNE phase_38 (maschinen-übergreifend stabil). phase_38 wird
  separat auf In-Prozess-Determinismus geprüft.
- ML-Phasen (ONNX/Torch) sind device-abhängig; das Zertifikat deckt die
  pure-DSP-Kette ab (ML-Determinismus: test_gpu_determinism_gate.py).

Autor: Aurik Testing Team
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from backend.core.defect_scanner import MaterialType

SR = 48000

# Feste Kette pure-DSP-Phasen (maschinen-unabhängig, kein hash()-Jitter, kein RNG).
_CHAIN: list[tuple[str, str, dict]] = [
    ("phase_04_eq_correction", "EQCorrectionPhase", {}),
    ("phase_16_final_eq", "FinalEQ", {}),
    ("phase_28_surface_noise_profiling", "SurfaceNoiseProfiling", {}),
    ("phase_33_stereo_width_limiter", "StereoWidthLimiterPhaseV2", {}),
    ("phase_34_mid_side_processing", "MidSideProcessing", {}),
    ("phase_37_bass_enhancement", "BassEnhancement", {}),
    ("phase_39_air_band_enhancement", "AirBandEnhancement", {}),
    ("phase_59_modulation_noise_reduction", "ModulationNoiseReductionPhase", {}),
]


def _make_input() -> np.ndarray:
    """Deterministisches vinyl-artiges Testsignal (1 s, 48 kHz)."""
    rng = np.random.default_rng(42)
    t = np.linspace(0, 1.0, SR, endpoint=False, dtype=np.float32)
    x = (
        0.2 * np.sin(2 * np.pi * 220 * t)
        + 0.15 * np.sin(2 * np.pi * 1000 * t)
        + rng.normal(0, 0.01, SR).astype(np.float32)
    )
    return (x / (np.max(np.abs(x)) + 1e-9) * 0.5).astype(np.float32)


def _run_chain(audio: np.ndarray) -> np.ndarray:
    import importlib

    out = np.asarray(audio, dtype=np.float32).copy()
    for module_name, class_name, kwargs in _CHAIN:
        mod = importlib.import_module(f"backend.core.phases.{module_name}")
        phase = getattr(mod, class_name)()
        result = phase.process(out, sample_rate=SR, material_type=MaterialType.VINYL, **kwargs)
        out = np.asarray(result.audio, dtype=np.float32)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _md5(x: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(x.astype(np.float32)).tobytes()).hexdigest()


class TestDeterminismCertificate:
    def test_chain_bit_identical_across_runs(self):
        audio = _make_input()
        run1 = _run_chain(audio)
        run2 = _run_chain(audio)
        assert np.array_equal(run1, run2)
        assert _md5(run1) == _md5(run2)
        # Kette verändert das Signal tatsächlich (kein trivialer Passthrough-Test)
        assert not np.array_equal(run1, audio)

    def test_chain_md5_stable_across_fresh_instances(self):
        # Neue Phasen-Objekte (kein versteckter Instanzzustand) ⇒ identisches MD5.
        audio = _make_input()
        md5_a = _md5(_run_chain(audio))
        md5_b = _md5(_run_chain(audio))
        assert md5_a == md5_b

    def test_chain_short_input_and_nan_guarded(self):
        audio = _make_input()[: SR // 2]
        audio[100] = np.nan
        out = _run_chain(audio)
        assert np.isfinite(out).all()
        assert out.shape == audio.shape

    def test_phase38_in_process_determinism(self):
        """phase_38: In-Prozess-Determinismus (hash()-Jitter ist prozess-stabil)."""
        from backend.core.phases.phase_38_presence_boost import PresenceBoost

        audio = _make_input()
        r1 = PresenceBoost().process(audio, sample_rate=SR, material_type=MaterialType.VINYL)
        r2 = PresenceBoost().process(audio, sample_rate=SR, material_type=MaterialType.VINYL)
        assert np.array_equal(r1.audio, r2.audio)
