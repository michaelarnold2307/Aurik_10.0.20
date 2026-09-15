"""§SOTA-PSY-A7 (2026-09-15) — Loudness-Cap-Rollout 10/11/40: Tests.

Deckt ``backend/core/dsp/perceptual_loudness_cap.py`` + die Phasen-Verdrahtung:
- kein Cap, wenn die Kurzzeit-Lautheit nicht über die Schwelle steigt
- Cap greift bei STL-Anhebung (proportionaler Blend Richtung Input)
- Headroom: Uniform-Gain (phase_40) ist legitim, Pumping darüber wird gekappt
- Determinismus, NaN-Schutz
- Phasen 10/11/40 melden ``loudness_cap``-Metadatum

Autor: Aurik Testing Team
"""

import numpy as np

from backend.core.dsp.perceptual_loudness_cap import perceptual_loudness_cap


def _audio(n: int = 48000, amp: float = 0.2) -> np.ndarray:
    t = np.linspace(0, n / 48000, n, endpoint=False, dtype=np.float32)
    return (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


class TestPerceptualLoudnessCap:
    def test_no_cap_when_not_raised(self):
        x = _audio()
        y = x * 0.8
        out, meta = perceptual_loudness_cap(x, y, 48000, phase_label="test")
        assert np.array_equal(out, y)
        assert meta["loudness_cap_applied"] is False

    def test_cap_backs_off_when_raised(self):
        x = _audio(amp=0.05)
        y = x * 4.0  # deutliche Lautheits-Anhebung
        out, meta = perceptual_loudness_cap(x, y, 48000, phase_label="test")
        assert meta["loudness_cap_applied"] is True
        assert 0.0 < meta["loudness_cap_wet"] < 1.0
        d_out = float(np.abs(out - x).mean())
        d_y = float(np.abs(y - x).mean())
        assert d_out < d_y

    def test_headroom_allows_uniform_gain(self):
        # Uniform-Gain um +6 dB bleibt erlaubt (phase_40-Ziel-LUFS-Anhebung)
        x = _audio(amp=0.1)
        y = x * 2.0  # +6,02 dB
        out, meta = perceptual_loudness_cap(x, y, 48000, phase_label="test", headroom_lin=2.0)
        assert meta["loudness_cap_applied"] is False
        assert np.allclose(out, y, atol=1e-6)

    def test_headroom_caps_pumping_beyond_uniform_gain(self):
        x = _audio(amp=0.2)
        y = x * 6.0  # +15,6 dB ≫ Headroom 1,5 (+3,5 dB)
        out, meta = perceptual_loudness_cap(x, y, 48000, phase_label="test", headroom_lin=1.5)
        assert meta["loudness_cap_applied"] is True
        d_out = float(np.abs(out - x).mean())
        d_y = float(np.abs(y - x).mean())
        assert d_out < d_y

    def test_deterministic(self):
        x = _audio(24000, amp=0.05)
        y = x * 3.0
        o1, _ = perceptual_loudness_cap(x, y, 48000)
        o2, _ = perceptual_loudness_cap(x, y, 48000)
        assert np.array_equal(o1, o2)

    def test_nan_input_guarded(self):
        x = _audio(16000)
        y = (x * 2.0).copy()
        y[100] = np.nan
        out, _ = perceptual_loudness_cap(x, y, 48000)
        assert np.isfinite(out).all()


class TestRolloutWiring:
    def test_phases_report_loudness_cap_metadata(self):
        from backend.core.defect_scanner import MaterialType

        x = _audio()
        cases = [
            ("phase_10_compression", "CompressionPhase"),
            ("phase_11_limiting", "LimitingPhase"),
            ("phase_40_loudness_normalization", "LoudnessNormalizationPhase"),
        ]
        import importlib

        for mod, cls in cases:
            m = importlib.import_module(f"backend.core.phases.{mod}")
            ph = getattr(m, cls)()
            res = ph.process(x, sample_rate=48000, material_type=MaterialType.VINYL)
            if mod == "phase_11_limiting" and res.metadata.get("limiting_applied") is False:
                # Early-Exit: TP ≤ Ceiling ⇒ unverändertes Audio — Guard bewusst unnötig.
                continue
            lc_meta = res.metadata.get("loudness_cap")
            assert lc_meta is not None, f"{mod} meldet kein loudness_cap-Metadatum"
            assert isinstance(lc_meta.get("loudness_cap_applied"), bool)
            assert np.isfinite(np.asarray(res.audio)).all()
            assert len(np.asarray(res.audio).ravel()) == len(x)
