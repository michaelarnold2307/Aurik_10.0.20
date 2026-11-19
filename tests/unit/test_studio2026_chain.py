"""Unit tests for studio2026_chain.py"""

import numpy as np
import pytest

_SR = 48000


def _audio(dur=2.0, stereo=True):
    t = np.arange(int(dur * _SR), dtype=np.float32) / _SR
    m = np.sin(2 * np.pi * 440 * t) * 0.3 + np.sin(2 * np.pi * 554 * t) * 0.2
    m = m.astype(np.float32)
    return np.stack([m, m * 0.9], axis=1) if stereo else m


@pytest.mark.unit
class TestStudio2026Chain:
    def test_01_init(self):
        from backend.core.studio2026_chain import Studio2026Result, reprocess_studio2026

        a = _audio(1.0)
        r = reprocess_studio2026(a, a.copy(), _SR, dry_run=True)
        assert r.stages_applied == ["dry_run"]
        assert isinstance(r, Studio2026Result)

    def test_02_all_stages_stereo(self):
        from backend.core.studio2026_chain import reprocess_studio2026

        a = _audio(1.5)
        r = reprocess_studio2026(a, a.copy(), _SR, material="tape")
        assert len(r.stages_applied) >= 6
        assert "freq_stereo" in r.stages_applied

    def test_03_all_stages_mono(self):
        from backend.core.studio2026_chain import reprocess_studio2026

        a = _audio(1.0, False)
        r = reprocess_studio2026(a, a.copy(), _SR)
        assert "freq_stereo" not in r.stages_applied

    def test_04_no_nan_inf(self):
        from backend.core.studio2026_chain import reprocess_studio2026

        a = _audio(1.0)
        r = reprocess_studio2026(a, a.copy(), _SR)
        assert not np.any(np.isnan(r.audio))
        assert not np.any(np.isinf(r.audio))

    def test_05_length_preserved(self):
        from backend.core.studio2026_chain import reprocess_studio2026

        a = _audio(1.5)
        r = reprocess_studio2026(a, a.copy(), _SR)
        assert len(r.audio) == len(a)

    def test_06_voiceprint_tracked(self):
        from backend.core.studio2026_chain import reprocess_studio2026

        a = _audio(1.0)
        r = reprocess_studio2026(a, a.copy(), _SR)
        assert 0 <= r.voiceprint_match <= 1.0

    def test_07_groove_tracked(self):
        from backend.core.studio2026_chain import reprocess_studio2026

        a = _audio(1.0)
        r = reprocess_studio2026(a, a.copy(), _SR)
        assert 0 <= r.groove_preserved <= 1.0

    def test_08_album_ref_accepted(self):
        from backend.core.studio2026_chain import reprocess_studio2026

        a = _audio(1.0)
        r = reprocess_studio2026(a, a.copy(), _SR, album_ref={"lufs_median": -16.0})
        assert len(r.stages_applied) >= 1

    def test_09_shellac_skips_sub_bass(self):
        from backend.core.studio2026_chain import reprocess_studio2026

        a = _audio(1.0)
        r = reprocess_studio2026(a, a.copy(), _SR, material="shellac")
        assert "sub_bass_synth" not in r.stages_applied

    def test_10_deterministic(self):
        from backend.core.studio2026_chain import reprocess_studio2026

        a = _audio(0.8)
        r1 = reprocess_studio2026(a, a.copy(), _SR)
        r2 = reprocess_studio2026(a, a.copy(), _SR)
        assert np.allclose(r1.audio, r2.audio, atol=1e-6)
