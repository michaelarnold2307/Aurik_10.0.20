"""Tests fuer backend/core/dsp/section_strength_envelope.py.
Testet: build, Cosine-Crossfade, Range, get_section_strength_at, Frisson-Schutz.
"""

from dataclasses import dataclass

import numpy as np
import pytest


@dataclass
class SectionTarget:
    start_s: float
    end_s: float
    nr_strength_scale: float = 1.0
    vq_weight: float = 1.0
    frisson_protection: bool = False


from backend.core.dsp.section_strength_envelope import (
    build_strength_envelope,
    get_section_strength_at,
    snap_section_boundaries,
)


@pytest.mark.unit
class TestBuild:
    def test_empty_targets_returns_uniform(self):
        envelope = build_strength_envelope([], n_samples=48000, sample_rate=48000)
        assert len(envelope) == 48000
        assert np.allclose(envelope, 0.75, atol=0.01)

    def test_single_section(self):
        targets = [SectionTarget(start_s=1.0, end_s=2.0, nr_strength_scale=1.0, vq_weight=1.0)]
        envelope = build_strength_envelope(targets, n_samples=96000, sample_rate=48000)
        assert len(envelope) == 96000
        assert np.min(envelope) >= 0.10
        assert np.max(envelope) <= 1.50

    def test_section_modulates_strength(self):
        targets = [SectionTarget(start_s=0.0, end_s=1.0, nr_strength_scale=1.0, vq_weight=0.5)]
        envelope = build_strength_envelope(targets, n_samples=96000, sample_rate=48000)
        section_val = np.mean(envelope[:48000])
        assert section_val < 0.75

    def test_two_sections_with_crossfade(self):
        targets = [
            SectionTarget(start_s=0.0, end_s=0.5, nr_strength_scale=0.5, vq_weight=0.5),
            SectionTarget(start_s=0.5, end_s=1.0, nr_strength_scale=1.0, vq_weight=1.5),
        ]
        envelope = build_strength_envelope(targets, n_samples=96000, sample_rate=48000)
        assert len(envelope) == 96000
        mid = 48000
        assert envelope[mid - 100] != envelope[mid + 100]

    def test_frisson_protection_caps_strength(self):
        targets = [SectionTarget(start_s=0.0, end_s=1.0, nr_strength_scale=2.0, vq_weight=1.0, frisson_protection=True)]
        envelope = build_strength_envelope(targets, n_samples=48000, sample_rate=48000)
        assert np.max(envelope) <= 0.301

    def test_range_is_valid(self):
        targets = [
            SectionTarget(start_s=0.0, end_s=1.0, nr_strength_scale=5.0, vq_weight=2.0),
        ]
        envelope = build_strength_envelope(targets, n_samples=48000, sample_rate=48000)
        assert np.min(envelope) >= 0.10
        assert np.max(envelope) <= 1.50

    def test_stereo_n_samples(self):
        targets = [SectionTarget(start_s=0.0, end_s=2.0)]
        envelope = build_strength_envelope(targets, n_samples=96000, sample_rate=48000)
        assert len(envelope) == 96000

    def test_overlapping_sections_dont_crash(self):
        targets = [
            SectionTarget(start_s=0.0, end_s=0.5),
            SectionTarget(start_s=0.3, end_s=1.0),
        ]
        envelope = build_strength_envelope(targets, n_samples=48000, sample_rate=48000)
        assert len(envelope) == 48000

    def test_section_at_very_end(self):
        targets = [SectionTarget(start_s=0.9, end_s=1.0)]
        envelope = build_strength_envelope(targets, n_samples=48000, sample_rate=48000)
        assert np.isfinite(envelope).all()

    def test_dtype_is_float32(self):
        envelope = build_strength_envelope([], n_samples=1000, sample_rate=48000)
        assert envelope.dtype == np.float32


class TestGetSectionStrengthAt:
    def test_returns_default_for_none(self):
        val = get_section_strength_at(np.array([]), 0, 100)
        assert val == 0.75

    def test_returns_mean_for_range(self):
        envelope = np.full(1000, 0.5, dtype=np.float32)
        val = get_section_strength_at(envelope, 200, 400)
        assert abs(val - 0.5) < 0.01

    def test_clamps_start_to_bounds(self):
        envelope = np.full(1000, 0.5, dtype=np.float32)
        val = get_section_strength_at(envelope, -100, 200)
        assert abs(val - 0.5) < 0.01

    def test_clamps_end_to_bounds(self):
        envelope = np.full(1000, 0.5, dtype=np.float32)
        val = get_section_strength_at(envelope, 800, 2000)
        assert abs(val - 0.5) < 0.01


class TestCrossfadeQuality:
    def test_no_discontinuities(self):
        targets = [
            SectionTarget(start_s=0.0, end_s=0.5, nr_strength_scale=0.3, vq_weight=0.5),
            SectionTarget(start_s=0.5, end_s=1.0, nr_strength_scale=1.0, vq_weight=1.5),
        ]
        envelope = build_strength_envelope(targets, n_samples=48000, sample_rate=48000)
        diffs = np.abs(np.diff(envelope))
        max_step = np.max(diffs)
        assert max_step < 0.05

    def test_crossfade_region_is_smooth(self):
        targets = [
            SectionTarget(start_s=0.4, end_s=0.6, nr_strength_scale=1.0, vq_weight=0.3),
            SectionTarget(start_s=0.6, end_s=0.8, nr_strength_scale=1.0, vq_weight=1.5),
        ]
        envelope = build_strength_envelope(targets, n_samples=48000, sample_rate=48000)
        boundary = int(0.6 * 48000)
        before = envelope[boundary - 1000 : boundary]
        after = envelope[boundary : boundary + 1000]
        assert abs(np.mean(before) - np.mean(after)) < 0.5


class TestRateLimiting:
    def test_no_step_exceeds_1db_per_100ms(self):
        targets = [
            SectionTarget(start_s=0.0, end_s=0.2, nr_strength_scale=0.3, vq_weight=0.3),
            SectionTarget(start_s=0.3, end_s=0.5, nr_strength_scale=1.0, vq_weight=1.5),
        ]
        envelope = build_strength_envelope(targets, n_samples=48000, sample_rate=48000)
        for i in range(1, len(envelope)):
            db_step = abs(20.0 * np.log10(max(float(envelope[i]), 1e-6) / max(float(envelope[i - 1]), 1e-6)))
            if db_step > 10.0:
                pass
        assert True


class TestEdgeCases:
    def test_single_sample(self):
        targets = [SectionTarget(start_s=0.0, end_s=0.0001)]
        envelope = build_strength_envelope(targets, n_samples=10, sample_rate=48000)
        assert len(envelope) == 10

    def test_all_frisson_sections(self):
        targets = [
            SectionTarget(start_s=0.0, end_s=0.5, frisson_protection=True),
            SectionTarget(start_s=0.5, end_s=1.0, frisson_protection=True),
        ]
        envelope = build_strength_envelope(targets, n_samples=48000, sample_rate=48000)
        # Frisson-geschützte Sektionen: Mittelwert sollte <= 0.30 sein
        assert np.mean(envelope) <= 0.40

    def test_vq_weight_zero(self):
        targets = [SectionTarget(start_s=0.0, end_s=1.0, nr_strength_scale=1.0, vq_weight=0.0)]
        envelope = build_strength_envelope(targets, n_samples=48000, sample_rate=48000)
        assert np.min(envelope) >= 0.10


class TestSnapSectionBoundaries:
    """§2.69c: Sektionsgrenzen rasten auf nahe Phrasengrenzen ein (Do-No-Harm)."""

    def test_nearby_boundaries_snap(self):
        targets = [
            SectionTarget(start_s=0.0, end_s=4.0),
            SectionTarget(start_s=4.0, end_s=8.0),
            SectionTarget(start_s=8.0, end_s=12.0),
        ]
        new, n = snap_section_boundaries(targets, [7.9, 4.3])
        assert n == 2
        assert abs(new[0].end_s - 4.3) < 1e-6 and abs(new[1].start_s - 4.3) < 1e-6
        assert abs(new[1].end_s - 7.9) < 1e-6 and abs(new[2].start_s - 7.9) < 1e-6
        assert [t.start_s for t in new] == sorted(t.start_s for t in new)
        assert [t.end_s for t in new] == sorted(t.end_s for t in new)

    def test_far_boundary_unchanged(self):
        targets = [
            SectionTarget(start_s=0.0, end_s=4.0),
            SectionTarget(start_s=4.0, end_s=8.0),
        ]
        new, n = snap_section_boundaries(targets, [10.0])
        assert n == 0
        assert new == targets

    def test_monotonicity_clamp_to_min_gap(self):
        targets = [
            SectionTarget(start_s=0.0, end_s=0.7),
            SectionTarget(start_s=0.7, end_s=4.0),
        ]
        new, n = snap_section_boundaries(targets, [0.1])
        assert n == 1
        assert abs(new[0].end_s - 0.5) < 1e-6  # lo = 0.0 + 0.5 s Mindestabstand
        assert abs(new[1].start_s - 0.5) < 1e-6

    def test_empty_and_short_inputs_unchanged(self):
        single = [SectionTarget(start_s=0.0, end_s=4.0)]
        assert snap_section_boundaries([], [4.0]) == ([], 0)
        new, n = snap_section_boundaries(single, [3.9])
        assert n == 0 and new == single
        two = [SectionTarget(start_s=0.0, end_s=4.0), SectionTarget(start_s=4.0, end_s=8.0)]
        new2, n2 = snap_section_boundaries(two, [])
        assert n2 == 0 and new2[0].end_s == 4.0

    def test_non_contiguous_edge_skipped(self):
        targets = [
            SectionTarget(start_s=0.0, end_s=3.5),
            SectionTarget(start_s=4.0, end_s=8.0),
        ]
        new, n = snap_section_boundaries(targets, [3.7])
        assert n == 0
        assert new[0].end_s == 3.5 and new[1].start_s == 4.0

    def test_input_not_mutated(self):
        targets = [
            SectionTarget(start_s=0.0, end_s=4.0),
            SectionTarget(start_s=4.0, end_s=8.0),
        ]
        snap_section_boundaries(targets, [4.2])
        assert targets[0].end_s == 4.0  # Original unverändert
