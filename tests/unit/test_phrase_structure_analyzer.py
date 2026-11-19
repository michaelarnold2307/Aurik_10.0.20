"""tests/unit/test_phrase_structure_analyzer.py — §v10.700 I5 / §2.69c Unit-Tests.

Testet den PhraseStructureAnalyzer: Sektions-Abdeckung, Monotonie,
Layout-Invariante §V7 (copilot-instructions.md) ((C, N) ≙ (N, C)),
Kurz-Audio, Determinismus, get_section_at. Synthetische Signale,
kein Datei-I/O.
"""

import numpy as np

from backend.core.phrase_structure_analyzer import PhraseStructureAnalyzer

SR = 48000


def _music_like(n: int, rng: np.random.Generator) -> np.ndarray:
    """Grobes Musik-Signal mit lauter Mittelzone (Verse/Chorus-Energiekontrast)."""
    t = np.arange(n) / SR
    base = 0.25 * np.sin(2 * np.pi * 220.0 * t) + 0.15 * np.sin(2 * np.pi * 440.0 * t)
    envelope = np.ones(n)
    mid = slice(n // 3, 2 * n // 3)
    envelope[mid] = 1.8
    return (base * envelope + 0.03 * rng.standard_normal(n)).astype(np.float32)


def test_sections_cover_duration_sorted_non_overlapping():
    rng = np.random.default_rng(11)
    n = 12 * SR
    mono = _music_like(n, rng)
    struct = PhraseStructureAnalyzer(sample_rate=SR).analyze(mono, sr=SR)
    assert len(struct.sections) >= 2
    starts = [s.start_s for s in struct.sections]
    ends = [s.end_s for s in struct.sections]
    assert starts == sorted(starts)
    assert ends == sorted(ends)
    for i in range(len(struct.sections) - 1):
        assert struct.sections[i].end_s <= struct.sections[i + 1].start_s + 1e-6
    assert starts[0] == 0.0
    assert abs(ends[-1] - (n / SR)) <= 4.0 + 1e-6  # Ende ≤ Duration + Segmentraster


def test_layout_invariance_channels_first_vs_samples_first():
    rng = np.random.default_rng(12)
    n = 8 * SR
    mono = _music_like(n, rng)
    cf = np.stack([mono, 0.9 * mono], axis=0).astype(np.float32)  # (C, N)
    sf = np.ascontiguousarray(cf.T)  # (N, C)
    a = PhraseStructureAnalyzer(sample_rate=SR).analyze(cf, sr=SR)
    b = PhraseStructureAnalyzer(sample_rate=SR).analyze(sf, sr=SR)
    assert [s.start_s for s in a.sections] == [s.start_s for s in b.sections]
    assert [s.end_s for s in a.sections] == [s.end_s for s in b.sections]
    assert a.bpm == b.bpm


def test_short_audio_single_section():
    rng = np.random.default_rng(13)
    mono = _music_like(SR // 2, rng)  # 0.5 s < segment_s
    struct = PhraseStructureAnalyzer(sample_rate=SR).analyze(mono, sr=SR)
    assert len(struct.sections) == 1
    assert struct.sections[0].start_s == 0.0


def test_deterministic():
    rng = np.random.default_rng(14)
    mono = _music_like(6 * SR, rng)
    a = PhraseStructureAnalyzer(sample_rate=SR).analyze(mono, sr=SR)
    b = PhraseStructureAnalyzer(sample_rate=SR).analyze(mono.copy(), sr=SR)
    assert [(s.label, s.start_s, s.end_s, s.confidence) for s in a.sections] == [
        (s.label, s.start_s, s.end_s, s.confidence) for s in b.sections
    ]
    assert a.bpm == b.bpm


def test_get_section_at():
    rng = np.random.default_rng(15)
    mono = _music_like(8 * SR, rng)
    struct = PhraseStructureAnalyzer(sample_rate=SR).analyze(mono, sr=SR)
    sec = struct.get_section_at(struct.sections[0].start_s)
    assert sec is not None and sec.label == struct.sections[0].label
    assert struct.get_section_at(-5.0) is None
    assert struct.get_section_at(1e9) is None
