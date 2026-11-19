from __future__ import annotations

import numpy as np
import pytest

from backend.core.phases import phase_55_diffusion_inpainting as phase55


@pytest.mark.unit
def test_process_channel_skips_ml_plugins_on_thrashing(monkeypatch):
    audio = np.full(512, 0.02, dtype=np.float32)
    audio[200:260] = 0.0

    calls = {"cqtdiff": 0, "flow": 0}

    monkeypatch.setattr(phase55, "_detect_gaps", lambda *_args, **_kwargs: [(200, 260)])
    monkeypatch.setattr(phase55, "_is_ml_thrashing", lambda: True)

    def _cqtdiff(*_args, **_kwargs):
        calls["cqtdiff"] += 1
        return np.full(60, 0.5, dtype=np.float32)

    def _flow(*_args, **_kwargs):
        calls["flow"] += 1
        return np.full(60, 0.5, dtype=np.float32)

    monkeypatch.setattr(phase55, "_try_cqtdiff_plus_plugin", _cqtdiff)
    monkeypatch.setattr(phase55, "_try_flow_matching_plugin", _flow)
    monkeypatch.setattr(phase55, "_inpaint_gap_dsp", lambda *_args, **_kwargs: np.zeros(60, dtype=np.float32))

    _repaired, stats = phase55._process_channel(audio, 48000, 20.0)

    assert calls["cqtdiff"] == 0
    assert calls["flow"] == 0
    assert stats["ml_thrashing_guard"] is True


def test_process_channel_damage_guard_replaces_risky_candidate(monkeypatch):
    audio = np.full(512, 0.02, dtype=np.float32)
    audio[200:260] = 0.0

    monkeypatch.setattr(phase55, "_detect_gaps", lambda *_args, **_kwargs: [(200, 260)])
    monkeypatch.setattr(phase55, "_is_ml_thrashing", lambda: True)
    monkeypatch.setattr(phase55, "_inpaint_gap_dsp", lambda *_args, **_kwargs: np.ones(60, dtype=np.float32))

    repaired, stats = phase55._process_channel(audio, 48000, 20.0)

    repaired_gap = repaired[200:260]
    assert stats["damage_guard_activations"] >= 1
    assert float(np.max(np.abs(repaired_gap))) < 0.2


def test_phase55_metadata_contains_damage_and_thrash_guards(monkeypatch):
    audio = np.full(512, 0.02, dtype=np.float32)
    audio[200:260] = 0.0

    monkeypatch.setattr(phase55, "_detect_gaps", lambda *_args, **_kwargs: [(200, 260)])
    monkeypatch.setattr(phase55, "_is_ml_thrashing", lambda: True)
    monkeypatch.setattr(phase55, "_inpaint_gap_dsp", lambda *_args, **_kwargs: np.ones(60, dtype=np.float32))

    phase = phase55.DiffusionInpaintingPhase()
    result = phase.process(audio, 48000)

    assert result.success is True
    assert int(result.metadata.get("damage_guard_activations", 0)) >= 1
    assert bool(result.metadata.get("ml_thrashing_guard", False)) is True


def test_fadeout_not_detected_as_dropout():
    """Regression: musikalischer Fadeout darf NICHT als Transport-Dropout erkannt werden.

    Bug: trailing silence nach graduellem Fadeout wurde als Gap klassifiziert →
    _conservative_boundary_fill füllte mit dem letzten non-zero Wert → 'Stille explodiert'.
    Fix: Fadeout-Slope-Check (-0.5 dB/frame) verhindert False-Positive-Erkennung.
    """
    sr = 48000
    # 0.5 s Musik (0.3 fadein auf 0.5), dann gradual fadeout über 0.3 s, dann 0.2 s Stille
    # Typisches Fadeout-Profil: Level sinkt langsam von 0.3 auf 0.0.
    n_music = int(0.5 * sr)
    n_fade = int(0.3 * sr)
    n_silence = int(0.2 * sr)

    t_music = np.ones(n_music, dtype=np.float32) * 0.3
    t_fade = np.linspace(0.3, 0.0, n_fade, dtype=np.float32)  # gradual decline
    t_silence = np.zeros(n_silence, dtype=np.float32)
    audio = np.concatenate([t_music, t_fade, t_silence]).astype(np.float32)

    # _detect_gaps soll keine Gaps finden — die trailing silence ist ein Fadeout
    from backend.core.phases.phase_55_diffusion_inpainting import _detect_gaps

    gaps = _detect_gaps(audio, sr, min_gap_ms=5.0)

    # Kein Dropout darf erkannt werden (die "Lücke" ist musikaler Fadeout + Stille)
    assert len(gaps) == 0, (
        f"Fadeout fälschlich als Dropout erkannt: gaps={gaps}. "
        "Root-Cause: fehlender Slope-Check in _detect_gaps trailing-gap-Block."
    )


def test_phase55_stereo_channel_first_axis_guard():
    """Regression: UV3 übergibt Audio als (2, N) channel-first; Phase_55 muss (N, 2) erwarten.

    Bug: `for ch in range(audio.shape[1])` bei (2, N) iteriert N-mal statt 2-mal →
    94K+ Thrashing-Warnungen + leere Gap-Detektion (2-Sample Arrays haben keine Gaps).
    Fix: Achsen-Guard transponiert (2,N)→(N,2) am Eingang und zurück am Ausgang.
    """
    sr = 48000
    n = 4800  # 0.1 s
    # Stereo-Audio im UV3-Format: (channels, samples) = (2, N)
    audio_channel_first = np.random.default_rng(42).random((2, n)).astype(np.float32) * 0.1

    phase = phase55.DiffusionInpaintingPhase()
    result = phase.process(audio_channel_first, sr)

    assert result.success is True
    # Output muss dieselbe Form wie Input haben: (2, N)
    assert result.audio.shape == audio_channel_first.shape, (
        f"Achsen-Guard-Fehler: Input {audio_channel_first.shape} → Output {result.audio.shape}. "
        "Erwartet: (2, N) channel-first bleibt erhalten."
    )


def test_phase55_full_nmf_fallback_when_all_channel_paths_fail(monkeypatch):
    """Wenn alle Kanalpfade ausfallen, muss der Ganzsignal-NMF-Fallback greifen."""
    sr = 48000
    audio = np.zeros(2048, dtype=np.float32)

    def _raise_channel(*_args, **_kwargs):
        raise RuntimeError("forced channel failure")

    monkeypatch.setattr(phase55, "_process_channel", _raise_channel)

    called = {"count": 0}

    def _fake_full_fallback(self, signal, _sr, strength=0.5):
        called["count"] += 1
        return np.asarray(signal, dtype=np.float32)

    monkeypatch.setattr(phase55.DiffusionInpaintingPhase, "_nmf_spectral_inpainting_fallback", _fake_full_fallback)

    phase = phase55.DiffusionInpaintingPhase()
    result = phase.process(audio, sr)

    assert result.success is True
    assert called["count"] == 1
    assert bool(result.metadata.get("full_nmf_fallback_used", False)) is True
    assert int(result.metadata.get("channel_failures", 0)) >= 1


def test_phase55_nmf_fallback_stereo_preserves_signed_ratio():
    """Stereo-Ratio darf negative Rekonstruktion nicht auf 0 wegclippen."""
    t = np.linspace(0.0, 0.1, int(0.1 * 48000), endpoint=False, dtype=np.float32)
    left = 0.2 * np.sin(2.0 * np.pi * 220.0 * t)
    right = 0.2 * np.sin(2.0 * np.pi * 330.0 * t)
    audio = np.column_stack([left, right]).astype(np.float32)

    mono = np.mean(audio, axis=1).astype(np.float32)
    mono_repaired = -mono  # explizit negatives Verhaeltnis
    repaired = phase55._apply_shared_stereo_ratio(audio, mono, mono_repaired)

    assert repaired.shape == audio.shape
    assert np.all(np.isfinite(repaired))
    # Beide Kanaele werden mit demselben signed ratio skaliert; Verhaeltnis bleibt stabil.
    mask = np.abs(audio[:, 0]) > 1e-4
    ratio_left = repaired[mask, 0] / audio[mask, 0]
    ratio_right = repaired[mask, 1] / audio[mask, 1]
    assert np.all(np.isfinite(ratio_left))
    assert np.all(np.isfinite(ratio_right))
    assert float(np.median(np.abs(ratio_left - ratio_right))) < 1e-3
    # Ohne signed clipping waere ein harter Null-Abschnitt zu erwarten.
    assert float(np.max(np.abs(repaired))) > 1e-4
