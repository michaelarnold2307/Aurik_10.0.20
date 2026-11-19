"""SOTA-M8 (D-K4-1) — Vertragstests des neuronalen A-SPADE-Klassen-Declipper (CHALLENGER zu Phase 07).

Verträge (Auftrag SOTA-M8 D-K4-1):
- Architektur-Shapes (A-SPADE-Klasse: unrolled ADMM + 1-D-CNN-Proximal)
- Determinismus: 2× gleicher Input ⇒ bit-identisch (§G5 (copilot-instructions.md));
  Seed-Fähigkeit: gleicher Seed ⇒ gleiche Gewichte/gleicher Output
- Never-worsen-Harness auf geclipptem Signal (Orakel akzeptiert,
  Verschlechterung bit-identisch abgelehnt)
- Stereo-Layout-Invariante: (2, N) ⇄ (N, 2) bit-identisch (AGENTS.md §3)
- Flag `use_aspade_declipper` AUS ⇒ bit-identisches Bestandsverhalten
  (Challenger wird nie gerufen; keine neuen Metrik-Schlüssel)
- Fehlerpfade: logger.warning + Ersatzpfad (§V6 (copilot-instructions.md)),
  kein stiller Fallback; ungültige Argumente werfen ValueError (laut)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import torch

import backend.core.music_model_flags as mmf
from backend.core.dsp import aspade_declipper as asp
from backend.core.dsp.aspade_declipper import AspadeClassDeclipper, AspadeClassNet, AspadeProximalNet
from backend.core.phases.phase_07_declipper import DeclipperPhase

SR = 48000


def _bursts_with_clean(amp: float = 6.0, dur_s: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """0.8-Sinus mit drei 25-ms-Bursts ×amp — (clean, hart bei ±1.0 geclippt).

    MUSTER: tests/unit/test_phase_07_declipper.py::_clipped_bursts; hier mit
    Ground-Truth-clean für den Never-worsen-Harness (SDR gegen clean).
    """
    n = int(SR * dur_s)
    t = np.linspace(0, 1, n, endpoint=False, dtype=np.float64)
    x = np.sin(2 * np.pi * 440 * t) * 0.8
    burst = int(SR * 0.025)
    for onset_s in (0.25, 0.5, 0.75):
        s0 = int(onset_s * n)
        x[s0 : s0 + burst] *= amp
    clean = x.astype(np.float32)
    clipped = np.clip(x, -1.0, 1.0).astype(np.float32)
    return clean, clipped


def _sdr_db(est: np.ndarray, ref: np.ndarray) -> float:
    """Test-Harness-Metrik (unabhängig vom Modul-Witness): SDR gegen Ground Truth."""
    err = float(np.mean((est.astype(np.float64) - ref.astype(np.float64)) ** 2))
    sig = float(np.mean(ref.astype(np.float64) ** 2))
    return 10.0 * np.log10((sig + 1e-12) / (err + 1e-12))


def _thd(y: np.ndarray) -> float:
    """THD-Verhältnis (Harmonische 2–10 vs. Grundton 440 Hz) — MUSTER test_phase_07_declipper."""
    spec = np.abs(np.fft.rfft(y.astype(np.float64)))
    freqs = np.fft.rfftfreq(len(y), 1.0 / SR)
    fund = float(spec[np.argmin(np.abs(freqs - 440))]) + 1e-12
    harm = 0.0
    for h in range(2, 11):
        i = int(np.argmin(np.abs(freqs - 440 * h)))
        harm += float(spec[i]) ** 2
    return float(np.sqrt(harm) / fund)


# ── Architektur-Shapes ──────────────────────────────────────────────────────


def test_proximal_net_shape_contract() -> None:
    """T_θ (1-D-CNN-Proximal): (B, 1, T) ⇒ (B, 1, T), Parameter vorhanden."""
    prox = AspadeProximalNet(channels=8, kernel_size=7, dilations=(1, 2, 4))
    x = torch.linspace(-1.0, 1.0, 2 * 1 * 2048).reshape(2, 1, 2048)
    out = prox(x)
    assert out.shape == x.shape
    assert sum(p.numel() for p in prox.parameters()) > 0
    assert bool(torch.isfinite(out).all())


def test_klassennetz_shape_und_stufen() -> None:
    """Unrolled ADMM: Stufenzahl == K, Forward (B, 1, T) ⇒ (B, 1, T) (auch ungerade T)."""
    net = AspadeClassNet(stages=3, channels=8, kernel_size=7, dilations=(1, 2, 4))
    assert len(net.stages) == 3
    y = torch.linspace(-1.0, 1.0, 1 * 1 * 2049).reshape(1, 1, 2049)
    theta = torch.tensor([[[0.5]]], dtype=torch.float32)
    out = net(y, theta)
    assert out.shape == y.shape
    assert bool(torch.isfinite(out).all())


def test_klassennetz_datenkonsistenz_ungeclippt() -> None:
    """Ungeclippte Samples bleiben exakt auf dem Eingang (Π_Γ-Projektion)."""
    net = AspadeClassNet(stages=2, channels=8, kernel_size=7, dilations=(1, 2))
    y = torch.linspace(-0.4, 0.4, 1024).reshape(1, 1, 1024)  # |y| < θ=0.5
    theta = torch.tensor([[[0.5]]], dtype=torch.float32)
    out = net(y, theta)
    torch.testing.assert_close(out, y, rtol=0.0, atol=0.0)  # exakt (bit-identisch)


# ── Determinismus (§G5 (copilot-instructions.md)) ───────────────────────────


def test_build_model_seed_reproduzierbar() -> None:
    """Gleicher Seed ⇒ bit-identische Gewichte; anderer Seed ⇒ andere Gewichte."""
    a = asp.build_model(stages=2, channels=8, seed=7)
    b = asp.build_model(stages=2, channels=8, seed=7)
    c = asp.build_model(stages=2, channels=8, seed=8)
    sd_a, sd_b, sd_c = a.state_dict(), b.state_dict(), c.state_dict()
    for key in sd_a:
        assert torch.equal(sd_a[key], sd_b[key]), f"Gewichte weichen bei {key} ab"
    assert any(not torch.equal(sd_a[key], sd_c[key]) for key in sd_a)


def test_inferenz_zwei_laeufe_bit_identisch() -> None:
    """2× gleicher Input ⇒ bit-identischer Output (§G5 (copilot-instructions.md))."""
    _, clipped = _bursts_with_clean(6.0)
    d = AspadeClassDeclipper(stages=2, channels=8, seed=5)
    r1 = d.declip(clipped, SR, 1.0)
    r2 = d.declip(clipped, SR, 1.0)
    np.testing.assert_array_equal(r1, r2)
    assert r1.shape == clipped.shape
    assert r1.dtype == clipped.dtype


def test_seed_faehigkeit_gleicher_seed_gleicher_output() -> None:
    """Zwei Instanzen mit demselben Seed liefern bit-identische Ergebnisse."""
    _, clipped = _bursts_with_clean(6.0)
    d1 = AspadeClassDeclipper(stages=2, channels=8, seed=11)
    d2 = AspadeClassDeclipper(stages=2, channels=8, seed=11)
    np.testing.assert_array_equal(d1.declip(clipped, SR, 1.0), d2.declip(clipped, SR, 1.0))


@pytest.mark.parametrize("chunk_samples", [1, 100, 101])
@pytest.mark.parametrize("extra_samples", [1, 50, 100, 137])
def test_chunked_identity_preserves_all_samples(monkeypatch, chunk_samples: int, extra_samples: int) -> None:
    """Auch ungerade Chunks und unvollständige Endstücke dürfen das Signal nicht ändern."""
    d = AspadeClassDeclipper(stages=1, channels=2, seed=5)
    d._chunk_samples = chunk_samples
    monkeypatch.setattr(d, "_infer_block", lambda segment, _theta: segment.copy())
    x = np.random.default_rng(42).uniform(-0.9, 0.9, chunk_samples + extra_samples).astype(np.float32)
    np.testing.assert_array_equal(d._infer_channel(x, 0.5), x)


def test_chunked_model_preserves_unclipped_samples() -> None:
    """Die ADMM-Datenkonsistenz gilt auch nach der Chunk-Assemblierung."""
    d = AspadeClassDeclipper(stages=1, channels=2, seed=5, never_worsen=False)
    d._chunk_samples = 100
    x = np.random.default_rng(42).uniform(-0.4, 0.4, 351).astype(np.float32)
    x[49:54] = 0.5
    x[198:203] = -0.5
    out = d.declip(x, SR, 0.5)
    np.testing.assert_array_equal(out[np.abs(x) < 0.5], x[np.abs(x) < 0.5])
    assert np.all(out[x >= 0.5] >= 0.5)
    assert np.all(out[x <= -0.5] <= -0.5)


@pytest.mark.parametrize("chunk_samples", [101, 257, 500])
def test_chunked_model_matches_whole_signal_at_boundaries(chunk_samples: int) -> None:
    """Unverfälschter Modellkontext auch an Chunk- und Songgrenzen."""
    d = AspadeClassDeclipper(stages=2, channels=4, seed=5, never_worsen=False)
    rng = np.random.default_rng(42)
    x = np.clip(rng.normal(0.0, 0.4, 1501), -0.35, 0.35).astype(np.float32)
    expected = d._infer_block(x, 0.35)
    d._chunk_samples = chunk_samples
    actual = d._infer_channel(x, 0.35)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_evaluation_report_separates_damaged_and_control_samples() -> None:
    from scripts.evaluate_aspade_declipper import _error_report

    clean = np.array([[0.8, 0.1, -0.7, -0.2]], dtype=np.float32)
    damaged = np.clip(clean, -0.5, 0.5)
    result = np.array([[0.8, 0.15, -0.7, -0.2]], dtype=np.float32)
    report = _error_report(clean, damaged, result)

    assert report["damaged_samples"] == 2
    assert report["control_changed_samples"] == 1
    assert report["control_max_abs_change"] == pytest.approx(0.05)
    assert report["error_energy_recovered_fraction"] < 1.0
    assert _error_report(clean, clean, clean)["delta_sdr_db"] is None


def test_evaluation_alternative_corpus_records_provenance(tmp_path, monkeypatch) -> None:
    import json
    import sys

    import soundfile as sf

    import scripts.evaluate_aspade_declipper as evaluation

    root = tmp_path / "music"
    (root / "second").mkdir(parents=True)
    (root / "first").mkdir()
    signal = np.sin(np.arange(16000) * 0.1).astype(np.float32)
    for folder in ("first", "second"):
        sf.write(root / folder / "same_name.wav", signal, 16000)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"test checkpoint")
    output = tmp_path / "report.json"
    monkeypatch.setattr(mmf, "use_aspade_declipper", False)
    monkeypatch.setattr(evaluation, "AspadeClassDeclipper", lambda **_kwargs: object())
    monkeypatch.setattr(evaluation, "evaluate_segment", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_aspade_declipper.py",
            "--audio-root",
            str(root),
            "--limit",
            "2",
            "--offset",
            "0",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
        ],
    )

    evaluation.main()

    report = json.loads(output.read_text())
    assert report["schema"] == "aurik.aspade-exploration/11"
    assert report["corpus_root"] == str(root.resolve())
    assert report["split"] is None and report["stem"] is None
    assert [track["track"] for track in report["tracks"]] == ["first/same_name.wav", "second/same_name.wav"]
    assert all(track["source_sample_rate"] == 16000 and track["source_channels"] == 1 for track in report["tracks"])
    assert all(len(track["segment_sha256"]) == 64 for track in report["tracks"])


@pytest.mark.parametrize(("option", "value"), [("--split", "test"), ("--stem", "vocals")])
def test_evaluation_alternative_corpus_rejects_musdb_options(tmp_path, monkeypatch, capsys, option, value) -> None:
    import sys

    import scripts.evaluate_aspade_declipper as evaluation

    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate_aspade_declipper.py", "--audio-root", str(tmp_path), option, value, "--output", "unused"],
    )
    with pytest.raises(SystemExit) as exc:
        evaluation.main()
    assert exc.value.code == 2
    assert f"{option} ist nur mit --musdb-root zulässig" in capsys.readouterr().err


def test_evaluation_right_channel_only_and_true_phase(monkeypatch) -> None:
    from scripts.evaluate_aspade_declipper import evaluate_segment

    class IdentityModel:
        def declip(self, audio, sample_rate, clip_threshold):
            assert sample_rate == SR
            assert clip_threshold > 0
            return audio.copy()

    import backend.core.music_model_flags as flags

    monkeypatch.setattr(flags, "use_aspade_declipper", False)
    x = np.stack(
        [
            np.linspace(-0.2, 0.2, 2000, dtype=np.float32),
            np.linspace(-0.9, 0.9, 2000, dtype=np.float32),
        ]
    )
    report = evaluate_segment(x, 0.5, IdentityModel(), right_only=True, phase_opt_in=True)
    assert report["raw_known_threshold"]["damaged_samples"] > 0
    assert report["raw_known_threshold"]["delta_sdr_db"] == pytest.approx(0.0)
    assert report["raw_known_threshold"]["control_changed_samples"] == 0
    assert report["raw_known_threshold_and_channel"]["control_changed_samples"] == 0
    assert "clip_threshold" in report["phase_automatic_threshold"]["metrics"]
    assert report["clean_control"]["delta_sdr_db"] is None
    assert report["phase_opt_in_automatic_threshold"]["changed_vs_default"] is False
    assert report["phase_opt_in_clean_control"]["changed_vs_default"] is False
    assert flags.use_aspade_declipper is False
    assert report["experimental_asymmetric"]["positive_threshold"] == 0.5
    assert report["experimental_asymmetric"]["negative_threshold"] == -0.75
    assert report["experimental_asymmetric"]["polarity_plateau"]["control_changed_samples"] == 0
    assert set(report["experimental_nonlinear_and_quantized_controls"]) == {"tanh4", "tanh8", "pcm16"}
    assert 1 not in report["experimental_resampled_detection"]["detected_channels"]
    oracle = report["experimental_resampled_detection"]["known_threshold_and_channel_oracle"]
    assert oracle["damaged_samples"] > 0
    assert oracle["delta_sdr_db"] == pytest.approx(0.0)
    assert oracle["control_changed_samples"] == 0
    before = report["experimental_resampled_detection"]["repair_before_resampling_oracle"]
    assert before["damaged_samples"] == oracle["damaged_samples"]
    assert before["delta_sdr_db"] == pytest.approx(0.0)
    assert before["control_changed_samples"] == 0


def test_evaluation_before_resampling_matches_known_clean_oracle(monkeypatch) -> None:
    from scipy.signal import resample_poly

    import backend.core.music_model_flags as flags
    from scripts.evaluate_aspade_declipper import _error_report, evaluate_segment

    t = np.arange(2000, dtype=np.float32) / SR
    clean = np.stack((0.3 * np.sin(2 * np.pi * 440 * t), 0.9 * np.sin(2 * np.pi * 440 * t)))
    damaged = clean.copy()
    damaged[1] = np.clip(damaged[1], -0.5, 0.5)

    class KnownCleanModel:
        def declip(self, audio, sample_rate, clip_threshold):
            if audio.ndim == 1 and np.array_equal(audio, damaged[1]):
                return clean[1].copy()
            return audio.copy()

    monkeypatch.setattr(flags, "use_aspade_declipper", False)
    report = evaluate_segment(clean, 0.5, KnownCleanModel(), right_only=True)

    def resample(x):
        return resample_poly(resample_poly(x, 147, 160, axis=-1), 160, 147, axis=-1).astype(np.float32)

    expected = _error_report(resample(clean), resample(damaged), resample(clean))
    before = report["experimental_resampled_detection"]["repair_before_resampling_oracle"]
    assert before == expected
    assert before["error_energy_after"] == 0.0
    assert np.isfinite(before["delta_sdr_db"])
    assert before["error_energy_recovered_fraction"] == pytest.approx(1.0)


def test_polarity_research_candidate_handles_asymmetric_plateaus() -> None:
    from scripts.evaluate_aspade_declipper import _polarity_plateau_candidate

    calls = []

    class Candidate:
        def declip(self, samples, sample_rate, clip_threshold):
            calls.append((samples.copy(), sample_rate, clip_threshold))
            return samples + np.float32(0.1)

    healthy = np.array([0.0, 0.2, 0.3, 0.0, -0.5, -0.4, 0.0, 0.1], dtype=np.float32)
    damaged = np.array([0.0, 0.3, 0.3, 0.0, -0.6, -0.6, -0.2, 0.0], dtype=np.float32)
    result, levels = _polarity_plateau_candidate(np.stack((healthy, damaged)), Candidate())
    assert len(calls) == 1
    assert calls[0][1] == SR
    assert calls[0][2] == pytest.approx(0.45)
    assert levels[0]["channel"] == 1
    assert levels[0]["positive_level"] == pytest.approx(0.3)
    assert levels[0]["negative_level"] == pytest.approx(-0.6)
    np.testing.assert_array_equal(result[0], healthy)
    np.testing.assert_array_equal(result[1, [0, 3, 6, 7]], damaged[[0, 3, 6, 7]])
    np.testing.assert_allclose(result[1, [1, 2, 4, 5]], damaged[[1, 2, 4, 5]] + 0.1)


def test_polarity_research_candidate_rejects_nonfinite_or_bad_shape() -> None:
    from scripts.evaluate_aspade_declipper import _polarity_plateau_candidate

    with pytest.raises(ValueError):
        _polarity_plateau_candidate(np.array([[np.nan]], dtype=np.float32), object())
    with pytest.raises(ValueError):
        _polarity_plateau_candidate(np.empty((1, 0), dtype=np.float32), object())


def test_polarity_centering_keeps_both_plateaus_above_model_threshold() -> None:
    from scripts.evaluate_aspade_declipper import _polarity_plateau_candidate

    class CheckThreshold:
        def declip(self, samples, sample_rate, clip_threshold):
            threshold = np.float32(clip_threshold)
            assert samples[1] >= threshold and samples[2] >= threshold
            assert samples[4] <= -threshold and samples[5] <= -threshold
            return samples.copy()

    damaged = np.array([[0, 0.665526, 0.665526, 0, -0.245813, -0.245813, 0]], dtype=np.float32)
    _, levels = _polarity_plateau_candidate(damaged, CheckThreshold())
    assert len(levels) == 1


def test_polarity_stereo_witness_protects_unaffected_samples_and_channel() -> None:
    from scripts.evaluate_aspade_declipper import _polarity_stereo_witness_candidate

    reference = np.linspace(-0.9, 0.9, 1001, dtype=np.float32)
    target = np.clip(reference, -0.5, 0.3)
    damaged = np.stack((reference, target))
    candidate = damaged.copy()
    mask = (target == np.float32(-0.5)) | (target == np.float32(0.3))
    candidate[1, mask] = reference[mask]
    levels = [{"channel": 1, "positive_level": float(np.float32(0.3)), "negative_level": float(np.float32(-0.5))}]

    result = _polarity_stereo_witness_candidate(damaged, candidate, levels, 501)

    np.testing.assert_array_equal(result[0], damaged[0])
    np.testing.assert_array_equal(result[1, ~mask], damaged[1, ~mask])
    assert np.sum((result[1] - reference) ** 2) < np.sum((target - reference) ** 2)
    assert np.all(np.abs(result[1] - target) <= np.abs(candidate[1] - target) + 1e-7)
    np.testing.assert_array_equal(_polarity_stereo_witness_candidate(damaged, candidate, [], 501), damaged)


def test_polarity_stereo_witness_rejects_bad_channel() -> None:
    from scripts.evaluate_aspade_declipper import _polarity_stereo_witness_candidate

    damaged = np.zeros((2, 20), dtype=np.float32)
    with pytest.raises(ValueError, match="Defektkanal"):
        _polarity_stereo_witness_candidate(damaged, damaged, [{"channel": 3}], 5)


def test_evaluation_identifies_known_healthy_channel_damage(monkeypatch) -> None:
    from scripts.evaluate_aspade_declipper import evaluate_segment

    class ThresholdModel:
        def declip(self, audio, sample_rate, clip_threshold):
            return np.where(np.abs(audio) >= clip_threshold, audio * 1.1, audio)

    import backend.core.music_model_flags as flags

    monkeypatch.setattr(flags, "use_aspade_declipper", False)
    clean = np.stack(
        [
            np.linspace(-0.9, 0.9, 2000, dtype=np.float32),
            np.linspace(-0.8, 0.8, 2000, dtype=np.float32),
        ]
    )
    report = evaluate_segment(clean, 0.5, ThresholdModel(), right_only=True)
    assert report["raw_known_threshold"]["control_changed_samples"] > 0
    assert report["raw_known_threshold_and_channel"]["control_changed_samples"] == 0


def test_experimental_plateau_detector_is_per_channel_and_exposes_quantization_risk() -> None:
    from scripts.evaluate_aspade_declipper import _plateau_candidate

    class ThresholdModel:
        def declip(self, audio, sample_rate, clip_threshold):
            return np.where(np.abs(audio) >= clip_threshold, audio * 1.1, audio)

    clean = np.array([[0.1, 0.8, 0.4, -0.3], [0.1, 0.9, 0.8, -0.1]], dtype=np.float32)
    clipped = clean.copy()
    clipped[1] = np.clip(clipped[1], -0.5, 0.5)
    result, detected = _plateau_candidate(clipped, ThresholdModel())
    assert detected == [1]
    np.testing.assert_array_equal(result[0], clipped[0])
    assert np.all(result[1, 1:3] > clipped[1, 1:3])
    assert _plateau_candidate(clean, ThresholdModel())[1] == []

    quantized_clean = np.array([[0.1, 0.5, 0.5, 0.1]], dtype=np.float32)
    assert _plateau_candidate(quantized_clean, ThresholdModel())[1] == [0]


def test_stereo_research_witness_respects_candidate_and_channels() -> None:
    from scripts.evaluate_aspade_declipper import _stereo_witness_candidate

    left = np.linspace(-0.9, 0.9, 101, dtype=np.float32)
    right = np.clip(left, -0.5, 0.5)
    damaged = np.stack((left, right))
    candidate = damaged.copy()
    candidate[1] = left
    result = _stereo_witness_candidate(damaged, candidate, [1], window_samples=41)
    np.testing.assert_array_equal(result[0], left)
    np.testing.assert_array_equal(result[1, np.abs(right) < 0.5], right[np.abs(right) < 0.5])
    assert np.all(np.abs(result[1] - right) <= np.abs(candidate[1] - right) + 1e-7)
    assert np.sum((result[1] - left) ** 2) < np.sum((right - left) ** 2)
    np.testing.assert_array_equal(_stereo_witness_candidate(damaged, candidate, [0, 1], 41), damaged)


def test_density_gated_research_candidate_routes_by_observed_density() -> None:
    from scripts.evaluate_aspade_declipper import _density_gated_candidate

    damaged = np.array([[0.1, 0.2, 0.3, 0.4], [0.1, 0.5, 0.5, -0.1]], dtype=np.float32)
    candidate = damaged + np.float32(0.1)
    witness = damaged + np.float32(0.01)
    np.testing.assert_array_equal(_density_gated_candidate(damaged, candidate, witness, [1], 0.25), candidate)
    np.testing.assert_array_equal(_density_gated_candidate(damaged, candidate, witness, [1], 0.5), witness)
    np.testing.assert_array_equal(_density_gated_candidate(damaged, candidate, witness, [0, 1], 0.5), damaged)
    with pytest.raises(ValueError):
        _density_gated_candidate(damaged, candidate, witness, [1], -0.1)


def test_correlation_gated_research_candidate_preserves_unrelated_channel() -> None:
    from scripts.evaluate_aspade_declipper import _correlation_gated_candidate

    x = np.linspace(-0.9, 0.9, 101, dtype=np.float32)
    damaged = np.stack((x, np.clip(x, -0.5, 0.5)))
    base = damaged.copy()
    base[1] = np.sign(damaged[1]) * 0.6
    witness = damaged.copy()
    witness[1] = x
    result = _correlation_gated_candidate(damaged, base, witness, [1], 41)
    np.testing.assert_array_equal(result[0], damaged[0])
    assert result[1, 20] == witness[1, 20]
    assert result[1, 0] == base[1, 0]
    np.testing.assert_array_equal(_correlation_gated_candidate(damaged, base, witness, [0, 1], 41), damaged)


def test_evaluation_opt_in_restores_flag_after_success_and_error(monkeypatch) -> None:
    import backend.core.music_model_flags as flags
    import scripts.evaluate_aspade_declipper as evaluation

    observed = []

    class FakePhase:
        def process(self, audio, **_kwargs):
            observed.append(flags.use_aspade_declipper)
            audio[0] = 0.0
            return audio

    monkeypatch.setattr(flags, "use_aspade_declipper", False)
    monkeypatch.setattr(evaluation, "DeclipperPhase", FakePhase)
    signal = np.ones(10, dtype=np.float32)
    evaluation._phase_opt_in(signal)
    assert observed == [True]
    assert flags.use_aspade_declipper is False
    np.testing.assert_array_equal(signal, np.ones(10, dtype=np.float32))

    class FailingPhase:
        def process(self, audio, **_kwargs):
            assert flags.use_aspade_declipper is True
            raise RuntimeError("expected")

    monkeypatch.setattr(evaluation, "DeclipperPhase", FailingPhase)
    with pytest.raises(RuntimeError, match="expected"):
        evaluation._phase_opt_in(signal)
    assert flags.use_aspade_declipper is False


# ── Never-worsen-Harness auf geclipptem Signal ──────────────────────────────


def test_never_worsen_harness_geclippt() -> None:
    """Der Kandidat darf das Signal nie verschlechtern (SDR gegen Ground Truth)."""
    clean, clipped = _bursts_with_clean(6.0)
    d = AspadeClassDeclipper(stages=2, channels=8, seed=5)
    out = d.declip(clipped, SR, 1.0)
    assert np.isfinite(out).all()
    assert out.shape == clipped.shape
    assert _sdr_db(out, clean) >= _sdr_db(clipped, clean) - 1e-9  # Never-worsen


def test_never_worsen_akzeptiert_orakel(monkeypatch) -> None:
    """Orakel-Kandidat (clean) wird übernommen — der Guard akzeptiert Gewinne."""
    clean, clipped = _bursts_with_clean(6.0)
    d = AspadeClassDeclipper(stages=2, channels=8, seed=5)
    torch_clean = torch.from_numpy(clean.astype(np.float32)).reshape(1, 1, -1)
    monkeypatch.setattr(d, "_net", lambda y_t, theta_t: torch_clean)
    out = d.declip(clipped, SR, 1.0)
    np.testing.assert_array_equal(out, clean)  # Orakel bit-identisch übernommen
    assert _sdr_db(out, clean) > _sdr_db(clipped, clean) + 20.0


def test_never_worsen_lehnt_verschlechterung_ab(monkeypatch, caplog) -> None:
    """Verschlechternder Kandidat wird abgelehnt — Eingang bleibt bit-identisch.

    Harness-Präzedenz: der Kandidat muss den Harmonik-Witness GARANTIERT
    erhöhen. Gewählt wird die 6. Harmonische (2640 Hz = 6 × 440) — außerhalb
    der Grundtonsuche (60–2000 Hz), aber in den gezählten Harmonischen (2–10)
    des Witnesses; ein Kandidat, der den Grundton selbst verschiebt (z. B.
    1320 Hz), würde den heuristischen f0-Suchlauf austricksen.
    """
    _, clipped = _bursts_with_clean(6.0)
    d = AspadeClassDeclipper(stages=2, channels=8, seed=5)
    t = torch.arange(clipped.size, dtype=torch.float32).reshape(1, 1, -1) / SR
    harmonic = 0.9 * torch.sin(2 * np.pi * 2640.0 * t)

    def _worse(y_t: torch.Tensor, theta_t: torch.Tensor) -> torch.Tensor:
        return y_t + harmonic

    # Präzedenz des Harness: der Stub-Kandidat ist per Witness messbar schlechter.
    cand_np = clipped + harmonic.numpy().reshape(-1)
    assert asp.harmonic_distortion_witness(cand_np, SR) > asp.harmonic_distortion_witness(clipped, SR)

    monkeypatch.setattr(d, "_net", _worse)
    with caplog.at_level(logging.INFO, logger="backend.core.dsp.aspade_declipper"):
        out = d.declip(clipped, SR, 1.0)
    np.testing.assert_array_equal(out, clipped)  # bit-identischer Passthrough
    assert any("abgelehnt" in r.getMessage() for r in caplog.records)


# ── Stereo-Layout-Invariante (C, N) (AGENTS.md §3) ─────────────────────────


def test_layout_stereo_2n_und_n2_bit_identisch() -> None:
    """(2, N) und (N, 2) liefern transponiert bit-identische Ergebnisse."""
    clean0, clipped0 = _bursts_with_clean(6.0)
    clean1, clipped1 = _bursts_with_clean(2.5)
    x_2n = np.stack([clipped0, clipped1])  # (2, N)
    x_n2 = np.ascontiguousarray(x_2n.T)  # (N, 2)
    d = AspadeClassDeclipper(stages=2, channels=8, seed=5)
    r_2n = d.declip(x_2n, SR, 1.0)
    r_n2 = d.declip(x_n2, SR, 1.0)
    assert r_2n.shape == x_2n.shape
    assert r_n2.shape == x_n2.shape
    np.testing.assert_array_equal(r_2n, r_n2.T)  # Layout-invariant, bit-identisch


def test_layout_mono_shape_und_dtype() -> None:
    """(N,) float64 ⇒ (N,) float64; (1, N) ⇒ (1, N)."""
    _, clipped = _bursts_with_clean(6.0)
    d = AspadeClassDeclipper(stages=2, channels=8, seed=5)
    out1 = d.declip(clipped.astype(np.float64), SR, 1.0)
    assert out1.shape == clipped.shape
    assert out1.dtype == np.float64
    out2 = d.declip(clipped.reshape(1, -1), SR, 1.0)
    assert out2.shape == (1, clipped.size)
    np.testing.assert_array_equal(out2[0], out1.astype(np.float32))


# ── Flag-Gate: use_aspade_declipper AUS ⇒ bit-identisches Bestandsverhalten ─


def test_default_checkpoint_uses_repository_models_directory() -> None:
    root = Path(asp.__file__).resolve().parents[3]
    assert root / asp.DEFAULT_CHECKPOINT_SUBPATH == asp.DEFAULT_CHECKPOINT


def test_flag_default_aus() -> None:
    """Flag ist aus (music_model_flags.py) — Challenger-Pfad läuft nie ohne Sign-off."""
    assert mmf.use_aspade_declipper is False


def test_flag_aus_challenger_nie_gerufen_und_bit_identisch(monkeypatch) -> None:
    """Flag AUS: die Phase 07 ruft den Challenger nie, Ausgabe/Metrik unverändert."""
    _, x = _bursts_with_clean(6.0)
    ref = DeclipperPhase().process(x, sample_rate=SR)

    calls: list[int] = []

    def _bomb(*_args, **_kwargs):
        calls.append(1)
        raise AssertionError("Challenger darf bei Flag AUS nicht gerufen werden")

    monkeypatch.setattr(asp, "declip_aspade_class", _bomb)
    out = DeclipperPhase().process(x, sample_rate=SR)

    assert calls == []  # nie aufgerufen
    np.testing.assert_array_equal(out.audio, ref.audio)  # bit-identisch
    assert out.metrics == ref.metrics  # keine neuen Metrik-Schlüssel
    assert "aspade_class_used" not in out.metrics


def test_flag_an_abstinenz_bit_identisch_zu_flag_aus(monkeypatch) -> None:
    """Flag AN, Challenger abstinent (None): PCHIP-Kaskade bleibt bit-identisch."""
    _, x = _bursts_with_clean(6.0)
    ref = DeclipperPhase().process(x, sample_rate=SR)

    monkeypatch.setattr(mmf, "use_aspade_declipper", True)
    monkeypatch.setattr(asp, "declip_aspade_class", lambda *_a, **_k: None)
    monkeypatch.setattr("backend.core.dsp.musik_declip_kopf.declip_musik_kopf", lambda *_a, **_k: None)
    out = DeclipperPhase().process(x, sample_rate=SR)

    np.testing.assert_array_equal(out.audio, ref.audio)
    assert out.metrics["aspade_class_used"] is False
    assert out.metrics["musik_kopf_used"] is False


def test_flag_an_orakel_wird_uebernommen(monkeypatch) -> None:
    """Flag AN, gewinnender Kandidat: Challenger übernimmt, Harmonik-Proxy sinkt."""
    clean, clipped = _bursts_with_clean(6.0)
    monkeypatch.setattr(mmf, "use_aspade_declipper", True)
    monkeypatch.setattr(
        asp,
        "declip_aspade_class",
        lambda audio, sample_rate, clip_threshold: clean.astype(np.float64).copy(),
    )
    out = DeclipperPhase().process(clipped, sample_rate=SR)
    assert out.metrics["aspade_class_used"] is True
    assert _thd(out.audio) < _thd(clipped)


def test_flag_an_fehler_pchip_kaskade_mit_warnung(monkeypatch, caplog) -> None:
    """Flag AN, Fehler im Challenger: §V6-Warnung + PCHIP-Kaskade bleibt bit-identisch (§V6, copilot-instructions.md)."""
    _, x = _bursts_with_clean(6.0)
    ref = DeclipperPhase().process(x, sample_rate=SR)

    monkeypatch.setattr(mmf, "use_aspade_declipper", True)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("kaputt")

    monkeypatch.setattr(asp, "declip_aspade_class", _boom)
    monkeypatch.setattr("backend.core.dsp.musik_declip_kopf.declip_musik_kopf", _boom)
    with caplog.at_level(logging.WARNING, logger="backend.core.phases.phase_07_declipper"):
        out = DeclipperPhase().process(x, sample_rate=SR)

    np.testing.assert_array_equal(out.audio, ref.audio)  # PCHIP-Kaskade bleibt
    assert out.metrics["aspade_class_used"] is False
    warn_text = " ".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
    assert "A-SPADE-Klassen-Declipper" in warn_text
    assert "PCHIP-Kaskade bleibt" in warn_text
    assert "§V6" in warn_text  # Textmarke des Logs (copilot-instructions.md)


# ── Fehlerpfade (§V6 (copilot-instructions.md): Warnung statt stiller Fallback) ─


def test_fassade_ohne_artefakt_none_mit_warnung(monkeypatch, caplog, tmp_path) -> None:
    """Ohne trainiertes Artefakt: None + §V6-Warnung — der Aufrufer fällt auf PCHIP zurück."""
    monkeypatch.setattr(asp, "DEFAULT_CHECKPOINT", tmp_path / "fehlt.pt")
    monkeypatch.setattr(asp, "_declipper_singleton", None)
    with caplog.at_level(logging.WARNING, logger="backend.core.dsp.aspade_declipper"):
        res = asp.declip_aspade_class(np.zeros(1024, dtype=np.float32), SR, 0.5)
    assert res is None
    assert any("§V6" in r.getMessage() for r in caplog.records)  # Textmarke des Logs (copilot-instructions.md)


def test_fehlerpfad_leeres_audio_warnung(caplog) -> None:
    """Leeres Audio: Warnung + unveränderter Passthrough (kein stilles Schweigen)."""
    d = AspadeClassDeclipper(stages=2, channels=8, seed=5)
    empty = np.zeros(0, dtype=np.float32)
    with caplog.at_level(logging.WARNING, logger="backend.core.dsp.aspade_declipper"):
        out = d.declip(empty, SR, 0.5)
    assert out.shape == (0,)
    assert any("leeres Audio" in r.getMessage() for r in caplog.records)


def test_fehlerpfad_nan_eingang_sanitisiert(caplog) -> None:
    """NaN/Inf im Eingang: Warnung + sanitisierte, finite Ausgabe (§0a/§V6, copilot-instructions.md)."""
    _, clipped = _bursts_with_clean(6.0)
    dirty = clipped.copy()
    dirty[100] = np.nan
    dirty[200] = np.inf
    d = AspadeClassDeclipper(stages=2, channels=8, seed=5)
    with caplog.at_level(logging.WARNING, logger="backend.core.dsp.aspade_declipper"):
        out = d.declip(dirty, SR, 1.0)
    assert out.shape == dirty.shape
    assert np.isfinite(out).all()
    assert any("NaN/Inf" in r.getMessage() for r in caplog.records)


def test_fehlerpfad_modelloutput_nicht_finit_warnung(monkeypatch, caplog) -> None:
    """Nicht-finites Modell-Ergebnis: §V6-Warnung + bit-identischer Eingangs-Passthrough."""
    _, clipped = _bursts_with_clean(6.0)
    d = AspadeClassDeclipper(stages=2, channels=8, seed=5)
    monkeypatch.setattr(d, "_net", lambda y_t, theta_t: torch.full_like(y_t, float("nan")))
    with caplog.at_level(logging.WARNING, logger="backend.core.dsp.aspade_declipper"):
        out = d.declip(clipped, SR, 1.0)
    np.testing.assert_array_equal(out, clipped)  # Eingang bleibt bit-identisch
    assert any("nicht-finites Modell-Ergebnis" in r.getMessage() for r in caplog.records)


def test_value_error_bei_falschen_argumenten() -> None:
    """Ungültige Argumente werfen laut ValueError — kein stilles Erraten."""
    d = AspadeClassDeclipper(stages=2, channels=8, seed=5)
    with pytest.raises(ValueError):
        d.declip(np.zeros((2, 2, 2), dtype=np.float32), SR, 0.5)  # falsches Layout
    with pytest.raises(ValueError):
        d.declip(np.zeros(16, dtype=np.int16), SR, 0.5)  # falscher dtype
    with pytest.raises(ValueError):
        d.declip(np.zeros(16, dtype=np.float32), SR, 0.0)  # θ muss > 0 sein
    with pytest.raises(ValueError):
        d.declip(np.zeros(16, dtype=np.float32), 0, 0.5)  # sample_rate muss > 0 sein


def test_checkpoint_laden_bit_identisch(tmp_path) -> None:
    """Trainiertes Artefakt (aspade_config + model_state_dict) lädt bit-identisch."""
    net = asp.build_model(stages=2, channels=8, dilations=(1, 2), seed=9)
    ckpt = tmp_path / "aspade_class_v1.pt"
    torch.save(
        {
            "aspade_config": {"stages": 2, "channels": 8, "kernel_size": 7, "dilations": (1, 2)},
            "model_state_dict": net.state_dict(),
        },
        ckpt,
    )
    d_loaded = AspadeClassDeclipper(checkpoint=ckpt)
    d_seeded = AspadeClassDeclipper(stages=2, channels=8, dilations=(1, 2), seed=9)
    _, clipped = _bursts_with_clean(6.0)
    np.testing.assert_array_equal(
        d_loaded.declip(clipped, SR, 1.0),
        d_seeded.declip(clipped, SR, 1.0),
    )
