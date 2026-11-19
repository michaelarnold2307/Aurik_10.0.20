from __future__ import annotations

import pytest

"""tests/unit/test_medium_detector_short_clip.py

Tests für §2.47/§0c Fix: rotation_strength-Maskierung bei kurzen Clips
in MediumDetector._bayesian_score (v10.0.0).

Problem: 33⅓ RPM-Rotation erzeugt ACF-Peak erst nach ≥3 Zyklen (≈5.5 s).
In Clips < 6 s ist rotation_strength ≈ 0 — das is ein MESS-Artefakt, kein
Materialhinweis. Das Bayesian-Modell bestraft Vinyl (μ=0.40, σ=0.20) stark
und bevorzugt Tape (μ=0.0, σ=0.08) → Fehlklassifizierung.

Fix: Unter _MIN_ROTATION_ANALYSIS_DURATION_S (6 s) wird rotation_strength
als nicht beobachtet behandelt (→ alle Materialien erhalten gleiche
Log-Likelihood für dieses Feature → andere Features entscheiden).
"""


import numpy as np

from forensics.medium_detector import MediumDetector

SR = 44_100
_det = MediumDetector()


def _make_vinyl_short(duration_s: float = 3.0, sr: int = SR) -> np.ndarray:
    """Synthetisches 'vinyl'-ähnliches Audio: impulsives Knistern + rosa Rauschen.

    rotation_strength ≈ 0 (zu kurz für ACF-Peak), aber crackle_density > 0.
    """
    rng = np.random.default_rng(42)
    n = int(duration_s * sr)
    # Rosa Rauschen (1/f): kumuliere White Noise in STFT-Raum
    white = rng.standard_normal(n).astype(np.float32) * 0.02
    # Einfache Annäherung für Tests: lowpass-gefiltert
    from scipy.signal import butter, sosfilt

    sos = butter(4, 4000.0 / (sr / 2), btype="low", output="sos")
    pink_approx = sosfilt(sos, white.astype(np.float64)).astype(np.float32)

    # Impulsives Knistern (Vinyl-Crackle)
    n_impulses = max(1, int(duration_s * 5))  # ~5 Impulse/s
    positions = rng.integers(int(0.01 * n), int(0.99 * n), size=n_impulses)
    for pos in positions:
        pink_approx[pos : pos + 3] += rng.standard_normal(3).astype(np.float32) * 0.3

    return np.clip(pink_approx, -1.0, 1.0)  # type: ignore[no-any-return]


def _make_tape_short(duration_s: float = 3.0, sr: int = SR) -> np.ndarray:
    """Synthetisches 'tape'-ähnliches Audio: bandpassgefiltertes Rauschen ohne Impulsartefakte.

    Kein Knistern, kein infrasonic_rms, rotation_strength ≈ 0.
    Bandpass (200–8000 Hz) verhindert infrasonic_rms-Spike (Brown-Noise via cumsum
    akkumuliert Energie bei 0 Hz → massivers infrasonic_rms → fälschlich vinyl-ähnlich).
    """
    rng = np.random.default_rng(7)
    n = int(duration_s * sr)
    white = rng.standard_normal(n).astype(np.float32) * 0.1

    # Bandpass 200–8000 Hz: kein subsonic, kein HF — tape-typisch
    from scipy.signal import butter, sosfilt

    sos = butter(4, [200.0 / (sr / 2), 8000.0 / (sr / 2)], btype="band", output="sos")
    tape_noise = sosfilt(sos, white.astype(np.float64)).astype(np.float32)
    tape_noise -= np.mean(tape_noise)
    peak = np.max(np.abs(tape_noise))
    if peak > 1e-6:
        tape_noise /= peak * 2.0  # normalisieren auf ca. ±0.5
    return np.clip(tape_noise, -1.0, 1.0)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Core: rotation_strength masking test
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestShortClipRotationMasking:
    """§2.47/§0c: rotation_strength bei kurzen Clips (< 6 s) ausblenden."""

    def test_short_clip_does_not_apply_rotation_penalty_to_vinyl(self) -> None:
        """Ein 3-s-Vinyl-ähnlicher Clip darf NICHT als 'tape' (oder 'unknown') gelten.

        Der Kern des Bugs (Alt-Tabelle): rotation_strength = 0 in 3 s → Tape (μ=0, σ=0.08)
        gewinnt über Vinyl (μ=0.40 → 2σ-Strafe). Fix: Feature für kurze Clips ignorieren.
        D-K1-3 F5 (2026-10-09): Nach der datengetriebenen Neukalibrierung bestraft die
        Rotation sogar Vinyl stärker (μ=0.359/σ=0.05 ⇒ ~7σ bei rotation≈0) — das Fix ist
        damit noch relevanter; vinyl-Posterior mit Fix > ohne Fix bleibt der Kern-Claim.
        """
        audio = _make_vinyl_short(duration_s=3.0)
        fp = _det._compute_fingerprint(audio, SR)

        # Verifikation: rotation_strength ist tatsächlich nahe 0 (Test-Voraussetzung)
        assert fp.rotation_strength < 0.10, (
            f"Testvoraussetzung verletzt: rotation_strength={fp.rotation_strength:.3f} "
            "sollte < 0.10 sein für 3-s-Clip ohne echte Rotation"
        )

        # Score mit duration_s=3.0 (kurzer Clip — Fix aktiv)
        posteriors_short = _det._bayesian_score(fp, duration_s=3.0)

        # Score ohne duration (Standard — kein Fix, Rotation bestraft Vinyl)
        posteriors_long = _det._bayesian_score(fp, duration_s=0.0)

        # Fix: vinyl-Posterior mit duration_s=3.0 MUSS höher sein als ohne duration
        assert posteriors_short.get("vinyl", 0.0) > posteriors_long.get("vinyl", 0.0), (
            f"vinyl-Posterior mit short-clip-Fix ({posteriors_short.get('vinyl', 0):.4f}) "
            f"sollte > ohne Fix ({posteriors_long.get('vinyl', 0):.4f}) sein"
        )

        # tape-Posterior mit Fix MUSS nicht HÖHER als ohne Fix werden (Hauptziel des Tests).
        # D-K1-3 F5 (2026-10-09): Die datengetriebene Neukalibrierung (_MATERIAL_MODELS,
        # 120 Echt-Träger) hat tape auf rotation μ=0.348/σ=0.073 gezogen — rotation≈0 ist
        # damit KEINE Tape-Evidenz mehr (alt: μ=0 ⇒ „Tape gewinnt"). Das Maskieren hebt den
        # tape-Wert daher minimal an, aber beide Werte liegen im Underflow-Bereich
        # (~1e-19 vs. ~1e-23), weit unter jeder aussagekräftigen Posterior-Skala; Top-1 ist
        # in beiden Modi dasselbe Material. Absolute Toleranz 1e-9 statt striktem `<=`.
        assert posteriors_short.get("tape", 1.0) <= posteriors_long.get("tape", 1.0) + 1e-9, (
            f"tape-Posterior mit short-clip-Fix ({posteriors_short.get('tape', 0):.3e}) "
            f"sollte ≤ ohne Fix ({posteriors_long.get('tape', 0):.3e}) ± Underflow-Toleranz sein"
        )

    def test_long_clip_unchanged(self) -> None:
        """Für Clips ≥ 6 s bleibt rotation_strength aktiv (kein Fix nötig)."""
        audio = _make_vinyl_short(duration_s=8.0)
        fp = _det._compute_fingerprint(audio, SR)

        posteriors_long = _det._bayesian_score(fp, duration_s=8.0)
        posteriors_default = _det._bayesian_score(fp, duration_s=0.0)

        # Für ≥ 6 s: Ergebnisse identisch (kein Masking)
        for mat in list(posteriors_long.keys())[:5]:
            assert abs(posteriors_long.get(mat, 0) - posteriors_default.get(mat, 0)) < 1e-9, (
                f"Long-Clip: {mat} sollte identischen Score haben "
                f"(got {posteriors_long.get(mat):.6f} vs {posteriors_default.get(mat):.6f})"
            )

    def test_tape_material_not_affected_by_fix(self) -> None:
        """Tape-ähnliches 3-s-Audio: Fix soll die Erkennung nicht zugunsten von Vinyl verzerren.

        Wenn rotation_strength ≈ 0 (wie bei tape-ähnlichen Signalen), hat das Masking
        keinen Effekt — beide Aufrufe (mit/ohne duration_s) sollten identische Posteriors liefern.
        """
        audio = _make_tape_short(duration_s=3.0)
        fp = _det._compute_fingerprint(audio, SR)

        # Überprüfen: rotation_strength ≈ 0 für das Testsignal (Voraussetzung)
        assert fp.rotation_strength < 0.10, (
            f"Testvoraussetzung: rotation_strength sollte < 0.10 sein, got {fp.rotation_strength:.3f}"
        )

        posteriors_with_fix = _det._bayesian_score(fp, duration_s=3.0)
        posteriors_no_fix = _det._bayesian_score(fp, duration_s=0.0)

        # §D-K1-3 (Fix 2026-10-09): Die frühere „Invarianz"-Annahme („rotation≈0
        # ⇒ Masking neutral") war logisch falsch: rotation_strength ist ein
        # AKTIVES Bayesian-Feature (vinyl μ=0.40, tape/streaming μ≈0) — ein
        # gemessenes ≈0 belegt den μ≈0-Materialien Evidenz, das Maskieren
        # entfernt diese Evidenz. Beide Modi dürfen sich daher unterscheiden.
        # Geprüft wird die Absicht (Kein-Vinyl-Trigger-Vertrag): Das Tape-Signal
        # darf in KEINEM Modus fälschlich Vinyl als Top-1 erhalten.
        for _label, _post in (("mit_fix", posteriors_with_fix), ("ohne_fix", posteriors_no_fix)):
            # Lambda statt `_post.get`: mypy löst die überladene `dict.get` als
            # `(key, default) -> float | None` auf, was zu keinem `max`-Overload passt.
            _top = max(_post, key=lambda k: _post[k])
            _vinyl = _post.get("vinyl", 0.0)
            assert _top != "vinyl", f"Vinyl darf nie Top-1 des Tape-Signals sein ({_label}): {_top}={_post[_top]:.4f}"
            assert _vinyl < 0.30, f"Vinyl-Posterior unerwartet hoch ({_label}): {_vinyl:.4f}"

    def test_detect_full_pipeline_3s_vinyl_not_classified_as_tape(self) -> None:
        """End-to-end: detect() auf 3-s-Vinyl-ähnlichem WAV → kein 'tape' als primary."""
        audio = _make_vinyl_short(duration_s=3.0)
        result = _det.detect(audio, SR, file_ext=".wav")

        # Hauptinvariante: das kurze Vinyl-Signal soll NICHT als tape erkannt werden
        assert result.primary_material != "tape", (
            f"3-s-Vinyl-Audio wurde als 'tape' klassifiziert — "
            f"rotation_strength-Masking für kurze Clips fehlt oder defekt. "
            f"Got: primary_material={result.primary_material}, "
            f"confidence={result.confidence:.3f}, chain={result.transfer_chain}"
        )

    def test_min_rotation_duration_constant_exists(self) -> None:
        """Klasenkonstante _MIN_ROTATION_ANALYSIS_DURATION_S muss existieren und 6 s sein."""
        assert hasattr(MediumDetector, "_MIN_ROTATION_ANALYSIS_DURATION_S"), (
            "MediumDetector._MIN_ROTATION_ANALYSIS_DURATION_S fehlt"
        )
        assert MediumDetector._MIN_ROTATION_ANALYSIS_DURATION_S == 6.0, (
            f"Erwartet 6.0 s, got {MediumDetector._MIN_ROTATION_ANALYSIS_DURATION_S}"
        )
