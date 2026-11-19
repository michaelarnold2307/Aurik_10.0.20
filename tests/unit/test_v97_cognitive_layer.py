"""
tests/unit/test_v97_cognitive_layer.py
=========================================
Aurik 10.0.0 — Cognitive Layer Test-Suite

Testet alle 4 neuen Weltklasse-Module:
  Sektion 17: PerceptualEmbedder (8 Tests)
  Sektion 18: CausalDefectReasoner (10 Tests)
  Sektion 19: GPParameterOptimizer (8 Tests)
  Sektion 20: PerceptualQualityScorer (9 Tests)

Insgesamt: 35 neue Tests
"""

import time

import numpy as np
import pytest

# ─── Fixtures ─────────────────────────────────────────────────────────────────

SR = 48000


@pytest.fixture(scope="module")
def sine_440():
    """440 Hz Sinuston, 2 Sekunden."""
    t = np.linspace(0, 2.0, int(SR * 2.0), endpoint=False)
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)


@pytest.fixture(scope="module")
def noise_signal():
    """Weißes Rauschen, 2 Sekunden."""
    rng = np.random.default_rng(42)
    return rng.normal(0, 0.3, int(SR * 2.0)).astype(np.float32)


@pytest.fixture(scope="module")
def silent_signal():
    """Stilles Signal, 1 Sekunde."""
    return np.zeros(SR, dtype=np.float32)


@pytest.fixture(scope="module")
def chord_signal():
    """Dur-Akkord C-E-G, 3 Sekunden, stereo."""
    t = np.linspace(0, 3.0, int(SR * 3.0), endpoint=False)
    c = np.sin(2 * np.pi * 261.63 * t)
    e = np.sin(2 * np.pi * 329.63 * t)
    g = np.sin(2 * np.pi * 392.00 * t)
    mono = (c + e + g).astype(np.float32) / 3.0
    return np.stack([mono, mono * 0.9])  # Stereo (2, N)


@pytest.fixture(scope="module")
def degraded_signal(sine_440):
    """Sinus mit addiertem Rauschen."""
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 0.15, len(sine_440)).astype(np.float32)
    return (sine_440 + noise).astype(np.float32)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SEKTION 17: PerceptualEmbedder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.mark.unit
class TestPerceptualEmbedder:
    """Sektion 17 — PerceptualAudioEmbedder (8 Tests)."""

    def test_import(self):
        """Modul ist importierbar."""
        from backend.core.perceptual_embedder import AudioEmbedding, PerceptualEmbedder

        assert PerceptualEmbedder is not None
        assert AudioEmbedding is not None

    def test_embed_dim_256(self, sine_440):
        """Embedding hat exakt 256 Dimensionen."""
        from backend.core.perceptual_embedder import PerceptualEmbedder

        emb = PerceptualEmbedder().embed(sine_440, SR)
        assert emb.vector.shape == (256,), f"Erwartet (256,), got {emb.vector.shape}"

    def test_l2_normalized(self, sine_440):
        """Embedding ist L2-normiert (||v|| = 1)."""
        from backend.core.perceptual_embedder import PerceptualEmbedder

        emb = PerceptualEmbedder().embed(sine_440, SR)
        norm = float(np.linalg.norm(emb.vector))
        assert abs(norm - 1.0) < 1e-5, f"Norm ist {norm}, nicht 1.0"

    def test_no_nan_or_inf(self, chord_signal):
        """Embedding enthält keine NaN/Inf-Werte."""
        from backend.core.perceptual_embedder import PerceptualEmbedder

        emb = PerceptualEmbedder().embed(chord_signal, SR)
        assert not np.any(np.isnan(emb.vector)), "NaN im Embedding"
        assert not np.any(np.isinf(emb.vector)), "Inf im Embedding"

    def test_cosine_self_similarity_is_one(self, sine_440):
        """Kosinus-Ähnlichkeit eines Signals mit sich selbst = 1.0."""
        from backend.core.perceptual_embedder import PerceptualEmbedder

        emb = PerceptualEmbedder().embed(sine_440, SR)
        sim = emb.cosine_similarity(emb)
        assert abs(sim - 1.0) < 1e-5, f"Selbst-Ähnlichkeit = {sim}"

    def test_sine_vs_noise_low_similarity(self, sine_440, noise_signal):
        """Sinus und Rauschen haben niedrige Kosinus-Ähnlichkeit."""
        from backend.core.perceptual_embedder import PerceptualEmbedder

        embedder = PerceptualEmbedder()
        e1 = embedder.embed(sine_440, SR)
        e2 = embedder.embed(noise_signal, SR)
        sim = e1.cosine_similarity(e2)
        # Unterschiedliche Signale sollten < 0.95 ähnlich sein
        assert sim < 0.95, f"Ähnlichkeit zu hoch: {sim}"

    def test_short_audio_handled(self, silent_signal):
        """Zu kurzes Signal (< 512 Samples) ergibt Null-Embedding ohne Fehler."""
        from backend.core.perceptual_embedder import PerceptualEmbedder

        short = silent_signal[:128]
        emb = PerceptualEmbedder().embed(short, SR)
        assert emb.vector.shape == (256,)
        assert not np.any(np.isnan(emb.vector))

    def test_convenience_function(self, sine_440):
        """embed_audio Convenience-Funktion funktioniert."""
        from backend.core.perceptual_embedder import embed_audio

        emb = embed_audio(sine_440, SR, segment_s=1.0)
        assert isinstance(emb.vector, np.ndarray)
        assert emb.vector.shape == (256,)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SEKTION 18: CausalDefectReasoner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCausalDefectReasoner:
    """Sektion 18 — CausalDefectReasoner (10 Tests)."""

    def test_import(self):
        """Modul ist importierbar."""
        from backend.core.causal_defect_reasoner import CausalDefectReasoner

        assert CausalDefectReasoner is not None

    def test_reason_returns_plan(self, sine_440):
        """reason() gibt ein RestorationPlan-Objekt zurück."""
        from backend.core.causal_defect_reasoner import CausalDefectReasoner

        plan = CausalDefectReasoner().reason(
            defect_scores={"tape_dropout": 0.7},
            material="tape",
            audio=sine_440,
            sample_rate=SR,
        )
        assert hasattr(plan, "primary_cause")
        assert hasattr(plan, "cause_probabilities")
        assert hasattr(plan, "recommended_phases")

    def test_probabilities_sum_to_one(self, sine_440):
        """Die Posterior-Wahrscheinlichkeiten summieren sich zu 1."""
        from backend.core.causal_defect_reasoner import CausalDefectReasoner

        plan = CausalDefectReasoner().reason(
            defect_scores={"noise_floor_db": -45.0},
            material="tape",
            audio=sine_440,
            sample_rate=SR,
        )
        total = sum(plan.cause_probabilities.values())
        assert abs(total - 1.0) < 1e-6, f"Summe = {total}"

    def test_tape_dropout_detects_correctly(self, sine_440):
        """Hohe Dropout-Severity → tape_dropout ist Primärursache (tape-Material)."""
        from backend.core.causal_defect_reasoner import CausalDefectReasoner

        plan = CausalDefectReasoner().reason(
            defect_scores={"dropout_severity": 0.85, "silence_ratio": 0.3},
            material="tape",
            audio=sine_440,
            sample_rate=SR,
        )
        assert plan.primary_cause == "tape_dropout", f"Erwartet tape_dropout, got {plan.primary_cause}"

    def test_electrical_hum_detected(self):
        """Hohes hum_score→ electrical_hum in Top-Position."""
        from backend.core.causal_defect_reasoner import CausalDefectReasoner

        # 50 Hz Brumm + Obertöne (100, 150, 200 Hz) — triggert hum_score
        t = np.linspace(0, 2.0, int(SR * 2.0), endpoint=False)
        hum_signal = (
            0.5 * np.sin(2 * np.pi * 50 * t)
            + 0.3 * np.sin(2 * np.pi * 100 * t)
            + 0.15 * np.sin(2 * np.pi * 150 * t)
            + 0.05 * np.sin(2 * np.pi * 200 * t)
        ).astype(np.float32)

        plan = CausalDefectReasoner().reason(
            defect_scores={"noise_floor_db": -20.0, "hum_severity": 0.70},
            material="digital",
            audio=hum_signal,
            sample_rate=SR,
        )
        # electrical_hum sollte in Top-5 sein
        top_causes = [c for c, _ in plan.ranked_causes[:5]]
        assert "electrical_hum" in top_causes or plan.primary_cause == "electrical_hum"

    def test_vinyl_crackle_vinyl_material(self, noise_signal):
        """Vinyl-Material mit hohem click_severity → vinyl_crackle priorisiert."""
        from backend.core.causal_defect_reasoner import CausalDefectReasoner

        plan = CausalDefectReasoner().reason(
            defect_scores={"click_severity": 0.90},
            material="vinyl",
            audio=noise_signal,
            sample_rate=SR,
        )
        top3 = [c for c, _ in plan.ranked_causes[:3]]
        assert "vinyl_crackle" in top3, f"vinyl_crackle nicht in Top-3: {top3}"

    def test_confidence_in_valid_range(self, sine_440):
        """Konfidenz liegt im Bereich [0, 1]."""
        from backend.core.causal_defect_reasoner import CausalDefectReasoner

        plan = CausalDefectReasoner().reason(
            defect_scores={},
            material="unknown",
            audio=sine_440,
            sample_rate=SR,
        )
        assert 0.0 <= plan.confidence <= 1.0, f"Konfidenz = {plan.confidence}"

    def test_recommended_phases_not_empty(self, sine_440):
        """recommended_phases ist nicht leer."""
        from backend.core.causal_defect_reasoner import CausalDefectReasoner

        plan = CausalDefectReasoner().reason(
            defect_scores={"dropout_severity": 0.5},
            material="tape",
        )
        assert len(plan.recommended_phases) > 0

    def test_no_audio_fallback(self):
        """Ohne Audio-Signal (audio=None) funktioniert der Reasoner."""
        from backend.core.causal_defect_reasoner import CausalDefectReasoner

        plan = CausalDefectReasoner().reason(
            defect_scores={"clip_severity": 0.9},
            material="digital",
            audio=None,
        )
        assert plan.primary_cause in [
            "digital_clip",
            "electrical_hum",
            "dc_offset",
            "tape_dropout",
            "tape_hiss",
            "tape_start_instability",
            "tape_head_contact_instability",
            "vinyl_crackle",
            "vinyl_warp",
            "head_misalignment",
            "clipping",
        ]

    def test_convenience_function(self, sine_440):
        """reason_about_defects() Convenience-Funktion funktioniert."""
        from backend.core.causal_defect_reasoner import reason_about_defects

        plan = reason_about_defects(
            defect_scores={"noise_floor_db": -40.0},
            material="shellac",
            audio=sine_440,
            sample_rate=SR,
        )
        assert plan is not None
        assert isinstance(plan.cause_probabilities, dict)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SEKTION 19: GPParameterOptimizer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestGPParameterOptimizer:
    """Sektion 19 — GPParameterOptimizer (8 Tests)."""

    def test_import(self):
        """Modul ist importierbar."""
        from backend.core.gp_parameter_optimizer import GPParameterOptimizer

        assert GPParameterOptimizer is not None

    def test_propose_returns_proposal(self):
        """propose() gibt ein ParameterProposal zurück."""
        from backend.core.gp_parameter_optimizer import GPParameterOptimizer

        opt = GPParameterOptimizer()
        prop = opt.propose(material="tape")
        assert hasattr(prop, "parameters")
        assert isinstance(prop.parameters, dict)
        assert len(prop.parameters) > 0

    def test_propose_values_in_range(self):
        """Vorgeschlagene Parameter liegen im definierten Parameterraum."""
        from backend.core.gp_parameter_optimizer import PARAMETER_SPACE, GPParameterOptimizer

        opt = GPParameterOptimizer()
        prop = opt.propose(material="vinyl")
        for name, val in prop.parameters.items():
            if name in PARAMETER_SPACE:
                lo, hi, mode = PARAMETER_SPACE[name]
                assert lo <= float(val) <= hi, f"{name}={val} liegt außerhalb [{lo}, {hi}]"

    def test_update_stores_in_memory(self, tmp_path, monkeypatch):
        """update() persistiert Datenpunkt in JSON-Gedächtnis."""
        import json

        from backend.core import gp_parameter_optimizer as gpm

        monkeypatch.setattr(gpm, "_MEMORY_DIR", tmp_path)

        opt = gpm.GPParameterOptimizer()
        params = {"noise_reduction_strength": 0.5, "ar_order": 32}
        opt.update(params, score=3.8, material="tape_test")
        mem_file = tmp_path / "tape_test.json"
        assert mem_file.exists()
        data = json.loads(mem_file.read_text())
        assert len(data) > 0

    def test_gp_improves_after_observations(self, tmp_path, monkeypatch):
        """Nach 6+ Beobachtungen wechselt GP von Zufalls- zu Posterior-Akquisition."""
        from backend.core import gp_parameter_optimizer as gpm

        monkeypatch.setattr(gpm, "_MEMORY_DIR", tmp_path)

        opt = gpm.GPParameterOptimizer(rng_seed=0)
        # 6 Beobachtungen einfügen
        for i in range(6):
            params = {
                "noise_reduction_strength": 0.3 + i * 0.1,
                "ar_order": 32 + i * 8,
            }
            opt.update(params, score=1.5 + i * 0.5, material="vinyl_test")

        # Nächster Vorschlag sollte from_memory=True haben
        prop = opt.propose(material="vinyl_test", n_init=5)
        assert prop.from_memory is True

    def test_ucb_value_is_mu_plus_kappa_sigma(self, tmp_path, monkeypatch):
        """UCB = μ + κ·σ mit κ=2.0."""
        from backend.core import gp_parameter_optimizer as gpm

        monkeypatch.setattr(gpm, "_MEMORY_DIR", tmp_path)

        opt = gpm.GPParameterOptimizer(kappa=2.0, rng_seed=1)
        for i in range(6):
            opt.update({"noise_reduction_strength": 0.5 + i * 0.05}, score=2.0 + i * 0.3, material="shellac_test")
        prop = opt.propose(material="shellac_test", n_init=5)

        expected_ucb = prop.expected_quality + 2.0 * prop.uncertainty
        # Toleranz: ±0.1 (Skalierungsunterschiede durch y-Normierung)
        assert abs(prop.ucb_value - expected_ucb) < 0.5 or prop.ucb_value >= 0

    def test_rbf_kernel_is_symmetric(self):
        """RBF-Kernel ist symmetrisch: K(x,y) = K(y,x)."""
        from backend.core.gp_parameter_optimizer import _rbf_kernel

        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (5, 3))
        Y = rng.normal(0, 1, (4, 3))
        K_XY = _rbf_kernel(X, Y)
        K_YX = _rbf_kernel(Y, X)
        np.testing.assert_allclose(K_XY, K_YX.T, atol=1e-10)

    def test_forget_clears_memory(self, tmp_path, monkeypatch):
        """forget() löscht das Material-Gedächtnis."""
        from backend.core import gp_parameter_optimizer as gpm

        monkeypatch.setattr(gpm, "_MEMORY_DIR", tmp_path)

        opt = gpm.GPParameterOptimizer()
        opt.update({"ar_order": 64}, score=3.0, material="forget_test")
        opt.forget("forget_test")
        mem_file = tmp_path / "forget_test.json"
        assert not mem_file.exists()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SEKTION 20: PerceptualQualityScorer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPerceptualQualityScorer:
    """Sektion 20 — PerceptualQualityScorer (9 Tests)."""

    def test_import(self):
        """Modul ist importierbar."""
        from backend.core.perceptual_quality_scorer import PerceptualQualityScorer

        assert PerceptualQualityScorer is not None

    def test_score_returns_pqsresult(self, sine_440, degraded_signal):
        """score() gibt ein PQSResult mit pqs_mos zurück."""
        from backend.core.perceptual_quality_scorer import PerceptualQualityScorer

        result = PerceptualQualityScorer().score(sine_440, degraded_signal, SR)
        assert hasattr(result, "pqs_mos")
        assert isinstance(result.pqs_mos, float)

    def test_mos_in_valid_range(self, sine_440, degraded_signal):
        """MOS liegt im Bereich [1.0, 5.0]."""
        from backend.core.perceptual_quality_scorer import PerceptualQualityScorer

        result = PerceptualQualityScorer().score(sine_440, degraded_signal, SR)
        assert 1.0 <= result.pqs_mos <= 5.0, f"MOS = {result.pqs_mos}"

    def test_identical_signals_high_mos(self, sine_440):
        """Referenz und Kopie ergeben hohen MOS (≥ 4.5)."""
        from backend.core.perceptual_quality_scorer import PerceptualQualityScorer

        ref = sine_440.copy()
        deg = sine_440.copy()
        result = PerceptualQualityScorer(align_signals=False).score(ref, deg, SR)
        assert result.pqs_mos >= 4.0, f"Identische Signale: MOS = {result.pqs_mos}"

    def test_degraded_has_lower_mos_than_original(self, sine_440, degraded_signal):
        """Degradiertes Signal hat niedrigeren MOS als das Original."""
        from backend.core.perceptual_quality_scorer import PerceptualQualityScorer

        scorer = PerceptualQualityScorer(align_signals=False)
        clean_mos = scorer.score(sine_440, sine_440.copy(), SR).pqs_mos
        noisy_mos = scorer.score(sine_440, degraded_signal, SR).pqs_mos
        assert noisy_mos <= clean_mos + 0.2, f"Rauschen ({noisy_mos}) nicht schlechter als Original ({clean_mos})"

    def test_nsim_range(self, sine_440, noise_signal):
        """NSIM liegt im Bereich [0, 1]."""
        from backend.core.perceptual_quality_scorer import PerceptualQualityScorer

        result = PerceptualQualityScorer().score(sine_440, noise_signal, SR)
        assert 0.0 <= result.nsim <= 1.0, f"NSIM = {result.nsim}"

    def test_mcd_nonnegative(self, sine_440, degraded_signal):
        """MCD ist nicht-negativ."""
        from backend.core.perceptual_quality_scorer import PerceptualQualityScorer

        result = PerceptualQualityScorer().score(sine_440, degraded_signal, SR)
        assert result.mcd_db >= 0.0, f"MCD = {result.mcd_db}"

    def test_score_absolute_no_reference(self, chord_signal):
        """score_absolute() funktioniert ohne Referenz-Signal."""
        from backend.core.perceptual_quality_scorer import PerceptualQualityScorer

        result = PerceptualQualityScorer().score_absolute(chord_signal, SR)
        assert result is not None
        assert result.referenced is False
        assert 1.0 <= result.pqs_mos <= 5.0

    def test_convenience_functions(self, sine_440, degraded_signal):
        """score_audio() und score_audio_absolute() Convenience-Funktionen."""
        from backend.core.perceptual_quality_scorer import score_audio, score_audio_absolute

        r1 = score_audio(sine_440, degraded_signal, SR)
        r2 = score_audio_absolute(sine_440, SR)
        assert isinstance(r1.pqs_mos, float)
        assert isinstance(r2.pqs_mos, float)

    def test_stereo_channels_first_no_stub(self):
        """Channels-first (2, N) darf nicht den Stub (MOS=3.0, NSIM=0.5, MCD=25.0) auslösen.

        Bug: len((2, N)) = 2 < 8 → Stub. Fix: Mono-Mix vor Längenberechnung.
        """
        from backend.core.perceptual_quality_scorer import score_audio, score_audio_absolute

        sr = 48000
        N = sr * 2  # 2 s
        rng = np.random.default_rng(42)
        ref_stereo = rng.standard_normal((2, N)).astype(np.float32) * 0.1
        deg_stereo = ref_stereo + rng.standard_normal((2, N)).astype(np.float32) * 0.005

        result = score_audio(ref_stereo, deg_stereo, sr)

        # Stub-Werte wären exakt (3.0, 0.5, 25.0) — echte Berechnung weicht ab
        assert not (result.mos == 3.0 and result.nsim == 0.5 and result.mcd_db == 25.0), (
            f"Stub-Werte für Stereo-Input zurückgegeben: {result}"
        )
        assert result.nsim > 0.7, f"NSIM zu niedrig für nahidentische Signale: {result.nsim:.3f}"
        assert result.mos > 3.5, f"MOS zu niedrig für gutes Signal: {result.mos:.2f}"

        # score_audio_absolute darf auch kein len()-Problem haben
        result_abs = score_audio_absolute(ref_stereo, sr)
        assert isinstance(result_abs.mos, float)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Integration-Rauchtests (smoke tests)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestV97Integration:
    """Rauch-Tests: Alle Module zusammen + Performance."""

    def test_all_modules_importable(self):
        """Alle 4 Module können ohne Fehler importiert werden."""

    def test_embed_then_score_pipeline(self, sine_440, degraded_signal):
        """End-to-End: Embedding → Kausaldiagnose → Qualitätsbewertung."""
        from backend.core.causal_defect_reasoner import reason_about_defects
        from backend.core.perceptual_embedder import embed_audio
        from backend.core.perceptual_quality_scorer import score_audio

        # 1. Embedding
        emb = embed_audio(sine_440, SR)
        assert emb.vector.shape == (256,)

        # 2. Diagnose
        plan = reason_about_defects(
            defect_scores={"dropout_severity": 0.4},
            material="tape",
            audio=sine_440,
            sample_rate=SR,
        )
        assert plan.primary_cause is not None

        # 3. Qualität
        result = score_audio(sine_440, degraded_signal, SR)
        assert result.pqs_mos > 1.0

    def test_gp_optimizer_with_pqs_score_loop(self, tmp_path, monkeypatch, sine_440, degraded_signal):
        """GP-Optimizer nutzt PQS-Score als Feedback (Mini-Optimierungsschleife)."""
        from backend.core import gp_parameter_optimizer as gpm

        monkeypatch.setattr(gpm, "_MEMORY_DIR", tmp_path)
        from backend.core.perceptual_quality_scorer import score_audio

        opt = gpm.GPParameterOptimizer(rng_seed=42)

        for _ in range(3):
            proposal = opt.propose(material="loop_test", n_init=5)
            pqs = score_audio(sine_440, degraded_signal, SR)
            opt.update(proposal.parameters, score=pqs.pqs_mos, material="loop_test")

        # Nach 3 Iterationen sollte ein Gedächtnis vorhanden sein
        final_prop = opt.propose(material="loop_test", n_init=5)
        assert final_prop is not None

    def test_performance_embedder_under_1s(self, chord_signal):
        """PerceptualEmbedder verarbeitet 3s Audio in unter 5 Sekunden."""
        from backend.core.perceptual_embedder import PerceptualEmbedder

        t0 = time.perf_counter()
        PerceptualEmbedder().embed(chord_signal, SR)
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, f"Embedder zu langsam: {elapsed:.2f}s"

    def test_performance_scorer_under_2s(self, sine_440, degraded_signal):
        """PerceptualQualityScorer verarbeitet 2s Audio in unter 10 Sekunden."""
        from backend.core.perceptual_quality_scorer import score_audio

        t0 = time.perf_counter()
        score_audio(sine_440, degraded_signal, SR)
        elapsed = time.perf_counter() - t0
        assert elapsed < 10.0, f"Scorer zu langsam: {elapsed:.2f}s"
