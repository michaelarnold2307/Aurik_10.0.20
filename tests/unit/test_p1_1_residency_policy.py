"""§SOTA-P1-1 (Modell-Residency & Warm-up-Policy) — Tests der Policy-Ebene.

Deckt ``backend/core/ml/residency_policy.py``:
- Residency-Tiers (ALWAYS/SESSION/ONESHOT) inkl. Namens-Normalisierung
- Warm-up-Amortisierung: EINMAL je Prozess, idempotent, fail-closed (§V6 (copilot-instructions.md))
- §G1-Seed-Ableitung: deterministisch, song-isoliert, Reihenfolge-unabhängig

Autor: Aurik Testing Team
"""

import backend.core.ml.residency_policy as rp


class TestResidencyTiers:
    def test_always_models(self):
        for name in ("Resemblyzer", "PANNs", "BEATs", "MuQ", "DeepFilterNet", "HiFiGAN"):
            assert rp.residency_tier_of(name) is rp.ResidencyTier.ALWAYS, name

    def test_session_models(self):
        for name in ("MelBandRoformer", "Demucs", "CQTdiff+", "GaCELA", "EAR-VAE"):
            assert rp.residency_tier_of(name) is rp.ResidencyTier.SESSION, name

    def test_oneshot_models(self):
        for name in ("FlashSR", "BigVGAN", "AudioLDM2", "UTMOSv2", "Miipher_DiT"):
            assert rp.residency_tier_of(name) is rp.ResidencyTier.ONESHOT, name

    def test_name_normalization(self):
        assert rp.residency_tier_of("panns_plugin") is rp.ResidencyTier.ALWAYS
        assert rp.residency_tier_of("bs_roformer_torch") is rp.ResidencyTier.SESSION
        assert rp.residency_tier_of("PAnNs-onnx") is rp.ResidencyTier.ALWAYS

    def test_unknown_is_conservative_session(self):
        assert rp.residency_tier_of("völlig_unbekanntes_modell") is rp.ResidencyTier.SESSION

    def test_should_keep_warm(self):
        assert rp.should_keep_warm("Resemblyzer") is True
        assert rp.should_keep_warm("FlashSR") is False


class TestWarmupAmortization:
    def teardown_method(self):
        rp.reset_warmup_registry()

    def test_warmup_runs_once(self):
        calls = {"n": 0}

        def warm():
            calls["n"] += 1

        assert rp.warmup_once("PANNs", warm) is True
        assert rp.warmup_once("PANNs", warm) is False
        assert rp.warmup_once("panns_plugin", warm) is False  # Normalisierung
        assert calls["n"] == 1
        assert rp.is_warmed("PANNs") is True

    def test_warmup_failure_is_logged_and_marked(self):
        def boom():
            raise RuntimeError("kaputt")

        assert rp.warmup_once("Resemblyzer", boom) is True  # ausgeführt (und markiert)
        assert rp.warmup_once("Resemblyzer", boom) is False  # kein erneuter Versuch
        assert rp.is_warmed("Resemblyzer") is True


class TestSongSeed:
    def test_deterministic(self):
        assert rp.song_seed("elke_best_30s", 42) == rp.song_seed("elke_best_30s", 42)

    def test_song_isolated(self):
        s1 = rp.song_seed("song_a", 42)
        s2 = rp.song_seed("song_b", 42)
        assert s1 != s2

    def test_order_independent(self):
        # Gleiche Ableitung unabhängig davon, in welcher Reihenfolge die Songs
        # im Batch verarbeitet werden (kein veränderlicher Zustand).
        a = rp.song_seed("song_a", 42)
        b = rp.song_seed("song_b", 42)
        assert rp.song_seed("song_a", 42) == a
        assert rp.song_seed("song_b", 42) == b

    def test_master_seed_affects_song_seed(self):
        assert rp.song_seed("song_a", 1) != rp.song_seed("song_a", 2)

    def test_range(self):
        for i in range(5):
            s = rp.song_seed(f"song_{i}", 42)
            assert 0 <= s < 2**31

    def test_master_seed_from_env_default(self, monkeypatch):
        monkeypatch.delenv("AURIK_MASTER_SEED", raising=False)
        assert rp.master_seed_from_env() == 42
        monkeypatch.setenv("AURIK_MASTER_SEED", "7")
        assert rp.master_seed_from_env() == 7
        monkeypatch.setenv("AURIK_MASTER_SEED", "kein-int")
        assert rp.master_seed_from_env() == 42
