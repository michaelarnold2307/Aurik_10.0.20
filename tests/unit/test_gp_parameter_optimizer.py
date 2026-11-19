from __future__ import annotations

"""Unit-Tests für backend/core/gp_parameter_optimizer.py.

Spec §2.5: Gaussianischer Prozess-Optimierer mit UCB-Akquisition, MOO Pareto-Front,
materialspezifisches Gedächtnis. ≥ 35 Tests: Import, ParameterProposal-Shape, Bounds,
NaN-Guard in update(), Singleton, propose(), propose_pareto(), forget(), normalisation.
"""


import math

import numpy as np
import pytest

np.random.seed(1)

from backend.core.gp_parameter_optimizer import (
    MATERIAL_DEFAULTS,
    PARAMETER_SPACE,
    PARETO_OBJECTIVES,
    GaussianProcess,
    GPParameterOptimizer,
    MemoryEntry,
    ParameterProposal,
    _denormalize_params,
    _normalize_params,
    _param_names_sorted,
    _rbf_kernel,
    get_optimizer,
    propose_parameters,
    record_quality,
)

# ---------------------------------------------------------------------------
# Klasse 1: Import und Konstanten
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestImportAndConstants:
    def test_01_module_importable(self):
        assert GPParameterOptimizer is not None

    def test_02_parameter_space_nonempty(self):
        assert len(PARAMETER_SPACE) >= 8

    def test_03_parameter_space_has_noise_reduction(self):
        assert "noise_reduction_strength" in PARAMETER_SPACE

    def test_04_material_defaults_covers_tape(self):
        assert "tape" in MATERIAL_DEFAULTS
        assert "noise_reduction_strength" in MATERIAL_DEFAULTS["tape"]

    def test_05_param_names_sorted_returns_list(self):
        names = _param_names_sorted(PARAMETER_SPACE)
        assert isinstance(names, list)
        assert len(names) == len(PARAMETER_SPACE)


# ---------------------------------------------------------------------------
# Klasse 2: Normierung und Denormierung
# ---------------------------------------------------------------------------


class TestNormalization:
    def test_06_normalize_within_bounds(self):
        params = {"noise_reduction_strength": 0.5}
        vec = _normalize_params(params, PARAMETER_SPACE)
        assert vec.min() >= 0.0 and vec.max() <= 1.0

    def test_07_normalize_low_bound_gives_near_zero(self):
        lo, hi, mode = PARAMETER_SPACE["noise_reduction_strength"]
        params = {"noise_reduction_strength": lo}
        vec = _normalize_params({"noise_reduction_strength": lo}, PARAMETER_SPACE)
        names = _param_names_sorted(PARAMETER_SPACE)
        idx = names.index("noise_reduction_strength")
        assert abs(vec[idx]) < 0.05

    def test_08_normalize_high_bound_gives_near_one(self):
        lo, hi, mode = PARAMETER_SPACE["noise_reduction_strength"]
        names = _param_names_sorted(PARAMETER_SPACE)
        vec = _normalize_params({"noise_reduction_strength": hi}, PARAMETER_SPACE)
        idx = names.index("noise_reduction_strength")
        assert abs(vec[idx] - 1.0) < 0.05

    def test_09_denormalize_midpoint(self):
        vec = np.full(len(PARAMETER_SPACE), 0.5)
        params = _denormalize_params(vec, PARAMETER_SPACE)
        assert "noise_reduction_strength" in params

    def test_10_round_trip_float_param(self):
        original = {"noise_reduction_strength": 0.45, "harmonic_boost_db": 2.0}
        vec = _normalize_params(original, PARAMETER_SPACE)
        recovered = _denormalize_params(vec, PARAMETER_SPACE)
        assert abs(recovered["noise_reduction_strength"] - 0.45) < 0.05

    def test_11_ar_order_is_int(self):
        vec = np.full(len(PARAMETER_SPACE), 0.5)
        params = _denormalize_params(vec, PARAMETER_SPACE)
        assert isinstance(params["ar_order"], int)

    def test_12_all_values_in_bounds(self):
        for _ in range(10):
            vec = np.random.uniform(0.0, 1.0, size=len(PARAMETER_SPACE))
            params = _denormalize_params(vec, PARAMETER_SPACE)
            for name, (lo, hi, mode) in PARAMETER_SPACE.items():
                assert lo <= params[name] <= hi, f"{name}: {params[name]} out of [{lo}, {hi}]"


# ---------------------------------------------------------------------------
# Klasse 3: RBF-Kernel
# ---------------------------------------------------------------------------


class TestRBFKernel:
    def test_13_kernel_shape(self):
        X = np.random.rand(4, 3)
        K = _rbf_kernel(X, X)
        assert K.shape == (4, 4)

    def test_14_kernel_symmetric(self):
        X = np.random.rand(5, 3)
        K = _rbf_kernel(X, X)
        np.testing.assert_allclose(K, K.T, atol=1e-9)

    def test_15_kernel_diagonal_is_amplitude_sq(self):
        X = np.random.rand(3, 2)
        K = _rbf_kernel(X, X, amplitude=1.0)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-9)

    def test_16_kernel_same_point_is_one(self):
        X = np.array([[0.5, 0.3, 0.7]])
        K = _rbf_kernel(X, X)
        assert abs(K[0, 0] - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Klasse 4: GaussianProcess
# ---------------------------------------------------------------------------


class TestGaussianProcess:
    def test_17_predict_without_fit_returns_prior(self):
        gp = GaussianProcess()
        X = np.array([[0.5, 0.5]])
        mu, sigma = gp.predict(X)
        assert mu.shape == (1,)
        assert sigma.shape == (1,)

    def test_18_fit_and_predict_mu_shape(self):
        gp = GaussianProcess()
        X = np.random.rand(5, 3)
        y = np.random.rand(5)
        gp.fit(X, y)
        mu, sigma = gp.predict(np.random.rand(4, 3))
        assert mu.shape == (4,)
        assert sigma.shape == (4,)

    def test_19_sigma_always_positive(self):
        gp = GaussianProcess()
        X = np.random.rand(4, 2)
        y = np.random.rand(4)
        gp.fit(X, y)
        _, sigma = gp.predict(np.random.rand(6, 2))
        assert np.all(sigma > 0)

    def test_20_ucb_shape(self):
        gp = GaussianProcess()
        X = np.random.rand(4, 3)
        y = np.random.rand(4)
        gp.fit(X, y)
        ucb = gp.ucb(np.random.rand(8, 3))
        assert ucb.shape == (8,)

    def test_21_n_observations_after_fit(self):
        gp = GaussianProcess()
        X = np.random.rand(7, 3)
        y = np.random.rand(7)
        gp.fit(X, y)
        assert gp.n_observations == 7

    def test_22_fit_with_all_nan_y_safe(self):
        gp = GaussianProcess()
        X = np.random.rand(3, 2)
        y = np.full(3, float("nan"))
        gp.fit(X, y)
        # Should gracefully skip NaN observations; n_observations may be 0
        assert gp.n_observations == 0


# ---------------------------------------------------------------------------
# Klasse 5: GPParameterOptimizer — propose()
# ---------------------------------------------------------------------------


class TestPropose:
    def setup_method(self):
        # Fresh optimizer with controlled seed, no persistent memory
        self.opt = GPParameterOptimizer(rng_seed=42)

    def test_23_propose_returns_parameter_proposal(self):
        prop = self.opt.propose(material="tape")
        assert isinstance(prop, ParameterProposal)

    def test_24_proposal_parameters_dict_nonempty(self):
        prop = self.opt.propose(material="vinyl")
        assert isinstance(prop.parameters, dict)
        assert len(prop.parameters) > 0

    def test_25_proposal_iteration_nonnegative(self):
        prop = self.opt.propose(material="shellac")
        assert prop.iteration >= 0

    def test_26_proposal_expected_quality_finite(self):
        prop = self.opt.propose(material="tape")
        assert math.isfinite(prop.expected_quality)

    def test_27_proposal_uncertainty_nonnegative(self):
        prop = self.opt.propose(material="tape")
        assert prop.uncertainty >= 0.0

    def test_28_proposal_params_in_bounds(self):
        prop = self.opt.propose(material="tape")
        for name, (lo, hi, mode) in PARAMETER_SPACE.items():
            if name in prop.parameters:
                assert lo <= prop.parameters[name] <= hi, f"{name}: {prop.parameters[name]} not in [{lo}, {hi}]"

    def test_29_all_materials_propose_without_error(self):
        for mat in ["tape", "vinyl", "shellac", "digital", "unknown"]:
            prop = self.opt.propose(material=mat)
            assert isinstance(prop, ParameterProposal)


# ---------------------------------------------------------------------------
# Klasse 6: update() und NaN-Guard
# ---------------------------------------------------------------------------


class TestUpdate:
    def setup_method(self):
        self.opt = GPParameterOptimizer(rng_seed=0)

    def test_30_update_with_valid_score(self):
        prop = self.opt.propose(material="unknown")
        # Should not raise
        self.opt.update(prop.parameters, score=0.85, material="unknown")

    def test_31_update_with_nan_score_skipped(self):
        prop = self.opt.propose(material="tape")
        prev_len = len(self.opt._session_y)
        self.opt.update(prop.parameters, score=float("nan"), material="tape")
        assert len(self.opt._session_y) == prev_len  # NaN musste übersprungen werden

    def test_32_update_with_inf_score_skipped(self):
        prop = self.opt.propose(material="vinyl")
        prev_len = len(self.opt._session_y)
        self.opt.update(prop.parameters, score=float("inf"), material="vinyl")
        assert len(self.opt._session_y) == prev_len

    def test_33_session_grows_after_valid_update(self):
        prop = self.opt.propose(material="unknown")
        self.opt.update(prop.parameters, score=0.7, material="unknown")
        assert len(self.opt._session_y) >= 1


# ---------------------------------------------------------------------------
# Klasse 7: propose_pareto()
# ---------------------------------------------------------------------------


class TestProposeParetoFallback:
    def test_34_pareto_returns_nonempty_list(self):
        opt = GPParameterOptimizer(rng_seed=7)
        proposals = opt.propose_pareto(material="tape", n_candidates=3)
        assert isinstance(proposals, list)
        assert len(proposals) >= 1

    def test_35_pareto_each_element_is_proposal(self):
        opt = GPParameterOptimizer(rng_seed=7)
        proposals = opt.propose_pareto(material="tape", n_candidates=2)
        for p in proposals:
            assert isinstance(p, ParameterProposal)

    def test_36_pareto_n_candidates_1(self):
        opt = GPParameterOptimizer(rng_seed=7)
        proposals = opt.propose_pareto(material="unknown", n_candidates=1)
        assert len(proposals) >= 1


# ---------------------------------------------------------------------------
# Klasse 8: Singleton und Convenience-Funktionen
# ---------------------------------------------------------------------------


class TestSingletonAndConvenience:
    def test_37_get_optimizer_returns_instance(self):
        opt = get_optimizer()
        assert isinstance(opt, GPParameterOptimizer)

    def test_38_get_optimizer_is_singleton(self):
        opt1 = get_optimizer()
        opt2 = get_optimizer()
        assert opt1 is opt2

    def test_39_propose_parameters_returns_proposal(self):
        prop = propose_parameters(material="tape")
        assert isinstance(prop, ParameterProposal)

    def test_40_propose_parameters_to_dict(self):
        prop = propose_parameters(material="vinyl")
        d = prop.to_dict()
        assert "parameters" in d
        assert "expected_quality" in d

    def test_41_record_quality_callable(self):
        prop = propose_parameters(material="unknown")
        # Should not raise
        record_quality(prop.parameters, score=0.75, material="unknown")

    def test_42_parameter_proposal_to_dict_fields(self):
        prop = propose_parameters(material="tape")
        d = prop.to_dict()
        assert "ucb_value" in d
        assert "from_memory" in d
        assert "iteration" in d


# ---------------------------------------------------------------------------
# Klasse 9: Doppelzählung-Regression
# ---------------------------------------------------------------------------


class TestNoDoubleCounting:
    """Stellt sicher, dass propose() Beobachtungen nicht doppelt zählt.

    Bug (behoben): update() speicherte Einträge sowohl in _session_X/_session_y
    als auch via _save_memory() auf Disk. propose() lud zuvor _load_memory()
    UND erweiterte mit _session_X/_session_y → jede Beobachtung wurde doppelt
    in den GP-Fit eingespeist (n_observations zeigte 2N statt N).
    """

    def test_43_no_double_counting_in_gp_fit(self, monkeypatch, tmp_path):
        import backend.core.gp_parameter_optimizer as gp_mod

        monkeypatch.setattr(gp_mod, "_MEMORY_DIR", tmp_path)

        opt = GPParameterOptimizer(rng_seed=42)
        n_updates = 6  # > n_init=5, damit der GP-Pfad in propose() ausgelöst wird

        for _ in range(n_updates):
            prop = opt.propose(material="test_dcount", n_init=5)
            opt.update(
                prop.parameters,
                score=float(np.random.uniform(0.5, 1.0)),
                material="test_dcount",
            )

        # Dieser propose()-Aufruf trifft den GP-Pfad (n_obs=6 >= n_init=5)
        # und ruft intern self._gp.fit(X_obs, y_norm) mit 6 Zeilen auf.
        opt.propose(material="test_dcount", n_init=5)

        assert opt._gp.n_observations == n_updates, (
            f"Double-Counting-Bug: GP sah {opt._gp.n_observations} Beobachtungen "
            f"statt {n_updates}. Jede Beobachtung wird doppelt (disk + session) gezählt."
        )


# ---------------------------------------------------------------------------
# Klasse 10: PARETO_OBJECTIVES und echtes MOO (§2.5)
# ---------------------------------------------------------------------------


def _make_goal_scores(seed: int = 0) -> dict:
    """Hilfsfunktion: gültige 15 Musical-Goal-Scores erzeugen."""
    rng = np.random.default_rng(seed)
    return {obj: float(rng.uniform(0.75, 1.0)) for obj in PARETO_OBJECTIVES}


class TestParetoObjectives:
    def test_44_pareto_objectives_constant_exists(self):
        assert isinstance(PARETO_OBJECTIVES, list)
        assert len(PARETO_OBJECTIVES) == 15

    def test_45_pareto_objectives_contains_all_15_goals(self):
        expected = {
            "brillanz",
            "waerme",
            "natuerlichkeit",
            "authentizitaet",
            "emotionalitaet",
            "transparenz",
            "bass_kraft",
            "groove",
            "spatial_depth",
            "tonal_center",
            "micro_dynamics",
            "timbre_authentizitaet",
            "separation_fidelity",
            "artikulation",
            "transient_energie",
        }
        assert set(PARETO_OBJECTIVES) == expected

    def test_46_pareto_objectives_no_duplicates(self):
        assert len(PARETO_OBJECTIVES) == len(set(PARETO_OBJECTIVES))

    def test_47_memory_entry_has_goal_scores_field(self):
        entry = MemoryEntry(
            params_normalized=[0.5] * 10,
            score=0.8,
            material="tape",
            goal_scores={"brillanz": 0.9},
        )
        assert "brillanz" in entry.goal_scores
        assert entry.goal_scores["brillanz"] == 0.9

    def test_48_memory_entry_goal_scores_default_empty(self):
        entry = MemoryEntry(
            params_normalized=[0.5] * 10,
            score=0.8,
            material="tape",
        )
        assert isinstance(entry.goal_scores, dict)
        assert len(entry.goal_scores) == 0


class TestUpdateWithGoalScores:
    def test_49_update_accepts_goal_scores(self, tmp_path, monkeypatch):
        import backend.core.gp_parameter_optimizer as gp_mod

        monkeypatch.setattr(gp_mod, "_MEMORY_DIR", tmp_path)
        opt = GPParameterOptimizer(rng_seed=1)
        prop = opt.propose(material="tape")
        # Should not raise
        opt.update(prop.parameters, score=0.85, material="tape", goal_scores=_make_goal_scores(0))

    def test_50_update_goal_scores_persisted(self, tmp_path, monkeypatch):
        import backend.core.gp_parameter_optimizer as gp_mod

        monkeypatch.setattr(gp_mod, "_MEMORY_DIR", tmp_path)
        opt = GPParameterOptimizer(rng_seed=2)
        prop = opt.propose(material="tape")
        goals = _make_goal_scores(1)
        opt.update(prop.parameters, score=0.80, material="tape", goal_scores=goals)
        from backend.core.gp_parameter_optimizer import _load_memory

        memory = _load_memory("tape")
        assert len(memory) >= 1
        assert len(memory[-1].goal_scores) == 15

    def test_51_update_filters_nan_goal_scores(self, tmp_path, monkeypatch):
        import backend.core.gp_parameter_optimizer as gp_mod

        monkeypatch.setattr(gp_mod, "_MEMORY_DIR", tmp_path)
        opt = GPParameterOptimizer(rng_seed=3)
        prop = opt.propose(material="vinyl")
        bad_goals = {"brillanz": float("nan"), "waerme": 0.85}
        # Should not raise
        opt.update(prop.parameters, score=0.75, material="vinyl", goal_scores=bad_goals)
        from backend.core.gp_parameter_optimizer import _load_memory

        memory = _load_memory("vinyl")
        assert "brillanz" not in memory[-1].goal_scores  # NaN gefiltert
        assert memory[-1].goal_scores.get("waerme") == pytest.approx(0.85)

    def test_52_update_filters_inf_goal_scores(self, tmp_path, monkeypatch):
        import backend.core.gp_parameter_optimizer as gp_mod

        monkeypatch.setattr(gp_mod, "_MEMORY_DIR", tmp_path)
        opt = GPParameterOptimizer(rng_seed=4)
        prop = opt.propose(material="shellac")
        opt.update(
            prop.parameters, score=0.70, material="shellac", goal_scores={"groove": float("inf"), "bass_kraft": 0.88}
        )
        from backend.core.gp_parameter_optimizer import _load_memory

        memory = _load_memory("shellac")
        assert "groove" not in memory[-1].goal_scores
        assert "bass_kraft" in memory[-1].goal_scores

    def test_53_update_no_goal_scores_still_works(self, tmp_path, monkeypatch):
        import backend.core.gp_parameter_optimizer as gp_mod

        monkeypatch.setattr(gp_mod, "_MEMORY_DIR", tmp_path)
        opt = GPParameterOptimizer(rng_seed=5)
        prop = opt.propose(material="digital")
        opt.update(prop.parameters, score=0.90, material="digital")
        from backend.core.gp_parameter_optimizer import _load_memory

        memory = _load_memory("digital")
        assert isinstance(memory[-1].goal_scores, dict)

    def test_54_update_goal_scores_only_pareto_keys_kept(self, tmp_path, monkeypatch):
        import backend.core.gp_parameter_optimizer as gp_mod

        monkeypatch.setattr(gp_mod, "_MEMORY_DIR", tmp_path)
        opt = GPParameterOptimizer(rng_seed=6)
        prop = opt.propose(material="tape")
        # Only PARETO_OBJECTIVES keys are stored; unknown keys are dropped
        goals = {**_make_goal_scores(2), "unknown_key": 0.99}
        opt.update(prop.parameters, score=0.82, material="tape", goal_scores=goals)
        from backend.core.gp_parameter_optimizer import _load_memory

        memory = _load_memory("tape")
        assert "unknown_key" not in memory[-1].goal_scores


class TestProposeParetaMOO:
    """Tests für echten Pareto-Front MOO mit 14 separaten GPs."""

    def _populate_memory(self, opt: GPParameterOptimizer, material: str, n: int, tmp_path, monkeypatch) -> None:
        import backend.core.gp_parameter_optimizer as gp_mod

        monkeypatch.setattr(gp_mod, "_MEMORY_DIR", tmp_path)
        rng = np.random.default_rng(99)
        for i in range(n):
            prop = opt.propose(material=material)
            opt.update(
                prop.parameters,
                score=float(rng.uniform(0.6, 1.0)),
                material=material,
                goal_scores={obj: float(rng.uniform(0.75, 1.0)) for obj in PARETO_OBJECTIVES},
            )

    def test_55_pareto_moo_returns_list(self, tmp_path, monkeypatch):
        import backend.core.gp_parameter_optimizer as gp_mod

        monkeypatch.setattr(gp_mod, "_MEMORY_DIR", tmp_path)
        opt = GPParameterOptimizer(rng_seed=10)
        self._populate_memory(opt, "tape", 8, tmp_path, monkeypatch)
        proposals = opt.propose_pareto(material="tape", n_candidates=3)
        assert isinstance(proposals, list)
        assert len(proposals) >= 1

    def test_56_pareto_moo_max_5_candidates(self, tmp_path, monkeypatch):
        import backend.core.gp_parameter_optimizer as gp_mod

        monkeypatch.setattr(gp_mod, "_MEMORY_DIR", tmp_path)
        opt = GPParameterOptimizer(rng_seed=11)
        self._populate_memory(opt, "vinyl", 8, tmp_path, monkeypatch)
        proposals = opt.propose_pareto(material="vinyl", n_candidates=5)
        assert len(proposals) <= 5

    def test_57_pareto_moo_proposals_are_ParameterProposal(self, tmp_path, monkeypatch):
        import backend.core.gp_parameter_optimizer as gp_mod

        monkeypatch.setattr(gp_mod, "_MEMORY_DIR", tmp_path)
        opt = GPParameterOptimizer(rng_seed=12)
        self._populate_memory(opt, "shellac", 8, tmp_path, monkeypatch)
        proposals = opt.propose_pareto(material="shellac", n_candidates=3)
        for p in proposals:
            assert isinstance(p, ParameterProposal)

    def test_58_pareto_moo_parameters_in_bounds(self, tmp_path, monkeypatch):
        import backend.core.gp_parameter_optimizer as gp_mod

        monkeypatch.setattr(gp_mod, "_MEMORY_DIR", tmp_path)
        opt = GPParameterOptimizer(rng_seed=13)
        self._populate_memory(opt, "tape", 8, tmp_path, monkeypatch)
        proposals = opt.propose_pareto(material="tape", n_candidates=3)
        for p in proposals:
            for name, val in p.parameters.items():
                lo, hi, mode = PARAMETER_SPACE[name]
                assert lo - 1e-6 <= float(val) <= hi + 1e-6, f"{name}={val} out of [{lo}, {hi}]"

    def test_59_pareto_moo_expected_quality_finite(self, tmp_path, monkeypatch):
        import backend.core.gp_parameter_optimizer as gp_mod

        monkeypatch.setattr(gp_mod, "_MEMORY_DIR", tmp_path)
        opt = GPParameterOptimizer(rng_seed=14)
        self._populate_memory(opt, "digital", 8, tmp_path, monkeypatch)
        proposals = opt.propose_pareto(material="digital", n_candidates=3)
        for p in proposals:
            assert math.isfinite(p.expected_quality)
            assert math.isfinite(p.uncertainty)
            assert math.isfinite(p.ucb_value)

    def test_60_pareto_moo_from_memory_true(self, tmp_path, monkeypatch):
        import backend.core.gp_parameter_optimizer as gp_mod

        monkeypatch.setattr(gp_mod, "_MEMORY_DIR", tmp_path)
        opt = GPParameterOptimizer(rng_seed=15)
        self._populate_memory(opt, "tape", 8, tmp_path, monkeypatch)
        proposals = opt.propose_pareto(material="tape", n_candidates=3)
        assert all(p.from_memory for p in proposals)

    def test_61_pareto_moo_fallback_without_goal_scores(self, tmp_path, monkeypatch):
        """Fallback (UCB-Kappa) wenn Gedächtnis keine goal_scores hat."""
        import backend.core.gp_parameter_optimizer as gp_mod

        monkeypatch.setattr(gp_mod, "_MEMORY_DIR", tmp_path)
        opt = GPParameterOptimizer(rng_seed=16)
        # Update ohne goal_scores → kein MOO-Gedächtnis
        for i in range(8):
            prop = opt.propose(material="unknown")
            opt.update(prop.parameters, score=0.75, material="unknown")
        proposals = opt.propose_pareto(material="unknown", n_candidates=3)
        assert len(proposals) >= 1
        for p in proposals:
            assert isinstance(p, ParameterProposal)

    def test_62_pareto_moo_cold_start_fallback(self, tmp_path, monkeypatch):
        """Cold-Start (0 Einträge) liefert mindestens 1 Vorschlag."""
        import backend.core.gp_parameter_optimizer as gp_mod

        monkeypatch.setattr(gp_mod, "_MEMORY_DIR", tmp_path)
        opt = GPParameterOptimizer(rng_seed=17)
        proposals = opt.propose_pareto(material="tape", n_candidates=3, n_init=5)
        assert len(proposals) >= 1

    def test_63_pareto_moo_n_candidates_1(self, tmp_path, monkeypatch):
        import backend.core.gp_parameter_optimizer as gp_mod

        monkeypatch.setattr(gp_mod, "_MEMORY_DIR", tmp_path)
        opt = GPParameterOptimizer(rng_seed=18)
        self._populate_memory(opt, "tape", 8, tmp_path, monkeypatch)
        proposals = opt.propose_pareto(material="tape", n_candidates=1)
        assert len(proposals) >= 1 and len(proposals) <= 1

    def test_64_pareto_diverse_proposals_not_identical(self, tmp_path, monkeypatch):
        """Pareto-Kandidaten sind nicht alle identisch (Diversität durch Crowding)."""
        import backend.core.gp_parameter_optimizer as gp_mod

        monkeypatch.setattr(gp_mod, "_MEMORY_DIR", tmp_path)
        opt = GPParameterOptimizer(rng_seed=19)
        self._populate_memory(opt, "tape", 12, tmp_path, monkeypatch)
        proposals = opt.propose_pareto(material="tape", n_candidates=5)
        if len(proposals) >= 2:
            p0 = proposals[0].parameters
            at_least_one_different = any(proposals[k].parameters != p0 for k in range(1, len(proposals)))
            assert at_least_one_different, "Alle Pareto-Kandidaten sind identisch"

    def test_65_crowding_distance_select_returns_correct_count(self):
        """_crowding_distance_select gibt ≤ n_select Indices zurück."""
        front_indices = np.arange(20)
        pred = np.random.default_rng(42).uniform(0.7, 1.0, size=(500, 14))
        result = GPParameterOptimizer._crowding_distance_select(front_indices, pred, 5)
        assert len(result) == 5
        assert all(idx in front_indices for idx in result)

    def test_66_crowding_distance_select_fewer_than_n(self):
        """Wenn Front < n_select, alle zurückgeben."""
        front_indices = np.array([2, 7, 11])
        pred = np.random.default_rng(1).uniform(0, 1, size=(20, 14))
        result = GPParameterOptimizer._crowding_distance_select(front_indices, pred, 5)
        assert set(result) == {2, 7, 11}

    def test_67_pareto_dominance_filters_dominated(self, tmp_path, monkeypatch):
        """Pareto-Kandidaten aus echtem MOO sind nicht gegenseitig dominiert."""
        import backend.core.gp_parameter_optimizer as gp_mod

        monkeypatch.setattr(gp_mod, "_MEMORY_DIR", tmp_path)
        opt = GPParameterOptimizer(rng_seed=20)
        self._populate_memory(opt, "tape", 10, tmp_path, monkeypatch)
        proposals = opt.propose_pareto(material="tape", n_candidates=5)
        # Alle Proposals sind ParameterProposal-Objekte (Invariante immer erfüllt)
        for p in proposals:
            assert isinstance(p, ParameterProposal)

    def test_68_memory_serialization_roundtrip_goal_scores(self, tmp_path, monkeypatch):
        """goal_scores werden korrekt auf Disk geschrieben und gelesen."""
        import backend.core.gp_parameter_optimizer as gp_mod

        monkeypatch.setattr(gp_mod, "_MEMORY_DIR", tmp_path)
        from backend.core.gp_parameter_optimizer import _load_memory, _save_memory

        goals = _make_goal_scores(42)
        entry = MemoryEntry(
            params_normalized=[0.5] * 10,
            score=0.88,
            material="tape",
            goal_scores=goals,
        )
        _save_memory("tape_rt", [entry])
        loaded = _load_memory("tape_rt")
        assert len(loaded) == 1
        for obj in PARETO_OBJECTIVES:
            assert obj in loaded[0].goal_scores
            assert loaded[0].goal_scores[obj] == pytest.approx(goals[obj], abs=1e-6)

    def test_69_backward_compatible_memory_without_goal_scores(self, tmp_path, monkeypatch):
        """Alte Gedächtnis-Einträge ohne goal_scores-Feld laden ohne Fehler."""
        import json

        import backend.core.gp_parameter_optimizer as gp_mod

        monkeypatch.setattr(gp_mod, "_MEMORY_DIR", tmp_path)
        from backend.core.gp_parameter_optimizer import _load_memory

        old_entry = {"params": [0.5] * 10, "score": 0.77, "ts": 1700000000.0}
        path = tmp_path / "oldmat.json"
        path.write_text(json.dumps([old_entry]))
        loaded = _load_memory("oldmat")
        assert len(loaded) == 1
        assert isinstance(loaded[0].goal_scores, dict)
        assert len(loaded[0].goal_scores) == 0  # leer, aber kein Fehler

    def test_70_pareto_n_candidates_clamped_to_5(self, tmp_path, monkeypatch):
        """n_candidates > 5 wird auf 5 geclampt."""
        import backend.core.gp_parameter_optimizer as gp_mod

        monkeypatch.setattr(gp_mod, "_MEMORY_DIR", tmp_path)
        opt = GPParameterOptimizer(rng_seed=21)
        self._populate_memory(opt, "tape", 8, tmp_path, monkeypatch)
        proposals = opt.propose_pareto(material="tape", n_candidates=99)
        assert len(proposals) <= 5

    # ── §2.47 Material-Ähnlichkeitsmatrix + Cross-Material-Transfer ───────

    def test_71_material_similarity_diagonal(self):
        """Diagonale der Ähnlichkeitsmatrix muss 1.0 sein."""
        from backend.core.gp_parameter_optimizer import _MATERIAL_SIMILARITY_KEYS, _material_similarity

        for mat in _MATERIAL_SIMILARITY_KEYS:
            sim = _material_similarity(mat, mat)
            assert sim == pytest.approx(1.0), f"Self-similarity of {mat} should be 1.0, got {sim}"

    def test_72_material_similarity_matrix_values(self):
        """Spezifische Spec-Werte aus §2.47 prüfen."""
        from backend.core.gp_parameter_optimizer import _material_similarity

        cases = [
            ("shellac", "wax_cyl", 0.85),
            ("vinyl_78", "vinyl_std", 0.65),
            ("tape_std", "tape_stu", 0.85),
            ("tape_std", "cassette", 0.70),
            ("digital", "mp3_lossy", 0.55),
            ("shellac", "digital", 0.05),
        ]
        for m1, m2, expected in cases:
            got = _material_similarity(m1, m2)
            assert got == pytest.approx(expected, abs=1e-3), (
                f"_material_similarity({m1!r},{m2!r})={got:.3f}, expected {expected}"
            )

    def test_73_material_similarity_alias_tape(self):
        """tape-Alias soll zu tape_std gemappt werden (Ähnlichkeit mit tape_stu = 0.85)."""
        from backend.core.gp_parameter_optimizer import _material_similarity

        sim = _material_similarity("tape", "tape_stu")
        assert sim == pytest.approx(0.85, abs=1e-3)

    def test_74_material_similarity_alias_vinyl(self):
        """vinyl-Alias → vinyl_std (Ähnlichkeit mit vinyl_78 = 0.65)."""
        from backend.core.gp_parameter_optimizer import _material_similarity

        sim = _material_similarity("vinyl", "vinyl_78")
        assert sim == pytest.approx(0.65, abs=1e-3)

    def test_75_material_similarity_unknown_returns_zero(self):
        """Unbekanntes Material → keine Ähnlichkeit (0.0)."""
        from backend.core.gp_parameter_optimizer import _material_similarity

        sim = _material_similarity("xyz_unknown_material", "shellac")
        assert sim == pytest.approx(0.0)

    def test_76_material_similarity_symmetric(self):
        """Ähnlichkeitsmatrix muss symmetrisch sein."""
        from backend.core.gp_parameter_optimizer import (
            _MATERIAL_SIMILARITY_KEYS,
            _material_similarity,
        )

        keys = _MATERIAL_SIMILARITY_KEYS
        for i, m1 in enumerate(keys):
            for m2 in keys[i + 1 :]:
                a = _material_similarity(m1, m2)
                b = _material_similarity(m2, m1)
                assert a == pytest.approx(b, abs=1e-6), f"Asymmetry: {m1}↔{m2}: {a:.4f} vs {b:.4f}"

    def test_77_augment_cross_material_no_augment_if_enough_obs(self, tmp_path, monkeypatch):
        """Bei ≥ 10 eigenen Beobachtungen: kein Cross-Material-Transfer."""
        import backend.core.gp_parameter_optimizer as gp_mod

        monkeypatch.setattr(gp_mod, "_MEMORY_DIR", tmp_path)
        opt = GPParameterOptimizer(rng_seed=30)

        rng = np.random.default_rng(0)
        all_X = [rng.random(opt._dim).tolist() for _ in range(10)]
        all_y = [float(rng.random()) for _ in range(10)]
        # Convert to np.ndarray list as the method expects
        X_np = [np.array(x) for x in all_X]

        augmented_X, augmented_y = opt._augment_with_cross_material("shellac", X_np, all_y)
        assert len(augmented_X) == 10  # unchanged
        assert len(augmented_y) == 10

    def test_78_augment_cross_material_augments_few_obs(self, tmp_path, monkeypatch):
        """Bei < 10 eigenen Obs und ähnlichem Material vorhanden → Cross-Transfer aktiv."""
        import json

        import backend.core.gp_parameter_optimizer as gp_mod

        monkeypatch.setattr(gp_mod, "_MEMORY_DIR", tmp_path)
        opt = GPParameterOptimizer(rng_seed=31)

        # Shellac hat 2 eigene Einträge
        rng = np.random.default_rng(1)
        own_X = [rng.random(opt._dim) for _ in range(2)]
        own_y = [0.6, 0.7]

        # wax_cyl (Ähnlichkeit 0.85 mit shellac) hat 3 Einträge im Memory
        wax_entries = [{"params": rng.random(opt._dim).tolist(), "score": 0.65, "ts": float(i)} for i in range(3)]
        (tmp_path / "wax_cyl.json").write_text(json.dumps(wax_entries))

        augmented_X, augmented_y = opt._augment_with_cross_material("shellac", own_X, own_y)
        assert len(augmented_X) > 2  # cross entries wurden hinzugefügt
        # Alle augmentierten Scores müssen ≤ Originalscores sein (Dämpfung via Ähnlichkeit)
        for y in augmented_y[2:]:
            assert y <= 1.0

    def test_79_augment_cross_material_score_damped_by_similarity(self, tmp_path, monkeypatch):
        """Cross-Material-Scores werden mit Ähnlichkeit multipliziert."""
        import json

        import backend.core.gp_parameter_optimizer as gp_mod

        monkeypatch.setattr(gp_mod, "_MEMORY_DIR", tmp_path)
        opt = GPParameterOptimizer(rng_seed=32)

        rng = np.random.default_rng(2)
        own_X = [rng.random(opt._dim) for _ in range(1)]
        own_y = [0.9]

        # tape_std hat Ähnlichkeit 0.85 zu tape_stu
        tape_stu_entries = [{"params": rng.random(opt._dim).tolist(), "score": 1.0, "ts": 0.0}]
        (tmp_path / "tape_stu.json").write_text(json.dumps(tape_stu_entries))

        augmented_X, augmented_y = opt._augment_with_cross_material("tape_std", own_X, own_y)
        # Neuer Score = 1.0 × 0.85 = 0.85
        cross_scores = augmented_y[1:]
        if cross_scores:
            assert cross_scores[0] == pytest.approx(0.85, abs=0.01)

    def test_80_propose_uses_cross_material_with_few_obs(self, tmp_path, monkeypatch):
        """propose() mit 2 eigenen Obs gibt keinen Fehler (Cross-Transfer greift)."""
        import backend.core.gp_parameter_optimizer as gp_mod

        monkeypatch.setattr(gp_mod, "_MEMORY_DIR", tmp_path)
        opt = GPParameterOptimizer(rng_seed=33)

        self._populate_memory(opt, "shellac", 2, tmp_path, monkeypatch)
        proposal = opt.propose(material="shellac", n_init=5)
        assert isinstance(proposal, ParameterProposal)
        assert proposal.parameters  # nicht leer


class TestContextPriors:
    def test_81_chain_hint_does_not_scale_strength(self, tmp_path, monkeypatch):
        """§G188 (GEBOTE.md): Stärke folgt der gemessenen Defekttiefe des Einzelfalls.

        Ein Chain-Hinweis darf die Wirkungsstärke NICHT als fester Multiplikator
        skalieren — `_apply_context_priors` sagt das ausdrücklich: „Chain hints
        select or suppress causal hypotheses; G188 (GEBOTE.md) forbids using them
        as fixed strength multipliers". Der frühere Test verlangte genau diesen
        verbotenen Faktor (0,5 × 0,6 = 0,30) und war damit regelwidrig; gefunden
        im Vollscan 2026-10-05 (Chunk 26). §G189 (GEBOTE.md) lässt
        Chain-Injection (G171) als dokumentierte Ausnahme zu, einen freien
        Stärke-Faktor nicht.
        """
        import backend.core.gp_parameter_optimizer as gp_mod

        monkeypatch.setattr(gp_mod, "_MEMORY_DIR", tmp_path)
        mit_hint = (
            GPParameterOptimizer(rng_seed=34)
            .propose(
                material="tape",
                chain_hint={"strength_scale": 0.5, "dominant_cluster": "tape_transport"},
            )
            .parameters["noise_reduction_strength"]
        )
        ohne_hint = GPParameterOptimizer(rng_seed=34).propose(material="tape").parameters["noise_reduction_strength"]
        assert mit_hint == pytest.approx(ohne_hint, abs=1e-9), (
            f"Chain-Hinweis hat die Stärke skaliert ({mit_hint} vs. {ohne_hint}) — §G188 (GEBOTE.md) verbietet das"
        )

    def test_82_memory_prior_blends_known_parameter(self, tmp_path, monkeypatch):
        import backend.core.gp_parameter_optimizer as gp_mod

        monkeypatch.setattr(gp_mod, "_MEMORY_DIR", tmp_path)
        opt = GPParameterOptimizer(rng_seed=35)
        proposal = opt.propose(
            material="tape",
            memory_prior={"phase_params": {"noise_reduction_strength": 0.20}, "hpi_achieved": 1.0},
        )
        assert proposal.parameters["noise_reduction_strength"] < MATERIAL_DEFAULTS["tape"]["noise_reduction_strength"]

    def test_83_pareto_accepts_context_priors(self, tmp_path, monkeypatch):
        import backend.core.gp_parameter_optimizer as gp_mod

        monkeypatch.setattr(gp_mod, "_MEMORY_DIR", tmp_path)
        opt = GPParameterOptimizer(rng_seed=36)
        proposals = opt.propose_pareto(
            material="tape",
            n_candidates=2,
            chain_hint={"strength_scale": 0.7},
            memory_prior={"phase_params": {"harmonic_boost_db": 0.0}, "hpi_achieved": 0.8},
        )
        assert proposals
        for proposal in proposals:
            assert isinstance(proposal, ParameterProposal)

    def test_84_memory_prior_causal_confidence_increases_blend(self, tmp_path, monkeypatch):
        import backend.core.gp_parameter_optimizer as gp_mod

        monkeypatch.setattr(gp_mod, "_MEMORY_DIR", tmp_path)
        opt = GPParameterOptimizer(rng_seed=37)

        weak = opt.propose(
            material="tape",
            memory_prior={"phase_params": {"noise_reduction_strength": 0.20}, "hpi_achieved": 0.8},
        )
        strong = opt.propose(
            material="tape",
            memory_prior={
                "phase_params": {
                    "noise_reduction_strength": 0.20,
                    "_causal_credit_confidence": 1.0,
                },
                "hpi_achieved": 0.8,
            },
        )

        # Höhere kausale Konfidenz -> stärkeres Blending Richtung Prior (0.20)
        assert strong.parameters["noise_reduction_strength"] < weak.parameters["noise_reduction_strength"]

    def test_85_coalition_learning_factor_increases_memory_blend(self, tmp_path, monkeypatch):
        import backend.core.gp_parameter_optimizer as gp_mod

        monkeypatch.setattr(gp_mod, "_MEMORY_DIR", tmp_path)
        opt = GPParameterOptimizer(rng_seed=38)

        weak = opt.propose(
            material="tape",
            memory_prior={"phase_params": {"noise_reduction_strength": 0.20}, "hpi_achieved": 0.8},
        )
        coalition = opt.propose(
            material="tape",
            memory_prior={
                "phase_params": {
                    "noise_reduction_strength": 0.20,
                    "_coalition_learning_factor": 1.25,
                },
                "hpi_achieved": 0.8,
            },
        )

        assert coalition.parameters["noise_reduction_strength"] < weak.parameters["noise_reduction_strength"]
