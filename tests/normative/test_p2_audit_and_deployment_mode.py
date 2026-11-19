from __future__ import annotations

"""P2-1 + P2-2 Normative CI-Gates.

P2-1 (AMRB Auditierbarkeit):
    - BenchmarkReport enthält run_seed, aurik_version, report_sha256
    - Gleicher run_seed → identischer SHA-256 (Reproduzierbarkeit)
    - scenario_type ist immer "synthetic" bei AMRB v1.0

P2-2 (Produkt-/Forschungsmodus):
    - RestorationConfig() Default-Modus ist DeploymentMode.PRODUCT
    - DeploymentMode hat genau PRODUCT und RESEARCH
    - Normative Invariante: kein Produktionsaufruf erzwingt RESEARCH-Modus
"""


import pytest

pytestmark = pytest.mark.normative


# ===========================================================================
# P2-1: AMRB Auditierbarkeit — Struktur-Tests (kein ML, kein Restore-Aufruf)
# ===========================================================================


class TestAMRBAuditability:
    """BenchmarkReport-Felder für Auditierbarkeit und Reproduzierbarkeit (P2-1)."""

    @staticmethod
    def _minimal_report():
        """Erzeugt einen minimalen BenchmarkReport mit Pass-Through-Funktion."""

        from benchmarks.musical_restoration_benchmark import BenchmarkConfig, run_benchmark

        config = BenchmarkConfig(
            restoration_fn=lambda audio, sr: audio,  # Pass-Through
            system_name="P2-1 CI Test",
            n_items_per_scenario=1,
            duration_s=1.0,
            scenarios=["AMRB-01-TAPE"],  # Nur 1 Szenario für Geschwindigkeit
            verbose=False,
            run_seed=99,
            aurik_version="9.10.57-test",
            enable_mushra_proxy=False,
            enable_musical_goals=False,
            enable_formal_session=False,
            enforce_min_fragment_guard=False,
        )
        return run_benchmark(config)

    def test_p2_1_report_has_run_seed(self):
        """BenchmarkReport muss run_seed-Feld enthalten."""
        report = self._minimal_report()
        assert hasattr(report, "run_seed"), "BenchmarkReport fehlt: run_seed"
        assert report.run_seed == 99

    def test_p2_1_report_has_aurik_version(self):
        """BenchmarkReport muss aurik_version-Feld enthalten."""
        report = self._minimal_report()
        assert hasattr(report, "aurik_version"), "BenchmarkReport fehlt: aurik_version"
        assert report.aurik_version == "9.10.57-test"

    def test_p2_1_report_has_sha256(self):
        """BenchmarkReport muss report_sha256 (64-stellig) enthalten."""
        report = self._minimal_report()
        assert hasattr(report, "report_sha256"), "BenchmarkReport fehlt: report_sha256"
        assert len(report.report_sha256) == 64, (
            f"SHA-256 soll 64 Zeichen haben, hat {len(report.report_sha256)}: {report.report_sha256!r}"
        )

    def test_p2_1_sha256_is_deterministic_for_same_seed(self):
        """Gleicher run_seed → identischer SHA-256 (Reproduzierbarkeit)."""

        from benchmarks.musical_restoration_benchmark import BenchmarkConfig, run_benchmark

        def make_report():
            config = BenchmarkConfig(
                restoration_fn=lambda audio, sr: audio,
                system_name="Repro-Test",
                n_items_per_scenario=1,
                duration_s=0.5,
                scenarios=["AMRB-01-TAPE"],
                verbose=False,
                run_seed=7,
                aurik_version="9.10.57-repro",
                enable_mushra_proxy=False,
                enable_musical_goals=False,
                enable_formal_session=False,
                enforce_min_fragment_guard=False,
            )
            return run_benchmark(config)

        r1 = make_report()
        r2 = make_report()
        assert r1.report_sha256 == r2.report_sha256, (
            f"SHA-256 ist nicht deterministisch:\n  Lauf 1: {r1.report_sha256}\n  Lauf 2: {r2.report_sha256}"
        )

    @pytest.mark.timeout(120)
    def test_p2_1_sha256_differs_for_different_seed(self):
        """Verschiedener run_seed → unterschiedliche SHA-256 (Seed ist eingebaut)."""

        from benchmarks.musical_restoration_benchmark import BenchmarkConfig, run_benchmark

        def make_report(seed: int):
            config = BenchmarkConfig(
                restoration_fn=lambda audio, sr: audio,
                system_name="Seed-Diff-Test",
                n_items_per_scenario=1,
                duration_s=0.5,
                scenarios=["AMRB-01-TAPE"],
                verbose=False,
                run_seed=seed,
                aurik_version="9.10.57-seed",
                enable_mushra_proxy=False,
                enable_musical_goals=False,
                enable_formal_session=False,
                enforce_min_fragment_guard=False,
            )
            return run_benchmark(config)

        r_a = make_report(seed=1)
        r_b = make_report(seed=2)
        assert r_a.report_sha256 != r_b.report_sha256, (
            "SHA-256 ist gleich für unterschiedliche Seeds — Seed fließt nicht in Hash ein."
        )

    def test_p2_1_scenario_type_is_synthetic(self):
        """Alle AMRB v1.0 Szenarien müssen scenario_type='synthetic' haben."""
        report = self._minimal_report()
        for sid, result in report.scenario_results.items():
            assert result.scenario_type == "synthetic", (
                f"Szenario '{sid}' hat scenario_type='{result.scenario_type}', erwartet 'synthetic'"
            )

    def test_p2_1_as_dict_includes_audit_fields(self):
        """as_dict() muss run_seed und aurik_version als Top-Level-Keys enthalten."""
        report = self._minimal_report()
        d = report.as_dict()
        assert "run_seed" in d, "as_dict() fehlt: run_seed"
        assert "aurik_version" in d, "as_dict() fehlt: aurik_version"
        assert d["run_seed"] == 99
        assert d["aurik_version"] == "9.10.57-test"

    def test_p2_1_scenario_dict_includes_scenario_type(self):
        """as_dict()['scenarios'][sid] muss scenario_type enthalten."""
        report = self._minimal_report()
        d = report.as_dict()
        for sid, scenario_data in d["scenarios"].items():
            assert "scenario_type" in scenario_data, f"Szenario '{sid}' in as_dict() fehlt: scenario_type"

    def test_p2_1_items_include_fallback_and_exception_flags(self):
        """AMRB-Items müssen technische Audit-Flags für Hörtest-Gates enthalten."""
        report = self._minimal_report()
        d = report.as_dict()
        for sid, scenario_data in d["scenarios"].items():
            for idx, item in enumerate(scenario_data.get("items", [])):
                assert "mushra_fallback_used" in item, f"Szenario '{sid}' Item {idx} fehlt: mushra_fallback_used"
                assert "restoration_exception" in item, f"Szenario '{sid}' Item {idx} fehlt: restoration_exception"
                assert isinstance(item["mushra_fallback_used"], bool), (
                    f"Szenario '{sid}' Item {idx}: mushra_fallback_used muss bool sein"
                )
                assert isinstance(item["restoration_exception"], bool), (
                    f"Szenario '{sid}' Item {idx}: restoration_exception muss bool sein"
                )

    def test_p2_1_as_dict_includes_external_validation_block(self):
        """AMRB-Bericht enthält External-Validation-Readiness-Felder."""
        report = self._minimal_report()
        d = report.as_dict()
        assert "external_validation" in d
        ext = d["external_validation"]
        assert "dataset" in ext
        assert "n_external_scenarios" in ext
        assert "ready" in ext
        assert "leadership_claim_ready" in ext
        assert "notes" in ext
        assert ext["n_external_scenarios"] == 0


# ===========================================================================
# P2-2: Produkt-/Forschungsmodus — Struktur-Tests (kein ML)
# ===========================================================================


class TestDeploymentMode:
    """DeploymentMode und RestorationConfig-Default (P2-2)."""

    def test_p2_2_deployment_mode_enum_exists(self):
        """DeploymentMode muss in performance_guard exportiert werden."""
        from backend.core.performance_guard import DeploymentMode

        assert DeploymentMode is not None

    def test_p2_2_deployment_mode_has_product_and_research(self):
        """DeploymentMode muss genau PRODUCT und RESEARCH definieren."""
        from backend.core.performance_guard import DeploymentMode

        values = {m.value for m in DeploymentMode}
        assert "product" in values, "DeploymentMode fehlt: PRODUCT"
        assert "research" in values, "DeploymentMode fehlt: RESEARCH"

    def test_p2_2_restoration_config_default_is_product(self):
        """RestorationConfig() Default muss DeploymentMode.PRODUCT sein (kein RESEARCH-Default)."""
        from backend.core.performance_guard import DeploymentMode
        from backend.core.unified_restorer_v3 import RestorationConfig

        cfg = RestorationConfig()
        assert cfg.deployment_mode == DeploymentMode.PRODUCT, (
            f"RestorationConfig() Default deployment_mode ist '{cfg.deployment_mode}', "
            f"erwartet DeploymentMode.PRODUCT — RESEARCH darf NIE der Produktions-Default sein."
        )

    def test_p2_2_research_mode_can_be_set_explicitly(self):
        """RESEARCH-Modus muss explizit opt-in möglich sein."""
        from backend.core.performance_guard import DeploymentMode
        from backend.core.unified_restorer_v3 import RestorationConfig

        cfg = RestorationConfig(deployment_mode=DeploymentMode.RESEARCH)
        assert cfg.deployment_mode == DeploymentMode.RESEARCH

    def test_p2_2_deployment_mode_importable_from_uv3(self):
        """DeploymentMode muss auch über UV3-Importpfad erreichbar sein."""
        from backend.core.unified_restorer_v3 import DeploymentMode

        assert DeploymentMode.PRODUCT.value == "product"
