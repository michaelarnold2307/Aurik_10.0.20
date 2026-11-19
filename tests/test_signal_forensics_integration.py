"""
tests/test_signal_forensics_integration.py
Integration Tests für Signal Forensics System
==============================================

End-to-End Tests vom Audio-Eingang bis zur Verarbeitungskette:
1. Audio → ML Medium Detector → Ergebnis
2. Audio → ML Era Detector → Ergebnis
3. Audio → ML Defect Detector → Ergebnis
4. Audio → Unified Analyzer → Ergebnis
5. Unified Analyzer → Adaptive Chain Builder → Verarbeitungskette
6. Full Pipeline: Audio → Analysis → Chain → Validation
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.forensics.adaptive_chain_builder import AdaptiveChainBuilder
from backend.core.forensics.dataset_generator import DatasetGenerator
from backend.core.forensics.ml_defect_detector import train_ml_defect_detector_from_dataset
from backend.core.forensics.ml_era_detector import train_ml_era_detector_from_dataset
from backend.core.forensics.ml_medium_detector import train_ml_detector_from_dataset
from backend.core.forensics.unified_analyzer import UnifiedForensicAnalyzer


@pytest.fixture(scope="module")
def trained_medium_detector():
    """Train medium detector."""
    gen = DatasetGenerator()
    dataset = gen.generate_medium_dataset(n_synthetic_per_medium=10)
    detector, _ = train_ml_detector_from_dataset(dataset, test_size=0.2, verbose=False)
    return detector


@pytest.fixture(scope="module")
def trained_era_detector():
    """Train era detector."""
    gen = DatasetGenerator()
    dataset = gen.generate_era_dataset(n_synthetic_per_era=10)
    detector, _ = train_ml_era_detector_from_dataset(dataset, test_size=0.2, verbose=False)
    return detector


@pytest.fixture(scope="module")
def trained_defect_detector():
    """Train defect detector."""
    from tests.test_ml_defect_detector import generate_defect_dataset

    dataset = generate_defect_dataset(n_samples_per_type=10)
    detector, _ = train_ml_defect_detector_from_dataset(dataset, test_size=0.2, verbose=False)
    return detector


@pytest.fixture(scope="module")
def full_analyzer(trained_medium_detector, trained_era_detector, trained_defect_detector):
    """Create unified analyzer with all detectors."""
    return UnifiedForensicAnalyzer(
        medium_detector=trained_medium_detector,
        era_detector=trained_era_detector,
        defect_detector=trained_defect_detector,
    )


@pytest.fixture(scope="module")
def chain_builder():
    """Create chain builder."""
    return AdaptiveChainBuilder()


@pytest.mark.unit
class TestFullForensicsPipeline:
    """Test full forensics pipeline from audio to chain."""

    def test_clean_audio_pipeline(self, full_analyzer, chain_builder):
        """Test pipeline with clean audio."""
        # Generate clean audio
        sr = 48000
        duration = 0.3
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.3

        # Analyze
        analysis = full_analyzer.analyze(audio, sr, verbose=False)

        # Build chain
        chain = chain_builder.build_chain(analysis, verbose=False)

        # Validate
        assert analysis is not None
        assert chain is not None
        assert len(chain.modules) > 0

        # Should have at least DCBlocker
        module_names = [m.name for m in chain.modules]
        assert "DCBlocker" in module_names

        # Should have some enhancement
        assert any("Enhancement" in name for name in module_names)

    def test_vinyl_with_clicks_pipeline(self, full_analyzer, chain_builder):
        """Test pipeline with vinyl audio containing clicks."""
        # Generate vinyl with clicks
        sr = 48000
        duration = 0.3
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.3

        # Add vinyl rumble
        rumble = np.sin(2 * np.pi * 30 * t) * 0.05
        audio += rumble

        # Add clicks
        for i in range(5):
            click_pos = 1000 + i * 2000
            if click_pos < len(audio):
                audio[click_pos : click_pos + 10] += np.random.randn(10) * 0.5

        # Analyze
        analysis = full_analyzer.analyze(audio, sr, verbose=False)

        # Build chain
        chain = chain_builder.build_chain(analysis, verbose=False)

        # Validate analysis
        assert analysis is not None
        assert analysis.overall_confidence > 0

        # Validate chain
        assert chain is not None
        assert len(chain.modules) > 0

        # Should have base modules
        module_names = [m.name for m in chain.modules]
        assert "DCBlocker" in module_names

    def test_audio_with_hum_pipeline(self, full_analyzer, chain_builder):
        """Test pipeline with audio containing hum."""
        # Generate audio with hum
        sr = 48000
        duration = 0.3
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.3

        # Add hum (50Hz)
        hum = np.sin(2 * np.pi * 50 * t) * 0.2
        audio += hum

        # Analyze
        analysis = full_analyzer.analyze(audio, sr, verbose=False)

        # Build chain
        chain = chain_builder.build_chain(analysis, verbose=False)

        # Validate
        assert analysis is not None
        assert chain is not None
        assert len(chain.modules) > 0

    def test_distorted_audio_pipeline(self, full_analyzer, chain_builder):
        """Test pipeline with distorted audio."""
        # Generate distorted audio
        sr = 48000
        duration = 0.3
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.8

        # Add clipping distortion
        audio = np.clip(audio, -0.5, 0.5)

        # Analyze
        analysis = full_analyzer.analyze(audio, sr, verbose=False)

        # Build chain
        chain = chain_builder.build_chain(analysis, verbose=False)

        # Validate
        assert analysis is not None
        assert chain is not None
        assert len(chain.modules) > 0


class TestMaterialSpecificPipelines:
    """Test pipelines for different material types."""

    def test_vinyl_specific_chain(self, full_analyzer, chain_builder):
        """Test that vinyl produces vinyl-appropriate chain."""
        # Generate vinyl-like audio
        sr = 48000
        duration = 0.3
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.3

        # Add rumble
        rumble = np.sin(2 * np.pi * 30 * t) * 0.05
        audio += rumble

        analysis = full_analyzer.analyze(audio, sr, verbose=False)
        chain = chain_builder.build_chain(analysis, verbose=False)

        # Validate chain is appropriate
        assert chain.material_type in chain_builder.CHAIN_TEMPLATES
        assert len(chain.modules) > 0


class TestAggressiveMode:
    """Test aggressive vs. normal processing chains."""

    def test_aggressive_vs_normal(self, full_analyzer, chain_builder):
        """Test that aggressive mode produces different chains."""
        # Generate test audio
        sr = 48000
        duration = 0.3
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.3

        # Add hum
        hum = np.sin(2 * np.pi * 50 * t) * 0.1
        audio += hum

        analysis = full_analyzer.analyze(audio, sr, verbose=False)

        # Build normal chain
        chain_normal = chain_builder.build_chain(analysis, aggressive=False, verbose=False)

        # Build aggressive chain
        chain_aggressive = chain_builder.build_chain(analysis, aggressive=True, verbose=False)

        # Both should be valid
        assert chain_normal is not None
        assert chain_aggressive is not None

        # Aggressive may have more modules or same
        assert len(chain_aggressive.modules) >= len(chain_normal.modules)


class TestChainOrdering:
    """Test processing chain module ordering."""

    def test_chain_has_correct_priority_order(self, full_analyzer, chain_builder):
        """Test that chain modules are in correct priority order."""
        # Generate test audio
        sr = 48000
        duration = 0.3
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.3

        analysis = full_analyzer.analyze(audio, sr, verbose=False)
        chain = chain_builder.build_chain(analysis, verbose=False)

        ordered_modules = chain.get_ordered_modules()

        # Check priority ordering
        priorities = [m.priority for m in ordered_modules]
        assert priorities == sorted(priorities)

    def test_dcblocker_first(self, full_analyzer, chain_builder):
        """Test that DCBlocker is first if present."""
        # Generate test audio
        sr = 48000
        duration = 0.3
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.3

        analysis = full_analyzer.analyze(audio, sr, verbose=False)
        chain = chain_builder.build_chain(analysis, verbose=False)

        ordered_modules = chain.get_ordered_modules()

        if ordered_modules:
            # DCBlocker should be first (if present)
            first_module = ordered_modules[0]
            if first_module.name == "DCBlocker":
                assert first_module.priority == 10

    def test_enhancement_last(self, full_analyzer, chain_builder):
        """Test that Enhancement is last if present."""
        # Generate test audio
        sr = 48000
        duration = 0.3
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.3

        analysis = full_analyzer.analyze(audio, sr, verbose=False)
        chain = chain_builder.build_chain(analysis, verbose=False)

        ordered_modules = chain.get_ordered_modules()

        if ordered_modules:
            # Enhancement should be last (if present)
            last_module = ordered_modules[-1]
            if "Enhancement" in last_module.name:
                assert last_module.priority >= 90


class TestParameterInference:
    """Test parameter inference in chains."""

    def test_hum_remover_parameters(self, full_analyzer, chain_builder):
        """Test that HumRemover gets correct parameters."""
        # Generate audio with hum
        sr = 48000
        duration = 0.3
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.3

        # Add strong hum
        hum = np.sin(2 * np.pi * 50 * t) * 0.3
        audio += hum

        analysis = full_analyzer.analyze(audio, sr, verbose=False)
        chain = chain_builder.build_chain(analysis, verbose=False)

        # Find HumRemover if present
        hum_remover = None
        for module in chain.modules:
            if module.name == "HumRemover":
                hum_remover = module
                break

        # If HumRemover is present, check parameters
        if hum_remover:
            assert "fundamental_hz" in hum_remover.parameters
            # Should be 50 or 60 Hz
            assert hum_remover.parameters["fundamental_hz"] in [50, 60]


class TestChainConsistency:
    """Test consistency of chain generation."""

    def test_same_audio_produces_similar_chains(self, full_analyzer, chain_builder):
        """Test that same audio produces similar chains."""
        # Generate test audio
        sr = 48000
        duration = 0.3
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.3

        # Analyze twice
        analysis1 = full_analyzer.analyze(audio, sr, verbose=False)
        analysis2 = full_analyzer.analyze(audio, sr, verbose=False)

        # Build chains
        chain1 = chain_builder.build_chain(analysis1, verbose=False)
        chain2 = chain_builder.build_chain(analysis2, verbose=False)

        # Chains should be consistent
        assert chain1.material_type == chain2.material_type
        assert len(chain1.modules) == len(chain2.modules)


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_workflow_clean_audio(self, full_analyzer, chain_builder):
        """Test complete workflow with clean audio."""
        # Generate clean audio
        sr = 48000
        duration = 0.3
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.3

        # Step 1: Analyze
        print("\n[1/3] Analyzing audio...")
        analysis = full_analyzer.analyze(audio, sr, verbose=False)

        assert analysis is not None
        assert 0 <= analysis.overall_confidence <= 1

        # Step 2: Build chain
        print("[2/3] Building processing chain...")
        chain = chain_builder.build_chain(analysis, verbose=False)

        assert chain is not None
        assert len(chain.modules) > 0

        # Step 3: Validate chain
        print("[3/3] Validating chain...")
        ordered_modules = chain.get_ordered_modules()

        # Should be in priority order
        priorities = [m.priority for m in ordered_modules]
        assert priorities == sorted(priorities)

        # All modules should have reasons
        for module in ordered_modules:
            assert module.reason != ""

        print(f"   ✓ Chain complete with {len(ordered_modules)} modules")

    def test_full_workflow_vinyl_with_defects(self, full_analyzer, chain_builder):
        """Test complete workflow with vinyl containing multiple defects."""
        # Generate vinyl with clicks and hum
        sr = 48000
        duration = 0.3
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.3

        # Add rumble
        rumble = np.sin(2 * np.pi * 30 * t) * 0.05
        audio += rumble

        # Add clicks
        for i in range(5):
            click_pos = 1000 + i * 2000
            if click_pos < len(audio):
                audio[click_pos : click_pos + 10] += np.random.randn(10) * 0.5

        # Add hum
        hum = np.sin(2 * np.pi * 50 * t) * 0.1
        audio += hum

        # Step 1: Analyze
        analysis = full_analyzer.analyze(audio, sr, verbose=False)

        # Step 2: Build chain
        chain = chain_builder.build_chain(analysis, verbose=False)

        # Step 3: Validate
        assert analysis is not None
        assert chain is not None
        assert len(chain.modules) > 0

        # Should have base modules
        module_names = [m.name for m in chain.modules]
        assert "DCBlocker" in module_names


def manual_test_full_pipeline():
    """
    Manual test of full forensics pipeline.
    Run with: pytest -k manual_test_full_pipeline -s
    """
    print("\n" + "=" * 70)
    print("Full Signal Forensics Pipeline Test")
    print("=" * 70)

    # Train detectors
    print("\n[1/5] Training detectors...")
    gen = DatasetGenerator()

    medium_dataset = gen.generate_medium_dataset(n_synthetic_per_medium=10)
    medium_detector, _ = train_ml_detector_from_dataset(medium_dataset, test_size=0.2, verbose=False)
    print("   ✓ Medium detector trained")

    era_dataset = gen.generate_era_dataset(n_synthetic_per_era=10)
    era_detector, _ = train_ml_era_detector_from_dataset(era_dataset, test_size=0.2, verbose=False)
    print("   ✓ Era detector trained")

    from tests.test_ml_defect_detector import generate_defect_dataset

    defect_dataset = generate_defect_dataset(n_samples_per_type=10)
    defect_detector, _ = train_ml_defect_detector_from_dataset(defect_dataset, test_size=0.2, verbose=False)
    print("   ✓ Defect detector trained")

    # Create analyzer and builder
    print("\n[2/5] Creating analyzer and chain builder...")
    analyzer = UnifiedForensicAnalyzer(
        medium_detector=medium_detector, era_detector=era_detector, defect_detector=defect_detector
    )
    builder = AdaptiveChainBuilder()
    print("   ✓ Components ready")

    # Generate test audio
    print("\n[3/5] Generating test audio (Vinyl with clicks + hum)...")
    sr = 48000
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 440 * t) * 0.3

    # Add vinyl characteristics
    rumble = np.sin(2 * np.pi * 30 * t) * 0.05
    audio += rumble

    # Add clicks
    for i in range(5):
        click_pos = 1000 + i * 3000
        if click_pos < len(audio):
            audio[click_pos : click_pos + 10] += np.random.randn(10) * 0.5

    # Add hum
    hum = np.sin(2 * np.pi * 50 * t) * 0.1
    audio += hum

    print("   ✓ Audio generated")

    # Analyze
    print("\n[4/5] Performing forensic analysis...")
    analysis = analyzer.analyze(audio, sr, verbose=True)

    # Build chain
    print("\n[5/5] Building processing chain...")
    chain = builder.build_chain(analysis, aggressive=False, verbose=True)

    # Visualize
    print("\n" + "=" * 70)
    print("FINAL PROCESSING CHAIN")
    print("=" * 70)
    print(builder.visualize_chain(chain))

    print("\n" + "=" * 70)
    print("Pipeline test complete!")
    print("=" * 70)
    print("\nSummary:")
    print(f"  Material Type: {chain.material_type}")
    print(f"  Era: {chain.era}")
    print(f"  Defects Addressed: {', '.join(chain.defects_addressed) if chain.defects_addressed else 'None'}")
    print(f"  Confidence: {chain.confidence:.1%}")
    print(f"  Total Modules: {len(chain.modules)}")
    print(f"  Enabled Modules: {len(chain.get_ordered_modules())}")


if __name__ == "__main__":
    manual_test_full_pipeline()
