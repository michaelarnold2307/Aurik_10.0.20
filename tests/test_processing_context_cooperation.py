"""
tests/test_processing_context_cooperation.py
Test Suite for Module Cooperation & Over-Processing Prevention
===============================================================

Tests:
- Module type tracking
- Over-processing detection
- Recommended strength calculation
- Material-aware adjustments
- Confidence-based dosing

Author: AURIK Team
Date: 10. Februar 2026
"""

import pytest

from backend.core.processing_context import ModuleState, ModuleType, ProcessingContext


@pytest.mark.unit
class TestModuleCooperation:
    """Test module cooperation features."""

    def setup_method(self):
        """Setup test context."""
        self.context = ProcessingContext(
            session_id="test_cooperation_001", sample_rate=48000, processing_mode="restoration"
        )

    def test_module_type_tracking(self):
        """Test that module types are tracked correctly."""
        # Register and complete a deesser module
        self.context.register_processing(
            module_name="DeEsser_v2", module_type=ModuleType.DEESSER, strength=1.0, confidence=0.85
        )
        self.context.set_module_state("DeEsser_v2", ModuleState.COMPLETED)

        # Check if module type was processed
        assert self.context.has_module_type_processed(ModuleType.DEESSER)
        assert not self.context.has_module_type_processed(ModuleType.COMPRESSOR)

    def test_processing_history(self):
        """Test processing history tracking."""
        # Apply deesser twice
        self.context.register_processing("DeEsser_v2", ModuleType.DEESSER, 1.0, 0.85)
        self.context.set_module_state("DeEsser_v2", ModuleState.COMPLETED)

        self.context.register_processing("DeEsser_context_aware", ModuleType.DEESSER, 0.8, 0.90)
        self.context.set_module_state("DeEsser_context_aware", ModuleState.COMPLETED)

        # Check history
        history = self.context.get_processing_history(ModuleType.DEESSER)
        assert len(history) == 2
        assert history[0].name == "DeEsser_v2"
        assert history[1].name == "DeEsser_context_aware"

    def test_accumulated_strength(self):
        """Test accumulated strength calculation."""
        # Apply deesser twice
        self.context.register_processing("DeEsser_v2", ModuleType.DEESSER, 1.0, 0.85)
        self.context.set_module_state("DeEsser_v2", ModuleState.COMPLETED)

        self.context.register_processing("DeEsser_context_aware", ModuleType.DEESSER, 0.5, 0.90)
        self.context.set_module_state("DeEsser_context_aware", ModuleState.COMPLETED)

        # Check accumulated strength
        total = self.context.get_accumulated_strength(ModuleType.DEESSER)
        assert total == 1.5  # 1.0 + 0.5


class TestOverProcessingPrevention:
    """Test over-processing prevention."""

    def setup_method(self):
        """Setup test context."""
        self.context = ProcessingContext(session_id="test_overprocessing_001", sample_rate=48000)

    def test_deesser_threshold(self):
        """Test deesser over-processing threshold (2.0)."""
        # Apply deesser at 1.5
        self.context.register_processing("DeEsser_v2", ModuleType.DEESSER, 1.5, 0.85)
        self.context.set_module_state("DeEsser_v2", ModuleState.COMPLETED)

        # Check if adding 0.3 is safe (total 1.8 < 2.0)
        risk_check = self.context.check_over_processing_risk(ModuleType.DEESSER, 0.3)
        assert risk_check["safe"] is True
        assert risk_check["current"] == 1.5
        assert risk_check["proposed_total"] == 1.8
        assert risk_check["threshold"] == 2.0

        # Check if adding 0.7 is unsafe (total 2.2 > 2.0)
        risk_check = self.context.check_over_processing_risk(ModuleType.DEESSER, 0.7)
        assert risk_check["safe"] is False
        assert risk_check["proposed_total"] == 2.2
        assert "OVER-PROCESSING RISK" in risk_check["recommendation"]

    def test_compressor_threshold(self):
        """Test compressor over-processing threshold (6.0)."""
        # Apply compression at 4.0 ratio
        self.context.register_processing("Compressor_v3", ModuleType.COMPRESSOR, 4.0, 0.88)
        self.context.set_module_state("Compressor_v3", ModuleState.COMPLETED)

        # Check if adding 1.5 is safe (total 5.5 < 6.0)
        risk_check = self.context.check_over_processing_risk(ModuleType.COMPRESSOR, 1.5)
        assert risk_check["safe"] is True

        # Check if adding 3.0 is unsafe (total 7.0 > 6.0)
        risk_check = self.context.check_over_processing_risk(ModuleType.COMPRESSOR, 3.0)
        assert risk_check["safe"] is False

    def test_noise_reduction_threshold(self):
        """Test noise reduction threshold (24 dB)."""
        # Apply 18 dB noise reduction
        self.context.register_processing("SpectralDenoise", ModuleType.NOISE_REDUCTION, 18.0, 0.92)
        self.context.set_module_state("SpectralDenoise", ModuleState.COMPLETED)

        # Check if adding 4 dB is safe (total 22 < 24)
        risk_check = self.context.check_over_processing_risk(ModuleType.NOISE_REDUCTION, 4.0)
        assert risk_check["safe"] is True

        # Check if adding 8 dB is unsafe (total 26 > 24)
        risk_check = self.context.check_over_processing_risk(ModuleType.NOISE_REDUCTION, 8.0)
        assert risk_check["safe"] is False


class TestRecommendedStrength:
    """Test recommended strength calculations."""

    def setup_method(self):
        """Setup test context."""
        self.context = ProcessingContext(session_id="test_recommended_001", sample_rate=48000)

    def test_confidence_based_reduction(self):
        """Test that low confidence reduces strength."""
        # Low confidence (0.6) should reduce strength by 30%
        result = self.context.get_recommended_strength(
            module_type=ModuleType.DEESSER, base_strength=1.0, confidence=0.6
        )

        # Expected: 1.0 * 0.7 = 0.7
        assert result["recommended_strength"] == pytest.approx(0.7, rel=0.01)
        assert "Low confidence" in result["adjustments"][0]

    def test_confidence_based_boost(self):
        """Test that high confidence boosts strength."""
        # High confidence (0.95) should boost strength by 10%
        result = self.context.get_recommended_strength(
            module_type=ModuleType.DEESSER, base_strength=1.0, confidence=0.95
        )

        # Expected: 1.0 * 1.1 = 1.1
        assert result["recommended_strength"] == pytest.approx(1.1, rel=0.01)
        assert "High confidence" in result["adjustments"][0]

    def test_vinyl_material_compression_reduction(self):
        """Test that vinyl material reduces compression."""
        result = self.context.get_recommended_strength(
            module_type=ModuleType.COMPRESSOR, base_strength=4.0, confidence=0.85, material_type="vinyl"
        )

        # Expected: 4.0 * 0.8 = 3.2 (20% reduction for vinyl)
        assert result["recommended_strength"] == pytest.approx(3.2, rel=0.01)
        assert any("Vinyl" in adj for adj in result["adjustments"])

    def test_vinyl_material_denoise_boost(self):
        """Test that vinyl material increases denoise."""
        result = self.context.get_recommended_strength(
            module_type=ModuleType.NOISE_REDUCTION, base_strength=12.0, confidence=0.85, material_type="vinyl"
        )

        # Expected: 12.0 * 1.1 = 13.2 (10% boost for vinyl surface noise)
        assert result["recommended_strength"] == pytest.approx(13.2, rel=0.01)
        assert any("surface noise" in adj for adj in result["adjustments"])

    def test_digital_material_denoise_boost(self):
        """Test that digital material increases denoise."""
        result = self.context.get_recommended_strength(
            module_type=ModuleType.NOISE_REDUCTION, base_strength=10.0, confidence=0.85, material_type="digital"
        )

        # Expected: 10.0 * 1.2 = 12.0 (20% boost for digital)
        assert result["recommended_strength"] == pytest.approx(12.0, rel=0.01)
        assert any("Digital" in adj for adj in result["adjustments"])

    def test_over_processing_prevention_in_recommendation(self):
        """Test that over-processing prevention limits recommended strength."""
        # Pre-existing deesser at 1.8
        self.context.register_processing("DeEsser_v2", ModuleType.DEESSER, 1.8, 0.85)
        self.context.set_module_state("DeEsser_v2", ModuleState.COMPLETED)

        # Try to apply another 1.0 (would exceed 2.0 threshold)
        result = self.context.get_recommended_strength(
            module_type=ModuleType.DEESSER, base_strength=1.0, confidence=0.85
        )

        # Should be capped at 0.2 (threshold 2.0 - current 1.8)
        assert result["recommended_strength"] == pytest.approx(0.2, rel=0.01)
        assert any("Over-processing prevention" in adj for adj in result["adjustments"])
        assert result["over_processing_check"]["safe"] is False

    def test_combined_adjustments(self):
        """Test multiple adjustments applied together."""
        result = self.context.get_recommended_strength(
            module_type=ModuleType.NOISE_REDUCTION,
            base_strength=15.0,
            confidence=0.65,  # Low confidence: -30%
            material_type="digital",  # Digital: +20%
        )

        # Expected: 15.0 * 0.7 (low conf) * 1.2 (digital) = 12.6
        assert result["recommended_strength"] == pytest.approx(12.6, rel=0.05)
        assert len(result["adjustments"]) >= 2


class TestIntegrationScenarios:
    """Integration tests for realistic processing scenarios."""

    def setup_method(self):
        """Setup test context."""
        self.context = ProcessingContext(
            session_id="test_integration_001", sample_rate=48000, processing_mode="restoration"
        )

    def test_full_restoration_chain(self):
        """Test full restoration chain with over-processing prevention."""
        # 1. Declicker
        recommended = self.context.get_recommended_strength(ModuleType.DECLICKER, 0.8, 0.90)
        self.context.register_processing("Declicker", ModuleType.DECLICKER, recommended["recommended_strength"], 0.90)
        self.context.set_module_state("Declicker", ModuleState.COMPLETED)

        # 2. Denoise
        recommended = self.context.get_recommended_strength(ModuleType.NOISE_REDUCTION, 12.0, 0.85, "vinyl")
        self.context.register_processing(
            "SpectralDenoise", ModuleType.NOISE_REDUCTION, recommended["recommended_strength"], 0.85
        )
        self.context.set_module_state("SpectralDenoise", ModuleState.COMPLETED)

        # 3. Deesser
        recommended = self.context.get_recommended_strength(ModuleType.DEESSER, 1.2, 0.88)
        self.context.register_processing("DeEsser_v2", ModuleType.DEESSER, recommended["recommended_strength"], 0.88)
        self.context.set_module_state("DeEsser_v2", ModuleState.COMPLETED)

        # 4. Compressor (should be reduced for vinyl)
        recommended = self.context.get_recommended_strength(ModuleType.COMPRESSOR, 4.0, 0.87, "vinyl")
        assert recommended["recommended_strength"] < 4.0  # Vinyl reduction

        # Verify all modules tracked
        stats = self.context.get_statistics()
        assert stats["completed_modules"] == 3
        assert stats["average_confidence"] >= 0.85


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
