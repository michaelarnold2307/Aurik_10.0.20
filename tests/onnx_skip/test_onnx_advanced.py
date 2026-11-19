"""
Tests for ONNX Converter, Quantizer, and Fallback Manager

Additional tests for ONNX optimization infrastructure.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

# Skip if torch not available
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except (ImportError, OSError):
    TORCH_AVAILABLE = False
    pytest.skip("torch not available", allow_module_level=True)

from backend.core.onnx.converter import ConversionConfig, ONNXConverter
from backend.core.onnx.fallback import FallbackManager, FallbackReason, ONNXModelWithFallback
from backend.core.onnx.quantizer import ModelQuantizer, QuantizationConfig, QuantizationType


# Simple PyTorch model for testing
class SimpleAudioModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


@pytest.fixture
def simple_torch_model():
    """Create simple PyTorch model."""
    model = SimpleAudioModel()
    model.eval()
    return model


@pytest.fixture
def sample_audio_tensor():
    """Create sample audio tensor."""
    return torch.randn(1, 1, 16000)


@pytest.mark.unit
class TestONNXConverter:
    """Test ONNX converter."""

    def test_conversion_config_defaults(self):
        """Test conversion config has reasonable defaults."""
        config = ConversionConfig()

        assert config.opset_version == 14
        assert config.do_constant_folding is True
        assert config.input_names == ["audio_input"]
        assert config.output_names == ["audio_output"]

    def test_converter_initialization(self):
        """Test converter initializes correctly."""
        converter = ONNXConverter()

        assert converter.config is not None
        assert converter.validation_tolerance == 1e-5
        assert converter.conversion_stats["total_conversions"] == 0

    @patch("torch.onnx.export")
    def test_convert_success(self, mock_onnx_export, simple_torch_model, sample_audio_tensor, tmp_path):
        """Test successful model conversion."""
        output_path = tmp_path / "test_model.onnx"

        # Mock forward pass
        with patch.object(simple_torch_model, "forward") as mock_forward:
            mock_forward.return_value = sample_audio_tensor

            converter = ONNXConverter()

            # Mock validation to avoid ONNX runtime dependency
            with patch.object(converter, "_validate_conversion", return_value=True):
                success = converter.convert(
                    pytorch_model=simple_torch_model,
                    output_path=output_path,
                    sample_input=sample_audio_tensor,
                    validate=True,
                )

        assert success
        assert converter.conversion_stats["successful_conversions"] == 1
        mock_onnx_export.assert_called_once()

    def test_converter_stats(self):
        """Test statistics tracking."""
        converter = ONNXConverter()

        stats = converter.get_stats()
        assert "total_conversions" in stats
        assert "successful_conversions" in stats
        assert "failed_conversions" in stats

        converter.reset_stats()
        assert converter.conversion_stats["total_conversions"] == 0


class TestModelQuantizer:
    """Test model quantizer."""

    def test_quantization_config_defaults(self):
        """Test quantization config defaults."""
        config = QuantizationConfig()

        assert config.quantization_type == QuantizationType.DYNAMIC
        assert config.optimize_model is True
        assert config.max_quality_loss_percent == 1.0

    def test_quantizer_initialization(self):
        """Test quantizer initializes."""
        quantizer = ModelQuantizer()

        assert quantizer.config is not None
        assert quantizer.quantization_stats["total_quantizations"] == 0

    def test_quantizer_missing_model(self, tmp_path):
        """Test quantizer handles missing model."""
        quantizer = ModelQuantizer()

        success = quantizer.quantize(
            model_path=tmp_path / "nonexistent.onnx", output_path=tmp_path / "output.onnx", validate_quality=False
        )

        assert not success

    def test_quantizer_stats(self):
        """Test statistics tracking."""
        quantizer = ModelQuantizer()

        stats = quantizer.get_stats()
        assert "total_quantizations" in stats
        assert "successful_quantizations" in stats
        assert "average_size_reduction" in stats

        quantizer.reset_stats()
        assert quantizer.quantization_stats["total_quantizations"] == 0


class TestFallbackManager:
    """Test fallback manager."""

    def test_fallback_manager_initialization(self):
        """Test fallback manager initializes."""
        manager = FallbackManager()

        assert manager.log_fallbacks is True
        assert len(manager.fallback_history) == 0
        assert len(manager.active_fallbacks) == 0

    def test_record_fallback(self):
        """Test recording fallback event."""
        manager = FallbackManager()

        manager.record_fallback(
            model_name="test_model", reason=FallbackReason.MODEL_NOT_FOUND, error_message="Model file missing"
        )

        assert len(manager.fallback_history) == 1
        assert "test_model" in manager.active_fallbacks
        assert manager.stats.total_fallbacks == 1
        assert manager.stats.active_fallbacks == 1

    def test_record_recovery(self):
        """Test recording recovery from fallback."""
        manager = FallbackManager()

        # Record fallback
        manager.record_fallback(
            model_name="test_model", reason=FallbackReason.INFERENCE_ERROR, error_message="Inference failed"
        )

        assert manager.is_fallback_active("test_model")

        # Record recovery
        manager.record_recovery("test_model")

        assert not manager.is_fallback_active("test_model")
        assert manager.stats.recovered_fallbacks == 1
        assert manager.stats.active_fallbacks == 0

    def test_is_fallback_active(self):
        """Test checking fallback status."""
        manager = FallbackManager()

        assert not manager.is_fallback_active("model1")

        manager.record_fallback(
            model_name="model1", reason=FallbackReason.ONNX_RUNTIME_ERROR, error_message="Runtime error"
        )

        assert manager.is_fallback_active("model1")
        assert not manager.is_fallback_active("model2")

    def test_get_fallback_reason(self):
        """Test getting fallback reason."""
        manager = FallbackManager()

        manager.record_fallback(
            model_name="model1", reason=FallbackReason.SHAPE_MISMATCH, error_message="Shape mismatch"
        )

        reason = manager.get_fallback_reason("model1")
        assert reason == FallbackReason.SHAPE_MISMATCH

        reason = manager.get_fallback_reason("model2")
        assert reason is None

    def test_health_check_success(self):
        """Test successful health check."""
        manager = FallbackManager()

        # Mock successful inference
        def successful_inference(audio):
            return audio * 0.9

        test_audio = np.random.randn(16000).astype(np.float32)

        success = manager.health_check_onnx(
            model_name="test_model", onnx_inference_func=successful_inference, test_input=test_audio
        )

        assert success
        assert not manager.is_fallback_active("test_model")

    def test_health_check_failure(self):
        """Test failed health check."""
        manager = FallbackManager()

        # Mock failing inference
        def failing_inference(audio):
            raise RuntimeError("Inference failed")

        test_audio = np.random.randn(16000).astype(np.float32)

        success = manager.health_check_onnx(
            model_name="test_model", onnx_inference_func=failing_inference, test_input=test_audio
        )

        assert not success
        assert manager.is_fallback_active("test_model")

    def test_get_stats(self):
        """Test getting statistics."""
        manager = FallbackManager()

        manager.record_fallback(model_name="model1", reason=FallbackReason.MODEL_NOT_FOUND, error_message="Not found")

        stats = manager.get_stats()

        assert stats["total_fallbacks"] == 1
        assert stats["active_fallbacks"] == 1
        assert "fallback_by_reason" in stats
        assert "fallback_by_model" in stats

    def test_fallback_history(self):
        """Test fallback history tracking."""
        manager = FallbackManager(max_fallback_history=5)

        # Record multiple fallbacks
        for i in range(10):
            manager.record_fallback(
                model_name=f"model{i}", reason=FallbackReason.INFERENCE_ERROR, error_message=f"Error {i}"
            )

        # Should only keep last 5
        assert len(manager.fallback_history) == 5

        # Get recent history
        recent = manager.get_fallback_history(limit=3)
        assert len(recent) == 3


class TestONNXModelWithFallback:
    """Test ONNX model with automatic fallback."""

    def test_initialization(self):
        """Test model with fallback initializes."""
        mock_onnx = Mock()
        mock_pytorch = Mock()

        model = ONNXModelWithFallback(name="test_model", onnx_model=mock_onnx, pytorch_model=mock_pytorch)

        assert model.name == "test_model"
        assert model.use_onnx is True
        assert model.inference_count == 0

    def test_process_onnx_success(self):
        """Test successful ONNX processing."""
        mock_onnx = Mock()
        mock_pytorch = Mock()

        mock_onnx.process.return_value = np.zeros(16000)

        model = ONNXModelWithFallback(name="test_model", onnx_model=mock_onnx, pytorch_model=mock_pytorch)

        audio = np.random.randn(16000).astype(np.float32)
        model.process(audio)

        # Should use ONNX
        mock_onnx.process.assert_called_once()
        mock_pytorch.process.assert_not_called()

        assert model.onnx_inference_count == 1
        assert model.pytorch_inference_count == 0

    def test_process_onnx_fallback(self):
        """Test fallback to PyTorch when ONNX fails."""
        mock_onnx = Mock()
        mock_pytorch = Mock()

        # ONNX fails
        mock_onnx.process.side_effect = RuntimeError("ONNX error")
        # PyTorch succeeds
        mock_pytorch.process.return_value = np.zeros(16000)

        model = ONNXModelWithFallback(name="test_model", onnx_model=mock_onnx, pytorch_model=mock_pytorch)

        audio = np.random.randn(16000).astype(np.float32)
        model.process(audio)

        # Should have tried ONNX, then used PyTorch
        mock_onnx.process.assert_called_once()
        mock_pytorch.process.assert_called_once()

        assert model.onnx_inference_count == 0  # Failed
        assert model.pytorch_inference_count == 1

        # Fallback should be recorded
        assert model.fallback_manager.is_fallback_active("test_model")

    def test_get_stats(self):
        """Test statistics retrieval."""
        mock_onnx = Mock()
        mock_pytorch = Mock()
        mock_onnx.process.return_value = np.zeros(16000)

        model = ONNXModelWithFallback(name="test_model", onnx_model=mock_onnx, pytorch_model=mock_pytorch)

        # Process audio
        audio = np.random.randn(16000).astype(np.float32)
        model.process(audio)

        stats = model.get_stats()

        assert stats["name"] == "test_model"
        assert stats["total_inferences"] == 1
        assert stats["onnx_inferences"] == 1
        assert stats["onnx_usage_percent"] == 100.0


class TestIntegration:
    """Integration tests for complete ONNX workflow."""

    def test_complete_workflow_simulation(self):
        """Test complete ONNX optimization workflow (simulated)."""
        # This is a high-level integration test that simulates
        # the workflow without actual ONNX/PyTorch models

        # 1. Create fallback manager
        manager = FallbackManager()

        # 2. Simulate ONNX model with fallback
        mock_onnx = Mock()
        mock_pytorch = Mock()
        mock_onnx.process.return_value = np.zeros(16000)
        mock_pytorch.process.return_value = np.zeros(16000)

        model = ONNXModelWithFallback(
            name="denoiser", onnx_model=mock_onnx, pytorch_model=mock_pytorch, fallback_manager=manager
        )

        # 3. Process multiple audio samples
        for i in range(5):
            audio = np.random.randn(16000).astype(np.float32)
            output = model.process(audio)
            assert output is not None

        # 4. Check statistics
        stats = model.get_stats()
        assert stats["total_inferences"] == 5
        assert stats["onnx_inferences"] == 5

        # 5. Verify no fallbacks occurred
        fallback_stats = manager.get_stats()
        assert fallback_stats["total_fallbacks"] == 0
