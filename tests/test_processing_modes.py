"""
Tests for Processing Modes System

Tests:
- ProcessingMode enum
- ProcessingConfig creation and validation
- Predefined mode configurations
- Mode selection by name/enum
- Custom config creation
- Parameter validation
"""

import pytest

from backend.core.processing_modes import (
    PROCESSING_CONFIGS,
    ProcessingConfig,
    ProcessingMode,
    create_custom_config,
    get_config_by_name,
    get_processing_config,
    list_available_modes,
)


@pytest.mark.unit
class TestProcessingMode:
    """Test ProcessingMode enum."""

    def test_all_modes_exist(self):
        """Verify all 5 modes exist."""
        # Prüfe nur gültige Modi
        assert len(ProcessingMode) == 2
        assert ProcessingMode.RESTORATION in ProcessingMode
        assert ProcessingMode.STUDIO_2026 in ProcessingMode

    def test_mode_values(self):
        """Verify mode string values."""
        assert ProcessingMode.RESTORATION.value == "restoration"
        assert ProcessingMode.STUDIO_2026.value == "studio_2026"

    def test_from_string(self):
        """Test converting string to ProcessingMode."""
        assert ProcessingMode.from_string("restoration") == ProcessingMode.RESTORATION
        assert ProcessingMode.from_string("studio_2026") == ProcessingMode.STUDIO_2026

    def test_from_string_invalid(self):
        """Test invalid mode name raises ValueError."""
        with pytest.raises(ValueError, match="Invalid processing mode"):
            ProcessingMode.from_string("invalid_mode")


class TestProcessingConfig:
    """Test ProcessingConfig dataclass."""

    def test_config_creation(self):
        """Test creating a ProcessingConfig."""
        config = ProcessingConfig(denoise_strength=0.40, preserve_breaths=True, compression_ratio=3.0)

        assert config.denoise_strength == 0.40
        assert config.preserve_breaths is True
        assert config.compression_ratio == 3.0

    def test_config_defaults(self):
        """Verify default values."""
        config = ProcessingConfig()

        assert config.denoise_strength == 0.30
        assert config.preserve_breaths is True
        assert config.compression_ratio == 2.0
        assert config.target_lufs is None

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = ProcessingConfig(denoise_strength=0.50)
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["denoise_strength"] == 0.50
        assert "preserve_breaths" in config_dict

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {"denoise_strength": 0.60, "preserve_breaths": False, "compression_ratio": 4.5}
        config = ProcessingConfig.from_dict(config_dict)

        assert config.denoise_strength == 0.60
        assert config.preserve_breaths is False
        assert config.compression_ratio == 4.5

    def test_config_validation_strength(self):
        """Test validation of strength parameters."""
        # Valid
        config = ProcessingConfig(denoise_strength=0.50)
        config.validate()  # Should not raise

        # Invalid: out of range
        config = ProcessingConfig(denoise_strength=1.5)
        with pytest.raises(ValueError, match="denoise_strength"):
            config.validate()

    def test_config_validation_compression(self):
        """Test validation of compression ratio."""
        # Valid
        config = ProcessingConfig(compression_ratio=3.0)
        config.validate()

        # Invalid: too high
        config = ProcessingConfig(compression_ratio=15.0)
        with pytest.raises(ValueError, match="compression_ratio"):
            config.validate()

    def test_config_validation_lufs(self):
        """Test validation of target LUFS."""
        # Valid
        config = ProcessingConfig(target_lufs=-14.0)
        config.validate()

        # Invalid: too high
        config = ProcessingConfig(target_lufs=-3.0)
        with pytest.raises(ValueError, match="target_lufs"):
            config.validate()

        # Valid: None
        config = ProcessingConfig(target_lufs=None)
        config.validate()

    def test_config_validation_freq_boost(self):
        """Test validation of high frequency boost."""
        # Valid
        config = ProcessingConfig(high_freq_boost_db=2.0)
        config.validate()

        # Invalid: too high
        config = ProcessingConfig(high_freq_boost_db=10.0)
        with pytest.raises(ValueError, match="high_freq_boost_db"):
            config.validate()


class TestPredefinedConfigs:
    """Test predefined mode configurations."""

    def test_all_modes_have_config(self):
        """Verify all modes have predefined configs."""
        assert len(PROCESSING_CONFIGS) == 2
        for mode in ProcessingMode:
            assert mode in PROCESSING_CONFIGS

    def test_restoration_mode(self):
        """Verify RESTORATION mode parameters."""
        config = PROCESSING_CONFIGS[ProcessingMode.RESTORATION]

        assert config.mode_name == "restoration"
        assert config.denoise_strength == 0.30
        assert config.preserve_breaths is True
        assert config.compression_ratio == 2.0
        assert config.target_lufs is None  # Keep original
        assert config.high_freq_boost_db == 0.0  # No boost

    def test_studio_2026_mode(self):
        """Verify STUDIO_2026 mode parameters."""
        config = PROCESSING_CONFIGS[ProcessingMode.STUDIO_2026]

        assert config.mode_name == "studio_2026"
        assert config.denoise_strength == 0.50  # Aggressive
        assert config.preserve_breaths is True  # Still preserve!
        assert config.compression_ratio == 4.0  # Competitive
        assert config.target_lufs == -14.0  # Streaming standard
        assert config.high_freq_boost_db == 2.0  # Modern "air"

    def test_forensic_mode(self):
        """FORENSIC mode wurde aus dem aktuellen System entfernt (nur 2 Modi: restoration + studio_2026)."""
        assert not hasattr(ProcessingMode, "FORENSIC")
        with pytest.raises(ValueError):
            ProcessingMode.from_string("forensic")

    def test_all_configs_valid(self):
        """Verify all predefined configs pass validation."""
        for mode, config in PROCESSING_CONFIGS.items():
            config.validate()  # Should not raise


class TestModeSelection:
    """Test mode selection functions."""

    def test_get_processing_config(self):
        """Test getting config by ProcessingMode enum."""
        config = get_processing_config(ProcessingMode.RESTORATION)

        assert config.mode_name == "restoration"
        assert isinstance(config, ProcessingConfig)

    def test_get_config_by_name(self):
        """Test getting config by mode name string."""
        config = get_config_by_name("studio_2026")

        assert config.mode_name == "studio_2026"
        assert config.compression_ratio == 4.0

    def test_get_config_by_name_invalid(self):
        """Test invalid mode name raises error."""
        with pytest.raises(ValueError):
            get_config_by_name("invalid_mode")

    def test_list_available_modes(self):
        """Test listing all available modes."""
        modes = list_available_modes()

        assert len(modes) == 2
        assert "restoration" in modes
        assert "studio_2026" in modes
        assert isinstance(modes["restoration"], str)  # Description


class TestCustomConfig:
    """Test custom config creation."""

    def test_create_custom_config(self):
        """Test creating a custom config."""
        config = create_custom_config(denoise_strength=0.40, preserve_breaths=True, target_lufs=-16.0)

        assert config.denoise_strength == 0.40
        assert config.preserve_breaths is True
        assert config.target_lufs == -16.0

    def test_create_custom_config_validates(self):
        """Test custom config is validated."""
        with pytest.raises(ValueError, match="denoise_strength"):
            create_custom_config(denoise_strength=2.0)  # Out of range

    def test_custom_config_inherits_defaults(self):
        """Test custom config inherits defaults for unspecified params."""
        config = create_custom_config(denoise_strength=0.40)

        # Should have defaults for other params
        assert config.preserve_breaths is True
        assert config.compression_ratio == 2.0


class TestCriticalRequirements:
    """Test critical requirements from roadmap."""

    def test_all_modes_preserve_breaths(self):
        """CRITICAL: All modes MUST preserve breaths."""
        for mode, config in PROCESSING_CONFIGS.items():
            assert config.preserve_breaths is True, f"Mode {mode.value} does not preserve breaths!"

    def test_studio_mode_streaming_ready(self):
        """STUDIO_2026 must be streaming-ready (LUFS -14)."""
        config = PROCESSING_CONFIGS[ProcessingMode.STUDIO_2026]
        assert config.target_lufs == -14.0

    def test_forensic_mode_minimal_processing(self):
        """FORENSIC mode wurde aus dem System entfernt."""
        assert "forensic" not in list_available_modes()
        with pytest.raises(ValueError):
            get_config_by_name("forensic")


def test_mode_demo():
    """Integration test: List all modes and print one config."""
    # List modes
    modes = list_available_modes()
    assert len(modes) == 2

    # Get one config
    config = get_config_by_name("restoration")
    config_dict = config.to_dict()

    # Should be JSON-serializable
    import json

    json_str = json.dumps(config_dict)
    assert len(json_str) > 0
