"""
Modern Configuration System for Aurik
Uses Pydantic Settings for type-safe, validated configuration with environment variable support.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DSPConfig(BaseModel):
    """DSP Processing Configuration"""

    default_sr: int = Field(
        default=48000, ge=8000, le=96000, description="Default sample rate in Hz (Aurik canonical SR)"
    )
    max_duration: int = Field(default=7200, ge=1, description="Maximum audio duration in seconds")
    buffer_size: int = Field(default=2048, ge=64, le=8192, description="Audio buffer size")
    hop_length: int = Field(default=512, ge=32, le=4096, description="STFT hop length")
    n_fft: int = Field(default=2048, ge=128, le=8192, description="FFT window size")

    @field_validator("default_sr")
    @classmethod
    def validate_sample_rate(cls, v: int) -> int:
        allowed = [8000, 16000, 22050, 32000, 44100, 48000, 88200, 96000]
        if v not in allowed:
            raise ValueError(f"Sample rate must be one of {allowed}, got {v}")
        return v


class DockerConfig(BaseModel):
    """Docker/Container Configuration"""

    enabled: bool = Field(
        default=False, description="Enable Docker-based processing (RELEASE_MUST: False for production)"
    )
    voice_conversion_container: str = Field(default="aurik_voice_conversion_container")
    codec_enhance_container: str = Field(default="aurik_hifigan_container")
    timeout: int = Field(default=300, ge=10, description="Docker operation timeout in seconds")
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")


class MLConfig(BaseModel):
    """Machine Learning Configuration"""

    # §4.4: versa_plugin ersetzt cdpam_plugin als non-reference MOS-Metrik (April 2026)
    feature_plugins: list[str] = Field(default_factory=lambda: ["panns_integration", "versa_plugin"])
    model_cache_dir: Path = Field(default=Path("models/"), description="ML model cache directory")
    use_gpu: bool = Field(
        default=True, description="GPU acceleration (True=auto-detect GPU, CPU fallback always active)"
    )
    batch_size: int = Field(default=1, ge=1, le=128, description="Batch size for ML inference")
    precision: Literal["fp32", "fp16", "int8"] = Field(default="fp32", description="Model precision mode")


class QualityConfig(BaseModel):
    """Quality Metrics Configuration"""

    sdr_metric: Literal["mir_eval"] = (
        Field(  # §4.4: pesq/dnsmos/nisqa/cdpam verboten für Musikrestaurierung — nur mir_eval
            default="mir_eval", description="Primary SDR metric"
        )
    )
    min_quality_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum quality threshold")
    enable_quality_gates: bool = Field(default=True, description="Enable quality gate checks")


class PolicyConfig(BaseModel):
    """Policy and Audit Configuration"""

    policy_mode: Literal["restore_only", "full", "forensic"] = Field(
        default="restore_only", description="Processing policy mode"
    )
    audit_log_path: Path = Field(default=Path("audit/voice_conversion_audit.log"))
    enable_audit_trail: bool = Field(default=True, description="Enable comprehensive audit logging")
    max_log_size_mb: int = Field(default=100, ge=1, le=10000, description="Max audit log size in MB")


class PerformanceConfig(BaseModel):
    """Performance Optimization Configuration"""

    enable_caching: bool = Field(default=True, description="Enable computation caching")
    cache_ttl_seconds: int = Field(default=3600, ge=60, description="Cache TTL in seconds")
    max_workers: int | None = Field(default=None, description="Max parallel workers (None = auto)")
    enable_profiling: bool = Field(default=False, description="Enable performance profiling")


class AurikSettings(BaseSettings):
    """
    Main Aurik Configuration
    Loads from environment variables, .env file, or defaults
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="Aurik Audio Restoration", description="Application name")
    version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")

    # Paths
    input_dir: Path = Field(default=Path("input/"), description="Input audio directory")
    output_dir: Path = Field(default=Path("output/"), description="Output audio directory")
    temp_dir: Path = Field(default=Path("temp/"), description="Temporary files directory")

    # Sub-configurations
    dsp: DSPConfig = Field(default_factory=DSPConfig)
    docker: DockerConfig = Field(default_factory=DockerConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)

    def validate_all(self) -> None:
        """Run comprehensive validation"""
        # Create directories if they don't exist
        for path in [self.input_dir, self.output_dir, self.temp_dir, self.ml.model_cache_dir]:
            path.mkdir(parents=True, exist_ok=True)

        # Validate policy mode
        if self.policy.policy_mode not in ["restore_only", "full", "forensic"]:
            raise ValueError(f"Invalid policy mode: {self.policy.policy_mode}")

    def to_legacy_config(self) -> dict:
        """Convert to legacy Config.py format for backward compatibility"""
        return {
            "docker_enabled": self.docker.enabled,
            "default_sr": self.dsp.default_sr,
            "voice_conversion_container": self.docker.voice_conversion_container,
            "codec_enhance_container": self.docker.codec_enhance_container,
            "feature_plugins": self.ml.feature_plugins,
            "sdr_metric": self.quality.sdr_metric,
            "audit_log_path": str(self.policy.audit_log_path),
            "policy_mode": self.policy.policy_mode,
        }


@lru_cache(maxsize=1)
def get_settings() -> AurikSettings:
    """
    Get singleton settings instance
    Cached for performance
    """
    return AurikSettings()


# Convenience function for legacy code
def cfg() -> dict:
    """Legacy compatibility function"""
    settings = get_settings()
    settings.validate_all()
    return settings.to_legacy_config()


if __name__ == "__main__":
    import logging as _logging

    _logging.basicConfig(level=_logging.DEBUG)
    _main_logger = _logging.getLogger("aurik.config.settings")
    # Sanity-check: dump configuration via logger (never print() in production code)
    settings = get_settings()
    _main_logger.info("=== Aurik Configuration ===")
    _main_logger.info("App: %s v%s", settings.app_name, settings.version)
    _main_logger.info("Debug: %s", settings.debug)
    _main_logger.info("Sample Rate: %d Hz", settings.dsp.default_sr)
    _main_logger.info("Docker Enabled: %s", settings.docker.enabled)
    _main_logger.info("GPU Enabled: %s", settings.ml.use_gpu)
    _main_logger.info("Policy Mode: %s", settings.policy.policy_mode)
    _main_logger.info("Caching: %s", settings.performance.enable_caching)
    _main_logger.info("=== Legacy Config Format ===")
    _main_logger.info("%s", settings.to_legacy_config())
