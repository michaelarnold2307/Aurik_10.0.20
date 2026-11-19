"""Benchmark ONNX vs PyTorch model performance

Compares inference speed and quality for:
- PyTorch (baseline)
- ONNX FP32
- ONNX INT8 (quantized)

Expected results:
- ONNX FP32: 1.5-2× faster
- ONNX INT8: 3-6× faster
- Quality loss: <1%
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.core.onnx.runtime import OptimizedONNXModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ModelBenchmark:
    """Benchmark ONNX models against PyTorch."""

    def __init__(self, num_warmup: int = 3, num_runs: int = 10):
        """
        Initialize benchmark.

        Args:
            num_warmup: Warmup iterations
            num_runs: Benchmark iterations
        """
        self.num_warmup = num_warmup
        self.num_runs = num_runs

    def benchmark_onnx_model(
        self, model_path: Path, sample_audio: np.ndarray, model_name: str = "model"
    ) -> dict[str, Any]:
        """
        Benchmark single ONNX model.

        Args:
            model_path: Path to ONNX model
            sample_audio: Input audio for benchmarking
            model_name: Model name for logging

        Returns:
            Benchmark results
        """
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Benchmarking: {model_name}")
        logger.info(f"Model: {model_path.name}")
        logger.info(f"{'=' * 60}")

        try:
            # Load ONNX model
            model = OptimizedONNXModel(model_path=model_path, enable_warmup=False)

            # Warmup
            logger.info(f"Warmup ({self.num_warmup} iterations)...")
            for _ in range(self.num_warmup):
                _ = model.process(sample_audio)

            # Benchmark
            logger.info(f"Benchmarking ({self.num_runs} iterations)...")
            times = []

            for i in range(self.num_runs):
                start = time.time()
                model.process(sample_audio)
                elapsed = time.time() - start
                times.append(elapsed)

            # Statistics
            avg_time = np.mean(times) * 1000  # ms
            std_time = np.std(times) * 1000  # ms
            min_time = np.min(times) * 1000  # ms
            max_time = np.max(times) * 1000  # ms

            # Calculate throughput (RTF - Real-Time Factor)
            audio_duration = len(sample_audio) / 48000  # assume 48kHz
            rtf = np.mean(times) / audio_duration

            results = {
                "model_name": model_name,
                "model_path": str(model_path),
                "avg_time_ms": avg_time,
                "std_time_ms": std_time,
                "min_time_ms": min_time,
                "max_time_ms": max_time,
                "real_time_factor": rtf,
                "throughput": 1.0 / rtf,
                "model_size_mb": model_path.stat().st_size / (1024 * 1024),
            }

            logger.info("\nResults:")
            logger.info(f"  Average time: {avg_time:.2f} ± {std_time:.2f} ms")
            logger.info(f"  Min/Max: {min_time:.2f} / {max_time:.2f} ms")
            logger.info(f"  Real-Time Factor: {rtf:.3f}× RT")
            logger.info(f"  Throughput: {results['throughput']:.2f}× real-time")
            logger.info(f"  Model size: {results['model_size_mb']:.2f} MB")

            return results

        except Exception as e:
            logger.error(f"Benchmark fehlgeschlagen: {e}")
            return {"model_name": model_name, "error": str(e)}

    def compare_models(self, model_paths: dict[str, Path], sample_audio: np.ndarray) -> dict[str, Any]:
        """
        Compare multiple models.

        Args:
            model_paths: Dict of {name: path}
            sample_audio: Input audio

        Returns:
            Comparison results
        """
        results = {}

        for name, path in model_paths.items():
            results[name] = self.benchmark_onnx_model(model_path=path, sample_audio=sample_audio, model_name=name)

        # Print comparison
        self._print_comparison(results)

        return results

    def _print_comparison(self, results: dict[str, Any]) -> None:
        """Print comparison table."""
        logger.info(f"\n{'=' * 80}")
        logger.info("COMPARISON SUMMARY")
        logger.info(f"{'=' * 80}")

        # Find baseline (usually first model or 'pytorch')
        baseline_name = list(results.keys())[0]
        if "pytorch" in results:
            baseline_name = "pytorch"

        baseline_time = results[baseline_name].get("avg_time_ms", 0)

        # Header
        print(f"\n{'Model':<30} {'Time (ms)':<15} {'RTF':<10} {'Speedup':<10} {'Size (MB)':<12}")
        print("-" * 80)

        # Rows
        for name, data in results.items():
            if "error" in data:
                print(f"{name:<30} ERROR: {data['error']}")
                continue

            time_ms = data["avg_time_ms"]
            rtf = data["real_time_factor"]
            size_mb = data["model_size_mb"]

            speedup = baseline_time / time_ms if time_ms > 0 else 0

            print(
                f"{name:<30} "
                f"{time_ms:>8.2f} ± {data['std_time_ms']:.2f}   "
                f"{rtf:>6.3f}×   "
                f"{speedup:>6.2f}×   "
                f"{size_mb:>8.2f}"
            )

        print("-" * 80)


def benchmark_deepfilternet(benchmark: ModelBenchmark, models_dir: Path) -> dict[str, Any]:
    """Benchmark DeepFilterNet models."""
    logger.info("\n" + "=" * 80)
    logger.info("DEEPFILTERNET BENCHMARK")
    logger.info("=" * 80)

    # Generate sample audio (1 second at 48kHz)
    sample_audio = np.random.randn(48000).astype(np.float32)

    # Model paths
    model_paths = {}

    dfn_dir = models_dir / "deepfilternet_v3_ii"
    if (dfn_dir / "deepfilternet_v3_II_enc.onnx").exists():
        model_paths["deepfilternet_encoder"] = dfn_dir / "deepfilternet_v3_II_enc.onnx"
    if (dfn_dir / "deepfilternet_v3_II_dec.onnx").exists():
        model_paths["deepfilternet_decoder"] = dfn_dir / "deepfilternet_v3_II_dec.onnx"

    if not model_paths:
        logger.warning("DeepFilterNet ONNX models not found")
        return {}

    return benchmark.compare_models(model_paths, sample_audio)


def benchmark_demucs(benchmark: ModelBenchmark, models_dir: Path) -> dict[str, Any]:
    """Benchmark Demucs models."""
    logger.info("\n" + "=" * 80)
    logger.info("DEMUCS BENCHMARK")
    logger.info("=" * 80)

    # Generate sample audio (2 channels, 5 seconds at 44.1kHz)
    sample_audio = np.random.randn(2, 44100 * 5).astype(np.float32)

    # Model paths
    model_paths = {}

    demucs_dir = models_dir / "demucs"
    if (demucs_dir / "htdemucs_6s.onnx").exists():
        model_paths["demucs_htdemucs"] = demucs_dir / "htdemucs_6s.onnx"

    if not model_paths:
        logger.warning("Demucs ONNX models not found")
        return {}

    return benchmark.compare_models(model_paths, sample_audio)


def benchmark_all_models(models_dir: Path) -> None:
    """Benchmark all available ONNX models."""
    benchmark = ModelBenchmark(num_warmup=3, num_runs=10)

    # Benchmark each model type
    deepfilternet_results = benchmark_deepfilternet(benchmark, models_dir)
    demucs_results = benchmark_demucs(benchmark, models_dir)

    # Overall summary
    logger.info("\n" + "=" * 80)
    logger.info("OVERALL SUMMARY")
    logger.info("=" * 80)
    logger.info(f"DeepFilterNet models benchmarked: {len(deepfilternet_results)}")
    logger.info(f"Demucs models benchmarked: {len(demucs_results)}")
    logger.info("=" * 80)


def main():
    """Main benchmark script."""
    parser = argparse.ArgumentParser(description="Benchmark ONNX model performance")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(__file__).parent.parent / "models",
        help="Directory containing ONNX models",
    )
    parser.add_argument("--model", type=str, help="Specific model to benchmark (optional)")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup iterations")
    parser.add_argument("--runs", type=int, default=10, help="Number of benchmark runs")

    args = parser.parse_args()

    if not args.models_dir.exists():
        logger.error(f"Models directory not found: {args.models_dir}")
        sys.exit(1)

    logger.info(f"Models directory: {args.models_dir}")
    logger.info(f"Warmup iterations: {args.warmup}")
    logger.info(f"Benchmark runs: {args.runs}")

    benchmark_all_models(args.models_dir)


if __name__ == "__main__":
    main()
