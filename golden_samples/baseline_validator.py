"""
Golden Sample Baseline Validator für AURIK v8
=============================================

Validate and update quality baselines for golden samples.

Purpose:
- Measure actual quality scores for golden samples
- Update metadata.json with measured values
- Validate baseline consistency
- Detect anomalies in baselines

Excellence Strategy #5: Golden Sample Library
- Accurate baselines for regression testing
- Consistent quality measurement
- Anomaly detection

Autor: AI Team
Datum: 11. Februar 2026
"""

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf

# Import AURIK components
from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker

logger = logging.getLogger(__name__)


class GoldenSampleBaselineValidator:
    """
    Validate and update quality baselines for golden samples.
    
    Workflow:
    1. Load golden sample metadata
    2. Load each audio file
    3. Measure Musical Goals scores
    4. Compare with existing baseline
    5. Update metadata.json if needed
    6. Report anomalies
    
    Features:
    - Automated baseline measurement
    - Consistency validation
    - Anomaly detection
    - Metadata update
    """

    def __init__(
        self,
        golden_samples_dir: Path,
        update_metadata: bool = False,
        anomaly_threshold: float = 0.20
    ):
        """
        Initialize baseline validator.
        
        Args:
            golden_samples_dir: Path to golden_samples/ directory
            update_metadata: If True, update metadata.json with measured values
            anomaly_threshold: Threshold for anomaly detection (default: 0.20 = 20%)
        """
        self.golden_samples_dir = Path(golden_samples_dir)
        self.update_metadata = update_metadata
        self.anomaly_threshold = anomaly_threshold

        # Load metadata
        metadata_path = self.golden_samples_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json not found: {metadata_path}")

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # Initialize quality checker
        self.musical_goals_checker = MusicalGoalsChecker()

        logger.info(
            f"BaselineValidator initialized: {len(self.metadata['golden_samples'])} samples, "
            f"update={update_metadata}, anomaly_threshold={anomaly_threshold}"
        )

    def validate_all_baselines(self) -> Dict:
        """
        Validate baselines for all golden samples.
        
        Returns:
            Validation report
        """
        samples = self.metadata["golden_samples"]

        logger.info(f"Validating baselines for {len(samples)} samples...")

        validation_results = []
        anomalies = []

        for i, sample_meta in enumerate(samples, 1):
            logger.info(f"[{i}/{len(samples)}] Validating {sample_meta['filename']}...")

            try:
                result = self._validate_sample_baseline(sample_meta)
                validation_results.append(result)

                if result["is_anomaly"]:
                    anomalies.append(result)
                    logger.warning(
                        f"  ANOMALY detected: max_deviation={result['max_deviation']:.3f}"
                    )
                else:
                    logger.info(f"  OK: max_deviation={result['max_deviation']:.3f}")

            except Exception as e:
                logger.error(f"  Failed to validate {sample_meta['filename']}: {e}")
                continue

        # Generate report
        report = self._generate_report(validation_results, anomalies)

        # Update metadata if needed
        if self.update_metadata:
            self._update_metadata(validation_results)

        logger.info(
            f"✓ Baseline validation complete: {len(anomalies)} anomalies detected"
        )

        return report

    def _validate_sample_baseline(self, sample_meta: Dict) -> Dict:
        """Validate baseline for a single sample."""
        # Load audio
        audio_path = self.golden_samples_dir / sample_meta["category"] / sample_meta["filename"]
        audio, sr = sf.read(audio_path)

        # Ensure mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Measure Musical Goals
        measured_scores = self.musical_goals_checker.measure_all(audio, sr)

        # Get existing baseline
        baseline_scores = sample_meta.get("quality_baseline", {})

        # Compare
        deviations = {}
        max_deviation = 0.0

        for goal in measured_scores.keys():
            baseline_value = baseline_scores.get(goal, 0.0)
            measured_value = measured_scores[goal]
            deviation = abs(measured_value - baseline_value)

            deviations[goal] = {
                "baseline": baseline_value,
                "measured": measured_value,
                "deviation": deviation
            }

            max_deviation = max(max_deviation, deviation)

        # Check for anomaly
        is_anomaly = max_deviation > self.anomaly_threshold

        return {
            "filename": sample_meta["filename"],
            "category": sample_meta["category"],
            "measured_scores": measured_scores,
            "baseline_scores": baseline_scores,
            "deviations": deviations,
            "max_deviation": max_deviation,
            "is_anomaly": is_anomaly
        }

    def _generate_report(self, validation_results: List[Dict], anomalies: List[Dict]) -> Dict:
        """Generate validation report."""
        total_samples = len(validation_results)

        # Calculate average deviation per goal
        all_goals = set()
        for result in validation_results:
            all_goals.update(result["measured_scores"].keys())

        goal_avg_deviations = {}
        for goal in all_goals:
            deviations = [
                result["deviations"].get(goal, {}).get("deviation", 0.0)
                for result in validation_results
                if goal in result["deviations"]
            ]
            goal_avg_deviations[goal] = np.mean(deviations) if deviations else 0.0

        report = {
            "summary": {
                "total_samples": total_samples,
                "anomalies_detected": len(anomalies),
                "anomaly_rate": len(anomalies) / total_samples if total_samples > 0 else 0.0,
                "avg_max_deviation": np.mean([r["max_deviation"] for r in validation_results]),
                "goal_avg_deviations": goal_avg_deviations,
                "timestamp": datetime.now().isoformat()
            },
            "anomalies": anomalies,
            "all_results": validation_results
        }

        return report

    def _update_metadata(self, validation_results: List[Dict]) -> None:
        """Update metadata.json with measured baselines."""
        logger.info("Updating metadata.json with measured baselines...")

        # Update each sample in metadata
        for result in validation_results:
            filename = result["filename"]
            measured_scores = result["measured_scores"]

            # Find sample in metadata
            for sample in self.metadata["golden_samples"]:
                if sample["filename"] == filename:
                    sample["quality_baseline"] = measured_scores
                    break

        # Recalculate metadata aggregates
        all_goals = set()
        for sample in self.metadata["golden_samples"]:
            all_goals.update(sample["quality_baseline"].keys())

        baseline_avgs = {}
        for goal in all_goals:
            values = [
                sample["quality_baseline"].get(goal, 0.0)
                for sample in self.metadata["golden_samples"]
                if goal in sample["quality_baseline"]
            ]
            baseline_avgs[goal] = np.mean(values) if values else 0.0

        self.metadata["metadata"]["quality_baseline"] = baseline_avgs
        self.metadata["metadata"]["last_updated"] = datetime.now().isoformat()

        # Save metadata
        metadata_path = self.golden_samples_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        logger.info(f"✓ Metadata updated: {metadata_path}")

    def export_report(self, report: Dict, output_path: Path) -> None:
        """Export validation report to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"✓ Validation report exported: {output_path}")

    def print_summary(self, report: Dict) -> None:
        """Print validation summary to console."""
        summary = report["summary"]

        print("\n" + "=" * 60)
        print("BASELINE VALIDATION SUMMARY")
        print("=" * 60)

        print(f"\nTotal Samples: {summary['total_samples']}")
        print(f"Anomalies Detected: {summary['anomalies_detected']} ({summary['anomaly_rate'] * 100:.1f}%)")
        print(f"Avg Max Deviation: {summary['avg_max_deviation']:.3f}")

        print("\nGoal-wise Avg Deviations:")
        for goal, deviation in summary["goal_avg_deviations"].items():
            print(f"  {goal}: {deviation:.3f}")

        if report["anomalies"]:
            print("\nAnomalies:")
            for anomaly in report["anomalies"]:
                print(f"  {anomaly['filename']} ({anomaly['category']}): max_deviation={anomaly['max_deviation']:.3f}")

                # Show top 3 deviations
                sorted_deviations = sorted(
                    anomaly["deviations"].items(),
                    key=lambda x: x[1]["deviation"],
                    reverse=True
                )[:3]

                for goal, dev_info in sorted_deviations:
                    print(
                        f"    {goal}: baseline={dev_info['baseline']:.3f}, "
                        f"measured={dev_info['measured']:.3f}, "
                        f"deviation={dev_info['deviation']:.3f}"
                    )

        print("\n" + "=" * 60)


def main():
    """Run baseline validation."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate golden sample baselines")
    parser.add_argument(
        "--golden-samples",
        type=str,
        default="golden_samples",
        help="Golden samples directory (default: golden_samples)"
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update metadata.json with measured values"
    )
    parser.add_argument(
        "--anomaly-threshold",
        type=float,
        default=0.20,
        help="Anomaly detection threshold (default: 0.20 = 20%%)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="baseline_validation_report.json",
        help="Output validation report JSON (default: baseline_validation_report.json)"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Initialize validator
    validator = GoldenSampleBaselineValidator(
        golden_samples_dir=Path(args.golden_samples),
        update_metadata=args.update,
        anomaly_threshold=args.anomaly_threshold
    )

    # Validate all baselines
    report = validator.validate_all_baselines()

    # Print summary
    validator.print_summary(report)

    # Export report
    validator.export_report(report, Path(args.output))

    print(f"\n✓ Validation report saved: {args.output}")

    if args.update:
        print("✓ Metadata.json updated with measured baselines")


if __name__ == "__main__":
    main()
