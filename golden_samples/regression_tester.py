"""
Golden Sample Regression Tester für AURIK v8
============================================

Regression testing across code versions.

Purpose:
- Compare benchmark results across versions
- Detect quality degradations
- Track performance trends
- Alert on regressions

Excellence Strategy #5: Golden Sample Library
- Continuous quality monitoring
- Version comparison
- Regression detection

Autor: AI Team
Datum: 11. Februar 2026
"""

from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RegressionAlert:
    """Regression alert for a specific goal/metric."""
    goal_name: str
    baseline_version: str
    current_version: str
    baseline_value: float
    current_value: float
    degradation: float
    severity: str  # "minor", "moderate", "critical"
    category: Optional[str] = None
    samples_affected: Optional[List[str]] = None


@dataclass
class RegressionSummary:
    """Summary of regression testing."""
    baseline_version: str
    current_version: str
    total_goals: int
    regressions_detected: int
    minor_regressions: int
    moderate_regressions: int
    critical_regressions: int
    improvements: int
    no_change: int
    alerts: List[RegressionAlert]
    timestamp: str


class GoldenSampleRegressionTester:
    """
    Regression tester for golden sample benchmarks.
    
    Workflow:
    1. Load baseline benchmark report (previous version)
    2. Load current benchmark report (current version)
    3. Compare Musical Goals scores
    4. Detect degradations (regressions)
    5. Generate alerts for significant regressions
    6. Export regression report
    
    Features:
    - Version comparison
    - Configurable thresholds (minor/moderate/critical)
    - Category-wise analysis
    - Sample-level tracking
    """
    
    def __init__(
        self,
        minor_threshold: float = 0.02,
        moderate_threshold: float = 0.05,
        critical_threshold: float = 0.10
    ):
        """
        Initialize regression tester.
        
        Args:
            minor_threshold: Degradation threshold for minor alert (default: 0.02 = 2%)
            moderate_threshold: Degradation threshold for moderate alert (default: 0.05 = 5%)
            critical_threshold: Degradation threshold for critical alert (default: 0.10 = 10%)
        """
        self.minor_threshold = minor_threshold
        self.moderate_threshold = moderate_threshold
        self.critical_threshold = critical_threshold
        
        logger.info(
            f"RegressionTester initialized: minor={minor_threshold}, "
            f"moderate={moderate_threshold}, critical={critical_threshold}"
        )
    
    def compare_reports(
        self,
        baseline_report_path: Path,
        current_report_path: Path,
        baseline_version: str = "v1",
        current_version: str = "v2"
    ) -> Tuple[RegressionSummary, List[RegressionAlert]]:
        """
        Compare two benchmark reports to detect regressions.
        
        Args:
            baseline_report_path: Path to baseline benchmark report JSON
            current_report_path: Path to current benchmark report JSON
            baseline_version: Version identifier for baseline
            current_version: Version identifier for current
        
        Returns:
            (summary, alerts)
        """
        # Load reports
        with open(baseline_report_path, 'r') as f:
            baseline_report = json.load(f)
        
        with open(current_report_path, 'r') as f:
            current_report = json.load(f)
        
        logger.info(
            f"Comparing {baseline_version} vs {current_version} "
            f"({baseline_report['summary']['total_samples']} samples)..."
        )
        
        # Compare summary-level metrics
        baseline_avg = baseline_report["summary"]["baseline_avg"]
        current_avg = current_report["summary"]["achieved_avg"]
        
        alerts = []
        
        # Get all goals
        all_goals = set(baseline_avg.keys()).union(set(current_avg.keys()))
        
        for goal in all_goals:
            baseline_value = baseline_avg.get(goal, 0.0)
            current_value = current_avg.get(goal, 0.0)
            degradation = current_value - baseline_value
            
            # Check for regression (degradation)
            if degradation < -self.minor_threshold:
                severity = self._classify_severity(abs(degradation))
                
                alert = RegressionAlert(
                    goal_name=goal,
                    baseline_version=baseline_version,
                    current_version=current_version,
                    baseline_value=baseline_value,
                    current_value=current_value,
                    degradation=degradation,
                    severity=severity
                )
                
                alerts.append(alert)
                
                logger.warning(
                    f"REGRESSION detected: {goal} = {baseline_value:.3f} → {current_value:.3f} "
                    f"({degradation:+.3f}) [{severity}]"
                )
        
        # Also check category-wise regressions
        category_alerts = self._compare_categories(
            baseline_report,
            current_report,
            baseline_version,
            current_version
        )
        
        alerts.extend(category_alerts)
        
        # Verbesserungen zählen: positiver Δ über minor_threshold
        improvements_count = sum(
            1 for g in all_goals
            if (current_avg.get(g, 0.0) - baseline_avg.get(g, 0.0))
            > self.minor_threshold
        )

        # Generate summary
        summary = self._generate_summary(
            alerts,
            baseline_version,
            current_version,
            len(all_goals),
            improvements_count,
        )
        
        logger.info(
            f"✓ Regression testing complete: {summary.regressions_detected} regressions detected "
            f"(minor={summary.minor_regressions}, moderate={summary.moderate_regressions}, "
            f"critical={summary.critical_regressions})"
        )
        
        return summary, alerts
    
    def _classify_severity(self, degradation: float) -> str:
        """Classify regression severity."""
        if degradation >= self.critical_threshold:
            return "critical"
        elif degradation >= self.moderate_threshold:
            return "moderate"
        else:
            return "minor"
    
    def _compare_categories(
        self,
        baseline_report: Dict,
        current_report: Dict,
        baseline_version: str,
        current_version: str
    ) -> List[RegressionAlert]:
        """Compare category-wise performance."""
        alerts = []
        
        baseline_categories = baseline_report["summary"].get("category_results", {})
        current_categories = current_report["summary"].get("category_results", {})
        
        all_categories = set(baseline_categories.keys()).union(set(current_categories.keys()))
        
        for category in all_categories:
            baseline_stats = baseline_categories.get(category, {})
            current_stats = current_categories.get(category, {})
            
            # Compare pass rates
            baseline_pass_rate = baseline_stats.get("pass_rate", 0.0)
            current_pass_rate = current_stats.get("pass_rate", 0.0)
            pass_rate_degradation = current_pass_rate - baseline_pass_rate
            
            if pass_rate_degradation < -self.minor_threshold:
                severity = self._classify_severity(abs(pass_rate_degradation))
                
                alert = RegressionAlert(
                    goal_name="pass_rate",
                    baseline_version=baseline_version,
                    current_version=current_version,
                    baseline_value=baseline_pass_rate,
                    current_value=current_pass_rate,
                    degradation=pass_rate_degradation,
                    severity=severity,
                    category=category
                )
                
                alerts.append(alert)
                
                logger.warning(
                    f"REGRESSION in {category}: pass_rate = {baseline_pass_rate:.1%} → "
                    f"{current_pass_rate:.1%} ({pass_rate_degradation:+.1%}) [{severity}]"
                )
            
            # Compare avg improvements
            baseline_improvement = baseline_stats.get("avg_improvement", 0.0)
            current_improvement = current_stats.get("avg_improvement", 0.0)
            improvement_degradation = current_improvement - baseline_improvement
            
            if improvement_degradation < -self.minor_threshold:
                severity = self._classify_severity(abs(improvement_degradation))
                
                alert = RegressionAlert(
                    goal_name="avg_improvement",
                    baseline_version=baseline_version,
                    current_version=current_version,
                    baseline_value=baseline_improvement,
                    current_value=current_improvement,
                    degradation=improvement_degradation,
                    severity=severity,
                    category=category
                )
                
                alerts.append(alert)
        
        return alerts
    
    def _generate_summary(
        self,
        alerts: List[RegressionAlert],
        baseline_version: str,
        current_version: str,
        total_goals: int,
        improvements_count: int = 0,
    ) -> RegressionSummary:
        """Generate regression summary."""
        minor_count = sum(1 for a in alerts if a.severity == "minor")
        moderate_count = sum(1 for a in alerts if a.severity == "moderate")
        critical_count = sum(1 for a in alerts if a.severity == "critical")
        
        return RegressionSummary(
            baseline_version=baseline_version,
            current_version=current_version,
            total_goals=total_goals,
            regressions_detected=len(alerts),
            minor_regressions=minor_count,
            moderate_regressions=moderate_count,
            critical_regressions=critical_count,
            improvements=improvements_count,
            no_change=max(0, total_goals - len(alerts) - improvements_count),
            alerts=alerts,
            timestamp=datetime.now().isoformat()
        )
    
    def export_report(
        self,
        summary: RegressionSummary,
        output_path: Path
    ) -> None:
        """
        Export regression report to JSON.
        
        Args:
            summary: Regression summary
            output_path: Path to output JSON file
        """
        report = {
            "summary": {
                "baseline_version": summary.baseline_version,
                "current_version": summary.current_version,
                "total_goals": summary.total_goals,
                "regressions_detected": summary.regressions_detected,
                "minor_regressions": summary.minor_regressions,
                "moderate_regressions": summary.moderate_regressions,
                "critical_regressions": summary.critical_regressions,
                "improvements": summary.improvements,
                "no_change": summary.no_change,
                "timestamp": summary.timestamp
            },
            "alerts": [
                {
                    "goal_name": alert.goal_name,
                    "baseline_version": alert.baseline_version,
                    "current_version": alert.current_version,
                    "baseline_value": alert.baseline_value,
                    "current_value": alert.current_value,
                    "degradation": alert.degradation,
                    "severity": alert.severity,
                    "category": alert.category,
                    "samples_affected": alert.samples_affected
                }
                for alert in summary.alerts
            ]
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✓ Regression report exported: {output_path}")
    
    def print_summary(self, summary: RegressionSummary) -> None:
        """Print regression summary to console."""
        print("\n" + "=" * 60)
        print("REGRESSION TESTING SUMMARY")
        print("=" * 60)
        
        print(f"\nBaseline Version: {summary.baseline_version}")
        print(f"Current Version: {summary.current_version}")
        print(f"Total Goals: {summary.total_goals}")
        
        print(f"\nRegressions Detected: {summary.regressions_detected}")
        print(f"  Minor: {summary.minor_regressions}")
        print(f"  Moderate: {summary.moderate_regressions}")
        print(f"  Critical: {summary.critical_regressions}")
        
        print(f"\nImprovements: {summary.improvements}")
        print(f"No Change: {summary.no_change}")
        
        if summary.alerts:
            print("\nRegression Alerts:")
            
            # Group by severity
            for severity in ["critical", "moderate", "minor"]:
                severity_alerts = [a for a in summary.alerts if a.severity == severity]
                
                if severity_alerts:
                    print(f"\n  {severity.upper()}:")
                    
                    for alert in severity_alerts:
                        category_str = f" ({alert.category})" if alert.category else ""
                        print(
                            f"    {alert.goal_name}{category_str}: "
                            f"{alert.baseline_value:.3f} → {alert.current_value:.3f} "
                            f"({alert.degradation:+.3f})"
                        )
        
        print("\n" + "=" * 60)


def main():
    """Run regression testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run golden sample regression testing")
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Baseline benchmark report JSON"
    )
    parser.add_argument(
        "--current",
        type=str,
        required=True,
        help="Current benchmark report JSON"
    )
    parser.add_argument(
        "--baseline-version",
        type=str,
        default="v1",
        help="Baseline version identifier (default: v1)"
    )
    parser.add_argument(
        "--current-version",
        type=str,
        default="v2",
        help="Current version identifier (default: v2)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="regression_report.json",
        help="Output regression report JSON (default: regression_report.json)"
    )
    parser.add_argument(
        "--minor-threshold",
        type=float,
        default=0.02,
        help="Minor regression threshold (default: 0.02 = 2%%)"
    )
    parser.add_argument(
        "--moderate-threshold",
        type=float,
        default=0.05,
        help="Moderate regression threshold (default: 0.05 = 5%%)"
    )
    parser.add_argument(
        "--critical-threshold",
        type=float,
        default=0.10,
        help="Critical regression threshold (default: 0.10 = 10%%)"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Initialize tester
    tester = GoldenSampleRegressionTester(
        minor_threshold=args.minor_threshold,
        moderate_threshold=args.moderate_threshold,
        critical_threshold=args.critical_threshold
    )
    
    # Compare reports
    summary, alerts = tester.compare_reports(
        baseline_report_path=Path(args.baseline),
        current_report_path=Path(args.current),
        baseline_version=args.baseline_version,
        current_version=args.current_version
    )
    
    # Print summary
    tester.print_summary(summary)
    
    # Export report
    tester.export_report(summary, Path(args.output))
    
    print(f"\n✓ Regression report saved: {args.output}")
    
    # Exit with error code if critical regressions detected
    if summary.critical_regressions > 0:
        print("\n⚠ CRITICAL REGRESSIONS DETECTED - BUILD SHOULD FAIL")
        exit(1)
    elif summary.moderate_regressions > 0:
        print("\n⚠ MODERATE REGRESSIONS DETECTED - REVIEW RECOMMENDED")
        exit(0)
    else:
        exit(0)


if __name__ == "__main__":
    main()
