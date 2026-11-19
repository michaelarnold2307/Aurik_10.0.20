import json
import os
from typing import Any


class SOTAMaximumAnalyzer:
    """
    Analysiert alle Reports (report.json) und extrahiert SOTA-Strategien, Fehlerquellen und Optimierungspotenziale.
    Liefert Empfehlungen für die SOTA-Maximum-Policy.
    """

    def __init__(self, report_dir="../results_sota"):
        self.report_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), report_dir))
        self.report_path = os.path.join(self.report_dir, "report.json")
        self.report = self._load_report()

    def _load_report(self) -> dict[str, Any]:
        if not os.path.isfile(self.report_path):
            raise FileNotFoundError(f"Report nicht gefunden: {self.report_path}")
        with open(self.report_path) as f:
            from typing import cast

            return cast(dict[str, Any], json.load(f))

    def get_top_policies(self, top_n=3) -> list[dict[str, Any]]:
        """Gibt die Policies mit den höchsten Scores zurück."""
        jobs = self.report.get("jobs", [])
        scores = self.report.get("scores", [])
        self.report.get("policy_events", [])
        # Kombiniere Jobs und Scores
        job_scores = [{"job": job, "score": scores[i] if i < len(scores) else None} for i, job in enumerate(jobs)]
        # Sortiere nach Score absteigend
        job_scores = [js for js in job_scores if js["score"] is not None]
        job_scores.sort(key=lambda x: x["score"], reverse=True)
        return job_scores[:top_n]

    def get_common_errors(self, min_count=2) -> list[str]:
        """Gibt die häufigsten Fehlermeldungen zurück."""
        errors = self.report.get("errors", [])
        from collections import Counter

        counter = Counter(errors)
        return [err for err, count in counter.items() if count >= min_count]

    def recommend_sota_policy(self) -> dict[str, Any]:
        """Leitet aus den Top-Jobs eine SOTA-Maximum-Policy ab."""
        top = self.get_top_policies(top_n=5)
        # Extrahiere Policy-Parameter, die bei Top-Jobs häufig vorkommen
        from collections import Counter, defaultdict

        param_counter: dict[str, Counter] = defaultdict(Counter)
        for entry in top:
            policy = entry["job"].get("policy", {})
            for k, v in policy.items():
                param_counter[k][str(v)] += 1
        # Wähle für jedes Policy-Attribut den häufigsten Wert
        sota_policy = {k: counter.most_common(1)[0][0] for k, counter in param_counter.items() if counter}
        return sota_policy
