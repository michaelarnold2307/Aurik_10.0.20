"""Pre-commit hook: verify 15 Musical Goals are configured."""
# pylint: disable=wrong-import-position

import sys

sys.path.insert(0, ".")
from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker

c = MusicalGoalsChecker()
n = len(c.thresholds)
assert n == 15, f"Erwartet 15 Musical Goals, erhalten {n}"
print("OK: 15 Musical Goals konfiguriert")
