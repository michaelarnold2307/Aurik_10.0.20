"""
Automatisierte Ketten-Optimierung für SOTA-DSP-Workflow
Findet die beste Reihenfolge und Auswahl der DSP-Module auf Basis vergangener Audit-Reports.
"""

import glob
import json
import logging
import os

import yaml

_logger = logging.getLogger(__name__)

AUDIT_DIR = "./"
POLICY_PATH = "config_dsp_chain_example.yaml"
MODULES = [
    "automatic_decrackler",
    "dehiss",
    "automatic_dehum",
    "shellac_declicker",
    "automatic_declipper",
    "sota_denoiser",
    "sota_dereverberator",
    "automatic_deesser",
    "limiter",
    "loudness_matching",
    "stereo_widener",
    "harmonic_exciter",
]

# Lade aktuelle Policy
with open(POLICY_PATH) as f:
    policy = yaml.safe_load(f)

# Sammle alle Audit-Reports
audit_files = glob.glob(os.path.join(AUDIT_DIR, "*_audit.json"))

# Sammle Ketten und Scores
dsp_chains = []
scores = []
for afile in audit_files:
    with open(afile) as f:
        report = json.load(f)
    chain = report.get("dsp_chain", [])
    score = 0.0
    # Score: +1 für passed, +hf_ratio, +correlation, -artefact_energy
    if report.get("passed", False):
        score += 1.0
    score += report.get("hf_ratio", 0.0)
    score += report.get("correlation", 0.0)
    score -= report.get("artefact_energy", 0.0)
    dsp_chains.append(chain)
    scores.append(score)

# Finde die häufigste und/oder beste Kette
from collections import Counter

chain_counts = Counter(tuple(chain) for chain in dsp_chains if chain)
if chain_counts:
    best_chain = list(chain_counts.most_common(1)[0][0])
    _logger.info("[Ketten-Optimierung] Häufigste Kette: %s", best_chain)
    policy["dsp_chain"] = best_chain
else:
    _logger.info("[Ketten-Optimierung] Keine Ketten gefunden. Nutze Default.")

# Optional: Suche nach bester Permutation (nur für kleine Modulzahlen praktikabel)
# permutations = list(itertools.permutations(MODULES, 4))
# ...

# Speichere die angepasste Policy
with open(POLICY_PATH, "w") as f:
    yaml.dump(policy, f)

_logger.info("[Ketten-Optimierung] Policy/Chain wurde angepasst:")
_logger.info("%s", policy)
