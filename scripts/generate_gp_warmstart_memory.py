"""
Aurik 10.0.0 — GP Warm-Start Memory Generator
==========================================
Generiert synthetische, aber fundierte GP-Memory-Dateien für alle Materialtypen.
Diese werden als vortrainierte Priors mit der Distribution ausgeliefert und beim
ersten Start in ~/.aurik/gp_memory/ kopiert (falls noch nicht vorhanden).

Methodik:
- Basis: MATERIAL_DEFAULTS aus gp_parameter_optimizer.py
- Für jeden Materialtyp: 20 Beobachtungen = 1 Zentrum + Perturbationen + Grenzen
- Scores basieren auf bekanntem Domänen-Wissen (nicht gemessen, aber plausibel)
- Die Beobachtungen geben dem GP einen validen Prior der sofort besser ist als
  Zufallsexploration, aber genug Unsicherheit lässt damit echte Daten dominieren

Ausführen:
    python scripts/generate_gp_warmstart_memory.py

Ausgabe: data/gp_warmstart/<material>.json
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

# --- Projekt-Root ins sys.path ---
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from backend.core.gp_parameter_optimizer import (
    MATERIAL_DEFAULTS,
    PARAMETER_SPACE,
    PARETO_OBJECTIVES,
    _normalize_params,
)

# ---------------------------------------------------------------------------
# Ausgabeverzeichnis
# ---------------------------------------------------------------------------
OUTPUT_DIR = _ROOT / "data" / "gp_warmstart"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Materialspezifische Expertenwissen-Scores (Musical Goals)
# Basieren auf Domänenwissen, nicht auf Messungen.
# Werte sind konservativ — reale Daten überschreiben rasch.
# ---------------------------------------------------------------------------
_MATERIAL_GOAL_BASE: dict[str, dict[str, float]] = {
    "tape": {
        "natuerlichkeit": 0.78,
        "authentizitaet": 0.76,
        "tonal_center": 0.82,
        "timbre_authentizitaet": 0.75,
        "artikulation": 0.74,
        "waerme": 0.78,
        "brillanz": 0.68,
        "emotionalitaet": 0.76,
        "transparenz": 0.72,
        "bass_kraft": 0.75,
        "groove": 0.73,
        "spatial_depth": 0.68,
        "micro_dynamics": 0.74,
        "separation_fidelity": 0.70,
    },
    "reel_tape": {
        "natuerlichkeit": 0.80,
        "authentizitaet": 0.78,
        "tonal_center": 0.84,
        "timbre_authentizitaet": 0.78,
        "artikulation": 0.76,
        "waerme": 0.80,
        "brillanz": 0.72,
        "emotionalitaet": 0.78,
        "transparenz": 0.75,
        "bass_kraft": 0.77,
        "groove": 0.75,
        "spatial_depth": 0.70,
        "micro_dynamics": 0.76,
        "separation_fidelity": 0.74,
    },
    "vinyl": {
        "natuerlichkeit": 0.76,
        "authentizitaet": 0.75,
        "tonal_center": 0.80,
        "timbre_authentizitaet": 0.74,
        "artikulation": 0.73,
        "waerme": 0.82,
        "brillanz": 0.70,
        "emotionalitaet": 0.77,
        "transparenz": 0.70,
        "bass_kraft": 0.78,
        "groove": 0.80,
        "spatial_depth": 0.72,
        "micro_dynamics": 0.76,
        "separation_fidelity": 0.71,
    },
    "shellac": {
        "natuerlichkeit": 0.72,
        "authentizitaet": 0.70,
        "tonal_center": 0.78,
        "timbre_authentizitaet": 0.70,
        "artikulation": 0.68,
        "waerme": 0.74,
        "brillanz": 0.55,
        "emotionalitaet": 0.72,
        "transparenz": 0.65,
        "bass_kraft": 0.65,
        "groove": 0.68,
        "spatial_depth": 0.55,
        "micro_dynamics": 0.68,
        "separation_fidelity": 0.62,
    },
    "wax_cylinder": {
        "natuerlichkeit": 0.65,
        "authentizitaet": 0.63,
        "tonal_center": 0.72,
        "timbre_authentizitaet": 0.62,
        "artikulation": 0.60,
        "waerme": 0.68,
        "brillanz": 0.45,
        "emotionalitaet": 0.66,
        "transparenz": 0.58,
        "bass_kraft": 0.55,
        "groove": 0.60,
        "spatial_depth": 0.45,
        "micro_dynamics": 0.60,
        "separation_fidelity": 0.55,
    },
    "cassette": {
        "natuerlichkeit": 0.74,
        "authentizitaet": 0.73,
        "tonal_center": 0.79,
        "timbre_authentizitaet": 0.72,
        "artikulation": 0.71,
        "waerme": 0.72,
        "brillanz": 0.65,
        "emotionalitaet": 0.74,
        "transparenz": 0.70,
        "bass_kraft": 0.72,
        "groove": 0.72,
        "spatial_depth": 0.65,
        "micro_dynamics": 0.70,
        "separation_fidelity": 0.68,
    },
    "digital": {
        "natuerlichkeit": 0.84,
        "authentizitaet": 0.83,
        "tonal_center": 0.88,
        "timbre_authentizitaet": 0.82,
        "artikulation": 0.82,
        "waerme": 0.76,
        "brillanz": 0.82,
        "emotionalitaet": 0.80,
        "transparenz": 0.84,
        "bass_kraft": 0.80,
        "groove": 0.80,
        "spatial_depth": 0.78,
        "micro_dynamics": 0.82,
        "separation_fidelity": 0.80,
    },
    "mp3_low": {
        "natuerlichkeit": 0.75,
        "authentizitaet": 0.74,
        "tonal_center": 0.80,
        "timbre_authentizitaet": 0.73,
        "artikulation": 0.72,
        "waerme": 0.72,
        "brillanz": 0.68,
        "emotionalitaet": 0.74,
        "transparenz": 0.72,
        "bass_kraft": 0.72,
        "groove": 0.73,
        "spatial_depth": 0.68,
        "micro_dynamics": 0.72,
        "separation_fidelity": 0.70,
    },
    "unknown": {
        "natuerlichkeit": 0.76,
        "authentizitaet": 0.75,
        "tonal_center": 0.80,
        "timbre_authentizitaet": 0.74,
        "artikulation": 0.73,
        "waerme": 0.74,
        "brillanz": 0.70,
        "emotionalitaet": 0.74,
        "transparenz": 0.72,
        "bass_kraft": 0.73,
        "groove": 0.73,
        "spatial_depth": 0.67,
        "micro_dynamics": 0.72,
        "separation_fidelity": 0.70,
    },
}

# Alias: gp_memory_key → MATERIAL_DEFAULTS key
_DEFAULTS_ALIAS: dict[str, str] = {
    "tape": "tape",
    "reel_tape": "tape",
    "tape_std": "tape",
    "tape_stu": "tape",
    "vinyl": "vinyl",
    "vinyl_std": "vinyl",
    "vinyl_78": "vinyl",
    "shellac": "shellac",
    "wax_cylinder": "shellac",
    "wax_cyl": "shellac",
    "cassette": "tape",
    "digital": "digital",
    "cd_digital": "digital",
    "dat": "digital",
    "mp3_low": "digital",
    "mp3_lossy": "digital",
    "minidisc": "digital",
    "unknown": "unknown",
}

# Alle zu generierenden Material-Keys (deckt gp_memory_key-Werte aus genre_classifier.py ab)
MATERIALS = [
    *_MATERIAL_GOAL_BASE.keys(),
    "tape_std",
    "tape_stu",
    "vinyl_std",
    "vinyl_78",
    "wax_cyl",
    "cd_digital",
    "dat",
    "mp3_lossy",
    "minidisc",
    # Genre-spezifische Keys aus genre_classifier.py
    "schlager",
    "pop",
    "jazz",
    "blues",
    "soul_rnb",
    "country",
    "folk",
    "rock",
    "orchestral",
    "opera",
    "electronic",
    "folk_world",
]

# Deduplizieren
MATERIALS = list(dict.fromkeys(MATERIALS))

# Parameter-Namen (sortiert — identisch mit _param_names_sorted())
_PARAM_NAMES = sorted(PARAMETER_SPACE.keys())
_DIM = len(_PARAM_NAMES)


def _get_defaults(material: str) -> dict[str, float]:
    """Liefert MATERIAL_DEFAULTS für einen gp_memory_key."""
    alias = _DEFAULTS_ALIAS.get(material, "unknown")
    base = dict(MATERIAL_DEFAULTS.get(alias, MATERIAL_DEFAULTS["unknown"]))
    # Fehlende Parameter auf Mittelpunkt setzen
    for name, (lo, hi, mode) in PARAMETER_SPACE.items():
        if name not in base:
            if mode == "log":
                base[name] = math.exp(0.5 * (math.log(lo + 1e-9) + math.log(hi)))
            else:
                base[name] = (lo + hi) / 2.0
    return base


def _get_goal_base(material: str) -> dict[str, float]:
    """Liefert Basis-Musical-Goals für einen Material-Key."""
    # Direkt vorhanden?
    if material in _MATERIAL_GOAL_BASE:
        return dict(_MATERIAL_GOAL_BASE[material])
    # Alias zu bekanntem Material
    alias = _DEFAULTS_ALIAS.get(material, "unknown")
    if alias in _MATERIAL_GOAL_BASE:
        return dict(_MATERIAL_GOAL_BASE[alias])
    return dict(_MATERIAL_GOAL_BASE["unknown"])


def _composite_score(goal_scores: dict[str, float]) -> float:
    """Aggregierter Score: musical_excellence-Approximation.
    Gewichtung: P1 × 0.30 + P2 × 0.25 + P3–P5 × 0.45
    """
    p1 = (goal_scores.get("natuerlichkeit", 0.0) + goal_scores.get("authentizitaet", 0.0)) / 2.0
    p2 = (
        goal_scores.get("tonal_center", 0.0)
        + goal_scores.get("timbre_authentizitaet", 0.0)
        + goal_scores.get("artikulation", 0.0)
    ) / 3.0
    _exclude = {"natuerlichkeit", "authentizitaet", "tonal_center", "timbre_authentizitaet", "artikulation"}
    rest_keys = [k for k in PARETO_OBJECTIVES if k not in _exclude]
    rest_vals = [goal_scores.get(k, 0.0) for k in rest_keys]
    p35 = float(np.mean(rest_vals)) if rest_vals else 0.0
    return round(0.30 * p1 + 0.25 * p2 + 0.45 * p35, 4)


def _generate_observations(material: str, n: int = 20, rng_seed: int = 42) -> list[dict]:
    """
    Generiert n synthetische Beobachtungen für ein Material.

    Strategie:
    - 1 Zentrum (Default-Parameter) mit gutem Score
    - 8 gute Perturbationen nahe den Defaults (±15%)
    - 6 moderate Perturbationen (±30%)
    - 3 schlechte (extreme Parameter — zeigen GP die Grenzen)
    - 2 zufällige (Exploration-Stichproben für GP-Diversität)
    """
    rng = np.random.default_rng(rng_seed)
    defaults = _get_defaults(material)
    goal_base = _get_goal_base(material)
    entries = []

    def _perturb(params: dict, scale: float, n_obs: int) -> list[dict]:
        """Erzeugt n_obs Perturbationen um params mit gegebenem Rausch-Scale."""
        results = []
        for _ in range(n_obs):
            perturbed = {}
            for pname, (lo, hi, mode) in PARAMETER_SPACE.items():
                base_val = params[pname]
                if mode == "log":
                    log_base = math.log(max(base_val, lo + 1e-9))
                    log_range = math.log(hi) - math.log(lo + 1e-9)
                    noise = rng.normal(0, scale * log_range)
                    new_val = math.exp(np.clip(log_base + noise, math.log(lo + 1e-9), math.log(hi)))
                elif mode == "int":
                    int_range = hi - lo
                    noise = rng.normal(0, scale * int_range)
                    new_val = float(np.clip(base_val + noise, lo, hi))
                    new_val = round(new_val)
                else:
                    float_range = hi - lo
                    noise = rng.normal(0, scale * float_range)
                    new_val = float(np.clip(base_val + noise, lo, hi))
                perturbed[pname] = new_val
            results.append(perturbed)
        return results

    def _entry(params: dict, score_delta: float = 0.0) -> dict:
        """Erzeugt einen Memory-Entry aus einem Parameter-Dict."""
        norm_vec = _normalize_params(params).tolist()
        # Score-Perturbation: repräsentiert Messunsicherheit
        meas_noise = float(rng.normal(0, 0.012))
        # Goal-Scores mit kleiner Perturbation
        goal_scores = {}
        for g in PARETO_OBJECTIVES:
            base = goal_base.get(g, 0.72)
            noise = float(rng.normal(0, 0.025))
            goal_scores[g] = round(float(np.clip(base + score_delta + noise, 0.0, 1.0)), 4)
        composite = _composite_score(goal_scores) + meas_noise
        return {
            "params": [round(v, 6) for v in norm_vec],
            "score": round(float(np.clip(composite, 0.0, 1.0)), 4),
            "ts": time.time() - rng.uniform(3600, 3600 * 24 * 60),  # 1h–60 Tage zurück
            "goal_scores": goal_scores,
        }

    # 1. Zentrum
    entries.append(_entry(defaults, score_delta=0.01))

    # 2–9. Gute Varianten (±15%)
    for p in _perturb(defaults, 0.10, 8):
        entries.append(_entry(p, score_delta=rng.uniform(-0.02, 0.02)))

    # 10–15. Moderate Varianten (±30%)
    for p in _perturb(defaults, 0.20, 6):
        entries.append(_entry(p, score_delta=rng.uniform(-0.05, 0.01)))

    # 16–18. Schlechte Parameter (Extremwerte — zeigen GP die Grenzen)
    bad_params_list = []
    for _ in range(3):
        bad = {}
        for pname, (lo, hi, mode) in PARAMETER_SPACE.items():
            # Extrem wenig oder extrem viel — wechselt zufällig
            if rng.random() < 0.5:
                bad[pname] = lo if mode != "int" else round(lo)
            else:
                bad[pname] = hi if mode != "int" else round(hi)
        bad_params_list.append(bad)
    for p in bad_params_list:
        entries.append(_entry(p, score_delta=rng.uniform(-0.15, -0.08)))

    # 19–20. Zufällige Exploration
    for _ in range(2):
        rand_params = {}
        for pname, (lo, hi, mode) in PARAMETER_SPACE.items():
            if mode == "log":
                rand_params[pname] = float(np.exp(rng.uniform(math.log(lo + 1e-9), math.log(hi))))
            elif mode == "int":
                rand_params[pname] = round(rng.uniform(lo, hi))
            else:
                rand_params[pname] = float(rng.uniform(lo, hi))
        entries.append(_entry(rand_params, score_delta=rng.uniform(-0.08, 0.00)))

    return entries


def generate_all(output_dir: Path = OUTPUT_DIR) -> None:
    """Generiert Memory-Dateien für alle Materialien."""
    for material in MATERIALS:
        observations = _generate_observations(material, n=20, rng_seed=hash(material) % (2**31))
        out_path = output_dir / f"{material}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(observations, f, indent=None)
        base_score = observations[0]["score"]
        print(f"  ✓ {material:20s}  n={len(observations)}  default_score={base_score:.4f}  → {out_path.name}")
    print(f"\n{len(MATERIALS)} Memory-Dateien in {output_dir}")


if __name__ == "__main__":
    print("Generiere GP Warm-Start Memory...")
    generate_all()
