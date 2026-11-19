"""Phase-spezifisches Strength-Orakel fuer UV3.

Dieses Modul liefert eine zentrale, phasenbewusste Steuerprofil-Berechnung,
die den lokalen Eingriff gegen den 15-Ziele-Teamvektor ausrichtet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

_ORACLE_CLASS_BY_FAMILY: dict[str, str] = {
    "general": "O1_general_repair",
    "subtractive_cleanup": "O2_subtractive",
    "harmonic_noise_control": "O9_periodic_cancellation",
    "tonal_restoration": "O3_spectral_balance",
    "spectral_restoration": "O8_generative_repair",
    "harmonic_reconstruction": "O3_spectral_balance",
    "transient_shaping": "O6_dynamics",
    "dynamics_repair": "O6_dynamics",
    "time_pitch_transport": "O4_time_pitch",
    "stereo_phase_geometry": "O5_stereo_field",
    "stereo_enhancement": "O5_stereo_field",
    "stereo_generation": "O5_stereo_field",
    "tonal_mastering": "O3_spectral_balance",
    "harmonic_enhancement": "O3_spectral_balance",
    "dynamics_control": "O6_dynamics",
    "tonal_enhancement": "O3_spectral_balance",
    "finalizer_output": "O10_output",
    "source_enhancement": "O3_spectral_balance",
    "semantic_guidance": "O10_output",
    "reconstruction_inpainting": "O8_generative_repair",
    "sibilance_control": "O7_vocal_articulation",
    "distortion_repair": "O9_periodic_cancellation",
}


_ORACLE_CLASS_POLICY: dict[str, dict[str, float]] = {
    "O1_general_repair": {"wet_base": 0.42, "wet_span": 0.30, "max_strength": 0.86, "max_strength_hi_rest": 0.78},
    "O2_subtractive": {"wet_base": 0.50, "wet_span": 0.42, "max_strength": 0.92, "max_strength_hi_rest": 0.84},
    "O3_spectral_balance": {
        "wet_base": 0.45,
        "wet_span": 0.38,
        "max_strength": 0.88,
        "max_strength_hi_rest": 0.80,
    },
    "O4_time_pitch": {"wet_base": 0.40, "wet_span": 0.32, "max_strength": 0.82, "max_strength_hi_rest": 0.72},
    "O5_stereo_field": {"wet_base": 0.38, "wet_span": 0.30, "max_strength": 0.78, "max_strength_hi_rest": 0.70},
    "O6_dynamics": {"wet_base": 0.44, "wet_span": 0.34, "max_strength": 0.84, "max_strength_hi_rest": 0.76},
    "O7_vocal_articulation": {
        "wet_base": 0.52,
        "wet_span": 0.40,
        "max_strength": 0.90,
        "max_strength_hi_rest": 0.82,
    },
    "O8_generative_repair": {
        "wet_base": 0.36,
        "wet_span": 0.30,
        "max_strength": 0.78,
        "max_strength_hi_rest": 0.68,
    },
    "O9_periodic_cancellation": {
        "wet_base": 0.46,
        "wet_span": 0.34,
        "max_strength": 0.82,
        "max_strength_hi_rest": 0.74,
    },
    "O10_output": {"wet_base": 0.34, "wet_span": 0.24, "max_strength": 0.72, "max_strength_hi_rest": 0.66},
}


_ORACLE_CLASS_DRIVER_GAIN: dict[str, float] = {
    "O1_general_repair": 0.78,
    "O2_subtractive": 1.00,
    "O3_spectral_balance": 0.94,
    "O4_time_pitch": 0.86,
    "O5_stereo_field": 0.80,
    "O6_dynamics": 0.88,
    "O7_vocal_articulation": 1.00,
    "O8_generative_repair": 0.74,
    "O9_periodic_cancellation": 0.92,
    "O10_output": 0.70,
}


_CHAIN_STAGE_FACTOR: dict[str, float] = {
    "wax_cylinder": 0.74,
    "shellac": 0.76,
    "wire_recording": 0.78,
    "vinyl": 0.86,
    "vinyl_78rpm": 0.84,
    "tape": 0.82,
    "reel_tape": 0.84,
    "cassette": 0.80,
    "mp3_low": 0.79,
    "aac": 0.83,
    "streaming": 0.86,
    "mp3_high": 0.90,
    "cd_digital": 0.96,
    "dat": 0.94,
    "unknown_analog": 0.82,
    "unknown": 0.88,
}


@dataclass(frozen=True)
class PhaseStrengthOracleProfile:
    """Normiertes Steuerprofil fuer eine Phase."""

    oracle_class: str
    control_strength: float
    wet_mix: float
    threshold_db: float | None
    ratio: float | None
    drive: float | None
    eq_gain_db: float | None
    band_profile: dict[str, float] | None
    team_contribution: dict[str, float]
    dominant_goal_guard: bool
    hard_caps: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Serialisiert das Profil in ein JSON-kompatibles Dictionary."""
        return {
            "oracle_class": self.oracle_class,
            "control_strength": self.control_strength,
            "wet_mix": self.wet_mix,
            "threshold_db": self.threshold_db,
            "ratio": self.ratio,
            "drive": self.drive,
            "eq_gain_db": self.eq_gain_db,
            "band_profile": dict(self.band_profile) if isinstance(self.band_profile, dict) else None,
            "team_contribution": dict(self.team_contribution),
            "dominant_goal_guard": self.dominant_goal_guard,
            "hard_caps": dict(self.hard_caps),
        }


def _extract_max_severity(defect_scores: object) -> float:
    if not isinstance(defect_scores, dict) or not defect_scores:
        return 0.0
    m = 0.0
    for _, val in defect_scores.items():
        sev = float(getattr(val, "severity", val) or 0.0)
        m = max(m, sev)
    return float(np.clip(m, 0.0, 1.0))


def _normalize_goal_contribution(
    goal_gaps: dict[str, float],
    goal_weights: dict[str, float] | None,
) -> dict[str, float]:
    if not goal_gaps:
        return {}
    weights = goal_weights if isinstance(goal_weights, dict) else {}
    raw: dict[str, float] = {}
    total = 0.0
    for goal, gap in goal_gaps.items():
        gap_v = float(max(0.0, gap))
        w = float(np.clip(float(weights.get(goal, 1.0)), 0.3, 2.0))
        v = gap_v * w
        if v <= 0.0:
            continue
        raw[goal] = v
        total += v
    if total <= 1e-9:
        return {}
    return {k: float(np.clip(v / total, 0.0, 1.0)) for k, v in raw.items()}


def _normalize_chain(material_key: str, transfer_chain: list[str] | None) -> list[str]:
    mats: list[str] = []
    if isinstance(transfer_chain, list):
        for stage in transfer_chain:
            s = str(stage or "").strip().lower()
            if s and s not in mats:
                mats.append(s)
    m0 = str(material_key or "").strip().lower()
    if m0 and m0 not in mats:
        mats.insert(0, m0)
    if not mats:
        mats.append("unknown")
    return mats


def _compute_chain_factor(
    material_key: str,
    transfer_chain: list[str] | None,
    chain_confidence: float | None,
) -> tuple[float, list[str]]:
    chain_mats = _normalize_chain(material_key, transfer_chain)
    stage_factors = [float(_CHAIN_STAGE_FACTOR.get(m, 0.88)) for m in chain_mats]
    strict_factor = float(np.clip(min(stage_factors) if stage_factors else 0.88, 0.60, 1.0))
    generation_penalty = float(np.clip(0.02 * max(0, len(chain_mats) - 1), 0.0, 0.12))
    strict_factor = float(np.clip(strict_factor - generation_penalty, 0.55, 1.0))

    conf = 0.65 if chain_confidence is None else float(np.clip(chain_confidence, 0.0, 1.0))
    conf_weight = float(np.clip((conf - 0.40) / 0.60, 0.0, 1.0))
    chain_factor = float(np.clip((1.0 - conf_weight) * 0.95 + conf_weight * strict_factor, 0.55, 1.0))
    return chain_factor, chain_mats


def _as_guard_scalar(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(np.clip(float(value), 0.0, 1.0))
    return float(np.clip(default, 0.0, 1.0))


def _compute_voice_guard_risk(panns_singing: float, vocal_guard_metrics: dict[str, Any] | None) -> float:
    """Berechnet ein konservatives Vocal-No-Harm-Risiko fuer die Staerkedrosselung."""
    singing = float(np.clip(panns_singing, 0.0, 1.0))
    if singing < 0.35:
        return 0.0

    metrics = vocal_guard_metrics if isinstance(vocal_guard_metrics, dict) else {}
    vqi = _as_guard_scalar(metrics.get("vqi", metrics.get("vqi_score")), 1.0)
    formant = _as_guard_scalar(
        metrics.get("formant_integrity", metrics.get("formant_fidelity")),
        1.0,
    )
    vibrato = _as_guard_scalar(
        metrics.get("vibrato_depth_preserved", metrics.get("vibrato_ok")),
        1.0,
    )
    micro = _as_guard_scalar(
        metrics.get("micro_dynamic_correlation", metrics.get("micro_dynamics_corr")),
        1.0,
    )

    risk = 0.0
    if vqi < 0.87:
        risk += 0.45 * float(np.clip((0.87 - vqi) / 0.35, 0.0, 1.0))
    if formant < 0.93:
        risk += 0.25 * float(np.clip((0.93 - formant) / 0.25, 0.0, 1.0))
    if vibrato < 0.90:
        risk += 0.20 * float(np.clip((0.90 - vibrato) / 0.30, 0.0, 1.0))
    if micro < 0.97:
        risk += 0.20 * float(np.clip((0.97 - micro) / 0.35, 0.0, 1.0))

    # Bei unsicherer Vocal-Metrik konservativ bleiben, aber nicht blockieren.
    risk *= float(0.60 + 0.40 * singing)
    return float(np.clip(risk, 0.0, 1.0))


# ---------------------------------------------------------------------------
# §0l Phase-ID-spezifische Strength-Override-Tabelle (Lücke-A-Fix v10.0.0)
# ---------------------------------------------------------------------------
# Die Family-basierte Orakel-Klasse (O1–O10) gibt grobe Grundparameter vor.
# Für Phasen mit stark unterschiedlichem physikalischem Eingriff innerhalb
# derselben Familie überschreibt diese Tabelle die Policy-Parameter.
#
# Erlaubte Override-Schlüssel:
#   oracle_class_override  → Erzwingt andere Orakel-Klasse (z. B. O7 für Vokal-Phasen)
#   max_strength_cap       → Absoluter Hard-Cap unabhängig von Restorability
#   threshold_db_min       → Obergrenze für threshold_db (weniger tief = konservativer)
#   eq_gain_db_cap         → Hard-Cap für eq_gain_db (additive Phasen)
#   drive_cap              → Hard-Cap für drive (generative/harmonische Phasen)
#   driver_gain_override   → Überschreibt _ORACLE_CLASS_DRIVER_GAIN für diese Phase
#
# Physikalische Begründung pro Gruppe:
#   Subtraktive NR-Phasen (phase_03/29/49/20): Carrier-Rauschboden muss erhalten
#     bleiben (V19/V21). threshold_db darf nicht zu tief; max_strength konservativ.
#   Additive BW-Phasen (phase_06/07/23): Hallucination-Guard (§2.46e) begrenzt.
#     eq_gain_db und drive konservativ gedeckelt.
#   Vokal-Phasen (phase_42/65): erzwingen O7-Klasse für korrekte Vokal-Parametrisierung.
#   Zeit/Pitch-Phasen (phase_12): mechanischer Defekt → aggressiver als generisches O4.
_PHASE_ID_STRENGTH_OVERRIDES: dict[str, dict[str, Any]] = {
    # --- Subtraktive Carrier-NR-Phasen ---
    "phase_29_tape_hiss_reduction": {
        # Tape-Hiss-Entfernung: Carrier-Chain-Inversion (§2.46 Stufe 4).
        # Coalition mit phase_07 (tape_transport, §2.67) — nicht zu aggressiv.
        "max_strength_cap": 0.78,
        "threshold_db_min": -28.0,  # V21: nie tiefer als −28 dB
        "driver_gain_override": 0.82,
    },
    "phase_03_denoise": {
        # Breitband-NR: breites Spektrum → konservativ; phase_06 folgt in Coalition.
        "max_strength_cap": 0.72,
        "threshold_db_min": -26.0,
        "driver_gain_override": 0.78,
    },
    "phase_49_advanced_dereverb": {
        # Hallentfernung: irreversibel, spatial_depth-Verlust intentional (V19).
        "max_strength_cap": 0.62,
        "threshold_db_min": -22.0,
        "driver_gain_override": 0.70,
    },
    "phase_02_hum_removal": {
        # Schmalbandige Notch-Operation: präzise, nicht zu breit.
        "max_strength_cap": 0.84,
        "driver_gain_override": 0.88,
    },
    "phase_20_reverb_reduction": {
        # Leichter als phase_49 (einfachere Reverb-Typen).
        "max_strength_cap": 0.70,
        "threshold_db_min": -24.0,
        "driver_gain_override": 0.74,
    },
    # --- Additive / BW-Erweiterungs-Phasen ---
    "phase_07_harmonic_restoration": {
        # Komplementärphase zu phase_29 (Coalition tape_transport, §2.67).
        # Muss Hallucination-Guard-konform bleiben (§2.46e).
        "max_strength_cap": 0.72,
        "eq_gain_db_cap": 2.0,
        "drive_cap": 0.62,
        "driver_gain_override": 0.86,
    },
    "phase_06_frequency_restoration": {
        # BW-Erweiterung: Material-BW-Ceiling (§6.2b) entscheidend.
        # Coalition broadband_nr_restoration mit phase_03.
        "max_strength_cap": 0.68,
        "eq_gain_db_cap": 1.8,
        "drive_cap": 0.58,
        "driver_gain_override": 0.80,
    },
    "phase_23_spectral_repair": {
        # Inpainting/ADMM: generativ, SSIP-geschützt (§2.68), konservativ.
        # Kein max_strength_cap hier: O8-Formel liefert bereits chain-aware Dynamic-Cap
        # (0.78 × (0.75 + 0.25 × chain_factor) ≤ 0.78) — statischer Override wäre redundant.
        "eq_gain_db_cap": 1.6,
        "drive_cap": 0.52,
        "driver_gain_override": 0.72,
    },
    # --- Vokal-spezifische Phasen ---
    "phase_42_vocal_enhancement": {
        # Studio 2026 only (§0a). VQI-Gate steuert; O7-Klasse für korrekte Parametrisierung.
        "oracle_class_override": "O7_vocal_articulation",
        "max_strength_cap": 0.76,
        "driver_gain_override": 0.90,
    },
    "phase_65_vocal_naturalness_restoration": {
        # §0a-konformer Vokal-Recovery in Restoration (HNR-Blend + Formant-Tilt).
        "oracle_class_override": "O7_vocal_articulation",
        "max_strength_cap": 0.60,
        "driver_gain_override": 0.72,
    },
    # --- Zeit/Pitch-Phasen ---
    "phase_12_wow_flutter_fix": {
        # Mechanischer Defekt: klarere physikalische Zielfunktion → aggressiver als O4.
        "max_strength_cap": 0.88,
        "drive_cap": 0.82,
        "driver_gain_override": 0.92,
    },
    "phase_25_azimuth_correction": {
        # Geometrische Korrektur: deterministisch, mittlere Stärke.
        "max_strength_cap": 0.82,
        "driver_gain_override": 0.84,
    },
}


def resolve_phase_strength_oracle(
    *,
    phase_id: str,
    phase_family: str,
    current_strength: float,
    goal_gaps: dict[str, float] | None,
    goal_weights: dict[str, float] | None,
    defect_scores: object,
    locality_factor: float,
    restorability_score: float,
    material_key: str,
    song_calibration_profile: dict[str, Any] | None,
    panns_singing: float,
    vocal_guard_metrics: dict[str, Any] | None = None,
    transfer_chain: list[str] | None = None,
    chain_confidence: float | None = None,
) -> PhaseStrengthOracleProfile:
    """Berechnet ein robustes, bounds-sicheres Steuerprofil je Phase."""
    _ = song_calibration_profile  # §0l: phase_id und panns_singing aktiv genutzt (Lücke-A-Fix)
    # §0l Lücke-A-Fix: Phase-ID-spezifische Overrides laden.
    _ph_override: dict[str, Any] = _PHASE_ID_STRENGTH_OVERRIDES.get(str(phase_id or "").strip(), {})
    oracle_class = _ORACLE_CLASS_BY_FAMILY.get(str(phase_family or "").strip().lower(), "O1_general_repair")
    # oracle_class_override: Erzwingt andere Klasse (z. B. phase_42 → O7_vocal_articulation).
    if "oracle_class_override" in _ph_override:
        oracle_class = str(_ph_override["oracle_class_override"])
    policy = _ORACLE_CLASS_POLICY.get(oracle_class, _ORACLE_CLASS_POLICY["O1_general_repair"])
    base = float(np.clip(current_strength, 0.0, 1.0))
    sev = _extract_max_severity(defect_scores)
    loc = float(np.clip(locality_factor, 0.35, 1.0))
    rest = float(np.clip(restorability_score, 0.0, 100.0))

    goal_gaps_map = goal_gaps if isinstance(goal_gaps, dict) else {}
    contrib = _normalize_goal_contribution(goal_gaps_map, goal_weights)
    weighted_gap = float(sum(float(v) for v in contrib.values()))
    chain_factor, chain_mats = _compute_chain_factor(material_key, transfer_chain, chain_confidence)
    voice_guard_risk = _compute_voice_guard_risk(panns_singing, vocal_guard_metrics)

    # Teamwork-orientierter Eingriffsfaktor: Defektlast + Goal-Luecken treiben hoch,
    # hohe Restorability bremst (Minimal-Intervention).
    # §0l driver_gain_override: Phasenspezifisches Antriebsniveau (Lücke-A-Fix).
    _driver_gain = float(_ph_override.get("driver_gain_override", _ORACLE_CLASS_DRIVER_GAIN.get(oracle_class, 0.80)))
    driver = float(
        np.clip(
            (0.58 * sev + 0.32 * weighted_gap + 0.10 * (1.0 - rest / 100.0)) * _driver_gain * chain_factor,
            0.0,
            1.0,
        )
    )
    control_strength = float(np.clip(max(base, 0.25 * base + 0.90 * driver) * loc, 0.0, 1.0))
    wet_mix = float(
        np.clip(
            (float(policy["wet_base"]) + float(policy["wet_span"]) * control_strength) * (0.88 + 0.12 * chain_factor),
            0.30,
            0.98,
        )
    )

    if voice_guard_risk > 0.0:
        # §0p Voice-First No-Harm: bei vokalem Risiko aktiv konservativer fahren.
        _voice_damp = float(np.clip(1.0 - 0.45 * voice_guard_risk, 0.55, 1.0))
        control_strength = float(np.clip(control_strength * _voice_damp, 0.0, 1.0))
        wet_mix = float(np.clip(wet_mix * (0.70 + 0.30 * _voice_damp), 0.30, 0.98))

    threshold_db = None
    ratio = None
    drive = None
    eq_gain_db = None
    band_profile = None

    if oracle_class == "O1_general_repair":
        ratio = float(np.clip(1.0 + 1.4 * control_strength, 1.0, 2.6))
    elif oracle_class == "O2_subtractive":
        threshold_db = float(np.clip(-18.0 - 16.0 * control_strength, -36.0, -18.0))
        # §0l threshold_db_min: Phase-ID-spezifische Obergrenze (weniger tief = konservativer, V21-Boden).
        if "threshold_db_min" in _ph_override:
            threshold_db = float(max(threshold_db, float(_ph_override["threshold_db_min"])))
        ratio = float(np.clip(1.0 + 2.2 * control_strength, 1.0, 3.4))
    elif oracle_class == "O7_vocal_articulation":
        threshold_db = float(np.clip(-16.0 - 14.0 * control_strength, -34.0, -14.0))
        ratio = float(np.clip(1.2 + 3.6 * control_strength, 1.2, 4.8))
    elif oracle_class == "O4_time_pitch":
        drive = float(np.clip(0.15 + 0.70 * control_strength, 0.0, 0.90))
    elif oracle_class == "O5_stereo_field":
        band_profile = {
            "low": float(np.clip(0.82 + 0.22 * control_strength, 0.75, 1.05)),
            "mid": float(np.clip(0.92 + 0.26 * control_strength, 0.85, 1.20)),
            "high": float(np.clip(0.96 + 0.30 * control_strength, 0.90, 1.26)),
        }
    elif oracle_class in {"O6_dynamics", "O10_output"}:
        ratio = float(np.clip(1.0 + 2.8 * control_strength, 1.0, 4.2))
    elif oracle_class == "O9_periodic_cancellation":
        threshold_db = float(np.clip(-14.0 - 12.0 * control_strength, -30.0, -14.0))
        drive = float(np.clip(0.10 + 0.70 * control_strength, 0.0, 0.85))
    elif oracle_class in {"O3_spectral_balance", "O8_generative_repair"}:
        _eq_cap = 3.4 if oracle_class == "O3_spectral_balance" else 2.6
        _drive_cap = 1.0 if oracle_class == "O3_spectral_balance" else 0.78
        eq_gain_db = float(np.clip(0.4 + 2.8 * control_strength, 0.0, _eq_cap))
        drive = float(np.clip(0.2 + 0.9 * control_strength, 0.0, _drive_cap))
        band_profile = {
            "low": float(np.clip(0.90 + 0.30 * control_strength, 0.80, 1.20)),
            "mid": float(np.clip(0.90 + 0.40 * control_strength, 0.80, 1.30)),
            "high": float(np.clip(0.85 + 0.55 * control_strength, 0.70, 1.35)),
        }

    # §0l Phase-ID Hard-Caps für additive Parameter (Lücke-A-Fix).
    if eq_gain_db is not None and "eq_gain_db_cap" in _ph_override:
        eq_gain_db = float(min(eq_gain_db, float(_ph_override["eq_gain_db_cap"])))
    if drive is not None and "drive_cap" in _ph_override:
        drive = float(min(drive, float(_ph_override["drive_cap"])))

    hard_caps = {
        "max_strength": float(
            np.clip(
                float(policy["max_strength_hi_rest"] if rest >= 75.0 else policy["max_strength"])
                * (0.75 + 0.25 * chain_factor),
                0.0,
                1.0,
            )
        ),
        "max_wet_mix": 0.98,
        "chain_factor": chain_factor,
        "chain_depth": float(len(chain_mats)),
        "chain_confidence": float(0.65 if chain_confidence is None else np.clip(chain_confidence, 0.0, 1.0)),
        "voice_guard_risk": voice_guard_risk,
    }
    control_strength = float(min(control_strength, hard_caps["max_strength"]))
    wet_mix = float(min(wet_mix, hard_caps["max_wet_mix"]))
    # §0l max_strength_cap: Phase-ID-spezifischer absoluter Hard-Cap (Lücke-A-Fix).
    if "max_strength_cap" in _ph_override:
        _ph_max_str = float(np.clip(float(_ph_override["max_strength_cap"]), 0.0, 1.0))
        control_strength = float(min(control_strength, _ph_max_str))
        hard_caps["max_strength"] = float(min(hard_caps["max_strength"], _ph_max_str))

    # Dominanz-Guard: kein einzelnes Ziel darf >65% Teamanteil tragen.
    dominant_goal_guard = True
    if contrib:
        dominant_goal_guard = float(max(contrib.values())) <= 0.65
        if not dominant_goal_guard:
            # Teamwork statt Einzelziel-Dominanz: konservative Daempfung.
            control_strength = float(np.clip(control_strength * 0.92, 0.0, hard_caps["max_strength"]))
            wet_mix = float(np.clip(wet_mix * 0.95, 0.30, hard_caps["max_wet_mix"]))

    return PhaseStrengthOracleProfile(
        oracle_class=oracle_class,
        control_strength=control_strength,
        wet_mix=wet_mix,
        threshold_db=threshold_db,
        ratio=ratio,
        drive=drive,
        eq_gain_db=eq_gain_db,
        band_profile=band_profile,
        team_contribution=contrib,
        dominant_goal_guard=bool(dominant_goal_guard),
        hard_caps=hard_caps,
    )
