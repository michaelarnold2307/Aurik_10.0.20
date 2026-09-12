"""H3: Witness-Veto → global_scalar/family_scalars Rückkopplung (§V7 (copilot-instructions.md), Hörordnung §8a).

Der Listening-Witness ist Zeuge, die Hör-Instanz entscheidet: Überschreitet
ein delta-basierter Hörbefund die JND-Schwelle (listening_witness.py),
wird die Stärke der VERANTWORTLICHEN Phase-Familie sofort reduziert —
nicht erst durch einen nachträglichen Re-Run (todo t11, „sofortige
Anpassung"). Alle Regeln sind delta-basiert (Regression vs. Referenz) und
klemmen auf [0.50, 1.50] bzw. Familie [0.5, 1.5] (Lücke-G-Fix v10.0.0).

Deterministisch, keine Nebenwirkungen außer der Profil-Mutation + Log.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# JND-Schwellen müssen den Konstanten in backend/core/listening_witness.py
# entsprechen (dort: _PITCH_DRIFT_CENTS=15, _HNR_DROP_DB=2.0, _HF_FLATNESS_RISE=0.15,
# _TRANSIENT_SMEAR_RATIO=0.35, _BASS_DROP_DB=1.5, _ROUGHNESS_RISE_ASPER=0.35,
# _PRE_ECHO_DB=-12.0, _AIR_LOSS_DB=2.0). ITD/IACC: konservative Veto-Grenzen.
_ITD_DRIFT_US = 100.0
_IACC_DROP = 0.15

# (attr, Schwelle, Familie, Faktor, Begründung)
_VETO_RULES: tuple[tuple[str, float, str, float, str], ...] = (
    ("roughness_rise_asper", 0.35, "denoise", 0.90, "Rauigkeits-Anstieg (Musical Noise)"),
    ("hf_flatness_rise", 0.15, "denoise", 0.95, "HF-Flatness-Anstieg (Spektral-Artefakte)"),
    ("transient_smear_ratio", 0.35, "denoise", 0.95, "Transienten-Verschmierung (Over-Denoising)"),
    ("bass_drop_db", 1.5, "denoise", 0.95, "Bass-Energieverlust"),
    ("hnr_drop_db", 2.0, "enhancement", 0.90, "Stimm-HNR-Abfall"),
    ("pitch_drift_cents", 15.0, "enhancement", 0.90, "Pitch-Drift"),
    ("pre_echo_db", -12.0, "repair", 0.90, "Pre-Echo (zu aggressive Reparatur)"),
)

_MIN_GLOBAL = 0.50
_MAX_GLOBAL = 1.50
_MIN_FAMILY = 0.5
_MAX_FAMILY = 1.5


@dataclass
class WitnessVetoResult:
    """Ergebnis einer Witness-Veto-Bewertung (deterministisch)."""

    stage: str = ""
    adjustments: dict[str, float] = field(default_factory=dict)
    global_factor: float = 1.0
    reasons: list[str] = field(default_factory=list)

    @property
    def applied(self) -> bool:
        return bool(self.adjustments) or self.global_factor < 1.0


def compute_witness_veto(witness: Any, stage: str = "") -> WitnessVetoResult:
    """Bewertet einen ListeningWitnessResult und liefert Familie-Anpassungen.

    Reine Funktion — mutiert nichts. Delta-basiert: nur Überschreitungen
    der JND-Schwellen zählen (Regression gegenüber der Referenz).
    """
    res = WitnessVetoResult(stage=stage)
    for attr, thr, family, factor, reason in _VETO_RULES:
        try:
            value = float(getattr(witness, attr, 0.0))
        except (TypeError, ValueError):
            value = 0.0
        if attr == "pre_echo_db":
            # Pre-Echo-Schwelle: Befunde liegen in (−12, 0) dB; −200.0 ist der
            # „kein Befund"-Sentinel (pre_echo_model) und 0.0 der „nicht
            # berechnet"-Default — beide lösen KEIN Veto aus.
            if -199.0 < value < 0.0 and value > thr:
                res.adjustments.setdefault(family, 1.0)
                res.adjustments[family] = round(res.adjustments[family] * factor, 4)
                res.reasons.append(f"{reason} ({attr}={value:.2f} > {thr:.2f})")
        else:
            if value > thr:
                res.adjustments.setdefault(family, 1.0)
                res.adjustments[family] = round(res.adjustments[family] * factor, 4)
                res.reasons.append(f"{reason} ({attr}={value:.2f} > {thr:.2f})")

    # Stereo-Kollaps: ITD-Drift oder IACC-Einbruch → räumlich wirkende Familien.
    try:
        itd = float(getattr(witness, "itd_drift_us", 0.0))
        iacc = float(getattr(witness, "iacc_drop", 0.0))
        findings = list(getattr(witness, "findings", []) or [])
    except (TypeError, ValueError):
        itd, iacc, findings = 0.0, 0.0, []
    if itd > _ITD_DRIFT_US or iacc > _IACC_DROP or any("stereo" in str(f).lower() for f in findings):
        for _fam in ("enhancement", "compression"):
            res.adjustments.setdefault(_fam, 1.0)
            res.adjustments[_fam] = round(res.adjustments[_fam] * 0.95, 4)
        res.reasons.append(f"Stereo-Kollaps (itd={itd:.1f} µs, iacc_drop={iacc:.3f})")

    if res.adjustments:
        res.global_factor = round(min(res.adjustments.values()), 4)
    return res


def apply_witness_veto(profile: Any, witness: Any, stage: str = "") -> WitnessVetoResult:
    """Wendet ein Witness-Veto auf ein Kalibrierungsprofil an (Mutation).

    ``profile`` ist dict-artig mit den Schlüsseln ``global_scalar`` und
    ``family_scalars`` (unified_restorer_v3._song_calibration_profile) oder
    ein SongCalibrationProfile-Dataclass. Clamping [0.50, 1.50] /
    Familie [0.5, 1.5] — Lücke-G-Fix v10.0.0.
    """
    res = compute_witness_veto(witness, stage)
    if not res.applied:
        return res

    fams = (
        profile.get("family_scalars", None) if isinstance(profile, dict) else getattr(profile, "family_scalars", None)
    )
    if isinstance(fams, dict):
        for family, factor in res.adjustments.items():
            cur = float(fams.get(family, 1.0))
            fams[family] = round(float(np_clip(cur * factor, _MIN_FAMILY, _MAX_FAMILY)), 3)
    gs_cur = (
        float(profile.get("global_scalar", 1.0))
        if isinstance(profile, dict)
        else float(getattr(profile, "global_scalar", 1.0))
    )
    gs_new = round(float(np_clip(gs_cur * res.global_factor, _MIN_GLOBAL, _MAX_GLOBAL)), 3)
    if isinstance(profile, dict):
        profile["global_scalar"] = gs_new
    else:
        profile.global_scalar = gs_new

    logger.warning(
        "§Witness-Veto (%s): %s → global_scalar %.3f→%.3f, Familien %s — "
        "Restaurierungs-Stärke SOFORT reduziert (delta-basiert, §V7 (copilot-instructions.md))",
        stage,
        "; ".join(res.reasons),
        gs_cur,
        gs_new,
        res.adjustments,
    )
    return res


def np_clip(x: float, lo: float, hi: float) -> float:
    """Clip ohne numpy-Import-Zwang (Tests ohne schweren Import)."""
    return lo if x < lo else hi if x > hi else x
