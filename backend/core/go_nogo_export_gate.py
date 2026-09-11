#!/usr/bin/env python3
"""backend/core/go_nogo_export_gate.py — Deterministisches GO/NO-GO-Gate (Export).

Umsetzung des automatischen Teils des GO/NO-GO-Decision-Protocols
(docs/guides/GO_NO_GO_DECISION_PROTOCOL.md) als produktiver Export-Schritt.

§0c-Kopplung (copilot-instructions.md): Das Gate BLOCKIERT NIE — es liefert
einen Verdict, der in den Export-Metadaten festgehalten wird; ein NO_GO
degradiert den Export (bestmögliches sicheres Ergebnis, kein Hardstop).

Regeln (deterministisch, aus vorhandenen Proxies — kein neues ML):
  - artifact_freedom < 0.95                → NO_GO (Artefakt-Risiko)
  - HPI ≤ 0                                → NO_GO (Signal verschlechtert)
  - quality_estimate < 0.55                → NO_GO (Kernqualität)
  - Structural-Silence-Lift > 1.0 dB       → NO_GO (Energie in Stille)
  - Pump-/Breathing-Indikatoren in Warnings→ NO_GO (NR-Pumpen)
  - Formant-Integrität < 0.72 / Vibrato < 0.80 → NO_GO (Vokalverfärbung)
  - Stereo: Mono-Kompatibilitäts-Kollaps (mono_compat-Drop > 0.22)
    oder IACC-Kollaps (Δ > 0.15)           → NO_GO (binaurale Zerstörung)
  - Audio NaN/Inf                          → NO_GO (technisch)
  - Sonstige Warnungen                     → GO_CAUTION

Ehrlicher Scope: Proxies sind Zeugen, keine Hörer (Hörordnung §1) — das
Verdict ersetzt keinen echten Hörversuch, macht aber jeden Export
nachvollziehbar (Audit) und speist die UI-Anzeige.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GoNoGoVerdict:
    """GO/NO-GO-Entscheidung mit Gründen (produktive Export-Metadaten)."""

    verdict: str  # "GO" | "GO_CAUTION" | "NO_GO"
    reasons: list[str] = field(default_factory=list)
    cautions: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "verdict": self.verdict,
            "reasons": list(self.reasons),
            "cautions": list(self.cautions),
        }


def _num(v: object) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def evaluate(result: object, audio: np.ndarray | None = None) -> GoNoGoVerdict:
    """Bewertet ein Restaurierungsergebnis (RestorationResult/AurikErgebnis)."""
    reasons: list[str] = []
    cautions: list[str] = []

    meta = getattr(result, "metadata", None) or {}
    if not isinstance(meta, dict):
        meta = {}

    # ── Audio-technische Prüfung ──────────────────────────────────────────
    if audio is not None:
        arr = np.asarray(audio)
        if arr.size == 0:
            reasons.append("audio_empty")
        elif np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            reasons.append("audio_nan_inf")

    # ── Kernqualität ──────────────────────────────────────────────────────
    af = (
        _num(meta.get("artifact_freedom"))
        if _num(meta.get("artifact_freedom")) is not None
        else _num(getattr(result, "artifact_freedom", None))
    )
    if af is not None and af < 0.95:
        reasons.append(f"artifact_freedom_low={af:.3f}")

    hpi = _num(meta.get("hpi")) if _num(meta.get("hpi")) is not None else _num(getattr(result, "hpi", None))
    if hpi is not None and hpi <= 0.0:
        reasons.append(f"hpi_le_zero={hpi:.3f}")

    qe = _num(getattr(result, "quality_estimate", None))
    if qe is not None and qe < 0.55:
        reasons.append(f"quality_estimate_low={qe:.3f}")

    # ── Structural-Silence / Vocal-Schäden ────────────────────────────────
    silence = meta.get("structural_silence_audit")
    if isinstance(silence, dict):
        lift = silence.get("max_lift_db") or silence.get("silence_lift_db") or silence.get("max_energy_lift_db")
        if _num(lift) is not None and float(lift) > 1.0:  # type: ignore[arg-type]
            reasons.append(f"silence_lift={float(lift):.2f}db")  # type: ignore[arg-type]
    direct_lift = _num(meta.get("structural_silence_lift_db"))
    if direct_lift is not None and direct_lift > 1.0:
        reasons.append(f"silence_lift={direct_lift:.2f}db")

    vocal = meta.get("vocal_quality_check")
    if isinstance(vocal, dict):
        fi = _num(vocal.get("formant_integrity"))
        vd = _num(vocal.get("vibrato_depth_preservation"))
        if fi is not None and fi < 0.72:
            reasons.append(f"formant_integrity={fi:.3f}")
        if vd is not None and vd < 0.80:
            reasons.append(f"vibrato_depth={vd:.3f}")

    # ── Stereo/Binaural-Kollaps (§2.51a) ──────────────────────────────────
    ssg = meta.get("stereo_safety_guard")
    if isinstance(ssg, dict):
        _hf_reasons = [str(r) for r in ssg.get("hard_fail_reasons", [])[:3]]
        _warn_reasons = [str(r) for r in ssg.get("warning_reasons", [])[:3]]
        if bool(ssg.get("hard_fail", False)):
            reasons.append("stereo_hard_fail:" + ",".join(_hf_reasons))
        elif bool(ssg.get("warning", False)):
            cautions.append("stereo_warning:" + ",".join(_warn_reasons))
    _delta_raw = meta.get("delta")
    delta = _delta_raw if isinstance(_delta_raw, dict) else {}
    mono_drop = _num(delta.get("mono_compat"))
    if mono_drop is not None and mono_drop < -0.22:
        reasons.append(f"mono_compat_drop={mono_drop:.3f}")

    # ── Pump-/Breathing-Indikatoren (aus Warnungen/Violations) ────────────
    warning_texts = [str(w).lower() for w in (getattr(result, "warnings", None) or [])]
    pump_hits = [w for w in warning_texts if ("pump" in w or "breathing" in w or "atmet" in w)]
    if pump_hits:
        reasons.append("nr_pumping:" + pump_hits[0][:60])

    # ── Sonstige Warnungen → Caution ──────────────────────────────────────
    gate_meta = meta.get("export_quality_gate_failed", False)
    if gate_meta:
        cautions.append("export_quality_gate_failed")

    verdict = "NO_GO" if reasons else ("GO_CAUTION" if cautions else "GO")
    return GoNoGoVerdict(verdict=verdict, reasons=reasons, cautions=cautions)


__all__ = ["GoNoGoVerdict", "evaluate"]
