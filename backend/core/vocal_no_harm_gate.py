"""Vocal no-harm gate for post-processing safety decisions.

This module combines the existing vocal safety signals into one small,
serializable decision object. It is intentionally non-blocking: unavailable
measurement modules add warnings, but measured violations create a rollback
recommendation for vocal material.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)

_ARTIFACT_FREEDOM_FLOOR = 0.95
_HNR_DELTA_MAX_DB = 3.0
_SINGER_IDENTITY_FLOOR = 0.92
_FORMANT_SHIFT_MAX_DB = 2.0
_BREATH_ATTENUATION_MAX_DB = 6.0
_BREATH_EMOTIONAL_ATTENUATION_MAX_DB = 3.0
_BREATH_MIN_RMS_DB = -70.0
_VOCAL_GATE_PANNS_FLOOR = 0.35
_VOCAL_MAX_TARGET_RESTORATION = 0.88

# §DENKER: Material-spezifische Vocal-No-Harm-Toleranzen
# Cassette/Band: höhere Toleranz (0.55) weil Defekte und Gesang untrennbar verwoben sind
# Vinyl/Shellac: mittlere Toleranz (0.45) wegen Oberflächengeräuschen
# Digital: strenge Toleranz (0.35) — saubere Quelle erwartet
_VOCAL_GATE_MATERIAL_FLOOR: dict[str, float] = {
    "cassette": 0.55,
    "tape": 0.55,
    "reel_tape": 0.50,
    "vinyl": 0.45,
    "shellac": 0.45,
    "wax_cylinder": 0.50,
    "wire_recording": 0.50,
}
_VOCAL_MAX_TARGET_STUDIO2026 = 0.92
_VOCAL_MAX_ALIGNMENT_RESTORATION_MIN = 0.90
_VOCAL_MAX_ALIGNMENT_STUDIO2026_MIN = 0.93
_VOCAL_MAX_ALIGNMENT_RESTORATION_TOL = 0.02
_VOCAL_MAX_ALIGNMENT_STUDIO2026_TOL = 0.015

_instance: VocalNoHarmGate | None = None
_lock = threading.Lock()


def _load_symbol(module_name: str, symbol_name: str) -> Any:
    """Lädt an optional symbol lazily."""
    return getattr(import_module(module_name), symbol_name)


def _as_float(value: object, default: float = 0.0) -> float:
    """Coerce numeric-like metadata values to finite float."""
    if isinstance(value, (int, float)):
        result = float(value)
        return result if np.isfinite(result) else default
    return default


@dataclass
class VocalNoHarmResult:
    """Serialisierbares Ergebnis von :class:`VocalNoHarmGate`."""

    active: bool
    passed: bool
    requires_rollback: bool
    reason: str = "ok"
    scores: dict[str, float] = field(default_factory=dict)
    checks: dict[str, bool] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Gibt JSON-safe gate metadata zurück."""
        return {
            "active": bool(self.active),
            "passed": bool(self.passed),
            "requires_rollback": bool(self.requires_rollback),
            "reason": str(self.reason),
            "scores": {key: float(value) for key, value in self.scores.items()},
            "checks": {key: bool(value) for key, value in self.checks.items()},
            "warnings": list(self.warnings),
        }


def compute_vocal_max_alignment(
    vqi: float,
    vqi_floor: float,
    material_type: str,
    mode: str,
    restorability_score: float | None,
) -> dict[str, float | bool]:
    """Berechnet VQI-Erreichung relativ zum material-adaptiven Maximum."""
    is_studio = str(mode).lower().replace(" ", "") == "studio2026"
    restorability = 70.0 if restorability_score is None else float(restorability_score)
    restorability = float(np.clip(restorability, 0.0, 100.0))
    rest_ratio = restorability / 100.0

    max_target = _VOCAL_MAX_TARGET_STUDIO2026 if is_studio else _VOCAL_MAX_TARGET_RESTORATION
    material_key = str(material_type or "").strip().lower()
    if material_key in {"wax_cylinder", "shellac", "wire_recording", "lacquer_disc"}:
        max_target = min(max_target, 0.78 + 0.06 * rest_ratio)
    elif material_key in {"mp3_low", "minidisc", "cassette", "kassette"}:
        max_target = min(max_target, 0.82 + 0.05 * rest_ratio)
    elif material_key in {"vinyl", "lp", "tape", "reel_tape"}:
        max_target = min(max_target, 0.86 + 0.04 * rest_ratio)

    vocal_max_target = float(np.clip(max(float(vqi_floor), max_target), float(vqi_floor), 0.95))
    alignment = float(np.clip(float(vqi) / max(vocal_max_target, 1e-6), 0.0, 1.0))
    min_alignment = _VOCAL_MAX_ALIGNMENT_STUDIO2026_MIN if is_studio else _VOCAL_MAX_ALIGNMENT_RESTORATION_MIN
    min_alignment = float(np.clip(min_alignment + 0.04 * rest_ratio, min_alignment, 0.98))
    tol = _VOCAL_MAX_ALIGNMENT_STUDIO2026_TOL if is_studio else _VOCAL_MAX_ALIGNMENT_RESTORATION_TOL
    effective_min_alignment = float(np.clip(min_alignment - tol, 0.0, 0.98))
    return {
        "vocal_max_target": vocal_max_target,
        "vocal_max_alignment": alignment,
        "vocal_max_alignment_percent": alignment * 100.0,
        "vocal_max_alignment_floor_percent": min_alignment * 100.0,
        "vocal_max_alignment_floor_percent_effective": effective_min_alignment * 100.0,
        "vocal_max_alignment_ok": alignment >= effective_min_alignment,
    }


class VocalNoHarmGate:
    """Aggregate vocal safety metrics into one no-harm decision."""

    def evaluate(
        self,
        audio_pre: np.ndarray,
        audio_post: np.ndarray,
        sr: int,
        *,
        panns_singing: float,
        material_type: str = "unknown",
        mode: str = "restoration",
        phase_id: str = "",
        genre: str | None = None,
        reference_audio: np.ndarray | None = None,
        skip_singer_identity: bool = False,
        goal_weights: dict[str, float] | None = None,
        restorability_score: float | None = None,
        breath_segments: list[Any] | None = None,
        era_decade: int | None = None,
        era_vocal_profile: Any | None = None,
    ) -> VocalNoHarmResult:
        """Bewertet whether a vocal processing result is safe to keep.

        The gate activates for ``panns_singing >= 0.35``. Below that threshold it
        returns an inactive pass so instrumental material is not blocked by vocal
        metrics.
        """
        panns = float(np.clip(panns_singing, 0.0, 1.0))
        _mat_floor = _VOCAL_GATE_MATERIAL_FLOOR.get(str(material_type or "").lower(), _VOCAL_GATE_PANNS_FLOOR)
        if panns < _mat_floor:
            return VocalNoHarmResult(active=False, passed=True, requires_rollback=False, reason="not_vocal")

        pre = np.nan_to_num(np.asarray(audio_pre, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        post = np.nan_to_num(np.asarray(audio_post, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        scores: dict[str, float] = {"panns_singing": panns}
        checks: dict[str, bool] = {}
        warnings: list[str] = []
        failures: list[str] = []

        pre_len = pre.shape[-1] if pre.ndim > 1 else pre.shape[0]
        post_len = post.shape[-1] if post.ndim > 1 else post.shape[0]
        length_match = pre_len == post_len
        checks["length_match"] = length_match
        if not length_match:
            failures.append("length_mismatch")

        self._evaluate_artifact_freedom(
            pre,
            post,
            sr,
            material_type,
            phase_id,
            goal_weights,
            restorability_score,
            scores,
            checks,
            warnings,
            failures,
        )
        self._evaluate_vqi(
            pre,
            post,
            sr,
            material_type,
            mode,
            genre,
            reference_audio,
            skip_singer_identity,
            restorability_score,
            scores,
            checks,
            warnings,
            failures,
            era_decade=era_decade,
            era_vocal_profile=era_vocal_profile,
        )
        self._evaluate_hnr(pre, post, sr, scores, checks, warnings, failures)
        self._evaluate_formants(pre, post, sr, scores, checks, warnings, failures, era_decade, era_vocal_profile)
        self._evaluate_breath_preservation(pre, post, sr, breath_segments, scores, checks, failures)
        if reference_audio is not None:
            reference = np.nan_to_num(
                np.asarray(reference_audio, dtype=np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            if reference.shape == post.shape:
                self._evaluate_cumulative_breath_preservation(
                    reference,
                    post,
                    sr,
                    breath_segments,
                    scores,
                    checks,
                    failures,
                )
                self._evaluate_cumulative_hnr(reference, post, sr, scores, checks, warnings, failures)
            else:
                warnings.append("cumulative_reference_shape_mismatch")

        if failures:
            reason = ";".join(failures)
            logger.debug("VocalNoHarmGate rollback recommended: %s", reason)
            return VocalNoHarmResult(
                active=True,
                passed=False,
                requires_rollback=True,
                reason=reason,
                scores=scores,
                checks=checks,
                warnings=warnings,
            )

        return VocalNoHarmResult(
            active=True,
            passed=True,
            requires_rollback=False,
            reason="ok",
            scores=scores,
            checks=checks,
            warnings=warnings,
        )

    @staticmethod
    def _evaluate_artifact_freedom(  # pylint: disable=too-many-positional-arguments
        pre: np.ndarray,
        post: np.ndarray,
        sr: int,
        material_type: str,
        phase_id: str,
        goal_weights: dict[str, float] | None,
        restorability_score: float | None,
        scores: dict[str, float],
        checks: dict[str, bool],
        warnings: list[str],
        failures: list[str],
    ) -> None:
        try:
            get_artifact_freedom_gate = _load_symbol(
                "backend.core.artifact_freedom_gate",
                "get_artifact_freedom_gate",
            )
            result = get_artifact_freedom_gate().evaluate(
                pre,
                post,
                sr,
                material_type=material_type,
                phase_id=phase_id,
                goal_weights=goal_weights,
                restorability_score=restorability_score,
            )
            artifact_freedom = _as_float(getattr(result, "artifact_freedom", 1.0), 1.0)
            scores["artifact_freedom"] = artifact_freedom
            checks["artifact_freedom_ok"] = artifact_freedom >= _ARTIFACT_FREEDOM_FLOOR
            if artifact_freedom < _ARTIFACT_FREEDOM_FLOOR:
                failures.append("artifact_freedom")
        except Exception as exc:  # pylint: disable=broad-except
            warnings.append(f"artifact_freedom_unavailable:{type(exc).__name__}")

    @staticmethod
    def _evaluate_vqi(  # pylint: disable=too-many-positional-arguments
        pre: np.ndarray,
        post: np.ndarray,
        sr: int,
        material_type: str,
        mode: str,
        genre: str | None,
        reference_audio: np.ndarray | None,
        skip_singer_identity: bool,
        restorability_score: float | None,
        scores: dict[str, float],
        checks: dict[str, bool],
        warnings: list[str],
        failures: list[str],
        era_decade: int | None = None,
        era_vocal_profile: Any | None = None,
    ) -> None:
        try:
            compute_vqi = _load_symbol("backend.core.musical_goals.vocal_quality_index", "compute_vqi")
            get_vqi_material_floor = _load_symbol(
                "backend.core.musical_goals.vocal_quality_index",
                "get_vqi_material_floor",
            )
            vqi_result = compute_vqi(
                pre,
                post,
                sr,
                skip_singer_identity=skip_singer_identity,
                reference_audio=reference_audio,
                genre=genre,
                era_profile=era_vocal_profile,
            )
            vqi = _as_float(vqi_result.get("vqi"), 1.0) if isinstance(vqi_result, dict) else 1.0
            singer_identity = (
                _as_float(vqi_result.get("singer_identity_cosine"), 1.0) if isinstance(vqi_result, dict) else 1.0
            )
            vqi_floor = float(
                get_vqi_material_floor(material_type, is_studio_2026=str(mode).lower().replace(" ", "") == "studio2026")
            )
            scores["vqi"] = vqi
            scores["vqi_floor"] = vqi_floor
            scores["singer_identity_cosine"] = singer_identity
            checks["vqi_ok"] = vqi >= vqi_floor
            checks["singer_identity_ok"] = skip_singer_identity or singer_identity >= _SINGER_IDENTITY_FLOOR
            VocalNoHarmGate._evaluate_vocal_max_alignment(
                vqi,
                vqi_floor,
                material_type,
                mode,
                restorability_score,
                scores,
                checks,
                failures,
            )
            if vqi < vqi_floor:
                failures.append("vqi")
            if not skip_singer_identity and singer_identity < _SINGER_IDENTITY_FLOOR:
                failures.append("singer_identity")
        except Exception as exc:  # pylint: disable=broad-except
            warnings.append(f"vqi_unavailable:{type(exc).__name__}")

    @staticmethod
    def _evaluate_vocal_max_alignment(
        vqi: float,
        vqi_floor: float,
        material_type: str,
        mode: str,
        restorability_score: float | None,
        scores: dict[str, float],
        checks: dict[str, bool],
        failures: list[str],
    ) -> None:
        alignment = compute_vocal_max_alignment(vqi, vqi_floor, material_type, mode, restorability_score)
        scores["vocal_max_target"] = float(alignment["vocal_max_target"])
        scores["vocal_max_alignment"] = float(alignment["vocal_max_alignment"])
        scores["vocal_max_alignment_percent"] = float(alignment["vocal_max_alignment_percent"])
        scores["vocal_max_alignment_floor_percent"] = float(alignment["vocal_max_alignment_floor_percent"])
        scores["vocal_max_alignment_floor_percent_effective"] = float(
            alignment["vocal_max_alignment_floor_percent_effective"]
        )
        checks["vocal_max_alignment_ok"] = bool(alignment["vocal_max_alignment_ok"])
        if not checks["vocal_max_alignment_ok"]:
            failures.append("vocal_max_alignment")

    @staticmethod
    def _evaluate_hnr(  # pylint: disable=too-many-positional-arguments
        pre: np.ndarray,
        post: np.ndarray,
        sr: int,
        scores: dict[str, float],
        checks: dict[str, bool],
        warnings: list[str],
        failures: list[str],
    ) -> None:
        try:
            check_hnr_delta = _load_symbol("backend.core.dsp.hnr_guard", "check_hnr_delta")
            hnr_result = check_hnr_delta(pre, post, sr)
            delta_hnr = _as_float(hnr_result.get("delta_hnr"), 0.0) if isinstance(hnr_result, dict) else 0.0
            scores["delta_hnr_db"] = delta_hnr
            checks["hnr_ok"] = delta_hnr <= _HNR_DELTA_MAX_DB
            if delta_hnr > _HNR_DELTA_MAX_DB:
                failures.append("hnr_overcleaned")
        except Exception as exc:  # pylint: disable=broad-except
            warnings.append(f"hnr_unavailable:{type(exc).__name__}")

    @staticmethod
    def _evaluate_cumulative_hnr(  # pylint: disable=too-many-positional-arguments
        reference: np.ndarray,
        post: np.ndarray,
        sr: int,
        scores: dict[str, float],
        checks: dict[str, bool],
        warnings: list[str],
        failures: list[str],
    ) -> None:
        try:
            check_hnr_delta = _load_symbol("backend.core.dsp.hnr_guard", "check_hnr_delta")
            hnr_result = check_hnr_delta(reference, post, sr)
            delta_hnr = _as_float(hnr_result.get("delta_hnr"), 0.0) if isinstance(hnr_result, dict) else 0.0
            scores["cumulative_delta_hnr_db"] = delta_hnr
            checks["cumulative_hnr_ok"] = delta_hnr <= _HNR_DELTA_MAX_DB
            if delta_hnr > _HNR_DELTA_MAX_DB:
                failures.append("cumulative_hnr_overcleaned")
        except Exception as exc:  # pylint: disable=broad-except
            warnings.append(f"cumulative_hnr_unavailable:{type(exc).__name__}")

    @staticmethod
    def _evaluate_formants(  # pylint: disable=too-many-positional-arguments
        pre: np.ndarray,
        post: np.ndarray,
        sr: int,
        scores: dict[str, float],
        checks: dict[str, bool],
        warnings: list[str],
        failures: list[str],
        era_decade: int | None = None,
        era_vocal_profile: Any | None = None,
    ) -> None:
        try:
            check_formant_shift_db = _load_symbol("backend.core.dsp.lpc_formant_tracker", "check_formant_shift_db")
            threshold_db = _FORMANT_SHIFT_MAX_DB
            try:
                resolve_formant_tolerance_db = _load_symbol(
                    "backend.core.musical_goals.era_vocal_profile", "resolve_formant_tolerance_db"
                )
                threshold_db = float(
                    resolve_formant_tolerance_db(
                        era_decade=era_decade,
                        era_profile=era_vocal_profile,
                        fallback_db=_FORMANT_SHIFT_MAX_DB,
                    )
                )
            except Exception as exc:  # pylint: disable=broad-except
                warnings.append(f"formant_tolerance_unavailable:{type(exc).__name__}")
            rollback_needed, max_shift_db = check_formant_shift_db(
                pre,
                post,
                sr,
                threshold_db=threshold_db,
            )
            scores["max_formant_shift_db"] = float(max_shift_db)
            scores["formant_tolerance_db"] = float(threshold_db)
            checks["formant_ok"] = not bool(rollback_needed)
            if rollback_needed:
                failures.append("formant_shift")
        except Exception as exc:  # pylint: disable=broad-except
            warnings.append(f"formant_unavailable:{type(exc).__name__}")

    @staticmethod
    def _evaluate_breath_preservation(  # pylint: disable=too-many-positional-arguments
        pre: np.ndarray,
        post: np.ndarray,
        sr: int,
        breath_segments: list[Any] | None,
        scores: dict[str, float],
        checks: dict[str, bool],
        failures: list[str],
    ) -> None:
        if not breath_segments:
            checks["breath_preservation_ok"] = True
            scores["max_breath_attenuation_db"] = 0.0
            return

        pre_mono = VocalNoHarmGate._to_mono(pre)
        post_mono = VocalNoHarmGate._to_mono(post)
        n = min(pre_mono.shape[-1], post_mono.shape[-1])
        max_attenuation_db = 0.0
        protected_count = 0

        for segment in breath_segments:
            start_s, end_s = VocalNoHarmGate._segment_bounds_s(segment)
            if end_s <= start_s:
                continue
            category = VocalNoHarmGate._segment_category(segment)
            if category == "mechanical_pop":
                continue

            start = int(np.clip(round(start_s * sr), 0, n))
            end = int(np.clip(round(end_s * sr), start, n))
            if end - start < max(16, int(0.02 * sr)):
                continue

            pre_db = VocalNoHarmGate._rms_db(pre_mono[start:end])
            if pre_db < _BREATH_MIN_RMS_DB:
                continue

            post_db = VocalNoHarmGate._rms_db(post_mono[start:end])
            attenuation_db = max(0.0, pre_db - post_db)
            max_attenuation_db = max(max_attenuation_db, attenuation_db)
            protected_count += 1
            limit_db = (
                _BREATH_EMOTIONAL_ATTENUATION_MAX_DB if category == "emotional_tension" else _BREATH_ATTENUATION_MAX_DB
            )
            if attenuation_db > limit_db:
                failures.append("breath_preservation")
                break

        scores["max_breath_attenuation_db"] = float(max_attenuation_db)
        scores["protected_breath_segments"] = float(protected_count)
        checks["breath_preservation_ok"] = "breath_preservation" not in failures

    @staticmethod
    def _evaluate_cumulative_breath_preservation(  # pylint: disable=too-many-positional-arguments
        reference: np.ndarray,
        post: np.ndarray,
        sr: int,
        breath_segments: list[Any] | None,
        scores: dict[str, float],
        checks: dict[str, bool],
        failures: list[str],
    ) -> None:
        if not breath_segments:
            checks["cumulative_breath_preservation_ok"] = True
            scores["cumulative_max_breath_attenuation_db"] = 0.0
            return

        before_failures = len(failures)
        cumulative_failures: list[str] = []
        cumulative_scores: dict[str, float] = {}
        cumulative_checks: dict[str, bool] = {}
        VocalNoHarmGate._evaluate_breath_preservation(
            reference,
            post,
            sr,
            breath_segments,
            cumulative_scores,
            cumulative_checks,
            cumulative_failures,
        )
        scores["cumulative_max_breath_attenuation_db"] = float(cumulative_scores.get("max_breath_attenuation_db", 0.0))
        scores["cumulative_protected_breath_segments"] = float(cumulative_scores.get("protected_breath_segments", 0.0))
        if "breath_preservation" in cumulative_failures:
            failures.append("cumulative_breath_preservation")
        checks["cumulative_breath_preservation_ok"] = len(failures) == before_failures

    @staticmethod
    def _to_mono(audio: np.ndarray) -> np.ndarray:
        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim == 1:
            return arr  # type: ignore[no-any-return]
        if arr.ndim == 2 and arr.shape[0] <= 8:
            return cast(np.ndarray, np.asarray(np.mean(arr, axis=0), dtype=np.float32))
        if arr.ndim == 2:
            return cast(np.ndarray, np.asarray(np.mean(arr, axis=1), dtype=np.float32))
        return arr.reshape(-1).astype(np.float32)  # type: ignore[no-any-return]

    @staticmethod
    def _rms_db(audio: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(np.asarray(audio, dtype=np.float32))) + 1e-12))
        return float(20.0 * np.log10(max(rms, 1e-12)))

    @staticmethod
    def _segment_bounds_s(segment: Any) -> tuple[float, float]:
        if isinstance(segment, dict):
            return float(segment.get("start_s", 0.0)), float(segment.get("end_s", 0.0))
        if isinstance(segment, (tuple, list)) and len(segment) >= 2:
            return float(segment[0]), float(segment[1])
        return float(getattr(segment, "start_s", 0.0)), float(getattr(segment, "end_s", 0.0))

    @staticmethod
    def _segment_category(segment: Any) -> str:
        raw = (
            segment.get("category", "natural") if isinstance(segment, dict) else getattr(segment, "category", "natural")
        )
        return str(getattr(raw, "value", raw)).lower()


def get_vocal_no_harm_gate() -> VocalNoHarmGate:
    """Gibt the thread-safe singleton VocalNoHarmGate zurück."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = VocalNoHarmGate()
    return _instance


def evaluate_vocal_no_harm(
    audio_pre: np.ndarray,
    audio_post: np.ndarray,
    sr: int,
    **kwargs: Any,
) -> VocalNoHarmResult:
    """Convenience wrapper for :meth:`VocalNoHarmGate.evaluate`."""
    return get_vocal_no_harm_gate().evaluate(audio_pre, audio_post, sr, **kwargs)
