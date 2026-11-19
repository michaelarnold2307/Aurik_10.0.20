"""Reproduzierbare Ground-Truth-Gegenprobe für den A-SPADE-Challenger.

Dies ist ein Forschungswerkzeug, kein Freigabe- oder Hörtest. Für den
Modellpfad wird die bekannte synthetische Schwelle verwendet; Phase 07 wird
mit ihrer automatischen Schwelle und dem Default-Flag gemessen. Optional wird
auch der bestehende Opt-in-Flagpfad gemessen.
Ein weiterer Forschungszweig testet exakte Plateaus je Kanal ohne Orakel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from scipy.ndimage import uniform_filter1d
from scipy.signal import resample_poly

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from backend.core.dsp.aspade_declipper import DEFAULT_CHECKPOINT, TARGET_SR, AspadeClassDeclipper
from backend.core.phases.phase_07_declipper import DeclipperPhase
from backend.core.phases.phase_interface import PhaseResult


def _error_report(clean: np.ndarray, damaged: np.ndarray, result: np.ndarray) -> dict[str, Any]:
    """Energiemaße gegen Referenz; keine psychoakustische Abnahme."""
    if clean.shape != damaged.shape or clean.shape != result.shape:
        raise ValueError("Referenz, Eingang und Ausgabe müssen dieselbe Shape haben")
    if not all(np.isfinite(a).all() for a in (clean, damaged, result)):
        raise ValueError("Messung enthält NaN/Inf")
    ref = clean.astype(np.float64)
    before = damaged.astype(np.float64)
    after = result.astype(np.float64)
    mask = before != ref
    error_before = float(np.sum((before - ref) ** 2))
    error_after = float(np.sum((after - ref) ** 2))
    control = after[~mask] - before[~mask]
    return {
        "damaged_samples": int(np.count_nonzero(mask)),
        "error_energy_before": error_before,
        "error_energy_after": error_after,
        "delta_sdr_db": (
            float(10 * (np.log10(error_before) - np.log10(max(error_after, np.finfo(float).tiny))))
            if error_before > 0
            else None
        ),
        "error_energy_recovered_fraction": (1.0 - error_after / error_before if error_before > 0 else None),
        "control_changed_samples": int(np.count_nonzero(control)),
        "control_max_abs_change": float(np.max(np.abs(control), initial=0.0)),
    }


def _read_segment(path: Path, offset: float, duration: float, layout: str) -> np.ndarray:
    with sf.SoundFile(str(path)) as stream:
        sr = stream.samplerate
        start = int(offset * sr)
        length = int(duration * sr)
        if start + length > len(stream):
            raise ValueError(f"{path}: angeforderter Ausschnitt liegt außerhalb der Datei")
        stream.seek(start)
        data = stream.read(length, dtype="float32", always_2d=True)
    if not np.isfinite(data).all():
        raise ValueError(f"{path}: Audio enthält NaN/Inf")
    if layout == "mono":
        data = data.mean(axis=1, keepdims=True)
    elif data.shape[1] != 2:
        raise ValueError(f"{path}: Stereo angefordert, aber {data.shape[1]} Kanäle gefunden")
    if sr != TARGET_SR:
        factor = gcd(sr, TARGET_SR)
        data = resample_poly(data, TARGET_SR // factor, sr // factor, axis=0).astype(np.float32)
    peak = float(np.max(np.abs(data), initial=0.0))
    if peak < 1e-6:
        raise ValueError(f"{path}: Ausschnitt ist still")
    return np.ascontiguousarray((data * (0.95 / peak)).T)


def _resampling_roundtrip(audio: np.ndarray) -> np.ndarray:
    return resample_poly(resample_poly(audio, 147, 160, axis=-1), 160, 147, axis=-1).astype(np.float32)


def _plateau_candidate(audio: np.ndarray, declipper: AspadeClassDeclipper) -> tuple[np.ndarray, list[int]]:
    """Versuch: identische benachbarte Extremwerte als Kanal- und Pegelhinweis.

    Dies erkennt nur exaktes Hard-Clipping und ist kein sicherer Detektor:
    Quantisierung oder beabsichtigte Sättigung können identische Plateaus bilden.
    """
    out = audio.copy()
    detected: list[int] = []
    for channel, samples in enumerate(audio):
        theta = float(np.max(np.abs(samples), initial=0.0))
        extrema = np.abs(samples) == theta
        if theta > 0 and np.any(extrema[:-1] & extrema[1:] & (samples[:-1] == samples[1:])):
            out[channel] = declipper.declip(samples, TARGET_SR, theta)
            detected.append(channel)
    return out, detected


def _polarity_plateau_candidate(
    audio: np.ndarray, declipper: AspadeClassDeclipper
) -> tuple[np.ndarray, list[dict[str, float | int | None]]]:
    """Forschung: unabhängige exakte positive und negative Kanalplateaus.

    Bei zwei detektierten Pegeln wird das Signal für das symmetrische Modell
    zentriert; nur detektierte Plateaus werden zurückgeschrieben. Identische
    Samples beweisen weder unbeabsichtigtes Clipping noch Hörgewinn.
    """
    if audio.ndim != 2 or audio.shape[0] not in (1, 2) or audio.shape[1] == 0:
        raise ValueError("Audio muss nichtleer und channels-first mit einem oder zwei Kanälen sein")
    if not np.isfinite(audio).all():
        raise ValueError("Audio enthält NaN/Inf")
    out = audio.copy()
    detected: list[dict[str, float | int | None]] = []
    for channel, samples in enumerate(audio):
        hi, lo = float(np.max(samples)), float(np.min(samples))
        hi_mask = samples == hi
        lo_mask = samples == lo
        positive = hi > 0 and bool(np.any(hi_mask[:-1] & hi_mask[1:]))
        negative = lo < 0 and bool(np.any(lo_mask[:-1] & lo_mask[1:]))
        if not positive and not negative:
            continue
        center = (hi + lo) / 2.0 if positive and negative else 0.0
        theta = (hi - lo) / 2.0 if positive and negative else (hi if positive else -lo)
        centered = samples - center
        # Float32 rounding must not turn a detected plateau into a reliable sample.
        if positive:
            centered[hi_mask] = theta
        if negative:
            centered[lo_mask] = -theta
        candidate = declipper.declip(centered, TARGET_SR, theta) + center
        if candidate.shape != samples.shape or not np.isfinite(candidate).all():
            raise ValueError("Polarity-Kandidat hat eine ungültige Shape oder enthält NaN/Inf")
        mask = (hi_mask if positive else np.zeros_like(hi_mask)) | (lo_mask if negative else np.zeros_like(lo_mask))
        out[channel, mask] = candidate[mask]
        detected.append(
            {"channel": channel, "positive_level": hi if positive else None, "negative_level": lo if negative else None}
        )
    return out, detected


def _local_stereo_estimate(
    target: np.ndarray, reference: np.ndarray, window_samples: int, valid_mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Lokale lineare Regression und Bestimmtheitsmaß auf beobachteten Samples."""
    if target.shape != reference.shape or target.ndim != 1 or target.size == 0:
        raise ValueError("Stereo-Kanäle müssen nichtleer, eindimensional und gleich lang sein")
    if window_samples < 1:
        raise ValueError("window_samples muss positiv sein")
    if valid_mask is not None and (valid_mask.shape != target.shape or valid_mask.dtype != np.bool_):
        raise ValueError("valid_mask muss eine gleich lange Bool-Maske sein")
    valid = (
        valid_mask.astype(np.float64)
        if valid_mask is not None
        else (np.abs(target) < np.max(np.abs(target))).astype(np.float64)
    )

    def average(values: np.ndarray) -> np.ndarray:
        return uniform_filter1d(values, size=window_samples, mode="nearest")

    weight = average(valid)
    sum_ref = average(valid * reference)
    sum_target = average(valid * target)
    sum_ref2 = average(valid * reference * reference)
    sum_target2 = average(valid * target * target)
    sum_cross = average(valid * reference * target)
    denominator = weight * sum_ref2 - sum_ref * sum_ref
    slope = np.divide(
        weight * sum_cross - sum_ref * sum_target,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 1e-12,
    )
    intercept = np.divide(sum_target - slope * sum_ref, weight, out=np.zeros_like(weight), where=weight > 1e-10)
    covariance = weight * sum_cross - sum_ref * sum_target
    target_variance = weight * sum_target2 - sum_target * sum_target
    variance_product = denominator * target_variance
    r_squared = np.divide(
        covariance * covariance, variance_product, out=np.zeros_like(weight), where=variance_product > 1e-14
    )
    return slope * reference + intercept, r_squared


def _polarity_stereo_witness_candidate(
    damaged: np.ndarray,
    candidate: np.ndarray,
    detected_levels: list[dict[str, float | int | None]],
    window_samples: int,
) -> np.ndarray:
    """Forschung: stereo-beschränktes Residuum bei einem erkannten Defektkanal."""
    if damaged.shape != candidate.shape or damaged.ndim != 2 or damaged.shape[0] != 2:
        raise ValueError("Stereo-Eingang und Kandidat müssen dieselbe channels-first Shape haben")
    if window_samples < 1:
        raise ValueError("window_samples muss positiv sein")
    if len(detected_levels) != 1:
        return damaged.copy()
    level = detected_levels[0]
    ch = level["channel"]
    if not isinstance(ch, int) or ch not in (0, 1):
        raise ValueError("Ungültiger Defektkanal")
    target = damaged[ch].astype(np.float64)
    reference = damaged[1 - ch].astype(np.float64)
    mask = np.zeros(target.shape, dtype=bool)
    for polarity in ("positive_level", "negative_level"):
        value = level[polarity]
        if value is not None:
            mask |= target == value
    prediction, _ = _local_stereo_estimate(target, reference, window_samples, ~mask)
    residual = candidate[ch].astype(np.float64) - target
    alpha = np.divide(prediction - target, residual, out=np.zeros_like(residual), where=np.abs(residual) > 1e-10)
    out = damaged.copy()
    out[ch] = (target + np.clip(alpha, 0.0, 1.0) * residual).astype(out.dtype)
    return out


def _stereo_witness_candidate(
    damaged: np.ndarray, candidate: np.ndarray, detected_channels: list[int], window_samples: int
) -> np.ndarray:
    """Forschung: beschränke das Residuum anhand eines lokalen Stereo-Zeugen.

    Der andere Kanal ist nicht zwingend unbeschädigt oder hinreichend ähnlich;
    diese Schätzung ist daher weder ein Sicherheitsbeweis noch produktiv.
    """
    if damaged.shape != candidate.shape or damaged.ndim != 2 or damaged.shape[0] != 2:
        raise ValueError("Stereo-Eingang und Kandidat müssen dieselbe channels-first Shape haben")
    if window_samples < 1:
        raise ValueError("window_samples muss positiv sein")
    if len(detected_channels) != 1 or detected_channels[0] not in (0, 1):
        return damaged.copy()
    ch = detected_channels[0]
    reference = damaged[1 - ch].astype(np.float64)
    target = damaged[ch].astype(np.float64)
    residual = candidate[ch].astype(np.float64) - target
    prediction, _ = _local_stereo_estimate(target, reference, window_samples)
    alpha = np.divide(prediction - target, residual, out=np.zeros_like(residual), where=np.abs(residual) > 1e-10)
    out = damaged.copy()
    out[ch] = (target + np.clip(alpha, 0.0, 1.0) * residual).astype(out.dtype)
    return out


def _density_gated_candidate(
    damaged: np.ndarray,
    candidate: np.ndarray,
    witness: np.ndarray,
    detected_channels: list[int],
    fraction: float,
) -> np.ndarray:
    """Forschung: bei niedriger Plateau-Dichte den Stereo-Zeugen verwenden."""
    if damaged.shape != candidate.shape or damaged.shape != witness.shape or damaged.ndim != 2:
        raise ValueError("Eingang und Kandidaten müssen dieselbe channels-first Shape haben")
    if not 0 <= fraction <= 1:
        raise ValueError("fraction muss zwischen 0 und 1 liegen")
    if len(detected_channels) != 1 or detected_channels[0] not in range(damaged.shape[0]):
        return damaged.copy()
    ch = detected_channels[0]
    samples = damaged[ch]
    if not samples.size:
        raise ValueError("Leerer Kanal")
    peak = np.max(np.abs(samples))
    density = np.count_nonzero(np.abs(samples) == peak) / samples.size
    return candidate.copy() if density > fraction else witness.copy()


def _correlation_gated_candidate(
    damaged: np.ndarray, base: np.ndarray, witness: np.ndarray, detected_channels: list[int], window_samples: int
) -> np.ndarray:
    """Forschungsvariante: vertraue dem Stereo-Zeugen bei hoher lokaler Korrelation."""
    if damaged.shape != base.shape or damaged.shape != witness.shape or damaged.ndim != 2 or damaged.shape[0] != 2:
        raise ValueError("Stereo-Eingang und Kandidaten müssen dieselbe Shape haben")
    if window_samples < 1:
        raise ValueError("window_samples muss positiv sein")
    if len(detected_channels) != 1 or detected_channels[0] not in (0, 1):
        return damaged.copy()
    ch = detected_channels[0]
    _, r_squared = _local_stereo_estimate(
        damaged[ch].astype(np.float64), damaged[1 - ch].astype(np.float64), window_samples
    )
    out = base.copy()
    out[ch] = np.where(r_squared >= 0.95, witness[ch], base[ch])
    return out


def _phase_opt_in(audio: np.ndarray) -> PhaseResult:
    """Misst den vorhandenen Phase-07-Flagpfad isoliert in diesem CLI-Prozess."""
    from backend.core import music_model_flags

    previous = music_model_flags.use_aspade_declipper
    try:
        music_model_flags.use_aspade_declipper = True
        return DeclipperPhase().process(audio.copy(), sample_rate=TARGET_SR, material="cd_digital")
    finally:
        music_model_flags.use_aspade_declipper = previous


def evaluate_segment(
    clean: np.ndarray,
    theta: float,
    declipper: AspadeClassDeclipper,
    *,
    right_only: bool = False,
    phase_opt_in: bool = False,
) -> dict[str, Any]:
    """Prüft Rohmodell mit bekannter und Phase mit geschätzter Schwelle."""
    if clean.ndim != 2 or clean.shape[0] not in (1, 2):
        raise ValueError("Audio muss channels-first mit einem oder zwei Kanälen sein")
    if not 0 < theta < 0.95:
        raise ValueError("Clipping-Schwelle muss zwischen 0 und 0.95 liegen")
    if right_only and clean.shape[0] != 2:
        raise ValueError("right_only benötigt Stereo")
    damaged = clean.copy()
    channels = [1] if right_only else range(clean.shape[0])
    for channel in channels:
        damaged[channel] = np.clip(clean[channel], -theta, theta)
    raw = declipper.declip(damaged, TARGET_SR, theta)
    raw_oracle_channel = None
    if right_only:
        raw_oracle_channel = damaged.copy()
        raw_oracle_channel[1] = declipper.declip(damaged[1], TARGET_SR, theta)
    plateau, plateau_channels = _plateau_candidate(damaged, declipper)
    polarity, polarity_levels = _polarity_plateau_candidate(damaged, declipper)
    asymmetric = clean.copy()
    negative_theta = min(0.95, theta * 1.5)
    for channel in channels:
        asymmetric[channel] = np.clip(clean[channel], -negative_theta, theta)
    asymmetric_plateau, asymmetric_channels = _plateau_candidate(asymmetric, declipper)
    asymmetric_polarity, asymmetric_levels = _polarity_plateau_candidate(asymmetric, declipper)
    asymmetric_polarity_stereo = (
        _polarity_stereo_witness_candidate(asymmetric, asymmetric_polarity, asymmetric_levels, 241)
        if asymmetric.shape[0] == 2
        else None
    )
    polarity_stereo = (
        _polarity_stereo_witness_candidate(damaged, polarity, polarity_levels, 241) if damaged.shape[0] == 2 else None
    )
    stereo_witness = (
        _stereo_witness_candidate(damaged, plateau, plateau_channels, window_samples=241)
        if damaged.shape[0] == 2
        else None
    )
    density_gated = (
        _density_gated_candidate(damaged, plateau, stereo_witness, plateau_channels, fraction=0.001)
        if stereo_witness is not None
        else None
    )
    correlation_gated = (
        _correlation_gated_candidate(damaged, density_gated, stereo_witness, plateau_channels, 241)
        if density_gated is not None and stereo_witness is not None
        else None
    )
    # Forschungsvergleich, keine zentrale Strength-Entscheidung oder Freigabe.
    plateau_blend = damaged + np.float32(0.75) * (plateau - damaged)
    resampled = _resampling_roundtrip(damaged)
    resampled_clean = _resampling_roundtrip(clean)
    pre_resampling_oracle = _resampling_roundtrip(raw_oracle_channel if right_only else raw)
    resampled_out, resampled_channels = _plateau_candidate(resampled, declipper)
    resampled_polarity, resampled_levels = _polarity_plateau_candidate(resampled, declipper)
    resampled_oracle = resampled.copy()
    for channel in channels:
        resampled_oracle[channel] = declipper.declip(resampled[channel], TARGET_SR, theta)
    plateau_clean, clean_channels = _plateau_candidate(clean, declipper)
    polarity_clean, polarity_clean_levels = _polarity_plateau_candidate(clean, declipper)
    polarity_stereo_clean = (
        _polarity_stereo_witness_candidate(clean, polarity_clean, polarity_clean_levels, 241)
        if clean.shape[0] == 2
        else None
    )
    stereo_witness_clean = (
        _stereo_witness_candidate(clean, plateau_clean, clean_channels, window_samples=241)
        if clean.shape[0] == 2
        else None
    )
    density_gated_clean = (
        _density_gated_candidate(clean, plateau_clean, stereo_witness_clean, clean_channels, fraction=0.001)
        if stereo_witness_clean is not None
        else None
    )
    correlation_gated_clean = (
        _correlation_gated_candidate(clean, density_gated_clean, stereo_witness_clean, clean_channels, 241)
        if density_gated_clean is not None and stereo_witness_clean is not None
        else None
    )
    controls = {
        "tanh4": (np.tanh(clean * 4.0) / np.float32(np.tanh(4.0))).astype(np.float32),
        "tanh8": (np.tanh(clean * 8.0) / np.float32(np.tanh(8.0))).astype(np.float32),
        "pcm16": (np.round(clean * 32767.0) / 32767.0).astype(np.float32),
    }
    control_reports = {}
    for label, control in controls.items():
        control_out, control_channels = _plateau_candidate(control, declipper)
        polarity_control, polarity_control_levels = _polarity_plateau_candidate(control, declipper)
        polarity_stereo_control = (
            _polarity_stereo_witness_candidate(control, polarity_control, polarity_control_levels, 241)
            if control.shape[0] == 2
            else None
        )
        stereo_control = (
            _stereo_witness_candidate(control, control_out, control_channels, window_samples=241)
            if control.shape[0] == 2
            else None
        )
        density_control = (
            _density_gated_candidate(control, control_out, stereo_control, control_channels, fraction=0.001)
            if stereo_control is not None
            else None
        )
        correlation_control = (
            _correlation_gated_candidate(control, density_control, stereo_control, control_channels, 241)
            if density_control is not None and stereo_control is not None
            else None
        )
        control_reports[label] = {
            **_error_report(control, control, control_out),
            "detected_channels": control_channels,
            "polarity_plateau": {
                **_error_report(control, control, polarity_control),
                "detected_levels": polarity_control_levels,
            },
            "polarity_stereo_witness": (
                _error_report(control, control, polarity_stereo_control)
                if polarity_stereo_control is not None
                else None
            ),
            "stereo_witness": (_error_report(control, control, stereo_control) if stereo_control is not None else None),
            "density_gated": (
                _error_report(control, control, density_control) if density_control is not None else None
            ),
            "correlation_gated": (
                _error_report(control, control, correlation_control) if correlation_control is not None else None
            ),
        }
    phase = DeclipperPhase().process(damaged.copy(), sample_rate=TARGET_SR, material="cd_digital")
    clean_phase = DeclipperPhase().process(clean.copy(), sample_rate=TARGET_SR, material="cd_digital")
    opt_in_phase = _phase_opt_in(damaged) if phase_opt_in else None
    opt_in_clean = _phase_opt_in(clean) if phase_opt_in else None
    return {
        "theta_known": theta,
        "right_only": right_only,
        "raw_known_threshold": _error_report(clean, damaged, raw),
        "raw_known_threshold_and_channel": (
            _error_report(clean, damaged, raw_oracle_channel) if raw_oracle_channel is not None else None
        ),
        "experimental_exact_plateau": {
            **_error_report(clean, damaged, plateau),
            "detected_channels": plateau_channels,
        },
        "experimental_polarity_plateau": {
            **_error_report(clean, damaged, polarity),
            "detected_levels": polarity_levels,
        },
        "experimental_polarity_stereo_witness": (
            _error_report(clean, damaged, polarity_stereo) if polarity_stereo is not None else None
        ),
        "experimental_asymmetric": {
            "positive_threshold": theta,
            "negative_threshold": -negative_theta,
            "exact_plateau": {
                **_error_report(clean, asymmetric, asymmetric_plateau),
                "detected_channels": asymmetric_channels,
            },
            "polarity_plateau": {
                **_error_report(clean, asymmetric, asymmetric_polarity),
                "detected_levels": asymmetric_levels,
            },
            "polarity_stereo_witness": (
                _error_report(clean, asymmetric, asymmetric_polarity_stereo)
                if asymmetric_polarity_stereo is not None
                else None
            ),
        },
        "experimental_exact_plateau_blend_075": _error_report(clean, damaged, plateau_blend),
        "experimental_stereo_witness": (
            _error_report(clean, damaged, stereo_witness) if stereo_witness is not None else None
        ),
        "experimental_density_gated": (
            _error_report(clean, damaged, density_gated) if density_gated is not None else None
        ),
        "experimental_correlation_gated": (
            _error_report(clean, damaged, correlation_gated) if correlation_gated is not None else None
        ),
        "experimental_resampled_detection": {
            "detected_channels": resampled_channels,
            "changed_samples": int(np.count_nonzero(resampled_out != resampled)),
            "polarity_plateau": {
                "detected_levels": resampled_levels,
                "changed_samples": int(np.count_nonzero(resampled_polarity != resampled)),
            },
            "known_threshold_and_channel_oracle": _error_report(resampled_clean, resampled, resampled_oracle),
            "repair_before_resampling_oracle": _error_report(resampled_clean, resampled, pre_resampling_oracle),
        },
        "experimental_clean_control": {
            **_error_report(clean, clean, plateau_clean),
            "detected_channels": clean_channels,
            "polarity_plateau": {
                **_error_report(clean, clean, polarity_clean),
                "detected_levels": polarity_clean_levels,
            },
            "polarity_stereo_witness": (
                _error_report(clean, clean, polarity_stereo_clean) if polarity_stereo_clean is not None else None
            ),
            "stereo_witness": (
                _error_report(clean, clean, stereo_witness_clean) if stereo_witness_clean is not None else None
            ),
            "density_gated": (
                _error_report(clean, clean, density_gated_clean) if density_gated_clean is not None else None
            ),
            "correlation_gated": (
                _error_report(clean, clean, correlation_gated_clean) if correlation_gated_clean is not None else None
            ),
        },
        "experimental_nonlinear_and_quantized_controls": control_reports,
        "phase_automatic_threshold": {
            **_error_report(clean, damaged, np.asarray(phase.audio)),
            "metrics": phase.metrics,
        },
        "phase_opt_in_automatic_threshold": (
            {
                **_error_report(clean, damaged, np.asarray(opt_in_phase.audio)),
                "metrics": opt_in_phase.metrics,
                "changed_vs_default": bool(not np.array_equal(opt_in_phase.audio, phase.audio)),
            }
            if opt_in_phase is not None
            else None
        ),
        "clean_control": _error_report(clean, clean, np.asarray(clean_phase.audio)),
        "phase_opt_in_clean_control": (
            {
                **_error_report(clean, clean, np.asarray(opt_in_clean.audio)),
                "metrics": opt_in_clean.metrics,
                "changed_vs_default": bool(not np.array_equal(opt_in_clean.audio, clean_phase.audio)),
            }
            if opt_in_clean is not None
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    roots = parser.add_mutually_exclusive_group(required=True)
    roots.add_argument("--musdb-root", type=Path)
    roots.add_argument("--audio-root", type=Path, help="alternativer Korpus mit WAV-Dateien in Unterordnern")
    parser.add_argument("--split", choices=("train", "test"))
    parser.add_argument("--stem", choices=("mixture", "vocals"))
    parser.add_argument("--layout", choices=("mono", "stereo"), default="mono")
    parser.add_argument("--right-only", action="store_true")
    parser.add_argument(
        "--phase-opt-in", action="store_true", help="zusätzlich bestehenden Phase-07-Flagpfad im Prozess messen"
    )
    parser.add_argument("--offset", type=float, default=30.0)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--threshold", type=float, nargs="+", default=[0.3, 0.6, 0.85])
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.musdb_root is not None and args.split is None:
        parser.error("--musdb-root benötigt --split")
    if args.audio_root is not None and args.split is not None:
        parser.error("--split ist nur mit --musdb-root zulässig")
    if args.audio_root is not None and args.stem is not None:
        parser.error("--stem ist nur mit --musdb-root zulässig")
    if args.offset < 0 or args.duration <= 0 or args.limit <= 0:
        parser.error("offset >= 0, duration > 0 und limit > 0 erforderlich")
    if args.right_only and args.layout != "stereo":
        parser.error("--right-only benötigt --layout stereo")
    if any(not 0 < theta < 0.95 for theta in args.threshold):
        parser.error("Jede Schwelle muss zwischen 0 und 0.95 liegen")
    root = args.musdb_root / args.split if args.musdb_root is not None else args.audio_root
    if root is None or not root.is_dir():
        parser.error(f"Korpusverzeichnis fehlt: {root}")
    stem = args.stem or "mixture"
    paths = (sorted(root.glob(f"*/{stem}.wav")) if args.musdb_root is not None else sorted(root.rglob("*.wav")))[
        : args.limit
    ]
    if len(paths) < args.limit:
        parser.error(f"Nur {len(paths)} von {args.limit} angeforderten Titeln gefunden")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint fehlt: {args.checkpoint}")
    if args.phase_opt_in and args.checkpoint.resolve() != DEFAULT_CHECKPOINT.resolve():
        parser.error("--phase-opt-in benötigt den Produktions-Checkpoint")
    from backend.core import music_model_flags

    if music_model_flags.use_aspade_declipper:
        parser.error("Phase-07-Vergleich erfordert das ausgeschaltete A-SPADE-Default-Flag")
    torch.set_num_threads(2)
    declipper = AspadeClassDeclipper(checkpoint=args.checkpoint, never_worsen=False)
    results = []
    for path in paths:
        clean = _read_segment(path, args.offset, args.duration, args.layout)
        source = sf.info(str(path))
        results.append(
            {
                "track": path.parent.name if args.musdb_root is not None else str(path.relative_to(root)),
                "source_sample_rate": source.samplerate,
                "source_channels": source.channels,
                "segment_sha256": hashlib.sha256(clean.tobytes()).hexdigest(),
                "cases": [
                    evaluate_segment(
                        clean, theta, declipper, right_only=args.right_only, phase_opt_in=args.phase_opt_in
                    )
                    for theta in args.threshold
                ],
            }
        )
    report = {
        "schema": "aurik.aspade-exploration/11",
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "corpus_root": str(root.resolve()),
        "split": args.split,
        "stem": stem if args.musdb_root is not None else None,
        "layout": args.layout,
        "offset_seconds": args.offset,
        "duration_seconds": args.duration,
        "sample_rate": TARGET_SR,
        "phase_opt_in": args.phase_opt_in,
        "note": "Exploration, keine Freigabe; keine Hörprüfung oder kanonische Reparaturquote.",
        "tracks": results,
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
