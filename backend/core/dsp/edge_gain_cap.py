"""§2.46f Edge-Gain-Cap — Randzonen (Intro/Outro) nie lauter als Original + Marge.

Normative Grundlage: §2.46f Natural-Performance-Artifacts-Guard, §0 Primum
non nocere. Additive Nach-Schritte der Denoise-Kette (V21 Noise-Floor,
Timbral-Resynth, HR-V1) laufen NACH dem Konvex-Edge-Taper und können die
Randzonen über die +2-dB-Marge heben. Dieser Cap skaliert die Randzone
sanft auf max(original × 10^(marge/20)) zurück:

  - Nur Absenkung (gain ≤ 1.0) — ein zu leises Intro wird NIE angehoben.
  - 50-ms-Crossfade an der Innenkante der Zone (kein Gain-Sprung).
  - Layout-sicher: channels-first (C, N) und channels-last (N, C), Mono.
  - Deterministisch (§G5 (GEBOTE.md)), rein numpy, §V6-fail-open (Fehler →
    unverändertes Signal zurück).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_EDGE_S = 0.5
_DEFAULT_MAX_GAIN_DB = 2.0
_DEFAULT_FADE_S = 0.05  # Crossfade an der Innenkante


def apply_edge_gain_cap(
    original: np.ndarray,
    processed: np.ndarray,
    sr: int,
    *,
    edge_s: float = _DEFAULT_EDGE_S,
    max_gain_db: float = _DEFAULT_MAX_GAIN_DB,
    fade_s: float = _DEFAULT_FADE_S,
) -> np.ndarray:
    """Kappt die Verstärkung der Intro-/Outro-Zone auf `max_gain_db` über dem Original.

    Args:
        original: Referenz (Vor-NR) — gleiches Layout wie processed.
        processed: Kandidat nach allen Nach-Schritten.
        sr: Abtastrate.
        edge_s: Zonenbreite in Sekunden (Default 0,5 s — wie der Edge-Taper).
        max_gain_db: erlaubte Maximal-Anhebung in dB (Default +2 dB).
        fade_s: Crossfade-Dauer an der Innenkante der Zone.

    Returns:
        np.ndarray — processed mit gekappten Randzonen (float32-Kopie).
    """
    try:
        ref = np.asarray(original, dtype=np.float32)
        out = np.asarray(processed, dtype=np.float32).copy()
        max_factor = float(10.0 ** (max_gain_db / 20.0))
        n_edge = int(edge_s * sr)
        n_fade = max(1, int(fade_s * sr))

        if ref.shape != out.shape or n_edge <= 0:
            if ref.ndim == 2 and out.ndim == 2 and ref.shape == out.shape[::-1]:
                # (C, N) vs (N, C) — Referenz transponieren statt Passthrough
                # (Stereo-Layout-Invariante AGENTS.md).
                ref = np.ascontiguousarray(np.transpose(ref))
            else:
                return out  # type: ignore[no-any-return]  # echter Mismatch/zu kurz → Passthrough (§V6-fail-open)

        # Zeitachse: Mono → 0; channels-first (2, N) → 1; channels-last (N, C) → 0
        if out.ndim == 2 and out.shape[0] <= 2 and out.shape[1] > 2:
            ch_first = True
            axis = 1
        else:
            ch_first = False
            axis = 0 if out.ndim == 2 else -1
        n_total = out.shape[axis]
        if n_total < n_edge * 4:
            return out  # type: ignore[no-any-return]  # zu kurz für Zonen-Definition

        for side in ("start", "end"):
            idx: Any
            if out.ndim == 1:
                idx = np.s_[:n_edge] if side == "start" else np.s_[-n_edge:]
            elif ch_first:
                idx = np.s_[:, :n_edge] if side == "start" else np.s_[:, -n_edge:]
            else:
                idx = np.s_[:n_edge, :] if side == "start" else np.s_[-n_edge:, :]
            rms_ref = float(np.sqrt(np.mean(ref[idx].astype(np.float64) ** 2) + 1e-15))
            rms_out = float(np.sqrt(np.mean(out[idx].astype(np.float64) ** 2) + 1e-15))
            if rms_out <= rms_ref * max_factor:
                continue
            gain = float((rms_ref * max_factor) / rms_out)  # < 1.0
            # 2 % Sicherheitsmarge unter die Bar: Crossfade-/Phasen-Interaktionen
            # können die Zonen-RMS sonst marginal über max_gain_db heben.
            gain *= 0.98
            zone = (out[idx] * gain).astype(np.float32)
            # Crossfade an der Innenkante zurück auf die REFERENZ (nicht auf
            # das ungekappte Signal!) — sonst hebt der Fade die Zonen-RMS
            # wieder über die Marge.
            zone_ref = ref[idx]
            if zone.shape[axis] > n_fade:
                fade = np.linspace(0.0, 1.0, n_fade, dtype=np.float32)
                if out.ndim == 1:
                    if side == "start":
                        zone[-n_fade:] = (zone[-n_fade:] * (1.0 - fade) + zone_ref[-n_fade:] * fade).astype(np.float32)
                    else:
                        zone[:n_fade] = (zone[:n_fade] * fade + zone_ref[:n_fade] * (1.0 - fade)).astype(np.float32)
                elif ch_first:  # channels-first (C, N)
                    if side == "start":
                        zone[:, -n_fade:] = (
                            zone[:, -n_fade:] * (1.0 - fade)[None, :] + zone_ref[:, -n_fade:] * fade[None, :]
                        ).astype(np.float32)
                    else:
                        zone[:, :n_fade] = (
                            zone[:, :n_fade] * fade[None, :] + zone_ref[:, :n_fade] * (1.0 - fade)[None, :]
                        ).astype(np.float32)
                else:  # channels-last (N, C)
                    if side == "start":
                        zone[-n_fade:, :] = (
                            zone[-n_fade:, :] * (1.0 - fade)[:, None] + zone_ref[-n_fade:, :] * fade[:, None]
                        ).astype(np.float32)
                    else:
                        zone[:n_fade, :] = (
                            zone[:n_fade, :] * fade[:, None] + zone_ref[:n_fade, :] * (1.0 - fade)[:, None]
                        ).astype(np.float32)
            out[idx] = zone
            logger.debug(
                "§2.46f Edge-Gain-Cap %s: gain=%.3f (out %.1f dB → cap %.1f dB über Referenz)",
                side,
                gain,
                20.0 * np.log10(rms_out + 1e-15),
                max_gain_db,
            )
        return out  # type: ignore[no-any-return]
    except Exception as _exc:
        logger.warning(
            "§2.46f Edge-Gain-Cap fehlgeschlagen (%s) — unverändert zurück (§V6 (copilot-instructions.md)).", _exc
        )
        return np.asarray(processed, dtype=np.float32)  # type: ignore[no-any-return]


def apply_edge_taper_convex(
    original: np.ndarray,
    processed: np.ndarray,
    sr: int,
    *,
    edge_s: float = _DEFAULT_EDGE_S,
) -> np.ndarray:
    """§2.46f Konvex-Edge-Taper: Randzonen konvex Richtung Original blenden.

    Genau der Taper des DSP-Pfades (0,5 s, lineare Fade) für Rückgabepfade,
    die die DSP-Kaskade umgehen (phase_03-ML-Pfad): Intro/Outro bleiben
    hochkorreliert mit dem Original und werden nie lauter (§0).
    Layout-sicher, deterministisch, §V6-fail-open.
    """
    try:
        ref = np.asarray(original, dtype=np.float32)
        out = np.asarray(processed, dtype=np.float32).copy()
        if ref.shape != out.shape:
            if ref.ndim == 2 and out.ndim == 2 and ref.shape == out.shape[::-1]:
                ref = np.ascontiguousarray(np.transpose(ref))
            else:
                return out  # type: ignore[no-any-return]
        n_edge = int(edge_s * sr)
        if out.ndim == 2 and out.shape[0] <= 2 and out.shape[1] > 2:
            axis = 1
        else:
            axis = 0 if out.ndim == 2 else -1
        n_total = out.shape[axis]
        if n_edge <= 0 or n_total < n_edge * 4:
            return out  # type: ignore[no-any-return]
        fade = np.linspace(0.0, 1.0, n_edge, dtype=np.float32)
        for side in ("start", "end"):
            idx: Any
            f: Any
            if out.ndim == 1:
                idx = np.s_[:n_edge] if side == "start" else np.s_[-n_edge:]
                f = fade if side == "start" else fade[::-1]
            elif axis == 1:
                idx = np.s_[:, :n_edge] if side == "start" else np.s_[:, -n_edge:]
                f = fade[None, :] if side == "start" else fade[::-1][None, :]
            else:
                idx = np.s_[:n_edge, :] if side == "start" else np.s_[-n_edge:, :]
                f = fade[:, None] if side == "start" else fade[::-1][:, None]
            out[idx] = (out[idx] * f + ref[idx] * (1.0 - f)).astype(np.float32)
        return out  # type: ignore[no-any-return]
    except Exception as _exc:
        logger.warning(
            "§2.46f Edge-Taper fehlgeschlagen (%s) — unverändert zurück (§V6 (copilot-instructions.md)).", _exc
        )
        return np.asarray(processed, dtype=np.float32)  # type: ignore[no-any-return]
