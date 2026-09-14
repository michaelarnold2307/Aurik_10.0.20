"""§SOTA-PSY-A8: Generische Hör-JND-Gate-Tabelle.

Hörbarkeits-JNDs (Just-Noticeable Differences) aus der Literatur als zentrale
Konstante für Never-worsen-Gates — statt willkürlicher absoluter dB-Grenzen.
Quellen: Zwicker/Fastl „Psychoacoustics“ (3. Aufl.), Moore „Introduction to
the Psychology of Hearing“, Mills (1960) für Pegel, Klumpp & Eady (1956) für
Frequenz, interaural_cues.py (ITD/ILD) für räumliche Größen.

Bestandteil der SOTA-Expansionsmatrix (Roadmap, PSY-A8, 2026-09-14).
Deterministisch und rein funktional (§G5 (GEBOTE.md)).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Jnd:
    value: float
    unit: str
    source: str


# Zentrales JND-Register (Hör-JNDs; Warn-/Hard-Fail-Abstufung dokumentiert).
_JNDS: dict[str, Jnd] = {
    "level_broadband": Jnd(1.0, "dB", "Mills 1960 (Breitband-Pegel ±1 dB)"),
    "level_tone_1khz": Jnd(0.5, "dB", "Zwicker/Fastl: Ton 1 kHz ±0,5 dB"),
    "frequency_1khz": Jnd(0.002, "rel", "Klumpp & Eady 1956: Δf/f ≈ 0,2 % @1 kHz"),
    "time_gap": Jnd(0.005, "s", "Zwicker/Fastl: Lücken-Detektion ≈ 5 ms"),
    "pan_angle": Jnd(2.0, "deg", "Mills 1958: laterale Winkelauflösung ≈ 2°"),
    "itd_tone": Jnd(30e-6, "s", "Hörordnung §8b (500 Hz–1 kHz)"),
    "ild_tone": Jnd(1.0, "dB", "Mills 1960 (Hörordnung §8b)"),
    "iacc": Jnd(0.08, "1", "Hörordnung §8b (räumliche Bündelung)"),
    "loudness_ratio": Jnd(0.25, "LU", "Zwicker/Fastl: Lautheitsverhältnis ≈ 25 %"),
}


def jnd(kind: str) -> float:
    """JND-Wert zu einer Fähigkeit; KeyError bei unbekannter ID (fail-closed)."""
    return _JNDS[kind].value


def jnd_info(kind: str) -> Jnd:
    return _JNDS[kind]


def below_jnd(delta: float, kind: str, factor: float = 1.0) -> bool:
    """Liegt |delta| unter der Hörschwelle (JND × factor)?

    factor < 1 = konservativ (strenger), > 1 = tolerant.
    """
    if delta != delta:  # NaN → als hörbar behandeln (fail-safe)
        return False
    return abs(delta) <= jnd(kind) * factor


def all_jnd_kinds() -> list[str]:
    return sorted(_JNDS.keys())
