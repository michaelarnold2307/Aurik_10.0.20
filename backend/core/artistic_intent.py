"""
§v10 Emotional-Artistic Intent Modulator — Musikalische Ausrichtung auf das menschliche Gehör.

Ein menschlicher Toningenieur hört zuerst das GANZE und fragt:
- Welches Genre? Welche Epoche? Welche Emotion?
- Was ist das WICHTIGSTE Element? (Stimme, Gitarre, Groove?)
- Soll es warm/kuschelig oder brillant/aufregend klingen?
- Ist Dynamik erwünscht (Ballade) oder komprimiert (Rock)?

Dieses Modul übersetzt Genre/Epoche/Emotion in konkrete DSP-Parameter-Modifikatoren,
die JEDE Phase der Pipeline beeinflussen können. Es arbeitet NACH dem Defect-Scan
und VOR der Phasen-Ausführung — wie der „erste Eindruck" des Toningenieurs.

Wissenschaftliche Basis:
- Reiss, J. (2016): „A Meta-Analysis of High Resolution Audio Perceptual Evaluation"
- Maempel, H.J. (2017): „Auditory and multimodal exploration of music"
- Juslin, P.N. & Västfjäll, D. (2008): „Emotional responses to music"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Genre-Profile: „Was würde ein Toningenieur für dieses Genre tun?"
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ArtisticIntent:
    """Künstlerische Intention — abgeleitet aus Genre + Epoche + Medium."""

    # Dynamik-Präferenz
    preserve_dynamics: bool = True  # True = Dynamik erhalten, False = komprimieren
    dynamic_range_target_lu: float = 10.0  # Ziel-LRA in LU

    # Tonale Präferenz
    warmth_target: float = 0.70  # 0.0 = kalt/klinisch, 1.0 = maximal warm
    brilliance_target: float = 0.50  # 0.0 = dunkel, 1.0 = maximal brillant
    presence_boost_db: float = 0.5  # Präsenz-Anhebung (2–6 kHz)

    # Bass-Präferenz
    bass_extension_hz: float = 40.0  # Untere Grenzfrequenz
    bass_boost_db: float = 1.0  # Bass-Anhebung

    # Räumlichkeit
    stereo_width_target: float = 0.85  # 0.0 = Mono, 1.0 = maximal breit
    reverb_tolerance: float = 0.60  # 0.0 = trocken, 1.0 = hallig

    # Vocal-Präferenz
    vocal_forward_db: float = 0.5  # Vocal-Präsenz (relativ zum Mix)
    de_ess_strength: float = 0.50  # 0.0 = kein De-Essing, 1.0 = aggressiv

    # Transienten
    transient_preserve: bool = True  # True = Transienten erhalten
    attack_emphasis_db: float = 0.0  # Attack-Betonung

    # „First, do no harm" — Gesamt-Risikobereitschaft
    risk_tolerance: float = 0.30  # 0.0 = maximal konservativ, 1.0 = experimentell

    # Metadaten
    genre: str = "unknown"
    era_decade: int = 1980
    emotion: str = "neutral"
    notes: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# Genre → ArtisticIntent Mapping
# ═══════════════════════════════════════════════════════════════════════════

_GENRE_PROFILES: dict[str, ArtisticIntent] = {
    # ── Ruhige, emotionale Musik ──
    "ballad": ArtisticIntent(
        preserve_dynamics=True,
        dynamic_range_target_lu=12.0,
        warmth_target=0.85,
        brilliance_target=0.35,
        presence_boost_db=0.0,
        bass_extension_hz=35.0,
        bass_boost_db=0.5,
        stereo_width_target=0.90,
        reverb_tolerance=0.70,
        vocal_forward_db=1.5,
        de_ess_strength=0.35,
        transient_preserve=True,
        risk_tolerance=0.20,
        genre="ballad",
        emotion="intimate",
        notes="Ballade: Dynamik maximal erhalten, Stimme im Vordergrund, warm und intim",
    ),
    "classical": ArtisticIntent(
        preserve_dynamics=True,
        dynamic_range_target_lu=14.0,
        warmth_target=0.60,
        brilliance_target=0.40,
        presence_boost_db=-0.5,
        bass_extension_hz=30.0,
        bass_boost_db=0.0,
        stereo_width_target=0.95,
        reverb_tolerance=0.80,
        vocal_forward_db=0.0,
        de_ess_strength=0.20,
        transient_preserve=True,
        risk_tolerance=0.10,
        genre="classical",
        emotion="majestic",
        notes="Klassik: Maximale Dynamik, keine künstliche Präsenz, natürlicher Raumklang",
    ),
    "jazz": ArtisticIntent(
        preserve_dynamics=True,
        dynamic_range_target_lu=9.0,
        warmth_target=0.80,
        brilliance_target=0.45,
        presence_boost_db=0.3,
        bass_extension_hz=35.0,
        bass_boost_db=1.5,
        stereo_width_target=0.88,
        reverb_tolerance=0.65,
        vocal_forward_db=0.8,
        de_ess_strength=0.40,
        transient_preserve=True,
        risk_tolerance=0.25,
        genre="jazz",
        emotion="sophisticated",
        notes="Jazz: Warmer Kontrabass, natürliche Dynamik, luftige Höhen",
    ),
    # ── Pop / Mainstream ──
    "schlager": ArtisticIntent(
        preserve_dynamics=False,
        dynamic_range_target_lu=6.0,
        warmth_target=0.75,
        brilliance_target=0.55,
        presence_boost_db=1.0,
        bass_extension_hz=40.0,
        bass_boost_db=2.0,
        stereo_width_target=0.82,
        reverb_tolerance=0.50,
        vocal_forward_db=2.0,
        de_ess_strength=0.55,
        transient_preserve=False,
        risk_tolerance=0.30,
        genre="schlager",
        emotion="joyful",
        notes="Schlager: Melodische Lead-Stimme, Schunkelrhythmus, präsent und lebendig",
    ),
    "pop": ArtisticIntent(
        preserve_dynamics=False,
        dynamic_range_target_lu=6.0,
        warmth_target=0.60,
        brilliance_target=0.65,
        presence_boost_db=1.5,
        bass_extension_hz=35.0,
        bass_boost_db=2.5,
        stereo_width_target=0.85,
        reverb_tolerance=0.45,
        vocal_forward_db=1.5,
        de_ess_strength=0.55,
        transient_preserve=False,
        risk_tolerance=0.35,
        genre="pop",
        emotion="energetic",
        notes="Pop: Modern, brillant, druckvoll, Stimme präsent",
    ),
    "electronic": ArtisticIntent(
        preserve_dynamics=False,
        dynamic_range_target_lu=5.0,
        warmth_target=0.50,
        brilliance_target=0.70,
        presence_boost_db=2.0,
        bass_extension_hz=25.0,
        bass_boost_db=3.0,
        stereo_width_target=0.92,
        reverb_tolerance=0.40,
        vocal_forward_db=1.0,
        de_ess_strength=0.50,
        transient_preserve=False,
        risk_tolerance=0.40,
        genre="electronic",
        emotion="hypnotic",
        notes="Electronic: Tiefer Sub-Bass, breites Stereo,现代, präzise Transienten",
    ),
    # ── Rock / Energie ──
    "rock": ArtisticIntent(
        preserve_dynamics=False,
        dynamic_range_target_lu=5.0,
        warmth_target=0.65,
        brilliance_target=0.60,
        presence_boost_db=1.0,
        bass_extension_hz=40.0,
        bass_boost_db=2.0,
        stereo_width_target=0.83,
        reverb_tolerance=0.40,
        vocal_forward_db=1.0,
        de_ess_strength=0.45,
        transient_preserve=False,
        attack_emphasis_db=1.0,
        risk_tolerance=0.35,
        genre="rock",
        emotion="powerful",
        notes="Rock: Druckvoll, Gitarren präsent, Attack betont, druckvoller Bass",
    ),
    "metal": ArtisticIntent(
        preserve_dynamics=False,
        dynamic_range_target_lu=4.0,
        warmth_target=0.55,
        brilliance_target=0.65,
        presence_boost_db=1.5,
        bass_extension_hz=30.0,
        bass_boost_db=3.0,
        stereo_width_target=0.88,
        reverb_tolerance=0.30,
        vocal_forward_db=0.8,
        de_ess_strength=0.50,
        transient_preserve=False,
        attack_emphasis_db=2.0,
        risk_tolerance=0.40,
        genre="metal",
        emotion="aggressive",
        notes="Metal: Maximale Energie, präzise Kick, aggressive Gitarren",
    ),
    # ── Akustisch / Folk ──
    "folk": ArtisticIntent(
        preserve_dynamics=True,
        dynamic_range_target_lu=9.0,
        warmth_target=0.80,
        brilliance_target=0.40,
        presence_boost_db=0.3,
        bass_extension_hz=45.0,
        bass_boost_db=0.5,
        stereo_width_target=0.80,
        reverb_tolerance=0.55,
        vocal_forward_db=1.5,
        de_ess_strength=0.35,
        transient_preserve=True,
        risk_tolerance=0.20,
        genre="folk",
        emotion="earthy",
        notes="Folk: Natürlich, warm, authentisch, Stimme im Mittelpunkt",
    ),
    "blues": ArtisticIntent(
        preserve_dynamics=True,
        dynamic_range_target_lu=8.0,
        warmth_target=0.85,
        brilliance_target=0.35,
        presence_boost_db=0.0,
        bass_extension_hz=40.0,
        bass_boost_db=1.0,
        stereo_width_target=0.78,
        reverb_tolerance=0.55,
        vocal_forward_db=1.2,
        de_ess_strength=0.30,
        transient_preserve=True,
        risk_tolerance=0.20,
        genre="blues",
        emotion="soulful",
        notes="Blues: Warm, erdig, emotional, roh und authentisch",
    ),
    "hiphop": ArtisticIntent(
        preserve_dynamics=False,
        dynamic_range_target_lu=5.0,
        warmth_target=0.50,
        brilliance_target=0.60,
        presence_boost_db=1.0,
        bass_extension_hz=25.0,
        bass_boost_db=4.0,
        stereo_width_target=0.85,
        reverb_tolerance=0.35,
        vocal_forward_db=2.5,
        de_ess_strength=0.60,
        transient_preserve=False,
        attack_emphasis_db=1.5,
        risk_tolerance=0.35,
        genre="hiphop",
        emotion="confident",
        notes="Hip-Hop: Tiefer 808-Bass, Stimme extrem präsent, druckvoll",
    ),
    # ── Default / Unbekannt ──
    "reggae": ArtisticIntent(
        preserve_dynamics=True,
        dynamic_range_target_lu=8.0,
        warmth_target=0.75,
        brilliance_target=0.45,
        presence_boost_db=0.0,
        bass_extension_hz=40.0,
        bass_boost_db=2.5,
        stereo_width_target=0.85,
        reverb_tolerance=0.65,
        vocal_forward_db=0.5,
        de_ess_strength=0.30,
        transient_preserve=True,
        risk_tolerance=0.20,
        genre="reggae",
        emotion="laid_back",
        notes="Reggae: Tiefer, satter Bass, entspannt, moderate Dynamik",
    ),
    "latin": ArtisticIntent(
        preserve_dynamics=True,
        dynamic_range_target_lu=7.0,
        warmth_target=0.70,
        brilliance_target=0.55,
        presence_boost_db=0.8,
        bass_extension_hz=35.0,
        bass_boost_db=1.5,
        stereo_width_target=0.88,
        reverb_tolerance=0.60,
        vocal_forward_db=1.0,
        de_ess_strength=0.35,
        transient_preserve=True,
        risk_tolerance=0.15,
        genre="latin",
        emotion="passionate",
        notes="Latin: Lebhaft, perkussiv, präsente Vocals, warme Mitten",
    ),
    "gospel": ArtisticIntent(
        preserve_dynamics=True,
        dynamic_range_target_lu=9.0,
        warmth_target=0.80,
        brilliance_target=0.50,
        presence_boost_db=0.3,
        bass_extension_hz=32.0,
        bass_boost_db=1.0,
        stereo_width_target=0.90,
        reverb_tolerance=0.70,
        vocal_forward_db=1.8,
        de_ess_strength=0.25,
        transient_preserve=True,
        risk_tolerance=0.15,
        genre="gospel",
        emotion="soulful",
        notes="Gospel: Kraftvolle Stimmen, viel Raum, warm, emotional",
    ),
    "country": ArtisticIntent(
        preserve_dynamics=True,
        dynamic_range_target_lu=7.0,
        warmth_target=0.70,
        brilliance_target=0.50,
        presence_boost_db=0.5,
        bass_extension_hz=38.0,
        bass_boost_db=0.8,
        stereo_width_target=0.82,
        reverb_tolerance=0.55,
        vocal_forward_db=1.2,
        de_ess_strength=0.35,
        transient_preserve=True,
        risk_tolerance=0.20,
        genre="country",
        emotion="heartfelt",
        notes="Country: Erzählende Stimme im Vordergrund, natürlich, bodenständig",
    ),
    "funk": ArtisticIntent(
        preserve_dynamics=True,
        dynamic_range_target_lu=6.0,
        warmth_target=0.72,
        brilliance_target=0.55,
        presence_boost_db=0.7,
        bass_extension_hz=30.0,
        bass_boost_db=2.0,
        stereo_width_target=0.85,
        reverb_tolerance=0.50,
        vocal_forward_db=0.5,
        de_ess_strength=0.40,
        transient_preserve=True,
        risk_tolerance=0.20,
        genre="funk",
        emotion="groovy",
        notes="Funk: Tight, perkussiv, satter Bass, punchy Drums",
    ),
    "ambient": ArtisticIntent(
        preserve_dynamics=False,
        dynamic_range_target_lu=10.0,
        warmth_target=0.65,
        brilliance_target=0.50,
        presence_boost_db=-0.2,
        bass_extension_hz=25.0,
        bass_boost_db=0.5,
        stereo_width_target=0.95,
        reverb_tolerance=0.90,
        vocal_forward_db=-0.5,
        de_ess_strength=0.15,
        transient_preserve=False,
        risk_tolerance=0.30,
        genre="ambient",
        emotion="atmospheric",
        notes="Ambient: Weit, atmosphärisch, keine harten Transienten",
    ),
    "world": ArtisticIntent(
        preserve_dynamics=True,
        dynamic_range_target_lu=8.0,
        warmth_target=0.75,
        brilliance_target=0.45,
        presence_boost_db=0.2,
        bass_extension_hz=35.0,
        bass_boost_db=1.0,
        stereo_width_target=0.88,
        reverb_tolerance=0.65,
        vocal_forward_db=0.8,
        de_ess_strength=0.30,
        transient_preserve=True,
        risk_tolerance=0.20,
        genre="world",
        emotion="authentic",
        notes="World: Authentische Instrumente, moderate Bearbeitung, natürlicher Klang",
    ),
    "unknown": ArtisticIntent(
        preserve_dynamics=True,
        dynamic_range_target_lu=8.0,
        warmth_target=0.65,
        brilliance_target=0.50,
        presence_boost_db=0.5,
        bass_extension_hz=40.0,
        bass_boost_db=1.0,
        stereo_width_target=0.85,
        reverb_tolerance=0.55,
        vocal_forward_db=0.8,
        de_ess_strength=0.50,
        transient_preserve=True,
        risk_tolerance=0.20,
        genre="unknown",
        emotion="neutral",
        notes='Default: Konservativ — "First, do no harm". Minimale Eingriffe.',
    ),
}


def get_artistic_intent(
    genre: str | None = None,
    era_decade: int | None = None,
    material: str | None = None,
    vocals_detected: bool = True,
) -> ArtisticIntent:
    """Ermittelt die künstlerische Intention aus Genre + Epoche + Medium.

    Wie ein Toningenieur, der zuerst fragt: „Welche Musik ist das?"

    Args:
        genre:     Genre-String (z.B. „schlager", „rock", „jazz")
        era_decade: Jahrzehnt (z.B. 1970, 1980, 2000)
        material:  Trägermedium (z.B. „vinyl", „mp3_low")
        vocals_detected: Ob Gesang erkannt wurde

    Returns:
        ArtisticIntent mit konkreten DSP-Parameter-Modifikatoren
    """
    genre_key = (genre or "unknown").lower().strip()
    # Normalisiere Genre-Namen
    genre_aliases = {
        "classic": "classical",
        "orchestral": "classical",
        "schlager_pop": "schlager",
        "deutsch_pop": "schlager",
        "country": "folk",
        "singer_songwriter": "folk",
        "acoustic": "folk",
        "indie": "rock",
        "alternative": "rock",
        "punk": "rock",
        "hard_rock": "rock",
        "heavy_metal": "metal",
        "death_metal": "metal",
        "edm": "electronic",
        "techno": "electronic",
        "house": "electronic",
        "trance": "electronic",
        "dubstep": "electronic",
        "rap": "hiphop",
        "trap": "hiphop",
        "rnb": "pop",
        "soul": "blues",
        "funk": "jazz",
        "latin": "pop",
        "reggae": "folk",
    }
    genre_key = genre_aliases.get(genre_key, genre_key)

    profile = _GENRE_PROFILES.get(genre_key, _GENRE_PROFILES["unknown"])

    # Epoche-Anpassungen
    if era_decade is not None:
        if era_decade < 1960:
            # Mono-Ära: schmaleres Stereo, mehr Wärme, weniger Brillanz
            profile.stereo_width_target = min(profile.stereo_width_target, 0.70)
            profile.warmth_target = min(1.0, profile.warmth_target + 0.10)
            profile.brilliance_target = max(0.20, profile.brilliance_target - 0.10)
            profile.risk_tolerance = max(0.10, profile.risk_tolerance - 0.10)
        elif era_decade < 1980:
            # Analog-Ära: moderate Anpassungen
            profile.warmth_target = min(1.0, profile.warmth_target + 0.05)
        elif era_decade >= 2000:
            # Digital-Ära: mehr Brillanz, mehr Kompression toleriert
            profile.brilliance_target = min(1.0, profile.brilliance_target + 0.05)
            profile.preserve_dynamics = False

    # Material-Anpassungen
    if material:
        mat = str(material).lower()
        if any(t in mat for t in ("vinyl", "shellac", "wax", "lacquer")):
            profile.warmth_target = min(1.0, profile.warmth_target + 0.10)
            profile.risk_tolerance = max(0.10, profile.risk_tolerance - 0.05)
        elif any(t in mat for t in ("mp3", "aac", "streaming")):
            profile.brilliance_target = max(0.30, profile.brilliance_target - 0.05)
            profile.risk_tolerance = min(0.50, profile.risk_tolerance + 0.05)

    # Ohne Gesang: Vocal-Präferenzen neutralisieren
    if not vocals_detected:
        profile.vocal_forward_db = 0.0
        profile.de_ess_strength = 0.30

    # „First, do no harm": Bei unbekanntem Genre IMMER konservativ
    if genre_key == "unknown":
        profile.risk_tolerance = 0.20

    profile.era_decade = era_decade or 1980

    logger.info(
        "ArtisticIntent: genre=%s era=%s material=%s vocals=%s → warmth=%.2f brilliance=%.2f dynamics=%s risk=%.2f",
        genre_key,
        era_decade,
        material,
        vocals_detected,
        profile.warmth_target,
        profile.brilliance_target,
        profile.preserve_dynamics,
        profile.risk_tolerance,
    )

    return profile


def apply_artistic_intent_to_params(
    intent: ArtisticIntent,
    base_params: dict[str, float],
) -> dict[str, float]:
    """Modifiziert DSP-Parameter basierend auf der künstlerischen Intention.

    Args:
        intent:      Künstlerische Intention (von get_artistic_intent())
        base_params: Basis-Parameter (z.B. von SelfLearningOptimizer)

    Returns:
        Modifizierte Parameter
    """
    params = dict(base_params)

    # Dynamik-Präferenz
    if not intent.preserve_dynamics:
        params["compression_ratio"] = params.get("compression_ratio", 1.5) * 1.2
        params["noise_reduction_strength"] = min(0.95, params.get("noise_reduction_strength", 0.5) * 1.1)

    # Tonale Präferenz
    params["warmth_target"] = intent.warmth_target
    params["brilliance_target"] = intent.brilliance_target
    params["presence_boost_db"] = intent.presence_boost_db

    # Bass-Präferenz
    params["bass_boost_db"] = intent.bass_boost_db

    # Räumlichkeit
    params["stereo_width_target"] = intent.stereo_width_target

    # Vocal-Präferenz
    params["vocal_forward_db"] = intent.vocal_forward_db
    params["de_ess_strength"] = intent.de_ess_strength

    # Transienten
    params["transient_preserve"] = 1.0 if intent.transient_preserve else 0.3
    params["attack_emphasis_db"] = intent.attack_emphasis_db

    # Risiko-Toleranz beeinflusst alle Stärken
    risk_factor = 0.7 + 0.3 * intent.risk_tolerance
    for key in list(params.keys()):
        if key.endswith("_strength") or key.endswith("_boost_db"):
            params[key] = float(np.clip(params[key] * risk_factor, 0.0, params.get(key, 1.0) * 1.5))

    return params
