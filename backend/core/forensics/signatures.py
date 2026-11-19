"""
sota/analysis/forensics/signatures.py
Akustische Fingerabdrücke aller Tonträger
"""

from dataclasses import dataclass, field
from enum import Enum, auto


class MediaCategory(Enum):
    """Hauptkategorien."""

    MECHANICAL = auto()  # Zylinder, frühe Schellack
    VINYL = auto()  # Alle Schallplatten
    TAPE_REEL = auto()  # Offene Bandmaschinen
    TAPE_CASSETTE = auto()  # Kassetten
    DIGITAL_PCM = auto()  # CD, DAT, etc.
    DIGITAL_LOSSY = auto()  # MP3, AAC, etc.
    BROADCAST = auto()  # Radio
    FILM = auto()  # Film/Video
    TELEPHONE = auto()  # Telefon


class EraType(Enum):
    """Aufnahmeepochen mit charakteristischen Merkmalen."""

    ERA_1950s = "1950s"
    ERA_1960s = "1960s"
    ERA_1970s = "1970s"
    ERA_1980s = "1980s"
    ERA_1990s = "1990s"
    ERA_2000s = "2000s"
    ERA_2010s = "2010s"
    ERA_2020s = "2020s"
    UNKNOWN = "unknown"


class MediaType(Enum):
    """Detaillierte Medientypen."""

    # Dokumentationskonforme Haupttypen als Alias für Mapping
    VINYL = 1001
    TAPE = 1002
    CASSETTE = 1003
    CD = 1004
    DIGITAL_NATIVE = 1005
    RADIO_BROADCAST = 1006
    # UNKNOWN bleibt am Ende eindeutig
    # ...alle bestehenden detaillierten Typen bleiben erhalten...
    # UNKNOWN nur einmal definieren:
    # (Falls oben schon vorhanden, hier nicht erneut definieren)
    CYLINDER_EDISON = auto()
    CYLINDER_PATHE = auto()
    SHELLAC_ACOUSTIC = auto()
    SHELLAC_ELECTRIC = auto()
    VINYL_LP_MONO = auto()
    VINYL_LP_STEREO = auto()
    VINYL_LP_QUAD = auto()  # DETECTION ONLY - Quadraphonic (4ch) not supported for processing
    VINYL_45_MONO = auto()
    VINYL_45_STEREO = auto()
    VINYL_DIRECT_TO_DISC = auto()
    FLEXI_DISC = auto()
    WIRE_RECORDING = auto()
    TAPE_30IPS = auto()
    TAPE_15IPS = auto()
    TAPE_7_5IPS = auto()
    TAPE_3_75IPS = auto()
    TAPE_1_875IPS = auto()
    CASSETTE_TYPE_I = auto()
    CASSETTE_TYPE_II = auto()
    CASSETTE_TYPE_IV = auto()
    CASSETTE_DOLBY_B = auto()
    CASSETTE_DOLBY_C = auto()
    CASSETTE_DOLBY_S = auto()
    CASSETTE_DBX = auto()
    EIGHT_TRACK = auto()
    ELCASET = auto()
    MICROCASSETTE = auto()
    CD_STANDARD = auto()
    CD_HDCD = auto()
    DAT_48K = auto()
    DAT_44K = auto()
    DAT_32K = auto()
    DCC = auto()
    MINIDISC = auto()
    MINIDISC_HIMD = auto()
    ADAT = auto()
    DVD_AUDIO = auto()
    SACD_DSD = auto()
    HIRES_PCM = auto()
    MP3_128 = auto()
    MP3_192 = auto()
    MP3_256 = auto()
    MP3_320 = auto()
    MP3_VBR = auto()
    AAC_128 = auto()
    AAC_256 = auto()
    OGG_VORBIS = auto()
    WMA = auto()
    ATRAC_SP = auto()
    ATRAC_LP2 = auto()
    ATRAC_LP4 = auto()
    OPUS = auto()
    AC3 = auto()
    DTS = auto()  # DETECTION ONLY - DTS 5.1/7.1 not supported, only DTS 2.0 stereo
    AM_MW = auto()
    AM_SW = auto()
    FM_MONO = auto()
    FM_STEREO = auto()
    DAB = auto()
    DAB_PLUS = auto()
    SATELLITE_RADIO = auto()
    INTERNET_STREAM = auto()
    OPTICAL_MONO = auto()
    OPTICAL_STEREO = auto()
    DOLBY_STEREO = auto()
    DOLBY_SR = auto()
    VHS_LINEAR = auto()
    VHS_HIFI = auto()
    BETAMAX = auto()
    LASERDISC = auto()
    PSTN = auto()
    GSM = auto()
    VOIP = auto()
    UNKNOWN = auto()


@dataclass
class MediaSignature:
    """
    Akustische Signatur eines Mediums.
    Jedes Medium hat charakteristische Merkmale.
    """

    # Pflichtfelder zuerst
    media_type: MediaType
    category: MediaCategory
    freq_low_hz: float
    freq_high_hz: float
    freq_rolloff_db_oct: float
    noise_floor_db: float
    noise_type: str
    dynamic_range_db: tuple[float, float]
    headroom_db: float
    stereo: bool = True

    # Optionale Felder mit sinnvollen Defaults
    pilot_tone_hz: float | None = None
    notch_frequencies: list[float] = field(default_factory=list)
    resonance_frequencies: list[float] = field(default_factory=list)
    eq_curve: str | None = None
    pre_emphasis_us: float | None = None
    hiss_spectrum: str | None = None
    rumble_hz: float | None = None
    hum_frequencies: list[float] = field(default_factory=list)
    clicks_per_minute: tuple[float, float] | None = None
    click_spectrum: str | None = None
    wow_hz: tuple[float, float] | None = None
    flutter_hz: tuple[float, float] | None = None
    scrape_flutter_hz: float | None = None
    codec_artifacts: list[str] | None = field(default_factory=list)
    compression_artifacts: bool = False
    channel_separation_db: float | None = None
    crosstalk_pattern: str | None = None
    era_range: tuple[int, int] = (1900, 2100)
    unique_markers: list[str] = field(default_factory=list)
    detection_weights: dict[str, float] = field(default_factory=dict)


@dataclass
class EraSignature:
    """
    Akustische Signatur einer Aufnahmeepoche.
    Jede Era hat charakteristische technische Merkmale.
    """

    era_type: EraType
    year_range: tuple[int, int]
    typical_media: list[MediaType]
    freq_bandwidth_hz: tuple[float, float]  # (low_cutoff, high_cutoff)
    dynamic_range_db: tuple[float, float]  # (typical_min, typical_max)
    noise_floor_db: tuple[float, float]  # (typical_min, typical_max)
    typical_artifacts: list[str]
    recording_technique: str
    mastering_style: str
    typical_eq_curves: list[str]
    mono_stereo_distribution: dict[str, float]  # {"mono": 0.7, "stereo": 0.3}
    loudness_range_lufs: tuple[float, float]
    peak_limiting: bool
    brick_wall_limiting: bool
    unique_markers: list[str]
    detection_weights: dict[str, float] = field(default_factory=dict)


# ...SIGNATUR-DATENBANK...
MEDIA_SIGNATURES: dict[MediaType, MediaSignature] = {
    MediaType.VINYL: MediaSignature(
        media_type=MediaType.VINYL,
        category=MediaCategory.VINYL,
        freq_low_hz=20,
        freq_high_hz=18000,
        freq_rolloff_db_oct=-18,
        noise_floor_db=-55,
        noise_type="rumble/hiss",
        dynamic_range_db=(45, 65),
        headroom_db=12,
        stereo=True,
        eq_curve="RIAA",
        rumble_hz=20,
        clicks_per_minute=(0, 10),
        wow_hz=(0.1, 0.5),
        flutter_hz=(0.2, 1.0),
        resonance_frequencies=[50, 3150],
        notch_frequencies=[60, 120],
        channel_separation_db=30,
        era_range=(1950, 2026),
        unique_markers=["Knackser", "Oberflächenrauschen", "RIAA-Kurve"],
        detection_weights={"spectral_rolloff": 0.7, "noise_floor": 0.6, "clicks": 0.5},
    ),
    MediaType.TAPE: MediaSignature(
        media_type=MediaType.TAPE,
        category=MediaCategory.TAPE_REEL,
        freq_low_hz=30,
        freq_high_hz=16000,
        freq_rolloff_db_oct=-12,
        noise_floor_db=-50,
        noise_type="hiss",
        dynamic_range_db=(40, 60),
        headroom_db=10,
        stereo=True,
        pilot_tone_hz=15000,
        wow_hz=(0.2, 0.8),
        flutter_hz=(0.5, 2.0),
        resonance_frequencies=[3150],
        notch_frequencies=[50, 100],
        channel_separation_db=25,
        era_range=(1955, 2026),
        unique_markers=["Bandrauschen", "Pilotton", "Spulenwechsel"],
        detection_weights={"spectral_rolloff": 0.6, "noise_floor": 0.7, "flutter": 0.5},
    ),
    MediaType.CASSETTE: MediaSignature(
        media_type=MediaType.CASSETTE,
        category=MediaCategory.TAPE_CASSETTE,
        freq_low_hz=40,
        freq_high_hz=14000,
        freq_rolloff_db_oct=-10,
        noise_floor_db=-45,
        noise_type="hiss",
        dynamic_range_db=(35, 55),
        headroom_db=8,
        stereo=True,
        pre_emphasis_us=120,
        wow_hz=(0.3, 1.0),
        flutter_hz=(1.0, 3.0),
        resonance_frequencies=[3150],
        notch_frequencies=[50, 120],
        channel_separation_db=20,
        era_range=(1963, 2026),
        unique_markers=["Dolby NR", "Kassettenrauschen", "Vorband"],
        detection_weights={"spectral_rolloff": 0.5, "noise_floor": 0.8, "flutter": 0.6},
    ),
    MediaType.CD: MediaSignature(
        media_type=MediaType.CD,
        category=MediaCategory.DIGITAL_PCM,
        freq_low_hz=20,
        freq_high_hz=22050,
        freq_rolloff_db_oct=0,
        noise_floor_db=-96,
        noise_type="none",
        dynamic_range_db=(90, 96),
        headroom_db=0,
        stereo=True,
        channel_separation_db=90,
        era_range=(1982, 2026),
        unique_markers=["TOC", "PCM", "keine Artefakte"],
        detection_weights={"spectral_rolloff": 0.9, "noise_floor": 0.9},
    ),
    MediaType.DIGITAL_NATIVE: MediaSignature(
        media_type=MediaType.DIGITAL_NATIVE,
        category=MediaCategory.DIGITAL_LOSSY,
        freq_low_hz=20,
        freq_high_hz=22050,
        freq_rolloff_db_oct=0,
        noise_floor_db=-90,
        noise_type="none",
        dynamic_range_db=(80, 90),
        headroom_db=0,
        stereo=True,
        codec_artifacts=["blockiness", "pre-echo"],
        compression_artifacts=True,
        channel_separation_db=80,
        era_range=(1995, 2026),
        unique_markers=["Bitrate", "Codec", "Artefakte"],
        detection_weights={"spectral_rolloff": 0.8, "noise_floor": 0.8, "codec_artifacts": 0.7},
    ),
    MediaType.RADIO_BROADCAST: MediaSignature(
        media_type=MediaType.RADIO_BROADCAST,
        category=MediaCategory.BROADCAST,
        freq_low_hz=50,
        freq_high_hz=15000,
        freq_rolloff_db_oct=-8,
        noise_floor_db=-40,
        noise_type="static/hiss",
        dynamic_range_db=(20, 40),
        headroom_db=5,
        stereo=False,
        channel_separation_db=5,
        era_range=(1920, 2026),
        unique_markers=["Mono", "Rauschen", "Senderkennung"],
        detection_weights={"spectral_rolloff": 0.4, "noise_floor": 0.6, "static": 0.7},
    ),
}

# Era-Signaturen für zeitliche Einordnung
ERA_SIGNATURES: dict[EraType, EraSignature] = {
    EraType.ERA_1950s: EraSignature(
        era_type=EraType.ERA_1950s,
        year_range=(1950, 1959),
        typical_media=[MediaType.VINYL, MediaType.TAPE, MediaType.RADIO_BROADCAST],
        freq_bandwidth_hz=(80, 12000),  # Eingeschränkte Bandbreite
        dynamic_range_db=(30, 50),  # Begrenzte Dynamik
        noise_floor_db=(-45, -35),  # Hoher Noise-Floor
        typical_artifacts=["clicks", "rumble", "wow", "flutter", "hum"],
        recording_technique="Tube microphones, mono",
        mastering_style="Minimal compression, natural dynamics",
        typical_eq_curves=["RIAA", "flat"],
        mono_stereo_distribution={"mono": 0.95, "stereo": 0.05},
        loudness_range_lufs=(-18, -12),  # Moderate Lautheit
        peak_limiting=False,
        brick_wall_limiting=False,
        unique_markers=["Mono dominant", "Tube warmth", "Natural room ambience", "Minimal processing"],
        detection_weights={"bandwidth": 0.8, "mono_stereo": 0.9, "noise_floor": 0.7, "dynamics": 0.6},
    ),
    EraType.ERA_1960s: EraSignature(
        era_type=EraType.ERA_1960s,
        year_range=(1960, 1969),
        typical_media=[MediaType.VINYL, MediaType.TAPE, MediaType.CASSETTE],
        freq_bandwidth_hz=(70, 14000),  # Etwas bessere Bandbreite
        dynamic_range_db=(35, 55),  # Verbesserte Dynamik
        noise_floor_db=(-50, -40),
        typical_artifacts=["tape hiss", "clicks", "wow", "flutter"],
        recording_technique="4-track tape, early stereo",
        mastering_style="Moderate compression, beginning of stereo",
        typical_eq_curves=["RIAA", "NAB", "flat"],
        mono_stereo_distribution={"mono": 0.6, "stereo": 0.4},
        loudness_range_lufs=(-16, -10),
        peak_limiting=False,
        brick_wall_limiting=False,
        unique_markers=["Stereo emergence", "4-track tape", "Moderate HF rolloff", "Early multitracking"],
        detection_weights={"bandwidth": 0.7, "mono_stereo": 0.8, "tape_hiss": 0.7, "dynamics": 0.6},
    ),
    EraType.ERA_1970s: EraSignature(
        era_type=EraType.ERA_1970s,
        year_range=(1970, 1979),
        typical_media=[MediaType.VINYL, MediaType.TAPE, MediaType.CASSETTE],
        freq_bandwidth_hz=(50, 16000),  # Bessere Bandbreite
        dynamic_range_db=(40, 60),  # Gute Dynamik
        noise_floor_db=(-55, -45),
        typical_artifacts=["tape hiss", "vinyl noise", "flutter"],
        recording_technique="16/24-track tape, stereo standard",
        mastering_style="Moderate compression, analog warmth",
        typical_eq_curves=["RIAA", "NAB", "IEC"],
        mono_stereo_distribution={"mono": 0.2, "stereo": 0.8},
        loudness_range_lufs=(-14, -8),
        peak_limiting=False,
        brick_wall_limiting=False,
        unique_markers=["Stereo dominant", "24-track tape", "Analog warmth", "Wide dynamic range"],
        detection_weights={"bandwidth": 0.6, "stereo_width": 0.8, "dynamics": 0.8, "warmth": 0.7},
    ),
    EraType.ERA_1980s: EraSignature(
        era_type=EraType.ERA_1980s,
        year_range=(1980, 1989),
        typical_media=[MediaType.CD, MediaType.VINYL, MediaType.CASSETTE, MediaType.DAT_48K],
        freq_bandwidth_hz=(30, 18000),  # CD-Era beginnt
        dynamic_range_db=(45, 70),  # Digital ermöglicht mehr Dynamik
        noise_floor_db=(-70, -50),  # Digitaler Noise-Floor
        typical_artifacts=["digital artifacts", "tape hiss", "clicks"],
        recording_technique="Digital multitrack, early CD mastering",
        mastering_style="Early digital compression, gate reverb",
        typical_eq_curves=["RIAA", "flat digital"],
        mono_stereo_distribution={"mono": 0.05, "stereo": 0.95},
        loudness_range_lufs=(-12, -6),  # Loudness War beginnt
        peak_limiting=True,
        brick_wall_limiting=False,
        unique_markers=["CD standard", "Digital reverb", "Gate effects", "Early digital artifacts"],
        detection_weights={"bandwidth": 0.7, "noise_floor": 0.8, "digital_artifacts": 0.7, "loudness": 0.6},
    ),
    EraType.ERA_1990s: EraSignature(
        era_type=EraType.ERA_1990s,
        year_range=(1990, 1999),
        typical_media=[MediaType.CD, MediaType.DIGITAL_NATIVE, MediaType.MP3_128],
        freq_bandwidth_hz=(20, 20000),  # Volle CD-Bandbreite
        dynamic_range_db=(40, 65),  # Loudness War intensiviert
        noise_floor_db=(-85, -60),
        typical_artifacts=["digital artifacts", "mp3 artifacts", "clipping"],
        recording_technique="DAW-based, digital effects",
        mastering_style="Increased compression, loudness maximization",
        typical_eq_curves=["flat digital", "hyped"],
        mono_stereo_distribution={"mono": 0.02, "stereo": 0.98},
        loudness_range_lufs=(-10, -4),  # Loudness War Höhepunkt
        peak_limiting=True,
        brick_wall_limiting=True,
        unique_markers=["MP3 era", "Loudness maximization", "Digital compression", "Brick-wall limiting"],
        detection_weights={"loudness": 0.9, "compression": 0.8, "mp3_artifacts": 0.7, "dynamics": 0.6},
    ),
    EraType.ERA_2000s: EraSignature(
        era_type=EraType.ERA_2000s,
        year_range=(2000, 2009),
        typical_media=[MediaType.DIGITAL_NATIVE, MediaType.MP3_320, MediaType.AAC_256],
        freq_bandwidth_hz=(20, 22050),  # Volle digitale Bandbreite
        dynamic_range_db=(30, 55),  # Peak der Loudness War
        noise_floor_db=(-90, -70),
        typical_artifacts=["clipping", "mp3 artifacts", "over-compression"],
        recording_technique="Digital DAW, plugins dominant",
        mastering_style="Extreme loudness, heavy limiting",
        typical_eq_curves=["hyped", "V-curve"],
        mono_stereo_distribution={"mono": 0.01, "stereo": 0.99},
        loudness_range_lufs=(-8, -2),  # Extremste Loudness
        peak_limiting=True,
        brick_wall_limiting=True,
        unique_markers=["Loudness War peak", "Extreme compression", "Digital perfection", "Clipping common"],
        detection_weights={"loudness": 1.0, "compression": 0.9, "clipping": 0.8, "dynamics": 0.5},
    ),
    EraType.ERA_2010s: EraSignature(
        era_type=EraType.ERA_2010s,
        year_range=(2010, 2019),
        typical_media=[MediaType.DIGITAL_NATIVE, MediaType.HIRES_PCM, MediaType.INTERNET_STREAM],
        freq_bandwidth_hz=(20, 22050),  # Digital standard
        dynamic_range_db=(35, 65),  # Leichte Erholung
        noise_floor_db=(-96, -80),  # Hi-Res Audio
        typical_artifacts=["streaming artifacts", "light compression"],
        recording_technique="Digital DAW, cloud collaboration",
        mastering_style="Louder but controlled, streaming optimization",
        typical_eq_curves=["balanced", "slight V-curve"],
        mono_stereo_distribution={"mono": 0.01, "stereo": 0.99},
        loudness_range_lufs=(-10, -5),  # Leichte Erholung
        peak_limiting=True,
        brick_wall_limiting=True,
        unique_markers=["Streaming era", "LUFS normalization", "Hi-Res audio", "Controlled loudness"],
        detection_weights={"loudness": 0.8, "streaming": 0.8, "hires": 0.7, "dynamics": 0.7},
    ),
    EraType.ERA_2020s: EraSignature(
        era_type=EraType.ERA_2020s,
        year_range=(2020, 2029),
        typical_media=[MediaType.DIGITAL_NATIVE, MediaType.HIRES_PCM, MediaType.INTERNET_STREAM],
        freq_bandwidth_hz=(20, 96000),  # Hi-Res standard
        dynamic_range_db=(40, 75),  # Renaissance der Dynamik
        noise_floor_db=(-110, -90),  # Extrem niedrig
        typical_artifacts=["minimal", "very clean"],
        recording_technique="Digital DAW, AI-assisted",  # Note: Atmos metadata detection only, no processing
        mastering_style="Dynamic, LUFS-aware, multiple masters",
        typical_eq_curves=["balanced", "transparent"],
        mono_stereo_distribution={"mono": 0.01, "stereo": 0.99},
        loudness_range_lufs=(-14, -8),  # EBU R128 / Streaming standards
        peak_limiting=True,
        brick_wall_limiting=False,
        unique_markers=["LUFS standardization", "Dynamic renaissance", "Immersive audio", "AI-assisted"],
        detection_weights={"loudness": 0.7, "dynamics": 0.9, "hires": 0.9, "cleanliness": 0.9},
    ),
}
