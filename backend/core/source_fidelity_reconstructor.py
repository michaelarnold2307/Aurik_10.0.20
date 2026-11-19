"""
SourceFidelityReconstructor — Klangtreue zum Aufnahmetag
=========================================================

Modelliert die Signallücke zwischen dem degradierten Ist-Zustand eines Tonträgers
und dem geschätzten Original-Klangbild zum Tag der Aufnahme.

Kernprinzip (§11.7a Denker-Rollendifferenzierung):
  Aurik restauriert nicht auf den Klang des letzten Tonträgers, sondern auf
  den Klang wie das Original am Tag der Aufnahme erklang — also auf die Quelle,
  nicht auf den Träger.

Modellierte Verlustquellen in der Tonträgerkette:
  1. Anzahl der analogen Überspielgenerationen → akkumulierter HF-Verlust,
     Dynamik-Komprimierung, Sättigungsakkumulation
  2. Ära-spezifische Original-Bandbreite (Mikrofon-/Gesamt-Kette am Tag der Aufnahme)
  3. Materialspezifische Übertragungsverluste der Kette (nicht nur der letzten Stufe)
  4. Rekonstruktionsziel für Phase 06 (Bandbreite), ExcellenceOptimizer
     (Oberton-Dichte), SongCalibrationProfile (reconstruction-Familien-Scalar)

Wissenschaftliche Grundlagen:
  - Eargle 2004: «The Microphone Book» — Mikrofon-Frequenzgänge nach Ära
  - Copeland 2008: «Manual of Analogue Sound Restoration Techniques» — Generationsverluste
  - Kefauver & Patschke 2007: «Fundamentals of Digital Audio» — HF-Verlust analog→digital
  - Blauert & Braasch 2011: «Auditory Virtual Environments» — Raumabdruck pro Ära

Singleton-Pattern (thread-safe, §3.x).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class SourceFidelityTarget:
    """
    Geschätztes Original-Klangbild zum Tag der Aufnahme.

    Alle Felder sind probabilistische Schätzungen — keine Fakten.
    confidence gibt die Verlässlichkeit an; bei niedrigem confidence < 0.4
    sollten Aufrufer konservativ auf die Standardwerte zurückfallen.
    """

    # -- Bandbreite ----------------------------------------------------------
    original_bandwidth_hz: float = 20000.0
    """Geschätzte Original-3dB-Bandbreite (Mikrofon → Mischpult → Masterband)."""

    current_bandwidth_hz: float = 20000.0
    """Gemessene Bandbreite des vorliegenden Signals (aus spectral_fingerprint)."""

    bandwidth_gap_hz: float = 0.0
    """original_bandwidth_hz − current_bandwidth_hz: das fehlende Band."""

    # -- Dynamik ------------------------------------------------------------
    original_dynamic_range_db: float = 60.0
    """Geschätzter Original-Dynamikumfang (DR) in dB. Ära × Material × Genre."""

    cumulative_dr_loss_db: float = 0.0
    """Geschätzter akkumulierter DR-Verlust durch Überspielgenerationen."""

    # -- Generationsmodell ---------------------------------------------------
    transfer_generation_count: int = 1
    """Geschätzte Anzahl analoger Überspielgenerationen (1 = Direktdigitalisierung)."""

    cumulative_hf_loss_db: float = 0.0
    """Akkumulierter HF-Verlust bei 8 kHz in dB (Referenz: 8 kHz als Kennfrequenz)."""

    # -- Harmonische Dichte --------------------------------------------------
    era_harmonic_density: float = 1.0
    """Ära-typische Oberton-Dichte [0–2]: 1930er Röhre = 0.8; 1970er Transistor = 1.3."""

    # -- Rekonstruktionskraft ------------------------------------------------
    reconstruction_strength: float = 0.0
    """Empfohlene Rekonstruktionsaggressivität [0–1] (0 = nicht nötig, 1 = maximal)."""

    bandwidth_extension_target_hz: float = 20000.0
    """Zielbandbreite für Phase 06 (AudioSR / SBR): min(original_bandwidth_hz, 20000 Hz)."""

    # -- Metadaten -----------------------------------------------------------
    era_decade: int = 1980
    """Aufnahme-Dekade (1920, 1930, …, 2020)."""

    material_key: str = "unknown"
    """Kanonischer Materialschlüssel (shellac, vinyl, tape, cd_digital, …)."""

    confidence: float = 0.5
    """Vertrauen in die Schätzung [0–1]. < 0.4 → nur schwache Anbindung."""

    notes: list[str] = field(default_factory=list)
    """Erläuterungskette für Audit (keine user-facing Texte)."""

    # -- Ära-Mikrofon-Typ (für Presence-Center-Shift in Phase 38) -----------
    era_mic_type: str = "condenser_modern"
    """Ära-typischer Mikrofon-Typ (acoustic/carbon/ribbon/condenser_early/…)."""

    presence_center_hz_lower: float = 4000.0
    """Era-passende untere Presence-Mittenfrequenz für Phase 38 (Hz)."""

    presence_center_hz_upper: float = 6500.0
    """Era-passende obere Presence-Mittenfrequenz für Phase 38 (Hz)."""


# ---------------------------------------------------------------------------
# Lookup-Tabellen (wissenschaftlich kalibriert)
# ---------------------------------------------------------------------------

# Ära-spezifische Original-Bandbreite:
# Aufnahmequell-Bandbreite (Mikrofon → Preamp → Master) bei −3 dB.
# Quellen: Eargle 2004, Katz & Milner 2007 (Mastering Audio), Clark 1998 (Tapehead)
_ERA_BANDWIDTH_HZ: dict[int, float] = {
    1900: 3500.0,  # Mechanisch-akustische Aufnahme (Grammophontrichter, kein Mikrofon)
    1910: 4000.0,  # Frühe elektrische Versuche (Lee de Forest, Audion)
    1920: 6000.0,  # Western Electric 394 (1925+), frühe Kondensatoren
    1930: 9000.0,  # RCA 44B Ribbon (1932), Neumann CMV3 (1928), AEG Magnetophon (1935)
    1940: 12000.0,  # Neumann U47 (1947), frühe Ampex-Bandmaschinen, BBC/EMI Studios
    1950: 14500.0,  # Telefunken ELA M251 (1957), Ampex 350 (15 ips → −3 dB @15 kHz)
    1960: 16500.0,  # Neumann U67 (1960), EMI TG12345, Studer A62, Beatles-Ära
    1970: 18500.0,  # SSL 4000 (1976), Neve 8078, Studer A80, Rolling Stones/Led Zeppelin
    1980: 20000.0,  # Volldigitale Monitorkette möglich, PCM-1600/Sony F1
    1990: 20000.0,  # CD/DAT Standardära
    2000: 20000.0,
    2010: 20000.0,
    2020: 20000.0,
}

# §v10.31: Physische Medium-Bandbreite (−3 dB Obergrenze).
# Die Ära-Tabelle gibt Studio-Equipment-Bandbreite. Das END-Medium
# (Consumer-Format) kann physikalisch weniger. Diese Caps verhindern,
# dass unrealistische Ziele gesetzt werden (z.B. 18 kHz für Gen-4-Kassette).
# Quellen: Eargle 2004, Camras 1988 (Magnetic Recording), IEC 60094.
_MATERIAL_BANDWIDTH_CAP_HZ: dict[str, float] = {
    "shellac": 8000.0,  # 78 rpm — mechanische Abtastung, keine HF
    "lacquer_disc": 10000.0,  # Heimaufnahme — primitive Schneidtechnik
    "vinyl": 18500.0,  # Stereo-LP — 18 kHz bei Innenrille möglich
    "cassette": 15500.0,  # Type II Chrome bei −3 dB, Type I ~12 kHz
    "tape": 16000.0,  # 7½ ips Viertelspur — IEC EQ limitiert HF
    "reel_tape": 18500.0,  # 15 ips Halbspur — Studer/Ampex Studioniveau
    "dat": 20000.0,  # Digital Audio Tape — 48 kHz Sampling
    "cd": 20000.0,  # Compact Disc — Nyquist 22.05 kHz
    "mp3_low": 15500.0,  # ≤128 kbps — Codec schneidet HF aggressiv
    "mp3_high": 18000.0,  # ≥256 kbps — HF bis ~18 kHz erhalten
    "wire_recording": 7000.0,  # Drahtton — sehr begrenzte HF
    "streaming": 20000.0,
    "minidisc": 18000.0,  # ATRAC 292 kbps — HF-Rolloff ab 18 kHz
    "aac": 20000.0,
    "unknown": 20000.0,
}

# Ära-spezifischer Original-Dynamikumfang (DR in dB):
# Vor digitaler Kompression: Aufnahme-DR = Mikrofon-DR (Kondensator ~130 dB) begrenzt
# durch Tape-Headroom + Rauschen. Erst nach 1990 DR-Krieg (Loudness War).
_ERA_DYNAMIC_RANGE_DB: dict[int, float] = {
    1900: 30.0,
    1910: 35.0,
    1920: 40.0,
    1930: 45.0,
    1940: 50.0,
    1950: 55.0,
    1960: 58.0,
    1970: 62.0,
    1980: 65.0,
    1990: 60.0,  # CD-Ära: gut, aber Loudness-War beginnt
    2000: 50.0,  # Loudness-War-Peak
    2010: 52.0,  # leichte Entspannung (Streaming-Normalisierung)
    2020: 56.0,
}

# Ära-typische Oberton-Dichte [0–2]:
# Röhrenelektronik erzeugt H2/H4 (geradzahlig, «warm»).
# Transistor/IC erzeugt H3/H5 (ungeradzahlig, «kühl», aber mehr Obertöne insgesamt).
_ERA_HARMONIC_DENSITY: dict[int, float] = {
    1900: 0.6,  # Mechanisch, mono, wenig Obertöne im Signal
    1910: 0.65,
    1920: 0.70,
    1930: 0.75,  # Erste Röhren-Amps, Western Electric warm
    1940: 0.80,  # Röhren-Warmklang voll entwickelt
    1950: 0.90,  # Vintage-Röhren-Studio Goldzeitalter
    1960: 1.05,  # Röhren → Transistor-Übergang, Oberton-Mix
    1970: 1.20,  # Transistor/IC voll dominierend, reichere Obertöne
    1980: 1.30,  # Digitale Referenz-Monitorketten, präzise Transiente
    1990: 1.15,  # Digital-Artefakte kompensieren Oberton-Natürlichkeit
    2000: 1.00,
    2010: 1.05,
    2020: 1.10,
}

# Überspielgenerationen pro Material (typisch, konservativ):
# Je mehr Generationen, desto mehr akkumulierter Verlust (HF, Dynamik, Rauschen).
# Quelle: Copeland 2008, IEC 60094-Kassetten-Norm
_MATERIAL_GENERATION_COUNT: dict[str, int] = {
    "shellac": 4,  # Aufnahme → Muttermatrize → Galvano → Pressung → ggf. Kassettendub
    "wire_recording": 4,  # Selten direkt — meist über Zwischen-Drahtspule
    "wax_cylinder": 5,  # Wachszylinder-Dub-Ketten besonders lang
    "vinyl": 3,  # Aufnahme (Band) → Schnitt (Lacquer) → Pressung
    "tape": 3,  # Masterband → Studio-Kopie → Kassette (Consumerband)
    "reel_tape": 2,  # Masterband → Archiv-Kopie (oft 1:1 an Profi-Maschine)
    "cassette": 3,
    "cd_digital": 1,  # Direkte Digitalisierung vom Masterband
    "dat": 1,
    "digital": 1,
    "mp3_low": 1,  # Bereits digital, nur Codec-Verlust
    "mp3_high": 1,
    "aac": 1,
    "streaming": 1,
    "minidisc": 2,  # MD = ATRAC-Kompression + häufiger 2. analog-Schritt
}

# HF-Verlust pro analoger Überspielgeneration bei 8 kHz (in dB):
# Cassette-Norm IEC 60094 Part 1: Kopierverlust 1,5–2,5 dB bei 10 kHz je Generation
# Studer-Interne Messung (Helical Scan 1982): 1,8 dB @ 8 kHz je 1st-gen open-reel copy
_HF_LOSS_PER_GENERATION_DB = 1.8  # bei 8 kHz Referenzfrequenz

# Dynamik-Verlust pro analoger Überspielgeneration (in dB DR-Reduction):
# Eada 1992 (SMPTE): Kassettendub-Verlust 2–4 dB DR je Generation
_DR_LOSS_PER_GENERATION_DB = 2.5

# Maximale Korrektur in dB (Sicherheits-Cap für alle Boost-Operationen)
_MAX_CORRECTION_DB = 12.0

# ---------------------------------------------------------------------------
# SOTA: Ära-spezifische Mikrofon-Typen (für Presence-Center-Shift)
# Quelle: Eargle 2004 "The Microphone Book", Huber & Runstein 2009
# ---------------------------------------------------------------------------

_ERA_MIC_TYPE: dict[int, str] = {
    1900: "acoustic",  # Mechanisch-akustischer Trichter — kein Mikrofon
    1910: "acoustic",
    1920: "carbon",  # Western Electric 394, Bell Labs carbon button
    1930: "ribbon",  # RCA 44B Ribbon (1932), Neumann CMV3
    1940: "ribbon_condenser",  # Übergangsära: RCA + frühe Kondensatoren (BK5, Neumann M7)
    1950: "condenser_early",  # Neumann U47 (1947), Telefunken ELA M251 (1957)
    1960: "condenser_mid",  # Neumann U67 (1960), AKG C12, Sony C37A
    1970: "condenser_modern",  # Neumann U87, AKG 414, SM57 — Standards bis heute
    1980: "condenser_modern",
    1990: "condenser_modern",
    2000: "condenser_modern",
    2010: "condenser_modern",
    2020: "condenser_modern",
}

# Ära-passende Presence-Mittenbereiche (Hz) pro Mikrofon-Typ:
# Lower/upper Presence center frequencies für Phase 38.
# Quellen: Katz 2007 "Mastering Audio", Moylan 2002 "The Art of Recording"
_MIC_PRESENCE_CENTER_HZ: dict[str, tuple[float, float]] = {
    "acoustic": (2000.0, 3500.0),  # Horn-Resonanz 1–3 kHz
    "carbon": (2500.0, 4000.0),  # Carbon-Presence-Peak 2–4 kHz
    "ribbon": (2800.0, 4300.0),  # Ribbon flat — warmth zone
    "ribbon_condenser": (3000.0, 4800.0),  # Mix: ribbon warmth + condenser clarity
    "condenser_early": (3200.0, 5500.0),  # U47 presence peak 5–8 kHz → upper zieht hoch
    "condenser_mid": (3500.0, 6000.0),  # U67/C12: klassische Presence-Zone
    "condenser_modern": (4000.0, 6500.0),  # Moderner Standard (Pultec, API 550A-Referenz)
}

# ---------------------------------------------------------------------------
# SOTA: Frequenz-abhängige Generationsverlust-Kompensationskurven
# Positive Werte = dB die pro Generation VERLOREN gehen → müssen KOMPENSIERT werden.
# Quellen: Copeland 2008 "Manual of Analogue Sound Restoration",
#          IEC 60094 (Compact Cassette Norm), Studer interne Messung 1982
# ---------------------------------------------------------------------------

_GENERATION_LOSS_DB_PER_GEN: dict[str, dict[int, float]] = {
    "shellac": {
        # Acetat-Scheibe → Galvano → Pressen: exponentieller Rolloff ab 5 kHz
        500: 0.10,
        1000: 0.20,
        2000: 0.50,
        3000: 0.90,
        5000: 2.00,
        6000: 3.50,
        8000: 5.50,
        10000: 7.50,
        12000: 10.0,
    },
    "wax_cylinder": {
        # Wachszylinder-Dub: sehr steiler Verlust
        500: 0.20,
        1000: 0.50,
        2000: 1.00,
        3000: 2.00,
        5000: 4.00,
        6000: 6.00,
        8000: 9.00,
        10000: 12.0,
        12000: 12.0,
    },
    "wire_recording": {
        500: 0.15,
        1000: 0.35,
        2000: 0.70,
        3000: 1.40,
        5000: 3.00,
        6000: 5.00,
        8000: 7.00,
        10000: 10.0,
        12000: 12.0,
    },
    "vinyl": {
        # Lacquer-Schnitt + Abtastung: moderater Verlust ab 8 kHz
        500: 0.05,
        1000: 0.10,
        2000: 0.20,
        3000: 0.35,
        5000: 0.60,
        6000: 1.00,
        8000: 2.00,
        10000: 3.50,
        12000: 5.50,
    },
    "tape": {
        # Consumer-Cassette (IEC 60094 Typ II): signifikanter Verlust ab 6 kHz
        500: 0.05,
        1000: 0.10,
        2000: 0.30,
        3000: 0.60,
        5000: 1.00,
        6000: 1.50,
        8000: 3.00,
        10000: 5.00,
        12000: 7.50,
    },
    "cassette": {
        500: 0.05,
        1000: 0.10,
        2000: 0.30,
        3000: 0.60,
        5000: 1.00,
        6000: 1.50,
        8000: 3.00,
        10000: 5.00,
        12000: 7.50,
    },
    "reel_tape": {
        # Professionelles Open-Reel (38 cm/s): sehr geringer Verlust je Kopie
        500: 0.02,
        1000: 0.05,
        2000: 0.12,
        3000: 0.25,
        5000: 0.40,
        6000: 0.65,
        8000: 1.10,
        10000: 1.80,
        12000: 3.00,
    },
    "cd_digital": {},  # Direkt digital: kein Generationsverlust
    "dat": {},
    "digital": {},
    "mp3_low": {},
    "mp3_high": {},
    "aac": {},
    "streaming": {},
    "minidisc": {  # MiniDisc ATRAC: leichter Hochton-Verlust durch Codec
        500: 0.0,
        1000: 0.0,
        2000: 0.10,
        3000: 0.20,
        5000: 0.50,
        6000: 0.80,
        8000: 1.50,
        10000: 2.50,
        12000: 4.00,
    },
}


# ---------------------------------------------------------------------------
# Lookup Helper
# ---------------------------------------------------------------------------


def _decade(year: int | None) -> int:
    """Rundet auf die nächste Dekade (1957 → 1950, 1982 → 1980)."""
    if year is None:
        return 1970  # Neutral-Fallback
    return max(1900, min(2020, (int(year) // 10) * 10))


def _lookup_era(table: dict[int, float], decade: int) -> float:
    """Lineare Interpolation zwischen benachbarten Einträgen der Ära-Tabelle (float)."""
    keys = sorted(table.keys())
    if decade <= keys[0]:
        return table[keys[0]]
    if decade >= keys[-1]:
        return table[keys[-1]]
    for i, k in enumerate(keys[:-1]):
        if k <= decade < keys[i + 1]:
            lo, hi = table[k], table[keys[i + 1]]
            frac = (decade - k) / (keys[i + 1] - k)
            return lo + frac * (hi - lo)
    return table[keys[-1]]


def _lookup_era_str(table: dict[int, str], decade: int) -> str:
    """Gibt den string-Wert für die Dekade zurück (nächstkleinerer oder gleicher Key)."""
    keys = sorted(table.keys())
    if decade <= keys[0]:
        return table[keys[0]]
    if decade >= keys[-1]:
        return table[keys[-1]]
    for k in reversed(keys):
        if k <= decade:
            return table[k]
    return table[keys[0]]


# ---------------------------------------------------------------------------
# Reconstructor
# ---------------------------------------------------------------------------

_instance: SourceFidelityReconstructor | None = None
_lock = threading.Lock()


class SourceFidelityReconstructor:
    """
    Schätzt die Signallücke zwischen dem Ist-Zustand und dem Original am Aufnahmetag.

    Ergebnis ``SourceFidelityTarget`` beinhaltet:
      - Welche Bandbreite das Original hatte (original_bandwidth_hz)
      - Wie viele HF-Generationen verloren gingen (cumulative_hf_loss_db)
      - Wie aggressiv die Rekonstruktion sein sollte (reconstruction_strength)
      - Zielbandbreite für Phase 06 (bandwidth_extension_target_hz)

    Verwendung in UV3 SongCalibration und Phase 06.
    """

    def estimate(
        self,
        *,
        era_decade: int | None = None,
        material_key: str = "unknown",
        current_bandwidth_hz: float | None = None,
        spectral_fingerprint: dict | None = None,
        transfer_chain: list[str] | None = None,
        mode: str = "restoration",
    ) -> SourceFidelityTarget:
        """
        Schätzt das Original-Klangbild auf Basis von Ära, Material und Transfer-Kette.

        Args:
            era_decade: Aufnahme-Dekade (z.B. 1960). None → 1970 (neutral).
            material_key: Kanonischer Materialschlüssel (shellac, vinyl, tape, …).
            current_bandwidth_hz: Gemessene Bandbreite des vorliegenden Signals.
                Wenn None: wird aus spectral_fingerprint entnommen oder 20kHz angenommen.
            spectral_fingerprint: DefectScanner-Spektral-Fingerabdruck (optional).
            transfer_chain: Liste erkannter Übertragungsträger in der Kette
                (jüngstes zuerst), z.B. ["mp3", "cassette", "vinyl"].
            mode: QualityMode-String (restauration/studio2026/…).

        Returns:
            SourceFidelityTarget mit dem geschätzten Originalklang.
        """
        notes: list[str] = []
        decade = _decade(era_decade)

        # -- 1. Ära-basierte Original-Bandbreite -----------------------------
        original_bw = _lookup_era(_ERA_BANDWIDTH_HZ, decade)
        notes.append(f"era_bw_{decade}={original_bw:.0f}Hz")

        # -- 1a. §v10.31: Physische Medium-Obergrenze -------------------------
        # Die Ära-Tabelle gibt Studio-Equipment-Bandbreite (z.B. Studer A80 = 18.5 kHz).
        # Das END-Medium (Kassette, Vinyl-Pressung, MP3) kann physikalisch weniger.
        # Clip auf das, was das konkrete Medium maximal erreichen kann.
        _material_bw_cap = _MATERIAL_BANDWIDTH_CAP_HZ.get(material_key, 20000.0)
        original_bw = min(original_bw, _material_bw_cap)
        notes.append(f"material_cap={_material_bw_cap:.0f}Hz → era_bw_capped={original_bw:.0f}Hz")

        # -- 2. Gemessene aktuelle Bandbreite --------------------------------
        _cur_bw: float
        if current_bandwidth_hz is not None:
            _cur_bw = float(np.clip(current_bandwidth_hz, 500.0, 24000.0))
        elif spectral_fingerprint:
            # effective_bandwidth_hz = last freq bin at -60 dBFS → actual HF cutoff
            # rolloff_95_hz = 95th-percentile cumulative energy → always low for
            # bass/mid-heavy music (Schlager, rock, classical) regardless of HF content.
            # Using min(rolloff_95, eff_bw) caused cur_bw=1475 Hz for normal MP3s.
            _ef = float(spectral_fingerprint.get("effective_bandwidth_hz", 20000.0))
            _cur_bw = float(np.clip(_ef, 500.0, 24000.0))
            notes.append(f"cur_bw_from_fingerprint={_cur_bw:.0f}Hz")
        else:
            _cur_bw = 20000.0
            notes.append("cur_bw_assumed_full")

        # -- 3. Überspielgenerationen schätzen --------------------------------
        if transfer_chain and len(transfer_chain) >= 2:
            # Correct generation count:
            # - Start with the SOURCE medium's internal stage count (recording → pressing etc.)
            # - Add +1 for each ADDITIONAL analog-medium boundary (each new copy generation)
            # - Digital media (mp3, cd, dat) do NOT add analog generation losses
            # Old code summed ALL media's gen counts → vinyl(3)+tape(3)=7 for a 2-step chain.
            _ANALOG_MEDIA = {"shellac", "vinyl", "tape", "reel_tape", "cassette", "wire_recording", "wax_cylinder"}
            _chain_lower = [m.lower() for m in transfer_chain]
            _first_analog = next((m for m in _chain_lower if m in _ANALOG_MEDIA), None)
            if _first_analog:
                _gen_count = _MATERIAL_GENERATION_COUNT.get(_first_analog, 2)
                _found_first = False
                for _m in _chain_lower:
                    if _m == _first_analog and not _found_first:
                        _found_first = True
                        continue
                    if _found_first and _m in _ANALOG_MEDIA:
                        _gen_count += 1  # Each additional analog medium = 1 copy generation
            else:
                _gen_count = 1
            _gen_count = min(_gen_count, 8)
            notes.append(f"gen_count_from_chain={_gen_count}")
        else:
            _gen_count = _MATERIAL_GENERATION_COUNT.get(material_key.lower(), 2)
            notes.append(f"gen_count_from_material={material_key}={_gen_count}")

        # -- 4. Kumulativer HF-Verlust ---------------------------------------
        _hf_loss = float(_gen_count - 1) * _HF_LOSS_PER_GENERATION_DB
        notes.append(f"hf_loss_accum={_hf_loss:.1f}dB")

        # -- 5. Kumulativer DR-Verlust ---------------------------------------
        _dr_loss = float(_gen_count - 1) * _DR_LOSS_PER_GENERATION_DB
        _orig_dr = _lookup_era(_ERA_DYNAMIC_RANGE_DB, decade)
        notes.append(f"dr_orig={_orig_dr:.1f}dB  dr_loss={_dr_loss:.1f}dB")

        # -- 6. Bandbreiten-Lücke --------------------------------------------
        _bw_gap = max(0.0, original_bw - _cur_bw)
        notes.append(f"bw_gap={_bw_gap:.0f}Hz")

        # -- 7. Rekonstruktionsstärke ----------------------------------------
        _bw_frac = float(np.clip(_bw_gap / max(original_bw, 1.0), 0.0, 1.0))
        _hf_frac = float(np.clip(_hf_loss / 12.0, 0.0, 1.0))
        _reconstruction_strength = float(np.clip(0.55 * _bw_frac + 0.45 * _hf_frac, 0.0, 1.0))

        if mode and "studio" in mode.lower():
            _reconstruction_strength = float(np.clip(_reconstruction_strength * 1.15, 0.0, 1.0))
            notes.append("studio2026_recon_boost×1.15")

        # -- 8. Zielbandbreite für Phase 06 ----------------------------------
        _bw_target = float(np.clip(original_bw, _cur_bw, _cur_bw + 5000.0))
        _bw_target = float(np.clip(_bw_target, _cur_bw, 20000.0))
        notes.append(f"bw_target={_bw_target:.0f}Hz")

        # -- 9. Ära-Oberton-Dichte -------------------------------------------
        _harm_density = _lookup_era(_ERA_HARMONIC_DENSITY, decade)
        notes.append(f"harm_density={_harm_density:.2f}")

        # -- 10. Konfidenz ---------------------------------------------------
        _conf = 0.30
        if era_decade is not None:
            _conf += 0.25
        if material_key not in ("unknown", ""):
            _conf += 0.25
        if transfer_chain and len(transfer_chain) >= 2:
            _conf += 0.20
        _conf = float(np.clip(_conf, 0.0, 1.0))
        notes.append(f"confidence={_conf:.2f}")

        # -- 11. Ära-Mikrofon-Typ und Presence-Center -------------------------
        _mic_type = _lookup_era_str(_ERA_MIC_TYPE, decade)
        _lower_center, _upper_center = _MIC_PRESENCE_CENTER_HZ.get(_mic_type, (4000.0, 6500.0))
        notes.append(f"mic_type={_mic_type} presence={_lower_center:.0f}/{_upper_center:.0f}Hz")

        logger.info(
            "SourceFidelityReconstructor: era=%s mat=%s bw_gap=%.0fHz gen=%d "
            "hf_loss=%.1fdB recon=%.2f target_bw=%.0fHz conf=%.2f mic=%s",
            decade,
            material_key,
            _bw_gap,
            _gen_count,
            _hf_loss,
            _reconstruction_strength,
            _bw_target,
            _conf,
            _mic_type,
        )

        return SourceFidelityTarget(
            original_bandwidth_hz=round(original_bw, 1),
            current_bandwidth_hz=round(_cur_bw, 1),
            bandwidth_gap_hz=round(_bw_gap, 1),
            original_dynamic_range_db=round(_orig_dr, 1),
            cumulative_dr_loss_db=round(min(_dr_loss, _orig_dr * 0.5), 1),
            transfer_generation_count=_gen_count,
            cumulative_hf_loss_db=round(_hf_loss, 2),
            era_harmonic_density=round(_harm_density, 3),
            reconstruction_strength=round(_reconstruction_strength, 3),
            bandwidth_extension_target_hz=round(_bw_target, 1),
            era_decade=decade,
            material_key=material_key,
            confidence=round(_conf, 2),
            notes=notes,
            era_mic_type=_mic_type,
            presence_center_hz_lower=round(_lower_center, 1),
            presence_center_hz_upper=round(_upper_center, 1),
        )

    def compute_correction_curve_db(
        self,
        target: SourceFidelityTarget,
        freqs_hz: np.ndarray,
    ) -> np.ndarray:
        """
        Frequenz-abhängige Korrekturkurve (dB) für Quelltreue-Restaurierung.

        Kompensiert akkumulierten frequenz-abhängigen Generationsverlust aus der
        Tonträgerkette (IEC 60094, Copeland 2008). Gibt ausschließlich Boosts zurück
        (≥ 0 dB). Gedeckelt auf _MAX_CORRECTION_DB. Skaliert mit confidence ×
        reconstruction_strength.

        Args:
            target: SourceFidelityTarget aus estimate().
            freqs_hz: Frequenz-Array in Hz (z.B. np.linspace(0, 24000, 129)).

        Returns:
            np.ndarray der dB-Korrekturwerte (≥ 0.0), gleiche Länge wie freqs_hz.
        """
        mat = target.material_key.lower().split("/")[0]
        extra_gens = max(0, target.transfer_generation_count - 1)
        correction = np.zeros(len(freqs_hz), dtype=np.float64)

        # --- Generationsverlust-Kompensation ---
        loss_curve = _GENERATION_LOSS_DB_PER_GEN.get(mat)
        if loss_curve is None:
            for key in _GENERATION_LOSS_DB_PER_GEN:
                if key in mat or mat in key:
                    loss_curve = _GENERATION_LOSS_DB_PER_GEN[key]
                    break

        if loss_curve and extra_gens > 0:
            loss_freqs = sorted(loss_curve.keys())
            loss_vals = [loss_curve[f] * extra_gens for f in loss_freqs]
            if loss_freqs:
                loss_interp = np.interp(
                    freqs_hz,
                    [float(f) for f in loss_freqs],
                    loss_vals,
                    left=0.0,
                    right=float(loss_vals[-1]),
                )
                if target.original_bandwidth_hz < 20000.0:
                    _bw_fade_start = target.original_bandwidth_hz * 0.80
                    _bw_fade_end = target.original_bandwidth_hz * 1.00
                    _fade = np.clip(
                        1.0 - (freqs_hz - _bw_fade_start) / max(_bw_fade_end - _bw_fade_start, 1.0),
                        0.0,
                        1.0,
                    )
                    loss_interp = loss_interp * _fade
                correction += np.maximum(0.0, loss_interp)

        # --- Skalierung mit Konfidenz × Rekonstruktionsstärke ---
        scale = float(np.clip(target.confidence * target.reconstruction_strength, 0.0, 1.0))
        correction *= scale

        # --- Sicherheits-Cap ---
        return np.minimum(correction, _MAX_CORRECTION_DB)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# EQ Processor
# ---------------------------------------------------------------------------

_eq_instance: SourceFidelityEQProcessor | None = None
_eq_lock = threading.Lock()


class SourceFidelityEQProcessor:
    """
    Wendet frequenz-abhängige Generationsverlust-Kompensation auf Audio an.

    Verwendet scipy.signal.firwin2 (linear-phase symmetrisches FIR, 257 Taps =
    5.3 ms @ 48 kHz). Frequenzgang = Korrekturkurve aus compute_correction_curve_db().
    Anwendung via scipy.signal.fftconvolve mode='same' (phasenerhaltend).

    Nur positive Verstärkung (gains ≥ 1.0). Max-Cap: _MAX_CORRECTION_DB.
    Überspringt sich wenn: confidence < 0.35, reconstruction_strength < 0.15,
    max Korrektur < 0.3 dB.

    Wissenschaftliche Basis:
    - Larsen & Aarts 2004 (Bandwidth Extension)
    - IEC 60094 (Cassette Norm — Kopierverlust-Messungen)
    - Copeland 2008 (Manual of Analogue Sound Restoration Techniques)
    """

    _FILTER_TAPS = 257
    _MIN_CORRECTION_DB = 0.30
    _MIN_CONFIDENCE = 0.35
    _MIN_RECON = 0.15

    def apply(
        self,
        audio: np.ndarray,
        sr: int,
        target: SourceFidelityTarget,
        *,
        strength: float = 1.0,
    ) -> np.ndarray:
        """
        Wendet Source-Fidelity-EQ-Korrektur auf Audio an.

        Args:
            audio: Eingabe-Audio (mono oder stereo, float32/float64), 48 kHz.
            sr: Sample-Rate (sollte 48000 sein für Verarbeitungsphasen).
            target: SourceFidelityTarget aus SourceFidelityReconstructor.estimate().
            strength: Gesamt-Korrektionsstärke [0–1].

        Returns:
            EQ-korrigiertes Audio, gleiche Shape wie Eingang.
            NaN/Inf-frei, begrenzt auf ±1.0.
        """
        if target.confidence < self._MIN_CONFIDENCE:
            return audio
        if target.reconstruction_strength < self._MIN_RECON:
            return audio
        eff_str = float(np.clip(strength, 0.0, 1.0))
        if eff_str < 0.05:
            return audio

        try:
            from scipy.signal import fftconvolve, firwin2
        except ImportError:
            logger.debug("SourceFidelityEQProcessor: scipy nicht verfügbar, übersprungen")
            return audio

        sfr = get_source_fidelity_reconstructor()
        n_pts = self._FILTER_TAPS // 2 + 1
        freqs_hz_fir = np.linspace(0.0, float(sr) / 2.0, n_pts)
        correction_db = sfr.compute_correction_curve_db(target, freqs_hz_fir) * eff_str

        max_corr = float(np.max(correction_db))
        if max_corr < self._MIN_CORRECTION_DB:
            logger.debug(
                "SourceFidelityEQProcessor: Korrektur vernachlässigbar (max=%.2f dB), übersprungen",
                max_corr,
            )
            return audio

        gains_linear = 10.0 ** (correction_db / 20.0)
        freqs_norm = np.clip(freqs_hz_fir / (float(sr) / 2.0), 0.0, 1.0)
        freqs_norm[0] = 0.0
        freqs_norm[-1] = 1.0

        try:
            fir = firwin2(
                self._FILTER_TAPS,
                freq=freqs_norm,
                gain=gains_linear,
                antisymmetric=False,
            )
        except Exception as _fir_exc:
            logger.debug("SourceFidelityEQProcessor: firwin2 fehlgeschlagen: %s", _fir_exc)
            return audio

        orig_dtype = audio.dtype
        audio_f64 = audio.astype(np.float64)
        if audio_f64.ndim == 1:
            result = fftconvolve(audio_f64, fir, mode="same")
        else:
            result = np.stack(
                [fftconvolve(audio_f64[ch], fir, mode="same") for ch in range(audio_f64.shape[0])],
                axis=0,
            )

        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        result = np.clip(result, -1.0, 1.0)
        logger.debug(
            "SourceFidelityEQProcessor: max_corr=%.1f dB mat=%s gen=%d conf=%.2f str=%.2f",
            max_corr,
            target.material_key,
            target.transfer_generation_count,
            target.confidence,
            eff_str,
        )
        return result.astype(orig_dtype)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Singleton-Zugriffsfunktionen (thread-safe, double-checked locking)
# ---------------------------------------------------------------------------


def get_source_fidelity_eq_processor() -> SourceFidelityEQProcessor:
    """Thread-safe Singleton-Zugriff auf SourceFidelityEQProcessor."""
    global _eq_instance
    if _eq_instance is None:
        with _eq_lock:
            if _eq_instance is None:
                _eq_instance = SourceFidelityEQProcessor()
    return _eq_instance


def get_source_fidelity_reconstructor() -> SourceFidelityReconstructor:
    """Thread-safe Singleton-Zugriff auf SourceFidelityReconstructor."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = SourceFidelityReconstructor()
    return _instance
