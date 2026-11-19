"""backend/core/broadcast_archive_db.py — Studio, Label & Broadcast-Archive Wissensbasis

Maps a ``provenance_hint`` string to a :class:`ProvenanceAdjustment` that is
layered **on top of** the generic era chain from :mod:`tonal_reference_profile`.
This gives Aurik deeper autonomous knowledge: not just "1970s recording" but
"1970s recording in NDR Hamburg's Studio 10 — EELA-Console, Studer A80 at 30 ips".

The adjustment is applied by
:meth:`TonalReferenceProfiler.get_curve_with_provenance`.

Covered organisations (provenance_hint key, case-insensitive partial match):
  abbey_road         — EMI Studios London (1931–present)
  capitol_studios    — Capitol Records Hollywood (1956–present)
  columbia_30th      — Columbia 30th Street NYC (1948–1981)
  ddr_rundfunk       — DDR-Rundfunk / DT64 (EELA C240 console, 1952–1991)
  deutsche_grammophon — DG / Archiv Produktion (Decca-Tree, 1950–present)
  mercury_living     — Mercury Living Presence (Ampex 201+351, 1952–1967)
  melodiya_soviet    — Melodiya / GOST 13699-55 (1964–1991)
  nbc_radio          — NBC Radio Studios USA (RCA 44B dominant, 1935–1975)
  ndr_hamburg        — Norddeutscher Rundfunk Hamburg (1945–present)
  philips_eindhoven  — Philips/PolyGram Eindhoven NL (1950–1975)
  rias_berlin        — RIAS Berlin West (1946–1994)

Scientific references:
  Doyle (2004) «Recording the Beatles» — Abbey Road equipment inventories
  Copeland (2008) «Manual of Analogue Sound Restoration» — BBC BTR machines
  Zeller (2012) «DDR Rundfunktechnik» — RIAS/DT64 EELA-Mischpulte, Telefunken-Tape
  Senior, Sound On Sound (2011) — Neve 1073 and SSL 4000 signature measurements
  Eargle (2004) «The Microphone Book» — Mercury Living Presence Ampex/RCA chain
  Katz (2007) «Mastering Audio» — Columbia 30th Street acoustics
  Morita & Takagi (1994) — NHK Tokyo studio archive standards
  Gronow & Saunio (1998) — Melodiya / Soviet recording industry history

Usage::

    from backend.core.broadcast_archive_db import get_provenance_adjustment
    adj = get_provenance_adjustment("ndr_hamburg", era_decade=1970)
    if adj is not None:
        # Overlay adj.mic_delta / console_delta / tape_delta on TonalCurve target

Singleton: this module is stateless (pure lookup table), no singleton needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProvenanceAdjustment:
    """Chain-specific EQ delta applied on top of the generic era chain.

    All delta arrays are (Hz, dB) breakpoint lists compatible with the
    :func:`tonal_reference_profile._interp_db_curve` helper.
    Positive dB = this studio's chain was brighter/warmer than the era average.
    Negative dB = it was darker/narrower.
    """

    provenance_label: str

    # Mic-response delta — adds to era mic FR
    mic_delta: list[tuple[float, float]] = field(default_factory=list)

    # Console/preamp EQ delta — adds to era console EQ
    console_delta: list[tuple[float, float]] = field(default_factory=list)

    # Tape machine delta — adds to era tape response
    tape_delta: list[tuple[float, float]] = field(default_factory=list)

    # Valid era range.  Outside this range, adjustment is linearly faded out.
    era_min: int = 1900
    era_max: int = 2030

    # Optional noise texture key override (overrides material default in
    # tonal_reference_profile._MATERIAL_NOISE_TEXTURE when not None).
    noise_texture_override: str | None = None

    # Human-readable note for logging / UI
    description: str = ""


# ---------------------------------------------------------------------------
# Provenance database
# ---------------------------------------------------------------------------
# Convention: keys are lowercase canonical names (matched via partial substring).

_PROVENANCE_DB: dict[str, ProvenanceAdjustment] = {
    # -----------------------------------------------------------------------
    # ABBEY ROAD / EMI London
    # Console: EMI REDD.37 (1957–1968), then TG12345 desk (1968–1983).
    # Tape: BTR2 (1950s–early 60s), Studer J37 (1960–late 60s), ATR102 later.
    # Characteristic: clean, slight bass warmth from BTR2 LF boost,
    # modest high-shelf roll from REDD.37 console.
    # Doyle (2004) ch. 4–6.
    # -----------------------------------------------------------------------
    "abbey_road": ProvenanceAdjustment(
        provenance_label="EMI Abbey Road Studios",
        mic_delta=[],  # EMI used standard Neumann U47/M49 — era curve sufficient
        console_delta=[
            # REDD.37: slight bass lift (200–500 Hz transformer warmth)
            (60.0, +0.5),
            (200.0, +0.8),
            (400.0, +0.5),
            (1000.0, 0.0),
            # REDD.37 slight mid-scoop (Doyle 2004)
            (1500.0, -0.3),
            (3000.0, -0.5),
            (6000.0, -0.3),
            (10000.0, 0.0),
            (16000.0, -0.5),
        ],
        tape_delta=[
            # BTR2 at 15 ips: modest LF emphasis, gentle HF rolloff vs generic
            (40.0, +0.3),
            (100.0, +0.5),
            (300.0, +0.2),
            (1000.0, 0.0),
            (6000.0, -0.3),
            (10000.0, -0.5),
            (14000.0, -1.0),
        ],
        era_min=1955,
        era_max=1985,
        description="EMI Abbey Road — REDD.37/TG12345 + BTR2/Studer J37",
    ),
    # -----------------------------------------------------------------------
    # CAPITOL STUDIOS Hollywood
    # Console: Altec 250 / Universal Audio 610 (1956–1970s), API later.
    # Tape: Ampex 300/350 (1950s), 440/ATR102 (1960s–70s) at 30 ips.
    # Characteristic: warm, slightly mid-forward (610 preamp character),
    # pronounced bass from Ampex 300 @ 15 ips.
    # Eargle (2004) §12.
    # -----------------------------------------------------------------------
    "capitol_studios": ProvenanceAdjustment(
        provenance_label="Capitol Studios Hollywood",
        mic_delta=[
            # Capitol favoured Neumann M50 overhead + RCA 77DX ribbon close-mic.
            # M50 is omnidirectional with rising HF above 8 kHz — slightly brighter.
            (6000.0, +0.5),
            (10000.0, +1.0),
            (14000.0, +0.5),
            (18000.0, 0.0),
        ],
        console_delta=[
            # UA 610 tube preamp: warm LF, slight 3 kHz presence peak (UA spec sheet)
            (60.0, +1.0),
            (200.0, +0.8),
            (600.0, 0.0),
            (1000.0, 0.0),
            (3000.0, +1.0),
            (6000.0, +0.5),
            (10000.0, 0.0),
        ],
        tape_delta=[
            # Ampex 350 @ 15 ips: warm LF bump (+1 dB) vs Studer
            (40.0, +0.8),
            (100.0, +1.0),
            (300.0, +0.5),
            (600.0, 0.0),
            (1000.0, 0.0),
            (8000.0, -0.3),
            (12000.0, -0.8),
        ],
        era_min=1955,
        era_max=1985,
        description="Capitol Hollywood — UA 610 + Ampex 350/440",
    ),
    # -----------------------------------------------------------------------
    # COLUMBIA 30TH STREET STUDIO — New York City ("The Church")
    # Console: Columbia custom / Altec 250. Tape: Ampex 200/300.
    # Characteristic: warm, diffuse room (ex-church), natural reverb tail,
    # slight LF bloom from large room modes, rolled HF (Altec 250 bandwidth).
    # Katz (2007) §5; Columbia 1956–1981.
    # -----------------------------------------------------------------------
    "columbia_30th": ProvenanceAdjustment(
        provenance_label="Columbia 30th Street NYC",
        mic_delta=[
            # Large church acoustic → distant mics → prominent room reverb in HF
            (3000.0, -0.3),
            (6000.0, -0.8),
            (10000.0, -1.5),
            (14000.0, -3.0),
        ],
        console_delta=[
            # Altec 250: warm, LF bloom from output transformers
            (40.0, +1.2),
            (100.0, +1.5),
            (300.0, +0.8),
            (600.0, +0.3),
            (1000.0, 0.0),
            (4000.0, -0.2),
            (8000.0, -0.5),
            (12000.0, -1.0),
        ],
        tape_delta=[
            # Ampex 200: modest LF emphasis, early HF rolloff (15 ips era)
            (60.0, +0.5),
            (200.0, +0.5),
            (600.0, 0.0),
            (1000.0, 0.0),
            (6000.0, -0.3),
            (10000.0, -1.5),
            (12000.0, -3.0),
        ],
        era_min=1948,
        era_max=1982,
        description="Columbia 30th Street NYC — Altec 250 + large church acoustic",
    ),
    # -----------------------------------------------------------------------
    # DDR-RUNDFUNK / DT64 / RIAS Berlin (East)
    # Console: EELA C240 (Swiss, distributed to DDR from ~1968).
    # Tape: ORWO-Tape (German Democratic Republic, ~equivalent BASF LH 900).
    # Characteristic: slight presence dip 800 Hz–1.5 kHz from EELA EQ topology;
    # HF limited to ~13 kHz (ORWO tape + DDR broadcast standard);
    # stable LF, minimal harmonic character (solid-state early adoption 1965+).
    # Zeller (2012) ch. 3, 7; Siwe (2003).
    # -----------------------------------------------------------------------
    "ddr_rundfunk": ProvenanceAdjustment(
        provenance_label="DDR-Rundfunk / DT64 (EELA C240)",
        mic_delta=[
            # DDR studios mostly used Neumann U87 / AKG C414 clones (RFT MV 701)
            # RFT MV 701: slight mid-forward, no extended HF
            (6000.0, -0.3),
            (10000.0, -1.0),
            (13000.0, -2.5),
            (16000.0, -8.0),
        ],
        console_delta=[
            # EELA C240: characterisitic presence dip ~800 Hz–1.5 kHz (Zeller 2012)
            (200.0, 0.0),
            (500.0, -0.3),
            (800.0, -1.2),
            (1200.0, -1.5),
            (1500.0, -0.8),
            (2000.0, -0.3),
            (3000.0, 0.0),
            (6000.0, +0.3),
            (10000.0, 0.0),
            (13000.0, -1.0),
            (16000.0, -4.0),
        ],
        tape_delta=[
            # ORWO LH-900 similar to BASF LH900 @ 19 cm/s (≈ 7.5 ips):
            # gentle LF, moderate HF cutoff vs. 30 ips
            (100.0, +0.3),
            (300.0, 0.0),
            (1000.0, 0.0),
            (6000.0, -0.5),
            (10000.0, -2.0),
            (12000.0, -4.0),
            (14000.0, -10.0),
        ],
        era_min=1952,
        era_max=1992,
        noise_texture_override="reel_tape",
        description="DDR-Rundfunk — EELA C240 + ORWO-Tape @ 19 cm/s",
    ),
    # -----------------------------------------------------------------------
    # DEUTSCHE GRAMMOPHON / Archiv Produktion
    # Microphone: Decca-Tree (3× Neumann M50 omni, 1956+), B&K 4006 later.
    # Console: Philips (custom NL), then SSL 4000 (1980s).
    # Tape: Telefunken 28 (1956–1965), Agfa PE36 (1965–1975), BASF 900 (1975+).
    # Characteristic: wide natural stereo (Decca-tree), very clean, slightly cool HF.
    # Eargle (2004) §11.
    # -----------------------------------------------------------------------
    "deutsche_grammophon": ProvenanceAdjustment(
        provenance_label="Deutsche Grammophon / DG Archiv",
        mic_delta=[
            # Neumann M50 (omni): extended flat from 20 Hz, rising above 8 kHz
            (6000.0, +0.3),
            (10000.0, +0.8),
            (14000.0, +1.0),
            (18000.0, +0.5),
        ],
        console_delta=[
            # Philips NL console: clean, flat, slight HF shelving roll at 12 kHz
            (1000.0, 0.0),
            (6000.0, 0.0),
            (10000.0, 0.0),
            (12000.0, -0.3),
            (16000.0, -1.0),
            (20000.0, -2.0),
        ],
        tape_delta=[
            # Agfa PE36 (1960s): "Agfa sound" subtle HF elevation (Kefauver 2007)
            (4000.0, +0.3),
            (6000.0, +0.5),
            (8000.0, +0.5),
            (10000.0, +0.2),
            (12000.0, 0.0),
            (14000.0, -0.5),
            (16000.0, -2.0),
        ],
        era_min=1950,
        era_max=1990,
        description="DG — Decca-Tree M50 + Agfa PE36",
    ),
    # -----------------------------------------------------------------------
    # MERCURY LIVING PRESENCE (1952–1967)
    # Microphone: single-point (Telefunken 201 or RCA 44B), minimal mics.
    # Console: custom RCA/Ampex mix bus. Tape: Ampex 201/351 at 30 ips.
    # Characteristic: natural acoustic, slightly warm, minimal EQ processing.
    # Eargle (2004) §10; "Living Presence" single-mic philosophy.
    # -----------------------------------------------------------------------
    "mercury_living": ProvenanceAdjustment(
        provenance_label="Mercury Living Presence",
        mic_delta=[
            # Single-point placement → strong room acoustics in signal
            # Telefunken 201 (1952–58): broad presence peak 2–7 kHz
            (2000.0, +0.5),
            (4000.0, +1.0),
            (7000.0, +1.0),
            (9000.0, +0.5),
            (12000.0, 0.0),
            (16000.0, -1.0),
        ],
        console_delta=[
            # Minimal processing philosophy — custom mix bus near-flat
            (1000.0, 0.0),
            (10000.0, 0.0),
            (20000.0, 0.0),
        ],
        tape_delta=[
            # Ampex 351 @ 30 ips: extended HF, slight LF warmth
            (40.0, +0.3),
            (100.0, +0.5),
            (1000.0, 0.0),
            (10000.0, +0.3),
            (15000.0, 0.0),
            (18000.0, -0.5),
        ],
        era_min=1952,
        era_max=1968,
        description="Mercury Living Presence — single-point Telefunken + Ampex 351 @ 30 ips",
    ),
    # -----------------------------------------------------------------------
    # MELODIYA / Soviet GOST standard
    # GOST 13699-55: limits audio bandwidth to 40 Hz – 12 500 Hz.
    # Console: Soviet RIAS/Neumann clones (RFT / TESLA), later Neve/Calrec imports.
    # Tape: TASMA 6 (Soviet, ~15 ips), later ORWO LH900.
    # Characteristic: restricted HF (<12.5 kHz), slight mid-forward presence,
    # occasional LF rumble from Soviet pressing equipment.
    # Gronow & Saunio (1998) ch. 8.
    # -----------------------------------------------------------------------
    "melodiya_soviet": ProvenanceAdjustment(
        provenance_label="Melodiya / Soviet GOST 13699-55",
        mic_delta=[
            # Soviet RFT/TESLA mics: mid-forward, limited HF extension
            (3000.0, +0.5),
            (6000.0, 0.0),
            (8000.0, -1.0),
            (10000.0, -3.0),
            (12000.0, -8.0),
            (14000.0, -20.0),
        ],
        console_delta=[
            # Soviet console: pronounced 3 kHz presence peak from circuit design
            (800.0, 0.0),
            (2000.0, +0.5),
            (3000.0, +1.5),
            (5000.0, +0.5),
            (8000.0, -0.5),
            (10000.0, -1.5),
            (12000.0, -4.0),
        ],
        tape_delta=[
            # TASMA-6 @ 15 ips: limited HF, moderate LF
            (60.0, +0.5),
            (200.0, +0.3),
            (1000.0, 0.0),
            (5000.0, -0.5),
            (8000.0, -2.0),
            (10000.0, -5.0),
            (12000.0, -15.0),
        ],
        era_min=1960,
        era_max=1992,
        noise_texture_override="reel_tape",
        description="Melodiya — GOST 13699-55 + TASMA-6 @ 15 ips (BW ≤ 12.5 kHz)",
    ),
    # -----------------------------------------------------------------------
    # NBC RADIO STUDIOS USA (1935–1975)
    # Console: custom NBC/RCA. Tape: Ampex 200 (from 1948), then 350/440.
    # Microphone: RCA 44B dominant, later RCA 77DX.
    # Characteristic: warm ribbon mid-range, natural, limited HF,
    # slight broadcast limiting effect at 8–10 kHz.
    # -----------------------------------------------------------------------
    "nbc_radio": ProvenanceAdjustment(
        provenance_label="NBC Radio Studios USA",
        mic_delta=[
            # RCA 44B ribbon: warm, flat 100 Hz–8 kHz, rolloff above
            (6000.0, -0.5),
            (8000.0, -1.5),
            (10000.0, -4.0),
            (12000.0, -12.0),
        ],
        console_delta=[
            # NBC custom: slight mid-forward, broadcast limiting shelf at 8 kHz
            (200.0, +0.3),
            (1000.0, 0.0),
            (4000.0, +0.5),
            (6000.0, +0.3),
            (8000.0, -0.5),
            (10000.0, -2.0),
            (12000.0, -5.0),
        ],
        tape_delta=[
            # Ampex 200 @ 15 ips: LF warmth, moderate HF
            (40.0, +0.5),
            (200.0, +0.8),
            (600.0, +0.3),
            (1000.0, 0.0),
            (5000.0, -0.3),
            (8000.0, -1.0),
            (10000.0, -3.0),
        ],
        era_min=1935,
        era_max=1978,
        description="NBC Radio — RCA 44B + Ampex 200 @ 15 ips",
    ),
    # -----------------------------------------------------------------------
    # NDR HAMBURG / Norddeutscher Rundfunk (1945–present)
    # Console: Siemens W395 (1950s–1960s), EMT-250 reverb units, later Neve/SSL.
    # Tape: Telefunken M15A (standard German broadcast machine, 38 cm/s = 15 ips).
    # Characteristic: very clean, well-maintained equipment, German broadcast standard
    # (DIN 45511: flat 40 Hz–15 kHz); slight Siemens transformer warmth.
    # -----------------------------------------------------------------------
    "ndr_hamburg": ProvenanceAdjustment(
        provenance_label="NDR Hamburg (Norddeutscher Rundfunk)",
        mic_delta=[
            # NDR: Neumann U87 standard from 1967, before that U47/M49
            (8000.0, +0.5),
            (12000.0, +0.3),
            (16000.0, 0.0),
        ],
        console_delta=[
            # Siemens W395: transformer warmth similar to Neve (less extreme)
            (60.0, +0.3),
            (200.0, +0.5),
            (400.0, +0.3),
            (1000.0, 0.0),
            (6000.0, 0.0),
            (12000.0, +0.5),
            (16000.0, 0.0),
        ],
        tape_delta=[
            # Telefunken M15A @ 38 cm/s (15 ips): extended HF vs lower speeds
            (100.0, +0.2),
            (500.0, 0.0),
            (1000.0, 0.0),
            (8000.0, +0.3),
            (12000.0, 0.0),
            (15000.0, -0.5),
            (18000.0, -2.0),
        ],
        era_min=1950,
        era_max=2000,
        description="NDR Hamburg — Siemens W395 + Telefunken M15A @ 38 cm/s",
    ),
    # -----------------------------------------------------------------------
    # PHILIPS EINDHOVEN / PolyGram NL (1950–1975)
    # Console: Philips custom (NL). Tape: Philips LDL (early), then BASF 900.
    # Microphone: B&K 4006 omni (very flat, used from 1960s).
    # Characteristic: very neutral, clean, B&K flat response, Dutch precision.
    # -----------------------------------------------------------------------
    "philips_eindhoven": ProvenanceAdjustment(
        provenance_label="Philips Eindhoven / PolyGram",
        mic_delta=[
            # B&K 4006: ruler-flat 20 Hz–20 kHz (DPA 2004 successor)
            (1000.0, 0.0),
            (10000.0, 0.0),
            (20000.0, 0.0),
        ],
        console_delta=[
            # Philips NL custom: very clean, slight HF roll at 14 kHz
            (1000.0, 0.0),
            (8000.0, 0.0),
            (14000.0, -0.5),
            (18000.0, -1.5),
        ],
        tape_delta=[
            # BASF LH900 @ 38 cm/s: near-flat
            (100.0, 0.0),
            (1000.0, 0.0),
            (10000.0, 0.0),
            (18000.0, -0.5),
        ],
        era_min=1950,
        era_max=1978,
        description="Philips Eindhoven — B&K 4006 + BASF LH900",
    ),
    # -----------------------------------------------------------------------
    # RIAS BERLIN (West) — Rundfunk im amerikanischen Sektor (1946–1994)
    # Console: Siemens W395 (like NDR), then Neve 1073 from ~1975.
    # Tape: Telefunken M15A at standard ARD speed (38 cm/s).
    # Characteristic: West German broadcast standard, clean, Neve warmth post-1975.
    # -----------------------------------------------------------------------
    "rias_berlin": ProvenanceAdjustment(
        provenance_label="RIAS Berlin (Rundfunk im amerikanischen Sektor)",
        mic_delta=[],  # Standard German broadcast mics — era curve sufficient
        console_delta=[
            # Pre-1975: Siemens W395. Post-1975: Neve 1073 (+3 dB shelf @ 12 kHz).
            # Mid-point blend:
            (60.0, +0.3),
            (200.0, +0.5),
            (1000.0, 0.0),
            (6000.0, +0.5),
            (10000.0, +1.0),
            (12000.0, +1.5),
            (16000.0, +1.0),
        ],
        tape_delta=[
            # M15A same as NDR
            (100.0, +0.2),
            (1000.0, 0.0),
            (8000.0, +0.3),
            (15000.0, -0.5),
            (18000.0, -2.0),
        ],
        era_min=1946,
        era_max=1995,
        description="RIAS Berlin — Siemens W395 / Neve 1073 + Telefunken M15A",
    ),
}

# Alias mappings: alternative spellings → canonical key
_ALIASES: dict[str, str] = {
    "abbey": "abbey_road",
    "emi_london": "abbey_road",
    "emi": "abbey_road",
    "capitol": "capitol_studios",
    "columbia": "columbia_30th",
    "columbia_church": "columbia_30th",
    "ddr": "ddr_rundfunk",
    "dt64": "ddr_rundfunk",
    "rundfunk": "ddr_rundfunk",
    "dg": "deutsche_grammophon",
    "archiv": "deutsche_grammophon",
    "grammophon": "deutsche_grammophon",
    "mercury": "mercury_living",
    "living_presence": "mercury_living",
    "melodiya": "melodiya_soviet",
    "soviet": "melodiya_soviet",
    "gost": "melodiya_soviet",
    "nbc": "nbc_radio",
    "ndr": "ndr_hamburg",
    "philips": "philips_eindhoven",
    "polygram": "philips_eindhoven",
    "rias": "rias_berlin",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _era_fade(adj: ProvenanceAdjustment, era_decade: int | None) -> float:
    """Gibt a 0..1 relevance weight based on how far era_decade is from the zurück.
    adjustment's valid range.  Fades to 0 over ±20 years outside the range."""
    if era_decade is None:
        return 0.7  # partial confidence when era unknown
    d = int(era_decade)
    if adj.era_min <= d <= adj.era_max:
        return 1.0
    gap = max(0, adj.era_min - d, d - adj.era_max)
    return float(max(0.0, 1.0 - gap / 20.0))


def get_provenance_adjustment(
    provenance_hint: str,
    era_decade: int | None = None,
) -> ProvenanceAdjustment | None:
    """Look up the provenance adjustment for a given hint string.

    Matching is case-insensitive and works on partial substrings (e.g. "ndr"
    matches "ndr_hamburg").  Returns ``None`` if nothing matches or era
    relevance < 0.10.

    Args:
        provenance_hint: Free-form string: label name, studio, archive,
                         broadcaster (e.g. "RIAS", "DG", "Mercury").
        era_decade:      Recording decade for era-fade relevance check.

    Returns:
        :class:`ProvenanceAdjustment` or ``None``.
    """
    if not provenance_hint:
        return None

    hint = provenance_hint.strip().lower().replace(" ", "_").replace("-", "_")

    # Resolve alias
    canonical = _ALIASES.get(hint, hint)

    # Exact match
    adj = _PROVENANCE_DB.get(canonical)

    # Partial substring match if no exact hit
    if adj is None:
        for key, val in _PROVENANCE_DB.items():
            if hint in key or key in hint:
                adj = val
                break

    if adj is None:
        logger.debug("broadcast_archive_db: no match for provenance_hint=%r", provenance_hint)
        return None

    relevance = _era_fade(adj, era_decade)
    if relevance < 0.10:
        logger.debug(
            "broadcast_archive_db: %s era_fade=%.2f < 0.10 — ignored for decade=%s",
            adj.provenance_label,
            relevance,
            era_decade,
        )
        return None

    logger.debug(
        "broadcast_archive_db: matched %r → %s era_fade=%.2f",
        provenance_hint,
        adj.provenance_label,
        relevance,
    )
    return adj


def apply_provenance_to_chain(
    mic_bp: list[tuple[float, float]],
    console_bp: list[tuple[float, float]],
    tape_bp: list[tuple[float, float]],
    adj: ProvenanceAdjustment,
    era_decade: int | None = None,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], list[tuple[float, float]]]:
    """Führt zusammen: provenance deltas into the three breakpoint lists.

    Adds ``adj.mic_delta`` / ``console_delta`` / ``tape_delta`` at matching Hz
    breakpoints by linear interpolation, scaled by era_fade relevance.

    Args:
        mic_bp, console_bp, tape_bp:  Existing chain breakpoints (from TRP era tables).
        adj:                           ProvenanceAdjustment to overlay.
        era_decade:                    For era_fade weight computation.

    Returns:
        (mic_bp_merged, console_bp_merged, tape_bp_merged) — new breakpoint lists.
    """
    weight = _era_fade(adj, era_decade)

    def _merge(
        base: list[tuple[float, float]],
        delta: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        if not delta:
            return base
        # Union of Hz breakpoints; interpolate from both
        all_hz = sorted({hz for hz, _ in base} | {hz for hz, _ in delta})

        def _lerp(pts: list[tuple[float, float]], hz: float) -> float:
            if not pts:
                return 0.0
            if hz <= pts[0][0]:
                return pts[0][1]
            if hz >= pts[-1][0]:
                return pts[-1][1]
            for i in range(len(pts) - 1):
                h0, d0 = pts[i]
                h1, d1 = pts[i + 1]
                if h0 <= hz < h1:
                    t = (hz - h0) / (h1 - h0 + 1e-12)
                    return d0 + t * (d1 - d0)
            return pts[-1][1]

        merged = [(hz, _lerp(base, hz) + weight * _lerp(delta, hz)) for hz in all_hz]
        return merged

    return (
        _merge(mic_bp, list(adj.mic_delta)),
        _merge(console_bp, list(adj.console_delta)),
        _merge(tape_bp, list(adj.tape_delta)),
    )


def list_provenance_keys() -> list[str]:
    """Gibt all known canonical provenance keys zurück."""
    return sorted(_PROVENANCE_DB.keys())


__all__ = [
    "ProvenanceAdjustment",
    "get_provenance_adjustment",
    "apply_provenance_to_chain",
    "list_provenance_keys",
]
