"""BWF Metadata Writer — Broadcast Wave Format Unterstützung.

§Format-Gap: Schreibt BWF-konforme bext- und iXML-Chunks in WAV-Dateien.
Wird nach dem soundfile-Export aufgerufen — soundfile/libndfile allein
kann keine benutzerdefinierten RIFF-Chunks schreiben.

Referenz:
    EBU Tech 3285 — Broadcast Wave Format (BWF)
    ITU-R BS.2088 — iXML in BWF
    AES31-3 — RIFF Chunk Specification

Usage::

    from backend.core.bwf_writer import write_bwf_chunks
    write_bwf_chunks("output.wav", originator="Aurik 10", description="Restauriert")

Autor: Aurik 10 — 11. Juli 2026
"""

from __future__ import annotations

import logging
import struct
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Chunk-Konstanten ────────────────────────────────────────────────────────
_BEXT_CHUNK_SIZE = 602  # Standard BWF bext chunk (EBU Tech 3285)
_RIFF_HEADER_SIZE = 12  # "RIFF" + size + "WAVE"


def _build_bext_chunk(
    description: str = "",
    originator: str = "",
    originator_ref: str = "",
    origination_date: str = "",
    origination_time: str = "",
    time_reference_low: int = 0,
    time_reference_high: int = 0,
    version: int = 1,
    umid: bytes = b"",
    loudness_value: int = 0,
    loudness_range: int = 0,
    max_truepeak: int = 0,
    max_momentary: int = 0,
    max_shortterm: int = 0,
) -> bytes:
    """Baut einen bext-Chunk gemäß EBU Tech 3285 (602 Bytes Payload).

    Der bext-Chunk enthält minimal: description, originator, originator_ref,
    origination_date, origination_time, time_reference. Wir erweitern um
    UMID (64 Bytes) und reservierte Felder.
    """
    # Felder auf feste Längen padden (ASCII, null-terminiert)
    desc = description[:256].ljust(256, "\x00")[:256]
    orig = originator[:32].ljust(32, "\x00")[:32]
    oref = originator_ref[:32].ljust(32, "\x00")[:32]
    odate = origination_date[:10].ljust(10, "\x00")[:10]
    otime = origination_time[:8].ljust(8, "\x00")[:8]

    # UMID: 64 Bytes (0 wenn nicht gesetzt)
    if len(umid) < 64:
        umid = umid + b"\x00" * (64 - len(umid))
    umid = umid[:64]

    # BWF v2 Loudness (optional, hier auf 0 = nicht gesetzt)
    loudness = struct.pack(
        "<HHHHH",
        loudness_value & 0xFFFF,
        loudness_range & 0xFFFF,
        max_truepeak & 0xFFFF,
        max_momentary & 0xFFFF,
        max_shortterm & 0xFFFF,
    )

    # Coding History: leer lassen (wird von Aurik nicht genutzt)
    _coding_history = b"\x00" * 602
    # Tatsächlich: bext ist 602 Bytes FEST (ohne Coding History variabel)
    # Für striktes BWF v1: 602 Bytes total

    chunk_data = (
        desc.encode("ascii", errors="replace")
        + orig.encode("ascii", errors="replace")
        + oref.encode("ascii", errors="replace")
        + odate.encode("ascii", errors="replace")
        + otime.encode("ascii", errors="replace")
        + struct.pack("<q", time_reference_low)  # 8 Byte
        + struct.pack("<q", time_reference_high)  # 8 Byte (BWF v2)
        + struct.pack("<H", version)  # Version (2 Byte)
        + umid  # 64 Byte
        + loudness  # 10 Byte (BWF v2 reserved)
        + b"\x00" * 148  # Reserved auf 602 Bytes auffüllen
    )

    # Auf exakte 602 Bytes trimmen/padden
    chunk_data = chunk_data[:602].ljust(602, b"\x00")

    # Chunk-Header: "bext" + size (602)
    return b"bext" + struct.pack("<I", 602) + chunk_data


def _build_ixml_chunk(xml_data: str = "") -> bytes:
    """Baut einen iXML-Chunk (variable Länge, UTF-8 XML).

    Wenn kein xml_data angegeben, wird ein minimaler Aurik-iXML-Block erzeugt.
    """
    if not xml_data:
        xml_data = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<BWFXML>\n"
            "  <IXML_VERSION>1.55</IXML_VERSION>\n"
            "  <PROJECT>Aurik 10 Restoration</PROJECT>\n"
            f"  <SCENE>Restored {time.strftime('%Y-%m-%d %H:%M:%S')}</SCENE>\n"
            "  <NOTE>Processed by Aurik 10 — Intelligent Music Restoration</NOTE>\n"
            "</BWFXML>\n"
        )

    xml_bytes = xml_data.encode("utf-8")
    # iXML muss auf ungerade Byte-Grenze aligniert werden (RIFF-Regel)
    if len(xml_bytes) % 2 == 0:
        xml_bytes += b"\x00"

    return b"iXML" + struct.pack("<I", len(xml_bytes)) + xml_bytes


def _write_bwf_chunks_raw(filepath: str | Path, bext: bytes, ixml: bytes) -> bool:
    """Appendet bext- und iXML-Chunks direkt in eine bestehende WAV-Datei.

    Dies ist ein Low-Level-Ansatz der die RIFF-Struktur parst und
    Chunks VOR dem data-Chunk einfügt.

    Args:
        filepath: Pfad zur WAV-Datei (muss existieren).
        bext:     Fertiger bext-Chunk (inkl. Header "bext" + size).
        ixml:     Fertiger iXML-Chunk (inkl. Header "iXML" + size).

    Returns:
        True wenn erfolgreich.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        logger.warning("BWF-Writer: Datei nicht gefunden: %s", filepath)
        return False

    try:
        with open(filepath, "r+b") as f:
            # RIFF-Header lesen und validieren
            riff_id = f.read(4)
            if riff_id != b"RIFF":
                logger.debug("BWF-Writer: Keine RIFF-Datei: %s", filepath)
                return False

            f.read(4)
            wave_id = f.read(4)
            if wave_id != b"WAVE":
                logger.debug("BWF-Writer: Kein WAVE-Format: %s", filepath)
                return False

            # Chunks scannen bis "data" gefunden wird
            # Wir schreiben bext + ixml VOR den data-Chunk
            chunk_positions: list[tuple[int, bytes, int]] = []  # (position, id, size)
            pos = f.tell()

            while True:
                chunk_header = f.read(8)
                if len(chunk_header) < 8:
                    break
                chunk_id = chunk_header[:4]
                chunk_size = struct.unpack("<I", chunk_header[4:8])[0]
                chunk_positions.append((pos, chunk_id, chunk_size))
                f.seek(chunk_size + (chunk_size % 2), 1)  # 16-bit alignment
                pos = f.tell()

            # "data"-Chunk finden
            data_pos = None
            for cpos, cid, csize in chunk_positions:
                if cid == b"data":
                    data_pos = cpos
                    break

            if data_pos is None:
                logger.warning("BWF-Writer: Kein data-Chunk in %s", filepath)
                return False

            # ----- Strategie: Neuschreiben -----
            # Da RIFF-Chunks sequenziell sind und wir Chunks EINFÜGEN müssen,
            # lesen wir den gesamten Inhalt ab data_pos und schreiben neu.
            f.seek(0)
            entire_file = f.read()

            # data-Chunk Offset und Größe aus der gescannten Liste
            data_chunk_offset = data_pos
            # Größe des data-Chunks (Header 8 + payload)
            data_size = None
            for cpos, cid, csize in chunk_positions:
                if cid == b"data":
                    data_size = csize
                    break

            if data_size is None:
                return False

            # Neuen Inhalt bauen: RIFF-Header + plugin-chunks + data-Chunk
            _riff_header = entire_file[:12]

            # Alle chunks zwischen RIFF-Header und data-Chunk sammeln
            pre_data_chunks = entire_file[12:data_chunk_offset]

            # data-Chunk (Header + Payload) extrahieren
            data_chunk = entire_file[data_chunk_offset : data_chunk_offset + 8 + data_size + (data_size % 2)]

            # Neuer Chunk-Bereich: bestehende pre-data + neue bext + ixml
            new_pre_data = pre_data_chunks + bext + ixml

            # Neue RIFF-Gesamtgröße berechnen
            new_riff_size = len(new_pre_data) + len(data_chunk)

            # Neuschreiben
            f.seek(0)
            f.write(b"RIFF")
            f.write(struct.pack("<I", new_riff_size))
            f.write(b"WAVE")
            f.write(new_pre_data)
            f.write(data_chunk)
            f.truncate()

        logger.info("BWF-Chunks geschrieben: bext=%d + iXML=%d → %s", len(bext), len(ixml), filepath)
        return True

    except Exception as exc:
        logger.warning("BWF-Writer Fehler für %s: %s", filepath, exc)
        return False


def write_bwf_chunks(
    filepath: str | Path,
    description: str = "",
    originator: str = "Aurik 10",
    originator_ref: str = "",
    transfer_chain: list[str] | None = None,
    **kwargs,
) -> bool:
    """Schreibt BWF bext- und iXML-Chunks in eine WAV-Datei.

    Args:
        filepath:       Pfad zur existierenden WAV-Datei.
        description:    Beschreibung (max 256 Zeichen).
        originator:     Erzeuger (z.B. "Aurik 10.0.0").
        originator_ref: Referenz-ID.
        **kwargs:       Weitere bext-Felder (siehe _build_bext_chunk).

    Returns:
        True wenn erfolgreich.
    """
    bext = _build_bext_chunk(
        description=description + (" | Chain: " + " → ".join(transfer_chain) if transfer_chain else ""),
        originator=originator,
        originator_ref=originator_ref,
        **kwargs,
    )
    ixml = _build_ixml_chunk()
    return _write_bwf_chunks_raw(filepath, bext, ixml)


__all__ = ["write_bwf_chunks"]
