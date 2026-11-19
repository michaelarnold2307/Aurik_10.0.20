"""shared.enums — UI-seitige Enums für Aurik 10.0.0.

Diese Datei definiert vereinfachte Enums für die Benutzeroberfläche.
Sie sind auf die Auswahloptionen der UI abgestimmt und dienen als
stabiles Interface-Paket zwischen Frontend und Backend-Core.

Kanonische Backend-Enums (mit vollständiger Detail-Ausprägung):
    - MediumType (full):    core.musical_quality_assurance.MediumType
    - ProcessingMode (full): core.processing_modes.ProcessingMode

Design-Entscheidung:  Die UI verwendet eine gröbere Granularität
(z.B. VINYL statt VINYL_33/VINYL_45), um die Bedienung für Nutzer
ohne Fachkenntnisse zu vereinfachen.
"""

from enum import Enum


class MediumType(Enum):
    """Vereinfachte Tonträger-Typen für die Benutzeroberfläche.

    Diese Werte entsprechen den UI-Dropdown-Optionen in main_window.py.
    Für die vollständige Liste forensischer Materialtypen siehe
    ``core.musical_quality_assurance.MediumType``.
    """

    VINYL = "VINYL"
    """Schallplatte (LP/Single — alle Drehzahlen)."""

    CASSETTE = "CASSETTE"
    """Compact Cassette / Magnetband-Kassette."""

    DAT = "DAT"
    """Digital Audio Tape."""

    CD = "CD"
    """Compact Disc (digitales Medium)."""

    MP3 = "MP3"
    """MP3 / verlustbehaftetes digitales Format."""

    SHELLAC = "SHELLAC"
    """Schellack-78rpm-Platte (historisch)."""

    WIRE = "WIRE"
    """Drahtbandaufnahme (Wire Recording, historisch)."""


class ProcessingMode(Enum):
    """Verarbeitungs-Modi für die Benutzeroberfläche.

    Diese Werte entsprechen den UI-Dropdown-Einträgen des Modus-Selektors
    in main_window.py. Sie sind von den Magic-Button-Modi (RESTORATION /
    STUDIO_2026 in ``core.processing_modes``) zu unterscheiden — hier
    stehen feingranulare, expertenseitige Optionen zur Verfügung.
    """

    GENTLE = "GENTLE"
    """Sanfte Verarbeitung — Klangcharakter maximal erhalten."""

    BALANCED = "BALANCED"
    """Ausgewogene Restaurierung (empfohlen für die meisten Aufnahmen)."""

    AGGRESSIVE = "AGGRESSIVE"
    """Aggressive Bereinigung — maximale Rauschunterdrückung."""

    ARCHIVE = "ARCHIVE"
    """Archivmodus — maximale Originalerhaltung, minimaler Eingriff."""

    MASTERING = "MASTERING"
    """Sanfte klangliche Aufwertung auf Mastering-Niveau."""
