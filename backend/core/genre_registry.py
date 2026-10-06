"""§G9 (copilot-instructions.md) — Kanonische Genre-Registry für Aurik.

**Befund 2026-10-06:** Genre-Labels wurden von vier Konsumenten mit vier
verschiedenen, jeweils unvollständigen Verfahren aufgelöst:

===============  ==========================================  =========================
Konsument        Verfahren                                   Ausfall (gemessen)
===============  ==========================================  =========================
``genre_classifier``      lokale ``label_map`` (dritte Kopie!)        ``Deutscher Schlager`` → ``{}``
``genre_goal_profile``    Fuzzy-Teilstring-Match                     ``Klassik/Oper/Soul/R&B/Hip-Hop/Country`` → ``unknown``
``tonal_reference_profile``  5-Einträge-``_ALIASES`` + Exakt-Lookup     ``Deutscher Schlager`` → Delta ``0,000``
``unified_restorer_v3``   if/elif-Kette auf 16 Labels               ``schlager``/``ambient``/``world`` ohne Zweig
===============  ==========================================  =========================

Der kanonische Klassifikator (``GermanSchlagerClassifier``) gibt ``Deutscher
Schlager``/``Internationaler Schlager`` aus — genau diese Labels hatte **kein**
Konsument aufgelöst. Folge: Für deutschsprachiges Schlager-Material (der
Hauptanwendungsfall, ``lang_de_score >= 0.55``) entfielen Genre-Spektralvorgabe
und Restaurierungsprofil vollständig, während das *unschärfere* Label
``Schlager`` (Sprache unsicher) sie erhielt — die Wirkung war invertiert zur
Konfidenz.

**Diese Datei ist die EINE Quelle** für:
  - die kanonische Genre-Menge (``CANONICAL_GENRES``),
  - die Alias-Auflösung (``resolve_genre``) inkl. der Sprach-Präfixe
    (``Deutscher …``/``Internationaler …``) und der Schreibvarianten,
  - die Zuordnung je Konsument (Restaurierungsprofil, Goal-Profil, Delta-Tabelle).

**Kein Fuzzy-Matching, keine zweite Kopie einer Tabelle, keine Alias-Duplikate
als Dict-Schlüssel.** Unbekannte Labels werden ehrlich als ``None`` gemeldet
(§V6 copilot-instructions.md: kein stilles Degradieren), statt per Teilstring
auf ein falsches Genre zu fallen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "CANONICAL_GENRES",
    "GenreResolution",
    "delta_key",
    "goal_profile_key",
    "jnd_key",
    "normalize_genre",
    "restoration_profile_key",
    "resolve_genre",
]

# ---------------------------------------------------------------------------
# Kanonische Genre-Menge
# ---------------------------------------------------------------------------
# Schlüsselraum von ``genre_classifier.GENRE_RESTORATION_PROFILES`` (klein
# geschrieben) plus die Schlager-Familie. Bewusst KEINE neuen Genres — die
# Registry konsolidiert nur, was fachlich bereits existiert (§V7 copilot-instructions.md).
CANONICAL_GENRES: tuple[str, ...] = (
    "schlager",
    "walzer",
    "marsch",
    "jazz",
    "klassik",
    "oper",
    "rock",
    "pop",
    "blues",
    "soul_rnb",
    "country",
    "folk",
    "funk",
    "electronic",
    "hiphop",
    "metal",
    "latin",
    "gospel",
    "reggae",
    "ambient",
    "world",
)

#: Schreibweise (normalisiert) → kanonische ID. Quelle: alle im Projekt beobachteten
#: Vorkommen (Klassifikator-Labels, Tabellen-Schlüssel, Aufrufer-Strings).
_GENRE_ALIASES: dict[str, str] = {
    # ── Schlager-Familie ────────────────────────────────────────────────────
    "schlager": "schlager",
    "german_pop": "schlager",
    "german_schlager": "schlager",
    "deutscher_schlager": "schlager",
    "internationaler_schlager": "schlager",
    "volkstuemlich": "schlager",
    "volkstümlich": "schlager",
    "volksmusik": "schlager",
    "schunkel": "schlager",
    "schunkelmusik": "schlager",
    "discoschlager": "schlager",
    "disco_schlager": "schlager",
    "schlager_1950s": "schlager",
    "schlager_modern": "schlager",
    # ── Weitere Genres ──────────────────────────────────────────────────────
    "walzer": "walzer",
    "marsch": "marsch",
    "jazz": "jazz",
    "bebop": "jazz",
    "big_band": "jazz",
    "swing": "jazz",
    "klassik": "klassik",
    "classical": "klassik",
    "classic": "klassik",
    "klassische_musik": "klassik",
    "orchestral": "klassik",
    "kammermusik": "klassik",
    "oper": "oper",
    "opera": "oper",
    "chor": "oper",
    "rock": "rock",
    "hard_rock": "rock",
    "punk": "rock",
    "rock_metal": "rock",
    "pop": "pop",
    "vocal_pop": "pop",
    "blues": "blues",
    "soul_rnb": "soul_rnb",
    "soul": "soul_rnb",
    "soul/r&b": "soul_rnb",
    "soul_r&b": "soul_rnb",
    "r&b": "soul_rnb",
    "rnb": "soul_rnb",
    "rhythm_and_blues": "soul_rnb",
    "country": "country",
    "bluegrass": "country",
    "folk": "folk",
    "singer_songwriter": "folk",
    "funk": "funk",
    "electronic": "electronic",
    "dance": "electronic",
    "edm": "electronic",
    "hiphop": "hiphop",
    "hip_hop": "hiphop",
    "rap": "hiphop",
    "trap": "hiphop",
    "metal": "metal",
    "heavy_metal": "metal",
    "latin": "latin",
    "samba": "latin",
    "bossa_nova": "latin",
    "tango": "latin",
    "flamenco": "latin",
    "gospel": "gospel",
    "reggae": "reggae",
    "dub": "reggae",
    "ambient": "ambient",
    "world": "world",
    "world_music": "world",
    "chanson": "schlager",
}

#: Kanonische ID → abweichender Schlüssel in ``genre_goal_profile._GENRE_PROFILES``.
#: Identische Fälle werden über ``canonical`` aufgelöst.
_GOAL_KEYS: dict[str, str] = {
    "klassik": "classical",
    "soul_rnb": "rnb",
    "hiphop": "hiphop",
}

#: Kanonische Genres OHNE Goal-Profil (heute kein Eintrag in ``_GENRE_PROFILES``).
#: Bewusst explizit: diese Genres bleiben im Goal-Profil ersatzpflichtig.
_GOAL_KEY_ABSENT: frozenset[str] = frozenset(
    {"walzer", "marsch", "oper", "blues", "country", "funk", "latin", "gospel", "reggae", "ambient", "world"}
)

#: Kanonische ID → abweichender Schlüssel in ``tonal_reference_profile._GENRE_DELTAS``.
_DELTA_KEYS: dict[str, str] = {
    "metal": "heavy_metal",
    "soul_rnb": "soul/r&b",
    "hiphop": "hip-hop",
}

#: Untergenre-Schreibweisen mit EIGENEM Delta-Eintrag — bleiben spezifisch
#: (``bebop`` bleibt ``bebop``, nicht ``jazz``).
_DELTA_SUBGENRES: frozenset[str] = frozenset(
    {
        "bebop",
        "big_band",
        "swing",
        "kammermusik",
        "chor",
        "hard_rock",
        "punk",
        "trap",
        "dub",
        "bluegrass",
        "singer_songwriter",
        "bossa_nova",
        "samba",
        "tango",
        "flamenco",
        "chanson",
    }
)

#: Schlager-Familie: das Basisprofil ist ``schlager``, die Abweichung kommt aus
#: ``_SUBGENRE_EXTENSIONS``. Diese Erweiterung wird separat transportiert, damit
#: die Registry keine zweite Kopie der Profil-Tabelle benötigt.
_RESTORATION_BASE: str = "schlager"
_RESTORATION_EXTENSIONS: dict[str, str] = {
    "walzer": "walzer",
    "marsch": "marsch",
    "disco_schlager": "discoschlager",
    "volksmusik": "volksmusik",
    "schunkel": "schunkel",
    "schunkelmusik": "schunkel",
    "schlager_1950s": "schlager_1950s",
    "schlager_modern": "schlager_modern",
}

#: Kanonische ID → Schlüssel in ``perceptual_tuning.GENRE_JND_FACTOR`` /
#: ``GENRE_DYNAMICS_PREFERENCE`` (englische Vokabel, vierte im Projekt).
#: Genau diese Zuordnung fehlte: ``Klassik`` fiel auf den generischen Faktor 1,00
#: statt auf ``classical`` = 0,80 zurück — bei Klassik liefen dadurch WENIGER
#: Phasen, das Gegenteil der Absicht (§G6 copilot-instructions.md).
_JND_KEYS: dict[str, str] = {
    "klassik": "classical",
    "oper": "opera",
    "hiphop": "hip_hop",
    "soul_rnb": "rnb",
}

#: Genres ohne Eintrag im JND-Vokabular (bleiben beim dokumentierten Default 1,0).
_JND_KEY_ABSENT: frozenset[str] = frozenset({"walzer", "marsch", "country", "gospel", "ambient"})


#: Sprach-Präfixe des kanonischen Klassifikators (``_determine_genre_label``).
#: ``Deutscher Schlager`` / ``Internationaler Walzer`` → Basislabels.
_PREFIXES: tuple[str, ...] = ("deutscher ", "deutsche ", "internationaler ", "internationale ")


def _normalize_raw(label: Any) -> str:
    """Schreibweise vereinheitlichen: trimmen, klein, Präfixe weg, Trenner einheitlich."""
    text = str(label or "").strip().lower()
    for prefix in _PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    # Umlaute/ß nicht transliterieren (deutsche Labels sind deutsche Labels),
    # aber Trennzeichen vereinheitlichen: "Hip Hop" / "Hip-Hop" / "hip_hop" → "hip_hop"
    text = text.replace("-", "_").replace(" ", "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


@dataclass(frozen=True)
class GenreResolution:
    """Aufgelöstes Genre-Label samt Ziel-Schlüsseln je Konsument.

    ``canonical is None`` bedeutet: **unbekanntes Genre** — die Konsumenten
    müssen ihren dokumentierten Ersatzpfad nehmen (und ihn protokollieren),
    statt es per Ähnlichkeit auf ein falsches Genre zu ziehen.
    """

    raw: str
    normalized: str
    canonical: str | None
    restoration_key: str | None
    restoration_extension_key: str | None
    goal_key: str | None
    delta_key: str | None
    is_schlager_family: bool


def resolve_genre(label: Any) -> GenreResolution:
    """Löst ein beliebiges Genre-Label in die kanonischen Ziel-Schlüssel auf.

    Args:
        label: Genre-Label in beliebiger Schreibweise (z. B. ``"Deutscher Schlager"``,
            ``"soul/r&b"``, ``"Hip-Hop"``, ``"classical"``).

    Returns:
        :class:`GenreResolution`. Bei unbekanntem Label sind ``canonical`` und alle
        Ziel-Schlüssel ``None``.
    """
    normalized = _normalize_raw(label)
    canonical = _GENRE_ALIASES.get(normalized)
    if canonical is None:
        return GenreResolution(str(label or ""), normalized, None, None, None, None, None, False)

    # Schlager-Familie: Basisprofil + Abweichung (Walzer/Marsch/Disco/Volksmusik …)
    extension = _RESTORATION_EXTENSIONS.get(normalized)
    restoration_key = _RESTORATION_BASE if extension is not None else canonical

    goal_key = _GOAL_KEYS.get(canonical)
    if goal_key is None and canonical not in _GOAL_KEY_ABSENT:
        goal_key = canonical

    delta_key = normalized if normalized in _DELTA_SUBGENRES else _DELTA_KEYS.get(canonical, canonical)
    return GenreResolution(
        raw=str(label or ""),
        normalized=normalized,
        canonical=canonical,
        restoration_key=restoration_key,
        restoration_extension_key=extension,
        goal_key=goal_key,
        delta_key=delta_key,
        is_schlager_family=canonical in {"schlager", "walzer", "marsch"},
    )


def normalize_genre(label: Any) -> str | None:
    """Kanonische Genre-ID oder ``None`` (unbekannt)."""
    return resolve_genre(label).canonical


def restoration_profile_key(label: Any) -> str | None:
    """Schlüssel in ``GENRE_RESTORATION_PROFILES`` oder ``None``."""
    return resolve_genre(label).restoration_key


def goal_profile_key(label: Any) -> str | None:
    """Schlüssel in ``genre_goal_profile._GENRE_PROFILES`` oder ``None``."""
    return resolve_genre(label).goal_key


def jnd_key(label: Any) -> str | None:
    """Schlüssel in ``perceptual_tuning.GENRE_JND_FACTOR`` oder ``None``.

    ``None`` bedeutet: kein genrespezifischer JND-Faktor vorhanden — der Aufrufer
    nimmt seinen dokumentierten Default (1,0) und protokolliert den Ersatzpfad.
    """
    canonical = resolve_genre(label).canonical
    if canonical is None or canonical in _JND_KEY_ABSENT:
        return None
    return _JND_KEYS.get(canonical, canonical)


def delta_key(label: Any) -> str | None:
    """Schlüssel in ``tonal_reference_profile._GENRE_DELTAS`` oder ``None``."""
    return resolve_genre(label).delta_key
