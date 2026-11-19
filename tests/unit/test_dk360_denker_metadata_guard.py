"""D-K3-60 (Bughunt 2026-10-08): §V6-Wächter am ndarray-Metadaten-Verlustpfad.

Der AurikDenker konvertiert ein rohes ndarray-Ergebnis des RestaurierDenkers in
einen SimpleNamespace OHNE Metadaten (``metadata={}``). Bis 10.12.x geschah das
STILL — der komplette §8.6/§8.7-Metadaten-Baum (WCS, hybrid_engineer_vector,
threshold_evidence, audibility_gate) kann den Export dadurch unbemerkt nicht
erreichen (D-K3-60-Familie: „Prüfung existiert, sieht aber nichts").

Fix: §V6-Warnung (copilot-instructions.md) im ndarray-Zweig — nie still
degradieren. Zusätzlich bleiben die verifizierten Kettenglieder des
Normalfalls (RestaurierErgebnis) gepinnt (Ergänzung zum Weltklasse-Validator).

Kontrakte:
1. Der ndarray-Zweig trägt den §V6-Warnaufruf (Source-Pin).
2. Die Warnung nennt die Metadaten-Folge (WCS/HPI/Audibility) — Diagnostik.
3. Normalfall-Kette bleibt vollständig: ``_rest_metadata = dict(_meta_raw)``
   + ``metadata={**_rest_metadata`` (kein Metadaten-Verlust).
"""

from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "denker" / "aurik_denker.py"


def _fenster_nach_ndarray_zweig(src: str) -> str:
    idx = src.find("if isinstance(rest, np.ndarray):")
    assert idx != -1, "ndarray-Zweig im AurikDenker nicht gefunden"
    return src[idx : idx + 1600]


def test_ndarray_zweig_traegt_v6_warnung() -> None:
    """Kontrakt 1: §V6-Warnung (copilot-instructions.md) im Verlustpfad."""
    fenster = _fenster_nach_ndarray_zweig(_SRC.read_text(encoding="utf-8"))
    assert "logger.warning(" in fenster
    assert "§V6 (copilot-instructions.md)" in fenster
    assert "D-K3-60" in fenster


def test_warnung_nennt_metadaten_folgen() -> None:
    """Kontrakt 2: Diagnostik benennt die betroffenen Metadaten."""
    fenster = _fenster_nach_ndarray_zweig(_SRC.read_text(encoding="utf-8"))
    assert "Metadaten" in fenster
    assert "WCS" in fenster
    assert "Audibility" in fenster


def test_normalfall_kette_bleibt_gepinnt() -> None:
    """Kontrakt 3: RestaurierErgebnis-Pfad erbt die Metadaten vollständig."""
    src = _SRC.read_text(encoding="utf-8")
    assert "_rest_metadata = dict(_meta_raw) if isinstance(_meta_raw, dict) else {}" in src
    assert "metadata={**_rest_metadata" in src
