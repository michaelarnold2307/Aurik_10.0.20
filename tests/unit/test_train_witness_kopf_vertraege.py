"""Vertragstests fuer den Zwei-Metrik-Fix + Smoke-Schutz in scripts/train_witness_kopf.py (2026-11-16).

Pinnt zwei Befunde:
1. Zwei-Metrik-Modellauswahl (Ablations-/Daempfungs-Befund 2026-11-16): der
   Holdout-Score MUSS den Spektral-Abstand zum Clean (Wohlklang-Proxy) enthalten —
   die alte Auswahl mass nur L1 + Witness-BCE und hat den Wohlklang nie gemessen.
2. Smoke-Schutz: --max-regionen > 0 MUSS einen separaten smoke_witness_*-Pfad
   schreiben — ein 1-Epochen-Smoke hat 2026-11-16 den kanonischen Checkpoint
   witness_groove_echo_best.pt (7,68 MB) mit 42 KB ueberschrieben.
"""

from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "scripts" / "train_witness_kopf.py"


def _text() -> str:
    return _SRC.read_text(encoding="utf-8")


def test_zwei_metrik_term_im_holdout_score() -> None:
    """Der Spektral-Abstand (Wohlklang-Proxy) ist Teil des Holdout-Scores."""
    src = _text()
    assert "def _spektral_abstand" in src
    assert "lam_spec * spec_h" in src
    assert "lambda_spektral" in src


def test_smoke_schreibt_separaten_pfad() -> None:
    """--max-regionen > 0 darf den kanonischen Checkpoint nicht ueberschreiben."""
    src = _text()
    assert "_ist_smoke" in src
    assert "smoke_witness_" in src
    assert 'f"witness_{klasse}_best.pt"' in src  # kanonischer Pfad bleibt fuer echte Laeufe
