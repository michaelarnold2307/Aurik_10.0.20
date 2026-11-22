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
    """Der Spektral-Abstand (Wohlklang-Proxy) ist Teil des Holdout-Scores UND des Trainings-Loss."""
    src = _text()
    assert "def _spektral_abstand" in src
    assert "lam_spec * spec_h" in src
    assert "lam_spec * spec_train" in src  # Loss-Term (Sweep-Befund: Auswahl allein reicht nicht)
    assert "lambda_spektral" in src


def test_smoke_schreibt_separaten_pfad() -> None:
    """--max-regionen > 0 darf den kanonischen Checkpoint nicht ueberschreiben."""
    src = _text()
    assert "_ist_smoke" in src
    assert "smoke_witness_" in src
    assert 'f"witness_{klasse}_best.pt"' in src  # kanonischer Pfad bleibt fuer echte Laeufe


def test_q_ziel_default_aus_und_pfad_klar() -> None:
    """--q-ziel ist Default-AUS: ohne Flag bleibt der Score-Pfad unveraendert (§G5 (GEBOTE.md))."""
    src = _text()
    assert "q_ziel: float | None = None" in src
    assert "and q_ziel is None" in src
    assert "def _q_nahe_auswahl" in src


def test_q_nahe_auswahl_anti_ueberrestauration() -> None:
    """Q-nahe Auswahl: Epoche mit q nahe 1,0 gewinnt; MOS-JND-Floor bleibt aktiv."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts import train_witness_kopf as twk

    kandidaten = [
        {"ep": 1, "q": 2.0, "mos": 4.9, "score": 0.5, "state": {"a": 1}},
        {"ep": 2, "q": 1.05, "mos": 4.85, "score": 0.6, "state": {"a": 2}},
        {"ep": 3, "q": 0.4, "mos": 4.4, "score": 0.4, "state": {"a": 3}},
    ]
    gew = twk._q_nahe_auswahl(kandidaten, 1.0)
    assert gew is not None and gew["ep"] == 2
    # Epoche 3 faellt aus dem MOS-Floor (4,9 − 0,05), Epoche 1 ist Ueberrestauration (q 2,0)
    assert twk._q_nahe_auswahl([{"ep": 1, "q": None, "mos": 4.0, "score": 0.5}], 1.0) is None
