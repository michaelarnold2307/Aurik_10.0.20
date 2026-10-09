"""D-K3-69 — Dauergewichteter Phasenfortschritt: Vertrags- und Wiring-Tests.

Befund (Tiefenanalyse 2026-10-09): `pct` war eine Zähl-Fraktion
(`_idx_ex / _n_sel`, Band 20–85) — gemessene Phasen-Imbalance ca. 11:1 ließ
den Balken bei Kurzphasen sprinten und bei Langphasen einfrieren.

Fix: Selbstkalibrierung über die GEMESSENEN Wand-Zeiten des Laufs
(`backend/core/phase_progress_weights.py`); keine statischen Fantasiegewichte
(§V7 copilot-instructions.md). Diese Tests pinnen die Mathematik (1-s-vs-100-s)
und die Verdrahtung in `_execute_pipeline` (§G9 copilot-instructions.md:
eine Quelle je Rechnung).
"""

from __future__ import annotations

from pathlib import Path

from backend.core.phase_progress_weights import weighted_phase_span, weighted_progress_fraction

RESTORER = Path(__file__).resolve().parents[2] / "backend" / "core" / "unified_restorer_v3.py"


def test_dauerbasis_statt_zaehlbasis() -> None:
    """Nach 1-s-Phase bei 100-s-Restphase: ≈1 % — Zählbasis hätte 50 % gezeigt."""
    d = {"fast": 1.0, "slow": 100.0}
    frac = weighted_progress_fraction(["fast"], d, 2)
    assert 0.005 <= frac <= 0.02, f"daueranteilig erwartet, war {frac}"
    assert frac < 0.5, "Zählbasis-Verhalten (0,5) darf nicht zurückkehren"


def test_vollstaendig_ist_eins() -> None:
    assert weighted_progress_fraction(["a", "b"], {"a": 5.0, "b": 5.0}, 2) == 1.0


def test_leer_und_dedupe() -> None:
    assert weighted_progress_fraction([], {}, 3) == 0.0
    assert weighted_progress_fraction(["a", "a"], {"a": 2.0}, 1) == 1.0, "Doppel-IDs dürfen nicht doppelt zählen"


def test_prior_ist_median_der_gemessenen() -> None:
    frac = weighted_progress_fraction(["slow"], {"slow": 100.0}, 3)
    assert abs(frac - (100.0 / 300.0)) < 1e-9


def test_spanne_grenzen_stimmen_mit_fraktion_ueberein() -> None:
    d = {"a": 10.0, "b": 30.0, "c": 30.0}
    s, e = weighted_phase_span(["a"], d, "b", 3)
    assert s == weighted_progress_fraction(["a"], d, 3)
    assert e == weighted_progress_fraction(["a", "b"], d, 3)


def test_spanne_klampft_auf_null_eins() -> None:
    s, e = weighted_phase_span(["x", "y"], {}, "z", 2)
    assert 0.0 <= s <= 1.0 and 0.0 <= e <= 1.0 and s <= e


def test_verdrahtung_in_execute_pipeline() -> None:
    """Wiring-Pins: 4 Abschluss-Emissionen + 2 Heartbeat-Spannen nutzen das Modul."""
    src = RESTORER.read_text(encoding="utf-8")
    assert src.count("_phase_pct_weighted([*executed, phase_id], _n_sel)") == 4
    assert src.count("_record_phase_duration(phase_id, result)") == 4
    assert src.count("_wps(executed, _phase_dur_acc, phase_id,") == 2
    assert "from backend.core.phase_progress_weights import" in src
    # Altpfad entfernt (Zähl-Fraktion über abgeschlossene Phasen):
    assert "_idx_ex / _n_sel" not in src
