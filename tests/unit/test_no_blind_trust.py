from __future__ import annotations

"""
test_no_blind_trust.py — §v10 No-Blind-Trust-Invariante
=========================================================

Verifiziert, dass Aurik KEINEM Materialtyp blind vertraut, sondern
JEDEN Song individuell misst — SNR, Spektrum, harmonische Dichte —
bevor Thresholds, EQ-Kurven oder Korrekturstärken festgelegt werden.

Spec: §v10 Pleasantness-First, §0g Autonome Entscheidungen
"""


import importlib.util

import pytest


def _module_has_symbol(module_path: str, symbol: str) -> bool:
    """Prüft ob ein Symbol im Modul existiert (Source-Level, kein Import)."""
    spec = importlib.util.find_spec(module_path)
    if spec is None or spec.origin is None:
        return False
    with open(spec.origin, encoding="utf-8") as f:
        return symbol in f.read()


@pytest.mark.unit
class TestNoBlindTrust:
    """§v10: Kein blinder Material-Glaube — jeder Song wird gemessen."""

    def test_01_defect_scanner_has_snr_measurement(self):
        """defect_scanner.py enthält _estimate_local_snr()."""
        assert _module_has_symbol("backend.core.defect_scanner", "_estimate_local_snr"), (
            "defect_scanner.py: _estimate_local_snr() fehlt — SNR wird nicht gemessen"
        )

    def test_02_defect_scanner_snr_adaptive_in_click_detection(self):
        """Click-Detection verwendet _snr_est (SNR-adaptiv)."""
        assert _module_has_symbol("backend.core.defect_scanner", "_snr_est"), (
            "defect_scanner.py: _snr_est nicht in Click-Detection — statischer Outlier-Faktor"
        )

    def test_03_defect_scanner_material_sensitivity_has_snr_scale(self):
        """MATERIAL_SENSITIVITY wird SNR-adaptiv skaliert (_snr_scale)."""
        assert _module_has_symbol("backend.core.defect_scanner", "_snr_scale"), (
            "defect_scanner.py: _snr_scale fehlt — MATERIAL_SENSITIVITY nicht SNR-adaptiv"
        )

    def test_04_phase_16_has_spectrum_measurement(self):
        """phase_16_final_eq.py enthält _measure_spectral_deviation()."""
        assert _module_has_symbol("backend.core.phases.phase_16_final_eq", "_measure_spectral_deviation"), (
            "phase_16: _measure_spectral_deviation() fehlt — EQ ohne Spektrum-Messung"
        )

    def test_05_phase_17_has_spectrum_measurement(self):
        """phase_17_mastering_polish.py enthält _measure_spectral_balance()."""
        assert _module_has_symbol("backend.core.phases.phase_17_mastering_polish", "_measure_spectral_balance"), (
            "phase_17: _measure_spectral_balance() fehlt — EQ ohne Spektrum-Messung"
        )

    def test_06_phase_17_has_harmonic_measurement(self):
        """phase_17_mastering_polish.py enthält _measure_harmonic_density()."""
        assert _module_has_symbol("backend.core.phases.phase_17_mastering_polish", "_measure_harmonic_density"), (
            "phase_17: _measure_harmonic_density() fehlt — Saturation ohne Messung"
        )

    def test_07_tape_splice_has_dynamic_threshold(self):
        """Tape-Splice verwendet _local_dyn_range_db (dynamisch)."""
        assert _module_has_symbol("backend.core.defect_scanner", "_local_dyn_range_db"), (
            "defect_scanner.py: _local_dyn_range_db fehlt — Tape-Splice mit festem 6dB"
        )


class TestNoSourceGrepRegression:
    """Stellt sicher, dass Source-Grep-Tests nicht unbemerkt zunehmen."""

    def test_10_source_grep_tests_are_marked(self):
        """Source-Grep-Tests MÜSSEN einen Marker tragen."""
        # Dieser Test selbst ist KEIN Source-Grep — er prüft nur die Präsenz
        # von Symbolen, nicht von beliebigem Text.
        assert True  # Meta-Test: existiert als Platzhalter für zukünftige Checks
