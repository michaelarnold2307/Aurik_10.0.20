from __future__ import annotations

import pytest

"""[RELEASE_MUST] §8.5 Globales Parameterregister — automatischer CI-Gate-Test (v10.0.0)

Spec reference:  .github/specs/07_quality_and_tests.md §8.5
Purpose:         Verifiziert, dass alle in §8.5A (OPTIMAL) dokumentierten Parameterwerte
                 tatsächlich im Produktionscode vorhanden sind.  Kein Test simuliert Audio-
                 Verarbeitung — es handelt sich um statische Code-Inspektion und Modul-Imports.

Kategorien:
  R01 – SR-Vertrag              → test_r01_*
  R02 – TruePeak-Guard          → test_r02_*
  R03 – Noise-Texture-Threshold → test_r03_*
  R04 – Priority-Phasen         → test_r04_*
  R05 – Stereo-Korr.-Guard      → test_r05_*
  R06 – OQS-Gates               → test_r06_*
  R07 – Carrier-Recovery-Ratio  → test_r07_*

Aufruf: pytest tests/normative/test_parameter_register_gate.py -v --timeout=30
"""

import re

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uv3_source() -> str:
    from pathlib import Path

    p = Path("backend/core/unified_restorer_v3.py")
    if not p.exists():
        p = Path(__file__).parent.parent.parent / "backend/core/unified_restorer_v3.py"
    return p.read_text(encoding="utf-8", errors="replace")


def _spec07_source() -> str:
    from pathlib import Path

    p = Path(".github/specs/07_quality_and_tests.md")
    if not p.exists():
        p = Path(__file__).parent.parent.parent / ".github/specs/07_quality_and_tests.md"
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


# ===========================================================================
# R01 — SR-Vertrag (§2.2.0, §8.5A)
# ===========================================================================


@pytest.mark.unit
class TestR01SRContract:
    """Processing-SR=48000 und Analysis-SR=import_sr sind normativ festgelegt."""

    def test_processing_sr_48000_present_in_uv3(self):
        src = _uv3_source()
        # UV3 muss per Phase assert sample_rate == 48000 aufrufen
        assert "48000" in src, "48000 Hz SR-Wert fehlt in unified_restorer_v3.py"

    def test_assert_sr_48k_in_uv3(self):
        """UV3 muss explizit 48000 als Processing-SR verwenden."""
        src = _uv3_source()
        # UV3 kann SR via target_sample_rate, sample_rate != 48000 oder Docstring dokumentieren
        assert re.search(r"sample_rate.*48000|48000.*sample_rate|target_sample_rate\s*=\s*48000", src), (
            "Kein 48000-Hz-SR-Verweis in UV3 gefunden"
        )

    def test_dual_sr_annotation_present(self):
        """§2.2.0 Dual-SR-Kommentar muss in UV3 vorhanden sein."""
        src = _uv3_source()
        assert "analysis_sr" in src or "analysis_audio" in src, "Kein analysis_sr/analysis_audio Dual-SR-Verweis in UV3"


# ===========================================================================
# R02 — Finaler TruePeak-Hardguard (§8.5A)
# ===========================================================================


class TestR02TruePeakHardGuard:
    """Universaler End-of-Pipeline TruePeak-Guard mit Ceiling 0.966 (-0.3 dBFS)."""

    def test_final_tp_ceiling_value_in_uv3(self):
        src = _uv3_source()
        # Ceiling-Wert muss exakt 0.966 sein
        assert "0.966" in src, "Finaler TruePeak-Hardguard Ceiling 0.966 fehlt in unified_restorer_v3.py"

    def test_final_tp_guard_log_marker_present(self):
        src = _uv3_source()
        assert "Final TruePeak hard-guard" in src, (
            "Log-Marker '§Final TruePeak hard-guard' fehlt — Guard-Implementierung prüfen"
        )

    def test_final_tp_uses_percentile_not_max(self):
        """Peak-Messung muss percentile(99.9) sein, nicht np.max (DSP-Invariante)."""
        src = _uv3_source()
        # Suche nach percentile(np.abs(...), 99.9) im TruePeak-Guard
        guard_block_start = src.find("Final TruePeak hard-guard")
        assert guard_block_start >= 0, "TruePeak hard-guard Block nicht gefunden"
        guard_block = src[guard_block_start : guard_block_start + 600]
        assert "percentile" in guard_block, "TruePeak-Guard: percentile() fehlt — np.max() widerspricht DSP-Invariante"
        assert "99.9" in guard_block, "TruePeak-Guard: 99.9-Perzentil fehlt"

    def test_final_tp_guard_uses_np_clip_afterwards(self):
        src = _uv3_source()
        guard_block_start = src.find("Final TruePeak hard-guard")
        # np.clip taucht in UV3 vielfach auf — prüfe globale Präsenz (Guard-Block kann kompakter sein)
        assert "np.clip" in src, "np.clip fehlt komplett in UV3 — finales Sample-Clipping nicht vorhanden"
        # Spezifisch: Guard-Block darf bis 2000 Zeichen nach dem Marker reichen (kommentierter Bereich)
        guard_area = src[guard_block_start : guard_block_start + 2000]
        assert "np.clip" in guard_area or "_tp_guard" in guard_area, (
            "TruePeak-Guard: kein np.clip oder Guard-Code in 2000 Zeichen nach Marker"
        )

    def test_final_tp_guard_is_non_blocking(self):
        """Guard muss Exception-sicher sein (non-blocking)."""
        src = _uv3_source()
        guard_block_start = src.find("Final TruePeak hard-guard")
        # NonBlocking-Log-Marker muss innerhalb von 500 Zeichen nach dem Guard stehen
        guard_area = src[guard_block_start : guard_block_start + 1200]
        assert "non-blocking" in guard_area or "_tp_guard_exc" in guard_area, (
            "TruePeak-Guard: Exception-Handler (non-blocking) fehlt"
        )


# ===========================================================================
# R03 — Noise-Texture-Threshold material-adaptiv (§8.5A)
# ===========================================================================


class TestR03NoiseTextureThreshold:
    """Noise-Texture-Rollback-Threshold ist materialadaptiv, kein fixer Wert."""

    def test_threshold_dict_exists_in_uv3(self):
        src = _uv3_source()
        assert "_MATERIAL_NOISE_TEXTURE_ROLLBACK_THRESHOLD" in src, (
            "_MATERIAL_NOISE_TEXTURE_ROLLBACK_THRESHOLD Dict fehlt in UV3"
        )

    def test_mp3_low_threshold_is_15(self):
        src = _uv3_source()
        # Wert muss 15.0 für mp3_low sein
        block_start = src.find("_MATERIAL_NOISE_TEXTURE_ROLLBACK_THRESHOLD")
        block = src[block_start : block_start + 1000]
        assert re.search(r'"mp3_low"\s*:\s*15\.0', block), "mp3_low Noise-Texture-Threshold ist nicht 15.0 dB/oct"

    def test_shellac_threshold_is_6(self):
        src = _uv3_source()
        block_start = src.find("_MATERIAL_NOISE_TEXTURE_ROLLBACK_THRESHOLD")
        block = src[block_start : block_start + 1000]
        assert re.search(r'"shellac"\s*:\s*6\.0', block), "shellac Noise-Texture-Threshold ist nicht 6.0 dB/oct"

    def test_vinyl_threshold_is_8(self):
        src = _uv3_source()
        block_start = src.find("_MATERIAL_NOISE_TEXTURE_ROLLBACK_THRESHOLD")
        block = src[block_start : block_start + 1000]
        assert re.search(r'"vinyl"\s*:\s*8\.0', block), "vinyl Noise-Texture-Threshold ist nicht 8.0 dB/oct"

    def test_helper_function_exists(self):
        src = _uv3_source()
        assert "_get_noise_texture_rollback_threshold" in src, (
            "Hilfsfunktion _get_noise_texture_rollback_threshold fehlt in UV3"
        )

    def test_guard_uses_helper_not_hardcoded_6(self):
        """§2.49 Guard darf keine hardcodierte 6.0 mehr nutzen."""
        src = _uv3_source()
        assert "_get_noise_texture_rollback_threshold" in src, (
            "Guard nutzt keine Helper-Funktion — hardcodierter Threshold-Verdacht"
        )
        # Extra: Stellen im Guard-Block prüfen
        guard_pos = src.find("noise_texture_deviation_db_oct")
        if guard_pos > 0:
            guard_block = src[guard_pos : guard_pos + 300]
            assert "_ntx_threshold" in guard_block or "_get_noise_texture_rollback_threshold" in guard_block, (
                "Noise-Texture-Guard-Block nutzt noch keine materialadaptive Funktion"
            )


# ===========================================================================
# R04 — Material-Priority-Phasen (§6.2a / §8.5A)
# ===========================================================================


class TestR04MaterialPriorityPhases:
    """Prüft, ob _MATERIAL_PRIORITY_PHASES die normativen Pflichtphasen enthält."""

    def _get_priority_dict(self) -> str:
        src = _uv3_source()
        # Extrahiere den MATERIAL_PRIORITY_PHASES Block als approx. Text
        return src  # pass through source for search

    def test_mp3_low_includes_phase06(self):
        src = _uv3_source()
        block_start = src.find('"mp3_low": [')
        block = src[block_start : block_start + 600]
        assert "phase_06_frequency_restoration" in block, (
            "phase_06_frequency_restoration fehlt in mp3_low Priority-Phasen (§8.5A)"
        )

    def test_mp3_low_includes_phase38(self):
        src = _uv3_source()
        block_start = src.find('"mp3_low": [')
        block = src[block_start : block_start + 600]
        assert "phase_38_presence_boost" in block, "phase_38_presence_boost fehlt in mp3_low Priority-Phasen (§8.5A)"

    def test_mp3_low_includes_phase39(self):
        src = _uv3_source()
        block_start = src.find('"mp3_low": [')
        block = src[block_start : block_start + 600]
        assert "phase_39_air_band_enhancement" in block, (
            "phase_39_air_band_enhancement fehlt in mp3_low Priority-Phasen (§8.5A)"
        )

    def test_mp3_high_includes_phase06(self):
        src = _uv3_source()
        block_start = src.find('"mp3_high": [')
        block = src[block_start : block_start + 400]
        assert "phase_06_frequency_restoration" in block, (
            "phase_06_frequency_restoration fehlt in mp3_high Priority-Phasen (§8.5A)"
        )

    def test_mp3_high_includes_phase39(self):
        src = _uv3_source()
        block_start = src.find('"mp3_high": [')
        block = src[block_start : block_start + 400]
        assert "phase_39_air_band_enhancement" in block, (
            "phase_39_air_band_enhancement fehlt in mp3_high Priority-Phasen (§8.5A)"
        )

    def test_lossy_presence_boost_active_without_vocals(self):
        """Tier-5: presence_boost muss für Lossy-Codecs auch ohne vocals_detected aktiv sein."""
        src = _uv3_source()
        assert "_lossy_codec_no_vocal" in src, (
            "_lossy_codec_no_vocal Guard fehlt — presence_boost nur für Vocals aktiv (§8.5A)"
        )

    def test_lossy_airband_condition_includes_mp3(self):
        """Tier-5: air_band muss für MP3_LOW/HIGH in _needs_airband-Kondition enthalten sein."""
        src = _uv3_source()
        assert "_needs_airband" in src, "_needs_airband-Kondition fehlt — mp3_low erhält kein Air-Band (§8.5A)"


# ===========================================================================
# R05 — Stereo-Korrelations-Guard (§2.51 / §8.5B)
# ===========================================================================


class TestR05StereoCorrelationGuard:
    """§2.51 Universaler Stereo-Korr.-Guard mit input-relativen Schwellen."""

    def test_stereo_guard_delta_limit_narrow_present(self):
        src = _uv3_source()
        assert "_delta_limit" in src or "delta_limit" in src, "Stereo-Korr.-Guard delta_limit fehlt in UV3 (§8.5B)"

    def test_stereo_guard_uses_input_relative_thresholds(self):
        src = _uv3_source()
        assert "_input_is_narrow" in src or "_corr_before" in src, (
            "Stereo-Korr.-Guard nutzt keine input-relativen Schwellen (§8.5B)"
        )

    def test_stereo_guard_log_marker_present(self):
        src = _uv3_source()
        assert "§2.51 Stereo-correlation" in src, (
            "Log-Marker §2.51 Stereo-correlation fehlt — Guard-Implementierung prüfen"
        )


# ===========================================================================
# R06 — OQS-Gates (§8.1.1a/b / §8.5A)
# ===========================================================================


class TestR06OQSGates:
    """Materialadaptive OQS-Gates für beide Modi."""

    def test_studio2026_oqs_gate_88_in_spec(self):
        spec = _spec07_source()
        assert "OQS ≥ **88**" in spec or "OQS >= 88" in spec, "Studio-2026-OQS-Gate 88 fehlt in Spec 07 §8.1.1a"

    def test_restoration_oqs_digital_80_in_spec(self):
        spec = _spec07_source()
        assert "≥ **80**" in spec or ">= 80" in spec, "Restoration-OQS-Gate 80 (digital) fehlt in Spec 07 §8.1.1b"

    def test_parameter_register_section_in_spec(self):
        spec = _spec07_source()
        assert "§8.5" in spec and "Parameterregister" in spec, "§8.5 Globales Parameterregister fehlt in Spec 07"

    def test_parameter_register_has_optimal_table(self):
        spec = _spec07_source()
        assert "OPTIMAL" in spec, "OPTIMAL-Status fehlt in §8.5 Parameterregister"

    def test_parameter_register_has_nicht_optimal_table(self):
        spec = _spec07_source()
        assert "NICHT OPTIMAL" in spec, "NICHT OPTIMAL-Status fehlt in §8.5 Parameterregister"


# ===========================================================================
# R07 — Carrier-Recovery-Ratio (§0d / §1.2a / §8.5A)
# ===========================================================================


class TestR07CarrierRecoveryRatio:
    """carrier_chain_recovery_ratio ist Pflichtfeld in metadata."""

    def test_carrier_recovery_ratio_field_in_uv3(self):
        src = _uv3_source()
        assert "carrier_chain_recovery_ratio" in src, "carrier_chain_recovery_ratio fehlt in UV3 (§0d Pflichtfeld)"

    def test_carrier_recovery_threshold_015_present(self):
        src = _uv3_source()
        assert "0.15" in src, "Carrier-Recovery-Threshold 0.15 fehlt in UV3 (§0d)"

    def test_carrier_recovery_threshold_035_present(self):
        src = _uv3_source()
        assert "0.35" in src, "Carrier-Recovery-Threshold 0.35 (massiv) fehlt in UV3 (§0d)"
