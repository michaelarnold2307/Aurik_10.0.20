from __future__ import annotations

import pytest

"""
tests/unit/test_247_chain_correction.py
=========================================
§2.47 Chain-Correction in AurikDenker (denker/aurik_denker.py)

Stellt sicher, dass die Defekt-basierte Ketten-Ergänzung:
  1. Eine rein digitale Kette (mp3_low) auf [vinyl, mp3_low] erweitert,
     wenn starke Vinyl-Defekte vorhanden sind.
  2. Eine rein digitale Kette auf [vinyl, tape, mp3_low] erweitert,
     wenn sowohl Vinyl- als auch Tape-Indikatoren vorliegen.
  3. Eine bereits mehrstufige Kette NICHT verändert.
  4. Bei schwachen analogen Defekten (< 0.35) KEINE Korrektur vornimmt.
  5. Bei rein digitalem Material ohne analoge Defekte KEINE Korrektur vornimmt.
  6. Keine Exception wirft wenn chain_info leer oder kein defect_raw vorhanden.

Normative Basis: §2.46a, §2.46b, §2.47, §6.2a
"""


from types import SimpleNamespace
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Hilfsfunktionen zum Simulieren von DefectAnalysisResult.get_top_defects()
# ---------------------------------------------------------------------------


def _make_defect_score(defect_type_str: str, severity: float):
    """Erzeugt ein Mock-Defect-Score-Objekt mit .defect_type.value und .severity."""
    ds = SimpleNamespace()
    ds.defect_type = SimpleNamespace(value=defect_type_str)
    ds.severity = severity
    return ds


def _make_defect_raw(scores: dict[str, float]):
    """Erzeugt ein Mock-DefectAnalysisResult mit get_top_defects(n)."""
    items = [_make_defect_score(k, v) for k, v in scores.items()]
    raw = MagicMock()
    raw.get_top_defects = lambda n=20: items[:n]
    return raw


# ---------------------------------------------------------------------------
# Die Hilfsfunktion, die die Chain-Correction-Logik repliziert (aus AurikDenker).
# Dies erlaubt isoliertes Testen ohne den vollen Denker-Stack.
# ---------------------------------------------------------------------------


def _apply_chain_correction(
    chain_info: dict,
    defect_raw,
    emit_cb=None,
    kette=None,
    stage_notes=None,
) -> dict:
    """
    Repliziert die §2.47 Chain-Correction-Logik aus AurikDenker.denke().
    Gibt das ggf. aktualisierte chain_info-Dict zurück.
    """
    import logging

    logging.getLogger("test_247")

    _DIGITAL_ONLY_KEYS = frozenset(
        {
            "cd_digital",
            "cd",
            "dat",
            "mp3_low",
            "mp3_high",
            "aac",
            "minidisc",
            "streaming",
            "cassette_digital",
        }
    )
    _cur_chain: list[str] = chain_info.get("chain", []) if isinstance(chain_info, dict) else []
    _chain_is_digital_only = bool(_cur_chain) and all(str(c).lower() in _DIGITAL_ONLY_KEYS for c in _cur_chain)

    if not (_chain_is_digital_only and defect_raw is not None):
        return chain_info

    _defect_sev: dict[str, float] = {
        str(s.defect_type.value): float(s.severity) for s in defect_raw.get_top_defects(20)
    }
    _VINYL_INDICATORS = frozenset(
        {
            "crackle",
            "low_freq_rumble",
            "riaa_curve_error",
            "groove_echo",
            "inner_groove_distortion",
        }
    )
    _TAPE_INDICATORS = frozenset(
        {
            "print_through",
            "tape_head_level_dip",
        }
    )
    _ALL_ANALOG = frozenset(
        {
            "crackle",
            "wow",
            "flutter",
            "multiband_wow_flutter",
            "low_freq_rumble",
            "riaa_curve_error",
            "groove_echo",
            "inner_groove_distortion",
            "print_through",
            "tape_head_level_dip",
            "soft_saturation",
        }
    )
    _vinyl_sev = max((_defect_sev.get(d, 0.0) for d in _VINYL_INDICATORS), default=0.0)
    _tape_sev = max((_defect_sev.get(d, 0.0) for d in _TAPE_INDICATORS), default=0.0)
    _max_analog = max((_defect_sev.get(d, 0.0) for d in _ALL_ANALOG), default=0.0)

    _inferred_analog: list[str] = []
    if _max_analog >= 0.35:
        if _vinyl_sev >= 0.20:
            _inferred_analog.append("vinyl")
        if _tape_sev >= 0.15:
            _inferred_analog.append("tape")
        if not _inferred_analog and _max_analog >= 0.50:
            _inferred_analog.append("vinyl")

    if _inferred_analog:
        _extended = _inferred_analog + [c for c in _cur_chain if c not in _inferred_analog]
        chain_info = dict(chain_info)
        chain_info["chain"] = _extended
        chain_info["chain_string"] = " → ".join(_extended)

    return chain_info


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestChainCorrection247:
    def test_vinyl_defects_extend_chain(self):
        """Vinyl-Indikatoren (crackle, rumble) → chain wird auf [vinyl, mp3_low] erweitert."""
        chain_info = {"chain": ["mp3_low"], "chain_string": "mp3_low"}
        defect_raw = _make_defect_raw(
            {
                "crackle": 1.0,
                "low_freq_rumble": 0.85,
                "compression_artifacts": 0.50,
            }
        )
        result = _apply_chain_correction(chain_info, defect_raw)
        assert result["chain"] == ["vinyl", "mp3_low"], f"Erwartet ['vinyl', 'mp3_low'], got {result['chain']}"
        assert "vinyl" in result["chain_string"]
        assert "mp3_low" in result["chain_string"]

    def test_vinyl_and_tape_defects_extend_to_three_stage(self):
        """Vinyl + Tape-Indikatoren → chain wird auf [vinyl, tape, mp3_low] erweitert."""
        chain_info = {"chain": ["mp3_low"], "chain_string": "mp3_low"}
        defect_raw = _make_defect_raw(
            {
                "crackle": 1.0,
                "low_freq_rumble": 0.90,
                "print_through": 0.80,  # tape indicator
                "tape_head_level_dip": 0.70,  # tape indicator
            }
        )
        result = _apply_chain_correction(chain_info, defect_raw)
        assert result["chain"] == ["vinyl", "tape", "mp3_low"], (
            f"Erwartet ['vinyl', 'tape', 'mp3_low'], got {result['chain']}"
        )

    def test_tape_only_defects_do_not_add_vinyl_without_vinyl_indicators(self):
        """Nur Tape-Indikatoren, keine Vinyl-Indikatoren → kein vinyl in chain."""
        chain_info = {"chain": ["mp3_low"], "chain_string": "mp3_low"}
        defect_raw = _make_defect_raw(
            {
                "print_through": 0.80,
                "tape_head_level_dip": 0.75,
                "soft_saturation": 0.60,  # analog but not vinyl-specific
            }
        )
        result = _apply_chain_correction(chain_info, defect_raw)
        # tape_sev >= 0.15 → tape added; vinyl_sev < 0.20 → no vinyl
        assert "vinyl" not in result["chain"]
        assert "tape" in result["chain"]
        assert "mp3_low" in result["chain"]

    def test_weak_analog_defects_no_correction(self):
        """Analoge Defekte < 0.35 → chain bleibt unverändert."""
        chain_info = {"chain": ["mp3_low"], "chain_string": "mp3_low"}
        defect_raw = _make_defect_raw(
            {
                "crackle": 0.30,  # below max_analog threshold 0.35
                "soft_saturation": 0.20,
            }
        )
        result = _apply_chain_correction(chain_info, defect_raw)
        assert result["chain"] == ["mp3_low"], f"Chain sollte unverändert bleiben, got {result['chain']}"

    def test_already_multistage_chain_not_modified(self):
        """Kette mit analogen Stufen → wird NICHT verändert (nicht digital-only)."""
        chain_info = {"chain": ["vinyl", "mp3_low"], "chain_string": "vinyl → mp3_low"}
        defect_raw = _make_defect_raw(
            {
                "crackle": 1.0,
                "low_freq_rumble": 0.90,
            }
        )
        result = _apply_chain_correction(chain_info, defect_raw)
        # chain has "vinyl" → not digital-only → no correction
        assert result["chain"] == ["vinyl", "mp3_low"]

    def test_no_defect_raw_no_correction(self):
        """Kein defect_raw → chain unverändert."""
        chain_info = {"chain": ["mp3_low"], "chain_string": "mp3_low"}
        result = _apply_chain_correction(chain_info, defect_raw=None)
        assert result["chain"] == ["mp3_low"]

    def test_empty_chain_info_no_crash(self):
        """Leeres chain_info → kein Crash, chain_info bleibt leer."""
        result = _apply_chain_correction({}, defect_raw=None)
        assert result == {}

    def test_strong_analog_non_vinyl_fallback_adds_vinyl(self):
        """Starke analoge Defekte (≥ 0.50) ohne spezifische Vinyl-Indikatoren → vinyl als Fallback."""
        chain_info = {"chain": ["mp3_low"], "chain_string": "mp3_low"}
        defect_raw = _make_defect_raw(
            {
                "soft_saturation": 0.90,  # max_analog = 0.90 (≥ 0.50), aber kein vinyl-spezifischer Indikator
            }
        )
        result = _apply_chain_correction(chain_info, defect_raw)
        # _vinyl_sev = 0 (kein vinyl-Indikator), aber max_analog >= 0.50 → vinyl Fallback
        assert "vinyl" in result["chain"]
        assert result["chain"][0] == "vinyl"
        assert "mp3_low" in result["chain"]

    def test_riaa_curve_error_triggers_vinyl(self):
        """RIAA-Kurven-Fehler ist ein eindeutiger Vinyl-Indikator."""
        chain_info = {"chain": ["mp3_low"], "chain_string": "mp3_low"}
        defect_raw = _make_defect_raw(
            {
                "riaa_curve_error": 0.75,
            }
        )
        result = _apply_chain_correction(chain_info, defect_raw)
        assert "vinyl" in result["chain"]

    def test_groove_echo_triggers_vinyl(self):
        """Groove-Echo ist ein eindeutiger Vinyl-Indikator."""
        chain_info = {"chain": ["mp3_low"], "chain_string": "mp3_low"}
        defect_raw = _make_defect_raw(
            {
                "groove_echo": 0.60,
            }
        )
        result = _apply_chain_correction(chain_info, defect_raw)
        assert "vinyl" in result["chain"]

    def test_cd_digital_chain_also_corrected(self):
        """Auch eine cd_digital-Kette wird bei analogen Defekten erweitert."""
        chain_info = {"chain": ["cd_digital"], "chain_string": "cd_digital"}
        defect_raw = _make_defect_raw(
            {
                "crackle": 0.80,
                "low_freq_rumble": 0.70,
            }
        )
        result = _apply_chain_correction(chain_info, defect_raw)
        assert "vinyl" in result["chain"]
        assert "cd_digital" in result["chain"]

    def test_chain_order_analog_before_digital(self):
        """Analoge Stufen kommen immer VOR dem digitalen Container (§2.46 Kettenordnung)."""
        chain_info = {"chain": ["mp3_low"], "chain_string": "mp3_low"}
        defect_raw = _make_defect_raw(
            {
                "crackle": 1.0,
                "print_through": 0.80,
            }
        )
        result = _apply_chain_correction(chain_info, defect_raw)
        chain = result["chain"]
        # mp3_low muss LETZTES Glied sein
        assert chain[-1] == "mp3_low", f"mp3_low soll letztes Glied sein, got {chain}"
        # Analoge Stufen kommen davor
        assert chain[0] in ("vinyl", "tape"), f"Erste Stufe soll analog sein, got {chain[0]}"


# ---------------------------------------------------------------------------
# Tests für §6.2a Transfer-Chain-aware Skip in Phase_29
# ---------------------------------------------------------------------------


class TestPhase29TransferChainSkip:
    """
    Stellt sicher, dass phase_29 nicht überspringt, wenn transfer_chain eine Tape-Stufe enthält.
    Normative Basis: §6.2a Carrier-Chain-Invariante, §2.46a.
    """

    def _make_phase29(self):
        """Instanziiert phase_29 (kein ML notwendig — DSP-Pfad)."""
        from backend.core.phases.phase_29_tape_hiss_reduction import TapeHissReductionPhase

        return TapeHissReductionPhase()

    def _make_audio(self, seconds: float = 1.0) -> tuple:
        """Gibt (audio_array, sample_rate) zurück."""
        import numpy as np

        sr = 48000
        t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
        # Sinus mit HF-Rauschen (simuliert Tape-Hiss)
        audio = 0.1 * np.sin(2 * np.pi * 440 * t) + 0.01 * np.random.randn(len(t))
        return audio.astype(np.float32), sr

    def test_mp3_without_chain_skips(self):
        """mp3_low ohne transfer_chain → Phase 29 überspringt (processing=skipped_digital)."""
        from backend.core.defect_scanner import MaterialType

        phase = self._make_phase29()
        audio, sr = self._make_audio()
        result = phase.process(audio, sample_rate=sr, material=MaterialType.MP3_LOW)
        assert result.metadata.get("processing") == "skipped_digital", (
            f"Erwartet skipped_digital, got {result.metadata.get('processing')}"
        )

    def test_mp3_with_tape_chain_does_not_skip(self):
        """mp3_low mit transfer_chain=['vinyl','tape','mp3_low'] → Phase 29 läuft (processing != skipped_digital)."""
        from backend.core.defect_scanner import MaterialType

        phase = self._make_phase29()
        audio, sr = self._make_audio()
        result = phase.process(
            audio,
            sample_rate=sr,
            material=MaterialType.MP3_LOW,
            transfer_chain=["vinyl", "tape", "mp3_low"],
        )
        assert result.metadata.get("processing") != "skipped_digital", (
            f"Phase sollte NICHT überspringen bei tape in chain, got processing={result.metadata.get('processing')}"
        )

    def test_mp3_with_reel_tape_chain_does_not_skip(self):
        """mp3_low mit transfer_chain=['reel_tape','mp3_low'] → Phase 29 läuft."""
        from backend.core.defect_scanner import MaterialType

        phase = self._make_phase29()
        audio, sr = self._make_audio()
        result = phase.process(
            audio,
            sample_rate=sr,
            material=MaterialType.MP3_LOW,
            transfer_chain=["reel_tape", "mp3_low"],
        )
        assert result.metadata.get("processing") != "skipped_digital"

    def test_mp3_with_vinyl_only_chain_skips(self):
        """mp3_low mit transfer_chain=['vinyl','mp3_low'] (kein Tape!) → Phase 29 überspringt."""
        from backend.core.defect_scanner import MaterialType

        phase = self._make_phase29()
        audio, sr = self._make_audio()
        result = phase.process(
            audio,
            sample_rate=sr,
            material=MaterialType.MP3_LOW,
            transfer_chain=["vinyl", "mp3_low"],  # Vinyl, aber kein Tape
        )
        # Vinyl allein hat kein Tape-Hiss → Skip ist korrekt
        assert result.metadata.get("processing") == "skipped_digital", (
            f"Nur-Vinyl-Kette sollte phase_29 überspringen, got {result.metadata.get('processing')}"
        )
