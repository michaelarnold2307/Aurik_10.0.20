from __future__ import annotations

"""tests/unit/test_material_key_normalization.py
=================================================
Unit-Tests für §6.1 [RELEASE_MUST] Material-Key-Normalisierung.

Prüft, dass MediumDetector.detect() ausschließlich SUPPORTED_MATERIALS-konforme
Keys zurückgibt (reel_wire → wire_recording, cassette_digital → dat, etc.).

Spec-Referenz:
  - Spec 05 §6.1: SUPPORTED_MATERIALS + Key-Mapping-Invariante
  - copilot-instructions.md: Fix X6 Material-Mapping
"""


import numpy as np
import pytest

# ---------------------------------------------------------------------------
# SUPPORTED_MATERIALS aus Spec 05 §6.1 (kanonische Quelle)
# ---------------------------------------------------------------------------

SUPPORTED_MATERIALS = frozenset(
    [
        "tape",
        "reel_tape",
        "cassette",
        "vinyl",
        "shellac",
        "wax_cylinder",
        "wire_recording",
        "lacquer_disc",
        "dat",
        "cd_digital",
        "mp3_low",
        "mp3_high",
        "aac",
        "minidisc",
        "streaming",
        "unknown",
    ]
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def detector():
    """Import MediumDetector singleton — skip if forensics not available."""
    pytest.importorskip("forensics.medium_detector")
    from forensics.medium_detector import get_medium_detector

    return get_medium_detector()


def _make_audio(duration_s: float = 2.0, sr: int = 48000) -> np.ndarray:
    """Simple white-noise mono clip for detector input."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(int(duration_s * sr)).astype(np.float32) * 0.05


# ---------------------------------------------------------------------------
# Tests: _normalize_material_key staticmethod
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNormalizeMaterialKey:
    """Unit-Tests für MediumDetector._normalize_material_key (§6.1 Fix X6)."""

    def test_cassette_passthrough(self, detector):
        assert detector._normalize_material_key("cassette") == "cassette"

    def test_reel_wire_maps_to_wire_recording(self, detector):
        assert detector._normalize_material_key("reel_wire") == "wire_recording"

    def test_cassette_digital_maps_to_dat(self, detector):
        assert detector._normalize_material_key("cassette_digital") == "dat"

    def test_vhs_audio_maps_to_tape(self, detector):
        assert detector._normalize_material_key("vhs_audio") == "tape"

    def test_passthrough_tape(self, detector):
        """Keys already in SUPPORTED_MATERIALS pass through unchanged."""
        assert detector._normalize_material_key("tape") == "tape"

    def test_passthrough_vinyl(self, detector):
        assert detector._normalize_material_key("vinyl") == "vinyl"

    def test_passthrough_shellac(self, detector):
        assert detector._normalize_material_key("shellac") == "shellac"

    def test_passthrough_wire_recording(self, detector):
        assert detector._normalize_material_key("wire_recording") == "wire_recording"

    def test_passthrough_dat(self, detector):
        assert detector._normalize_material_key("dat") == "dat"

    def test_passthrough_cd_digital(self, detector):
        assert detector._normalize_material_key("cd_digital") == "cd_digital"

    def test_passthrough_mp3_low(self, detector):
        assert detector._normalize_material_key("mp3_low") == "mp3_low"

    def test_passthrough_mp3_high(self, detector):
        assert detector._normalize_material_key("mp3_high") == "mp3_high"

    def test_passthrough_unknown(self, detector):
        assert detector._normalize_material_key("unknown") == "unknown"

    def test_passthrough_reel_tape(self, detector):
        assert detector._normalize_material_key("reel_tape") == "reel_tape"

    def test_unknown_key_passes_through_unchanged(self, detector):
        """Unbekannte Keys (z. B. future keys) werden nicht verändert."""
        result = detector._normalize_material_key("some_unknown_future_key")
        assert result == "some_unknown_future_key"


# ---------------------------------------------------------------------------
# Tests: detect() Rückgabe-Invariante — primary_material ∈ SUPPORTED_MATERIALS
# ---------------------------------------------------------------------------


class TestDetectSupportedMaterialsInvariant:
    """Stellt sicher, dass detect() nur SUPPORTED_MATERIALS-Keys zurückgibt."""

    def test_primary_material_in_supported(self, detector):
        """primary_material muss immer in SUPPORTED_MATERIALS sein."""
        audio = _make_audio()
        result = detector.detect(audio, 48000)
        assert result.primary_material in SUPPORTED_MATERIALS, (
            f"primary_material='{result.primary_material}' ist nicht in SUPPORTED_MATERIALS"
        )

    def test_transfer_chain_keys_in_supported(self, detector):
        """Jedes Element der transfer_chain muss in SUPPORTED_MATERIALS sein."""
        audio = _make_audio()
        result = detector.detect(audio, 48000)
        for key in result.transfer_chain:
            assert key in SUPPORTED_MATERIALS, f"transfer_chain enthält '{key}' — nicht in SUPPORTED_MATERIALS"

    def test_no_raw_internal_alias_keys_in_result(self, detector):
        """Roh-Alias-Keys des Bayesian-Scorers dürfen nie in detect()-Ergebnissen erscheinen.

        D-K1-3 F5 (2026-10-09): Die frühere Variante dieses Tests verbot auch den
        kanonischen 'cassette'-Key ("interner Key statt tape") — das widersprach dem
        eigenen §6.1-Vertrag in dieser Datei ('cassette' ∈ SUPPORTED_MATERIALS,
        `test_cassette_passthrough`: `_normalize_material_key("cassette") == "cassette"`)
        und war nur zufällig grün, solange Weißrauschen auf 'tape' mündete. Die
        datengetriebene _MATERIAL_MODELS-Neukalibrierung (120 Echt-Träger) klassifiziert
        den Testclip jetzt legitimerweise als 'cassette' ⇒ der Vertrag wird präzisiert:
        verboten sind die ROHEN Scorer-Aliasse, nicht das First-Class-Material.
        """
        audio = _make_audio()
        result = detector.detect(audio, 48000)
        raw_aliases = {"reel_wire", "cassette_digital", "vhs_audio"}
        for key in result.transfer_chain:
            assert key not in raw_aliases, f"transfer_chain enthält Roh-Alias-Key '{key}' (nicht normalisiert)"
        assert result.primary_material not in raw_aliases, (
            f"primary_material='{result.primary_material}' ist ein Roh-Alias-Key (nicht normalisiert)"
        )

    def test_no_reel_wire_key_in_result(self, detector):
        """Das interne 'reel_wire'-Key darf nie in detect()-Ergebnissen erscheinen."""
        audio = _make_audio()
        result = detector.detect(audio, 48000)
        assert "reel_wire" not in result.transfer_chain
        assert result.primary_material != "reel_wire"

    def test_detect_mp3_file_ext_digital(self, detector):
        """Bei .mp3-Dateiendung → digitale Posterior-Anpassung → Ergebnis trotzdem normiert."""
        audio = _make_audio()
        result = detector.detect(audio, 48000, file_ext=".mp3")
        assert result.primary_material in SUPPORTED_MATERIALS
        for key in result.transfer_chain:
            assert key in SUPPORTED_MATERIALS

    def test_detect_wav_file_ext_digital(self, detector):
        """Bei .wav-Dateiendung → digitale Posterior → normiertes Ergebnis."""
        audio = _make_audio()
        result = detector.detect(audio, 48000, file_ext=".wav")
        assert result.primary_material in SUPPORTED_MATERIALS

    def test_detect_returns_confidence_in_range(self, detector):
        """confidence muss ∈ [0.0, 1.0] sein."""
        audio = _make_audio()
        result = detector.detect(audio, 48000)
        assert 0.0 <= result.confidence <= 1.0, f"confidence={result.confidence} außerhalb [0, 1]"

    def test_detect_transfer_chain_nonempty(self, detector):
        """transfer_chain darf nie leer sein."""
        audio = _make_audio()
        result = detector.detect(audio, 48000)
        assert len(result.transfer_chain) >= 1

    def test_detect_primary_matches_chain_first(self, detector):
        """primary_material muss gleich transfer_chain[0] sein."""
        audio = _make_audio()
        result = detector.detect(audio, 48000)
        assert result.primary_material == result.transfer_chain[0], (
            f"primary_material='{result.primary_material}' != transfer_chain[0]='{result.transfer_chain[0]}'"
        )

    def test_detect_stereo_input(self, detector):
        """Auch Stereo-Input liefert normierte SUPPORTED_MATERIALS-Keys."""
        rng = np.random.default_rng(7)
        stereo = rng.standard_normal((2, 48000 * 2)).astype(np.float32) * 0.05
        result = detector.detect(stereo, 48000)
        assert result.primary_material in SUPPORTED_MATERIALS


# ---------------------------------------------------------------------------
# Tests: Bridge-Integration
# ---------------------------------------------------------------------------


class TestBridgeMediumDetector:
    """Prüft get_medium_detector() via Bridge (\u00a711.1 Spec 08)."""

    def test_bridge_get_medium_detector_importable(self):
        """get_medium_detector muss über Bridge importierbar sein."""
        from backend.api.bridge import get_medium_detector as bridge_fn

        assert callable(bridge_fn), "get_medium_detector muss callable sein"

    def test_bridge_returns_detector_instance(self):
        """Bridge gibt MediumDetector-Singleton zurück (oder None wenn nicht verfügbar)."""
        from backend.api.bridge import get_medium_detector as bridge_fn

        instance = bridge_fn()
        # None ist akzeptabel (optional module), aber wenn vorhanden muss detect() existieren
        if instance is not None:
            assert hasattr(instance, "detect"), "MediumDetector-Instanz muss detect()-Methode haben"

    def test_bridge_in_all_list(self):
        """get_medium_detector muss in bridge.__all__ stehen."""
        from backend.api import bridge

        assert "get_medium_detector" in bridge.__all__, "get_medium_detector fehlt in bridge.__all__"
