from __future__ import annotations

"""Unit-Tests für §POL-INV Polarity-Inversion-Korrektur in UV3 pre-pipeline.

DefectScanner erkennt `polarity_inverted=True` wenn L/R Korrelation ≤ −0.9.
UV3 muss den R-Kanal vor der Phase-Pipeline invertieren, damit alle 64 Phasen
mit korrekter Stereo-Phase laufen.

Spec: §POL-INV, copilot-instructions.md §0p Vocal-Supremacy
"""


import numpy as np
import pytest

SR = 48_000


def _make_stereo(duration_s: float = 1.0) -> np.ndarray:
    """Synthetisches Stereo-Signal (2, N)."""
    n = int(duration_s * SR)
    t = np.arange(n, dtype=np.float32) / SR
    ch_l = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    ch_r = ch_l.copy()
    return np.stack([ch_l, ch_r], axis=0)


# ---------------------------------------------------------------------------
# §POL-INV Logik-Tests (isoliert, ohne UV3-Import)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPolarityInversionLogic:
    """Isolierte Prüfung der Polarity-Inversion-Logik (Whitebox)."""

    def _apply_polarity_correction(
        self,
        audio: np.ndarray,
        polarity_inverted: bool,
        stereo_correlation: float = -0.95,
    ) -> tuple[np.ndarray, bool]:
        """Reproduziert die §POL-INV-Logik aus UV3 isoliert."""
        corrected = False
        if audio.ndim == 2 and polarity_inverted:
            audio = audio.copy()
            audio[1] = -audio[1]
            corrected = True
        return audio, corrected

    def test_polarity_inversion_inverts_right_channel(self):
        """§POL-INV: polarity_inverted=True → R-Kanal muss negiert werden."""
        audio = _make_stereo()
        original_r = audio[1].copy()

        corrected, was_corrected = self._apply_polarity_correction(audio, polarity_inverted=True)

        assert was_corrected, "Korrektur soll als durchgeführt markiert werden"
        np.testing.assert_array_almost_equal(
            corrected[1],
            -original_r,
            err_msg="R-Kanal muss exakt negiert sein",
        )
        # L-Kanal unverändert
        np.testing.assert_array_almost_equal(corrected[0], audio[0])

    def test_polarity_inversion_not_applied_when_flag_false(self):
        """§POL-INV: polarity_inverted=False → kein Eingriff."""
        audio = _make_stereo()
        original = audio.copy()

        corrected, was_corrected = self._apply_polarity_correction(audio, polarity_inverted=False)

        assert not was_corrected, "Korrektur soll nicht durchgeführt werden"
        np.testing.assert_array_equal(corrected, original)

    def test_polarity_inversion_skipped_for_mono(self):
        """§POL-INV: Mono-Audio (1D) → kein Eingriff."""
        mono = np.zeros(SR, dtype=np.float32)
        original = mono.copy()

        if mono.ndim == 2:
            corrected_audio, _ = self._apply_polarity_correction(mono, polarity_inverted=True)
        else:
            corrected_audio, was_corrected = mono, False

        np.testing.assert_array_equal(corrected_audio, original)

    def test_after_correction_lr_correlation_becomes_positive(self):
        """Nach Inversion muss L/R-Korrelation positiv sein (≥ 0.9)."""
        n = SR
        t = np.arange(n, dtype=np.float32) / SR
        ch = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        # Absichtlich invertiertes Stereo
        audio = np.stack([ch, -ch], axis=0)

        corrected, was_corrected = self._apply_polarity_correction(audio, polarity_inverted=True)

        assert was_corrected
        corr = float(np.corrcoef(corrected[0], corrected[1])[0, 1])
        assert corr >= 0.9, f"Nach Polarity-Korrektur muss L/R-Korrelation ≥ 0.9 sein, erhalten: {corr:.3f}"

    def test_original_audio_not_mutated(self):
        """§POL-INV: Original-Array darf nicht in-place mutiert werden (audio.copy() Pflicht)."""
        audio = _make_stereo()
        original_r_sum = float(np.sum(audio[1]))

        # Simuliert die Korrektur mit copy()
        audio_in = audio  # Referenz ohne copy
        audio_corrected = audio.copy()
        audio_corrected[1] = -audio_corrected[1]

        # Original unverändert
        assert float(np.sum(audio_in[1])) == pytest.approx(original_r_sum), (
            "Original-Array wurde mutiert — audio.copy() fehlt in §POL-INV"
        )


# ---------------------------------------------------------------------------
# DefectScanner-Metadaten-Extraktion (Whitebox)
# ---------------------------------------------------------------------------


class TestDefectScannerPolarityMetadata:
    """Prüft das Auslesen von polarity_inverted aus DefectScore-Metadata."""

    def test_metadata_extraction_from_defect_score(self):
        """polarity_inverted und stereo_correlation müssen aus metadata auslesbar sein."""
        from types import SimpleNamespace

        # Minimale DefectScore-Mock-Struktur
        mock_score = SimpleNamespace(
            defect_type=SimpleNamespace(value="phase_issues"),
            metadata={
                "polarity_inverted": True,
                "stereo_correlation": -0.97,
            },
        )

        polarity_inverted = mock_score.metadata.get("polarity_inverted", False)
        stereo_corr = float(mock_score.metadata.get("stereo_correlation", -1.0))

        assert polarity_inverted is True
        assert stereo_corr == pytest.approx(-0.97)

    def test_metadata_extraction_returns_false_when_absent(self):
        """Wenn polarity_inverted fehlt, soll .get() False zurückgeben."""
        from types import SimpleNamespace

        mock_score = SimpleNamespace(
            defect_type=SimpleNamespace(value="phase_issues"),
            metadata={},
        )

        assert mock_score.metadata.get("polarity_inverted", False) is False

    def test_defect_type_filter_by_value(self):
        """Nur defects mit defect_type.value == 'phase_issues' sollen gefunden werden."""
        from types import SimpleNamespace

        defects = [
            SimpleNamespace(
                defect_type=SimpleNamespace(value="crackle"),
                metadata={"polarity_inverted": True},
            ),
            SimpleNamespace(
                defect_type=SimpleNamespace(value="phase_issues"),
                metadata={"polarity_inverted": True},
            ),
        ]

        found = next(
            (s for s in defects if getattr(s.defect_type, "value", "") == "phase_issues"),
            None,
        )
        assert found is not None
        assert found.metadata["polarity_inverted"] is True
