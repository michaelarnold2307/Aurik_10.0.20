"""tests/unit/test_denker/test_rekonstruktions_denker.py

Unit-Tests für RekonstruktionsDenker.rekonstruiere() — ECHTE Verträge, keine
`try/except: pass`-Hüllen (Lehre D-K3-20): fällt `rekonstruiere()` aus, fällt
der Test aus. Der Phantom-Mock `denker.rekonstruktions_denker.UnifiedRestorerV3`
(existierte nie) ist entfernt.
"""

from __future__ import annotations

import math
import threading

import numpy as np

from denker.rekonstruktions_denker import (
    RekonstruktionsDenker,
    RekonstruktionsErgebnis,
    get_rekonstruktions_denker,
)

SR = 48_000


def _sine(dur: float = 1.0) -> np.ndarray:
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    return (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)  # type: ignore[no-any-return]


def _make_rekonstruktions_ergebnis() -> RekonstruktionsErgebnis:
    return RekonstruktionsErgebnis(
        audio=_sine(0.5),
        gaps_found=3,
        gaps_repaired=3,
        gaps_skipped=0,
        total_repaired_ms=45.0,
        processing_time_ms=12.0,
        detail_note="3 Lücken geschlossen.",
        gap_details=[],
        warnings=[],
        phases_applied=["phase_24_dropout_repair"],
        reconstruction_risk_profile={},
    )


class TestRekonstruktionsErgebnisFields:
    def test_01_audio_is_ndarray(self) -> None:
        assert isinstance(_make_rekonstruktions_ergebnis().audio, np.ndarray)

    def test_02_gaps_filled_non_negative_int(self) -> None:
        gaps = _make_rekonstruktions_ergebnis().gaps_filled
        assert isinstance(int(gaps), int)
        assert int(gaps) >= 0

    def test_03_reconstruction_quality_bounded(self) -> None:
        q = float(_make_rekonstruktions_ergebnis().reconstruction_quality)
        assert math.isfinite(q)
        assert 0.0 <= q <= 1.0

    def test_04_phases_applied_is_list(self) -> None:
        assert isinstance(_make_rekonstruktions_ergebnis().phases_applied, list)

    def test_05_detail_note_is_str(self) -> None:
        assert isinstance(_make_rekonstruktions_ergebnis().detail_note, str)

    def test_06_audio_finite(self) -> None:
        assert np.isfinite(_make_rekonstruktions_ergebnis().audio).all()


class TestRekonstruktionsDenkerSingleton:
    def test_07_get_returns_instance(self) -> None:
        assert isinstance(get_rekonstruktions_denker(), RekonstruktionsDenker)

    def test_08_get_returns_same_instance(self) -> None:
        assert get_rekonstruktions_denker() is get_rekonstruktions_denker()

    def test_09_singleton_thread_safe(self) -> None:
        results: list[int] = []
        lock = threading.Lock()

        def _grab() -> None:
            for _ in range(12):
                inst_id = id(get_rekonstruktions_denker())
                with lock:
                    results.append(inst_id)

        threads = [threading.Thread(target=_grab) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(set(results)) == 1, "Singleton-Verletzung im Multithreading"


class TestRekonstruktionsDenkerRekonstruiere:
    """10–20: direkte Aufrufe gegen den ECHTEN `rekonstruiere()`-Pfad."""

    def test_10_returns_rekonstruktions_ergebnis(self) -> None:
        result = RekonstruktionsDenker().rekonstruiere(_sine(1.0), SR)
        assert isinstance(result, RekonstruktionsErgebnis)

    def test_11_audio_is_ndarray(self) -> None:
        result = RekonstruktionsDenker().rekonstruiere(_sine(1.0), SR)
        assert isinstance(result.audio, np.ndarray)

    def test_12_audio_finite(self) -> None:
        result = RekonstruktionsDenker().rekonstruiere(_sine(1.0), SR)
        assert np.isfinite(result.audio).all()

    def test_13_gaps_filled_non_negative(self) -> None:
        result = RekonstruktionsDenker().rekonstruiere(_sine(1.0), SR)
        assert int(result.gaps_filled) >= 0

    def test_14_reconstruction_quality_bounded(self) -> None:
        result = RekonstruktionsDenker().rekonstruiere(_sine(1.0), SR)
        q = float(result.reconstruction_quality)
        assert math.isfinite(q)
        assert 0.0 <= q <= 1.0

    def test_15_silence_no_false_repair(self) -> None:
        """Stille enthält keine Dropout-Lücken ⇒ keine Füllung (never-worsen)."""
        silence = np.zeros(SR, dtype=np.float32)
        result = RekonstruktionsDenker().rekonstruiere(silence, SR)
        assert isinstance(result, RekonstruktionsErgebnis)
        assert int(result.gaps_filled) == 0

    def test_16_stereo_input_no_crash(self) -> None:
        mono = _sine(1.0)
        stereo = np.stack([mono, mono], axis=0)
        result = RekonstruktionsDenker().rekonstruiere(stereo, SR)
        assert isinstance(result, RekonstruktionsErgebnis)
        assert np.isfinite(result.audio).all()

    def test_17_short_audio_no_crash(self) -> None:
        short = np.zeros(512, dtype=np.float32)
        result = RekonstruktionsDenker().rekonstruiere(short, SR)
        assert isinstance(result, RekonstruktionsErgebnis)

    def test_18_audio_shape_preserved(self) -> None:
        audio = _sine(1.0)
        result = RekonstruktionsDenker().rekonstruiere(audio, SR)
        assert result.audio.shape == audio.shape

    def test_19_clipped_input_no_crash(self) -> None:
        clipped = np.ones(SR, dtype=np.float32) * 1.5
        result = RekonstruktionsDenker().rekonstruiere(clipped, SR)
        assert isinstance(result, RekonstruktionsErgebnis)
        assert np.isfinite(result.audio).all()

    def test_20_phases_applied_list_of_strings(self) -> None:
        result = RekonstruktionsDenker().rekonstruiere(_sine(1.0), SR)
        assert isinstance(result.phases_applied, list)
        for item in result.phases_applied:
            assert isinstance(item, str)
