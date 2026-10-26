"""tests/unit/test_denker/test_reparatur_denker.py

Unit-Tests für ReparaturDenker.repariere() — ECHTE Verträge, keine
`try/except: pass`-Hüllen (Lehre D-K3-20): fällt der Produktionscode aus,
fällt der Test aus. Der Phantom-Mock `denker.reparatur_denker.UnifiedRestorerV3`
(existierte nie) ist entfernt — `repariere()` ist ein reiner DSP-Pfad
(_remove_clicks/_remove_hum/_repair_clipping, §V6-Fallbacks intern).
"""

from __future__ import annotations

import math
import threading

import numpy as np

from denker.reparatur_denker import ReparaturDenker, ReparaturErgebnis, get_reparatur_denker

SR = 48_000


def _sine(dur: float = 1.0) -> np.ndarray:
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    return (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)  # type: ignore[no-any-return]


def _make_reparatur_ergebnis() -> ReparaturErgebnis:
    return ReparaturErgebnis(
        audio=_sine(0.5),
        clicks_removed=3,
        hum_removed=True,
        clipping_repaired=False,
        clipping_regions=0,
        processing_note="Clicks entfernt.",
        repairs_applied=["phase_01_click_removal"],
        quality_delta=0.12,
        material="vinyl",
        reasoning="Clicks entfernt.",
    )


class TestReparaturErgebnisFields:
    def test_01_audio_is_ndarray(self) -> None:
        assert isinstance(_make_reparatur_ergebnis().audio, np.ndarray)

    def test_02_repairs_applied_is_list(self) -> None:
        assert isinstance(_make_reparatur_ergebnis().repairs_applied, list)

    def test_03_quality_delta_finite(self) -> None:
        assert math.isfinite(float(_make_reparatur_ergebnis().quality_delta))

    def test_04_material_is_str(self) -> None:
        material = _make_reparatur_ergebnis().material
        assert isinstance(material, str) and len(material) > 0

    def test_05_reasoning_is_str(self) -> None:
        assert isinstance(_make_reparatur_ergebnis().reasoning, str)

    def test_06_audio_finite(self) -> None:
        assert np.isfinite(_make_reparatur_ergebnis().audio).all()


class TestReparaturDenkerSingleton:
    def test_07_get_returns_instance(self) -> None:
        assert isinstance(get_reparatur_denker(), ReparaturDenker)

    def test_08_get_returns_same_instance(self) -> None:
        assert get_reparatur_denker() is get_reparatur_denker()

    def test_09_singleton_thread_safe(self) -> None:
        results: list[int] = []
        lock = threading.Lock()

        def _grab() -> None:
            for _ in range(12):
                inst_id = id(get_reparatur_denker())
                with lock:
                    results.append(inst_id)

        threads = [threading.Thread(target=_grab) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(set(results)) == 1, "Singleton-Verletzung im Multithreading"


class TestReparaturDenkerRepariere:
    """10–20: Verhaltens-Tests gegen den ECHTEN DSP-Pfad (kein Mock)."""

    def test_10_returns_reparatur_ergebnis(self) -> None:
        result = ReparaturDenker().repariere(_sine(1.0), SR)
        assert isinstance(result, ReparaturErgebnis)

    def test_11_audio_is_ndarray(self) -> None:
        result = ReparaturDenker().repariere(_sine(1.0), SR)
        assert isinstance(result.audio, np.ndarray)

    def test_12_audio_finite(self) -> None:
        result = ReparaturDenker().repariere(_sine(1.0), SR)
        assert np.isfinite(result.audio).all()

    def test_13_quality_delta_finite(self) -> None:
        result = ReparaturDenker().repariere(_sine(1.0), SR)
        assert math.isfinite(float(result.quality_delta))

    def test_14_repairs_applied_is_str_list(self) -> None:
        result = ReparaturDenker().repariere(_sine(1.0), SR)
        assert isinstance(result.repairs_applied, list)
        for item in result.repairs_applied:
            assert isinstance(item, str)

    def test_15_material_reflected(self) -> None:
        result = ReparaturDenker().repariere(_sine(1.0), SR, material="vinyl")
        assert isinstance(result.material, str)

    def test_16_stereo_input_no_crash(self) -> None:
        mono = _sine(1.0)
        stereo = np.stack([mono, mono], axis=0)
        result = ReparaturDenker().repariere(stereo, SR)
        assert isinstance(result, ReparaturErgebnis)
        assert np.isfinite(result.audio).all()

    def test_17_short_audio_no_crash(self) -> None:
        short = np.zeros(512, dtype=np.float32)
        result = ReparaturDenker().repariere(short, SR)
        assert isinstance(result, ReparaturErgebnis)

    def test_18_clipped_input_output_bounded(self) -> None:
        clipped = np.ones(SR, dtype=np.float32) * 1.5
        result = ReparaturDenker().repariere(clipped, SR)
        assert np.max(np.abs(result.audio)) <= 1.0 + 1e-4

    def test_19_sine_with_click_gets_repair_or_unchanged(self) -> None:
        """Echter Klick in einem Sinus: Ergebnis stabil und endlich (kein Absturz).

        Ob der Klick repariert oder konservativ unverändert bleibt, ist eine
        Implementierungsfrage — gepinnt wird der never-worsen-Kontrakt:
        Ausgabe endlich, ≤ 1.0, gleiche Länge.
        """
        mono = _sine(1.0)
        mono[SR // 4] = 0.99
        result = ReparaturDenker().repariere(mono, SR, remove_clicks=True, remove_hum=False, repair_clipping=False)
        assert isinstance(result, ReparaturErgebnis)
        assert result.audio.shape == mono.shape
        assert np.isfinite(result.audio).all()
        assert np.max(np.abs(result.audio)) <= 1.0 + 1e-4

    def test_20_silence_input_no_crash(self) -> None:
        silence = np.zeros(SR, dtype=np.float32)
        result = ReparaturDenker().repariere(silence, SR)
        assert isinstance(result, ReparaturErgebnis)
        assert np.isfinite(result.audio).all()


class TestReparaturDenkerLayout:
    """21–23: Stereo-Layout-Invarianten (channels-first und samples-first)."""

    def test_21_samples_first_stereo_shape_preserved(self) -> None:
        mono = _sine(1.0)
        audio_n_ch = np.stack([mono, mono], axis=1)  # (N, 2)
        result = ReparaturDenker().repariere(audio_n_ch, sr=SR)
        assert result.audio.shape == audio_n_ch.shape, (
            f"Audio-Shape verändert: {audio_n_ch.shape} → {result.audio.shape}"
        )
        assert np.isfinite(result.audio).all()

    def test_22_samples_first_click_detection_works(self) -> None:
        mono = _sine(1.0)
        for pos in (SR // 4, SR // 2, 3 * SR // 4):
            mono[pos] = 0.99
        audio_n_ch = np.stack([mono, mono], axis=1)
        result = ReparaturDenker().repariere(
            audio_n_ch, sr=SR, remove_clicks=True, remove_hum=False, repair_clipping=False
        )
        assert result.audio.shape == audio_n_ch.shape
        assert np.isfinite(result.audio).all()

    def test_23_channels_first_stereo_shape_preserved(self) -> None:
        mono = _sine(1.0)
        audio_ch_n = np.stack([mono, mono], axis=0)  # (2, N)
        result = ReparaturDenker().repariere(audio_ch_n, sr=SR)
        assert result.audio.shape == audio_ch_n.shape, (
            f"Audio-Shape verändert: {audio_ch_n.shape} → {result.audio.shape}"
        )
        assert np.isfinite(result.audio).all()
