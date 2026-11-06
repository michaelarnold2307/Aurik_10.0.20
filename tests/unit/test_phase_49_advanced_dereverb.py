"""tests/unit/test_phase_49_advanced_dereverb.py — §v10.700 I1."""

import numpy as np
import pytest

from backend.core.phases.phase_49_advanced_dereverb import AdvancedDereverbPhase


@pytest.fixture
def phase():
    return AdvancedDereverbPhase()


@pytest.fixture
def audio():
    rng = np.random.RandomState(42)
    t = np.linspace(0, 1, 48000, endpoint=False, dtype=np.float32)
    return (np.sin(2 * np.pi * 440 * t) * 0.5 + rng.randn(48000) * 0.01).astype(np.float32)


def test_returns_ndarray(phase, audio):
    result = phase.process(audio, sample_rate=48000, material_type="vinyl")
    assert isinstance(result.audio, np.ndarray)


def test_no_nan_inf(phase, audio):
    result = phase.process(audio, sample_rate=48000, material_type="vinyl")
    assert np.isfinite(result.audio).all()


def test_not_silent(phase, audio):
    result = phase.process(audio, sample_rate=48000, material_type="vinyl")
    assert float(np.sqrt(np.mean(result.audio**2))) > 1e-10


def test_length_preserved(phase, audio):
    result = phase.process(audio, sample_rate=48000, material_type="vinyl")
    assert len(result.audio) == len(audio)


class TestWpeBandsBatch:
    """§PERF-R9: _predict_reverb_bands_batch muss numerisch äquivalent zum
    alten Per-Bin-Loop sein (qualitätsneutral) und die Semantik von
    _predict_reverb_band erhalten (Silent-Bins = 0, LinAlgError ⇒ 0,
    §2.61-Budget-Teil-Ergebnis) — deterministisch nach §G5 (GEBOTE.md)."""

    @staticmethod
    def _ref_loop(phase, y_stft, power, D, K, strength):
        T, F = y_stft.shape
        n_eq = T - D - K
        reverb = np.zeros_like(y_stft)
        if n_eq < K:
            return reverb
        for f in range(F):
            if power[:, f].max() < 1e-12:
                continue
            reverb[:, f] = phase._predict_reverb_band(y_stft[:, f], power[:, f], D, K, strength)
        return reverb

    @staticmethod
    def _reverb_signal(seed: int, n: int = 48000) -> np.ndarray:
        rng = np.random.RandomState(seed)
        t = np.arange(n) / 48000.0
        x = 0.4 * np.sin(2 * np.pi * 220 * t) + 0.1 * np.sin(2 * np.pi * 440 * t) + 0.05 * rng.randn(n)
        tail = np.exp(-np.arange(6000) / 2400.0)
        x = np.convolve(x, tail / tail.sum())[:n]
        return (x / (np.abs(x).max() + 1e-9) * 0.7).astype(np.float64)

    def test_equivalent_to_per_band_loop(self, phase):
        x = self._reverb_signal(3)
        win = np.hanning(phase._WINDOW_SIZE)
        stft = phase._stft(x, win)
        power = phase._smooth_power(np.abs(stft) ** 2, alpha=0.90)
        for D, K in ((2, 3), (4, 8), (6, 12)):
            old = self._ref_loop(phase, stft, power, D, K, 0.7)
            new, exhausted = phase._predict_reverb_bands_batch(stft, power, D, K, 0.7)
            assert not exhausted
            # Numerische Äquivalenz (Float-Rauschen der batched BLAS-Ordnung),
            # weit unter jeder Gate-Schwelle.
            assert np.abs(old - new).max() < 1e-9

    def test_silent_bins_stay_zero(self, phase):
        rng = np.random.RandomState(9)
        y = (rng.randn(300, 32) + 1j * rng.randn(300, 32)).astype(np.complex128)
        power = np.abs(y) ** 2 + 1.0
        power[:, 5:8] = 0.0  # Silent-Bins
        reverb, exhausted = phase._predict_reverb_bands_batch(y, power, 3, 4, 0.7)
        assert not exhausted
        assert np.all(reverb[:, 5:8] == 0.0)
        assert np.any(np.abs(reverb[:, 0]) > 0.0)

    def test_budget_exhaustion_returns_partial(self, phase):
        rng = np.random.RandomState(11)
        y = (rng.randn(400, 64) + 1j * rng.randn(400, 64)).astype(np.complex128)
        power = np.abs(y) ** 2 + 1.0
        # Budget bereits abgelaufen ⇒ sofortiger Abbruch, Teil-Ergebnis (Nullen).
        reverb, exhausted = phase._predict_reverb_bands_batch(y, power, 3, 4, 0.7, budget_start=0.0, budget_s=0.0)
        assert exhausted
        assert np.all(reverb == 0.0)

    def test_deterministic(self, phase):
        x = self._reverb_signal(17)
        win = np.hanning(phase._WINDOW_SIZE)
        stft = phase._stft(x, win)
        power = phase._smooth_power(np.abs(stft) ** 2, alpha=0.90)
        a, _ = phase._predict_reverb_bands_batch(stft, power, 4, 8, 0.7)
        b, _ = phase._predict_reverb_bands_batch(stft, power, 4, 8, 0.7)
        assert np.array_equal(a, b)  # §G5 (GEBOTE.md): bit-identisch bei gleichem Input


class TestSgmseTier0PfadW3:
    """Pfad-Klaerung W-3 (2026-11-06):

    Report docs/reports/current/2026-11-06_w3_pfadklaerung.md. Der
    SGMSE+-Tier-0 in Verarbeitungsschritt 49 muss fuer Stereo-Fenster
    ERREICHBAR sein, sobald music_model_flags.use_sgmse_musik gesetzt ist.
    Bis zum Fix berechnete die Mindest-Signallaengen-Pruefung die Dauer auf
    der Kanal-Achse (shape[-1] auf dem channels-last-Array nach
    to_channels_last): jedes Stereo-Fenster galt als „< 1 s“, der Tier-0
    blieb unerreichbar und der W-3-Schalter war wirkungslos (Delta == 0 in
    der B3-Messung 2026-11-05). Die Tests pinnen die Schalter-Verdrahtung
    (Setzung wirkt, Sperre bleibt Sperre) und die Kurzsignal-Schwelle.
    """

    _SAMPLE_RATE = 48000

    @staticmethod
    def _stub_factory(calls):
        class _StubResult:
            def __init__(self, audio):
                self.audio = audio
                self.model_used = "stub"

        class _StubPlugin:
            def enhance(self, audio, sample_rate, sigma=0.5, max_runtime_s=None):
                calls.append({"sigma": float(sigma), "shape": tuple(audio.shape)})
                return _StubResult(np.asarray(audio, dtype=np.float32).copy())

        def _factory():
            return _StubPlugin()

        return _factory

    def _run(self, monkeypatch, samples, flag):
        calls = []
        monkeypatch.setattr("backend.core.music_model_flags.use_sgmse_musik", flag)
        monkeypatch.setattr("plugins.sgmse_plugin.get_sgmse_plus_plugin", self._stub_factory(calls))
        # System-RAM-abhaengige Budget-Entscheidung deterministisch festnageln
        # (§G5 (copilot-instructions.md)).
        monkeypatch.setattr("backend.core.ml_memory_budget.try_allocate", lambda name, gb: True)
        monkeypatch.setattr("backend.core.ml_memory_budget.release", lambda name: None)
        rng = np.random.RandomState(7)
        audio = (rng.randn(samples, 2) * 0.1).astype(np.float32)
        phase = AdvancedDereverbPhase()
        # Reverb-Presence Gate (§2.46f) wie der UV3-Orchestrator mit
        # DefectScanner-Ergebnis speisen — sonst überspringt Schritt 49 komplett.
        result = phase.process(
            audio,
            sample_rate=self._SAMPLE_RATE,
            material_type="vinyl",
            defect_scores={"reverb_excess": 0.5},
        )
        return audio, result, calls

    def test_sgmse_tier0_erreichbar_stereo(self, monkeypatch):
        # 2.5 s Stereo = Fenster-Laenge der B3-Messung (scripts/b3_stufenmessung.py).
        audio, result, calls = self._run(monkeypatch, 120000, flag=True)
        assert calls, "SGMSE+-Tier-0 bei freigeschaltetem W-3-Schalter nicht aufgerufen"
        assert result.audio.shape == audio.shape

    def test_sgmse_tier0_gesperrt_ohne_flag(self, monkeypatch):
        # §V7 (copilot-instructions.md): ohne Freigabe bleibt der Sprach-Core gesperrt.
        _audio, _result, calls = self._run(monkeypatch, 120000, flag=False)
        assert calls == []

    def test_sgmse_tier0_kurzsignal_skip_bleibt(self, monkeypatch):
        # 0.5 s < 1.0 s Mindest-Signallaenge → WPE-DSP-Fallback, kein SGMSE+.
        _audio, _result, calls = self._run(monkeypatch, 24000, flag=True)
        assert calls == []
