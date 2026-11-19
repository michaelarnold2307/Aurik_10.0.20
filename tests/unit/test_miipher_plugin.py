import pytest

"""Unit tests for the productive MIIPHER vocal SOTA adapter."""

import types
from typing import Any, cast

import numpy as np


def _patch_vocal_guards(monkeypatch) -> None:
    sys_modules = __import__("sys").modules
    monkeypatch.setitem(
        sys_modules,
        "backend.core.dsp.hnr_guard",
        types.SimpleNamespace(apply_hnr_blend=lambda pre, post, sr: (post, {"over_cleaned": False})),
    )
    monkeypatch.setitem(
        sys_modules,
        "backend.core.dsp.hallucination_guard",
        types.SimpleNamespace(
            check_hallucination=lambda pre, post, sr=48000, mode="restoration": types.SimpleNamespace(
                requires_rollback=False,
                spectral_novelty=0.0,
            )
        ),
    )


@pytest.mark.unit
def test_miipher_adapter_uses_loaded_sgmse_plus(monkeypatch):
    import plugins
    from plugins.miipher_plugin import MiipherPlugin

    _patch_vocal_guards(monkeypatch)

    class _FakeSgmseResult:
        def __init__(self, audio: np.ndarray) -> None:
            self.audio = (audio * 0.4).astype(np.float32)
            self.model_used = "sgmse_plus_torchscript"

    class _FakeSgmsePlus:
        _model_loaded = True

        @staticmethod
        def enhance(audio: np.ndarray, sr: int, **kwargs):  # pylint: disable=unused-argument
            return _FakeSgmseResult(audio)

    fake_sgmse_mod = types.SimpleNamespace(get_sgmse_plus_plugin=lambda: _FakeSgmsePlus())
    monkeypatch.setitem(__import__("sys").modules, "plugins.sgmse_plugin", fake_sgmse_mod)
    monkeypatch.setattr(plugins, "sgmse_plugin", fake_sgmse_mod, raising=False)

    plugin = MiipherPlugin()
    audio = np.linspace(-0.2, 0.2, 4800, dtype=np.float32)
    result = plugin.enhance(audio, 48000, noise_snr_db=4.0)

    np.testing.assert_allclose(result, audio * 0.4, atol=1e-7)
    metadata = plugin.route_metadata
    assert metadata["model_used"] in ("miipher_sgmse_plus_fullmix", "miipher_sgmse_plus_stem")
    assert metadata["capability_status"] == "sota_fallback"
    assert metadata["native_miipher_loaded"] is False
    assert metadata["activation_reason"] == "sgmse_plus_chain"


def test_resolve_miipher_onnx_path_uses_env_override_when_file_exists(monkeypatch, tmp_path):
    from plugins import miipher_plugin as mod

    model_file = tmp_path / "miipher.onnx"
    model_file.write_bytes(b"dummy-onnx")
    monkeypatch.setenv("AURIK_MIIPHER_ONNX_PATH", str(model_file))

    resolved = mod._resolve_miipher_onnx_path()

    assert resolved == model_file


def test_resolve_miipher_onnx_path_returns_none_when_env_file_missing(monkeypatch):
    from plugins import miipher_plugin as mod

    monkeypatch.setenv("AURIK_MIIPHER_ONNX_PATH", "/tmp/does-not-exist-miipher.onnx")

    resolved = mod._resolve_miipher_onnx_path()

    assert resolved is None


def test_miipher_adapter_falls_back_to_dfn_when_sgmse_unloaded(monkeypatch):
    import plugins
    from plugins.miipher_plugin import MiipherPlugin

    _patch_vocal_guards(monkeypatch)

    class _FakeSgmsePlus:
        _model_loaded = False

        @staticmethod
        def enhance(audio: np.ndarray, sr: int):  # pylint: disable=unused-argument
            raise AssertionError("unloaded SGMSE+ must not run")

    class _FakeDfn:
        @staticmethod
        def enhance(audio: np.ndarray, sr: int, energy_bias_db: float = -6.0, **kwargs):  # pylint: disable=unused-argument
            return (audio * 0.7).astype(np.float32)

    fake_sgmse_mod = types.SimpleNamespace(get_sgmse_plus_plugin=lambda: _FakeSgmsePlus())
    fake_dfn_mod = types.SimpleNamespace(get_deepfilternet_plugin=lambda: _FakeDfn())
    sys_modules = __import__("sys").modules
    monkeypatch.setitem(sys_modules, "plugins.sgmse_plugin", fake_sgmse_mod)
    monkeypatch.setitem(sys_modules, "plugins.deepfilternet_v3_ii_plugin", fake_dfn_mod)
    monkeypatch.setattr(plugins, "sgmse_plugin", fake_sgmse_mod, raising=False)
    # OMLSA post-filter (compute_imcra_noise_estimate) im DFN-Fallback umgehen,
    # damit der rohe DFN-Ausgang (audio * 0.7) unverändert zurückkommt.

    def _raise_omlsa(*a, **k):
        raise ImportError("omlsa disabled in test")

    monkeypatch.setitem(
        sys_modules,
        "backend.core.dsp.noise_estimator",
        types.SimpleNamespace(compute_imcra_noise_estimate=_raise_omlsa),
    )

    plugin = MiipherPlugin()
    audio = np.linspace(-0.2, 0.2, 4800, dtype=np.float32)
    result = plugin.enhance(audio, 48000, noise_snr_db=4.0)

    np.testing.assert_allclose(result, audio * 0.7, atol=1e-7)
    metadata = plugin.route_metadata
    assert metadata["model_used"] == "miipher_deepfilternet_v3_ii"
    assert metadata["capability_status"] == "sota_fallback"
    fallback_chain = metadata.get("fallback_chain", [])
    assert isinstance(fallback_chain, list)
    assert any(str(item).startswith("sgmse_plus:") for item in fallback_chain)


# ---------------------------------------------------------------------------
# Stem-based SGMSE+ (v10.0.0)
# ---------------------------------------------------------------------------


def _make_stem_sep_result(vocal_mono: np.ndarray, sdri: float = 5.0):
    """Minimaler Mock für StemSeparationResult."""
    return types.SimpleNamespace(stems={"vocals": vocal_mono}, sdri=sdri)


def test_stem_sgmse_succeeds_when_mbr_available(monkeypatch):
    """Stem-SGMSE+ Pfad wird genommen wenn MBR verfügbar und SDRi ≥ 1.0."""
    import plugins
    from plugins.miipher_plugin import MiipherPlugin

    _patch_vocal_guards(monkeypatch)

    audio = np.random.default_rng(42).uniform(-0.3, 0.3, 9600).astype(np.float32)
    # Vocal-Stem ≈ 60 % des Mix-Pegels (ausreichend Energie, gute Separation)
    vocal_mono = (audio * 0.6).astype(np.float32)
    mbr_called = []

    class _FakeMbr:
        def separate(self, a, sr, stems=None):
            mbr_called.append(True)
            return _make_stem_sep_result(vocal_mono, sdri=5.0)

    fake_mbr_mod = types.SimpleNamespace(get_bs_roformer_plugin=lambda: _FakeMbr())
    monkeypatch.setitem(__import__("sys").modules, "plugins.bs_roformer_plugin", fake_mbr_mod)

    class _FakeSgmse:
        _model_loaded = True

        @staticmethod
        def enhance(audio_in: np.ndarray, sr: int, **kwargs):
            return types.SimpleNamespace(audio=(audio_in * 0.8).astype(np.float32))

    fake_sgmse_mod = types.SimpleNamespace(get_sgmse_plus_plugin=lambda: _FakeSgmse())
    monkeypatch.setitem(__import__("sys").modules, "plugins.sgmse_plugin", fake_sgmse_mod)
    monkeypatch.setattr(plugins, "sgmse_plugin", fake_sgmse_mod, raising=False)

    plugin = MiipherPlugin()
    result = plugin.enhance(audio, 48000, noise_snr_db=5.0, panns_singing=0.6)

    assert mbr_called, "BS-RoFormer must have been called for stem separation"
    assert result.shape == audio.shape
    assert np.all(np.isfinite(result))
    assert np.max(np.abs(result)) <= 1.0
    assert plugin.route_metadata["model_used"] == "miipher_sgmse_plus_stem"


def test_stem_sgmse_falls_back_to_fullmix_when_mbr_unavailable(monkeypatch):
    """Wenn MBR nicht verfügbar → Full-Mix-SGMSE+ (bisheriger Pfad)."""
    import plugins
    from plugins.miipher_plugin import MiipherPlugin

    _patch_vocal_guards(monkeypatch)

    # MBR nicht vorhanden → ImportError beim Laden
    monkeypatch.setitem(
        __import__("sys").modules,
        "plugins.bs_roformer_plugin",
        types.SimpleNamespace(get_bs_roformer_plugin=lambda: None),
    )

    full_mix_called = []

    class _FakeSgmse:
        _model_loaded = True

        @staticmethod
        def enhance(audio_in: np.ndarray, sr: int, **kwargs):
            full_mix_called.append(True)
            return types.SimpleNamespace(audio=(audio_in * 0.5).astype(np.float32))

    fake_sgmse_mod = types.SimpleNamespace(get_sgmse_plus_plugin=lambda: _FakeSgmse())
    monkeypatch.setitem(__import__("sys").modules, "plugins.sgmse_plugin", fake_sgmse_mod)
    monkeypatch.setattr(plugins, "sgmse_plugin", fake_sgmse_mod, raising=False)

    plugin = MiipherPlugin()
    audio = np.linspace(-0.2, 0.2, 4800, dtype=np.float32)
    result = plugin.enhance(audio, 48000, noise_snr_db=5.0, panns_singing=0.5)

    assert full_mix_called, "Full-Mix-SGMSE+ muss als Fallback aufgerufen werden"
    assert result.shape == audio.shape
    assert plugin.route_metadata["model_used"] == "miipher_sgmse_plus_fullmix"


def test_stem_sgmse_falls_back_when_sdri_too_low(monkeypatch):
    """SDRi < 1.0 → RuntimeError in _enhance_stem_sgmse → Full-Mix-Fallback."""
    import plugins
    from plugins.miipher_plugin import MiipherPlugin

    _patch_vocal_guards(monkeypatch)

    audio = np.random.default_rng(7).uniform(-0.2, 0.2, 4800).astype(np.float32)
    vocal_mono = (audio * 0.5).astype(np.float32)

    class _FakeMbrLowSdri:
        def separate(self, a, sr, stems=None):
            return _make_stem_sep_result(vocal_mono, sdri=0.3)  # unter Mindestschwelle

    fake_mbr_mod = types.SimpleNamespace(get_bs_roformer_plugin=lambda: _FakeMbrLowSdri())
    monkeypatch.setitem(__import__("sys").modules, "plugins.bs_roformer_plugin", fake_mbr_mod)

    class _FakeSgmse:
        _model_loaded = True

        @staticmethod
        def enhance(audio_in: np.ndarray, sr: int, **kwargs):
            return types.SimpleNamespace(audio=(audio_in * 0.5).astype(np.float32))

    fake_sgmse_mod = types.SimpleNamespace(get_sgmse_plus_plugin=lambda: _FakeSgmse())
    monkeypatch.setitem(__import__("sys").modules, "plugins.sgmse_plugin", fake_sgmse_mod)
    monkeypatch.setattr(plugins, "sgmse_plugin", fake_sgmse_mod, raising=False)

    plugin = MiipherPlugin()
    # Muss trotzdem erfolgreich sein (Full-Mix-Fallback greift)
    result = plugin.enhance(audio, 48000, noise_snr_db=5.0, panns_singing=0.5)
    assert result.shape == audio.shape
    assert plugin.route_metadata["model_used"] == "miipher_sgmse_plus_fullmix"


# ---------------------------------------------------------------------------
# FCPE F0-guided Harmonik-Guard im DFN-Fallback
# ---------------------------------------------------------------------------


def _patch_dfn_deps(monkeypatch, audio_out_scale: float = 0.7):
    """Patcht DFN + OMLSA + Vocal-Guards für DFN-Fallback-Tests."""
    sys_modules = __import__("sys").modules
    _patch_vocal_guards(monkeypatch)

    class _FakeDfn:
        @staticmethod
        def enhance(audio: np.ndarray, sr: int, energy_bias_db: float = -6.0, **kwargs):
            return (audio * audio_out_scale).astype(np.float32)

    monkeypatch.setitem(
        sys_modules,
        "plugins.deepfilternet_v3_ii_plugin",
        types.SimpleNamespace(get_deepfilternet_plugin=lambda: _FakeDfn()),
    )
    monkeypatch.setitem(
        sys_modules,
        "backend.core.dsp.noise_estimator",
        types.SimpleNamespace(
            compute_imcra_noise_estimate=lambda audio, sr, **kw: np.zeros((1025, 10), dtype=np.float32)
        ),
    )


def test_fcpe_harmonic_guard_applied_when_panns_singing_high(monkeypatch):
    """FCPE F0-Guard wird aktiviert wenn panns_singing ≥ 0.25."""
    from plugins.miipher_plugin import MiipherPlugin

    sys_modules = __import__("sys").modules
    _patch_dfn_deps(monkeypatch, audio_out_scale=0.7)

    # MBR + SGMSE+ nicht verfügbar → DFN-Pfad wird erzwungen
    monkeypatch.setitem(
        sys_modules, "plugins.bs_roformer_plugin", types.SimpleNamespace(get_bs_roformer_plugin=lambda: None)
    )
    monkeypatch.setitem(sys_modules, "plugins.sgmse_plugin", types.SimpleNamespace(get_sgmse_plus_plugin=lambda: None))

    fcpe_called = []

    class _FakeFcpe:
        def analyze(self, audio, sr):
            fcpe_called.append(True)
            n_frames = max(1, len(audio) // 160)
            f0 = np.full(n_frames, 220.0, dtype=np.float32)  # A3 = 220 Hz
            return types.SimpleNamespace(f0_hz=f0)

    monkeypatch.setitem(sys_modules, "plugins.fcpe_plugin", types.SimpleNamespace(get_fcpe_plugin=lambda: _FakeFcpe()))

    plugin = MiipherPlugin()
    audio = np.random.default_rng(11).uniform(-0.3, 0.3, 9600).astype(np.float32)
    result = plugin.enhance(audio, 48000, noise_snr_db=5.0, panns_singing=0.5)

    assert fcpe_called, "FCPE muss aufgerufen worden sein (panns_singing=0.5 ≥ 0.25)"
    assert result.shape == audio.shape
    assert np.all(np.isfinite(result))
    assert np.max(np.abs(result)) <= 1.0
    assert plugin.route_metadata["model_used"] == "miipher_deepfilternet_v3_ii"


def test_fcpe_harmonic_guard_nonblocking_when_fcpe_fails(monkeypatch):
    """FCPE-Fehler darf DFN-Fallback nicht blockieren."""
    from plugins.miipher_plugin import MiipherPlugin

    sys_modules = __import__("sys").modules
    _patch_dfn_deps(monkeypatch, audio_out_scale=0.7)

    monkeypatch.setitem(
        sys_modules, "plugins.bs_roformer_plugin", types.SimpleNamespace(get_bs_roformer_plugin=lambda: None)
    )
    monkeypatch.setitem(sys_modules, "plugins.sgmse_plugin", types.SimpleNamespace(get_sgmse_plus_plugin=lambda: None))

    def _raise(*a, **k):
        raise RuntimeError("FCPE unavailable in test")

    monkeypatch.setitem(sys_modules, "plugins.fcpe_plugin", types.SimpleNamespace(get_fcpe_plugin=_raise))

    plugin = MiipherPlugin()
    audio = np.linspace(-0.2, 0.2, 4800, dtype=np.float32)
    # Darf nicht crashen
    result = plugin.enhance(audio, 48000, noise_snr_db=5.0, panns_singing=0.5)
    assert result.shape == audio.shape
    assert plugin.route_metadata["model_used"] == "miipher_deepfilternet_v3_ii"


def test_fcpe_harmonic_guard_skipped_when_panns_singing_low(monkeypatch):
    """FCPE wird nicht aufgerufen wenn panns_singing < 0.25."""
    from plugins.miipher_plugin import MiipherPlugin

    sys_modules = __import__("sys").modules
    _patch_dfn_deps(monkeypatch, audio_out_scale=0.7)

    monkeypatch.setitem(
        sys_modules, "plugins.bs_roformer_plugin", types.SimpleNamespace(get_bs_roformer_plugin=lambda: None)
    )
    monkeypatch.setitem(sys_modules, "plugins.sgmse_plugin", types.SimpleNamespace(get_sgmse_plus_plugin=lambda: None))

    fcpe_called = []

    class _FakeFcpe:
        def analyze(self, audio, sr):
            fcpe_called.append(True)
            return types.SimpleNamespace(f0_hz=np.zeros(10, dtype=np.float32))

    monkeypatch.setitem(sys_modules, "plugins.fcpe_plugin", types.SimpleNamespace(get_fcpe_plugin=lambda: _FakeFcpe()))

    plugin = MiipherPlugin()
    audio = np.linspace(-0.2, 0.2, 4800, dtype=np.float32)
    plugin.enhance(audio, 48000, noise_snr_db=5.0, panns_singing=0.1)  # < 0.25

    assert not fcpe_called, "FCPE darf nicht aufgerufen werden bei panns_singing < 0.25"


# ---------------------------------------------------------------------------
# Option 1: Interharmonische Dämpfung (v10.0.0)
# ---------------------------------------------------------------------------


def test_interharmonic_gain_cap_attenuates_nonharmonic_bins(monkeypatch):
    """Interharmonic NR cap senkt Energie in Nicht-Harmonik-Bändern bei voiced Frames.

    Mit panns_singing ≥ 0.25 + FCPE F0=220 Hz werden Bins zwischen Harmoniken
    auf max. 50 % Wiener-Gain begrenzt → messbar niedrigere Energie im
    non-harmonischen Band (300–380 Hz) im Vergleich zum Lauf ohne FCPE.
    """
    import scipy.signal

    from plugins.miipher_plugin import MiipherPlugin

    sys_modules = __import__("sys").modules

    # DFN gibt Signal zurück (kein Gain-Verlust, damit OMLSA-Effekt direkt messbar)
    _patch_dfn_deps(monkeypatch, audio_out_scale=1.0)

    monkeypatch.setitem(
        sys_modules, "plugins.bs_roformer_plugin", types.SimpleNamespace(get_bs_roformer_plugin=lambda: None)
    )
    monkeypatch.setitem(sys_modules, "plugins.sgmse_plugin", types.SimpleNamespace(get_sgmse_plus_plugin=lambda: None))

    class _FakeFcpe220:
        def analyze(self, audio, sr):
            n_frames = max(1, len(audio) // 160)
            return types.SimpleNamespace(f0_hz=np.full(n_frames, 220.0, dtype=np.float32))

    monkeypatch.setitem(
        sys_modules, "plugins.fcpe_plugin", types.SimpleNamespace(get_fcpe_plugin=lambda: _FakeFcpe220())
    )

    sr = 48000
    rng = np.random.default_rng(99)
    audio = rng.uniform(-0.3, 0.3, sr).astype(np.float32)  # White noise, 1 Sekunde

    # Run 1: OHNE interharmonic cap (panns_singing < 0.25 → FCPE inaktiv)
    plugin_no_cap = MiipherPlugin()
    result_no_cap = plugin_no_cap.enhance(audio.copy(), sr, noise_snr_db=5.0, panns_singing=0.1)

    # Run 2: MIT interharmonic cap (panns_singing=0.5 → FCPE aktiv, F0=220 Hz)
    plugin_with_cap = MiipherPlugin()
    result_with_cap = plugin_with_cap.enhance(audio.copy(), sr, noise_snr_db=5.0, panns_singing=0.5)

    # Energie im non-harmonischen Band 300–380 Hz (zwischen H1=220 Hz und H2=440 Hz)
    f, psd_no_cap = scipy.signal.welch(result_no_cap, sr, nperseg=4096)
    _, psd_with_cap = scipy.signal.welch(result_with_cap, sr, nperseg=4096)

    nh_mask = (f >= 300) & (f <= 380)
    energy_no_cap = float(np.mean(psd_no_cap[nh_mask]))
    energy_with_cap = float(np.mean(psd_with_cap[nh_mask]))

    assert energy_with_cap < energy_no_cap, (
        f"Interharmonic cap muss Energie im non-harmonischen Band (300–380 Hz) senken: "
        f"ohne={energy_no_cap:.6e} mit={energy_with_cap:.6e}"
    )
    assert result_with_cap.shape == audio.shape
    assert np.all(np.isfinite(result_with_cap))
    assert np.max(np.abs(result_with_cap)) <= 1.0


# ---------------------------------------------------------------------------
# Option 2: phase_65 in CAUSE_TO_PHASES für Noise-Causes
# ---------------------------------------------------------------------------


def test_run_native_miipher_onnx_uses_model_session_and_sets_path(monkeypatch):
    """_run_native_miipher_onnx: nutzt _model_session, setzt last_miipher_path='native_onnx'
    und gibt Audio gleicher Shape zurück (\u00a74.4 SOTA-Primary Pfad).
    """
    import sys

    from plugins.miipher_plugin import MiipherPlugin

    _patch_vocal_guards(monkeypatch)

    # Fake ONNX session: gibt input direkt zurück (noise-free passthrough)
    class _FakeOrtSession:
        def get_inputs(self):
            return [types.SimpleNamespace(name="waveform", shape=[1, None])]

        def get_modelmeta(self):
            return types.SimpleNamespace(custom_metadata_map={"sample_rate": "48000"})

        def run(self, output_names, feed_dict):
            wav = list(feed_dict.values())[0]
            return [wav * 0.9]  # leichte Dämpfung als Nachweis der Inferenz

    # Fake librosa (kein Resampling nötig bei model_sr==48000)
    monkeypatch.setitem(
        sys.modules,
        "librosa",
        types.SimpleNamespace(
            resample=lambda x, orig_sr, target_sr: x,
        ),
    )

    plugin = MiipherPlugin()
    plugin._model_loaded = True
    plugin._model_session = cast(Any, _FakeOrtSession())

    audio = np.linspace(-0.3, 0.3, 4800, dtype=np.float32)
    result = plugin._run_native_miipher_onnx(audio, sr=48000)

    assert plugin._last_miipher_path == "native_onnx"
    assert result.shape == audio.shape
    assert np.all(np.isfinite(result))
    assert np.max(np.abs(result)) <= 1.0


def test_enhance_routes_to_native_onnx_when_model_loaded(monkeypatch):
    """enhance(): setzt capability_status='sota_real' und model_used='miipher_native_onnx'
    wenn _model_loaded=True und native ONNX erfolgreich läuft (\u00a74.4 Routing-Korrektur).
    """
    import sys

    from plugins.miipher_plugin import MiipherPlugin

    _patch_vocal_guards(monkeypatch)

    class _FakeOrtSession:
        def get_inputs(self):
            return [types.SimpleNamespace(name="wav", shape=[1, None])]

        def get_modelmeta(self):
            return types.SimpleNamespace(custom_metadata_map={"sample_rate": "48000"})

        def run(self, output_names, feed_dict):
            wav = list(feed_dict.values())[0]
            return [wav * 0.85]

    monkeypatch.setitem(
        sys.modules,
        "librosa",
        types.SimpleNamespace(resample=lambda x, orig_sr, target_sr: x),
    )

    plugin = MiipherPlugin()
    plugin._model_loaded = True
    plugin._model_session = cast(Any, _FakeOrtSession())

    audio = np.linspace(-0.2, 0.2, 4800, dtype=np.float32)
    result = plugin.enhance(audio, 48000, noise_snr_db=3.0, panns_singing=0.7)

    meta = plugin.route_metadata
    assert meta["model_used"] == "miipher_native_onnx", (
        f"Wenn native ONNX läuft, muss model_used='miipher_native_onnx' sein, war: {meta['model_used']}"
    )
    assert meta["capability_status"] == "sota_real"
    assert meta["native_miipher_loaded"] is True
    assert plugin.is_productive()
    assert meta["activation_reason"] == "native_onnx"
    assert result.shape == audio.shape
    assert np.all(np.isfinite(result))


def test_is_productive_with_loaded_sgmse_plus(monkeypatch):
    import plugins
    from plugins.miipher_plugin import MiipherPlugin

    class _FakeSgmsePlus:
        _model_loaded = True

    fake_sgmse_mod = types.SimpleNamespace(get_sgmse_plus_plugin=lambda: _FakeSgmsePlus())
    monkeypatch.setitem(__import__("sys").modules, "plugins.sgmse_plugin", fake_sgmse_mod)
    monkeypatch.setattr(plugins, "sgmse_plugin", fake_sgmse_mod, raising=False)

    plugin = MiipherPlugin()

    assert plugin.is_productive()


def test_phase65_in_secondary_phases_for_noise_causes():
    """phase_65_vocal_naturalness_restoration muss in secondary_phases der
    Noise-Causes stehen, die schwere ML-NR auslösen (§7.10 Spec).
    """
    from backend.core.defect_phase_mapper import DefectPhaseMapper
    from backend.core.defect_scanner import DefectType

    mapper = DefectPhaseMapper()
    causes = [
        DefectType.HIGH_FREQ_NOISE,
        DefectType.NR_BREATHING_ARTIFACT,
        DefectType.GENERATION_LOSS,
        DefectType.MODULATION_NOISE,
    ]
    for cause in causes:
        assignment = mapper.get_assignment(cause)
        assert assignment is not None, f"{cause.name}: PhaseAssignment fehlt komplett"
        all_phases = list(assignment.primary_phases) + list(assignment.secondary_phases)
        assert "phase_65_vocal_naturalness_restoration" in all_phases, (
            f"{cause.name}: phase_65_vocal_naturalness_restoration fehlt in phases (§7.10)"
        )
