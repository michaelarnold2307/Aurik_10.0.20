"""Historischer Vergleich zweier Denker-Aufrufformen.

Dieser Test ruft weder die reale GUI noch die CLI oder deren Export auf und
belegt keine End-to-End-Parität. Der reale GUI-Batchpfad läuft in-process.

Normative Grundlage (AGENTS.md):
    - Bridge-Verbot (§V4 (copilot-instructions.md)): CLI und GUI rufen die
      gesamte Verarbeitung ausschließlich über `backend/api/bridge.py` auf.
    - CLI/GUI-Funktionsgleichheit: funktionelle Abweichungen zwischen beiden
      Pfaden sind verboten.
    - §2.40/§G5 (GEBOTE.md): gleicher Input + gleiche Umgebung + gleicher
      Modus ⇒ bitnahe Ausgabe.

Der Test vergleicht historische Aufrufformen (Stand 2026-09-14):

  CLI-Form (cli/aurik_cli.py):
      run_pre_analysis(audio_native, sr_native, audio_48k, file_path,
                       store_in_bridge_cache=True)
      → denker.denke(audio_48k, sr=48000, mode=…, no_rt_limit=…,
                     input_path, output_path, pre_analysis_result=pre,
                     phase_strength_oracle_rollout=…)

  GUI-Form (Aurik10/ipc/pipeline_process.py):
      denker.denke(audio=…, sr=…, mode=job.mode, material=job.material,
                   variant=job.variant, quality_target=job.quality_target,
                   input_path, output_path, progress_callback=…,
                   audio_update_callback=…, cached_defect_result=…)

Der Vergleich verwendet max_abs ≤ 1e-5, rms ≤ 1e-5 und identische Samplezahl.

Ausführung: pytest tests/normative/test_cli_gui_output_parity.py -v
"""

from __future__ import annotations

import os

import numpy as np
import pytest

# CPU-Pin (§G5 (GEBOTE.md), §III.9 (copilot-instructions.md)): Gemessen 2026-10-05 auf
# der Zielhardware — mit aktivem ROCm-Pfad dauert dieser Paritätstest 113,7 s und der
# Prozess endet nach „1 passed“ mit Exit 1 (HIP-OOM beim Teardown); CPU-deterministisch
# sind es 10,4 s und Exit 0. Der Test vergleicht Ausgaben, nicht GPU-Geschwindigkeit —
# und GPU-Inferenz ist laut §G5 (GEBOTE.md) nur toleranz-gleich, nicht bit-identisch, wäre hier
# also sogar die falsche Referenz. `AURIK_FORCE_CPU=1` ist der dokumentierte Schalter.
os.environ.setdefault("AURIK_FORCE_CPU", "1")

# Toleranzen:
#   Identische Aufruf-Formen erreichen §2.40 (max_abs ≤ 1e-6, rms ≤ 1e-7).
#   Cross-Form (CLI vs. GUI) erlaubt zusätzlich numerisches Rauschen aus den
#   unterschiedlichen Analysepfaden (Voranalyse vs. interne Analyse, GPU/CPU-
#   Kernel-Nichtassoziativität) — Befund 2026-09-14: rms ~1e-6. Beide Grenzen
#   liegen ~100 dB unter dem Signal und damit weit unter jeder JND/Hörbarkeit
#   (Hörordnung §8a). max_abs bleibt hart bei 1e-5 (−100 dBFS).
MAX_ABS_ERR: float = 1e-5
RMS_ERR: float = 1e-5

_SR = 48_000
_DURATION_S = 2.0


def _make_deterministic_audio(sr: int, duration_s: float, seed: int = 42) -> np.ndarray:
    """Deterministisches Stereo-Testsignal (channels-first (2, N), §G5 (GEBOTE.md))."""
    n = int(sr * duration_s)
    t = np.linspace(0.0, duration_s, n, endpoint=False, dtype=np.float64)
    rng = np.random.default_rng(seed)
    left = 0.25 * np.sin(2.0 * np.pi * 220.0 * t) + 0.05 * rng.standard_normal(n)
    right = 0.22 * np.sin(2.0 * np.pi * 330.0 * t + 0.3) + 0.05 * rng.standard_normal(n)
    return np.stack([left, right]).astype(np.float32)


def _noop_progress(
    pct: float, phase_name: str, phase_idx: int = 0, total: int = 0, metrics: dict | None = None
) -> None:
    """GUI-Fortschritts-Callback — muss für die Parität wirkungslos bleiben."""
    del pct, phase_name, phase_idx, total, metrics


def _noop_audio_update(phase_audio: np.ndarray, phase_sr: int, phase_id: str) -> None:
    """GUI-Audio-Update-Callback — muss für die Parität wirkungslos bleiben."""
    del phase_audio, phase_sr, phase_id


@pytest.mark.timeout(600)
def test_cli_gui_bridge_output_parity(tmp_path) -> None:
    """Gleiche User-Intention ⇒ CLI-Form und GUI-Form liefern bitnahe Ausgabe.

    Beide Formen gehen über dieselbe Bridge-Singleton-Denker-Instanz.
    Unterschiede in den historischen Aufruf-Argumenten
    (pre_analysis_result vs. material-Kwargs, no_rt_limit, Callbacks) dürfen
    die Ausgabe nicht verändern.
    """
    import soundfile as sf

    from backend.api import bridge

    audio = _make_deterministic_audio(_SR, _DURATION_S)
    assert audio.ndim == 2 and audio.shape[0] == 2

    wav_path = tmp_path / "parity_input.wav"
    sf.write(str(wav_path), audio.T, _SR, subtype="PCM_24")

    denker = bridge.get_aurik_denker_instance()
    assert denker is not None, "AurikDenker über Bridge nicht verfügbar"

    # ── GUI-Form (Aurik10/ipc/pipeline_process.py, Job-Defaults) ──────────
    gui_result = denker.denke(
        audio=audio.copy(),
        sr=_SR,
        mode="restoration",
        material="vinyl",
        variant="balanced",
        quality_target=80.0,
        input_path=str(wav_path),
        output_path=str(tmp_path / "parity_gui.wav"),
        progress_callback=_noop_progress,
        audio_update_callback=_noop_audio_update,
        cached_defect_result=None,
    )

    # ── CLI-Form (cli/aurik_cli.py) ───────────────────────────────────────
    # Voranalyse exakt wie in process_audio(): store_in_bridge_cache=True.
    pre = bridge.run_pre_analysis(
        audio_native=audio.copy(),
        sr_native=_SR,
        audio_48k=audio.copy(),
        file_path=str(wav_path),
        store_in_bridge_cache=True,
    )
    cli_result = denker.denke(
        audio.copy(),
        sr=_SR,
        mode="restoration",
        # CLI-Default: AURIK_NO_RT_LIMIT ungesetzt ⇒ no_rt_limit=False — identisch
        # zur GUI-Form, die no_rt_limit gar nicht übergibt. no_rt_limit=True hätte
        # force-execute-Semantik (§2.45-Skip, unified_restorer_v3.py) und würde
        # beide Pfade künstlich divergieren lassen (Befund 2026-09-14: 0.365).
        input_path=str(wav_path),
        output_path=str(tmp_path / "parity_cli.wav"),
        pre_analysis_result=pre,
        phase_strength_oracle_rollout=None,
    )

    gui_audio = np.asarray(gui_result.audio, dtype=np.float32)
    cli_audio = np.asarray(cli_result.audio, dtype=np.float32)

    assert gui_audio.shape == cli_audio.shape, f"Formunterschied: {gui_audio.shape} vs. {cli_audio.shape}"
    assert np.isfinite(gui_audio).all() and np.isfinite(cli_audio).all()

    diff = gui_audio.astype(np.float64) - cli_audio.astype(np.float64)
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    rms = float(np.sqrt(np.mean(diff**2) + 1e-12))

    assert max_abs <= MAX_ABS_ERR, (
        f"CLI/GUI-Parität verletzt: max_abs={max_abs:.3e} > {MAX_ABS_ERR:.0e} "
        f"(§2.40 — die Aufruf-Formen dürfen die Ausgabe nicht funktional verändern)"
    )
    assert rms <= RMS_ERR, f"CLI/GUI-Parität verletzt: rms={rms:.3e} > {RMS_ERR:.0e} (§2.40)"


@pytest.mark.skipif(os.environ.get("AURIK_GUI_CLI_FULL_E2E") != "1", reason="Ressourcenintensiver Vollpipeline-Test")
@pytest.mark.timeout(900)
def test_real_gui_cli_full_pipeline_pcm_parity(tmp_path, monkeypatch) -> None:
    """Reale Voranalyse, Denker und Exporte; kein Ersatz des Restaurierungspfads."""
    import soundfile as sf
    from PyQt5.QtCore import QCoreApplication

    from Aurik10.ui import modern_window
    from backend.api import bridge
    from cli import aurik_cli

    monkeypatch.setenv("AURIK_NO_RT_LIMIT", "1")
    app = QCoreApplication.instance() or QCoreApplication([])
    assert app is not None
    wav_path = tmp_path / "input.wav"
    sf.write(wav_path, _make_deterministic_audio(_SR, _DURATION_S).T, _SR, subtype="PCM_24")
    native, sr = modern_window._load_audio_robust(str(wav_path))
    pre = bridge.run_pre_analysis(
        audio_native=native, sr_native=sr, audio_48k=native, file_path=str(wav_path), store_in_bridge_cache=True
    )
    assert pre.defects is not None

    gui_path = tmp_path / "gui.wav"
    cli_path = tmp_path / "cli.wav"
    queue = modern_window.SimpleBatchQueue()
    item = queue.add_item(
        str(wav_path),
        str(gui_path),
        {
            "mode": "RESTORATION",
            "format_key": "wav24",
            "export_sample_rate": _SR,
            "pre_analysis_result": pre,
            "cached_defect_result": pre.defects,
        },
    )
    modern_window.BatchProcessingThread(queue).run()
    assert item.status == "completed" and item.error is None, item.error
    assert any(str(phase).startswith("phase_") for phase in item.restoration_result.phases_executed), (
        "GUI hat keine UV3-Restaurierungsphase ausgeführt"
    )
    assert not item.restoration_result.metadata.get("restaurierdenker_stage_error")
    cli_result = aurik_cli.process_audio(str(wav_path), str(cli_path), verbose=False)
    assert any(str(phase).startswith("phase_") for phase in cli_result.phases_executed), (
        "CLI hat keine UV3-Restaurierungsphase ausgeführt"
    )
    assert not cli_result.metadata.get("restaurierdenker_stage_error")
    gui_pcm, gui_sr = sf.read(gui_path, dtype="int32")
    cli_pcm, cli_sr = sf.read(cli_path, dtype="int32")
    assert gui_sr == cli_sr == _SR
    np.testing.assert_array_equal(gui_pcm, cli_pcm)
