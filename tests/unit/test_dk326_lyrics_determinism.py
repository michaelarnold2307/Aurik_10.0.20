"""D-K3-26: Lyrics-Modelle laden deterministisch (PLM-Eviction / ML-Budget).

Pinnen den §G5-Fix (copilot-instructions.md) aus Release 10.12.8:

  - Winzige §2.36-Pflichtmodelle (Whisper-Tiny, wav2vec2, HF-Whisper) überleben
    die ML-Budget-Kappung — der Phonem-Masken-Pfad hängt nicht mehr vom
    Plugin-Bestand ab (gemessene Streuung vorher: corr 0,96606–0,99949,
    ML-Roh-NR 5,27 vs. 6,74 dB).
  - ``_ensure_models_loaded`` lädt nach einer PLM-Eviction deterministisch nach
    (idempotent, Lock-geschützt).
  - ``keep_warm=True`` schützt die Modelle vor der Phasen-Eviction.
  - Dieselbe Eingabe liefert zweimal dieselbe Phonem-Maske.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_deterministic_tiny_models_survive_budget_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """§D-K3-26: §2.36-Pflichtmodelle laden auch bei vollem Budget deterministisch."""
    from backend.core import ml_memory_budget as budget

    monkeypatch.setattr(budget, "ML_MAX_GB", 0.10)
    monkeypatch.setattr(budget, "is_system_thrashing", lambda: False)
    monkeypatch.setattr(budget, "_should_block_heavy_ml_load", lambda _size: False)
    monkeypatch.setattr(budget, "_preflight_system_memory", lambda required_mb: True)
    monkeypatch.setattr(budget, "_allocated", {})
    monkeypatch.setattr(budget, "_total_gb", 0.08)

    # OOM-Schutz bleibt scharf: großes Modell über der Kappung wird blockiert.
    assert budget.try_allocate("BigModel", 0.5) is False
    # §2.36-Pflichtmodelle überleben die Kappung (deterministischer Pfad).
    assert budget.try_allocate("lyrics_transcriber_whisper", 0.04) is True
    assert budget.try_allocate("lyrics_aligner_wav2vec2", 0.13) is True
    assert budget.try_allocate("lyrics_whisper_hf", 0.25) is True


def test_ensure_models_loaded_reloads_after_eviction(monkeypatch: pytest.MonkeyPatch) -> None:
    """§D-K3-26: Nach PLM-Eviction lädt der nächste Bedarf deterministisch nach."""
    from backend.core import lyrics_guided_enhancement as lge

    inst = object.__new__(lge.LyricsGuidedEnhancement)
    inst._ort_session = None
    inst._aligner_session = None
    inst._whisper_hf_processor = None
    inst._whisper_hf_model = None

    calls: list[str] = []

    def _load_hf() -> None:
        calls.append("hf")  # Modell-Datei fehlt → bleibt None (simuliert)

    def _load_onnx() -> None:
        calls.append("onnx")
        inst._ort_session = object()

    def _load_aligner() -> None:
        calls.append("aligner")
        inst._aligner_session = object()

    monkeypatch.setattr(inst, "_try_load_hf_whisper", _load_hf)
    monkeypatch.setattr(inst, "_try_load_onnx", _load_onnx)
    monkeypatch.setattr(inst, "_try_load_aligner", _load_aligner)

    assert inst._ensure_models_loaded() is True
    assert calls == ["hf", "onnx", "aligner"]

    # Idempotent: geladene Modelle lösen keinen weiteren Load aus.
    assert inst._ensure_models_loaded() is True
    assert calls == ["hf", "onnx", "aligner"]

    # PLM-Eviction (Sessions genullt) → nächster Bedarf lädt erneut.
    inst._ort_session = None
    inst._aligner_session = None
    assert inst._ensure_models_loaded() is True
    assert calls == ["hf", "onnx", "aligner", "hf", "onnx", "aligner"]


def test_keep_warm_protects_lyrics_models_from_phase_eviction(monkeypatch: pytest.MonkeyPatch) -> None:
    """§D-K3-26: keep_warm-Registrierung überlebt die Phasen-Eviction des PLM."""
    from backend.core.plugin_lifecycle_manager import (
        get_plugin_lifecycle_manager,
        register_plugin,
    )

    mgr = get_plugin_lifecycle_manager()
    monkeypatch.setattr(mgr, "_entries", {})
    monkeypatch.setattr(mgr, "_lookahead_models", frozenset())

    evicted: list[str] = []
    register_plugin(
        "fake_lyrics_keepwarm",
        size_gb=0.01,
        unload_fn=lambda: evicted.append("keepwarm"),
        keep_warm=True,
    )
    register_plugin(
        "fake_regular_plugin",
        size_gb=0.01,
        unload_fn=lambda: evicted.append("regular"),
    )

    n = mgr.evict_for_phase("phase_03_denoise")
    assert n == 1
    assert evicted == ["regular"]
    assert "fake_lyrics_keepwarm" in mgr._entries
    assert "fake_regular_plugin" not in mgr._entries


def test_phoneme_mask_is_deterministic_without_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """§D-K3-26/§G5 (copilot-instructions.md): Dieselbe Eingabe → dieselbe Phonem-Maske (DSP-Ersatzpfad)."""
    from backend.core import lyrics_guided_enhancement as lge

    monkeypatch.setattr(lge.LyricsGuidedEnhancement, "_try_load_hf_whisper", lambda self: None)
    monkeypatch.setattr(lge.LyricsGuidedEnhancement, "_try_load_onnx", lambda self: None)
    monkeypatch.setattr(lge.LyricsGuidedEnhancement, "_try_load_aligner", lambda self: None)
    monkeypatch.setattr(lge, "_lge_instance", None)

    rng = np.random.default_rng(7)
    sig = (rng.standard_normal(48_000) * 0.1).astype(np.float32)

    m1 = lge.get_phoneme_mask(sig, 48_000)
    m2 = lge.get_phoneme_mask(sig, 48_000)

    assert m1.dtype == np.bool_
    assert m1.shape == m2.shape
    assert np.array_equal(m1, m2)
