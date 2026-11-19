"""Unit-Tests für den BEATs-Encoder-Pfad in plugins/beats_plugin.py.

Befund 2026-09-14: Der lokale ONNX ist ein Encoder-Export (fbank→768-Tokens)
ohne Tagger-Head. Geprüft: Erkennungs-Flag/Property, echte Embeddings statt
Nullen, ehrliche Cache-Herkunft (kein falsches „beats_onnx_cached“),
Null-Fallback ohne Encoder.
"""

from __future__ import annotations

import numpy as np
import pytest

import backend.core.dsp.beats_onset_detector as bod
import plugins.beats_plugin as bp


@pytest.fixture
def plugin() -> bp.BeatsPlugin:
    obj = bp.BeatsPlugin.__new__(bp.BeatsPlugin)  # ohne echtes ONNX-Loading
    obj._session = None
    obj._model_loaded = False
    obj._encoder_only = False
    return obj


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    with bp._tags_cache_lock:
        bp._tags_cache.clear()
    yield
    with bp._tags_cache_lock:
        bp._tags_cache.clear()


def _audio() -> np.ndarray:
    return (0.1 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, 48000))).astype(np.float32)


def test_encoder_only_property_default_false(plugin: bp.BeatsPlugin) -> None:
    assert plugin.beats_encoder_only is False
    plugin._encoder_only = True
    assert plugin.beats_encoder_only is True


def test_get_tags_wires_real_embeddings(plugin: bp.BeatsPlugin, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_emb = np.arange(768, dtype=np.float32)
    fake_emb = (fake_emb / np.linalg.norm(fake_emb)).astype(np.float32)
    monkeypatch.setattr(bod, "beats_pooled_embedding", lambda audio, sr: fake_emb)
    monkeypatch.setattr(
        plugin,
        "_panns_fallback",
        lambda audio, sr, top_k: bp.BeatsResult(
            tags={"Vocals": 0.8}, embeddings=np.zeros(768, dtype=np.float32), model_used="panns_fallback"
        ),
    )
    plugin._encoder_only = True
    result = plugin.get_tags(_audio(), 48000, top_k=3)
    assert result.model_used == "panns_fallback"
    assert np.array_equal(result.embeddings, fake_emb)
    assert float(np.linalg.norm(result.embeddings)) > 0.99


def test_cached_path_never_claims_beats_tagger(plugin: bp.BeatsPlugin, monkeypatch: pytest.MonkeyPatch) -> None:
    """Der Cache speichert die Tag-Quelle mit — kein falsches „beats_onnx_cached“."""
    fake_emb = np.ones(768, dtype=np.float32) / np.sqrt(768.0)
    monkeypatch.setattr(bod, "beats_pooled_embedding", lambda audio, sr: fake_emb)
    plugin._encoder_only = True
    audio = _audio()
    key = bp._cache_key(audio, 48000)
    with bp._tags_cache_lock:
        bp._tags_cache[key] = ({"Vocals": 0.7}, "panns_fallback")
    result = plugin.get_tags(audio, 48000, top_k=3)
    assert result.model_used == "panns_fallback_cached"
    assert result.tags == {"Vocals": 0.7}
    assert np.array_equal(result.embeddings, fake_emb)


def test_encoder_embeddings_zeros_without_encoder(plugin: bp.BeatsPlugin, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bod, "beats_pooled_embedding", lambda audio, sr: None)
    emb = plugin._encoder_embeddings(_audio(), 48000)
    assert emb.shape == (768,)
    assert np.all(emb == 0.0)
