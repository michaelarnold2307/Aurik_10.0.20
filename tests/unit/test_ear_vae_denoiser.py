"""Tests für das EAR-VAE-Denoiser-Plugin (§V6 (copilot-instructions.md)-Ersatzpfad)."""

from __future__ import annotations

import numpy as np

import plugins.ear_vae_denoiser as ev_mod


def test_missing_model_falls_back_to_none(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(ev_mod, "_MODEL_DIR", "models/ear_vae_does_not_exist")
    monkeypatch.setattr(ev_mod, "_INSTANCE", None)
    plugin = ev_mod.get_ear_vae_denoiser()
    assert plugin.available is False
    assert plugin.denoise(np.zeros(44100, dtype=np.float32), 44100) is None


def test_layout_normalization_passthrough_shape(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """(N, C)-Layout wird normalisiert; ohne Modell bleibt der None-Pfad."""
    monkeypatch.setattr(ev_mod, "_MODEL_DIR", "models/ear_vae_does_not_exist")
    monkeypatch.setattr(ev_mod, "_INSTANCE", None)
    plugin = ev_mod.get_ear_vae_denoiser()
    stereo_nc = np.zeros((44100, 2), dtype=np.float32)
    assert plugin.denoise(stereo_nc, 44100) is None
