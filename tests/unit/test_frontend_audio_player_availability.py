from __future__ import annotations

from types import SimpleNamespace

import pytest

import Aurik10.ui.audio_player as audio_player_mod
from Aurik10.ui.audio_player import StreamingAudioPlayer


@pytest.mark.unit
def test_streaming_player_unavailable_when_sounddevice_missing(monkeypatch) -> None:
    monkeypatch.setattr(audio_player_mod, "_SD_AVAILABLE", False)
    monkeypatch.setattr(audio_player_mod, "sd", None)

    assert StreamingAudioPlayer().available is False


def test_streaming_player_unavailable_without_output_device(monkeypatch) -> None:
    fake_sd = SimpleNamespace(query_devices=lambda kind=None: {"max_output_channels": 0, "default_samplerate": 48000.0})
    monkeypatch.setattr(audio_player_mod, "_SD_AVAILABLE", True)
    monkeypatch.setattr(audio_player_mod, "sd", fake_sd)

    assert StreamingAudioPlayer().available is False


def test_streaming_player_available_with_output_device(monkeypatch) -> None:
    fake_sd = SimpleNamespace(query_devices=lambda kind=None: {"max_output_channels": 2, "default_samplerate": 48000.0})
    monkeypatch.setattr(audio_player_mod, "_SD_AVAILABLE", True)
    monkeypatch.setattr(audio_player_mod, "sd", fake_sd)

    assert StreamingAudioPlayer().available is True
