import numpy as np
import pytest
import soundfile as sf

from backend.file_import import load_audio_file


@pytest.mark.unit
def test_load_audio_file_reports_effective_metadata_after_downmix_resample(tmp_path):
    sr_in = 44100
    duration_s = 0.25
    t = np.arange(int(sr_in * duration_s), dtype=np.float32) / float(sr_in)
    audio = np.stack(
        [
            0.10 * np.sin(2.0 * np.pi * 220.0 * t),
            0.08 * np.sin(2.0 * np.pi * 330.0 * t),
            0.06 * np.sin(2.0 * np.pi * 440.0 * t),
            0.04 * np.sin(2.0 * np.pi * 550.0 * t),
        ],
        axis=-1,
    ).astype(np.float32)
    path = tmp_path / "four_channel.wav"
    sf.write(path, audio, sr_in)

    result = load_audio_file(str(path), target_sr=48000, do_carrier_analysis=False)

    assert result is not None
    assert result.get("error") is None
    imported = np.asarray(result["audio"], dtype=np.float32)
    assert result["input_channels"] == 4
    assert result["channels"] == 2
    assert result["sr"] == 48000
    assert result["format"] == "WAV"
    assert isinstance(result["meta"].get("extra_info"), str)
    assert imported.ndim == 2
    assert imported.shape[-1] == 2
    assert result["duration"] == imported.shape[0] / 48000
