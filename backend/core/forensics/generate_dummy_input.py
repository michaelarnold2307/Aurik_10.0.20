import numpy as np
import soundfile as sf

sr = 16000
duration = 1.0
audio = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sr * duration)))
sf.write("input/test.wav", audio, sr)
