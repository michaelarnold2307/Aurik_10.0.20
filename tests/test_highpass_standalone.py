#!/usr/bin/env python3
"""Quick test of HighpassFilter with stereo audio"""

import numpy as np

from backend.core.dsp.classic_filters import HighpassFilter

# Create test stereo audio: (2, 1000) format
audio_stereo = np.random.randn(2, 1000).astype(np.float64)
print(f"Input audio shape: {audio_stereo.shape}, dtype: {audio_stereo.dtype}")

# Apply highpass filter
hpf = HighpassFilter(cutoff_hz=20.0, sr=48000)
output = hpf.process(audio_stereo)

print(f"Output audio shape: {output.shape}, dtype: {output.dtype}")
print("✓ Test erfolgreich!" if output.shape == (2, 1000) else "❌ Test fehlgeschlagen!")
