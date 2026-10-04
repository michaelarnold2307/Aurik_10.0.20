
# Aurik 6.0 – Policy-Engine & Zielvorgaben

**Hinweis zur Implementierungsqualität:**
Alle DSP-Module sollen bereits im ersten Durchgang auf maximal möglichem SOTA-Niveau umgesetzt werden. Es werden keine Platzhalter, Dummys oder vereinfachte Algorithmen akzeptiert. Ziel ist es, dass jede DSP-Komponente direkt produktiv und mit bestmöglicher Qualität arbeitet (z. B. durch Nutzung aktueller Deep-Learning-Modelle, adaptiver Filter, spektraler Maskierung, Multiband-Prozessoren etc.).

## Zielvorgaben (Goals)

Die Policy-Engine steuert die Auswahl und Kombination der DSP-Module adaptiv anhand von Zielvorgaben ("goals").

**Beispiel-Policy:**

```python
policy = {
    "aggressiveness": 1.0,
    "goal": "max_sprachverstaendlichkeit",  # Alternativen: "min_artefakte", "max_lautheit", ...
}
```

## DSP-Kettenbildung

Die Policy-Engine wählt je nach Zielvorgabe automatisch die passende DSP-Kette:

- `max_sprachverstaendlichkeit`: ZeroCrossingRate, SpectralCentroid
- `min_artefakte`: RMSEnergy, SpectralRolloff
- `max_lautheit`: (z. B. RMSEnergy, weitere Module)
- Standard: RMSEnergy, ZeroCrossingRate

## Erweiterung (Plug-and-Play)

Neue Zielvorgaben und DSP-Kombinationen können ohne Codeänderung an der PolicyEngine registriert werden:

```python
from aurik6.policy.policy_engine import PolicyEngine

# Beispiel: CustomCompressor als DSP-Modul registrieren
PolicyEngine.register_goal("kompression", ["CustomCompressor", "RMSEnergy"])
# Beispiel: SotaDenoiser als DSP-Modul registrieren
PolicyEngine.register_goal("denoise", ["SotaDenoiser"])
# Beispiel: Speech Enhancement mit SpectralGate und SotaDenoiser
PolicyEngine.register_goal("speech_enhancement", ["SpectralGate", "SotaDenoiser"])
# Beispiel: Noise Reduction Chain mit SpectralSubtractor, SpectralGate, SotaDenoiser
PolicyEngine.register_goal("noise_reduction_chain", ["SpectralSubtractor", "SpectralGate", "SotaDenoiser"])
# Beispiel: Dynamic Enhancement Chain mit DynamicRangeExpander, SpectralGate, SotaDenoiser
PolicyEngine.register_goal("dynamic_enhancement_chain", ["DynamicRangeExpander", "SpectralGate", "SotaDenoiser"])
# Beispiel: Transient Processing Chain mit TransientShaper, DynamicRangeExpander, SotaDenoiser
PolicyEngine.register_goal("transient_processing_chain", ["TransientShaper", "DynamicRangeExpander", "SotaDenoiser"])
# Beispiel: Harmonic Enhancement Chain mit HarmonicExciter, DynamicRangeExpander, SotaDenoiser
PolicyEngine.register_goal("harmonic_enhancement_chain", ["HarmonicExciter", "DynamicRangeExpander", "SotaDenoiser"])
# Beispiel: Stereo Enhancement Chain mit StereoWidener, HarmonicExciter, SotaDenoiser
PolicyEngine.register_goal("stereo_enhancement_chain", ["StereoWidener", "HarmonicExciter", "SotaDenoiser"])
# Beispiel: Finalization Chain mit Limiter, StereoWidener, SotaDenoiser
PolicyEngine.register_goal("finalization_chain", ["Limiter", "StereoWidener", "SotaDenoiser"])
# Beispiel: De-Essing Chain mit DeEsser, Limiter, SotaDenoiser
PolicyEngine.register_goal("de_essing_chain", ["DeEsser", "Limiter", "SotaDenoiser"])
```

Beim nächsten Aufruf mit `goal="kompression"` oder `goal="denoise"` wird automatisch die neue DSP-Kette verwendet.

## Beispiel: De-Essing Chain Policy

```python
import numpy as np
from aurik6.policy.policy_engine import PolicyEngine

sr = 16000
t = np.linspace(0, 1, sr)
audio = 0.3 * np.sin(2 * np.pi * 7000 * t)

policy = {"goal": "de_essing_chain"}
engine = PolicyEngine(policy)
result = engine.process(audio, sr, policy)
de_essed = result["dsp_results"]["DeEsser"]
print("DeEsser max:", np.abs(de_essed).max())
```

## Beispiel: Finalization Chain Policy

```python
import numpy as np
from aurik6.policy.policy_engine import PolicyEngine

sr = 16000
t = np.linspace(0, 1, sr)
audio = 1.2 * np.sin(2 * np.pi * 440 * t)

policy = {"goal": "finalization_chain"}
engine = PolicyEngine(policy)
result = engine.process(audio, sr, policy)
limited = result["dsp_results"]["Limiter"]
print("Limiter max:", np.abs(limited).max())
```

## Beispiel: Stereo Enhancement Chain Policy

```python
import numpy as np
from aurik6.policy.policy_engine import PolicyEngine

sr = 16000
t = np.linspace(0, 1, sr)
audio = 0.3 * np.sin(2 * np.pi * 330 * t)

policy = {"goal": "stereo_enhancement_chain"}
engine = PolicyEngine(policy)
result = engine.process(audio, sr, policy)
stereo = result["dsp_results"]["StereoWidener"]
print("Stereo shape:", stereo.shape, "Max:", np.abs(stereo).max())
```

## Beispiel: Harmonic Enhancement Chain Policy

```python
import numpy as np
from aurik6.policy.policy_engine import PolicyEngine

sr = 16000
t = np.linspace(0, 1, sr)
audio = 0.2 * np.sin(2 * np.pi * 220 * t)

policy = {"goal": "harmonic_enhancement_chain"}
engine = PolicyEngine(policy)
result = engine.process(audio, sr, policy)
print("Shape:", result.shape, "Max:", np.abs(result).max())
```

## Beispiel: Transient Processing Chain Policy

```python
import numpy as np
from aurik6.policy.policy_engine import PolicyEngine

sr = 16000
t = np.linspace(0, 1, sr)
audio = np.sign(np.sin(2 * np.pi * 10 * t)) * 0.5 + 0.01 * np.random.randn(sr)

policy = {"goal": "transient_processing_chain"}
engine = PolicyEngine(policy)
result = engine.process(audio, sr, policy)
print("Shape:", result.shape, "Max:", np.abs(result).max())
```

## Beispiel: Dynamic Enhancement Chain Policy

```python
import numpy as np
from aurik6.policy.policy_engine import PolicyEngine

sr = 16000
t = np.linspace(0, 1, sr)
audio = 0.05 * np.sin(2 * np.pi * 440 * t) + 0.01 * np.random.randn(sr)

policy = {"goal": "dynamic_enhancement_chain"}
engine = PolicyEngine(policy)
result = engine.process(audio, sr, policy)
print("Shape:", result.shape, "Max:", np.abs(result).max())
```

## Beispiel: Noise Reduction Chain Policy

```python
import numpy as np
from aurik6.policy.policy_engine import PolicyEngine

sr = 16000
t = np.linspace(0, 1, sr)
audio = 0.2 * np.sin(2 * np.pi * 440 * t) + 0.15 * np.random.randn(sr)

policy = {"goal": "noise_reduction_chain"}
engine = PolicyEngine(policy)
result = engine.process(audio, sr, policy)
print("Shape:", result.shape, "Max:", np.abs(result).max())
```

## Beispiel: Speech Enhancement Policy

```python
import numpy as np
from aurik6.policy.policy_engine import PolicyEngine

sr = 16000
t = np.linspace(0, 1, sr)
audio = 0.1 * np.sin(2 * np.pi * 440 * t) + 0.03 * np.random.randn(sr)

policy = {"goal": "speech_enhancement"}
engine = PolicyEngine(policy)
result = engine.process(audio, sr, policy)
print("Shape:", result.shape, "Max:", np.abs(result).max())
```

## Vollständiges Beispiel: Import und Nutzung eines externen DSP-Moduls

```python
import numpy as np
from aurik6.policy.policy_engine import PolicyEngine

# Dummy-Audio erzeugen
sr = 16000
t = np.linspace(0, 1, sr)
audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.random.randn(sr)

# SotaDenoiser als Zielvorgabe registrieren
PolicyEngine.register_goal("denoise", ["SotaDenoiser"])
policy = {"goal": "denoise"}

# PolicyEngine mit SotaDenoiser nutzen
engine = PolicyEngine()
result = engine.process(audio, sr, policy)

print("Shape:", result.shape, "Max:", np.abs(result).max())
```

## Beispielaufruf

```python
from aurik6.policy.policy_engine import PolicyEngine
import numpy as np

audio = np.random.randn(16000).astype(np.float32)
policy = {"aggressiveness": 1.0, "goal": "max_sprachverstaendlichkeit"}
engine = PolicyEngine(policy, quality_threshold=3.0)
result = engine.process(audio, 16000, user_score=4.0)
print(result["dsp_chain"])  # ['ZeroCrossingRate', 'SpectralCentroid']
```
