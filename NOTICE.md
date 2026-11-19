# NOTICE — Aurik 10.0.8

Copyright 2026 Michael Arnold

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at:

```text
http://www.apache.org/licenses/LICENSE-2.0
```

---

## Third-Party Components

This software bundles or depends upon the following open-source components.
Each component retains its original license as specified below.

### ML Models (bundled in `models/`)

| Model | License | Source |
| --- | --- | --- |
| **Vocos 24 kHz** (vocos_mel_spec_24khz.onnx) | MIT | Siuzdak (2023), github.com/hubert-siuzdak/vocos |
| **MDX23C Kim_Vocal_2 / Kim_Inst** | MIT | UVR / KimberleyJensen |
| **Apollo** (apollo_model.onnx) | Apache-2.0 | Zhang et al. (2024) |
| **DeepFilterNet v3.II** (3 ONNX files) | MIT (code) / CC BY-NC-SA 4.0 (weights) | Schröter et al. (2022) |
| **CREPE full** (model-full.onnx) | MIT | Kim et al. (2018), github.com/marl/crepe |
| **PANNs CNN14** | Apache-2.0 | Kong et al. (2020) |
| **DiffWave** (diffwave_model.onnx) | MIT | Kong et al. (2020) |
| **HiFi-GAN** (hifi_gan.onnx) | MIT | Kong et al. (2020) |
| **Resemble-Enhance** | MIT | Resemble AI (2023) |
| **Banquet Vinyl** | CC BY-NC-SA 4.0 | Bai et al. (2024) |
| **DCCRN** | MIT | Hu et al. (2020) |
| **BS-RoFormer / Mel-RoFormer** | MIT | Lu et al. (2023) |
| **UVR MDX-Net HQ 1–4** | MIT | UVR Team |
| **HTDemucs 6s** | MIT | Défossez et al. (2023) |
| **CDPAM** | MIT | Bitterman et al. (2021) |
| **Whisper Tiny** | MIT | Radford et al. (2022), OpenAI |

> **Hinweis MERT-v1-330M**: MERT wird NICHT gebündelt (CC BY-NC-SA 4.0,
> nicht-kommerziell). Es ist ein optionales Opt-in-Modul mit DSP-Fallback.
> Bei Aktivierung erscheint ein expliziter NC-Hinweis in der UI.
> **Hinweis Banquet Vinyl / DeepFilterNet-Gewichte**: CC BY-NC-SA 4.0 gilt
> für die Modellgewichte. Der Quellcode ist unter MIT/Apache-2.0 lizenziert.
> Für kommerzielle Nutzung der Gewichte bitte die jeweiligen Autoren kontaktieren.

### Python-Bibliotheken (Auswahl, vollständige Liste via `pip-licenses`)

| Paket | Lizenz |
| --- | --- |
| numpy | BSD-3-Clause |
| scipy | BSD-3-Clause |
| librosa | ISC |
| soundfile | BSD-3-Clause |
| PyQt5 | GPL-3.0 (LGPL-3.0 für Extensions) |
| onnxruntime | MIT |
| torch (CPU) | BSD-3-Clause |
| torchaudio (CPU) | BSD-2-Clause |
| FastAPI | MIT |
| pydantic | MIT |
| madmom | BSD-3-Clause |

> Vollständige, automatisch generierte SBOM:
> `scripts/generate_sbom.py --output sbom.json`
> Aktuelle SBOM je Release als `sbom-<version>.json` im Release-Anhang.

### Benchmark-Daten (NICHT im Installer)

- **MUSDB18-HQ**: CC BY-NC-SA 4.0 — Stöter et al. (2019).
  Nur in `tests/` und `benchmarks/` für Entwickler-Tests verwendet.
  Enthält keinerlei MUSDB18-HQ-Daten.

---

## Trademark-Hinweis

„Aurik" und das Aurik-Logo sind nicht-eingetragene Markenzeichen von Michael Arnold.
Die Apache-2.0-Lizenz gewährt keine Rechte zur Nutzung des Namens „Aurik" oder der
zugehörigen Logos für abgeleitete Produkte oder Dienste ohne schriftliche Genehmigung.

Detaillierte Trademark-Policy: [TRADEMARK.md](TRADEMARK.md)
