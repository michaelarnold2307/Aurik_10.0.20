# Vendored MuQ — Provenienz

Unverändert kopierter Drittanbieter-Code aus dem offiziellen MuQ-Repository.

- **Quelle:** <https://github.com/tencent-ailab/MuQ> (Branch `main`, Stand 2026-09-11)
  Dateien: `src/muq/muq/` → `plugins/_vendor_muq/`
- **Paper:** Zhu et al. (2025), "MuQ: Self-Supervised Music Representation Learning
  with Mel Residual Vector Quantization", arXiv:2501.01108.
- **Code-Lizenz:** MIT — siehe `LICENSE`.
- **Gewichte-Lizenz:** CC-BY-NC-4.0 (nicht-kommerziell) — siehe `LICENSE_weights`.
  Die Gewichte selbst werden NICHT mit dem Repo verteilt; sie liegen lokal im
  HuggingFace-Cache (`OpenMuQ/MuQ-large-msd-iter`) oder unter `models/muq/`.
- **Nutzung:** `plugins/muq_plugin.py` importiert `MuQ`/`MuQConfig` aus diesem
  Paket. Keine Modifikationen am vendorten Code (Linter-Ausnahme `SKIP_DIRS`).
- **Abhängigkeiten:** torch, torchaudio, einops, easydict, transformers,
  huggingface_hub (alle im Projekt bereits vorhanden; `easydict` seit
  2026-09-11 in `requirements/requirements_aurik.txt`).
