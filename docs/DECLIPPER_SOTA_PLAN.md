# Declipper-SOTA-Plan: A-SPADE-Klasse als Phase-07-Upgrade

> Status: Umsetzungsplan (2026-09-12) · Auftrag: Lücke 1 aus
> `docs/TIEFENANALYSE_SOTA_ABGLEICH.md` §6 schließen („Größter Einzelgewinn
> bei stark geclipptem Material"). Arbeitssprache: Deutsch.

## 1. Ausgangslage (Ist-Zustand, mit Dateibeweis)

| Fakt | Beleg |
|---|---|
| Phase 07 ist klassisch: PCHIP-Interpolation (Déger & Duhamel 2002) + adaptive Kantenblendung, logarithmische Shift-Schätzung | `backend/core/phases/phase_07_declipper.py` |
| **A-SPADE-lite existiert bereits, ist aber nicht verdrahtet** (0 Aufrufer): Sparsity-constrained iterative STFT-Rekonstruktion, validiert THD −14.7 → +12.1 dB, deterministisch, kein ML | `backend/core/dsp/sparse_declipper.py` (§v10.755, 2026-09-09) |
| DefectType `digital_clip` mit Prioritäts-/Schwellen-Mapping vorhanden | `backend/core/causal_defect_reasoner.py` (Z. 20, 75, 161) |
| Phase 07 läuft im UV3-Pfad mit `phase_07_declipper` ∈ `distortion_repair` | `backend/core/unified_restorer_v3.py` (Z. 6945, 27931) |
| Tiefenanalyse: „Declipping ist klassisch — größte Einzel-Lücke" | `docs/TIEFENANALYSE_SOTA_ABGLEICH.md` §6 Nr. 1 |

## 2. Zielbild

**Ein dreistufiger Declip-Router in Phase 07:**

1. **Sparse (A-SPADE-lite)** — milde Fälle (existiert, wird verdrahtet)
2. **Neuronal (A-SPADE, ONNX)** — mittlere/starke Fälle, Quality-Mode
3. **PCHIP** — Fallback/Edge-Fälle + wenn ML nicht verfügbar (§V6 (copilot-instructions.md): logger.warning + Begründung)

Randbedingungen (nicht verhandelbar): Determinismus §G5 (copilot-instructions.md)
— A-SPADE ist Feed-Forward/ADMM-unrolled, kein Sampling; ONNX-Float32-Inferenz
ist bit-deterministisch. Never-worsen via Listening-Witness-Gate (flat_top_rise,
HNR-Drop, THD) — nie schlechter als der Input. Layout-sicher (C, N).

## 3. Modell-Wahl (mit Evidenz)

- **A-SPADE** (Gaultier et al., „A Sparsity-based Approach to Deep Declipping",
  IEEE/ACM TASLP 2023): unrolled ADMM, 1-D-CNN-Stacks, Top-Rankings der
  SDR-Declipping-Challenges; deterministische Inferenz (kein Diffusion-Sampling).
- Alternativen geprüft und verworfen: SGMSE-Declip/Diffusions-Declipper
  (Sampling → §G5-Bruch), DCUNet-artige End-to-End-Modelle (schwächere
  publizierte SDR-Werte als A-SPADE).
- **ONNX-Verfügbarkeit:** kein offizielles ONNX-Release des Autors →
  PyTorch→ONNX-Export nötig; der Exportprozess ist im Repo etabliert
  (Spec `v10.18_onnx_export_roadmap.md`, ONNX-Sessions +
  `ml_device_manager.py`/`plugin_lifecycle_manager.py`).

## 4. Implementierungs-Slices

### Slice A — Quick Win ohne ML ✅ (2026-09-12 umgesetzt)
`sparse_declip` in `phase_07_declipper.py` verdrahtet — aber NICHT pauschal:
empirische THD-Messung zeigte, dass Sparse bei mildem Clipping (2.5×-Bursts)
verschlechtert (THD 0.0255 → 0.051), bei starkem Flat-Top (6×/12×) dagegen
massiv gewinnt (0.0424 → 0.0219 bzw. 0.0472 → 0.0096), während PCHIP dort
nichts bewirkt. Daher: **Never-worsen-Harmonik-Proxy** (`_harmonic_distortion_proxy`,
delta-basiert, AGENTS.md Guard-Kalibrierung) — der Sparse-Kandidat wird pro
Kanal nur übernommen, wenn der Proxy sinkt, sonst bleibt PCHIP. Tests:
88 grün (mild → abgelehnt, stark → übernommen, Determinismus, Fallback).

### Slice B — A-SPADE-ONNX-Plugin ✅ (2026-09-12 umgesetzt, Modell-Artefakt offen)
- `plugins/aspade_declipper_plugin.py` nach dem Muster von
  `banquet_vinyl_plugin.py`/`deepfilternet_v3_ii_plugin.py`: ONNX-Session über
  `ml_device_manager`/Providers, Chunked-Inferenz (10 s + 0.5 s Hann-Overlap-Add),
  Float32, RAM-Budget, deterministisch; Waveform→Waveform-Vertrag (1,1,T)/(1,T).
- **Gewichte-Beschaffung (2026-09-12): NICHT MÖGLICH — die neuronalen
  A-SPADE/APPLADE-Gewichte sind nicht öffentlich veröffentlicht.** Evidenz:
  GitHub-Suche (nur klassisches MATLAB-SPADE, MIT — bereits als
  `sparse_declipper`-DSP repliziert), HAL (nur SPADE-Toolbox/CASCADE),
  Hugging Face (kein Audio-Declipper). Das Plugin bleibt als fertiger
  Adapter liegen (`is_available()=False` → CQT-Diff/PCHIP, §V6
  (copilot-instructions.md)) und lädt das Modell automatisch, sobald es
  unter `models/aspade/aspade_declipper.onnx` liegt.
- **APPLADE-Code-Fund (2026-09-12):** Das offizielle APPLADE-Repo des
  Erstautors (TomoroTanaka/APPLADE, Zenodo/MATLAB-File-Exchange) enthält
  die trainierten DNN-Parameter (MATLAB .mat, 16-kHz-Sprache/LibriSpeech)
  — aber unter **NTT-Software-Lizenz „for Evaluation"**: keine Weitergabe,
  keine kommerzielle Nutzung, keine abgeleiteten Werke. Einbinden in Aurik
  (öffentliches Repo) wäre ein Lizenzverstoß → nicht integrierbar ohne
  schriftliche Freigabe von NTT. Technisch wäre der PnP-ADMM-Ansatz
  determinismus-tauglich (§G5 (copilot-instructions.md)), die Domäne aber
  Sprache/16 kHz statt Musik/48 kHz.
- **Operativer neuronaler Declip-Zweig ist CQT-Diff+** (250 MB, lokal unter
  models/cqtdiff vorhanden, §v10.752 mit KL-Guard + fixem Seed) — der
  schwere Fall ist damit neuronal abgedeckt; A-SPADE wäre die
  deterministische (sampling-freie) Ergänzung gewesen.
- Tests: 93 grün inkl. synthetischem Identity-ONNX (dynamische Zeitachse),
  Fallback ohne Modell, Determinismus.

### Slice C — Phase-07-Router ✅ (2026-09-12 umgesetzt)
Severity-Router: schwere Fälle (Clip-Runs ≥ 50 ms, Anteil ≥ 1 %) →
**A-SPADE zuerst**, dann CQT-Diff, sonst PCHIP; milde/moderate Fälle → Sparse
(Never-worsen-Harmonik-Proxy pro Kanal). ML→DSP-Fallback mit `logger.warning`
(§V6 (copilot-instructions.md)); `global_scalar` bleibt zentrale Stärke
(§V7 (copilot-instructions.md)).

### Slice D — Kalibrierung + Evidenzblock
- Witness-Gate-Kalibrierung: Schwellen auf P90 der No-Harm-Deltas über
  **N≥3 Songs** (Evidenzblock, Seed, 95 %-CI, Maintainer Sign-off —
  PR-Vertrag §4 AGENTS.md).
- Deterministischer Referenzlauf (2 Läufe bit-identisch) als Test.

## 5. Tests

1. Synthetisches Clipping (Hard-Clip bei −6/−3 dB): THD-Verbesserung
   Sparse < A-SPADE < PCHIP-Grenzen, kein Flat-Top-Rest über Input-Niveau.
2. Determinismus: zwei Inferenzläufe bit-identisch (§G5 (copilot-instructions.md)).
3. Fallback: Plugin deaktiviert → PCHIP-Pfad + Warnung, kein Crash (§V6 (copilot-instructions.md)).
4. Layout: (C, N) und (N, C) identisches Ergebnis; Stereo-Kollaps-Gate (C3) bleibt grün.
5. Budget: RT-Faktor der Phase innerhalb des Performance-Budgets dokumentieren.

## 6. Risiken

| Risiko | Mitigation |
|---|---|
| ONNX-Export-Operator-Inkompatibilität (ADMM-Custom-Ops) | Export-Frühtest in Slice B, sonst ONNX-äquivalente Reimplementierung der unrolled-Stufen (deterministisch, gleiche Gewichte) |
| CPU-RT (A-SPADE ist rechenintensiv) | Quality-Mode-Opt-in + Chunk-Parallelisierung; RT-Budget-Kaskade (Denker, Spec §9.5) |
| Über-Glättung bei leisen Passagen | Never-worsen-Witness-Gate + global_scalar; nur an geclippten Segmenten anwenden |

## 7. Reihenfolge-Empfehlung

**Slice A sofort** (ein Arbeitsschritt, kein ML-Risiko) → **Slice B/C** als
nächster Block mit ONNX-Export → **Slice D** mit überwachtem Run auf realem
Clipping-Material (Kandidaten: Pop 90er–2000er aus der Testbibliothek).
