# HR-V1 BigVGAN A/B-Validierung (F3-Aktivierungsvertrag) — 2026-09-16

> §SOTA-HR-V1 (Q6/F3) — Beleg für den F3-Aktivierungsvertrag in
> `plugins/bigvgan_v2_plugin.py` / `backend/core/phases/phase_07_harmonic_restoration.py`.
> Skript: `scripts/validate_hr_v1.py` (Exit 0 = bestehen, 1 = verschlechtert, 2 = Setup-Fehler).

## Aufbau

| Feld | Wert |
|---|---|
| Eingabe | `test_audio/Testkünstlerin (Schlager) - 30 Sekunden.mp3` (20,0 s, Mono 48 kHz) |
| DSP-Pfad | `HarmonicRestorationPhase.process()` (Status quo, ohne BigVGAN) |
| BigVGAN-Pfad | `synthesize_audio` (bigvgan_v2.pth, ONNX-Runtime, Torch 2.11 ROCm 7.2) + `additive_synthesis_gate` (model=bigvgan_v2) |
| Gates | af-Delta ≥ −0,02 (click/pre-echo-Proxy aus `artifact_freedom_guard`), HNR-Delta ≥ −0,5 dB (Boersma-1993-Autokorrelation) |
| Datum/Laufzeit | 2026-09-16, ~5,5 min Wand (BigVGAN-GPU-Inferenz dominant) |

## Ergebnis — BESTANDEN (Exit 0)

| Metrik | DSP-Pfad | BigVGAN-Pfad | Delta | Toleranz | Urteil |
|---|---|---|---|---|---|
| af | 0,4582 | 0,4655 | **+0,0073** | ≥ −0,02 | ✅ |
| HNR | −4,62 dB | −0,20 dB | **+4,42 dB** | ≥ −0,5 dB | ✅ |
| PQS-MOS | — | 4,52 | — | — | Zeuge |
| bands_released | — | 26 | — | — | Zeuge |

```text
Input: test_audio/Testkünstlerin (Schlager) - 30 Sekunden.mp3 (20.0 s, Mono 48000 Hz)
DSP-Pfad:       af=0.4582  HNR=-4.62 dB
BigVGAN-Pfad:   af=0.4655  HNR=-0.20 dB  (PQS=4.52, bands_released=26)
Deltas: af +0.0073 (Toleranz −0.02) | HNR +4.42 dB (Toleranz −0.5)
✅ BigVGAN-Pfad BESTEHT die F3-Validierung — Aktivierungs-Flag darf gesetzt werden.
```

Roh-Log: `output/validate_hr_v1_2026-09-16.txt` (gitignored).

## Entscheid

Der A/B-Befund erfüllt beide Vertrags-Gates ⇒ der BigVGAN-Pfad verschlechtert
den DSP-Pfad auf diesem Material nicht (HNR sogar deutlich besser). Das
Aktivierungs-Flag (`BIGVGAN_V2_HR_ACTIVATED`) bleibt trotzdem **bewusst OFF**
(fail-closed) — die Freigabe ist ein Produktions-Rollout mit zwei noch offenen
Voraussetzungen:

1. **Test-Suite-Anpassung:** Mit Flag ON versuchen sämtliche phase_07-Tests
   (unit + integration, > 10 Dateien) echte GPU-Synthese; die
   Status-quo-Assertions (attempted=False) müssten auf den aktivierten Pfad
   umgestellt und die Synthese in der Suite gemockt werden.
2. **Performance-Budget:** Die BigVGAN-Inferenz dominiert die Laufzeit
   (~Minuten je 20 s ⇒ ≫ 10× RT); ein UV3-Budget-Nachweis
   (`scripts/benchmark_effizienz_matrix.py --enforce-budget`, §Performance-Budget
   in copilot-instructions.md) muss mit aktivem Pfad wiederholt werden.

Der Rollout ist über die EINZIGE Schaltstelle (`BIGVGAN_V2_HR_ACTIVATED`)
minimal; der A/B-Beleg hier ist die geforderte Validierungs-Hürde des
Aktivierungsvertrags und liegt damit vor.
