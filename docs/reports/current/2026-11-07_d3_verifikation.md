# D3-Verifikation — DiffWave-Vokal-Inpaint, S1-/Never-worsen-Gate bestanden

**Datum:** 2026-11-07
**Gegenstand:** D3-1 `d3_diffwave_vokal` (LUECKEN-MATRIX D.3, SOTA-M12/F1)
**Ergebnis:** **Gate bestanden** — kein Deploy, kein Flag-Flip

## 1. Kanonischer Beleg

Das Trainer-interne S1-Gate ist die Messung (ein separates
`verify_d3_quality_gate.sh` existiert nicht). Der Trainer schreibt seinen
Report unabhängig vom Job-Runner:

`docs/reports/current/2026-11-07_diffwave_vocal_finetune.json`

| Feld | Wert | Bedeutung |
| --- | --- | --- |
| `never_worsen_per_gap` | `true` | keine Lücke schlechter als Zero-Shot |
| `best_val_mean_sdr_db` | **+3,23 dB** | Referenzwert aus der Handover-Erwartung |
| `baseline_zero_shot_mean_sdr_db` | **−3,01 dB** | Zero-Shot-Baseline |
| `baseline_zero_shot_min_sdr_db` | −9,11 dB | schlechteste Einzellücke der Baseline |
| `best_epoch` | 19 | bester Validierungsstand |
| `epochs` | 14 | fortgesetzte Epochen (Resume) |
| `seed` | 42 | §G5-Determinismus |

Zusätzlicher Lauf-Beleg: `/tmp/k2.log` der Kette 2 —

```
[matrix_d3-20261106T173428Z-d8fa12] Ergebnis: erfolgreich -- Manifest: .../train_job_d3_diffwave_vokal.json
JOB_EXIT=0
```

Versuch 1 lief nach 21600 s in den Timeout (Exit −15), Versuch 2 setzte mit
`models/diffwave/checkpoint_latest.pt` fort und beendete den Job erfolgreich.

## 2. Manifest-Hinweis (kein Gate-Widerspruch)

`docs/reports/current/train_job_d3_diffwave_vokal.json` trägt aktuell nur
Versuch 1 (Timeout, `ergebnis` fehlt). Ursache: Eine parallele Session hat
während der Sitzung mit `git reset`/Pre-Commit-Stash den Arbeitsbaum auf einen
älteren Stand zurückgesetzt; der vom Runner geschriebene Erfolgsstand
(Versuch 2 + `ergebnis: erfolgreich`) ging dabei verloren. Der Trainer-Report
und das Ketten-Log sind davon unberührt und belegen den Erfolg. Es wurde
**kein** Manifest-Eintrag manuell erfunden oder nachgetragen.

## 3. Einordnung

- **Qualitäts-Gate:** bestanden (Never-worsen + Val-SDR deutlich über der
  Zero-Shot-Baseline).
- **Deploy:** NICHT automatisch. Ein D3-Artefakt ist klangwirksam; ein Deploy
  erfordert ein Never-worsen-A/B gegen den deployten Checkpoint und das
  §4-Verfahren (`docs/reports/current/2026-10-28_f_reihe_laufzeit_optimierung.md` §9).
- **RT-Budget:** Die RT-Messung 2026-11-05 (Median 32,19× RT, Baseline 53,9×)
  bleibt gültig; sie ist nicht Teil dieses Gates.
- **Kette 2:** D3-2 (BigVGAN-v2 Musik+Vokal) läuft; D3-3 (FlashSR-Musik) folgt.
  Kein paralleles Training wurde gestartet.

## 4. offen

- D3-2/D3-3 Qualitäts-Gates nach Abschluss der jeweiligen Jobs prüfen.
- Deploy-/Hörabnahme-Entscheidung für das DiffWave-Vokal-Artefakt (menschlich,
  §4-Verfahren).
- Kette-1-Neulauf (M9/Cantus) und Kette 4 (WF-V4/TP-V2/F9/F10/vocos) im freien
  GPU-Fenster.
