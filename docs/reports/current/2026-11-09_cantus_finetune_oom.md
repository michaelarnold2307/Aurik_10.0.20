# Cantus fine_tune: CUDA-OOM bei Batch 8 (D-K3-88)

**Datum:** 2026-11-09 · **Klasse:** K3 (Ketten-/Taktungsdefekt, kein Modellfehler)
**Betroffen:** Kette 1, R4 `c2_cantus_finetune` (`scripts/train_cantus.py`)
**Status:** Fix implementiert + unit-getestet; **GPU-Wirksamkeitsnachweis offen**
(nächster Kette-1-Lauf). Kein Flag-Flip, keine Hör-/Lizenzentscheidung.

## 1. Befund (am Manifest gemessen)

`docs/reports/current/train_job_c2_cantus_finetune.json` nach dem Kettenstart
2026-11-09 20:47:

| Versuch | Dauer | Exit | Ergebnis |
| --- | --- | --- | --- |
| 1 | 55,1 s | 1 | `torch.OutOfMemoryError` |
| 2 (Resume) | 30,6 s | 1 | `torch.OutOfMemoryError` |

Der Fehler tritt im Forward des Cantus-Blocks auf
(`models/cantus/cantus_model.py:169`, AdaLN-Modulation):

```
CUDA out of memory. Tried to allocate 750.00 MiB. GPU 0 has a total capacity
of 23.98 GiB of which 280.00 MiB is free. Of the allocated memory 22.77 GiB is
allocated by PyTorch, and 454.09 MiB is reserved by PyTorch but unallocated.
```

**Das ist kein Timeout und kein Modellproblem:** der Job stirbt reproduzierbar
nach wenigen Sekunden, beide Male identisch. Der Runner hat korrekt
zurückgemeldet (`ergebnis: fehlgeschlagen`, 2 Versuche) — die Kette hat den Job
also *richtig* als offen geführt.

## 2. Ursache

`c2_cantus_pretrain` läuft mit demselben Full-Preset (`--preset=full`) und
derselben Default-Batch-Größe 8 **erfolgreich** (Manifest `erfolgreich`). Der
Unterschied des fine_tune ist das Flag `--singmos-loss`: es lädt zusätzlich das
SingMOS-Lehrer-Modell in den VRAM und hält dessen Kalibrierungs-Pfad aktiv
(`opt_proxy`). Batch 8 × diese Zusatzlast überschreitet die 24-GiB-Karte.

Damit ist der Defekt **strukturell**: in der konfigurierten Form kann
`c2_cantus_finetune` auf dieser GPU nie `erfolgreich` werden — unabhängig von
der Modellgüte (dieselbe Klasse wie D-K3-87 bei den D3-Finetunes).

## 3. Entscheidung (autonom, ohne Qualitätsreduktion)

Die **effektive Batch bleibt 8** — es wird nur die Speicher-Taktung geändert:

- `scripts/train_cantus.py`: neues Flag `--grad-accum` (Default 1).
  Effektive Batch = `batch_size × grad_accum`. Der Loss wird je Mikro-Batch
  durch `grad_accum` geteilt, `clip_grad_norm_(5.0)` und `opt.step()` laufen am
  Ende des Akkumulations-Fensters; am Epochenende werden Rest-Gradienten
  angewendet (kein Step geht verloren). Der §0a-NaN/Inf-Guard setzt das Fenster
  zurück.
- `scripts/gpu_kette1_welle_c.sh` (R4): `--batch-size 2 --grad-accum 4` in
  Erst- **und** Resume-Befehl. Aktivierungs-Spitzenbedarf sinkt um Faktor 4,
  die Rezeptur (effektive Batch, LR, Loss-Gewichte, Epochen) bleibt unverändert.
  Muster wie SOTA-M9 (D-K3-78-Klasse: `--batch-size 1 --grad-accum 4`).

Da sich die Befehlszeile ändert, startet `train_job_runner` den Job beim
nächsten Kettenstart neu (Manifest-Vergleich ist literal) — genau das ist hier
gewollt; das alte Manifest bleibt als Beleg des OOM erhalten.

## 4. Verifikation

- `tests/unit/test_gpu_kette1_wiring.py` — **24/24 grün**, davon neu:
  `test_cantus_finetune_haelt_das_vram_budget_ein` (pinnt
  `--batch-size 2 --grad-accum 4` in beiden Befehlen und das Trainer-Flag).
- `python3 -m py_compile scripts/train_cantus.py` — clean.
- `bash -n scripts/gpu_kette1_welle_c.sh` — clean.
- Die generische Ketten-Prüfung `test_ketten_flags_existieren_in_den_trainern`
  deckt `--grad-accum` jetzt automatisch mit ab (alle vier Ketten gegen die
  `add_argument`-Flags der Trainer).

**Nicht verifiziert (ehrlich):** dass der Lauf mit 2×4 tatsächlich in den
24 GiB bleibt und 30 Epochen durchläuft. Der Beweis ist der nächste
Kette-1-Lauf; bis dahin bleibt D-K3-88 `in-arbeit`.

## 5. Offen / Grenzen

- Sollte 2×4 noch nicht reichen, ist der nächste Schritt 1×8 (gleiche effektive
  Batch, halber Peak) — dieselbe Mechanik, ein Zahlenwert.
- `--singmos-loss` bleibt aktiv: die Kalibrierung ist Teil des F10-Vertrags
  (D-K4-6) und wird nicht stillschweigend abgeschaltet (§V7).
- Keine Änderung an Pretrain/domain_adapt (dort kein SingMOS-Lehrer, kein OOM
  beobachtet).
