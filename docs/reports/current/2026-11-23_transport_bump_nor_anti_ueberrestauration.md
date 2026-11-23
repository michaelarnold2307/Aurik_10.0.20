# transport_bump — Anti-Überrestauration im Loss (--lambda-nor) — Re-Gate BESTANDEN

**Datum:** 2026-11-23 · **Kontext:** Spec 27, Register-Zeile 8 (ANT-A) ·
**Klasse:** transport_bump · **Gerät:** CPU (AURIK_FORCE_CPU=1, nice 19)

## Befund (Runde 1, Spec-27-Zeile-8-Signatur)

Der transport_bump-Witness-Kopf überrestaurierte mit lokalisiertem Loss
(`--lambda-lokal 1.0`): alle Epochen lagen am Q-Clip **q_median 2,0**
(= Überrestauration, DRT-3-Fehlerbild; Beleg
`2026-11-23_witness_kopf_gate_transport_bump.json` vom 18:06, vor dem
Retrain). Der Schutz-Arm (never_worsen) hielt 1,0 — überrestauriert wurde
nur auf den Schadens-Regionen selbst (Gewinn-Arm zu aggressiv).

## Ursache und Fix (Ursache statt Symptom, §V7)

Der kombinierte Loss enthielt keinen Anti-Überrestaurations-Term: der
Ausgang durfte witness-seitig „sauberer“ werden als die clean-Referenz
(q > 1). Neuer Term im Trainer `scripts/train_witness_kopf.py`:

- **`--lambda-nor`** (Default 0.0 = bit-identisch, §G5): `nor = relu(p_clean − p_out)`
  mit `p_clean` aus demselben Witness auf der clean-Referenz, `p_out` aus dem
  Witness auf dem Kopf-Ausgang. Penalty nur, wenn der Ausgang sauberer als
  clean wirkt (q > 1-Fehlerbild); `lambda_nor > 0` addiert den Term exakt.
- Report-Feld `lambda_nor` im Trainings-Report.
- 17 Vertragstests in `tests/unit/test_witness_mos_loss.py` (inkl.
  Bit-Identität des Default-Pfads) — 17/17 grün.

## Retrain und Re-Gate

```
nice -n 19 .venv_aurik/bin/python -B scripts/train_witness_kopf.py \
  --klasse transport_bump --epochen 30 --lambda-id 0.2 --lambda-lokal 1.0 \
  --lambda-witness 0.1 --lambda-spektral 0.3 --lambda-mos 0.0 \
  --q-ziel 1.0 --nw-min 0.95 --lambda-nor 0.5 --device cpu
```

30/30 Epochen (Log `/tmp/tb_nor.log`), Checkpoint
`models/klassen_koepfe/witness_transport_bump_best.pt`.

Re-Gate über den Produktions-Pfad (DRT-7, `--mos`):

| Größe | Runde 1 | Runde 2 (NOR) |
| --- | --- | --- |
| q_median | 2,0 (Clip) | **1,4034** |
| q_anteil ≥ 0,5 | 1,0 | **1,0** |
| never_worsen_anteil | 1,0 | **1,0** |
| musik_mos Δ-Mittel (report-only, DRT-8/10) | +1,29 | **+1,0345** |
| musik_mos-Anteil ≥ −JND | 1,0 | **1,0** |
| gate_bestanden | true | **true** |

Beleg: `2026-11-23_witness_kopf_gate_transport_bump.json`
(n_q_bewertbar 90, n_never_worsen_bewertbar 169, n_regionen 171,
übersprungen 2, `produktions_semantik_verifiziert: true`).

## Bindung und Grenzen

- **Kein Flag-Flip:** `use_witness_kopf_transport_bump` bleibt Default AUS
  (ANT-5); Aktivierung nur über §4 (DRT-6/8/12).
- Der NOR-Term ist ein DRT-3-Mechanismus im Trainings-Loss; Gate-Schwellen
  sind unverändert (DRT-5).
- Spec 27, Register-Zeile 8 ist damit **GELÖST** (jitter_artifacts war seit
  Runde 2 gelöst; transport_bump folgt mit dieser Runde).
