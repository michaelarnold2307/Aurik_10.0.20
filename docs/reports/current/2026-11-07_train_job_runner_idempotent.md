# Trainings-Runner idempotent — Wiederaufnahme überspringt erledigte Jobs

**Datum:** 2026-11-07
**Anlass:** Pflicht-Check `/tmp/kun_pgrep.txt` + Manifest-Audit
**Status:** behoben, Tests grün, **kein Flag-Flip, kein GPU-Eingriff**

## 1. Befund

Der Wiederaufnahme-Vertrag in `docs/SOTA_HANDOVER.md` §3 stützt sich auf
Manifeste (`docs/reports/current/train_job_*.json`) und Job-Locks. Gemessen am
Ist-Stand:

| Manifest | `ergebnis` | `befehl` (Auszug) |
| --- | --- | --- |
| `train_job_m8_aspade.json` | `erfolgreich` | `train_aspade_declipper.py --musdb-root data/musdb18hq --epochs 30 --device cuda --out-dir models/aspade` |
| `train_job_m9_tf_gridnet.json` | `fehlgeschlagen` | `train_tf_gridnet_denoiser.py --epochs 30 --device cuda --synthetic 4000` (Altstand) |
| `train_job_c2_cantus_{pretrain,finetune,domain}.json` | `fehlgeschlagen` | `train_cantus.py` ohne `--data` (Altstand) |
| `train_job_c3_rt60.json` | `erfolgreich` | `train_rt60_regressor.py --out models/rt60_regressor/rt60_regressor.joblib` |

`scripts/train_job_runner.py` prüfte vor dem Start **kein** vorhandenes
Manifest. Ein Neustart der Kette 1 hätte damit:

1. M8 A-SPADE erneut trainiert (bis zu 6 h GPU-Zeit) und
2. das Erfolgs-Manifest von M8 mit einem neuen, zunächst leeren `versuche`-Array
   überschrieben — der Erfolgsbeleg wäre verloren gegangen.

Die Job-Locks verhindern nur **parallele** Läufe, nicht die Wiederholung eines
bereits abgeschlossenen Jobs.

## 2. Fix

`scripts/train_job_runner.py`:

- liest vor dem Start das vorhandene Manifest (`_load_manifest`);
- überspringt den Job **ohne GPU-Zeit**, wenn `ergebnis == "erfolgreich"` und
  `befehl`, `phase` und `seed` mit dem aktuellen Aufruf übereinstimmen;
- lässt das Erfolgs-Manifest dabei **unverändert** (Beleg bleibt erhalten);
- `--force` erzwingt einen Neulauf (z. B. nach geänderten Trainingsdaten);
- weicht Befehl/Phase/Seed ab, wird mit `logger.warning()` neu trainiert;
- ein unlesbares Manifest blockiert nicht: `logger.warning()` + Neulauf
  (§V6 (copilot-instructions.md), kein stilles Überspringen).

Rückwärtskompatibel: `force` ist ein Keyword-Argument mit Default `False`,
die CLI erhält nur das additive Flag `--force`.

Zusätzlich: `scripts/gpu_kette3_k4_rest.sh` trug in der Kopfzeile noch
„Läuft parallel zur D3-Kette“ — das widerspricht §5 (keine parallelen
Trainings). Kommentar an die harte Regel angeglichen; Kette 3 startet nur im
freien GPU-Fenster (kein Verhaltens-/CLI-Eingriff).

## 3. Tests

| Test | Ergebnis |
| --- | --- |
| `python3 -m pytest tests/unit/test_train_job_runner.py -q` | **8 passed** (4 bestehende + 4 neue) |
| `python3 -m pytest tests/unit/test_gpu_kette1_wiring.py -q` | **7 passed** |

Neue Tests:

- `test_erfolgreicher_job_wird_uebersprungen` — kein zweiter Trainingslauf,
  Manifest byte-identisch;
- `test_force_erzwingt_neulauf` — `--force` trainiert erneut;
- `test_geaenderter_befehl_trainiert_neu` — geänderter Befehl ist ein neuer Job;
- `test_unlesbares_manifest_blockiert_nicht` — kaputtes JSON führt zum Neulauf,
  nicht zum Stillstand.

## 4. Wirkung auf den offenen GPU-Neulauf

Der nächste Start von `scripts/gpu_kette1_welle_c.sh`:

- **überspringt M8** (`models/aspade`, `erfolgreich`, Befehl identisch),
- **überspringt RT60** (`models/rt60_regressor`, `erfolgreich`, Befehl identisch),
- **startet M9** mit den echten Musik-Paaren (`data/tfgrid_pairs`, `--batch-size 1
  --grad-accum 4`) und **Cantus R3/R4** mit `--data data/cantus_pairs/manifest.jsonl`,
- M9/Cantus bleiben offen bis zum freien GPU-Fenster; R5 (`domain_adapt`) bleibt
  datenblockiert (keine echten `*_degraded.wav`-Paare).

Damit sinkt die Wartezeit bis zum M9-/Cantus-Neulauf um den M8-Block; die
GPU-Zeit fließt in die noch offenen Jobs.

## 5. GPU-/Kettenstand bei Abschluss (unverändert)

- Kette 2 läuft: `gpu_kette2_matrix_d3.sh` (PID 2254102), Job
  `d3_diffwave_vokal`, **Versuch 2/2** nach 21600-s-Timeout von Versuch 1;
  `models/diffwave/checkpoint_latest.pt` existiert (Auto-Resume-Pfad korrekt).
- Kette 1/3/4 wurden **nicht** gestartet (kein paralleles Training, §5
  SOTA_HANDOVER).
- Kein Modell-Artefakt verändert, kein Flag-Flip, keine Hör-/Lizenzentscheidung.

## 6. Offen (unverändert)

- D3-Verifikation (`train_job_d3_diffwave_vokal.json`: `ergebnis: erfolgreich` +
  `never_worsen_per_gap: true`) im trainingsfreien Fenster.
- Kette 1 GPU-Neulauf (M9/Cantus), Kette 3 (M8/M9-Rest), Kette 4
  (WF-V4/TP-V2/F9/F10/vocos) — sequenziell nach Freiwerden der GPU.
- M9-Never-worsen-Gate/Hörabnahme und M9-Datenmenge (25 Paare) bleiben
  menschlich bzw. datenabhängig.
