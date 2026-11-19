# 2026-11-08 — FlashSR-Resume-Anker: sechs GPU-Stunden ohne Wiederaufnahme (Kette 2, D3-3)

**Lücke:** `docs/SOTA_HANDOVER.md` §2 Zeile 4/11 (D3-Matrix, Ketten-Verdrahtung)
**Typ:** Verdrahtungs-/Wiederaufnahme-Defekt (kein Modellproblem, kein Flag-Flip)
**Status:** behoben; FlashSR-Training selbst bleibt GPU-offen

## 1. Befund aus dem Manifest

`docs/reports/current/train_job_d3_flashsr_musik.json` (`--phase matrix_d3`,
`--name d3_flashsr_musik`) verzeichnet zwei Versuche:

| Versuch | Dauer | Ergebnis | Kern der Ausgabe |
| --- | --- | --- | --- |
| 1 | 21 600 s | Timeout (`exit_code -15`) | Epoche 0–3 fertig (4840 + 4956 + 4770 + 4769 s ≈ **5,37 h**), Val fiel 0,4756 → 0,3726 |
| 2 | **61,3 s** | `exit_code 1` | `FileNotFoundError: [Errno 2] No such file or directory: 'models/flashsr/checkpoint_latest.pt'` |

Der Retry hat also nicht trainiert, sondern ist **nach 61 s** an einem fehlenden
Checkpoint gestorben. Die 5,37 h aus Versuch 1 waren nicht wiederaufnehmbar.

## 2. Ursache (drei Teile, ein Fehlerbild)

1. **Der Trainer schrieb den Anker nie.** `scripts/train_flashsr_f4.py` legte
   nur `checkpoint_epoch<N>.pt` und `best.pt` ab. Die Repo-Konvention
   (alle anderen Trainer: `train_bigvgan_f3.py`, `train_diffwave_vocal_inpaint.py`,
   `train_cantus.py`, `train_aspade_declipper.py` …) ist
   `<out-dir>/checkpoint_latest.pt` — genau darauf zeigte der Ketten-Befehl.
2. **Der Retry war nicht datei-geschützt.** `gpu_kette2_matrix_d3.sh` übergab
   `--resume models/flashsr/checkpoint_latest.pt` unbedingt; ohne Datei wurde
   daraus ein harter `FileNotFoundError` statt eines frischen Starts.
3. **Der Erstlauf resumte grundsätzlich nicht.** `train_job_runner.py` nutzt bei
   Versuch 1 `--cmd` (dort fehlte `--resume`) und erst danach `--resume-cmd` —
   jeder Ketten-Neustart begann deshalb bei Epoche 0, obwohl bereits trainierte
   Epochen auf der Platte lagen.

Dasselbe Fehlerbild war bei BigVGAN (D3-2) bereits einmal behoben worden
(Kommentarblock in `gpu_kette2_matrix_d3.sh`: „der alte Pfad startete den Lauf
bei jedem Start von vorn"); FlashSR war die verbliebene Stelle derselben Klasse.

## 3. Fix

- `scripts/train_flashsr_f4.py`
  - schreibt nach jeder Epoche `save_guarded(out_dir / "checkpoint_latest.pt", ckpt)`
    (atomar + rotierendes Backup, `backend/core/training_artifacts.py`);
  - `--resume` prüft die Datei **vor** dem Laden und bricht sonst mit `Exit 2`
    und klarer Meldung ab (statt Traceback nach dem Modellaufbau);
  - `val_loss` nutzt `reshape(-1)` statt doppeltem `squeeze(1)` — die
    Broadcasting-Warnung aus dem Lauf (`target [144000]` vs. `input [1,144000]`)
    ist damit beseitigt (Wert identisch, Form jetzt eindeutig).
- `scripts/gpu_kette2_matrix_d3.sh` (D3-3)
  - **beide** Befehle (`--cmd` und `--resume-cmd`) sind datei-geschützt
    (`if [ -f models/flashsr/checkpoint_latest.pt ]`), Muster wie A-SPADE in
    Kette 3: mit Anker wird fortgesetzt, ohne Anker frisch gestartet.
- **Anker restauriert:** `models/flashsr/checkpoint_latest.pt` aus dem
  vorhandenen `checkpoint_epoch3.pt` (Epoche 3, `val_a1` 0,372558) gesetzt —
  identische Payload-Struktur (`model`, `optimizer`, `epoch`, `val_a1`),
  am Artefakt geprüft. Der nächste Kettenlauf startet damit bei Epoche 4
  statt bei 0.

## 4. Tests (neu, `tests/unit/test_gpu_kette1_wiring.py`)

| Test | Pinnt |
| --- | --- |
| `test_jeder_ketten_resume_anker_wird_vom_trainer_geschrieben` | **alle** `--resume <pfad>` der vier Ketten: der genannte Dateiname muss im Ziel-Trainer vorkommen (generische Klassen-Sperre gegen den FlashSR-Fehler) |
| `test_flashsr_trainer_schreibt_anker_und_faellt_fail_closed` | `save_guarded(... "checkpoint_latest.pt")` + `is_file()`-Guard im Trainer |
| `test_flashsr_resume_ist_datei_geschuetzt` | beide Befehle der Kette prüfen den Anker, `--out-dir` = Anker-Verzeichnis |

Ergebnis: `tests/unit/test_gpu_kette1_wiring.py` **10/10 grün** (7 bestehende +
3 neue). `bash -n` der Kette und `py_compile` des Trainers sind sauber.

### Laufzeit-Beleg (CPU-only, GPU-Kette unberührt)

`HIP_VISIBLE_DEVICES="" PYTHONPATH=. python -B scripts/train_flashsr_f4.py
--smoke --data-dir data/musdb18hq --out-dir /tmp/flashsr_smoke` (2 Threads,
nice-frei, nur CPU):

- nach jeder Epoche entsteht `checkpoint_latest.pt` — geprüft: Epoche 4,
  `val_a1` 0,732647, Schlüssel `['epoch','model','optimizer','val_a1']`
  (identisch zur Struktur der Produktions-Checkpoints);
- Negativfall `--resume /tmp/gibt_es_nicht.pt` → **`PY_EXIT=2`** mit
  `ERROR Resume-Checkpoint fehlt: /tmp/gibt_es_nicht.pt — Lauf NICHT gestartet`,
  also kein Traceback und keine GPU-Zeitverbrennung mehr.

Der Smoke wurde nach 900 s abgebrochen (CPU-Resampling der Tracks ist teuer);
die Epochen 0–4 waren zu diesem Zeitpunkt geschrieben — der Beleg ist damit
vollständig, ohne dass ein GPU-Prozess gestartet wurde.

## 5. Ehrlich offen

- **Kein neues Training gestartet.** Kette 3 (`m8_aspade`) belegte die GPU zum
  Zeitpunkt des Fixes; paralleles Training ist untersagt. Der reparierte
  Kette-2-Lauf (D3-1 erfolgreich, D3-2 BigVGAN weiter ohne Erfolg, D3-3
  FlashSR) startet im nächsten freien GPU-Fenster über
  `nohup bash scripts/gpu_kette2_matrix_d3.sh > /tmp/k2.log 2>&1 &`.
- **Das F4-Gate selbst ist nicht gemessen** (ΔSDR ≥ +2 dB, Never-worsen) — das
  braucht den fertigen Finetune; unverändert offen in Roadmap-Zeile F4.
- **Wächter für Kette 4 gesetzt:** `/tmp/k4_warte.sh` (PID 3840206) pollt alle
  300 s und startet `gpu_kette4_modellkoepfe.sh`, sobald **kein**
  `train_job_runner` und **keine** `gpu_kette<N>` mehr läuft — das ist der
  ausdrücklich gewünschte Pfad für §2 Zeile 7, ohne die laufende Kette 3 zu
  stören. **Kette 2 startet bewusst nicht automatisch:** sie beginnt mit D3-2
  (BigVGAN, bisher 0 Erfolge, ~6 h je Versuch) und würde FlashSR um ~12 h
  verzögern. Wer FlashSR zuerst will, startet den Einzeljob
  (`train_job_runner.py --name d3_flashsr_musik …`, Befehl aus
  `gpu_kette2_matrix_d3.sh`) — dann greift der restaurierte Anker (Epoche 4).
- **Testumfang:** die fokussierten Ketten-Tests (10/10) **und** der
  Pre-Commit-Unit-Smoke (`tests/unit`, `maxfail=3`) sind grün; die teuren
  Voll-Suiten und der D-K3-13-UAT bleiben während der GPU-Kette gesperrt
  (Handover §5) und werden im trainingsfreien Fenster nachgezogen.
