# D3-Taktung: Zeitbudget statt 6-h-SIGTERM (D-K3-87)

**Datum:** 2026-11-09 · **Klasse:** K3 (Ketten-/Taktungsdefekt, kein Modellfehler)
**Betroffen:** Kette 2 (LUECKEN-MATRIX D.3), D3-2 BigVGAN-v2, D3-3 FlashSR
**Status:** Fix implementiert + unit-getestet; **GPU-Wirksamkeitsnachweis offen**
(nächster Kette-2-Lauf). Kein Flag-Flip, keine Hör-/Lizenzentscheidung.

## 1. Befund (am Manifest gemessen)

| Job | Epochen-Dauer (gemessen) | 30 Epochen | Konfiguriert | Erreicht |
| --- | --- | --- | --- | --- |
| D3-2 BigVGAN (`train_job_d3_bigvgan_f3.json`) | 4418,5–4424 s (73,6–73,7 min) | ≈ 36,8 h | `--timeout-s 21600`, `--retries 1` = 2×6 h | Epoche 8/30, Manifest `fehlgeschlagen` |
| D3-3 FlashSR (`models/flashsr/report.json`, Manifest) | 4798,6–4843,3 s (80,0–80,7 min) | ≈ 40,3 h | `--timeout-s 21600`, `--retries 1` = 2×6 h | Epoche 11/30 (Checkpoint 16:52); Versuch 2/2 um 17:29 in Epoche 12 gekillt |

Vier Defekte, alle **nicht** Modellqualität:

1. **Strukturelle Unerreichbarkeit:** Die Kapazität (12 h) ist kleiner als der
   Bedarf (36,8/40,3 h). Der Job kann in der konfigurierten Taktung **nie**
   `ergebnis: erfolgreich` werden — unabhängig von der Modellgüte.
2. **BigVGAN-Erstlauf ohne Anker (neu gefunden):** Der `--cmd` für D3-2 rief
   `train_bigvgan_f3.py --epochs 30` **ohne** `--resume`. Jeder Kettenstart
   begann deshalb bei Epoche 0 und überschrieb `checkpoint_latest.pt` — der Job
   kam über ~8 Epochen nie hinaus, obwohl `checkpoint_epoch8.pt` bereits
   existierte. FlashSR hatte den Datei-Guard seit 2026-11-08, BigVGAN nicht.
3. **Verlust mitten in der Epoche:** Der Runner killt per SIGTERM nach 6 h.
   BigVGAN/FlashSR schreiben den Anker nur am Epochenende ⇒ bis zu **80 min
   GPU-Zeit** gehen je Versuch verloren. FlashSR Versuch 2 wurde um 17:29 in
   Epoche 12 gekillt; Epoche 11 war um 16:52 fertig — **37 min** Fortschritt
   verloren.
4. **Beleg-Verlust beim Resume:** Jeder Resume begann `history` leer und
   überschrieb `report.json`. Der FlashSR-Report enthielt am Ende nur die
   Epochen 8–11 von 30 — das Beleg-Artefakt für das ΔSDR-Gate war unvollständig.

## 2. Entscheidung (autonom, ohne Qualitätsreduktion)

Die 30 Epochen bleiben das Ziel (kein `--epochs`-Eingriff, kein Early-Stop-
Vorziehen). Geändert wird nur die **Taktung und Wiederaufnahme**:

- **12-h-Versuche, 4 Versuche** (`--timeout-s 43200 --retries 3`) ⇒ 48 h
  Kapazität. Das deckt 36,8 h (BigVGAN) bzw. 40,3 h (FlashSR) mit Reserve.
- **BigVGAN-Erstlauf resumt:** `--cmd` prüft jetzt `output/f3_bigvgan/checkpoint_latest.pt`
  und setzt fort, statt bei Epoche 0 zu beginnen (A-SPADE-Muster wie FlashSR).
- **`--max-seconds 37800` (10,5 h)** je Prozess: Der Trainer stoppt **an der
  Epochen-Grenze**, bevor der harte Timeout greift (Exit 75). Die Schätzung
  nutzt die zuletzt gemessene Epochen-Dauer; die erste Epoche läuft immer.
- **Reihenfolge D3-1 → D3-3 → D3-2:** FlashSR ist näher am Ziel (Epoche
  11/30, bestes Val 0,3587 in Epoche 11) und blockiert nicht mehr hinter dem
  37-h-BigVGAN. Genau das war der dokumentierte Grund, Kette 2 nicht
  automatisch zu starten.
- **§G5-RNG-Fortsetzung:** `rng_state` (NumPy-PCG64) und `torch_rng_state`
  liegen jetzt im Checkpoint. Ein Zeitbudget-Stopp setzt die Crop-Folge fort
  statt sie neu zu beginnen — der unterbrochene Lauf folgt derselben
  Zufallsfolge wie ein durchgehender Lauf (Seed 42). **`best_val` liegt
  ebenfalls im Checkpoint:** ein Resume behält den echten Bestwert, statt ihn
  mit dem letzten (oft schlechteren) Val zu überschreiben.
- **Resume-feste Historie:** `report_historie_laden` + `historie_zusammenfuehren`
  führen die Epochen über Versuchs-Grenzen zusammen (eine Zeile je Epoche,
  jüngster Versuch gewinnt; fremder Seed wird verworfen).
- **Randfall fail-closed:** Anker **hinter** dem Ziel (`start_epoch > epochs`)
  ⇒ Exit 3. Anker **auf** dem Ziel (`==`) ⇒ `completed: true`,
  `stop_reason: ziel_epochen_bereits_erreicht`, Exit 0 — ein Kill zwischen
  letztem Checkpoint und Report ist kein No-Op-Erfolg (D-K3-84-Klasse), sondern
  nachweislich fertig.

## 3. Änderungen

| Datei | Änderung |
| --- | --- |
| `backend/core/training_artifacts.py` | `zeitbudget_erreicht`, `report_historie_laden`, `historie_zusammenfuehren` (eine Quelle für alle Trainer) |
| `scripts/train_bigvgan_f3.py` | `--max-seconds`, Budget-Stopp (Exit 75), RNG-Stand im Checkpoint, Historie-Merge, Fail-closed-Anker-Guard, Report atomar via `write_text_guarded` |
| `scripts/train_flashsr_f4.py` | identisch (Checkpoint-Schlüssel `model` statt `generator`) |
| `scripts/gpu_kette2_matrix_d3.sh` | D3-2/D3-3: `--timeout-s 43200 --retries 3`, `--max-seconds 37800` in Erst- und Resume-Befehl; D3-3 vor D3-2; **D3-2-Erstlauf datei-geschützt** (vorher Start bei Epoche 0) |
| `tests/unit/test_gpu_kette1_wiring.py` | 8 neue Tests: Kapazität gegen Messwerte, Budget in allen Befehlen, Reihenfolge, BigVGAN-Anker-Guard, Budget-Helfer, Historie-Merge, Seed-Verwurf, Trainer-Pins |
| `scripts/train_job_runner.py` | **Manifest-Status `zeitbudget`:** Exit 75 (Budget-Stopp) gilt nicht mehr als „fehlgeschlagen“, sondern als gesunder, fortsetzbarer Lauf; nur `erfolgreich` wird beim Ketten-Neustart übersprungen. Log-Zeile entsprechend getrennt. |
| `tests/unit/test_train_job_runner.py` | Neuer Test `test_zeitbudget_stopp_ist_kein_fehlschlag` (Exit 75 ⇒ `ergebnis: zeitbudget`, Resume wird versucht) |

## 4. Verifikation

- `python3 -m pytest tests/unit/test_gpu_kette1_wiring.py -q` — **23/23 grün**
  (15 bestehende + 8 neue).
- `python3 -m ruff check` auf allen vier geänderten Python-Dateien — clean.
- `python3 -m py_compile` — clean.
- RNG-Probe (CPU): `torch.save`/`torch.load(weights_only=True)` trägt
  `rng_state` (PCG64) + `torch_rng_state` (uint8[5056]); Zustands-Restore
  erzeugt dieselbe Zufallszahl.
- Kapazitätsrechnung als Test gepinnt: `(retries+1) * max_seconds ≥ 30 * Epoche_s`
  mit den gemessenen 4424 s / 4843 s.
- `python3 -m pytest tests/unit/test_train_job_runner.py -q` — **9/9 grün**
  (8 bestehende + `test_zeitbudget_stopp_ist_kein_fehlschlag`).

**Nicht verifiziert (ehrlich):** der eigentliche 30-Epochen-Lauf. Der Beweis,
dass BigVGAN/FlashSR mit der neuen Taktung `erfolgreich` enden, ist der nächste
Kette-2-Lauf (GPU). Bis dahin bleibt D-K3-87 `in-arbeit`.

## 5. Grenzen / Risiken

- Die GPU ist im schlimmsten Fall **48 h** durch einen Job belegt. Das ist der
  Preis der bereits geplanten 30 Epochen, keine neue Qualitätsentscheidung.
- Early-Stop kann früher enden (FlashSR: bestes Val 0,3587 in Epoche 11,
  Patience 5); dann ist der Job nach wenigen Epochen fertig, sonst nach ~40 h.
  Epoche 11 verbesserte sich erneut — der Lauf wird voraussichtlich die vollen
  30 Epochen brauchen.
- Ein echter Hänger ohne Epochen-Fortschritt wird weiterhin vom 12-h-Timeout
  beendet (4 Versuche = 48 h Obergrenze). Der Anker begrenzt den Verlust auf
  die laufende Epoche.
- **BigVGAN-Fortschritt akkumuliert jetzt:** Vorher warf jeder Kettenstart die
  ersten ~4 Epochen weg (Start bei Epoche 0). Erst der Datei-Guard macht die
  4×12-h-Kapazität überhaupt nutzbar.
- **Nicht geprüft:** Kette 1 (Cantus) und Kette 3/4 haben andere Timeouts
  (12 h/4 h) und keine belegte 74–80-min-Epoche; dort wurde nichts geändert.
  Sollte ein Cantus-Lauf dieselbe Klasse zeigen, gilt derselbe Fix.

## 6. Offen (nicht autonom)

- **D-K3-13:** UAT-Evidenz älter als 48 h ⇒ `test_daily_gate_stored_recently`
  rot; echter UAT-Lauf ist GPU-gebunden und wartet auf das trainingsfreie
  Fenster.
- **D3-Deploy:** Ein D3-Artefakt zu deployen ist klangwirksam; nur mit
  Never-worsen-A/B gegen den deployten Checkpoint und §4-Verfahren.
- **R6 Gender-Head / BEATs / MERT:** daten-blockiert (Labels/Lizenz).
