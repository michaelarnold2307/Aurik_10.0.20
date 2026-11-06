# WF-V4-Echt-Pfad verdrahtet + D-K0-9 geschlossen — 2026-11-06

**Auftrag (Wiederaufnahme):** §2 des `docs/SOTA_HANDOVER.md` abarbeiten, ohne
laufende GPU-Ketten zu stören; keine Flag-Flips, keine Hör-/Lizenz-Entscheidung
eigenmächtig.

**Zustand bei Sitzungsbeginn (Pflicht-Check):** `pgrep -f gpu_kette` →
`gpu_kette2_matrix_d3.sh` **läuft** (D3-1 `d3_diffwave_vokal`, GPU 70 % / VRAM
43 %). Die D3-Verifikation (RT-Messung + Qualitäts-Gate) braucht laut
`docs/reports/current/2026-10-28_f_reihe_laufzeit_optimierung.md` §9 ein
**trainingsfreies Fenster** ⇒ in dieser Sitzung **begründet offen**. Kein
Training gestartet, keine Kette angefasst.

---

## 1. Befund: drei Phantom-Verdrahtungen im WF-V4/TP-V2-Echt-Pfad

Ausgangspunkt war der Manager-Ausführungsschritt 5 aus
`docs/reports/current/2026-11-06_trainingswelle_modelle.md` („Kette 4 starten“).
Die Prüfung der Kette vor dem Start ergab, dass **R1 und R2 niemals auf
Musik trainiert hätten** — die Kette wäre „grün“ durchgelaufen und hätte das
Gegenteil ihres Etiketts getan (§III.11 (copilot-instructions.md)):

| # | Befund | Wirkung ohne Fix | Beleg |
| --- | --- | --- | --- |
| 1 | `train_wf4_warp_head.py` hatte **kein `--real-dir`** (argparse-Liste, `manifest["real_dir"]` fest `None`); trainiert ausschließlich aus `make_warp_curve`/`make_degraded` (synthetische Harmonische) | R1 hätte trotz MUSDB-Paaren auf synthetischen Sinus-Signalen trainiert; die Docstring-Behauptung „konsumiert die Paare über `--real-dir`“ war unwahr | `scripts/train_wf4_warp_head.py` (vor Fix), Docstring `scripts/prepare_wf4_warp_pairs.py` |
| 2 | `gpu_kette4_modellkoepfe.sh` prüfte `[ -d data/wf4_warp_pairs/degraded ]` — das Verzeichnis heißt aber `audio/` | R1 wäre **immer** übersprungen worden (stiller Skip) | `scripts/gpu_kette4_modellkoepfe.sh` (vor Fix) |
| 3 | `train_tp_v2_phase.py`: `--synthetic` (Default **16**) überstimmte ein gesetztes `--real-dir` still (`if self._synthetic > 0 or not self._real_paths`) | R2 hätte mit `--real-dir data/tp2_transient_pairs` auf 16 synthetischen Segmenten trainiert | `scripts/train_tp_v2_phase.py:166` (vor Fix) |

Zusätzlich: `prepare_wf4_warp_pairs.py` schrieb **keine** Tracker-Beobachtung
(`x_f0`), also fehlte der Modelleingang für Echt-Training vollständig.

## 2. Umsetzung

### 2.1 Datensatz: F0-Beobachtung statt Zirkelschluss (`prepare_wf4_warp_pairs.py`)

- Neu `beobachte_f0(clean, degraded, n_frames)` — `x_f0 = log2(f0_degradiert /
  f0_clean)` mit dem **Produktions-Tracker**
  `backend.core.dsp.pyin_viterbi_fast.pyin_fast` (bit-identisch zu librosa 0.11.0,
  CPU, deterministisch). Die GT-Warp-Kurve wird **nicht** verwendet — sonst wäre
  der Trainings-Eingang ein Zirkelschluss (§V6).
- Ausreißer-Regel: nicht beidseitig stimmhaft oder |Δ| > `F0_MAX_ABWEICHUNG_OCT`
  (0,5 log2 = 600 Cent) ⇒ `x_f0 = 0`, `x_conf = 0` („keine Beobachtung“).
- Neue Ausgaben `f0/<name>.npy`, `conf/<name>.npy` + `sha256_f0`/`sha256_conf`
  je Paar; Manifest-Feld `f0_beobachtung` (Tracker, fmin/fmax, frame_length,
  hop_length, max_abweichung_oct). CLI-Schalter `--ohne-f0` (nur für
  Nicht-Trainings-Zwecke; der Trainer lehnt solche Paare ab).
- `sys.path`-Bootstrap ergänzt (Backend-Import aus `scripts/`).

### 2.2 Trainer: echter, fail-closed Real-Pfad (`train_wf4_warp_head.py`)

- Neuer Lader `lade_real_paare(real_dir)`: prüft Manifest, Schema
  `aurik.wf4_warp_pairs/1`, Vorhandensein von `f0_beobachtung`, jede einzelne
  Datei und **einheitliche Frame-Anzahl**; bricht sonst ab (`FileNotFoundError`/
  `ValueError`) — **kein** stiller Rückfall auf synthetisch (§V6).
- Batches: deterministische Permutation je `(epoch, step)` aus
  `_seed_for(seed, "batch", epoch, step)` (§G5), kein Zeitbezug.
- Stereo-Layout-Invariante (AGENTS.md §3): `sf.read(..., always_2d=True)` →
  expliziter Layout-Check, Mono nur über `y[:, 0]` bzw. `mean(axis=1)`.
- Manifest führt jetzt `real_dir`, `real_pairs`, `material: "musik"|"synthetisch"`.
  Der synthetische Pfad bleibt, loggt aber ausdrücklich als Ersatzpfad.

### 2.3 TP-V2: Vorrang der echten Daten (`train_tp_v2_phase.py`)

`TransientPhaseDataset` setzt bei vorhandenen realen `*.wav` `_synthetic = 0` und
warnt laut, wenn `--synthetic > 0` ignoriert wurde; ein leeres `--real-dir`
fällt **mit Warnung** auf synthetisch zurück (§V6). Kette 4 übergibt zusätzlich
explizit `--synthetic 0`.

### 2.4 Kette 4 (`gpu_kette4_modellkoepfe.sh`)

- R1: Guard auf `data/wf4_warp_pairs/manifest.json` **und** `f0/`, Aufruf mit
  `--real-dir data/wf4_warp_pairs`; fehlt etwas ⇒ Job wird **übersprungen** mit
  Hinweis auf den Synthesebefehl (fail-closed statt Scheintraining).
- R2: Guard verlangt mindestens ein `*.wav`, Aufruf mit `--synthetic 0`.

## 3. Zweiter Befund: WAV-Container ist byte-instabil (Determinismus §G5)

Zwei identische Synthese-Läufe (Seed 42) erzeugten **bit-identische Audiodaten**,
aber **eine Byte-Differenz** im WAV-Container: libsndfile schreibt einen
`PEAK`-Chunk, dessen Float-Wert im letzten Bit abweicht (gemessen: 1 Byte bei Offset 61, `95 34 ee 6a` vs. `9a 34 ee 6a`). Konsequenz: `sha256_wav` ist als
Provenienzwert **nicht reproduzierbar**.

**Fix:** verbindlicher `sha256_pcm` (Hash der float32-PCM-Daten) je Paar;
`sha256_wav` bleibt informativ, das Manifest dokumentiert die Semantik
(`provenienz`-Feld). Der Regressionstest
`test_erzeuge_paare_deterministisch_ueber_pcm` pinnt das: zwei Läufe müssen in
`sha256_pcm`, `sha256_npy`, `sha256_f0`, `sha256_conf` übereinstimmen.

## 4. Ausgeführte Manager-Schritte (Belege)

| Schritt | Ergebnis |
| --- | --- |
| `bash -n scripts/gpu_kette4_modellkoepfe.sh` | OK |
| `python3 -m py_compile` (3 Skripte) | OK |
| `train_job_runner.py --help` | OK (Argparse-Vertrag `--name --ckpt-dir --cmd --resume-cmd --phase`) |
| Datensynthese WF-V4 | **300 Paare** `data/wf4_warp_pairs/` (audio/warp/f0/conf, Seed 42); `f0_beobachtet_anteil` **Median 0,45** (min 0,00 / max 1,00) — polyphone MUSDB-Mixturen, der Tracker ist dort erwartungsgemäß selektiv; jedes Paar trägt `sha256_pcm` |
| Datensynthese TP-V2 | **200 Segmente** `data/tp2_transient_pairs/` (Seed 42, Manifest) |
| Lader-Probe (Produktionspfad) | `lade_real_paare` → 300 Paare je 500 Frames; `x_f0 (500,)`, `x_conf (500,)`, `x_feat (500, 8)`, `y_warp (500,)`, `y_conf (500,)` |
| Tests | `tests/unit/test_wf4_warp_real_pairs.py` + `tests/unit/test_train_tp_v2_real_dir_vorrang.py` → **13 passed** |

**Nicht ausgeführt (bewusst):** R1/R2-Trainingslauf (GPU durch Kette 2 belegt,
§5 „keine parallelen Trainings“); D3-Verifikation (trainingsfreies Fenster
nötig); Kette 4 Start (pgrep-Guard: `train_job_runner` aktiv).

## 5. D-K0-9 geschlossen — Buchführung einer bereits gefallenen Entscheidung

Das Register führte D-K0-9 (EAR-VAE v1 ↔ v2) weiter als „offen
(Rollout-/Ersatz-Entscheidung offen)“. Die Entscheidung ist jedoch am
**2026-11-05** von der Hör-Instanz gefallen:

- objektive Blind-Messung (seed 42, 17 Triale): never_worsen ✓, 16/17
  v2-vorteilhaft, Vorzeichen-Test **p = 0,000275** (signifikant **pro v2**) —
  `docs/reports/current/2026-11-03_ear_vae_v1_v2_vergleich.json`
- menschliches Blind-Urteil: **12:3 für v1** (p = 0,035; „v1 dumpfer/sauberer,
  v2 heller/rauschiger“) — `docs/reports/current/2026-11-05_hoer_signoff.md`
  (Block C + Entblindung), Commit `7d2e3441`
- Ergebnis: **NO-GO v2, v1 bleibt** (Hörordnung §7: Metriken sind Zeugen, die
  Hör-Instanz entscheidet); Wunsch-Richtung künftiger Iterationen: heller **und**
  weniger Rauschen.

Diese Sitzung hat **keine** Entscheidung getroffen, sondern den Entscheid mit
beiden Belegen in Register (§Status `geschlossen`), Roadmap (3 Stellen,
`SOTA-M2` auf „teilweise“) und §2 des Handovers nachgetragen. `sota_deficit_gate.py`
bleibt grün (99 Einträge, keine Verstöße).

## 6. Offen (nicht autonom schließbar in dieser Sitzung)

1. **W-3-Neumessung** über §4 mit Vertrags-Referenz-Songs — braucht GPU
   (Produktions-Restore-Pfad), Kette 2 läuft.
2. **D3-Verifikation** (RT + Qualitäts-Gate) — trainingsfreies Fenster.
3. **R1/R2-Trainingslauf** — startet über Kette 4, sobald `train_job_runner`
   leer ist; Daten und Verdrahtung liegen bereit.
4. **F9/F10/D-K2-3-Trainer fehlen weiterhin** (`train_voiceencoder_finetune.py`,
   `calibrate_singmos_music.py`, `train_vocos_48khz_finetune.py`) — R3/R4/R5
   bleiben dadurch inert; das ist **code-offen**, nicht daten-blockiert.

## 7. Reproduktion

```
# Datensätze (deterministisch, Seed 42)
env PYTHONPATH=. python3 -B scripts/prepare_wf4_warp_pairs.py \
    --musdb-root data/musdb18hq --out-dir data/wf4_warp_pairs --anzahl 300 --seed 42
python3 -B scripts/prepare_tp2_transient_pairs.py \
    --musdb-root data/musdb18hq --out-dir data/tp2_transient_pairs --seed 42

# Vertragstests
.venv_aurik/bin/python -m pytest tests/unit/test_wf4_warp_real_pairs.py \
    tests/unit/test_train_tp_v2_real_dir_vorrang.py -q

# Register-Gate
python3 scripts/sota_deficit_gate.py
```
