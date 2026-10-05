# Offene Maßnahmen — Abarbeitungsliste (Stand 2026-10-04)

> **Zweck:** Verbindliche Abarbeitungsliste für die im Zuge des Ausführungsauftrags
> („Wo Training möglich → Training. Wo nur Ersetzung möglich → Ersetzung. Wo A/B
> möglich → A/B. Wo Ausführung möglich → Ausführung.") verbliebenen Punkte.
>
> **Endzustands-Gebot (Zero-Gap-Regel):** Jede Position endet in GENAU einem der
> drei zulässigen Endzustände — **GESCHLOSSEN** (Artefakt + Beweis), **ERSETZT**
> (Substitut + Paritäts-/Never-worsen-Nachweis) oder **GESTRICHEN** (Sign-off +
> Begründung). „Extern blockiert", „offen", „mal schauen" sind keine Endzustände.
>
> **Beweisführung:** Verifikation je Position ist ausführtbar (Test-/Check-Befehl
> steht dabei). Keine Position wird abgehakt, ohne dass ihr Beweis geführt wurde.

## 0. Ist-Stand (gemessen 2026-10-04, Bash-Beweis in dieser Session)

| Zustand | Messung |
|---|---|
| MUSAN-Download (`data/musan.tar.gz`) | **11.086.114.085 B — vollständig** (mtime 2026-10-04 12:57; kein curl-Prozess mehr aktiv) |
| `data/musan/noise/` (Extraktion) | **930 WAVs / 687 MB — extrahiert** (2026-10-04 12:58) |
| F7-Training | **LÄUFT & trainiert** (Batch-4 UND Batch-2 starben in OOM → **Batch 1** seit 01:50 stabil: Ep 1/200 nachweislich, ~11 min/Epoch, VRAM ≈ 22/24 GB; OOM-Belege `output/train_sgmse_musik_oom_batch4_20261005.log` + `…_oom_batch2_20261005.log`; Fortschritt `output/train_sgmse_musik.log`) |
| Watchdog | Task `8f60bc26-51a3-4058-a436-4246ab3483a8`, alle 6 h (Download-Resume, Extraktion, Training-Start); **2026-10-04 gehärtet:** Start nur bei freiem VRAM ≥ 8 GB; **2026-10-05:** Startkommando final auf `--batch-size 1` + `.venv_aurik`-Python nachgezogen (Batch 4/2 OOM-belegt; Batch 1 läuft stabil); letzter Lauf Status `error` (provider-seitig: 402/SSE-Frame-Limit — extern) |
| Lane-C-Runner (`scripts/eval_scnet_vs_mdx23c.py`) | **erstellt & gelaufen** (Parent-Übernahme 2026-10-05; Kind `child_muucc2nd_s4p56d` ohne Lieferung; Matrix `output/scnet_ab_2026-10-05` — SCNet dominiert Demucs-Baseline) |
| Lane-D-Runner (`scripts/run_t61_ab.py`) | **Akzeptanzmatrix läuft** (`output/t61_ab_2026-10-05`, 20-s-Exzerpte, `AURIK_FORCE_CPU=1`; FORCE_CPU-Nachweis im Log: „… CPU erzwungen (globaler GPU-Stopp)") |
| `models/sgmse_plus/finetuned/sgmse_musik_best.ckpt` | fehlt (erwartet — ist das Trainingsziel) |

Bereits abgeschlossen (nur Kontext, keine Abarbeitung nötig): flow_matching →
`cqtdiff_plus`/diffwave **ERSETZT** (bewiesen: `check_core_model_sources` „ERSETZT",
Exit 0; `test_hybrid_release_mode` 14 grün) · miipher → MIIPHER-DiT **ERSETZT**
(Manifest `replaced_by`, §v10.14) · Gewichte-Beschaffung (`inpainting_best.pt` SHA
`f8765d14…` = Pin; `MERT-v1-330M_fairseq.pt` SHA `13d9b884…` = Pin; MERT-HF-Set) ·
Demucs-v5-Klärschlag (existiert nie; `htdemucs_6s.onnx` = v4).

---

## A. Laufende Maßnahmen (automatisiert / asynchron)

### M1 · Lane A: F7-SGMSE-Musik-Finetune → `sgmse_musik.ts` — TRAINING
- [x] **A1** MUSAN-Download abschließen (`data/musan.tar.gz`, Quelle `https://www.openslr.org/resources/17/musan.tar.gz`; Watchdog resumed per `curl -C -`). **EVIDENZ:** 11.086.114.085 B, mtime 2026-10-04 12:57, kein curl-Prozess.
- [x] **A2** `musan/noise/*` nach `data/musan/` extrahieren (Watchdog; nur Noise-Subset). **EVIDENZ:** 930 WAVs / 687 MB unter `data/musan/noise/` (2026-10-04 12:58).
- [ ] **A3** Finaler Trainingslauf (Watchdog startet ihn, sobald A2 erfüllt und kein Prozess läuft):
      `python3 -u -B scripts/train_sgmse_musik.py --epochs 200 --batch-size 4 --lr 3e-5 --steps-per-epoch 200 --seed 42 --out-dir models/sgmse_plus/finetuned [--resume models/sgmse_plus/finetuned/checkpoint_latest.ckpt]`
      (Datenvertrag §v10.16: 50 % synthetisch / 50 % MUSAN; Laufzeit ca. 3–7 Tage GPU.)
      **Gate 2026-10-04:** Start nur bei freiem VRAM ≥ 8 GB — das Cantus-Pretrain belegt aktuell ≈ 18/24 GB (Doppel-Training-Risiko); Watchdog prüft das seit 2026-10-04.
      **STATUS 2026-10-05:** Batch 4 UND Batch 2 → OOM auf 24 GB (Belege `output/train_sgmse_musik_oom_batch4_20261005.log`, `…_oom_batch2_20261005.log`); **Batch 1 läuft stabil** (Start 01:50; 4-s-Datenvertrag bleibt; gemessen Ep 1: ~11 min/Epoch ⇒ ~1,5 Tage für 200 Epochen; VRAM ≈ 22/24 GB); Checkpoints je Epoche → `models/sgmse_plus/finetuned/checkpoint_latest.ckpt`.
- [ ] **A4** TorchScript-Export `sgmse_musik_best.ckpt` → `models/sgmse_plus/finetuned/sgmse_musik.ts`
      (analog zur Herstellung von `sgmse_plus.ts`, Geometrie n_fft=510/hop=128 wie `plugins/sgmse_plugin.py`) + SHA-256 in `models/manifest.json` (Eintrag `sgmse_musik`) eintragen.
- [ ] **A5** Abnahme: Parität/Never-worsen vs. `sgmse_plus.ts` (§V6: nie schlechter als der Ausgangspfad), dann **Sign-off** für Flag-Flip `use_sgmse_musik=True` (`backend/core/music_model_flags.py:25`) + `plugins/sgmse_plugin.py:_SGMSE_TS_PATH` auf `sgmse_musik.ts` (§v10.19). ⚠ Flag-Flip erst nach menschlichem Sign-off.
- **Verifikation:** Manifest-SHA gesetzt · `pytest tests/normative/test_primary_paths_no_fallback.py --run-heavy-tests` (15 grün bleibt) · Laufprotokoll `output/train_sgmse_musik.log`.
- **Endzustand:** GESCHLOSSEN (mit Flag) ODER GESTRICHEN (Sign-off, falls Finetune das Never-worsen-Gate verfehlt).

### M2 · Lane C: P1-2 A/B — SCNet vs MDX23C (Child `child_mut9x9hg_zx2yti`, läuft)
- [x] **C1** Lieferung erstellt (Parent-Übernahme, Kind `child_muucc2nd_s4p56d` lieferte keine Artefakte): `scripts/eval_scnet_vs_mdx23c.py` + Report `docs/reports/current/2026-10-04_p1_2_scnet_vs_mdx23c_ab.md`. **Vollmatrix 3×30 s** (`output/scnet_ab_2026-10-05/`, auto-Vocals-Fenster 165/169/107 s, Seed 42, CPU): SCNet **SI-SDR +10,3…+16,4 dB · Singer-cos 0,978/0,992/0,988 · Fidelity 0,946–0,967** vs. Baseline Demucs v4 (MDX23C fehlt §v10.73): −20,1…+1,5 dB · 0,45–0,63 · 0,0–0,271. Zusatzbefund: MelBandRoformer-ONNX-Pfad liefert duplizierte Instrument-Stems (Residual-Skalierungen 0,40/0,30/0,15/0,10/0,05) und unkorrelierte Vocals — nicht als Baseline verwendet.
      (3 MUSDB-Test-Songs „AM Contra – Heart Peripheral", „Al James – Schoolboy Facination", „Motor Tapes – Shore"; Metriken SI-SDR, singer_identity_cosine, separation_fidelity; CPU, §G5-Seeds).
- [x] **C2** Parent-Integration: FILE_REGISTRY-Einträge gesetzt (Runner + vendored `zfturbo_scnet`); Ledger-Snapshot 2026-10-05 (siehe `TASK_CHANGES.md`).
- [x] **C3** Verdikt in `docs/TODOS_SOTA_ROADMAP.md` (Zeile „TODO-P1-2") eingehängt — Statusblock 2026-10-05 mit Matrix, Gates und MelBand-Befund.
- [ ] **C4** **Hörstichprobe (menschlich)** + Sign-off: SCNet-Kandidat `models/scnet_4stems/huge_scnet_4stems_v1.2.ckpt` (SHA `807f470b…`, Apache-2.0) integrieren oder verwerfen. Schwellen: separation_fidelity ≥ 0,80/0,83 · singer_identity_cosine ≥ 0,92 (Never-Below-Gate, Hörordnung §62).
- [ ] **C5** VS-1/GSEP-Ziel **final entscheiden** (Sign-off): als ERSETZT (durch SCNet-Rollenübernahme) oder GESTRICHEN (keine offiziellen lizenzklaren Weights auffindbar) eintragen — kein „extern blockiert" stehen lassen.
- **Verifikation:** Report-Tabelle + `validate_core_model_presence.py` unverändert runtime_ready · Norm-Gate `test_hybrid_release_mode` 14 grün.
- **Falls das Child dauerhaft ausfällt** (Retries verbraucht): Auftragstext M2 liegt in dieser Session; Lauf ist ohne Produktions-Änderungen reproduzierbar (CPU, feste Songs).

### M3 · Lane D: T6-1 A/B-Abnahme (Child `child_mut9x9hh_vtuqrz`, läuft)
- [ ] **D1** Matrix-Lauf über die 3 Songs (Stand-in MUSDB-Test-Trio, im Report als Ersatz dokumentiert) → CSV `config,score_ref,score_var`; Runner `scripts/run_t61_ab.py` existiert (Umschaltmethode Monkeypatch, keine Produktions-Änderung). (STATUS 2026-10-05: Akzeptanzmatrix läuft `output/t61_ab_2026-10-05` — 20-s-Exzerpte (Auszugslänge im Abnahme-Report dokumentiert), `AURIK_FORCE_CPU=1`; ONNX-ROCm-Abbruch-Ursache behoben: `get_onnx_providers`/`apply_gpu_policy` erzwingen jetzt CPU — Log-Nachweis „§v10.40c … CPU erzwungen".)
- [ ] **D2** `python3 scripts/validate_t61_ab.py <matrix.csv>` — prä-registrierte Regel (Gewinn ≥ min_delta + Bootstrap-95 %-CI ohne 0 + Hörordnung-Ebene-1-Veto); JSON-Verdikt in den Report.
- [ ] **D3** Report `docs/reports/current/2026-10-04_t61_abnahme.md` (Per-Song-Kernziel-Deltas: spatial_depth, natuerlichkeit, timbre_authentizitaet, harm …; Never-worsen-Log).
- [ ] **D4** Parent-Integration: FILE_REGISTRY + Ledger + Roadmap „Rest (A/B-Abnahme)" abhaken → **T6-1 GESCHLOSSEN**.
- [ ] **D5** **Hörordnungs-Abnahme (menschlich)** + Sign-off Optionsvertrag A.
- [ ] **D6** (optional) Nachlauf mit den echten „3 Referenz-Songs", sobald sie namentlich benannt sind — dann Matrix erneut fahren, Entscheidung ggf. revidieren.

---

## B. Braucht menschliche Entscheidung (Sign-off — Termine/Urteil)

- [ ] **B1** Hörstichprobe (C4) + Hörordnungs-Abnahme (D5) terminieren.
- [ ] **B2** MUSHRA-Studie n ≥ 30 (Status „Vorbereitung", `docs/reports/studies/mushra_2026q2/` inkl. preregistration): **entweder** Studie terminieren **oder** formal beschließen, dass die Zwischenabnahme über Objektivmetriken läuft und die Studie nachgelagert ist — beides ist ein gültiger Endzustand, „extern blockiert (Hörer)" nicht.
- [ ] **B3** F11 BW-Reconstructor v5: Re-Training freigeben (A1-Gate 0,73 < 1,02 nicht bestanden) — GPU-Lauf, danach Gate erneut.
- [ ] **B4** F8 UTMOSv2-Musik-Orakel priorisieren (höchster Wohlklang-Oracle-Hebel; Gewichte vorhanden: `models/utmosv2/fold0_s42_best_model.pth`, SHA `c8149d98…`): Anbindung/Kalibrierung als Musik-MOS statt Sprach-MOS (BVCC).

---

## C. Rechenbar, noch nicht gestartet (Ausführungs-Lanes)

- [ ] **E1** F12 BANQUET Real-Vinyl-Finetune: **Lauf starten** (`scripts/train_banquet_vinyl_finetune.py`; Code + 10 Tests fertig). VORBEDINGUNG prüfen: reale Vinyl↔Reissue-Paare im Trainingsdata-Pfad vorhanden; endet mit Report, KEIN automatischer Flag-Flip.
- [ ] **E2** F4 FlashSR HF-Finetune (> 12,9 kHz): Fortsetzung prüfen (Roadmap: Epoche 6→14) und zu Ende trainieren.
- [ ] **E3** F3 BigVGAN-v2-Finetune (GPU-gebunden) — nach M1/E1 in die GPU-Reihenfolge einordnen.

---

## Empfohlene Abarbeitungsreihenfolge

1. **sofort, läuft schon:** M2, M3 (Children) + M1 A1/A2 (Watchdog) — nichts zu tun als abwarten/prüfen.
2. **nach M2/M3:** C2–C5, D4–D6 (Integration + die zwei menschlichen Abnahmen) → P1-2 und T6-1 sind dann endgültig entschieden.
3. **GPU-Kaskade:** A3/A4/A5 (läuft automatisch) → E1 → E2 → E3 → B3.
4. **jederzeit per Termin:** B1, B2 (brauchen nur eure Entscheidung, keine Rechenzeit).

## Quellen / Belege

- Roadmap: `docs/TODOS_SOTA_ROADMAP.md` (T6-1 Zeilen ~86–131, P1-2 ~224–252, Abschluss-Matrix ~1898 ff.)
- F7-Rezept: `.github/specs/v10.16_sgmse_musik_finetune.md`, `scripts/train_sgmse_musik.py`, `scripts/prepare_sgmse_musik_data.py`
- Ersetzungsnachweise: `scripts/check_core_model_sources.py` („ERSETZT", Exit 0), `tests/normative/test_hybrid_release_mode.py` (14 grün), `models/manifest.json` (`replaced_by`)
- Entscheidungs-Harness: `scripts/validate_t61_ab.py` + `tests/unit/test_validate_t61_ab.py` (7 grün)
- Watchdog: Kun-Scheduled-Task `8f60bc26-51a3-4058-a436-4246ab3483a8`

---

## D. Erledigt 2026-10-04 abends (Ausführungssitzung)

- [x] **Hörordnungs-Kanonisierung angewandt** (`.github/instructions/hoerordnung.instructions.md`):
      §3-Umsetzung (Level-1-Guard-Blend + §SCK-R/§WBG-R), §4-Instanzen
      (`audibility_targets`, Golden-Ear/Hörpanel) und §8-Tabelle (Zeilen 1/2) —
      Entwurfsnamen durch verifizierte Implementierungen ersetzt (Details im
      2026-10-03-Report). Prüfungen: Kalibrierung 14/14, kein Spec-Drift,
      ID-Registry 0 Warnungen.
- [x] **Evidenz-Generator-Defekt behoben** (`scripts/run_real_audio_corpus_test.py`):
      echte UV3-Pipeline (`QualityMode.QUALITY`), ehrliche Fehler-/Exit-Semantik
      (§V6), Korpus-Paarung `<base>_clean`, `samples` = Elementzahl; Smoke-Lauf
      dokumentiert (Reverb-Paar, CPU-erzwungen).
- [x] **FILE_REGISTRY R6-Warnungen behoben** (Backtick-/`**init**`-Mangling bei
      `models/cantus/__init__.py` + `models/symphonia/__init__.py`):
      `scripts/file_registry_check.py` → 0 ERROR / 0 WARNUNG.
- [x] **F7-Watchdog gehärtet:** Start nur bei freiem VRAM ≥ 8 GB (verhindert
      Doppel-Training/OOM, solange Cantus-Pretrain läuft).
- [x] **ORT-ROCm/FORCE_CPU-Lücke geschlossen** (`backend/core/gpu_model_registry.py`):
      `AURIK_FORCE_CPU=1` erzwang bei direkten Registry-Aufrufern (bigvgan/flashsr,
      `get_onnx_providers`/`apply_gpu_policy`) keine CPU — unter GPU-Last endete das
      in einem ORT-Hart-Abort (`no ROCm-capable device is detected`). Fix + 2
      Regressionstests (`tests/unit/test_gpu_model_registry.py`, 18/18 grün);
      Nachweis im T6-1-Log „§v10.40c … CPU erzwungen (globaler GPU-Stopp)".
- [x] **Corpus-Smoke gemessen** (Evidenz-Generator): Reverb-Paar → MUSHRA
      **75,6/100 (Fair)**, NSIM 0,651, `error_cases: 0`; JSON/Log im
      2026-10-03-Report verlinkt.
- [x] **F7-Training gestartet und stabil** (Batch 1 nach Batch-4/2-OOM; Ep-1-Schritte belegt, ~11 min/Epoch) und
      **T6-1-Akzeptanzmatrix gestartet** (`output/t61_ab_2026-10-05`).
- [x] **M2/Vollmatrix + Baseline-Klärung** (2026-10-05, s. C1): Parent-Übernahme nach Kind-Ausfall;
      SCNet-Kandidat strikt geladen (0/0, 65.292.464 Params), Demucs-v4-Baseline (ONNX) verifiziert,
      MelBand-ONNX-Defekt belegt; Vollmatrix + Hör-Artefakte (32-bit-Float) für C4 erzeugt.
