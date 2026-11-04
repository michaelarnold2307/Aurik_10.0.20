# Sitzung 2026-11-04 (2. Hälfte) — Transient-Mess-Lauf fortgesetzt + B3-Bestand

> **Auftrag:** Autonomer Lückenschluss nach `docs/SOTA_HANDOVER.md` (§2
> priorisiert), Pflicht-Wiederherstellungs-Check, „Fallback-/Mess-Läufe
> fortsetzen wo sie endeten“, AGENTS.md-Regeln, alle Pre-Commit-Gates grün,
> Conventional Commit. Hör-/Lizenz-Entscheidungen nie eigenmächtig (§5
> SOTA_HANDOVER). Quelle je Zeile (Zitierdisziplin `AGENTS.md` §5).

## 1. Wiederherstellungs-Check (Pflicht, SOTA_HANDOVER §3)

| Prüfung | Befund | Beleg |
| --- | --- | --- |
| `pgrep -f gpu_kette` | Kette 2 läuft (PID 3560120, `bash scripts/gpu_kette2_matrix_d3.sh`, seit 2026-11-03 21:55) | Prozessliste, `ps -o pid,etime,cmd` |
| Trainings-Runden offen? | `d3_diffwave_vokal` **erfolgreich** (Gate bestanden, Checkpoint `diffwave_vocal_ft.ckpt` aktiviert; Val-SDR +3,23 dB vs Zero-Shot −3,01 dB, `never_worsen_per_gap: true`) | `docs/reports/current/train_job_d3_diffwave_vokal.json`, `2026-11-04_diffwave_vocal_finetune.json` |
| Nächster Job | `d3_bigvgan_f3` läuft (run_id `matrix_d3-20261104T092240Z-b734d7`, `versuche: []`) | `docs/reports/current/train_job_d3_bigvgan_f3.json` |
| Konsequenz | **Keine Ketten-Neustarts** (Regel a greift nicht — Kette läuft), **kein paralleles Training** (§5); Mess-Lauf CPU-only (`python3` = torch 2.7.0+cpu, `torch.cuda.is_available() == False`) mit `nice -n 19` (host-sicher, Report `2026-11-04_transient_energie_ursache_fix.md` §7) | diese Sitzung |

## 2. Gewählte Lücke und Begründung

Aus `docs/SOTA_HANDOVER.md` §2 sind die Punkte 1, 2, 3, 5 erledigt; **#4
(F-Reihe 53×→32× RT) ist GPU-gebunden** (RT-Messung erst im nächsten
trainingsfreien Fenster, Verifikation nach D3-Lauf — Kette läuft); **#6
(W-1…W-13)** ist bis auf die menschlichen Hör-Sessions und gesperrte Stufen
nicht autonom weiterzubringen (siehe §5); **#7** extern-/daten-blockiert,
**#8** menschlich. Die höchste autonom abarbeitbare Rest-Arbeit ist daher
Wiederherstellungsschritt (b): **der im Report
`2026-11-04_transient_energie_ursache_fix.md` §7/§9.3 offen verbliebene
Never-worsen-Mess-Lauf** (`b3_transient_energie_neverworsen.py run --label neu`)
— er endete zuvor an einem Harness-Defekt (siehe §3) und war nie gelaufen
(keine `b3_transient_energie_*.json` in `docs/reports/current/`).

## 3. Umgesetzt: Layout-Fix der Mess-Rate-Normalisierung (§V7 (copilot-instructions.md): Ursache statt Symptom)

**Symptom des vorherigen Lauf-Abbruchs:** `ValueError: Signal zu kurz für
MusikMOS (mindestens 2048 Samples)` in `_bewerte` nach Warnung
`Wohlklang-Garantie: … len=110250→3` (Log `/tmp/b3_transient_neu.log`,
Laufversuch 2026-11-04).

**Ursache (statisch + Repro belegt):** Die Resampelung 44,1 → 48 kHz
(Report `2026-11-04_transient_energie_ursache_fix.md` §7, Nachtrag) nutzte
`resample_poly(..., axis=-1)`. `_load_clip` (`scripts/release_candidate_pipeline.py`)
liefert aber **`(N, C)` channels-last** (gemessen: `shape=(110250, 2),
sr=44100`) — das Resample traf die **Kanal-Achse** und erzeugte 3
Schein-Kanäle (`ceil(2·160/147)=3`); die Pipeline erhielt ein defektes
Array, der Restore kollabierte (`len=110250→3`). Genau die Falle der
**Stereo-Layout-Invariante** (`AGENTS.md` §3: „`audio[:, 0]`/`mean(axis=0)`
ohne Layout-Check kollabiert“).

**Fix (§V7 (copilot-instructions.md) — Ursache statt Symptom):** `scripts/b3_transient_energie_neverworsen.py`
`_auf_messrate_48k` bestimmt die Sample-Achse layout-sicher
(`(C, N)` ⇒ Achse 1; `(N,)`/`(N, C)` ⇒ Achse 0) und resampelt deterministisch
(`resample_poly`, §G5 (copilot-instructions.md)) auf 48 kHz
(01_musical_goals.md §1.4.6: Mess-Rate 48 kHz). Nach dem Fix:
`(110250, 2)@44,1 kHz → (120000, 2)@48 kHz` (Repro, diese Sitzung).

**Tests ergänzt** (`tests/unit/test_transient_energy_messkontrakt.py`):
`test_messfenster_resampeln_deterministisch_auf_48k` (Länge ±1 Sample,
Determinismus §G5 (copilot-instructions.md), Passthrough) und
`test_resample_trifft_die_sample_achse_in_beiden_stereo_layouts`
(Kanalzahl erhalten in `(N, C)` **und** `(C, N)`, Sample-Achse resampelt).
Ergebnis: **24/24 grün** (`pytest tests/unit/test_transient_energy_metric.py
tests/unit/test_transient_energy_messkontrakt.py`), `ruff --select
F821,F601,B009,I001` **sauber** über die Report-§7-Dateiliste.

## 4. Mess-Lauf `run --label neu` — Ergebnis (§9.3 Zielwert-Frage)

**Lauf:** `nice -n 19 python3 -B scripts/b3_transient_energie_neverworsen.py
run --label neu` (2026-11-04, Laufzeit ~42 min, Exit 0; Log
`/tmp/b3_transient_neu.log`). Ausgabe:
`docs/reports/current/b3_transient_energie_neu.json` (Schema
`aurik.transient_energie.messung/1`, seed 42, **6 Fenster** aus 3
Corpus-Clips — Fensterplan Starts 0,0/2,5/5,0/7,5 s; `_load_clip` verwirft
Fenster jenseits der Clip-Länge per ValueError, daher 4+1+1 statt 12).

| Kennzahl | Wert | Beleg |
| --- | --- | --- |
| B1 ungeplante ML→DSP-Rückfälle | **0** | `b3_transient_energie_neu.json` `defekt_fallbacks` |
| transient_neu (korrigierter Vertrag) — Mittel | **0,8589** | Zeilen `transient_neu.score` |
| transient_alt (alter Vertrag) — Mittel | 0,7705 | Zeilen `transient_alt.score` |
| Δ (neu − alt), Doppelmessung je Fenster auf selbem Audio | **+0,0885** | `b3_transient_energie_neu.json` |
| Never-worsen der Vertragsänderung (JND 0,05) | **✓** — min Δ = +0,0000; 4/6 Fenster besser (+0,07…+0,18), 2/6 identisch (Score-Decke 1,0) | ebd. |
| Zielkorridor 0,70 (`backend/core/calibration_matrix.py`) | **erreicht** — Mittel 0,86; **5/6 Fenster** ≥ 0,70 | ebd. |

**Antwort §9.3 (Zielwert-Frage):** Der Zielkorridor 0,70 wird mit dem
korrigierten Vertrag **erreicht**. Der Korrektur-Effekt tritt genau dort auf,
wo der alte Vertrag Wow-/Flutter-Timing-Korrekturen und isolierte
Defekt-Impulse als Energieverlust wertete (Fenster `cassette_1980s_wow__w2/w3`:
+0,18/+0,18; `reel_1940s_dropout__w4`: +0,11; `jazz_1950s_scratched__w5`:
+0,07). Der einzige Einzelfall unter 0,70 (`jazz_1950s_scratched__w5`, 0,6335)
ist damit **echter Signalfaden statt Messartefakt** und verbleibt als
U4-Kandidat (§7) — kein Stärke-Pflaster (§V7 (copilot-instructions.md)).

**Zum Harness-`compare`-Modus (bewusst nicht gelaufen):** Die
`_kontrakt_doppelmessung` misst beide Verträge je Fenster auf selbem Audio —
die Wirkung der §1.4.6-Änderung ist damit im `neu`-Lauf vollständig belegt.
`compare --alt … --neu …` verlangt zwei Laufdateien **verschiedener
Code-Stände** (vor/nach Messkontrakt-Fix); ein echter `alt`-Lauf existiert
nicht (die Messung war vor dieser Sitzung nie gelaufen) — ein zweiter Lauf
mit demselben Code würde eine triviale 0-Differenz-„Bestätigung“ erzeugen
(§V7 (copilot-instructions.md): keine erfundenen Zahlen). Die
Vergleichsbasis 2026-11-03 bleibt als `BASELINE_2026_11_03` im Harness
hinterlegt (Docstring `scripts/b3_transient_energie_neverworsen.py`).

## 5. B3-Bestand W-1…W-13 (Lücke #6) — ehrliche Bestandsaufnahme

Gemessen an `docs/reports/current/b3_objektivbeschluss_W-*.json` (Schema
`aurik.objektivbeschluss/1`) und `.github/WOHLKLANG_CLAIMS.md`:

| Stufe | Beschluss | Never-worsen | Signifikant | Defekt-Fallbacks | Status/Blocker |
| --- | --- | --- | --- | --- | --- |
| W-1 HR-V1 | ABGELEHNT (`b3_objektivbeschluss_W-1.json`) | ✗ (verfehlt) | ✗ (2/5, p=1,0) | 0 | widersprüchliche Objektiv-Belege (HF-Gain-Beschluss BESTÄTIGT vs. musik_mos-Beschluss ABGELEHNT) → **Hör-Instanz entscheidet** (Hörordnung §7) |
| W-4 MP-SENet | ABGELEHNT (`b3_objektivbeschluss_W-4.json`) | ✓ | ✗ (2/5, p=1,0) | 0 | Hör-Session vorbereitet = **menschlich offen** |
| W-5 DFN | ABGELEHNT (`b3_objektivbeschluss_W-5.json`) | ✓ | ✗ (0/5, p=0,0625) | 0 | Hör-Session vorbereitet = **menschlich offen** |
| W-6 MIIPHER-DiT | ABGELEHNT (`b3_objektivbeschluss_W-6.json`) | ✓ | ✗ (1/6, p=0,22) | 364 (Zähl-Artefakte, vor B1-Fix `ef3795b7`) | Signifikanzkriterium B3 nicht erfüllt (WOHLKLANG_CLAIMS.md, Abschnitt „Formeller Objektivmetrik-Beschluss“ Punkt 3) |
| W-7 Harmonic-Inpainting | ABGELEHNT (`b3_objektivbeschluss_W-7.json`) | ✓ | ✗ (1/6, p=0,22) | 2236 (Zähl-Artefakte, vor B1-Fix) | Signifikanzkriterium B3 nicht erfüllt (WOHLKLANG_CLAIMS.md, ebd.) |
| W-2 SCNet / W-3 SGMSE | keine Messung | — | — | — | `gesperrt` (W-2 Budget offen, CPU ~7× langsamer; W-3 Pfad gesperrt) |
| W-8…W-13 | — | — | — | — | `gesperrt` (Vertrag `.github/WOHLKLANG_CLAIMS.md`) |

- **Kein Flag-Flip** (§5 SOTA_HANDOVER, §V7 (copilot-instructions.md)):
  `scripts/flag_freigabe.py` ist fail-closed und verlangt
  `never_worsen: true` ∧ `signifikant: true` — für **keine** Stufe erfüllt.
- **Hör-Sessions W-1/W-4/W-5** sind vorbereitet (`docs/reports/current/w*_blind/`
  mit key/items/fit-JSONs; Stimuli `study_stimuli/model_ab/*.wav` + `manifest.json`
  **bewusst ungetrackt** — dieses Repo tracked keine Audio-Dateien
  (`git ls-files '*.wav'` = 0), die Manifeste nennen die Pfade).
- **EAR-VAE v1↔v2 (D-K0-9):** Objektivbeschluss formal erfüllt (Blind-Messung
  seed 42, 16/17 Trials v2-vorteilhaft, Sign-Test p = 0,000275, `never_worsen ✓`,
  Δ musik_mos +0,10 / Δ defektlast −0,12; `docs/reports/current/2026-11-03_ear_vae_v1_v2_vergleich.json`)
  — **Rollout-Entscheidung bleibt menschlich** (D-K0-9/SOTA-M2: Hör-Abnahme vor
  jedem Austausch), daher **kein Flag-Flip** in dieser Sitzung.

## 6. „4-Modelle“-B3-Lauf (Material/EAR/Defekt/Loudness) — begründet offen

Der Abbruch-Report `docs/reports/current/2026-11-02_b3_modelle_metrik.json`
fordert „dann B3-Lauf wiederholen“ (Abbruch durch `roformer_fallback_result`,
behoben Commit `e0b61d5b`). Eine Wiederholung ist **nicht** mechanisch
möglich und wurde bewusst **nicht** nachgebaut:

1. Der Lauf war **ad-hoc** — kein Treiber-Skript im Repo, kein gelöschter
   Blob in der Git-History (`git log --all --diff-filter=D` zeigt kein
   `*b3*/*modelle*`-Skript); nur `scripts/flag_freigabe.py` und
   `scripts/b3_stufenmessung.py` schreiben Objektivbeschlüsse.
2. Das Mess-Design „4 Modelle (Material/EAR/Defekt/Loudness)“ ist aus den
   Artefakten nicht rekonstruierbar (Commit `0f528943` nennt die Zusammensetzung,
   aber keine Auswertungslogik) — **unklare Datenlage** ⇒ Auftragsteil (3):
   abbrechen und begründen statt neues Verfahren erfinden.
3. Korrekturhinweis: `models/material_crnn.pt` **existiert** (seit 2026-11-01,
   1,5 MB); die Meldung „fehlt“ im Report `2026-11-03_d_k1_3_material_crnn_wiring.md`
   Z. 15 ist worktree-abhängig veraltet. Der Flag-Flip `use_material_crnn`
   bleibt dennoch offen: nur §4-Verfahren (`scripts/flag_freigabe.py`) +
   Hörordnungs-Sign-off (Report Z. 17).

## 7. Bewusst nicht angefasst

- **Laufende GPU-Kette 2:** unberührt (keine parallelen Trainings, keine
  Drosselung; Messung CPU-only).
- **Flag-Aktivierungen/Flag-Flips:** keine (nur §4-Verfahren, §5 SOTA_HANDOVER).
- **§2/7 Modelle (WF-V4, TP-V2, BEATs-/MERT-/Gender-Head):** extern-/daten-
  blockiert (Tiefenanalyse E2), keine autonome Schließung möglich.
- **§2/8 Hör-/Lizenz-Entscheidungen** (MUSHRA n≥30, Vinyl-Lizenz, AERO):
  menschlich — nie eigenmächtig.
- **RT-Messung Referenz-Song (F-Reihe 53×→32×):** erst im nächsten
  trainingsfreien Fenster (SOTA_HANDOVER §2/4) — Kette 2 läuft.
- **Spec-Revision V27 (Onset-Set-Vertrag):** Spec-Änderung ⇒ PR-Evidenzblock +
  Maintainer-Sign-off (Report `2026-11-04_transient_energie_ursache_fix.md` §9.1)
  — bewusst offen.
- **Pipeline-Baustein U4 (`phase_08_transient_preservation` ≈ 1 % Wirkung):**
  Behebung ist ein eigener Pipeline-Fix mit Never-worsen-Beleg (§V7 (copilot-instructions.md):
  keine blinden Stärke-Pflaster) — offener Rest, siehe §9.3 des Ursachen-Reports.

## 8. Reproduzierbarkeit

- Seeds: Mess-Lauf seed 42 (Harness-Default); Unit-Tests deterministisch
  (§G5 (copilot-instructions.md)).
- Umgebungen: Messung `python3` (torch 2.7.0+cpu, GPU-frei); Tests
  `.venv_aurik/bin/python -B -m pytest`; `nice -n 19` (Host-Schutz).
- Mess-Befehle: `nice -n 19 python3 -B scripts/b3_transient_energie_neverworsen.py
  run --label neu` (Report `2026-11-04_transient_energie_ursache_fix.md` §7).
