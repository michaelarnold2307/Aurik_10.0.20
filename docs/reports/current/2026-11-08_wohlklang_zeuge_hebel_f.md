# SOTA-Lückenschluss 2026-11-08 — Wohlklang-Zeuge je Lauf (§2b Hebel F) + Cantus-No-Op

> **Auftrag:** `docs/SOTA_HANDOVER.md` §2b Hebel F („Forensik-Monitor:
> WP-3-Forensik + musik_mos/defektlast als Dauer-Wohlklang-Zeuge je Lauf") und
> §2 Zeile 11 (Ketten-Verdrahtung). **GPU-Lage:** Kette 2 (`d3_bigvgan_f3`,
> `train_job_runner`, PID 392298) belegte die GPU ⇒ kein Ketten-Neustart, kein
> paralleles Training, keine Voll-Suite, keine UAT-Neuerzeugung (D-K3-13).
> Alle Änderungen sind CPU-seitig verifiziert.

## 1. Befund A — die zertifizierten Metriken sahen den Produktionspfad nie

`musik_mos` (referenzfrei, Spearman 0,95) und `defektlast` sind seit
2026-10-28 **richtungszertifiziert**
(`docs/reports/current/qualitaetsmetrik_richtungszertifikat.json`, Seed 42,
Schwelle 0,8; gemessen 0,9487 auf beiden Materialien). Verbraucher waren
**ausschließlich Mess-Skripte**:

* `scripts/b3_stufenmessung.py`, `scripts/b3_komposition_suche.py`,
  `scripts/b3_transient_energie_neverworsen.py`,
* `scripts/ear_vae_v1_v2_vergleich.py`, `scripts/golden_ear_prototyp.py`,
* `scripts/quality_metric_direction_gate.py` (das Zertifikat selbst).

`grep -rn "musik_mos\|defektlast" backend/` lieferte **null Treffer**. Folge:
Für einen einzelnen Restaurierungslauf konnte niemand sagen, ob er das Signal
unter der Hörschwelle verschlechtert hat — der Wohlklang-Vertrag hatte Zeugen
nur im Messlabor, nicht am Fließband (§2b Hebel F: der letzte Hebel ohne
Status).

## 2. Umsetzung A — `backend/core/dsp/wohlklang_zeuge.py`

Neues DSP-Modul (Write-Gate vorher geprüft:
`scripts/repo_search.py --before-create` → „keine kanonische Alternative"):

| Baustein | Semantik |
| --- | --- |
| `messe_wohlklang_zeuge(eingang, ausgang, sr, jnd=0.05)` | misst `musik_mos` + `defektlast` an **Eingang und Ausgang**, liefert Deltas, `regression` je Metrik, `gerichteter_fall`, `hoer_instanz`, `grund`, `pegel_delta_db` (Diagnose), `laenge_delta_samples` |
| `zeuge_meta(...)` | **nie werfende** Produktions-Fassung (§V6): `verfuegbar=False` + Warnung + Grund statt Abbruch |
| `defektlast(audio, sr)` | **kanonische** Definition (aus `scripts/quality_metric_direction_gate.py` hierher gewandert; das Skript delegiert) |
| `JND_TOLERANZ = 0.05` | **EINE** Quelle — Stufenmessung und Kompositions-Suche importieren sie jetzt |

**Richtungs-Semantik** (Hörordnung §7/§8a — Metriken sind Zeugen):
`musik_mos` höher = besser ⇒ Regression bei `delta < −JND`; `defektlast`
höher = schlechter ⇒ Regression bei `delta > +JND`. Ein **gerichteter Fall**
(`hoer_instanz = "vorzulegen"`) ist genau dann gesetzt, wenn eine der beiden
Metriken über die JND hinaus in die falsche Richtung läuft. **REPORT-ONLY:**
kein Sample-Eingriff, kein Rollback, keine Export-Blockade, kein Flag.

### Verdrahtung (`backend/core/unified_restorer_v3.py`)

`UnifiedRestorerV3._wohlklang_zeuge_meta()` wird an **drei** Produktionsenden
aufgerufen und schreibt `metadata["wohlklang_zeuge"]`:

1. **Einzelpass** (`restore()`, vor `return result`, Zeile ~24308),
2. **Chunked-Pfad** (`_restore_chunked`, bei der Song-Assembly neben
   `inviting_gate_song_level`, Zeile ~48460) — Eingang ist die Original-Audio,
   genau wie beim Einladungs-Gate,
3. **Options-/Closed-Loop-Pfad** (`_restore_optimized`, Metadaten-Dict).

Quell-Pin im Test: genau **drei** Schreibstellen und **keine** Lesestelle —
kein Gate, kein Skip-Pfad und kein Rollback konsumiert den Befund.

## 3. Befund B — Cantus-`fine_tune` trainierte NULL Epochen (D-K3-84)

Beim Prüfen der Kette-1-Manifeste fiel auf: `train_job_c2_cantus_finetune.json`
stand auf `erfolgreich` — mit **11,9 s** Laufzeit und identischem `best_val`
wie das Pretrain:

```
Resume: epoch=50 step=10000 best_val=4.5922 (seed=42)
Training: phase=fine_tune epochs=30 lr=2.0e-05 …
Fertig: best_val=4.5922
```

Ursache: `scripts/train_cantus.py` übernahm `start_epoch` **ohne Phasenbezug**
aus dem Checkpoint. Pretrain `epoch=50` gegen `--epochs=30` ⇒
`range(50, 30)` ist leer ⇒ **null Gradienten-Schritte**, aber Exit 0 und
„erfolgreich" im Manifest (dieselbe Klasse wie D-K3-80, Manifest-Lüge).
Zweiter Effekt derselben Wurzel: `best_val` (4,5922, Pretrain-Verlust) hing an
einer fremden Verlust-Achse ⇒ `checkpoint_best.pt` wäre in `fine_tune` **nie**
geschrieben worden.

### Fix

* Neue reine Funktion `_phasen_lokale_resume_groessen()`: bei Phasenwechsel
  (`ckpt["phase"] != --phase`, auch `None` bei Alt-Checkpoints) werden
  `start_epoch`, `global_step` und `best_val` auf `0/0/inf` zurückgesetzt —
  die **Gewichte bleiben geladen** (das ist der Sinn des Finetunes); der
  Optimizer war bereits vorher bewusst phasen-lokal verworfen.
* §V6-Guard: **0 trainierte Epochen ⇒ kein Erfolg.** `report["epochen_trainiert"]`
  + `stopped_reason = "no_epochs"` ⇒ **Exit 3** (bestehender stopped_reason-Pfad).
* Der No-Op-Lauf wurde als
  `docs/reports/current/train_job_c2_cantus_finetune_noop_20261108.json`
  **archiviert** (Beleg bleibt erhalten, kein stiller Erfolg). Ohne das hätte
  `train_job_runner` den Job beim nächsten Kette-1-Start als „bereits
  erfolgreich" übersprungen und Cantus wäre dauerhaft untrainiert geblieben.
  Das Pretrain-Manifest bleibt gültig (Checkpoint `epoch=50`, 2026-10-27).

## 4. Verifikation (CPU, während Kette 2 lief)

| Prüfung | Ergebnis |
| --- | --- |
| `tests/unit/test_wohlklang_zeuge.py` | **13/13 grün** (Richtung, JND-Grenze, Determinismus §G5, Stereo-Layout (C,N)/(N,C), kein Sample-Eingriff, fail-safe bei zu kurzem Signal/ungültiger Rate, Bit-Identität zur Mess-Referenz, JND-EINE-Quelle, UV3-Verdrahtung) |
| `tests/unit/test_gpu_kette1_wiring.py` | **14/14 grün** (4 neue D-K3-84-Tests) |
| `test_quality_metric_direction_gate.py` + `test_b3_komposition_suche.py` + `test_b3_pfad_wirksamkeit.py` + `test_b3_full_song_defect_merge.py` | **31/31 grün** (3+9+14+5; Delegation/JND-Import bricht die Messpfade nicht); B3-Familie gesamt `-k b3`: **37 passed** |
| `py_compile` aller geänderten Module/Trainer | sauber |
| `_wohlklang_zeuge_meta` direkt aufgerufen (Dummy-Self, 1 s Sinus) | `verfuegbar=True`, `delta={0.0, 0.0}`, `gerichteter_fall=False` |

**Bit-Identität:** `wz.defektlast(x, sr) == scripts.quality_metric_direction_gate._defektlast(x, sr)`
für drei Signale (Test) — die Zertifikatszahlen bleiben damit gültig.

## 5. Ehrliche Grenzen (nicht gemessen / nicht getan)

* **Kein End-to-End-Lauf** des Zeugen: jeder echte Restaurierungslauf braucht
  die GPU/Modelle; Kette 2 belegte die GPU (kein paralleles Training, §5).
  Verifiziert ist die Wirkungs-Semantik am Modul und die Verdrahtung per
  Quell-Pin.
* **Kein RT-Delta gemessen.** Erwartung: zwei numpy-Metrikfenster
  (`musik_mos` sieht 32 k Samples, `defektlast` 16 k) — deutlich unter 100 ms
  je Lauf; die 32×-RT-Norm bleibt unberührt, weil der Zeuge **kein** Audio
  verändert. Die Messung gehört ins trainingsfreie Fenster.
* **Kein Cantus-Neulauf.** Der Trainer ist gefixt und getestet, der echte
  `fine_tune` (30 Epochen) läuft beim nächsten Kette-1-Start.
* **Kein Flag-Flip, kein Deploy.** `use_wf4_warp_head`/`use_tp_v2_phase`
  bleiben AUS (B3-M12-Messung 2026-11-08: 1/6 bzw. 0/6, fail-closed §4).

## 6. Berührte Dateien

```
backend/core/dsp/wohlklang_zeuge.py                 (neu)
backend/core/unified_restorer_v3.py                 (3 Verdrahtungspunkte + Hook)
scripts/quality_metric_direction_gate.py            (_defektlast delegiert)
scripts/b3_stufenmessung.py                         (JND aus dem Zeugen)
scripts/b3_komposition_suche.py                     (JND aus dem Zeugen)
scripts/train_cantus.py                             (phasen-lokale Resume-Größen + No-Op-Guard)
tests/unit/test_wohlklang_zeuge.py                  (neu, 13 Tests)
tests/unit/test_gpu_kette1_wiring.py                (+4 Tests, D-K3-84)
.github/FILE_REGISTRY.md                            (2 Einträge)
.github/SOTA_DEFICIT_REGISTER.md                    (D-K3-83, D-K3-84; Trainings-Stand nachgezogen)
docs/TODOS_SOTA_ROADMAP.md                          (Hebel F, F9, SOTA-M8/M9/M12, WF-V4/TP-V2)
docs/reports/current/train_job_c2_cantus_finetune_noop_20261108.json (archiviert)
```

## 7. Reproduktion

```bash
AURIK_FORCE_CPU=1 python3 -m pytest tests/unit/test_wohlklang_zeuge.py -q
AURIK_FORCE_CPU=1 .venv_aurik/bin/python -m pytest tests/unit/test_gpu_kette1_wiring.py -q
python3 -c "import numpy as np; from backend.core.dsp.wohlklang_zeuge import zeuge_meta; \
x=np.sin(2*np.pi*220*np.arange(48000)/48000).astype('float32'); print(zeuge_meta(x, x, 48000))"
```

**Register:** D-K3-83 (Zeuge), D-K3-84 (Cantus-No-Op).
