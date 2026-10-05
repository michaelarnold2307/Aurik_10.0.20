# Evidenz-Generator-Defekt: `run_real_audio_corpus_test.py` lief Schein-Ergebnisse

**Datum:** 2026-10-03 · **Schwere:** hoch (Schein-Evidenz in der Beweiskette) ·
**Status:** Ursache gefunden, Fix bereit, Anwendung wartet auf Read-Werkzeug-Guard

## Befund (gemessen, nicht vermutet)

1. `_run_restoration()` importiert `backend.core.e2e_pipeline.E2EPipeline` —
   **dieses Modul existiert nirgends im Repo** (Grep über `backend/`: 0 Treffer).
2. Folge: JEDER Lauf fängt die `ModuleNotFoundError` ab und fällt auf
   **Identität (Input zurück)** — mit einer bloßen Warnung, Exit-Code 0.
3. Damit misst der „MUSHRA-Score" dieses Skripts **beschädigt vs. sauber**,
   also die Degradations-Schwere — nicht die Restaurierungsqualität.
4. Zusatzfehler: `material_type="vinyl"` hartkodiert (Dosierungs-Bias je
   Material), Paarung per Namensgleichheit (kennt die Korpus-Konvention
   `<base>_*` → `<base>_clean` nicht → „Keine Paare"), `samples`-Feld misst
   `len()` = Kanalzahl, Fehlerfälle bleiben Exit 0.

## Widerruf früherer Zahlen (ausdrücklich)

Die heute berichteten Werte **76,9/100 (Hall-Fall)** und **70,75 Mittel (57 Paare)**
sind KEINE Restaurierungsqualität — sie sind die Roh-Ähnlichkeit der
beschädigten Eingaben gegenüber den sauberen Referenzen (Standardabweichung
8,73, Shellac-Schlechteste 37,8 = schwerste Degradation). Sie sind als
**Degradations-Baseline** wertvoll, aber jeder Qualitätsanspruch daraus ist
zurückzuziehen.

## Anzuwendender Fix (exakt, kopierfertig)

1. **`_run_restoration` gegen die echte Pipeline tauschen** — das bewährte
   Muster aus `scripts/mushra_harness.py::prepare_pair`:
   ```python
   from backend.core.unified_restorer_v3 import QualityMode, RestorationConfig, UnifiedRestorerV3

   cfg = RestorationConfig(mode=QualityMode("restoration"))
   engine = UnifiedRestorerV3(cfg)
   result = engine.restore(audio, sample_rate=sr)
   ```
   plus Layout-Normalisierung wie dort (T/2,2-T-Korrektur).
2. **Kein stilles Fallback** (§V6 (VERBOTEN.md)): Exception ⇒ Fall als
   `error` protokollieren und am Ende Exit ≠ 0, wenn Fehlerfälle existieren.
   Ein Beweis-Generator darf nie still Identität ausgeben.
3. **Paarung auf Korpus-Konvention:** längster Clean-Präfix (`<base>_clean`),
   `damaged.stem.startswith(base)` — identisch zur Logik im Hör-Panel-Setup.
4. **`sys.path`-Insert** am Kopf (Muster `scripts/validate_hr_v1.py`):
   ```python
   _PROJECT_ROOT = Path(__file__).resolve().parent.parent
   if str(_PROJECT_ROOT) not in sys.path:
       sys.path.insert(0, str(_PROJECT_ROOT))
   ```
5. **`samples`-Feld** auf `int(np.asarray(x).size)` umstellen.

## Nach dem Anwenden (verbindlich)

- Reverb-Smoke erneut: echte Laufzeit (≈ 5 s/Fall), echte Score-Differenz.
- Vollen Korpus (57 Paare) neu laufen lassen → `real_audio_corpus_results_full_*.json`.
- Danach erst die Worldclass-Reports (corpus_gate/trusted_vocal) neu fahren.
- Die Degradations-Baseline dieses Laufs bleibt als `…_2026-10-03.json` erhalten
  (umetikettiert: keine Qualitätsaussage).

---

## ANGEWANDT + GEMESSEN 2026-10-05 (Ausführungssitzung)

1. **Patch angewandt** (`scripts/run_real_audio_corpus_test.py`): `_run_restoration`
   jetzt über `UnifiedRestorerV3` (Muster `mushra_harness.prepare_pair`, inkl.
   Layout-Normalisierung), kein stiller Fallback mehr (Laden/Restaurieren/
   Bewerten eskalieren; der Fall wird als `status:"error"` protokolliert und der
   Lauf endet mit Exit ≠ 0, §V6), Korpus-Paarung `<base>_clean` (längster
   Präfix), `sys.path`-Insert am Kopf, `samples` = `int(np.asarray(x).size)`.
2. **Entwurfs-Fehler gefunden:** `QualityMode("restoration")` ist ungültig
   (Enum: fast/balanced/quality/maximum) → auf `QualityMode.QUALITY` korrigiert.
   Der erste Smoke-Lauf bewies die neue Ehrlichkeit sofort: `FALL-FEHLER …
   ValueError … als error protokolliert`, Exit 1 statt Schein-Score.
3. **Zweiter Smoke-Lauf (verbindlich, Reverb-Paar, CPU-erzwungen):**
   `reverb_jazz_1960s_hall.wav` → MUSHRA **75,6/100 (Fair)**, NSIM 0,651,
   Anchor 62,7, `error_cases: 0`, Laufzeit 2699,7 s. JSON:
   `reports/real_audio_corpus_reverb_smoke_20261004.json`; Log:
   `output/corpus_smoke_20261004b.log`.
4. **Zusatzbefund (Produktionsbug, behoben):** `AURIK_FORCE_CPU=1` wurde von
   `get_onnx_providers()`/`apply_gpu_policy()` nicht respektiert — direkte
   Aufrufer (bigvgan/flashsr) erhielten trotz CPU-Vertrag eine ROCm-Session;
   unter GPU-Last endet das in einem ORT-Hart-Abort
   (`no ROCm-capable device is detected`). Zentral in
   `backend/core/gpu_model_registry.py` gefixt (CPU erzwungen bei
   `AURIK_FORCE_CPU=1`, Log `§v10.40c … CPU erzwungen`) + 2 Regressionstests
   (18/18 grün).
5. **Folgemaßnahmen:** Voller Korpus (57 Paare) noch nicht neu gefahren —
   45 min/Paar bei CPU-only (165× RT; GPU war durch Training belegt).
