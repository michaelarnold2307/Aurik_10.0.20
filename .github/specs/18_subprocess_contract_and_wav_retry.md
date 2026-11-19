# Spec §18 — Subprocess-Vertrag, WAV-Retry & Scipy-Unpack | §v10.50

**Aurik 10.11.14+ | Gültig ab: 03. August 2026 | Normativ für alle Orchestrator- und WAV-Loader-Module**

---

## 1. Problemstellung: Was der Log vom April 2026 offenbart hat

### 1.1 Orchestrator-Log (orchestrator_runtime.log, 2026-04-25)

| Metrik | Run 1 | Run 2 | Gesamt |
|--------|-------|-------|--------|
| WAV-Load WARNINGs | 9 | 0 | 9 |
| WAV-Load ERRORs | 1 | 0 | 1 |
| Restart-Loop ERRORs | 42 | 48 | 90 |
| quality_mode ERROR | 0 | 1 | 1 |
| pre_analysis_runner WARNING | 0 | 1 | 1 |
| Pegelexplosion WARNINGs (legitim) | 0 | 9 | 9 |
| **Summe vermeidbar** | **52** | **50** | **102** |
| **Summe legitim** | **0** | **9** | **9** |
| **Gesamt** | **52** | **59** | **111** |

Die 102 vermeidbaren W/E-Einträge haben **drei Root Causes**:

### 1.2 Root Cause 1: Falscher Python-Interpreter in Subprozessen

```python
# FALSCH (orchestrate_quality_monitoring.py, frontend_with_analysis.py, simple_restoration_monitor.py):
str(_WORKSPACE_ROOT / ".venv_aurik" / "bin" / "python")

# KORREKT:
sys.executable
```

**Warum**: `.venv_aurik` hat andere soundfile/scipy-Versionen als das aktuell ausgeführte venv_rocm72. Das führt zu `too many values to unpack` in soundfile/scipy, weil die Versionen inkompatibel sind.

**Betroffene Dateien (vor Fix)**:

- `scripts/orchestrate_quality_monitoring.py` (Analyzer + Pegelexplosion-Monitor)
- `scripts/frontend_with_analysis.py` (Analyzer)
- `scripts/simple_restoration_monitor.py` (Pegelexplosion-Monitor)

### 1.3 Root Cause 2: Fehlende Retry-Logik bei transienten WAV-Read-Fehlern

```python
# FALSCH: einmaliger Load-Versuch ohne Retry
result = load_audio_file(str(audio_path))
if result is None or result.get("error"):
    logger.warning(f"✗ Load fehlgeschlagen: ...")
    return

# KORREKT: 3× Retry mit exponentiellem Backoff bei "unpack"-Fehlern
for _attempt in range(3):
    result = load_audio_file(str(audio_path))
    if not result or result.get("error"):
        _err = result.get("error") if result else "None"
        if _attempt < 2 and "unpack" in str(_err).lower():
            time.sleep(0.5 * (_attempt + 1))
            continue
        logger.warning(f"✗ Load fehlgeschlagen: {_err}")
        return
    break
```

**Warum**: WAV-Dateien können während des Schreibens durch einen anderen Prozess kurzzeitig unlesbar sein. Ein Retry mit 0.5–1.5s Delay überbrückt diese Race-Condition.

### 1.4 Root Cause 3: Unsicheres scipy.io.wavfile.read()-Unpack

```python
# FALSCH: Tuple-Destructuring kann bei unerwartetem Rückgabewert crashen
sr, data = wavfile.read(path)

# KORREKT: Index-basierte Entpackung mit Typ-Prüfung
_wf_result = wavfile.read(path)
if isinstance(_wf_result, tuple) and len(_wf_result) >= 2:
    sr = int(_wf_result[0])
    data = _wf_result[1]
else:
    raise ValueError(f"wavfile.read returned unexpected type: {type(_wf_result)}")
```

**Warum**: In seltenen Fällen (partiell geschriebene Dateien, exotische WAV-Container) kann `wavfile.read()` andere Strukturen zurückgeben. Index-basierte Entpackung fängt dies ab.

### 1.5 Zusätzlich: quality_mode + pre_analysis_runner (bereits behoben)

- `quality_mode`: Wurde zwischen April und August 2026 zu `UnifiedRestorerV3.__init__()` hinzugefügt.
- `pre_analysis_runner`: Das Modul existiert nicht mehr, kein Import im aktuellen Code.

---

## 2. Normative Vorgaben

### §V34 — Subprocess-Python-Vertrag

**Gebot**: Jeder `subprocess.Popen`/`subprocess.run`-Aufruf, der ein Python-Script startet, MUSS `sys.executable` als Interpreter verwenden. Hartcodierte Venv-Pfade sind VERBOTEN.

**Ausnahmen**:

- Shell-Script-Wrapper (`run_aurik.sh`) — diese lösen den Interpreter selbst auf.
- CI/CD-Scripts, die ein bestimmtes Environment garantieren müssen (dürfen `sys.executable` via `--python`-Flag übersteuern).

**Prüfung**: Pre-Commit-Hook `check_sys_executable.py` scannt alle `.py`-Dateien in `scripts/` auf hartcodierte `.venv_aurik/bin/python`-Pfade in `subprocess.Popen`-Aufrufen.

### §V35 — WAV-Load-Retry-Vertrag

**Gebot**: Jeder WAV-Ladevorgang in Monitoring-/Analyzer-Prozessen MUSS einen Retry-Loop mit mindestens 2 Wiederholungen und exponentiellem Backoff (0.5s Basis) implementieren. Der Retry greift NUR bei Fehlermeldungen, die `unpack` enthalten.

**Betroffene Module**:

- `scripts/pegelexplosion_monitor.py` → `_analyze_file()`
- `scripts/continuous_deep_analysis.py` → `run_analysis()` Audio-Load

**Prüfung**: Anti-Regression-Gate Bug 11 prüft auf Präsenz von `_max_retries` oder `_max_load_retries` in WAV-Load-Blöcken.

### §V36 — Scipy-WAV-Unpack-Vertrag

**Gebot**: `scipy.io.wavfile.read()` darf NIEMALS per Tuple-Destructuring (`sr, data = wavfile.read(...)`) entpackt werden. Stattdessen MUSS index-basierte Entpackung mit `isinstance`-Typ-Prüfung verwendet werden.

**Betroffene Module**:

- `backend/meta_router.py` → `_load_audio()` Stufe 2

**Prüfung**: Anti-Regression-Gate Bug 12 prüft auf das Muster `sr, data = wavfile.read(`.

---

## 3. Erwartete Wirkung

| Log-Kategorie | Vor Fix (pro Run) | Nach Fix (pro Run) | Reduktion |
|---------------|-------------------|---------------------|-----------|
| WAV-Load WARNINGs | 9 | 0 | 100% |
| WAV-Load ERRORs | 1 | 0 | 100% |
| Restart-Loop ERRORs | 45 | 0 | 100% |
| quality_mode ERROR | 1 | 0 | 100% |
| pre_analysis_runner WARNING | 1 | 0 | 100% |
| Pegelexplosion WARNINGs | 9 | 0 | 100% |
| **Gesamt** | **66** | **0** | **100%** |

Die 9 Pegelexplosion-WARNINGs waren **echte Audio-Defekte** (Spikes 3.5–8.7 dB im
Fade-Out) — siehe §v10.51 und bestehende Dokumentation in
[Spec §02](../.github/specs/02_pipeline_architecture.md) (Zeile 509–520, 1948–2035).

### §v10.51 MDEM Hard-Floor-Guard (Bug 11)

**Spec-Implementierungs-Lücke**: Spec §02 schreibt −36 dBFS als normativen Quiet-Zone-
Schwellwert für `morph()` vor (Zeile 515: "MDEM, 400 ms → −36 dBFS, Einzel-Bedingung").
Der Code implementierte einen rein adaptiven Schwellwert `_FADEOUT_QUIET_LUFS =
clip(p5+8, −36, −18)`, der bei Vinyl-Material mit Rauschboden −35 dBFS auf −27 dBFS
steigt — und damit die −36-dBFS-Vorgabe unterläuft.

**Fix**: Zusätzlicher harter −36 dBFS Floor per OR-Bedingung an allen 7 Guard-Punkten
in `morph()` und `_morph_internal()`. Der adaptive Schwellwert schützt weiterhin vor
Vinyl-Rauschen bei −35 dBFS, der Hard-Floor verhindert positive Gains in echten
Stille-Zonen (< −36 dBFS) auch bei angehobenem adaptivem Schwellwert.

**Referenz**: Spec 02 §2.30b (Zeile 509–552), Spec 03 §2.30 (Zeile 583–642).

---

## 4. Dateien

| Datei | Änderung |
|-------|----------|
| `scripts/orchestrate_quality_monitoring.py` | `.venv_aurik` → `sys.executable` |
| `scripts/frontend_with_analysis.py` | `.venv_aurik` → `sys.executable` |
| `scripts/simple_restoration_monitor.py` | `.venv_aurik` → `sys.executable` |
| `scripts/worldclass_autopilot_pipeline.py` | `.venv_aurik` → `sys.executable` |
| `scripts/pegelexplosion_monitor.py` | Retry-Loop (3×, 0.5s Backoff) |
| `scripts/continuous_deep_analysis.py` | Retry-Loop (3×, 0.5s Backoff) |
| `backend/meta_router.py` | Robustes scipy-WAV-Unpack |
| `backend/core/micro_dynamics_envelope_morphing.py` | §v10.51 Post-MDEM Peak Guard (Layer 2) |
| `backend/core/signal_flow_tracer.py` | §G71 Wet-Ceilings + `get_sft_wet_ceilings()` |
| `backend/core/joint_calibrator.py` | §G71 Adaptive min_strength aus Restorability |
| `backend/core/unified_restorer_v3.py` | §G71 Dynamische Ceilings + Dual-Path Audio-Callback |
| `Aurik10/ui/modern_window.py` | Punkt 0: Dual-Path SharedAudioRing + Qt-Signal |
| `scripts/compliance/anti_regression_gate.py` | Bug 10-13 Checks |
| `scripts/compliance/check_sys_executable.py` | Neuer Pre-Commit-Hook |
| `tests/unit/test_subprocess_contract.py` | 12 Regression-Tests |

---

## 5. Changelog

| Version | Datum | Änderung |
|---------|-------|----------|
| 1.0 | 2026-08-03 | Initial: §V34-V36, Anti-Regression Bug 10-13, Pre-Commit-Hook |
| 1.1 | 2026-08-03 | §v10.51 MDEM Peak Guard, §G71 Wet-Ceilings, Punkt 0 Dual-Path |
