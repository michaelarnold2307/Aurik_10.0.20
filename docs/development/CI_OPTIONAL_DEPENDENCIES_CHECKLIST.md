# Aurik 10.0.8 — CI Checkliste fuer optionale Test-Abhaengigkeiten

**Ziel:** Vermeidbare `skipped`-Tests in CI minimieren, ohne absichtliche Environment-Skips zu brechen.

## 1. Basisprofil (immer)

Installieren in jedem CI-Job:

```bash
.venv_aurik/bin/python -m pip install -r requirements/requirements_optimization.txt
```

Warum:

- Verhindert vermeidbare Module-Skips fuer Optimierungs-Tests (`optuna`).

## 2. Optional-Profile je Jobtyp

### 2.1 Unit-Only (headless, schnell)

Soll enthalten:

- `optuna` (bereits in requirements_optimization)
- `hypothesis`
- `soundfile`
- `librosa`

Soll nicht zwingend enthalten:

- GUI/Display-Stack (PyQt5 + Xvfb), wenn GUI-Tests bewusst nicht Teil dieses Jobs sind.

### 2.2 Unit+GUI (headless GUI)

Zusatz:

- `PyQt5`
- virtueller Display-Runner (z. B. `xvfb-run`)

Beispiel:

```bash
xvfb-run -a .venv_aurik/bin/python -m pytest tests/unit -m "not ml and not slow" -q
```

### 2.3 Performance/Benchmark-Job

Zusatz:

- `pytest-benchmark`

### 2.4 ONNX/ML-Job

Zusatz:

- `onnxruntime`
- model artifacts/pfade gemaess Projektkonfiguration

Hinweis:

- Heavy-Tests nur in dediziertem Job mit `--run-heavy-tests`.

## 3. Schneller Import-Precheck (vor pytest)

Empfohlen als frueher CI-Schritt:

```bash
.venv_aurik/bin/python - <<'PY'
mods = ["optuna", "hypothesis", "soundfile", "librosa", "onnxruntime"]
missing = []
for m in mods:
    try:
        __import__(m)
    except Exception:
        missing.append(m)
if missing:
    raise SystemExit(f"Fehlende optionale Module fuer diesen Job: {missing}")
print("Optional dependency precheck: OK")
PY
```

## 4. Skip-Report in CI sichtbar machen

Immer mit `-rs` laufen lassen, damit Gruende im Log stehen:

```bash
.venv_aurik/bin/python -m pytest tests/unit -q -rs
```

## 5. Einstufung von Skips

### 5.1 Behebbar im Setup

- `optuna nicht installiert`
- `hypothesis nicht installiert`
- `soundfile nicht installiert`
- `pytest-benchmark not installed`

### 5.2 Absichtlich / umgebungsabhaengig

- `PyQt5`/Display fehlt in headless Non-GUI Job
- Externe Lizenztools oder externe Modelle nicht vorhanden
- Heavy-Tests in Standard-Run ohne `--run-heavy-tests`

## 6. Konkrete Regression aus 2026-03-28

- Befund: `optuna` fehlte im aktiven venv -> vermeidbare Skip-Gefahr in Optimierungs-Tests.
- Fix: `optuna` installiert.
- Verifikation: `tests/test_optimization.py` + `tests/test_optimization_phase2.py` liefen mit `54 passed`, `0 skipped`, `0 failed`.

## 7. CI-Definition of Done

Ein Job gilt als sauber konfiguriert, wenn:

1. Precheck der fuer den Job erforderlichen optionalen Module gruene Ausgabe liefert.
2. `pytest -rs` keine vermeidbaren Skips zeigt.
3. Verbleibende Skips explizit als absichtlich/umgebungsbedingt dokumentiert sind.
