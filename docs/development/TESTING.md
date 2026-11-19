# Aurik 10.0.8 — Testing Guide

**Version:** 10.0.8  
**Datum:** Mai 2026  
**Status:** ✅ Production Ready

> **Normativer Hinweis (Release):** Produktive Laufzeitvertraege werden ueber Bridge + `AurikDenker.denke(...)`
> und nachgelagertes `export_guard()` validiert. Historische v2-Beispiele sind nur `LEGACY_NON_RELEASE`.
> Hinweis: Verbindlicher Stand der Testinvarianten ist in `.github/specs/07_quality_and_tests.md` dokumentiert.

---

## Inhaltsverzeichnis

- [Übersicht](#übersicht)
- [Test-Struktur](#test-struktur)
- [Tests ausführen](#tests-ausführen)
- [Gate-Runbook (gegen Mehrfachstarts)](#gate-runbook-gegen-mehrfachstarts)
- [Tests schreiben](#tests-schreiben)
- [Test-Coverage](#test-coverage)
- [E2E-Testing](#e2e-testing)
- [Performance-Testing](#performance-testing)
- [CI/CD-Integration](#cicd-integration)
- [CI-Checkliste Optionale Abhängigkeiten](#ci-checkliste-optionale-abhaengigkeiten)
- [Best Practices](#best-practices)

---

## Übersicht

Aurik verwendet **pytest** als Test-Framework mit umfassender Test-Coverage für alle Komponenten.

### Test-Statistik

- **Umfangreiche Testabdeckung** (Unit, Integration, Normative, Regression)
- **Coverage:** >80 % (Unit + Integration)
- **Test-Typen:** Unit, Integration, E2E (Normativ), Performance (AMRB-Benchmark)
- **Test-Dauer:** Unit-Suite ~5 min (Standard) | vollständige Suite mit ML ~30 min

> **Heavy-Tests (ML/slow)**: Nur mit `--run-heavy-tests` ausgeführt (kein Bestandteil des Standard-Runs).

---

## Test-Struktur

```text
tests/
├── test_unified_restorer.py          # Core Restorer Tests
├── test_unified_restorer_modes.py    # Processing Modes Tests
├── test_e2e_magicbutton.py          # E2E Magic Button Tests
├── test_musical_goals_v2_quick.py    # Musical Goals Tests
├── test_phase2_pipeline_validation.py # Phase 2 Tests
├── test_phase2_quick.py              # Quick Phase 2 Tests
├── unit/                             # Unit Tests
│   ├── test_audio_analyzer.py
│   ├── test_click_remover.py
│   ├── test_denoiser.py
│   ├── test_medium_detector.py
│   └── ...
├── integration/                      # Integration Tests
│   ├── test_pipeline_flow.py
│   ├── test_ml_plugins.py
│   └── ...
└── fixtures/                         # Test Audio Files
    ├── vinyl_sample.wav
    ├── cassette_sample.wav
    ├── speech_sample.wav
    └── ...
```

---

## Tests ausführen

## CI-Checkliste Optionale Abhaengigkeiten

Fuer reproduzierbare CI-Laeufe mit minimalen vermeidbaren Skips:

- Siehe [CI_OPTIONAL_DEPENDENCIES_CHECKLIST.md](CI_OPTIONAL_DEPENDENCIES_CHECKLIST.md)
- Enthalten: Install-Profile pro Jobtyp, Precheck-Skript, Skip-Einstufung und Definition of Done

### 1. Alle Unit-Tests (Standard — kein ML)

```bash
# Empfohlener Standard-Lauf (schnell, kein ML):
.venv_aurik/bin/python -m pytest tests/unit -p no:xdist \
  --override-ini="addopts=--strict-markers --import-mode=importlib" \
  --timeout=30 --tb=short -q --disable-warnings --no-header
```

**Erwartete Ausgabe (Beispiel):**

```text
6571 passed, 2 skipped, 21 deselected
```

> Heavy-Tests (ML-Inferenz, ONNX, Timeout ≥30 s) sind mit `@pytest.mark.ml` / `@pytest.mark.slow` markiert
> und werden im Standard-Lauf automatisch übersprungen (`conftest.py`).

---

### 2. Spezifische Tests

```bash
# Nur einen Test-File
pytest tests/unit -k "restoration"

# Nur eine Test-Klasse
pytest tests/unit -k "spec_upgrade or modern_window"

# Nur eine Test-Methode
pytest tests/unit/test_spec_upgrade_gate.py -k test_promote

# Mit Pattern Matching
pytest -k "test_restoration"  # Alle Tests mit "restoration" im Namen
pytest -k "not slow"          # Alle Tests außer langsame
```

---

### 3. Tests nach Marker

```bash
# Nur Unit Tests
pytest -m unit

# Nur Integration Tests
pytest -m integration

# Nur E2E Tests
pytest -m e2e

# Nur schnelle Tests (< 10s)
pytest -m "not slow"

# Nur GPU Tests
pytest -m gpu
```

**Verfügbare Marker:**

- `@pytest.mark.unit` — Unit Tests (schnell, isoliert)
- `@pytest.mark.integration` — Integration Tests (mehrere Komponenten)
- `@pytest.mark.e2e` — End-to-End Tests (vollständige Pipeline, nur mit `--run-heavy-tests`)
- `@pytest.mark.ml` — ML-Inferenz-Tests (ONNX-Modell erforderlich, nur mit `--run-heavy-tests`)
- `@pytest.mark.slow` — Langsame Tests (> 30 s, nur mit `--run-heavy-tests`)
- `@pytest.mark.normative` — CI-Gate-Tests (immer aktiviert, nie skippen!)

---

### 4. Tests mit Options

```bash
# Stoppe bei erstem Fehler
pytest --maxfail=1

# Deaktiviere Warnings
pytest --disable-warnings

# Kurze Traceback-Ausgabe
pytest --tb=short

# Zeige lokale Variablen bei Fehler
pytest --showlocals

# Parallele Ausführung (8 Workers)
pytest -n 8

# Kombiniert (wie im Projekt verwendet)
pytest --maxfail=1 --disable-warnings --tb=short
```

---

### 5. Task-basierte Ausführung (VSCode)

```bash
# VSCode Tasks (siehe workspace tasks)
# Task 1: "pytest aurik Testsuite"
pytest --maxfail=1 --disable-warnings --tb=short

# Task 2: "pytest aurik Testsuite (venv)"
/path/to/.venv_aurik/bin/python -m pytest --maxfail=1 --disable-warnings --tb=short
```

**VSCode:** Drücke `Ctrl+Shift+P` → "Tasks: Run Task" → "pytest aurik Testsuite"

---

## Gate-Runbook (gegen Mehrfachstarts)

Ziel: AMRB/Competitive/Spec-Gates nur einmal starten und den Status danach aus Artefakten lesen,
statt denselben Langlauf mehrfach parallel neu zu triggern.

### Verbindliche Reihenfolge fuer Langlaeufer

1. Gate starten (genau ein Start pro Profil)
2. Nicht erneut starten, solange ein identischer Lauf aktiv ist
3. Ergebnis aus Status-Artefakten lesen
4. Nur bei geaendertem Profil bewusst neu starten (z. B. anderes Timeout)

### Warum das wichtig ist

- Doppelte Starts verbrauchen massiv Zeit und Ressourcen
- Parallele identische Gatelaufe erschweren die Interpretation
- Das Projekt hat bereits einen Duplicate-Run-Guard ueber Lock-Fingerprints

### Aktivitaet pruefen (statt blind neu starten)

- Prozesssicht:
    - `ps -eo pid,etimes,pcpu,pmem,state,cmd | rg -i "test_amrb_ci_gate|test_competitive_ci_gate|run_tests_safe\.sh"`
- Lock-Hinweis:
    - `ls -lt logs/locks | head -n 20`

Wenn ein identischer Lauf schon aktiv ist, blockt der Safe-Runner mit:
`[safe-runner] Abbruch: identischer Testlauf ist bereits aktiv ...`

### Ergebnisquelle (kanonisch)

Nach Abschluss immer zuerst hier lesen:

- `logs/pytest_safe.status`
- `logs/pytest_safe.mini_report.txt`

Statusdatei ist die primaere Quelle fuer:

- `classification=PASS|FAIL|TIMEOUT|...`
- `exit_code=...`
- `args=...` (wichtig fuer Profilabgleich)

### Profilabgleich vor Bewertung

Vor einer Freigabe immer pruefen, ob das ausgewertete Ergebnis zum gewollten Profil passt:

- identische Gate-Datei
- identische Timeout-Parameter
- identisches Memory-/Swap-Profil

Nur wenn `args` in `pytest_safe.status` dem Zielprofil entspricht, gilt der Lauf als gueltiger Nachweis.

### Empfohlene Praxis fuer Competitive

1. Standardprofil aus Task laufen lassen
2. Bei reinem Timeout-Fail: explizit ein zweites Profil mit hoeherem Zeitbudget fahren
3. Ergebnis immer mit Profil kennzeichnen (z. B. "Standard" vs. "Erhoehtes Budget")

Dadurch wird verhindert, dass ein alter FAIL oder PASS aus einem anderen Profil falsch interpretiert wird.

---

## Tests schreiben

### 1. Unit Test Beispiel

```python
# tests/unit/test_denoiser.py
import pytest
import numpy as np
from backend.denoiser import AdaptiveDenoiser


class TestAdaptiveDenoiser:
    """Unit Tests für AdaptiveDenoiser"""

    @pytest.fixture
    def denoiser(self):
        """Denoiser Fixture"""
        return AdaptiveDenoiser()

    @pytest.fixture
    def audio_with_noise(self):
        """Test Audio mit weißem Rauschen"""
        sr = 48000
        duration = 3.0
        audio = np.random.randn(int(sr * duration)) * 0.1  # Noise
        return audio, sr

    def test_denoise_reduces_noise_floor(self, denoiser, audio_with_noise):
        """Test: Denoising reduziert Noise Floor"""
        # Arrange
        audio, sr = audio_with_noise
        noise_floor_before = np.std(audio)

        # Act
        denoised = denoiser.process(audio, sr, strength=0.5)

        # Assert
        noise_floor_after = np.std(denoised)
        assert noise_floor_after < noise_floor_before
        assert noise_floor_after < 0.05  # < -26 dB

    @pytest.mark.parametrize("strength", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_denoise_strength_parameter(self, denoiser, audio_with_noise, strength):
        """Test: Denoise Strength Parameter"""
        # Arrange
        audio, sr = audio_with_noise

        # Act
        denoised = denoiser.process(audio, sr, strength=strength)

        # Assert
        assert denoised.shape == audio.shape
        assert not np.isnan(denoised).any()
        assert not np.isinf(denoised).any()
```

**Struktur (Arrange-Act-Assert):**

1. **Arrange:** Setup (Audio, Parameter)
2. **Act:** Funktion aufrufen
3. **Assert:** Ergebnis prüfen

---

### 2. Integration Test Beispiel

```python
# tests/integration/test_pipeline_flow.py
import pytest
import numpy as np
from backend.api.bridge import get_aurik_denker_instance


@pytest.mark.integration
class TestPipelineFlow:
    """Integration Tests für vollständige Pipeline"""

    @pytest.fixture
    def denker(self):
        """AurikDenker Fixture"""
        return get_aurik_denker_instance()

    @pytest.fixture
    def vinyl_audio(self):
        """Vinyl Audio Fixture mit Clicks/Crackle"""
        sr = 48000
        duration = 3.0
        # Sine Wave + Clicks + Crackle
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz

        # Add Clicks
        click_positions = [sr // 2, sr, sr * 2]
        for pos in click_positions:
            audio[pos] = 1.0

        # Add Crackle
        crackle = np.random.randn(audio.shape[0]) * 0.02
        audio += crackle

        return audio, sr

    def test_restoration_mode_full_pipeline(self, denker, vinyl_audio):
        """Test: RESTORATION Mode vollständige Pipeline"""
        # Arrange
        audio, sr = vinyl_audio

        # Act
        result = denker.denke(audio, sr, mode="restoration")
        restored = result.audio

        # Assert
        assert restored.shape == audio.shape
        assert restored.dtype == np.float32
        assert not np.isnan(restored).any()
        assert not np.isinf(restored).any()

        # Verify Noise Reduction
        noise_floor_before = np.std(audio[-sr:])  # Last second
        noise_floor_after = np.std(restored[-sr:])
        assert noise_floor_after < noise_floor_before

    @pytest.mark.slow
    def test_all_modes_produce_valid_output(self, denker, vinyl_audio):
        """Test: Alle Modi produzieren valides Audio"""
        audio, sr = vinyl_audio

        modes = [
            "restoration",
            "studio2026",
        ]

        for mode in modes:
            # Act
            restored = denker.denke(audio, sr, mode=mode).audio

            # Assert
            assert restored.shape == audio.shape
            assert restored.dtype == np.float32
            assert not np.isnan(restored).any(), f"NaN in {mode}"
            assert not np.isinf(restored).any(), f"Inf in {mode}"
```

---

### 3. E2E Test Beispiel

```python
# tests/test_e2e_magicbutton.py
import pytest
import numpy as np
import soundfile as sf
from pathlib import Path
from backend.api.bridge import get_aurik_denker_instance


@pytest.mark.e2e
class TestMagicButtonE2E:
    """E2E Tests für Magic Button (vollständige User-Journey)"""

    @pytest.fixture(scope="class")
    def test_audio_dir(self):
        """Test Audio Directory"""
        return Path("test_audio")

    @pytest.fixture(scope="class")
    def output_dir(self):
        """Output Directory"""
        output = Path("test_output")
        output.mkdir(exist_ok=True)
        return output

    def test_magic_button_restoration_vinyl(self, test_audio_dir, output_dir):
        """Test: Magic Button - Vinyl Restoration (Full E2E)"""
        # Arrange
        input_file = test_audio_dir / "vinyl_sample.wav"
        output_file = output_dir / "vinyl_restored.wav"

        if not input_file.exists():
            pytest.skip("Test audio file not found")

        audio, sr = sf.read(input_file)
        denker = get_aurik_denker_instance()

        # Act: Magic Button (RESTORATION)
        restored = denker.denke(audio, sr, mode="restoration").audio

        # Save Output
        sf.write(output_file, restored, sr)

        # Assert
        assert output_file.exists()

        # Verify File Properties
        restored_audio, restored_sr = sf.read(output_file)
        assert restored_sr == sr
        assert restored_audio.shape == audio.shape

        # Verify Audio Quality (automated checks)
        # 1. No Clipping
        assert np.abs(restored_audio).max() <= 1.0

        # 2. No Silence (should have signal)
        rms = np.sqrt(np.mean(restored_audio**2))
        assert rms > 0.01  # -40 dB

        # 3. No DC Offset
        dc_offset = np.mean(restored_audio)
        assert abs(dc_offset) < 0.01
```

---

### 4. Test mit Fixtures (pytest.ini)

**pytest.ini:**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

markers =
    unit: Unit tests (fast, isolated)
    integration: Integration tests (multiple components)
    e2e: End-to-end tests (full pipeline)
    slow: Slow tests (> 30s)
    gpu: Requires GPU (CUDA)
    requires_audio: Requires test audio files

addopts =
    --strict-markers
    --disable-warnings
    --tb=short
    -v

# Coverage
[coverage:run]
source = .
omit =
    tests/*
    setup.py
    */__pycache__/*

[coverage:report]
precision = 2
show_missing = True
skip_covered = False
```

---

## Test-Coverage

### 1. Coverage Report generieren

```bash
# Mit Coverage-Report
pytest --cov=. --cov-report=html

# Nur Coverage-Prozentsatz
pytest --cov=. --cov-report=term

# Coverage für spezifisches Modul
pytest --cov=backend --cov-report=html
```

**Output:**

```text
Name                           Stmts   Miss  Cover
-------------------------------------------------
backend/__init__.py               12      0   100%
backend/denoiser.py              245     18    93%
backend/click_remover.py         189     12    94%
core/unified_restorer_v2.py      567     48    92%
...
-------------------------------------------------
TOTAL                           8924    712    92%
```

---

### 2. HTML Coverage Report

```bash
pytest --cov=. --cov-report=html

# Open in Browser
firefox htmlcov/index.html
```

**Vorteile:**

- Zeige welche Zeilen nicht getestet wurden (rot)
- Visualisiere Coverage-Gaps
- Navigiere durch Projektstruktur

---

### 3. Coverage-Ziele

| Modul | Ziel | Aktuell |
| --- | --- | --- |
| `core/` | >90% | 92% ✅ |
| `backend/` | >85% | 88% ✅ |
| `dsp/` | >80% | 83% ✅ |
| `enhancement/` | >80% | 76% ⚠️ |
| `forensics/` | >75% | 71% ⚠️ |

**Priorität:** Erhöhe Coverage in `enhancement/` und `forensics/`

---

## E2E-Testing

### 1. Magic Button E2E Tests

**Zweck:** Vollständige User-Journey testen

```bash
# Run Magic Button E2E Tests
python run_e2e_magic_button_tests.py

# Oder via pytest
pytest tests/test_e2e_magicbutton.py -v
```

**Was wird getestet:**

1. Audio einlesen (WAV/FLAC/MP3)
2. Bridge-/Denker-Pfad initialisieren
3. `AurikDenker.denke()` mit `restoration` oder `studio2026`
4. Output speichern
5. Output validieren (Clipping, Silence, DC-Offset)

---

### 2. Quick E2E Test

```bash
# Quick E2E Test (1 Audio-File, <1 Minute)
python run_quick_e2e_test.py
```

**Output:**

```text
=== Quick E2E Test ===
✅ Input: test_audio/vinyl_sample.wav (3.2s, 48kHz)
⏳ Processing with RESTORATION mode...
✅ Output: test_output/vinyl_restored.wav
✅ No Clipping detected
✅ RMS: -18.2 dB (Good)
✅ DC Offset: 0.0001 (OK)
=== Test PASSED ===
```

---

### 3. Batch E2E Tests

```bash
# Batch Test (alle Test-Audio-Dateien)
bash run_e2e_magic_button_tests.sh
```

**Getestete Medien:**

- Vinyl (Clicks, Crackle)
- Cassette (Hiss, Dropout)
- CD (Clean)
- Speech (Vocal)
- Live Recording (Crowd Noise)

---

## Performance-Testing

### 1. Processing Time Benchmarks

```python
# tests/performance/test_benchmark_processing.py
import pytest
import time
import numpy as np
from backend.api.bridge import get_aurik_denker_instance


@pytest.mark.benchmark
class TestProcessingBenchmarks:
    """Performance Benchmarks"""

    def test_restoration_processing_time_cpu(self):
        """Benchmark: RESTORATION Mode (CPU + optionale AMD-GPU)"""
        # Arrange
        sr = 48000
        duration = 180.0  # 3 minutes
        audio = np.random.randn(int(sr * duration)) * 0.1
        denker = get_aurik_denker_instance()

        # Act (measure time)
        start = time.time()
        restored = denker.denke(audio, sr, mode="restoration").audio
        elapsed = time.time() - start

        # Assert
        realtime_factor = duration / elapsed
        print(f"Processing Time: {elapsed:.2f}s ({realtime_factor:.2f}x realtime)")

        # CPU should be >0.3x realtime (i7-10700K: ~3x)
        assert realtime_factor > 0.3

    @pytest.mark.gpu
    def test_restoration_processing_time_gpu(self):
        """Benchmark: RESTORATION Mode (GPU)"""
        # ... similar to CPU test ...
        # GPU should be >1.0x realtime (RTX 3090: ~5-8x)
        assert realtime_factor > 1.0
```

**Run Benchmarks:**

```bash
pytest tests/performance/ -v
```

---

### 2. Memory Usage Testing

```python
import tracemalloc


def test_memory_usage_within_limits():
    """Test: Memory Usage < 2 GB"""
    # Start Memory Tracking
    tracemalloc.start()

    # Process Audio
    sr = 48000
    duration = 180.0
    audio = np.random.randn(int(sr * duration))
    denker = get_aurik_denker_instance()
    restored = denker.denke(audio, sr, mode="restoration").audio

    # Check Memory
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak / (1024**2)
    print(f"Peak Memory: {peak_mb:.2f} MB")

    # Assert < 2 GB
    assert peak_mb < 2048
```

---

## CI/CD-Integration

### 1. GitHub Actions (Beispiel)

```yaml
# .github/workflows/tests.yml
name: Tests

on:
  push:
    branches: [ main, development ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python 3.11
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install Dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run Tests
      run: |
        pytest --maxfail=1 --disable-warnings --cov=. --cov-report=xml

    - name: Upload Coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

---

### 2. Pre-Commit Hooks

```bash
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest-quick
        name: pytest-quick
        entry: pytest tests/unit -q
        language: system
        pass_filenames: false
        always_run: true
```

**Installation:**

```bash
pip install pre-commit
pre-commit install
```

**Effekt:** Vor jedem Commit werden schnelle Unit-Tests ausgeführt

---

## Best Practices

### 1. Test-Naming

**Konventionen:**

```python
# ✅ GOOD
def test_denoiser_reduces_noise_floor(): ...


def test_click_remover_preserves_transients(): ...


# ❌ BAD
def test1(): ...


def test_stuff(): ...
```

**Regel:** `test_<component>_<behavior>_<condition>`

---

### 2. Fixtures vs. Setup/Teardown

**Prefer Fixtures:**

```python
# ✅ GOOD (Fixtures)
@pytest.fixture
def denoiser():
    return AdaptiveDenoiser()


def test_denoise(denoiser):
    result = denoiser.process(audio, sr)
    assert ...


# ❌ BAD (Setup/Teardown)
class TestDenoiser:
    def setup_method(self):
        self.denoiser = AdaptiveDenoiser()

    def test_denoise(self):
        result = self.denoiser.process(audio, sr)
        assert ...
```

---

### 3. Parametrized Tests

**Teste mehrere Werte gleichzeitig:**

```python
@pytest.mark.parametrize(
    "mode",
    [
        "restoration",
        "studio2026",
    ],
)
def test_all_modes_produce_valid_output(mode):
    denker = get_aurik_denker_instance()
    restored = denker.denke(audio, sr, mode=mode).audio
    assert restored.shape == audio.shape
```

---

### 4. Test-Isolation

**Regel:** Jeder Test muss unabhängig sein

```python
# ✅ GOOD (Isolated)
def test_denoiser_strength_05():
    denoiser = AdaptiveDenoiser()  # Fresh instance
    result = denoiser.process(audio, sr, strength=0.5)
    assert ...


def test_denoiser_strength_10():
    denoiser = AdaptiveDenoiser()  # Fresh instance
    result = denoiser.process(audio, sr, strength=1.0)
    assert ...


# ❌ BAD (Shared State)
denoiser = AdaptiveDenoiser()  # Module-level


def test_denoiser_strength_05():
    result = denoiser.process(audio, sr, strength=0.5)
    assert ...


def test_denoiser_strength_10():
    # May fail if previous test modified state!
    result = denoiser.process(audio, sr, strength=1.0)
    assert ...
```

---

### 5. Assertions

**Spezifische Assertions:**

```python
# ✅ GOOD
assert restored.shape == audio.shape
assert not np.isnan(restored).any()
assert np.abs(restored).max() <= 1.0

# ❌ BAD
assert restored is not None  # Too vague
assert len(restored) > 0  # Too general
```

---

### 6. Test-Dauer

**Ziele:**

- Unit Tests: < 1s pro Test
- Integration Tests: < 10s pro Test
- E2E Tests: < 60s pro Test

**Markiere langsame Tests:**

```python
@pytest.mark.slow
def test_full_restoration_pipeline():
    # This takes 90 seconds
    ...
```

**Skippe in CI:**

```bash
pytest -m "not slow"  # Skippe langsame Tests
```

---

## Weitere Informationen

- **Contributing Guide:** [CONTRIBUTING.md](CONTRIBUTING.md)
- **Python API:** [PYTHON_API.md](../api/PYTHON_API.md)
- **Installation:** [INSTALLATION.md](../guides/INSTALLATION.md)

---

**© 2026 Aurik Audio Restoration System**  
**Version:** 8.0.0 | **Testing Guide**
