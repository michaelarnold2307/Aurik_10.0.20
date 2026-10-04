# Bugfix: file_ext-Prior & SongCalibration & PhasePruner — 2026-07-09

## Executive Summary

Vier Bugs in Kaskade führten dazu, dass MP3-Imports fälschlich als analoges
Material klassifiziert wurden, eine leere Transfer-Chain produzierten, mit starr
neutralem `global_scalar=1.000` liefen und zu wenige Phasen ausführten:

1. `@staticmethod` zerstörte `self` → `global_scalar=1.000` (unified_restorer_v3.py)
2. `input_path` fehlte im REST-API-Endpoint → kein Digital-Prior (batch_endpoints.py)
3. `cached_cached_era_result` Tippfehler (aurik_denker.py)
4. **PhasePruner: 76% der Defekt-Namen existierten nicht im Scanner**
   → Phasen fälschlich geprunt (phase_pruner.py — siehe separates Dokument)

## Gefundene Bugs

### Bug 1: `@staticmethod` auf `_build_song_calibration_profile`
- **Datei:** `backend/core/unified_restorer_v3.py:2183`
- **Symptom:** `TypeError: takes 0 positional arguments but 1 was given`
- **Ursache:** Methode hatte Signatur `def _build_song_calibration_profile(*, material_type: ...)` — der `*` machte alle Parameter keyword-only, aber `self` fehlte vor dem `*`.
- **Kaskade:** `NameError` → Fallback → `global_scalar=1.000` → keine adaptive Dämpfung → Over-Processing ohne Rücksicht auf Materialqualität.
- **Fix:** `@staticmethod` entfernt, `self` vor `*` eingefügt: `def _build_song_calibration_profile(self, *, material_type: ...)`.

### Bug 2: `input_path` fehlt im REST-API-Batch-Endpoint
- **Datei:** `backend/api/rest/batch_endpoints.py:68`
- **Symptom:** `denker.denke(audio, sr, mode="restoration")` — ohne `input_path`.
- **Ursache:** `in_path` war verfügbar (wird für `load_audio_file` genutzt), wurde aber nicht an `denke()` durchgereicht.
- **Kaskade:** Ohne `input_path` → `file_ext=""` → die Digital-Extension-Prüfung in `MediumDetector.detect()` (`_ext_lower in _DIGITAL_FILE_EXTS`) evaluiert zu `False` → analoge Bayesian-Posteriors werden **nicht** mit ×0.25 bestraft → physische Merkmale können fälschlich analoge Träger (vinyl, tape) klassifizieren → leere/fehlerhafte `transfer_chain` → Fallback auf `reel_tape` → Tonband-Phasen auf MP3 aktiv.
- **Fix:** `input_path=str(in_path)` zum Aufruf hinzugefügt.

### Bug 3: `cached_cached_era_result` → `cached_era_result` (bereits gefixt)
- **Datei:** `denker/aurik_denker.py`
- **Symptom:** Doppeltes Präfix `cached_cached_` im Variablennamen.
- **Ursache:** Copy-Paste-Fehler bei Refactoring.
- **Fix:** Variable umbenannt.

## File-Extension-Propagation-Chain (vollständig)

```
Batch-Endpoint / CLI / GUI
  │
  ├─ input_path = "/path/to/song.mp3"    ← NOW FIXED (Bug 2)
  │
  └─ denke(input_path="song.mp3")
       └─ restauriere(input_path="song.mp3")
            └─ _denke_impl(input_path="song.mp3")
                 │
                 ├─ TontraegerDenker.erkenne(file_path="song.mp3")
                 │    └─ _file_ext = os.path.splitext(file_path)[1]  → ".mp3"
                 │         └─ MediumDetector.detect(file_ext=".mp3")
                 │              ├─ _ext_lower = ".mp3"
                 │              ├─ _ext_lower in _DIGITAL_FILE_EXTS  → True ✅
                 │              └─ analog posteriors × 0.25 penalty
                 │
                 └─ TontraegerketteDenker.analysiere(file_path="song.mp3")
                      └─ _file_ext = ".mp3"
                           └─ MediumDetector.detect(file_ext=".mp3")  ✅
```

### Call-Site-Audit (alle 5 Einstiegspunkte)

| Call-Site | `input_path` | Status |
|---|---|---|
| `cli/aurik_cli.py:534` | `input_path=input_path` | ✅ OK |
| `batch_processor.py:256` | `input_path=str(input_file)` | ✅ OK |
| `_aurik_run_excellence.py:225` | `input_path=str(input_path)` | ✅ OK |
| `Aurik10/ui/ml_refinement_thread.py:236` | `input_path=job.input_path` | ✅ OK |
| `backend/api/rest/batch_endpoints.py:68` | Fehlte → `input_path=str(in_path)` | 🔧 GEFIXT |

### Digital-Extension-Prior (MediumDetector)

```python
_DIGITAL_FILE_EXTS = frozenset({".mp3", ".mp2", ".aac", ".m4a", ".ogg", ".opus", ".mpc", ".wma"})
_ANALOG_PENALTY = 0.25  # ×0.25 auf analoge Bayesian-Posteriors
```

**Wichtig:** Lossless-Container (`.wav`, `.flac`, `.aiff`) werden NICHT als
digital-bestätigt behandelt — sie sind neutrale Storage-Container, die häufig
für Vinyl-Rips und Tape-Transfers genutzt werden (§6.7b).

## SongCalibration: Erwartete Werte nach Fix

| Material | `global_scalar` | Begründung |
|---|---|---|
| Shellac 1940, SNR=8dB, def=0.9 | **0.75** | Starke Dämpfung, Preservation Mode |
| MP3 2005, SNR=35dB, def=0.25 | **0.95** | Leichte Dämpfung, Codec-Artefakte |
| CD 2010, SNR=55dB, def=0.02 | **0.95** | Nahe neutral, kaum Defekte |

Bereich: `[0.50, 1.50]` gemäß Lücke-G-Fix v10.0.0.

## Empfehlungen

### 1. Regression-Test für `input_path`
```python
# In test_pipeline_integration.py oder neuem Test:
def test_file_ext_reaches_medium_detector():
    """Stellt sicher, dass input_path als file_ext im MediumDetector ankommt."""
    with patch("forensics.medium_detector.MediumDetector.detect") as mock_detect:
        mock_detect.return_value = make_fake_result()
        denker.denke(audio, sr, mode="restoration", input_path="/tmp/test.mp3")
        call_kwargs = mock_detect.call_args.kwargs
        assert call_kwargs.get("file_ext") == ".mp3", f"file_ext sollte '.mp3' sein, war: {call_kwargs.get('file_ext')}"
```

### 2. Assertion-Guard in `_denke_impl`
```python
# In aurik_denker.py, vor TontraegerDenker-Aufruf:
if input_path and not os.path.splitext(input_path)[1]:
    logger.warning("input_path ohne Dateiendung: %s — MediumDetector bekommt keinen Digital-Prior", input_path)
```

### 3. Kein `@staticmethod` auf Methoden mit >10 Parametern
Der `@staticmethod`-Dekorator auf `_build_song_calibration_profile` wurde
wahrscheinlich gesetzt, weil die Methode lange keinen `self`-Zugriff brauchte.
Als später `self._restoration_context` hinzukam, wurde der Dekorator nicht
entfernt. Empfehlung: Linter-Regel, die `@staticmethod` auf Methoden mit
`self.X`-Zugriffen flaggt.

### 4. Canary-Log für `file_ext=""`
Wenn `MediumDetector.detect()` mit `file_ext=""` aufgerufen wird und die
Bayesian-Top-3 ein analoges Material enthalten, ein `logger.info` ausgeben:
„Kein file_ext — bei .mp3/.aac/.ogg würde Digital-Prior ×0.25 greifen."

## Betroffene Dateien

| Datei | Änderung |
|---|---|
| `backend/core/unified_restorer_v3.py` | `@staticmethod` entfernt, `self` eingefügt |
| `backend/api/rest/batch_endpoints.py` | `input_path=str(in_path)` hinzugefügt |
| `denker/aurik_denker.py` | `cached_cached_era_result` → `cached_era_result` |

## Datum

2026-07-09 — Analyse & Fix durch Kun
