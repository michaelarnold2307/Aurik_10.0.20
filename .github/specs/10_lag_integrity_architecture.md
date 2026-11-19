# Aurik 10 — Lag-Integritäts-Architektur (§G60–§G67, §V27–§V33)

> **Erkenntnisse aus der Lag-Root-Cause-Analyse (2026-07-13)**
> 13 Commits, 8 Root Causes gefunden und behoben.
> Diese Spec verhindert das Wiedereinschleichen.

---

## Gefundene Root Causes (Chronik)

### RC1: STCG Mid-Window misst variablen Lag nicht

**Symptom**: LAG_PROBE_0B → korrigiert zu 0, LAG_PROBE_1 → wieder -8900
**Ursache**: STCG maß nur 10s Mid-Window. Bei 0%→-8900, 50%→0, 100%→-7297 sah Mid-Window 0 und korrigierte nichts.
**Fix**: STCG verwendet Multi-Point-Median (3 Positionen) als PRIMÄRE Messung.
**Commits**: `0ab8a7f1`, `b0132490`

### RC2: Phase 12 xcorr-Fallback überschrieb STCG

**Symptom**: Trotz STCG-Korrektur war Lag nach Phase 12 wieder da.
**Ursache**: `_preserve_phase_loudness` führte NACH erfolgreichem STCG einen Onset-Energy-Fallback durch, der die Sub-Sample-Korrektur mit grobem `np.concatenate` überschrieb.
**Fix**: Fallback nur bei STCG-Exception (`_stcg_applied`-Flag).
**Commit**: `785fa2c0`

### RC3: LAG_PROBE_0B nutzte np.roll

**Symptom**: Zirkuläre Audio-Artefakte, Längenänderung.
**Ursache**: Lag-Korrektur verwendete `np.roll` (zirkulär) + Audio-Trunkierung statt `scipy.ndimage.shift`.
**Fix**: STCG-Aufruf mit Sub-Sample-Präzision.
**Commit**: `8a127caf`

### RC4: Multi-Point-Funktion ignorierte Channels-First

**Symptom**: `LAG_PROBE 0B: points=[n/a]` – leere Messung.
**Ursache**: `_estimate_interchannel_lag_multi_point` nahm `arr.shape[0]` als Sample-Anzahl. Bei (2,N) war das 2.
**Fix**: Orientierungserkennung analog Single-Point-Funktion.
**Commit**: `b0132490`

### RC5: Phase 24 nutzte signal.correlate (primitiv)

**Symptom**: Lag-Deltas bis 1303 Samples, aber max Suchraum 960 – Lags >20ms unsichtbar.
**Ursache**: Phase 24 hatte eigene Lag-Erkennung mit `signal.correlate` (kein GCC-PHAT, kein Whitening, ganzzahliger Shift).
**Fix**: Vollständiger Ersatz durch STCG (Multi-Point, GCC-PHAT, ±200ms, cubic spline).
**Commit**: `81e2762a`

### RC6: STCG 20ms Pre-Pipeline-Guard

**Symptom**: 185ms Source-Lag wurde als "False-Positive" verworfen.
**Ursache**: Guard verwarf Lags >20ms pauschal ohne Verifikation.
**Fix**: Multi-Point-Verifikation ersetzt den Blind-Guard.
**Commit**: `b0132490`

### RC7: Soft-Saturation pauschal preserved

**Symptom**: Generation-Loss-Saturation (Kassette, MP3) wurde als "künstlerisch" eingestuft.
**Ursache**: Genre-Profil + Fallback setzten `soft_saturation_preserve=True` ohne Chain-Depth-Prüfung.
**Fix**: Chain-Depth-Guard + SaturationDiscriminator (H2/H3/H5-Signalanalyse).
**Commits**: `ac251532`, `8648b759`, `d07167f8`

### RC8: Scipy-STFT-Warnungen (62+ ungeschützte Aufrufer)

**Symptom**: "nperseg=2048 > input length=2" flutete das Log.
**Ursache**: 62+ `scipy.signal.stft`-Aufrufe ohne Längen-Guard in 30+ Dateien.
**Fix**: Zentraler Guard in `backend/__init__.py` (patched `scipy.signal.stft`).
**Commit**: `4b756bc8`

---

## Neue GEBOTE — Kategorie VII: Stereo-Lag-Integrität (§G60–§G67)

| ID | Gebot | Begründung | Fundstelle |
|----|-------|-----------|------------|
| **§G60** | **STCG Multi-Point-Primär** | STCG MUSS Multi-Point (≥3 Song-Positionen) als PRIMÄRE Lag-Messmethode verwenden. Single-Mid-Window nur als Fallback bei Audio < 30s. | RC1: Mid-Window sah lag=0, aber Start/Ende hatten -8900/-7297 |
| **§G61** | **Chunk-Phasen-STCG-Pflicht** | Jede Chunk-basierte Phase (Phase 12, 24, 31) MUSS für Stereo-Lag-Korrektur ausschließlich den STCG verwenden. Eigene Korrelations-Implementierungen sind VERBOTEN (§V27). | RC5: Phase 24s signal.correlate erkannte 185ms-Lags nicht |
| **§G62** | **Sub-Sample-Lag-Korrektur** | Lag-Korrektur MUSS `scipy.ndimage.shift` (cubic spline) oder STCG verwenden. `np.roll`, `np.concatenate` und ganzzahlige Shifts sind VERBOTEN (§V32). | RC3: np.roll erzeugte zirkuläre Artefakte |
| **§G63** | **Lag-Messung orientierungsunabhängig** | Alle Lag-Messfunktionen MÜSSEN `(2,N)` und `(N,2)` korrekt erkennen. `arr.shape[0]` als implizite Sample-Anzahl ist VERBOTEN. | RC4: Multi-Point gab leere Ergebnisse bei Channels-First |
| **§G64** | **Keine konkurrierenden Lag-Korrekturen** | Nach einer erfolgreichen STCG-Korrektur darf KEINE zweite Lag-Korrektur (Onset-Energy, xcorr-Fallback) auf dasselbe Audio angewendet werden. Ausnahme: STCG warf Exception. | RC2: Phase 12s Fallback überschrieb STCG |
| **§G65** | **Lag-Messbereich ≥ ±200ms** | Jeder Lag-Suchraum MUSS mindestens ±200ms (±9600 samples @48kHz) abdecken. Kleinere Bereiche ignorieren reale Hardware-Versätze. | RC5: Phase 24 suchte nur ±20ms |
| **§G66** | **STFT-Längen-Guard zentral** | Jeder `scipy.signal.stft`-Aufruf MUSS durch den zentralen Guard in `backend/__init__.py` geschützt sein. Individuelle Guards sind redundant aber erlaubt. | RC8: 62+ ungeschützte Aufrufer |
| **§G67** | **Lag-Plausibilität nur mit Multi-Point** | Ein Lag >20ms darf NUR dann als False-Positive verworfen werden, wenn eine Multi-Point-Prüfung (≥2 Positionen) Inkonsistenz zeigt (spread > 50 samples). Blinde Schwellwerte sind VERBOTEN (§V33). | RC6: 185ms Source-Lag wurde pauschal verworfen |

---

## Neue VERBOTE — Kategorie Stereo-Lag (§V27–§V33)

| ID | Verbot | Begründung | Fundstelle |
|----|--------|-----------|------------|
| **§V27** | **Kein signal.correlate für Lag** | `signal.correlate` (Standard-Kreuzkorrelation ohne PHAT-Whitening) ist für Stereo-Lag-Messung UNZULÄSSIG. PHAT-Whitening eliminiert Periodenambiguitäten und ist Pflicht. | Phase 24: `_estimate_stereo_lag_samples` mit `signal.correlate` |
| **§V28** | **Kein begrenzter Lag-Suchraum** | Der Lag-Suchraum darf NICHT auf < ±200ms begrenzt werden. Echte Tonkopf-Dejustierungen erreichen 180+ms. | Phase 24: `max_lag_samples = 960` (20ms) |
| **§V29** | **Keine Zweitkorrektur nach STCG** | Nach erfolgreicher STCG-Korrektur darf KEINE unabhängige Lag-Korrektur erfolgen. STCG-Ergebnis ist bindend. | Phase 12: xcorr-Fallback nach STCG |
| **§V30** | **Kein Single-Window bei >30s** | Bei Audio >30s ist die Lag-Messung an NUR einer Position unzulässig. Minimum: 3 Positionen. | RC1: Mid-Window Blind-Spot |
| **§V31** | **Kein STFT ohne Längenprüfung** | `scipy.signal.stft` ohne vorherige `len(x) >= nperseg`-Prüfung ist verboten. | RC8 |
| **§V32** | **Kein np.roll für Stereo-Shift** | `np.roll` (zirkulär), `np.concatenate` (grob) und vergleichbare ganzzahlige Verschiebungen sind für Stereo-Lag-Korrektur VERBOTEN. | RC3 |
| **§V33** | **Kein blinder Lag-Schwellwert** | Ein pauschaler Schwellwert (z.B. 20ms) zum Verwerfen von Lag-Messungen ohne Multi-Point-Verifikation ist VERBOTEN. | RC6 |

---

## Linter-Regeln (automatisierte Durchsetzung)

Diese Regeln werden durch Pre-Commit-Hooks und den `scripts/aurik_verboten_linter.py` durchgesetzt:

### L1: STCG-Import-Check (→ §G61, §V27)

```python
# VERBOTEN:
import scipy.signal

lag = signal.correlate(l, r, mode="full")
# oder: from scipy.signal import correlate

# GEBOTEN:
from backend.core.stereo_temporal_coherence_guard import get_stereo_temporal_coherence_guard
```

### L2: np.roll-Stereo-Check (→ §G62, §V32)

```python
# Pattern: np.roll( ... audio[:, 1] ... ) oder np.roll( ... audio[1] ... )
# → WARNING: "np.roll für Stereo-Kanal-Shift – verwende STCG oder scipy.ndimage.shift"
```

### L3: Multi-Point-Pflicht (→ §G60, §V30)

```python
# Pattern: _estimate_interchannel_lag_samples(audio, sr)  # Single-Point
# → INFO: "Single-Point Lag-Messung – §G60 empfiehlt Multi-Point"
# Gilt NICHT für LAG_PROBE quick-checks (< 50ms Laufzeit)
```

### L4: STFT-Längen-Guard (→ §G66, §V31)

```python
# Pattern: signal.stft(x, ...) ohne vorheriges len(x) >= nperseg
# → WARNING: "scipy.signal.stft ohne Längen-Guard – §G66"
# Ausnahme: wenn von zentralem Guard in __init__.py geschützt
```

### L5: Orientierungs-Check (→ §G63)

```python
# Pattern: arr.shape[0] in Lag-Messfunktion ohne ndim-Check
# → WARNING: "shape[0] als implizite Sample-Anzahl – §G63"
```

---

## Test-Vorgaben

### T1: Lag-Regressionstest (Pflicht bei jeder STCG-/Phase-12/24-Änderung)

```python
def test_lag_varied_across_song():
    """Simuliert variablen Lag: 0%→-8900, 50%→0, 100%→-7297."""
    ...
    assert abs(multi["median_lag"]) < 50  # Median muss korrigiert sein
```

### T2: Orientierungs-Test

```python
def test_lag_both_orientations():
    """Test channels-first (2,N) und channels-last (N,2)."""
    ...
    assert result_cf == result_cl  # Gleiches Ergebnis
```

### T3: Kein np.roll in Lag-Pfad

```python
def test_no_np_roll_in_lag_correction():
    """Assert: kein Aufruf von np.roll mit audio[:, 1] im Lag-Korrektur-Pfad."""
    # Via AST-Analyse
```

---

## Änderungshistorie

| Version | Datum | Änderung |
|---------|-------|----------|
| 10.1.0 | 2026-07-13 | Initial: 8 Root Causes dokumentiert. §G60–§G67, §V27–§V33, L1–L5, T1–T3. |
