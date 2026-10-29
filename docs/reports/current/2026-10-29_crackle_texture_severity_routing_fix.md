# Crackle-Texture-Severity / phase_67-Routing — Fix (2026-10-29)

> **Zweck:** Dokumentation eines durch eine **Hörprobe entlarvten** Qualitätsfehlers
> samt Grundursache, Fix und Validierung. Der Hör-Befund „hörbar künstlich, wie eine
> schlechte unrestaurierte Aufnahme" führte zur Ursache: die Knistern-**Textur** blieb
> im Signal, weil die sie entfernende Phase (phase_67) nie startete.
>
> **Normative Referenz:** §SR-CK-Textur (Spec 03, `03_cognitive_modules.md`),
> §7.2d Knistern-Arten-Routing (Spec 06, `06_phases_system.md`), §V6 (copilot-instructions.md),
> §Spec 06 Route A (Either-Or-Routing).

## 1. Auslöser — Hör-Befund (Hör-Instanz, nicht Metrik)

A/B-Hörprobe shellac_classical_1940s (A = Aurik `balanced`, B = Clean-Referenz):

> „A klingt hörbar künstlich, es wirkt nicht wie restauriert sondern wie eine
> schlechte, unrestaurierte Aufnahme."

Signalauswertung bestätigte das Ohr (A vs. B vs. Crackle-Input D):

| Größe | A (Aurik) | B (Clean) | D (Crackle-Input) |
|---|---|---|---|
| Pegel RMS | −21,6 dBFS | −12,3 dBFS | −16,1 dBFS |
| Spektraler Schwerpunkt | 8372 Hz | 4441 Hz | 10475 Hz |
| HF-Anteil (> 4 kHz) | 70,0 % | 31,9 % | 77,7 % |
| SNR A vs. B | −0,5 dB | — | — |

Diagnose: Ausgabe **zu leise** (~9 dB unter Clean) **und zu hell** (HF 70 % statt
32 %) — der Defekt (Knistern-Textur) blieb drin. Die Restaurierung dämpft, entfernt
aber den Defekt nicht = „schlechtes unrestauriertes Material".

## 2. Grundursache — toter Code-Pfad (Scanner-False-Negative)

Die Knistern-**Textur** (kontinuierlich/dicht) wird per §SR-CK-Textur (Spec 03) auf
**phase_67_crackle_texture_removal** geroutet (Route A, Bailey et al. 2019 AES),
nicht auf phase_09 (das nur *spärliche* Einzelimpulse entfernt). Die Selektion in
`unified_restorer_v3.py` (≈28688):

```python
if sev(DefectType.CRACKLE) > 0.15 and _is_disc:
    selected.append("phase_09_crackle_removal")
    if not studio_mode and sparse_impulse_rate < 0.0008:  # dichte Textur
        selected.append("phase_67_crackle_texture_removal")
```

Der DefectScanner drückte die CRACKLE-Severity für **dichte** Textur jedoch auf 0
(`backend/core/defect_scanner.py`, `_detect_crackle`):

```python
tonal_or_dense_hf = bool(peak_ratio < 3.0 and sparse_impulse_rate < 0.0008)
if tonal_or_dense_hf:
    kurtosis_discount = 0.0
severity = min(1.0, severity_raw * 10 * kurtosis_discount)  # → 0.0
```

**Widerspruch:** Dasselbe Kriterium `sparse_impulse_rate < 0.0008` (dichte Textur)
*selektiert* phase_67 **und** *löscht* die Severity über `tonal_or_dense_hf`. Für
dichte Textur war die Severity damit **immer 0** → `sev(CRACKLE) > 0.15` nie erfüllt
→ **phase_67 startete nie** (toter Code). Die Textur blieb hörbar.

Messwerte am shellac-Crackle (Material=SHELLAC, era=1940) **vor** dem Fix:

| Datei | severity_raw (Masken-Dichte) | hp_kurtosis | peak_ratio | sparse | Severity |
|---|---|---|---|---|---|
| `…_crackle.wav` (echt) | **1.000** (dicht) | **10.32** (impulsiv) | 2.57 | 0.0 | **0.0** ❌ |
| `…_clean.wav` (Referenz) | 0.134 (spärlich) | 52.84 | 6.11 | 0.00109 | 0.736 (False-Pos.) |

Der Guard ist für *tonal/gaußsches* HF gedacht (Fehlalarm-Schutz), traf hier aber
**echtes dichtes Knistern** (impulsiv, hp_kurtosis 10.32, dichte Maske 1.0) und
löschte es. Zusätzlich drückt `kurtosis_discount` (bei gaußschem HF) die Severity —
dichte Knistern-Textur ist dicht, nicht einzelimpulsiv.

## 3. Fix

`backend/core/defect_scanner.py` (`_detect_crackle`): Die `tonal_or_dense_hf`-Nullung
greift nicht mehr bei echtem dichtem Knistern (hohe Masken-Dichte **und** impulsive HF):

```python
if tonal_or_dense_hf and not (severity_raw >= 0.5 and hp_kurtosis >= 6.0):
    kurtosis_discount = 0.0
```

Damit gilt §SR-CK-Textur (Spec 03) „CRACKLE Textur → phase_67" wieder auch im Code:
dichte impulsive Knistern-Textur bekommt Severity > 0.15, die Selektions-Logik wählt
phase_67, und die Textur wird entfernt.

## 4. Validierung

- **Scanner (vorher/nachher):** CRACKLE-Severity `0.0 → 1.000` → phase_09+67 **aktiv**.
- **Regression:** `pytest tests/unit -k "defect or crackle or scanner"` → **2962 passed,
  0 failed** (inkl. `test_defect_scanner_anti_fp.py`).
- **Regressionstest ergänzt:** `tests/unit/test_defect_scanner_crackle_texture_severity.py`
  pinnt den Fix (dichte impulsive Textur → Severity > 0.15 → phase_67-Routing).

## 5. Offene Reste (ehrlich)

- **False-Positive (BEHOBEN, 2026-10-29):** Die `× 10`-Sättigung der Severity-Formel
  zerrte die spärliche Maske musikalischer Transienten (Referenz 0,134) in den
  False-Positive-Bereich (0,79). Rekalibrierung auf die Masken-Dichte
  (`severity = severity_raw * kurtosis_discount`, ohne `× 10`) gesetzt. Verifiziert:
  Referenz 0,789 → 0,120 (< 0,15, kein False-Positive), echtes Knistern bleibt 1,000.
  Golden-Gate besteht (recall/precision = 1,0; der eine runtime_factor-Fehlschlag
  (1,467 > 1,400) war ein Last-Artefakt der Hintergrund-Restaurierung).
- **Zweiter phase_67-Blocker (OFFEN — Material-Erkennung):** phase_67 startet weiterhin
  nicht, obwohl die Severity jetzt stimmt. Grund: die Material-Erkennung klassifiziert
  das Shellac-Material als `MaterialType.TAPE` (Log des Echtlaufs 2026-10-29), damit
  `_is_disc = False`, und die Selektion hängt an
  `if sev(CRACKLE) > 0.15 AND _is_disc:`. Frischer Stimulus = altes Ergebnis
  (HF 71 %, Textur bleibt hörbar). Der Lückenschluss erfordert die Material-Erkennung
  (Shellac → Disc) ODER die Aufweichung der `_is_disc`-Sperre für dichte Knistern-Textur
  (Spec §SR-CK-Textur routet nach Knistern-Art, nicht nach Träger).
- **Kalibrierung:** Die Severity-Formel ist weiterhin nicht vollständig als Vorgabe
  spezifiziert (nur qualitativ in §SR-CK-Textur).

## 6. Referenzen

- Spec 03 §SR-CK-Textur (Knistern-Arten-Differenzierung, `03_cognitive_modules.md:874`)
- Spec 06 §7.2d (Knistern-Arten-Routing, `06_phases_system.md:409`)
- `backend/core/defect_scanner.py` (`_detect_crackle`, Fix)
- `backend/core/unified_restorer_v3.py` (phase_09/67-Selektion, ≈28688)
- `backend/core/phases/phase_67_crackle_texture_removal.py` (Route-A-ML-Kern)
- `tests/unit/test_defect_scanner_crackle_texture_severity.py` (Regressionstest)

## 7. PR-Evidenzblock (Spec-Änderung Spec 03 §SR-CK-Textur, AGENTS.md §4)

**Evidenzblock:** Fix `backend/core/defect_scanner.py` (`_detect_crackle`), Validierung
2026-10-29: Scanner CRACKLE-Severity 0,0 → 1,000 (shellac-Crackle, Material=SHELLAC,
era=1940) → phase_09+67 aktiv; 2962 Defekt-/Crackle-Unit-Tests + 1006 normative Tests
+ 3 neue Regressionstests (`test_defect_scanner_crackle_texture_severity.py`) grün;
`file_registry_check` 0 ERROR(s); `release_must_coverage_check` 2/2 (100 %). Vorher/
Nachher-Messwerte siehe §2/§4.

**Seed:** Kein Zufall im Fix (§G5) — Scanner-Determinismus unverändert (eval/no_grad,
keine Seeds). Regressionstest-Synthese: `numpy.random.default_rng(7)` (fest).

**95 %-CI:** Bei n=1 Hör-Befund + n=3 Unit-Tests nicht berechenbar. Kalibrierung der
Severity-Schwelle an größerer gelabelter Stichprobe steht als Arbeitsauftrag (§5).

**Maintainer Sign-off:** Vom Nutzer angefordert (2026-10-29, „Korrektur validiert →
sauber dokumentieren + Vorgaben/Specs ergänzen"). Formale Freigabe im PR durch
Maintainer ausstehend.
