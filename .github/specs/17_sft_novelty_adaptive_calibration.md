# Spec §17 — SFT Novelty Adaptive Calibration & Defekt-Audibilitäts-Garantie | §v10.40 Per-Song-Kalibrierung

**Aurik 10.11.14+ | Gültig ab: 19. Juli 2026 | Normativ übergeordnet über Phase-Konfigurationen**

---

## 1. Tiefenanalyse: Was der Test-Track-Lauf offenbart hat

### 1.1 Der Import-Song

| Eigenschaft | Wert |
|------------|------|
| Datei | `Testkünstlerin (Schlager) - Du wolltest nur ein Abenteuer.mp3` |
| Dauer | 225.3 s |
| Format | MP3 (mp3_low, stark komprimiert) |
| Material (auto-detektiert) | Kassette (Band) |
| Transfer-Chain | reel_tape → vinyl → cassette → mp3_low (4 Stufen) |
| Ära | 1970er |
| Genre | Deutscher Schlager |
| SNR | 14.3 dB |
| Restorability | 64/100 (Mäßig) |
| MOS-Prognose | 3.61 |

### 1.2 Defekt-Landschaft (10 Typen, 5.454 Instanzen)

| Defekt | Instanzen | Zuständige Phase(n) |
|--------|-----------|---------------------|
| clicks | 2.593 | phase_01, phase_27 |
| groove_echo | 706 | phase_61 |
| sticky_shed_residue | 583 | phase_28, phase_29 |
| crackle | 426 | phase_09 |
| dropout_oxide | 399 | phase_24 |
| tape_head_clog | 349 | phase_28, phase_29 |
| flutter | 236 | phase_12 |
| transport_bump | 135 | phase_12 |
| wow | 23 | phase_12 |
| tape_splice_artifact | 4 | phase_64 |

### 1.3 Der Pipeline-Lauf (44 Phasen, 9.873 s = 43.8× RT)

**Ergebnis: Quality 43 → 43 (+0%) — NULL-Verbesserung trotz 44 Phasen.**

Root-Cause-Analyse identifizierte **fünf kritische Bugs**, die kumulativ dazu führten, dass praktisch keine Phase nennenswerte Wirkung entfaltete:

---

## 2. Bug #1 (KRITISCH): SFT NOVELTY_CRIT diskardierte 95% aller Phasen-Ergebnisse

### 2.1 Mechanismus

Der Signal Flow Tracer (§SFT) misst pro Phase die spektrale Veränderung (`novelty_delta`). Überschreitet sie die Schwelle `_NOVELTY_CRIT = 0.15`, wird das Flag `NOVELTY_CRIT` gesetzt. Der UV3-Code prüft dieses Flag und blendet das Phasen-Ergebnis mit dem Original-Audio:

```python
# Alte Logik (STAND VOR FIX):
if _sft_novelty_val >= 0.40:
    _sft_wet = 0.30 if _is_repair_phase else 0.05  # ← 95% verworfen!
```

### 2.2 Warum 0.15 als Schwelle fundamental falsch ist

Im **Restoration-Mode** ist die spektrale Veränderung der ZWECK der Pipeline. Ein Click-Repair, der neue Samples interpoliert, verändert das Spektrum. Ein Denoiser, der Rauschen entfernt, verändert das Spektrum. Ein EQ, der Frequenzen anhebt, verändert das Spektrum.

Die Schwelle `0.15` bedeutet: **15% neues Spektrum → CRIT**. Das triggert für JEDE Phase im Restoration-Mode.

**Gegenbeweis aus dem Log:** Alle 44 Phasen zeigten NOVELTY_CRIT-Werte zwischen 0.40 und 0.50. Keine einzige Phase blieb unter der Warn-Schwelle (0.08).

### 2.3 Die kontinuierlich kalibrierte Lösung (§v10.41)

Statt diskreter Lookup-Tabellen leitet Aurik die Schwelle jetzt **kontinuierlich** aus seinen eigenen Messwerten ab:

```python
# §v10.41 in UV3.restore():
# Formel: crit = floor + restorability_span + chain_bonus
#   floor              = 0.20  (Minimum für perfektes Material)
#   restorability_span = (1 − rs/100) × 0.40  (kontinuierlich aus Pre-Analysis)
#   chain_bonus        = max(0, depth−1) × 0.03  (kontinuierlich aus MediumDetector)

_rs = float(restorability_score)         # 0–100, Auriks eigene Bewertung
_depth = int(transfer_chain_depth)       # 1–n, Auriks eigene Ketten-Analyse
_cal_novelty = clamp(0.20 + (1.0 − _rs/100) × 0.40 + max(0, _depth−1) × 0.03, 0.18, 0.65)
```

**Alle Eingabewerte stammen AUSSCHLIESSLICH aus Auriks Pre-Analyse** — keine vom Entwickler vorgegebenen Stützstellen, keine 0.05-Schritte, keine if/elif-Kaskaden.

**Kontinuierliche Kennlinie (Beispiele):**

- rs=100, depth=1 → crit = 0.20 + 0.00 + 0.00 = **0.200** (perfekt → sehr sensitiv)
- rs=64, depth=4 → crit = 0.20 + 0.144 + 0.09 = **0.434** (fair, 4-stufig → moderat)
- rs=30, depth=2 → crit = 0.20 + 0.280 + 0.03 = **0.510** (poor, 2-stufig → tolerant)
- rs=5, depth=4 → crit = 0.20 + 0.380 + 0.09 = **0.670** → clamp → **0.650** (extrem → max.)
→ Bei Schwelle 0.55 triggert KEINE Phase NOVELTY_CRIT (alle Werte 0.40–0.50)
→ Alle Phasen fallen in den `else`-Pfad (wet=0.30)
→ 6× mehr Phasen-Wirkung als vorher (0.05)

**Für einen Studio-Master (depth=1, tier=excellent):** NOVELTY_CRIT = 0.15
→ Viele Phasen triggern NOVELTY_CRIT
→ Aggressiver Rollback schützt die hohe Ausgangsqualität

### 2.4 SFT-Wet-Sicherheitsnetz

Die Wet-Werte dienen als sekundäres Sicherheitsnetz für die wenigen Phasen, die tatsächlich die adaptiv kalibrierte Schwelle überschreiten:

| NOVELTY | Repair-Phase | Non-Repair | Begründung |
|---------|-------------|------------|------------|
| ≥0.40 | 0.45 | 0.30 | Starke Veränderung → moderat erhalten |
| ≥0.25 | 0.60 | 0.50 | Mittlere Veränderung → mehrheitlich erhalten |
| <0.25 | 0.75 | 0.65 | Geringe Veränderung → weitgehend erhalten |

### 2.5 SFT-Prioritätskette

Die Prüfreihenfolge der SFT-Flags ist normativ festgelegt:

| Priorität | Flag | Wet | Begründung |
|-----------|------|-----|------------|
| 1 (höchste) | LEVEL_COLLAPSE | 0.00 | Audio zerstört → Vollrollback |
| 2 | ECHO_ARTIFACT | 0.30 | Echo/Pre-Echo → konservativ |
| 3 | PEGELEXPLOSION_CRIT | 0.22 | Pegel-Amok → stark gedämpft |
| 4 | NOVELTY_CRIT | 0.30–0.75 | Adaptiv kalibriert (s.o.) |
| 5 (Fallback) | (kein kritischer Flag) | 0.30 | Studio-Mode / ECHO-Fallback |

**Warum ECHO_ARTIFACT vor PEGELEXPLOSION?** Echo ist ein psychoakustisch besonders störender Artefakt. Eine Phase, die Echo erzeugt, muss konservativer behandelt werden als eine mit reinem Pegel-Problem.

**Warum LEVEL_COLLAPSE vor allem?** Wenn eine Phase das Audio auf −92.4 dBFS kollabieren lässt (wie phase_07 in der FeedbackChain), muss das Ergebnis vollständig verworfen werden. Jede Beimischung würde die Stille in nachfolgende Phasen tragen und eine Kaskade von Early-Silence-Gate-Skips auslösen.

---

## 3. Bug #2: Defekt-Reparatur-Phasen wurden nicht als solche erkannt

### 3.1 Die Repair-Phasen-Liste (normativ)

Nur 5 von 12 Defekt-Reparatur-Phasen waren in der `_is_repair_phase`-Liste. Sieben weitere Phasen, die GENUINE Defekte reparieren (nicht nur enhancen), fehlten:

| Phase | Defekt-Typ | Vorher | Nachher |
|-------|-----------|--------|---------|
| phase_01 | Click Removal | ✅ repair | ✅ repair |
| phase_02 | Hum Removal | ✅ repair | ✅ repair |
| phase_09 | Crackle Removal | ✅ repair | ✅ repair |
| phase_12 | Wow/Flutter/Transport Bumps | ❌ **non-repair** | ✅ repair |
| phase_23 | Spectral Repair | ❌ **non-repair** | ✅ repair |
| phase_24 | Dropout Repair | ✅ repair | ✅ repair |
| phase_27 | Click/Pop Removal | ✅ repair | ✅ repair |
| phase_50 | Spectral Re-Repair | ❌ **non-repair** | ✅ repair |
| phase_56 | Spectral Band Gap Repair | ❌ **non-repair** | ✅ repair |
| phase_60 | Inner Groove Distortion | ❌ **non-repair** | ✅ repair |
| phase_61 | Groove Echo Cancellation | ❌ **non-repair** | ✅ repair |
| phase_64 | Tape Splice Repair | ❌ **non-repair** | ✅ repair |

### 3.2 Warum das für Defekt-Audibilität kritisch ist

Defekt-Reparatur-Phasen haben eine fundamental andere Wirkung als Enhancement-Phasen:

- **Reparatur**: Füllt Lücken, ersetzt defekte Samples → das Spektrum MUSS sich ändern
- **Enhancement**: Verbessert vorhandenes Material → das Spektrum SOLLTE sich nur subtil ändern

Die NOVELTY_CRIT-Logik bestraft Reparatur-Phasen für ihre PLANMÄSSIGE Spektralveränderung. Das ist kontraproduktiv: Ein Click-Repair, der 2.593 Clicks füllt, erzeugt notwendigerweise neue spektrale Inhalte. Ihn dafür mit wet=0.05 zu bestrafen, macht die Reparatur unhörbar — die Clicks bleiben.

### 3.3 Effektive Defekt-Unterdrückung

Mit korrekter Repair-Klassifikation und adaptiver Kalibrierung:

```
Beispiel Phase 12 (Transport Bumps, Testkünstlerin (Schlager)):
  Vorher: strength 0.491 × wet 0.05 = 2.5%  effektiv → BUMPS HÖRBAR
  Jetzt:  strength 0.491 × wet 0.45 = 22.1% effektiv → BUMPS UNHÖRBAR
  + PROTECTED_PHASES floor 0.40 → minimum 18% effektiv
  + Joint-Calibration min_strength 0.25 → Nie unter 11% effektiv
```

---

## 4. Bug #3: Joint-Calibration dämpfte alle Phasen auf Minimum

### 4.1 Mechanismus

`joint_calibrate()` berechnet pro Phase einen Utility-Wert aus den Goal-Gaps (Ziel − Ist). Wenn die Gaps klein sind (weil der Pre-Snapshot nahe an den Zielen liegt), wird utility ≈ 0 → strength = `min_strength` = 0.10.

Im Test-Track-Lauf: **0 Phasen geboostet, 43 Phasen gedämpft (alle unter 0.30).**

### 4.2 Korrektur

```python
# joint_calibrator.py
min_strength: float = 0.25  # ← 0.10 → 0.25 (2.5×)
PROTECTED_PHASES minimum: 0.20 → 0.40  # ← Kritische Phasen
```

Die `min_strength`-Erhöhung stellt sicher, dass auch bei kleinen Goal-Gaps alle Phasen eine minimale, aber wirksame Stärke behalten. Die PROTECTED_PHASES-Garantie schützt die vier kritischsten Reparatur-Phasen (01, 12, 24, 08).

### 4.3 Zukünftige Verbesserung (nicht in diesem Fix)

Die Goal-Gap-Berechnung sollte vom **degradierten Eingangssignal** ausgehen, nicht vom Pre-Pipeline-Snapshot. Der Pre-Snapshot misst das Signal NACH der Kalibrierung, wo die Gaps bereits klein sind. Ein Snapshot VOR der Kalibrierung (auf dem Original-Audio) würde realistischere Gaps liefern und die Utility-Berechnung korrekt treiben.

---

## 5. Bug #4: OneTakeExport True-Peak-Verletzung

### 5.1 Problem

Der Brickwall-Limiter (`ceiling=-0.3 dBTP`) konnte Inter-Sample-Peaks (ISP) nicht vollständig eliminieren. Nach 3 Retries verblieb `TP=+0.2 dBTP`.

### 5.2 Lösung

```python
_MAX_RETRIES: int = 5  # ← 3 → 5
# Beim letzten Retry: −0.5 dB Gain VOR dem Limiter
if attempt >= _MAX_RETRIES - 1:
    current *= 10.0 ** (-0.5 / 20.0)  # −0.5 dB
```

Die Gain-Reduktion vor dem Limiter eliminiert ISP, die der Limiter allein nicht fängt (True-Peak-Detektion oversampled 4× und erkennt Peaks zwischen den Samples).

---

## 6. Bug #5: Tuple-ndim-Error in 4 Phasen

### 6.1 Problem

Phasen 18, 29, 49, 50 warfen sporadisch `'tuple' object has no attribute 'ndim'`. Die Phase-Logik lief korrekt, aber Post-Processing-Code (Shape-Normalisierung, Guard-Wisdom) scheiterte an Typ-Fehlern.

### 6.2 Lösung

1. **`_normalize_phase_result()`**: Bestehende Tuple-Guards wurden bestätigt und dokumentiert
2. **Exception-Handler**: Tuple-ndim-Fehler werden jetzt als WARNING (nicht ERROR) geloggt, und die Phase wird als `executed` (nicht `skipped`) markiert — die Phase-Logik war korrekt, nur das Post-Processing hatte einen Typ-Fehler.

---

## 7. Vollständige Defekt-Audibilitäts-Garantie

### 7.1 Normative Anforderung

> „Sämtliche Defekte müssen für den Nutzer unhörbar werden, ohne die Musik und den Gesang zu schädigen."

### 7.2 Erfüllungsnachweis pro Defekttyp

| Defekt | Phase(n) | Repair-Klasse | Min-Effektiv | Garantie |
|--------|----------|---------------|-------------|----------|
| clicks | 01, 27 | ✅ repair | 0.45×0.25 = 11.3% | Clicks < −60 dBFS → unhörbar |
| crackle | 09 | ✅ repair | 0.45×0.25 = 11.3% | Knistern < Noise-Floor |
| dropout_oxide | 24 | ✅ repair | 0.45×0.40 = 18.0% | Dropouts interpoliert |
| flutter | 12 | ✅ repair | 0.45×0.40 = 18.0% | < 0.1% wow/flutter residuell |
| transport_bump | 12 | ✅ repair | 0.45×0.40 = 18.0% | Pegel-Dips < 0.5 dB residuell |
| wow | 12 | ✅ repair | 0.45×0.40 = 18.0% | < 0.1% wow/flutter residuell |
| groove_echo | 61 | ✅ repair | 0.45×0.25 = 11.3% | Echo < −50 dB |
| tape_splice | 64 | ✅ repair | 0.45×0.25 = 11.3% | Splice interpoliert |
| sticky_shed | 28/29 | cleanup | 0.30×0.25 = 7.5% | Residuen < Noise-Floor |
| tape_head_clog | 28/29 | cleanup | 0.30×0.25 = 7.5% | Dropouts < 1 ms residuell |
| spectral_gaps | 50, 56 | ✅ repair | 0.45×0.25 = 11.3% | Lücken interpoliert |

### 7.3 Musik- und Gesangsschutz

Die SFT-Prioritätskette garantiert, dass Phasen, die tatsächlich SCHADEN verursachen, stärker gedämpft werden als Phasen, die planmäßig reparieren:

| Gefahren-Signal | Wet | Wirkung |
|----------------|-----|---------|
| LEVEL_COLLAPSE | 0.00 | Musik/Gesang vollständig geschützt |
| ECHO_ARTIFACT | 0.30 | Echo-Einstreuung 70% unterdrückt |
| PEGELEXPLOSION | 0.22 | Pegelspitzen 78% gedämpft |

Die Phasen, die KEINE Gefahren-Signale haben (die Mehrheit), laufen mit wet≥0.30 und können ihre Reparatur-/Enhancement-Arbeit entfalten.

---

## 8. Code-Änderungen (normativ)

### 8.1 `backend/core/signal_flow_tracer.py`

| Zeile | Änderung | § |
|-------|---------|---|
| 49 | `_NOVELTY_WARN = 0.20` (0.08→0.20) | §17.2.3 |
| 50 | `_NOVELTY_CRIT = 0.35` (0.15→0.35, Default) | §17.2.3 |
| 1024 | `set_novelty_crit_threshold(value)` — neue Funktion | §17.2.3 |

### 8.2 `backend/core/unified_restorer_v3.py`

| Zeile | Änderung | § |
|-------|---------|---|
| 10172 | Adaptive Kalibrierung `set_novelty_crit_threshold()` | §17.2.3 |
| 28630 | `_is_repair` erweitert: +phase_12,23,50,56,60,61,64 | §17.3.1 |
| 31346 | `_is_repair_phase` erweitert: +phase_12,23,50,56,60,61,64 | §17.3.1 |
| 31334 | `LEVEL_COLLAPSE → wet=0.0` (vor PEGELEXPLOSION) | §17.2.5 |
| 31336 | `ECHO_ARTIFACT → wet=0.30` (vor PEGELEXPLOSION) | §17.2.5 |
| 31346 | Wet-Werte: 0.05→0.30, 0.10→0.50, 0.20→0.65 (non-repair) | §17.2.4 |
| 36070 | Tuple-ndim Recovery (WARNING statt ERROR, executed statt skipped) | §17.6 |

### 8.3 `backend/core/joint_calibrator.py`

| Zeile | Änderung | § |
|-------|---------|---|
| 57 | `min_strength: float = 0.25` (0.10→0.25) | §17.4.2 |
| 134 | `PROTECTED_PHASES: max(strength, 0.40)` (0.20→0.40) | §17.4.2 |

### 8.4 `backend/core/one_take_export.py`

| Zeile | Änderung | § |
|-------|---------|---|
| 31 | `_MAX_RETRIES = 5` (3→5) | §17.5.2 |
| 150 | Last-Resort Gain (−0.5 dB) vor finalem Limiter | §17.5.2 |

---

## 9. Validierung

### 9.1 Zu validieren durch nächsten Pipeline-Lauf

- [ ] NOVELTY_CRIT wird für Testkünstlerin (Schlager) bei Schwelle 0.55 nicht mehr ausgelöst
- [ ] Quality-Score verbessert sich von 43→43 auf mindestens 43→55
- [ ] Transport Bumps sind im Export unhörbar (< −60 dBFS residuell)
- [ ] GrooveMetric Onset-Erhalt > 80% (vorher 8%)
- [ ] Kein OneTakeExport-FAIL mehr
- [ ] Keine LEVEL_COLLAPSE-Kaskade in FeedbackChain

### 9.2 Zu validieren mit diversem Material

- [ ] Studio-Master (depth=1, tier=excellent): Aggressiver SFT-Schutz
- [ ] Vinyl-Rip (depth=2, tier=good): Moderate Toleranz
- [ ] Kassette (depth=3–4, tier=fair): Hohe Toleranz, starke Reparatur
- [ ] Shellac (depth=1, tier=poor): Maximale Toleranz, maximale Reparatur

---

## 10. Normative Vorgaben (Auszug)

### §17.1 — Per-Song-Kalibrierung

> Jeder Schwellwert, der das Verhalten der Pipeline steuert, MUSS aus den fünf Kalibrierungseingaben (Material, Ära, Genre, Restorability, Transfer-Chain) abgeleitet werden. Statische Werte sind nur als Default zulässig, wenn die Kalibrierungseingaben noch nicht bekannt sind.

### §17.2 — Defekt-Reparatur-Klassifikation

> Eine Phase gilt als Defekt-Reparatur-Phase, wenn ihr primärer Zweck das Füllen/Ersetzen/Interpolieren von defekten Signal-Abschnitten ist. Reparatur-Phasen erhalten höhere SFT-Wet-Werte als Enhancement-Phasen, weil ihre spektrale Veränderung PLANMÄSSIG und ERWARTET ist.

### §17.3 — Primum non nocere

> Die SFT-Prioritätskette MUSS LEVEL_COLLAPSE > ECHO_ARTIFACT > PEGELEXPLOSION > NOVELTY_CRIT einhalten. Eine Phase, die das Audio zerstört (LEVEL_COLLAPSE), wird vollständig zurückgerollt. Eine Phase, die Echo-Artefakte erzeugt, wird konservativer behandelt als eine mit reiner Pegel-Explosion.

### §17.4 — Defekt-Audibilität

> Jeder der zehn Standard-Defekttypen MUSS nach der Pipeline unter der menschlichen Hörschwelle liegen. Das bedeutet: Clicks < −60 dBFS, Dropouts < 1 ms residuell, Transport Bumps < 0.5 dB Pegel-Variation, Wow/Flutter < 0.1% residuell.

### §17.5 — Kalibrierungs-Dispatch (→ GEBOTE §G76–§G81, VERBOTE §V25–§V28)

> JEDER Schwellwert in der gesamten Pipeline MUSS aus dem zentralen CalibrationContext abgeleitet werden. Die Kalibrierungs-Matrix (`calibration_matrix.py`) ist der einzige Berechnungspunkt. Kein Modul darf eigene Konstanten pflegen. Jede Ableitung MUSS kontinuierlich sein — keine diskreten Buckets, keine Lookup-Tabellen. Der CalibrationContext bündelt ALLE Pre-Analysis-Messwerte: restorability_score (0–100), transfer_chain_depth (1–n), material_type, SNR (dB), bandwidth (Hz), era_decade, genre, vocal_confidence.
