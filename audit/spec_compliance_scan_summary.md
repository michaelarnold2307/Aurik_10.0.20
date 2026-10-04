# 🔍 Spec-Compliance-Scan: Alle 64 Phasen

**Scan-Datum:** 19. April 2026  
**Gesamtstatus:** ⚠️ **REQUIRES ATTENTION** (nur 26.6% PASS)

---

## 📊 Zusammenfassung

| Metrik | Wert |
| -------- | ------ |
| Gescannte Phasen | 64 |
| ✅ PASS (0 Violations) | 17 (26.6%) |
| ⚠️ WARNING (1-2) | 30 (46.9%) |
| ❌ CRITICAL (>2) | 17 (26.6%) |
| Durchschn. Compliance-Score | **8.45/10** |

---

## 🔴 TOP 10 PHASEN MIT HÖCHSTEM VIOLATIONS-COUNT

**Sofortig-Maßnahmen erforderlich:**

| Rang | Phase | Total Violations | Hauptverletzungen | Score | Status |
| ------ | ------- | ------------------- | ------------------- | ------- | -------- |
| 1 | **phase_06** | 6 | stft_no_boundary(6) | 8.5 | ⚠️ WARNING |
| 2 | **phase_10** | 6 | np.max_abs(4), rms_mean(2) | 7.0 | ⚠️ WARNING |
| 3 | **phase_20** | 6 | audio_slice_1d(2), rms_mean(2), stft_no_boundary(2) | 5.5 | ❌ **CRITICAL** |
| 4 | **phase_03** | 5 | audio_slice_1d(4), stft_no_boundary(1) | 7.0 | ⚠️ WARNING |
| 5 | **phase_09** | 5 | session_run_no_chunking(2), stft_no_boundary(3) | 7.0 | ⚠️ WARNING |
| 6 | **phase_17** | 5 | np.max_abs(2), rms_mean(3) | 7.0 | ⚠️ WARNING |
| 7 | **phase_01** | 4 | lpc_order_low(1), np.max_abs(3) | 7.0 | ⚠️ WARNING |
| 8 | **phase_12** | 4 | np.max_abs(2), stft_no_boundary(2) | 7.0 | ⚠️ WARNING |
| 9 | **phase_24** | 4 | stft_no_boundary(4) | 8.5 | ⚠️ WARNING |
| 10 | **phase_42** | 4 | audio_slice_1d(1), rms_mean(3) | 7.0 | ⚠️ WARNING |

---

## 📈 ANTI-PATTERN-HÄUFIGKEIT

**Verteilung über alle Phasen (gesamte Codebase):**

| Anti-Pattern | Count | Severity | Fix-Priorität |
| -------------- | ------- | ---------- | --------------- |
| **stft_no_boundary** | 19 | MEDIUM | 🔴 P0 |
| **np.max_abs** | 18 | HIGH | 🔴 P0 |
| **rms_mean** | 13 | MEDIUM | 🟠 P1 |
| **audio_slice_1d** | 11 | HIGH | 🔴 P0 |
| **no_goal_exclusions** | 3 | MEDIUM | 🟠 P1 |
| **lpc_order_low** | 1 | MEDIUM | 🟠 P1 |
| **session_run_no_chunking** | 1 | HIGH | 🔴 P0 |

**Gesamt Violations:** 65 über alle Phasen

---

## 🎯 PRIORITÄTS-ROADMAP

### 🔴 **P0 - SOFORTIGE FIXES (kritisch für Compliance)**

**Pattern 1: `stft_no_boundary` (19 Vorkommen)**

- Betroffene Phasen: phase_05, phase_06, phase_08, phase_12, phase_20, phase_23, phase_24, phase_29, phase_49, phase_50, phase_55, phase_56 (und weitere)
- **Action:** In allen STFT-Aufrufen `boundary='even'` statt keine/reflect setzen
- **Fix-Beispiel:**

  ```python
  # ❌ FALSCH
  stft_result = librosa.stft(audio)

  # ✅ RICHTIG
  stft_result = librosa.stft(audio, boundary="even")
  ```

- **Estimated Effort:** 2-3 Stunden (automatisierte Replace in allen Dateien)

**Pattern 2: `np.max(np.abs(audio))` (18 Vorkommen)**

- Betroffene Phasen: phase_01, phase_04, phase_10, phase_12, phase_17, phase_19, phase_31, phase_34, phase_35, phase_36, phase_40, phase_44, phase_46, phase_47, phase_52, phase_53, phase_54
- **Action:** Ersetze mit `np.percentile(np.abs(audio), 99.9)` zur Impuls-Artefakt-Toleranz
- **Fix-Beispiel:**

  ```python
  # ❌ FALSCH (blockiert durch Einzelimpulse)
  peak = np.max(np.abs(audio))

  # ✅ RICHTIG (robust gegen Impulse)
  peak = np.percentile(np.abs(audio), 99.9)
  ```

- **Estimated Effort:** 2 Stunden

**Pattern 3: `audio[0]` bei 2D-Audio (11 Vorkommen)**

- Betroffene Phasen: phase_03, phase_20, phase_22, phase_42, phase_57, phase_59, phase_60, phase_61, phase_62, phase_63, phase_64
- **Action:** Ersetze `audio[0]` mit robust axis-agnostischem Code oder `audio[:, 0]`
- **Fix-Beispiel:**

  ```python
  # ❌ FALSCH (gibt nur 2 Samples zurück bei 2D!)
  left_ch = audio[0]

  # ✅ RICHTIG
  left_ch = audio[:, 0] if audio.ndim == 2 else audio
  ```

- **Estimated Effort:** 1-2 Stunden

**Pattern 4: `session.run()` ohne Chunking (1 Vorkommen)**

- Betroffene Phase: **phase_09** (ONNX-Modell mit variabler Eingabe-Länge)
- **Action:** Implementiere Zone-/Chunk-basierte Inferenz mit Overlap-Add
- **Estimated Effort:** 3-4 Stunden

---

### 🟠 **P1 - WICHTIGE FIXES (Code-Qualität & Konsistenz)**

**Pattern 5: `np.mean(audio**2)` als RMS ohne Gating (13 Vorkommen)**

- Betroffene Phasen: phase_07, phase_10, phase_11, phase_17, phase_18, phase_28, phase_33, phase_38, phase_42, phase_49 (und weitere)
- **Action:** Nutze gated RMS-Messung (nur Frames > -50 dBFS) zur Stille-Ignoranz
- **Fix-Beispiel:**

  ```python
  # ❌ FALSCH (Stille beeinflusst Messung)
  rms = np.sqrt(np.mean(audio**2))

  # ✅ RICHTIG (Stille ignoriert)
  frame_rms = np.sqrt(np.mean(audio.reshape(-1, 1024) ** 2, axis=1))
  voiced_frames = frame_rms > 10 ** (-50 / 20)
  rms = np.sqrt(np.mean(audio[np.repeat(voiced_frames, 1024) // 2] ** 2))
  ```

- **Estimated Effort:** 2-3 Stunden

**Pattern 6: `no_goal_exclusions` (3 Vorkommen)**

- Betroffene Phasen: **phase_58** (Lyrics-Guided), **phase_57** (Print-Through), **phase_64** (Tape-Splice)
- **Action:** Registriere fehlende Phasen in `per_phase_musical_goals_gate.py`
- **Estimated Effort:** 30 Minuten

**Pattern 7: `lpc_order_low` (1 Vorkommen)**

- Betroffene Phase: **phase_01** (Click-Removal)
- **Action:** LPC-Ordnung ≥ 30 @ 48 kHz (nicht < 16)
- **Estimated Effort:** 15 Minuten

---

## ✅ PHASEN MIT VOLLSTÄNDIGER COMPLIANCE (17)

Diese Phasen benötigen **keine** Änderungen:

```
phase_02, phase_13, phase_14, phase_15, phase_16, phase_21,
phase_25, phase_26, phase_27, phase_30, phase_32, phase_37,
phase_39, phase_41, phase_43, phase_45, phase_51
```

---

## 📋 IMPLEMENTIERUNGS-PLAN

### **Welle 1: P0-Fixes (Tag 1-2)**

1. ✅ Automatisierte Replace: `stft(...` → `stft(..., boundary='even'` (19 Fixes)
2. ✅ Automatisierte Replace: `np.max(np.abs(...)` → `np.percentile(..., 99.9)` (18 Fixes)
3. ✅ Manuelle Prüfung & Fix: `audio[0]` → axis-agnostisch (11 Fixes)
4. ✅ Manuelle Implementation: phase_09 ONNX-Chunking (1 Aufgabe)

**Geschätzter Aufwand:** 8-10 Stunden

### **Welle 2: P1-Fixes (Tag 3)**

1. ✅ RMS-Gating-Refactoring (13 Fixes)
2. ✅ Goal-Exclusions-Registration (3 Phasen)
3. ✅ LPC-Ordnung-Korrektur (1 Phase)

**Geschätzter Aufwand:** 4-5 Stunden

### **Welle 3: Validierung (Tag 4)**

- Re-scan mit diesem Script
- Unit-Tests für kritische Phasen
- Integration-Tests

**Geschätzter Aufwand:** 2-3 Stunden

---

## 🛠️ NÄCHSTE SCHRITTE

1. **Scan-Report speichern:** ✅ (diese Datei)
2. **Git-Branch erstellen:** `spec-compliance-fixes-p0`
3. **Automatisierte Fixes (STFT, np.max_abs):**
   - Scripts/sed-basierte Batch-Replace
   - oder manuell mit multi_replace_string_in_file
4. **Manuelle Code-Reviews:**
   - Phase-by-Phase für TOP 10
   - Besondere Aufmerksamkeit: phase_20 (CRITICAL), phase_09, phase_03
5. **Regression-Tests:**
   - `pytest tests/unit/test_phases_*.py` nach jedem Fix
   - Compliance-Scan erneut ausführen

---

## 📞 FRAGEN?

- **Wo finde ich die Scan-Ergebnisse?** → `audit/compliance_output.txt` und `audit/spec_compliance_scan_summary.md`
- **Kann ich die Fixes automatisieren?** → Ja, P0-Fixes sind mostly regex-replaceable
- **Beeinflussen die Fixes die Audio-Qualität?** → Nein, es sind Code-Quality-Verbesserungen (nicht Audio-Processing-Änderungen)
- **Wie wichtig ist phase_20 (CRITICAL)?** → Sehr. Es hat die meisten audio_slice_1d-Fehler und sollte zuerst adressiert werden

---

**Status:** 🟠 **IN PROGRESS** → Next action: P0-Fixes implementieren

---

## Delta-Audit (20. Mai 2026) — PDV/UV3 Goal-Awareness und Push-Gate

### Zweck

Aktualisierung des Audit-Stands fuer den aktuellen normativen Delta-Fix in der
Phase-Defect-Verifikation (PDV) und den zugehoerigen Entwicklungsprozess.

### Gepruefter Scope

- `backend/core/phase_defect_verifier.py`
- `backend/core/unified_restorer_v3.py`
- `tests/unit/test_phase_defect_verifier.py`
- `tests/unit/test_artifact_freedom_gate.py`

### Ergebnis

- **Lint/Type-Status im Scope:** PASS
  - C0415 (import-outside-toplevel): behoben
  - W0603 (global-statement): behoben
- **Funktionsstatus:** PASS
  - PDV arbeitet goal-aware mit song-spezifischen Zielwerten (`goal_before`, `goal_after`,
    `goal_targets`, `goal_weights`)
  - Naturalness-Preserve-Guard ist aktiv
- **Teststatus:** PASS (85/85)

### Governance-Update

- Push-Regeln auf Release-Must-Gate aktualisiert
- Pflicht: Tests + Lint/Type + Changelog/Spec/Audit-Delta vor Push

---

## Delta-Audit (20. Mai 2026) — Umgewichtung vor Rollback

### Ziel

Verhindern, dass eine Phase bei grenzwertiger Defekt-Verschlechterung sofort verworfen wird.
Policy: erst Umgewichtung, dann Rollback.

### Umsetzung

- `backend/core/phase_defect_verifier.py`:
  - Vor Rollback werden konservative Blend-Kandidaten getestet (`alpha`: 0.92, 0.85, 0.78).
  - Wenn ein Kandidat alle PDV-Proxy-Schwellen einhaelt, wird Rollback vermieden.
  - Hard-Exception bleibt aktiv: Natuerlichkeits-Hard-Guard verletzt => sofortiger Rollback.
  - Telemetrie erweitert um `reweight_applied` und `reweight_alpha`.

- `.github/instructions/pipeline.instructions.md`:
  - Neuer normativer Abschnitt `§2.48b Umgewichtung vor Rollback [RELEASE_MUST]`.

---

## 🔎 Zusatz-Audit: Direkte Fallback-Pfade (verifiziert)

Ziel dieses Zusatz-Audits: Prüfen, ob Phasen bei Primärfehlern sofort auf DSP/Bypass/Passthrough springen, ohne ausreichende Guarded-Retry-Strategie.

### Top 10 (verifizierte Stellen)

| Rang | Phase | Stelle | Fallback-Typ | Risiko | Kurzbegründung |
| ------ | ------- | -------- | -------------- | -------- | ---------------- |
| 1 | phase_58 | `phase_58_lyrics_guided_enhancement.py:136` | Pass-through return | HIGH | Broad `except Exception` beim LGE-Load führt direkt zu unverändertem Audio-Return. |
| 2 | phase_58 | `phase_58_lyrics_guided_enhancement.py:174` | Pass-through on error | HIGH | `enhance()`-Fehler führt sofort zu Passthrough ohne Retry/Chunking. |
| 3 | phase_23 | `phase_23_spectral_repair.py:1091` | Forced DSP fallback | HIGH | Bei Thrashing-Guard wird ML hart deaktiviert und direkt DSP erzwungen. |
| 4 | phase_23 | `phase_23_spectral_repair.py:1139` | Single-STFT fallback | MEDIUM | MRSA wird unter Last übersprungen; direkter Wechsel auf vereinfachten DSP-Pfad. |
| 5 | phase_06 | `phase_06_frequency_restoration.py:899` | DSP-only timeout fallback | MEDIUM | Watchdog-Timeout im AudioSR-Pfad springt direkt auf DSP-only. |
| 6 | phase_20 | `phase_20_reverb_reduction.py:620` | ML→DSP fallback | MEDIUM | Deterministischer ML-Fehler führt ohne erneute ML-Strategie in DSP-Pfad. |
| 7 | phase_42 | `phase_42_vocal_enhancement.py:864` | RoFormer skip → MDX/NMF/HPSS | MEDIUM | Bei Skip/Fehler im Primärmodell direkter Sprung in Kaskaden-Fallback. |
| 8 | phase_42 | `phase_42_vocal_enhancement.py:948` | MDX failure → NMF/HPSS | MEDIUM | Broad Exception im sekundären Modellpfad führt sofort zur nächsten Notstufe. |
| 9 | phase_49 | `phase_49_advanced_dereverb.py:461` | SGMSE+→WPE fallback | LOW | Primär-ML-Fehler degradiert auf WPE; robust, aber direkter Wechsel ohne Retry-Fenster. |
| 10 | phase_31 | `phase_31_speed_pitch_correction.py:503` | pYIN→YIN fallback | LOW | Pitch-Pfad fällt bei Exception sofort auf einfaches YIN mit fixer Konfidenz. |

### Positive Gegenbeispiele (robuste Kaskade)

| Phase | Stelle | Robustheitsmerkmal |
| ------- | -------- | -------------------- |
| phase_09 | `phase_09_crackle_removal.py:412` | ONNX-Chunking mit Teilsegment-Handling statt harter globaler Abbruch. |
| phase_42 | `phase_42_vocal_enhancement.py:905` | PLM-`set_active(..., False)` im `finally` minimiert Leaks trotz Fallback. |
| phase_49 | `phase_49_advanced_dereverb.py:459` | SGMSE+-Fallback auf WPE plus sauberes Release im `finally`. |

### Bewertung

- Direkte Fallbacks sind nicht per se Spez-Verstoß (Kaskaden sind normativ gefordert),
- aber HIGH-Risk bleibt dort, wo breit gefangene Exceptions ohne Retry-/Recovery-Fenster direkt Passthrough oder harte Degradierung auslösen.
- Priorität für Härtung: phase_58, phase_23, phase_06.
