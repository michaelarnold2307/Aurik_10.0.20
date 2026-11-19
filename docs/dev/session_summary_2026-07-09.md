# Session Summary: Systemische Verbesserungen — 2026-07-09

## Übersicht

Diese Session identifizierte und behob **4 Bugs**, **3 systemische Muster** und
entfernte **toten Legacy-Code** — alles im Dienst der Architektur-Konsistenz und
Betriebssicherheit von Aurik.

---

## Teil 1: Bugs (dokumentiert in separaten Dateien)

| # | Bug | Dokument |
|---|---|---|
| 1 | `@staticmethod` zerstörte `self` in SongCalibration | `bugfix_2026-07-09_file_ext_and_song_calibration.md` |
| 2 | `input_path` fehlte in `batch_endpoints.py` | selbes Dokument |
| 3 | `cached_cached_era_result` Tippfehler | selbes Dokument |
| 4 | PhasePruner: 76% falsche Defekt-Namen | `bugfix_2026-07-09_phase_pruner_defect_mismatch.md` |

## Teil 2: Systemische Muster (dieses Dokument)

### Muster 1: Cross-Module ContractValidator

**Problem:** Module kommunizieren über String-Keys (Defekt-Namen, Phasen-IDs,
Material-Schlüssel), aber niemand prüft, ob die Keys zwischen Produzent und
Konsument kompatibel sind.

**Lösung:** `backend/core/defect_contract_validator.py`
- Läuft einmal pro `restore()`-Aufruf
- Prüft: PhasePruner vs DefectType, DefectPrecisionEnhancer-Methoden,
  DefectManifest-Sync, tote Dateien
- Logged WARNING bei Mismatches, wirft nie Exceptions
- Ergebnis in `restoration_context["contract_validation"]`

### Muster 2: Stille `except Exception: pass`-Blöcke

**Problem:** 51 von 114 `except Exception:`-Blöcken in `unified_restorer_v3.py`
hatten kein Logging. Features wie DefectPrecisionEnhancer waren jahrelang tot,
ohne dass es jemand bemerkte.

**Lösung:** Automatisches Script (`scripts/compliance/fix_silent_excepts.py`)
fügte `logger.debug(..., exc_info=True)` vor jedes stille `pass`/`return`/
`continue` ein. Kein einziger Fehler bleibt jetzt unbemerkt.

### Muster 3: Zentrale DefectManifest-Registry

**Problem:** Jedes Modul (PhasePruner, SongGoalImportance,
DefectPrecisionEnhancer, CausalDefectReasoner) pflegte eigene Listen von
Defekt→Phase/Goal/Strength-Mappings. Neue DefectTypes wurden nicht überall
registriert — Divergenz war garantiert.

**Lösung:** `backend/core/defect_manifest.py`
- Kanonische Registry für alle 54+ DefectTypes
- Mappings: DefectType → Phasen, Goals, Repair-Stärke
- ContractValidator prüft Sync zwischen Manifest und PhasePruner
- Neue DefectTypes werden hier registriert und sind automatisch überall sichtbar

---

## Teil 2.5: Weitere Bugs (nach der ersten Dokumentation entdeckt)

### Bug 5: defekt_hint-Datenfluss (KRITISCH)
`_defekt_hint` enthielt nur `recommended_phases` + `confidence` — keine Defekt-Daten.
PhasePruner bekam `list(dict)` = `["recommended_phases", "confidence"]` als "Defekte".
→ 0 Matches → 29/42 Phasen geprunt → ML-Phasen nie aktiv.

### Bug 6: Preservation Mode zu aggressiv
Schwelle `bw_loss >= 0.90` triggerte bei Kassette mit 13 kHz. → Alle Enhancements blockiert,
aber Harmonic Restoration lief trotzdem mit voller Stärke → Verzerrung wurde verstärkt.

### Bug 7: Falsches Feld für Bandbreiten-Cross-Check
`source_fidelity_bandwidth_hz` (nie geschrieben) → `source_fidelity_bandwidth_target_hz` (korrekt).

### Bug 8: QualityModeConfig-Regression (selbst verursacht)
Neues `quality_mode.py` exportierte `QualityModeConfig` nicht → ~50 Phasen Import-Fehler.
Pre-Commit Hook `check_import_breaking.py` hätte es geblockt — wurde nachträglich installiert.

### Bug 9: 51 stille except-Blöcke in UV3
`DefectPrecisionEnhancer` war jahrelang tot — `analyze_defects()` existierte nicht.
Auto-Fix-Script fügte `logger.debug(..., exc_info=True)` vor jedes `pass`/`return`/`continue`.

---

## Teil 3: PhasePruner vervollständigt

| Metrik | Vorher | Nachher |
|---|---|---|
| Phasen im Pruner | 24 | **66** (alle) |
| Defekt-Requirements | 19/25 falsch (76%) | **0 falsch** |
| Digital-Material-Skips | 2/7 | **7/7** |
| Tape-Defekte abgedeckt | 14/26 | **synced via DefectManifest** |
| Digital-Defekte abgedeckt | 2/9 | **synced via DefectManifest** |
| Fehlende Phasen | 42 | **0** |

---

## Teil 4: Toter Code entfernt

| Datei/Verzeichnis | Grund |
|---|---|
| `backend/adaptive_pipeline.py` (2247 Zeilen) | V8.2 Legacy-Pipeline, nie instanziiert |
| `backend/defect_detection/` (12 Dateien) | V8.2 Detektoren, nur von adaptive_pipeline genutzt |
| `backend/region_analysis.py` (770 Zeilen) | Von niemandem importiert |
| `tests/test_adaptive_pipeline.py` | Obsolet |
| `tests/defect_detection/` | Obsolet |
| `tests/unit/test_adaptive_pipeline_canonical_policy_guard.py` | Obsolet |

Referenz in `tests/normative/test_no_production_stubs.py` bereinigt.

---

## Teil 5: Neue Werkzeuge

| Script | Zweck |
|---|---|
| `scripts/compliance/check_staticmethod_self.py` | @staticmethod + self.X Detektor |
| `scripts/compliance/check_defect_name_strings.py` | Defekt-String-Literale vs DefectType |
| `scripts/compliance/fix_silent_excepts.py` | Stille except-Blöcke automatisch fixen |

---

## Teil 6: Neue Dateien

| Datei | Zweck |
|---|---|
| `backend/core/defect_contract_validator.py` | Cross-Module-Konsistenzprüfung |
| `backend/core/defect_manifest.py` | Kanonische Defekt-Registry |
| `tests/unit/test_file_ext_digital_prior.py` | Regression-Test für file_ext → Digital-Prior |

---

## Architektur-Prinzipien (Lessons Learned)

1. **Defekt-Namen NUR aus `DefectType`-Enum ableiten**, nie ad hoc erfinden.
2. **Kein `except Exception: pass` ohne `logger.debug(..., exc_info=True)`.**
3. **Jedes neue DefectType → zuerst in `DefectManifest`, dann in Module spiegeln.**
4. **ContractValidator fängt Divergenzen SOFORT, nicht erst beim Kunden.**
5. **`input_path` muss durch alle Call-Sites bis zum MediumDetector propagiert werden.**

---

## Datum

2026-07-09 — Session durch Kun
