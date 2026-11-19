# Aurik Audit — Finale Erkenntnisse 2026-07-09

## Neue Bugs (seit der letzten Dokumentation)

### Bug 5: defekt_hint-Datenfluss (KRITISCH)
- **Dateien:** `denker/aurik_denker.py:1350`, `backend/core/unified_restorer_v3.py:11509`
- **Symptom:** PhasePruner erhielt `["recommended_phases", "confidence"]` als Defekt-Liste
- **Ursache:** `_defekt_hint` enthielt nur `recommended_phases` und `confidence` — keine `defect_scores`. UV3 rief `list(dict)` auf, was die **Keys** des Dicts lieferte, nicht die Defekt-Namen.
- **Kaskade:** 0 echte Defekte → PhasePruner matchte nichts → 29/42 Phasen geprunt → 13 übrig → ML-Phasen nie aktiv
- **Fix:** `_defekt_hint` um `"defect_types"` (DefectType.name.lower()-Keys) und `"defect_severities"` (Severity-Werte) erweitert. UV3 liest mit `.get("defect_types", [])`.

### Bug 6: Preservation Mode zu aggressiv
- **Datei:** `backend/core/unified_restorer_v3.py:2536`
- **Symptom:** Kassette mit 13 kHz Bandbreite wurde als "zu degradiert" eingestuft
- **Ursache:** Schwelle `bw_loss >= 0.90` triggerte bei fast jeder analogen Aufnahme
- **Fix:** Schwelle auf `0.97` angehoben + Cross-Check mit `source_fidelity_bandwidth_target_hz` (>6 kHz → kein Preservation Mode)

### Bug 7: Falsches Feld für Bandbreiten-Cross-Check
- **Datei:** `backend/core/unified_restorer_v3.py` (Preservation Mode)
- **Symptom:** Cross-Check las `source_fidelity_bandwidth_hz` — dieses Feld wird NIE geschrieben
- **Fix:** Korrigiert auf `source_fidelity_bandwidth_target_hz` (wird von SourceFidelityReconstructor befüllt)

### Bug 8: QualityModeConfig-Regression (SELBST VERURSACHT)
- **Datei:** `backend/core/quality_mode.py`
- **Symptom:** ~50 Phasen konnten nicht laden: `cannot import name 'QualityModeConfig'`
- **Ursache:** Unser neues `quality_mode.py` ersetzte das Original, exportierte aber `QualityModeConfig`, `is_phase_ml_enabled`, `log_mode_decision` nicht.
- **Fix:** `QualityMode`-Enum + `QualityModeConfig`-Klasse + Hilfsfunktionen wiederhergestellt
- **Prävention:** `check_import_breaking.py` Pre-Commit-Hook hätte es geblockt

### Bug 9: 51 stille except-Blöcke in UV3
- **Datei:** `backend/core/unified_restorer_v3.py`
- **Symptom:** `DefectPrecisionEnhancer` war jahrelang tot — `analyze_defects()` existierte nicht, `AttributeError` wurde von `except Exception: pass` verschluckt
- **Fix:** Automatisches Script fügte `logger.debug(..., exc_info=True)` vor jedes stille `pass`/`return`/`continue` ein

---

## Korrigierte System-Architektur

### Datenfluss Defekterkennung (vollständig)

```
DefectScanner.scan()
  → scores: Dict[DefectType, DefectScore]
  → DefectType.name.lower() = "wow", "clicks", ...
      ↓
DefektDenker._extract_scores()
  → defect_type.name → name.lower()
  → DefektErgebnis.defect_scores: {"wow": 1.0, "clicks": 0.8, ...}
      ↓
aurik_denker.py _defekt_hint
  → {"recommended_phases": [...], "confidence": 0.39,
     "defect_types": ["wow", "clicks", ...],
     "defect_severities": {"wow": 1.0, ...}}  ← NEU (war Lücke)
      ↓
UV3.restore() kwargs["defekt_hint"]
  → self._active_defekt_hint
      ↓
PhasePruner.prune(defect_types=defekt_hint["defect_types"],
                   defect_severities=defekt_hint["defect_severities"])
  → Substring-Match: "wow" in "wow" → True
  → Phasen mit Requirements matchen → werden behalten
      ↓
ContractValidator (jeder restore()-Start)
  → Prüft: PhasePruner-Requirements ⊆ DefectType.values
  → Prüft: DefectManifest → PhasePruner Sync
  → Prüft: Keine toten Dateien
```

### Preservation Mode (korrigiert)

```
Alte Logik:
  bw_loss >= 0.90 AND SNR < 16 dB → Preservation Mode
  → Alle Enhancements blockiert
  → ABER Harmonic Restoration + Mastering liefen TROTZDEM mit voller Stärke
  → Verzerrung wurde VERSTÄRKT statt behoben

Neue Logik:
  bw_loss >= 0.97 AND SNR < 16 dB
  AND (eff_bw unbekannt ODER eff_bw < 6000 Hz) → Preservation Mode
  → Nur bei WIRKLICH katastrophalem Material
  → Cross-Check mit tatsächlicher Bandbreite aus RekonstruktionsDenker
```

---

## Neue Schutzmechanismen (Pre-Commit)

| Hook | Schützt vor |
|---|---|
| `check_contracts.py` | Defekt-Phasen-Mismatches, tote Dateien |
| `check_import_breaking.py` | Entfernte kritische Exports (QualityModeConfig, etc.) |
| `check_import_breaking.py` (strukturell) | defekt_hint ohne defect_types, list(dict)-Anti-Pattern |
| `check_spec_refs.py` | Undokumentierte §-Referenzen |
| `check_staticmethod_self.py` | @staticmethod + self.X-Zugriffe |
| `check_format_strings.py` | Logger-Format-String-Mismatches |
| `check_debug_guard.py` | logger.debug() in Loop-Bodies |
| `check_defect_name_strings.py` | Defekt-Namen gegen DefectType-Enum |

---

## Gelöschter Code

| Datei | Grund | Zeilen |
|---|---|---|
| `backend/adaptive_pipeline.py` | Tote v8.2-Legacy-Pipeline | 2247 |
| `backend/defect_detection/` | 12 Dateien, nur von adaptive_pipeline genutzt | ~3000 |
| `backend/region_analysis.py` | Von niemandem importiert | 770 |
| 3 Test-Dateien | Obsolet | ~500 |

---

## Neue Module

| Modul | Zweck |
|---|---|
| `backend/core/defect_manifest.py` | Kanonische Defekt→Phase/Goal/Strength-Registry |
| `backend/core/defect_contract_validator.py` | Cross-Module-Konsistenz (jeder restore()) |
| `backend/core/safe_dict.py` | Dict-Wrapper mit Key-Validierung |
| `backend/core/quality_mode.py` | QualityMode-Validierung + ML-Steuerung |
| `backend/core/periodic_health.py` | Batch-Gesundheits-Metriken (alle 50 Runs) |

---

## Audit-Ergebnisse (Stand: Ende Session)

| Test | Ergebnis |
|---|---|
| 12 kritische Module kompilieren | ✅ |
| ContractValidator | ✅ 0 violations |
| DefectManifest → PhasePruner Sync | ✅ |
| PhasePruner Edge Cases (None/leer/unbekannt) | ✅ |
| Preservation Mode (alt vs. neu) | ✅ Fix wirkt |
| Bandbreiten-Cross-Check-Feld | ✅ Korrigiert |
| defekt_hint Datenfluss | ✅ Konsistent |
| 62 DefectTypes name==value | ✅ 0 Mismatches |
| 5 kritische Phasen importierbar | ✅ |
| psutil verfügbar | ✅ |
| 21 Unit-Tests | ✅ Alle grün |
