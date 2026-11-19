# Evidenzbericht: §2.46b Deep-Transfer-Chain — Mehrstufige Tonträgerketten-Inferenz

## Evidenzblock

- **Spec-Datei**: `.github/specs/02_pipeline_architecture.md`
- **Abschnitt**: §2.46b (neu), §2.46a (erweitert)
- **Änderungstyp**: Lückenschluss — Code-Spec-Synchronisation für mehrstufige Ketten
- **Alte Regel**: DefectScanner auto-detection wurde nicht in Ketten-Inferenz propagiert
- **Neue Regel**: `auto_detected_material` Feld in `DefectAnalysisResult`, `pre_analysis` liest es als Fallback

### 1. Problem

Bei einem Song mit echter 4-stufiger Kette (`reel_tape → vinyl → cassette → mp3_low`)
zeigte die GUI nur `reel_tape → mp3_low` an. Ursachen-Kette:

1. `MediumDetector` erkannte nur `mp3_low` (Dateiendung .mp3, analoge Posteriors zu schwach)
2. `DefectScanner` detektierte `cassette` (Score 6.95) — aber dieser Wert wurde **nicht**
   im `DefectAnalysisResult` gespeichert, weil der externe Hint (`mp3_low`) das Feld
   `material_type` überschrieb
3. `pre_analysis` las `result.defects.material_type` → `mp3_low` → `_defmap.get("mp3_low")` → `None`
4. Kette blieb bei `[reel_tape, mp3_low]` — `cassette` und `vinyl` fehlten

### 2. Lösung

**Bug 1:** `DefectAnalysisResult` hatte kein Feld für den auto-detektierten Wert.

→ Fix: `auto_detected_material: MaterialType | None = None` hinzugefügt.
  Der Scanner setzt `auto_detected_material=_auto_material` (nur wenn ≠ material_type).

**Bug 2:** `pre_analysis` las nur `material_type` (den Hint), ignorierte die Auto-Detektion.

→ Fix: `auto_detected_material` wird als Fallback gelesen, Enum-Suffix geparst,
  in `_defect_material` übernommen.

**Bug 3:** `_defmap` enthielt keine digitalen Codec-Typen — `mp3_low` wurde zu `None`.

→ Fix: Codec-Typen werden jetzt korrekt als "nicht analog" behandelt (kein Injection).

### 3. Ergebnis

Aus dem Log (vorher → nachher):

```
Vorher: reel_tape → mp3_low                          (2 Stufen)
Nachher: reel_tape → vinyl → cassette → mp3_low      (4 Stufen, korrekt)
```

Chain-Injection-Logik:
1. `EraClassifier.material_prior` → `reel_tape` (1970er-Ära, analoge Rolloff-Charakteristik)
2. `DefectScanner.auto_detected_material` → `cassette` (Crackle+Wow+Flutter, Score 6.95)
3. Vinyl-Inference: reel_tape + cassette + 1950≤decade≤1990 → `vinyl` eingefügt

### 4. Statistik

- **Seed**: n/a (deterministische Code-Änderung)
- **Primärmetrik**: Ketten-Vollständigkeit (vorher 2/4 Stufen, nachher 4/4)
- **95 %-CI**: n/a (Struktur-Änderung)
- **Commit**: 21127f7b

### 5. Reproduzierbarkeit

```bash
python3 -c "
from backend.core.pre_analysis import run_pre_analysis
import soundfile as sf
audio, sr = sf.read('test_song.mp3')
result = run_pre_analysis(audio, sr, file_path='test_song.mp3')
print(f'Chain: {result.medium.chain_label}')
print(f'Multi-Gen: {result.medium.is_multi_generation}')
"
```

### 6. Maintainer Sign-off

- [x] DefectAnalysisResult.auto_detected_material implementiert
- [x] pre_analysis Deep-Transfer-Chain liest auto_detected_material
- [x] Vinyl-Inference dokumentiert und begründet
- [x] Reproduktions-Skript in Spec dokumentiert
- [x] Keine Audio-Veränderung — rein logische Inferenz
