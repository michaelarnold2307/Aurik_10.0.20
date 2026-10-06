# 1c — F3-BigVGAN-Rollout-Entscheid (D-K0-1 / D-K0-2)

**Datum:** 2026-10-06 · **Version:** 10.5.0 · **Auftrag:** Arbeitspaket 1c aus
`docs/reports/current/2026-10-06_steps_0-5_preparation.md` (Schritt 1c)
**Regelbezug:** §III.9 (copilot-instructions.md) Paritätsnachweis auf
strukturierten Feeds; §III.13 (copilot-instructions.md) Domänen-Evidenz;
§V6/§V7 (copilot-instructions.md) Fallback-Logging / Ursache statt Symptom;
§v10.802 (copilot-instructions.md) Versionierungs- und Aktivierungsvertrag;
§Performance-Budget (copilot-instructions.md, synchron zu Spec 07 §9).

---

## 1. Auftrag

Entscheiden, ob der MUSDB18-HQ-Finetune `bigvgan/bigvgan_v2_f3` (A/B-Beleg
`docs/reports/current/2026-09-16_hr_v1_bigvgan_ab_validation.md`: HNR
**+4,42 dB**, af +0,0073) die **Basis** im Produktionspfad ersetzt.

Der Auftrag setzte als Prämisse: „Flag `BIGVGAN_V2_HR_ACTIVATED` ist OFF, der
Rollout ist durch Test-Suite-Umstellung und Performance-Budget blockiert“
(`.github/SOTA_DEFICIT_REGISTER.md`, D-K0-1).

**Diese Prämisse ist widerlegt** (Abschnitt 5).

## 2. Durchgeführt: Export-Rezept an der Wurzel gehärtet

`scripts/export_bigvgan_v2_onnx.py` hatte drei Wurzel-Defekte, die einen
belastbaren F3-Export verhinderten. Alle drei sind behoben:

| # | Defekt | Wurzel-Fix |
| --- | --- | --- |
| 1 | `import numpy` **innerhalb** von `export()` — die Modulebene hatte kein `np`, damit war kein wiederverwendbarer Paritäts-Helfer möglich und der Import fiel bei jedem Aufruf neu an | Modul-Level-`import numpy as np` |
| 2 | Architektur war eine **handkopierte Replik** der Basis-Config, nicht die des Checkpoints ⇒ ein Finetune mit abweichender Config wäre still falsch gebaut worden (Silent-Failure, §V6 (copilot-instructions.md)) | `_build_model()` liest jetzt `raw["cfg"]` **aus dem Checkpoint** (`AttrDict`), validiert `num_mels == 128` und `sampling_rate == 44100` und meldet die Quelle als `Architektur-Quelle: …`; die Replik bleibt nur Fallback für Checkpoints **ohne** `cfg` |
| 3 | Paritätsprüfung ohne **strukturierte Feeds** — die §III.9-Lehre aus basicpitch („weißes Rauschen täuscht OK vor“) war im Skript nicht umgesetzt | neuer Helfer `_parity_feeds()` mit drei deterministischen Feeds (`sane` seed 42, `harmonic` Sinus über Band/Zeit, `const05` konstant 0,5); der Export bricht bei **jedem** Feed mit rel ≥ 1e-3 ab |

Zusätzlich prüft der Export, dass die Eingangsform des exportierten Artefakts
byte-genau dem deployten Vertrag `[1, 128, 64]` entspricht.

## 3. F3-Export: Ergebnis

```bash
AURIK_FORCE_CPU=1 .venv_aurik/bin/python scripts/export_bigvgan_v2_onnx.py \
    --checkpoint output/_training_archive_20260920/f3_bigvgan/best.pt \
    --output models/bigvgan/bigvgan_v2_f3.onnx
```

| Größe | Wert |
| --- | --- |
| Artefakt | `models/bigvgan/bigvgan_v2_f3.onnx` (2 856 056 B) + `.onnx.data` (491 323 392 B) |
| Architektur-Quelle | **Checkpoint-cfg** — `epoch=7`, `val_a1=0.5756958723068237`, 17 Config-Schlüssel |
| Eingang | `mel` `[1, 128, 64]` — **identisch** zum deployten Basis-Vertrag |
| Ausgang | `audio` `[1, 1, 32768]` |

**Paritätsnachweis gegen Torch (strukturierte Feeds, §III.9 (copilot-instructions.md)):**

| Feed | Charakter | max&nbsp;abs | **rel** | Grenze |
| --- | --- | --- | --- | --- |
| `sane` | Normalverteilung µ=−3,0 σ=2,0, seed 42 | 6,5e-04 | **6,521e-05** | ≤ 1e-3 ✅ |
| `harmonic` | Sinus über Band- und Zeitachse | 4,9e-06 | **9,705e-06** | ≤ 1e-3 ✅ |
| `const05` | konstante 0,5 (degenerierter Feed) | 8,3e-05 | **1,693e-05** | ≤ 1e-3 ✅ |

Alle drei Feeds liegen um **1,5 bis 2 Größenordnungen** unter der Grenze. Das
Artefakt ist damit numerisch paritätsbewiesen und **ersetzt nichts** — die
Basis `bigvgan_v2.onnx` bleibt unangetastet.

**Git-Status:** `models/*` ist per `.gitignore:37` ausgenommen. Die Artefakte
bleiben **lokal** (494 MB); committet werden Rezept, Paritätsbeleg und
Report — reproduzierbar über das Kommando oben (§G5
(copilot-instructions.md) Determinismus-Kopplung).

## 4. RT-Kosten — der eigentliche Gate-Keeper

### 4.1 ONNX-Kern (Warmmessung, Median aus 3 Läufen, CPU)

| Artefakt | ms je Block (64 Mel-Frames, Overlap 18) | × RT | je Audio-**Minute** |
| --- | --- | --- | --- |
| Basis `bigvgan_v2.onnx` | 6 768,9 | 12,67× | **760,5 s** |
| F3 `bigvgan_v2_f3.onnx` | 6 826,2 | 12,78× | **766,9 s** |

Block-Dichte: 1,872 Blöcke je Audio-Sekunde. **F3 ist +0,9 % — Rauschen.** Der
Aufwand ist **architektonisch** (122,19 M Parameter, SnakeBeta, 44,1 kHz), nicht
finetune-spezifisch. Ein F3-Rollout bringt **keinen** Laufzeitgewinn.

### 4.2 Produktionshelfer `apply_hr_v1_additive` (echter Pfad, CPU)

Gemessen am kanonischen Helfer, 2,0 s Testton (440 Hz):

| Zustand | Wandzeit für 2,0 s Audio | × RT |
| --- | --- | --- |
| kalt (inkl. Session-Load) | 32,7 s | **16,3×** |
| warm (Median aus 3) | 26,4 s | **13,2×** |

Der Helfer bestätigt die Mikromessung: **eine** HR-V1-Passage kostet
**792 s je Audio-Minute**.

### 4.3 Budget-Vergleich

| Norm (copilot-instructions.md) | Wert je Audio-**Minute** | HR-V1, 1 Aufruf | Verhältnis |
| --- | --- | --- | --- |
| Phase-Pipeline **gesamt** | ≤ 240 s | 792 s | **3,3× Überzug** |
| End-to-End (alle Modi) | 32× RT = 1 920 s | 792 s = 41 % | ein Aufruf frisst 41 % |
| DefectScanner | ≤ 4 s | — | nicht betroffen |
| FeedbackChain (alle Iter.) | ≤ 120 s | — | nicht betroffen |

**Aufrufstellen:** `apply_hr_v1_additive` wird **unbedingt** (nur `try/except`)
aus **fünf** Stellen gerufen:

| Phase | Zeile | Art |
| --- | --- | --- |
| `phase_03_denoise.py` | 2058 | ML-Zweig |
| `phase_03_denoise.py` | 2874 | Hauptpfad |
| `phase_07_harmonic_restoration.py` | 1336 | Hauptpfad |
| `phase_23_spectral_repair.py` | 1550 | Hauptpfad |
| `phase_50_spectral_repair.py` | 882 | Hauptpfad |

Rechnerisch **1…5 Passagen je Song** ⇒ 792 s bis **3 960 s** je Audio-Minute
(13,2× bis 66× RT). Das 32×-RT-End-to-End-Norm ist damit **auch bei einem
einzigen** Aufruf nur mehr rechnerisch erreichbar, bei zwei Aufrufen
überschritten.

## 5. Befund: Aktivierungs-Drift (der Kern des Entschids)

**Gemessen am Produktionspfad** (nicht gelesen, sondern ausgeführt):

```
status: {'activated': True, 'checkpoint_present': True, 'reason': 'activated'}
meta:   {'attempted': True, 'applied': True, 'activated': True,
         'bands_released': 18, 'pqs_mos': 3.859}
Ausgabe verändert:  True
```

| Quelle | Aussage | Wahrheit |
| --- | --- | --- |
| `plugins/bigvgan_v2_plugin.py:108` | `BIGVGAN_V2_HR_ACTIVATED = True` | **aktiv** ✅ |
| `tests/unit/test_hr_v1_activation_contract.py:26` | erwartet `True` | deckt den Code |
| `.github/SOTA_DEFICIT_REGISTER.md` D-K0-1 | „Flag … OFF“, „Rollout blockiert“ | **falsch** ❌ |
| `docs/TODOS_SOTA_ROADMAP.md` Q1 | „UPDATE 2026-09-27 FLAG-FLIP DURCHGEFÜHRT … produktiv aktiv“ | richtig, aber ohne Kostenzahl |
| `docs/TODOS_SOTA_ROADMAP.md` SOTA4-2 | „OFFEN — Budget-Urteil“ | **widerspricht** Q1 ❌ |
| `docs/TODOS_SOTA_ROADMAP.md` Matrix A, Zeile 4 | „nicht deployt“ | **falsch** ❌ |

**Wie es dazu kam:** Der Flip wurde 2026-09-27 auf einen A/B-Beleg gestützt, der
auf **20 s Material auf der GPU** lief („Synthese 2,5× RT GPU“) und zusätzlich
mit „§PERF-R15/17 −32 % Laufzeit“ begründet wurde. Der Produktionspfad ist
jedoch **CPU/ONNX** und kostet **13,2× RT**, also das **5,3-fache** der
GPU-Annahme. Eine 32-%-Laufzeitersparnis kann einen 3,3-fachen Budgetbruch
nicht kompensieren. Zusätzlich nennt der A/B-Beleg selbst zwei offene
Vorbedingungen (Test-Suite-Umstellung >10 `phase_07`-Tests; Performance-Budget
BigVGAN ≫ 10× RT) und schließt mit „bleibt trotzdem **bewusst OFF**“.

**Klasse:** R-BLOCKER-nah. Der Pfad verändert das Ausgangssignal
(`bands_released: 18`, `applied: True`) und verletzt das Per-Operation-Budget
um Faktor 3,3 — beides ohne gültigen Budget-Nachweis und gegen die eigene
Aktenlage. Kein stilles Akzeptieren (§V6/§V7 (copilot-instructions.md)).

## 6. Entscheid

1. **F3 → ONNX: ERFÜLLT.** Exportiert, Architektur aus der Checkpoint-Config,
   Parität auf drei strukturierten Feeds belegt (rel ≤ 6,521e-05 ≪ 1e-3,
   §III.9 (copilot-instructions.md)). Das Artefakt bleibt lokal und **ersetzt
   nichts**.
2. **Rollout/Aktivierung: GESPERRT.** Der Blocker ist **nicht** das Artefakt,
   sondern die **Kostenarchitektur**: 12,7–13,2× RT je Passage, unbedingter
   Aufruf aus bis zu fünf Phasen, 3,3-facher Überzug des gesamten
   Phase-Pipeline-Budgets. F3 bietet **keinen** Laufzeitgewinn (+0,9 %). Damit
   ist ein F3-Rollout derzeit **weder möglich noch sinnvoll** — er würde die
   Kosten nicht ändern und die QualitätsFrage nicht entscheiden.
3. **Die Aktivierungs-Frage ist vorgezogen und offen:** Weil der Pfad bereits
   aktiv ist und die Aktenlage ihn als OFF führt, ist der nächste Schritt
   **keine** F3-Entscheidung, sondern die **Budget-/Aktivierungs-Entscheidung
   für HR-V1 selbst**. Sie verändert das Ausgangssignal und braucht daher
   Maintainer-Sign-off (§v10.802 (copilot-instructions.md)) — sie wird hier
   **nicht** einseitig vollzogen.
4. **Empfehlung:** `BIGVGAN_V2_HR_ACTIVATED = False` (fail-closed, wie im
   A/B-Beleg und im Register gefordert), bis **ein** Aufruf budgetgedeckt ist
   — oder alternativ eine explizite Budget-Ausnahme plus Reduktion auf eine
   einzige Aufrufstelle mit Längen-Deckel. Beide Wege sind Code-Änderungen am
   Signalpfad und damit sign-off-pflichtig.

## 7. Aktenkorrekturen (in diesem Commit)

| Datei | Korrektur |
| --- | --- |
| `.github/SOTA_DEFICIT_REGISTER.md` | D-K0-1: Flag-Zustand und Blockertexte auf die Messung gezogen; D-K0-2: F3-ONNX existiert (Basis bleibt aktiv); D-K3-1: Hinweis auf den realen F3-Artefaktnamen |
| `docs/TODOS_SOTA_ROADMAP.md` | Matrix A (Zeile 34/131): „F3 geplant“ → exportiert+paritätsbewiesen, Rollout gesperrt; Q1: Kostenzahl + Drift-Warnung; SOTA4-2 mit Q1 abgeglichen; Phase-Zeilen 03/07/23/50 um die RT-Kosten ergänzt |
| `docs/reports/current/2026-10-06_steps_0-5_preparation.md` | Schritt 1c von „Arbeitspaket“ auf Ist-Stand gezogen |
| `.github/FILE_REGISTRY.md` | neuer Report + Probeneintrag |
| `CHANGELOG.md` | `## 10.5.1`-Block |

## 8. Offene Punkte

- **`MENSCH`:** Aktivierungs-Entscheid HR-V1 (Flag OFF vs. Budget-Ausnahme mit
  einer Aufrufstelle). Ohne diesen Entschid bleibt ein 3,3-facher
  Budgetbruch im Produktionspfad.
- **`CPU`:** Test-Suite-Umstellung (>10 `phase_07`-Tests) — unverändert offen
  und **erst nach** dem Aktivierungs-Entscheid sinnvoll.
- **`CPU`:** `.github/ML_ARTIFACT_FINGERPRINTS.md` nimmt `bigvgan_v2_f3.onnx`
  beim nächsten `scripts/model_artifact_probe.py`-Lauf auf (66 Verzeichnisse,
  153 → 154 Artefakte).
- **Beleg-Lücke bleibt:** Der Basis-Korpus von `bigvgan_v2.onnx` ist ohne
  Modell-Karte/Hparams/SHA nicht belegbar ⇒ Domäne `unbekannt`
  (D-K0-2, §III.13 (copilot-instructions.md)). Der F3-Zweig ist dagegen
  lückenlos belegt (MUSDB18-HQ, `scripts/train_bigvgan_f3.py`, Checkpoint-cfg).

## 9. Reproduktion

```bash
# Export + Parität (§III.9, strukturierte Feeds)
AURIK_FORCE_CPU=1 .venv_aurik/bin/python scripts/export_bigvgan_v2_onnx.py \
    --checkpoint output/_training_archive_20260920/f3_bigvgan/best.pt \
    --output models/bigvgan/bigvgan_v2_f3.onnx

# Produktionshelfer: Kosten + Aktivierungszustand
AURIK_FORCE_CPU=1 .venv_aurik/bin/python -c "
import statistics, time, numpy as np
from plugins.bigvgan_v2_plugin import apply_hr_v1_additive, hr_v1_activation_status
sr = 48000; x = (0.2*np.sin(2*np.pi*440.0*np.arange(2*sr)/sr)).astype(np.float32)
print(hr_v1_activation_status())
ts = [(lambda t0: (apply_hr_v1_additive(x, sr), time.perf_counter()-t0)[1])(time.perf_counter()) for _ in range(3)]
print('Median RT:', statistics.median(ts)/2.0)
"
```
