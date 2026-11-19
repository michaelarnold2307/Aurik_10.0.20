# M12-Pfad-Erreichbarkeit — zwei B3-Ablehnungen sind Mess-Artefakte (2026-11-09)

**Klasse:** D-K3-85 (K3, Ketten-/Phantom-Verdrahtung) · **Status:** Code-Teil
geschlossen, Neumessung offen · **Kein Flag-Flip.**

## 1. Anlass

Nach dem Abschluss von R1–R5 (Kette 4, 2026-11-08) waren WF-V4 und TP-V2
trainiert, aber die B3-Stufen `M12-WF4`/`M12-TPV2` wurden am selben Tag formal
**ABGELEHNT**. Die Prüfung, ob die Stufen überhaupt das trainierte Modell
ausführen (Muster „Phantom-Verdrahtung", §2b Hebel C), ergab: **nein**.

## 2. Befund (am Artefakt gemessen, nicht aus Doku geschlossen)

### (a) WF-V4: Ladepfad zeigte auf ein Artefakt, das nie entsteht

| Prüfung | Ergebnis |
| --- | --- |
| `DEFAULT_MODEL_SUBPATH` | `models/wf4_warp_head/wf4_warp_head.onnx` |
| existiert? | **nein** — repo-weit kein `wf4*.onnx` |
| Trainer-Ausgabe | `wf4_warp_head_v1.pt`, `checkpoint_latest.pt` (Torch) |
| `WF4WarpHead().is_available()` | `False` |
| `apply_wf4_warp_stage(..., Flag AN)` | `(None, False)` — identisch zu Flag AUS |

⇒ Die AN/AUS-Messung vom 2026-11-08 verglich den **heuristischen Phase-12-Pfad
mit sich selbst**. Gemessen wurden 1/6 „Vorteile" bei `fussabdruck_max_rms =
7,86e-4` — das ist Pipeline-Rauschen, keine Modellwirkung.

### (b) TP-V2: falsche Projekt-Wurzel ⇒ untrainierte Seed-Gewichte

```
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent   # = .../Aurik_Standalone/backend  (FALSCH)
DEFAULT_CHECKPOINT = backend/models/tp_v2/tp_v2_phase_v1.pt     # existiert nicht
TPv2PhaseEstimator(...).source == "seed"                        # Zufalls-/Seed-Gewichte
```

`backend/models/` existiert legitim (Trainer-Skripte `de_crackle`,
`loudness_dpn`) — daher fiel der Pfad-Fehler nicht auf. Der Flag-Pfad lief also
mit **untrainierten Gewichten**: die Messung `M12-TPV2` (0/6) beurteilt nicht
das trainierte Modell, sondern ein Zufallsnetz.

### (c) Das Pfad-Gate war blind — und protokollierte trotzdem „ABGELEHNT"

`scripts/b3_pfad_wirksamkeit.py` prüft (b) „Fußabdruck": AN==AUS bit-identisch
⇒ Messung ungültig (W-3-Lehre). Diese Prüfung setzt eine **bit-reproduzierbare**
Pipeline voraus. Sie ist es nicht: derselbe Input zweier Läufe differiert um
~6,5e-4…7,9e-4 RMS. Folge: `pfad_wirksam: true` bei totem Modellpfad, und
`beschluss_schreiben` schrieb `ergebnis: "ABGELEHNT"` — obwohl
`messung_ungueltig` bekannt war.

## 3. Fix (Code)

1. **`backend/core/dsp/wf4_warp_head.py`** — zweiter, fail-closed Ladepfad:
   `DEFAULT_CHECKPOINT_SUBPATH = models/wf4_warp_head/wf4_warp_head_v1.pt`,
   `_try_load_checkpoint()` (ONNX zuerst, dann Checkpoint), Architektur aus
   `build_wf4_warp_model()` — **eine** Quelle für Training und Inferenz, kein
   Export-Drift. Neu: `source`-Property (`onnx:…`/`checkpoint:…`/`nicht_geladen`),
   `_predict_torch` (`eval()` + `inference_mode()`, §G5). Fehlt beides, bleibt
   die §V6-Warnung + Heuristik-Fallback.
2. **`backend/core/dsp/tp_v2_phase.py`** — `_PROJECT_ROOT = parents[3]`
   (Repo-Wurzel, wie in `wf4_warp_head.py`).
3. **`scripts/b3_pfad_wirksamkeit.py`** — Modell-Zeuge für Modell-Stufen:
   `PFAD_ZEUGEN["M12-WF4"|"M12-TPV2"]["modell"]` lädt das Modell und verlangt
   Quelle `checkpoint:…`. Seed-Gewichte oder „nicht geladen" ⇒ Zeuge inaktiv ⇒
   Messung ungültig (fail-closed).
4. **`scripts/b3_stufenmessung.py`** — ungültige Messung schreibt
   `ergebnis: "NICHT_BEWERTBAR"` statt `"ABGELEHNT"` (eine Ablehnung ist ein
   Qualitätsurteil und darf nicht aus einem Mess-Artefakt entstehen).

## 4. Verifikation (CPU, ohne GPU-Eingriff)

| Prüfung | Ergebnis |
| --- | --- |
| `WF4WarpHead().source` | `checkpoint:wf4_warp_head_v1.pt`, `is_available() = True` |
| `apply_wf4_warp_stage` Flag AUS / AN | `(None, False)` / `True`, Stretch 0,99889–1,00773 |
| Determinismus zweier `predict`-Aufrufe | bit-identisch (§G5) |
| `TPv2PhaseEstimator(DEFAULT_CHECKPOINT).source` | `checkpoint:tp_v2_phase_v1.pt` (vorher `seed`) |
| `zeugnis({}, "M12-WF4"/"M12-TPV2")` | `aktiv: True`, Quelle `checkpoint:…` |
| Tests | `test_wf4_warp_head.py` + `test_tp_v2_phase.py` + `test_b3_pfad_wirksamkeit.py` **41/41 grün** |

Neue Regressionstests pinnen genau die Fehlerklasse:

- `test_ladepfad_checkpoint_wenn_onnx_fehlt` (Checkpoint trägt, §G5-deterministisch),
- `test_echtes_trainingsartefakt_ist_erreichbar` (deploytes Artefakt, kein Phantom),
- `test_default_checkpoint_zeigt_in_die_repo_wurzel` (Wurzel-Pin, vorher `backend/`),
- `test_trainer_dateiname_passt_zum_default_checkpoint` (Consumer/Trainer-Namen),
- `test_modellzeuge_meldet_toten_pfad` / `…seed_gewichte_als_inaktiv`,
- `test_beschluss_ungueltige_messung_ist_nicht_bewertbar`.

## 5. Was das für die beiden Beschlüsse bedeutet

`docs/reports/current/b3_objektivbeschluss_M12-WF4.json` und
`…_M12-TPV2.json` (2026-11-08) sind **keine Qualitätsurteile** über die Modelle
— dieselbe Klasse wie der W-3-Befund 2026-11-05 („Δ ≈ 0 war ein No-Op"):

- `M12-WF4`: Modell wurde nie ausgeführt (Vergleich Heuristik ↔ Heuristik).
- `M12-TPV2`: ausgeführt wurden **untrainierte Seed-Gewichte**.

Die Dateien bleiben als Zeitdokument unverändert; die Invalidierung ist
maschinenlesbar in `docs/reports/current/b3_messung_ungueltig_20261109.json`
festgehalten. **Beide Flags bleiben AUS** (§V7) — eine Aktivierung ist erst nach
einer gültigen Neumessung über §4 möglich.

## 6. Offen (nicht autonom in dieser Sitzung)

- **Neumessung** `b3_stufenmessung.py --stages M12-WF4,M12-TPV2` im
  **trainingsfreien GPU-Fenster** (Kette 2 läuft: D3-2 BigVGAN, 6-h-Versuche).
  Erst danach ist ein §4-Beschluss über die Modelle zulässig.
- Die Messung selbst ist nicht bit-reproduzierbar; das Fußabdruck-Gate (b)
  bleibt deshalb nur ein Grobfilter — der Modell-Zeuge (d) ist der belastbare
  Nachweis. Eine Determinismus-Härtung der Pipeline ist ein eigener Auftrag
  (nicht Teil dieses Fixes, keine stille Änderung am Klangpfad).

## 7. Belege

- Register: `.github/SOTA_DEFICIT_REGISTER.md` → `D-K3-85`.
- Code: `backend/core/dsp/wf4_warp_head.py`, `backend/core/dsp/tp_v2_phase.py`,
  `scripts/b3_pfad_wirksamkeit.py`, `scripts/b3_stufenmessung.py`.
- Tests: `tests/unit/test_wf4_warp_head.py`, `tests/unit/test_tp_v2_phase.py`,
  `tests/unit/test_b3_pfad_wirksamkeit.py`.
- Mess-Artefakte: `docs/reports/current/b3_stufenmessung_M12-WF4.json`
  (`fussabdruck_max_rms = 0.0007859073333625735`),
  `…_M12-TPV2.json` (`0.0006504640738223078`).
