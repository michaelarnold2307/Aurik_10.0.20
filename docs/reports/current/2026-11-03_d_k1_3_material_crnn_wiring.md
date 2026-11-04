# D-K1-3 Material-CRNN-Verdrahtung — Abschluss-Report (2026-11-03)

**Aufgabe:** D-K1-3 (Material-CRNN) strikt nach Register-Kriterien schließen — gelernte
Material-Repräsentation in den Material-Entscheidungspfad verdrahten.
**Artefakt (Register):** `forensics/medium_detector.py` — `MediumDetector.detect`, Roh-Aufruf.
**Branch:** `kun/material-crnn-d-k1-3-wiring-dn0w3m`

---

## 1. Register-Kriterien → Status

| Kriterium (`.github/SOTA_DEFICIT_REGISTER.md`, D-K1-3) | Status | Grund / Beleg |
|---|---|---|
| Lern-Material-CRNN nach WF3 — gelernte Material-Repräsentation **statt** statischer Merkmale im Material-Entscheidungspfad | **erfüllt (Verdrahtung)** | CRNN-Entscheidung sitzt jetzt im Roh-Aufruf `MediumDetector.detect()` (Register-Artefakt), nicht mehr nur in einem Wrapper. `forensics/medium_detector.py`, §D-K1-3-Block vor der `ClassificationResult`-Konstruktion; Override von `primary_material` + `confidence`, Passthrough konsistent (`classifier_source="material_crnn"`). |
| Ziel ≥ 80 % echte Korpus (9 Skalar-Features ≈ 28–36 %) | **erfüllt (Messung), Artefakt-Deployment offen** | Gemessen 2026-10-31: **95,3 %** echte Korpus vs. 59–63 % Gauss/Mixture (held-out, `scripts/train_material_crnn.py`, FILE_REGISTRY Z.40). ABER: `models/material_crnn.pt` existiert in diesem Worktree **nicht** (Modell-Artefakte liegen nicht im Git). Verdrahtung ist ab Modell-Lage wirksam; bis dahin greift §V6-Fallback = bisheriges Verfahren. Training/Durchführung des Artefakts ist offener Rest (siehe §4). |
| Material als eigene Klasse im Entscheidungspfad (K0–K4: material vs. defect vs. signal vs. era) | **erfüllt** | Die CRNN-Klassenliste (`_CRNN_MATERIALS`, 10 Materialien) ersetzt die Material-Entscheidung; Defekt-/Era-Pfade unberührt. |
| Aktivierung nur über §4-Verfahren (`scripts/flag_freigabe.py`) | **offen (bewusst)** | **Kein Flag-Flip.** `use_material_crnn` bleibt `False` (Default = bisheriges Verhalten, bit-identisch). Flag-Flip-Verdikt + Hörordnungs-Sign-off bleiben offen (Register D-K0-8 zeigt: Material-Entscheidungen wirken auf die ganze Kette; §v10.802). |
| Blocker (Datenlage/SOTA-Pfad 3-fach) | **teilweise** | (1) diskriminativere Features → erfüllt (Log-Mel-Spektrogramm-CRNN); (2) gelernter Klassifikator statt pro-Material-Gauss → erfüllt (verdrahtet, Challenger-Logik); (3) mehr Echt-Träger → offen (D-K1-2). |

---

## 2. Architektur der Verdrahtung

**Vorher:** `detect_medium_chain()` (Wrapper) trug einen CRNN-Challenger-Block *nach* dem
`detect()`-Aufruf. Der Register-Roh-Aufruf `MediumDetector.detect()` selbst kannte den CRNN
nicht; bei Era-Konstrain-Second-Pass kehrte der Wrapper frühzeitig zurück und der Challenger
sprang nie an (verdrahtete Lücke).

**Jetzt (single decision point):**

1. `MediumDetector.detect()` berechnet wie bisher den Bayes-/Gauss-Mixture-Pfad
   (9 handgebaute Skalar-Features, `SpectralFingerprint`).
2. **§D-K1-3-Block** (hinter `use_material_crnn`, **Default AUS**):
   - `_crnn_predict(audio, sr)` → `(Klasse, Softmax-Konfidenz)` oder `None`;
   - Treffer: Key-Normierung (Spec 05 §6.1), Override von `primary_material`/`confidence`,
     Evidence-Eintrag `Material-CRNN: …`, `classifier_source="material_crnn"`;
   - Danach bauen Kette, Dolby/DBX-, RIAA-Gates und Konsens auf dem CRNN-primär auf
     (Roh-Aufruf-Konsument `backend/api/pre_analysis.py` erbt den Pfad).
3. Wrapper `detect_medium_chain()`: redundanter Challenger-Block entfernt — beide
   `detect()`-Pässe erben die Entscheidung aus dem Roh-Aufruf.

**CRNN-Inferenz** (`_crnn_predict`): Log-Mel 64×96 (25 s, 24-kHz-Resample, Kanal 0 wie im
Training), Modell `models/material_crnn.pt` (`state_dict` + `materials`), Conv2d×2 + GRU +
FC(10), Softmax-Konfidenz, CPU-deterministisch (§G5 (copilot-instructions.md): kein Sampling,
kein `time.time()`).

**§V6-Fallback (copilot-instructions.md)** — jeder Ausfall kehrt in den bisherigen
Gauss/Mixture-Pfad zurück, mit `logger.warning` + Begründung (`logging.getLogger(__name__)`):

- `models/material_crnn.pt` fehlt → Warnung inkl. Trainingshinweis (Pfad-Check *vor*
  torch-Import, damit der Fallback auch ohne torch getestet/ausgeführt wird);
- Ladefehler (torch/Checkpoint) → Warnung;
- Inferenzfehler → Warnung;
- Flag-Importfehler bzw. Exception an der Entscheidungsstelle → Warnung, Gauss-Pfad bleibt.

**Stereo-Layout-Invariante (AGENTS.md §3):** `_crnn_mono()` bedient **beide** Layouts
`(C, N)` und `(N, C)` und nimmt Kanal 0 (Trainingsverteilung `scripts/train_material_crnn.py::
_load_dataset`). Korrigierter Produktionsbefund: die alte `_crnn_predict`-Implementierung
nutzte `np.asarray(audio).ravel()` — bei Stereo mischt das die Kanäle interleaved und
verfälscht das Spektrogramm (AGENTS.md §3: Layout ohne Check kollabiert).

**Key-Normierung (Spec 05 §6.1 [RELEASE_MUST]):** CRNN-Klasse `digital` (Korpus-Label,
mp3-Born-Netlabels) ist nicht in `SUPPORTED_MATERIALS`; Mapping `digital → mp3_high`
(Ground-Truth der WP-7-Validierung war `mp3_high`, Register D-K1-3, 2026-10-09). Alle
weiteren CRNN-Klassen sind bereits supported; `_KEY_MAP` fängt zusätzlich
`reel_wire → wire_recording`, `cassette_digital → dat`, `vhs_audio → tape` ab.

**Flag-Kontrakt (`backend/core/music_model_flags.py`):** `use_material_crnn = False` —
Default AUS = bit-identisch zum bisherigen Verhalten. Flag-Flip nur nach §4-Verfahren
(`scripts/flag_freigabe.py`: Verdikt → Register-Buchung → Flag-Flip → Commit), dokumentiert
offen.

---

## 3. Belege

**Geändert:**
- `forensics/medium_detector.py` — CRNN-Entscheidung im Roh-Aufruf (`detect()`),
  `_load_material_crnn` (§V6-Warnung, Pfad-Check vor Import), `_crnn_predict` (Tuple
  `(Klasse, Konfidenz)`, Softmax), `_crnn_mono` (Stereo-Layout), `_KEY_MAP` (`digital`),
  Wrapper-Cleanup, `classifier_source`-Durchreichung.
- `backend/core/music_model_flags.py` — Flag-Doku: Verdrahtungsort, Default AUS, Flag-Flip
  offen.
- `.github/FILE_REGISTRY.md` — zwei neue Einträge (Write-Gate).

**Neu:**
- `tests/unit/test_dk13_material_crnn_wiring.py` — 11 Vertragstests, fokussiert,
  synthetische Spektrogramme (64×96) + Mock-Inferenz:
  Flag-Default-Vertrag (kein Flag-Flip), Flag AUS ⇒ CRNN wird nicht konsultiert
  (bit-identisch), Flag AN ⇒ Override (primary/Konfidenz/Evidence/ClassificationResult),
  `digital → mp3_high`, §V6-Fallback ohne Modell (Warnung + Gauss-Baseline),
  §V6-Fallback bei Inferenzfehler (Warnung + Gauss-Baseline), Loader ohne Modell
  (None + Warnung), `_crnn_mono` in beiden Stereo-Layouts (Kanal 0, kein Interleave),
  `_crnn_predict` Layout-Parität + Softmax-Konfidenz + Tensor-Shape (1,1,64,96),
  Determinismus §G5.
- `docs/reports/current/2026-11-03_d_k1_3_material_crnn_wiring.md` — dieser Report.

**Belege im Register (unverändert herangezogen):** SOTA-REVANCHEN2026.md P10,
SOTA_B3_Design.md §D-K1, docs/SOTA_DK13_FORSCHUNGSVORHABEN.md,
docs/reports/current/2026-11-02_b3_modelle_metrik.json (95,3 % vs. 40,2 %),
docs/reports/current/2026-10-31_material_crnn_rekalibrierung.md, FILE_REGISTRY Z.40.

---

## 4. Offene Reste (ehrlich, mit Grund)

1. **Modell-Artefakt:** `models/material_crnn.pt` fehlt im Worktree — die 95,3 %-Messung
   entstand aus einem Training ohne Deployment des Artefakts. Vor Flag-Flip:
   `python scripts/train_material_crnn.py --epochs 30 --out models/material_crnn.pt`
   (Korpus-Grundlage D-K1-2) und Artefakt-Fingerprint
   (`scripts/model_artifact_probe.py`, §III.13 (copilot-instructions.md)) erzeugen.
2. **Flag-Flip:** offen — nur nach §4-Verfahren (`python scripts/flag_freigabe.py
   --flag use_material_crnn`) inkl. Hör-/Sign-off-Bedingung analog Register D-K0-8
   (§v10.802). Dieser Report ändert bewusst kein Flag.
3. **Daten-Bedingung SOTA-M18 (Welle D):** Vinyl↔Reissue-Paare, Echt-Träger ≥ 50/je
   Klasse (D-K1-2) — datenseitige Grundlage für die Rekalibrierung bleibt offen.
4. **Register-Buchung D-K1-3:** Status-/Beleg-Aktualisierung im
   `.github/SOTA_DEFICIT_REGISTER.md` erfolgt im Verifikationslauf (Pre-Commit
   `aurik-sota-deficit-gate`) — siehe §5.

## 5. Ausführungs-Hinweis (Verifikation)

Diese Worker-Session verfügte **nicht über eine Shell/Command-Ausführung** (kein
Ausführungswerkzeug im Tool-Surface; nur Datei-/Inspektionstools). Daher wurde **nichts
gemessen, gelintet oder committed**. Statt Schein-Verifikation: die Befehle für den
Verifikationslauf (aus dem Repo-Root, `nice -n 19` wegen parallelem B3-Messlauf):

```bash
nice -n 19 python -m pytest tests/unit/test_dk13_material_crnn_wiring.py -v   # fokussiert
nice -n 19 ruff --select F821,F601,B009,I001 forensics/medium_detector.py \
    backend/core/music_model_flags.py tests/unit/test_dk13_material_crnn_wiring.py
python scripts/model_inventory.py --wiring   # Verwaisten-Check (erwartet: keine neuen)
python scripts/repo_search.py --before-create tests/unit/test_dk13_material_crnn_wiring.py
python scripts/repo_search.py --before-create docs/reports/current/2026-11-03_d_k1_3_material_crnn_wiring.py
python scripts/change_ledger.py snapshot     # TASK_CHANGES.md
git add -A && git commit -m "feat(forensics): verdrahte Material-CRNN in den Material-Entscheidungspfad (D-K1-3)"
```

Statische Hinweise zur erwarteten Lage:
- `model_inventory.py --wiring` auditiert nur **Verzeichnisse** unter `models/` gegen die
  Consumer-Roots; flaches `models/material_crnn.pt` und diese Änderung erzeugen keine neuen
  Modell-Verzeichnisse → „ohne neue Verwaiste“ per Konstruktion (aber nicht per Lauf
  verifiziert).
- Write-Gate: die beiden neuen Dateien wurden vorab per Namens-/Inhalts-Suche geprüft
  (keine vorhandene kanonische Alternative; `tests/unit/test_dk13_*` bzw.
  `docs/reports/current/2026-11-03_*` sind neu) — `repo_search.py --before-create` ist
  im Verifikationslauf zu bestätigen.
- Sicht-Review der geänderten Stellen (Read-back) erfolgt; Python-Syntax und I001-Ordnung
  der Importblöcke sind nach Konvention gesetzt, aber **nicht per ruff/pytest gemessen**.
