# Kette-4-Trainer: drei Defekte, die R3/R4/R5 am Training hinderten (D-K3-81)

**Datum:** 2026-11-08 · **Klasse:** K3 (Verdrahtung/Determinismus, kein Modellproblem)
**Auslöser:** Wiederaufnahme nach Stillstand — `pgrep -f gpu_kette`/`train_job_runner`
leer, drei Kette-4-Jobs `fehlgeschlagen` in Sekunden bis 65 s.
**Kein Flag-Flip. Kein Deploy. Kein Training gestartet (nur CPU-Smokes).**

## 1. Befund aus den Manifesten

| Job | Ergebnis | Dauer | Fehlertext (Auszug) |
| --- | --- | --- | --- |
| `train_job_f9_voiceencoder_finetune.json` | fehlgeschlagen (2×) | 65,1 s / 48,4 s | `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` |
| `train_job_f10_singmos_calibrate.json` | fehlgeschlagen (2×) | 56,0 s / 53,8 s | `AssertionError: SR muss 48000 Hz sein, erhalten: 16000` |
| `train_job_d_k2_3_vocos_finetune.json` | fehlgeschlagen (2×) | 2,1 s / 2,2 s | `Vocos-Checkpoint nicht ladbar: Repo id must be in the form 'repo_name' or 'namespace/repo_name'` |

Alle drei sind **Trainer-Defekte**, keine Daten- oder GPU-Blocker: die
MUSDB18-HQ-Daten lagen vor, die Kette-4-Guards griffen, die GPU war frei.

## 2. F9 — der Trainer konnte strukturell keine Gradienten bilden

`VoiceEncoder.embed_utterance()` kapselt den Forward-Pass in
`torch.no_grad()` (`models/resemblyzer/resemblyzer/voice_encoder.py`). Die
Trainingsschleife baute die d-Vektoren genau damit und rief danach
`loss.backward()` — das musste bei jedem Lauf scheitern, unabhängig von Daten
und Gerät.

**Fix:** `_embed_utterance_diff(encoder, wav, torch)` geht denselben Pfad
(Partial-Slices → Log-Mel → Modell → Mittelwert → L2-Norm) **ohne** `no_grad`.
Nachweis der Äquivalenz: Kosinus **0,99999994** gegen `embed_utterance`,
`requires_grad=True`, `backward()` läuft. Zusätzlich fail-closed, wenn der
Encoder keine trainierbaren Parameter hat (Exit 2 statt Traceback nach 60 s).

**Zweiter Defekt im selben Trainer (erst durch den Smoke sichtbar):**
`_verification_accuracy` enthielt zwei `while`-Schleifen, die bei **einem**
Validierungs-Track nie terminieren konnten. Der Smoke-Modus (4 Tracks ⇒ 1
Val-Track) lief **7 min mit 100 % CPU** ohne Ergebnis; ein Stack-Dump
(`faulthandler`) zeigte die Stelle. Fix: zweiter Index wird über den
Rest-Bereich gezogen (bias-frei, deterministisch, garantiert ≠ erster).

**Verifikation:** Smoke-Lauf vorher `Exit 124` (Timeout), nachher **Exit 0 in
7,7 s** — `Basis-Validierungs-Genauigkeit 1.000`, `Epoch 1/1 loss=3.129583`,
`checkpoint_latest.pt` + `voiceencoder_vocal_ft.pt` + `manifest.json`
geschrieben.

## 3. F10 — 16 kHz an einem 48-kHz-Vertrag, danach ein entarteter Fit

`_SR_MOS = 16_000` trug den Kommentar „Produktionspfad score_mos: I/O @ 16 kHz",
aber `plugins/versa_plugin.py::score()` asserted `sr == 48_000` (und resampelt
intern selbst auf die 16-kHz-Modellrate). Der Kalibrierlauf starb nach 56 s an
der Assertion. **Vertragsquelle ist das Plugin, nicht der Kommentar**
(§V1 (copilot-instructions.md)).

**Fix 1:** `_SR_MOS = 48_000`, `_mono_16k` → `_mono_ziel_sr` (Name sagt jetzt
die Wahrheit), Form-Invariante in `_codec` (Trim auf Eingangslänge, sonst
16002 vs. 16000 Samples).

**Fix 2 (Mess-Validität):** Der Produktions-Score kappt auf einen
**30-s-Mittelausschnitt**. Ein ganzer 238-s-Vocal wurde damit an einer
beliebigen Stelle bewertet — gemessen: derselbe Track **clean 1,303 vs.
`codec` 4,858**. Eine solche Zahl misst den Ausschnitt, nicht die Degradation.
Neu: `_pick_window()` wählt deterministisch das energiereichste 20-s-Fenster
(`_WINDOW_S = 20.0` < 30-s-Kappung, 5-s-Hop, §G5). Clean und verarbeitet werden
jetzt auf **derselben Stelle und Länge** verglichen.

**Fix 3 (fail-closed statt Schein-Artefakt):** `fit_calibration` klemmte den
Nenner auf `1e-6` und meldete im ersten Lauf **a = 2,4e6, b = −7,4e6**. Jetzt
gilt `span < _MIN_SPAN_MOS (0,05)` als **entartet**: `degeneriert: true`,
`a = 0.0`, `grund`, und `main()` endet mit **Exit 3** — das Artefakt bleibt als
Beleg liegen, ist aber nicht als gültige Kalibrierung ausgegeben.

### 3.1 Substantiver Messbefund (der eigentliche Punkt)

Volle Kalibrierung, 320 Paare / 20 Tracks, Seed 42 (`/tmp/f10_voll/calibration.json`):

| Kette | MOS-Mittel | Std |
| --- | --- | --- |
| **clean** | **4,959** | 0,023 |
| mixpegel | 4,284 | 0,181 |
| hall | 4,778 | 0,243 |
| saettigung | 4,945 | 0,027 |
| bandlimit | **4,983** | 0,012 |
| codec | **4,995** | 0,004 |

- `per_pair_violations = 144 / 320` (45 %): der verarbeitete Clip wird **höher**
  bewertet als der unverarbeitete.
- `monotonicity_violations = 3` von 4 möglichen Stufen.
- Nur die Pegel-Kette (`mixpegel`) bewegt den Score in die richtige Richtung.

**Interpretation (ehrlich):** Der Produktions-Proxy SingMOS-Pro ist auf
MUSDB-Gesang am **Ceiling** (4,96 von 5,00, Std 0,02) und für Bandbegrenzung
und Codec-Artefakte **nicht sensitiv** — er bewertet sie sogar leicht besser als
clean. Damit ist eine affine Kalibrierung auf dieser Grundlage **nicht
belastbar**; das Artefakt trägt die Grund-Ordnung nicht. Das ist **kein
Verdrahtungsfehler mehr**, sondern eine Eigenschaft der Metrik — und damit eine
Entscheidung über die Messgröße (Kandidaten: `musik_mos`/`defektlast`, beide
zertifiziert, Spearman 0,95), nicht ein autonomer Code-Schritt.

**D-K4-6 bleibt offen** — mit jetzt präziser Ursache: nicht „Lauf fehlt“,
sondern „Proxy trägt die Ordnung nicht“. Keine Flag-Aktivierung, keine
Read-Side-Verdrahtung.

## 4. D-K2-3 — `Vocos.from_pretrained()` ist ein Hugging-Face-Aufruf

`vocos==0.1.0` implementiert `from_pretrained(repo_id)` als
`hf_hub_download(...)`; ein lokaler Pfad wird von der Repo-ID-Validierung
abgelehnt (`Repo id must be in the form …`) — Exit 1 nach 2,1 s, obwohl
`models/vocos_48khz/{config.yaml,pytorch_model.bin}` vollständig deployt sind.

**Fix:** Offline-Pfad `Vocos.from_hparams(config.yaml)` +
`torch.load(pytorch_model.bin, weights_only=True)` + `load_state_dict`
(dieselbe Reihenfolge wie das Vorbild, ohne Netzwerk; die `EncodecFeatures`-
Sonderbehandlung des Originals ist nicht nötig, die Config nutzt
`MelSpectrogramFeatures`). **Verifikation:** Smoke-Lauf **Exit 0 in 7,6 s**,
`Epoch 1/1 — Verlust 0.16940`, Artefakte `vocos_48khz_music_ft.pt` +
`checkpoint_latest.pt` + `manifest.json`.

## 5. Verifikation (alle Zahlen aus diesem Lauf)

| Prüfung | Ergebnis |
| --- | --- |
| F9-Smoke (CPU, venv_rocm72) | Exit 0, 7,7 s, `val_acc 1.000`, 3 Artefakte |
| F9-Äquivalenz differenzierbarer Pfad | Kosinus 0,99999994, `requires_grad=True`, `backward()` ok |
| F10-Smoke (48 kHz) | Exit 0, 41 s, a = 4,2318, `degeneriert=false` |
| F10 voll (320 Paare) | Exit 0, a = 3,5562, `span 0,675`, **144/320 Verletzungen** |
| vocos-Smoke (CPU) | Exit 0, 7,6 s, Verlust 0,16940, 3 Artefakte |
| `pytest` der 4 berührten Testdateien | **43 passed, 1 skipped** (System-Python ohne `webrtcvad`) |
| `pytest` F9 im Produktions-venv | **12 passed** (inkl. Gradienten-Test) |
| `ruff check` (6 Dateien) | clean |
| `py_compile` (3 Trainer) | OK |

Neue Regressionstests: 13 (F9: 5 — Terminierung, Determinismus, Gradienten,
Quelltext-Pin; F10: 6 — 48-kHz-Vertrag, Fenster ≤ 30 s, Fensterwahl,
Entartungs-Guard, Nicht-Entartung; vocos: 2 — Offline-Pin, Offline-Laden).

## 6. Ehrliche Grenzen

- **Kein Training behauptet.** Die drei Trainer sind jetzt *startbar*; das ist
  nicht dasselbe wie „trainiert“. R3/R4/R5 laufen im nächsten freien
  GPU-Fenster über Kette 4.
- **F10 ist inhaltlich nicht gelöst** (siehe §3.1): die Kalibrierung ist
  diagnostisch, nicht anwendbar.
- **F9-Never-worsen-Gate** und **Hörabnahme** bleiben offen (menschlich/GPU).
- Die volle F10-Kalibrierung lief auf `/tmp` (CPU, `nice 15`) — die GPU-Kette 1
  (m8/A-SPADE) lief parallel, es wurde **kein** zweites Training gestartet.
