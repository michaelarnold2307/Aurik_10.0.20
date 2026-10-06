# Spec 24: ML-Fallback-Architektur — Kein stiller Ausfall mehr

> **Version:** Aurik 10.0.24 · **Scope:** Systemische Stabilität
> **Status:** In Umsetzung
> **Erstellt:** 2026-08-16 · **Abgeschlossen:** —

## Prämisse

ROCm/numba-Defekte ließen ML-Pfade im Betrieb still ausfallen: PANNs-Tags leer
(`PANNs tags: {}`), Genre-CLAP auf Fallback-Konstante 0.350, Readiness-Selbsttest
CRITICAL („NIEMALS als bereit erkannt"), EraClassifier-Traceback-Spam. Zusätzlich
verletzte der §3.0-CrossPhase-Consensus-Cap die §v10.53-Invariante (explizite
Stärke wurde überschrieben). Ziel: EINE kanonische Fallback-Architektur, die
jeden ML→CPU/DSP-Ausfall sichtbar (§V6) und hörqualitäts-neutral macht.

## Maßnahme

Sechs kanonische Muster, je an EINEM zentralen Ort — Plugins nutzen die Helfer,
statt eigene Fallback-Kopien zu pflegen. Keine Komponente wird deaktiviert;
jeder Fallback ist additiv und loggt Warnung + Begründung.

### Implementierung

1. `backend/core/ml_device_manager.py`: `ort_run_with_cpu_fallback(session, feeds,
   rebuild_cpu_factory, label)` — MIOpen-Kernel-Fehler („Code object build failed“)
   → einmaliger CPU-Session-Rebuild. Verdrahtet in LAION-CLAP (`embed_audio`),
   MelBandRoformer (`separate`), PANNs (`get_tags`).
2. `backend/core/resampling_utils.py`: `resample_audio(audio, orig_sr, target_sr)` —
   librosa zuerst; bei `get_call_template`-AttributeError scipy.signal.resample_poly.
   **Never-Pass-Through bei falscher Samplerate** (korrumpiert ML-Embeddings).
   `genre_classifier._resample` darauf umgestellt. Ebenso: DSP-Ersatzpfade für
   `_onset_rate` (Energie-Flux) und `_estimate_key` (FFT-Pitch-Class-Profile) —
   echte Messwerte statt Konstanten-Fallbacks (2.0 / „Unbekannt“).
3. `backend/core/ml_model_readiness.py`: Readiness-Checks werfen nie — Import-Ketten-
   Fehler melden „nicht bereit“ mit §V6-Warnung statt CRITICAL-Selbsttest-Abbruch.
4. `backend/core/era_classifier.py`: Tier-1 ruft CLAP nur bei geladenem Modell
   (eine Warnung, kein Traceback-Spam); `EraResult`-Label/Decade-Invariante
   (Snap auf VALID_DECADES zieht das Label nach).
5. `backend/core/genre_classifier.py`: Stille CLAP-Tag-Fallbacks loggen §V6-
   Warnungen (positiv UND negativ) — Degradation ist nie wieder unsichtbar.
6. `backend/core/unified_restorer_v3.py`: §v10.53-Invariante — §DENKER- und
   §3.0-CrossPhase-Modulation sind mit `not _strength_explicit` geschützt;
   explizite Stärke bleibt autoritativ (Spec 23-Nachtrag).
7. `backend/core/pre_analysis.py`: Chain-Depth-Cap nach wörtlicher
   v10.19-Regel — `_md_confidence < 0.50 → max_chain_depth=2` (obere Stufen
   nutzen die geboostete Konfidenz). Verhindert kettenadaptive Tape-Detektoren
   auf ungeklärten Ketten (Befund: 658 Head-Dip-False-Positives bei
   md_conf=0.31).
8. `backend/core/pre_analysis.py`: Toter Code-Block entfernt — der
   §v10.220-DefectConsensusPipeline-Aufruf war in den Dataclass-Body gerückt
   (NameError still geschluckt → lief nie). Integration als Roadmap dokumentiert
   (Manifest→DefectAnalysisResult-Adapter nötig).
9. `backend/core/defect_scanner.py` + `forensics/medium_detector.py`:
   Zero-Length-Guards — bei Audio < 0.05 s liefern scan()/detect() ehrliche
   Leer-Ergebnisse statt Unsinn (Befund: „0.0s Audio“ → vinyl=1.00 aus Stille,
   8 Consensus-Defekte).
10. `backend/core/unified_restorer_v3.py`: §v10.705-B6-Warnpflicht umgesetzt —
    Chain-Injection fremder Material-Pflichtphasen loggt jetzt sichtbar
    („Material-Fremdlauf“); zuvor still (Befund: phase_64_tape_splice_repair
    auf vinyl-Primary ohne Spur).

### Erfolgskriterium

- Kein CRITICAL im Readiness-Selbsttest bei numba/librosa-defekter Umgebung.
- `PANNs tags` nicht leer bei MIOpen-Kernel-Fehler (CPU-Retry greift).
- Genre-CLAP rechnet mit korrekt resampledtem Audio (kein Pass-through).
- Kein Traceback-Spam im EraClassifier-Log; eine §V6-Warnung pro Fallback.
- Alle bestehenden Tests grün (Repro-Ketten: 101/101 Genre+Resample,
  103/103 Era, 11/11 ROCm-Fallback, 50/50 FeedbackChain).

### Aufwand

3h | **Wohlklang-Wirkung:** Indirekt

### Risiken & Gegenmaßnahmen

| Risiko | Eintrittswkt. | Gegenmaßnahme |
|--------|---------------|---------------|
| CPU-Retry verlängert Inferenz-Latenz | Hoch (bei jedem MIOpen-Defekt) | Einmaliger Rebuild; Session bleibt dauerhaft CPU — Folge-Inferenzen normal |
| scipy.resample_poly weicht minimal von librosa ab | Niedrig | Phasenlineare Polyphasen-Semantik identisch; Längen-Konvention ceil() getestet |
| „Nie werfen“ verdeckt echte Registrierungsfehler | Niedrig | Selftest prüft weiterhin ALLE Checks und loggt Attribut-Fehler; nur der AST-Perceptual-Check degradiert explizit |

---

## Ziel-Matrix

| Ziel | Betroffen? | Wie? |
|------|-----------|------|
| Hörbarer Wohlklang | Nein | Indirekt: ML-Pfade fallen nicht mehr still aus |
| Systemische Stabilität | Ja | Ein kanonischer Fallback-Ort; Ausfälle sichtbar statt still |
| Nachhaltige Wartbarkeit | Nein | Sekundär: fünf Flake-Muster in tests.instructions.md verankert |

> **Regel:** Eine Maßnahme adressiert GENAU EIN Ziel als primäres Ziel.
> Die anderen beiden dürfen als sekundäre Ziele profitieren, aber nicht
> im Fokus stehen. Keine Maßnahme adressiert alle drei gleichzeitig.

---

## Lokale Dritt-Clones: Patch-Vertrag (2026-10-06)

> **Anlass:** Der lokale LAION-CLAP-Clone unter `models/clap/src/` ist aus
> **allen** Gates ausgenommen — `.gitignore` (`/models/*`), `pyrightconfig.json`
> (`exclude: models`), `pyproject.toml` (mypy `ignore_errors` für `models.*`,
> ruff `exclude: models`) und `SKIP_DIRS` des VERBOTEN-Linters. Dadurch blieben
> **drei echte Laufzeitdefekte** unbemerkt (§V6 (copilot-instructions.md):
> stiller Ausfall; §V7 (copilot-instructions.md): Symptom statt Ursache).

### Behobene Defekte (1:1 zum dokumentierten Upstream-Vertrag)

| # | Defekt | Wirkung | Behebung |
| :-: | ------ | ------- | -------- |
| 1 | `CLAP.audio_infer` nutzt `output_dict[key]`, ohne `key` als Parameter zu führen | `NameError` — die dokumentierte Aufrufkonvention (`evaluate/eval_dcase.py`: `audio_infer(audio, hopsize=…, key="embedding", …)`) verlangt `key` | `key="embedding"` als Parameter ergänzt (identisch zum Schlüssel, den `encode_audio`/`get_audio_embedding`/`forward` verwenden) |
| 2 | `hopsize = min(hopsize, audio_len)` mit `hopsize is None` | `TypeError: '<' not supported between instances of 'NoneType' and 'int'` | Default ist die Modell-Clip-Fensterweite: `min(self.audio_cfg.clip_samples, audio_len)` |
| 3 | `convert_weights_to_fp16` schreibt `attr.data` auf `text_projection` | `AttributeError: 'Sequential' object has no attribute 'data'` — in den Zweigen bert/roberta/bart ist `text_projection` ein `nn.Sequential` | `isinstance(attr, torch.Tensor)`-Guard; die fp16-Konvertierung der sub-Module leistet weiterhin `model.apply` |

**Vorher-Nachher-Belege (gemessen 2026-10-06):**

- Defekt 3, alte Codezeile isoliert: `hasattr(nn.Sequential(…), 'data') = False` →
  `AttributeError: 'Sequential' object has no attribute 'data'`.
- Defekt 3, nach dem Patch: `convert_weights_to_fp16(Dummy mit Sequential)`
  läuft fehlerfrei; `text_projection[0].weight.dtype = torch.float16`,
  `proj.dtype = torch.float16` (Parameter-Pfad unverändert).
- Defekt 1, nach dem Patch:
  `inspect.signature(CLAP.audio_infer) = (self, audio, hopsize=None, key='embedding', device=None)`.

### Vertrag

1. **Der Clone bleibt ungetrackt.** `models/clap/` ist ein vollständiger
   LAION-CLAP-Upstream-Clone (README, LICENSE, Gewichte, `roberta-base/`) und
   **kein** projekt-eigener Vendored-Baustein wie `plugins/_vendor_*`. Die
   Anchoring-Ausnahme in `.gitignore` ist hier deshalb **nicht** anzuwenden —
   das Haus-Muster gilt für schlanke, unverzichtbare Architektur-Kopien
   (`models/scnet_4stems/`), nicht für komplette Fremd-Repos.
2. **Lokale Korrekturen brauchen einen committbaren Wächter.** Da der Clone
   selbst nicht committbar ist, ist der **Test** das dauerhafte Artefakt:
   `tests/unit/test_clap_vendored_contract.py` prüft als Quelltext-Invariante
   (stdlib `symtable`, ohne Gewichte lauffähig), dass im Modul **kein**
   referenzierter Name undefiniert ist, dass `audio_infer` den `key`-Parameter
   führt, dass kein `min(hopsize, …)`-Selbstbezug steht und dass der
   fp16-Guard intakt ist. Ohne Clone **überspringt** er sich
   (`skipif`-Idiom wie `test_laion_clap_onnx_guard.py`, §v10.761).
3. **Der Wächter testet sich selbst** (§G176 (copilot-instructions.md):
   Auslösebedingung muss erreichbar sein): die historische Defektform wird
   synthetisch injiziert und muss erkannt werden.
4. **Fremd-Code wird nur defektbezogen gepatcht** — keine Typ-Kosmetik, damit
   die Divergenz zum Upstream messbar klein bleibt (das Modul ist als
   Vendored-Code deklariert, nicht als Projekt-Code).

**Rolle im Fallback:** Der ONNX-Pfad (`models/clap/audio_encoder.onnx`) ist
primär; der PyTorch-Pfad über `models/clap/src/` ist der Fallback
(`plugins/laion_clap_plugin.py`, `sys.path`-Erweiterung). Genau der
Fallback-Pfad war ungetestet — dieser Vertrag schließt ihn.

**Erfolgskriterium:** `pytest tests/unit/test_clap_vendored_contract.py` grün
(7 Tests, 2 davon Selbsttest); auf Checkouts ohne Clone sauber übersprungen
(kein False-Green, kein Fehlschlag).
