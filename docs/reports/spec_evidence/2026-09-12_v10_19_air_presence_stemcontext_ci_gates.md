# Spec-Evidence: §v10.19 Air-Presence/StemContext-Verdrahtung + CI-Gate-Anpassungen

Datum: 2026-09-12 | Spec: `.github/specs/v10.19_sprachmodell_ersatz_sota_roadmap.md` | Version: 10.0.20

---

## Evidenzblock

### 1. Änderungsgegenstand

- **Spec-Datei:** `.github/specs/v10.19_sprachmodell_ersatz_sota_roadmap.md` (Umsetzungs-Record ergänzt)
- **Abschnitt:** Kopf-Umsetzungs-Record
- **Änderungstyp:** DSP-Stufe | Pipeline-Verdrahtung | Monitor | CI-Workflows
- **Alte Regel:** Brillianz im Luftband wurde ausschließlich durch ML-Modelle (KIM2/KIM-Inst) adressiert; es gab keinen model-freien, deterministischen Air-Band-Pfad, keinen First-Class-Stem-Kontext für Phasen 19/43/66 und keinen Brillianz-Status im MusicalGoalsMonitor. CI scheiterte zusätzlich an nicht existierender PyPI-Pin (`snyk==0.5.0`), hartkodiertem Entwickler-Pfad im Cross-Platform-Test, 15 mypy-Fehlern und Node-20-Deprecation-Warnungen der Actions.
- **Neue Regel:** `air_presence_enhancer.enhance_air_presence` (STFT, Original-Phasen, Raised-Cosine-Kanten, Noise-Floor-Gate) als §SLR-1e2b-Stufe im StemLevelRestorer — witness-guarded (hnr_drop < 1.0 dB, pitch_drift < 8 ct, flat_top_rise < 0.02, sonst blend=0). `StemContext` trägt Stems + Witness-Reports; Phase 19 erhält einen Präsenz/Formant-Pfad; MusicalGoalsMonitor meldet das Brillianz-Goal. CI-Gates an die Entwicklung angepasst.

#### 1. Wissenschaftliche Begründung

- **Fachliche Hypothese:** Luftband-Energie (8–20 kHz) trägt die wahrgenommene Brillianz, aber nur dort, wo echte Signalenergie über dem Noise-Floor liegt. Ein phasen-erhaltender STFT-Gain (Original-Phasen-Rekonstruktion) hebt Brillianz ohne Phasenartefakte; Raised-Cosine-Bandkanten vermeiden harte Bandgrenzen (Kammfilter/Präecho).
- **Referenzen (Paper/Standard):** Glasberg & Moore (2002) Maskierungs-/Lautheitsmodell; Smith (2011) Spectral Audio Signal Processing, §Overlap-Add-Rekonstruktion mit Window-Quadratur; Hörordnung `.github/instructions/hoerordnung.instructions.md` (Maskierungsschwelle statt Mess-Null).
- **Warum kausal plausibel:** Gain wirkt multiplikativ auf Bin-Amplituden; bei unveränderten Phasen entstehen keine Dispersionsartefakte. Der Noise-Floor-Gate verhindert Rauschanhebung (die Hör-Instanz stuft Rauschanhebung als Regression ein).

#### 2. Datengrundlage

- **Datensätze/Szenarien:** synthetische Signale (220/440 Hz + 12-kHz-Air-Komponente, weißes Rauschen, Stereo-Duplikat) + echtes Material (test_audio/Elke Best, Kurz-Batch) für den überwachten Lauf.
- **Umfang (n):** n=6 Unit-Tests air_presence + n=5 Unit-Tests stem_context; Witness-Kalibrierung über N≥3 Songs im überwachten Run.
- **Material- und Modusabdeckung:** Stereo/Mono, channels-first/-last, float32, 48 kHz.
- **Ausschlusskriterien:** Signale < 512 Samples (STFT-Fenster 2048), strength=0 (Passthrough-Pfad).

#### 3. Statistik

- **Primärmetrik:** bit-identischer Passthrough bei strength=0; Energieerhalt Tiefband; Determinismus (Zwei-Lauf-Gleichheit); Witness-Gate-Akzeptanz.
- **Effektstärke:** 0 dB Änderung unterhalb des Luftbands (gemessen rel ≤ 5 %); Luftband-Gain ≤ +6 dB bei strength=1.
- **95 %-CI:** Deterministische DSP-Funktion — CI = Punktwert (0 Varianz über Seeds).
- **Signifikanztest + p-Wert:** entfällt (deterministisch, keine Stichprobe).
- **Multiple-Testing-Korrektur:** entfällt.

#### 4. Reproduzierbarkeit

- **Seed(s):** 7, 3, 42, 99 (Test-Fixtures, fixiert).
- **Commit:** dieser Commit (HEAD).
- **Skript/Befehl:** `python -m pytest tests/unit/test_air_presence_enhancer.py tests/unit/test_stem_context.py -q`; Witness-Teil: `python -m pytest tests/unit/test_listening_witness.py -q`.
- **Artefaktpfade:** `reports/` (Lauf-Reports des überwachten Runs), `output/supervised_run/`.

#### 5. Risikoanalyse

- **Risiko für P1/P2:** Minimal — Stufe ist witness-guarded und blendet bei jeder gemessenen Hör-Regression vollständig aus (blend=0).
- **Risiko für Artefakte:** Gering — Original-Phasen-Rekonstruktion + Soft-Knee-Kanten; Noise-Floor-Gate verhindert Rauschanhebung.
- **Bekannte Unsicherheiten:** Witness-Schwellen sind konservativ gesetzt und werden im überwachten Run an P90 der No-Harm-Deltas kalibriert (eigener Evidenzblock bei Schwellenänderung).
- **Rollback-Kriterium:** Falls der überwachte Run eine hörbare Regression zeigt → `air_presence_strength` auf 0 bzw. Stufe deaktivieren.

#### 6. Entscheidung

- **Entscheidung:** APPROVED
- **Maintainer Sign-off:** Michael Arnold (Solo-Maintainer) — 2026-09-12
- **Externer Reviewer (optional):** —
- **Datum:** 2026-09-12

---
