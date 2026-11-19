# Aurik 10 — Architektur-Überblick

**Stand:** 2026-10-07
**Version:** 10.11.0
**Status:** RELEASE_MUST-konform | §v10 Pleasantness-First aktiv | Hör-Gates Ebenen 1/2/4 aktiv

> Verbindlicher Wahrheitsstand (normative Kette, `AGENTS.md` §1): `.github/copilot-instructions.md` →
> `.github/VERBOTEN.md` → `.github/instructions/` (hoerordnung + Domain) → `.github/specs/`
> (Index: `00_SPEC_INDEX.md`) → `CLAUDE.md`.

## Kernprinzip (§v10)

**Aurik optimiert JEDEN individuellen Song autonom.** Kein blinder Material-Glaube,
keine statischen Schwellwerte ohne Messung. Die Tonträgerkette, das gemessene SNR,
das tatsächliche Spektrum und die harmonische Dichte des Songs bestimmen ALLE
Parameter — nicht der erkannte Materialtyp allein.

## Kernzahlen (am Code gemessen, 2026-10-07)

- **71 Phasen-Module** (`backend/core/phases/phase_*.py`); im Restoration-Modus sind
  `phase_21_exciter`, `phase_35_multiband_compression` und `phase_42_vocal_enhancement`
  **normativ verboten** (§0a (copilot-instructions.md))
- **65 DefectTypes** (DefectScanner) — alle SNR-adaptiv
- **72 Kausal-Ursachen** (CausalDefectReasoner) — `CAUSES`/`LIKELIHOOD_FNS`/`CAUSE_TO_PHASES`
  sind 72/72/72 synchron; Spec 06 §7.2 ist die Spiegel-Liste (Code ⇄ Spec per Gate erzwungen)
- **13 Denker-Module** (`denker/`) — Orchestrierung, Strategie, Defekt, Reparatur,
  Rekonstruktion, Restaurierung, Exzellenz, Phasen-Interaktion, Cross-Phase,
  Tonträger, Tonträgerkette, Perceptual Council, `__init__`
- **16 Träger-Materialien** (`MaterialType`) — Priors für 17 Schlüssel in `MATERIAL_PRIORS`
  (inkl. `unknown`)
- **15 Musical Goals** (Spec 01) + 2 vokal-exklusive P0-Gates
- **Hör-Gates Ebenen 1/2/4** (`level_1_invariants_guard`, `defect_audibility_gate`,
  `vocal_overdrive_guard`, `einladungs_gate`)
- **Budget-Wahrheit:** harter Guard **32× RT** (FAST **8×**, §2.38 KMV), gemessene Ist-Lage
  der Voll-Pipeline **~53× RT** (Matrix-Endlauf 2026-09-07/08). Die Lücke ist Performance-Arbeit
  (TODO-P0-1: Analytik/End-Gate von per-Chunk auf Song-Ebene), kein Planungsziel —
  das Plan-Budget ist auf die Guard-Grenze gedeckelt
- **~18.400 Tests** (511 mit Markern)

## Kanonischer Release-Vertrag

```text
Audio-Import  -> backend.api.bridge.get_load_audio_fn()
Voranalyse    -> backend.api.bridge.run_pre_analysis() genau einmal
Pipeline      -> get_aurik_denker_instance().denke(...)
Modus         -> restoration | studio2026
Export        -> export_guard() + validate_export_quality() + AudioExporter
```

## Zentrale Komponenten

| Komponente | Zweck |
| --- | --- |
| `AurikDenker` | Kognitive Orchestrierung der Gesamtpipeline (Stufen 1–10) |
| `StrategieDenker` | Budgetplanung und Stufenwahl; Budget ist auf die Guard-Grenze gedeckelt |
| `DefektDenker` | Defekt-Detektion (65 Typen) + kausale Einordnung |
| `ReparaturDenker` | Chirurgische Vorab-Reparatur (Klicks, Brummen, Clipping) |
| `RekonstruktionsDenker` | Lücken-Schließung (Dropouts, Bandbreite) |
| `RestaurierDenker` | UV3-Instanz und Kern-Skalierung |
| `ExzellenzDenker` | 15-Goal-Bewertung + konservative Ziel-Reparatur **unter der lexikografischen Hörordnung** |
| `PhaseInteractionDenker` | Phasenplan: Konflikte, Constraints, Kahn-DAG, **zentrale Guard-Modulation** (Stärke) |
| `CrossPhaseCoordinator` | Kumulative Wirkung pro Frequenzband (2–8 kHz) deckeln; Reset je Song |
| `TontraegerDenker` / `TontraegerketteDenker` | Trägererkennung und Kettenableitung |
| `PerceptualQualityCouncil` | Holistisches Qualitätsurteil als **Zeuge** (ersetzt keine normative Formel) |
| `UnifiedRestorerV3` | Phase-Orchestrierung und Kontextsteuerung |
| `DefectScanner` | Defekt-Detektion (65 Typen) |
| `CausalDefectReasoner` | Kausalkette und Mapping auf Phasen (72 Ursachen) |
| `MusicalGoalsChecker` | 15-Goal-Bewertung |
| `HolisticPerceptualGate` | HPI/AFG/VQI-basierte Freigabelogik |
| Hör-Gates E1/2/4 | Level-1-Invarianten, Defect-Audibility, Vocal-Overdrive, Einladungs-Gate (`backend/core/dsp/`) |

## Datenfluss (vollständig)

```mermaid
flowchart TD
    IN["🎵 Audio-Eingang"] --> BRIDGE["backend/api/bridge.py
Mode-Normalisierung (kanonischer Vertrag)"]
    BRIDGE --> PRE["🔍 run_pre_analysis (genau einmal)"]

    subgraph PRE_LAYER["Voranalyse"]
        MD["MediumDetector
Träger + Transferkette"]
        EC["EraClassifier
Ära + Material-Prior"]
        GC["GenreClassifier"]
        DS["DefectScanner
65 DefectTypes, SNR-adaptiv"]
        RE["RestorabilityEstimator"]
    end
    PRE --> PRE_LAYER

    PRE_LAYER --> STRAT["StrategieDenker
Budget ≤ Guard-Grenze (32×; FAST 8×)"]
    STRAT --> DENKER["AurikDenker.denke()
zentrale Entscheidungsintelligenz"]

    subgraph DENKER_LAYER["Denker-Schicht (13 Module)"]
        DD["DefektDenker"] --> CDR["CausalDefectReasoner
72 Ursachen, Bayes"]
        CDR --> RD["ReparaturDenker"]
        RD --> ReD["RekonstruktionsDenker"]
        ReD --> PID["PhaseInteractionDenker
Konflikte · Constraints · Kahn-DAG"]
        PID --> CPC["CrossPhaseCoordinator
Band-Budgets (deterministisch)"]
    end
    DENKER --> DENKER_LAYER

    DENKER_LAYER --> UV3["UnifiedRestorerV3"]

    subgraph PHASE0["Phase 0 — ML-Vorverarbeitung"]
        E0A["EAR_VAE
Neural Clean-Pass"] --> E0B["Apollo
Codec-Restauration"]
        E0B --> E0C["DeepFilterNet v3
Rauschen"]
        E0C --> E0D["Resemble Enhance
Spektrum"]
    end
    UV3 --> PHASE0

    subgraph PHASES["Phasen-Engine (71 Module)"]
        P1["Korrektiv
Klicks · Brummen · Rumpeln · Knistern"]
        P2["Subtraktiv
Denoise · Hiss · Tape"]
        P3["Dynamik · Transienten"]
        P4["Additiv
EQ · Air-Band · Harmonics"]
        GLUE["Glue Stage (vorletzte Phase, in ALLEN Modi)"]
    end
    PHASE0 --> PHASES

    subgraph FORBIDDEN["⛔ §0a — im Restoration VERBOTEN"]
        F1["phase_21_exciter"]
        F2["phase_35_multiband_compression"]
        F3["phase_42_vocal_enhancement"]
    end

    subgraph GUARDS["🛡️ Hör-Ordnung & Gates"]
        E1["Ebene 1 — Hör-Invarianten
Level1InvariantsGuard
Stimm-Identität · Vibrato · Atem"]
        E2["Ebene 2 — Audibility
Maskierungsschwelle statt Mess-Null
is_audible · repair_objective_met"]
        E3["Ebene 3 — Lexikografische Ordnung
Natürlichkeit vor Wärme
vor Klarheit vor Brillanz
GoalPriorityProtocol"]
        E4["Ebene 4 — Einladungs-Gate
nie ermüden"]
        G1["Per-Phase Musical Goals Gate"]
        G2["Artifact Freedom Gate"]
        G3["Holistic Perceptual Gate"]
        G4["Wohlklang-Vertrag (5 Belege)"]
    end
    PHASES --> GUARDS
    GUARDS -->|Phase verletzt Ebene 1| UV3

    PHASES --> FC["FeedbackChain"]
    FC --> UV3
    UV3 --> EXPORT_GATE["Export-Gate
§0c: degraded exportieren statt Hardstop ohne Datei"]

    subgraph EXPORT_PIPE["📤 Export-Reihenfolge (§IV)"]
        X1["_export_guard
NaN/Inf + True-Peak"]
        X2["_export_nuance_guard
perzeptuelle Politur"]
        X3["cd_noise_profile_inject
psychoakustisch maskiert"]
        X4["apply_dither
POW-r Type 3 / TPDF"]
        X5["Atomic write .tmp → os.replace"]
        X6["Metadata + Aurik-Provenance"]
    end
    EXPORT_GATE --> EXPORT_PIPE
    EXPORT_PIPE --> OUT["💾 Ausgabe-Datei"]

    OUT --> REPORT["GUI/CLI-Narrativ
t() i18n · kontextbewusste Statusmeldungen"]
```

## Kanonischer Pfad (GUI und CLI identisch, §G9 (copilot-instructions.md))

```mermaid
sequenceDiagram
    autonumber
    participant U as 👤 Nutzer (GUI/CLI)
    participant B as 🌉 backend/api/bridge.py
    participant P as 🔍 Pre-Analysis
    participant D as 🧠 AurikDenker
    participant V as ⚙️ UnifiedRestorerV3
    participant E as 📤 Exporter

    U->>B: Datei + Modus (restoration | studio2026)
    B->>B: normalize_user_mode() — EINE Aliasquelle
    B->>P: run_pre_analysis()
    P-->>B: Medium · Ära · Genre · Defekte · Restaurierbarkeit
    B->>D: denke(audio, Modus, Kontext)
    D->>D: StrategieDenker: Budget ≤ Guard-Grenze
    D->>D: DefektDenker → CausalDefectReasoner (Bayes, 72 Ursachen)
    D->>D: Reparatur-/Rekonstruktions-/RestaurierDenker
    D->>V: precomputed_phase_plan + Kontext
    loop je Phase
        V->>V: Guard-Modulation (gewichtete Mittelung)
        V->>V: Band-Budget-Deckel (CrossPhaseCoordinator)
        V->>V: Hör-Invarianten (Ebene 1) → ggf. Phase zurücknehmen
        V->>V: Audibility (Ebene 2): Defekt unter die Maske drücken
    end
    V->>V: Glue Stage (vorletzte Phase, alle Modi)
    V-->>D: Audio + Metadaten + Zeugen
    D-->>B: AurikErgebnis
    B->>E: Export-Pipeline (§IV)
    E-->>U: Datei + Narrativ (Narrativ in beiden Pfaden gleich)
```

## Qualitäts- und Sicherheitsinvarianten

- `artifact_freedom < 0.95` blockiert Freigabe.
- Vokalpfad nutzt VQI als zusaetzlichen Recovery-Trigger.
- Kein paralleler Produktpfad ausserhalb des kanonischen Vertrags; GUI, CLI, REST und Batch
  durchlaufen denselben Pfad (Canonical-Contract-Drift-Gate).
- **Hörordnung Ebene 3 ist im Denker erzwungen:** `hearing_order_violation()` lehnt jeden
  Kandidaten ab, der ein höherrangiges Ziel für ein niederrangigeres senkt — auch bei
  positiver Gesamtbilanz.
- **§V8 Song-Isolation:** alle zustandstragenden Module werden je Song zurückgesetzt
  (u. a. `FallbackAuditor`, `CrossPhaseCoordinator`, Narrativ-Zähler).
- **§G5 Determinismus:** keine Zufalls-/Zeitquelle in Entscheidungen; Mengen werden
  sortiert iteriert, wo Fließkommawerte aufsummiert werden.
- **§0c Export-Vertrag:** Bei fehlgeschlagenem Export-Gate wird das bestmögliche sichere
  Ergebnis mit Status `degraded` exportiert — kein Hardstop ohne Datei.

## Produktgrenzen

- Desktop-only
- Offline-first
- Mono/Stereo als produktiver Zielpfad
- Keine Cloud-/Serverpflicht im Endnutzerbetrieb
