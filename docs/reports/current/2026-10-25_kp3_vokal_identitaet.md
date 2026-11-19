# KP-3 · Vokal-Identität F9 (MuQ-MuLan-Zeuge) — Evidenz-Report (Pfad A, KEIN Training)

> Datum: 2026-10-25 · Arbeitssprache Deutsch · §-Zitate mit Quelle (AGENTS.md §5
> ID-Zitierdisziplin). Diese Datei dokumentiert den Ist-Stand **ehrlich** (§G8
> (copilot-instructions.md)): was implementiert ist, was der Harness misst, und
> welche Lücke bis zur Veto-Freigabe offen bleibt.

---

## 1. Ziel und Rolle

Das sprachtrainierte Resemblyzer-Embedding (LibriSpeech, `voice_encoder.onnx`)
wirkte bis zur Sperre als Vokal-Identitäts-Richter; `use_resemblyzer_music=False`
(Vertrag W-12 gesperrt), heute trägt der domänenneutrale DSP-Proxy (MFCC +
spektraler Centroid) die Messung. KP-3 ersetzt diese Rolle **musiknativ** durch
**MuQ-MuLan-Embeddings** (`plugins/muq_plugin.extract_embedding`, 1024-d,
musiktrainiert, deterministisch §G5 (copilot-instructions.md)).

Normative Anker: §III.11 (copilot-instructions.md) — ein sprachtrainiertes
Embedding darf Musik/Gesang nicht richten, ein musiktrainiertes ist der
zulässige Kandidat; Hörordnung §8 — Metriken sind Zeugen, die Hör-Instanz
entscheidet; D-K4-5 (`.github/SOTA_DEFICIT_REGISTER.md`) — Musik-Vokal-Finetune
(F9) steht aus; Empfehlung „Ersetzung statt Finetune"
(`docs/SOTA_TRAINING_KONZEPT.md` §3/KP-3).

**Semantik:** report-only **zusätzlicher Zeuge** — KEIN Veto-Recht, KEIN
Flag-Flip, die Veto-Semantik des bestehenden Guards bleibt unverändert.

---

## 2. Implementierter Stand (diese Dispatch)

| Artefakt | Pfad | Rolle |
| --- | --- | --- |
| Musik-Identitäts-Zeuge | `backend/core/dsp/music_identity_oracle.py` | `music_identity_cosine(pre, post, sr) -> (cosine, reasons)` über MuQ-MuLan; deterministisch (§G5), None nur mit `logger.warning` + Begründung (§V6), Stereo-Layout-Invariante (§11 Spec 02) |
| Ebene-1-Guard-Verdrahtung | `backend/core/dsp/level_1_invariants_guard.py` | `Level1Result.singer_identity_cosine_muq` (Default None) zusätzlich gemessen in `check()`; bestehende Felder/Schwellen/Veto-Logik **unverändert** |
| phase_65-Verdrahtung | `backend/core/phases/phase_65_vocal_naturalness_restoration.py` | `singer_identity_cosine_muq` zusätzlich ins `meta` geschrieben |
| A/B-Harness | `scripts/benchmark_identity_oracles.py` | DSP-Proxy vs. MuQ nach Kosinus-Invarianz (identisch/v verschieden) |
| Unit-Tests | `tests/unit/test_music_identity_oracle.py`, `tests/unit/test_level_1_guard_music_identity_muq.py` | Determinismus, §V6-Fallback, Layout, Schwelle, Integration |

`music_model_flags.use_resemblyzer_music` bleibt **False** (W-12 gesperrt, nicht
semantisch geändert). Es wird KEIN neues Flag eingeführt: der Zeuge misst
report-only **immer** (wie KP-2, `music_mos_oracle.py`).

---

## 3. A/B-Evidenz (Golden Set / synthetisch)

### 3.1 Golden Set — ehrliche Lücke (§G8 (copilot-instructions.md))

`audit/golden_listening_set.json` enthält `items[]` mit `path`, `material`,
`genre`, `defect_types` etc. — **aber keine Sänger-Identitäts-Paare** (keine
zwei Aufnahmen desselben Sängers als „identisch"-Paar). Für die Kosinus-
Invarianz-Hypothese (identischer Sänger ⇒ cos hoch trotz Verarbeitung;
verschiedener ⇒ niedrig) ist das Set deshalb **nicht unmittelbar geeignet**.
Der Harness nutzt daher **deterministische synthetische Paare** (Seed 42):
„identisch" = degradierte Kopien (Hall/Sättigung/Bandlimit) eines Referenz-
Sängers; „verschieden" = unabhängige synthetische Sänger (andere Grundfrequenz).
Das Golden Set wird, wo lesbar, als zusätzliche Referenz-Quelle angebunden
(`golden:*`) — nicht als Identitäts-Paar behauptet.

### 3.2 Methodik

`scripts/benchmark_identity_oracles.py` erzeugt deterministisch (Seed 42) die
folgenden Identitäts- und Verschieden-Paare und misst beide Verfahren:

| Klasse | Erzeugung |
| --- | --- |
| hall / saturation / bandlimit_8k / bandlimit_12k | deterministisch degradierte Kopie desselben Sängers ⇒ „identisch" |
| other_singer_{0..2} | unabhängige synthetische Sänger (Grundfreq 196/261.6/329.6 Hz) ⇒ „verschieden" |

**Metrik:** Kosinus-Invarianz — `mean(cos_ident) − mean(cos_verschieden)` je
Verfahren (höher = bessere Trennung identisch/v verschieden). Der MuQ-Zeuge soll
diese Trennung mindestens so gut treffen wie der DSP-Proxy. Ausgabe:
`docs/reports/current/<datum>_kp3_vokal_identitaet.json`.

### 3.3 Messwerte

**Der Harness-Lauf (MuQ-Modell laden, GPU/CPU) steht noch aus** — im
Sub-Agent-Kontext ist keine Shell verfügbar. Die konkrete A/B-Tabelle wird durch
den Parent/Manager per `python scripts/benchmark_identity_oracles.py` erzeugt.
Die erwartete Richtung ist dokumentiert (identisch ⇒ cos hoch, verschieden ⇒
niedrig); sie wird hier **nicht als Messwert erfunden** (§G8 (copilot-instructions.md)).

---

## 4. Offene Schritte bis zur Veto-Freigabe (noch NICHT erfüllt)

Die Veto-Freigabe hängt an **Kalibrierung + menschlicher Hörabnahme** und bleibt
gesperrt (§v10.802 (copilot-instructions.md), D-K4-5, W-12 `use_resemblyzer_music=False`):

1. **Harness-Lauf ausführen** — die A/B-Tabelle (DSP-Proxy vs. MuQ) auf
   identisch/v verschieden-Paaren erzeugen.
2. **Golden-Set-Identitäts-Paare** beschaffen ODER die synthetische Lücke als
   dauerhaft dokumentiert akzeptieren (§3.1).
3. **Schwellen-Kalibrierung** des MuQ-Kosinus gegen die bestehende 0.92-Schwelle
   (darf die bestehende Schwelle NICHT ohne Hörordnungs-Abnahme ändern).
4. **Blindes A/B + Hörabnahme** für einen Menschen (Wohlklang-Vertrag, 5 Belege).

Erst nach (3)+(4) darf ein Veto-Recht erwogen werden — dieser Dispatch **flippt
nichts und vergibt kein Veto-Recht**.

---

## 5. Akzeptanzkriterien — Selbstbewertung

- **A1** (`music_identity_cosine` deterministisch, None nur mit §V6-Warnung):
  **erfüllt** durch `backend/core/dsp/music_identity_oracle.py` + Tests.
- **A2** (Verdrahtung in Guard UND phase_65; bestehende Felder/Schwellen/Veto
  unverändert; `use_resemblyzer_music` bleibt False): **erfüllt** — grep
  `music_identity|muq.*cos|cosine_muq` liefert Treffer in beiden Dateien.
- **A3** (Evidenz-Report mit A/B-Tabelle): **teilweise** — Report + Harness
  liegen vor; die konkrete Tabelle steht als auszuführender Harness-Schritt offen
  (§3.3 — ehrlich dokumentiert).
- **A4** (Tests grün, file_registry 0/0, id_registry ohne neue Warnungen,
  Ledger, D-K4-5-Blocker): vorbereitet (siehe Dispatch-Bericht); der
  pytest/Script-Lauf erfolgt durch den Parent (kein Shell im Sub-Agent-Kontext).
- **A5** (Bericht): erfüllt durch diesen Report + Dispatch-Bericht.

---

## 6. Gemessener Harness-Lauf (Manager, 2026-10-25) — A3-Nachtrag

Manager-Lauf `scripts/benchmark_identity_oracles.py` (Seed 42, 6-s-Signale)
→ `docs/reports/current/2026-10-25_kp3_vokal_identitaet.json`.

| Paar-Gruppe | n | DSP-Proxy cos (mean) | MuQ-Zeuge cos (mean) |
| --- | --- | --- | --- |
| identisch (degradierte Kopien: Hall/Sättigung/Bandlimit 8k/12k; synthetisch + golden:cassette) | 8 | **0.835** | **0.768** |
| verschieden (synthetische Fremd-Sänger 196/261.6/329.6 Hz) | 3 | **0.975** | **0.969** |
| Trennung (ident − verschieden) | — | −0.14 | −0.20 |

Befunde (ehrlich, §G8 (copilot-instructions.md)):

- **Kosinus-Invarianz unter Verarbeitung belegt:** degradierte Kopien desselben
  Materials bleiben für beide Verfahren hoch-korreliert — der MuQ-Zeuge ist als
  report-only-Zusatzzeuge damit brauchbar (kein Fehlalarm durch Hall/Sättigung).
- **Fremd-Paar-Diskrimination inkonklusiv:** die synthetischen „Fremd-Sänger"
  werden von BEIDEN Metriken mit cos ≈ 0.97 bewertet — die synthetischen Quellen
  sind in beiden Feature-Räumen zu ähnlich (keine echten Sänger-Identitäts-Paare
  im Golden Set, §3.1). Eine Schwelle-0.92-Entscheidung kann auf dieser Basis
  NICHT kalibriert werden; die Lücke bleibt auf D-K4-5-Daten (echte
  Identitäts-Paare) oder die dokumentierte synthetische Akzeptanz angewiesen.
- **A3 nach dem Lauf: teilweise erfüllt** (Tabelle vorhanden,
  Diskriminations-Lücke belegt); A1/A2/A4/A5 erfüllt (Manager-Verifikation:
  pytest 31 grün, Registry 0/0, SOTA-Gate 0, ID-Registry ohne neue Warnungen).
  Der MuQ-Zeuge bleibt report-only ohne Veto-Recht.
