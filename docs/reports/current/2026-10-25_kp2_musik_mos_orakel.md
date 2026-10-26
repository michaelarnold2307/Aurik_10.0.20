# KP-2 · Musik-MOS-Orakel — Evidenz-Report (Pfad A, kein Training)

> Datum: 2026-10-25 · Arbeitssprache Deutsch · §-Zitate mit Quelle (AGENTS.md §5
> ID-Zitierdisziplin). Diese Datei ist der geforderte Evidence-Report
> (Akzeptanzkriterium A1). Sie dokumentiert den Ist-Stand **ehrlich**: was bereits
> gemessen ist (mit Pfad auf die Mess-Reports), was der neu gebaute Harness abdeckt,
> und welche Lücke für die spätere Veto-Freigabe noch offen ist (§G8 (copilot-instructions.md):
> Lücken nie still, sondern dokumentiert).

---

## 1. Ziel und Rolle

Das sprachtrainierte UTMOSv2-Orakel (BVCC) wird durch ein **musiknatives**
MOS-Orakel ersetzt: **MuQ + A1-Eval-Head** (SRCC 0.957 gegen menschliche MOS,
`plugins/muq_plugin.py`) kombiniert mit **SingMOS Pro**
(`models/singmos/singmos_pro.onnx`). Pfad A (empfohlen in
`docs/SOTA_TRAINING_KONZEPT.md` §3/KP-2): Zeugen-Paare + Schwellen-Kalibrierung
auf vorhandener Degradations-Evidenz; Anbindung **report-only** an den
Witness-Pfad (kein Veto, kein Flag-Flip).

Normative Anker: §III.11 (copilot-instructions.md) — ein sprachtrainiertes
Orakel darf bei Musik/Gesang nicht richten; Hörordnung §8 — Metriken sind
Zeugen, die Hör-Instanz entscheidet; D-K4-4
(`.github/SOTA_DEFICIT_REGISTER.md`) — Musik-MOS-Finetune/‑Kalibrierung (F8)
steht aus.

---

## 2. Implementierter Stand (diese Dispatch)

| Artefakt | Pfad | Rolle |
| --- | --- | --- |
| Musik-MOS-Orakel (report-only) | `backend/core/dsp/music_mos_oracle.py` | lädt MuQ+A1 (`estimate_muq_mos`) und SingMOS Pro (`get_versa_plugin().score`), berechnet ein kalibriertes Musik-MOS-Delta; **`applied` ist konstant False** — kein Veto, kein Flag-Flip, keine Profil-Mutation |
| Richtungsstabilitäts-Harness | `scripts/benchmark_music_mos_oracles.py` | vergleicht UTMOS vs. MuQ+A1 vs. SingMOS über ≥4 Degradationsklassen auf MUSDB-test + Vinyl-Korpus; Richtungsstabilität + Non-Inferiority vs. UTMOS |
| Unit-Tests | `tests/unit/test_music_mos_oracle.py` | Determinismus, Fallback (§V6), Stereo-Layout, Schwellen-/Delta-Logik — ohne `try/except: pass` (Lehre D-K3-20) |

Der Witness-Pfad bleibt **unverändert** in der Veto-Semantik. `music_model_flags.use_utmos_music`
bleibt `False` (W-10 gesperrt); es wird kein neues Flag eingeführt, weil das
Orakel report-only **immer** misst. Der bestehende UTMOS-Witness
(`backend/core/dsp/utmos_delta_gate.py`) bleibt unangetastet — das Musik-MOS-Orakel
ist ein **zusätzlicher** Zeuge, kein Ersatz Laufendes im Regelkreis.

---

## 3. Vorhandene Mess-Evidenz (belegt, mit Pfad)

### 3.1 UTMOS — Negativbefund (richtungsblind/invertiert, gemessen)

`docs/reports/current/2026-09-13_utmos_music_calibration.json`
(`scripts/validate_utmos_music_calibration.py`, MUSDB-test, Seed 42, 3 Tracks):

| Klasse | mean_delta (ref−deg) | direction_correct |
| --- | --- | --- |
| band12 | −0.163 | **false** |
| band16 | +0.011 | **false** |
| noise10 | −0.207 | **false** |
| noise0 | −0.395 | **false** |

Interpretation: UTMOS liefert **negative** Deltas für degradiertes Material —
es bewertet verrauschtes Material *besser* als die Referenz. Das ist der
dokumentierte Vertragsgrund für W-10 `gesperrt` (Sprach-Orakel, kein Veto-Recht
bis Musik-Kalibrierung) und bestätigt §III.11 (copilot-instructions.md).

### 3.2 MuQ + A1-Head — richtungsvalidiert (gemessen)

`docs/reports/current/2026-09-13_muq_plugin_direction.json`
(`scripts/validate_muq_plugin_direction.py`, MUSDB-test, Seed 42, 3 Tracks):

| Klasse | Δ_ref (1:1 MuQ-Eval) | Δ_plugin |
| --- | --- | --- |
| band12 | −0.017 | −0.020 (neutral) |
| band8 | −0.281 | −0.227 (neutral/leicht) |
| noise10 | **+2.220** | **+2.213** |
| noise0 | **+3.662** | **+3.237** |

`direction_ok=3`, `direction_total=3`, `direction_neutral=2`. MuQ+A1 erkennt die
Rausch-Degradation korrekt und in die **richtige Richtung** (MOS(ref) > MOS(deg)).
Der Modul-Docstring belegt SRCC 0.957 (MuQ-Eval 2026, A1-Head auf frozen Features).

### 3.3 SingMOS Pro — noch NICHT richtungsgemessen (Lücke, ehrlich §G8)

SingMOS Pro ist deployt (`models/singmos/singmos_pro.onnx`, 1,26 GB externe
Gewichte, ONNX-CPU §v10.17), aber eine **Richtungsstabilitäts-Messung über die
KP-2-Degradationsklassen liegt noch nicht vor.** Der bisherige Einsatz ist
referenzfrei als Vokal-MOS-Gate (D-K4-6/F10, Kalibrierung auf verarbeitete
Musik-Vocals offen). Diese Lücke wird unten (§5) explizit als offener Schritt
geführt — nicht als Messwert erfunden.

---

## 4. Harness — Methodik und Abdeckung

`scripts/benchmark_music_mos_oracles.py` erzeugt pro Klasse deterministisch
(Seed 42, §G5 (copilot-instructions.md)) folgende Degradationen aus der Referenz
und misst **alle drei** Orakel auf (ref, deg):

| Klasse | Erzeugung |
| --- | --- |
| noise_10dB / noise_0dB | additives weißes Rauschen (SNR 10/0 dB) |
| clip_0.3 / clip_0.1 | Hard-Clipping ±0.3 / ±0.1 |
| bandlimit_12k / bandlimit_8k | Butterworth-Tiefpass 5. Ordnung, 12/8 kHz |
| codec_16k | Resample down→up 16 kHz (deterministische Codec-Klasse) |

**Metrik:** Richtungsstabilität = Anteil korrekt geordneter Paare
`MOS(ref) > MOS(deg)` je Klasse und Orakel. **Non-Inferiority:** ΔRate =
`rate(oracle) − rate(utmos)` je Klasse — ein musiknatives Orakel gilt als
mindestens gleichwertig, wenn es die (gemessen invertierte) UTMOS-Rate nicht
unterschreitet. Ausgabe:
`docs/reports/current/<datum>_kp2_musik_mos_orakel.json`.

**Evidenz-Quellen:** MUSDB18-test (musiknativ, §6 `docs/SOTA_TRAINING_KONZEPT.md`)
und `corpus/vinyl` (CC0, clean↔damaged-Paare). Bei Fehlen von MUSDB nutzt der
Harness ausschließlich den Vinyl-Korpus und dokumentiert den Rückfall.

---

## 5. Offene Schritte bis zur Veto-Freigabe (noch NICHT erfüllt)

Die Veto-Freigabe hängt an **Kalibrierung + menschlicher Hörabnahme** und bleibt
gesperrt (§v10.802 (copilot-instructions.md), D-K4-4, W-10 `use_utmos_music=False`):

1. **Harness-Lauf ausführen** (GPU/CPU mit geladenen Modellen) — die Richtungs-
   stabilitäts-Tabelle über alle ≥4 Klassen für alle drei Orakel erzeugen.
2. **SingMOS-Pro-Richtung** erstmals für die KP-2-Klassen messen (bisher fehlt der
   Beleg, §3.3).
3. **Schwellen-Kalibrierung** je Degradationsklasse auf einem MUSHRA-Musik-Subset
   (hängt an D-K4-9-Daten, n≥30 — externer Flaschenhals).
4. **Blindes A/B + Hörabnahme** für einen Menschen (Wohlklang-Vertrag, 5 Belege).

Erst nach (3)+(4) darf ein Flag-Flip erwogen werden — dieser Dispatch **flippt
nichts**.

---

## 6. Akzeptanzkriterien — Selbstbewertung

- **A1** (Evidenz-Report + Richtungsstabilitäts-Tabelle ≥4 Klassen): **teilweise**
  — Report liegt vor; die UTMOS- und MuQ+A1-Richtung ist durch frühere, belegte
  Messungen abgedeckt (§3); die **SingMOS-Richtung und der frische 3-Wege-Lauf**
  stehen als auszuführender Harness-Schritt offen (§5/1 — ehrlich dokumentiert).
- **A2** (report-only, kein Veto, `use_utmos_music=False`): **erfüllt** durch
  Design (`applied` konstant False) + Test `test_witness_always_false_veto_no_model`
  und `test_measure_and_log_does_not_mutate_flags`.
- **A3** (Tests grün, deterministischer Wiederholungslauf): Unit-Tests liegen vor;
  der pytest-Lauf selbst erfolgt durch den Parent (kein Shell im Sub-Agent-Kontext).
- **A4** (FILE_REGISTRY 0/0, id_registry ohne neue Warnungen, Ledger-Snapshot,
  D-K4-4-Blocker): vorbereitet (siehe Dispatch-Bericht).
- **A5** (Bericht): erfüllt durch diesen Report + Dispatch-Report.

---

## 7. Gemessener Harness-Lauf (Manager, 2026-10-25) — A1-Nachtrag

Harness `scripts/benchmark_music_mos_oracles.py` (Seed 42, 4 MUSDB-Test-Tracks,
10-s-Ausschnitte, 7 Degradationsklassen; 48-kHz-Resampling-Fix in Harness + Witness
nachgezogen) → `docs/reports/current/2026-10-25_kp2_musik_mos_orakel.json`.

| Klasse | UTMOS-Rate | MuQ+A1-Rate | SingMOS-Pro |
| --- | --- | --- | --- |
| noise_10dB | **0.0** (richtungsblind, bestätigt) | **1.0** | 0 Messwerte |
| noise_0dB | **0.0** | **1.0** | 0 Messwerte |
| clip_0.3 | 0.25 | 0.25 | 0 Messwerte |
| clip_0.1 | 0.25 | 0.5 | 0 Messwerte |
| bandlimit_12k | 0.5 | 0.25 | 0 Messwerte |
| bandlimit_8k | 0.25 | 0.5 | 0 Messwerte |
| codec_16k | 0.75 | 0.75 | 0 Messwerte |

- **UTMOS-Negativbefund reproduziert:** auf Rausch-Degradationen Rate 0/4 in
  beiden Klassen (bewertet degradiert ≥ Referenz) — deckt sich mit der
  Sept-Evidenz und W-10 (`gesperrt`).
- **MuQ+A1 non-inferior zu UTMOS in 5/7 Klassen**, gleichauf in 1/7, nur
  `bandlimit_12k` −0.25 (Delta-Rate) — auf Rauschen (Kernklasse des
  Restaurierungs-Regelkreises) klar überlegen (1.0 vs. 0.0).
- **SingMOS-Pro: 0 Messwerte** — `model_used != "singmos_pro"` auf diesen
  Tracks (vocal-gated, kein klares Gesangsmaterial) ⇒ Zeuge liefert None mit
  §V6 (copilot-instructions.md)-Begründung; Richtungsmessung bleibt bis
  KP-12/D-K4-6 (Kalibrierung auf verarbeitete Musik-Vocals) offen —
  **ehrlich dokumentiert, kein Messwert erfunden** (§G8 (copilot-instructions.md)).
- Bewertung A1 nach dem Lauf: **teilweise erfüllt** (UTMOS + MuQ+A1 über
  7 Klassen gemessen; SingMOS-Lücke belegt und auf D-K4-6-Daten wartend).
  A2–A5 erfüllt (Manager-Verifikation: pytest 31 grün, Registry 0/0,
  SOTA-Gate 0, ID-Registry ohne neue Warnungen).
