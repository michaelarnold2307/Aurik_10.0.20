# Vorbereitungspaket Schritte 0–5 (2026-10-06)

> **Zweck:** Alle Maßnahmen der priorisierten Reihenfolge aus
> `docs/TODOS_SOTA_ROADMAP.md` (Lücken-Matrix §D) sind hier als ausführbare
> Arbeitspakete vorbereitet — mit **Ziel**, **vorbereitetem Bestand**,
> **durchzuführender Aktion**, **Akzeptanzkriterium** und **Blockerklasse**.
>
> **Blockerklassen:** `CPU` (hier sofort machbar) · `GPU` (7900 XTX nötig) ·
> `DATEN` (Korpus/Labels fehlen) · `MENSCH` (Hörpanel/Sign-off) · `EXTERN`
> (Quelle/Download unklar).
>
> **Regel für alle Pakete:** Nichts wird aktiviert, was den Klang ändert, ohne
> A/B **und** Hörordnungs-Sign-off (§v10.802 copilot-instructions.md); jeder
> Statuswechsel im Defizit-Register braucht einen **existierenden** Belegpfad
> (`scripts/sota_deficit_gate.py`, fail-closed).

---

## Schritt 0 — Commit-Abschluss

| Punkt | Stand |
| --- | --- |
| Commit | **erledigt** — `b09c1f91` (`feat(sota): Artefakt-Probe aller Modelle + Herkunftsbelege + beschaffte Bestände`) |
| Push | **erledigt** — `9f8b6d54..b09c1f91  main -> main`; `HEAD == origin/main` |
| Arbeitsbaum | sauber (0 offene Dateien) |
| Blockerrelevante Lehre | Zwei fail-closed Hooks haben den Commit zweimal abgebrochen: `markdownlint` (MD033/MD045 aus **Fremd-README-HTML** in generierter Doku) und `aurik-id-registry` (R2 nackte `§G8`/`§G9`/`§V7`-Zitate). Beides an der **Quelle** behoben (§V7 copilot-instructions.md): Generator sanitized Fremd-Text, Zitate qualifiziert. |

**Resümee für künftige Commits:** Bei generierten Markdown-Dateien mit
Fremd-Inhalten ist der Generator die Wurzel, nicht die Datei; bei §-IDs immer
`§<ID> (<Quelle>)` in **derselben** Zeile.

---

## Schritt 1a — **konsolidiert** (2026-10-06): flacher Material-/Depth-Schätzer (D-K0-8)

**Status: konsolidiert — kein Arbeitspaket und kein offener Messschritt mehr.**

**Beschluss (2026-10-06):**

- **Depth: zurückgezogen.** Die Zahl 85,7 % ist **nicht bewertbar** — die
  Vergleichsgröße `detected_depth` ist `len(result.transfer_chain)`
  (Kettenlänge), nicht die kuratierte Depth-Label-Größe; und der heutige
  Konsument `backend/core/medium_classifier.py` gibt **gar keine** Depth aus
  (`ClassificationResult` ohne `depth`-Feld). Für Depth existiert damit **kein**
  applikationsgleicher Partner. Das Artefakt bleibt unverändert auf der Platte
  (kein Löschen); nur die Doku-Aussage wird richtiggestellt.
- **Material: bleibt `offen`, aber mit dem _richtigen_ Blocker.** Blockerklasse
  **`DATEN`** (Label-Korpus ≥ ~50 je Klasse; heute 6 Klassen, klein und
  unbalanciert — `reel_tape` durchgehend als `tape` fehlklassifiziert); danach
  **Wiring-Entscheidung** (CLAP-Zeuge vs. trainierter Kopf) + **Hörordnungs-
  Sign-off** (§v10.802 (copilot-instructions.md)). Material steuert
  Bandbreiten-Ceilings, Defekt-Schwellen und Era-Priors der ganzen Kette — der
  Hebel ist real, er hängt an **Daten**, nicht an einer Messung.

**Warum kein Re-Measurement mehr:** Für Depth verlangte ein fairer Vergleich
  einen Detektor, der Depth **gar nicht liefert**; für Material wäre der
  Erzeuger derselbe (`forensics.medium_detector`), die Stichprobe aber zu
  klein/unbalanciert für eine belastbare Aussage. Die Konsolidierung ist damit
  ehrlicher als eine Zahl, die eine andere Messgröße beschreibt (§V7
  (copilot-instructions.md) — Ursache statt Symptom).

**Verifizierte Messbefunde (2026-10-06, am Code gelesen — nicht angenommen):**

1. **Erzeuger der Baseline:** `scripts/golden_set_tool.py` schreibt
   `detected_material` = `forensics.medium_detector.get_medium_detector().detect(...).primary_material`
   und `detected_depth` = `len(result.transfer_chain)`.
2. **Depth ist eine andere Messgröße:** `len(transfer_chain)` (Kettenlänge) ≠
   kuratiertes `depth`-Label (1/2/3/4+). Die Zahl 51,8 % ist als Baseline
   **unzulässig**.
3. **Kein Depth-Konsument:** `backend/core/medium_classifier.py::ClassificationResult`
   hat **kein** `depth`-Feld — der Depth-Head hat im heutigen Produktionspfad
   **keinen** Vergleichspartner.
4. **Taxonomie-Bruch (Material):** Der Träger-Space enthält Klassen ohne
   Pendant in den kuratierten 6 (`lacquer_disc`, `wax_cylinder`, …) — die alte
   Baseline hat sie als Fehler gezählt.

**Vorbereiteter Bestand (bleibt gültig, falls Daten kommen):**

- Ground Truth + Items: `audit/golden_listening_set.json` (56 Items mit
  `path`, `material`, `depth`, `era_year`).
- Trainings-/Feature-Rezept (kanonisch, deterministisch §G5 (copilot-instructions.md)):
  `scripts/train_medium_classifier.py::extract_features`.
- Artefakt + Report: `models/medium_shallow_v1.joblib`, `models/medium_shallow_v1_report.json`.
- Erzeuger-Code der Baseline: `scripts/golden_set_tool.py`.

**Akzeptanz (erreicht):** Register **D-K0-8** führt die zurückgezogene
Depth-Angabe nicht mehr als Vorbehalt, sondern als **erledigten**
Konsolidierungspunkt; der Blocker nennt die **echte** Klasse (`DATEN`). Keine
Klang-/Flag-Änderung, kein Artefakt gelöscht.

**Blocker:** `DATEN` (Label-Korpus) für einen späteren Rollout; `MENSCH` für den
Sign-off. **Die Konsolidierung selbst ist abgeschlossen.**

---

## Schritt 1b — **verdrahtet** (2026-10-06): SCNet hinter Never-worsen (D-K0-5 / TODO-P1-2)

**Ziel:** `models/scnet_4stems/huge_scnet_4stems_v1.2.ckpt` als **Qualitäts-Tier**
verfügbar machen, ohne Demucs v4 als Default zu verdrängen.

**Gemessene Ausgangslage (belegt):** Faire Neumessung SI-SDR **+1,96…+3,59 dB**
gegen Demucs v4 (`docs/reports/current/2026-10-06_p1_2_scnet_vs_demucs_fair_ab.md`);
CPU **~7× langsamer** (68,9 s vs. 9,8 s je 30 s).

**Vorbereiteter Bestand:**

- A/B-Harness mit **kanonischem** Demucs-Aufruf: `scripts/eval_scnet_vs_mdx23c.py`.
- Hör-Artefakte je Song: `output/scnet_ab_2026-10-06_fair/` (`mix`, `gt_vocals`,
  `scnet_vocals`, `baseline_vocals`).
- Never-worsen-Arbiter: `resolve_never_worsen` (Spec v10.25, im Denker verdrahtet).

**Durchzuführende Aktion:**

1. Sperr-Flag nach bestehendem Muster anlegen (`use_scnet_music=False`) und den
   Pfad über `resolve_model_path` auflösen — **kein** hartkodierter Pfad.
2. Verdrahtung **hinter** dem Never-worsen-Arbiter: SCNet ersetzt Demucs nur,
   wenn es **auf demselben Song** gewinnt.
3. Sperr-Pin-Test nach dem Muster von
   `test_router_vocal_nr_locks_speech_core_and_reports_reason`: Flag `False`
   ⇒ nachweislich **kein** SCNet-Aufruf.
4. RT-Nachweis gegen die Budget-Tabelle (32×-Guard ist die End-to-End-Norm).

**Ergebnis (2026-10-06) — umgesetzt:**

- **Kanonische Implementierung:** `plugins/scnet_plugin.py` (§G9 copilot-instructions.md:
  _eine_ Quelle) — Laden, Inferenz und Rate-Konvertierung kommen von dort; der
  A/B-Harness `scripts/eval_scnet_vs_mdx23c.py` enthält keine Kopie mehr und die
  Treue-Formel stammt aus `backend/core/dsp/stem_separator.reconstruction_fidelity`.
- **Sperr-Flag** `use_scnet_music=False` in `backend/core/music_model_flags.py`
  (+ `MUSIC_MODEL_PATHS["scnet"]`, bewusst **ohne** Legacy-Gegenstück ⇒ ohne
  Freigabe liefert `resolve_model_path("scnet")` `None`).
- **Verdrahtung hinter Never-worsen:** `MLStemSeparator.separate` wertet SCNet nur
  bei Freigabe als Kandidaten aus; zwei referenzfreie Zeugen entscheiden
  (Rekonstruktions-Treue + Vokal-Erhalt, §v10.26). Ohne Freigabe **kein Import,
  kein Modell-Load** ⇒ verhaltensneutral.
- **Sperr-Pin-Test:** `tests/unit/test_scnet_wiring.py` (10 Fälle).
- **Normativ gepinnt:** `tests/normative/test_primary_paths_no_fallback.py` führt
  `use_scnet_music` als **bewusst gesperrten** Flag.
- **Audit:** Modell-Verdrahtungs-Audit → `scnet_4stems` **verdrahtet**, weiterhin
  **0 undokumentierte Verwaiste**.

**Akzeptanz:** Verdrahtung mit Flag `False` ist **verhaltensneutral** (kein
Plugin-Zugriff — der Test pinnt das), Sperr-Pin-Test grün, A/B-Report verlinkt.
**Offen:** C4-Hörstichprobe (Mensch) und der RT-Nachweis gegen die Budget-Tabelle
(§9.5) — beides **vor** einer Aktivierung.

**Blocker:** `MENSCH` (Hörstichprobe + Sign-off) und die RT-Messung; die
Verdrahtung selbst ist abgeschlossen.

---

## Schritt 1c — F3-BigVGAN-Rollout-Entscheid (D-K0-1 / D-K0-2)

**Ziel:** Entscheiden, ob der MUSDB-Finetune `f3_bigvgan/best.pt` (A/B: HNR
**+4,42 dB**, af +0,0073) die Basis im Produktionspfad ersetzt.

**Belegter Ist-Stand:** Der produktiv geladene ONNX ist **bytescharf die Basis**
(469/783 Tensoren ≠ F3; `ups.1.0.weight_v`: max|Δ| = 0 gegen Basis, 1,0e-1 gegen
F3; ONNX 2026-09-10 < Finetune 2026-10-03) ⇒ der Export kann ihn nicht enthalten.

**Vorbereiteter Bestand:**

- A/B-Beleg: `docs/reports/current/2026-09-16_hr_v1_bigvgan_ab_validation.md`.
- Artefakt-Fingerabdruck: `.github/ML_ARTIFACT_FINGERPRINTS.md`.
- Paritäts-Regel: §III.9 (rel ≤ 1e-3 auf strukturiertem Feed).

**Ergebnis 2026-10-06 (durchgeführt — Beleg:
`docs/reports/current/2026-10-06_p1_3_bigvgan_f3_rollout_entscheid.md`):**

1. ✅ **Export-Rezept an der Wurzel gehärtet** — `scripts/export_bigvgan_v2_onnx.py`
   baut die Architektur jetzt aus der **Checkpoint-Config** (statt Handkopie),
   nutzt Modul-Level-`numpy` und prüft die Parität auf **drei strukturierten
   Feeds** (§III.9 (copilot-instructions.md)-Lehre aus basicpitch).
2. ✅ **F3 → ONNX exportiert, ersetzt nichts** — `models/bigvgan/bigvgan_v2_f3.onnx`
   (2,86 MB + 491 MB `.onnx.data`), Quelle `Checkpoint-cfg (epoch=7,
   val_a1=0.5756958723068237)`, Eingang `[1,128,64]` wie der Basis-Vertrag.
3. ✅ **Paritätsnachweis** — rel `6,521e-05` / `9,705e-06` / `1,693e-05`
   (sane/harmonic/const05) ⇒ 1,5–2 Größenordnungen unter der Grenze 1e-3.
4. ✅ **RT-Messung + Budget-Entscheid — ERGEBNIS NEGATIV.** 12,67× RT (Basis) /
   12,78× RT (F3) im ONNX-Kern; **13,2× RT warm / 16,3× RT kalt** am
   Produktionshelfer ⇒ **792 s je Audio-Minute** gegen ≤ 240 s für die
   **gesamte** Phase-Pipeline (3,3× Überzug). F3 ist +0,9 % ⇒ **kein**
   Laufzeitgewinn. Der Rollout F3 bleibt damit **gesperrt** — nicht wegen des
   Artefakts, sondern wegen der Kostenarchitektur (122 M Parameter auf CPU,
   unbedingter Aufruf aus 5 Phasen).
5. ⛔ **Nebenbefund mit Vorrang:** Die Annahme „Flag OFF“ war **falsch** — der
   Pfad ist am Produktionspfad **aktiv** (`applied: True`, 18 Bänder, Signal
   verändert) und lädt die **Basis** statt F3. Der Flip vom 2026-09-27 stützt
   sich auf eine GPU-Kostenzahl (2,5× RT) und trägt den Budgetbruch nicht.
   ⇒ **Der Aktivierungs-Entscheid für HR-V1 selbst ist jetzt der offene
   Schritt** (Sign-off nach §v10.802 (copilot-instructions.md)), nicht der
   F3-Rollout. Empfehlung im Report: Flag auf `False` bis ein Aufruf
   budgetgedeckt ist — oder Budget-Ausnahme + Reduktion auf eine Aufrufstelle
   mit Längen-Deckel.

**Akzeptanz:** Parität ✅ belegt · Budget ✅ **belegt (negativ)** ·
Hörordnungs-Sign-off ⏳ offen (`MENSCH`). Ohne Budget-Nachweis bleibt der
Rollout **gesperrt** (kein „A/B bestanden“ ⇒ deployt).

**Blocker:** `CPU` für Export + Parität **erledigt**; `MENSCH` für den
Aktivierungs-Entscheid; **Budget** ist der eigentliche Gate-Keeper — und er
sagt derzeit Nein.

---

## Schritt 1d — EAR-VAE v1↔v2 (D-K0-9)

**Ziel:** Entscheiden, ob der v2-Upstream (`models/ear_vae2_upstream/`, 298 M)
den v1-MUSDB-Finetune ersetzt.

**Belegter Ist-Stand:** `models/ear_vae/{encoder,decoder}.onnx` sind
**byte-identisch** mit `ear_vae_ft_*_inline.onnx` ⇒ Produktion **ist** der
MUSDB-Finetune; ein Re-Export reproduziert max|Δ| = 0,0.

**Vorbereiteter Bestand:** Upstream-Klon inkl. `inference.py`/`configs`/`docs`
liegt lokal; v1-Finetune + Re-Export-Rezept liegen in `models/ear_vae/`.

**Durchzuführende Aktion:** Never-worsen-Benchmark v1 vs. v2 auf **demselben**
Song/Korpus (identische Fenster, Seed 42); Kriterium ist ein belegter Gewinn
plus Hörordnungs-Sign-off — sonst **kein** Austausch (§V7).

**Akzeptanz:** Benchmark-Report + Sign-off. **Blocker:** `CPU` + `MENSCH`.

---

## Schritt 2 — Beschaffung (extern)

| Was | Rolle im Wohlklang | Vorbereitet | Blocker |
| --- | --- | --- | --- |
| **AudioLDM2-Plugin** | generative Baseline für destruktive Fälle | ONNX liegt **lokal**; es fehlt nur das Plugin | `CPU` |
| **WF-V4 (Warp-Schätzer)** | Rest-Wow/Flutter unter die Hörschwelle | — | `EXTERN` (Quelle/Download unklar) |
| **TP-V2 (Phasen-Schätzung)** | Transienten-Phasenkohärenz (Kammfilter-Freiheit) | — | `EXTERN` (Modell + Quelle fehlen) |

**Vorgehen:** Beide Extern-Positionen brauchen zuerst eine **Quellenklärung**
(Upstream-Repo, Lizenz, SHA) — ohne Artefakt + Domänenbeleg darf nichts
verdrahtet werden (§III.13 Evidenzpflicht). Für AudioLDM2 ist der nächste
Schritt ein Plugin **nach** dem Muster bestehender ONNX-Plugins
(`resolve_model_path` + EP-Policy + §V6-Fallback), nicht ein Parallelpfad.

---

## Schritt 3 — GPU-Finetunes (7900 XTX)

**Feste Reihenfolge:** DiffWave-Vokal → GaCELA-Vokal → BigVGAN-v2 (Musik+Vokal)
→ FlashSR-Musik → DDSP (BEATs-Encoder statt CLAP) → BEATs-Tagger-Head →
Gender-Head (F13).

**Vorbereiteter Bestand:** Trainingsskripte liegen vor (u. a.
`scripts/train_gender_head.py` mit **Label-Pflicht**, `scripts/train_bigvgan_f3.py`,
`scripts/train_ddsp_predictor_c4.py`); Schutz vor Artefakt-Überschreiben:
`backend/core/training_artifacts.py` (`save_guarded`, rotierendes Backup,
Schrumpf-Warnung).

**Akzeptanz je Lauf:** ΔSDR ≥ **+2 dB**, Never-worsen, Parität (rel ≤ 1e-3),
Registry-Eintrag. Kein Rollout ohne alle drei.

**Blocker:** `GPU` ❌ — zusätzlich `DATEN` für F13/F14 (Label-Korpus ≥ ~50 je
Klasse fehlt; `train_gender_head.py` bricht ohne Labels bewusst mit Exit 2 ab).

---

## Schritt 4 — Hygiene & Performance (CPU, parallel lauffähig)

| Paket | Inhalt | Vorbereitet | Blocker |
| --- | --- | --- | --- |
| **WP-2 Phantome** | 2 tote Referenzen (`ab_test_exports.py` → `bigvgan_v2_f3e29.onnx`; `models/wav2vec2/quality_predict.py` CTC↔Classification-Mismatch) | Fundstellen benannt im Defizit-Register (D-K3-1/-2) | `CPU` |
| **WP-2 Stubs** | 17 TorchScript-Stubs + 1 echter No-Op (`bandwidth_artifact_remover.py:94`) | Fundliste verifiziert (D-K3-3/-4) | `CPU` |
| **P0-1** | Analytik/End-Gate von per-Chunk auf **Song-Ebene** (53× → 32×-Guard) | Roadmap-TODO-P0-1 | `CPU` |
| **P0-3** | drei Budget-Zahlen in **eine** Norm konvergieren | Roadmap-TODO-P0-3 | `CPU` |
| **P1-3** | Audibility (JND/Masking) auf **alle** Schwellwert-Guards | Roadmap-TODO-P1-3 | `CPU` |

**Regel:** „Ehrlich deklarieren **oder** implementieren“ — kein Modul darf mit
einem ML-Pfad werben, der nie läuft (§V6, §G8). Ein No-Op ist zulässig, wenn er
als Never-worsen deklariert ist (Audibility-Prinzip).

---

## Schritt 5 — Menschliches Hörpanel (MUSHRA)

**Ziel:** Kalibrierung des MUSHRA-Proxys (§3.6) und Entscheidbarkeit von F8.

**Warum nicht ersetzbar:** Der Referenz-SCORER ist auf Musik **richtungsblind
bzw. invertiert** (UTMOS-Negativbefund; `use_utmos_music=False`); MuQ ist der
einzige richtungsvalidierte Zeuge (3/3). Ohne Hörstudie bleibt F8
unentscheidbar.

**Vorbereiteter Bestand:** `mushra_harness` (ITU-R BS.1534),
`docs/guides/GO_NO_GO_DECISION_PROTOCOL.md` (beratend), ABX-Contract-Tests
(`tests/.../test_listener_contract.py`).

**Durchzuführen:** Panel-Protokoll (n ≥ 30 Hörer, Blind, randomisierte
Reihenfolge) → Ridge-Regression auf den Hör-Ergebnissen → kalibrierter Proxy.

**Blocker:** `MENSCH` (extern). Nicht durch CPU/GPU-Arbeit substituierbar.

---

## Abhängigkeitsreihenfolge (empfohlen)

```text
0  Commit+Push                        ✔ erledigt
1a Konsolidiert (D-K0-8)              ✔ erledigt (Depth zurückgezogen; Material auf DATEN)
1b SCNet hinter Never-worsen ──┐      CPU-Verdrahtung sofort; Aktivierung nach Sign-off
1c F3-BigVGAN  ────────────────┼────→ Export+Parität CPU; Rollout am Budget-Gate
1d EAR-VAE v1↔v2 ──────────────┘      Benchmark CPU; Austausch nur mit Gewinn
4  Hygiene/P0-1/P0-3/P1-3             CPU, jederzeit parallel
2  Beschaffung                        AudioLDM2-Plugin CPU; WF-V4/TP-V2 extern
3  GPU-Finetunes                      erst nach Beschaffung + Daten
5  Hörpanel                           unabhängig, aber entscheidend für 1b/1c/3
```

**Kritischer Pfad:** 1b/1c (CPU-Verdrahtung) → 5 (Sign-off) → Aktivierung.
Schritt 3 und 5 sind die einzigen **echten** externen Blocker; **1a ist
abgeschlossen** (konsolidiert, D-K0-8).
