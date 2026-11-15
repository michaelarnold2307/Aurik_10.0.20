# Aurik 10 — Spec 26: Defekt-Reparatur-Trainingsvertrag (DRT)

> **Zweck:** Normative Leitplanken für das Training defekt-spezifischer
> Reparatur-Köpfe (Witness-Kopf-Verfahren, D-K3-95-Folge). Verhindert
> unnötige Trainingsläufe, indem jede Verlustrunde der Vergangenheit
> (M9-V1/V2 2026-11-14, Witness-Kopf-V1 2026-11-15, Schutz-Arm-Versager
> 2026-11-15, Produktions-Paritäts-No-Op 2026-11-15) strukturell
> ausgeschlossen wird. Maschinell erzwungen durch den Pre-Commit-Hook
> `aurik-drt-vertrag` (`scripts/compliance/drt_contract_gate.py`, fail-closed).
> Arbeitssprache: Deutsch. Regeländerung ⇒ Gate-Skript nachziehen (AGENTS.md §2).

## Warum diese Spec — die vier Verlustrunden

1. **M9-V1/V2 (TF-GridNet-Denoiser):** optimiert wurde Mask-MSE, gemessen wurde
   Wohlklang — Zielfunktion ≠ Gate-Ziel ⇒ zwei volle GPU-Läufe ohne Wirkung.
2. **Witness-Kopf-V1:** der Witness-Term war durch falsche Mel-Skalierung
   (log10 statt slaney + 10·log10 + amin) numerisch wirkungslos ⇒ Q=None ⇒ Neulauf.
3. **Schutz-Arm-Versager (6 Klassen):** der Kopf überrestaurierte auf clean-Regionen,
   weil das Training nie clean→clean gesehen hatte ⇒ Gate verfehlt ⇒ Nacharbeit.
4. **Produktions-Paritäts-No-Op:** k398-Regionen sind 0,1 s; Trainer/Gate padden auf
   1 s, die Produktions-Trigger-Schleife tat es nicht ⇒ Kanal war ein stiller No-Op.

Jede Runde hätte eine der folgenden Regeln abgefangen — **vor** dem GPU-Lauf.

## DRT-1 — Zielfunktion = Gate-Ziel

Jeder Trainer für einen defekt-spezifischen Reparatur-Kopf MUSS:

- ein **eingefrorenes Gate-Witness** nutzen (Parameter `requires_grad=False`;
  BatchNorm auf eval, GRU im Training-Modus — MIOpen-Backward-Befund 2026-11-15),
- den Witness als **differenzierbaren Loss-Term** führen (Gradient durch die
  Mel-Repräsentation: slaney-Norm + `10·log10` mit `amin`-Clipping wie
  `compute_mel_produktion` — jede andere Skalierung sättigt das Witness),
- die **Checkpoint-Auswahl nach dem Gate-Score beider Arme** treffen (DRT-5),
  niemals nach einem Einzel-Verlust.

## DRT-2 — Zwei-Arm-Loss (Never-worsen ist ein Trainings-Ziel)

Der Verlust MUSS zwei Arme enthalten:

- **Gewinn-Arm:** `L1(schaden→clean)` + λ_w · BCE(Witness(out), 0);
- **Schutz-Arm:** Identitäts-Anker `λ_id · L1(kopf(clean), clean)` mit λ_id > 0.

Der Never-worsen-Schutz des Gates ist damit eine TRAININGS-Eigenschaft, kein
Nach-Filter. Verfehlt ein Kanal den Schutz-Arm, ist die Ursache das Training
(§V7 (copilot-instructions.md): Ursache statt Symptom — niemals die
Gate-Schwelle senken).

## DRT-3 — Pre-flight vor jedem GPU-Lauf

Vor jedem vollen Lauf MUSS ein CPU-Smoke belegen:

- Witness-Kontrast auf den geladenen Regionen: `p_schaden − p_clean ≥ 0,5`
  (sonst kann der Witness das Gate-Ziel nicht tragen — Abbruch VOR dem Training),
- beide Loss-Arme numerisch aktiv (Kontrast-/Größen-Check),
- Determinismus (§G5 (copilot-instructions.md)) und Witness-Checkpoint
  fail-closed (§V6 (copilot-instructions.md)).

## DRT-4 — Parität Training/Gate/Produktion

Fenster, Zero-Padding auf 1 s, RMS-Normalisierung (`RMS_ZIEL` aus einer Quelle,
Import statt Duplikat — §V25 (copilot-instructions.md)), Mel-Repräsentation und
Resampling müssen zwischen Trainer, Gate und Produktions-DSP **identisch** sein.
Jeder Kanal braucht einen **Produktions-Paritäts-Test** (kurze Region triggert,
Ausgabe-Länge = Eingangs-Länge). Muster:
`tests/unit/test_witness_kopf_produktion_paritaet.py`.

## DRT-5 — Gate-Schwellen unantastbar, Gate-Ziel-Auswahl

- Aktivierungs-Gate: **Witness-Q ≥ 0,5** (Median, schaden-Regionen des
  Holdout-Titels, leak-frei) **UND** **Never-worsen ≥ 0,95** (clean-Regionen).
  Beide Schwellen sind normativ fix.
- Modellauswahl im Training über denselben Score (Gewinn- + Schutz-Arm).
- Verfehlen ⇒ Nacharbeit nur über Ursache (Training, Trigger-Schärfe), nie über
  Schwellwert-Senkung oder phasen-individuelle Ausnahmen.

## DRT-6 — Ehrliches Scheitern, kein toter Code

- Gate verfehlt ⇒ Kanal bleibt **report-only**: kein Flag, keine Verdrahtung,
  Q-Erstbeleg mit `status: gate_verfehlt`, Allowlist-Buchung „bewusst-verwaist"
  mit Beleg-Verweis (Modell-Verdrahtungs-Audit).
- Gate bestanden ⇒ Verdrahtung hinter `music_model_flags.use_witness_kopf_<klasse>`
  (Default AUS = bit-identisch; Flag-Flip NUR über §4-Hör-Sign-off), Produktions-
  Konsument in phase_07 mit Doppel-Guard (Witness-Trigger + Harmonik-Proxy gegen
  Input und Amtsinhaber), §4-Hörproben (nur wirksame Regionen, A≠B verifiziert).
- Trainings-/Gate-Infrastruktur bleibt wiederverwendbar (kein toter Code).

## Normative Einordnung

Diese Spec konkretisiert für das Reparatur-Training: §G5 (copilot-instructions.md)
(Determinismus), §V6 (copilot-instructions.md) (kein Silent-Failure), §V7
(copilot-instructions.md) (Ursache statt Symptom), §V25 (copilot-instructions.md)
(kein hartcodierter Schwellwert), §III (copilot-instructions.md) (Soft-Knee
6 dB/200 ms), Hörordnung §4 (Sign-off für klangwirksame Aktivierungen).
