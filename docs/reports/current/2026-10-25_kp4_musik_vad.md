# KP-4 · Musik-VAD (PANNs-Singing-Offset) — Evidenz-Report (Pfad A, KEIN Training)

> Datum: 2026-10-25 · Arbeitssprache Deutsch · §-Zitate mit Quelle (AGENTS.md §5
> ID-Zitierdisziplin). Diese Datei dokumentiert den Ist-Stand **ehrlich** (§G8
> (copilot-instructions.md)): was implementiert ist, was gemessen wurde, und
> welche Lücke bis zu einer Aktivierung offen bleibt.
> Hinweis: Der KP-4-Worker-Dispatch endete mit einem Provider-Stream-Fehler
> nach ~44 Tool-Calls; der Manager hat die verbliebenen Schritte (A/B-Harness,
> Report, Registry, Verifikation) selbst ausgeführt.

---

## 1. Ziel und Rolle

Silero-VAD (English Speech) steuerte in phase_18 einen Gate-Schwellen-Offset
(bis −15 dB). `music_model_flags.use_silero_vad_music == False` (Vertrag W-11
gesperrt) — das Gate läuft heute **ohne Offset** (neutral, §V7
(copilot-instructions.md)). KP-4 baut den **musikadaptierten Ersatz**:
PANNs-Singing-Wahrscheinlichkeit als Offset-Quelle (Muster phase_19) —
**report-only**: Der Offset wird berechnet, geloggt und als Zusatzfeld
ausgegeben; das produktive Gate-Verhalten bleibt unverändert. KEIN Flag-Flip,
KEIN Veto, `use_silero_vad_music` bleibt False.

Normative Anker: §III.11 (copilot-instructions.md) — ein sprachtrainiertes VAD
darf Musik-Gates nicht steuern; Hörordnung §8 — Metriken sind Zeugen, die
Hör-Instanz entscheidet; W-11 (`.github/WOHLKLANG_CLAIMS.md`) — gesperrt bis
Never-worsen-Beweis + Hörordnungs-Sign-off.

---

## 2. Implementierter Stand

| Artefakt | Pfad | Rolle |
| --- | --- | --- |
| Musik-VAD-Offset-Orakel | `backend/core/dsp/music_vad_oracle.py` | `compute_music_vad_offset_db(panns_singing, panns_tags)` + `compute_music_vad_offset_from_audio(audio, sr)`; Offset = affine Skalierung der Singing-Konfidenz auf das bestehende 0…−15-dB-Band; deterministisch (§G5 (copilot-instructions.md)); None/fehlende Evidenz ⇒ neutraler Offset 0 dB **nur mit** `logger.warning` + Begründung (§V6 (copilot-instructions.md)); `applied` konstant False |
| phase_18-Verdrahtung | `backend/core/phases/phase_18_noise_gate.py` | Report-only-Block nach dem Gate: liest `panns_singing`/`panns_tags` aus den Phase-kwargs, berechnet den Zeugen-Offset, loggt ihn und schreibt `panns_vad_offset_db` ins `meta`. **`gated_audio` wird nicht verändert** — der aktive Offset bleibt der bestehende Pfad (bei Flag=False neutral 0) |
| A/B-Harness | `scripts/benchmark_vad_oracles.py` | MUSDB18-Teststems (vocals vs. drums/other) + deterministische synthetische Klassen; Richtungs- und Never-worsen-Metriken |
| Unit-Tests | `tests/unit/test_music_vad_oracle.py`, `tests/unit/test_phase_18_noise_gate.py` (erweitert) | Determinismus, §V6-Fallbacks, Offset-Skalierung, Layout, phase_18-Integration |

---

## 3. Messung (Manager-Lauf 2026-10-25)

`scripts/benchmark_vad_oracles.py` (Seed 42, 6-s-Ausschnitte, 3 MUSDB-Tracks
vocals/drums/other + 3 synthetische Vokal- + 3 Instrumental-Klassen) →
`docs/reports/current/2026-10-25_kp4_musik_vad.json`.

| Klasse | n | PANNs-Singing (max) | Offset |
| --- | --- | --- | --- |
| MUSDB vocals (real) | 3 | **0,09** (Al James), sonst ~1e-4 | 0 dB |
| MUSDB drums/other | 6 | ~5e-4 | 0 dB |
| synthetisch vocal | 3 | ~2e-4 | 0 dB |
| synthetisch instrumental | 3 | ~1e-5 | 0 dB |

- **Richtung: 9/15 (Rate 0.6)** — Instrumental wird konservativ neutral
  gehalten (korrekt), Vokal-Material wird von PANNs auf 6-s-Ausschnitten
  **nicht** als „Singing voice“ erkannt (max. Konfidenz 0,09, überall unter der
  Mindest-Konfidenz 0,25).
- **Never-worsen: 9/9** auf Instrumental — per Konstruktion (Offset 0 ⇒
  identische Gate-Entscheidung wie der neutrale Pfad) **und** gemessen.
- **Ehrlicher Befund (§G8 (copilot-instructions.md)):** Die AudioSet-
  „Singing voice“-Klasse des PANNs-Plugins ist auf kurzen Musik-Ausschnitten
  **nicht diskriminativ** — der Zeuge liefert fast überall Offset 0. Damit ist
  er harmlos (Never-worsen sicher), aber der **vokale Nutzen** ist mit dieser
  Schnittstelle unbelegt.

---

## 4. Offene Schritte bis zur Aktivierung (noch NICHT erfüllt)

Die Aktivierung eines musiknativen Gate-Offsets hängt an (W-11,
`docs/SOTA_TRAINING_KONZEPT.md` §3/KP-4):

1. **Diskriminative Gesangs-Evidenz:** Statt der rohen AudioSet-Klasse die
   plugin-interne Multi-Window-Singing-Score-Logik (`plugins/panns_plugin.py`
   Z. ~601–645, für lange Songs) bzw. AST nutzen und auf ≥3 echten
   Vokal-Stems + Instrumental-Stems kalibrieren — mit Schwellen-Evidenz je
   Klasse.
2. **Never-worsen-Beweis** auf den 3 Referenz-Songs (Gate-Entscheidungen mit
   vs. ohne Offset, 95 %-CI).
3. **Blindes A/B + Hörordnungs-Sign-off** (Wohlklang-Vertrag, 5 Belege).

Erst danach darf der Offset aktiviert werden — dieser Dispatch **flippt
nichts, vergibt kein Veto und ändert das Gate-Verhalten nicht**.

---

## 5. Akzeptanzkriterien — Bewertung (Manager)

- **A1** (Provider deterministisch, None nur mit §V6-Warnung): **erfüllt**
  (`music_vad_oracle.py` + Tests).
- **A2** (Verdrahtung nachweisbar, Gate-Verhalten unverändert,
  `use_silero_vad_music` False, kein Flag-Flip): **erfüllt** — grep
  `panns|music_vad` in `phase_18_noise_gate.py` (23 Treffer), `gated_audio`
  unberührt, Flag unverändert False.
- **A3** (A/B-Tabelle oder ehrlich dokumentierte Lücke): **teilweise erfüllt**
  — Tabelle vorhanden; Diskriminations-Lücke (PANNs auf Kurz-Ausschnitten)
  belegt und als offener Schritt §4/1 geführt.
- **A4** (Tests grün, Registry 0/0, keine neuen ID-Warnungen, Ledger,
  W-11-Kommentar): **erfüllt** nach Manager-Verifikation (pytest 25 grün
  in `test_music_vad_oracle` + `test_phase_18_noise_gate`).
- **A5** (Bericht): erfüllt durch diesen Report.
