# WP-3 · Wohlklang-Ordnung-Forensik — Vocoder-Hypothese (Evidenz-Report)

> Datum: 2026-10-25 · Arbeitssprache Deutsch · §-Zitate mit Quelle (AGENTS.md §5
> ID-Zitierdisziplin). Messwerte sind **ehrlich** dokumentiert (§G8
> (copilot-instructions.md)); dieses Dokument ist der WP-3-Beleg
> (`docs/TODOS_SOTA_ROADMAP.md` Arbeitspaket WP-3).

---

## 1. Hypothese und Methode

**Hypothese (WP-3):** Der sprach-trainierte Vocos-48k-Primärpfad des
§1.4-Studio-Finishers (D-K2-3) ist ein Kandidat für Hörordnungs-Verstöße:
Er verbessert niederrangige Ziele (tonale Zentrierung, Brillanz), während er
höherrangige (Authentizität, Emotionalität, Mikrodynamik, Natürlichkeit)
degradiert — genau das verbietet §5 der Hörordnung („ein höherrangiges Ziel
darf für kein niederrangiges gesenkt werden“).

**Methode (`scripts/forensik_vocoder_ordnung.py`, Seed 42, §G5
(copilot-instructions.md)):**
1. MUSDB18-Teststems (vocals + mixture, 2 Tracks), 8-s-Ausschnitte ab 30 s,
   deterministisch auf 48 kHz resampliert (Produktions-Äquivalenz: UV3-intern
   48 kHz).
2. Vocoder-Aufruf identisch zum §1.4-Block:
   `plugins/vocos_plugin.get_vocos_plugin().vocode(audio, 48000, mode="studio2026")`
   — die §V6-Domänen-Warnung des Plugins greift (Log bestätigt).
3. Goal-Scores vor/nach via
   `backend.core.musical_goals.musical_goals_metrics.measure_all(audio, sr)`.
4. Ordnungs-Prüfung je Paar (verbessert, degradiert) mit |Δ| ≥ 0.02
   (Messrauschen-Schwelle, §v10.306 (copilot-instructions.md)) über
   `GoalPriorityProtocol.would_violate_hearing_order`.

Ergebnisdaten: `docs/reports/current/2026-10-25_wp3_wohlklang_forensik.json`.

---

## 2. Ergebnis — Hypothese BESTÄTIGT (36 Verstöße, 4/4 Signale)

| Signal | Verstöße | Muster |
| --- | --- | --- |
| AM Contra/vocals | 6 | tonal_center +0.21 **auf Kosten** von emotionalitaet −0.40, micro_dynamics −0.45, natuerlichkeit −0.11, authentizitaet −0.06 (Tier 1), bass_kraft −0.03 (Tier 2) |
| AM Contra/mixture | 18 | brillanz +0.06 (Tier 4) **auf Kosten** von emotionalitaet −0.42, micro_dynamics −0.45, natuerlichkeit −0.16, authentizitaet −0.10 (Tier 1), artikulation −0.15 (Tier 3) |
| Al James/vocals | 6 | tonal_center +0.25 **auf Kosten** von authentizitaet −0.42, emotionalitaet −0.40, micro_dynamics −0.45, natuerlichkeit −0.14 (Tier 1), bass_kraft −0.89 (Tier 2) |
| Al James/mixture | 6 | tonal_center +0.08 **auf Kosten** von authentizitaet −0.44, emotionalitaet −0.44, micro_dynamics −0.45, natuerlichkeit −0.25 (Tier 1), bass_kraft −0.67 (Tier 2) |

**Befund:** Der sprach-trainierte Vocoder gewinnt systematisch
Tier-3/4-Ziele (tonale Zentrierung, Brillanz) und verliert dabei
Tier-1-Ziele (Emotionalität −0.39…−0.44, Mikrodynamik −0.45 konstant über
alle Signale, Authentizität bis −0.44, Natürlichkeit bis −0.25). Das ist ein
**struktureller Hörordnungs-Verstoß** der primären Finisher-Stufe — nicht
Messrauschen (alle Deltas weit über 0.02).

---

## 3. Konsequenz

- Der Befund bestätigt D-K2-3 (Vocos-48k sprach-trainiert als PRIMÄRE Stufe)
  als **realen Wohlklang-Risikopfad** — nicht nur Domänen-Hygiene.
- Die Gegenmaßnahme ist bereits vorbereitet (2026-10-25, KP-5): der
  BigVGAN(-F3)-Kandidat als Studio-Finisher-Stufe-1 hinter
  `use_bigvgan_f3_studio` (W-13, Default AUS). Die Aktivierung wartet auf die
  **W-1-Hörprobe** (Timbre-Konsistenz, D-K0-1) + Hörordnungs-Sign-off —
  genau das Gate, das dieser Befund normativ erforderlich macht
  (§v10.802 (copilot-instructions.md)).
- Bis zur Aktivierung bleibt der Vocos-Pfad unverändert aktiv (Bestands-
  verhalten); der Forensic-Harness steht dauerhaft bereit, um nach der
  Umstellung den Never-worsen-Nachweis je Goal zu führen (gleiche Messung,
  BigVGAN vs. Vocos).

---

## 4. Akzeptanz (WP-3-Arbeitspaket)

- **Hypothese geprüft:** ja — 36 Verstöße in 4/4 Signalen, Muster
  reproduzierbar (Seed 42, deterministisch).
- **Beleg:** `scripts/forensik_vocoder_ordnung.py` + JSON-Report (registriert
  in `.github/FILE_REGISTRY.md`).
- **Folgeaktion:** W-1-Hörprobe (Nutzer) → KP-5-Aktivierung → erneuter
  Forensic-Lauf als Never-worsen-Gate.
