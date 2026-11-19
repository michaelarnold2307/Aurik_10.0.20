# Hör-Runde 2026-11-05 — Gesamtpaket (Entscheidungsblatt)

**Für:** die menschliche Hör-Instanz (Hörordnung §4/§7). **Dauer:** ca. 45–60 Min.
**Regel:** Es gibt keine richtigen/falschen Ohren — Ihr Eindruck entscheidet. Bei
Widerspruch zwischen Mess und Ohr gilt Hörordnung §7 (Konfliktregel).

## So buchen Sie Ihr Ergebnis

Tragen Sie pro Block Ihr Urteil in **eine** Datei ein
(`docs/reports/current/2026-11-05_hoer_signoff.md`), Format je Block:

```
## <Block-Name>
Urteil: <GO | NO-GO | Unsicher — siehe Kommentar>
Kommentar: <Ihr Höreindruck, ein bis zwei Sätze genügen>
```

Danach führe ich für jedes Urteil das §4-Verfahren aus
(`scripts/flag_freigabe.py --flag … --verdikt … --signoff …`), buche die
Entscheidungen in die Register und committen — **ohne Flag-Flip ohne Ihr
explizites GO**.

---

## Block A — W-1 „Höhen-Behandlung heraushören" (PFLICHT, ~10 Min)

**Warum:** Die einzige Stufe mit widersprüchlichen Objektiv-Belegen
(HF-Gain-Metrik BESTÄTIGT vs. musik_mos ABGELEHNT) — §7 verlangt Ihr Ohr.
**Stimuli:** `output_audio/hr_v1_hoerprobe/` (11 Dateien: ex1–ex3 + elke,
jeweils `*_ohne.wav` / `*_mit.wav`).
**Prüf-Blatt:** `docs/reports/current/2026-10-07_hoerprobe_w1_hrv1.json`
(10 blinde Trials; je Trial: hören, in welchem der beiden Intervalle steckt die
zusätzliche Höhen-Behandlung — oder „nicht hörbar").
**Frage:** Ist die Zusatz-Behandlung überhaupt hörbar — und wenn ja: bereichert
sie (klarere Höhen) oder schadet sie (hart/künstlich)?
**Entscheidung:** W-1 = „behutsam behalten" oder „zurück"? Das Urteil steuert
auch W-13 (BigVGAN-F3-Studio, folgt W-1).

## Block B — W-2 „SCNet-Musik freigeben?" (~10 Min)

**Stand objektiv:** Nachmessung 2026-11-05 ausgewertet: never-worsen ✓,
Vorteile 2/6, p = 0,6875 → **objektiv kein belegter Nutzen**, aber sicher
(Ebene 1–3 unverletzt). Ihre Entscheidung ist Ermessen.
**Stimuli:** `output/scnet_ab_2026-10-06_fair/` (Blind-Paare SCNet vs.
Demucs4-Baseline).
**Prüf-Blatt:** `docs/reports/current/2026-10-08_hoerprobe_scnet_c4_r2.json`
(14 blinde Trials, Frage: welcher Gesang klingt klarer/natürlicher).
**Entscheidung:** `use_scnet_music` aktivieren (WO) oder gesperrt lassen (NO)?

## Block C — EAR-VAE v1 → v2 (~15 Min)

**Stand objektiv:** Blind-Messung 2026-11-04: **16/17 Trials v2-vorteilhaft**,
Sign-Test p = 0,000275, never_worsen ✓ (Δ musik_mos +0,10 / Δ defektlast −0,12).
**Stimuli:** `output/ear_vae_v1_v2/hoer_stimuli/` (blinde Paare v1/v2 je Fenster,
`key.json` zur Entblindung NACH dem Hören).
**Frage:** Bestätigt Ihr Ohr den V2-Vorteil (weniger Artefakte, klarere Textur)?
**Entscheidung:** Austausch v1 → v2 (D-K0-9) — GO / NO-GO.

## Block D — Ermessen ohne Hören (5 Min, aus den Belegen)

Diese Punkte sind dokumentierte Governance-Entscheidungen (Evidenz steht in
Klammern); ein Hören ist optional:

1. **W-8 BW-v5** (`use_bw_v5`): A1-HF-Gain-Gate 2026-10-27 **bestanden**
   (kalibrierte Metrik), aber **+17 dB HF-Boost-Artefakt-Risiko**. GO = freigeben
   (Risiko im Ohr akzeptiert), NO-GO = gesperrt lassen. *(Hören auf Wunsch: ich
   erzeuge ein Vergleichs-Stimulus-Paar.)*
2. **W-10** (`use_utmos_music`): Sprach-Orakel bleibt gesperrt; ersetzen wir das
   Veto durch das **zertifizierte musik_mos** (Spearman 0,95)? GO = Umstellung
   (als Flag-Entscheidung dokumentiert), NO-GO = Status quo.
3. **W-11** (`use_silero_vad_music`): musikadaptiver PANNs-Singing-Offset liegt
   report-only vor. GO = Freigabe als Beratungs-Signal, NO-GO = bleibt aus.
4. **Material-CRNN** (`use_material_crnn`): 95,3 % vs. 40,2 % Genauigkeit
   (objektiv gemessen), verdrahtet hinter Flag. GO = aktivieren, NO-GO = warten.

## Was NICHT in dieser Runde liegt

- W-3 (SGMSE-Musik): **objektiv gemessen 2026-11-05: ABGELEHNT** (6 Trials,
  0/6 Vorteile, p = 0,031 — signifikant gegen eine Aktivierung, never_worsen ✓).
  Hör-Prüfung nur auf Wunsch; objektiv ist die Akte geschlossen.
- W-4/W-5 (MP-SENet, DF-Musik): bereits als „ausnahme" aktiv; Review-Bedingungen
  (Evidenz-Matrix WP-2) sind noch nicht erfüllt.
- W-9/W-12: dauerhaft gesperrt (deprecated / sprachtrainiert).
- MUSHRA-Panel (D-K4-9) und Preference-Labels (D-K4-11): eigene Sessions nach
  `docs/guides/MUSHRA_STUDIENPROTOKOLL.md` (Stimuli-Builder liegt bereit).

---

## Nach Ihrem GO

Für jedes GO führe ich aus: `python3 -B scripts/flag_freigabe.py --flag <name>
--verdikt <Objektivdatei> --signoff docs/reports/current/2026-11-05_hoer_signoff.md`
(vorbereitet, Commit erst nach Ihrer Freigabe), buche das Urteil in
`.github/SOTA_DEFICIT_REGISTER.md` + `docs/SOTA_HANDOVER.md` §2 und aktualisiere
die W-Verträge in `.github/WOHLKLANG_CLAIMS.md`.
