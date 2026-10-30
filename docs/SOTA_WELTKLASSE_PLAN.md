# SOTA-Weltklasse-Plan — Restaurierung für das menschliche Gehör

> **Zweck:** Geschlossener, begrenzter Plan mit klaren Endzuständen für die Behauptung
> „weltweit hochwertigste Musik-Restaurierung für das menschliche Gehör". Drei Stränge:
> **Verifikation** (wann dürfen wir das behaupten) + **Substanz** (was liefert das Niveau)
> + **Hör-Instanz** (wer entscheidet). Jeder Strang ist endlich — danach operativer
> Maßstab, keine Dauerforderung (§G8, Entscheidung dokumentieren).
>
> **Stand:** 2026-10-30. Keine offene Forderung: jeder Punkt hat ein Fertig-Kriterium.

## 0. Ehrliche Ziel-Definition

„Weltweit hochwertigst für das menschliche Gehör" ist **beweglich** (SOTA entwickelt
sich, „hochwertigst" hängt am Ohr). Operativ erreichbar + ehrlich definiert als:

> **Die aktuelle Welt-SOTA perzeptiv schlagen + den kalibrierten Wohlklang-Balken
> erreichen — robust über Hör-Variabilität.**

Alles im Plan dient diesem prüfbaren Zustand.

## 1. Strang V — Verifikation (zwei Pfeiler)

### Pfeiler A — Absoluter Wohlklang-Balken (MUSHRA-kalibriert)
- **Ziel:** absoluter, wahrnehmungskalibrierter Qualitäts-Balken (nicht relativ zur Vorversion).
- **Scope (fix):** 10 Ausschnitte (4–8 s) aus `golden_set`/`output_audio/mushra`, divers
  über Genre/Ära/Defekt. MUSHRA (ITU-R BS.1534-orientiert), Panel = Hör-Instanz + ≤3.
- **Durchführung (Kun):** referenzfreie Metrik (`musik_mos`+`defektlast`, Spearman 0,95)
  gegen MUSHRA-Scores kalibrieren (monoton, §G5-Seed).
- **Fertig-Kriterium:** (1) Kalibrierung reproduzierbar, (2) Zahlen-Balken steht
  (z.B. „MUSHRA ≥ 80 = hochwertig"), (3) Sign-off Hör-Instanz. → **operativer Maßstab.**
- **Owner:** Kun (Kalibrierung) + Hör-Instanz (Sign-off). Aufwand: ~1 Hörnachmittag.

### Pfeiler B — Perzeptiver Welt-SOTA-Vergleich
- **Ziel:** „weltweit hochwertigst" perzeptiv gegen publizierten Welt-SOTA belegt.
- **Scope (fix):** 1–2 öffentliche Datensätze (MUSDB18-HQ + Restaurierungs-Set) mit
  definierten Defekt-Injektionen; 2–3 öffentliche Baselines (AERO, DDSP-basiert,
  CD-Restaurierungs-Baseline); perzeptive Metriken (MUSHRA/ViSQOL) + interne Proxy
  (SI-SDR nur Zeuge, §4 Hörordnung).
- **Durchführung (Kun):** Benchmark-Auswertung, 95 %-CI (§G5-Seed).
- **Fertig-Kriterium:** Aurik ≥ beste Baseline perzeptiv, CI belegt, Sign-off
  Hör-Instanz. → **Nachweis erbracht.**
- **Owner:** Kun (Auswertung) + Hör-Instanz (Sign-off). Aufwand: 1 Auswertungs-Lauf + Runde.

## 2. Strang S — Qualitäts-Substanz (liefert das Niveau)

### S-1 — K0-Musik-Rollouts (hinter Hör-Sign-off)
- **Was:** `BigVGAN-F3`, `sgmse_musik`, `scnet_4stems` — trainiert + paritätsbewiesen
  (rel ≤ 6,5e-05) + fair-A/B gemessen (+1,96…+3,59 dB bei scnet). C4/MUSHRA-Pakete gebaut.
- **Fertig-Kriterium:** Hör-Abnahme + Sign-off (§v10.802, §III.11) → Aktivierung
  (`use_scnet_music`, BigVGAN-F3-Rollout, sgmse). **Endzustand: alle drei entschieden.**
- **Owner:** Hör-Instanz (entscheidet), Kun (stellt Sign-off-Paket bereit).

### S-2 — Masking-sparse Verarbeitung
- **Was:** nur **hörbare** TF-Regionen verarbeiten, maskierte/unhörbare Regionen
  unangetastet lassen (natürlicherer Klang) — beginnend mit `phase_03_denoise` (global).
- **Fertig-Kriterium:** hörbare Defekte repariert, maskierte unangetastet; A/B-Hörprobe
  bestätigt natürlicheren Klang; Performance-Gewinn gemessen. → **über Pfeiler A prüfbar.**
- **Owner:** Kun (Implementierung) + Hör-Instanz (A/B-Abnahme).

## 3. Strang H — Hör-Instanz (deine Entscheidungen)

Gebündelte Sign-off-Pakete (eine Runde je Paket, kein Dauer-Hören):
- **Paket K0:** BigVGAN-F3 + sgmse + scnet (C4/MUSHRA-Runden 2) → GO/NO-GO je Modell.
- **Paket A/B:** Pfeiler-A-Balken bestätigen.
- **Paket S-2:** Masking-sparse A/B.

## 4. Reihenfolge + Endzustand

1. **K0-Sign-off-Paket** (Strang H) → aktiviert S-1 → hebt das Niveau (Substanz).
2. **Masking-sparse** (S-2) → weitere Qualitäts- + Performance-Hebung.
3. **Pfeiler A** (V) → kalibrierter Balken → macht „hochwertig" messbar.
4. **Pfeiler B** (V) → perzeptiver Welt-SOTA-Vergleich → **Beweis** „weltweit hochwertigst".

**Endzustand (alles erledigt):** K0 aktiviert (Sign-off), Masking-sparse aktiv (A/B),
Balken kalibriert + signiert, Welt-SOTA-Vergleich bestanden (CI + Sign-off). Danach ist
„weltweit hochwertigst fürs menschliche Gehör" ein **belegter, operativer Maßstab** —
nicht mehr und nicht weniger. Keine der vier hat einen offenen Rest.

## 5. Harte Regeln (durchgängig)

- Hör-Entscheidungen nur durch die Hör-Instanz (§1 normative Kette: Hörordnung > Kun).
- Metriken sind Zeugen, die Hör-Instanz entscheidet (§4 Hörordnung).
- Determinismus §G5 (Seeds), §V6 Fallback-Warnung, §V7 Ursache statt Symptom.
- Keine Messung ohne Preflight (Dependency + Smoke-Lauf) — Lektion aus dem RT-Nachtlauf.
