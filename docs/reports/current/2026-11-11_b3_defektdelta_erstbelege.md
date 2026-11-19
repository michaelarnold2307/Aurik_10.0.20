# Defekt-Delta-Erstbelege — Jets·clipping (Golden-Test, §G191 (GEBOTE.md))

> Stand 2026-11-11. Erste Messfahrten des Defekt-Delta-Witness
> (`aurik.defektdelta/1`, Spec 07 §8.6) auf dem Sinn-Beweis-Paar
> Jets·clipping. Report-Only-Zeuge — kein Gate, kein Flag, keine
> Stärke-Änderung (Hörordnung §7).

## 1. Setup

- Schadens-Tripel: clean-Snippet (deterministisch: `_decode_48k` +
  `_mittel_ausschnitt` 10 s, Mittelteil, gleiche Kette wie
  `erstelle_schadens_proben.py`), Schaden + restauriert aus
  `/home/michael/Musik/schadens_proben/` (Jets·clipping, 480000 Samples
  @ 48 kHz).
- Regionen: `DefectScanner().scan()` auf dem Schaden →
  `regionen_aus_defekt_scores` (Schwelle severity 0,1) +
  Kontrollregionen (feste Anteile, max. 3).
- Metrik: `defektlast_lang` (Voll-Fenster, auf dem Zertifikats-Fenster
  bit-identisch, Test-gepinnt), JND 0,05 (eine Wahrheit
  `wohlklang_zeuge.JND_TOLERANZ`).
- CPU-only (`AURIK_FORCE_CPU=1`), deterministisch (§G5
  (copilot-instructions.md)); Belege mit sha256 der Eingänge.

## 2. Beleg 1 — echtes Paar (No-Op-Befund auf Regionsebene)

`docs/reports/current/b3_defektdelta_jets_clipping.json`

- 377 Scanner-Regionen extrahiert, **173** ≥ 2048 Samples in 11
  Defekttypen (keine Kontrollregionen — die Defektabdeckung traf alle
  festen Kandidaten-Positionen).
- **Ergebnis: alle Δ = 0,0, alle `unter_jnd`, Q = null** — der
  restaurierte Export ist für dieses Paar **bit-identisch zum Schaden**
  (Gain 1,0). Das ist der auf Regionsebene belegte D-K3-97-Folgebefund:
  die Pipeline entschied „nichts tun" (No-Op). Der Witness macht den
  No-Op **sichtbar** statt ihn zu verschweigen — genau die §G191-Pflicht.
- Pegel-Befund: clean-RMS 0,152 vs. Schaden-RMS 0,570 (die
  Clipping-Rezeptur komprimiert ≈ +11,5 dB bei gleichem Peak) —
  Gain-Matching je Region greift (Faktor ≈ 3,8, innerhalb der
  4,0-Klammer).

## 3. Beleg 2 — Kontrast-Messung (Metrik-Grenze dokumentiert)

`docs/reports/current/b3_defektdelta_jets_kontrast.json`

- Kontrast: Schaden vs. **deterministische Flat-Top-Variante**
  (`clip(x·1,5; ±0,99)`, 35,8 % der Samples > 0,98).
- **Befund: keine Verschlechterung messbar** — nach dem RMS-Matching
  (Gain < 1, weil die Flat-Tops die RMS anheben) liegen die Flat-Tops
  wieder unter der absoluten 0,98-Schwelle der zertifizierten
  Clip-Statistik; Δ bleibt sub-JND (+0,0079). Die Sättigung wird vom
  Matching **nivelliert**.

### Methodische Konsequenz (Zertifikats-Grenzen, D-K3-97-Klasse)

1. **RMS-Matching nivelliert Sättigungs-Signaturen** — Sättigung erhöht
   die RMS; das Matching zieht den Beweis wieder unter die Schwelle.
   Künftige Iteration: Matching auf ein robustes **Body-Perzentil**
   (z. B. 25. Perzentil von |x|) statt RMS, plus eine **relative**
   Clip-Statistik (Flat-Top-Anteil relativ zum gematchten Body-Niveau).
2. **Absolute 0,98-Schwelle + 2048er-Spektralfenster** der zertifizierten
   Referenz sind als Grenzen zu dokumentieren (Spec 07 §8.6 verlangt die
   Zertifikats-Grenzen-Dokumentation); `defektlast_lang` erbt beide.
3. Punkt-Events des Scanners (Clicks: (t, t)) und Regionen < 2048 Samples
   werden §V6 (copilot-instructions.md)-geschützt übersprungen — kein
   stiller Ausfall.

## 4. Status & offene Punkte

- **Erbracht:** erster End-to-End-Beleg des Witness auf echtem Material
  (No-Op sichtbar) + Kontrast-Beleg, der die Metrik-Grenzen
  **maschinenlesbar** dokumentiert.
- **UMSGESETZT (gleicher Zyklus):** relative Sättigungs-Komponente —
  Schema `aurik.defektdelta/2` (Body-Perzentil-Matching p25 statt RMS +
  `flat_top_anteil` relativ zur eigenen Hüllen-Decke, gain-invariant);
  `/1`-Belege werden als `veraltet` markiert statt still interpretiert
  (ohr_prior-Lehre, D-K3-89).
- **Offen:** Richtungs-Zertifizierung über Hör-Labels (D-K4-9) — bleibt
  Voraussetzung für jede B3-Nutzung (Report-Only-Zeuge).

## 5. Iteration /2 — Ergebnis (v2-Belege)

- **Beleg 1 v2** (`b3_defektdelta_jets_clipping_v2.json`): No-Op-Befund
  bestätigt — alle Δ = 0, alle `delta_flat_top` = 0 (restauriert ist
  bit-identisch zum Schaden).
- **Beleg 2 v2** (`b3_defektdelta_jets_kontrast_v2.json`): Die
  Flat-Top-Variante wird jetzt **lokalisiert**: **172/173 Regionen**
  `jnd_klasse_flat_top = verschlechtert` (Median Δ ≈ −0,33, Anteil 1,0 in
  allen 11 Defekttypen; Beispiel phase_rotation: flat_top 0,015 → 0,345).
  Die relative Statistik sieht die Sättigung, die das RMS-Matching unter
  die absolute 0,98-Schwelle gezogen hatte — die in §3 dokumentierten
  Grenzen sind damit behoben bzw. als Zertifikats-Grenzen der Referenz
  dokumentiert.
