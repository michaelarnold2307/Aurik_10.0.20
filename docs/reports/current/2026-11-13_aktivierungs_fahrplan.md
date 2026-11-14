# Aktivierungs-Fahrplan — exakt geplant, mit Stopp-Kriterien (2026-11-13)

**Anlass:** Keine langen, erfolglosen Trainings mehr. Dieser Plan definiert für
jede offene Aktivierungskette die verifizierte Wurzelursache, den exakten
nächsten Schritt und das billige Stopp-Kriterium VOR jedem GPU-Lauf.

## Verifizierte Wurzelursachen (gemessen, nicht vermutet)

### Kette A — Musik-Declip-Kopf (D-K4-1): technisch KOMPLETT, nur §4 offen
- Produktionspfad-Erreichbarkeit: 6/6 Schichten gefixt, Probe v11 belegt
  Ausführung + Flag-Wirkung (delta 1,86).
- Objektive Gates: MUSDB leak-frei 4/10 besser / 0/10 schlechter (JND 0,05);
  k398-LOO 3/3 besser (Boogie +15,6 dB).
- **Rest: blindes A/B + Hörordnungs-Sign-off (menschlich, §4).**
  → Schritt A-1 (kein Training): Hörpaare generieren + Protokoll.

### Kette B — groove_echo-Klassenkopf: Objektiv-Metrik-Mismatch BEWIESEN
- **In-Sample-Q-Orkel: median 0,0101** (n=110 Q-fähige Trainingsregionen).
  Das Modell erzeugt nicht einmal auf den Trainingsdaten Q ⇒ mehr Daten/mehr
  Epochen können NICHT helfen. Das L1-Wellenform-Objektiv ist blind für die
  Gate-Metrik.
- **Metrik-Konstruktion (defektlast, Zertifikats-Referenz, erste 16384 Samples):**
  `clip(≥0,98)·2 + flatness + hf_verlust·0,5`. Die k398-Rezeptur klippt bei
  ±0,90 ⇒ der Clip-Term zählt die Schäden GAR NICHT; nur Flatness und
  HF-Verlust können Q tragen.
- **Daten-Filter-Lücke:** nur ~37 % der Trainingsregionen haben
  defektlast-Kontrast > JND ⇒ 2/3 des Trainingssignals ist für das Gate
  wertlos.

## Exakter Plan (Reihenfolge verbindlich)

### Schritt B-0 (billig, vor jedem Training): Daten-Filter + Metrik-Proxy
1. Trainingsregionen NUR mit `|defektlast(schaden) - defektlast(clean)| > 0,05`
   behalten (Kontrast-Filter). Manifest `klassen_paare_kombi_gefiltert`.
2. Differenzierbarer defektlast-Proxy als Auxiliary-Loss im Trainer:
   - Soft-Clip-Term: `mean(sigmoid((|x| - 0.90) / tau))` (Schwelle 0,90,
     passend zur Rezeptur statt der Metrik-0,98 — die Metrik honoriert den
     Effekt über Flatness/HF trotzdem, weil de-clippte Plateaus die
     Spektralform ändern),
   - Flatness-Proxy: `exp(mean(log(spec))) / mean(spec)` (identisch zur
     Metrik, differenzierbar über die STFT des Modell-Outputs),
   - HF-Proxy: Energie > 8 kHz / Gesamtenergie (identisch zur Metrik).
3. **Stopp-Kriterium (das neue Pre-Flight-Gate, Laufzeit ~30 s):**
   In-Sample-Q (Train-Split, 300 Regionen, Skript liegt vor) muss **≥ 0,5**
   sein, BEVOR der Holdout bewertet wird. Sonst: Ansatz verwerfen, nicht mehr
   Epochen.

### Schritt B-1: Training mit gefilterten Daten + Metrik-Proxy-Loss
- Nur starten, wenn B-0 (1) und (2) implementiert und per Unit-Test gepinnt
  sind. Budget: 40 Epochen × 300 Schritte (~5 min GPU). Abbruchkriterium
  in-Epoch: In-Sample-Q alle 10 Epochen messen; kein Fortschritt (> +0,02)
  nach 20 Epochen ⇒ Stop.

### Schritt B-2: Gehärtete Holdout-Bewertung
- `bewerte_klassen_kopf.py` auf dem gefilterten Manifest.
- **Aktivierungs-Gate:** Q ≥ 0,5 UND Never-worsen 100 % UND Kontrollregionen
  innerhalb JND. Nur dann: Produktions-Verdrahtung (Challenger hinter
  `use_aspade_declipper`-Muster, Flag AUS) + §4.

### Schritt A-1 (parallel, kein Training): §4-Hörpaket für den Declipper
- Blinde A/B-Paare: `erstelle_aspade_hoerpaare.py` auf PCHIP-vs.-Kopf-Basis
  erweitern (Flag-AUS vs. Flag-AN über den ECHTEN Produktionspfad,
  10-s-Clipping-Fenster, Seed-Seiten, Schlüssel getrennt).
- `hoer_entscheidung.py`-Protokoll + Evidenz-Bundle (alle Gate-Belege).

## Verbindliche Regeln
- **Kein GPU-Lauf ohne erfülltes Pre-Flight-Gate.**
- Jede Stufe bucht ihren Beleg; bei Stopp wird der Befund (nicht der Erfolg)
  committet.
- Flag-Flips bleiben §4 — die Fahrplan-Schritte liefern nur die Voraussetzungen.
