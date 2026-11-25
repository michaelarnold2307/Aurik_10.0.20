# §4-Hörsitzung „Hörbarkeit zuerst" — Plan (2026-11-25)

**Auftrag:** „Alles, was zum maximalen Wohlklang beiträgt, ohne Schaden"
(Freigabe 2026-11-25). **Muster-Befund (Prioritäts-Umkehr):** Die Gate-Metrik
ist für die hörbarsten Defekte am blindesten (Sichtbarkeit: clipping±0.9 =
4 %, mp3_128 = 4 %, Rauschen@24dB = 5 % — Q-Mission 2026-11-14). Die
Hörordnung (Ebene 1) priorisiert genau diese Defekte. Deshalb wird die
§4-Sitzung um die **hörbarsten Defekte** gebündelt — nicht um den 17er-
Witness-Satz (der ist auf echtem Material metrisch inert, Paar 2/18:
ΔMOS +0.000 / ΔDL +0.000).

## 1. Sitzungs-Blöcke (Reihenfolge = lexikografische Wohlklang-Ordnung)

| Block | Defekt | Amtsinhaber | Paket-Status | Vergleich |
| --- | --- | --- | --- | --- |
| A | Clipping | `phase_07_declipper` (+ ASPADE-Kandidat `use_aspade_declipper`) | **FERTIG** — `study_stimuli/hoerproben/declip_produktionspfad/` (3 Paare, 10 s, p25-Matching, Schlüssel getrennt; MUSDB-Gate 4/10 besser, 0/10 schlechter, k398-LOO 3/3) | Flag AUS vs. AN |
| B | Wow/Flutter | `phase_12_wow_flutter_fix` | **Generator nötig** — nach A-1-Konvention (Schlimmstfenster, p25-Matching, Schlüssel getrennt, `--produktionspfad`) | Amtsinhaber AN vs. Pass-Through |
| C | Brumm | `phase_02_hum_removal` | **Generator nötig** — wie Block B | Amtsinhaber AN vs. Pass-Through |

## 2. Warum genau diese drei zuerst

- Die drei Defekte sind die hörbarsten Klassen (Hörordnung §8/§8a) und
  zugleich die metrik-blinden (4–5 % Sichtbarkeit) — die Hör-Instanz ist
  hier die **einzige** kalibrierte Instanz.
- Für jeden Block existiert bereits ein produktiver Amtsinhaber — die
  Sitzung entscheidet über **Qualität der Amtsinhaber** (und ggf. des
  ASPADE-Kandidaten), nicht über ungesicherte Witness-Köpfe. Das ist die
  niedrigste Risikostufe mit der höchsten Hörbarkeits-Wirkung.
- Die DRT8-Einzelkanal-Befunde (hum −0.097, wow −0.0035 Defektlast,
  kein Ganz-Titel-Durchbruch) sind metrik-seitig **nicht entscheidend** —
  genau deshalb trägt hier die Hör-Instanz.

## 3. Generator-Konvention (Blöcke B/C)

Wiederverwendet wird das Muster aus `scripts/erstelle_aspade_hoerpaare.py`
(`--produktionspfad`-Modus, 7 Vertragstests):

1. Schadens-Schnipsel mit dem Ziel-Rezept (wowflutter bzw. hum) aus
   `data/schadens_proben`, Schlimmstfenster (max. Defekt-Delta) je Titel.
2. A-Seite = Produktionspfad mit Amtsinhaber-Phase AKTIV (Status quo),
   B-Seite = identischer Pfad mit der Phase DEAKTIVIERT (Pass-Through) —
   oder umgekehrt, Schlüssel getrennt, Seeds fixiert.
3. p25-Pegelmatching (Lautheit darf nicht entscheiden — Hörordnung §7).
4. Manifest + Schlüssel getrennt ablegen (fail-closed gegen Überschreiben),
   Determinismus §G5, CPU-only.

Hinweis: Phase-Deaktivierung prozess-lokal (kein Flag-Datei-Effekt, §G1) —
das existierende Muster aus `scripts/messe_kanalsatz_ab.py` nutzen.

## 4. Ablauf nach Generator-Fertigstellung

1. Pakete B/C erzeugen (je 3–6 Paare), Schlüssel sichern.
2. **Eine** kalibrierte Sitzung mit allen drei Blöcken (A–C) durch die
   Hör-Instanz.
3. Verdikte über `scripts/flag_freigabe.py --flag … --verdikt …` buchen —
   kein Flag-Flip ohne §4-Sign-off.
4. Ergebnis in diesen Plan eintragen; danach erst Metrik-Sichtbarkeits-
   Arbeit (defekt_delta/4-Kanäle, D-K3-95/97) für die nächsten Klassen
   (mp3, Rauschen).

## 5. Explizit NICHT in dieser Sitzung

- Der 17er-Witness-Satz (metrisch inert, keine Hör-Kandidatur).
- SCNet, W-2/W-3/M12-Kandidaten (objektiv abgelehnt — erneute Vorlage nur
  mit neuem Beleg).
- DRT-16/RX-12-Externvergleich (extern blockiert, unverändert).
