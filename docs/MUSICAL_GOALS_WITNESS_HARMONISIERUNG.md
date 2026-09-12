# Analyse: Musical-Goals ↔ Witness-Reinhören — Harmonisierungs-Urteil

> Status: Analyse + Umsetzung (2026-09-12) · Bezug: Hörordnung
> (`hoerordnung.instructions.md`), `musical_goals_metrics.py`,
> `listening_witness.py`, `musical_goals_monitor.py`, §v10.19-Paket
> Datum: 2026-09-12

## 1. Die Frage

> „Wollen Musical-Goals und Witness-Reinhören dasselbe? Ist eine Harmonisierung
> der beiden im Sinne des Wohlklangs für das menschliche Gehör sinnvoll,
> erforderlich, obsolet oder kontraproduktiv?"

## 2. Was die beiden Systeme tatsächlich sind (Code-Befund)

| | **Listening-Witness** | **Musical-Goals** |
|---|---|---|
| Datei | `backend/core/listening_witness.py` | `backend/core/musical_goals/musical_goals_metrics.py` (+ Monitor) |
| Frage, die es beantwortet | „Hat DIESE Phase etwas **verschlechtert**?" | „Wie **gut** klingt das Ergebnis (in 15 Dimensionen)?" |
| Bezugsbasis | **Delta** (before/after der Phase), signiert | **Absolut** (Crest-Faktor, Balken, Dynamik …) |
| Metriken | Pitch-Drift/-Modulation (Cent), HNR-Drop, HF-Flatness, Flat-Top, Pumpen (STL), Bass-Drop, Transienten-Schärfe, seit 2026-09-12: `air_gain_db` | bass_kraft, brillianz, waerme, natuerlichkeit, authentizitaet, emotionalitaet, transparenz, groove, spatial_depth, timbral_authenticity, tonal_center, micro_dynamics, separation_fidelity, articulation, … |
| Normative Rolle | **No-Harm-Zeuge** (Primum non nocere, §0; Hör-Invarianten Ebene 1 der Hörordnung) | **Wohlklang-Kompass** (lexikografische Wohlklang-Ordnung) |
| Entscheidungsmacht | REPORT-ONLY als System (§8a); als Zeuge von SLR-Gates befragt (hnr/pitch/flat-top) | Optimierungsrichtung, Monitor-Reporting; keine Hard-Fails auf Absolutwerte (Guard-Kalibrierung) |

## 3. Urteil zur Frage

1. **Wollen beide dasselbe?** Nur im letzten Ziel („Wohlklang fürs menschliche
   Ohr") — **nicht in der Rolle**. Der Witness ist der Wächter der
   Hör-Invarianten („nichts verschlechtern"), die Goals sind der Kompass
   („wohin verbessern"). Ein Schritt kann den Goal-Score heben und trotzdem
   vom Witness abgelehnt werden (z. B. Brillanz-Boost, der HNR kippt) —
   genau für diesen Konflikt existiert die Trennung.
2. **Harmonisierung sinnvoll?** **Ja — auf zwei Ebenen, in dieser Reihenfolge:**
   - **Erforderlich (Rollen-Hierarchie):** Der Witness schlägt die Goals im
     Konfliktfall (Ebene 1 der Hörordnung), die Goals optimieren nur innerhalb
     der vom Witness freigegebenen Menge. Das wird heute praktiziert
     (§SLR-1e2/1e2b/1e3-Gates fragen den Witness, nie den Goal-Score), war
     aber nirgends als Vertrag festgeschrieben — **jetzt dokumentiert** (§5).
   - **Sinnvoll (gemeinsame Mess-Sprache):** Wo beide dasselbe Phänomen
     beobachten, dieselbe Wahrnehmungsdomäne nutzen. Konkret umgesetzt:
     - `air_gain_db` (Luftband 8–20 kHz, signiertes Delta in dB) im
       Witness-Report → der Witness spricht jetzt über „Brillianz" in der
       gleichen Domäne wie das Brillianz-Goal — aber delta-basiert, seiner
       Rolle treu (Regression, nicht Zielwert).
     - `MusicalGoalsMonitor.get_status()` meldet `brillanz` → Monitor und
       Witness berichten komplementär (absolut vs. delta) über dieselbe
       Domäne.
   - **Kontraproduktiv (nicht tun):** Fusion zu einem Gesamt-Score,
     gemeinsame Schwellen, Goal-Maximierung als Witness-Override oder absolute
     Goal-Schwellen als Hard-Fail. Eine Fusion würde den No-Harm-Vertrag
     verwässern (Zielgewinn könnte Regression maskieren); Goal-Schwellen sind
     absolut/kontextabhängig, Witness-Schwellen deltabasiert.
3. **Obsolet ist nichts:** Ein Ziel ohne Zeugen ist blind, ein Zeuge ohne Ziel
   ist ziellos. Beide bleiben nötig — der Witness als Ebene-1-Wächter, die
   Goals als lexikografische Wohlklang-Ordnung darüber.

## 4. Überlappungs-Karte (Mess-Ebene)

| Wahrnehmungsdomäne | Witness-Metrik | Goal-Metrik | Harmonisierungsstatus |
|---|---|---|---|
| Brillianz/Luftband | `air_gain_db` (Delta) + `hf_flatness_rise` (Musical-Noise-Proxy) | `brillanz` (2–16 kHz Crest-Faktor) | **Harmonisiert** (2026-09-12): gemeinsame Domäne, rollengetreue Maße |
| Bass/Wärme | `bass_drop_db` (Delta) | `bass_kraft` (20–250 Hz), `waerme` | Getrennt, komplementär (Delta vs. absolut) — korrekt |
| Stimmtreue | `hnr_drop_db`, `pitch_drift_cents` | `authentizitaet`, `natuerlichkeit` | Getrennt, komplementär — korrekt |
| Dynamik | `loud_mod_rise_db` (Pumpen) | `micro_dynamics`, `emotionalitaet` | Getrennt, komplementär — korrekt |

## 5. Der Harmonisierungs-Vertrag (verbindlich ab 2026-09-12)

1. **Konfliktregel:** Witness-Befund (Hör-Invarianten, Ebene 1) schlägt immer
   den Goal-Score. Ein Gate darf nie eine Witness-Regression mit einem
   Goal-Gewinn verrechnen.
2. **Mess-Sprache:** Wenn ein neues Phänomen in beide Systeme aufgenommen
   wird, definiert der Witness die **delta-basierte** Variante, die Goals die
   **absolute** Variante derselben Domäne — nie zwei konkurrierende
   Definitionen derselben Größe.
3. **Reporting:** Der Monitor berichtet beide Sprachen nebeneinander
   (`brillanz` absolut, Witness `air_gain_db` als Delta) — keine
   Umrechnung, keine Fusion.
4. **Schwellen-Hoheit:** Witness-Schwellen werden nur an No-Harm-Deltas
   kalibriert (Analyse-Plan §5: P90 der No-Harm-Deltas, Evidenzblock-Pflicht);
   Goal-Schwellen bleiben absolutes Reporting ohne Hard-Fail.

## 6. Evidenz (Umsetzung)

- `listening_witness.py`: `air_gain_db`-Feld + `air_loss`-Befund (Schwelle
  −2 dB, konservativ) + `as_dict`-Erweiterung.
- `tests/unit/test_listening_witness.py`: 2 neue Tests (Luftband-Boost ⇒
  positives Delta; Tiefpass-Dämpfung ⇒ `air_loss`) — 19/19 grün, deterministisch.
- `musical_goals_monitor.py`: `get_status()` meldet `brillanz` + letzte Goals
  (im selben Arbeitspaket verdrahtet, §v10.19).
