# groove_echo-Kette B: Pre-Flight implementiert — Stopp-Kriterium GEGRIFFEN (2026-11-14)

**Fahrplan-Referenz:** `docs/reports/current/2026-11-13_aktivierungs_fahrplan.md`,
Kette B (groove_echo-Klassenkopf), Schritte B-0 bis B-2.

## Was passiert ist

1. **B-0(2) Proxy-Loss implementiert** (Commit `d1e08fd6`): differenzierbarer
   defektlast-Proxy (Soft-Clip 0,90 + Flatness + HF) als Auxiliary-Loss mit
   Gewicht 0,1 im Klassenkopf-Trainer, 3 neue Vertragstests.
2. **B-1 Training** auf dem gefilterten Kontrast-Manifest
   (`klassen_paare_kombi_gefiltert`, 4.265 Einträge) als `groove_echo_proxy`:
   40 Epochen × 300 Schritte, Seed 42, val_mae 0,04979 (GPU, ~4 min).
3. **B-0(3) Pre-Flight NACHGERÜSTET**: Der Fahrplan verlangt ein
   In-Sample-Q-Gate (300 deterministische Train-Regionen, Q ≥ 0,5) VOR jeder
   Holdout-Bewertung — das Skript fehlte jedoch im Repo (der Fahrplan
   behauptete „Skript liegt vor"). Neu: `scripts/bewerte_in_sample_q.py`
   (Exit 0 = bestanden / 1 = verfehlt / 2 = Pfadfehler) + 5 Vertragstests
   + FILE_REGISTRY-Eintrag.

## Messergebnis — Stopp-Kriterium GEGRIFFEN

| Gate | Soll | Ist | Befund |
| --- | --- | --- | --- |
| In-Sample-Q (Pre-Flight, 300 Regionen) | ≥ 0,5 | **0,0046** | VERFEHLT |
| Holdout-Q (19 Q-fähige Regionen, 1 Titel) | ≥ 0,5 | **0,0115** | VERFEHLT |
| Never-worsen | 100 % | 95,45 % | VERFEHLT |
| Kontrollregionen | alle ≤ JND | verfehlt | VERFEHLT |

Belege: `docs/reports/current/in_sample_q_groove_echo_proxy.json`,
`docs/reports/current/klassen_kopf_q_groove_echo_proxy.json`.

## Befund

Der defektlast-Proxy-Loss hat den bewiesenen Objektiv-Metrik-Mismatch
(In-Sample-Q-Orkel 0,0101) **nicht geheilt**: Das Modell erzeugt nicht
einmal auf den Trainingsdaten Q (0,0046 — schlechter als der Orkel), die
Holdout-Q liegt bei Rauschen (0,0115). Die Fahrplan-Regel greift:
**Ansatz verwerfen, nicht mehr Epochen.** Der L1-Wellenform-/Proxy-Loss-Weg
für groove_echo ist damit dreifach unabhängig widerlegt (Echttraining,
MUSDB-Expansion, Proxy-Loss).

Konsequenz für die nächste Runde (ansatzseitig, kein Training): ein
wellenform-basierter Per-Klassen-Kopf lernt für groove_echo keine
§G13 (copilot-instructions.md)-Q-wirksame Reparatur. Erst wenn ein
Kandidaten-Ansatz das In-Sample-Q-Pre-Flight (≥ 0,5) besteht, ist eine
Holdout-Bewertung zulässig — das Pre-Flight-Gate steht jetzt als
ausführbares Werkzeug bereit.

Kein Flag-Flip, keine Verdrahtung geändert. Der Musik-Declip-Kopf
(Kette A, objektiv 4/10 besser / 0/10 schlechter, k398-LOO 3/3) ist von
diesem Stopp nicht betroffen; sein §4-Hörpaket (Schritt A-1) läuft separat.
