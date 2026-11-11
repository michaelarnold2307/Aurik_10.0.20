# Defekt-Delta-Metrik — Design-Skizze (Muster 3: Reparatur messbar machen)

> Stand 2026-11-11. Skizze als nächster Baustein nach dem Sinn-Beweis
> (D-K3-95) und den ersten echten B3-Ablehnungen (W-3/M12, alle
> pfad_wirksam ✓). **Report-only-Zeuge**, kein Gate, kein Flag-Flip,
> keine Stärke-Änderung — Aktivierung im Entscheidungspfad wäre
> klangwirksam und bleibt §4/Hör-Instanz vorbehalten.

## 0. Problem (Muster 3)

Der Sinn-Beweis 2026-11-11 zeigt: **3/12** Schadens-Paare besser,
**clipping 0/3** (Jets·clipping: defektlast 0,478 → 0,716, musik_mos
4,06 → 3,68). Drei Ganzclip-Metriken sehen Reparatur strukturell nicht:

- `defektlast` liest nur die **ersten 16384 Samples** (0,34 s) und ist
  für **Sättigung blind** (Schaden 0,273 < clean 0,280 — D-K3-95/97).
- `musik_mos` integriert über den ganzen Clip — eine verbesserte
  Defektregion (≈ 1 % der Fläche) geht im Gesamteindruck unter.
- SingMOS-Pro liegt am Ceiling (144/320 Paar-Verletzungen — D-K3-81).

Folge: Solange kein Beleg „diese Defektregion ist besser geworden"
existiert, kann kein B3-Beschluss und kein Hör-Panel gezielt bestätigen,
**wo** Reparatur wirkt. Das Projekt hat einen Never-worsen-Bremser,
aber keinen messenden Beschleuniger.

## 1. Kernidee

Metriken **nur in den annotierten Defektregionen** messen, vorher
(Schaden) und nachher (restauriert), pegel-gematcht — und die
**Reparaturquote** je Defektklasse ausweisen:

- Δ_Metrik(r) = Qualitätsdifferenz schaden→restauriert in Region r
- **Reparaturquote Q(r)** = Δ(schaden→restauriert) / Δ(schaden→clean) —
  „wie viel des vorhandenen Schadens wurde zurückgewonnen" (0 = nichts,
  1 = vollständig, < 0 = verschlechtert; nur mit clean-Referenz).
- JND-Klassifikation je Region mit der einen Wahrheit **JND 0,05** aus
  `backend/core/dsp/wohlklang_zeuge.py`.

Damit wird aus „musik_mos 4,06 → 3,68" ein diagnostisch brauchbarer
Satz: *„Jets·clipping: 4 Sättigungsregionen, Q median −0,2, 3/4 unter
JND verschlechtert — die Verschlechterung sitzt in den Regionen X/Y/Z"*.

## 2. Eingaben

- **Schadens-Tripel** (clean, schaden, restauriert) aus
  `scripts/erstelle_schadens_proben.py` / `scripts/bewerte_schadens_proben.py`
  (deterministische Rezepte, Seed 42, §G5).
- **DefectScanner-Annotationen** des Schadens: Zeitfenster, DefectType
  (65 Typen), per-Instanz-Stärke (§AD — im Lauf-Log sichtbar, z.B.
  „129 echte Defekt-Instanzen in 4 Typen").
- **Pegel-Matching** je Region (RMS/EBU auf den Schaden) — die
  D-K3-95-Lehre: ungematcht misst man den Loudness-Regelkreis, nicht
  die Reparatur.

## 3. Algorithmus (Skizze)

1. **Regionen-Extraktion:** DefectScanner-Fenster (start/end/typ/stärke)
   mit Merge-Regeln für Überlappungen; Mindestlänge je Metrik
   (dokumentiert); Regionen mit mehreren Typen → dominanter Typ + Liste.
2. **Pegel-Matching** je Region: restauriert (und clean) auf den
   Schadens-Pegel der Region (Gain, kein Limiter — Report-Only muss
   bit-treu bleiben).
3. **Regionen-Metriken:**
   - `defektlast_lang` — **NEU**: Voll-Längen-Variante der zertifizierten
     Kurzform; Äquivalenz auf dem Zertifikats-Set nachweisen, sonst
     `verfuegbar=false` + Grund (§V6 (copilot-instructions.md)).
   - `musik_mos` auf der Region (Mindestfenster dokumentieren;
     kürzere Regionen mit Padding-Regel, nie still verwerfen).
4. **Deltas & Reparaturquote** je Region und Metrik; JND-Klasse
   (≥ 0,05 = über Hörschwelle).
5. **Aggregate:** je DefectType (n, Median Δ, Median Q, Anteil ≥ JND),
   je Schadens-Rezept, plus **Kontrollregionen** (annotiert defektfrei):
   dort muss Δ ≈ 0 sein — die Never-worsen-Leitplanke der Metrik selbst.

## 4. Schema & Determinismus

- Schema `aurik.defektdelta/1` — fail-closed: unbekanntes/altes Schema
  ⇒ `verfuegbar=false` + Grund (Muster `ohr_prior`/2-Ablehnung).
- Seed 42, sha256 der Eingangs-WAVs im Beleg, kein `time.time()` in
  Entscheidungslogik (§G5 (copilot-instructions.md)).
- Stereo-Layout-Invariante: (C, N) intern, Kanal-Mittelung explizit
  (Muster `speech_intelligibility` — D-K3-86-Lehre).

## 5. Einordnung in die normative Kette

- **Status: Report-Only-Zeuge** (Muster Wohlklang-Zeuge/SII-Zeuge):
  kein Gate, kein Rollback, keine Export-Blockade, kein Flag.
- **Richtungs-Zertifizierung** (Spearman auf Regionsurteilen) braucht
  Hör-Labels — hängt an D-K4-9 (MUSHRA n≥30)/Hör-Instanz. Bis dahin ist
  die Metrik ein **Zeuge fürs Panel**, kein Richter (Hörordnung:
  „Metriken sind Zeugen, die Hör-Instanz entscheidet").
- Jede Nutzung für Stärke-/Schwellen-Änderungen ist klangwirksam ⇒
  nur über §4/`flag_freigabe.py` mit Hör-Sign-off.

## 6. Bekannte Grenzen → Arbeitspakete

| Grenze | Quelle | Arbeitspaket |
| --- | --- | --- |
| defektlast nur 0,34 s + Sättigungs-Blindheit | D-K3-95/97 | `defektlast_lang` + Äquivalenznachweis + Sättigungs-Grenze als Zertifikats-Grenze dokumentieren |
| musik_mos Mindestfenster | — | Padding-Regel + Test |
| Regionen ohne clean-Referenz | — | Q entfällt, Δ bleibt (nur mit clean sinnvoll) |
| Mehrfach-Defekte je Region | DefectScanner | Typ-Liste statt Einzeltyp; dominanter Typ fürs Aggregat |

## 7. Verifikationsplan (modell-frei, §G5 (GEBOTE.md))

1. Synthetische deterministische Fälle: bekannte Regionen, bekannte
   Deltas ⇒ exakte Erwartungswerte (JND-Grenzen 0,049/0,051).
2. Pegel-Matching-Regression: Gain 1,0 vs. Gain 0,3 je Region
   (D-K3-95-Lehre gepinnt).
3. Stereo-Layout-Invariante; leere Regionen; Schema-Ablehnung.
4. **Diagnostischer Golden-Test:** Jets·clipping reproduzieren — das
   Delta muss die Verschlechterung lokalisieren (defektlast 0,478→0,716
   auf Regionsebene) und die Kontrollregionen ≈ 0 liefern.
5. CPU-only, kein GPU-Fenster nötig; Ruff/py_compile wie immer.

## 8. Integration

- `scripts/bewerte_defekt_delta.py` (neu) — re-used das Pegel-Matching
  von `bewerte_schadens_proben.py`; schreibt Beleg
  `docs/reports/current/b3_defektdelta_<datum>.json` + Report.
- `bewerte_schadens_proben.py` bekommt optional `--defekt-delta`.
- **Später** (nach Zertifizierung): Wohlklang-Zeuge je Lauf um
  per-Defekt-Deltas ergänzen; Kompositions-Suche als objektives
  Kriterium. Beides bewusst NICHT vor Zertifikat.

## 9. Nicht-Ziele

- Keine automatische Stärke-Änderung, keine Flags, kein Deploy.
- Kein Ersatz der zertifizierten Metriken als B3-Zeuge vor Zertifikat.
- Kein „besser/schlechter"-Urteil über den ganzen Song — nur über
  Regionen, nur als Zeuge.
