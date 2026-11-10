# M5-Hörbefund + C4-Teilbefund (Hör-Instanz) — Konsolidierung 2026-11-10

**Anlass:** Im trainingsfreien CPU-Fenster (Kette 2 fährt D3-3 FlashSR) wurde die
B3/C4-Hörproben-Kette gebaut und **eine menschliche Hör-Session** darauf gefahren.
Dieser Report belegt den Stand und zieht die Buchführung nach; er **trifft keine
Hör-Entscheidung** — er protokolliert die bereits ergangenen.

## 1. Was neu ist (Werkzeuge, CPU-only)

| Artefakt | Zweck |
| --- | --- |
| `scripts/erstelle_hoerproben.py` | 10-s-Hörproben **jetzt auf CPU** erzeugen (kein GPU-Fenster nötig): `c4_vocos` (vocos-Base vs. Musik-Finetune-ONNX) und `m5_rekonstruktion` (lineare Interpolation vs. vocos-Resynthese einer 2-s-Lücke); deterministische Auswahl aus `~/Musik` (Seed 42, §G5), Mel train/serve-konsistent (torchaudio, 48 kHz/128 Mel/2048/256); Manifest `aurik.hoerproben/1` |
| `scripts/hoer_entscheidung.py` | Verblindetes A/B-Panel: `--status` (offene Entscheidungen + Proben), `--hoeren <set>` (ffplay, links→rechts), Urteil → `aurik.ab_stats/1`-Beleg, den `flag_freigabe.py` direkt als Hör-Sign-off konsumiert |
| `tests/unit/test_hoerproben_b3c4.py` | 5 Vertragstests (10-s-Mitte, 44,1→48 kHz, Fail-closed ohne ONNX, Status ohne Proben, Manifest-Schema) |

Beide Skripte **flippen kein Flag** und erfinden keine Proben (§G8); der
B3-Stufen-Pfad bleibt ausdrücklich auf das objektive Messfenster verwiesen.

## 2. Hör-Befunde (Hör-Instanz, interaktive Session)

**M5 Lücken-Synthese — `use_vocos_ft_reconstruction`:**

| Kennzahl | Wert |
| --- | --- |
| Verdikte | n = 6 (0 Gleichstände) |
| Siege | linear 6 / vocos-Finetune 0 |
| Vorzeichen-Test | p = 0,03125 ⇒ **signifikant** |
| Beleg | `docs/reports/current/hoer_verdikt_m5_rekonstruktion.json` |

⇒ **Negativer Hör-Sign-off**: die lineare Interpolation ist der bestätigte
SOTA-Zustand. Das Flag bleibt **AUS**; ein §4-Flip entfällt. Der Challenger
`RekonstruktionsDenker._vocos_ft_fill` bleibt als dokumentierter Negativbefund
inaktiv (Default AUS ⇒ bit-identisch).

**C4 Studio-Finisher (Teil) — vocos-Base vs. Musik-Finetune:**

| Kennzahl | Wert |
| --- | --- |
| Verdikte | n = 3 (2 Gleichstände, 1 entscheidend) |
| Siege | A (Base) 0 / B (Finetune) 1 |
| Vorzeichen-Test | p = 1,0 ⇒ **nicht signifikant** |
| Beleg | `docs/reports/current/hoer_verdikt_c4_vocos.json` |

⇒ **kein hörbarer Vorteil** des 30-Epochen-Finetunes auf diesem Material —
konsistent mit der Upstream-Qualitätswarnung (D-K2-3). Die C4-Rollen-Entscheidung
reduziert sich auf den **BigVGAN-Vergleich** (D3-2) und bleibt menschlich.

## 3. Konsequenzen für die Register-/Roadmap-Führung

- **Kein Flag-Flip.** Beide Befunde sind negativ bzw. nicht signifikant; der
  Buchführungszustand „AUS" war bereits korrekt.
- **D-K2-3** bleibt `in-arbeit`: verbleibend ist ausschließlich die menschliche
  C4-Primärrollen-Entscheidung (vocos vs. BigVGAN-F3), nicht mehr der M5-Konsum.
- **B3-Stufen (W-1…W-13)** sind von diesen Proben **nicht** berührt — sie
  brauchen die objektive Messung im trainingsfreien GPU-Fenster (§4-Verfahren).

## 4. Ehrliche Grenzen

- **n = 6 bzw. n = 3** ist klein; die M5-Richtung ist mit p = 0,03125 formal
  signifikant, die C4-Teilfrage ausdrücklich nicht. Beide Sätze gelten nur für
  das gehörte Material (8 bzw. 3 Titel, überwiegend Vertrags-Referenz-Songs).
- Die Urteile wurden **interaktiv** über das Panel eingegeben (menschliche
  Hör-Instanz). Die Belegdateien sind inhaltlich unverändert; ihre neue mtime
  stammt aus der EOF-Normalisierung des Pre-Commit-Hooks, nicht aus einer
  Neumessung.
- Über BigVGAN (D3-2) sagt dieser Report nichts — das Training läuft.

## 5. Verifikation dieser Konsolidierung

- `pytest tests/unit/test_hoerproben_b3c4.py tests/unit/test_ohr_panel.py
  tests/unit/test_denker/test_rekonstruktions_denker.py
  tests/unit/test_vocos_music_ft_flag.py` ⇒ **52 passed**
- Pre-Commit: alle Hooks grün; **`aurik-unit-smoke` bewusst ausgesetzt**
  (Voll-Suite über `tests/unit` bei laufender GPU-Kette — Nutzer-Vorgabe
  „keine Voll-Suites während der GPU-Kette"; Nachzug im trainingsfreien Fenster).
- Sprach-Guard-Befund behoben: die neue Log-Meldung in
  `denker/rekonstruktions_denker.py` war englisch („Fallback") und ist jetzt
  deutsch („lineare Füllung bleibt", §v10.51).
