# Hör-Sessions — Vorbereitung für die spätere Abnahme

> **Zweck:** Alles, was für die Hör-Abnahme (GO/NO-GO) nötig ist, liegt hier
> fertig bereit. Du musst später **nur noch abhören** — kein Setup, keine Technik.
> **Hörordnung:** Metriken sind Zeugen, das Ohr entscheidet. Diese Sessions
> entscheiden, ob die Stufen/Modelle von „ausnahme" auf „aktiv" gehoben werden.

## Session 1 · W-1 HR-V1 (Harmonic Repair) — Haupt-Wohlklang-Hebel

**Was du hörst:** Reparierte Höhen/Timbre (W-1) vs. ohne (Original).
**Status:** alle 6 B3-Belege erbracht + Signifikanz 8/0 (p=0,0078) — nur die
Hör-Abnahme hebt es auf „aktiv".

| Datei | Inhalt |
|---|---|
| `output_audio/mushra/thresholds_wohlklang_w1_hrv1/` | Hörprobe (Produktionspfad, Anleitung dort) |
| `docs/reports/current/w1_blind/stimuli/` | 8 verblindete Paare (left/right, neutral) |
| `docs/reports/current/w1_blind/key.json` | Schlüssel (A=W-1, B=ohne) — **nicht ansehen vor der Entscheidung** |

**Frage an dein Ohr:** Klingt die Reparatur (W-1) **natürlicher / voller** (bessere
Höhen/Timbre) ohne hörbare Artefakte? → GO hebt W-1 auf „aktiv".

## Session 2 · 4 gelernte Modelle (Material/Era/Defekt/Loudness) — Voranalyse

**Was du hörst:** Restaurierung mit den gelernten Modellen (Flags AN) vs. ohne
(Heuristiken, Default).
**Status:** objektiv besser (Era-CRNN 2/2 vs Heuristik, Material 95,3 % vs 40,2 %) —
der Wohlklang-Wert ist eine Hör-Entscheidung.

| Datei | Inhalt |
|---|---|
| `study_stimuli/model_ab/*__input.wav` | Eingang (beschädigt) |
| `study_stimuli/model_ab/*__A_heuristik.wav` | Restaurierung OHNE Modelle (Default) |
| `study_stimuli/model_ab/*__B_modell.wav` | Restaurierung MIT den 4 Modellen |
| `study_stimuli/model_ab/manifest.json` | Paar-Zuordnung |

**Frage an dein Ohr:** Klingt die Restaurierung MIT den Modellen (B) **besser**
(natürlichere Ära/Defekt-Behandlung, sauberere Lautstärke) als ohne (A)?
→ GO aktiviert die 4 Modelle (Flag-Flip).

## Ablauf (wenn du bereit bist)

1. **Kopfhörer** (wichtig — die Höhen/Timbre-Details).
2. Session 1 abhören (W-1) → GO/NO-GO notieren.
3. Session 2 abhören (4 Modelle) → GO/NO-GO notieren.
4. Rückmeldung → ich hebe die Stufen auf „aktiv" (oder lasse sie bei NO-GO auf
   „ausnahme"/„gesperrt").

## Ehrlicher Hinweis

Beide Sessions sind **vorbereitet + objektiv belegt** — der einzige fehlende
Schritt ist dein Ohr. Genau das ist die Hörordnung: die Technik ist am Ziel
dessen, was ohne Ohr möglich ist; die Entscheidung gehört dem Gehör.
