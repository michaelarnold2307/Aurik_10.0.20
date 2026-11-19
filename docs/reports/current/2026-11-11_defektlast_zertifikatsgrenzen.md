# defektlast — Zertifikats-Grenzen (D-K3-97-Punkt 4, erledigt 2026-11-11)

> Dokumentiert die gemessenen Grenzen der richtungszertifizierten Referenz
> `backend/core/dsp/wohlklang_zeuge.defektlast` (Richtungszertifikat
> `docs/reports/current/qualitaetsmetrik_richtungszertifikat.json`).
> Dieses Dokument ist **Buchführung über die Grenzen**, keine
> Neu-Zertifizierung — eine erweiterte Zertifizierung (Sättigungs-Sensitivität,
> Voll-Fenster) braucht Hör-Labels (D-K4-9) und ist eine
> Metrik-Zertifizierungs-Entscheidung.

## 1. Gemessene Grenzen der zertifizierten Referenz

| Grenze | Befund | Quelle |
| --- | --- | --- |
| **Fenster: erste 16384 Samples** | Clip-Statistik läuft nur über die ersten 0,34 s; Schäden dahinter sind für die Referenz unsichtbar (Test-gepinnt in `tests/unit/test_defekt_delta.py::test_defektlast_lang_sieht_spaete_clips_die_die_referenz_verpasst`) | D-K3-97 |
| **Spektral-Fenster: n=2048** | Flatness-/HF-Anteile nutzen nur die ersten 2048 Samples des Eingangs (Konstruktion der Referenz) | Code `wohlklang_zeuge.defektlast` |
| **Sättigungs-Blindheit (absolute 0,98-Schwelle)** | Sättigung unter der absoluten Schwelle ist unsichtbar: Schaden-Rezept mit Peak 0,9 erzeugt Clip-Anteil 0; der Sinn-Beweis maß Sättigungs-Schaden **besser** als clean (0,273 < 0,280) | D-K3-95/97, Erstbeleg `2026-11-11_b3_defektdelta_erstbelege.md` §3 |
| **RMS-Matching nivelliert Sättigung** | In der Regionen-Messung hob die RMS der Sättigung die Referenz an, das Matching zog den Beweis wieder unter die Schwelle | Erstbeleg §3 |

## 2. Konsequenz für die Regionen-Messung (§G191 (GEBOTE.md))

- `defektlast_lang` (`backend/core/dsp/defekt_delta.py`): identische
  Konstruktion, Clip-Statistik über die **volle Region**; auf dem
  16384-Samples-Fenster bit-identisch zur Referenz (Test-gepinnt). Erbt die
  Grenzen 2–4 — trägt **kein** Richtungszertifikat.
- `flat_top_anteil` (Schema `aurik.defektdelta/2`): relative,
  gain-invariante Sättigungs-Komponente (Anteil an der eigenen
  Hüllen-Decke) — schließt die Sättigungs-Blindheit der Referenz für die
  Regionen-Messung (Report-Only, nicht zertifiziert).
- Pegel-Matching mit **Body-Perzentil p25** statt RMS (Erstbeleg §3).

## 3. Status

- Grenzen sind dokumentiert und Test-gepinnt; die zertifizierte Referenz
  bleibt unverändert und bleibt der B3-Zeuge für Ganzclip-Urteile.
- **Offen (menschlich/Zertifizierung):** erweiterte Richtungs-Zertifizierung
  einer sättigungssensitiven Variante gegen Hör-Labels (D-K4-9) — bis dahin
  ist `flat_top_anteil` reiner Zeuge, kein Richter (Hörordnung §7).
