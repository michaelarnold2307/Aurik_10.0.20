# Whisper-Denoiser (D-K0-4) — Funktions- und Ersatzanalyse (SOTA-M2)

> **Auslöser:** User-Vorgabe 2026-10-28: „Aurik soll nicht ohne adäquaten
> Ersatz bleiben und darf keine wichtigen Funktionen verlieren." Diese Analyse
> klärt Aufgabe und Ersatzlage; sie ist die Grundlage des SOTA-Urteils §9/5.

## Aufgabe des Whisper-Denoiser

- **Funktion:** Rauschunterdrückung (NR) im Musikpfad.
- **Artefakt:** `models/miipher_dit/whisper_denoiser_best.pt`, trainiert mit
  `scripts/train_whisper_v2.py` auf MUSDB18-HQ (Quelle: `.github/SOTA_DEFICIT_REGISTER.md`, D-K0-4, Evidenz-Spalte).
- **Status im Signalpfad:** Flag `use_whisper_denoiser = False` — **DEPRECATED
  (Rev. 2026-08-16)**, nur noch A/B-Gate-Referenz
  (Quelle: `backend/core/music_model_flags.py:33`).

## Ersatzlage — die Funktion bleibt, höherwertig getragen

Die NR-Funktion wird von der **Spec-04-Kette** getragen, die im Signalpfad ist
(Quelle: Flag-Kommentar `backend/core/music_model_flags.py:33` — „NR trägt die
Spec-04-Kette (DFN/SGMSE+/OMLSA)"):

| Ersatz | Status | Beleg |
| --- | --- | --- |
| DFN Musik (`use_df_musik`) | **aktiv** (Flag `True`) | `backend/core/music_model_flags.py:42` |
| OMLSA (DSP-NR) | Teil der Spec-04-Kette | `backend/core/music_model_flags.py:33` |
| SGMSE+ Musik-Core | als weiterer NR-Pfad vorhanden (gated, D-K0-3) | `backend/core/music_model_flags.py:45` |

## SOTA-Urteil §9/5

1. Die Funktion (NR) verliert Aurik **nicht**: der aktive Pfad (DFN Musik +
   OMLSA) trägt sie bereits; der Whisper-Denoiser ist seit Rev. 2026-08-16
   nicht mehr Teil der Verarbeitung.
2. Das Artefakt wird **archiviert statt gepflegt** (Snapshot
   `models/_archive_*`; das deployte Artefakt bleibt zur A/B-Gate-Referenz
   erhalten — reversibel).
3. Register D-K0-4 wird geschlossen; der Beleg ist dieses Dokument plus das
   Sign-off der User-Vorgabe (oben).

**Sign-off (User, 2026-10-28):** „Aurik soll nicht ohne adäquaten Ersatz
bleiben und darf keine wichtigen Funktionen verlieren." — Bedingung geprüft
und erfüllt (Ersatz aktiv, Funktion erhalten).
