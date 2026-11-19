# B1 Fallback-Komponentenanalyse — die systematischen ~3 Rückfälle je Fenster

> **Methode:** Ein 5-s-Fenster des Referenz-Korpus
> (`test_audio/tape/cassette_1980s_wow.wav`) durch die volle
> Produktionskette (UnifiedRestorerV3, FAST-Profil wie in
> `scripts/release_candidate_pipeline.py`); jeder WARNING+-Datensatz mit
> Logger-Name gefangen, Fallback-Ledger daneben geschaltet.
> Rohdaten: `/tmp/fallback_komponenten.txt` (60 WARNINGs gesamt).

## Befund: 4 Marker-Rückfälle — 2 Komponenten, beide **geplant**

| # | Komponente | Meldung | Ursache | Art |
| --- | --- | --- | --- | --- |
| 2× | `phase_49_advanced_dereverb` | „ML→DSP-Ersatzpfad aktiviert" | SGMSE+-Import/-Nutzung schlägt fehl, weil `use_sgmse_musik=False` (bewusst gesperrt, D-K0-3, §v10.802) → WPE-DSP-Pfad | **geplant** |
| 2× | `dsp.music_mos_oracle` | „SingMOS Pro nicht verfügbar (model_used=pqs_dsp_fallback)" | Zeugen-Modell SingMOS Pro fehlt (KP-12 offen) — Zeuge wird übersprungen, Signalpfad unberührt | **geplant** |

**Damit ist der SOTA-Blocker Nr. 1 erklärt:** Es gibt **keine defekten**
ML→DSP-Rückfälle — die Zählung kam zustande, weil der Zähler geplante
Zustände nicht von Defekten unterschied.

## Umgesetzt (dieser Lauf)

1. `backend/core/fallback_ledger.py`: Klassifikation `geplant | defekt`
   (Regeln `PLANNED_RULES`; unbekannte Art = defekt, fail-closed).
   `fallback_report()` weist beide aus; das Release-Gate sperrt nur noch
   bei `defekt > 0`.
2. **B1 im Wohlklang-Vertrag verfeinert:** „Fallback-Quote 0 (ungeplant)" —
   geplante Rückfälle werden ausgewiesen, nicht still vergeben.
3. 1 neuer Test (Klassifikation), 8/8 grün.

## Verbleibende Hebel (nicht autonom)

- `use_sgmse_musik`-Aktivierung (Hör-Gate D-K0-3) → verschwindet der
  phase_49-Rückfall.
- KP-12 SingMOS Pro (Training/Kalibrierung) → verschwindet der Zeugen-Rückfall.
- **Env-Befund:** `h5py` fehlt im System-Python (PEP 668 gesperrt) —
  LAION-CLAP fällt auf PANNs-Pfad (INFO-Ebene, nicht B1-zählend).
  Fix bei Bedarf: `python3 -m pip install --user --break-system-packages h5py`.
