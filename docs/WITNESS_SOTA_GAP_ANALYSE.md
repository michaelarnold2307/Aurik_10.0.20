# Witness-SOTA-Gap-Analyse: Bildet das Reinhören das menschliche Gehör SOTA nach — und nutzt es SOTA aus?

> Status: Analyse (2026-09-12) · Bezug: `listening_witness.py`, Hörordnung
> (`hoerordnung.instructions.md`), Harmonisierungs-Urteil
> (`docs/MUSICAL_GOALS_WITNESS_HARMONISIERUNG.md`)
> Datum: 2026-09-12

## 1. Die Frage

> „Besitzt Witness-Reinhören alle mit heutiger Technologie möglichen
> Fähigkeiten, das menschliche Gehör SOTA nachzuempfinden — und nutzt
> Witness-Reinhören diese SOTA aus?"

**Kurz-Antwort: Nein — der Witness besitzt nicht ALLE heute möglichen
Fähigkeiten der Gehör-Nachempfindung; er nutzt einen soliden, deterministischen
Teil davon. Der größte Einzelabstand ist nicht fehlende „Tiefe", sondern ein
normativer Gap: Der Witness misst rohe Deltas statt maskierter Hörbarkeit —
die Hörordnung selbst (Ebene 2: Audibility, Maskierungsschwelle statt
Mess-Null) ist im Witness noch nicht implementiert.**

## 2. Bestandsaufnahme: Was der Witness heute misst (9 Metriken)

| Metrik | Technik | Hör-Domäne |
|---|---|---|
| `pitch_drift_cents` | Cross-Signal-ΔF0, Median über stimmhafte Frames | Tonhöhen-Treue (JND-basiert) |
| `pitch_mod_depth_cents` | 3–8-Hz-Modulation der F0-Trajektorie, per Phrase | Vibrato/Stabilität |
| `hnr_drop_db` | De-Krom-HNR via FFT-Autokorrelation | Stimm-Klarheit/Verzerrung |
| `hf_flatness_rise` | Spektrale Flatness im HF-Stimmband | Musical Noise/Rauigkeit |
| `flat_top_rise` | Anteil ±0.98 gepinnter Samples | Clipping/Limiter-Härte |
| `loud_mod_rise_db` | STL-Modulationstiefe 0.2–6 Hz | Lautstärke-Pumpen |
| `bass_drop_db` | Band-Energie-Anteil 20–250 Hz | Bass-Verlust |
| `transient_smear_ratio` | 5-ms-Envelope-Steigung | Anschlags-Schärfe |
| `air_gain_db` | Luftband-Delta 8–20 kHz (2026-09-12) | Brillianz-Verlust |

Bewertung: **ingenieurtechnisch solide, deterministisch (§G5), schnell
(224-s-Song ≈ 1.5 s), layout-sicher, report-only.** Alle neun sind
DELTA-basierte Regressions-Proxies — exakt die Witness-Rolle.

## 3. SOTA-Inventar: Was heutige Technologie kann (und was der Witness davon hat)

| SOTA-Fähigkeit | Stand der Technik | Im Witness? |
|---|---|---|
| Tonhöhen-/Modulations-Analyse | FFT-Autokorrelation, YIN/pYIN | ✅ (de-Krom-Variante) |
| Lautheitsmodell (frequenzabhängige Maskierung) | Glasberg & Moore 2002, ISO 532-2, Johnston 1988 (Maskierungsschwelle) | ❌ **fehlt** |
| Rauigkeit (Roughness) | Daniel & Weber 1997, Vassilakis 2001 (AM 15–300 Hz) | ⚠️ nur indirekt via HF-Flatness |
| Schärfe (Sharpness) | DIN 45692, Aures | ❌ fehlt |
| Tonalität/Harmonizität | Terhardt, Tonhöhen-Salienz | ⚠️ implizit via HNR |
| Objektive Audioqualität (ODG) | PEAQ (ITU-R BS.1387), ViSQOL, DPAM | ❌ fehlt (bewusst: absolut, nicht delta) |
| Sprach-Maße | POLQA (ITU-T P.863), STOI, PESQ | ❌ fehlt (bewusst: speech-only) |
| Gelernte Qualitäts-Prädiktoren | NORESQA-MOS (TorchAudio-Squim), MOSNet | ❌ fehlt (bewusst: ML im Gate-Loop) |
| Binaural/Räumlich | ITD/ILD, Interaural Correlation | ❌ fehlt |
| Trennungs-Qualität | SI-SDR, BSS-Eval | ❌ fehlt (bewusst: gehört in Stem-Ebene) |
| Pre-Echo/Zeitliche Maskierung | Forward-Masking-Modelle | ❌ fehlt |
| Sibilanz-Härte | 5–8-kHz-Rauigkeit/Resonanz-Schätzung | ❌ fehlt |
| Muddiness (LF-Verdeckung) | 250-Hz-Maskierungs-Analyse | ❌ fehlt |

## 4. Gap-Urteil

1. **Vollständigkeit:** Nein. Der Witness deckt ~9 gezielt gewählte
   Regressions-Domänen ab; die Hörordnung verlangt aber Ebene 2
   (Audibility) — d. h. Befunde müssten gegen die **Maskierungsschwelle**
   geprüft werden, nicht gegen starre dB/Cent-Schwellen. Heute meldet der
   Witness z. B. `air_loss` auch dort, wo der Verlust unter der
   Maskierungsschwelle liegt und kein Mensch ihn hören würde.
2. **SOTA-Nutzung:** Teilweise. Genutzte SOTA-Bausteine: FFT-Autokorrelation
   (§V08-konform), de-Krom-HNR, per-Phrase-Modulations-Statistik,
   Transienten-Hüllkurven. Nicht genutzte, aber verfügbare: Maskierungsmodelle,
   Rauigkeit, Schärfe, ITD/ILD. Bewusst ausgeschlossen (und richtig so):
   PEAQ/POLQA-Absolutwerte (Witness ist delta-basiert), gelernte Prädiktoren
   (Determinismus §G5 + Performance-Budget, kein ML im Gate-Loop).
3. **Der wichtigste Einzelschritt** ist damit nicht „mehr Metriken", sondern
   **maskierte Hörbarkeit**: jedes Finding bekommt eine Audibility-Einstufung
   („über Maskierungsschwelle" vs. „Mess-Null-Artefakt"). Das ist genau die
   Konfliktregel der Hörordnung („Metriken sind Zeugen — die Hör-Instanz
   entscheidet"), heute fehlt dem Zeugen das Maskierungs-Wissen.

## 5. Roadmap (priorisiert, determinismus-tauglich)

- **P1 — Maskierungs-Modul ✅ (2026-09-12):** Johnston-1988-vereinfachte
  Maskierungsschwelle (Bark, SFM-Tonalität, Spreading, Sicherheitsmarge) in
  `backend/core/dsp/masking_model.py`; Witness meldet `masked_residual_db` +
  `air_audible` und unterscheidet `air_loss` von `air_loss_audible`.
- **P2 — Rauigkeit ✅ (2026-09-12):** Hilbert-Hüllkurven-Fluktuation
  (20–150 Hz, deterministisch) in `backend/core/dsp/roughness_model.py`;
  Witness meldet `roughness_rise_asper` + Finding `roughness_increase`.
  (Band-weise Vassilakis-Verfeinerung + 5–8-kHz-Sibilanz bleiben Folgeschritt.)
- **P3 — Räumlich ✅ (2026-09-12):** `interaural_cue_integrity` (ITD/ILD/IACC
  vs. Hör-JNDs) im Witness verdrahtet — Felder `itd_drift_us`/`ild_drift_db`/
  `iacc_drop` + Finding `stereo_collapse` bei JND-Verletzung.
- **P4 — Zeitliche Maskierung ✅ (2026-09-12):** Pre-Echo-Proxy
  (`backend/core/dsp/pre_echo_model.py`) — Delta-Energie vor starken Onsets;
  Witness-Feld `pre_echo_db` + Finding `pre_echo` (> −12 dB).
- **Nicht tun:** PEAQ/POLQA/MOS-Prädiktoren im Witness (Rollenbruch:
  absolut statt delta; ML-Bruch: Determinismus), Fusion mit den Goals
  (siehe Harmonisierungs-Urteil §5).

## 6. Fazit

Der Witness ist ein **deterministischer Delta-Zeuge mit 9 scharf gewählten
Sinnen** — er „hört" bewusst wie ein Mess-Protokoll, nicht wie ein
Qualitäts-Prädiktor. SOTA nachzuempfinden ist er heute **nicht vollständig
fähig**; die Lücke ist aber **schließbar ohne Rollenbruch**: zuerst
Maskierungsschwelle (Hörordnung Ebene 2), dann Rauigkeit/Räumlichkeit.
Die Analyse empfiehlt P1 als nächsten Ausbau-Schritt mit Evidenzblock-Pflicht.
