# D-K3-57 — Reinhör-Witness pre_echo: Repro-Werte des 10.12.25-Laufs (2026-10-09)

**Frage (Register D-K3-57):** Der Walzer-Lauf („Trio Schweizer – 13 Tage", 10.12.17) zeigte
`pre_echo`-Befunde auf `phase_01` bei komplett neutralen Nachbarwerten; ohne den echten
`pre_echo_db`-Wert war der Trigger nicht entscheidbar. Nach dem Diagnostik-Fix (10.12.17:
`pre_echo=%.1fdB` in der Witness-Logzeile) wurde der **nächste volle Lauf derselben Quelle**
mit aktuellem Code (10.12.25) gefahren — dieser Lauf liefert die Repro-Werte.

## Messwerte (Log `trio_lauf_101225.log`, Witness-Zeilen mit Satzzeit)

| Zeile | Phase | pre_echo | Nachbar-Messwerte |
| ---: | --- | ---: | --- |
| 1373 | `phase_01_click_removal` | **−2,8 dB** | pitch=0.0c mod=0.0c hnr=0.0dB hf=0.003 loud=0.9dB |
| 1517 | `phase_12_wow_flutter_fix` | **−8,4 dB** | pitch=0.0c mod=0.0c hnr=−0.0dB hf=0.002 loud=0.0dB |
| 1667 | `phase_24_dropout_repair` | **−10,5 dB** | pitch=0.0c mod=0.0c hnr=0.0dB hf=0.000 loud=0.5dB |
| 3892 | `phase_12_wow_flutter_fix` (später Chunk) | **+0,1 dB** | pitch=0.0c mod=0.0c hnr=0.0dB hf=0.002 loud=0.0dB |
| 6188 | `phase_12_wow_flutter_fix` (später Chunk) | **+0,2 dB** | pitch=0.0c mod=0.2dB hnr=0.001 loud=0.0dB |

## Einordnung

1. **Deterministisch identisch zum 10.12.17-Lauf** (−2,83 / −8,36 / −10,46 dB aus den
   `AURIK_WITNESS_DEBUG`-Dumps, D-K3-62-Evidenz): derselbe Trigger, dieselben Werte —
   **kein Phantom**, kein Zufall, keine Chunk-Nichtdeterminismus-Reste im 5-ms-Raster.
2. **Ursache = D-K3-62-Klasse:** Hinzugefügte Reparatur-Energie (Klick-Ersatz phase_01,
   Wow-Resampling phase_12, Dropout-Füllung phase_24) passiert die 5-ms-Pre-Fenster-Gates.
   Die Dumps des 10.12.17-Laufs belegten positionsidentische Wellenform-Peaks (kein
   Echo-Duplikat); die späten +0,1/+0,2-dB-Werte sind Chunk-Kontext-Varianten derselben Klasse.
3. **Kein Nachverfolgen als Defekt (Audibility, Hörordnung Ebene 2):** Die Energie ist die
   GEWÜNSCHTE Reparatur; der Diskriminator (`analyze_pre_echo`, 10.12.27) klassifiziert
   solche Fälle als `repair_fill` (report-only). Eine Verschärfung/Veto braucht erst die
   D-K3-62-Hörstichprobe.
4. Der ursprüngliche Walzer-Wert −2,8 dB ist damit vollständig erklärt; die offene
   Frage „Fix metrik- oder phasenseitig?" beantwortet sich als **keiner von beiden**:
   die Metrik-Sicht ist korrekt für ihr Fenster — die Einordnung übernimmt der Diskriminator.

_Beleg: Log-Zeilen oben (Lauf 10.12.25, Quelle `test_audio/Trio Schweizer - 13 Tage.mp3`),
D-K3-62-Dump-Reproduktion `tmp_repro/pe_verify.py` (2026-10-07). Stand 2026-10-09._
