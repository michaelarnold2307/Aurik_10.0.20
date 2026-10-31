# D-K1-3 Forschungsvorhaben — Material-Roh-Erkennung SOTA

> **Zweck:** Eigenständiges, datengetriebenes Vorhaben, um die Material-Roh-Erkennung
> (`forensics/medium_detector.py`) von ~43 % (strikt) auf SOTA-Niveau anzuheben.
> Kein Schnellschuss — die Session 2026-10-31 hat gemessen, dass die Feature-Trennung
> die Grenze ist (Gauss ≈ LDA ≈ 40 %; Rekalibrierung/Ära/noise_low_ratio = inkrementell).
>
> **Status-Disziplin:** Jede Phase hat ein messbares Gate. Kein „erledigt" ohne
> gemessene Zahl. Kein Zahl-Trick (§V7 Ursache statt Symptom).

## 0. Ausgangslage (gemessen, 2026-10-31, 122-Datei-Korpus)

| Metrik | Wert |
| --- | --- |
| Strikt (Label == pred) | 52/122 = **42,6 %** |
| Fair (digital→Codec mp3/aac = korrekt) | 57/122 = **46,7 %** |
| Größter Einzel-Hebel bisher | `noise_low_ratio` (Rausch-Tief-Anteil), +3,3 |
| Classifier-Vergleich | Gauss ≈ LDA ≈ 40 % → Classifier nicht der Flaschenhals |
| Register-Befund D-K1-3 | „auf Reals unzuverlässig", nur 1/9 hart korrekt (WP-7) |

**Bekannte Ursachen der Restfehler (gemessen):**
1. **Feature-Trennung:** bandwidth/snr/noise/crackle/wow/block/rotation/infrasonic/nc
   überlappen breit zwischen shellac/vinyl/tape/mp3.
2. **Ära-Erkennung konfundiert:** EraClassifier liest die **mp3-Digitalisierung**,
   nicht die Aufnahme-Ära (shellac→1980 für 29/45, vinyl→1890 für 6/15).
3. **Korpus klein + teils synthetisch:** reel_tape n=7, viele synthetische Testdateien,
   vinyl strukturell 0 echte Träger (D-K1-2).

## 1. Zielbild (SOTA-Definition)

- **Strikt ≥ 70 %** auf dem Echt-Korpus (15 Materialien), fair ≥ 80 %.
- **Keine Regression** auf den 15 Material-Paaren (Spec 05 §6.1).
- **Erklärbar:** die Klassifikation nennt die treibenden Merkmale (für die Hör-Instanz).
- **Deterministisch** (§G5), Guard-konform, §-Zitate mit Quelle.

## 2. Phasen (je mit messbarem Gate)

### Phase A — EraClassifier de-konfundieren (größter einzelner Hebel)
**Problem:** Die Dekaden-Erkennung nutzt die Bandbreite (HF-Rolloff) → die mp3-Digitalisierung
lässt shellac „modern" (1980) aussehen. Die Ära *könnte* shellac/vinyl/digital sauber trennen.
**Auftrag:** Die Aufnahme-Ära von der Digitalisierungs-Ära trennen.
- (A1) Codec-Signatur (MDCT/Pre-Echo) als *Negativ-Merker* für die Ära: wenn ein Codec-Layer
  vorliegt, die Bandbreiten-basierte Dekaden-Schätzung dämpfen + auf die Rausch-/Inhalt-Merkmale
  gehen (§2.13 `_ANALOG_ERA_CEILING` konsequent anwenden).
- (A2) `MEDIUM_DECADE_FLOOR` + `constrain_era_to_medium()` nutzen, damit die Ära-Kette
  (shellac Pre-1950, vinyl Post-1950) die Dekade begrenzt.
- **Gate A:** Ära-Erkennung auf dem Korpus: shellac-Dekade ≤ 1950 für ≥ 80 % (heute 29/45 = 1980
  falsch), vinyl ≥ 1950 für ≥ 80 % (heute 6/15 = 1890 falsch). Danach Material-Genauigkeit neu messen.

### Phase B — Diskriminative Features (Rausch-Spektralprofil + Harmonische)
**Problem:** Die 11 Features trennen nicht genug.
**Auftrag:** Physikalisch diskriminative Merkmale ergänzen.
- (B1) Volles **Rausch-Bark-Profil** (24 Bänder, aus `EraResult.noise_profile`) statt nur
  `noise_low_ratio` — die Rausch-FORM (shellac breit vs. vinyl Rumble vs. tape Hiss).
- (B2) **Harmonische-/Transienten-Merkmale** (Oberflächenrauschen vs. Rillenrauschen).
- (B3) **Codec-Signatur** (MDCT/Pre-Echo) als eigenes Merkmal (trennt digital sauber).
- **Gate B:** Material-Genauigkeit +≥ 10 Prozentpunkte über Phase A (gemessen, strikt).

### Phase C — Korpus + gelernter Classifier
**Problem:** kleiner/synthetischer Korpus; Gauss-Independence schwach.
**Auftrag:** (C1) mehr Echt-Träger (D-K1-2: 63 Echt-Träger, vinyl-Rechtepolitik);
(C2) gelernter Classifier (LDA/Ensemble) auf den Phasen-A/B-Features, sobald die Features
trennen. **Gate C:** Ziel ≥ 70 % strikt erreicht + 15-Material-Regression grün.

## 3. Harte Regeln
- Spec > Code > Kommentar (AGENTS.md §1). Determinismus §G5. Silent-Failure §V6 (logger.warning).
- Keine phasen-individuellen Schwellwerte (§V7). Kein Zahl-Trick (kein Assert-Lockern,
  keine Metric-Kosmetik). Jede Zahl aus einer reproduzierbaren Messung (Befehl im Report).
- Write-Gate bei neuen Dateien (FILE_REGISTRY), §-Zitate mit Quelle (id_registry fail-closed).
- Hörordnung: die Hör-Instanz bleibt finale Autorität; Metriken sind Zeugen.

## 4. Nächster Schritt (sofort)
**Phase A (EraClassifier de-konfundieren)** — größter einzelner Hebel.
Erste Messung: Ära-Genauigkeit shellac/vinyl auf dem Korpus (Gate A-Baseline),
dann A1/A2 umsetzen + Gate A messen.
