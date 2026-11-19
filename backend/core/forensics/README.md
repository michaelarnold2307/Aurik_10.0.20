
# Aurik 6.0 – Forensics

SOTA-konforme forensische Analyse, Medium-Erkennung und Defektbewertung.

## SOTA-Kernmodule & Modularität

**1. PolicyManager (analysis_and_modules.py):**
Adaptives, auditierbares Policy-Management mit Quality-Gates, Eskalationslogik und vollständigem Logging. Steuert alle Qualitätsentscheidungen und ist gender-neutral.

**2. FeatureExtractor (analysis_and_modules.py):**
SOTA-Feature-Extraktor für Audioanalyse. Extrahiert alle relevanten Metriken (F0, RMS, ZCR, Spektralwerte, SNR, SI-SDR, Loudness, etc.) und integriert PolicyManager/QualityEvaluator.

**3. Detector (detector.py):**
Multi-Layer Forensik-Engine für Hypothesenbildung, Evidenz-Management und Transfer-Erkennung.

**4. Signatures (signatures.py):**
Zentrale Definition aller MediaTypes, MediaCategories und akustischer Fingerabdrücke.

---

## Redundanzprüfung & SOTA-Konformität

- analysis_and_modules.py: Enthält PolicyManager und FeatureExtractor, keine Überschneidung mit detector.py oder signatures.py. SOTA-konform und notwendig.
- detector.py: Haupt-Engine, modular und SOTA-konform.
- signatures.py: Typen, Fingerabdrücke, Kategorien, SOTA-konform.
- Keine redundanten oder überflüssigen Dateien vorhanden. Minderwertige/alte Medium-Erkennungen wurden entfernt. Nur MLMediumDetector und MediumDetector sind zulässig.

---

## Regel & Erweiterung

- Neue forensische Funktionen werden als Layer/Funktion in detector.py oder als Modul in analysis_and_modules.py angelegt und hier dokumentiert.
- README regelmäßig aktualisieren, Herkunft und Zweck der Module transparent halten.

---
