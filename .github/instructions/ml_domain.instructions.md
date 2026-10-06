---
applyTo: "{plugins/**/*.py,backend/**/*.py,denker/**/*.py,models/**/*.py,scripts/*.py,Aurik10/**/*.py,cli/**/*.py}"
---

# ML-Domänen-Regel (normativ) — Aurik 10

> Kanonische Quelle: **`.github/ML_MODEL_DOMAIN_REGISTRY.md`**.
> Diese Datei ist der Kurz-Zwang, sie zu lesen, bevor du über ML-Modelle
> sprichst oder sie verdrahtest.

## ⛔ Bevor du behauptest, ein Modell sei „Sprache" — prüfe die Domäne

**Der häufigste Fehlschluss in diesem Repo ist FALSCH:**

> „Ein Großteil der lokal verfügbaren ML-Modelle sind Sprachmodelle."

**Fakt (belegt):** Von **61** Modellverzeichnissen in `models/` sind
**37 musik-trainiert (≈ 61 %)**, nur **11 sprach-trainiert (≈ 18 %)**.
Aurik ist ein **Musik-Restaurierungssystem** — der Signalpfad-Kern
(Separation, Pitch, Gesang, Inpainting, Deklipper) ist **Musik**.

## Regeln

1. **Kein Domänen-Urteil aus dem Namen.** `whisper`, `silero`, `utmos`,
   `bigvgan`, `deepfilternet`, `sgmse_plus`, `miipher`, `versa` klingen nach
   Sprache — die **produktiv geladenen** Artefakte sind teils Musik
   (`miipher_dit` = MUSDB-Singstimme, `ear_vae` = offizielles Musik-Modell,
   `deepfilternet/finetuned/` = Musik-Finetune).
2. **Evidenz vor Behauptung.** Nur gültig mit: Trainings-Skript-Korpuspfad,
   vendored Modell-Karte, SHA-256-Identität oder Plugin-Header **ohne
   Widerspruch**. Fehlt das → Domäne = **unbekannt**, **nicht** „Sprache".
   **Belegter Fall (2026-10-06, am Artefakt gemessen):** `bigvgan_v2.onnx`
   (128 Mel-Bänder/512×/1536 Kanäle = 44,1-kHz-Konfiguration) und
   `hifi_gan.onnx` (80 Mel-Bänder/256×/128 Kanäle, 0,93 M Parameter = kein
   offizieller V1) sind **`unbekannt`** — die alte Zuordnung
   „LibriTTS"/„UNIVERSAL_V1 = Sprache" ist architektur-**widerlegt**
   (Mess-Block §4a in `.github/ML_MODEL_DOMAIN_REGISTRY.md`).
3. **Signalpfad-Regel (§III.11 copilot-instructions.md).** Ein sprach-trainierter
   Kern darf als Musik-Signalpfad **nur** mit belegtem Musik-Finetune + A/B
   laufen. Sonst: musik-trainierter Kern oder DSP.
4. **Zeugen ≠ Signalpfad.** Whisper/UTMOS/Silero/Resemblyzer sind Metriken/VAD/
   Embeddings und dürfen das Ausgangssignal **nicht** verändern.
5. **Sprach-Vocoder/SR (Vocos, NVSR, AERO)** sind als Musik-Pfad ein
   Risiko und nur nach §V6 (copilot-instructions.md) (Warnung + Grund) als
   Fallback zulässig. Für `hifi_gan`/`bigvgan` ist die Sprach-Herkunft
   **widerlegt bzw. unbelegt** (`unbekannt`, s. Regel 2) — sie bleiben als
   **unbelegte** Signalpfade im Defizit-Register (`.github/SOTA_DEFICIT_REGISTER.md`,
   D-K0-2, D-K2-1) und ebenso nur §V6-Fallback.
6. **`nara_wpe`** ist unüberwacht (kein Training) → domain-neutral, **kein**
   Sprach-Modell.

## Pflege

- Neue Domäne prüfen: `python scripts/model_inventory.py --domains`
- Registry aktualisieren **im selben Commit** wie ein neues Modell/Finetune
  (§G9 copilot-instructions.md).
- Vollständige Tabellen und Belege: `.github/ML_MODEL_DOMAIN_REGISTRY.md`.
