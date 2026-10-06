# ML-Modell-Domänen-Registry (normativ) — Aurik 10

> **Status: Aktiv — normative Referenz.** Kanonische Quelle für die
> **Trainings-Domäne** jedes lokal verfügbaren ML-Modells unter `models/`.
> Abgeleitet aus Trainings-Skripten (`scripts/train_*.py`), vendored
> Modell-Karten (`models/*/README.md`, `MODEL_CARD.md`), Export-Skripten
> (`scripts/export_*.py`) und SHA-256-Vergleichen — **nicht** aus Vermutungen
> über Datei- oder Ordnernamen.
> Einordnung in die normative Kette: `AGENTS.md` §1 (Punkt 4, Domain-Regeln).
> Maschinelle Re-Ableitung: `scripts/model_inventory.py --domains`.
> Zugehörige, automatisch geladene Instruktion:
> `.github/instructions/ml_domain.instructions.md`.

Arbeitssprache: Deutsch. Jedes §-Zitat mit Quelle.

---

## 0. ⛔ Häufigster KI-Fehlschluss — zuerst lesen

**Behauptung (FALSCH):** „Ein Großteil der lokal verfügbaren ML-Modelle sind
Sprachmodelle."

**Fakt (belegt am Ist-Bestand 2026-10-06):**

| Domäne | Anzahl Verzeichnisse | Anteil |
| --- | --- | --- |
| **Musik-trainiert** | **37** | **≈ 61 %** |
| Gemischt (Sprach-Basis + Musik-Variante) | 4 | ≈ 7 % |
| Sprach-trainiert | 11 | ≈ 18 % |
| Audio-allgemein (AudioSet / LAION-Audio / Codec) | 8 | ≈ 13 % |
| Unüberwacht (kein Training) | 1 | ≈ 2 % |
| **Summe** | **61** | 100 % |

**Merksatz:** Aurik ist ein **Musik-Restaurierungssystem**. Der überwiegende
Teil der lokalen ML-Modelle ist **musik-trainiert**. Wer die Modelle nach ihren
Ordnernamen („whisper", „silero", „utmos") sortiert, übersieht, dass die
tatsächlich im Signalpfad liegenden Kerne (Separation, Pitch, Gesang, Inpainting,
Deklipper) **Musik-Modelle** sind.

**Zusatz-Beleg — kuratiertes Produktions-Manifest** (`scripts/model_inventory.py
--domains`): Von den **30** kuratierten Produktionsmodellen sind **20 musik**
(≈ 67 %), **3 sprach** und **7 audio-allgemein**. Auch im aktiven Pfad dominiert
Musik — nicht Sprache.

### Warum die Verwechslung entsteht

1. **Sprach-Basis + lokaler Musik-Finetune**: Ein Upstream-Modell heißt nach
   seiner Sprach-Herkunft (z. B. DeepFilterNet, SGMSE+, BigVGAN), das
   **produktiv geladene Artefakt** ist aber der lokale Musik-Finetune.
2. **Generische Namen**: `models/versa/` enthält ein Evaluations-Toolkit, der
   Aurik-Plugin `versa_plugin.py` nutzt aber **SingMOS Pro** (Gesang).
3. **Rollen-Verwechslung**: Whisper/UTMOS/Silero/Resemblyzer sind **Zeugen**
   (Wortgrenzen, MOS-Proxy, VAD, Sprecher-Ähnlichkeit) — sie sind **nicht** der
   Signalpfad. Der Signalpfad ist musik-trainiert
   (§III.11 (copilot-instructions.md)).

---

## 1. Musik-trainiert (37) — der Signalpfad-Kern

| Modell (`models/…`) | Beleg (Evidenz) | Produktion |
| --- | --- | --- |
| `apollo` | Modell-Karte: „trained on the **MUSDB18-HQ and MoisesDB** datasets" | Phase-0 (Carrier-Chain) |
| `applade` | `port/` == `models/aspade/aspade_declipper.onnx` (SHA `8161ed93…`) → Musik-Deklipper | via A-SPADE |
| `aspade` | A-SPADE Deklipper, Musik-Restaurierung | phase_07 |
| `banquet` | `banquet_vinyl_final.onnx`; `train_banquet_vinyl_finetune.py` (Vinyl-Korpus) | BANQUET-Vinyl |
| `basicpitch` | spotify/basic-pitch (Multi-Pitch, Musik) | Pitch-Witness |
| `bs_roformer` | BS-RoFormer-Separation, MUSDB-Training (SDR 12,98) | Torch-ROCm-Kern |
| `bw_reconstructor` | `train_bw_reconstructor.py` → **MUSDB18-HQ** | BW-Extender |
| `cantus` | Modell-Karte: MUSDB, Musik; Gesangsrestaurierung (MERT-conditioned DiT) | Gesangsrestaurierung |
| `cqtdiff` | Modell-Karte: **MAESTRO** (Musik) | Lücken-Inpainting |
| `crackle_texture_removal` | `phase_67` — Bailey et al., **Vinyl-Knistern** | phase_67 |
| `crepe` | Modell-Karte: **MedleyDB**, Musik-Pitch | Pitch-Witness |
| `ddsp_predictor` | `train_ddsp_predictor_c4.py` → **MUSDB18-HQ** | nicht verdrahtet (val_MAE ohne Skill) |
| `demucs` | `htdemucs_6s`, **MUSDB**-Training (Stem-Separation) | Stem-Separation |
| `ear_vae` | Upstream: „44.1 kHz **music** signal reconstruction model" (phase-aware, Stereo) | Phase-0 Clean-Pass |
| `era_classifier` | `era_anchors.npy` — Musik-Ära-Anker | Ära-Klassifikation |
| `fcpe` | `fcpe.pt` aus CNChTu/Diffusion-SVC — Musik/Vokal-Pitch | Pitch (Wow/Flutter) |
| `flashsr` | `train_flashsr_f4.py` → **MUSDB18-HQ** (F4) | BW-Extension (F4) |
| `forensics` | `clap_material_head.npz` — Material-Kopf auf CLAP, Musik-Träger | Material-Forensik |
| `gacela` | Modell-Karte: **MAESTRO**, Musik — Gap-Inpainting | Gap-Inpainting |
| `gacela_upstream` | Upstream-Architekturkopie (Musik) | kein Laufzeitpfad |
| `harmonic_inpainting` | `train_harmonic_inpainting.py` → **MUSDB18-HQ**+corpus | Harmonik-Inpainting |
| `kim_inst` | KimberleyJSN/KIM-INST — Instrumental-Enhancer (Musik) | KIM-Inst |
| `kim_vocal_2` | KimberleyJSN/Kim-Vocal-2 — Gesangs-Enhancer (Musik) | KIM-Vocal |
| `matchering2.0` | Referenz-Matching-Werkzeug (Musik-Mastering) | Referenz-Vergleich |
| `melbandroformer` | Mel-Band-RoFormer, **MUSDB**-Musik-Separation | Separation |
| `mert` | MERT-v1 (auf **160 k+ Stunden Musik** vortrainiert); `clean_music_centroid.npy` | Qualitäts-Proxy/MUSHRA |
| `mert-v1-330m` | HF: m-a-p/MERT-v1-330M (Musik) | Alternativ-Pfad |
| `mert_denoiser` | MERT-basierter Musik-Denoiser | Denoise |
| `miipher_dit` | `train_miipher_dit.py` → **MUSDB18-HQ vocals** + musan; Plugin: „Flow-Matching **Singing Voice** Enhancement" | **AKTIV** (`use_miipher_dit=True`) |
| `mp_senet` | `train_mp_senet_musik.py` (Musik-Finetune) | Musik-Denoise |
| `muq_eval` | Modell-Karte: Musik; MuQ-Eval-A1 MOS-Head | MOS-Witness |
| `muq_mulan` | TencentARC/MuQ-MuLan (Musik-Embedding) | Harmonie-Kontext |
| `rmvpe` | RMVPE Vokal-Pitch (Musik/MIR) | Pitch Stufe 2 |
| `scnet_4stems` | SCNet Stem-Separation (Musik); A/B-gewonnen 2026-10-05 | Verdrahtung offen |
| `singmos` | South-TP-AI-Lab/SingMOS-Pro — **Gesangs**-MOS | MOS-Witness Gesang |
| `symphonia` | Instrumentalrestaurierung aus MERT + MuQ-MuLan + Flow-DiT (Musik) | Instrumental-Restaurierung |
| `uvr_mdx_net` | UVR MDX-Net (Musik-Separation) | Separation |

---

## 2. Gemischt — Sprach-Basis mit Musik-Variante (4)

> **Warnung:** Bei diesen Modellen ist die **Basis** sprach-trainiert, es
> existiert aber eine (lokale) **Musik-Variante**. Die Domänen-Regel
> §III.11 (copilot-instructions.md) verlangt, den **Musik-Pfad** zu wählen —
> ein sprach-trainierter Kern darf als Signalpfad nur mit belegter
> Musik-Variante laufen.

| Modell | Sprach-Basis (Upstream) | Musik-Variante | Deployt? |
| --- | --- | --- | --- |
| `bigvgan` | BigVGAN v2 (LibriTTS, Sprache) | `train_bigvgan_f3.py` → **MUSDB18-HQ** (`output/_training_archive_20260920/f3_bigvgan/best.pt`), A/B bestanden (HNR **+4,42 dB**) | ❌ Flag `BIGVGAN_V2_HR_ACTIVATED` OFF, kein `bigvgan_v2_f3e29.onnx` |
| `deepfilternet_v3_ii` | DFN-3 (Sprach-Denoise) | `finetuned/` (`train_df_musik.py` → `dfn_musik_best.pt`) | ✅ `finetuned/` geladen |
| `sgmse_plus` | SGMSE+ (WSJ0/CHiME3, Sprache) | `finetuned/` + `sgmse_musik_core.onnx` (`train_sgmse_musik.py`) | ⚠️ Default AUS (`use_sgmse_musik=False`) |
| `diffwave` | DiffWave (Sprach-Vocoder) | `train_diffwave_vocal_inpaint.py` → Gesangs-Inpaint-Finetune (`diffwave_vocal_ft.ckpt`) | nur Vocal-Inpaint-Pfad |

**EAR-VAE-Zusatzbefund:** `models/ear_vae/` ist die **offizielle Musik-Basis**
(SHA `8f97ef67…`/`ed64a1f1…`). Ein interner **MUSDB-Finetune** existiert im Backup
(`ear_vae_ft_*.onnx`, SHA `4168db14…`/`48dad212…`), ist aber **nicht** deployt —
kein Defekt (die Basis ist bereits Musik), aber ungenutztes Potenzial.

---

## 3. Sprach-trainiert (11) — **Zeugen, nicht Signalpfad**

> Diese Modelle sind in Aurik **ausschließlich** als **Zeugen** (Metriken,
> Grenzen, VAD, Embeddings) oder als **ausdrücklich als Sprache markierter**
> Pfad zulässig. Ein Einsatz als Musik-Signalpfad ist nur nach §III.11
> (copilot-instructions.md) (belegter Musik-Finetune + A/B) erlaubt.

| Modell | Upstream-Korpus | Rolle in Aurik | Risiko |
| --- | --- | --- | --- |
| `whisper` | Whisper-Tiny (Sprache) | Wortgrenzen (Lyrics-Timeline) — Nebenrolle | niedrig (Witness) |
| `wav2vec2` | Wav2Vec2-**CTC** (Sprache) | Phonem-Erkennung (§III.10) | Witness; `quality_predict.py` ist inkonsistent (Classification statt CTC) |
| `wav2vec2-base` | Wav2Vec2 (Sprache) | Offline-Fallback für UTMOS | niedrig |
| `silero` | Silero VAD **English v5** (Sprache) | Sprach-/Gesangspräsenz | VAD auf Musik unzuverlässig |
| `utmosv2` | **VoiceMOS** Challenge (Sprache) | MOS-Proxy (Plugin „Musik-orientiert") | Modell ist sprach-trainiert; `use_utmos_music=False` |
| `resemblyzer` | GE2E/LibriSpeech (Sprache) | Sprecher-Identität | `use_resemblyzer_music=False` |
| `nvsr` | Neuronales SR (Sprache) | FlashSR-Fallback (8–16 kHz) | **Signalpfad-Risiko**: Sprache auf Musik |
| `aero` | slp-rl/aero SR 12→48 kHz (Sprache) | BWE-Challenger (nicht verdrahtet) | würden Musik-F4 verdrängen |
| `vocos` | Vocos (LibriTTS, Sprache) | Vocoder-Stufe 24 kHz | Signalpfad-Risiko |
| `vocos_48khz` | `kittn/vocos-mel-48khz-alpha1` — **Trainings-Domain undokumentiert** (Karte: „Training details: TODO") | Vocoder-Stufe 48 kHz (primär) | **Unbekannte Domäne — prüfen** |
| `versa` | wavlab-speech VERSA (Sprach-Eval-Toolkit) | Verzeichnis = Toolkit; der gleichnamige Aurik-Plugin nutzt **SingMOS Pro** (Gesang) | Namens-Verwechslung (§0) |

---

## 4. Audio-allgemein (8) — weder rein Musik noch rein Sprache

| Modell | Korpus | Anmerkung |
| --- | --- | --- |
| `ast` | AudioSet | 527 Klassen (Musik-Klassen enthalten) |
| `ast_perceptual_base` | AudioSet (fine-tuned) | Genre/Scene |
| `audioldm2` | Text-to-Audio (allgemein) | Generativ |
| `beats` | AudioSet | **Encoder-only** (`[B,T,768]`), kein 527-Tagger |
| `clap` | LAION-Audio-630k (Musik+Sprache+Allgemein) | Tagging; **nie alleiniger Entscheider** (§6.8) |
| `dac` | `dac_44khz` (44,1 kHz = CD-/Musik-Rate) | Musik-inklusive; Inpainting-Conditioning |
| `hifi_gan` | HiFi-GAN **UNIVERSAL_V1** (sprach-trainierter Multi-Speaker-Vocoder) | Vocoder-Fallback, `sota_speech_superres.py` — **Signalpfad-Risiko** auf Musik (§V6) |
| `panns` | AudioSet (PANNs CNN14) | Liefert die 527 Tags für BEATs-Ersatz |

---

## 5. Unüberwacht (1) — kein Training, domain-neutral

| Modell | Anmerkung |
| --- | --- |
| `nara_wpe` | WPE-Dereverberation ist **unüberwacht** (kein Trainingskorpus) → für Musik **und** Sprache gleichermaßen gültig. **Kein** Domänen-Problem. |

---

## 6. Verbotene Fehlschlüsse (quotierbar)

1. **„Die meisten Modelle sind Sprache."** → FALSCH. 37/61 sind Musik (Tabelle §0).
2. **„Ein Modell heißt wie ein Sprachmodell, also ist es Sprache."** → FALSCH.
   Prüfe die **lokale** Evidenz (Trainings-Skript, `finetuned/`, SHA-Vergleich),
   nicht den Upstream-Namen (`miipher_dit`, `ear_vae`, `deepfilternet`,
   `sgmse_plus` sind bzw. haben Musik-Varianten).
3. **„Der Sprach-Kern ist der Signalpfad."** → VERBOTEN, wenn eine Musik-Variante
   existiert (§III.11 (copilot-instructions.md)).
4. **„Ein Sprach-Modell als Vocoder/SR auf Musik ist harmlos."** → FALSCH.
   Sprach-trainierte Vocoder/SR (HiFi-GAN, Vocos, NVSR, AERO) sind ein
   Signalpfad-Risiko für Musik; sie sind nur als Fallback mit
   `logger.warning` (§V6 (copilot-instructions.md)) zulässig.
5. **„Ein Witness verändert das Signal."** → FALSCH. Whisper/UTMOS/Silero/
   Resemblyzer sind Zeugen und dürfen das Ausgangssignal nicht verändern.
6. **„`models/versa` = VERSA = Sprache, also Musik ungeeignet."** → FALSCH für
   den Signalpfad: der Aurik-Plugin nutzt SingMOS (Gesang/Musik).

---

## 7. Pflege und Verifikation

- **Re-Ableitung:** `python scripts/model_inventory.py --domains` gibt die
  maschinenlesbare Domänen-Tabelle aus (JSON mit `domain` + `evidence`).
- **Evidenz-Regel:** Ein Domänen-Eintrag ist nur gültig mit **einer** der
  folgenden Quellen: (a) Trainings-Skript-Korpuspfad, (b) vendored Modell-Karte,
  (c) SHA-256-Identität mit einem dokumentierten Artefakt, (d) Plugin-Header
  ohne Widerspruch zur Karte. Sonst gilt die Domäne als **unbekannt** — nicht
  als „Sprache".
- **Änderung der Domäne** eines produktiven Kerns ⇒ A/B-Nachweis + Hör-Sign-off
  (§v10.802, §III.11 (copilot-instructions.md)).
- **Pflege-Pflicht:** Wird ein neues Modellverzeichnis unter `models/` angelegt
  oder ein Finetune produktiv geschaltet, MUSS diese Registry im selben
  Commit aktualisiert werden (§G9 (copilot-instructions.md), Konsistenz).

---

## 8. Referenzen

- Domänen-Gap-Regel: §III.11 (copilot-instructions.md)
- ML-GPU-Numerik/Parität: §III.9 (copilot-instructions.md)
- Kein stiller ML→DSP-Fallback: §V6 (copilot-instructions.md)
- Physical-over-statistical (CLAP nie alleiniger Entscheider):
  §6.8 (CLAUDE.md)
- Instruktion (auto-geladen): `.github/instructions/ml_domain.instructions.md`
- Maschinen-Ableitung: `scripts/model_inventory.py --domains`
- Verdrahtungs-Audit: `scripts/model_inventory.py --wiring`

_Letzte Verifikation: 2026-10-06 — Bestand 61 Modellverzeichnisse._
