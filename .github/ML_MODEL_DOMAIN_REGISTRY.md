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
--domains`, Stand 2026-10-06): Von den **32** kuratierten Produktionsmodellen
sind **22 musik** (≈ 69 %), **3 sprach**, **5 audio-allgemein** und
**2 unbekannt** (`models/bigvgan/bigvgan_v2.onnx`,
`models/hifi_gan/hifi_gan.onnx` — Sprach-Zuordnung am Artefakt widerlegt,
Mess-Block §4a). Auch im aktiven Pfad dominiert Musik — nicht Sprache.

### Warum die Verwechslung entsteht

1. **Sprach-Basis + lokaler Musik-Finetune**: Ein Upstream-Modell heißt nach
   seiner Sprach-Herkunft (z. B. DeepFilterNet, SGMSE+), das
   **produktiv geladene Artefakt** ist aber der lokale Musik-Finetune.
   **Achtung Gegenrichtung (2026-10-06):** Ein Name wie „BigVGAN-v2" oder
   „UNIVERSAL_V1" belegt **keine** Sprach-Herkunft — bei `bigvgan` und
   `hifi_gan` ist die Doku-Zuordnung „Sprache" am Artefakt **widerlegt**
   (s. Mess-Block §4a), die Domäne gilt dort als **unbekannt**.
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
| `bigvgan` | BigVGAN v2 in der **44,1-kHz-Konfiguration** — **lokal am Artefakt gemessen** (128 Mel-Bänder, 512×, 1536 Kanäle, SnakeBeta + Anti-Alias, 122,19 M Parameter; s. Mess-Block §4a). Der **Upstream-Korpus ist nicht belegbar** ⇒ §III.13 (copilot-instructions.md): `unbekannt`. Die frühere Angabe „LibriTTS (Sprache)" ist architektur-**widerlegt** (LibriTTS-Konfiguration = 24 kHz/100 Bänder/256×/512 Kanäle/≈14 M) | `train_bigvgan_f3.py` → **MUSDB18-HQ** (`output/_training_archive_20260920/f3_bigvgan/best.pt`), A/B bestanden (HNR **+4,42 dB**) | ⚠️ Flag `BIGVGAN_V2_HR_ACTIVATED = True` (§P1-2, `plugins/bigvgan_v2_plugin.py`) — der aktive Pfad nutzt aber den **Basis**-ONNX; der **F3-Finetune ist nicht deployt** (469 von 783 Tensoren Basis≠F3, deployter ONNX == Basis mit max\|Δ\|=0), kein `bigvgan_v2_f3e29.onnx` |
| `deepfilternet_v3_ii` | DFN-3 (Sprach-Denoise) | `finetuned/` (`train_df_musik.py` → `dfn_musik_best.pt`) | ✅ `finetuned/` geladen |
| `sgmse_plus` | SGMSE+ (WSJ0/CHiME3, Sprache) | `finetuned/` + `sgmse_musik_core.onnx` (`train_sgmse_musik.py`) | ⚠️ Default AUS (`use_sgmse_musik=False`) |
| `diffwave` | DiffWave (Sprach-Vocoder) | `train_diffwave_vocal_inpaint.py` → Gesangs-Inpaint-Finetune (`diffwave_vocal_ft.ckpt`) | nur Vocal-Inpaint-Pfad |

**EAR-VAE-KORREKTUR (2026-10-06, SHA-belegt):** Die frühere Fassung dieses
Absatzes war **falsch** — sie hielt die deployten Dateien für die Upstream-Basis
und den Finetune für „nicht deployt". Gemessen (SHA-256, byte-identisch):

| deployt in `models/ear_vae/` | SHA-256 | identisch mit (Backup `ear_vae_upstream/`) |
| --- | --- | --- |
| `encoder.onnx` | `8f97ef67…c387` | `ear_vae_ft_encoder_inline.onnx` |
| `decoder.onnx` | `ed64a1f1…11ab` | `ear_vae_ft_decoder_inline.onnx` |

Der Produktionsstand ist also der **MUSDB-Musik-Finetune** („inline"-Export mit
eingebetteten Gewichten). Die zuvor genannten SHAs `4168db14…`/`48dad212…` sind
derselbe Finetune in der **External-Data-Variante** (`ear_vae_ft_encoder.onnx` +
`ear_vae_ft_encoder.onnx.data`, `ear_vae_ft_decoder.onnx` + `.onnx.data`) — sie
liegt seit 2026-10-06 **ebenfalls in `models/ear_vae/`**, ist aber **nicht** der
Produktionspfad (die Begleit-`.data`-Dateien müssen mitgeliefert werden; der
Inline-Export benötigt nur je eine Datei). Numerische Parität beider Varianten
gegen die deployte Paarung: **max|Δ| = 0,0**. Belege:
`finetune_music.py` (Rezept v2, validierungsgesteuert), `masking_loss.py`
(A1-Loss), `export_finetuned_onnx.py` (Export-Paar), `benchmark_finetuned.py`
sowie der Checkpoint `ear_vae_music_finetuned.pyt` (SHA `993e69be…`), seit
2026-10-06 **in-repo** unter `models/ear_vae/`. Reproduzierbarkeit geprüft:
Re-Export aus diesem Checkpoint ergibt **max|Δ| = 0,0** gegen die deployten
ONNX-Artefakte (Encoder, Decoder, Ende-zu-Ende) — deckungsgleich mit
`docs/PHASE_SOTA_GAP_ANALYSE.md` §1.2 (ΔSDR +4,83 dB) und
`plugins/ear_vae_denoiser.py`.

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
| `hifi_gan` | HiFi-GAN-**Topologie** lokal gemessen (80 Mel-Bänder, [8,8,2,2]/[16,16,4,4] = 256×, Kanäle 128/64 ⇒ 0,93 M Parameter, 22 050 Hz; s. Mess-Block §4a); **kein offizieller Checkpoint** (V1: 512 Kanäle/13,9 M; V2: [4,4,2,2]/[8,8,4,4]) ⇒ §III.13 (copilot-instructions.md): `unbekannt`; die Zuordnung „UNIVERSAL_V1" ist architektur-**widerlegt** | Vocoder-Fallback, `sota_speech_superres.py` — Signalpfad mit **unbelegter** Domäne (Risiko bleibt; §V6 (copilot-instructions.md)-Fallback + §III.13-Beleg offen) |
| `panns` | AudioSet (PANNs CNN14) | Liefert die 527 Tags für BEATs-Ersatz |

---

## 4a. Artefakt-Messung 2026-10-06 — Vocoder `bigvgan` / `hifi_gan`

> **Warum dieser Block:** Die Sprach-Zuordnung dieser beiden Vocoder stammte
> ausschließlich aus Doku-Übernahme (Registry-Selbstbezug + Skript-Kopf),
> nicht aus einem Artefakt-Beleg. Nachgemessen **am Artefakt** (§III.13
> (copilot-instructions.md) — kein Domänen-Urteil aus Namen oder Doku):

| Messgröße | `models/bigvgan/bigvgan_v2.onnx` (+ `.pth`) | `models/hifi_gan/hifi_gan.onnx` |
| --- | --- | --- |
| Mel-Bänder | **128** (`conv_pre.weight_v` = `[1536, 128, 7]`; ONNX-Input `mel [1, 128, 64]`) | **80** (`conv_pre.weight` = `[128, 80, 7]`) |
| Upsample | 6 Stufen `ups.0…5` ⇒ **512×** (ONNX: 32 768 Samples aus 64 Frames) | `[8, 8, 2, 2]` ⇒ **256×** |
| Kanäle | `upsample_initial_channel` **1536** | **128** (ResBlocks 64) |
| Aktivierung | **SnakeBeta** (109×`alpha` **und** 109×`beta`) + Anti-Alias (`activation_post.downsample.lowpass.filter`) | ReLU/LeakyReLU |
| Parameter | **122,19 M** (Checkpoint) / 122,81 M (ONNX) | **0,93 M** |
| Sample-Rate | 44 100 Hz (Konfigurationsfamilie; `plugins/bigvgan_v2_plugin.py`) | 22 050 Hz (`models/hifi_gan/hifigan_infer.py`) |
| Identität (**belegt**) | **BigVGAN v2, 44,1-kHz-Konfiguration** (`bigvgan_v2_44khz_128band_512x`) | **HiFi-GAN-Topologie** (V1-Zeitplan in ~¼ Kanalbreite) |
| **widerlegt** | „LibriTTS/Sprache" — LibriTTS-Konfiguration = 24 kHz/100 Bänder/256×/512 Kanäle/≈14 M | „UNIVERSAL_V1" — offizieller V1 = 512 Kanäle/13,9 M; offizieller V2 = `[4,4,2,2]`/`[8,8,4,4]` |
| **nicht belegbar** | Trainingskorpus (keine Modell-Karte; Checkpoint = nur `{"generator": …}` ohne Hparams; kein Upstream-SHA) | Trainingskorpus (keine Karte, keine Hparams, kein Upstream-SHA) |
| **Domänen-Urteil (§III.13)** | **`unbekannt`** | **`unbekannt`** |

**Deployter Gewichtssatz (bewiesen):** Das produktiv geladene
`models/bigvgan/bigvgan_v2.onnx` enthält die **Basis**-Gewichte, nicht den
F3-Musik-Finetune: 469 von 783 Tensoren unterscheiden sich zwischen
`models/bigvgan/bigvgan_v2.pth` und
`output/_training_archive_20260920/f3_bigvgan/best.pt`, und das ONNX stimmt
exakt mit der Basis (`ups.1.0.weight_v`: max|Δ| = 0 gegen Basis, 1,0e-1 gegen
F3). Das ONNX datiert auf **2026-09-10**, der Finetune auf **2026-10-03** — der
Export kann ihn nicht enthalten. Damit bleibt Defizit D-K0-2 (Basis statt
Musik-Finetune) bestehen, aber mit korrigierter Begründung: die Basis-Domäne ist
**unbelegt**, nicht „sprach-trainiert".

**Grenze der Aussage (§G8 (copilot-instructions.md)):** Aus Bytes ist die
**Architektur-Identität** beweisbar, **nicht** der **Trainingskorpus** — eine
ONNX-Datei trägt kein Trainingsmaterial. Ein Korpus-Beleg entsteht erst mit
SHA-Identität gegen das offizielle Upstream-Artefakt (offen, s.
`.github/SOTA_DEFICIT_REGISTER.md`).

---

## 4b. Provenienz-Belege (Archiv `models/_archive_20260920` ↔ deployt)

> **Quelle:** `scripts/model_artifact_probe.py` →
> `.github/ML_ARTIFACT_FINGERPRINTS.md` (+ generierter JSON-Report). Gemessen
> werden **SHA-256-Gleichheit** („byte-identisch“) und **Parameterzahl-Gleichheit
> bei abweichendem SHA** („architektur-gleich ⇒ Finetune/Generation“). Das ist
> die maschinelle Form des EAR-VAE-Nachweises — für **alle** 26 Artefakte des
> Archivs, nicht mehr nur für ein Einzelmodell.

**Byte-identisch (8) — gleiche Gewichte, nur anderer Pfad:**

| Archiv | deployt |
| --- | --- |
| `bs_roformer/bsr317_core_reexport.onnx` | `bs_roformer/bs_roformer_317_core.onnx` |
| `ear_vae_upstream/…/ear_vae_ft_{encoder,decoder}_inline.onnx` | `ear_vae/{encoder,decoder}.onnx` |
| `ear_vae_upstream/…/ear_vae_ft_{encoder,decoder}.onnx` | `ear_vae/ear_vae_ft_{encoder,decoder}.onnx` |
| `ear_vae_upstream/…/ear_vae_music_finetuned.pyt` | `ear_vae/ear_vae_music_finetuned.pyt` |
| **`flashsr/flashsr_onnx_prod_20260810.onnx`** | **`nvsr/nvsr.onnx`** |
| `harmonic_inpainting/inpainting_best.pt` | `harmonic_inpainting/inpainting_best.pt` |

**Architektur-gleich, Gewichte verschieden (6) — Finetune-/Generationsbelege:**

| Archiv-Artefakt | deploytes Gegenstück | Bedeutung |
| --- | --- | --- |
| **`ear_vae_upstream/…/pretrained_weight/ear_vae_44k.pyt`** (147,86 M) | `ear_vae/ear_vae_music_finetuned.pyt` | **Upstream-Basis liegt jetzt in-repo** ⇒ „offizielles Musik-Modell + lokaler MUSDB-Finetune“ ist lokal belegt (bis v10.3.23 nur über Skripte) |
| **`sgmse_plus/sgmse_wsj0_reverb.ckpt`** (327,95 M) | `sgmse_plus/sgmse_plus_src_1.ckpt` | **Sprach-Basis-Core in-repo** ⇒ §III.11-Sperre (`use_sgmse_musik=False`) ist gegen die echte Basis prüfbar |
| `bs_roformer/bsr317_core_fp16.onnx` (159,76 M) | `bs_roformer/bs_roformer_317_core.onnx` | fp16-Variante desselben Kerns |
| `diffwave/diffwave_vocal_ft_rejected_epoch{2,7}*.ckpt` (2,62 M) | `diffwave/diffwave.ckpt` | **verworfene** Finetunes (gatefail/aborted) — Prüf-Historie |
| `miipher_dit/whisper_denoiser_latest.pt` (3,38 M) | `miipher_dit/whisper_denoiser_best.pt` | Zwischenstand vs. Best-Checkpoint |

**Ohne Gegenstück im deployten Bestand (14) — Herkunft noch zuzuordnen:**
`banquet/banquet_vinyl_batch.onnx` (23,69 M) ·
`bw_reconstructor/best_model{,_v3,_v4,_v5_pre_retrain}.pt` (je ≈1,0 M) ·
`clean_music_centroid.npy` · `ear_vae/{encoder,decoder}_orig_20260730_backup.onnx`
(84,1/84,3 M — **andere Parameterzahl** als der deployte Finetune mit 73,8 M:
Zuordnung offen, keine Gleichsetzung behauptet) ·
`harmonic_inpainting/inpainting_mask_best.pt` (201,89 M) ·
`medium_classifier/medium_shallow_v1.joblib` ·
`melbandroformer/melbandroformer_optimized.onnx.rocm_safe*.onnx` (225,13 M) ·
`singmos/singmos_pro.pt` · `gacela_venv/…/distutils-precedence.pth`
(vom Prober korrekt als _kein_ Modellartefakt klassifiziert).

**Wirkung:** Die beim Aufbau dieser Registry oft fehlende Evidenzklasse
„SHA-Identität/Nachbarartefakt“ liegt jetzt **lokal** vor — für 14 von 26
Archivartefakten ist die Basis↔Finetune-Relation bewiesen, die Restmenge ist
ehrlich als offen dokumentiert (keine Vermutung, §V7 (copilot-instructions.md)).

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
   Sprach-trainierte Vocoder/SR (Vocos, NVSR, AERO) sind ein
   Signalpfad-Risiko für Musik; sie sind nur als Fallback mit
   `logger.warning` (§V6 (copilot-instructions.md)) zulässig. Bei
   `bigvgan`/`hifi_gan` ist die Sprach-Herkunft sogar **widerlegt bzw.
   unbelegt** (Mess-Block §4a) — die Kandidaten bleiben deshalb als
   **unbelegte** Signalpfade im Defizit-Register
   (`.github/SOTA_DEFICIT_REGISTER.md`, D-K0-2, D-K2-1), nicht als
   „sprach-trainiert".
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
