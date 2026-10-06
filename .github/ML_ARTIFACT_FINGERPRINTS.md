# ML-Artefakt-Fingerabdrücke (`models/`) — am Artefakt gemessen

> **Erzeugt von** `scripts/model_artifact_probe.py` (§III.13 (copilot-instructions.md) — Evidenzpflicht).
> Die Tabelle nennt **gemessene** Architektur-Fingerabdrücke (Mel-Bänder, Upsample-Faktor, Kanalbreiten, Parameterzahl, Hparams, Dtype-Mix) — **keine** Domänen-Behauptung. Ob ein Artefakt Musik- oder sprach-trainiert ist, entscheidet sich an Trainings-Skript, Modell-Karte, SHA-Identität oder einem widerspruchsfreien Plugin-Header; fehlt das, gilt die Domäne als `unbekannt`.

**Bestand:** 66 Verzeichnisse · 153 Artefakte · 41.1 GB.

## Verzeichnis-Übersicht

| Verzeichnis | Artefakte | Größe | Lokale Doku-Evidenz |
| --- | --- | --- | --- |
| `_archive_20260920` | 28 | 8.9 GB | — |
| `aero` | 1 | 77.6 MB | — |
| `apollo` | 2 | 129.0 MB | LICENSE, README.md |
| `applade` | 3 | 9.8 MB | LICENSE.pdf, README.md |
| `aspade` | 1 | 3.3 MB | — |
| `ast` | 1 | 287.5 KB | — |
| `ast_perceptual_base` | 0 | 0 B | README.md; config.json, preprocessor_config.json |
| `audioldm2` | 2 | 1.4 GB | — |
| `banquet` | 1 | 91.8 MB | — |
| `basicpitch` | 2 | 450.1 KB | — |
| `beats` | 1 | 344.8 MB | — |
| `bigvgan` | 2 | 469.1 MB | — |
| `bs_roformer` | 2 | 1.2 GB | — |
| `bw_reconstructor` | 3 | 33.2 MB | — |
| `cantus` | 3 | 788.8 MB | README.md; cantus_config.json |
| `clap` | 5 | 2.5 GB | LICENSE, README.md |
| `cqtdiff` | 3 | 182.8 MB | LICENSE, README.md |
| `crackle_texture_removal` | 2 | 2.3 MB | — |
| `crepe` | 1 | 84.9 MB | LICENSE, README.md |
| `dac` | 2 | 293.4 MB | — |
| `ddsp_predictor` | 1 | 647.1 KB | — |
| `deepfilternet_v3_ii` | 8 | 67.5 MB | LICENSE, LICENSE-APACHE, LICENSE-MIT |
| `demucs` | 1 | 2.5 MB | — |
| `diffwave` | 2 | 10.6 MB | — |
| `ear_vae` | 5 | 1.1 GB | — |
| `ear_vae2_upstream` | 0 | 0 B | LICENSE, README.md |
| `era_classifier` | 1 | 30.2 KB | — |
| `fcpe` | 2 | 131.8 MB | LICENSE, README.md |
| `flashsr` | 6 | 9.6 MB | README.md |
| `forensics` | 0 | 0 B | — |
| `gacela` | 3 | 440.9 MB | LICENSE, README.md |
| `gacela_upstream` | 0 | 0 B | LICENSE, README.md |
| `harmonic_inpainting` | 3 | 2.3 GB | — |
| `hifi_gan` | 1 | 3.6 MB | — |
| `hubert` | 1 | 1.2 MB | — |
| `kim_inst` | 1 | 63.7 MB | — |
| `kim_vocal_2` | 1 | 63.7 MB | — |
| `matchering2.0` | 0 | 0 B | DOCKER.md, LICENSE, README.md |
| `melbandroformer` | 1 | 859.5 MB | — |
| `mert` | 3 | 677.1 MB | — |
| `mert-v1-330m` | 2 | 4.9 GB | —config.json, preprocessor_config.json |
| `mert_denoiser` | 1 | 13.3 MB | — |
| `miipher_dit` | 2 | 782.1 MB | — |
| `mp_senet` | 3 | 44.1 MB | LICENSE, README.md; config.json |
| `mpsenet` | 0 | 0 B | LICENSE |
| `muq_eval` | 0 | 0 B | LICENSE, README.md |
| `muq_mulan` | 5 | 3.7 GB | —config.json |
| `nara_wpe` | 0 | 0 B | LICENSE, README.rst |
| `nvsr` | 2 | 1.1 MB | — |
| `panns` | 1 | 79.1 KB | — |
| `resemblyzer` | 3 | 38.0 MB | LICENSE, README.md |
| `rmvpe` | 1 | 344.9 MB | — |
| `rvc` | 2 | 353.5 MB | — |
| `scnet_4stems` | 1 | 377.3 MB | README.md |
| `sgmse_plus` | 6 | 2.9 GB | LICENSE, README.md |
| `silero` | 1 | 112.1 MB | — |
| `singmos` | 1 | 2.7 MB | LICENSE, README.md |
| `symphonia` | 1 | 1.3 MB | README.md; symphonia_config.json |
| `utmosv2` | 2 | 1.1 GB | LICENSE, README.md |
| `uvr_mdx_net` | 4 | 247.3 MB | — |
| `versa` | 1 | 1.2 GB | LICENSE, README (1).md, README.md |
| `vocos` | 1 | 51.6 MB | — |
| `vocos_48khz` | 2 | 296.6 MB | README.md |
| `wav2vec2` | 1 | 339.2 MB | README.md; config.json, preprocessor_config.json |
| `wav2vec2-base` | 1 | 360.0 MB | —config.json, preprocessor_config.json |
| `whisper` | 3 | 1.9 GB | —config.json, preprocessor_config.json, tokenizer.json |

## Messungen je Verzeichnis

### `_archive_20260920`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `banquet/banquet_vinyl_batch.onnx` | 91.6 MB | `cf681f0a41741bef…` | 23.69 M Params; IN ['batch', 128, 128, 128]; OUT ['batch', 128, 128, 128]; ops: Slice×384, Concat×148, Transpose×96, Unsqueeze×96 |
| `bs_roformer/bsr317_core_fp16.onnx` | 308.5 MB | `151a64daf84ea269…` | 159.76 M Params; IN ['batch', 2, 1025, 'time', 2]; OUT ['batch', 'Transposemasks_dim_1', 'Transposemasks_dim_2', 'time', 'Transposemasks_dim_4']; ops: Constant×4019, Unsqueeze×1140, Shape×1091, Gather×953 |
| `bs_roformer/bsr317_core_reexport.onnx` | 615.2 MB | `430d1971687f6077…` | 159.76 M Params; IN ['batch', 2, 1025, 'time', 2]; OUT ['batch', 'Transposemasks_dim_1', 'Transposemasks_dim_2', 'time', 'Transposemasks_dim_4']; ops: Constant×4019, Unsqueeze×1140, Shape×1091, Gather×953 |
| `bw_reconstructor/best_model.pt` | 4.2 MB | `96b40a11083c5f54…` | 1.09 M Params; 14 Stubs ohne Formangabe |
| `bw_reconstructor/best_model_v3.pt` | 4.2 MB | `85943655b6d5ada7…` | 1.09 M Params; 14 Stubs ohne Formangabe |
| `bw_reconstructor/best_model_v4.pt` | 4.2 MB | `c44ff60798739f56…` | 1.09 M Params; 14 Stubs ohne Formangabe |
| `bw_reconstructor/best_model_v5_pre_retrain.pt` | 4.0 MB | `9ec868098067db95…` | 1.03 M Params; 8 Stubs ohne Formangabe |
| `clean_music_centroid.npy` | 3.1 KB | `7b098b2e53307c3d…` | npy [768] float32 |
| `diffwave/diffwave_vocal_ft_rejected_epoch2_gatefail.ckpt` | 10.1 MB | `5c81b41467db7878…` | 2.62 M Params |
| `diffwave/diffwave_vocal_ft_rejected_epoch7_aborted.ckpt` | 10.1 MB | `c0292a3e2f06a184…` | 2.62 M Params |
| `ear_vae/decoder_orig_20260730_backup.onnx` | 321.5 MB | `e4c99b994387c87d…` | 84.25 M Params; IN ['batch', 64, 'latent_time']; OUT ['batch', 2, 'audio_time']; ops: Mul×93, Constant×78, Add×72, Exp×42 |
| `ear_vae/encoder_orig_20260730_backup.onnx` | 321.0 MB | `4274e75c478d1f72…` | 84.12 M Params; IN ['batch', 2, 'audio_time']; OUT ['batch', 'Addlatent_dim_1', 'latent_time']; ops: Mul×96, Constant×84, Add×74, Exp×42 |
| `ear_vae_upstream/ear_vae_upstream/ear_vae_ft_decoder.onnx` | 1.1 MB | `48dad212c0e9708c…` | 74.05 M Params; IN ['batch', 64, 'latent_time']; OUT ['batch', 2, '1024*latent_time']; ops: Mul×124, Add×63, Sin×37, ReduceL2×36 |
| `ear_vae_upstream/ear_vae_upstream/ear_vae_ft_decoder_inline.onnx` | 283.6 MB | `ed64a1f12b685bb1…` | 74.05 M Params; IN ['batch', 64, 'latent_time']; OUT ['batch', 2, '1024*latent_time']; ops: Mul×124, Add×63, Sin×37, ReduceL2×36 |
| `ear_vae_upstream/ear_vae_upstream/ear_vae_ft_encoder.onnx` | 893.7 KB | `4168db14dec26806…` | 73.78 M Params; IN ['batch', 2, 'audio_time']; OUT ['batch', 64, '(audio_time//1024)']; ops: Mul×108, Add×51, Conv×37, Sin×36 |
| `ear_vae_upstream/ear_vae_upstream/ear_vae_ft_encoder_inline.onnx` | 282.3 MB | `8f97ef673d4120ed…` | 73.78 M Params; IN ['batch', 2, 'audio_time']; OUT ['batch', 64, '(audio_time//1024)']; ops: Mul×108, Add×51, Conv×37, Sin×36 |
| `ear_vae_upstream/ear_vae_upstream/ear_vae_music_finetuned.pyt` | 564.1 MB | `993e69bea97bd9c9…` | 147.86 M Params |
| `ear_vae_upstream/ear_vae_upstream/pretrained_weight/ear_vae_44k.pyt` | 564.1 MB | `0362dc7e96566869…` | 147.86 M Params |
| `flashsr/flashsr_onnx_prod_20260810.onnx` | 389.3 KB | `a469b590a882bc6c…` | 0.09 M Params; IN ['batch', 1, 'time']; OUT ['batch', 1, 'time']; ops: Constant×69, Mul×39, Add×33, Resize×26 |
| `gacela_venv/lib/python3.10/site-packages/distutils-precedence.pth` | 152 B | `7ea7ffef3fe2a117…` | Textdatei: import os; var = 'SETUPTOOLS_USE_DISTUTILS'; enabled = os.environ.get(var, 'stdl |
| `harmonic_inpainting/inpainting_best.pt` | 768.7 MB | `f8765d14564e65c0…` | 201.50 M Params |
| `harmonic_inpainting/inpainting_mask_best.pt` | 770.2 MB | `c4d89bbe0df00116…` | 201.89 M Params |
| `medium_classifier/medium_shallow_v1.joblib` | 1.7 MB | `bf3f7ab1e9af6154…` | joblib: dict |
| `melbandroformer/melbandroformer_optimized.onnx.rocm_safe.onnx` | 859.5 MB | `6f090d26d1adbcb1…` | 225.13 M Params; IN [1, 'duration', 60, 384]; OUT ['Reshapeoutput_dim_0', 'duration', 'Reshapeoutput_dim_2', 'Reshapeoutput_dim_3', 'Reshapeoutput_dim_4']; ops: Unsqueeze×613, Gather×502, Mul×350, MatMul×264 |
| `melbandroformer/melbandroformer_optimized.onnx.rocm_safe.onnx.probe.onnx` | 859.7 MB | `6e8d64f04a6e02bd…` | 225.13 M Params; IN [1, 'duration', 60, 384]; OUT ['Reshapeoutput_dim_0', 'duration', 'Reshapeoutput_dim_2', 'Reshapeoutput_dim_3', 'Reshapeoutput_dim_4']; ops: Unsqueeze×613, Gather×502, Mul×350, MatMul×264 |
| `miipher_dit/whisper_denoiser_latest.pt` | 12.9 MB | `c66aec0bf57971fe…` | 3.38 M Params |
| `sgmse_plus/sgmse_wsj0_reverb.ckpt` | 1.2 GB | — | 327.95 M Params; 1 Stubs ohne Formangabe; hparams: 31 Felder; [hash_note: übersprungen (> 1024 MB)] |
| `singmos/singmos_pro.pt` | 1.2 GB | — | 1 Stubs ohne Formangabe; [hash_note: übersprungen (> 1024 MB)] |

### `aero`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `aero_12_48.onnx` | 77.6 MB | `3381f1dcd1299272…` | 19.86 M Params; IN [1, 1, 120000]; OUT [1, 1, 480000]; ops: Slice×230, Reshape×125, Mul×95, Gather×86 |

### `apollo`

**Lokale Karten/Lizenzen:**

- `LICENSE` — `(Lizenz)`
- `README.md` — `/ / / / Kai Li 1,2 , Yi Luo 2`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `apollo_core.onnx` | 64.4 MB | `75b4407ae58df555…` | 16.55 M Params; IN [1, 11, 'time'], [1, 11, 'time']; OUT ['Concatri_masks_dim_0', 'Concatri_masks_dim_1', 'time', 'Concatri_masks_dim_3']; ops: Constant×3630, Unsqueeze×1350, Gather×802, Shape×728 |
| `apollo_model.pt` | 64.6 MB | `f7d460d3debc1e2b…` | 1 Stubs ohne Formangabe |

### `applade`

**Lokale Karten/Lizenzen:**

- `LICENSE.pdf` — `(Lizenz)`
- `README.md` — `# APPLADE (Adjustable Plug-and-PLay Audio DEclipper) / **Tomoro Tanaka (Department of Intermedia Art and Science, Waseda University, Tokyo, Japan)**\ / / This README file describes the MATLAB codes pr`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `port/applade_dnn.onnx` | 3.3 MB | `8e6a161741814c4e…` | 0.85 M Params; IN ['batch', 1, 512, 'time']; OUT ['batch', 1, 512, 'time']; ops: Constant×29, ReduceMean×27, Add×19, Sub×19 |
| `port/applade_music_finetuned.onnx` | 3.3 MB | `8161ed93dd28362e…` | 0.86 M Params; IN ['batch', 1, 512, 'time']; OUT ['batch', 1, 512, 'time']; ops: ReduceMean×27, Add×19, Sub×19, Mul×18 |
| `port/applade_music_finetuned.pth` | 3.3 MB | `8352aeda06befe17…` | 0.86 M Params |

### `aspade`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `aspade_declipper.onnx` | 3.3 MB | `8161ed93dd28362e…` | 0.86 M Params; IN ['batch', 1, 512, 'time']; OUT ['batch', 1, 512, 'time']; ops: ReduceMean×27, Add×19, Sub×19, Mul×18 |

### `ast`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `ast_model.onnx` | 287.5 KB | `5b7a7079ed3438ea…` | 86.59 M Params; IN [1, 1024, 128]; OUT [1, 527]; ops: Add×86, MatMul×72, Transpose×38, Gather×38 |

### `ast_perceptual_base`

**Lokale Konfigurations-Evidenz:**

- `config.json`: `architectures`=['ASTForAudioClassification'], `hidden_size`=768, `model_type`=audio-spectrogram-transformer, `num_hidden_layers`=12
- `preprocessor_config.json`: `sampling_rate`=16000

**Lokale Karten/Lizenzen:**

- `README.md` — `--- / license: bsd-3-clause / tags: / - audio-classification / ---`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |

### `audioldm2`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `audioldm2.onnx` | 1.3 GB | — | 346.94 M Params; IN ['batch', 8, 'height', 'width'], [1]; OUT ['batch', 8, 'height', 'width']; ops: Constant×3805, Unsqueeze×1154, Add×921, Mul×863; [hash_note: übersprungen (> 1024 MB)] |
| `vae_decoder.onnx` | 125.9 MB | `3a42aed4dfe14432…` | 32.98 M Params; IN ['batch', 8, 'height', 'width']; OUT ['batch', 1, 'mel_bins', 'time_frames']; ops: Constant×118, Reshape×54, Mul×50, Add×40 |

### `banquet`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `banquet_vinyl_final.onnx` | 91.8 MB | `cf76124d399ee4d4…` | 23.69 M Params; IN [1, 128, 128, 128]; OUT [1, 128, 128, 128]; ops: Slice×432, Concat×144, Transpose×96, Unsqueeze×96 |

### `basicpitch`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `basicpitch.onnx` | 225.0 KB | `2c3c1d144bfa61ad…` | 0.04 M Params; IN ['unk__749', 43844, 1]; OUT ['unk__750', 172, 88]; ops: Reshape×67, Unsqueeze×37, Conv×32, Pad×24 |
| `basicpitch_nmp.onnx` | 225.0 KB | `2c3c1d144bfa61ad…` | 0.04 M Params; IN ['unk__749', 43844, 1]; OUT ['unk__750', 172, 88]; ops: Reshape×67, Unsqueeze×37, Conv×32, Pad×24 |

### `beats`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `beats_iter3.onnx` | 344.8 MB | `82c1b7c0c97422a9…` | 90.31 M Params; IN ['batch', 'frames', 128]; OUT ['batch', 'Transposeoutput_dim_1', 768]; ops: Constant×698, Unsqueeze×215, Mul×172, Add×162 |

### `bigvgan`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `bigvgan_v2.onnx` | 2.7 MB | `43a6bb965dee2151…` | 122.81 M Params; IN [1, 128, 64]; OUT [1, 1, 32768]; ops: Mul×413, Conv×219, Pad×206, Add×175 |
| `bigvgan_v2.pth` | 466.4 MB | `d9fe7ec6bd0b44ed…` | 122.19 M Params |

### `bs_roformer`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `bs_roformer_317_core.onnx` | 615.2 MB | `430d1971687f6077…` | 159.76 M Params; IN ['batch', 2, 1025, 'time', 2]; OUT ['batch', 'Transposemasks_dim_1', 'Transposemasks_dim_2', 'time', 'Transposemasks_dim_4']; ops: Constant×4019, Unsqueeze×1140, Shape×1091, Gather×953 |
| `model_bs_roformer_ep_317_sdr_12.9755.ckpt` | 609.7 MB | `5b84f37e8d444c8c…` | 159.76 M Params |

### `bw_reconstructor`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `best_model_v5.pt` | 15.7 MB | `2a01a6823ad4eff2…` | 4.12 M Params; 8 Stubs ohne Formangabe |
| `bw_reconstructor.onnx` | 1.8 MB | `4a4e4f58b9679abe…` | 0.48 M Params; IN ['batch', 1, 'freq', 'time']; OUT ['batch', 1, 'freq', 'time']; ops: Conv×15, Relu×14, MaxPool×3, ConvTranspose×3 |
| `bw_reconstructor_v5.onnx` | 15.7 MB | `7888e65678234fa3…` | 4.10 M Params; IN ['batch', 1, 'time'], ['batch', 1]; OUT ['batch', 1, 'time']; ops: Constant×22, Conv×14, Mul×8, Split×7 |

### `cantus`

**Lokale Konfigurations-Evidenz:**

- `cantus_config.json`: `training.sr`=48000

**Lokale Karten/Lizenzen:**

- `README.md` — `# models/cantus — Cantus Vocal Restorer / Gesangsrestaurierung (Vocal-Stem-in → Vocal-Stem-out) mit music-aware / MERT-Features, Pitch-/Harmony-Konditionierung und Multi-Scale-Flow-Matching-DiT. / ##`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `cantus_dit.onnx` | 779.7 MB | `ea2556b23dd511bb…` | 204.27 M Params; IN ['batch', 'time', 1], ['batch']; OUT ['Addv_dim_0', 'Addv_dim_1', 1]; ops: Constant×935, Mul×393, Add×333, Unsqueeze×223 |
| `checkpoint_best.pt` | 4.5 MB | `c121f8849cfdc835…` | 1.17 M Params; 67 Stubs ohne Formangabe; hparams: 60 Felder |
| `checkpoint_latest.pt` | 4.5 MB | `3169ecb1cce814a8…` | 1.17 M Params; 67 Stubs ohne Formangabe; hparams: 60 Felder |

### `clap`

**Lokale Karten/Lizenzen:**

- `LICENSE` — `(Lizenz)`
- `README.md` — `# CLAP / / / /`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `audio_encoder.onnx` | 272.0 MB | `68cd50a8d4aa5a13…` | 71.12 M Params; IN ['batch', 'samples']; OUT ['batch', 'Divaudio_embedding_dim_1']; ops: Constant×1352, Unsqueeze×382, Gather×239, Concat×211 |
| `clap_text_encoder.onnx` | 95.2 KB | `23e9db0e502f9faf…` | 125.30 M Params; IN ['batch', 'seq'], ['batch', 'seq']; OUT ['batch', 512]; ops: Add×123, MatMul×96, Reshape×48, Transpose×48 |
| `music_audioset_epoch_15_esc_90.14.pt` | 2.2 GB | — | 587.89 M Params; 466 Stubs ohne Formangabe; [hash_note: übersprungen (> 1024 MB)] |
| `src/laion_clap/training/audioset_textmap.npy` | 82.5 KB | `bada103070d92f9e…` | npy [527] <U40 |
| `text_embeddings.npy` | 86.1 KB | `c1009fff45ec82af…` | npy [43, 512] float32 |

### `cqtdiff`

**Lokale Karten/Lizenzen:**

- `LICENSE` — `(Lizenz)`
- `README.md` — `# CQTDiff: Solving audio inverse problems with a diffusion model / Official repository of the paper: / > E. Moliner,J. Lehtinen and V. Välimäki, "Solving audio inverse problems with a diffusion model"`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `score_network.onnx` | 1.9 MB | `828b8acf610296fd…` | 15.49 M Params; IN [1, 2, 450, 704], [1, 1]; OUT [1, 2, 450, 704]; ops: Mul×254, Add×246, Div×246, Conv×157 |
| `score_network.pt` | 62.7 MB | `7d5c327f85c3ef67…` | 1 Stubs ohne Formangabe |
| `src/models/cqt_weights.pt` | 118.3 MB | `8aaff0349e6a0c6e…` | 30.97 M Params |

### `crackle_texture_removal`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `crackle_texture_removal_base.onnx` | 1.2 MB | `1cd7501d933c7472…` | 0.30 M Params; IN [1, 2, 288000]; OUT [1, 2, 288000]; ops: Conv×9, Add×8, Mul×8, Div×4 |
| `crackle_texture_removal_base.pt` | 1.2 MB | `a5ebc44f1f128707…` | 0.30 M Params |

### `crepe`

**Lokale Karten/Lizenzen:**

- `LICENSE` — `(Lizenz)`
- `README.md` — `CREPE Pitch Tracker / =================== / / /`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `crepe.onnx` | 84.9 MB | `fe3013ca57e83dc8…` | 22.24 M Params; IN ['unk__102', 1024]; OUT ['unk__103', 360]; ops: Conv×6, Relu×6, BatchNormalization×6, MaxPool×6 |

### `dac`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `decoder_model.onnx` | 207.1 MB | `df9d3700d34b9622…` | 54.26 M Params; IN ['batch_size', 9, 'time_steps']; OUT ['ConvTranspose_185_o0__d0', 1, 'ConvTranspose_185_o0__d2']; ops: Mul×58, Add×49, Conv×35, Sin×29 |
| `encoder_model.onnx` | 86.4 MB | `d81082939351ed88…` | 22.59 M Params; IN ['batch_size', 'num_channels', 'sequence_length']; OUT ['batch_size', 9, 'floor(floor(floor(floor(sequence_length/2)/4)/8)/8)']; ops: Add×90, Mul×67, Conv×47, Pow×46 |

### `ddsp_predictor`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `c4_head.pth` | 647.1 KB | `761f5d0e7c004b35…` | 0.16 M Params |

### `deepfilternet_v3_ii`

**Lokale Karten/Lizenzen:**

- `LICENSE` — `(Lizenz)`
- `LICENSE-APACHE` — `(Lizenz)`
- `LICENSE-MIT` — `(Lizenz)`
- `README.md` — `# DeepFilterNet / A Low Complexity Speech Enhancement Framework for Full-Band Audio (48kHz) using on Deep Filtering. /`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `dec.onnx` | 18.4 MB | `fd5d9e15cbbd66f8…` | 4.81 M Params; IN [1, 'S', 512], [1, 64, 'S', 96]; OUT ['Addcoefs_dim_0', 'S', 'Addcoefs_dim_2', 10]; ops: Constant×45, Unsqueeze×9, Concat×9, Reshape×9 |
| `enc.onnx` | 1.8 MB | `a561b2decbdb69e4…` | 0.47 M Params; IN ['batch', 1, 'time', 32], ['batch', 2, 100, 'time']; OUT ['batch', 16, 'time', 'Relue0_dim_3']; ops: Constant×63, Reshape×12, Shape×12, Concat×11 |
| `erb_dec.onnx` | 1.8 MB | `5da71a3234cdfbbc…` | 0.46 M Params; IN ['batch', 'time', 128], ['batch', 16, 'time', 8]; OUT ['batch', 1, 'time', 'Sigmoiderb_mask_dim_3']; ops: Constant×32, Shape×10, Relu×9, Conv×9 |
| `finetuned/dec.onnx` | 5.5 MB | `a58c4b34d0c3ff0c…` | 1.43 M Params; IN [1, 'S', 128], [1, 16, 'S', 96]; OUT [1, 'S', 96, 10]; ops: Constant×40, Shape×7, Unsqueeze×7, Concat×6 |
| `finetuned/dfn_musik_best.pt` | 9.2 MB | `e257bacf5b7107c1…` | 2.40 M Params; 15 Stubs ohne Formangabe |
| `finetuned/dfn_musik_latest.pt` | 27.3 MB | `e5272e2fef219d08…` | 7.12 M Params; 95 Stubs ohne Formangabe |
| `finetuned/enc.onnx` | 1.8 MB | `8c68a24b28802c16…` | 0.47 M Params; IN [1, 1, 'S', 32], [1, 2, 'S', 96]; OUT ['Relue0_dim_0', 16, 'S', 'Relue0_dim_3']; ops: Constant×63, Reshape×12, Shape×12, Concat×11 |
| `finetuned/erb_dec.onnx` | 1.8 MB | `6da28729ac7fc28f…` | 0.46 M Params; IN [1, 'S', 128], [1, 16, 'S', 8]; OUT ['Sigmoidm_dim_0', 1, 'S', 'Sigmoidm_dim_3']; ops: Constant×32, Shape×10, Relu×9, Conv×9 |

### `demucs`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `htdemucs_6s.onnx` | 2.5 MB | `00e07237c92d2010…` | 29.29 M Params; IN [1, 2, 343980], [1, 4, 2048, 336]; OUT [1, 6, 4, 2048, 336]; ops: Mul×314, Reshape×261, Add×231, Transpose×135 |

### `diffwave`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `diffwave.ckpt` | 10.1 MB | `b434ac56f45486ec…` | 2.62 M Params |
| `diffwave_model.onnx` | 550.8 KB | `a07f3ca160873b7d…` | 2.63 M Params; IN [1, 16384], [1]; OUT [1, 1, 16384]; ops: Add×118, Conv×93, Split×60, Unsqueeze×33 |

### `ear_vae`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `decoder.onnx` | 283.6 MB | `ed64a1f12b685bb1…` | 74.05 M Params; IN ['batch', 64, 'latent_time']; OUT ['batch', 2, '1024*latent_time']; ops: Mul×124, Add×63, Sin×37, ReduceL2×36 |
| `ear_vae_ft_decoder.onnx` | 1.1 MB | `48dad212c0e9708c…` | 74.05 M Params; IN ['batch', 64, 'latent_time']; OUT ['batch', 2, '1024*latent_time']; ops: Mul×124, Add×63, Sin×37, ReduceL2×36 |
| `ear_vae_ft_encoder.onnx` | 893.7 KB | `4168db14dec26806…` | 73.78 M Params; IN ['batch', 2, 'audio_time']; OUT ['batch', 64, '(audio_time//1024)']; ops: Mul×108, Add×51, Conv×37, Sin×36 |
| `ear_vae_music_finetuned.pyt` | 564.1 MB | `993e69bea97bd9c9…` | 147.86 M Params |
| `encoder.onnx` | 282.3 MB | `8f97ef673d4120ed…` | 73.78 M Params; IN ['batch', 2, 'audio_time']; OUT ['batch', 64, '(audio_time//1024)']; ops: Mul×108, Add×51, Conv×37, Sin×36 |

### `ear_vae2_upstream`

**Lokale Karten/Lizenzen:**

- `LICENSE` — `(Lizenz)`
- `README.md` — `# εar-VAE2 / **Fourier is Frontier: Frequency-Aware Autoencoding for High-Fidelity Music Reconstruction** / Kangdi Wang 1 · Yusheng Dai 2 · Jin Xu 1†`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |

### `era_classifier`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `era_anchors.npy` | 30.2 KB | `6691c15aaad73dcd…` | npy [15, 513] float32 |

### `fcpe`

**Lokale Karten/Lizenzen:**

- `LICENSE` — `(Lizenz)`
- `README.md` — `TorchFCPE / &nbsp; / ## Overview`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `fcpe.onnx` | 66.0 MB | `c1314254106a9c39…` | 17.24 M Params; IN ['batch', 'time', 128]; OUT ['batch', 'time', 360]; ops: Constant×473, Shape×127, Gather×114, Unsqueeze×108 |
| `fcpe.pt` | 65.8 MB | `c3a8dd2dbd51baf1…` | 17.24 M Params; hparams: 60 Felder |

### `flashsr`

**Lokale Karten/Lizenzen:**

- `README.md` — `# FlashSR / This is a tiny audio super-resolution model based on hierspeech++ that upscales 16khz audio into much clearer 48khz audio at speed over 200x realtime to 400x realtime! / FlashSR is release`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `FastAudioSR/SR48k.pth` | 1.6 MB | `62c70874ac4efeb4…` | 0.40 M Params; 96 Stubs ohne Formangabe |
| `flashsr.onnx` | 2.4 MB | `41396e3d58fc24fa…` | 0.29 M Params; IN [1, 1, 68000]; OUT [1, 1, 204000]; ops: Mul×47, Transpose×39, Concat×39, Add×37 |
| `flashsr_f4.onnx` | 2.4 MB | `41396e3d58fc24fa…` | 0.29 M Params; IN [1, 1, 68000]; OUT [1, 1, 204000]; ops: Mul×47, Transpose×39, Concat×39, Add×37 |
| `models/model.onnx` | 759.7 KB | `db8f28d1babc905a…` | 0.09 M Params; IN [1, 1, 's0']; OUT [1, 1, '3*s0']; ops: Mul×47, Conv×27, Pad×25, Add×23 |
| `models/model_lite.onnx` | 889.5 KB | `4cf7e3041fc4b99c…` | 0.09 M Params; IN [1, 1, 's0']; OUT [1, 1, '3*s0']; ops: Mul×49, Transpose×39, Add×37, Conv×27 |
| `models/upsampler.pth` | 1.6 MB | `62c70874ac4efeb4…` | 0.40 M Params; 96 Stubs ohne Formangabe |

### `forensics`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |

### `gacela`

**Lokale Karten/Lizenzen:**

- `LICENSE` — `(Lizenz)`
- `README.md` — `# GACELA - Generative adversarial context encoder for audio inpainting / We introduce GACELA, a generative adversarial network (GAN) designed to restore missing musical audio data with a duration rang`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `model/01_400000.pt` | 386.0 MB | `edbd8b03b7ae9ef0…` | 101.17 M Params |
| `model/gacela_core.onnx` | 117.1 KB | `18c5115ae620ab67…` | 24.13 M Params; IN [1, 1, 80, 240], [1, 1, 80, 240]; OUT [1, 1, 256, 256]; ops: Relu×12, ConvTranspose×11, Conv×8, ReduceL2×7 |
| `model/gacela_core_ft39.onnx` | 54.8 MB | `f33e1b807bba3dcd…` | 14.33 M Params; IN [1, 1, 80, 120], [1, 1, 80, 120]; OUT [1, 1, 512, 64]; ops: Relu×12, ConvTranspose×11, Conv×8, ReduceL2×7 |

### `gacela_upstream`

**Lokale Karten/Lizenzen:**

- `LICENSE` — `(Lizenz)`
- `README.md` — `# GACELA - Generative adversarial context encoder for audio inpainting / We introduce GACELA, a generative adversarial network (GAN) designed to restore missing musical audio data with a duration rang`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |

### `harmonic_inpainting`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `inpainting_best.onnx` | 769.2 MB | `77f3e663fbdd56ff…` | 201.50 M Params; IN ['batch', 'time', 1], ['batch']; OUT ['batch', 'time', 1]; ops: Constant×806, Mul×314, Add×239, Unsqueeze×209 |
| `inpainting_best.pt` | 768.7 MB | `f8765d14564e65c0…` | 201.50 M Params |
| `inpainting_mask_best.onnx` | 770.7 MB | `17e8f7e524c802bc…` | 201.89 M Params; IN ['batch', 'time', 2], ['batch']; OUT ['batch', 'time', 1]; ops: Constant×806, Mul×314, Add×239, Unsqueeze×209 |

### `hifi_gan`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `hifi_gan.onnx` | 3.6 MB | `b36be9f4b23132ba…` | 0.93 M Params; IN [1, 80, 'seq_length']; OUT [1, 'seq_length', 2560]; ops: LeakyRelu×77, Conv×74, Add×44, ConvTranspose×4 |

### `hubert`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `hubert_model.onnx` | 1.2 MB | `22f0ff140cb1ef9b…` | 94.37 M Params; IN ['s6', 's76']; OUT [1, '((((((((s76//80)) - 3)//2)) - 1)//2)) + 1', 768]; ops: Add×122, MatMul×97, Reshape×75, Mul×66 |

### `kim_inst`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `kim_inst.onnx` | 63.7 MB | `86b1940e7122fbdd…` | 16.68 M Params; IN ['batch_size', 4, 3072, 256]; OUT ['batch_size', 4, 3072, 256]; ops: Relu×66, Conv×40, BatchNormalization×27, MatMul×22 |

### `kim_vocal_2`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `kim_vocal_2.onnx` | 63.7 MB | `ce74ef3b6a6024ce…` | 16.68 M Params; IN ['batch_size', 4, 3072, 256]; OUT ['batch_size', 4, 3072, 256]; ops: Relu×66, Conv×40, BatchNormalization×27, MatMul×22 |

### `matchering2.0`

**Lokale Karten/Lizenzen:**

- `DOCKER.md` — `# Docker Image - The Easiest Way / **Matchering 2.0** works on all major platforms using **Docker**. / ## Choose yours`
- `LICENSE` — `(Lizenz)`
- `README.md` — `/ / /`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |

### `melbandroformer`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `melbandroformer_optimized.onnx` | 859.5 MB | `1a85b53bce2e90e4…` | 225.13 M Params; IN [1, 'duration', 60, 384]; OUT ['Reshapeoutput_dim_0', 'duration', 'Reshapeoutput_dim_2', 'Reshapeoutput_dim_3', 'Reshapeoutput_dim_4']; ops: Unsqueeze×613, Gather×502, Mul×350, MatMul×264 |

### `mert`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `clean_music_centroid.npy` | 4.1 KB | `199f39a5eecb94ca…` | npy [1024] float32 |
| `mert.onnx` | 338.5 MB | `0985d2f9cda1f5ef…` | 315.43 M Params; IN ['batch', 'samples']; OUT ['batch', 'frames', 1024]; ops: Constant×684, Mul×453, Add×327, Cast×193 |
| `mert_330m.onnx` | 338.5 MB | `0985d2f9cda1f5ef…` | 315.43 M Params; IN ['batch', 'samples']; OUT ['batch', 'frames', 1024]; ops: Constant×684, Mul×453, Add×327, Cast×193 |

### `mert-v1-330m`

**Lokale Konfigurations-Evidenz:**

- `config.json`: `_name_or_path`=m-a-p/MERT-v1-330M, `architectures`=['MERTModel'], `hidden_size`=1024, `model_type`=mert_model, `num_hidden_layers`=24, `sample_rate`=24000, `vocab_size`=32
- `preprocessor_config.json`: `sampling_rate`=24000

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `MERT-v1-330M_fairseq.pt` | 3.7 GB | — | 997.67 M Params; 16 Stubs ohne Formangabe; hparams: 60 Felder; [hash_note: übersprungen (> 1024 MB)] |
| `pytorch_model.bin` | 1.2 GB | — | 315.43 M Params; [hash_note: übersprungen (> 1024 MB)] |

### `mert_denoiser`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `mert_decoder.onnx` | 13.3 MB | `ac6396691eded5d8…` | 3.48 M Params; IN ['batch', 2, 481, 'time'], ['batch', 'time_mert', 768]; OUT ['batch', 2, 481, 'time']; ops: Constant×56, Mul×18, Shape×12, Concat×12 |

### `miipher_dit`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `flow_matching_dit.onnx` | 769.2 MB | `077f6d755c07d807…` | 201.50 M Params; IN ['batch', 'time', 1], ['batch']; OUT ['batch', 'time', 1]; ops: Constant×806, Mul×314, Add×239, Unsqueeze×209 |
| `whisper_denoiser_best.pt` | 12.9 MB | `b1c00ad50d657bf9…` | 3.38 M Params |

### `mp_senet`

**Lokale Konfigurations-Evidenz:**

- `config.json`: `hop_size`=100, `n_fft`=400, `sampling_rate`=16000, `win_size`=400

**Lokale Karten/Lizenzen:**

- `LICENSE` — `(Lizenz)`
- `README.md` — `# Explicit Estimation of Magnitude and Phase Spectra in Parallel for High-Quality Speech Enhancement / ### Ye-Xin Lu, Yang Ai, Zhen-Hua Ling / In our paper, we proposed MP-SENet: a TF-domain monaural`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `finetuned/checkpoint_latest.pt` | 26.2 MB | `f03683c71dd71ccc…` | 6.79 M Params; 247 Stubs ohne Formangabe |
| `finetuned/mp_senet_musik.onnx` | 9.0 MB | `fe3cae1adf2146d2…` | 2.26 M Params; IN ['batch', 201, 'time'], ['batch', 201, 'time']; OUT ['batch', 'Muldenoised_amp_dim_1', 'time']; ops: Constant×559, Unsqueeze×194, Reshape×118, Concat×108 |
| `mp_senet.onnx` | 9.0 MB | `f630f8628574f912…` | 2.26 M Params; IN ['batch', 201, 'time'], ['batch', 201, 'time']; OUT ['batch', 'Muldenoised_amp_dim_1', 'time']; ops: Constant×559, Unsqueeze×194, Reshape×118, Concat×108 |

### `mpsenet`

**Lokale Karten/Lizenzen:**

- `LICENSE` — `(Lizenz)`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |

### `muq_eval`

**Lokale Karten/Lizenzen:**

- `LICENSE` — `(Lizenz)`
- `README.md` — `# MuQ-Eval: Per-Sample Quality Prediction for Generated Music / Open-source neural quality metric for generated music using frozen MuQ-310M representations. / **Paper:** "Frozen Music Representations`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |

### `muq_mulan`

**Lokale Konfigurations-Evidenz:**

- `config.json`: `hop_length`=240, `n_mels`=128, `w2v2_config.architectures`=['Wav2Vec2ConformerForCTC'], `w2v2_config.hidden_size`=1024, `w2v2_config.model_type`=wav2vec2-conformer, `w2v2_config.num_hidden_layers`=24, `w2v2_config.vocab_size`=32

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `model.safetensors` | 1.2 GB | — | 525 Tensoren; dtypes: F32×507, I64×18; [hash_note: übersprungen (> 1024 MB)] |
| `mulan/pytorch_model.bin` | 2.5 GB | — | 663.40 M Params; 18 Stubs ohne Formangabe; [hash_note: übersprungen (> 1024 MB)] |
| `muq_eval_a1_head.pt` | 1.5 MB | `2ba1e9048d0eb948…` | 0.39 M Params |
| `muq_eval_bn_stats.pt` | 138.1 KB | `2e461b82bc2de756…` | 0.03 M Params; 18 Stubs ohne Formangabe |
| `muq_mulan.onnx` | 978.3 KB | `bfd8ea005ecf65ed…` | 317.28 M Params; IN [1, 240000]; OUT [1, 768]; ops: Add×160, MatMul×123, Transpose×100, Mul×97 |

### `nara_wpe`

**Lokale Karten/Lizenzen:**

- `LICENSE` — `(Lizenz)`
- `README.rst` — `======== / nara_wpe / ======== / .. image:: https://readthedocs.org/projects/nara-wpe/badge/?version=latest / :target: http://nara-wpe.readthedocs.io/en/latest/`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |

### `nvsr`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `nvsr.onnx` | 389.3 KB | `a469b590a882bc6c…` | 0.09 M Params; IN ['batch', 1, 'time']; OUT ['batch', 1, 'time']; ops: Constant×69, Mul×39, Add×33, Resize×26 |
| `nvsr.ts` | 702.4 KB | `c3dea9caa15308af…` | 396 Zip-Einträge |

### `panns`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `panns_wavegram_logmel_cnn14.onnx` | 79.1 KB | `12b573e62e0fe491…` | 82.14 M Params; IN [1, 320000]; OUT [1, 527]; ops: Conv×23, Relu×22, AveragePool×7, Transpose×5 |

### `resemblyzer`

**Lokale Karten/Lizenzen:**

- `LICENSE` — `(Lizenz)`
- `README.md` — `Resemblyzer allows you to derive a **high-level representation of a voice** through a deep learning model (referred to as the voice encoder). Given an audio file of speech, it creates a summary vector`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `build/lib/resemblyzer/pretrained.pt` | 16.3 MB | `39373b86598fa3da…` | 4.27 M Params |
| `resemblyzer/pretrained.pt` | 16.3 MB | `39373b86598fa3da…` | 4.27 M Params |
| `resemblyzer_voice_encoder.onnx` | 5.4 MB | `895799ddc62de9ea…` | 1.42 M Params; IN ['batch', 'time', 40]; OUT ['batch', 256]; ops: Constant×25, Slice×6, LSTM×3, Gather×2 |

### `rmvpe`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `rmvpe.onnx` | 344.9 MB | `5370e71ac80af8b4…` | 90.40 M Params; IN [1, 128, 'time']; OUT ['Sigmoidoutput_dim_0', 'Sigmoidoutput_dim_1', 'time']; ops: Conv×124, Relu×117, Add×57, Constant×8 |

### `rvc`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `hubert_base.pt` | 180.7 MB | `f54b40fd2802423a…` | 94.70 M Params; 9 Stubs ohne Formangabe; hparams: 60 Felder |
| `rmvpe.pt` | 172.8 MB | `6d62215f4306e3ca…` | 90.47 M Params; 118 Stubs ohne Formangabe |

### `scnet_4stems`

**Lokale Karten/Lizenzen:**

- `README.md` — `--- / license: apache-2.0 / ---`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `huge_scnet_4stems_v1.2.ckpt` | 377.3 MB | `807f470b13fe0734…` | 196.47 M Params; 1 Stubs ohne Formangabe |

### `sgmse_plus`

**Lokale Karten/Lizenzen:**

- `LICENSE` — `(Lizenz)`
- `README.md` — `# Speech Enhancement and Dereverberation with Diffusion-based Generative Models / / This repository contains the official PyTorch implementations for the papers:`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `finetuned/checkpoint_latest.ckpt` | 751.4 MB | `6dd93b2c9158cabc…` | 196.77 M Params; 646 Stubs ohne Formangabe |
| `finetuned/sgmse_musik_best.ckpt` | 250.4 MB | `1044979b4b043751…` | 65.59 M Params |
| `sgmse_musik_core.onnx` | 251.4 MB | `4e1015f519f7eacc…` | 65.59 M Params; IN ['batch', 2, 'freq', 'frames'], ['batch', 2, 'freq', 'frames']; OUT ['batch', 2, 'freq', 'frames']; ops: Constant×3274, Unsqueeze×784, Reshape×587, Add×527 |
| `sgmse_plus.ts` | 251.2 MB | `feabbfb3acee8ab5…` | 1631 Zip-Einträge |
| `sgmse_plus_core.onnx` | 251.4 MB | `6c7961e7d02e7400…` | 65.59 M Params; IN ['batch', 2, 'freq', 'frames'], ['batch', 2, 'freq', 'frames']; OUT ['batch', 2, 'freq', 'frames']; ops: Constant×3274, Unsqueeze×784, Reshape×587, Add×527 |
| `sgmse_plus_src_1.ckpt` | 1.2 GB | — | 327.95 M Params; 11 Stubs ohne Formangabe; hparams: 30 Felder; [hash_note: übersprungen (> 1024 MB)] |

### `silero`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `silero_en_v5.onnx` | 112.1 MB | `352559c6e9b310d2…` | 29.37 M Params; IN ['batch', 'samples']; OUT ['batch', 'frames', 999]; ops: Constant×527, Unsqueeze×195, Add×187, Transpose×135 |

### `singmos`

**Lokale Karten/Lizenzen:**

- `LICENSE` — `(Lizenz)`
- `README.md` — `/ # SingMOS / /`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `singmos_pro.onnx` | 2.7 MB | `1c11517b6d8571df…` | 315.51 M Params; IN ['batch', 1, 'time'], [1]; OUT [1]; ops: Reshape×250, Add×234, MatMul×171, Transpose×142 |

### `symphonia`

**Lokale Konfigurations-Evidenz:**

- `symphonia_config.json`: `training.sr`=48000

**Lokale Karten/Lizenzen:**

- `README.md` — `# Symphonia — Instrumentalrestaurierung / Symphonia ist das Gegenstück zu Cantus für Instrumentalstems. Es restauriert / 'drums + bass + other' nach der Separation und vor KIM-Inst. / Bei rein instrum`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `symphonia_tiny.onnx` | 1.3 MB | `3c56e737e484663e…` | 0.31 M Params; IN ['batch', 'time', 1], ['batch']; OUT ['Addv_dim_0', 'Addv_dim_1', 1]; ops: Constant×199, Mul×81, Add×65, Unsqueeze×39 |

### `utmosv2`

**Lokale Karten/Lizenzen:**

- `LICENSE` — `(Lizenz)`
- `README.md` — `/ / / / UTMOSv2: UTokyo-SaruLab MOS Prediction System`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `fold0_s42_best_model.pth` | 780.6 MB | `c8149d988e4bbf3f…` | 204.29 M Params; 440 Stubs ohne Formangabe |
| `utmosv2_ssl_encoder.onnx` | 360.2 MB | `cfabccf4a06da972…` | 94.37 M Params; IN ['batch', 'seq_len']; OUT ['batch', 'seq_frames', 768]; ops: Constant×308, Add×119, MatMul×97, Unsqueeze×73 |

### `uvr_mdx_net`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `uvr_mdx_net_inst_hq_1.onnx` | 63.7 MB | `38a045c4ded87e3b…` | 16.68 M Params; IN ['batch_size', 4, 3072, 256]; OUT ['batch_size', 4, 3072, 256]; ops: Relu×66, Conv×40, BatchNormalization×27, MatMul×22 |
| `uvr_mdx_net_inst_hq_2.onnx` | 63.7 MB | `197f8ab296df850f…` | 16.68 M Params; IN ['batch_size', 4, 3072, 256]; OUT ['batch_size', 4, 3072, 256]; ops: Relu×66, Conv×40, BatchNormalization×27, MatMul×22 |
| `uvr_mdx_net_inst_hq_3.onnx` | 63.7 MB | `317554b07fe1ea52…` | 16.68 M Params; IN ['batch_size', 4, 3072, 256]; OUT ['batch_size', 4, 3072, 256]; ops: Relu×66, Conv×40, BatchNormalization×27, MatMul×22 |
| `uvr_mdx_net_inst_hq_4.onnx` | 56.3 MB | `3c4b5b9b05090fdf…` | 14.76 M Params; IN ['batch_size', 4, 2560, 256]; OUT ['batch_size', 4, 2560, 256]; ops: Relu×66, Conv×40, BatchNormalization×27, MatMul×22 |

### `versa`

**Lokale Karten/Lizenzen:**

- `LICENSE` — `(Lizenz)`
- `README (1).md` — `--- / license: cc-by-4.0 / language: / - zh / - ja / tags:`
- `README.md` — `/ # VERSA: Versatile Evaluation of Speech and Audio / /`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `hub_cache/checkpoints/ft_wav2vec2_large_ll60k_mdf_p1_200epochs_all_192epochs.pth` | 1.2 GB | — | 317.47 M Params; [hash_note: übersprungen (> 1024 MB)] |

### `vocos`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `vocos_mel_spec_24khz.onnx` | 51.6 MB | `a84c58728a769e8a…` | 13.53 M Params; IN ['batch_size', 100, 'time']; OUT ['Clipmag_dim_0', 'Clipmag_dim_1', 'Clipmag_dim_2']; ops: Add×34, Constant×31, Mul×26, Transpose×20 |

### `vocos_48khz`

**Lokale Karten/Lizenzen:**

- `README.md` — `--- / license: mit / tags: / - audio / library_name: pytorch / ---`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `pytorch_model.bin` | 140.5 MB | `3315c87d130922df…` | 36.83 M Params |
| `vocos_48khz.onnx` | 156.0 MB | `6b3ecba89e4f4ef8…` | 40.89 M Params; IN ['batch', 128, 'T']; OUT ['batch', 'S']; ops: Constant×101, Add×48, Mul×30, Transpose×25 |

### `wav2vec2`

**Lokale Konfigurations-Evidenz:**

- `config.json`: `architectures`=['Wav2Vec2ForCTC'], `hidden_size`=1024, `model_type`=wav2vec2, `num_hidden_layers`=24, `vocab_size`=392
- `preprocessor_config.json`: `sampling_rate`=16000

**Lokale Karten/Lizenzen:**

- `README.md` — `--- / language: multilingual / datasets: / - common_voice / tags: / - speech`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `wav2vec2_forced_alignment.onnx` | 339.2 MB | `0569a28db36d2fd7…` | 315.84 M Params; IN ['batch', 'seq_len']; OUT ['batch', 'seq_frames', 392]; ops: Constant×983, Mul×461, Add×341, Unsqueeze×265 |

### `wav2vec2-base`

**Lokale Konfigurations-Evidenz:**

- `config.json`: `_name_or_path`=facebook/wav2vec2-base, `architectures`=['Wav2Vec2Model'], `hidden_size`=768, `model_type`=wav2vec2, `num_hidden_layers`=12, `vocab_size`=32
- `preprocessor_config.json`: `sampling_rate`=16000

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `model.safetensors` | 360.0 MB | `bc12da95c473f8f3…` | 211 Tensoren; dtypes: F32×211 |

### `whisper`

**Lokale Konfigurations-Evidenz:**

- `config.json`: `_name_or_path`=openai/whisper-base, `architectures`=['WhisperForConditionalGeneration'], `model_type`=whisper, `num_hidden_layers`=6, `vocab_size`=51865
- `preprocessor_config.json`: `hop_length`=160, `n_fft`=400, `sampling_rate`=16000
- `tokenizer.json`: `version`=1.0
- `vocab.json`: `version`=29153
- `whisper_tiny_vocab.json`: `version`=1

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `decoder_model_merged.onnx` | 656.2 MB | `7f4f078c567a125f…` | 158.80 M Params; IN ['batch_size', 'decoder_sequence_length'], ['batch_size', 'encoder_sequence_length / 2', 1280]; OUT ['batch_size', 'decoder_sequence_length', 51866]; ops: If×1 |
| `whisper_large_v3_turbo_encoder_fp16.onnx` | 1.2 GB | — | 636.97 M Params; IN ['batch_size', 128, 3000]; OUT ['batch_size', 1500, 1280]; ops: Add×389, MatMul×256, Mul×165, ReduceMean×130; [hash_note: übersprungen (> 1024 MB)] |
| `whisper_tiny.onnx` | 31.4 MB | `6642befb640f950d…` | 8.21 M Params; IN ['batch_size', 'feature_size', 'encoder_sequence_length']; OUT ['batch_size', 'encoder_sequence_length / 2', 384]; ops: Constant×164, Add×53, Unsqueeze×44, MatMul×32 |

## Lose Artefakte in `models/`

| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |
| --- | --- | --- | --- |
| `medium_shallow_v1.joblib` | 1.7 MB | `bf3f7ab1e9af6154…` | joblib: dict |

## Herkunftsbelege (`_archive_20260920` ↔ deployt)

> Maschinell gemessen (SHA-256 bzw. Parameterzahl) — Muster des EAR-VAE-Nachweises, hier für **alle** archivierten Artefakte. Das Archiv enthält die Generationen VOR dem jeweiligen Finetune; die Relation ist damit ein Evidenz-Beleg nach §III.13 (copilot-instructions.md).

### Byte-identisch (gleiche Gewichte, nur anderer Pfad) — 8

| Archiv-Artefakt | Parameter | Partner (deployt) |
| --- | --- | --- |
| `_archive_20260920/bs_roformer/bsr317_core_reexport.onnx` | 159.76 M | `bs_roformer/bs_roformer_317_core.onnx` |
| `_archive_20260920/ear_vae_upstream/ear_vae_upstream/ear_vae_ft_decoder.onnx` | 74.05 M | `ear_vae/ear_vae_ft_decoder.onnx` |
| `_archive_20260920/ear_vae_upstream/ear_vae_upstream/ear_vae_ft_decoder_inline.onnx` | 74.05 M | `ear_vae/decoder.onnx` |
| `_archive_20260920/ear_vae_upstream/ear_vae_upstream/ear_vae_ft_encoder.onnx` | 73.78 M | `ear_vae/ear_vae_ft_encoder.onnx` |
| `_archive_20260920/ear_vae_upstream/ear_vae_upstream/ear_vae_ft_encoder_inline.onnx` | 73.78 M | `ear_vae/encoder.onnx` |
| `_archive_20260920/ear_vae_upstream/ear_vae_upstream/ear_vae_music_finetuned.pyt` | 147.86 M | `ear_vae/ear_vae_music_finetuned.pyt` |
| `_archive_20260920/flashsr/flashsr_onnx_prod_20260810.onnx` | 0.09 M | `nvsr/nvsr.onnx` |
| `_archive_20260920/harmonic_inpainting/inpainting_best.pt` | 201.50 M | `harmonic_inpainting/inpainting_best.pt` |

### Architektur-gleich, Gewichte verschieden (Finetune/Generation) — 6

| Archiv-Artefakt | Parameter | Partner (deployt) |
| --- | --- | --- |
| `_archive_20260920/bs_roformer/bsr317_core_fp16.onnx` | 159.76 M | `bs_roformer/bs_roformer_317_core.onnx` |
| `_archive_20260920/diffwave/diffwave_vocal_ft_rejected_epoch2_gatefail.ckpt` | 2.62 M | `diffwave/diffwave.ckpt` |
| `_archive_20260920/diffwave/diffwave_vocal_ft_rejected_epoch7_aborted.ckpt` | 2.62 M | `diffwave/diffwave.ckpt` |
| `_archive_20260920/ear_vae_upstream/ear_vae_upstream/pretrained_weight/ear_vae_44k.pyt` | 147.86 M | `ear_vae/ear_vae_music_finetuned.pyt` |
| `_archive_20260920/miipher_dit/whisper_denoiser_latest.pt` | 3.38 M | `miipher_dit/whisper_denoiser_best.pt` |
| `_archive_20260920/sgmse_plus/sgmse_wsj0_reverb.ckpt` | 327.95 M | `sgmse_plus/sgmse_plus_src_1.ckpt` |

### Ohne Gegenstück im deployten Bestand — 14

| Archiv-Artefakt | Parameter | Partner (deployt) |
| --- | --- | --- |
| `_archive_20260920/banquet/banquet_vinyl_batch.onnx` | 23.69 M | — |
| `_archive_20260920/bw_reconstructor/best_model.pt` | 1.09 M | — |
| `_archive_20260920/bw_reconstructor/best_model_v3.pt` | 1.09 M | — |
| `_archive_20260920/bw_reconstructor/best_model_v4.pt` | 1.09 M | — |
| `_archive_20260920/bw_reconstructor/best_model_v5_pre_retrain.pt` | 1.03 M | — |
| `_archive_20260920/clean_music_centroid.npy` | — | — |
| `_archive_20260920/ear_vae/decoder_orig_20260730_backup.onnx` | 84.25 M | — |
| `_archive_20260920/ear_vae/encoder_orig_20260730_backup.onnx` | 84.12 M | — |
| `_archive_20260920/gacela_venv/lib/python3.10/site-packages/distutils-precedence.pth` | — | — |
| `_archive_20260920/harmonic_inpainting/inpainting_mask_best.pt` | 201.89 M | — |
| `_archive_20260920/medium_classifier/medium_shallow_v1.joblib` | — | — |
| `_archive_20260920/melbandroformer/melbandroformer_optimized.onnx.rocm_safe.onnx` | 225.13 M | — |
| `_archive_20260920/melbandroformer/melbandroformer_optimized.onnx.rocm_safe.onnx.probe.onnx` | 225.13 M | — |
| `_archive_20260920/singmos/singmos_pro.pt` | — | — |
