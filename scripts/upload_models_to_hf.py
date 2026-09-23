#!/usr/bin/env python3
"""
Aurik 10.0.0 — Upload all custom/converted model files to HuggingFace Hub.
Skips files that are available from official upstream repos.
Resumes automatically if interrupted (checks existing files on HF).

Usage:
    HF_TOKEN=hf_... python scripts/upload_models_to_hf.py
    HF_TOKEN=hf_... python scripts/upload_models_to_hf.py --dry-run
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/tmp/aurik_hf_upload.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

HF_REPO = "michaelarnold2307/aurik-models"
MODELS = Path(__file__).resolve().parents[1] / "models"
DRY_RUN = "--dry-run" in sys.argv

# ---------------------------------------------------------------------------
# Files to upload: (local_relative_to_models, path_in_hf_repo)
# Files from official upstream repos are SKIPPED (users download from source).
# ---------------------------------------------------------------------------
UPLOAD_LIST: list[tuple[str, str]] = [
    # ── DeepFilterNet v3.II (custom ONNX export) ───────────────────────────
    ("deepfilternet_v3_ii/enc.onnx", "deepfilternet_v3_ii/enc.onnx"),
    ("deepfilternet_v3_ii/dec.onnx", "deepfilternet_v3_ii/dec.onnx"),
    ("deepfilternet_v3_ii/erb_dec.onnx", "deepfilternet_v3_ii/erb_dec.onnx"),
    # ── MDX23C / Kim (custom ONNX export) ─────────────────────────────────
    ("mdx23c/models/Kim_Vocal_2.onnx", "mdx23c/Kim_Vocal_2.onnx"),
    ("mdx23c/models/Kim_Inst.onnx", "mdx23c/Kim_Inst.onnx"),
    ("mdx23c/models/MDX23C-8KFFT-InstVoc_HQ.ckpt", "mdx23c/MDX23C-8KFFT-InstVoc_HQ.ckpt"),
    ("mdx23c/models/MDX23C-8KFFT-InstVoc_HQ_2.ckpt", "mdx23c/MDX23C-8KFFT-InstVoc_HQ_2.ckpt"),
    # ── Kim standalone copies ──────────────────────────────────────────────
    ("kim_inst/kim_inst.onnx", "mdx23c/Kim_Inst.onnx"),
    ("kim_vocal_2/kim_vocal_2.onnx", "mdx23c/Kim_Vocal_2.onnx"),
    # ── CREPE full (custom ONNX export) ───────────────────────────────────
    ("crepe/crepe/model-full.onnx", "crepe/model-full.onnx"),
    # ── Apollo (codec restoration) ─────────────────────────────────────────
    ("apollo/apollo_model.onnx", "apollo/apollo_model.onnx"),
    # ── HiFi-GAN (custom ONNX export) ─────────────────────────────────────
    ("hifi_gan/hifi_gan.onnx", "hifi_gan/hifi_gan.onnx"),
    # ── Banquet Vinyl (Aurik-original) ────────────────────────────────────
    ("banquet/banquet_vinyl_final.onnx", "banquet/banquet_vinyl_final.onnx"),
    ("banquet/banquet_vinyl_final.onnx.data", "banquet/banquet_vinyl_final.onnx.data"),
    ("banquet/ev-pre-aug.ckpt", "banquet/ev-pre-aug.ckpt"),
    # ── DCCRN ── VERBOTEN (§4.4: ersetzt durch MP-SENet / DeepFilterNet) ──────
    # ("dccrn/dccrn.onnx", ...)  # REMOVED — DCCRN ist VERBOTEN in Aurik 10.0.0
    # ── DiffWave (custom ONNX export) ─────────────────────────────────────
    ("diffwave/diffwave_model.onnx", "diffwave/diffwave_model.onnx"),
    ("diffwave/diffwave_model.onnx.data", "diffwave/diffwave_model.onnx.data"),
    # ── FullSubNet+ (custom ONNX export) ──────────────────────────────────
    ("fullsubnet_plus/fullsubnet_plus_1x1x257x100.onnx", "fullsubnet_plus/fullsubnet_plus_1x1x257x100.onnx"),
    ("fullsubnet_plus/fullsubnet_plus_1x1x257x200.onnx", "fullsubnet_plus/fullsubnet_plus_1x1x257x200.onnx"),
    ("fullsubnet_plus/fullsubnet_plus_1x1x257x500.onnx", "fullsubnet_plus/fullsubnet_plus_1x1x257x500.onnx"),
    ("fullsubnet_plus/fullsubnet_plus_1x1x257x1000.onnx", "fullsubnet_plus/fullsubnet_plus_1x1x257x1000.onnx"),
    ("fullsubnet_plus/fullsubnet_plus_1x1x257x2000.onnx", "fullsubnet_plus/fullsubnet_plus_1x1x257x2000.onnx"),
    ("fullsubnet_plus/fullsubnet_plus_1x1x257x3000.onnx", "fullsubnet_plus/fullsubnet_plus_1x1x257x3000.onnx"),
    ("fullsubnet_plus/fullsubnet_plus_1x1x257x5000.onnx", "fullsubnet_plus/fullsubnet_plus_1x1x257x5000.onnx"),
    ("fullsubnet_plus/fullsubnet_plus_1x1x257x7000.onnx", "fullsubnet_plus/fullsubnet_plus_1x1x257x7000.onnx"),
    ("fullsubnet_plus/fullsubnet_plus_1x1x257x8000.onnx", "fullsubnet_plus/fullsubnet_plus_1x1x257x8000.onnx"),
    ("fullsubnet_plus/fullsubnet_plus_1x1x257x10000.onnx", "fullsubnet_plus/fullsubnet_plus_1x1x257x10000.onnx"),
    ("fullsubnet_plus/model.onnx", "fullsubnet_plus/model.onnx"),
    # ── PANNs (custom ONNX export) ────────────────────────────────────────
    ("panns/wavegram_logmel_cnn14.onnx", "panns/wavegram_logmel_cnn14.onnx"),
    ("panns/wavegram_logmel_cnn14.onnx.data", "panns/wavegram_logmel_cnn14.onnx.data"),
    ("panns/panns_wavegram_logmel_cnn14.onnx", "panns/panns_wavegram_logmel_cnn14.onnx"),
    ("panns/panns_wavegram_logmel_cnn14.onnx.data", "panns/panns_wavegram_logmel_cnn14.onnx.data"),
    # ── HTDemucs 6s (custom ONNX export) ─────────────────────────────────
    ("demucs/htdemucs_6s.onnx", "demucs/htdemucs_6s.onnx"),
    ("demucs/htdemucs_6s.onnx.data", "demucs/htdemucs_6s.onnx.data"),
    # ── AudioLDM2 (custom ONNX export) ────────────────────────────────────
    ("audioldm2/audioldm2.onnx", "audioldm2/audioldm2.onnx"),
    # ── AudioLM (custom ONNX export) ──────────────────────────────────────
    ("audiolm/clap.rvq.950_no_fusion.semantic.onnx", "audiolm/clap.rvq.950_no_fusion.semantic.onnx"),
    ("audiolm/clap.rvq.950_no_fusion.semantic.onnx.data", "audiolm/clap.rvq.950_no_fusion.semantic.onnx.data"),
    ("audiolm/coarse.transformer.18000.semantic.onnx", "audiolm/coarse.transformer.18000.semantic.onnx"),
    ("audiolm/coarse.transformer.18000.semantic.onnx.data", "audiolm/coarse.transformer.18000.semantic.onnx.data"),
    ("audiolm/fine.transformer.24000.semantic.onnx", "audiolm/fine.transformer.24000.semantic.onnx"),
    ("audiolm/fine.transformer.24000.semantic.onnx.data", "audiolm/fine.transformer.24000.semantic.onnx.data"),
    ("audiolm/semantic.transformer.14000.semantic.onnx", "audiolm/semantic.transformer.14000.semantic.onnx"),
    ("audiolm/semantic.transformer.14000.semantic.onnx.data", "audiolm/semantic.transformer.14000.semantic.onnx.data"),
    # ── MelBandRoformer (custom ONNX exports) ─────────────────────────────
    ("melbandroformer/melbandroformer.onnx", "melbandroformer/melbandroformer.onnx"),
    ("melbandroformer/melbandroformer_optimized.onnx", "melbandroformer/melbandroformer_optimized.onnx"),
    ("melbandroformer/bs_conformer_medium.onnx", "melbandroformer/bs_conformer_medium.onnx"),
    ("melbandroformer/bs_conformer_medium_stft_decomp.onnx", "melbandroformer/bs_conformer_medium_stft_decomp.onnx"),
    (
        "melbandroformer/bs_conformer_medium_stft_decomp.onnx.data",
        "melbandroformer/bs_conformer_medium_stft_decomp.onnx.data",
    ),
    (
        "melbandroformer/bs_conformer_medium_stft_decomp_emb.onnx",
        "melbandroformer/bs_conformer_medium_stft_decomp_emb.onnx",
    ),
    ("melbandroformer/scnet_masked_xl_ihf.onnx", "melbandroformer/scnet_masked_xl_ihf.onnx"),
    ("melbandroformer/scnet_masked_xl_ihf.fixed.onnx", "melbandroformer/scnet_masked_xl_ihf.fixed.onnx"),
    ("melbandroformer/scnet_masked_xl_ihf_stft_decomp.onnx", "melbandroformer/scnet_masked_xl_ihf_stft_decomp.onnx"),
    (
        "melbandroformer/scnet_masked_xl_ihf_stft_decomp.onnx.data",
        "melbandroformer/scnet_masked_xl_ihf_stft_decomp.onnx.data",
    ),
    (
        "melbandroformer/scnet_masked_xl_ihf_stft_decomp_emb.onnx",
        "melbandroformer/scnet_masked_xl_ihf_stft_decomp_emb.onnx",
    ),
    (
        "melbandroformer/scnet_masked_xl_ihf_stft_decomp_fix_scatter.onnx",
        "melbandroformer/scnet_masked_xl_ihf_stft_decomp_fix_scatter.onnx",
    ),
    (
        "melbandroformer/scnet_masked_xl_ihf_stft_decomp_fix_scatter.fixed.onnx",
        "melbandroformer/scnet_masked_xl_ihf_stft_decomp_fix_scatter.fixed.onnx",
    ),
    (
        "melbandroformer/scnet_masked_xl_ihf_stft_decomp_fix_scatter.reordered.onnx",
        "melbandroformer/scnet_masked_xl_ihf_stft_decomp_fix_scatter.reordered.onnx",
    ),
    # ── UVR MDX-Net (custom ONNX export) ─────────────────────────────────
    ("uvr_mdx_net/uvr_mdx_net_inst_hq_1.onnx", "uvr_mdx_net/uvr_mdx_net_inst_hq_1.onnx"),
    ("uvr_mdx_net/uvr_mdx_net_inst_hq_2.onnx", "uvr_mdx_net/uvr_mdx_net_inst_hq_2.onnx"),
    ("uvr_mdx_net/uvr_mdx_net_inst_hq_3.onnx", "uvr_mdx_net/uvr_mdx_net_inst_hq_3.onnx"),
    ("uvr_mdx_net/uvr_mdx_net_inst_hq_4.onnx", "uvr_mdx_net/uvr_mdx_net_inst_hq_4.onnx"),
    # ── UTMOSv2 (pretrained checkpoints) ──────────────────────────────────
    ("utmosv2/fold0_s42_best_model.pth", "utmosv2/fold0_s42_best_model.pth"),
    ("utmosv2/fold1_s42_best_model.pth", "utmosv2/fold1_s42_best_model.pth"),
    ("utmosv2/fold2_s42_best_model.pth", "utmosv2/fold2_s42_best_model.pth"),
    ("utmosv2/fold3_s42_best_model.pth", "utmosv2/fold3_s42_best_model.pth"),
    ("utmosv2/fold4_s42_best_model.pth", "utmosv2/fold4_s42_best_model.pth"),
    # ── CDPAM ── VERBOTEN (§4.4: ersetzt durch VERSA MOS-Metrik) ───────────────
    # ("cdpam/...", ...)  # REMOVED — CDPAM ist VERBOTEN in Aurik 10.0.0 (kein Musiktraining)
    # Nachfolger: versa_plugin (VERSA 2024 MOS) + utmos_plugin (Gesangs-MOS)
    # ── Silero EN v5 (custom ONNX) ────────────────────────────────────────
    ("silero/silero_en_v5.onnx", "silero/silero_en_v5.onnx"),
    # ── Voice Cloning Detection ───────────────────────────────────────────
    ("voice-cloning-detection/model/model.safetensors", "voice-cloning-detection/model.safetensors"),
    # ── Whisper ONNX (custom export) ──────────────────────────────────────
    ("whisper/whisper-base_beamsearch.onnx", "whisper/whisper-base_beamsearch.onnx"),
    ("whisper/whisper_tiny.onnx", "whisper/whisper_tiny.onnx"),
    # ── Resemble Enhance (also on official HF, but bundled locally) ───────
    ("resemble_enhance/model.onnx", "resemble_enhance/model.onnx"),
    ("resemble_enhance/ds/G/default/mp_rank_00_model_states.pt", "resemble_enhance/mp_rank_00_model_states.pt"),
    # ── Vocos (also on official HF, but bundled locally) ──────────────────
    ("vocos/vocos_mel_spec_24khz.onnx", "vocos/vocos_mel_spec_24khz.onnx"),
]


def main() -> None:
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        log.error("HF_TOKEN nicht gesetzt. Abbruch.")
        sys.exit(1)

    try:
        from huggingface_hub import HfApi
    except ImportError:
        log.error("huggingface_hub nicht installiert: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi(token=token)

    # Bereits hochgeladene Dateien ermitteln (für Resume)
    log.info("Lade Dateiliste von %s …", HF_REPO)
    try:
        existing = {f.rfilename for f in api.list_repo_files(HF_REPO)}  # type: ignore[attr-defined]
        log.info("%d Dateien bereits im Repo.", len(existing))
    except Exception as exc:
        log.warning("Kann Repo-Inhalt nicht abrufen: %s — nehme an, leer.", exc)
        existing = set()

    total = len(UPLOAD_LIST)
    done = skipped = failed = 0

    for i, (local_rel, hf_path) in enumerate(UPLOAD_LIST, 1):
        local = MODELS / local_rel
        prefix = f"[{i:3}/{total}]"

        if not local.exists():
            log.warning("%s FEHLT lokal: %s", prefix, local_rel)
            continue

        if hf_path in existing:
            log.info("%s OK (bereits vorhanden): %s", prefix, hf_path)
            skipped += 1
            continue

        size_mb = local.stat().st_size / 1_048_576
        log.info("%s Uploading %.1f MB: %s → %s", prefix, size_mb, local_rel, hf_path)

        if DRY_RUN:
            log.info("   [DRY-RUN] übersprungen")
            continue

        t0 = time.time()
        try:
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=hf_path,
                repo_id=HF_REPO,
                repo_type="model",
                commit_message=f"upload {hf_path}",
            )
            elapsed = time.time() - t0
            speed = size_mb / elapsed if elapsed > 0 else 0
            log.info("   ✓ fertig in %.0fs (%.1f MB/s)", elapsed, speed)
            existing.add(hf_path)
            done += 1
        except Exception as exc:
            log.error("   ✗ FEHLER: %s", exc)
            failed += 1
            time.sleep(5)  # kurze Pause bei Fehler, dann weiter

    log.info("─" * 60)
    log.info("Upload abgeschlossen: %d neu, %d übersprungen, %d Fehler", done, skipped, failed)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
