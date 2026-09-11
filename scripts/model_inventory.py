#!/usr/bin/env python3
"""scripts/model_inventory.py — Modell-Inventar-Status mit Quellen-Empfehlung.

§Modell-Transparenz (2026-09-11): Ehrlicher Überblick, welche SOTA-ML-Modelle
lokal vorhanden sind und welche fehlen — mit konkreter Empfehlung, woher
(HuggingFace-Repo, GitHub-Release oder internes Export-Script) das jeweilige
Modell bezogen wird. Kein Download — reine Status-/Empfehlungs-Ausgabe.

Aufruf:
    python scripts/model_inventory.py          # Konsolen-Bericht
    python scripts/model_inventory.py --json   # Maschinenlesbar (GUI)
    python scripts/model_inventory.py --one-line  # 1-Zeilen-Zusammenfassung (Pipeline-Start)

Design-Prinzip: Die Pipeline läuft NIE mit einem Modell, dessen Datei fehlt —
die Plugins haben Availability-Guards (§V6-DSP-Fallback). Dieses Inventar
macht die dadurch verbleibende Qualitäts-Lücke SICHTBAR (Anzeigen-Wahrheit).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


@dataclass
class ModelEntry:
    """Ein erwartetes ML-Modell: Pfad, Zweck, Domäne, Lizenz, Bezugsquelle."""

    path: str
    purpose: str
    domain: str  # "musik" | "sprache" | "audio-allgemein"
    license_note: str
    source: str  # konkrete Empfehlung (HF-Repo-ID / GitHub-URL / internes Script)
    required: bool = False  # True = im aktiven Pfad für Kern-Restaurierung


# ── Manifest: alle Produktions-Modelle mit Bezugsempfehlung ────────────────
_MODELS: list[ModelEntry] = [
    # Kern-Modelle (aktiv, vorhanden)
    ModelEntry(
        "models/demucs/htdemucs_6s.onnx",
        "Stem-Separation (htdemucs_6s)",
        "musik",
        "MIT",
        "github: facebookresearch/demucs (v4.0.2, htdemucs_ft)",
        required=True,
    ),
    ModelEntry(
        "models/deepfilternet_v3_ii/finetuned/enc.onnx",
        "DFN-3-Denoiser Encoder (musik-finetuned)",
        "musik",
        "MIT",
        "intern: scripts/train_df_musik.py → export (dfn_musik_best.pt)",
    ),
    ModelEntry(
        "models/deepfilternet_v3_ii/finetuned/dec.onnx",
        "DFN-3-Denoiser Decoder (musik-finetuned)",
        "musik",
        "MIT",
        "intern: scripts/train_df_musik.py → export (dfn_musik_best.pt)",
    ),
    ModelEntry(
        "models/deepfilternet_v3_ii/finetuned/erb_dec.onnx",
        "DFN-3 ERB-Decoder",
        "musik",
        "MIT",
        "intern: scripts/train_df_musik.py → export (dfn_musik_best.pt)",
    ),
    ModelEntry(
        "models/muq_mulan/muq_mulan.onnx",
        "MuQ-MuLan Audio-Embedding (768-d)",
        "musik",
        "CC-BY-NC-4.0 (Gewichte)",
        "github: TencentARC/MuQ-MuLan → intern: scripts/export_muq_mulan_onnx.py",
    ),
    ModelEntry(
        "models/whisper/whisper_tiny.onnx",
        "Whisper-Tiny Encoder (Lyrics-Timeline)",
        "sprache",
        "MIT",
        "HF: openai/whisper-tiny → ONNX-Export",
    ),
    ModelEntry(
        "models/fcpe/fcpe.onnx",
        "FCPE Pitch (schnell, Wow/Flutter-Detektion)",
        "musik",
        "MIT",
        "github: CNChTu/Diffusion-SVC (fcpe.pt) → ONNX-Export",
    ),
    ModelEntry(
        "models/rmvpe/rmvpe.onnx",
        "RMVPE Vokal-Pitch (SOTA, Stufe 2)",
        "musik",
        "MIT",
        "github: yxlllc/RMVPE (rmvpe.pt) → ONNX-Export",
    ),
    ModelEntry(
        "models/crepe/crepe.onnx",
        "CREPE Pitch (full, 89 MB)",
        "musik",
        "MIT",
        "github: marl/crepe (model-full.h5) → ONNX-Export",
    ),
    ModelEntry(
        "models/kim_vocal_2/kim_vocal_2.onnx",
        "KIM Vocal Enhancer (MDX23C-Maske)",
        "musik",
        "MIT",
        "HF: KimberleyJSN/Kim-Vocal-2 (kim_vocal_2.onnx)",
    ),
    ModelEntry(
        "models/kim_inst/kim_inst.onnx",
        "KIM Instrumental Enhancer (MDX23C-Maske)",
        "musik",
        "MIT",
        "HF: KimberleyJSN/KIM-INST (kim_inst.onnx)",
    ),
    ModelEntry(
        "models/panns/panns_wavegram_logmel_cnn14.onnx",
        "PANNs Tagging (Voranalyse)",
        "audio-allgemein",
        "MIT",
        "github: qiuqiangkong/audioset_tagging_cnn → ONNX-Export",
    ),
    ModelEntry(
        "models/ast/ast_model.onnx",
        "AST Audio-Transformer (Genre/Scene)",
        "audio-allgemein",
        "MIT",
        "HF: MIT/ast-finetuned-audioset-10-10-0.4593 → ONNX-Export",
    ),
    ModelEntry(
        "models/basicpitch/basicpitch.onnx",
        "Basic-Pitch (Multi-Pitch)",
        "musik",
        "Apache-2.0",
        "github: spotify/basic-pitch → ONNX-Export",
    ),
    ModelEntry(
        "models/bw_reconstructor/bw_reconstructor.onnx",
        "Bandbreiten-Rekonstruktion (BW-Extender)",
        "musik",
        "intern",
        "intern: scripts/train_bw_reconstructor.py (best_model.pt)",
    ),
    ModelEntry(
        "models/singmos/singmos_pro.onnx",
        "SingMOS Pro (Gesangs-MOS)",
        "musik",
        "MIT",
        "HF: South-TP-AI-Lab/SingMOS-Pro → scripts/export_singmos_onnx.py",
    ),
    ModelEntry(
        "models/hifi_gan/hifi_gan.onnx",
        "HiFi-GAN Vocoder",
        "audio-allgemein",
        "MIT",
        "github: jik876/hifi-gan (UNIVERSAL_V1)",
    ),
    ModelEntry(
        "models/bigvgan/bigvgan_v2.onnx", "BigVGAN Vocoder", "audio-allgemein", "MIT", "github: NVIDIA/BigVGAN (v2)"
    ),
    ModelEntry(
        "models/gacela/model/gacela_core.onnx",
        "GACELA Gap-Inpainting-Kern",
        "musik",
        "MIT",
        "HF: TencentARC/GACELA → scripts/export_gacela_onnx.py",
    ),
    # Fehlende Modelle — mit Empfehlung
    ModelEntry(
        "models/mert/mert.onnx",
        "MERT-MUSIK-Transformer (Qualitäts-Proxy/MUSHRA)",
        "musik",
        "CC-BY-NC-4.0 (Gewichte) — kommerzielle Nutzung prüfen!",
        "HF: m-a-p/MERT-v1-330M (oder -95M) → intern: scripts/export_mert*.py",
        required=True,
    ),
    ModelEntry(
        "models/mert-v1-330m/pytorch_model.bin",
        "MERT-v1-330M Checkpoint (Alternativ-Pfad)",
        "musik",
        "CC-BY-NC-4.0 (Gewichte)",
        "HF: m-a-p/MERT-v1-330M",
    ),
    ModelEntry(
        "models/mert-95m/MERT-v1-95M_fairseq.pt",
        "MERT-v1-95M Checkpoint (leicht)",
        "musik",
        "CC-BY-NC-4.0 (Gewichte)",
        "HF: m-a-p/MERT-v1-95M",
    ),
    ModelEntry(
        "models/utmos/utmos.onnx",
        "UTMOSv2 Sprach-MOS (Vokal-Qualität)",
        "sprache",
        "MIT",
        "HF: sarulab-speech/UTMOSv2 → ONNX-Export (nur für Vokal-Qualität, nicht Instrumental)",
    ),
    ModelEntry(
        "models/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "BS-RoFormer-Separation (SDR 12.98)",
        "musik",
        "MIT",
        "github: lucadellalib/BS-RoFormer (model_bs_roformer_ep_317_sdr_12.9755.ckpt)",
    ),
    ModelEntry(
        "models/cqtdiff_plus/score_network.onnx",
        "CQTdiff+ Score-Netz (Lücken-Inpainting)",
        "musik",
        "MIT",
        "github: sjc0423/CQTdiff (ckpt) → intern: scripts/export_cqtdiff*.py",
    ),
    ModelEntry(
        "models/dac/encoder_model.onnx",
        "DAC Encoder (Neural-Codec)",
        "audio-allgemein",
        "MIT",
        "github: descriptinc/descript-audio-codec (weights) → ONNX-Export",
    ),
    ModelEntry(
        "models/dac/decoder_model.onnx",
        "DAC Decoder (Neural-Codec)",
        "audio-allgemein",
        "MIT",
        "github: descriptinc/descript-audio-codec (weights) → ONNX-Export",
    ),
    ModelEntry(
        "models/artifact_detector.pt",
        "Artefakt-Detektor (intern trainiert)",
        "audio-allgemein",
        "intern — keine öffentliche Quelle",
        "intern: Trainings-Script erneut ausführen (Trainingsdaten im Repo)",
    ),
    ModelEntry(
        "models/muq/muq_eval_a1_head.pt",
        "MuQ-Eval-A1 MOS-Head",
        "musik",
        "MIT",
        "HF: zhudi2825/MuQ-Eval-A1 → intern: scripts/extract_muq_eval_a1_head.py",
    ),
    # Sprachmodell-Platzhalter — für Gesang UNGEEIGNET (bewusst NICHT empfohlen)
    ModelEntry(
        "models/resemble_enhance/resample_enhance.pt",
        "Resemble-Enhance Resampler (SPRACH-Modell)",
        "sprache",
        "MIT",
        "NICHT FÜR GESANG EMPFEHLEN — Sprach-Enhancer; für Vokal-SOTA stattdessen "
        "models/kim_vocal_2/kim_vocal_2.onnx oder CQTdiff+ verwenden",
    ),
]

# Ggf. weitere im Repo referenzierte Modelle dynamisch ergänzen? Nein —
# das Manifest ist die autoritative Liste (Pflege hier).


def scan() -> dict[str, dict]:
    """Prüft Existenz + Größe aller Manifest-Einträge."""
    out: dict[str, dict] = {}
    for m in _MODELS:
        p = _ROOT / m.path
        present = p.exists() and p.stat().st_size > 1000  # >1 KB = kein Platzhalter
        size_mb = round(p.stat().st_size / 1e6, 1) if p.exists() else 0.0
        placeholder = (
            present and p.stat().st_size < 50_000 and p.read_text(encoding="utf-8", errors="replace").startswith("#")
        )
        out[m.path] = {
            "present": bool(present and not placeholder),
            "placeholder": bool(placeholder),
            "size_mb": size_mb,
            "purpose": m.purpose,
            "domain": m.domain,
            "license": m.license_note,
            "source": m.source,
            "required": m.required,
        }
    return out


def summary_line() -> str:
    """1-Zeilen-Zusammenfassung für den Pipeline-Start (Anzeigen-Wahrheit)."""
    s = scan()
    total = len(s)
    present = sum(1 for v in s.values() if v["present"])
    missing_required = [k for k, v in s.items() if v["required"] and not v["present"]]
    missing_opt = [k for k, v in s.items() if not v["required"] and not v["present"]]
    msg = f"🧠 ML-Modelle: {present}/{total} vorhanden"
    if missing_required:
        msg += f" — FEHLEND (Kern): {', '.join(missing_required)}"
    if missing_opt:
        msg += f" — optional fehlend: {len(missing_opt)} (model_inventory.py --list für Details)"
    return msg


def main() -> int:
    parser = argparse.ArgumentParser(description="Modell-Inventar-Status mit Quellen-Empfehlung.")
    parser.add_argument("--json", action="store_true", help="Maschinenlesbare Ausgabe (GUI)")
    parser.add_argument("--one-line", action="store_true", help="Nur die 1-Zeilen-Zusammenfassung")
    parser.add_argument("--list", action="store_true", help="Nur die fehlenden Modelle auflisten")
    args = parser.parse_args()

    s = scan()
    if args.one_line:
        print(summary_line())
        return 0

    missing = {k: v for k, v in s.items() if not v["present"]}
    placeholders = {k: v for k, v in s.items() if v["placeholder"]}

    if args.json:
        print(
            json.dumps(
                {"models": s, "missing": sorted(missing), "summary": summary_line()}, ensure_ascii=False, indent=2
            )
        )
        return 0

    print("═" * 100)
    print("🧠 AURIK MODELL-INVENTAR — Status + Bezugsempfehlung")
    print("═" * 100)
    present = sum(1 for v in s.values() if v["present"])
    print(f"\n✅ Vorhanden: {present}/{len(s)}  ❌ Fehlend: {len(missing)}  ⚠️ Platzhalter: {len(placeholders)}\n")
    if args.list:
        for k, v in sorted(missing.items()):
            print(f"  ❌ {k}  ({v['domain']}, {v['license']})")
            print(f"     → Quelle: {v['source']}")
        return 0
    for k, v in sorted(s.items(), key=lambda kv: (not kv[1]["present"], kv[0])):
        mark = "✅" if v["present"] else ("⚠️" if v["placeholder"] else "❌")
        req = " [KERN]" if v["required"] else ""
        print(f"  {mark} {k} ({v['size_mb']} MB){req}")
        print(f"      Zweck: {v['purpose']} | Domäne: {v['domain']} | Lizenz: {v['license']}")
        print(f"      Quelle: {v['source']}")
    print("\n" + summary_line())
    print("\nHinweis: Fehlende Modelle ⇒ Pipeline läuft auf DSP-Fallback (§V6 (copilot-instructions.md), geloggt).")
    print("Für Gesang: resemble_enhance/AnyEnhance (Sprache) NICHT verwenden — KIM-Vocal/CQTdiff+ (Musik) nutzen.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
