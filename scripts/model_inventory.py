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
    python scripts/model_inventory.py --wiring          # Verdrahtungs-Audit (§G9 copilot-instructions.md)
    python scripts/model_inventory.py --fail-on-orphans # Gate: Exit 1 bei verwaistem Artefakt

Design-Prinzip: Die Pipeline läuft NIE mit einem Modell, dessen Datei fehlt —
die Plugins haben Availability-Guards (§V6-DSP-Fallback). Dieses Inventar
macht die dadurch verbleibende Qualitäts-Lücke SICHTBAR (Anzeigen-Wahrheit).

Zwei Achsen (Befund 2026-10-06):
  * ``_MODELS`` — **Vorhandensein** je Pfad (30 kuratierte Produktions-Modelle).
  * ``--wiring`` — **Nutzung** je ``models/``-Verzeichnis: lokal vorhandenes
    Kapital, das von keiner Produktionsstelle geladen wird, ist ein stiller
    Verlust (Produktionsbefund: ``ddsp_predictor/c4_head.pth`` trainiert, aber
    nirgends verdrahtet; ``scnet_4stems`` A/B-gewonnen, aber nirgends verdrahtet).
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
    # §III.13 (copilot-instructions.md) + §G9 (copilot-instructions.md):
    # KORREKTUR 2026-10-06 — lokal AM ARTEFAKT gemessen, nicht aus der Doku
    # übernommen. Die frühere Einstufung "sprache" war eine reine Doku-Übernahme
    # (Registry + Skript-Kopf) und ist durch die Architektur widerlegt:
    #   * bigvgan_v2.onnx/.pth: `conv_pre.weight_v` = [1536, 128, 7] ⇒ 128
    #     Mel-Bänder, 6 Up-Stufen ⇒ 512×, 109×alpha UND 109×beta ⇒ SnakeBeta +
    #     Anti-Alias (`activation_post.downsample.lowpass.filter`), 122,19 M
    #     Parameter ⇒ die 44,1-kHz-Konfiguration (bigvgan_v2_44khz_128band_512x).
    #     Die LibriTTS-Konfiguration derselben Familie ist 24 kHz/100 Bänder/
    #     256×/512 Kanäle/≈14 M ⇒ "LibriTTS/Sprache" ist damit WIDERLEGT.
    #   * hifi_gan.onnx: 80 Mel-Bänder, [8,8,2,2]/[16,16,4,4] = 256×, aber nur
    #     128/64 Kanäle ⇒ 0,93 M Parameter (offizieller V1: 512 Kanäle/13,9 M;
    #     offizieller V2: [4,4,2,2]/[8,8,4,4]) ⇒ kein offizieller Checkpoint;
    #     die Zuordnung "UNIVERSAL_V1" ist WIDERLEGT.
    # Ohne Trainings-Skript-Korpus, Modell-Karte oder SHA-Identität gegen
    # Upstream gilt nach §III.13 "unbekannt" — niemals "Sprache"
    # (Evidenzpflicht; Signalpfad-Risiko bleibt als Defizit bestehen).
    ModelEntry(
        "models/hifi_gan/hifi_gan.onnx",
        "HiFi-GAN Vocoder (Topologie belegt, Domäne unbelegt)",
        "unbekannt",
        "MIT",
        "lokal: models/hifi_gan/hifigan_infer.py (22 050 Hz) — kein Upstream-SHA-Nachweis",
    ),
    ModelEntry(
        "models/bigvgan/bigvgan_v2.onnx",
        "BigVGAN v2 Vocoder (44,1-kHz-Konfiguration, Domäne unbelegt)",
        "unbekannt",
        "MIT",
        "lokal: plugins/bigvgan_v2_plugin.py (bigvgan_v2_44khz_128band_512x) — kein Upstream-SHA-Nachweis",
    ),
    # EAR-VAE (Phase-0 Clean-Pass): deployter Stand = MUSDB-Musik-Finetune
    # (SHA-belegt 2026-10-06): `encoder.onnx`/`decoder.onnx` sind byte-identisch
    # mit `ear_vae_ft_encoder_inline.onnx`/`ear_vae_ft_decoder_inline.onnx` aus
    # `models/ear_vae/export_finetuned_onnx.py`.
    ModelEntry(
        "models/ear_vae/encoder.onnx",
        "EAR-VAE Encoder (musik-finetunt)",
        "musik",
        "intern (Upstream MIT) + MUSDB-Finetune",
        "intern: models/ear_vae/export_finetuned_onnx.py (aus ear_vae_music_finetuned.pyt)",
    ),
    ModelEntry(
        "models/ear_vae/decoder.onnx",
        "EAR-VAE Decoder (musik-finetunt)",
        "musik",
        "intern (Upstream MIT) + MUSDB-Finetune",
        "intern: models/ear_vae/export_finetuned_onnx.py (aus ear_vae_music_finetuned.pyt)",
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


# ── Verdrahtungs-Audit (§G9 copilot-instructions.md): jedes Modellverzeichnis braucht einen Konsumenten ──
# Befund 2026-10-06: `models/` enthielt 61 Verzeichnisse, `_MODELS` pflegte 30
# Einträge — verwaiste Artefakte blieben damit unsichtbar.
_CONSUMER_ROOTS = ("backend", "plugins", "denker", "cli", "Aurik10")

# Bewusst ohne Konsumenten — JEDER Eintrag ist ein Entscheid mit Begründung
# und Aufgabenziel, kein vergessener Rest (§G8 copilot-instructions.md, Transparenz).
_WIRING_ALLOWLIST: dict[str, str] = {
    "applade": (
        "port/ ist ein Zwilling von models/aspade/aspade_declipper.onnx (identischer SHA 8161ed93…) — "
        "das Musik-Finetune ist bereits über A-SPADE in phase_07 PRODUKTIV; nur applade_dnn.onnx (Basis) ist verwaist"
    ),
    "ddsp_predictor": "trainiert (c4_head.pth), aber val_MAE 0.2447 ≈ Baseline 0.2455 (kein Skill) — NICHT aktivierbar (F5/C4, §V7-Blindaktivierungs-Verbot)",
    "gacela_upstream": "Upstream-Architekturkopie als Vergleichsquelle — kein Laufzeitpfad",
    "matchering2.0": "Referenz-Matching-Werkzeug (Mastering-Vergleich) — Rolle ungeklaert",
    "scnet_4stems": "A/B gewonnen 2026-10-05 (SI-SDR +10,3…+16,4 dB) — Verdrahtung offen (TODO-P1-2)",
    # Beschaffung 2026-10-06 (Bestandsabgleich Repo ↔ Sicherung): drei
    # Herkunfts-/Belegbestände ohne Laufzeitpfad — bewusst vorgehalten, damit
    # Basis ↔ Finetune am Artefakt prüfbar ist (§III.13 Evidenzpflicht).
    "ear_vae2_upstream": (
        "EAR-VAE-v2-Upstream-Klon (Phase-0-Clean-Pass) als Herkunfts-/Vergleichsquelle "
        "in-repo; kein Laufzeitpfad — Rollout-Entscheidung offen (Register D-K0-7)"
    ),
    "hubert": (
        "HuBERT-ONNX (+ External Data) als Herkunftsbeleg zum RVC-Backbone "
        "`models/rvc/hubert_base.pt`; kein Produktionskonsument über `resolve_model_path`"
    ),
    "mpsenet": (
        "MP-SENet-Trainingsquelle (Upstream-Repo-Kopie mit `best_ckpt/`) als "
        "Herkunftsbeleg für `models/mp_senet/`; Laufzeit läuft ausschließlich über "
        "`models/mp_senet/` (§G9 (copilot-instructions.md) — eine Quelle)"
    ),
}


# ── Domänen-Registry (kanonisch: .github/ML_MODEL_DOMAIN_REGISTRY.md) ──────
# Evidenz je kuratiertem Modellpfad — NUR aus Trainings-Skript-Korpus, vendored
# Modell-Karte oder SHA-/Export-Beleg, NIE aus Ordner-/Dateinamen
# (§III.13 + §III.11 copilot-instructions.md; §V7 kein Workaround, §G8 Transparenz).
_DOMAIN_EVIDENCE: dict[str, str] = {
    "models/demucs/htdemucs_6s.onnx": "MUSDB-HQ Stem-Separation (Musik)",
    "models/deepfilternet_v3_ii/finetuned/enc.onnx": "train_df_musik.py → dfn_musik_best.pt (Musik-Finetune)",
    "models/deepfilternet_v3_ii/finetuned/dec.onnx": "train_df_musik.py → dfn_musik_best.pt (Musik-Finetune)",
    "models/deepfilternet_v3_ii/finetuned/erb_dec.onnx": "train_df_musik.py → dfn_musik_best.pt (Musik-Finetune)",
    "models/muq_mulan/muq_mulan.onnx": "TencentARC/MuQ-MuLan (Musik-Embedding)",
    "models/whisper/whisper_tiny.onnx": "openai/whisper-tiny (Sprache; nur Wortgrenzen-Zeuge)",
    "models/fcpe/fcpe.onnx": "CNChTu/Diffusion-SVC fcpe.pt (Musik-Pitch)",
    "models/rmvpe/rmvpe.onnx": "yxlllc/RMVPE (Musik-Vokal-Pitch)",
    "models/crepe/crepe.onnx": "marl/crepe full (MedleyDB, Musik-Pitch)",
    "models/kim_vocal_2/kim_vocal_2.onnx": "KimberleyJSN/Kim-Vocal-2 (Musik)",
    "models/kim_inst/kim_inst.onnx": "KimberleyJSN/KIM-INST (Musik)",
    "models/panns/panns_wavegram_logmel_cnn14.onnx": "AudioSet PANNs CNN14 (Audio-allgemein)",
    "models/ast/ast_model.onnx": "AudioSet AST (Audio-allgemein)",
    "models/basicpitch/basicpitch.onnx": "spotify/basic-pitch (Musik)",
    "models/bw_reconstructor/bw_reconstructor.onnx": "train_bw_reconstructor.py → MUSDB18-HQ (Musik)",
    "models/singmos/singmos_pro.onnx": "SingMOS Pro (Gesang/Musik)",
    # GEMESSEN 2026-10-06 (ONNX-Introspection; §III.13-Evidenzpflicht).
    "models/hifi_gan/hifi_gan.onnx": (
        "lokal gemessen: 80 Mel-Bänder, Upsample [8,8,2,2]/Kernel [16,16,4,4] = 256×, "
        "Kanäle 128/64 ⇒ 0,93 M Parameter, 22 050 Hz (hifigan_infer.py) ⇒ HiFi-GAN-"
        "Topologie, aber KEIN offizieller V1-Checkpoint (512 Kanäle/13,9 M) und kein V2 "
        "([4,4,2,2]/[8,8,4,4]); Korpus ohne Karte/Hparams/SHA nicht belegbar ⇒ "
        "§III.13: Domäne unbekannt (nicht 'Sprache')"
    ),
    "models/bigvgan/bigvgan_v2.onnx": (
        "lokal gemessen: 128 Mel-Bänder (conv_pre.weight_v [1536,128,7]), 6 Up-Stufen "
        "⇒ 512×, upsample_initial_channel 1536, SnakeBeta (109×alpha + 109×beta) + "
        "Anti-Alias, 122,19 M Parameter ⇒ BigVGAN v2 in der 44,1-kHz-Konfiguration "
        "(bigvgan_v2_44khz_128band_512x); die LibriTTS-Zuordnung (24 kHz/100 Bänder/"
        "256×/512 Kanäle/≈14 M) ist architektur-widerlegt; Korpus ohne Karte/Hparams/"
        "SHA nicht belegbar ⇒ §III.13: Domäne unbekannt (nicht 'Sprache')"
    ),
    "models/ear_vae/encoder.onnx": (
        "MUSDB18-Musik-Finetune (finetune_music.py) — SHA == ear_vae_ft_encoder_inline.onnx (Musik)"
    ),
    "models/ear_vae/decoder.onnx": (
        "MUSDB18-Musik-Finetune (finetune_music.py) — SHA == ear_vae_ft_decoder_inline.onnx (Musik)"
    ),
    "models/gacela/model/gacela_core.onnx": "TencentARC/GACELA (MAESTRO, Musik)",
    "models/mert/mert.onnx": "m-a-p/MERT-v1 (160k h Musik)",
    "models/mert-v1-330m/pytorch_model.bin": "m-a-p/MERT-v1-330M (Musik)",
    "models/mert-95m/MERT-v1-95M_fairseq.pt": "m-a-p/MERT-v1-95M (Musik)",
    "models/utmos/utmos.onnx": "sarulab-speech/UTMOSv2 (VoiceMOS/Sprache; nur MOS-Zeuge)",
    "models/model_bs_roformer_ep_317_sdr_12.9755.ckpt": "BS-RoFormer 317 (MUSDB, Musik)",
    "models/cqtdiff_plus/score_network.onnx": "sjc0423/CQTdiff (MAESTRO, Musik)",
    "models/dac/encoder_model.onnx": "descript-audio-codec dac_44khz (Musik-inklusiv)",
    "models/dac/decoder_model.onnx": "descript-audio-codec dac_44khz (Musik-inklusiv)",
    "models/artifact_detector.pt": "intern (audio-allgemein) — kein Trainingskorpus belegt",
    "models/muq/muq_eval_a1_head.pt": "MuQ-Eval-A1 (Musik-MOS-Head)",
    "models/resemble_enhance/resample_enhance.pt": "Resemble-Enhance (Sprach-Enhancer; für Gesang NICHT empfohlen)",
}

# Kopfzahlen des GESAMTbestands (61 Verzeichnisse) — kanonisch in der Registry.
_DOMAIN_TOTALS: dict[str, int] = {
    "musik": 37,
    "gemischt": 4,
    "sprache": 11,
    "audio-allgemein": 8,
    "unueberwacht": 1,
}
_DOMAIN_TOTAL_DIRS = 61


def audit_wiring() -> dict[str, str]:
    """Ordnet jedes Verzeichnis unter ``models/`` einem Verdrahtungs-Status zu.

    Status: ``verdrahtet`` | ``bewusst-verwaist: <Begründung>`` | ``verwaist``.

    Ehrlichkeit zur Methode: geprüft wird die **Namensreferenz** im
    Produktionscode (untere Schranke — ein Kommentar-Treffer zählt). Das Gate
    findet damit vollständig unbekannte Artefakte, NICHT falsch verdrahtete.
    """
    models_dir = _ROOT / "models"
    if not models_dir.is_dir():
        return {}

    consumers: list[str] = []
    for sub in _CONSUMER_ROOTS:
        base = _ROOT / sub
        if not base.is_dir():
            continue
        for src in base.rglob("*.py"):
            if "_vendor" in src.parts:
                continue
            consumers.append(src.read_text(encoding="utf-8", errors="ignore"))
    blob = "\n".join(consumers)

    out: dict[str, str] = {}
    for entry in sorted(p for p in models_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
        if re.search(rf"\b{re.escape(entry.name)}\b", blob):
            out[entry.name] = "verdrahtet"
        elif entry.name in _WIRING_ALLOWLIST:
            out[entry.name] = f"bewusst-verwaist: {_WIRING_ALLOWLIST[entry.name]}"
        else:
            out[entry.name] = "verwaist"
    return out


def wiring_allowlist() -> dict[str, str]:
    """Kopie der begründeten Ausnahmen (öffentliche API für Tests/GUI, §G9 copilot-instructions.md)."""
    return dict(_WIRING_ALLOWLIST)


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
            "evidence": _DOMAIN_EVIDENCE.get(
                m.path, "Trainings-Domäne nicht einzeln belegt — siehe .github/ML_MODEL_DOMAIN_REGISTRY.md"
            ),
            "license": m.license_note,
            "source": m.source,
            "required": m.required,
        }
    return out


def domain_report() -> dict:
    """Maschinenlesbare Domänen-Sicht (Musik vs. Sprache) aus der kanonischen Registry.

    Kernaussage (§III.13 copilot-instructions.md): Aurik ist ein
    Musik-Restaurierungssystem — 37 von 61 Modellverzeichnissen sind
    musik-trainiert, nur 11 sprach-trainiert. Der KI-Fehlschluss
    „die meisten Modelle sind Sprache" ist VERBOTEN.
    """
    s = scan()
    curated_counts: dict[str, int] = {}
    for v in s.values():
        curated_counts[v["domain"]] = curated_counts.get(v["domain"], 0) + 1
    return {
        "registry": ".github/ML_MODEL_DOMAIN_REGISTRY.md",
        "instruction": ".github/instructions/ml_domain.instructions.md",
        "normalized_chain": "AGENTS.md §1 (Punkt 4) · copilot-instructions.md §III.13",
        "misconception_forbidden": "Ein Großteil der lokal verfügbaren ML-Modelle seien Sprachmodelle",
        "fact": "37/61 musik-trainiert (≈ 61 %), 11/61 sprach-trainiert (≈ 18 %)",
        "totals_all_dirs": _DOMAIN_TOTAL_DIRS,
        "totals_by_domain": _DOMAIN_TOTALS,
        "curated_counts": curated_counts,
        "models": {k: {"domain": v["domain"], "evidence": v["evidence"]} for k, v in s.items()},
    }


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
    parser.add_argument(
        "--wiring",
        action="store_true",
        help="Verdrahtungs-Audit aller models/-Verzeichnisse (§G9 copilot-instructions.md)",
    )
    parser.add_argument(
        "--fail-on-orphans",
        action="store_true",
        help="Exit 1, wenn ein Modellverzeichnis ohne Konsumenten und ohne Begründung existiert (Gate)",
    )
    parser.add_argument(
        "--domains",
        action="store_true",
        help="Trainings-Domänen (Musik vs. Sprache) — kanonische Registry-Sicht (§III.13 copilot-instructions.md)",
    )
    args = parser.parse_args()

    if args.domains:
        print(json.dumps(domain_report(), ensure_ascii=False, indent=2))
        return 0

    if args.wiring or args.fail_on_orphans:
        wiring = audit_wiring()
        orphans = sorted(k for k, v in wiring.items() if v == "verwaist")
        print("═" * 100)
        print("🔌 MODELL-VERDRAHTUNGS-AUDIT (§G9 (copilot-instructions.md)) — jedes Artefakt braucht einen Konsumenten")
        print("═" * 100)
        print(f"\nVerzeichnisse: {len(wiring)}  verwaist (undokumentiert): {len(orphans)}\n")
        for name, status in sorted(wiring.items()):
            mark = "✅" if status == "verdrahtet" else ("ℹ️" if status.startswith("bewusst") else "❌")
            print(f"  {mark} {name:24s} {status}")
        if orphans:
            print(
                "\nEntscheid nötig (§G8 copilot-instructions.md): verdrahten, entfernen "
                "oder mit Begründung in _WIRING_ALLOWLIST aufnehmen."
            )
        return 1 if (orphans and args.fail_on_orphans) else 0

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
