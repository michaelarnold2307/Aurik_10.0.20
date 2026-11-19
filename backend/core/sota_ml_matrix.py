"""SOTA-ML-Entscheidungsmatrix (§Spec 04 §4.4).

Dokumentiert welche ML-Modelle in welchen Aurik-Phasen zum Einsatz kommen,
mit Alternativen und Entscheidungsbegründung.

Autor: Aurik 10 — 11. Juli 2026
"""

SOTA_ML_MATRIX = {
    "denoising": {
        "aurik_model": "IMCRA + OMLSA (Cohen 2002/2003)",
        "phase": "phase_03_denoise",
        "alternatives": ["DeepFilterNet3 (Schröter 2022)", "Demucs HT (Défossez 2021)", "RNNoise (Valin 2017)"],
        "decision": "IMCRA/OMLSA gewählt wegen deterministischer Reproduzierbarkeit, "
        "kein Training-Bias, beweisbare Konvergenz. DeepFilterNet3 ist "
        "installiert aber nur als optionaler Plugin-Pfad aktiv.",
        "performance": "<1× Echtzeit (DSP), ~2.5× (ML-Hybrid)",
    },
    "crackle_removal": {
        "aurik_model": "BANQUET ONNX (Vinyl-Restoration)",
        "phase": "phase_09_crackle_removal",
        "alternatives": ["iZotope RX De-crackle", "SpectralDecrackler (DSP)"],
        "decision": "ML-Hybrid: BANQUET für Vinyl, DSP-Fallback für andere Materialien. "
        "ONNX-Direktinferenz ohne Docker-Overhead.",
        "performance": "~2.5× Echtzeit (ONNX), <1× (DSP-Fallback)",
    },
    "vocal_enhancement": {
        "aurik_model": "DSP-3-Stufen-Korrektiv (Spectral-Tilt + HNR-Blend + Formant-Korrektur)",
        "phase": "phase_65_vocal_naturalness_restoration",
        "alternatives": ["iZotope RX Voice De-noise", "Adobe Podcast Enhance"],
        "decision": "DSP-only wegen §0a-Invariante (kein Energiegewinn über Input). "
        "PANNS-Singing-Gate aktiviert nur bei Vocal-Präsenz.",
        "performance": "<0.5× Echtzeit",
    },
    "vocal_quality": {
        "aurik_model": "VQI (VocalQualityIndex) + VocalQualityGate (6-dim Delta-Rollback)",
        "phase": "PhaseInterface._safe_process",
        "alternatives": ["DNSMOS (Microsoft)", "MOSNet (Lo 2019)"],
        "decision": "DNSMOS/MOSNet sind auf Telefonsprache trainiert — für Musik ungeeignet (§V14). "
        "Aurik VQI ist multi-dimensional und material-adaptiv.",
        "performance": "<1 ms pro Phase",
    },
    "genre_classification": {
        "aurik_model": "LAION-CLAP (HTSAT-base, music_audioset)",
        "phase": "Pre-Analysis (backend.core.genre_classifier)",
        "alternatives": ["PANNs (CNN14)", "CLAP (LAION-Audio)"],
        "decision": "LAION-CLAP gewählt wegen Music-Audioset-Fine-Tuning. PANNs dient als Genre-Tagging-Fallback.",
        "performance": "~2s pro 30s Audio",
    },
    "era_classification": {
        "aurik_model": "LAION-CLAP → Decade-Boundary-Softener (DSP-Korrektur)",
        "phase": "Pre-Analysis (backend.core.era_classifier)",
        "alternatives": ["None (kein vergleichbares Open-Source-Tool)"],
        "decision": "3-Tier-Kaskade: Tier-1 CLAP → Tier-2 DSP-Fingerprint → Tier-3 Mikrofon-Heuristik. "
        "Decade-Boundary-Softener korrigiert CLAP-Bias mit physikalischer Bandbreite.",
        "performance": "~15s pro 30s Audio (inkl. LAION-CLAP-Laden)",
    },
    "speaker_identity": {
        "aurik_model": "SpeakerEmbeddingGuard (72-dim Multi-Window MFCC + CMVN)",
        "phase": "PhaseInterface._safe_process (Vokal-Phasen)",
        "alternatives": ["ECAPA-TDNN (Desplanques 2020)", "RawNet3 (Jung 2023)"],
        "decision": "Reines NumPy/C — keine externen ML-Frameworks als Pflicht. "
        "CMVN für Kanal-Robustheit, 3 Fensterlängen für Stimmregister.",
        "performance": "<50 ms pro Prüfung",
    },
}

SUMMARY = """
SOTA-ML-Entscheidungsmatrix für Aurik 10.

Design-Prinzipien:
  • Deterministische Reproduzierbarkeit hat Vorrang vor ML-Genauigkeit
  • DSP-Fallback für jeden ML-Pfad (Graceful Degradation)
  • Keine Speech-Metriken für Musikqualität (§V14)
  • ONNX-Direktinferenz wo möglich (kein Docker, kein Cloud-Roundtrip)
  • ML-Budget-Guard verhindert OOM durch kumulative Modell-Ladung
"""
