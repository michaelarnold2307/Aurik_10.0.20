"""
core/hybrid/ — DSP+ML-Brückenmodule
=====================================

Dieses Paket enthält Hybrid-Module, die reine DSP-Algorithmen aus dsp/
mit optionalen ML-Plugins aus plugins/ kombinieren.

Architektur-Begründung:
    - dsp/ = reine DSP, kein ML, keine Plugin-Importe
    - plugins/ = ML-Adapter mit DSP-Fallback
    - core/hybrid/ = Orchestrierung beider Schichten (darf beide importieren)

Module:
    hybrid_ml_denoiser      — OMLSA (DSP) + DeepFilterNet (ML)
    hybrid_dereverb         — DSP-Dereverb + DCCRN (ML)
    hybrid_nvsr             — Bandbreiten-Erweiterung + AudioSR (ML)
    hybrid_speed_pitch_ml   — Zeitdehnung/Pitch + CREPE (ML)
    hybrid_wow_flutter      — Wow/Flutter-Korrektur + CREPE (ML)
    hybrid_vocal_enhancer   — Formant/Atem-Analyse + Plugins
"""
