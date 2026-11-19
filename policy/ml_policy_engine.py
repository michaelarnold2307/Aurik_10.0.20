"""
ML Model Policy Engine - AURIK 6.0
===================================

Intelligente Modellauswahl basierend auf Audio-Kontext.
Wählt optimales SOTA-Modell für jede Restaurierungs-Aufgabe.

Verwendung:
    policy = MLModelPolicyEngine()
    model_name = policy.select_denoise_model(context, goal)
    plugin = getattr(pipeline, model_name)
    plugin.process(input, output)
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


CANONICAL_VOCAL_NR_ROUTE = "sota_vocal_model_router.enhance_vocal"
CANONICAL_INSTRUMENTAL_NR_ROUTE = "sota_vocal_model_router.enhance_instrumental"
CANONICAL_SEPARATION_ROUTE = "sota_vocal_model_router.separate_vocal_instrumental"
CANONICAL_REPAIR_ROUTE = "uv3.phase_repair_chain"
CANONICAL_BW_EXTENSION_ROUTE = "phase_06_frequency_restoration"
CANONICAL_INPAINTING_ROUTE = "phase_55_diffusion_inpainting"
CANONICAL_VOCODER_ROUTE = "vocos"
CANONICAL_TAGGING_ROUTE = "panns"
CANONICAL_PITCH_ROUTE = "fcpe"
CANONICAL_QUALITY_ROUTES = ["versa", "vqi"]


class MLModelPolicyEngine:
    """
    Policy-Engine für automatische SOTA-Modellauswahl.

    Wählt basierend auf Audio-Kontext das optimale ML-Modell:
    - Kontext: detected_medium, genre, has_vocals, noise_type, etc.
    - Goal: quality_level, specific requirements
    """

    def __init__(self):
        """Initialisiert Policy-Engine."""
        self.logger = logging.getLogger("MLModelPolicyEngine")
        self.logger.info("Policy-Engine initialisiert (ML-First-Modus: ML primär, DSP als Fallback)")

    # §ML-FIRST: ML-Modelle sind primär, DSP nur als Fallback.
    # Diese Policy wurde von DSP-First auf ML-First umgestellt (v10.1).
    # Begründung: FlashSR v2, Demucs v5, DFN v4, MDX23C liefern nachweislich
    # bessere Ergebnisse als DSP-Äquivalente und sind GPU-beschleunigt verfügbar.
    _ML_FIRST: bool = True

    def select_denoise_model(self, context: dict[str, Any], goal: dict[str, Any]) -> str:
        """
        Wählt bestes Denoise/Restoration-Modell basierend auf semantischem Audio-Verständnis.

        Args:
            context: Policy-Kontext mit semantischen Feldern:
                - detected_medium, has_vocals, noise_type
                - dominant_instrument, content_character, processing_strategy (🔬 Innovation #3)
                - preserve_transients, enhance_clarity, reduce_harshness
                - has_drums, has_guitar, has_keys, has_ambient
            goal: Restaurierungs-Ziel

        Returns:
            Canonical route name, not a direct legacy plugin name.

        The legacy policy layer no longer names individual obsolete plugins.
        It returns canonical Aurik 10.0.0 routes so execution is centralized in UV3,
        SotaVocalModelRouter, ModelCapabilityGate, and phase-level guards.
        """
        del goal
        if context.get("has_vocals", False) or context.get("dominant_instrument") in {"VOCALS", "SPEECH"}:
            self.logger.info("Vocal NR → SotaVocalModelRouter.enhance_vocal")
            return CANONICAL_VOCAL_NR_ROUTE
        self.logger.info("Instrumental/general NR → SotaVocalModelRouter.enhance_instrumental")
        return CANONICAL_INSTRUMENTAL_NR_ROUTE

    def select_repair_model(self, context: dict[str, Any], goal: dict[str, Any]) -> str:
        """
        Wählt bestes Repair/Declipping-Modell.

        Args:
            context: Audio-Kontext
            goal: Restaurierungs-Ziel

        Returns:
            Plugin-Name: 'mp_senet'

        Entscheidungslogik:
        - Speech/Music → mp_senet (normatives SOTA-Modell, ersetzt DCCRN/FullSubNet+)
        """
        del context
        del goal
        self.logger.info("Repair/Declipping → UV3 phase repair chain")
        return CANONICAL_REPAIR_ROUTE

    def select_stem_separation_model(self, context: dict[str, Any], goal: dict[str, Any]) -> str:
        """
        Wählt bestes Stem Separation-Modell.

        Args:
            context: Audio-Kontext
            goal: Separation-Ziel (vocals, instruments, stems)

        Returns:
            Canonical route name for the central SOTA vocal separation router.

        Entscheidungslogik:
        - Vocals/Instrument Split → MDX23C (SOTA quality)
        - 6-Stem Separation → Demucs v4 (most stems)
        - HQ Mastering → UVR MDX-Net HQ4 (best quality)
        - Speech Separation → Conv-TasNet (speech-optimized)
        """
        del goal
        del context
        self.logger.info("Stem separation → SotaVocalModelRouter.separate_vocal_instrumental")
        return CANONICAL_SEPARATION_ROUTE

    def select_enhancement_model(self, context: dict[str, Any], goal: dict[str, Any]) -> str:
        """
        Wählt bestes Enhancement/Upsampling-Modell.

        Args:
            context: Audio-Kontext
            goal: Enhancement-Ziel

        Returns:
            Canonical route name.

        Entscheidungslogik:
        - Speech → Resemble Enhance (voice clarity)
        - Super-Resolution → FlashSR (16/24 kHz → 48 kHz)
        - Diffusion Enhancement → WPE Dereverberation (Nakatani 2010)
        - General Enhancement → GACELA (audio enhancement)
        """
        enhancement_type = goal.get("enhancement_type", "general")

        # Vocal enhancement is routed through the same vocal-first model router.
        if context.get("has_vocals", False) or enhancement_type == "speech":
            self.logger.info("Vocal enhancement → SotaVocalModelRouter.enhance_vocal")
            return CANONICAL_VOCAL_NR_ROUTE

        # Super-Resolution (Upsampling)
        if enhancement_type == "super_resolution":
            self.logger.info("Super-Resolution → phase_06_frequency_restoration")
            return CANONICAL_BW_EXTENSION_ROUTE

        # Diffusion-based Enhancement
        if enhancement_type == "diffusion":
            self.logger.info("Diffusion/inpainting enhancement → phase_55_diffusion_inpainting")
            return CANONICAL_INPAINTING_ROUTE

        # General Enhancement
        self.logger.info("General enhancement → SotaVocalModelRouter.enhance_instrumental")
        return CANONICAL_INSTRUMENTAL_NR_ROUTE

    def select_quality_assessment_model(self, context: dict[str, Any], goal: dict[str, Any]) -> list[str]:
        """
        Wählt Quality Assessment-Modelle für Musikqualität.

        Args:
            context: Audio-Kontext
            goal: Assessment-Ziel

        Returns:
            List[Plugin-Name]: ['versa', 'utmos', 'visqol', 'peaq']

        Entscheidungslogik (§4.4/§10.2 — Verbotene Metriken berücksichtigen):
        - Immer: VERSA 2024 (primäre MOS ohne Referenz, Music+Speech trainiert)
        - Bei Gesang (has_vocals=True): zusätzlich UTMOS (MOS-Verifikation Gesang)
        - Mit Referenz: +ViSQOL v3 (zwingend --audio Mode, kein Speech-Default)
        - Vollständig: VERSA + UTMOS + ViSQOL + PEAQ (erweiterte Metriken, nur Reporting)

        VERBOTEN (§4.4/§10.2/§11.3):
        - CDPAM: ersetzt durch VERSA 2024 (§4.4, ABSOLUTVERBOTEN als MOS-Primär)
        - DNSMOS: trainiert auf 16 kHz DNS-Challenge-Sprachkorpus, bewertet Musik falsch
        - NISQA: Sprachqualitäts-CNN, keine Musik-Trainingsdaten
        - PESQ: Telefonband 300–3400 Hz, strukturell ungeeignet für Vollband-Musik
        - STOI: Sprachverständlichkeit, sinnlos für Instrumentalmusik
        """
        has_vocals = context.get("has_vocals", False)
        del goal
        models = list(CANONICAL_QUALITY_ROUTES if has_vocals else ["versa"])
        self.logger.info("Quality assessment → %s", models)
        return models

    def select_vocoder_model(self, context: dict[str, Any], goal: dict[str, Any]) -> str:
        """
        Wählt bestes Vocoder/Synthesis-Modell.

        Args:
            context: Audio-Kontext
            goal: Synthesis-Ziel

        Returns:
            Plugin-Name: 'hifigan', 'diffwave'

        Entscheidungslogik:
        - Fast/Real-time → HiFi-GAN (GAN-based, fast)
        - High-Quality → DiffWave (Diffusion-based, slower but better)
        """
        del context
        del goal
        self.logger.info("Vocoding → Vocos primary vocoder")
        return CANONICAL_VOCODER_ROUTE

    def select_audio_tagging_model(self, context: dict[str, Any], goal: dict[str, Any]) -> str:
        """
        Wählt Audio Tagging/Classification-Modell.

        Args:
            context: Audio-Kontext
            goal: Tagging-Ziel

        Returns:
            Plugin-Name: 'panns' (527 AudioSet classes)
        """
        del context
        del goal
        self.logger.info("Audio Tagging → PANNS")
        return CANONICAL_TAGGING_ROUTE

    def select_mastering_model(self, context: dict[str, Any], goal: dict[str, Any]) -> str:
        """
        Wählt Automated Mastering-Modell.

        Args:
            context: Audio-Kontext
            goal: Mastering-Ziel (with reference track)

        Returns:
            Plugin-Name: 'matchering'
        """
        del context
        del goal
        self.logger.info("Mastering policy → UV3 phase planning, no standalone Matchering route")
        return "uv3.phase_plan"

    def select_generative_model(self, context: dict[str, Any], goal: dict[str, Any]) -> str:
        """
        Wählt Generative Audio-Modell.

        Args:
            context: Audio-Kontext
            goal: Generation-Ziel

        Returns:
            Plugin-Name: 'audioldm2', 'flow_matching'

        Entscheidungslogik:
        - Text-to-Audio → AudioLDM2 (text prompt)
        - Music Generation → flow_matching (generative inpainting, SOTA 2024)
        """
        del context
        generation_type = goal.get("generation_type", "text_to_audio")

        if generation_type == "music":
            self.logger.info("Music inpainting → phase_55_diffusion_inpainting")
            return CANONICAL_INPAINTING_ROUTE
        self.logger.info("Text-to-audio is outside one-button restoration; route disabled")
        return "unsupported.text_to_audio"

    def select_pitch_detection_model(self, context: dict[str, Any], goal: dict[str, Any]) -> str:
        """
        Wählt Pitch Detection-Modell.

        Args:
            context: Audio-Kontext
            goal: Pitch detection-Ziel

        Returns:
            Plugin-Name: 'fcpe' (monophonic pitch tracking, primary)
        """
        del context
        del goal
        self.logger.info("Pitch Detection → FCPE")
        return CANONICAL_PITCH_ROUTE

    def select_medium_specific_model(self, context: dict[str, Any], goal: dict[str, Any]) -> str:
        """
        Wählt Medium-spezifisches Restoration-Modell.

        Args:
            context: Audio-Kontext
            goal: Medium-specific restoration

        Returns:
            Plugin-Name: 'banquet' (vinyl-specialized)
        """
        detected_medium = context.get("detected_medium", "unknown")
        self.logger.info("Medium-specific routing for %s stays inside UV3/phase planning", detected_medium)
        return self.select_denoise_model(context, goal)

    def select_all_models(self, context: dict[str, Any], tasks: list[str]) -> dict[str, Any]:
        """
        Intelligente Auswahl aller benötigten Modelle basierend auf Tasks.

        Args:
            context: Audio-Kontext
            tasks: Liste von Tasks ['denoise', 'separation', 'enhancement', 'quality']

        Returns:
            Dict mit gewählten Modellen pro Task
        """
        selected_models: dict[str, Any] = {}

        for task in tasks:
            if task == "denoise":
                selected_models["denoise"] = self.select_denoise_model(context, {})
            elif task == "repair":
                selected_models["repair"] = self.select_repair_model(context, {})
            elif task == "separation":
                selected_models["separation"] = self.select_stem_separation_model(context, {})
            elif task == "enhancement":
                selected_models["enhancement"] = self.select_enhancement_model(context, {})
            elif task == "quality":
                selected_models["quality"] = self.select_quality_assessment_model(context, {})
            elif task == "vocoder":
                selected_models["vocoder"] = self.select_vocoder_model(context, {})
            elif task == "tagging":
                selected_models["tagging"] = self.select_audio_tagging_model(context, {})
            elif task == "mastering":
                selected_models["mastering"] = self.select_mastering_model(context, {})
            elif task == "generation":
                selected_models["generation"] = self.select_generative_model(context, {})
            elif task == "pitch":
                selected_models["pitch"] = self.select_pitch_detection_model(context, {})
            elif task == "medium_specific":
                selected_models["medium_specific"] = self.select_medium_specific_model(context, {})

        self.logger.info("Selected models for tasks %s: %s", tasks, selected_models)
        return selected_models

    def select_separation_model(self, context: dict[str, Any], goal: dict[str, Any]) -> str:
        """
        Wählt bestes Source-Separation-Modell.

        Args:
            context: Audio-Kontext
            goal: Restaurierungs-Ziel

        Returns:
            Plugin-Name: 'demucs', 'mdx23c', 'uvr_mdxnet'

        Entscheidungslogik:
        - Maximal Quality → mdx23c (SOTA)
        - 4+ Stems needed → demucs (6-Stem: vocals, drums, bass, other, piano, guitar)
        - Fast 2-Stem → uvr_mdxnet (HQ4 für Qualität, HQ1 für Speed)
        """
        # Priorität 1: Maximal Quality explizit gewünscht
        del context
        del goal
        self.logger.info("Source separation → SotaVocalModelRouter.separate_vocal_instrumental")
        return CANONICAL_SEPARATION_ROUTE

    def select_enhancement_model_alt(self, context: dict[str, Any], goal: dict[str, Any]) -> str:
        """
        Wählt bestes allgemeines Enhancement-Modell.

        Args:
            context: Audio-Kontext
            goal: Restaurierungs-Ziel

        Returns:
            Plugin-Name für allgemeines Enhancement

        Entscheidungslogik:
        - Speech → resemble_enhance (SOTA Speech Enhancement)
        - Classical → wpe (Dereverberation)
        - General → waveunet oder gacela
        """
        return self.select_enhancement_model(context, goal)

    def select_super_resolution_model(self, context: dict[str, Any], goal: dict[str, Any]) -> str:
        """
        Wählt bestes Super-Resolution-Modell.

        Returns:
            'flashsr' - Aktuell nur ein SOTA-Modell verfügbar
        """
        del context
        del goal
        self.logger.info("Super-Resolution → phase_06_frequency_restoration")
        return CANONICAL_BW_EXTENSION_ROUTE

    def select_quality_metrics(self, context: dict[str, Any], goal: dict[str, Any]) -> list:
        """
        Wählt relevante Quality-Metriken.

        Args:
            context: Audio-Kontext
            goal: Restaurierungs-Ziel

        Returns:
            Liste von Metrik-Plugin-Namen

        Metrik-Auswahl (§4.4 — nur musik-geeignete Metriken):
        - Standard: VERSA 2024 (primäre MOS, kein Referenzsignal nötig)
        - Gesang: + UTMOS (Gesangs-MOS-Verifikation)
        - Mit Referenz: + ViSQOL v3 (--audio Mode)
        - Maximal Quality: VERSA + UTMOS + ViSQOL + PEAQ

        VERBOTEN (§4.4): CDPAM, DNSMOS, NISQA, PESQ, STOI (kein Musiktraining)
        """
        has_vocals = context.get("has_vocals", False)
        has_reference = goal.get("has_reference", False)
        metrics = ["versa"]  # VERSA 2024 ist immer Basis (§4.4)

        if has_vocals:
            metrics.append("utmos")  # UTMOS MOS-Verifikation Gesang

        if has_reference:
            metrics.append("visqol")  # ViSQOL v3 --audio Mode

        if goal.get("quality_level") == "maximal":
            if "peaq" not in metrics:
                metrics.append("peaq")
            self.logger.info("Maximal Quality → VERSA + UTMOS + ViSQOL + PEAQ")
        elif has_vocals:
            self.logger.info("Gesang → VERSA + UTMOS")
        else:
            self.logger.info("Standard → VERSA")

        return list(dict.fromkeys(metrics))  # Deduplizieren, Reihenfolge erhalten

    def should_use_diffusion_models(self, context: dict[str, Any], goal: dict[str, Any]) -> bool:
        """
        Entscheidet ob Diffusion-basierte Modelle genutzt werden sollen.

        Diffusion-Modelle (WPE, DiffWave, FlashSR, AudioLDM2):
        - Sehr hohe Qualität
        - Langsamer (600s Timeout)
        - Besonders gut für Musik/Classical

        Returns:
            True wenn Diffusion-Modelle empfohlen werden
        """
        # Maximal quality explizit gewünscht
        if goal.get("quality_level") == "maximal":
            return True

        # Classical/Jazz/Acoustic genres profitieren besonders
        if context.get("genre") in ["classical", "jazz", "acoustic"]:
            return True

        # High-End Context (hohe Sample Rate, low noise floor)
        if context.get("sample_rate", 0) >= 48000:
            return True

        return False


# ===== CONVENIENCE FUNCTIONS =====


def get_recommended_models(context: dict[str, Any]) -> dict[str, Any]:
    """
    Quick helper: Get recommended models for common restoration workflow.

    Args:
        context: Audio context dictionary

    Returns:
        Dict with recommended models:
        - denoise: Best denoise model
        - quality: List of quality assessment models
        - separation: Best stem separation model (if needed)
    """
    policy = MLModelPolicyEngine()

    recommendations = {
        "denoise": policy.select_denoise_model(context, {}),
        "quality": policy.select_quality_assessment_model(context, {"has_reference": False}),
    }

    # Add separation if multi-track content detected
    if context.get("has_vocals", False):
        recommendations["separation"] = policy.select_stem_separation_model(context, {"num_stems": 2})

    return recommendations


# Convenience-Funktionen für direkte Nutzung
def get_optimal_denoise_plugin(context: dict[str, Any], goal: dict[str, Any]) -> str:
    """Shortcut: Gibt bestes Denoise-Plugin zurück."""
    policy_engine = MLModelPolicyEngine()
    return policy_engine.select_denoise_model(context, goal)


def get_optimal_separation_plugin(context: dict[str, Any], goal: dict[str, Any]) -> str:
    """Shortcut: Gibt bestes Separation-Plugin zurück."""
    policy_engine = MLModelPolicyEngine()
    return policy_engine.select_stem_separation_model(context, goal)


def get_optimal_repair_plugin(context: dict[str, Any], goal: dict[str, Any]) -> str:
    """Shortcut: Gibt bestes Repair-Plugin zurück."""
    policy_engine = MLModelPolicyEngine()
    return policy_engine.select_repair_model(context, goal)


if __name__ == "__main__":
    # Test-Beispiele
    logging.basicConfig(level=logging.INFO)

    engine = MLModelPolicyEngine()

    # Test 1: Speech Restoration
    context1: dict[str, Any] = {"has_vocals": True, "noise_type": "broadband", "genre": "speech"}
    goal1 = {"quality_level": "high"}
    print(f"Speech Restoration: {engine.select_denoise_model(context1, goal1)}")

    # Test 2: Vinyl Restoration
    context2: dict[str, Any] = {"detected_medium": "vinyl", "genre": "jazz"}
    goal2 = {"quality_level": "maximal"}
    print(f"Vinyl Restoration: {engine.select_denoise_model(context2, goal2)}")

    # Test 3: Classical Music Enhancement
    context3: dict[str, Any] = {"genre": "classical", "has_vocals": False}
    goal3 = {"quality_level": "maximal"}
    print(f"Classical Enhancement: {engine.select_denoise_model(context3, goal3)}")

    # Test 4: Source Separation
    context4: dict[str, Any] = {"stem_count": 4}
    goal4 = {"quality_level": "maximal"}
    print(f"Source Separation: {engine.select_stem_separation_model(context4, goal4)}")
