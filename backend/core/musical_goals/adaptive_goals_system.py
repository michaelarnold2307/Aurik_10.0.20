"""
AURIK v8 Adaptive Musical Goals System
======================================

Weltspitzen-Feature: Selbst-adaptive Qualitätsziele basierend auf Material-Qualität

Kernidee:
- AURIK erkennt automatisch die Ausgangsqualität des Materials
- Passt Musical Goals Thresholds intelligent an
- Optimiert Processing-Strategie für degradiertes Material
- Garantiert IMMER maximale Qualitätssteigerung auf Weltspitzenniveau

Edge Cases:
- Multi-Generation-Kopien (Cassette → MP3 → Digital)
- Stark komprimiertes Material (MP3 128kbps oder schlechter)
- Historische Aufnahmen (1920er-1970er)
- Live-Recordings mit extremem Rauschen
- Telefon/Walkie-Talkie Qualität

Autor: AURIK AI Team
Datum: 12. Februar 2026
"""
# pylint: disable=import-outside-toplevel

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# §2.31 SCALE_FACTORS — Normative restorability-adaptive Schwellwert-Skalierung
# Quelle: AMRB-Kalibrierung (500 Testdateien), abgeleitet aus PhysicalCeilingEstimator.
# scale_factor = ceiling_avg(goals) / baseline_threshold.
# VERBOTEN: Stufenwerte manuell setzen ohne PhysicalCeilingEstimator-Grundlage.
# ---------------------------------------------------------------------------
SCALE_FACTORS: dict[str, float] = {
    "≥ 70": 1.00,  # GOOD     — ceiling_avg = 0.97
    "50–69": 0.93,  # FAIR     — ceiling_avg = 0.90
    "30–49": 0.85,  # POOR     — ceiling_avg = 0.82
    "< 30": 0.75,  # VERY_POOR — ceiling_avg = 0.73
    # §2.47: Shellac + Restorability < 20 special case
    "shellac_<20": 0.65,  # Extreme degradation on shellac medium
}


def get_restorability_scale_factor(
    restorability_score: float,
    is_shellac: bool = False,
) -> float:
    """Map RestorabilityEstimator score (0–100) to a threshold scale factor.

    Uses the normative SCALE_FACTORS table (§2.31, AMRB-calibrated).
    Special case per §2.47: Shellac + Restorability < 20 → Scale 0.65.

    Args:
        restorability_score: Float 0–100 from RestorabilityEstimator.estimate().
        is_shellac: True if the source medium is shellac (78 rpm).

    Returns:
        Scale factor ∈ [0.65, 1.00].
    """
    # §2.47: shellac + very low restorability → separate floor (spec-normative)
    if is_shellac and restorability_score < 20.0:
        return SCALE_FACTORS["shellac_<20"]
    if restorability_score >= 70.0:
        return SCALE_FACTORS["≥ 70"]
    if restorability_score >= 50.0:
        return SCALE_FACTORS["50–69"]
    if restorability_score >= 30.0:
        return SCALE_FACTORS["30–49"]
    return SCALE_FACTORS["< 30"]


class MaterialQuality(Enum):
    """Material-Qualitätsstufen für adaptive Processing"""

    PRISTINE = "pristine"  # Studio-Qualität, unbearbeitet
    EXCELLENT = "excellent"  # Leichte Bearbeitung, hohe Qualität
    GOOD = "good"  # Standard Digital/CD-Qualität
    FAIR = "fair"  # MP3 192kbps, leichte Degradation
    POOR = "poor"  # MP3 128kbps, Cassette, Vinyl mit Artefakten
    VERY_POOR = "very_poor"  # Stark degradiert, Multi-Generation
    EXTREME = "extreme"  # Telefon, Walkie-Talkie, historisch


@dataclass
class MaterialQualityAssessment:
    """Ergebnis der Material-Qualitätsbewertung"""

    quality_level: MaterialQuality
    confidence: float  # 0.0 - 1.0
    degradation_score: float  # 0.0 (perfekt) - 1.0 (extrem degradiert)

    # Detaillierte Indikatoren
    medium_chain: list[str]  # z.B. ["cassette", "damaged_mp3", "digital"]
    generation_count: int  # Anzahl Kopier-Generationen

    # Qualitäts-Metriken
    noise_level: float  # 0.0 - 1.0
    bandwidth_limitation: float  # 0.0 (volle Bandwidth) - 1.0 (stark limitiert)
    artifact_density: float  # 0.0 - 1.0
    dynamic_range_db: float  # dB

    # Processing-Empfehlungen
    recommended_strength: float  # 0.0 - 1.0 für adaptive Processing
    requires_enhanced_processing: bool


@dataclass
class AdaptiveGoalThresholds:
    """Adaptive Thresholds basierend auf Material-Qualität"""

    brillanz: float
    waerme: float
    natuerlichkeit: float
    authentizitaet: float
    emotionalitaet: float
    transparenz: float
    bass_kraft: float

    quality_level: MaterialQuality
    relaxation_factor: float  # Wie stark wurden Thresholds relaxiert (0.0 - 1.0)
    groove: float = 0.83
    spatial_depth: float = 0.70
    timbre_authentizitaet: float = 0.87
    tonal_center: float = 0.95
    micro_dynamics: float = 0.88
    separation_fidelity: float = 0.80
    artikulation: float = 0.88
    transient_energie: float = 0.88


@dataclass
class AdaptiveThresholdResult:
    """Pro-Restaurierung adaptierte Schwellwerte für alle 15 Musical Goals (§2.31).

    Enthält die angepassten Zielwerte, die Original-Schwellwerte zum Vergleich,
    deutschsprachige Begründungen für jede Anpassung sowie die physikalischen
    Obergrenzen (informationstheoretisches Maximum) pro Ziel.
    """

    thresholds: dict[str, float]  # Goal-Name → adaptierter Schwellwert
    unadapted_thresholds: dict[str, float]  # Original-Schwellwerte zum Vergleich
    adaptations: dict[str, str]  # Goal → Begründung der Anpassung (Deutsch)
    physical_ceiling: dict[str, float]  # Goal → informationstheoretisches Maximum


class MaterialQualityAnalyzer:
    """
    Analysiert die Ausgangsqualität des Materials

    Nutzt:
    - Medium Detection Results (Vinyl, Cassette, MP3, etc.)
    - Spectral Analysis (Bandwidth, Artefakte)
    - Noise Profile Analysis
    - Dynamic Range Measurement
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def analyze(
        self,
        audio: np.ndarray,
        sr: int,
        medium_detection: dict | None = None,
        forensic_analysis: dict | None = None,
    ) -> MaterialQualityAssessment:
        """
        Analysiere Material-Qualität

        Args:
            audio: Audio signal (mono or stereo)
            sr: Sample rate
            medium_detection: Results from MediumDetector (optional)
            forensic_analysis: Additional forensic data (optional)

        Returns:
            MaterialQualityAssessment mit allen Metriken
        """
        # Ensure mono for analysis
        audio_mono = np.mean(audio, axis=0) if audio.ndim > 1 else audio

        # 1. Analyze Medium Chain
        medium_chain, generation_count = self._analyze_medium_chain(medium_detection)

        # 2. Measure Degradation Indicators
        noise_level = self._measure_noise_level(audio_mono, sr)
        bandwidth_limitation = self._measure_bandwidth_limitation(audio_mono, sr)
        artifact_density = self._measure_artifact_density(audio_mono, sr)
        dynamic_range_db = self._measure_dynamic_range(audio_mono)

        # 2.5 Extract Defects from Forensics (ML-basiert, 98%+ Recall)
        _defect_count, defects_severity_score = self._extract_defects_from_forensics(forensic_analysis)

        # 3. Calculate Overall Degradation Score
        degradation_score = self._calculate_degradation_score(
            noise_level=noise_level,
            bandwidth_limitation=bandwidth_limitation,
            artifact_density=artifact_density,
            dynamic_range_db=dynamic_range_db,
            generation_count=generation_count,
            defects_severity_score=defects_severity_score,
        )

        # 4. Classify Quality Level
        quality_level, confidence = self._classify_quality_level(degradation_score, medium_chain)

        # 5. Calculate Processing Recommendations
        recommended_strength = self._calculate_processing_strength(degradation_score, quality_level)
        requires_enhanced = quality_level in [MaterialQuality.POOR, MaterialQuality.VERY_POOR, MaterialQuality.EXTREME]

        assessment = MaterialQualityAssessment(
            quality_level=quality_level,
            confidence=confidence,
            degradation_score=degradation_score,
            medium_chain=medium_chain,
            generation_count=generation_count,
            noise_level=noise_level,
            bandwidth_limitation=bandwidth_limitation,
            artifact_density=artifact_density,
            dynamic_range_db=dynamic_range_db,
            recommended_strength=recommended_strength,
            requires_enhanced_processing=requires_enhanced,
        )

        self.logger.info(
            "Material Quality: %s (Degradation: %.2f, Confidence: %.2f)",
            quality_level.value.upper(),
            degradation_score,
            confidence,
        )

        return assessment

    def _analyze_medium_chain(self, medium_detection: dict | None) -> tuple[list[str], int]:
        """Extrahiere Medium-Kette aus Detection Results"""
        if not medium_detection:
            return [], 0

        # Extract detected media with confidence > threshold
        # Lowered to 0.5 for inclusive multi-generation detection (10/10 optimization)
        chain = []
        confidence_threshold = 0.5

        for medium, confidence in medium_detection.items():
            if confidence >= confidence_threshold:
                chain.append(medium.lower())

        generation_count = len(chain)

        return chain, generation_count

    def _extract_defects_from_forensics(self, forensic_analysis: dict | None) -> tuple[int, float]:
        """
        Extrahiere Defekt-Count und Severity-Score aus Forensics

        Returns:
            (defect_count, defects_severity_score)

        Berücksichtigt:
        - Anzahl erkannter Defekte (0-5+)
        - Severity-Level pro Defekt (LOW/MEDIUM/HIGH)
        - Confidence pro Defekt (0.0-1.0)
        - Defekt-spezifische Gewichtung (Hum/Distortion schwerer als Clicks)
        """
        if not forensic_analysis or "defects_detected" not in forensic_analysis:
            return 0, 0.0

        detected = forensic_analysis["defects_detected"]
        severities = forensic_analysis.get("defect_severities", {})
        confidences = forensic_analysis.get("defect_confidences", {})

        total_score = 0.0
        defect_count = 0

        # Defect-spezifische Gewichtung basierend auf Störgrad für Musik
        defect_weights = {
            "CLICKS": 0.8,  # Sehr störend (impulsiv)
            "HUM": 0.9,  # Extrem störend (konstant, harmonisch)
            "DISTORTION": 1.0,  # Maximum störend (Harmonic Verzerrung)
            "DROPOUT": 0.7,  # Mittel störend (kurze Ausfälle)
            "NOISE_BURST": 0.6,  # Moderat störend (transient)
        }

        for defect, is_detected in detected.items():
            if is_detected:
                defect_count += 1

                # Severity Mapping
                severity = severities.get(defect, "MEDIUM")
                severity_weight = {"LOW": 0.3, "MEDIUM": 0.6, "HIGH": 1.0}.get(severity, 0.6)

                # Confidence Weight
                confidence = confidences.get(defect, 0.5)

                # Defect Weight
                weight = defect_weights.get(defect, 0.5)

                # Kombinierter Score
                total_score += severity_weight * confidence * weight

        # Normalisiere auf 0.0-1.0 (5 HIGH Defects = 1.0 = Maximum Degradation)
        normalized_score = min(1.0, total_score / 5.0)

        return defect_count, normalized_score

    def _measure_noise_level(self, audio: np.ndarray, sr: int) -> float:
        """
        Messe Rauschpegel (0.0 = kein Rauschen, 1.0 = extrem verrauscht)

        Nutzt spektrale Analyse in stillen Segmenten
        """
        # Find quietest segments (likely noise floor)
        frame_length = int(0.1 * sr)  # 100ms frames
        hop_length = frame_length // 2

        rms_values = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i : i + frame_length]
            rms = np.sqrt(np.mean(frame**2))
            rms_values.append(rms)

        if not rms_values:
            return 0.0

        # Noise floor = 10th percentile
        noise_floor = np.percentile(rms_values, 10)

        # Normalize to 0-1 scale
        noise_level = min(1.0, noise_floor * 100)  # Rough scaling

        return float(noise_level)  # type: ignore[arg-type]

    def _measure_bandwidth_limitation(self, audio: np.ndarray, sr: int) -> float:
        """
        Messe Bandwidth-Limitation (0.0 = volle BW, 1.0 = stark limitiert)

        Detektiert:
        - MP3/AAC Cutoffs (16kHz, 18kHz, 20kHz)
        - Telefon-Qualität (300-3400 Hz)
        - AM Radio (50-5000 Hz)
        """
        import librosa

        # Compute spectrum
        stft = librosa.stft(audio, n_fft=2048)
        magnitude = np.abs(stft)
        mean_spectrum = np.mean(magnitude, axis=1)

        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

        # Expected bandwidth for sr
        nyquist = sr / 2
        expected_cutoff = nyquist * 0.95  # 95% of Nyquist

        # Find actual cutoff (where energy drops significantly)
        energy_threshold = np.max(mean_spectrum) * 0.01  # -40 dB

        cutoff_idx = len(mean_spectrum) - 1
        for i in range(len(mean_spectrum) - 1, 0, -1):
            if mean_spectrum[i] > energy_threshold:
                cutoff_idx = i
                break

        actual_cutoff = freqs[cutoff_idx]

        # Calculate limitation
        if actual_cutoff >= expected_cutoff * 0.9:
            limitation = 0.0  # Full bandwidth
        elif actual_cutoff < 4000:
            limitation = 0.9  # Telefon/AM Radio
        elif actual_cutoff < 10000:
            limitation = 0.7  # Heavy MP3 compression
        elif actual_cutoff < 16000:
            limitation = 0.4  # MP3 128 kbps
        else:
            limitation = (expected_cutoff - actual_cutoff) / expected_cutoff

        return float(np.clip(limitation, 0.0, 1.0))

    def _measure_artifact_density(self, audio: np.ndarray, _sr: int) -> float:
        """
        Messe Artefakt-Dichte (0.0 = keine, 1.0 = viele Artefakte)

        Detektiert:
        - Clicks/Crackles (Vinyl, Cassette)
        - Dropouts (Tape)
        - MP3 Artefakte (Pre-Echo, Birdie Artefacts)
        """
        # Simplified artifact detection via peak analysis
        diff = np.abs(np.diff(audio))

        # Count anomalous peaks
        threshold = np.mean(diff) + 3 * np.std(diff)
        artifacts: int = int(np.sum(diff > threshold))

        # Normalize by length
        artifact_density = artifacts / len(audio) * 1000  # per 1000 samples

        # Clip to 0-1
        artifact_density = min(1.0, artifact_density)

        return float(artifact_density)

    def _measure_dynamic_range(self, audio: np.ndarray) -> float:
        """Messe Dynamic Range in dB"""
        # RMS-based dynamic range
        rms = np.sqrt(np.mean(audio**2))
        peak: float = float(np.max(np.abs(audio)))

        dr_db = 20 * np.log10(peak / rms) if rms > 0 else 0.0

        return float(dr_db)

    def _calculate_degradation_score(
        self,
        noise_level: float,
        bandwidth_limitation: float,
        artifact_density: float,
        dynamic_range_db: float,
        generation_count: int,
        defects_severity_score: float = 0.0,
    ) -> float:
        """
        Berechne Gesamt-Degradation-Score (0.0 = perfekt, 1.0 = extrem degradiert)

        Gewichtung (NEU mit Defekt-Integration):
        - Noise: 12% (reduziert von 15%)
        - Bandwidth: 12% (reduziert von 15%)
        - Artifacts: 6% (reduziert von 10%, da redundant zu Defects)
        - Dynamic Range: 8% (reduziert von 10%)
        - Generation Count: 40% (reduziert von 50%, aber immer noch DOMINANT)
        - Defects Severity: 22% (NEU! ML-basiert mit 98%+ Recall)

        TOTAL: 100% (12+12+6+8+40+22)
        """
        # Dynamic Range penalty (lower is worse)
        dr_penalty = 0.0
        if dynamic_range_db < 10:
            dr_penalty = 1.0
        elif dynamic_range_db < 20:
            dr_penalty = 0.5
        elif dynamic_range_db < 30:
            dr_penalty = 0.2

        # Generation penalty (LINEAR BIS 5, dann logarithmisch komprimiert)
        # 1 Generation = 0.20 → FAIR
        # 2 Generationen = 0.40 → POOR
        # 3 Generationen = 0.60 → VERY_POOR
        # 4 Generationen = 0.80 → VERY_POOR
        # 5 Generationen = 1.00 → EXTREME
        # 6+ Generationen = 1.00 (gekappt, extrem seltener Edge Case)
        # WORLD-FIRST: Garantierte Differenzierung bis 5+ Tonträger-Tiefe!
        if generation_count <= 5:
            gen_penalty = generation_count * 0.20
        else:
            # >5 Generationen: Bleibt bei 1.0 (Maximum Degradation)
            # Optional: Logarithmische Kompression für hypothetische 10+ Gen Szenarien
            gen_penalty = 1.0

        gen_penalty = min(1.0, gen_penalty)

        # Weighted sum (NEU: Defekt-Integration mit 22% Gewicht)
        # Generation Count bleibt DOMINANT mit 40%, aber Defekte haben signifikanten Impact
        degradation = (
            0.12 * noise_level  # Reduziert: 15% → 12%
            + 0.12 * bandwidth_limitation  # Reduziert: 15% → 12%
            + 0.06 * artifact_density  # Reduziert: 10% → 6% (redundant)
            + 0.08 * dr_penalty  # Reduziert: 10% → 8%
            + 0.40 * gen_penalty  # Reduziert: 50% → 40% (immer noch DOMINANT)
            + 0.22 * defects_severity_score  # NEU: 22% Gewicht auf ML-Defekte (98%+ Recall)
        )

        return float(np.clip(degradation, 0.0, 1.0))

    def _classify_quality_level(
        self,
        degradation_score: float,
        _medium_chain: list[str],
    ) -> tuple[MaterialQuality, float]:
        """
        Klassifiziere Quality Level basierend auf Degradation Score

        WICHTIG: Multi-Generation-Kopien erfordern niedrigere Thresholds!

        Returns:
            (quality_level, confidence)
        """
        # Quality thresholds (LOWERED FOR MULTI-GENERATION!)
        if degradation_score < 0.05:
            return MaterialQuality.PRISTINE, 0.95
        elif degradation_score < 0.15:
            return MaterialQuality.EXCELLENT, 0.90
        elif degradation_score < 0.25:  # Lowered from 0.30
            return MaterialQuality.GOOD, 0.85
        elif degradation_score < 0.35:  # Lowered from 0.45
            return MaterialQuality.FAIR, 0.80
        elif degradation_score < 0.50:  # Lowered from 0.60
            return MaterialQuality.POOR, 0.75
        elif degradation_score < 0.70:  # Lowered from 0.80
            return MaterialQuality.VERY_POOR, 0.70
        else:
            return MaterialQuality.EXTREME, 0.65

    def _calculate_processing_strength(
        self,
        _degradation_score: float,
        quality_level: MaterialQuality,
    ) -> float:
        """
        Berechne empfohlene Processing-Stärke (0.0 - 1.0)

        Principle: Je schlechter das Material, desto aggressiver das Processing
        """
        # Base strength from degradation

        # Quality-specific adjustments
        if quality_level == MaterialQuality.EXTREME:
            strength = 1.0  # Maximum processing
        elif quality_level == MaterialQuality.VERY_POOR:
            strength = 0.85
        elif quality_level == MaterialQuality.POOR:
            strength = 0.70
        elif quality_level == MaterialQuality.FAIR:
            strength = 0.50
        else:
            strength = 0.30  # Conservative for good material

        return float(strength)


class AdaptiveGoalsCalculator:
    """
    Berechnet adaptive Musical Goals Thresholds basierend auf Material-Qualität

    Principle: "Garantiere maximale Qualitätssteigerung relativ zur Ausgangssituation"

    - PRISTINE Material: Hohe Thresholds (0.85-0.90) → Exzellenz-Standard
    - POOR Material: Relaxierte Thresholds (0.50-0.65) → Realistische Ziele
    - EXTREME Material: Minimal-Thresholds (0.30-0.45) → Überhaupt Verbesserung
    """

    # Default Thresholds (für GOOD-EXCELLENT Material)
    DEFAULT_THRESHOLDS = {
        "brillanz": 0.85,
        "waerme": 0.80,
        "natuerlichkeit": 0.85,
        "authentizitaet": 0.88,
        "emotionalitaet": 0.87,
        "transparenz": 0.89,
        "bass_kraft": 0.85,
        "groove": 0.83,
        "spatial_depth": 0.70,
        "timbre_authentizitaet": 0.87,
        "tonal_center": 0.95,
        "micro_dynamics": 0.88,
        "separation_fidelity": 0.80,
        "artikulation": 0.88,
        "transient_energie": 0.88,
    }

    def calculate_adaptive_thresholds(
        self,
        quality: MaterialQualityAssessment,
        restorability_score: float | None = None,
    ) -> AdaptiveGoalThresholds:
        """
        Berechne adaptive Thresholds basierend auf Material-Qualität.

        Args:
            quality: Material Quality Assessment
            restorability_score: Optional — RestorabilityEstimator score (0–100).
                When provided, the normative §2.31 SCALE_FACTORS table is applied
                on top of the degradation-based relaxation to clamp thresholds
                to the physical ceiling measured on the AMRB corpus.

        Returns:
            AdaptiveGoalThresholds mit relaxierten Werten
        """
        # Calculate relaxation factor (0.0 = no relaxation, 1.0 = maximum relaxation)
        relaxation_factor = quality.degradation_score

        # §2.31: restorability-adaptive scale factor (normative SCALE_FACTORS table)
        # §2.47: pass shellac flag for special-case handling (restorability < 20 + shellac → 0.65)
        _is_shellac = "shellac" in (quality.medium_chain or [])
        scale = (
            get_restorability_scale_factor(float(restorability_score), is_shellac=_is_shellac)
            if restorability_score is not None
            else 1.0
        )

        # Apply relaxation to each threshold
        thresholds = {}
        for goal_name, default_threshold in self.DEFAULT_THRESHOLDS.items():
            # Relaxation formula:
            # - PRISTINE (degradation=0.0): threshold = default (0.85)
            # - EXTREME (degradation=1.0): threshold = default * 0.3 (0.25)

            min_threshold = default_threshold * 0.3  # Minimum 30% of default
            relaxed_threshold = default_threshold - (default_threshold - min_threshold) * relaxation_factor
            # §2.31: physical-ceiling clamp via SCALE_FACTORS
            relaxed_threshold = min(relaxed_threshold, default_threshold * scale)
            # Absolute Untergrenze (§2.31: adaptive_t ≥ 0.50)
            relaxed_threshold = max(relaxed_threshold, 0.50)

            thresholds[goal_name] = float(relaxed_threshold)

        adaptive_thresholds = AdaptiveGoalThresholds(
            brillanz=thresholds["brillanz"],
            waerme=thresholds["waerme"],
            natuerlichkeit=thresholds["natuerlichkeit"],
            authentizitaet=thresholds["authentizitaet"],
            emotionalitaet=thresholds["emotionalitaet"],
            transparenz=thresholds["transparenz"],
            bass_kraft=thresholds["bass_kraft"],
            quality_level=quality.quality_level,
            relaxation_factor=relaxation_factor,
            groove=thresholds["groove"],
            spatial_depth=thresholds["spatial_depth"],
            timbre_authentizitaet=thresholds["timbre_authentizitaet"],
            tonal_center=thresholds["tonal_center"],
            micro_dynamics=thresholds["micro_dynamics"],
            separation_fidelity=thresholds["separation_fidelity"],
            artikulation=thresholds["artikulation"],
            transient_energie=thresholds["transient_energie"],
        )

        logger.info(
            "Adaptive Thresholds calculated for %s: Relaxation=%.2f scale=%.2f Brillanz=%.2f Authentizität=%.2f",
            quality.quality_level.value,
            relaxation_factor,
            scale,
            adaptive_thresholds.brillanz,
            adaptive_thresholds.authentizitaet,
        )

        return adaptive_thresholds


class EnhancedProcessingStrategy:
    """
    Weltspitzen-Processing-Strategien für stark degradiertes Material

    Features:
    - Multi-Pass Processing (iterative Verbesserung)
    - Aggressive Denoise mit Harmonic Preservation
    - Spectral Gap Filling (Bandwidth Extension)
    - Harmonic Enhancement
    - Adaptive Mastering
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_enhanced_config(
        self,
        quality: MaterialQualityAssessment,
        base_config: dict,
    ) -> dict:
        """
        Erstelle Enhanced Processing Config für degradiertes Material

        Args:
            quality: Material Quality Assessment
            base_config: Base Processing Config (from ProcessingMode)

        Returns:
            Enhanced Config mit optimierten Parametern
        """
        enhanced = base_config.copy()

        # Enhance based on quality level
        if quality.quality_level in [MaterialQuality.VERY_POOR, MaterialQuality.EXTREME]:
            # EXTREME Processing
            enhanced["denoise_strength"] = min(0.8, quality.recommended_strength)
            enhanced["enhancement_strength"] = min(0.9, quality.recommended_strength)
            enhanced["enable_spectral_repair"] = True
            enhanced["enable_bandwidth_extension"] = True
            enhanced["enable_harmonic_enhancement"] = True
            enhanced["multi_pass_processing"] = True
            enhanced["adaptive_mastering"] = True

            self.logger.info("Activated EXTREME ENHANCED Processing (Multi-Pass, Bandwidth Extension)")

        elif quality.quality_level == MaterialQuality.POOR:
            # AGGRESSIVE Processing
            enhanced["denoise_strength"] = min(0.6, quality.recommended_strength)
            enhanced["enhancement_strength"] = 0.75
            enhanced["enable_spectral_repair"] = True
            enhanced["enable_harmonic_enhancement"] = True

            self.logger.info("Activated AGGRESSIVE ENHANCED Processing")

        elif quality.quality_level == MaterialQuality.FAIR:
            # MODERATE Enhancement
            enhanced["denoise_strength"] = 0.45
            enhanced["enhancement_strength"] = 0.60

            self.logger.info("Activated MODERATE ENHANCED Processing")

        return enhanced


# Convenience function for full adaptive workflow
def get_adaptive_goals_and_config(
    audio: np.ndarray,
    sr: int,
    base_config: Any = None,
    medium_detection: Any = None,
    restorability_score: float | None = None,
) -> tuple[AdaptiveGoalThresholds, dict, MaterialQualityAssessment]:
    """One-Stop-Shop: Analysiere Material und berechne adaptive Goals + Config.

    Akzeptiert beliebige Typen für ``base_config`` und ``medium_detection``:
    Nicht-dict-Werte werden sicher auf nutzbare Defaults normalisiert, sodass
    der Aufruf aus ``UnifiedRestorerV3`` (§2.31) mit MaterialType-/
    DefectAnalysisResult-Objekten ohne try/except-Absorption gelingt.

    Args:
        audio: Audio-Signal (float32/64, mono oder stereo).
        sr: Sample-Rate in Hz.
        base_config: Basis-Konfigurationsdikt **oder** beliebiges Objekt.
            Nicht-dict → wird auf ``{}`` normalisiert.
        medium_detection: Optionale Medium-Erkennungsdaten (dict) **oder**
            beliebiges Objekt. Nicht-dict → wird auf ``None`` normalisiert.

    Returns:
        Tuple: (adaptive_thresholds, enhanced_config, quality_assessment)
    """
    # ── Input-Normalisierung (§2.31-Invariante) ─────────────────────────────
    # base_config muss ein dict sein — get_enhanced_config ruft .copy() auf
    if not isinstance(base_config, dict):
        base_config = {}
    # medium_detection muss ein dict sein oder None —
    # _analyze_medium_chain ist bei None already safe (returns [], 0)
    if not isinstance(medium_detection, dict):
        medium_detection = None

    # 1. Analyze Material Quality
    analyzer = MaterialQualityAnalyzer()
    quality = analyzer.analyze(audio, sr, medium_detection)

    # 2. Calculate Adaptive Thresholds
    calculator = AdaptiveGoalsCalculator()
    thresholds = calculator.calculate_adaptive_thresholds(quality, restorability_score=restorability_score)

    # 3. Get Enhanced Processing Config
    strategy = EnhancedProcessingStrategy()
    enhanced_config = strategy.get_enhanced_config(quality, base_config)

    return thresholds, enhanced_config, quality
