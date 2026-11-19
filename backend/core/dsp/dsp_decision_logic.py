"""
dsp_decision_logic.py - Zentrale Entscheidungslogik für DSP- und KI-Modelle in Aurik 6.0

- Moderne, transparente und musikalisch präsente Ergebnisse
- Auditierbarkeit und SOTA-Konformität
"""

import json
import logging
import os
from typing import Any

import numpy as np
import yaml

from backend.core.data_models import AnalysisProfile

from .dsp_module_registry import DSPModuleRegistry

logger = logging.getLogger(__name__)


class DSPDecisionLogic:
    def analyze_adaptive(self, profile: AnalysisProfile) -> list[dict[str, Any]]:
        """
        SOTA-konforme, adaptive Entscheidungslogik: Verarbeitet ein vollständiges AnalysisProfile-Objekt
        und wählt Maßnahmenkette, DSP/ML-Module und Parameter optimal für musikalische Ziele.
        """
        # Extrahiere alle relevanten Parameter
        chain_info = profile.material_chain
        defects = profile.detected_defects
        musical = profile.musical_context
        vocals = profile.vocal_analysis
        features = profile.raw_features
        dynamics = profile.dynamics
        stereo = profile.stereo
        # Beispiel: adaptive Maßnahmenkette
        chain = []
        # 1. Medium- und Kettenlogik
        getattr(chain_info, "detected_medium", None)
        # 2. Defektlogik
        for defect in defects:
            if defect.defect_type == "broadband_noise" and defect.severity > 0.2:
                chain.append({"module": "sota_denoiser", "params": {"strength": min(1.0, defect.severity + 0.2)}})
            if defect.defect_type == "clipping" and defect.severity > 0.1:
                chain.append({"module": "automatic_declipper", "params": {}})
            if defect.defect_type == "hum" and defect.severity > 0.1:
                chain.append({"module": "automatic_dehum", "params": {}})
        # 3. Gesang/Geschlecht/Sibilanz
        if vocals.has_vocals:
            if features.get("f0_median", -1) > 180:
                chain.append(
                    {"module": "female_voice_deesser", "params": {"sibilance": features.get("sib_ratio", 1.0)}}
                )
            elif features.get("f0_median", -1) > 0:
                chain.append({"module": "male_voice_deesser", "params": {"sibilance": features.get("sib_ratio", 1.0)}})
            if features.get("f0_median", -1) > 250:
                chain.append({"module": "child_voice_deesser", "params": {"sibilance": features.get("sib_ratio", 1.0)}})
        # 4. Musikalische Features (z. B. Genre, Tempo, Harmonie)
        if musical.genre and str(musical.genre).lower() == "jazz":
            chain.append({"module": "jazz_eq", "params": {"chroma": features.get("chroma_mean", 0.5)}})
        if musical.tempo_bpm and musical.tempo_bpm > 120:
            chain.append({"module": "transient_shaper", "params": {"tempo": musical.tempo_bpm}})
        # 5. Stereo/Spatial
        if stereo.stereo_width < 0.5:
            chain.append({"module": "stereo_widener", "params": {"target_width": 1.0}})
        # 6. Adaptive Parameter für DSP/ML
        # (Beispiel: Sibilanz, Lautheit, Dynamik, etc.)
        if features.get("sib_ratio", 1.0) > 1.2:
            chain.append({"module": "adaptive_deesser", "params": {"sibilance": features["sib_ratio"]}})
        if dynamics.lufs_integrated > -14:
            chain.append({"module": "limiter", "params": {"target_lufs": -14}})
        # 7. Policy/Quality Gates (z. B. Zielwerte aus Policy)
        # ... weitere adaptive Logik ...
        # Rückgabe: Liste von Maßnahmen (Module + Parameter)
        return chain

    def maximum_quality_control(
        self,
        before,
        after,
        sr: int,
        out_path: str | None = None,
        policy: dict | None = None,
    ):
        """
        SOTA-Maximum-Qualitätskontrolle: Prüft Einhaltung musikalischer Ziele und dokumentiert Audit-Report.
        """

        # 1. Hochtonanteil (Dumpfheitsschutz)
        def band_energy(audio, sr, f_low, f_high):
            spec = np.abs(np.fft.rfft(audio))
            freqs = np.fft.rfftfreq(len(audio), 1 / sr)
            mask = (freqs >= f_low) & (freqs <= f_high)
            return float(np.sum(spec[mask]))

        hf_before = band_energy(before, sr, 6000, 11000)
        hf_after = band_energy(after, sr, 6000, 11000)
        hf_ratio = hf_after / (hf_before + 1e-9) if hf_before > 0 else 1.0

        # 2. Lautheit (LUFS, grob)
        loudness_before = float(np.mean(np.abs(before)))
        loudness_after = float(np.mean(np.abs(after)))

        # 3. Dynamik (Peak/RMS)
        peak_before = float(np.max(np.abs(before)))
        peak_after = float(np.max(np.abs(after)))
        rms_before = float(np.sqrt(np.mean(before**2)))
        rms_after = float(np.sqrt(np.mean(after**2)))
        dynamic_range_before = peak_before / (rms_before + 1e-9)
        dynamic_range_after = peak_after / (rms_after + 1e-9)

        # 4. Sibilanz (Energie 6-10kHz)
        sib_before = band_energy(before, sr, 6000, 10000)
        sib_after = band_energy(after, sr, 6000, 10000)
        sib_ratio = sib_after / (sib_before + 1e-9) if sib_before > 0 else 1.0

        # 5. Artefakte (Delta-Energie)
        artefact_energy = float(np.mean(np.abs(after - before)))

        # 6. Transparenz (Korrelation) — guarded dot-product (§VERBOTEN: np.corrcoef)
        if before.shape == after.shape:
            _b = before.flatten().astype(float)
            _a = after.flatten().astype(float)
            _bc = _b - float(np.mean(_b))
            _ac = _a - float(np.mean(_a))
            _nb = float(np.linalg.norm(_bc))
            _na = float(np.linalg.norm(_ac))
            corr = float(np.clip(np.dot(_bc, _ac) / (_nb * _na), -1.0, 1.0)) if _nb > 1e-12 and _na > 1e-12 else 0.0
        else:
            corr = 0.0

        # Zielwerte aus Policy oder Defaults
        pol = policy or {}
        min_hf = pol.get("quality", {}).get("min_hf_ratio", 0.7)
        min_trans = pol.get("quality", {}).get("min_transparency", 0.8)
        max_artefact = pol.get("quality", {}).get("max_artefact", 0.2)
        min_corr = min_trans

        passed = hf_ratio >= min_hf and corr >= min_corr and artefact_energy <= max_artefact

        report = {
            "passed": passed,
            "hf_ratio": hf_ratio,
            "loudness_before": loudness_before,
            "loudness_after": loudness_after,
            "dynamic_range_before": dynamic_range_before,
            "dynamic_range_after": dynamic_range_after,
            "sib_ratio": sib_ratio,
            "artefact_energy": artefact_energy,
            "correlation": corr,
            "policy": pol,
            "output_file": out_path,
        }
        # Audit-Report speichern
        if out_path:
            base, _ = os.path.splitext(out_path)
            audit_path = base + "_audit.json"
            with open(audit_path, "w") as f:
                json.dump(report, f, indent=2)
            logger.info("[Audit] SOTA-Qualitätsreport gespeichert: %s", audit_path)
        else:
            logger.info("[Audit] SOTA-Qualitätsreport: %s", report)
        return passed, report

    def quality_check(self, before, after, sr: int) -> bool:
        """
        Einfache Qualitätsprüfung: Dumpfheitsschutz (Hochtonanteil), Transparenz, musikalische Präsenz.
        Gibt True zurück, wenn die Qualitätsziele erreicht werden.
        """

        def band_energy(audio: np.ndarray, sr: int, f_low: float, f_high: float) -> float:
            spec = np.abs(np.fft.rfft(audio))
            freqs = np.fft.rfftfreq(len(audio), 1 / sr)
            mask = (freqs >= f_low) & (freqs <= f_high)
            return float(np.sum(spec[mask]))

        hf_b = band_energy(before, sr, 6000, 11000)
        hf_a = band_energy(after, sr, 6000, 11000)
        # Dumpfheit vermeiden: HF-Anteil muss erhalten bleiben
        if hf_b == 0:
            return True  # Kein Hochton im Original, keine Dumpfheitsprüfung nötig
        if hf_a / (hf_b + 1e-9) < 0.75:
            logger.info("[Audit] Dumpfheit erkannt: Hochtonverlust >25%")
            return False
        # Weitere Checks (z.B. Lautheit, Dynamik) können ergänzt werden
        return True

    def __init__(
        self,
        policy: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
        config_path: str | None = None,
    ):
        self.policy = policy or {}
        self.meta = meta or {}
        if config_path:
            self._load_config(config_path)
        self.registry = DSPModuleRegistry()
        self.available_modules = self.registry.list_modules()

    def _load_config(self, config_path: str):
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            # Übernehme Felder aus YAML in meta/policy
            if "medium" in config:
                self.meta["medium"] = config["medium"]
            if "defects" in config:
                self.meta["defects"] = config["defects"]
            if "policy" in config:
                self.policy.update(config["policy"])
        except Exception as e:
            logger.error("[DecisionLogic] Fehler beim Laden der YAML-Konfiguration: %s", e)

    def analyze_context(self, audio, sr: int) -> list[str]:
        """
        Analysiert Policy, Metadaten, Tonträgerart, Defekt-Hypothesen und wählt die optimale DSP-Kette.
        Unterscheidet explizit zwischen Kassette und Reel-to-Reel.
        """
        medium = str(self.meta.get("medium", "") or self.policy.get("medium", "")).lower()
        tape_type = self.meta.get("tape_type") or self.policy.get("tape_type")
        defects = self.meta.get("defects", []) or self.policy.get("defects", [])
        chain: list[str] = []
        if "vinyl" in medium:
            if "automatic_decrackler" in self.available_modules:
                chain.append("automatic_decrackler")
            if "dehiss" in self.available_modules:
                chain.append("dehiss")
        # --- Tape-Logik differenziert ---
        if "cassette" in (medium or "") or (tape_type and "cassette" in tape_type.lower()):
            if "tape_noise_reduction" in self.available_modules:
                chain.append("tape_noise_reduction")
            if "tape_equalizer" in self.available_modules:
                chain.append("tape_equalizer")
            if "automatic_dehum" in self.available_modules:
                chain.append("automatic_dehum")
        elif "reel" in (medium or "") or (
            tape_type and ("reel" in tape_type.lower() or "tonband" in tape_type.lower())
        ):
            if "reel_to_reel_noise_reduction" in self.available_modules:
                chain.append("reel_to_reel_noise_reduction")
            if "reel_to_reel_equalizer" in self.available_modules:
                chain.append("reel_to_reel_equalizer")
            if "automatic_dehum" in self.available_modules:
                chain.append("automatic_dehum")
        elif "8-track" in (medium or ""):
            if "eight_track_noise_reduction" in self.available_modules:
                chain.append("eight_track_noise_reduction")
            if "eight_track_equalizer" in self.available_modules:
                chain.append("eight_track_equalizer")
        elif "elcaset" in (medium or ""):
            if "elcaset_noise_reduction" in self.available_modules:
                chain.append("elcaset_noise_reduction")
            if "elcaset_equalizer" in self.available_modules:
                chain.append("elcaset_equalizer")
        elif "wire" in (medium or ""):
            if "wire_noise_reduction" in self.available_modules:
                chain.append("wire_noise_reduction")
        elif "dat" in (medium or ""):
            if "dat_error_correction" in self.available_modules:
                chain.append("dat_error_correction")
        elif "minidisc" in (medium or ""):
            if "minidisc_artifact_removal" in self.available_modules:
                chain.append("minidisc_artifact_removal")
        elif "dsd" in (medium or "") or "sacd" in (medium or ""):
            if "dsd_noise_shaper" in self.available_modules:
                chain.append("dsd_noise_shaper")
        elif "dvd-audio" in (medium or ""):
            if "dvd_audio_decoder" in self.available_modules:
                chain.append("dvd_audio_decoder")
        elif "hires" in (medium or ""):
            if "hires_audio_optimizer" in self.available_modules:
                chain.append("hires_audio_optimizer")
        elif "radio" in (medium or "") or "broadcast" in (medium or ""):
            if "radio_noise_reduction" in self.available_modules:
                chain.append("radio_noise_reduction")
        elif "pstn" in (medium or "") or "gsm" in (medium or "") or "voip" in (medium or ""):
            if "telephone_bandwidth_expander" in self.available_modules:
                chain.append("telephone_bandwidth_expander")
        elif "hybrid" in (medium or "") or "/" in (medium or ""):
            if "hybrid_chain_optimizer" in self.available_modules:
                chain.append("hybrid_chain_optimizer")
        elif "tape" in medium:
            if "automatic_dehum" in self.available_modules:
                chain.append("automatic_dehum")
        if "shellac" in medium and "shellac_declicker" in self.available_modules:
            chain.append("shellac_declicker")
        if "clipping" in defects and "automatic_declipper" in self.available_modules:
            chain.append("automatic_declipper")
        if "noise" in defects and "sota_denoiser" in self.available_modules:
            chain.append("sota_denoiser")
        if "reverb" in defects and "sota_dereverberator" in self.available_modules:
            chain.append("sota_dereverberator")
        for mod in [
            "automatic_deesser",
            "limiter",
            "loudness_matching",
            "stereo_widener",
            "harmonic_exciter",
        ]:
            if mod in self.available_modules:
                chain.append(mod)
        for mod in self.available_modules:
            if mod not in chain:
                chain.append(mod)
        return chain

    def process(self, audio, sr: int, model_path: str | None = None):
        # DSP-Kette dynamisch bestimmen (Policy, Signal, Tonträger, Defektanalyse)
        chain = self.analyze_context(audio, sr)
        out = audio
        for mod_name in chain:
            try:
                mod = self.registry.instantiate(mod_name)
                # Standardisierte Schnittstelle: .process(audio, sr) oder .deess/.dereverberate etc.
                if hasattr(mod, "process"):
                    out = mod.process(out, sr)
                elif hasattr(mod, "deess"):
                    out = mod.deess(out, sr)
                elif hasattr(mod, "dereverberate"):
                    out = mod.dereverberate(out, sr)
                else:
                    logger.info("[DecisionLogic] Modul %s hat keine bekannte Verarbeitungsmethode.", mod_name)
            except Exception as e:
                logger.error("[DecisionLogic] Fehler bei %s: %s", mod_name, e)
        # Qualitätsprüfung: Dumpfheit, Transparenz, musikalische Präsenz
        # SOTA-Maximum-Qualitätskontrolle nach der Kette
        out_path = None
        if hasattr(self, "output_path_hint"):
            out_path = self.output_path_hint
        passed, _report = self.maximum_quality_control(audio, out, sr, out_path=out_path, policy=self.policy)
        if not passed:
            logger.info("[Audit] SOTA-Qualitätsziel nicht erreicht, Rollback auf das Originalsignal.")
            return audio
        return out
