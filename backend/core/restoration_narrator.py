"""
core/restoration_narrator.py — RestorationNarrator v1.0 (Aurik 10.0.0)

Übersetzt die technischen Restaurierungsergebnisse von UnifiedRestorerV3 in
laienverständliche, emotional ansprechende Sprache.

Ziel: Ein Nutzer ohne jede Technik-Kenntnis versteht sofort —
  • ob die Restaurierung geholfen hat
  • wie schwierig das Material war
  • dass Aurik mit jeder Restaurierung besser für ihn wird
  • wie er das Ergebnis im Vergleich hören kann

Alle Ausgaben sind auf Deutsch, laienverständlich, niemals technisch.

Singleton-Pattern (Thread-safe, Double-Checked Locking gemäß Spec §3.2).
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schwellwert-Konstanten
# ---------------------------------------------------------------------------

_HIGH_CONFIDENCE = 0.70  # Tier "high" ab dieser Konfidenz
_LOW_CONFIDENCE = 0.45  # Tier "low" unter dieser Konfidenz
_EXCELLENT_QUALITY = 0.82  # Quality-Estimate: Weltklasse
_GOOD_QUALITY = 0.65  # Quality-Estimate: gut bis sehr gut
_FAIR_QUALITY = 0.45  # Quality-Estimate: deutliche Verbesserung
_POOR_QUALITY = 0.25  # Quality-Estimate: geringe Verbesserung

# GP-Lernkurve: Schwellwerte für Nutzermeldungen
_GP_NEWBIE = 0  # noch gar nichts gelernt
_GP_EARLY = 3  # erste Erfahrungen
_GP_KNOWS = 10  # kennt den Geschmack
_GP_EXPERT = 30  # verlässlicher Klang-Experte


# ---------------------------------------------------------------------------
# Ergebnis-Datenklasse
# ---------------------------------------------------------------------------


@dataclass
class NarratorResult:
    """
    Menschenlesbare Restaurierungs-Erzählung für Laien-Nutzer.

    Alle Felder sind auf Deutsch, emotional ansprechend und
    frei von technischem Fachvokabular.
    """

    verdict: str
    """Ein-Satz-Antwort auf: 'Hat es geholfen?' — immer vorhanden."""

    emotional_summary: str
    """Begeisterte, ehrliche Einschätzung des Restaurierungserfolgs."""

    comparison_hint: str
    """Laiengerechter Hinweis wie man Vorher/Nachher hören kann."""

    trust_message: str | None = None
    """Nur bei schwierigem Material: ehrliche, beruhigende Einschätzung."""

    learning_message: str | None = None
    """Hinweis auf den Lernfortschritt des GP-Systems — wenn sinnvoll."""

    difficulty_stars: int = 3
    """Schwierigkeitsgrad 1–5 Sterne (für UI-Anzeige)."""

    quality_stars: int = 3
    """Restaurierungsqualität 1–5 Sterne (für UI-Anzeige)."""

    confidence_tier: str = "medium"
    """'high' | 'medium' | 'low' — für UI-Farbkodierung."""

    era_context: str | None = None
    """Kontext-Satz zur erkannten Aufnahme-Ära — wenn bekannt."""

    defects_found: list[str] = field(default_factory=list)
    """Liste der erkannten Defekte in laienverständlicher Sprache."""

    defects_fixed: list[str] = field(default_factory=list)
    """Liste der behobenen Defekte in laienverständlicher Sprache."""

    gp_observations: int = 0
    """Anzahl der GP-Lernbeobachtungen für dieses Material."""

    def as_dict(self) -> dict:
        """Serialisierungsformat für Metadata-Dict und JSON."""
        return {
            "verdict": self.verdict,
            "emotional_summary": self.emotional_summary,
            "comparison_hint": self.comparison_hint,
            "trust_message": self.trust_message,
            "learning_message": self.learning_message,
            "difficulty_stars": self.difficulty_stars,
            "quality_stars": self.quality_stars,
            "confidence_tier": self.confidence_tier,
            "era_context": self.era_context,
            "defects_found": self.defects_found,
            "defects_fixed": self.defects_fixed,
            "gp_observations": self.gp_observations,
        }


# ---------------------------------------------------------------------------
# Übersetzungstabellen: technisch → laienverständlich
# ---------------------------------------------------------------------------

_DEFECT_LABELS_DE: dict[str, str] = {
    # DefectType-Enum-Werte
    "clicks": "Knackser und Klicks",
    "crackle": "Knistern",
    "hum": "Brummen",
    "wow_flutter": "Tonhöhen-Schwankungen",
    "low_freq_rumble": "dumpfes Grundrauschen",
    "dropouts": "Tonaussetzer",
    "clipping": "Übersteuerungen",
    "soft_saturation": "weiche Klangsättigung",
    "dc_offset": "Gleichspannungs-Versatz",
    "bandwidth_loss": "fehlende Höhen",
    "high_freq_noise": "Rauschen in den Höhen",
    "stereo_imbalance": "ungleicher Links/Rechts-Pegel",
    "phase_issues": "Phasenfehler",
    "pitch_drift": "langsame Tonhöhendrift",
    "reverb_excess": "unnatürlicher Hall",
    "print_through": "magnetisches Vorecho",
    "digital_artifacts": "digitale Artefakte",
    "compression_artifacts": "Kompressionsartefakte",
    "quantization_noise": "Digitalisierungsrauschen",
    "jitter_artifacts": "Zeitversatz-Fehler",
    "dynamic_compression_excess": "überpresste Dynamik",
    # DefectType-Werte aus data_models.py
    "broadband_noise": "Breitrauschen",
    "crackle_pops": "Knistern und Knacken",
    "dropout": "Tonaussetzer",
    "distortion": "Verzerrung",
    "compression": "Dynamik-Kompression",
    # Leerstring-Fallback
    "": "unbekannter Defekt",
}

_MATERIAL_LABELS_DE: dict[str, str] = {
    "tape": "Magnetband-Kassette",
    "reel_tape": "Spulenband",
    "vinyl": "Schallplatte",
    "shellac": "Schellack-Schallplatte",
    "wax_cylinder": "Phonographen-Wachswalze",
    "wire_recording": "Drahtbandaufnahme",
    "lacquer_disc": "Acetat-Lackfolie",
    "dat": "Digital-Audio-Tape",
    "cd_digital": "CD / digitale Aufnahme",
    "mp3_low": "stark komprimierte MP3-Datei",
    "mp3_high": "MP3-Datei",
    "aac": "AAC-Datei",
    "minidisc": "MiniDisc",
    "streaming": "Streaming-Aufnahme",
    "unknown": "Aufnahme unbekannter Herkunft",
    # Aus data_models.py MediaType:
    "cassette": "Kassette",
    "cd": "CD",
    "digital_native": "digitale Aufnahme",
    "radio_broadcast": "Rundfunkaufnahme",
}


def _defect_label(defect_key: str) -> str:
    """Gibt den deutschen Laien-Namen eines Defekttyps zurück."""
    key = defect_key.lower().replace("-", "_")
    return _DEFECT_LABELS_DE.get(key, defect_key.replace("_", " "))


def _material_label(material: str) -> str:
    """Gibt die deutsche Laien-Bezeichnung des Trägermaterials zurück."""
    return _MATERIAL_LABELS_DE.get(material.lower(), material)


# ---------------------------------------------------------------------------
# Hauptklasse: RestorationNarrator
# ---------------------------------------------------------------------------


class RestorationNarrator:
    """
    Erzeugt laienverständliche, emotionale Restaurierungsberichte.

    Algorithmus:
        1. Qualitätsstufe ermitteln (quality_estimate, musical_goal_scores, PQS-MOS)
        2. Konfidenz-Tier auswerten (high / medium / low → trust_message)
        3. Erkannte Defekte in Alltagssprache übersetzen
        4. Ära- und Material-Kontext ergänzen (era_label, decade)
        5. GP-Gedächtnis-Stand auslesen → learning_message
        6. Alle Texte zu NarratorResult zusammensetzen

    Thread-Sicherheit: Stateless — keine Instanzvariablen werden geschrieben.
    Alle Methoden sind rein funktional und können parallel aufgerufen werden.
    """

    # -------------------------------------------------------------------
    # Öffentliche API
    # -------------------------------------------------------------------

    def narrate(
        self,
        *,
        quality_estimate: float,
        material: str = "unknown",
        confidence: float = 0.60,
        confidence_tier: str = "medium",
        musical_goal_scores: dict[str, float] | None = None,
        musical_goals_passed: dict[str, bool] | None = None,
        top_defects: list[tuple[str, float]] | None = None,
        executed_phases: int = 0,
        era_decade: int | None = None,
        era_label: str | None = None,
        pqs_mos: float | None = None,
        gp_observations: int = 0,
    ) -> NarratorResult:
        """
        Erzeugt den vollständigen NarratorResult aus den technischen Metriken.

        Args:
            quality_estimate:      Restaurierungsqualität [0, 1]
            material:              Materialtyp-String (z.B. "vinyl")
            confidence:            Pipeline-Konfidenz [0, 1]
            confidence_tier:       'high' | 'medium' | 'low'
            musical_goal_scores:   Dict[goal_name → score ∈ [0, 1]]
            musical_goals_passed:  Dict[goal_name → bool]
            top_defects:           Liste (defect_type, severity) — absteigend
            executed_phases:       Anzahl ausgeführter Phasen
            era_decade:            Erkanntes Aufnahme-Jahrzehnt (z.B. 1940)
            era_label:             Textbeschriftung der Ära (z.B. "1940er Jahre")
            pqs_mos:               PQS-MOS-Wert [1.0, 5.0]
            gp_observations:       Anzahl GP-Lernbeobachtungen für dieses Material

        Returns:
            NarratorResult mit allen Texten und Sternbewertungen
        """
        # --- NaN-Guard ---
        quality_estimate = float(quality_estimate) if math.isfinite(float(quality_estimate)) else 0.5
        confidence = float(confidence) if math.isfinite(float(confidence)) else 0.5
        pqs_mos = float(pqs_mos) if pqs_mos is not None and math.isfinite(float(pqs_mos)) else None
        musical_goal_scores = musical_goal_scores or {}
        musical_goals_passed = musical_goals_passed or {}
        top_defects = top_defects or []

        # Kombinierte Qualitätsgröße (bevorzugt PQS-MOS wenn vorhanden)
        eff_quality = self._effective_quality(quality_estimate, pqs_mos, musical_goal_scores)

        # Erkannte / behobene Defekte in Alltagssprache
        defects_found, defects_fixed = self._translate_defects(top_defects)

        # Sternbewertungen
        q_stars = self._quality_stars(eff_quality)
        d_stars = self._difficulty_stars(confidence, top_defects)

        # Kern-Texte
        verdict = self._build_verdict(eff_quality, defects_fixed)
        emotional_summary = self._build_emotional_summary(
            eff_quality, material, era_decade, era_label, defects_fixed, q_stars
        )
        comparison_hint = self._build_comparison_hint()
        trust_message = self._build_trust_message(confidence, confidence_tier, top_defects)
        learning_message = self._build_learning_message(gp_observations, material, eff_quality)
        era_context = self._build_era_context(era_decade, era_label, material) if era_decade else None

        logger.debug(
            "🗣️ Narrator: q=%.2f conf=%.2f stars=%d/%d gp_obs=%d era=%s material=%s",
            eff_quality,
            confidence,
            q_stars,
            d_stars,
            gp_observations,
            era_decade,
            material,
        )

        return NarratorResult(
            verdict=verdict,
            emotional_summary=emotional_summary,
            comparison_hint=comparison_hint,
            trust_message=trust_message,
            learning_message=learning_message,
            difficulty_stars=d_stars,
            quality_stars=q_stars,
            confidence_tier=confidence_tier,
            era_context=era_context,
            defects_found=defects_found,
            defects_fixed=defects_fixed,
            gp_observations=gp_observations,
        )

    # -------------------------------------------------------------------
    # Effektive Qualitätsgröße
    # -------------------------------------------------------------------

    def _effective_quality(
        self,
        quality_estimate: float,
        pqs_mos: float | None,
        musical_goal_scores: dict[str, float],
    ) -> float:
        """
        Kombiniert quality_estimate, PQS-MOS und Musical-Goal-Durchschnitt
        zu einer einheitlichen Qualitätsgröße ∈ [0, 1].

        Formel:
            q_eff = 0.50·q_est + 0.30·q_mos + 0.20·q_goals
        wobei:
            q_mos   = (MOS − 1) / 4   (normiert [1,5] → [0,1])
            q_goals = Durchschnitt der Musical-Goal-Scores
        """
        weights = [0.50, 0.30, 0.20]
        components = [quality_estimate, None, None]

        if pqs_mos is not None:
            components[1] = max(0.0, min(1.0, (pqs_mos - 1.0) / 4.0))

        if musical_goal_scores:
            valid_scores = [v for v in musical_goal_scores.values() if math.isfinite(v)]
            if valid_scores:
                components[2] = float(np.mean(valid_scores))

        total_weight = 0.0
        total_value = 0.0
        for w, c in zip(weights, components):
            if c is not None:
                total_weight += w
                total_value += w * c

        if total_weight < 1e-9:
            return quality_estimate
        eff = total_value / total_weight
        return float(np.clip(eff, 0.0, 1.0))

    # -------------------------------------------------------------------
    # Sternbewertungen
    # -------------------------------------------------------------------

    def _quality_stars(self, eff_quality: float) -> int:
        """Restaurierungsqualität → 1–5 Sterne."""
        if eff_quality >= _EXCELLENT_QUALITY:
            return 5
        if eff_quality >= _GOOD_QUALITY:
            return 4
        if eff_quality >= _FAIR_QUALITY:
            return 3
        if eff_quality >= _POOR_QUALITY:
            return 2
        return 1

    def _difficulty_stars(
        self,
        confidence: float,
        top_defects: list[tuple[str, float]],
    ) -> int:
        """
        Schwierigkeitsgrad 1–5: Je mehr Defekte und je niedriger die Konfidenz,
        desto höher (schwieriger). 1 = einfach, 5 = extrem schwierig.
        """
        n_severe = sum(1 for _, sev in top_defects if sev >= 0.7)
        base = 1
        if confidence < _LOW_CONFIDENCE:
            base += 2
        elif confidence < _HIGH_CONFIDENCE:
            base += 1
        base += min(2, n_severe)
        return min(5, base)

    # -------------------------------------------------------------------
    # Defekt-Übersetzung
    # -------------------------------------------------------------------

    def _translate_defects(
        self,
        top_defects: list[tuple[str, float]],
        threshold: float = 0.25,
    ) -> tuple[list[str], list[str]]:
        """
        Konvertiert technische Defektbezeichnungen in Alltagssprache.

        Returns:
            (defects_found, defects_fixed):
                defects_found: alle erkannten Defekte mit Severity ≥ threshold
                defects_fixed: Teilmenge, die als behoben gilt (severity ≥ 0.35)
        """
        found, fixed = [], []
        for defect_key, severity in top_defects:
            if severity < threshold:
                continue
            label = _defect_label(defect_key)
            if label not in found:
                found.append(label)
            if severity >= 0.35:
                fixed.append(label)
        return found, fixed

    # -------------------------------------------------------------------
    # Text-Builder
    # -------------------------------------------------------------------

    def _build_verdict(self, eff_quality: float, defects_fixed: list[str]) -> str:
        """
        Kurze Antwort auf 'Hat es geholfen?' — immer ein einziger Satz.
        """
        n_fixed = len(defects_fixed)

        if eff_quality >= _EXCELLENT_QUALITY:
            if n_fixed >= 3:
                return (
                    f"Ja — Aurik hat {n_fixed} verschiedene Klangfehler beseitigt "
                    "und die Aufnahme auf Weltklasse-Niveau gebracht."
                )
            return "Ja, eindeutig — die Aufnahme klingt jetzt auf Weltklasse-Niveau."

        if eff_quality >= _GOOD_QUALITY:
            if n_fixed >= 2:
                return f"Ja — {n_fixed} Klangfehler wurden behoben, das Ergebnis ist deutlich besser als das Original."
            if n_fixed == 1:
                return (
                    f"Ja — {defects_fixed[0].capitalize()} wurde erfolgreich entfernt, "
                    "die Aufnahme klingt deutlich klarer."
                )
            return "Ja — die Aufnahme klingt deutlich besser als zuvor."

        if eff_quality >= _FAIR_QUALITY:
            return "Ja, mit gutem Ergebnis — die auffälligsten Klangfehler wurden spürbar reduziert."

        if eff_quality >= _POOR_QUALITY:
            return (
                "Ja, wenn auch das Ausgangsmaterial sehr schwierig war — "
                "eine messbare Verbesserung konnte erzielt werden."
            )

        return (
            "Das Ausgangsmaterial war außergewöhnlich stark beschädigt. "
            "Aurik hat das Maximum herausgeholt, das physikalisch möglich war."
        )

    def _build_emotional_summary(
        self,
        eff_quality: float,
        material: str,
        era_decade: int | None,
        era_label: str | None,
        defects_fixed: list[str],
        quality_stars: int,
    ) -> str:
        """
        Begeisterte, ehrliche Zusammenfassung des Restaurierungserfolgs.
        Bezieht Ära, Material und die wichtigsten Verbesserungen ein.
        """
        mat_label = _material_label(material)

        # Ära-Einleitung
        if era_decade and era_label:
            era_intro = f"Diese Aufnahme aus den {era_label}"
        elif era_decade:
            era_intro = f"Diese Aufnahme aus dem Jahr {era_decade}"
        else:
            era_intro = f"Diese {mat_label}"

        # Haupt-Botschaft nach Qualitätsstufe
        if quality_stars == 5:
            core = (
                "klingt nach der Restaurierung so, als wäre sie gestern in einem "
                "modernen Tonstudio aufgenommen worden. "
                "Die Klangqualität ist außergewöhnlich."
            )
        elif quality_stars == 4:
            core = (
                "hat durch die Restaurierung eine beeindruckende Verwandlung erlebt. "
                "Was vorher verborgen war, ist jetzt deutlich und lebendig hörbar."
            )
        elif quality_stars == 3:
            if defects_fixed:
                core = (
                    f"klingt nach der Entfernung von {defects_fixed[0]} "
                    "wesentlich angenehmer. Die wichtigsten Störungen wurden beseitigt."
                )
            else:
                core = "wurde spürbar verbessert — die störendsten Klangfehler sind nicht mehr hörbar."
        elif quality_stars == 2:
            core = (
                "war eine echte Herausforderung. Das Ausgangsmaterial war stark beschädigt, "
                "aber Aurik hat die bestmögliche Verbesserung erzielt."
            )
        else:
            core = (
                "gehört zu den schwierigsten Restaurierungsaufgaben überhaupt. "
                "Aurik hat alle verfügbaren Mittel eingesetzt, "
                "um das Beste aus dem Original herauszuholen."
            )

        return f"{era_intro} {core}"

    def _build_comparison_hint(self) -> str:
        """
        Laiengerechter, motivierender Hinweis auf den Vorher/Nachher-Vergleich.
        Kein Fachwissen nötig — jeder versteht es sofort.
        """
        return (
            "Hör einfach beide Versionen ab: Original und restauriert. "
            "Du wirst den Unterschied sofort spüren — "
            "besonders an Stellen, die vorher gedämpft oder verrauscht klangen."
        )

    def _build_trust_message(
        self,
        confidence: float,
        tier: str,
        top_defects: list[tuple[str, float]],
    ) -> str | None:
        """
        Erscheint nur, wenn die Restaurierung schwierig war oder die Konfidenz gering.
        Ehrlich und beruhigend — keine falschen Versprechungen.
        """
        if tier == "high" and confidence >= _HIGH_CONFIDENCE:
            return None  # Kein Bedarf — alles sicher

        if tier == "low" or confidence < _LOW_CONFIDENCE:
            return (
                "Diese Aufnahme war sehr schwierig zu analysieren — "
                "das System arbeitet in solchen Fällen besonders vorsichtig, "
                "damit nichts verschlechtert wird. "
                "Das Ergebnis zeigt das Maximum, das aus diesem Material möglich ist. "
                "Manche Stellen könnten noch leichte Spuren des Originals tragen."
            )

        if tier == "medium":
            n_severe = sum(1 for _, sev in top_defects if sev >= 0.8)
            if n_severe >= 2:
                return (
                    "Manche Stellen waren schwer zu beurteilen — "
                    "das System hat dort besonders vorsichtig gearbeitet, "
                    "damit nichts am Original-Klangcharakter verloren geht."
                )
        return None

    def _build_learning_message(
        self,
        gp_observations: int,
        material: str,
        eff_quality: float,
    ) -> str | None:
        """
        Sagt dem Nutzer, dass Aurik mit jeder Restaurierung besser für ihn wird.
        Nur wenn es etwas Bedeutungsvolles zu sagen gibt.

        Schwellen (§2.5):
            0:  noch kein Lernstand — keine Meldung
            1–3:  erste Erfahrungen
            4–9:  Aurik kennt bereits den Geschmack
            10–29: verlässlicher Klang-Experte
            ≥30:   persönlicher Klangspezialist
        """
        mat_label = _material_label(material)

        if gp_observations <= _GP_NEWBIE:
            # Erste Restaurierung — sanfte Einführung
            return (
                "Dies ist eine der ersten Aufnahmen, die Aurik für dich restauriert. "
                f"Je mehr {mat_label}n du bearbeitest, desto besser "
                "lernt Aurik dein Material kennen."
            )

        if gp_observations <= _GP_EARLY:
            return (
                f"Aurik hat {gp_observations} Erfahrungen mit "
                f"{mat_label}n gesammelt und "
                "beginnt, dein Material kennen zu lernen."
            )

        if gp_observations <= _GP_KNOWS:
            return (
                f"Aurik kennt {mat_label}n bereits gut — "
                f"nach {gp_observations} Restaurierungen "
                "passen sich die Klangeinstellungen immer besser an dein Material an."
            )

        if gp_observations <= _GP_EXPERT:
            return (
                f"Mit {gp_observations} Restaurierungen hat Aurik "
                "einen verlässlichen Klangsinn für dein Material entwickelt. "
                "Die Ergebnisse werden mit jeder weiteren Aufnahme noch präziser."
            )

        return (
            f"Aurik ist durch {gp_observations} Restaurierungen "
            f"ein erprobter Spezialist für {mat_label}n geworden — "
            "der Klang wird auf dein Material maßgeschneidert."
        )

    def _build_era_context(
        self,
        era_decade: int,
        era_label: str | None,
        material: str,
    ) -> str:
        """
        Liefert einen historischen Kontext-Satz zur erkannten Aufnahme-Ära.
        Fasziniert den Nutzer — verbindet die Technik mit Geschichte.
        """
        label = era_label or f"den {era_decade}er Jahren"
        mat_label = _material_label(material)

        if era_decade <= 1910:
            return (
                f"Diese Aufnahme stammt aus {label} — "
                "einer Zeit, als Musik noch auf Wachswalzen festgehalten wurde. "
                "Aurik hat die Klanggrenzen der damaligen Aufnahmetechnik berücksichtigt."
            )
        if era_decade <= 1930:
            return (
                f"Aufnahmen aus {label} wurden oft mit frühen Kohlenmikrofonen gemacht. "
                "Aurik hat den charakteristischen Klang dieser Ära bewahrt "
                "und nur störende Geräusche entfernt."
            )
        if era_decade <= 1950:
            return (
                f"Aus {label} stammt ein wertvoller Klangzeuge der frühen Tonstudio-Ära. "
                "Aurik hat die warme Röhrenklang-Patina erhalten "
                "und das störende Rauschen dieser Zeit entfernt."
            )
        if era_decade <= 1970:
            return (
                f"Goldene Studio-Ära: Aufnahmen aus {label} haben einen unverwechselbaren Charakter. "
                "Aurik hat den Originalklang bewahrt und nur die Schäden des Tons behoben."
            )
        if era_decade <= 1990:
            return (
                f"Aus {label} — der Zeit der analogen {mat_label}s. "
                "Aurik kannte die typischen Defekte dieser Epoche und hat gezielt behoben, "
                "was stört, und bewahrt, was Charakter verleiht."
            )
        if era_decade <= 2010:
            return f"Eine digitale Aufnahme aus {label}. Aurik hat Kompressionsartefakte und digitale Fehler beseitigt."
        return (
            f"Eine moderne Aufnahme aus {label}. "
            "Aurik hat verborgene Klangfehler behoben und die Klangqualität verfeinert."
        )


# ---------------------------------------------------------------------------
# Thread-sicherer Singleton (§3.2 Spec: Double-Checked Locking)
# ---------------------------------------------------------------------------

_instance: RestorationNarrator | None = None
_lock = threading.Lock()


def get_narrator() -> RestorationNarrator:
    """Thread-sicherer Singleton-Accessor (Double-Checked Locking)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = RestorationNarrator()
    return _instance


# ---------------------------------------------------------------------------
# Convenience-Funktion (§3.2 Spec)
# ---------------------------------------------------------------------------


def narrate_restoration(
    *,
    quality_estimate: float,
    material: str = "unknown",
    confidence: float = 0.60,
    confidence_tier: str = "medium",
    musical_goal_scores: dict[str, float] | None = None,
    musical_goals_passed: dict[str, bool] | None = None,
    top_defects: list[tuple[str, float]] | None = None,
    executed_phases: int = 0,
    era_decade: int | None = None,
    era_label: str | None = None,
    pqs_mos: float | None = None,
    gp_observations: int = 0,
) -> NarratorResult:
    """
    Convenience-Wrapper: Erzeugt laienverständlichen Restaurierungsbericht.

    Verwendet den Singleton-RestorationNarrator.

    Args:
        quality_estimate:      Restaurierungsqualität [0, 1]
        material:              Materialtyp-String (z.B. "vinyl")
        confidence:            Pipeline-Konfidenz [0, 1]
        confidence_tier:       'high' | 'medium' | 'low'
        musical_goal_scores:   Dict[goal_name → score ∈ [0, 1]]
        musical_goals_passed:  Dict[goal_name → bool]
        top_defects:           Liste (defect_type, severity) — absteigend
        executed_phases:       Anzahl ausgeführter Phasen
        era_decade:            Erkanntes Aufnahme-Jahrzehnt (z.B. 1940)
        era_label:             Textbeschriftung der Ära
        pqs_mos:               PQS-MOS-Wert [1.0, 5.0]
        gp_observations:       Anzahl GP-Lernbeobachtungen für dieses Material

    Returns:
        NarratorResult mit allen laienverständlichen Texten
    """
    return get_narrator().narrate(
        quality_estimate=quality_estimate,
        material=material,
        confidence=confidence,
        confidence_tier=confidence_tier,
        musical_goal_scores=musical_goal_scores,
        musical_goals_passed=musical_goals_passed,
        top_defects=top_defects,
        executed_phases=executed_phases,
        era_decade=era_decade,
        era_label=era_label,
        pqs_mos=pqs_mos,
        gp_observations=gp_observations,
    )
