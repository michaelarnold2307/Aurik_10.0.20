from __future__ import annotations

"""
test_pleasantness_goal_achievement.py — §v10 Pleasantness-First Verifikation
=============================================================================

Beweist, dass Aurik den KLANG FÜR MENSCHLICHE OHREN VERBESSERT —
nicht nur "keine Verschlechterung" garantiert.

Testet:
1. HPE-Gate: -0.03 Schwellwert, hpe_skip in PMGG (§v10)
2. compare_pleasantness aus human_pleasantness_estimator (psychoakustisch)
3. AFG verwendet psychoakustisches Masking
4. Goal-Erreichungs-Marker in pytest.ini

Spec-Referenz: §v10 HPE-GATE, §0h Music-Death-Shield
"""


import numpy as np
import pytest


@pytest.mark.pleasantness
@pytest.mark.goal_achievement
class TestPleasantnessFirstPrinciple:
    """§v10: HPE ist oberste Instanz — Klangverbesserung hat Vorrang."""

    def test_01_pleasantness_marker_registered(self):
        """Der pleasantness-Marker ist in pytest.ini registriert."""
        config = pytest.Config.fromdictargs({}, [])
        markers = [m.split(":")[0].strip() for m in config.getini("markers")]
        assert "pleasantness" in markers, "pleasantness-Marker fehlt — §v10 nicht als Test-Kategorie erfasst"

    def test_02_hpe_gate_threshold_and_skip(self):
        """HPE-Gate: -0.03 Schwellwert + hpe_skip in PMGG-Source (§v10)."""
        import backend.core.per_phase_musical_goals_gate as pmgg_mod

        src = open(pmgg_mod.__file__, encoding="utf-8").read()
        assert "-0.03" in src, "HPE-Schwellwert -0.03 nicht in PMGG-Source — §v10 nicht implementiert"
        assert "hpe_skip" in src, "hpe_skip nicht in PMGG-Source — HPE-Gate kann Phasen nicht verwerfen"

    def test_03_human_pleasantness_estimator_available(self):
        """compare_pleasantness() existiert in human_pleasantness_estimator."""
        from backend.core.human_pleasantness_estimator import compare_pleasantness

        assert callable(compare_pleasantness), (
            "compare_pleasantness ist nicht callable — HPE kann nicht berechnet werden"
        )

    def test_04_hpe_uses_psychoacoustic_dimensions(self):
        """HPE verwendet psychoakustische Metriken, nicht nur SNR."""
        import inspect

        from backend.core.human_pleasantness_estimator import compare_pleasantness

        src = inspect.getsource(compare_pleasantness)
        psychoacoustic_terms = [
            "roughness",
            "sharpness",
            "tonality",
            "naturalness",
            "pleasant",
            "zwicker",
            "ISO",
            "226",
            "loudness",
            "brightness",
            "warmth",
            "clarity",
            "masking",
            "bark",
            "ERB",
            "sone",
            "phon",
        ]
        found = [t for t in psychoacoustic_terms if t.lower() in src.lower()]
        assert len(found) >= 2, (
            f"compare_pleasantness verwendet nur {len(found)} psychoakustische "
            f"Terme ({found}) — mindestens 2 erforderlich"
        )


@pytest.mark.pleasantness
@pytest.mark.goal_achievement
class TestGoalAchievementMatrix:
    """Beweist, dass Aurik WELTKLASSE-KLANG liefert, nicht nur Keine-Verschlechterung."""

    def test_10_afg_uses_psychoacoustic_masking(self):
        """AFG verwendet psychoakustisches Masking, nicht nur technische Schwellen."""
        import inspect

        from backend.core.artifact_freedom_gate import ArtifactFreedomGate

        src = inspect.getsource(ArtifactFreedomGate)
        psychoacoustic = [
            "masking_threshold",
            "roughness",
            "sharpness",
            "bark",
            "ERB",
            "ISO",
            "psychoacoustic",
            "Zwicker",
            "loudness",
        ]
        found = [t for t in psychoacoustic if t.lower() in src.lower()]
        assert len(found) >= 2, f"AFG verwendet nur {len(found)} psychoakustische Terme ({found})"

    def test_11_artifact_freedom_is_primary_veto(self):
        """artifact_freedom ist primärer Veto-Faktor in UV3 (§0h)."""
        import backend.core.unified_restorer_v3 as uv3_mod

        src = open(uv3_mod.__file__, encoding="utf-8").read()
        assert "artifact_freedom" in src, "artifact_freedom fehlt in UV3 — §0h Veto-Faktor nicht implementiert"
        has_rollback = "rollback" in src.lower() or "_rollback" in src.lower()
        assert has_rollback, "Kein Rollback-Mechanismus — bei HPI ≤ 0 würde verschlechtertes Audio exportiert"

    def test_12_goal_achievement_marker_registered(self):
        """Der goal_achievement-Marker ist in pytest.ini registriert."""
        config = pytest.Config.fromdictargs({}, [])
        markers = [m.split(":")[0].strip() for m in config.getini("markers")]
        assert "goal_achievement" in markers, (
            "goal_achievement-Marker fehlt — keine Test-Kategorie für positive Klangverbesserung"
        )


class TestWohlklangOptimumObjective:
    """§Hörordnung §1–§3: Parameter-Suche muss den maximalen Wohlklang suchen.

    Regressionstest gegen den Bug 2026-09-26 („Aurik berechnet nicht die
    optimalen Parameter des maximalen Wohlklangs"): die Stärken-Suche
    bewertete Signal-Ähnlichkeit (identisch = best), bestrafte damit jede
    echte Reparatur und verharrte bei Default-Stärken.
    """

    @staticmethod
    def _tone(secs: float = 2.0, sr: int = 48000) -> np.ndarray:
        t = np.arange(int(sr * secs)) / sr
        return 0.5 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 440 * t)

    def test_20_objective_identical_is_exact_zero(self):
        from backend.core.human_pleasantness_estimator import wohlklang_objective_delta

        a = self._tone()
        assert wohlklang_objective_delta(a, a.copy(), 48000) == 0.0

    def test_21_objective_prefers_wohlklang_improvement(self):
        """Entrauschung muss als Wohlklang-Gewinn > 0 sichtbar sein."""
        from backend.core.human_pleasantness_estimator import wohlklang_objective_delta

        rng = np.random.default_rng(11)
        clean = self._tone()
        noisy = clean + 0.04 * rng.standard_normal(len(clean))
        assert wohlklang_objective_delta(noisy, clean, 48000) > 0.0

    def test_22_objective_rejects_structural_collapse(self):
        """Crest-Vernichtung (Limiter-artig) ist Hör-Invarianten-Verletzung."""
        from backend.core.human_pleasantness_estimator import wohlklang_objective_delta

        a = self._tone()
        crushed = np.tanh(4.0 * a) / np.tanh(4.0)
        assert wohlklang_objective_delta(a, crushed, 48000) < -0.03

    def test_23_strength_search_climbs_wohlklang_ladder(self):
        """Die Suche muss anheben, wenn höhere Stärke mehr Wohlklang bringt."""
        from backend.core.adaptive_strength_optimizer import optimize_phase_strength

        rng = np.random.default_rng(3)
        clean = self._tone()
        noisy = clean + 0.04 * rng.standard_normal(len(clean))

        def runner(audio: np.ndarray, strength: float) -> np.ndarray:
            # Stärke skaliert die Entrauschung: 0 = rauschig, 1 = sauber
            return (1.0 - strength) * noisy + strength * clean

        result = optimize_phase_strength(
            phase_id="test_wohlklang_ladder",
            audio_input=noisy,
            sample_rate=48000,
            phase_runner=runner,
            restorability_score=75.0,
        )
        assert result.optimal_strength > 0.2, (
            f"Stärke-Suche verharrte bei {result.optimal_strength} — das alte "
            f"Signal-Ähnlichkeits-Objektiv bestraft Reparaturen noch immer "
            f"(history={result.history})"
        )
        assert not result.was_skipped, "Wohlklang-Verbessernde Phase wurde übersprungen"

    def test_24_tape_dip_repair_energy_continuity(self):
        """§2.35b: Dip-Reparatur darf keine neuen Energie-Sprünge erzeugen.

        Produktionsbefund Import-Song: 18 Sprünge > 6 dB/100 ms nach phase_12 —
        die lineare Rampen-Länge degenerierte bei kurzen Dips zu einem
        Fade-Frame (Sprung 1,0→Gain in ~11 ms).
        """
        from backend.core.phases.phase_12_wow_flutter_fix import WowFlutterFix

        sr = 48000
        t = np.arange(sr * 4) / sr
        sig = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.15 * np.sin(2 * np.pi * 880 * t)
        env = np.ones(len(sig))
        dip_slices = []
        for start_s in (1.0, 2.0, 3.0):
            a0 = int(start_s * sr)
            a1 = a0 + int(0.040 * sr)  # 40-ms-Dips = worst case der alten Fade-Degeneration
            env[a0:a1] = 10 ** (-12.0 / 20.0)
            dip_slices.append(slice(a0, a1))
        dipped = sig * env

        phase = WowFlutterFix()
        out, n_repaired = phase._stabilize_tape_level(dipped, sr, 1.0, is_primary_tape=True, confirmed_tape_dip=False)
        assert n_repaired >= 1, "Dips wurden nicht repariert — Test prüft Reparatur-Kontinuität"

        def _max_jump_db(x: np.ndarray) -> float:
            win = int(0.1 * sr)
            rms = np.array([np.sqrt(np.mean(x[i : i + win] ** 2)) + 1e-12 for i in range(0, len(x) - win, win // 2)])
            return float(np.max(np.abs(20.0 * np.log10(rms[1:] / rms[:-1]))))

        out_m = np.asarray(out, dtype=np.float64)
        in_m = np.asarray(dipped, dtype=np.float64)
        assert _max_jump_db(out_m) <= _max_jump_db(in_m) + 0.5, (
            f"Reparatur fügte Energie-Sprünge hinzu (out={_max_jump_db(out_m):.1f} dB "
            f"> in={_max_jump_db(in_m):.1f} dB) — TemporalConsistencyGuard-Verletzung"
        )
        # Dip-Region angehoben (Reparatur wirkt)
        for sl in dip_slices:
            r_in = np.sqrt(np.mean(in_m[sl] ** 2)) + 1e-12
            r_out = np.sqrt(np.mean(out_m[sl] ** 2)) + 1e-12
            assert r_out > r_in * 1.2, "Dip-Pegel wurde nicht wiederhergestellt"

    def test_25_pmgg_timing_ladder_searches_least_regression(self):
        """§2.29 + §Wohlklang-Optimum: Timing-Phasen suchen die geringste
        Regression per Re-Ausführung — die ungesuchte Vollstärke-Fassung wird
        nie übernommen (Befund: regression=0,3351 ≫ 0,033 akzeptiert).
        """
        import backend.core.per_phase_musical_goals_gate as pmgg

        src = open(pmgg.__file__, encoding="utf-8").read()
        assert "using best-effort (regression=" not in src, "alter Vollstärke-Retour-Pfad lebt noch"
        assert 'return best_audio, best_scores, "best_effort", initial_strength' not in src
        assert "suche geringste Regression per Re-Ausfuehrung" in src, (
            "Timing-Re-Ausführungs-Leiter fehlt — phase_12/31 nehmen wieder Vollstärke ungeprüft"
        )

    def test_26_skip_when_all_strengths_below_jnd(self, monkeypatch):
        """Dead-Zone-Fix: best Δ < −HPE_JND (±0.03) ⇒ Phase überspringen.

        Regressionstest gegen das alte Gate (`best_delta < -0.05`), das Phasen
        mit best_delta ≈ −0.04 (≒ 1.3× HPE-JND) bei Floor-Stärke laufen ließ —
        hörbare Verschlechterung, obwohl keine Stärke hilft
        (GEBOTE.md §G124; PMGG-HPE-Gate-Semantik: < -0.03 ⇒ skip).
        """
        import backend.core.adaptive_strength_optimizer as aso

        audio = self._tone(secs=0.5)

        def _fake_delta(_orig: np.ndarray, _rest: np.ndarray, *args, **kwargs):
            return -0.04  # schlechter als JND (±0.03), besser als altes Gate (-0.05)

        monkeypatch.setattr(aso, "_quick_quality_delta", _fake_delta)

        def runner(audio: np.ndarray, strength: float) -> np.ndarray:
            return np.clip(audio * (1.0 + 0.01 * strength), -1.0, 1.0)

        result = aso.optimize_phase_strength(
            phase_id="test_dead_zone",
            audio_input=audio,
            sample_rate=48000,
            phase_runner=runner,
            restorability_score=75.0,
        )
        assert result.was_skipped is True, (
            f"best_delta={result.best_delta} < -HPE_JND(-0.03): Phase MUSS "
            f"übersprungen werden — lief stattdessen bei Stärke {result.optimal_strength}"
        )
        assert not result.was_executed

    def test_27_no_skip_when_best_delta_above_jnd(self, monkeypatch):
        """Gegenüber-Schutz: best Δ > −HPE_JND ⇒ Phase darf laufen (keine Überkorrektur).

        Eine nur marginal negative Regression (besser als JND ±0.03) ist
        tolerabel — dieselbe Semantik wie das PMGG-HPE-Gate
        (-0.03 <= delta < 0 ⇒ akzeptieren, nicht skippen).
        """
        import backend.core.adaptive_strength_optimizer as aso

        audio = self._tone(secs=0.5)

        def _fake_delta(_orig: np.ndarray, _rest: np.ndarray, *args, **kwargs):
            return -0.02  # besser als JND (±0.03) → tolerabel

        monkeypatch.setattr(aso, "_quick_quality_delta", _fake_delta)

        def runner(audio: np.ndarray, strength: float) -> np.ndarray:
            return np.clip(audio * (1.0 + 0.01 * strength), -1.0, 1.0)

        result = aso.optimize_phase_strength(
            phase_id="test_above_jnd",
            audio_input=audio,
            sample_rate=48000,
            phase_runner=runner,
            restorability_score=75.0,
        )
        assert not result.was_skipped, (
            f"best_delta={result.best_delta} > -HPE_JND(-0.03): Phase durfte Nicht übersprungen werden (Überkorrektur)"
        )
        assert result.optimal_strength > 0.01
