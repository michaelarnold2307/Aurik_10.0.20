"""§P1-3 (2026-10-06) — HR-V1-Budget-Vertrag (Längen-Deckel) und kanonische Aufrufstelle.

Belege die zwei Entscheidungen des Arbeitspakets 1c (Weg 2):
1. **Längen-Deckel**: HR-V1 synthetisiert höchstens ``BIGVGAN_V2_HR_MAX_DUTY``
   der Signallänge in gleichmäßig verteilten Ausschnitten; alles außerhalb
   bleibt BIT-IDENTISCH (keine Nahtkante, §G3/§V2 copilot-instructions.md).
2. **Eine Aufrufstelle**: nur ``phase_07_harmonic_restoration`` ruft den
   Helfer; 03/23/50 führen nur noch eine Audit-Spur (§G9 copilot-instructions.md).

Ohne diese Tests wäre die Budget-Ausnahme unbelegt und die Aufrufstellen
könnten unbemerkt zurückkehren.

Autor: Aurik Testing Team
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

import plugins.bigvgan_v2_plugin as bvg

SR = 48000
_REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Deckel-Arithmetik + Fensterwahl (deterministisch, §G5 copilot-instructions.md)
# ---------------------------------------------------------------------------


class TestBudgetDeckel:
    def test_floor_for_short_signals(self):
        assert bvg.hr_v1_budget_seconds(1.0) == pytest.approx(bvg.BIGVGAN_V2_HR_MIN_SECONDS)

    def test_proportional_in_the_middle(self):
        # 30 s × 5 % = 1,5 s — zwischen Floor und Ceiling
        assert bvg.hr_v1_budget_seconds(30.0) == pytest.approx(1.5)

    def test_ceiling_for_long_signals(self):
        assert bvg.hr_v1_budget_seconds(600.0) == pytest.approx(bvg.BIGVGAN_V2_HR_MAX_SECONDS)

    def test_degenerate_inputs(self):
        assert bvg.hr_v1_budget_seconds(0.0) == 0.0
        assert bvg.hr_v1_budget_seconds(-3.0) == 0.0

    def test_budget_never_exceeds_duty_or_ceiling(self):
        for duration in (0.1, 1.0, 7.0, 30.0, 120.0, 600.0, 3600.0):
            budget = bvg.hr_v1_budget_seconds(duration)
            assert budget <= bvg.BIGVGAN_V2_HR_MAX_SECONDS
            assert budget <= max(bvg.BIGVGAN_V2_HR_MIN_SECONDS, duration * bvg.BIGVGAN_V2_HR_MAX_DUTY) + 1e-9


class TestWindowSelection:
    @staticmethod
    def _spans(duration_s: float) -> list[tuple[int, int]]:
        n = int(duration_s * SR)
        return bvg.select_hr_v1_windows(n, SR, bvg.hr_v1_budget_seconds(duration_s))

    def test_deterministic_and_in_bounds(self):
        n = 30 * SR
        budget = bvg.hr_v1_budget_seconds(30.0)
        first = bvg.select_hr_v1_windows(n, SR, budget)
        second = bvg.select_hr_v1_windows(n, SR, budget)
        assert first == second, "Fensterwahl muss deterministisch sein (§G5 copilot-instructions.md)"
        assert first, "Fenster müssen existieren"
        for s, e in first:
            assert 0 <= s < e <= n

    def test_sorted_and_non_overlapping(self):
        spans = self._spans(224.0)
        assert spans == sorted(spans)
        for (_, e_prev), (s_next, _) in itertools.pairwise(spans):
            assert s_next >= e_prev, "Fenster dürfen sich nicht überlappen"

    def test_budget_fully_used(self):
        for duration in (10.0, 30.0, 60.0, 224.0):
            spans = self._spans(duration)
            used = sum(e - s for s, e in spans) / SR
            budget = bvg.hr_v1_budget_seconds(duration)
            assert used <= budget + 1e-6, "Deckel darf nie überschritten werden"
            assert used >= budget * 0.99, "Deckel muss ausgeschöpft werden"

    def test_spread_over_the_whole_signal(self):
        n = 224 * SR
        spans = self._spans(224.0)
        assert len(spans) >= 2
        assert spans[0][0] == 0, "erster Ausschnitt beginnt am Signalstart"
        assert spans[-1][1] == n, "letzter Ausschnitt endet am Signalende"

    def test_degenerate_inputs(self):
        assert bvg.select_hr_v1_windows(0, SR, 1.0) == []
        assert bvg.select_hr_v1_windows(SR, SR, 0.0) == []
        assert bvg.select_hr_v1_windows(SR, 0, 1.0) == []

    def test_short_signal_uses_single_window(self):
        spans = self._spans(1.0)
        assert len(spans) == 1
        s, e = spans[0]
        assert e - s == int(round(bvg.BIGVGAN_V2_HR_MIN_SECONDS * SR))


class TestRampMask:
    def test_zero_outside_and_no_hard_edge(self):
        n = 30 * SR
        spans = bvg.select_hr_v1_windows(n, SR, bvg.hr_v1_budget_seconds(30.0))
        mask = bvg._hr_v1_ramp_mask(n, spans, SR)
        assert mask.max() <= 1.0 + 1e-6
        covered = np.zeros(n, dtype=bool)
        for s, e in spans:
            covered[s:e] = True
        assert np.all(mask[~covered] == 0.0), "außerhalb der Ausschnitte keine Wirkung"
        # Keine Kante: der Sprung zwischen Nachbar-Samples bleibt klein
        # (200-ms-Cosinus über 9600 Samples ⇒ Δ ≪ 0,01).
        assert float(np.max(np.abs(np.diff(mask)))) < 0.01
        # Plateau vorhanden (Rampe frisst nicht den ganzen Ausschnitt).
        assert float(mask.max()) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# End-to-End des Helfers (ohne Checkpoint, ohne ONNX — Stubs)
# ---------------------------------------------------------------------------


class _StubVoc:
    """Minimaler VocoderResult-Ersatz mit gleicher Länge wie der Ausschnitt."""

    def __init__(self, audio: np.ndarray, *, model_used: str = "bigvgan_v2", pqs: float = 3.9) -> None:
        self.audio = audio
        self.model_used = model_used
        self.pqs_mos = pqs


@pytest.fixture()
def _ready(monkeypatch: pytest.MonkeyPatch) -> None:
    """HR-V1 als bereit markieren, ohne den (gitignorierten) Checkpoint zu brauchen."""
    monkeypatch.setattr(bvg, "bigvgan_v2_ready", lambda: True)


@pytest.fixture()
def _release(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Stub-Gate, das 3 Bänder freigibt; Rückgabe = Liste der Syntheselängen."""
    lengths: list[int] = []

    def _gate(
        candidate: np.ndarray, baseline: np.ndarray, sample_rate: int, **_kwargs: object
    ) -> tuple[np.ndarray, dict[str, object]]:
        # Annotierte Typ-Grenze (Repo-Muster): numpy-Stubs liefern hier Any.
        scaled: np.ndarray = np.asarray(baseline, dtype=np.float32) * np.float32(1.5)
        out: np.ndarray = scaled
        report: dict[str, object] = {"bands_released": 3, "model": "bigvgan_v2"}
        return out, report

    import backend.core.dsp.additive_synthesis_gate as gate_mod

    monkeypatch.setattr(gate_mod, "additive_synthesis_gate", _gate)

    def _synth(span_audio, _sr, **_kwargs):  # type: ignore[no-untyped-def]
        lengths.append(int(np.asarray(span_audio).size))
        return _StubVoc(np.asarray(span_audio, dtype=np.float32))

    monkeypatch.setattr(bvg, "synthesize_audio", _synth)
    return lengths


class TestApplyHrV1Budget:
    @staticmethod
    def _signal(duration_s: float = 30.0) -> np.ndarray:
        t = np.arange(int(duration_s * SR)) / SR
        wave: np.ndarray = 0.25 * np.sin(2 * np.pi * 440.0 * t)
        out: np.ndarray = np.asarray(wave, dtype=np.float32)
        return out

    def test_synthesized_seconds_bounded_by_budget(self, _ready, _release):
        x = self._signal(30.0)
        _, meta = bvg.apply_hr_v1_additive(x, SR)
        budget = bvg.hr_v1_budget_seconds(30.0)
        windows = meta["windows"]
        processed = meta["processed_seconds"]
        assert isinstance(windows, int)
        assert isinstance(processed, float)
        # Guard-Padding erlaubt Zusatzlänge, aber nur um _HR_GUARD_SECONDS je Fenster.
        pad = bvg._HR_GUARD_SECONDS * SR
        synthesized_s = sum(_release) / SR
        assert synthesized_s <= budget + windows * (2 * pad / SR) + 1e-6
        assert processed <= budget + 1e-6
        assert meta["capped"] is True

    def test_outside_windows_bit_identical(self, _ready, _release):
        x = self._signal(30.0)
        y, meta = bvg.apply_hr_v1_additive(x, SR)
        assert meta["applied"] is True
        spans = bvg.select_hr_v1_windows(len(x), SR, bvg.hr_v1_budget_seconds(30.0))
        inside = np.zeros(len(x), dtype=bool)
        for s, e in spans:
            inside[s:e] = True
        assert np.array_equal(y[~inside], x[~inside]), "außerhalb der Ausschnitte darf nichts passieren"
        assert not np.array_equal(y[inside], x[inside]), "in den Ausschnitten muss etwas passieren"

    def test_deterministic_output(self, _ready, _release):
        x = self._signal(30.0)
        y1, _ = bvg.apply_hr_v1_additive(x, SR)
        y2, _ = bvg.apply_hr_v1_additive(x, SR)
        assert np.array_equal(y1, y2), "gleicher Input ⇒ gleicher Output (§G5 copilot-instructions.md)"

    def test_model_used_none_changes_nothing(self, _ready, monkeypatch: pytest.MonkeyPatch):
        x = self._signal(30.0)
        monkeypatch.setattr(bvg, "synthesize_audio", lambda span, _sr, **_: _StubVoc(span * 0, model_used="none"))

        y, meta = bvg.apply_hr_v1_additive(x, SR)
        assert meta["attempted"] is True
        assert meta["applied"] is False
        assert np.array_equal(y, x)

    def test_length_mismatch_fails_closed(self, _ready, monkeypatch: pytest.MonkeyPatch):
        """Synthese-Länge ≠ Eingabe ⇒ verwerfen + melden (§V6 copilot-instructions.md), nichts mischen."""
        x = self._signal(30.0)
        monkeypatch.setattr(
            bvg,
            "synthesize_audio",
            lambda span, _sr, **_: _StubVoc(np.zeros(1234, dtype=np.float32)),
        )
        y, meta = bvg.apply_hr_v1_additive(x, SR)
        skipped = meta["skipped_length_mismatch"]
        assert isinstance(skipped, int)
        assert skipped >= 1
        assert meta["applied"] is False
        assert np.array_equal(y, x), "Längen-Mismatch darf nie mischen"

    def test_synthesis_failure_fails_closed(self, _ready, monkeypatch: pytest.MonkeyPatch):
        x = self._signal(30.0)

        def _raise(*_a, **_k):
            raise RuntimeError("simulierter Synthese-Fehler")

        monkeypatch.setattr(bvg, "synthesize_audio", _raise)
        y, meta = bvg.apply_hr_v1_additive(x, SR)
        assert meta["attempted"] is True
        assert meta["applied"] is False
        assert np.array_equal(y, x)
        assert np.isfinite(np.asarray(y)).all()

    def test_not_ready_is_bit_identical(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(bvg, "bigvgan_v2_ready", lambda: False)
        x = self._signal(5.0)
        y, meta = bvg.apply_hr_v1_additive(x, SR)
        assert meta["attempted"] is False
        assert y is x, "ohne Freigabe darf kein Kopie-/Rechenpfad entstehen"


# ---------------------------------------------------------------------------
# Eine Aufrufstelle (§G9 copilot-instructions.md) — Guard gegen Rückkehr der fünf Aufrufe
# ---------------------------------------------------------------------------


class TestSingleCanonicalCallSite:
    def test_only_phase_07_imports_the_helper(self):
        """§P1-3: Der Helfer wird ausschließlich von phase_07 gerufen."""
        callers: list[str] = []
        for root in (_REPO / "backend", _REPO / "denker", _REPO / "Aurik10", _REPO / "cli", _REPO / "plugins"):
            for path in root.rglob("*.py"):
                text = path.read_text(encoding="utf-8", errors="ignore")
                if "apply_hr_v1_additive" in text:
                    callers.append(str(path.relative_to(_REPO)))
        assert sorted(callers) == sorted(
            [
                "backend/core/phases/phase_07_harmonic_restoration.py",
                "plugins/bigvgan_v2_plugin.py",  # Definition
            ]
        ), "HR-V1 darf genau EINE Aufrufstelle haben (§G9 copilot-instructions.md); gefunden: " + repr(sorted(callers))

    def test_non_canonical_phases_report_audit_witness(self):
        """03/23/50 verschwinden nicht still: sie melden die Zentralisierung."""
        for rel in (
            "backend/core/phases/phase_03_denoise.py",
            "backend/core/phases/phase_23_spectral_repair.py",
            "backend/core/phases/phase_50_spectral_repair.py",
        ):
            text = (_REPO / rel).read_text(encoding="utf-8")
            assert '"reason": "centralized_g9"' in text, f"{rel} meldet die Zentralisierung nicht"
            assert "phase_07_harmonic_restoration" in text, f"{rel} nennt die kanonische Stelle nicht"
