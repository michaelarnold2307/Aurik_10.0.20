"""Tests für backend/core/defect_audibility_gate.py (Hörbarkeits-Gate, m1).

Hörordnung Ebene 2 (§4): „Reparatur gilt als abgeschlossen, wenn ein Defekt
unter der Maskierungsschwelle liegt — nicht wenn sein Messwert Null ist."
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core.defect_audibility_gate import (
    AUDIBLE_BASE,
    MATERIAL_JND_OFFSET,
    DefectAudibilityReport,
    audible_threshold,
    canonical_audibility_verdicts,
    evaluate_defect_audibility,
)


def _entry(pre: float, post: float, masked: int = 0) -> dict:
    return {
        "pre": pre,
        "post": post,
        "reduction": round(max(0.0, pre - post), 4),
        "reduction_pct": round(max(0.0, pre - post) / max(pre, 0.001) * 100, 1),
        "masked_events": masked,
    }


class TestAudibleThreshold:
    def test_vinyl_base(self) -> None:
        assert audible_threshold("vinyl", 1) == pytest.approx(0.06)  # 0.08 - 0.02

    def test_cassette_higher_floor(self) -> None:
        assert audible_threshold("cassette", 1) == pytest.approx(0.12)

    def test_mp3_low_offset(self) -> None:
        assert audible_threshold("mp3_low", 1) == pytest.approx(0.13)

    def test_chain_depth_adds_masking(self) -> None:
        assert audible_threshold("vinyl", 3) == pytest.approx(0.08)

    def test_clipping_bounds(self) -> None:
        assert audible_threshold("cassette", 5) <= 0.15
        assert audible_threshold("vinyl", 1) >= 0.03

    def test_unknown_material_neutral(self) -> None:
        assert audible_threshold("banana", 1) == pytest.approx(AUDIBLE_BASE)

    def test_offset_table_complete(self) -> None:
        # In unified_restorer_v3.py wird die Tabelle aus diesem Modul importiert —
        # sie muss die historischen Materialien weiterhin abdecken.
        for mat in ("cassette", "vinyl", "shellac", "reel_tape", "cd_digital", "mp3_low", "aac"):
            assert mat in MATERIAL_JND_OFFSET


class TestEvaluateAudibility:
    def test_all_resolved_gate_passes(self) -> None:
        data = {"clicks": _entry(0.5, 0.02), "hiss": _entry(0.4, 0.03)}
        rep = evaluate_defect_audibility(data, material_key="vinyl")
        assert rep.gate_passed is True
        assert rep.n_resolved == 2
        assert rep.n_audible_unmasked == 0

    def test_audible_remaining_fails_gate(self) -> None:
        data = {"clicks": _entry(0.5, 0.20)}  # post 0.20 > thr 0.06, keine Maskierung
        rep = evaluate_defect_audibility(data, material_key="vinyl")
        assert rep.gate_passed is False
        assert rep.n_audible_unmasked == 1
        assert rep.improvable_types == ["clicks"]

    def test_masked_events_cover_residual(self) -> None:
        # post über Schwelle, aber ERB-Masking hat Events abgedeckt und post ist
        # moderat → gilt als abgedeckt, Gate besteht.
        data = {"clicks": _entry(0.5, 0.15, masked=12)}
        rep = evaluate_defect_audibility(data, material_key="vinyl")
        assert rep.gate_passed is True
        assert rep.n_masked == 1

    def test_masked_but_very_loud_still_fails(self) -> None:
        # Maskierte Events hin oder her: post >= 0.35 bleibt sicher exponiert.
        data = {"clicks": _entry(0.8, 0.60, masked=3)}
        rep = evaluate_defect_audibility(data, material_key="vinyl")
        assert rep.gate_passed is False
        assert rep.n_audible_unmasked == 1

    def test_physical_cap_types_accepted(self) -> None:
        data = {"bandwidth_loss": _entry(0.7, 0.35)}
        rep = evaluate_defect_audibility(data, material_key="mp3_low", physical_cap_types={"codec_artifacts"})
        assert rep.gate_passed is True
        assert rep.n_physical_cap == 1

    def test_never_audible_ignored(self) -> None:
        data = {"wow": _entry(0.02, 0.01)}
        rep = evaluate_defect_audibility(data, material_key="vinyl")
        assert rep.gate_passed is True
        assert rep.n_never_audible == 1

    def test_empty_and_malformed(self) -> None:
        # §G8 (copilot-instructions.md): fail-open fuer den BLOCK, aber KEIN
        # Nachweis — ohne Post-Scan ist der Evidenzstand "scan_missing".
        rep = evaluate_defect_audibility(None, material_key="vinyl")
        assert rep.gate_passed is True
        assert rep.evidence_state == "scan_missing"
        assert rep.gate_verified is False
        rep2 = evaluate_defect_audibility({"x": "kaputt", "y": {"pre": np.nan, "post": None}}, material_key="vinyl")
        assert rep2.gate_passed is True
        assert rep2.gate_verified is False

    def test_scan_missing_is_unverified(self) -> None:
        """Ein bestandener Block ohne Messung ist kein Nachweis — §G8 und §V6 (copilot-instructions.md)."""
        rep = evaluate_defect_audibility({"clicks": _entry(0.5, 0.02)}, material_key="vinyl", post_scan_ran=False)
        assert rep.gate_passed is True  # kein hörbarer Restdefekt behauptet
        assert rep.gate_verified is False  # ...aber auch nicht belegt
        assert rep.evidence_state == "scan_missing"
        assert rep.to_metadata()["gate_verified"] is False
        assert rep.to_metadata()["evidence_state"] == "scan_missing"

    def test_no_residual_defects_after_real_scan_is_verified(self) -> None:
        # Post-Scan lief, fand nichts Ueber-der-Schwelle-Liegendes -> bestanden.
        rep = evaluate_defect_audibility({}, material_key="vinyl", post_scan_ran=True)
        assert rep.evidence_state == "no_residual_defects"
        assert rep.gate_verified is True

    def test_evaluated_scan_is_verified(self) -> None:
        rep = evaluate_defect_audibility({"clicks": _entry(0.5, 0.02)}, material_key="vinyl", post_scan_ran=True)
        assert rep.evidence_state == "evaluated"
        assert rep.n_verified == 1
        assert rep.gate_verified is True

    def test_failed_gate_is_never_verified(self) -> None:
        rep = evaluate_defect_audibility({"clicks": _entry(0.5, 0.20)}, material_key="vinyl", post_scan_ran=True)
        assert rep.gate_passed is False
        assert rep.gate_verified is False

    def test_report_metadata_jsonable(self) -> None:
        data = {"clicks": _entry(0.5, 0.20)}
        rep = evaluate_defect_audibility(data, material_key="vinyl")
        meta = rep.to_metadata()
        assert meta["gate_passed"] is False
        assert meta["improvable_types"] == ["clicks"]
        assert isinstance(meta["per_type"]["clicks"]["pre"], float)


class TestCanonicalConsolidation:
    """Konsistenz-Slice 2 (2026-09-27): eine Hör-Instanz, eine Wahrheit.

    Der kanonische Pfad nutzt DIESELBE Maskierungs-Instanz wie die Phasen-Gates
    (dsp/audibility_gate.defect_audibility_from_signal). Aufbau: 2-kHz-Masker
    (Amplitude 0.3) im ganzen Signal; Defekt-Location = 2-kHz-Burst in der
    Location. Gleiche kritische Bande ⇒ starke Maskierung: 0.01-Burst bleibt
    unter der Schwelle (masked), 0.6-Burst liegt darüber (audible).
    """

    SR = 48000

    @staticmethod
    def _masked_burst_signal(amp: float) -> np.ndarray:
        n = int(1.5 * TestCanonicalConsolidation.SR)
        t = np.arange(n) / TestCanonicalConsolidation.SR
        audio = 0.3 * np.sin(2.0 * np.pi * 2000.0 * t)
        d0, d1 = int(1.0 * TestCanonicalConsolidation.SR), int(1.03 * TestCanonicalConsolidation.SR)
        audio[d0:d1] = amp * np.sin(2.0 * np.pi * 2000.0 * t[d0:d1])
        return audio.astype(np.float32)

    @staticmethod
    def _candidate_data() -> dict:
        # post 0.2 >= thr 0.06 (vinyl) ⇒ Severity-Kandidat für den kanonischen Pfad.
        return {"clicks": _entry(0.5, 0.20)}

    def test_loud_burst_audible_canonical(self) -> None:
        # Kalibriert (2026-09-27): Maskierungs-Schwelle des 0.3-Maskers liegt bei
        # ~38 dB (Modell-Einheiten); Burst 1.0 (~42 dB) liegt ~4 dB darüber,
        # Burst 0.01 (~2 dB) ~36 dB darunter - robuste Trennung.
        audio = self._masked_burst_signal(1.0)
        rep = evaluate_defect_audibility(
            self._candidate_data(),
            material_key="vinyl",
            audio=audio,
            sample_rate=self.SR,
            defect_locations={"clicks": [(1.0, 1.03)]},
        )
        assert rep.gate_passed is False
        assert rep.n_audible_unmasked == 1
        pt = rep.per_type["clicks"]
        assert pt["status"] == "audible"
        assert pt["evidence"] == "canonical_masking"
        assert pt["canon_audible"] is True
        assert pt["canon_checked"] == 1

    def test_quiet_burst_masked_canonical(self) -> None:
        audio = self._masked_burst_signal(0.01)
        rep = evaluate_defect_audibility(
            self._candidate_data(),
            material_key="vinyl",
            audio=audio,
            sample_rate=self.SR,
            defect_locations={"clicks": [(1.0, 1.03)]},
        )
        assert rep.gate_passed is True
        assert rep.n_masked == 1
        assert rep.per_type["clicks"]["status"] == "masked"
        assert rep.per_type["clicks"]["evidence"] == "canonical_masking"
        assert rep.per_type["clicks"]["canon_audible"] is False

    def test_physical_cap_wins_over_canonical(self) -> None:
        # bandwidth_loss ist PHYSICAL_CAP: auch ein kanonisch hörbarer Rest
        # wird als „erfüllt mit Dokumentation“ akzeptiert (Bestandsverhalten).
        audio = self._masked_burst_signal(1.0)
        data = {"bandwidth_loss": _entry(0.7, 0.35)}
        rep = evaluate_defect_audibility(
            data,
            material_key="mp3_low",
            audio=audio,
            sample_rate=self.SR,
            defect_locations={"bandwidth_loss": [(1.0, 1.03)]},
        )
        assert rep.gate_passed is True
        assert rep.n_physical_cap == 1
        assert rep.per_type["bandwidth_loss"]["status"] == "physical_cap"
        assert rep.per_type["bandwidth_loss"]["evidence"] == "canonical_masking"

    def test_stereo_layout_equals_mono_verdict(self) -> None:
        mono = self._masked_burst_signal(0.01)
        stereo = np.stack([mono, mono], axis=1)  # (N,2)-Layout
        rep = evaluate_defect_audibility(
            self._candidate_data(),
            material_key="vinyl",
            audio=stereo,
            sample_rate=self.SR,
            defect_locations={"clicks": [(1.0, 1.03)]},
        )
        assert rep.per_type["clicks"]["status"] == "masked"
        # (2,N)-Layout ebenso (Stereo-Layout-Invariante, AGENTS.md §3).
        rep_cn = evaluate_defect_audibility(
            self._candidate_data(),
            material_key="vinyl",
            audio=np.stack([mono, mono], axis=0),
            sample_rate=self.SR,
            defect_locations={"clicks": [(1.0, 1.03)]},
        )
        assert rep_cn.per_type["clicks"]["status"] == "masked"

    def test_no_locations_severity_fallback(self) -> None:
        audio = self._masked_burst_signal(0.01)
        rep = evaluate_defect_audibility(
            self._candidate_data(),
            material_key="vinyl",
            audio=audio,
            sample_rate=self.SR,
            defect_locations={},
        )
        assert rep.per_type["clicks"]["status"] == "audible"  # Severity-Pfad
        assert rep.per_type["clicks"]["evidence"] == "severity_scale"

    def test_fm_types_stay_severity_scale(self) -> None:
        # Wow/Flutter haben keine Energie-Delta-Domäne ⇒ kanonisches Band = None.
        audio = self._masked_burst_signal(0.01)
        data = {"wow": _entry(0.5, 0.20)}
        rep = evaluate_defect_audibility(
            data,
            material_key="vinyl",
            audio=audio,
            sample_rate=self.SR,
            defect_locations={"wow": [(1.0, 1.03)]},
        )
        assert rep.per_type["wow"]["evidence"] == "severity_scale"

    def test_fail_open_on_bad_sample_rate(self) -> None:
        audio = self._masked_burst_signal(0.01)
        rep = evaluate_defect_audibility(
            self._candidate_data(),
            material_key="vinyl",
            audio=audio,
            sample_rate=0,
            defect_locations={"clicks": [(1.0, 1.03)]},
        )
        assert rep.per_type["clicks"]["evidence"] == "severity_scale"

    def test_determinism_bit_identical(self) -> None:
        audio = self._masked_burst_signal(0.01)
        kw = {
            "audio": audio,
            "sample_rate": self.SR,
            "defect_locations": {"clicks": [(1.0, 1.03)]},
        }
        meta_a = evaluate_defect_audibility(self._candidate_data(), material_key="vinyl", **kw).to_metadata()
        meta_b = evaluate_defect_audibility(self._candidate_data(), material_key="vinyl", **kw).to_metadata()
        assert meta_a == meta_b

    def test_canonical_verdicts_fm_returns_empty(self) -> None:
        audio = self._masked_burst_signal(0.01)
        out = canonical_audibility_verdicts(audio, self.SR, {"wow": [(1.0, 1.03)]}, types=["wow"])
        assert out == {}
