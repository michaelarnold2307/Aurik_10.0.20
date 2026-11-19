"""§SOTA-R8 (2026-09-15) — Sparse-Repair-Rollout phase_59: Tests.

Deckt die Defekt-Maske-als-Rechen-Maske-Verdrahtung in
``phase_59_modulation_noise_reduction``:
- mit defect_locations: sparse Pfad — ``sparse_repair``-Metadatum, Audio-Länge/Finite\n  erhalten, Defekt-Region repariert (regions_repaired ≥ 1 oder full_repair)\n- ohne defect_locations: Vollrepair-Fallback (coverage 1,0) — Verhalten wie vorher\n- sauberes Signal bleibt nahezu unverändert (Never-worsen)\n\nAutor: Aurik Testing Team\n"""

import numpy as np

from backend.core.defect_scanner import MaterialType
from backend.core.phases.phase_59_modulation_noise_reduction import ModulationNoiseReductionPhase


def _audio(n: int = 96000, amp: float = 0.2) -> np.ndarray:
    t = np.linspace(0, n / 48000, n, endpoint=False, dtype=np.float32)
    base = (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    # leises Modulationsrauschen im Band 200 Hz–8 kHz (wie NR-Atmen)
    noise = (0.01 * np.sin(2 * np.pi * 3000 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t))).astype(np.float32)
    return (base + noise).astype(np.float32)


class TestSparseRepairRollout59:
    def test_with_defect_locations_reports_sparse_metadata(self):
        x = _audio()
        ph = ModulationNoiseReductionPhase()
        res = ph.process(
            x,
            sample_rate=48000,
            material_type=MaterialType.VINYL,
            defect_locations={"modulation_noise": [(0.5, 0.7)]},
        )
        sp = res.metadata.get("sparse_repair")
        assert sp is not None, "phase_59 meldet kein sparse_repair-Metadatum"
        assert isinstance(sp.get("regions_repaired"), int)
        assert isinstance(sp.get("full_repair"), bool)
        assert sp["regions_repaired"] >= 1 or sp["full_repair"] is True
        assert np.isfinite(np.asarray(res.audio)).all()
        assert len(np.asarray(res.audio).ravel()) == len(x.ravel())

    def test_without_defect_locations_is_full_repair(self):
        x = _audio()
        ph = ModulationNoiseReductionPhase()
        res = ph.process(x, sample_rate=48000, material_type=MaterialType.VINYL)
        sp = res.metadata.get("sparse_repair")
        assert sp is not None
        # Ohne defect_locations ist die Lokalitäts-Maske all-ones ⇒ ein Vollrepair.
        assert sp["full_repair"] is True or sp["coverage"] >= 0.85

    def test_clean_signal_stays_close(self):
        x = _audio()
        ph = ModulationNoiseReductionPhase()
        res = ph.process(
            x,
            sample_rate=48000,
            material_type=MaterialType.VINYL,
            defect_locations={"modulation_noise": [(0.5, 0.7)]},
        )
        out = np.asarray(res.audio)
        delta_db = float(20.0 * np.log10(np.sqrt(np.mean(out**2)) / (np.sqrt(np.mean(x**2)) + 1e-12)))
        assert abs(delta_db) < 1.0

    def test_deterministic(self):
        x = _audio(48000)
        ph = ModulationNoiseReductionPhase()
        kw = {
            "sample_rate": 48000,
            "material_type": MaterialType.VINYL,
            "defect_locations": {"modulation_noise": [(0.3, 0.5)]},
        }
        r1 = ph.process(x, **kw)
        r2 = ph.process(x, **kw)
        assert np.array_equal(np.asarray(r1.audio), np.asarray(r2.audio))
