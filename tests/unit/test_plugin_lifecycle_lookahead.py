from __future__ import annotations

"""Unit-Tests für §Perf Look-Ahead-Eviction im PluginLifecycleManager.

Validiert, dass die Look-Ahead-Logik Reload-Thrashing eliminiert, indem Modelle
geschützt werden, die eine *baldige* Phase im aktuellen Pipeline-Fenster erneut braucht
— OHNE das restaurierte Audio, die Phasenreihenfolge oder eine Messung zu berühren.

Die Tests bauen den Manager direkt (kein Singleton-Interferenz) mit Fake-Plugins,
deren unload_fn nur ein Flag setzt — keine echten ML-Modelle, keine Audio-Verarbeitung.
"""


import pytest

from backend.core.plugin_lifecycle_manager import PluginLifecycleManager


def _make_manager() -> PluginLifecycleManager:
    mgr = PluginLifecycleManager()
    # Auto-Eviction-Monitor unterdrücken: Pipeline-aktiv markieren, damit der
    # Hintergrund-Thread während des Tests nicht spontan evictet.
    mgr.enter_pipeline()
    return mgr


def _register_fake(mgr: PluginLifecycleManager, name: str, unloaded: dict[str, bool]) -> None:
    unloaded[name] = False

    def _unload() -> None:
        unloaded[name] = True

    mgr.register(name, size_gb=1.0, unload_fn=_unload)


@pytest.mark.unit
class TestEvictForPhaseWindow:
    """evict_for_phase_window schützt Modelle aller anstehenden Phasen."""

    def test_protects_model_needed_by_later_phase(self, monkeypatch) -> None:
        """AudioSR (phase_06 → phase_23) darf vor phase_06 NICHT entladen werden.

        §v10.742: Proaktive Look-Ahead-Entladung läuft nur bei echtem
        Speicherdruck (≥ 75 % des ML-Limits) — ohne Druck gilt Residency.
        """
        mgr = _make_manager()
        unloaded: dict[str, bool] = {}
        try:
            import backend.core.ml_memory_budget as _mbb

            monkeypatch.setattr(_mbb, "_total_gb", 9.0)
            monkeypatch.setattr(_mbb, "ML_MAX_GB", 10.0)  # 90 % ≥ 75 % → Druck
            _register_fake(mgr, "AudioSR", unloaded)
            _register_fake(mgr, "DemucsV4", unloaded)  # von keiner Fensterphase gebraucht
            # Fenster: phase_06 (AudioSR) → DSP → phase_23 (Apollo, AudioSR).
            window = [
                "phase_06_frequency_restoration",
                "phase_16_final_eq",
                "phase_23_spectral_repair",
            ]
            mgr.evict_for_phase_window(window)
            assert unloaded["AudioSR"] is False, "AudioSR von phase_23 gebraucht → nicht entladen"
            assert unloaded["DemucsV4"] is True, "DemucsV4 von keiner Fensterphase gebraucht → entladen"
        finally:
            mgr.leave_pipeline()
            mgr.shutdown()

    def test_empty_window_is_noop(self) -> None:
        mgr = _make_manager()
        unloaded: dict[str, bool] = {}
        try:
            _register_fake(mgr, "AudioSR", unloaded)
            assert mgr.evict_for_phase_window([]) == 0
            assert unloaded["AudioSR"] is False
        finally:
            mgr.leave_pipeline()
            mgr.shutdown()

    def test_active_model_never_evicted(self) -> None:
        """Ein aktives (in Inferenz befindliches) Modell bleibt geschützt."""
        mgr = _make_manager()
        unloaded: dict[str, bool] = {}
        try:
            _register_fake(mgr, "DemucsV4", unloaded)
            mgr.set_active("DemucsV4", True)
            mgr.evict_for_phase_window(["phase_06_frequency_restoration"])
            assert unloaded["DemucsV4"] is False, "aktives Modell darf nicht entladen werden"
        finally:
            mgr.leave_pipeline()
            mgr.shutdown()


class TestPhaseInternalInheritsWindow:
    """Phasen-interne evict_for_phase()-Calls erben das gespeicherte Look-Ahead-Fenster."""

    def test_phase_internal_call_respects_stored_window(self) -> None:
        """Nach evict_for_phase_window darf evict_for_phase('phase_03') AudioSR NICHT entladen,
        obwohl phase_03 selbst AudioSR nicht braucht — phase_06 im Fenster braucht es."""
        mgr = _make_manager()
        unloaded: dict[str, bool] = {}
        try:
            _register_fake(mgr, "AudioSR", unloaded)
            _register_fake(mgr, "SGMSE+", unloaded)
            # Orchestrator setzt Fenster: aktuelle phase_03 → phase_06 (AudioSR).
            mgr.evict_for_phase_window(["phase_03_denoise", "phase_06_frequency_restoration"])
            assert unloaded["AudioSR"] is False  # vom Fenster (phase_06) geschützt
            # Jetzt der phasen-interne Call von phase_03 — ohne Fenster-Vererbung würde er
            # AudioSR entladen (phase_03 braucht es nicht). Mit Vererbung bleibt es resident.
            mgr.evict_for_phase("phase_03_denoise")
            assert unloaded["AudioSR"] is False, (
                "phasen-interner evict_for_phase darf AudioSR nicht entladen — "
                "phase_06 im Look-Ahead-Fenster braucht es noch"
            )
        finally:
            mgr.leave_pipeline()
            mgr.shutdown()

    def test_window_cleared_on_leave_pipeline(self) -> None:
        """Nach Pipeline-Ende ist das Fenster leer → Originalverhalten (nur eigenes Modell)."""
        mgr = PluginLifecycleManager()
        unloaded: dict[str, bool] = {}
        try:
            mgr.enter_pipeline()
            _register_fake(mgr, "AudioSR", unloaded)
            mgr.evict_for_phase_window(["phase_06_frequency_restoration"])
            mgr.leave_pipeline()  # Pipeline-Refcount 0 → Fenster zurückgesetzt
            # Standalone-Call von phase_03 (braucht AudioSR nicht) → AudioSR jetzt entladen.
            mgr.evict_for_phase("phase_03_denoise")
            assert unloaded["AudioSR"] is True, "nach leave_pipeline kein veraltetes Fenster mehr"
        finally:
            mgr.shutdown()

    def test_force_evict_all_clears_window(self) -> None:
        mgr = _make_manager()
        unloaded: dict[str, bool] = {}
        try:
            _register_fake(mgr, "AudioSR", unloaded)
            mgr.evict_for_phase_window(["phase_06_frequency_restoration"])
            mgr.force_evict_all()
            assert unloaded["AudioSR"] is True, "force_evict_all entlädt alles + verwirft Fenster"
            # Fenster verworfen: ein erneuter Standalone-Call findet ohnehin nichts mehr.
            assert mgr._lookahead_models == frozenset()
        finally:
            mgr.leave_pipeline()
            mgr.shutdown()


class TestBackwardCompatibility:
    """Ohne gesetztes Fenster verhält sich evict_for_phase exakt wie zuvor."""

    def test_evict_for_phase_without_window_unchanged(self) -> None:
        mgr = _make_manager()
        unloaded: dict[str, bool] = {}
        try:
            _register_fake(mgr, "AudioSR", unloaded)
            _register_fake(mgr, "DeepFilterNetV3", unloaded)
            # Kein Fenster gesetzt. phase_03 braucht DFN, nicht AudioSR.
            mgr.evict_for_phase("phase_03_denoise")
            assert unloaded["DeepFilterNetV3"] is False, "phase_03 braucht DFN → behalten"
            assert unloaded["AudioSR"] is True, "phase_03 braucht AudioSR nicht → entladen (Originalverhalten)"
        finally:
            mgr.leave_pipeline()
            mgr.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
