"""Tests des Song-Plan-Pins im PluginLifecycleManager (§G196 (GEBOTE.md), Hörordnung §8c Muster 5).

Der Pin schließt die D-K3-GoNoGo-Klasse: Unter RAM-/Swap-Druck evakuierte der
PLM auch Modelle des laufenden Song-Plans (Notfall-Pfad), betroffene Phasen
liefen über §V6-Ersatzpfade, der Klang kippte zwischen Läufen. Gepinnte
Modelle werden von JEDER Eviction (normal UND Notfall) verschont; der Reset
erfolgt am Song-Ende in force_evict_all (§V8 (copilot-instructions.md)).
"""

from __future__ import annotations

import backend.core.plugin_lifecycle_manager as plm


def _neu_manager() -> plm.PluginLifecycleManager:
    m = plm.PluginLifecycleManager.__new__(plm.PluginLifecycleManager)  # ohne Auto-Monitor-Thread
    import threading

    m._lock = threading.Lock()
    m._entries = {}
    m._pipeline_active = 0
    m._lookahead_models = frozenset()
    m._pinned_models = frozenset()
    m._song_window_active = False
    m._song_touched = set()
    m._last_swap_warn_ts = 0.0
    m._last_evict_was_empty = False
    return m


def test_pin_schuetzt_vor_evict_if_needed(monkeypatch) -> None:
    m = _neu_manager()
    entladen: list[str] = []
    m.register("ModellA", 1.0, lambda: entladen.append("ModellA"))
    m.register("ModellB", 1.0, lambda: entladen.append("ModellB"))
    m.pin_song_models({"ModellA"})

    # RAM-Druck simulieren: frei bleibt knapp → evict_if_needed evakuiert alles
    # außer gepinnten Modellen.
    monkeypatch.setattr(plm.PluginLifecycleManager, "_ram_percent", lambda self: 99.0)
    monkeypatch.setattr(plm.PluginLifecycleManager, "_free_mb", lambda self: 100.0)
    monkeypatch.setattr(plm.PluginLifecycleManager, "_swap_percent", lambda self: 0.0)

    n = m.evict_if_needed(required_mb=0.0)
    assert "ModellB" in entladen
    assert "ModellA" not in entladen
    assert n == 1


def test_pin_wirkt_auch_im_notfall_do_evict(monkeypatch) -> None:
    m = _neu_manager()
    entladen: list[str] = []
    m.register("Gepinnt", 1.0, lambda: entladen.append("Gepinnt"))
    m.register("Frei", 1.0, lambda: entladen.append("Frei"))
    m.pin_song_models({"Gepinnt"})

    monkeypatch.setattr(plm.PluginLifecycleManager, "_ram_percent", lambda self: 95.0)
    monkeypatch.setattr(plm.PluginLifecycleManager, "_free_mb", lambda self: 50.0)
    monkeypatch.setattr(plm.PluginLifecycleManager, "_swap_percent", lambda self: 90.0)

    m._do_evict(target_pct=0.0)
    assert "Frei" in entladen
    assert "Gepinnt" not in entladen


def test_force_evict_all_setzt_pin_zurueck() -> None:
    m = _neu_manager()
    m.register("ModellA", 1.0, lambda: None)
    m.pin_song_models({"ModellA"})
    assert m._pinned_models == frozenset({"ModellA"})
    m.force_evict_all()
    assert m._pinned_models == frozenset()
    assert "ModellA" not in m._entries  # nach Reset evakuiert


def test_evict_for_phase_respektiert_pin() -> None:
    m = _neu_manager()
    entladen: list[str] = []
    m.register("BanquetVinyl", 1.0, lambda: entladen.append("BanquetVinyl"))
    m.register("SGMSE+", 1.0, lambda: entladen.append("SGMSE+"))
    m.pin_song_models({"BanquetVinyl"})
    # phase_03_denoise braucht SGMSE+/DeepFilterNetV3/MIIPHER_DiT — BanquetVinyl wäre
    # ohne Pin Kandidat (nicht benötigt), mit Pin geschützt.
    m.evict_for_phase("phase_03_denoise")
    assert "BanquetVinyl" not in entladen


def test_unpin_gibt_modelle_frei() -> None:
    m = _neu_manager()
    entladen: list[str] = []
    m.register("ModellA", 1.0, lambda: entladen.append("ModellA"))
    m.pin_song_models({"ModellA"})
    m.unpin_song_models()
    assert m._pinned_models == frozenset()
    m._do_evict(target_pct=0.0, force_all=True)
    assert "ModellA" in entladen


def test_pin_on_touch_schuetzt_beruehrte_modelle(monkeypatch) -> None:
    """§G196-Ausbau: Modelle, die der Song berührt (touch/set_active), sind
    im Song-Fenster vor Eviction geschützt — auch ohne Plan-Eintrag."""
    m = _neu_manager()
    entladen: list[str] = []
    m.register("Teacher", 0.5, lambda: entladen.append("Teacher"))
    m.register("Stale", 0.5, lambda: entladen.append("Stale"))
    m.pin_song_models(set())  # Plan leer — alles Weitere kommt aus dem Touch
    m.touch("Teacher")

    monkeypatch.setattr(plm.PluginLifecycleManager, "_ram_percent", lambda self: 99.0)
    monkeypatch.setattr(plm.PluginLifecycleManager, "_free_mb", lambda self: 100.0)
    monkeypatch.setattr(plm.PluginLifecycleManager, "_swap_percent", lambda self: 0.0)

    m.evict_if_needed(required_mb=0.0)
    assert "Teacher" not in entladen  # berührt ⇒ geschützt
    assert "Stale" in entladen  # unberührt ⇒ evakuierbar (RAM-Entlastung bleibt)


def test_pin_on_touch_reset_am_song_ende(monkeypatch) -> None:
    """force_evict_all beendet das Song-Fenster — berührte Modelle werden
    danach wieder evakuierbar (§V8 (copilot-instructions.md)-Reset)."""
    m = _neu_manager()
    entladen: list[str] = []
    m.register("Teacher", 0.5, lambda: entladen.append("Teacher"))
    m.pin_song_models(set())
    m.touch("Teacher")
    m.force_evict_all()  # Song-Ende: Fenster zu, dann volle Eviction
    assert "Teacher" in entladen
    assert m._song_window_active is False
    assert m._song_touched == set()
