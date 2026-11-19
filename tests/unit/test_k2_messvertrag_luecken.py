from __future__ import annotations

import logging
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from backend.core.perceptual_quality_scorer import PerceptualQualityScorer
from plugins.vocos_plugin import VocosPlugin

"""K2-Messvertrags-Lücken D-K2-2 / D-K2-3 / D-K2-5 (SOTA-Defizit-Register).

Prüft die Verträge, die über den K2-Signalpfad (Sprach-Vocoder auf Musik)
entscheiden — §V6 (copilot-instructions.md)-Gate, Ladepriorität, PQS-Timing
und die D-K2-5-Verdrahtungs-Invariante. Ausführung durch den Manager
(Worker-Env ohne Shell):
    nice -n 15 python3 -B -m pytest tests/unit/test_k2_messvertrag_luecken.py -q
    ruff --select F821,F601,B009,I001 tests/unit/test_k2_messvertrag_luecken.py
"""


# ---------------------------------------------------------------------------
# D-K2-2 — §V6 (copilot-instructions.md)-Gate (WP-1: „keine unflagged
# Sprach-Vocoder-Synthese auf Musik")
# ---------------------------------------------------------------------------


def test_restoration_modus_sperre_keine_sprach_vocoder_synthese_auf_musik(
    tmp_path: Path,
) -> None:
    """WP-1: Im Restoration-Pfad darf der Sprach-Vocoder Vocos gar nicht laufen."""
    plugin = VocosPlugin(model_path=str(tmp_path / "fehlt.onnx"))
    assert plugin.model_loaded is False
    with pytest.raises(ValueError, match="Restoration-Modus deaktiviert"):
        plugin.vocode(np.zeros(16, dtype=np.float32), 48_000, mode="restoration")


def test_ladeprioritaet_48k_44k_24k_24k_bleibt_fallback_stufe_3(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Register D-K2-2: Priorität 48k→44.1→24k — 24 kHz ist ausschließlich Fallback-Stufe 3."""
    rufe: list[str] = []

    def _recorder(self: VocosPlugin, path: str) -> None:
        rufe.append(Path(path).name)

    monkeypatch.setattr(VocosPlugin, "_try_load", _recorder)
    monkeypatch.setattr("plugins.vocos_plugin._MODEL_48K", str(tmp_path / "vocos_48khz.onnx"))
    monkeypatch.setattr("plugins.vocos_plugin._MODEL_44K", str(tmp_path / "vocos_mel_spec_44khz.onnx"))
    monkeypatch.setattr("plugins.vocos_plugin._MODEL_24K", str(tmp_path / "vocos_mel_spec_24khz.onnx"))
    (tmp_path / "vocos_mel_spec_24khz.onnx").write_bytes(b"stub")
    VocosPlugin()
    assert rufe == ["vocos_mel_spec_24khz.onnx"]  # nur 24k da → Stufe 3 greift

    (tmp_path / "vocos_mel_spec_44khz.onnx").write_bytes(b"stub")
    rufe.clear()
    VocosPlugin()
    assert rufe == ["vocos_mel_spec_44khz.onnx"]  # 44.1 vor 24

    (tmp_path / "vocos_48khz.onnx").write_bytes(b"stub")
    rufe.clear()
    VocosPlugin()
    assert rufe == ["vocos_48khz.onnx"]  # 48 kHz nativ zuerst


@pytest.mark.parametrize(
    "dateiname",
    ["vocos_48khz.onnx", "vocos_mel_spec_44khz.onnx", "vocos_mel_spec_24khz.onnx"],
)
def test_sprach_vocoder_warnen_domaene_beim_laden_gilt_fuer_jedes_tier(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    dateiname: str,
) -> None:
    """D-K2-2-Gate-Lücke (2026-11-05 geschlossen): die §V6-Domänen-Warnung muss beim
    Laden JEDES Vocos-Tiers erscheinen — auch des 44.1-kHz-Tiers (Domäne unbelegt),
    das zuvor über `if "44" not in bname` ohne Warnung als Stufe 1 laufen konnte."""

    class _FakeTensor:
        shape = ["batch", 128, "T"]

    class _FakeSession:
        def __init__(self, path: str, sess_options: object = None, providers: object = None) -> None:
            pass

        def get_inputs(self) -> list[_FakeTensor]:
            return [_FakeTensor()]

        def get_outputs(self) -> list[_FakeTensor]:
            return [_FakeTensor()]

    ort_stub = types.ModuleType("onnxruntime")
    ort_stub.SessionOptions = type("SessionOptions", (), {})  # type: ignore[attr-defined]
    ort_stub.InferenceSession = _FakeSession  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "onnxruntime", ort_stub)
    monkeypatch.setattr("backend.core.ml_memory_budget.try_allocate", lambda *a, **k: True)
    monkeypatch.setattr("backend.core.ml_memory_budget.release", lambda *a, **k: None)
    monkeypatch.setattr(
        "backend.core.plugin_lifecycle_manager.get_plugin_lifecycle_manager",
        lambda: types.SimpleNamespace(register=lambda *a, **k: None),
    )
    monkeypatch.setattr(
        "backend.core.ml_device_manager.get_ort_providers",
        lambda name: ["CPUExecutionProvider"],
        raising=False,
    )
    pfad = tmp_path / dateiname
    pfad.write_bytes(b"stub")
    caplog.set_level(logging.WARNING)

    plugin = VocosPlugin(model_path=str(pfad))
    assert plugin.model_loaded is True, f"Stub-Load fehlgeschlagen für {dateiname}"
    warnungen = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        "§V6 (copilot-instructions.md)" in w and "sprach-trainiert" in w and "Gesangs" in w for w in warnungen
    ), f"§V6-Domänen-Warnung fehlt beim Laden von {dateiname} (Gate muss JEDES Tier melden)"


# ---------------------------------------------------------------------------
# D-K2-2/D-K2-3 — PQS-Messvertrag des §1.4-Finisher-Triggers (PQS-MOS < 4.3).
# Vertragsbruch „Timing-Konflation": `align_signals=True` („zeitlich
# ausgerichtet") war dokumentiert, aber nie implementiert (§V7 (copilot-instructions.md):
# Metrik korrigieren statt Pipeline verbiegen).
# ---------------------------------------------------------------------------


def _probe_signale(n: int = 16_384, sr: int = 48_000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    t = np.arange(n, dtype=np.float64) / sr
    ref = 0.2 * np.sin(2.0 * np.pi * 997.0 * t) + 0.05 * rng.standard_normal(n)
    versatz = np.concatenate([np.zeros(240), ref[: n - 240]])  # reine Latenz +240 Samples
    return ref.astype(np.float32), versatz.astype(np.float32), ref.astype(np.float32)


def test_pqs_vertrag_alt_punished_reinen_timing_offset() -> None:
    """Vertrag ALT (align_signals=False, Trunkierung ohne Ausrichtung): ein reiner
    Latenz-Offset wird als Qualitätsverlust gemessen — genau die Konflation, die
    den §1.4-Studio-Finisher (K2-Sprach-Vocoder auf Musik) unnötig auslösen konnte."""
    ref, versatz, identisch = _probe_signale()
    vertrag_alt = PerceptualQualityScorer(align_signals=False)
    mos_versatz = vertrag_alt.score(ref, versatz, 48_000).pqs_mos
    mos_identisch = vertrag_alt.score(ref, identisch, 48_000).pqs_mos
    # Konflation ist richtungsklar messbar (gemessen 0,0148 bei diesen
    # Probesignalen — auch sub-JND kann sie Trigger/Schwellwerte kippen):
    assert mos_versatz < mos_identisch - 0.005


def test_pqs_vertrag_neu_korrigiert_timing_und_bleibt_never_worsen() -> None:
    """Vertrag NEU (align_signals=True): Timing-Offset wird korrigiert; Never-worsen
    ist konstruktiv garantiert (Lag 0 ist Kandidat der max-Auswahl);
    ausgerichtetes Material bleibt unverändert."""
    ref, versatz, identisch = _probe_signale()
    vertrag_alt = PerceptualQualityScorer(align_signals=False)
    vertrag_neu = PerceptualQualityScorer(align_signals=True)
    mos_alt_versatz = vertrag_alt.score(ref, versatz, 48_000).pqs_mos
    mos_neu_versatz = vertrag_neu.score(ref, versatz, 48_000).pqs_mos
    mos_alt_identisch = vertrag_alt.score(ref, identisch, 48_000).pqs_mos
    mos_neu_identisch = vertrag_neu.score(ref, identisch, 48_000).pqs_mos

    # Never-worsen (A/B-Vertrag alt/neu — Muster b3_transient_energie_neverworsen.py):
    assert mos_neu_versatz >= mos_alt_versatz - 1e-9
    assert mos_neu_identisch >= mos_alt_identisch - 1e-9
    # Korrektur wirkt in die richtige Richtung (gemessen +0,0148 bei diesen
    # Probesignalen; die Ausrichtung stellt den reinen Offset vollständig her):
    assert mos_neu_versatz > mos_alt_versatz + 0.005
    # Ausrichtung stellt die Qualität des rein verschobenen Signals wieder her:
    assert mos_neu_versatz >= mos_neu_identisch - 0.2
    # Kein Effekt auf bereits ausgerichtetes Material:
    assert abs(mos_neu_identisch - mos_alt_identisch) <= 1e-9


# ---------------------------------------------------------------------------
# D-K2-5 — Designtreue: kein AERO im Produktionspfad bis zur bestandenen
# Challenger-Runde (Register D-K2-5, SOTA-M2).
# ---------------------------------------------------------------------------


def test_aero_bleibt_unverdrahtet_bis_challenger_runde_bestanden() -> None:
    root = Path(__file__).resolve().parents[2]
    erlaubt = {"plugins/aero_plugin.py"}
    muster = re.compile(
        r"from\s+plugins\.aero_plugin|import\s+aero_plugin|from\s+aero_plugin"
        r"|get_aero_plugin|AeroPlugin"
    )
    verstoße: list[str] = []
    for basis in ("backend", "plugins", "cli", "Aurik10", "denker", "processing"):
        for pfad in sorted((root / basis).rglob("*.py")):
            rel = pfad.relative_to(root).as_posix()
            if rel in erlaubt:
                continue
            if muster.search(pfad.read_text(encoding="utf-8", errors="replace")):
                verstoße.append(rel)
    assert not verstoße, (
        f"AERO-Verdrahtung gefunden — D-K2-5 erlaubt erst nach bestandener Challenger-Runde (SOTA-M2): {verstoße}"
    )


# ---------------------------------------------------------------------------
# B3-Harness-Logik (Vertrag alt/neu — Nie-schlechter-Werden, JND 0.05).
# ---------------------------------------------------------------------------


def test_b3_pqs_vergleich_bewerten_wertet_never_worsen_aus() -> None:
    from scripts.b3_pqs_mos_neverworsen import vergleich_bewerten

    alt = {
        "seed": 42,
        "zeilen": [
            {"trial_id": "a_identisch", "mos": 4.5, "pqs_norm": 0.875, "trigger": False},
            {"trial_id": "b_versatz", "mos": 2.0, "pqs_norm": 0.25, "trigger": True},
        ],
    }
    neu_gut = {
        "seed": 42,
        "zeilen": [
            {"trial_id": "a_identisch", "mos": 4.5, "pqs_norm": 0.875, "trigger": False},
            {"trial_id": "b_versatz", "mos": 2.4, "pqs_norm": 0.35, "trigger": True},
        ],
    }
    gut = vergleich_bewerten(alt, neu_gut)
    assert gut["never_worsen"] is True
    assert gut["ergebnis"] == "ERFUELLT"

    neu_schlecht = {
        "seed": 42,
        "zeilen": [
            {"trial_id": "a_identisch", "mos": 4.5, "pqs_norm": 0.875, "trigger": False},
            {"trial_id": "b_versatz", "mos": 1.5, "pqs_norm": 0.125, "trigger": True},
        ],
    }
    schlecht = vergleich_bewerten(alt, neu_schlecht)
    assert schlecht["never_worsen"] is False
    assert schlecht["ergebnis"] == "ABGELEHNT"
