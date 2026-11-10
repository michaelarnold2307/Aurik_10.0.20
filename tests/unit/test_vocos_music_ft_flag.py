"""Vertragstests M5 (D-K2-3): Vocos-48-kHz-Musik-Finetune im §1.4-Finisher.

Prüft die vollständige Reachability-Kette (Export ⇒ Plugin ⇒ Flag ⇒ UV3),
die Default-AUS-Sperre (§V7 (copilot-instructions.md)) und das
fail-closed-Verhalten (§V6 (copilot-instructions.md)) — ohne die 163 MB
Gewichte zu laden (die echten Lade-Tests brauchen onnxruntime + Artefakt).

Hintergrund D-K3-85: Ein Modell, das kein Pfad lädt, ist ein Mess-Artefakt.
Diese Tests pinnen genau diese Klasse: Artefaktname, Flag-Default, Auswahl
im Produktionspfad.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.core import music_model_flags
from plugins import vocos_plugin

_EXPORT_BASENAME = "vocos_48khz_music_ft.onnx"


class _FakeVocos:
    """Minimal-Ersatz für VocosPlugin (kein ONNX, kein Speicher)."""

    constructions: int = 0

    def __init__(self, model_path: str, model_loaded: bool = True) -> None:
        type(self).constructions += 1
        self._loaded = model_loaded
        self._weights_basename = Path(model_path).name if model_loaded else ""

    @property
    def model_loaded(self) -> bool:
        return self._loaded

    @property
    def weights_basename(self) -> str:
        return self._weights_basename


def _reset(monkeypatch, *, path: str | None = None) -> None:
    """Setzt den FT-Singleton-Zustand je Test zurück (Modulglobale!)."""
    monkeypatch.setattr(vocos_plugin, "_music_ft_instance", None)
    monkeypatch.setattr(vocos_plugin, "_music_ft_unavailable", False)
    if path is not None:
        monkeypatch.setattr(vocos_plugin, "_MODEL_48K_MUSIC_FT", path)


# ── 1. Flag-Sperre (§V7 (copilot-instructions.md)) ───────────────────────


def test_flag_default_aus():
    """Das Finetune darf nie unbeaufsichtigt klangwirksam werden."""
    assert music_model_flags.use_vocos_music_ft is False


def test_flag_default_ist_im_quelltext_gepinnt():
    """Fail-closed-Default auch dann, wenn das Modul neu importiert wird."""
    text = (ROOT / "backend" / "core" / "music_model_flags.py").read_text(encoding="utf-8")
    assert "use_vocos_music_ft: bool = False" in text


# ── 2. Artefakt-/Export-Kette ───────────────────────────────────────────────


def test_ft_pfad_ist_das_exportziel():
    expected = ROOT / "models" / "vocos_48khz" / "finetune" / _EXPORT_BASENAME
    assert Path(vocos_plugin._MODEL_48K_MUSIC_FT) == expected


def test_exportskript_schreibt_genau_dieses_artefakt():
    """Konsum- und Erzeugerpfad müssen denselben Dateinamen nennen."""
    text = (ROOT / "scripts" / "export_vocos_48khz_onnx.py").read_text(encoding="utf-8")
    assert f'FINETUNE_OUT = os.path.join(FINETUNE_DIR, "{_EXPORT_BASENAME}")' in text
    assert 'FINETUNE_DIR = os.path.join(MODEL_DIR, "finetune")' in text


# ── 3. Fail-closed-Verhalten (§V6 (copilot-instructions.md)) ─────────────


def test_fehlendes_artefakt_faellt_auf_basis_ohne_ladeversuch(monkeypatch, tmp_path):
    _reset(monkeypatch, path=str(tmp_path / "gibt_es_nicht.onnx"))
    sentinel = object()
    monkeypatch.setattr(vocos_plugin, "get_vocos_plugin", lambda: sentinel)

    def _verboten(*_a, **_k):  # pragma: no cover - darf nie laufen
        raise AssertionError("VocosPlugin darf ohne Artefakt nicht gebaut werden")

    monkeypatch.setattr(vocos_plugin, "VocosPlugin", _verboten)
    assert vocos_plugin.get_vocos_music_ft_plugin() is sentinel


def test_nicht_ladbarer_finetune_faellt_auf_basis_und_cacht_negativ(monkeypatch, tmp_path):
    fake_file = tmp_path / _EXPORT_BASENAME
    fake_file.write_bytes(b"kein echtes onnx")
    _reset(monkeypatch, path=str(fake_file))
    sentinel = object()
    monkeypatch.setattr(vocos_plugin, "get_vocos_plugin", lambda: sentinel)
    _FakeVocos.constructions = 0
    monkeypatch.setattr(
        vocos_plugin,
        "VocosPlugin",
        lambda model_path: _FakeVocos(model_path, model_loaded=False),
    )

    assert vocos_plugin.get_vocos_music_ft_plugin() is sentinel
    assert _FakeVocos.constructions == 1
    # Negativ-Cache: kein zweiter 163-MB-Ladeversuch je Aufruf.
    assert vocos_plugin.get_vocos_music_ft_plugin() is sentinel
    assert _FakeVocos.constructions == 1


def test_geladener_finetune_ist_singleton(monkeypatch, tmp_path):
    fake_file = tmp_path / _EXPORT_BASENAME
    fake_file.write_bytes(b"kein echtes onnx")
    _reset(monkeypatch, path=str(fake_file))
    _FakeVocos.constructions = 0
    monkeypatch.setattr(vocos_plugin, "VocosPlugin", lambda model_path: _FakeVocos(model_path))

    erst = vocos_plugin.get_vocos_music_ft_plugin()
    zweit = vocos_plugin.get_vocos_music_ft_plugin()
    assert erst is zweit
    assert _FakeVocos.constructions == 1
    assert erst.weights_basename == _EXPORT_BASENAME


# ── 4. Auswahl im Produktionspfad ───────────────────────────────────────────


def test_uv3_waehlt_finetune_nur_hinter_dem_flag():
    """Quelltext-Pin: der §1.4-Block liest das Flag und den FT-Accessor."""
    text = (ROOT / "backend" / "core" / "unified_restorer_v3.py").read_text(encoding="utf-8")
    block = text.index("§1.4 Vocos-Synthese")
    kp5 = text.index("if _kp5_audio is None:", block)
    flag = text.index('getattr(_mmf_m5, "use_vocos_music_ft", False)', kp5)
    accessor = text.index("get_vocos_music_ft_plugin", flag)
    assert block < kp5 < flag < accessor
    # Der Basis-Pfad muss VOR der Flag-Prüfung gesetzt sein (Default-Verhalten).
    assert text.index("_vplug = _get_vocos_plg()", kp5) < flag
