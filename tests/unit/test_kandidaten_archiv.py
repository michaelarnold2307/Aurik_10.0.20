"""Tests für die Kandidaten-Archivierung (Pfeiler 3: archivieren statt pflegen)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.kandidaten_archiv import archiv_plan, fuehre_aus

_ROW = "| D-K0-4 | `models/probe/probe_denoiser.onnx` | K0 | Sprache | 2/2 | in-arbeit | schwach | — |"


def _projekt(tmp_path: Path) -> Path:
    (tmp_path / ".github").mkdir()
    (tmp_path / "models" / "probe").mkdir(parents=True)
    (tmp_path / "models" / "probe" / "probe_denoiser.onnx").write_bytes(b"modell")
    reg = (
        "# Register\n"
        "| ID | Artefakt | Klasse | Domäne | Evidenz | Status | Blocker | Beleg |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |\n" + _ROW + "\n"
    )
    (tmp_path / ".github" / "SOTA_DEFICIT_REGISTER.md").write_text(reg, encoding="utf-8")
    return tmp_path


def test_plan_und_trockenlauf(tmp_path: Path) -> None:
    root = _projekt(tmp_path)
    plan = archiv_plan("D-K0-4", projekt_root=root)
    assert plan["quellen"] == ["models/probe/probe_denoiser.onnx"]
    assert plan["ziel"].startswith("models/_archive_") and "D-K0-4" in plan["ziel"]
    reg = (root / ".github" / "SOTA_DEFICIT_REGISTER.md").read_text(encoding="utf-8")
    assert _ROW in reg, "Trockenlauf schreibt nichts"
    assert not (root / plan["ziel"]).exists(), "Trockenlauf legt kein Archiv an"


def test_ausfuehren_ohne_signoff_abgelehnt(tmp_path: Path) -> None:
    root = _projekt(tmp_path)
    plan = archiv_plan("D-K0-4", projekt_root=root)
    code, plan2 = fuehre_aus(plan, projekt_root=root, signoff=root / "fehlt.md")
    assert code == 2 and "Sign-off" in plan2["fehler"]
    assert _ROW in (root / ".github" / "SOTA_DEFICIT_REGISTER.md").read_text(encoding="utf-8")


def test_ausfuehren_bucht_und_archiviert(tmp_path: Path) -> None:
    root = _projekt(tmp_path)
    signoff = root / "signoff.md"
    signoff.write_text("User-Sign-off: archivieren (§9/5 der A2-Liste).", encoding="utf-8")
    plan = archiv_plan("D-K0-4", projekt_root=root)
    code, plan2 = fuehre_aus(plan, projekt_root=root, signoff=signoff, gates=lambda: True)
    assert code == 0 and plan2["ergebnis"] == "archiviert"
    ziel = root / plan["ziel"]
    assert (ziel / "probe_denoiser.onnx").is_file(), "Snapshot liegt im Archiv"
    assert (root / "models" / "probe" / "probe_denoiser.onnx").is_file(), "Deployt bleibt stehen"
    reg = (root / ".github" / "SOTA_DEFICIT_REGISTER.md").read_text(encoding="utf-8")
    assert "geschlossen" in reg and plan["ziel"] in reg and "signoff.md" in reg


def test_rollback_bei_gatefehler(tmp_path: Path) -> None:
    root = _projekt(tmp_path)
    signoff = root / "signoff.md"
    signoff.write_text("Sign-off.", encoding="utf-8")
    plan = archiv_plan("D-K0-4", projekt_root=root)
    code, plan2 = fuehre_aus(plan, projekt_root=root, signoff=signoff, gates=lambda: False)
    assert code == 1 and "Nach-Gate" in plan2["fehler"]
    assert not (root / plan["ziel"]).exists(), "Snapshot zurückgerollt"
    assert _ROW in (root / ".github" / "SOTA_DEFICIT_REGISTER.md").read_text(encoding="utf-8"), "Register unverändert"
