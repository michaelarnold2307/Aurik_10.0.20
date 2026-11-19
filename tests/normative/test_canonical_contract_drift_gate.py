from __future__ import annotations

"""[RELEASE_MUST] Canonical Contract Drift Gate.

Dieses Gate verhindert Parallelpfade neben dem kanonischen Aurik-Vertrag:
Import -> Pre-Analysis -> AurikDenker.denke -> Bridge/Exporter-Export.
"""


import ast
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_COPILOT_INSTRUCTIONS = _ROOT / ".github" / "copilot-instructions.md"
_SPEC_08 = _ROOT / ".github" / "specs" / "08_architecture_and_distribution.md"
_SPEC_07 = _ROOT / ".github" / "specs" / "07_quality_and_tests.md"
_CLI = _ROOT / "cli" / "aurik_cli.py"
_CLI_DEBUG = _ROOT / "cli" / "aurik_debug.py"
_FRONTEND = _ROOT / "Aurik10" / "ui" / "modern_window.py"
_BATCH = _ROOT / "batch_processor.py"
_AUDIO_EXPORTER = _ROOT / "backend" / "core" / "audio_exporter.py"
_EXPORT_WORKFLOW = _ROOT / "backend" / "core" / "export_workflow.py"
_AURIK_RESTORE_LEGACY = _ROOT / "backend" / "aurik_restore.py"
_REST_LEGACY = [
    _ROOT / "backend" / "api" / "rest" / "batch_api.py",
    _ROOT / "backend" / "api" / "rest" / "batch_endpoints.py",
]
_DEBUG_LEGACY = [
    _ROOT / "cli" / "aurik_debug.py",
    _ROOT / "backend" / "api" / "debug_api.py",
]


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_canonical_contract_is_documented_in_vorgaben_and_specs() -> None:
    """Canonical Contract Drift muss in Vorgaben und Specs normativ verankert sein."""
    expected = "Canonical Contract Drift Gate"
    for path in (_COPILOT_INSTRUCTIONS, _SPEC_08, _SPEC_07):
        assert path.exists(), f"{path} fehlt."
        text = path.read_text(encoding="utf-8")
        assert expected in text, f"{expected} fehlt in {path}."


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_cli_release_path_uses_full_canonical_contract() -> None:
    """CLI muss Import, Voranalyse, Denker und Export ueber den Bridge-Vertrag fuehren."""
    src = _CLI.read_text(encoding="utf-8")
    required_tokens = {
        "get_load_audio_fn": "CLI muss Audio ueber den kanonischen Bridge-Loader laden.",
        "run_pre_analysis": "CLI muss die Voranalyse vor dem Denker ausfuehren.",
        "get_aurik_denker_instance": "CLI muss den Denker-Singleton nutzen.",
        "denker.denke(": "CLI muss den Pflicht-Einstieg AurikDenker.denke() nutzen.",
        "export_guard(": "CLI-Export muss vor Schreiboperationen export_guard() nutzen.",
        "validate_export_quality(": "CLI muss das Bridge-Export-Quality-Gate nutzen.",
        "build_export_quality_gate_payload(": "CLI muss den strukturierten Export-Gate-Payload erzeugen.",
        "get_audio_exporter_class": "CLI muss den AudioExporter als Primaerpfad nutzen.",
        "os.replace(tmp_path, out_path)": "CLI-Fallback muss atomic WAV schreiben.",
    }
    for token, message in required_tokens.items():
        assert token in src, message

    forbidden_tokens = (
        "UnifiedRestorerV3.restore(",
        "get_restorer().restore(",
        "librosa.load(",
    )
    for token in forbidden_tokens:
        assert token not in src, f"CLI enthaelt verbotenen Parallelpfad: {token}"


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_frontend_release_path_uses_bridge_contract() -> None:
    """Frontend muss Bridge/Denker/Export-Gate statt direkter Core-Bypaesse nutzen."""
    src = _FRONTEND.read_text(encoding="utf-8")
    tree = ast.parse(src)
    required_tokens = {
        "_bridge_run_pre_analysis": "Frontend muss die Bridge-Voranalyse nutzen.",
        "_bridge_get_load_audio_fn": "Frontend muss den Bridge-Loader nutzen.",
        "_bridge_get_aurik_denker_instance": "Frontend muss den Denker-Singleton nutzen.",
        ".denke(": "Frontend muss AurikDenker.denke() nutzen.",
        "_export_guard(": "Frontend-Export muss export_guard nutzen.",
        "_validate_export_quality(": "Frontend muss das Export-Quality-Gate nutzen.",
        "_build_export_quality_gate_payload": "Frontend muss strukturierten Gate-Payload erzeugen.",
        "_bridge_get_audio_exporter_class": "Frontend muss den Bridge-AudioExporter nutzen.",
    }
    for token, message in required_tokens.items():
        assert token in src, message

    assert "get_restorer().restore(" not in src, (
        "Frontend darf UV3 nicht direkt ueber get_restorer().restore() aufrufen."
    )
    assert "UnifiedRestorerV3.restore(" not in src, "Frontend darf UV3 nicht direkt aufrufen."

    forbidden_import_calls: list[tuple[str, ast.expr]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if (func.value.id, func.attr) in {("sf", "SoundFile"), ("librosa", "load")}:
                forbidden_import_calls.append((f"{func.value.id}.{func.attr}", func))
            if (func.value.id, func.attr) == ("_PydubAudioSegment", "from_file"):
                forbidden_import_calls.append(("_PydubAudioSegment.from_file", func))
        elif isinstance(func, ast.Name) and func.id == "_PedalboardAudioFile":
            forbidden_import_calls.append((func.id, func))

    assert not forbidden_import_calls, (
        "Frontend darf keine eigene Audio-Import-Kaskade neben "
        f"backend.api.bridge.get_load_audio_fn() pflegen: {forbidden_import_calls[:3]}"
    )


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_batch_release_path_uses_bridge_contract() -> None:
    """Batch-Pfad muss den kanonischen Bridge-Vertrag vollstaendig einhalten."""
    src = _BATCH.read_text(encoding="utf-8")
    required_tokens = {
        "_get_load_audio_fn": "Batch muss den Bridge-Loader verwenden.",
        "_run_pre_analysis": "Batch muss run_pre_analysis() vor Denker ausfuehren.",
        "_get_aurik_denker_instance": "Batch muss den Denker-Singleton verwenden.",
        "denker.denke(": "Batch muss den Pflicht-Einstieg AurikDenker.denke() verwenden.",
        "_export_guard(": "Batch-Export muss export_guard() nutzen.",
        "_validate_export_quality(": "Batch muss das Bridge-Export-Quality-Gate nutzen.",
        "_build_export_quality_gate_payload(": "Batch muss den strukturierten Export-Gate-Payload erzeugen.",
        "_get_audio_exporter_class": "Batch muss den Bridge-AudioExporter einbinden.",
        "os.replace(tmp_path, output_file)": "Batch-Fallback muss atomic WAV schreiben.",
    }
    for token, message in required_tokens.items():
        assert token in src, message

    forbidden_tokens = (
        "UnifiedRestorerV3.restore(",
        "get_restorer().restore(",
        "librosa.load(",
    )
    for token in forbidden_tokens:
        assert token not in src, f"Batch enthaelt verbotenen Parallelpfad: {token}"


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_release_surfaces_share_same_bridge_contract_symbols() -> None:
    """GUI, CLI und Batch duerfen keinen auseinanderlaufenden Release-Vertrag pflegen."""
    surfaces = {
        "cli": _CLI.read_text(encoding="utf-8"),
        "frontend": _FRONTEND.read_text(encoding="utf-8"),
        "batch": _BATCH.read_text(encoding="utf-8"),
    }
    canonical_tokens = {
        "load_audio": ("get_load_audio_fn", "_bridge_get_load_audio_fn", "_get_load_audio_fn"),
        "pre_analysis": ("run_pre_analysis", "_bridge_run_pre_analysis", "_run_pre_analysis"),
        "denker": ("get_aurik_denker_instance", "_bridge_get_aurik_denker_instance", "_get_aurik_denker_instance"),
        "export_guard": ("export_guard", "_export_guard", "_export_guard"),
        "validate_export_quality": ("validate_export_quality", "_validate_export_quality", "_validate_export_quality"),
        "audio_exporter": ("get_audio_exporter_class", "_bridge_get_audio_exporter_class", "_get_audio_exporter_class"),
    }
    for contract_name, tokens in canonical_tokens.items():
        for surface_name, token in zip(surfaces.keys(), tokens, strict=True):
            assert token in surfaces[surface_name], (
                f"{surface_name} fehlt Canonical-Contract-Symbol {contract_name}: {token}"
            )


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_legacy_rest_server_paths_are_explicitly_non_release() -> None:
    """REST-Altpfade mit eigenem IO muessen klar als nicht release-faehig markiert sein."""
    for path in _REST_LEGACY:
        assert path.exists(), f"{path} fehlt."
        src = path.read_text(encoding="utf-8")
        assert "LEGACY_NON_RELEASE" in src, (
            f"{path} nutzt historische Server-/Batch-Logik und muss als LEGACY_NON_RELEASE markiert sein "
            "oder vollstaendig auf den Canonical Bridge Contract migriert werden."
        )


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_legacy_rest_exports_still_use_bridge_export_guard() -> None:
    """Auch LEGACY_NON_RELEASE-Batchpfade duerfen keine ungeguardeten Audiodateien schreiben."""
    for path in _REST_LEGACY:
        src = path.read_text(encoding="utf-8")
        assert "export_guard" in src, f"{path} muss export_guard vor sf.write nutzen."
        assert ".tmp" in src and ".replace(out_path)" in src, f"{path} muss atomic schreiben."


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_fastapi_legacy_batch_uses_aurik_ergebnis_contract_safely() -> None:
    """FastAPI-Legacy-Batch darf keine nicht existierenden AurikErgebnis-Felder voraussetzen."""
    src = (_ROOT / "backend" / "api" / "rest" / "batch_endpoints.py").read_text(encoding="utf-8")

    assert "denker_result: Any = denker.denke" in src
    assert 'getattr(denker_result, "sample_rate", sr)' in src
    assert 'getattr(denker_result, "material", "")' in src
    assert 'getattr(denker_result, "deferred_phases", [])' in src


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_legacy_debug_paths_are_explicitly_non_release() -> None:
    """Debug-Bypasspfade muessen klar als LEGACY_NON_RELEASE markiert bleiben."""
    for path in _DEBUG_LEGACY:
        assert path.exists(), f"{path} fehlt."
        src = path.read_text(encoding="utf-8")
        assert "LEGACY_NON_RELEASE" in src, (
            f"{path} ist ein Debug-/Bypasspfad und muss als LEGACY_NON_RELEASE markiert sein, "
            "damit keine Release-Parallelwelt entsteht."
        )


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_legacy_debug_audio_output_uses_bridge_export_guard() -> None:
    """Debug --out darf trotz LEGACY_NON_RELEASE keine ungeguardete Audiodatei schreiben."""
    debug_src = _CLI_DEBUG.read_text(encoding="utf-8")

    out_idx = debug_src.index("if args.out and not args.no_audio:")
    guard_idx = debug_src.index("from backend.api.bridge import export_guard", out_idx)
    write_idx = debug_src.index("sf.write(tmp_path, export_guard(_audio_out), 48000)", guard_idx)
    replace_idx = debug_src.index("tmp_path.replace(out_path)", write_idx)

    assert out_idx < guard_idx < write_idx < replace_idx


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_aurik6_restore_legacy_path_is_marked_and_atomic() -> None:
    """Der historische Aurik-6-E2E-Pfad darf nicht als Release-Parallelwelt driften."""
    src = _AURIK_RESTORE_LEGACY.read_text(encoding="utf-8")

    assert "LEGACY_NON_RELEASE" in src
    export_idx = src.index("def export(")
    guard_idx = src.index("np.nan_to_num", export_idx)
    tmp_idx = src.index('tmp_path = out_path + ".tmp"', guard_idx)
    replace_idx = src.index("os.replace(tmp_path, out_path)", tmp_idx)

    assert export_idx < guard_idx < tmp_idx < replace_idx


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_core_export_workflow_sanitizes_and_writes_atomically() -> None:
    """Core-Exportworkflow muss final numerisch guard-railed und atomic schreiben."""
    src = _EXPORT_WORKFLOW.read_text(encoding="utf-8")

    fn_idx = src.index("def export_audio(")
    guard_idx = src.index("np.nan_to_num", fn_idx)
    write_idx = src.index("sf.write(tmp_export_path", guard_idx)
    replace_idx = src.index("os.replace(tmp_export_path, export_path)", write_idx)

    assert fn_idx < guard_idx < write_idx < replace_idx


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_audio_exporter_sanitizes_and_writes_atomically() -> None:
    """AudioExporter ist der Release-Primärpfad und muss unmittelbar vor Disk-Write sichern."""
    src = _AUDIO_EXPORTER.read_text(encoding="utf-8")

    fn_idx = src.index("def export(")
    guard_idx = src.index("np.nan_to_num(audio_export", fn_idx)
    atomic_fn_idx = src.index("def _atomic_write_audio(", guard_idx)
    write_idx = src.index("sf.write(tmp_path", atomic_fn_idx)
    replace_idx = src.index("tmp_path.replace(path)", write_idx)

    assert fn_idx < guard_idx < atomic_fn_idx < write_idx < replace_idx


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_release_mode_surface_remains_two_button_only() -> None:
    """Release-Oberflaechen duerfen nur Restoration und Studio 2026 als Nutzerentscheidung anbieten."""
    cli_src = _CLI.read_text(encoding="utf-8")
    ui_src = _FRONTEND.read_text(encoding="utf-8")
    assert "Restoration" in cli_src and "Studio 2026" in cli_src
    assert '"RESTORATION"' in ui_src and '"STUDIO_2026"' in ui_src
    assert "--strength" not in cli_src
    assert "--phase" not in cli_src
    assert "--policy" not in cli_src


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_cli_mode_aliases_use_bridge_normalizer() -> None:
    """Mode-Alias-Normalisierung darf nicht parallel in der CLI driften."""
    cli_src = _CLI.read_text(encoding="utf-8")
    bridge_src = (_ROOT / "backend" / "api" / "bridge.py").read_text(encoding="utf-8")

    assert "def normalize_user_mode(" in bridge_src, (
        "Bridge muss den zentralen normalize_user_mode()-Resolver bereitstellen."
    )
    assert "normalize_user_mode = _bridge.normalize_user_mode" in cli_src, (
        "CLI muss den Bridge-Resolver für Mode-Aliase verwenden."
    )
    assert "return normalize_user_mode(mode)" in cli_src, (
        "CLI _normalize_mode() muss an die Bridge delegieren, statt eigene Alias-Tabellen zu pflegen."
    )


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_frontend_mode_aliases_use_bridge_normalizer() -> None:
    """Frontend darf Mode-Aliase nicht lokal driften lassen."""
    ui_src = _FRONTEND.read_text(encoding="utf-8")
    bridge_src = (_ROOT / "backend" / "api" / "bridge.py").read_text(encoding="utf-8")

    assert "def normalize_user_mode(" in bridge_src, (
        "Bridge muss den zentralen normalize_user_mode()-Resolver bereitstellen."
    )
    assert "normalize_user_mode as _bridge_normalize_user_mode" in ui_src, (
        "Frontend muss normalize_user_mode aus der Bridge importieren."
    )
    assert "def _normalize_denker_mode(" in ui_src, (
        "Frontend muss einen zentralen Denker-Mode-Mapper statt verstreuter Alias-Heuristiken nutzen."
    )
    assert "canonical = _bridge_normalize_user_mode(raw_mode)" in ui_src, (
        "Frontend-Mode-Mapping muss auf dem Bridge-Normalizer basieren."
    )


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_legacy_debug_mode_aliases_are_bridge_harmonized() -> None:
    """Auch Legacy-Debug-Pfade sollen den zentralen Mode-Resolver nutzen."""
    debug_src = _CLI_DEBUG.read_text(encoding="utf-8")

    assert "def _resolve_debug_modes(" in debug_src, (
        "Debug-CLI muss einen zentralen Mode-Resolver besitzen statt verstreuter Alias-Checks."
    )
    assert "from backend.api.bridge import normalize_user_mode" in debug_src, (
        "Debug-CLI soll normalize_user_mode aus der Bridge zur Alias-Harmonisierung verwenden."
    )
