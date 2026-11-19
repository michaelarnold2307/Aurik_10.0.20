from __future__ import annotations

from Aurik10.core.preset_manager import PresetManager


def test_factory_preset_descriptions_do_not_contain_known_text_artifacts(tmp_path) -> None:
    manager = PresetManager(presets_dir=tmp_path)

    descriptions = [preset.description for preset in manager.get_all_presets()]

    forbidden_fragments = [
        "Verformungormung",
        "Gleichlaufschwankungenhlaufschwankungen",
        "VerschiebungVerschiebung",
    ]
    for description in descriptions:
        for fragment in forbidden_fragments:
            assert fragment not in description
