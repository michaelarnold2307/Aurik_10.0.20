from backend.core.material_canonical import canonical_material_key, material_label_de


def test_canonical_material_key_normalizes_aliases():
    assert canonical_material_key("cassette") == "tape"
    assert canonical_material_key("vinyl_standard") == "vinyl"
    assert canonical_material_key("tape_studio") == "reel_tape"
    assert canonical_material_key("digital") == "cd_digital"


def test_canonical_material_key_accepts_enum_like_objects():
    class _Mat:
        value = "cassette_chrome"

    assert canonical_material_key(_Mat()) == "tape"


def test_material_label_de_is_unambiguous_for_tape_terms():
    assert material_label_de("tape") == "Kassette (Band)"
    assert material_label_de("cassette") == "Kassette (Band)"
    assert material_label_de("reel_tape") == "Spulenband (Reel-to-Reel)"
