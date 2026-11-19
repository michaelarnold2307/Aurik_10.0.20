from backend.core.hybrid.hybrid_nvsr import NVSRStrategy, create_nvsr_config


def test_studio_2026_alias_maps_to_maximum_behavior():
    studio = create_nvsr_config("studio_2026", material_type="shellac")
    maximum = create_nvsr_config("maximum", material_type="shellac")

    assert studio.strategy == NVSRStrategy.HYBRID
    assert studio.strategy == maximum.strategy
    assert studio.target_bandwidth_hz == maximum.target_bandwidth_hz
    assert studio.confidence_threshold == maximum.confidence_threshold
    assert studio.blend_ratio == maximum.blend_ratio


def test_restoration_alias_maps_to_balanced_behavior():
    restoration = create_nvsr_config("restoration", material_type="vinyl")
    balanced = create_nvsr_config("balanced", material_type="vinyl")

    assert restoration.strategy == balanced.strategy
    assert restoration.bandwidth_threshold_hz == balanced.bandwidth_threshold_hz
    assert restoration.confidence_threshold == balanced.confidence_threshold
    assert restoration.blend_ratio == balanced.blend_ratio
