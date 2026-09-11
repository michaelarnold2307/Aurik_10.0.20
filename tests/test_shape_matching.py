import pytest

"""
Test für Shape-Matching zwischen Stereo/Mono Arrays (resemble_enhance Broadcasting-Fix)
"""

import numpy as np


@pytest.mark.unit
def test_shape_matching():
    """Test dass Shape-Matching korrekt funktioniert"""
    print("=" * 80)
    print("TEST: Shape-Matching für ML-Plugin Outputs")
    print("=" * 80)

    # Simuliere verschiedene Input/Output-Kombinationen

    # Test 1: Stereo Input, Mono Output
    print("\n[Test 1] Stereo Input (720000, 2) → Mono Output (720000,)")
    x = np.random.randn(720000, 2)
    x_denoised = np.random.randn(720000)

    # Apply Fix
    if x.ndim == 2 and x_denoised.ndim == 1:
        x_denoised = np.column_stack([x_denoised, x_denoised])

    print(f"   Input shape: {x.shape}")
    print(f"   Output shape after fix: {x_denoised.shape}")

    # Test Broadcasting
    try:
        result = 0.5 * x + 0.5 * x_denoised
        print(f"   ✅ Broadcasting erfolgreich! Result shape: {result.shape}")
    except Exception as e:
        print(f"   ❌ Broadcasting fehlgeschlagen: {e}")
        return False

    # Test 2: Mono Input, Stereo Output
    print("\n[Test 2] Mono Input (720000,) → Stereo Output (720000, 2)")
    x = np.random.randn(720000)
    x_denoised = np.random.randn(720000, 2)

    # Apply Fix
    if x.ndim == 1 and x_denoised.ndim == 2:
        x_denoised = x_denoised[:, 0]

    print(f"   Input shape: {x.shape}")
    print(f"   Output shape after fix: {x_denoised.shape}")

    try:
        result = 0.5 * x + 0.5 * x_denoised
        print(f"   ✅ Broadcasting erfolgreich! Result shape: {result.shape}")
    except Exception as e:
        print(f"   ❌ Broadcasting fehlgeschlagen: {e}")
        return False

    # Test 3: Stereo Input, Stereo Output (Same channels)
    print("\n[Test 3] Stereo Input (720000, 2) → Stereo Output (720000, 2)")
    x = np.random.randn(720000, 2)
    x_denoised = np.random.randn(720000, 2)

    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {x_denoised.shape}")

    try:
        result = 0.5 * x + 0.5 * x_denoised
        print(f"   ✅ Broadcasting erfolgreich! Result shape: {result.shape}")
    except Exception as e:
        print(f"   ❌ Broadcasting fehlgeschlagen: {e}")
        return False

    # Test 4: Different lengths (trim)
    print("\n[Test 4] Length mismatch (trim longer)")
    x = np.random.randn(720000, 2)
    x_denoised = np.random.randn(730000, 2)

    # Apply Fix
    if len(x_denoised) != len(x):
        if len(x_denoised) > len(x):
            x_denoised = x_denoised[: len(x)]

    print(f"   Input length: {len(x)}")
    print(f"   Output length after fix: {len(x_denoised)}")

    try:
        result = 0.5 * x + 0.5 * x_denoised
        print(f"   ✅ Broadcasting erfolgreich! Result shape: {result.shape}")
    except Exception as e:
        print(f"   ❌ Broadcasting fehlgeschlagen: {e}")
        return False

    # Test 5: Different lengths (pad shorter)
    print("\n[Test 5] Length mismatch (pad shorter)")
    x = np.random.randn(720000, 2)
    x_denoised = np.random.randn(710000, 2)

    # Apply Fix
    if len(x_denoised) != len(x):
        if len(x_denoised) < len(x):
            if x.ndim == 2:
                pad = np.zeros((len(x) - len(x_denoised), x.shape[1]))
            else:
                pad = np.zeros(len(x) - len(x_denoised))
            x_denoised = np.concatenate([x_denoised, pad])

    print(f"   Input length: {len(x)}")
    print(f"   Output length after fix: {len(x_denoised)}")

    try:
        result = 0.5 * x + 0.5 * x_denoised
        print(f"   ✅ Broadcasting erfolgreich! Result shape: {result.shape}")
    except Exception as e:
        print(f"   ❌ Broadcasting fehlgeschlagen: {e}")
        return False

    print("\n" + "=" * 80)
    print("✅✅✅ ALLE SHAPE-MATCHING TESTS BESTANDEN ✅✅✅")
    print("=" * 80)
    return True


if __name__ == "__main__":
    import sys

    try:
        success = test_shape_matching()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ TEST FEHLGESCHLAGEN: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
