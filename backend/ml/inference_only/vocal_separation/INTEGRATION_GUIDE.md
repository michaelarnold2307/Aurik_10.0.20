"""
Integration instructions for Vocal Separation Module into adaptive_pipeline.py

## NEW FEATURES (v8.1 - Vocal Excellence Phase)

1. **Advanced Vocal Separation** (CRITICAL Priority)
   - MDX-Net wrapper (spectral domain, U-Net)
   - Demucs v5 wrapper (hybrid transformer)
   - Hybrid ensemble (adaptive fusion)
   - HIPS-compliant safety wrapper

## INTEGRATION STEPS

### Step 1: Import new vocal separation modules

Add to imports section (after line 31):

```python
# v8.1: Advanced Vocal Separation (Feature #1)
from backend.ml.inference_only.vocal_separation import HybridVocalSeparator, MDXNetSeparator, DemucsV5Separator
from backend.ml.safety_wrappers.vocal_separation_safety import VocalSeparationSafetyWrapper, HIPSViolationError
```

### Step 2: Initialize separators in `__init__`

Add to `__init__` method (after line 105):

```python
# v8.1: Advanced Vocal Separation
self.vocal_separator_v8 = HybridVocalSeparator(
    fusion_strategy="adaptive",
    sample_rate=44100,
    device=None,  # Auto-detect
)
self.vocal_safety_wrapper = VocalSeparationSafetyWrapper(self.vocal_separator_v8, strict_mode=False)

self.logger.info("v8.1 Vocal Separation initialized (Hybrid: MDX-Net + Demucs v5)")
```

### Step 3: Add separate_vocals_v8 method

Add new method to AdaptiveProcessingPipeline class:

```python
    def separate_vocals_v8(
        self,
        audio: np.ndarray,
        sr: int,
        use_safety_wrapper: bool = True
    ) -> Dict[str, np.ndarray]:
        \"\"\"
        v8.1: Advanced vocal separation with HIPS compliance

        Uses Hybrid ensemble (MDX-Net + Demucs v5) with adaptive fusion.

        Args:
            audio: Audio array (stereo or mono)
            sr: Sample rate  
            use_safety_wrapper: Enable HIPS compliance checking

        Returns:
            Dictionary with 'vocals' and 'instrumental' stems
        \"\"\"
        self.logger.info("Starting v8.1 vocal separation (Hybrid)")

        if use_safety_wrapper:
            # HIPS-compliant separation with validation
            stems = self.vocal_safety_wrapper.safe_separate(
                audio,
                sr,
                return_individual=False
            )
        else:
            # Direct separation (bypass safety checks)
            stems = self.vocal_separator_v8.separate(
                audio,
                sr,
                return_individual=False
            )

        # Log metrics
        metrics = self.vocal_separator_v8.get_metrics()
        self.logger.info(
            f"Vocal separation complete: "
            f"{metrics['total_separations']} total, "
            f"fusion={metrics['fusion_strategy']}"
        )

        return stems
```

### Step 4: Replace old vocal separation calls

Search for existing vocal separation calls (Demucs v4, Spleeter, etc.) and replace with:

```python
# OLD (v7.x):
# stems = self.demucs.separate_stems(audio, sr)

# NEW (v8.1):
stems = self.separate_vocals_v8(audio, sr, use_safety_wrapper=True)
```

### Step 5: Update policy engine recommendations

Modify policy/ml_policy_engine.py to recommend v8.1 separators:

```python
# Add to _recommend_vocal_separation():
if context.content_type == "music_vocal":
    recommendations.append(
        {
            "model": "hybrid_vocal_separator_v8",
            "reason": "SOTA hybrid ensemble (MDX-Net + Demucs v5)",
            "priority": "CRITICAL",
            "quality_target": 0.95,
        }
    )
```

## TESTING

Run integration tests:

```bash
pytest tests/vocal_separation/test_integration.py -v
pytest tests/test_adaptive_pipeline_v8.py::test_vocal_separation_v8 -v
```

## VALIDATION CRITERIA

1. ✅ Stems recombine to original (reconstruction error < -40 dB)
2. ✅ Phase coherence preserved (> 75%)
3. ✅ Stereo width maintained (change < 30%)
4. ✅ Energy conservation (loss < 15%)
5. ✅ HIPS audit log created
6. ✅ Processing time < 10s for 3-minute song

## NEXT STEPS

After integration:

1. Benchmark against LALAL.AI, AudioShake on test dataset
2. Tune fusion strategy weights based on results
3. Implement Feature #2: Pitch Correction (Conservative)
4. Add formant-aware enhancement (Feature #3)

---
Generated: 2026-02-07
Feature: #1 Vocal Source Separation Upgrade (Phase 1, Month 1-2)
Status: Implementation Complete, Integration Pending
"""
