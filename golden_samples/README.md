# Golden Samples Directory

This directory contains golden audio samples for quality validation and testing.

## Structure

```
golden_samples/
├── vocal/          # Vocal-heavy tracks (60 samples target)
├── instrumental/   # Instrumental tracks (20 samples target)
├── classical/      # Classical recordings (10 samples target)
├── jazz/           # Jazz recordings (10 samples target)
├── references/     # Studio-quality reference audio
└── metadata.json   # Metadata für alle samples
```

## Usage

### Adding New Golden Samples

1. Place audio files in the appropriate category folder
2. Add corresponding reference audio (if available) in `references/`
3. Update `metadata.json` with:
   - File name
   - Category (vocal/instrumental/classical/jazz)
   - Duration
   - Sample rate
   - Source (optionalOptional: Artist, track name, etc.)
   - Quality baseline score (if known)

### Running Quality Validation

```bash
# Validate all golden samples
python optimization/profiling.py --golden-samples golden_samples/ --references golden_samples/references/

# Validate specific category
python optimization/profiling.py --golden-samples golden_samples/vocal/ --references golden_samples/references/
```

### Golden Sample Requirements

- **Format**: WAV or FLAC (lossless)
- **Sample Rate**: 44.1 kHz or 48 kHz preferred
- **Duration**: 30-60 seconds (full tracks okay but slower)
- **Quality**: As high as possible (studio masters preferred)
- **Diversity**: Cover various genres, instruments, recording conditions

## Current Status

- **Total Samples**: 1 (example)
- **Target**: 100 samples
- **Vocal**: 0 / 60
- **Instrumental**: 0 / 20
- **Classical**: 0 / 10  
- **Jazz**: 0 / 10

## Next Steps

1. Collect 100 golden samples (see Roadmap Week 2-3)
2. Obtain reference audio for each sample
3. Run comprehensive quality validation
4. Establish quality baseline (88-90%)
5. Measure improvement after optimizations (target: 95-97%)

## Automated Testing

Golden samples are automatically tested in CI/CD:

```bash
# Run nightly quality regression tests
pytest tests/test_golden_samples.py
```

This ensures that code changes don't degrade quality on known test cases.
