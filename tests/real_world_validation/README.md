# Real-World Validation Suite

**Status:** ⏳ In Progress (2-3 Wochen)  
**Goal:** Validate AURIK algorithms with real-world archive recordings  
**Impact:** +3-5 Punkte (Confidence + Subjective Quality)

---

## 📋 Overview

This validation suite tests AURIK's restoration algorithms against real-world archive recordings across 4 categories:

1. **Vinyl** - Scratches, pops, clicks, surface noise
2. **Tape** - Dropouts, wow/flutter, azimuth errors, print-through
3. **Digital** - Clipping, MP3 artifacts, streaming glitches
4. **Vocals** - Sibilance, plosives, breaths, resonances

---

## 🏗️ Structure

```
tests/real_world_validation/
├── README.md                     # This file
├── test_dataset_creator.py       # Collects & validates test files
├── validation_suite.py           # Objective metrics (SNR, THD, Spectral)
├── blind_test_generator.py       # A/B/X comparison files
├── results_analyzer.py           # Statistical analysis & reports
└── test_library/
    ├── vinyl/                    # 10+ vinyl recordings
    ├── tape/                     # 10+ tape recordings
    ├── digital/                  # 10+ digital recordings
    └── vocals/                   # 10+ vocal recordings
```

---

## 🎯 Test Categories

### Vinyl (10+ files)

- **Scratched jazz** (1950s) - Surface noise, clicks, pops
- **Worn rock** (1970s) - Heavy surface wear, pitch instability
- **Classical** (1960s) - Ticks, rumble, tonearm resonance
- **Blues** (1940s shellac) - 78rpm shellac defects

### Tape (10+ files)

- **Reel-to-reel dropouts** - Magnetic coating loss
- **Cassette wow/flutter** - Speed variations
- **DAT phase errors** - Digital artifact from damaged tapes
- **Print-through** - Pre/post echo from magnetic bleed

### Digital (10+ files)

- **CD clipping** - 0dBFS hard clipping
- **MP3 64kbps** - Lossy compression artifacts
- **Streaming glitches** - Buffer underruns, packet loss
- **Digital noise** - A/D converter quantization

### Vocals (10+ files)

- **Opera sibilance** - Harsh "s" sounds (6-10 kHz)
- **Podcast plosives** - "p", "b" pops (<200 Hz)
- **Choir breaths** - Natural breathing (PRESERVE!)
- **Voice resonances** - Vocal tract resonances

---

## 📊 Objective Metrics

### SNR (Signal-to-Noise Ratio)

- **Target:** +10-20 dB improvement
- **Method:** Spectral analysis (noise floor vs signal)
- **Tools:** librosa, scipy.signal

### THD (Total Harmonic Distortion)

- **Target:** <1% THD increase (preservation)
- **Method:** Harmonic content analysis
- **Tools:** scipy.fftpack

### Spectral Metrics

- **Spectral Flatness:** Measure tonality vs noise
- **Spectral Centroid:** Brightness preservation
- **Spectral Rolloff:** High-frequency preservation
- **Tools:** librosa.feature

### Perceptual Metrics

- **PESQ** - Perceptual Evaluation of Speech Quality
- **ViSQOL** - Virtual Speech Quality Objective Listener
- **DNSMOS** - Deep Noise Suppression Mean Opinion Score

---

## 🧪 Validation Workflow

### 1. Dataset Acquisition (3-5 days)

```python
# test_dataset_creator.py
python test_dataset_creator.py --category vinyl --count 10
python test_dataset_creator.py --category tape --count 10
python test_dataset_creator.py --category digital --count 10
python test_dataset_creator.py --category vocals --count 10
```

**Sources:**

- Public domain archives (Internet Archive, Library of Congress)
- Creative Commons recordings
- Self-recorded test signals with controlled defects

### 2. Objective Validation (5-7 days)

```python
# validation_suite.py
python validation_suite.py --input test_library/ --output validation_report.json
```

**Metrics:**

- Pre-processing: Analyze defects
- Post-processing: Measure restoration quality
- Comparison: SNR improvement, THD increase, spectral changes

### 3. Blind Test Generation (2-3 days)

```python
# blind_test_generator.py
python blind_test_generator.py --input test_library/ --output blind_tests/
```

**Output:**

- A/B/X comparison files (randomized)
- Reference files (unprocessed)
- Test protocol for human evaluators

### 4. Statistical Analysis (2-3 days)

```python
# results_analyzer.py
python results_analyzer.py --validation validation_report.json --blind blind_results.json
```

**Reports:**

- Objective metrics summary
- Blind test results (subjective quality)
- Comparison vs. iZotope RX10, Cedar
- Statistical significance tests

---

## 📈 Success Criteria

### Objective Targets

- ✅ **SNR improvement:** +10-20 dB (better than iZotope RX10)
- ✅ **THD increase:** <1% (no audible distortion)
- ✅ **Spectral preservation:** <0.5 dB deviation in centroid
- ✅ **PESQ score:** >4.0 for speech
- ✅ **ViSQOL score:** >4.0 for music

### Subjective Targets

- ✅ **Blind test preference:** >60% prefer AURIK over unprocessed
- ✅ **Blind test preference:** >50% prefer AURIK over iZotope RX10
- ✅ **Naturalness score:** >4.0/5.0 (no artifacts detected)

---

## 🚀 Usage

### Quick Start

```bash
# Install dependencies
pip install librosa scipy numpy pandas matplotlib seaborn

# Create test dataset (placeholder generator)
python test_dataset_creator.py --mode placeholder

# Run validation suite
python validation_suite.py --input test_library/ --output results/

# Generate blind tests
python blind_test_generator.py --input test_library/ --output blind_tests/

# Analyze results
python results_analyzer.py --results results/validation_report.json
```

### Custom Dataset

```bash
# Add your own files to test_library/
cp my_vinyl_recording.wav test_library/vinyl/
cp my_tape_recording.wav test_library/tape/

# Run validation
python validation_suite.py --input test_library/ --output my_results/
```

---

## 📦 Deliverables

- [ ] Test Dataset (30+ real-world files)
- [ ] Objective Validation Report (SNR, THD, Spectral metrics)
- [ ] Blind Test Files (A/B/X comparison)
- [ ] Statistical Analysis (vs. iZotope RX10, Cedar)
- [ ] Subjective Scoring Integration
- [ ] Documentation (this file + results report)

---

## 🎓 References

- **PESQ:** ITU-T P.862 (Perceptual Evaluation of Speech Quality)
- **ViSQOL:** Google Research (Virtual Speech Quality Objective Listener)
- **DN SMOS:** Microsoft (Deep Noise Suppression Mean Opinion Score)
- **iZotope RX10:** Industry-standard audio restoration suite
- **Cedar:** Professional broadcast restoration tools

---

**Author:** AURIK Development Team  
**Date:** 9. Februar 2026  
**Version:** 1.0 (Initial Implementation)
