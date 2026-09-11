# Contributing to Aurik 10.0.8

Thank you for your interest in contributing to Aurik 10.0.8! This document provides guidelines and instructions for contributing to this project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Issue Reporting & Management](#issue-reporting--management)
- [Coding Guidelines](#coding-guidelines)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)
- [Community Guidelines](#community-guidelines)

---

## Getting Started

Aurik 10.0.8 is a professional-grade audio restoration and enhancement software with a hybrid DSP + ML architecture. Before contributing, please:

1. Read the [README.md](README.md)
2. Review the [Roadmap](docs/aurik9_roadmap.md)
3. Check [existing issues](../../issues) and [pull requests](../../pulls)
4. Join our community discussions (if available)

## Development Setup

### Prerequisites

- Python 3.10+
- Virtual environment tool (venv)
- Git

### Setup Instructions

```bash
# Clone the repository
git clone https://github.com/aurik-audio/Aurik_Standalone.git
cd Aurik_Standalone

# Create and activate virtual environment
python -m venv .venv_aurik
source .venv_aurik/bin/activate  # On Windows: .venv_aurik\Scripts\activate

# Install dependencies
pip install -r requirements/requirements_aurik.txt

# Run tests to verify setup
pytest tests/unit --maxfail=1 --disable-warnings --tb=short -q

# Run the application
python start_aurik_90.py
```

### Project Structure

```
Aurik_Standalone/
├── core/                    # Core processing engine
│   ├── phases/             # 64 processing phases (Phase 01–64)
│   ├── unified_restorer_v3.py  # Main pipeline orchestrator
│   ├── defect_scanner.py   # Defect detection (62 DefectTypes)
│   └── musical_goals/      # 14 perceptual quality goals
├── dsp/                     # DSP algorithms (OMLSA, NMF-β, PGHI, …)
├── plugins/                 # ML plugins (ONNX/local, all with DSP fallback)
├── denker/                  # Cognitive orchestration layer
├── backend/                 # API and backend utilities
├── tests/                   # Test suite (~11598 unit tests)
├── docs/                    # Documentation
├── Aurik10/                # GUI (PyQt5-based, frameless dark-theme)
├── models/                  # ML model weights (not tracked in Git)
└── README.md
```

---

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

1. **Bug Reports:** Submit detailed bug reports with reproduction steps
2. **Feature Requests:** Propose new features or enhancements
3. **Code Contributions:** Fix bugs, implement features, optimize performance
4. **Documentation:** Improve docs, add examples, fix typos
5. **Testing:** Write tests, improve test coverage
6. **Reviews:** Review pull requests, provide feedback

### Finding Issues to Work On

- Look for issues labeled `good first issue` for beginners
- Check `help wanted` for issues needing contributors
- Review the [Roadmap](docs/aurik9_roadmap.md) for planned features

**Browse Issues:**

```
is:open is:issue label:"good first issue" no:assignee
is:open is:issue label:"help wanted" no:assignee
is:open is:issue label:"priority: high" no:assignee
```

### Before You Start

1. **Check existing work:** Search issues/PRs to avoid duplicates
2. **Discuss major changes:** Open an issue first for significant changes
3. **Follow conventions:** Match the existing code style and patterns

---

## Issue Reporting & Management

### Reporting Bugs 🐛

Found a bug? Help us fix it by creating a detailed bug report.

**Use the Bug Report Template:**

- Go to [Issues](../../issues/new/choose)
- Select "🐛 Bug Report"
- Fill out all required fields

**Include:**

- Clear title: `[Bug]: Crash when processing 32-bit WAV files`
- Steps to reproduce
- Expected vs actual behavior
- Aurik version, OS, audio format
- Log output or error messages

**See:** [.github/ISSUE_MANAGEMENT.md](.github/ISSUE_MANAGEMENT.md) for detailed guidelines

### Requesting Features ✨

Have an idea? We'd love to hear it!

**Use the Feature Request Template:**

- Go to [Issues](../../issues/new/choose)
- Select "✨ Feature Request"
- Describe the problem and proposed solution

**Good Feature Requests:**

- Clear problem statement
- Specific use cases
- Examples or references
- Implementation ideas (if you have technical knowledge)

### Performance Issues ⚡

Experiencing slow processing?

**Use the Performance Issue Template:**

- Include RT factor, processing time
- Audio file specifications
- System specifications (CPU, RAM, GPU)
- Processing mode used (FAST/BALANCED/MAXIMUM)

### Documentation Issues 📚

Found unclear or missing documentation?

**Use the Documentation Template:**

- Specify file/page location
- Describe what's wrong or missing
- Suggest improvements

**Quick Docs Fixes:**

- Simple typos or grammar fixes can be submitted directly as PRs
- No need to create an issue first for minor documentation improvements

### Issue Labels

We use labels to organize and prioritize issues:

**Type Labels:**

- `bug` - Something isn't working
- `enhancement` - New feature request
- `performance` - Performance optimization
- `documentation` - Documentation improvement

**Priority Labels:**

- `priority: critical` - Must fix immediately
- `priority: high` - Should fix soon
- `priority: medium` - Fix when possible
- `priority: low` - Nice to have

**Area Labels:**

- `area: dsp` - DSP algorithms
- `area: ml` - Machine Learning
- `area: gui` - GUI
- `area: cli` - Command Line
- `area: testing` - Tests and QA

**See:** [.github/LABELS.md](.github/LABELS.md) for complete label reference

### Claiming Issues

Ready to work on an issue?

1. Comment: "I'd like to work on this issue"
2. Wait for maintainer confirmation (usually within 48 hours)
3. Self-assign if you have permission, or ask maintainer
4. Provide updates at least every 2 weeks

**Note:** Issues inactive for >4 weeks may be reassigned to keep project momentum.

---

## Coding Guidelines

### Python Code Style

- **PEP 8:** Follow Python's style guide
- **Type Hints:** Use type annotations for function signatures
- **Docstrings:** Document all public functions/classes
- **Comments:** Explain complex logic, not obvious code

**Example:**

```python
def process_audio(
    audio: np.ndarray,
    sample_rate: int,
    material: MaterialType = MaterialType.VINYL
) -> PhaseResult:
    """
    Process audio with material-adaptive restoration.

    Args:
        audio: Input audio samples (mono or stereo)
        sample_rate: Sample rate in Hz
        material: Source material type

    Returns:
        PhaseResult with processed audio and metadata
    """
    # Implementation
    pass
```

### Code Organization

- **Modularity:** Keep functions small and focused
- **Separation of Concerns:** DSP logic separate from ML logic
- **Error Handling:** Use try-except with graceful fallbacks
- **Logging:** Use Python logging module, not print statements

```python
import logging
logger = logging.getLogger(__name__)

def process():
    try:
        # ML processing
        result = ml_algorithm()
    except Exception as e:
        logger.warning(f"ML failed: {e}, falling back to DSP")
        result = dsp_fallback()
    return result
```

### Performance Considerations

- **Profile first:** Measure before optimizing
- **Vectorization:** Use NumPy operations, avoid Python loops
- **Parallelization:** Use ThreadPoolExecutor/ProcessPoolExecutor for CPU-bound tasks
- **Memory:** Be mindful of large arrays, use generators when possible

---

## Testing Requirements

### Test Coverage

All contributions must include tests:

- **Unit Tests:** Test individual functions/classes
- **Integration Tests:** Test phase interactions
- **End-to-End Tests:** Test full pipelines

### Running Tests

```bash
# Run all unit tests
pytest tests/unit --disable-warnings --tb=short -q

# Run specific test file
pytest tests/unit/test_v99_genre_schlager.py -v

# Run with coverage
pip install coverage
coverage run -m pytest tests/unit/
coverage report

# Run fast tests only (skip slow ML tests)
pytest tests/unit -m "not slow"
```

### Writing Tests

```python
import pytest
import numpy as np

def test_denoise_phase():
    """Test Phase 03 Denoise with synthetic audio"""
    # Arrange
    audio = np.random.randn(48000)  # 1 second at 48kHz
    phase = DenoisePhase()

    # Act
    result = phase.process(audio, sample_rate=48000, material_type='vinyl')

    # Assert
    assert result.success
    assert result.audio is not None
    assert len(result.audio) == len(audio)
    assert result.execution_time_seconds < 2.0  # <2× realtime
```

### Test Data

- **Synthetic Audio:** Generate test signals programmatically
- **Golden Samples:** Use `golden_samples/` for reference audio
- **Fixtures:** Share test data with pytest fixtures

---

## Pull Request Process

### 1. Fork and Branch

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/aurik-audio/Aurik_Standalone.git
cd Aurik_Standalone

# Create a feature branch
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write code following guidelines above
- Add/update tests
- Update documentation if needed
- Test thoroughly locally

### 3. Commit

Use clear, descriptive commit messages:

```bash
# Good commit messages
git commit -m "Fix: Resolve Phase 20 reverb detection bug"
git commit -m "Feature: Add NVSR frequency restoration (Phase 06/07)"
git commit -m "Docs: Update ML-Hybrid integration guide"
git commit -m "Test: Add end-to-end test for tape material"

# Bad commit messages (avoid)
git commit -m "fix bug"
git commit -m "update"
git commit -m "WIP"
```

### 4. Push and Create PR

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create Pull Request on GitHub with:
# - Clear title and description
# - Reference related issues (#123)
# - List changes made
# - Mention any breaking changes
```

### 5. PR Review Process

1. **Automated Checks:** CI/CD runs tests automatically
2. **Code Review:** Maintainers review your code
3. **Feedback:** Address review comments
4. **Approval:** Once approved, PR is merged

### 6. Push Rules (Release-Must)

Before every push, the local gate must be green.

Mandatory checks:

1. Run changed-scope unit tests and affected integration tests.
2. Ensure no new lint/type diagnostics in touched files.
3. Update changelog/spec/audit docs for normative behavior changes.
4. Verify no contract bypass around bridge/denker/export path.

Minimum commands (project standard):

```bash
"${workspaceFolder}/.venv_aurik/bin/python" -m pytest tests/unit -p no:xdist --override-ini="addopts=--strict-markers --import-mode=importlib" --timeout=30 --tb=short -q --disable-warnings --no-header
"${workspaceFolder}/.venv_aurik/bin/python" -m pytest tests -m "not e2e and not ml" -p no:xdist --override-ini="addopts=--strict-markers --import-mode=importlib" --timeout=30 --tb=short -q --disable-warnings --no-header --maxfail=20 --ignore=tests/unit
```

Push is blocked when:

- a mandatory test fails
- touched files introduce new lint/type findings
- normative code change has no matching changelog/spec/audit delta

### PR Template

```markdown
## Description
Brief description of the changes

## Related Issues
Fixes #123, Related to #456

## Changes Made
- Added feature X
- Fixed bug Y
- Updated docs Z

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing completed

## Performance Impact
- Estimated impact: +5% speed, -2 MB memory

## Breaking Changes
None / List any breaking changes

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests added and passing
- [ ] Documentation updated
- [ ] No linting errors
```

---

## Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:

- **Be respectful:** Treat everyone with respect and kindness
- **Be constructive:** Provide helpful feedback, not personal attacks
- **Be collaborative:** We're all working toward the same goal
- **Be patient:** Remember that everyone is learning

### Communication Channels

- **GitHub Issues:** Bug reports, feature requests
- **GitHub Discussions:** General questions, ideas
- **Pull Requests:** Code contributions, reviews

### Getting Help

- **Documentation:** Check `docs/` folder
- **Examples:** See `tests/` for usage examples
- **Issues:** Search existing issues or open a new one
- **Discussions:** Ask questions in GitHub Discussions

---

## Development Workflow

### Typical Workflow

1. **Identify Work:** Choose issue or feature from roadmap
2. **Discuss:** Comment on issue, discuss approach
3. **Branch:** Create feature branch
4. **Develop:** Write code, tests, docs
5. **Test:** Run tests locally, verify functionality
6. **Commit:** Make clear, atomic commits
7. **Push:** Push to your fork
8. **PR:** Create pull request
9. **Review:** Address feedback
10. **Merge:** Maintainer merges PR

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation changes
- `test/description` - Test additions/fixes
- `refactor/description` - Code refactoring

### Release Process

- **Semantic Versioning:** MAJOR.MINOR.PATCH (e.g., 10.0.0)
- **Release Candidates:** 10.0.0-rc1, 10.0.0-rc2
- **Changelog:** Updated in `docs/aurik9_roadmap.md`

---

## Areas Needing Contribution

### High Priority

1. **GUI Development:** Electron-based interface (`aurik_90/`)
2. **Real-World Testing:** Validate with actual degraded recordings
3. **ML Plugin Deployment:** Docker containers for DCCRN, DeepFilterNet, CREPE
4. **Performance Optimization:** Further speed improvements
5. **Documentation:** User guides, API documentation

### Medium Priority

1. **Tier 2 ML-Hybrid:** Phase 06/07 (NVSR), Phase 19 (Phoneme Detection)
2. **Musical Excellence Features:** Vocal Enhancement Suite
3. **CI/CD Pipeline:** Automated builds, tests, releases
4. **Benchmark Suite:** Compare with iZotope RX, CEDAR
5. **Platform Support:** Windows, macOS packaging

### Low Priority / Future

1. **Custom ML Models:** Train Aurik-specific models
2. **GPU Acceleration:** CUDA support (currently CPU-only)
3. **Real-time Processing:** Live audio processing
4. **Plugin Formats:** VST/AU for DAW integration
5. **Cloud Processing:** Backend API for web/mobile apps

---

## Recognition

Contributors will be recognized in:

- `CONTRIBUTORS.md` file
- Release notes
- Project documentation

Thank you for contributing to Aurik 10.0.8! 🎵✨

---

**Questions?** Open an issue or discussion on GitHub.

**Found a bug?** Please report it with reproduction steps.

**Have an idea?** We'd love to hear it - open a feature request!
