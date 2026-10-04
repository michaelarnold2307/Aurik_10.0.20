# CLI Accessibility Guide for AURIK v8

## Overview

AURIK v8 provides comprehensive accessibility features for command-line interfaces, ensuring that all users can effectively use the batch processor and other command-line tools.

## Features

### 🎨 Multiple Visual Themes

- **Plain** (Screen Reader Mode)
  - No colors or visual formatting
  - Descriptive text prefixes: `[SUCCESS]`, `[ERROR]`, `[WARNING]`, `[INFO]`
  - Clean table formatting with text markers
  - Example:

    ```text
    [SUCCESS] Operation completed
    [TABLE: Name, Age, City]
    Alice | 30 | NYC
    [END TABLE]
    ```

- **Colorful** (Default Visual Mode)
  - Color-blind safe palette (deuteranopia/protanopia tested)
  - Visual icons: ✓ ✗ ⚠ ℹ
  - Progress bars with filled blocks: `█████████░░░`
  - Enhanced tables with color-coded headers

- **High Contrast**
  - Bright colors optimized for dark backgrounds
  - Maximum contrast for low vision users
  - Essential information highlighted

### ⌨️ Keyboard Navigation

All interactive elements support keyboard-only navigation:

- Single-key selection for menu options
- Enter/Return to confirm
- Ctrl+C to cancel
- Tab completion where available

### 🔊 Audio Feedback

Optional audio cues for important events:

- Single beep for success
- Double beep for errors
- Customizable via `AURIK_AUDIO_FEEDBACK=1`

### 📊 Accessible Progress Indicators

- Text-based progress: `Processing: 50/100 (50.0%)`
- Visual progress bars automatically adapt to theme
- Both count and percentage displayed

## Environment Variables

Control accessibility features via environment variables:

```bash
# Disable all colors (screen reader mode)
export NO_COLOR=1
# or
export AURIK_NO_COLOR=1

# Enable high contrast mode
export AURIK_HIGH_CONTRAST=1

# Enable audio feedback
export AURIK_AUDIO_FEEDBACK=1
```

## Usage Examples

### Basic Usage in Python Scripts

```python
from usability.cli_accessibility import AccessibleCLI

# Initialize (auto-detects best theme)
cli = AccessibleCLI()

# Display messages
cli.header("Processing Audio Files")
cli.info("Starting batch processing...")
cli.success("File processed successfully!")
cli.warning("Skipping corrupted file")
cli.error("Processing failed: invalid format")

# Progress tracking
for i in range(100):
    cli.progress("Converting", i + 1, 100)

# Tables
cli.table(
    headers=["File", "Status", "Time"],
    rows=[["audio_01.wav", "Success", "2.3s"], ["audio_02.wav", "Failed", "0.1s"]],
    alignments=["left", "center", "right"],
)

# Interactive prompts
name = cli.prompt("Enter your name", default="User")
proceed = cli.confirm("Continue processing?", default=True)

# Audio feedback
cli.play_sound("success")  # Beep notification
```

### Batch Processor CLI

The batch processor automatically uses AccessibleCLI:

```bash
# Default mode (auto-detects theme)
python batch_processor_ui.py --auto

# Force plain mode for screen readers
NO_COLOR=1 python batch_processor_ui.py --auto

# High contrast mode
AURIK_HIGH_CONTRAST=1 python batch_processor_ui.py --auto

# With audio feedback
AURIK_AUDIO_FEEDBACK=1 python batch_processor_ui.py --auto
```

### Interactive Menu Example

```python
cli = AccessibleCLI()

cli.list_options(
    {"1": "Process all files", "2": "Select specific files", "3": "Change settings", "q": "Quit"}, title="Main Menu"
)

choice = cli.prompt("Select option", valid_choices=["1", "2", "3", "q"])
```

## Keyboard Shortcuts

### Global Shortcuts

| Key      | Action                    |
| -------- | ------------------------- |
| `1-9`    | Select menu option        |
| `y/n`    | Confirm/decline prompts   |
| `Enter`  | Confirm current selection |
| `Ctrl+C` | Cancel operation          |
| `Ctrl+D` | Exit program              |

### Batch Processor Shortcuts

| Key | Action             |
| --- | ------------------ |
| `1` | Discover files     |
| `2` | Show queue         |
| `3` | Filter queue       |
| `4` | Start processing   |
| `5` | Save report        |
| `0` | Exit               |

## Screen Reader Compatibility

AURIK CLI is compatible with popular screen readers:

### NVDA (Windows)

```bash
# Enable NVDA-friendly mode
set NO_COLOR=1
python batch_processor_ui.py
```

### JAWS (Windows)

```bash
# Enable JAWS-friendly mode
set AURIK_NO_COLOR=1
python batch_processor_ui.py
```

### VoiceOver (macOS)

```bash
# Enable VoiceOver-friendly mode
export NO_COLOR=1
python batch_processor_ui.py
```

### Orca (Linux)

```bash
# Enable Orca-friendly mode
NO_COLOR=1 python batch_processor_ui.py
```

## Best Practices for Developers

### 1. Always Use AccessibleCLI for Output

**Bad:**

```python
print("Processing files...")
print("✓ Done")
```

**Good:**

```python
cli = AccessibleCLI()
cli.info("Processing files...")
cli.success("Done")
```

### 2. Provide Descriptive Progress Updates

**Bad:**

```python
for i in range(100):
    print(f"{i}%")
```

**Good:**

```python
for i in range(100):
    cli.progress("Converting audio", i + 1, 100)
```

### 3. Use Semantic Message Types

```python
cli.success("Operation completed")  # Positive outcome
cli.error("Failed to load file")  # Fatal error
cli.warning("File already exists")  # Non-fatal issue
cli.info("Processing 10 files")  # Neutral information
cli.dim("Metadata: created 2024")  # Secondary information
```

### 4. Make Tables Accessible

```python
# Always provide headers
cli.table(
    headers=["Metric", "Value", "Unit"],
    rows=data,
    alignments=["left", "right", "left"],  # Align numbers right
)
```

### 5. Support Keyboard-Only Interaction

```python
# Use valid_choices for validation
choice = cli.prompt("Select mode", valid_choices=["1", "2", "3"], default="1")

# Use confirm for yes/no questions
if cli.confirm("Delete file?", default=False):
    delete_file()
```

## Testing Accessibility

### Manual Testing

```bash
# Test plain mode
NO_COLOR=1 python your_script.py

# Test high contrast
AURIK_HIGH_CONTRAST=1 python your_script.py

# Test with audio
AURIK_AUDIO_FEEDBACK=1 python your_script.py

# Test with screen reader
# 1. Enable screen reader software
# 2. Run: NO_COLOR=1 python your_script.py
# 3. Verify all output is read correctly
```

### Automated Testing

```python
import pytest
from usability.cli_accessibility import AccessibleCLI


def test_accessible_output(capsys):
    """Test CLI output is screen reader friendly"""
    cli = AccessibleCLI(theme="plain")

    cli.success("Test message")
    captured = capsys.readouterr()

    # Verify descriptive prefix
    assert "[SUCCESS]" in captured.out
    assert "Test message" in captured.out
```

Run tests:

```bash
pytest tests/test_cli_accessibility.py -v
```

## Compliance

AURIK CLI accessibility features comply with:

- **WCAG 2.1 Level AA** for terminal interfaces
- **Section 508** US accessibility standards
- **EN 301 549** European accessibility requirements
- **NO_COLOR standard** (<https://no-color.org/>)

## Troubleshooting

### Colors Not Showing

**Problem:** Output is plain text even without `NO_COLOR`

**Solution:** Check if stdout is a TTY:

```python
import sys

print(sys.stdout.isatty())  # Should be True
```

### Audio Feedback Not Working

**Problem:** No beeps on success/error

**Solution:**

1. Enable audio feedback: `export AURIK_AUDIO_FEEDBACK=1`
2. Check system audio volume
3. Some terminals may not support `\a` (bell character)

### Screen Reader Reads Too Fast

**Problem:** Screen reader skips information

**Solution:**

1. Use `NO_COLOR=1` for plain mode
2. Adjust screen reader verbosity settings
3. Use slower speech rate in screen reader preferences

### Unicode Characters Not Displaying

**Problem:** Progress bars show `?` instead of blocks

**Solution:**

1. Set terminal encoding to UTF-8: `export LANG=en_US.UTF-8`
2. Use plain mode: `export NO_COLOR=1`
3. Update terminal font to one with Unicode support

## Examples

### Complete Batch Processing Example

```python
#!/usr/bin/env python3
"""Accessible batch audio processor"""

from pathlib import Path
from usability.cli_accessibility import AccessibleCLI


def main():
    cli = AccessibleCLI()

    # Welcome header
    cli.header("AURIK Batch Processor")

    # Discover files
    cli.info("Discovering audio files...")
    files = list(Path("input").glob("*.wav"))
    cli.success(f"Found {len(files)} files")

    # Show files in table
    rows = [[str(i + 1), f.name, f"{f.stat().st_size / 1e6:.1f} MB"] for i, f in enumerate(files)]
    cli.table(["#", "Filename", "Size"], rows, ["right", "left", "right"])

    # Confirm processing
    if not cli.confirm(f"Process {len(files)} files?", default=True):
        cli.warning("Operation cancelled")
        return

    # Process files
    failed = 0
    for i, file in enumerate(files):
        cli.progress("Processing", i, len(files))

        try:
            process_audio(file)  # Your processing function
            cli.success(f"✓ {file.name}")
        except Exception as e:
            cli.error(f"✗ {file.name}: {e}")
            failed += 1

    # Summary
    cli.header("Processing Complete")
    cli.info(f"Processed: {len(files) - failed}")
    if failed > 0:
        cli.error(f"Failed: {failed}")
    else:
        cli.success("All files processed successfully!")
        cli.play_sound("success")


if __name__ == "__main__":
    main()
```

## Contributing

When adding new CLI features to AURIK:

1. **Always use AccessibleCLI** instead of `print()`
2. **Test with `NO_COLOR=1`** to ensure screen reader compatibility
3. **Provide keyboard alternatives** for all mouse interactions
4. **Add tests** to `tests/test_cli_accessibility.py`
5. **Document** new features in this guide

## Support

For accessibility issues or feature requests:

- File an issue on GitHub
- Tag with `accessibility` label
- Provide terminal/OS information
- Specify screen reader software (if applicable)

## References

- [NO_COLOR Standard](https://no-color.org/)
- [WCAG 2.1 Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
- [Section 508 Standards](https://www.section508.gov/)
- [Terminal ANSI Color Codes](https://en.wikipedia.org/wiki/ANSI_escape_code)
- [Screen Reader Compatibility](https://webaim.org/articles/screenreader_testing/)

---

**Version:** 8.0  
**Last Updated:** February 2026  
**Authors:** AURIK Development Team
