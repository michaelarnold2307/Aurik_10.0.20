# AURIK Professional - Packaging Documentation

## Distribution Package

### Build Information

- **Build Tool**: PyInstaller 6.18.0
- **Python**: 3.10.12
- **Platform**: Linux x86_64
- **Package Type**: Directory bundle (onedir)
- **Package Size**: 7.0 GB

### Location

```
dist/AURIK_Professional/
├── AURIK_Professional          # Main executable
├── base_library.zip            # Python standard library
├── _internal/                  # Dependencies folder
│   ├── PyQt5/
│   ├── numpy/
│   ├── scipy/
│   ├── librosa/
│   ├── torch/                  # PyTorch (large!)
│   ├── matplotlib/
│   └── ...
```

### Execution

```bash
./dist/AURIK_Professional/AURIK_Professional
```

### Known Limitations

#### 1. PortAudio Not Bundled

- **Issue**: Audio playback requires PortAudio shared library
- **Warning**: "sounddevice not available (PortAudio library not found)"
- **Impact**: Audio preview playback disabled
- **Workaround**: Install system package:
  ```bash
  sudo apt install libportaudio2
  ```

#### 2. Large Package Size (7.0 GB)

- **Cause**: PyTorch bundled with all CUDA dependencies
- **Note**: AURIK doesn't use PyTorch directly, it's pulled in by librosa
- **Solution**: See "Size Optimization" below

### Size Optimization (Future)

#### Option 1: Exclude PyTorch

Create custom PyInstaller hook to exclude torch:
```python
# exclude_torch.py hook
hiddenimports = []
excludes = ["torch", "nvidia.*"]
```

#### Option 2: Lightweight Installation

- Strip CUDA libraries (if not using GPU)
- Use CPU + optionale AMD-GPU PyTorch build
- Exclude test/debug modules

Expected size reduction: **7.0 GB → 1.5 GB**

### Platform-Specific Builds

#### Linux (Current)

- ✅ Built and tested on Ubuntu/Zorin OS
- Format: ELF 64-bit executable
- Dependencies: glibc 2.35+

#### Windows (Future)

```bash
pyinstaller aurik_professional.spec --windowed
```
- Output: `.exe` executable
- Requires: Visual C++ Redistributable

#### macOS (Future)

```bash
pyinstaller aurik_professional.spec --onedir --windowed
```
- Output: `.app` bundle
- Code signing required for distribution

### Distribution Formats

#### Current: Directory Bundle

- Fast startup
- Easy debugging
- Larger disk footprint
- Users see folder structure

#### Alternative: Single File

```python
# In .spec file:
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,  # Include all in one file
    a.zipfiles,
    a.datas,
    ...
    onefile=True  # Single executable
)
```
- Slower startup (extracts to temp)
- Cleaner distribution
- Larger memory usage

### Testing Checklist

✅ Application launches
✅ GUI renders correctly
✅ File loading works
✅ Processing pipeline functional
✅ Waveform visualization displays
✅ Batch queue operates
✅ Presets load/save
⚠️ Audio playback (requires PortAudio)

### Deployment Steps

1. **Local Testing**
   ```bash
   ./dist/AURIK_Professional/AURIK_Professional
   ```

2. **Package for Distribution**
   ```bash
   cd dist
   tar -czf AURIK_Professional_Linux_v1.0.tar.gz AURIK_Professional/
   ```

3. **Create Installer (Optional)**
   - Linux: Create `.deb` package or AppImage
   - Windows: Use NSIS or Inno Setup
   - macOS: Create `.dmg` with installer

4. **Documentation**
   - Include README with installation instructions
   - List system requirements (Qt5, X11, PortAudio optional)
   - Provide sample audio files

### System Requirements

**Minimum:**
- OS: Ubuntu 20.04+ / Zorin OS 17+ / equivalent
- CPU: Dual-core 2.0 GHz
- RAM: 4 GB
- Disk: 8 GB free space
- Display: 1024x768, X11 server

**Recommended:**
- OS: Ubuntu 22.04+ / Zorin OS 17+
- CPU: Quad-core 3.0 GHz+
- RAM: 16 GB
- Disk: 10 GB SSD
- Display: 1920x1080, X11/Wayland
- Audio: PortAudio installed for playback

### Build Reproducibility

1. **Activate Virtual Environment**
   ```bash
   source .venv_aurik/bin/activate
   ```

2. **Install PyInstaller**
   ```bash
   pip install pyinstaller
   ```

3. **Run Build Script**
   ```bash
   ./build_executable.sh
   ```

Build time: ~3 minutes on modern hardware

### Troubleshooting

#### "libpython3.10.so not found"

```bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

#### "Qt platform plugin not found"

- Included in bundle, should not occur
- If occurs: Install `libqt5gui5`, `libqt5widgets5`

#### "No module named 'backend'"

- Indicates missing hidden imports
- Add to `.spec` file `hiddenimports` list

### Next Steps

- [ ] Optimize package size (remove PyTorch)
- [ ] Bundle PortAudio library
- [ ] Create installer packages (.deb, .rpm)
- [ ] Build Windows version
- [ ] Build macOS version
- [ ] Set up CI/CD for automated builds
- [ ] Add update mechanism
- [ ] Create desktop integration (.desktop file, icon)
