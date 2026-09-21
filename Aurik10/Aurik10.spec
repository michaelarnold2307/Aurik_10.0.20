# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller Specification for AURIK 9.10.57
Packages the application as a standalone executable
Target platforms: Linux (AppImage) and Windows 10/11 (.exe)
macOS is NOT supported (spec §13 — Desktop-only, Linux + Windows).
"""

block_cipher = None

# All Python dependencies that need to be included
hiddenimports = [
    # Scientific stack
    'scipy',
    'scipy.signal',
    'scipy.fft',
    'numpy',
    'soundfile',
    'librosa',
    'matplotlib',
    'matplotlib.backends.backend_qt5agg',
    'PyQt5',
    'PyQt5.QtCore',
    'PyQt5.QtWidgets',
    'PyQt5.QtGui',
    'pedalboard',
    'pedalboard.io',
    # Backend core (§11 — Schichten)
    'backend',
    'backend.core',
    'backend.api',
    'backend.api.bridge',
    'backend.core.unified_restorer_v3',
    'backend.core.feedback_chain',
    'backend.core.plugin_lifecycle_manager',
    'backend.core.perceptual_quality_scorer',
    'backend.core.mushra_evaluator',
    'backend.core.restorability_estimator',
    'backend.core.musical_goals',
    'backend.core.musical_goals.musical_goals_metrics',
    'backend.core.musical_goals.adaptive_goals_system',
    'backend.core.phases',
    # Denker-Orchestrierung (§2.2)
    'denker',
    'denker.aurik_denker',
    'denker.defekt_denker',
    'denker.exzellenz_denker',
    'denker.rekonstruktions_denker',
    'denker.reparatur_denker',
    'denker.restaurier_denker',
    'denker.strategie_denker',
    'denker.tontraeger_denker',
    'denker.tontraegerkette_denker',
    # Shared Enums
    'shared',
    'shared.enums',
    # DSP-Module
    'dsp',
    'dsp.bass_enhancement',
    'dsp.drums_enhancement',
    'dsp.guitar_enhancement',
    'dsp.piano_restoration',
    'dsp.brass_enhancement',
    'dsp.spatial_enhancement',
    # Plugins (ML-Engines, §13 — alle Primär-Plugins)
    'plugins',
    'plugins.panns_plugin',
    'plugins.crepe_plugin',
    'plugins.deepfilternet_v3_ii_plugin',
    'plugins.rmvpe_plugin',
    'plugins.beats_plugin',
    'plugins.sgmse_plugin',
    'plugins.silero_plugin',
    'plugins.vocos_plugin',
    'plugins.demucs_v4_plugin',
    'plugins.bs_roformer_plugin',
    'plugins.uvr_mdxnet_plugin',
    'plugins.apollo_plugin',
    'plugins.cqtdiff_plus_plugin',
    'plugins.flow_matching_plugin',
    'plugins.gacela_plugin',
    'plugins.mp_senet_plugin',
    'plugins.versa_plugin',
    'plugins.bigvgan_v2_plugin',
    'plugins.flashsr_plugin',
    'plugins.fcpe_plugin',
    'plugins.rmvpe_plugin',
    'plugins.laion_clap_plugin',
    'plugins.utmos_plugin',
    'plugins.matchering_plugin',
    'plugins.artifact_detection_plugin',
    'plugins.lyrics_transcriber_plugin',
    # Startup-Check (§13 Out-of-the-Box)
    'backend.core.startup_model_check',
    # i18n / Frontend
    'Aurik10',
    'Aurik10.i18n',
    'Aurik10.ui',
    'Aurik10.core',
]

# Data files to include (models, configs, resources)
# NOTE: Large lazy-load models (FlashSR 5.9 GB, MERT 3.9 GB) are excluded here;
# they are loaded on-demand from ~/.aurik/models/ or the app bundle's models/ dir.
import glob as _glob
import os as _os

def _model_glob(src_pattern: str, dst_dir: str):
    """Only add datas entry if at least one matching file exists at build time."""
    matches = _glob.glob(src_pattern, recursive=True)
    if matches:
        return [(src_pattern, dst_dir)]
    return []

datas = [
    # UI resources (icons, QSS stylesheets)
    ('Aurik10/resources/*', 'resources'),
    # Model manifest — required for startup integrity check (§13 Out-of-the-Box)
    ('models/manifest.json', 'models'),
    # ── Core ML models (< 200 MB each, DSP-critical, always bundled) ────────────
    *_model_glob('models/deepfilternet_v3_ii/**/*', 'models/deepfilternet_v3_ii'),
    *_model_glob('models/silero-vad/**/*',          'models/silero-vad'),
    *_model_glob('models/silero/**/*',              'models/silero'),
    *_model_glob('models/panns/**/*',               'models/panns'),
    *_model_glob('models/crepe/**/*',               'models/crepe'),
    *_model_glob('models/vocos/**/*',               'models/vocos'),
    *_model_glob('models/hifi_gan/**/*',            'models/hifi_gan'),
    *_model_glob('models/fcpe/**/*',                'models/fcpe'),
    *_model_glob('models/era_classifier/**/*',      'models/era_classifier'),
    *_model_glob('models/whisper/**/*',             'models/whisper'),
    *_model_glob('models/wav2vec2/**/*',            'models/wav2vec2'),
    # ── Medium-weight models (bundled when present at build time) ───────────────
    *_model_glob('models/sgmse_plus/**/*',          'models/sgmse_plus'),
    *_model_glob('models/mp_senet/**/*',            'models/mp_senet'),
    *_model_glob('models/versa/**/*',               'models/versa'),
    *_model_glob('models/utmosv2/**/*',             'models/utmosv2'),
    *_model_glob('models/uvr_mdx_net/**/*',         'models/uvr_mdx_net'),
    *_model_glob('models/matchering2.0/**/*',       'models/matchering2.0'),
    *_model_glob('models/madmom/**/*',              'models/madmom'),
    # ── Large models excluded from bundle — loaded lazily from ~/.aurik/models/ ─
    # models/melbandroformer  (860 MB)  → lazy via bs_roformer_plugin
    # models/htdemucs         (320 MB)  → lazy via htdemucs_plugin
    # models/flashsr          (5.9 GB)  → lazy via flashsr_plugin
    # models/mert             (3.9 GB)  → lazy via mert_plugin
    # models/apollo           (≈80 MB)  → lazy via apollo_plugin
]

a = Analysis(
    ['Aurik10/main.py'],
    pathex=['.'],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    # Sets OMP/BLAS/MKL thread limits + disables CUDA probing before ML init.
    runtime_hooks=['Aurik10/hooks/runtime_hook_threading.py'],
    excludes=[
        'tkinter',
        'test',
        'tests',
        'pytest',
        'setuptools',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AURIK910',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No console window
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='Aurik10/resources/icon.ico' if os.path.exists('Aurik10/resources/icon.ico') else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AURIK910',
)

# macOS ist NICHT unterstützt (spec §13 — Linux AppImage + Windows 10/11 only).
# Das BUNDLE-Statement wurde entfernt, da es den Build auf Linux/Windows bricht.
