"""
Aurik10/ui/export_presets.py — Einfache Export-Voreinstellungen für Laien.

Zeigt freundliche Preset-Karten statt technischer Formate.
Integration: wird vor dem technischen ExportConfigDialog gezeigt.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets

from Aurik10.i18n import t


class ExportPresetDialog(QtWidgets.QDialog):
    """Zeigt Export-Presets als klickbare Karten."""

    preset_selected = QtCore.pyqtSignal(dict)

    PRESETS = [
        {
            "id": "whatsapp",
            "icon": "\U0001f4f1",
            "title_key": "export.preset.whatsapp",
            "desc_key": "export.preset.whatsapp_desc",
            "fmt": "mp3",
            "bitrate": "192k",
            "sr": 44100,
            "bits": 16,
        },
        {
            "id": "cd",
            "icon": "\U0001f4bf",
            "title_key": "export.preset.cd",
            "desc_key": "export.preset.cd_desc",
            "fmt": "wav",
            "sr": 44100,
            "bits": 16,
        },
        {
            "id": "email",
            "icon": "\U0001f4e7",
            "title_key": "export.preset.email",
            "desc_key": "export.preset.email_desc",
            "fmt": "mp3",
            "bitrate": "V2",
            "sr": 44100,
            "bits": 16,
        },
        {
            "id": "mobile",
            "icon": "\U0001f3a7",
            "title_key": "export.preset.mobile",
            "desc_key": "export.preset.mobile_desc",
            "fmt": "aac",
            "bitrate": "256k",
            "sr": 48000,
            "bits": 16,
        },
        {
            "id": "archive",
            "icon": "\U0001f3db",
            "title_key": "export.preset.archive",
            "desc_key": "export.preset.archive_desc",
            "fmt": "flac",
            "sr": 48000,
            "bits": 24,
        },
        {
            "id": "youtube",
            "icon": "\U0001f3ac",
            "title_key": "export.preset.youtube",
            "desc_key": "export.preset.youtube_desc",
            "fmt": "wav",
            "sr": 48000,
            "bits": 24,
        },
        {
            "id": "custom",
            "icon": "\U0001f527",
            "title_key": "export.preset.custom",
            "desc_key": "export.preset.custom_desc",
            "fmt": None,
        },
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("export.preset.title"))
        self.setMinimumSize(550, 420)
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)

        title = QtWidgets.QLabel(t("export.preset.title"))
        title.setStyleSheet("font-size: 16pt; font-weight: bold;")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QtWidgets.QLabel(t("export.preset.subtitle"))
        subtitle.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        grid = QtWidgets.QGridLayout()
        grid.setSpacing(8)
        row, col = 0, 0

        for p in self.PRESETS:
            btn = QtWidgets.QPushButton()
            btn.setMinimumSize(240, 80)
            btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
            btn_text = f"{p['icon']}  {t(p['title_key'])}\n{t(p['desc_key'])}"
            btn.setText(btn_text)
            btn.setStyleSheet("""
                QPushButton {
                    text-align: left;
                    padding: 12px;
                    border: 2px solid rgba(102,126,234,0.4);
                    border-radius: 12px;
                    font-size: 10pt;
                }
                QPushButton:hover {
                    border-color: #667eea;
                    background: rgba(102,126,234,0.15);
                }
            """)
            if p["fmt"] is None:
                btn.clicked.connect(lambda checked, preset=dict(p): self._on_custom(preset))
            else:
                btn.clicked.connect(lambda checked, preset=dict(p): self._on_select(preset))
            grid.addWidget(btn, row, col)
            col += 1
            if col > 1:
                col = 0
                row += 1

        layout.addLayout(grid)
        layout.addStretch()

    def _on_select(self, preset: dict):
        self.preset_selected.emit(preset)
        self.accept()

    def _on_custom(self, preset: dict):
        self.preset_selected.emit(preset)
        self.accept()
