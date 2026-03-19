from __future__ import annotations
# -*- coding: utf-8 -*-

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.config import SETTINGS


class SettingsPanel(QWidget):
    """Settings form widget used by the Settings dialog."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)

        grp_gen = QGroupBox("General & Performance")
        form_gen = QFormLayout(grp_gen)

        theme_label = QLabel("Light Mode Only")
        form_gen.addRow("Theme:", theme_label)

        self.chk_preview = QCheckBox("Enable live preview")
        self.chk_preview.setToolTip(
            "Renders a small PNG preview when properties change."
        )
        self.chk_preview.setChecked(
            bool(SETTINGS.get("ENABLE_PREVIEW", True, type=bool))
        )
        form_gen.addRow("Live Preview:", self.chk_preview)

        self.fps = QSpinBox()
        self.fps.setRange(15, 60)
        self.fps.setValue(int(SETTINGS.get("FPS", 15) or 15))
        form_gen.addRow("Preview FPS:", self.fps)

        self.quality = QComboBox()
        self.quality.addItems(["Low (ql)", "Medium (qm)", "High (qh)"])
        self.quality.setCurrentText(
            str(SETTINGS.get("QUALITY", "Low (ql)") or "Low (ql)")
        )
        form_gen.addRow("Quality:", self.quality)

        layout.addWidget(grp_gen)

        grp_ai = QGroupBox("Google Gemini AI")
        form_ai = QFormLayout(grp_ai)

        self.api_key = QLineEdit(str(SETTINGS.get("GEMINI_API_KEY", "") or ""))
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key.setPlaceholderText("Paste API key")
        form_ai.addRow("API Key:", self.api_key)

        self.gemini_model = QComboBox()
        self.gemini_model.addItems(["gemini-3-flash-preview", "gemini-3-pro-preview"])
        self.gemini_model.setCurrentText(
            str(SETTINGS.get("GEMINI_MODEL", "gemini-3-flash-preview") or "")
        )
        form_ai.addRow("Code Model:", self.gemini_model)

        self.tts_model = QComboBox()
        self.tts_model.addItems(
            ["gemini-2.5-flash-preview-tts", "gemini-2.5-pro-preview-tts"]
        )
        self.tts_model.setCurrentText(
            str(SETTINGS.get("TTS_MODEL", "gemini-2.5-flash-preview-tts") or "")
        )
        form_ai.addRow("TTS Model:", self.tts_model)

        layout.addWidget(grp_ai)

    def apply_settings(self) -> None:
        SETTINGS.set("GEMINI_API_KEY", self.api_key.text())
        SETTINGS.set("FPS", self.fps.value())
        SETTINGS.set("QUALITY", self.quality.currentText())
        SETTINGS.set("GEMINI_MODEL", self.gemini_model.currentText())
        SETTINGS.set("TTS_MODEL", self.tts_model.currentText())
        SETTINGS.set("ENABLE_PREVIEW", self.chk_preview.isChecked())
