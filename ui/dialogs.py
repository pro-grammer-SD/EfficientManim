from __future__ import annotations
# -*- coding: utf-8 -*-

from PySide6.QtWidgets import QDialog, QHBoxLayout, QPushButton, QTextEdit, QVBoxLayout

from ui.panels.settings_panel import SettingsPanel
from utils.tooltips import apply_tooltip


class KeyboardShortcutsDialog(QDialog):
    _SHORTCUTS_TEXT = """Global:
  Ctrl+N          New Project
  Ctrl+O          Open Project
  Ctrl+S          Save Project
  Ctrl+Q          Quit

Editing:
  Ctrl+Z          Undo
  Ctrl+Y          Redo
  Delete          Delete Selected
  Ctrl+L          Auto-Layout Nodes

Canvas:
  Middle Mouse    Pan Canvas
  Ctrl+Scroll     Zoom In/Out
  Ctrl+A          Select All Nodes

Use Help -> Edit Keybindings... to customize shortcuts.
"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.resize(500, 400)
        layout = QVBoxLayout(self)

        text_display = QTextEdit()
        text_display.setReadOnly(True)
        text_display.setPlainText(self._SHORTCUTS_TEXT)
        layout.addWidget(text_display)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        apply_tooltip(
            close_btn,
            "Close this dialog",
            "Returns to the editor",
            "Esc",
            "Close",
        )
        layout.addWidget(close_btn)


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(520, 460)
        layout = QVBoxLayout(self)

        self.panel = SettingsPanel(self)
        layout.addWidget(self.panel)

        btns = QHBoxLayout()
        b_save = QPushButton("Save and Close")
        b_save.setStyleSheet(
            "background-color: #2ecc71; color: white; font-weight: bold;"
        )
        b_save.clicked.connect(self._save)
        apply_tooltip(
            b_save,
            "Apply settings and close",
            "Changes take effect immediately",
            "Enter",
            "Save",
        )

        b_cancel = QPushButton("Cancel")
        b_cancel.clicked.connect(self.reject)
        apply_tooltip(
            b_cancel,
            "Discard changes",
            "Close without saving",
            "Esc",
            "Cancel",
        )

        btns.addStretch()
        btns.addWidget(b_save)
        btns.addWidget(b_cancel)
        layout.addLayout(btns)

    def _save(self):
        self.panel.apply_settings()
        self.accept()
