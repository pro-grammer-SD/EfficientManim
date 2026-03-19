from __future__ import annotations
# -*- coding: utf-8 -*-

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget

from utils.tooltips import apply_tooltip


class QuickExportBar(QWidget):
    """One-click export actions."""

    export_requested = Signal(str)  # format

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        for label, fmt, tip in [
            (
                "Export .py",
                "py",
                ("Save Python file", "Choose a location and save code", "Ctrl+E"),
            ),
            (
                "Copy Code",
                "copy",
                ("Copy to clipboard", "Use in your editor", "Ctrl+Shift+C"),
            ),
            (
                "Render MP4",
                "mp4",
                ("Render video", "Open the render panel and start", "Ctrl+R"),
            ),
        ]:
            btn = QPushButton(label)
            btn.setFixedHeight(28)
            btn.clicked.connect(lambda _=False, f=fmt: self.export_requested.emit(f))
            apply_tooltip(btn, tip[0], tip[1], tip[2], label)
            layout.addWidget(btn)
        layout.addStretch()
