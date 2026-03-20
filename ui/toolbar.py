from __future__ import annotations
# -*- coding: utf-8 -*-

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget

from utils.tooltips import apply_tooltip


class QuickExportBar(QWidget):
    """One-click export actions."""

    export_requested = Signal(str)  # format
    undo_requested = Signal()
    redo_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        # Undo / Redo
        self.btn_undo = QPushButton("Undo")
        self.btn_undo.setFixedHeight(28)
        self.btn_undo.clicked.connect(lambda: self.undo_requested.emit())
        apply_tooltip(self.btn_undo, "Undo last action", "Step back", "Ctrl+Z", "Undo")
        layout.addWidget(self.btn_undo)

        self.btn_redo = QPushButton("Redo")
        self.btn_redo.setFixedHeight(28)
        self.btn_redo.clicked.connect(lambda: self.redo_requested.emit())
        apply_tooltip(
            self.btn_redo, "Redo last action", "Step forward", "Ctrl+Y", "Redo"
        )
        layout.addWidget(self.btn_redo)

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

    def set_history_enabled(self, can_undo: bool, can_redo: bool) -> None:
        try:
            self.btn_undo.setEnabled(can_undo)
            self.btn_redo.setEnabled(can_redo)
        except Exception:
            pass
