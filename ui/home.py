from __future__ import annotations
# -*- coding: utf-8 -*-

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.file_manager import get_recents
from utils.tooltips import apply_tooltip


class HomeScreen(QWidget):
    """Modern, beginner-friendly home screen."""

    new_project_requested = Signal()
    open_project_requested = Signal()
    open_example_requested = Signal()
    open_docs_requested = Signal()
    recent_project_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        # Explicit light background — HomeScreen may be hosted in an unstyled
        # QMainWindow before the editor theme is applied, so never leave it
        # transparent (inherits system dark palette on many platforms).
        self.setStyleSheet(
            "HomeScreen { background-color: #ffffff; }"
            "HomeScreen QWidget { background-color: #ffffff; color: #1f2937; }"
            "HomeScreen QListWidget { background-color: #f9fafb; border: 1px solid #e5e7eb; border-radius: 6px; }"
            "HomeScreen QListWidget::item { padding: 6px 10px; border-radius: 4px; }"
            "HomeScreen QListWidget::item:hover { background-color: #dbeafe; }"
            "HomeScreen QListWidget::item:selected { background-color: #2563eb; color: #ffffff; }"
            "HomeScreen QPushButton { background-color: #2563eb; color: #ffffff; border: none; border-radius: 6px; padding: 6px 16px; font-weight: 600; }"
            "HomeScreen QPushButton:hover { background-color: #1d4ed8; }"
            "HomeScreen QPushButton:pressed { background-color: #1e40af; }"
        )
        self._build()
        self.refresh_recents()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 28, 32, 28)
        root.setSpacing(18)

        title = QLabel("EfficientManim")
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title.setFont(title_font)
        root.addWidget(title)

        subtitle = QLabel(
            "Build Manim animations visually with a node-based editor, "
            "live previews, and structured rendering."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #4b5563; font-size: 12px;")
        root.addWidget(subtitle)

        actions = QFrame()
        actions_layout = QHBoxLayout(actions)
        actions_layout.setContentsMargins(0, 8, 0, 8)
        actions_layout.setSpacing(12)

        self.btn_new = QPushButton("New Project")
        self.btn_open = QPushButton("Open Project")
        self.btn_example = QPushButton("Example Project")
        self.btn_docs = QPushButton("Documentation")

        for btn in [self.btn_new, self.btn_open, self.btn_example, self.btn_docs]:
            btn.setMinimumHeight(36)
            actions_layout.addWidget(btn)

        self.btn_new.clicked.connect(self.new_project_requested.emit)
        self.btn_open.clicked.connect(self.open_project_requested.emit)
        self.btn_example.clicked.connect(self.open_example_requested.emit)
        self.btn_docs.clicked.connect(self.open_docs_requested.emit)

        apply_tooltip(
            self.btn_new,
            "Create a new project",
            "Start with an empty scene",
            "Ctrl+N",
            "New Project",
        )
        apply_tooltip(
            self.btn_open,
            "Open an existing project",
            "Choose a .efp file",
            "Ctrl+O",
            "Open Project",
        )
        apply_tooltip(
            self.btn_example,
            "Open a guided example",
            "Explore a ready-made graph",
            None,
            "Example Project",
        )
        apply_tooltip(
            self.btn_docs,
            "Open documentation",
            "Read the guide and references",
            None,
            "Documentation",
        )

        root.addWidget(actions)

        sections = QHBoxLayout()
        sections.setSpacing(18)

        # Recent Projects
        recent_frame = QFrame()
        recent_layout = QVBoxLayout(recent_frame)
        recent_layout.setContentsMargins(0, 0, 0, 0)
        recent_layout.setSpacing(8)

        recent_title = QLabel("Recent Projects")
        recent_title.setStyleSheet("font-weight: 600; font-size: 13px;")
        recent_layout.addWidget(recent_title)

        self.recents_list = QListWidget()
        self.recents_list.setMinimumWidth(320)
        self.recents_list.itemDoubleClicked.connect(self._open_recent)
        recent_layout.addWidget(self.recents_list, 1)

        self.empty_recents = QLabel("No recent projects yet.")
        self.empty_recents.setStyleSheet("color: #6b7280; font-size: 11px;")
        recent_layout.addWidget(self.empty_recents)

        sections.addWidget(recent_frame, 2)

        # Getting Started
        start_frame = QFrame()
        start_layout = QVBoxLayout(start_frame)
        start_layout.setContentsMargins(0, 0, 0, 0)
        start_layout.setSpacing(8)

        start_title = QLabel("Getting Started")
        start_title.setStyleSheet("font-weight: 600; font-size: 13px;")
        start_layout.addWidget(start_title)

        steps = [
            "1. Create or open a project.",
            "2. Add Mobjects and Animations from the Elements tab.",
            "3. Connect nodes to define the timeline.",
            "4. Preview and render from the Video panel.",
        ]
        for step in steps:
            lbl = QLabel(step)
            lbl.setWordWrap(True)
            lbl.setStyleSheet("color: #374151; font-size: 11px;")
            start_layout.addWidget(lbl)

        start_layout.addStretch(1)

        sections.addWidget(start_frame, 1)

        root.addLayout(sections)
        root.addStretch(1)

    def refresh_recents(self):
        self.recents_list.clear()
        recents = get_recents()
        if not recents:
            self.empty_recents.show()
            return
        self.empty_recents.hide()
        for path in recents:
            item = QListWidgetItem(Path(path).name)
            item.setToolTip(str(path))
            item.setData(Qt.ItemDataRole.UserRole, path)
            self.recents_list.addItem(item)

    def _open_recent(self, item: QListWidgetItem):
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            self.recent_project_requested.emit(path)
