"""
screen_manager.py — Editor Screen Manager

Manages the persistent Editor screen instance.
Switching uses visibility/stacking, never recreation.

Rules:
  - Screen instantiated at app startup
  - State preservation across interactions
  - Zero re-initialization overhead
"""

import logging
from typing import Optional

try:
    from PySide6.QtWidgets import (
        QWidget,
        QStackedWidget,
        QVBoxLayout,
        QFrame,
    )
    from PySide6.QtCore import Signal, QObject

    HAS_PYSIDE = True
except ImportError:
    HAS_PYSIDE = False
    QWidget = object
    QStackedWidget = object
    Signal = None
    QObject = object


LOGGER = logging.getLogger("screen_manager")


class EditorScreen(QWidget if HAS_PYSIDE else object):
    """
    Editor workspace screen — instantiated once, never destroyed.

    Contains:
    - Canvas (node graph editor)
    - Properties panel
    - Inspector
    - Asset manager
    - LaTeX editor
    - Voiceover manager
    - Code view
    - Logs
    """

    def __init__(self, parent=None):
        if HAS_PYSIDE:
            super().__init__(parent)

        self.setObjectName("EditorScreen")
        self._scroll_position = 0
        self._zoom_level = 1.0
        self._panel_visibility: dict = {}

        if HAS_PYSIDE:
            self._setup_layout()

    def _setup_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        main_frame = QFrame()
        main_frame.setObjectName("EditorMainFrame")
        layout.addWidget(main_frame)

    def save_state(self) -> dict:
        return {
            "scroll_position": self._scroll_position,
            "zoom_level": self._zoom_level,
            "panel_visibility": dict(self._panel_visibility),
        }

    def restore_state(self, state: dict) -> None:
        if not state:
            return
        self._scroll_position = state.get("scroll_position", 0)
        self._zoom_level = state.get("zoom_level", 1.0)
        self._panel_visibility = state.get("panel_visibility", {})


class ScreenManager(QObject if HAS_PYSIDE else object):
    """
    Manages the editor screen with a persistent instance.

    Rules:
    - Screen created at startup
    - State persisted across interactions
    - No recreation
    """

    if HAS_PYSIDE:
        state_changed = Signal(str)

    def __init__(self):
        if HAS_PYSIDE:
            super().__init__()

        self._editor_screen: Optional[EditorScreen] = None
        self._stacked_widget: Optional[QStackedWidget] = None
        self._saved_states: dict = {}

    def setup(self, stacked_widget: "QStackedWidget") -> None:
        """
        Initialize screen manager with stacked widget.

        CRITICAL: Call this ONCE at app startup.
        """
        if not HAS_PYSIDE:
            return

        self._stacked_widget = stacked_widget
        self._editor_screen = EditorScreen()
        self._stacked_widget.addWidget(self._editor_screen)
        self._stacked_widget.setCurrentWidget(self._editor_screen)

        LOGGER.info("✓ ScreenManager initialised")

    def get_editor_screen(self) -> Optional[EditorScreen]:
        return self._editor_screen

    def save_all_state(self) -> dict:
        return {
            "editor": self._editor_screen.save_state() if self._editor_screen else {},
        }

    def restore_all_state(self, state: dict) -> None:
        if not state:
            return
        if self._editor_screen:
            self._editor_screen.restore_state(state.get("editor", {}))


SCREEN_MANAGER = ScreenManager()
