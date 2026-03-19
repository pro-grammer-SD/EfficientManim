"""
screen_manager.py — Stateful Dual-Screen Architecture

Manages Editor and Timeline screens with persistent instances.
Screens are created once at startup and never recreated.
Switching uses visibility/stacking, never recreation.

Rules:
  - Both screens instantiated at app startup
  - Switching hides/shows — never destroys
  - State preservation across switches
  - Independent layout maintenance
  - Zero re-initialization overhead
"""

import logging
from typing import Optional
from enum import Enum, auto

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


class ScreenType(Enum):
    """Screen identifier."""

    EDITOR = auto()
    TIMELINE = auto()


class EditorScreen(QWidget if HAS_PYSIDE else object):
    """
    Editor workspace screen.

    Contains:
    - Canvas (node graph editor)
    - Properties panel
    - Inspector
    - Keybindings panel
    - Asset manager
    - LaTeX editor
    - Voiceover manager
    - Code view
    - Logs

    Instantiated once, never destroyed.
    Layout state persisted.
    """

    def __init__(self, parent=None):
        if HAS_PYSIDE:
            super().__init__(parent)

        self.setObjectName("EditorScreen")
        self._layout: Optional[QVBoxLayout] = None
        self._scroll_position = 0
        self._zoom_level = 1.0
        self._panel_visibility = {}

        if HAS_PYSIDE:
            self._setup_layout()

    def _setup_layout(self) -> None:
        """Setup editor screen layout."""
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        # Main container will be populated by __import__('ui.main_window').main_window.EfficientManimWindow
        # This is just the framework
        main_frame = QFrame()
        main_frame.setObjectName("EditorMainFrame")
        self._layout.addWidget(main_frame)

    def save_state(self) -> dict:
        """Save editor screen state."""
        return {
            "scroll_position": self._scroll_position,
            "zoom_level": self._zoom_level,
            "panel_visibility": dict(self._panel_visibility),
        }

    def restore_state(self, state: dict) -> None:
        """Restore editor screen state."""
        if not state:
            return
        self._scroll_position = state.get("scroll_position", 0)
        self._zoom_level = state.get("zoom_level", 1.0)
        self._panel_visibility = state.get("panel_visibility", {})


class TimelineScreen(QWidget if HAS_PYSIDE else object):
    """
    Timeline workspace screen.

    Contains:
    - Full multi-track deterministic timeline
    - Video preview monitor
    - Waveform display
    - Camera track editor
    - Render controls
    - Timeline inspector
    - Version timeline view

    Instantiated once, never destroyed.
    Layout state persisted.
    """

    def __init__(self, parent=None):
        if HAS_PYSIDE:
            super().__init__(parent)

        self.setObjectName("TimelineScreen")
        self._layout: Optional[QVBoxLayout] = None
        self._timeline_zoom = 1.0
        self._track_expansions = {}
        self._playhead_position = 0

        if HAS_PYSIDE:
            self._setup_layout()

    def _setup_layout(self) -> None:
        """Setup timeline screen layout."""
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        # Main timeline container will be populated by __import__('ui.main_window').main_window.EfficientManimWindow
        main_frame = QFrame()
        main_frame.setObjectName("TimelineMainFrame")
        self._layout.addWidget(main_frame)

    def save_state(self) -> dict:
        """Save timeline screen state."""
        return {
            "timeline_zoom": self._timeline_zoom,
            "track_expansions": dict(self._track_expansions),
            "playhead_position": self._playhead_position,
        }

    def restore_state(self, state: dict) -> None:
        """Restore timeline screen state."""
        if not state:
            return
        self._timeline_zoom = state.get("timeline_zoom", 1.0)
        self._track_expansions = state.get("track_expansions", {})
        self._playhead_position = state.get("playhead_position", 0)


class ScreenManager(QObject if HAS_PYSIDE else object):
    """
    Manages dual-screen system with persistent instances.

    Rules:
    - Both screens created at startup
    - Switching changes visibility only
    - No recreation
    - No re-initialization
    - State persisted across switches
    """

    if HAS_PYSIDE:
        screen_switched = Signal(int)  # (screen_type)
        state_changed = Signal(str)  # (screen_name)

    def __init__(self):
        if HAS_PYSIDE:
            super().__init__()

        self._current_screen: Optional[ScreenType] = None
        self._editor_screen: Optional[EditorScreen] = None
        self._timeline_screen: Optional[TimelineScreen] = None
        self._stacked_widget: Optional[QStackedWidget] = None
        self._saved_states: dict = {}

    def setup(self, stacked_widget: QStackedWidget) -> None:
        """
        Initialize screen manager with stacked widget.

        CRITICAL: Call this ONCE at app startup.
        Both screens are created here and NEVER RECREATED.
        """
        if not HAS_PYSIDE:
            return

        self._stacked_widget = stacked_widget

        # Create both screens ONCE
        self._editor_screen = EditorScreen()
        self._timeline_screen = TimelineScreen()

        # Add to stacked widget
        self._stacked_widget.addWidget(self._editor_screen)
        self._stacked_widget.addWidget(self._timeline_screen)

        # Start with editor screen
        self._current_screen = ScreenType.EDITOR
        self._stacked_widget.setCurrentWidget(self._editor_screen)

        LOGGER.info("✓ ScreenManager initialized with persistent screens")

    def switch_to_editor(self) -> None:
        """
        Switch to editor screen.

        No recreation. Just changes visibility.
        State automatically preserved.
        """
        if not HAS_PYSIDE or not self._stacked_widget:
            return

        if self._current_screen == ScreenType.EDITOR:
            return  # Already on editor

        # Save timeline state before switching away
        if self._current_screen == ScreenType.TIMELINE and self._timeline_screen:
            self._saved_states[ScreenType.TIMELINE.name] = (
                self._timeline_screen.save_state()
            )

        # Show editor
        self._stacked_widget.setCurrentWidget(self._editor_screen)
        self._current_screen = ScreenType.EDITOR

        # Restore editor state if previously saved
        if ScreenType.EDITOR.name in self._saved_states:
            self._editor_screen.restore_state(
                self._saved_states[ScreenType.EDITOR.name]
            )

        if HAS_PYSIDE:
            self.screen_switched.emit(ScreenType.EDITOR.value)

        LOGGER.info("✓ Switched to Editor Screen")

    def switch_to_timeline(self) -> None:
        """
        Switch to timeline screen.

        No recreation. Just changes visibility.
        State automatically preserved.
        """
        if not HAS_PYSIDE or not self._stacked_widget:
            return

        if self._current_screen == ScreenType.TIMELINE:
            return  # Already on timeline

        # Save editor state before switching away
        if self._current_screen == ScreenType.EDITOR and self._editor_screen:
            self._saved_states[ScreenType.EDITOR.name] = (
                self._editor_screen.save_state()
            )

        # Show timeline
        self._stacked_widget.setCurrentWidget(self._timeline_screen)
        self._current_screen = ScreenType.TIMELINE

        # Restore timeline state if previously saved
        if ScreenType.TIMELINE.name in self._saved_states:
            self._timeline_screen.restore_state(
                self._saved_states[ScreenType.TIMELINE.name]
            )

        if HAS_PYSIDE:
            self.screen_switched.emit(ScreenType.TIMELINE.value)

        LOGGER.info("✓ Switched to Timeline Screen")

    def get_current_screen(self) -> Optional[ScreenType]:
        """Get currently active screen."""
        return self._current_screen

    def get_editor_screen(self) -> Optional[EditorScreen]:
        """Get editor screen (persistent instance)."""
        return self._editor_screen

    def get_timeline_screen(self) -> Optional[TimelineScreen]:
        """Get timeline screen (persistent instance)."""
        return self._timeline_screen

    def save_all_state(self) -> dict:
        """Save state of both screens."""
        return {
            "current_screen": self._current_screen.name
            if self._current_screen
            else "EDITOR",
            "editor": self._editor_screen.save_state() if self._editor_screen else {},
            "timeline": self._timeline_screen.save_state()
            if self._timeline_screen
            else {},
        }

    def restore_all_state(self, state: dict) -> None:
        """Restore state of both screens."""
        if not state:
            return

        if self._editor_screen:
            self._editor_screen.restore_state(state.get("editor", {}))
        if self._timeline_screen:
            self._timeline_screen.restore_state(state.get("timeline", {}))

        # Restore current screen
        current = state.get("current_screen", "EDITOR")
        if current == "TIMELINE":
            self.switch_to_timeline()
        else:
            self.switch_to_editor()


# Global screen manager instance
SCREEN_MANAGER = ScreenManager()
