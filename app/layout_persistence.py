"""
layout_persistence.py — UI Layout State Persistence

Saves and restores:
- Dock positions and sizes
- Panel visibility
- Splitter states
- Window geometry
- Tab indexes
- Screen positions
- Zoom levels
- Scroll positions

Persists to JSON config file.
Survives app restarts.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

LOGGER = logging.getLogger("layout_persistence")


class LayoutPersistenceManager:
    """
    Manages persistence of UI layout state.

    Saves:
    - Main window geometry
    - Dock widget positions
    - Splitter states
    - Panel visibility
    - Tab selections
    - Scroll/zoom positions
    """

    CONFIG_FILE = Path.home() / ".efficientmanim" / "layout.json"

    def __init__(self):
        self._state: Dict[str, Any] = {}
        self._enabled = True
        self._load()

    def save_geometry(self, window_state: bytes, window_geometry: bytes) -> None:
        """
        Save main window state and geometry.

        Args:
            window_state: QMainWindow.saveState()
            window_geometry: QMainWindow.saveGeometry()
        """
        if not self._enabled:
            return

        try:
            # Convert bytes to hex string for JSON serialization
            self._state["window_state"] = window_state.hex() if window_state else ""
            self._state["window_geometry"] = (
                window_geometry.hex() if window_geometry else ""
            )
            self._save()
            LOGGER.debug("Saved window geometry")
        except Exception as e:
            LOGGER.error(f"Failed to save geometry: {e}")

    def restore_geometry(self, window) -> Tuple[Optional[bytes], Optional[bytes]]:
        """
        Restore main window state and geometry.

        Returns:
            (state_bytes, geometry_bytes) or (None, None)
        """
        try:
            state_hex = self._state.get("window_state", "")
            geometry_hex = self._state.get("window_geometry", "")

            state_bytes = bytes.fromhex(state_hex) if state_hex else None
            geometry_bytes = bytes.fromhex(geometry_hex) if geometry_hex else None

            if state_bytes and geometry_bytes:
                LOGGER.debug("Restored window geometry from cache")

            return state_bytes, geometry_bytes
        except Exception as e:
            LOGGER.error(f"Failed to restore geometry: {e}")
            return None, None

    def save_panel_visibility(self, panel_states: Dict[str, bool]) -> None:
        """
        Save visibility state of all panels.

        Args:
            panel_states: {panel_name: is_visible}
        """
        if not self._enabled:
            return

        self._state["panel_visibility"] = panel_states
        self._save()
        LOGGER.debug(f"Saved visibility for {len(panel_states)} panels")

    def get_panel_visibility(self) -> Dict[str, bool]:
        """Get saved panel visibility states."""
        return self._state.get("panel_visibility", {})

    def save_tab_index(self, tab_widget_name: str, index: int) -> None:
        """
        Save current tab selection.

        Args:
            tab_widget_name: Name of QTabWidget
            index: Currently selected tab index
        """
        if not self._enabled:
            return

        if "tab_indexes" not in self._state:
            self._state["tab_indexes"] = {}

        self._state["tab_indexes"][tab_widget_name] = index
        self._save()
        LOGGER.debug(f"Saved tab index for {tab_widget_name}: {index}")

    def get_tab_index(self, tab_widget_name: str) -> Optional[int]:
        """Get saved tab index."""
        tab_indexes = self._state.get("tab_indexes", {})
        return tab_indexes.get(tab_widget_name)

    def save_splitter_state(self, splitter_name: str, state: bytes) -> None:
        """
        Save splitter geometry.

        Args:
            splitter_name: Name of QSplitter
            state: splitter.saveState()
        """
        if not self._enabled:
            return

        try:
            if "splitter_states" not in self._state:
                self._state["splitter_states"] = {}

            self._state["splitter_states"][splitter_name] = state.hex() if state else ""
            self._save()
            LOGGER.debug(f"Saved splitter state for {splitter_name}")
        except Exception as e:
            LOGGER.error(f"Failed to save splitter state: {e}")

    def get_splitter_state(self, splitter_name: str) -> Optional[bytes]:
        """Get saved splitter state."""
        try:
            splitter_states = self._state.get("splitter_states", {})
            state_hex = splitter_states.get(splitter_name, "")
            return bytes.fromhex(state_hex) if state_hex else None
        except Exception as e:
            LOGGER.error(f"Failed to restore splitter state: {e}")
            return None

    def save_scroll_position(self, widget_name: str, pos_x: int, pos_y: int) -> None:
        """Save scroll position for a widget."""
        if not self._enabled:
            return

        if "scroll_positions" not in self._state:
            self._state["scroll_positions"] = {}

        self._state["scroll_positions"][widget_name] = {"x": pos_x, "y": pos_y}
        self._save()

    def get_scroll_position(self, widget_name: str) -> Optional[tuple]:
        """Get saved scroll position."""
        scroll_positions = self._state.get("scroll_positions", {})
        pos = scroll_positions.get(widget_name)
        if pos:
            return (pos.get("x", 0), pos.get("y", 0))
        return None

    def save_zoom_level(self, widget_name: str, zoom: float) -> None:
        """Save zoom level for a widget."""
        if not self._enabled:
            return

        if "zoom_levels" not in self._state:
            self._state["zoom_levels"] = {}

        self._state["zoom_levels"][widget_name] = zoom
        self._save()

    def get_zoom_level(self, widget_name: str) -> float:
        """Get saved zoom level."""
        zoom_levels = self._state.get("zoom_levels", {})
        return zoom_levels.get(widget_name, 1.0)

    def save_screen_state(self, screen_name: str, state: dict) -> None:
        """
        Save full screen state.

        Args:
            screen_name: "EDITOR" or "TIMELINE"
            state: Screen-specific state dict
        """
        if not self._enabled:
            return

        if "screen_states" not in self._state:
            self._state["screen_states"] = {}

        self._state["screen_states"][screen_name] = state
        self._save()
        LOGGER.debug(f"Saved state for {screen_name} screen")

    def get_screen_state(self, screen_name: str) -> Optional[dict]:
        """Get saved screen state."""
        screen_states = self._state.get("screen_states", {})
        return screen_states.get(screen_name)

    def reset_layout(self) -> None:
        """
        Reset all layout to defaults.
        Useful for corrupted/broken layouts.
        """
        self._state = {}
        self._save()
        LOGGER.info("Layout reset to defaults")

    def _save(self) -> None:
        """Persist state to JSON file."""
        try:
            self.CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(self.CONFIG_FILE, "w") as f:
                json.dump(self._state, f, indent=2, default=str)
            LOGGER.debug(f"Saved layout to {self.CONFIG_FILE}")
        except Exception as e:
            LOGGER.error(f"Failed to save layout: {e}")

    def _load(self) -> None:
        """Load state from JSON file."""
        if not self.CONFIG_FILE.exists():
            LOGGER.debug(f"No layout config found at {self.CONFIG_FILE}")
            return

        try:
            with open(self.CONFIG_FILE, "r") as f:
                self._state = json.load(f)
            LOGGER.debug(f"Loaded layout from {self.CONFIG_FILE}")
        except Exception as e:
            LOGGER.error(f"Failed to load layout: {e}")


# Global layout persistence instance
LAYOUT_PERSISTENCE = LayoutPersistenceManager()
