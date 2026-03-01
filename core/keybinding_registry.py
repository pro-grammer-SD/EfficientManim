"""
keybinding_registry.py — Unified Keybinding System

Single source of truth for all application keybindings.

GOVERNANCE RULES:
  1. One authoritative registry
  2. Persistent storage (config file)
  3. No duplicate definitions
  4. Dynamic QAction rebinding without restart
  5. Conflict prevention and warning
  6. Both UI panels read from SAME registry

Usage:
    from keybinding_registry import KeybindingRegistry, KEYBINDINGS

    # Register an action
    KEYBINDINGS.register_action("Save Project", "Ctrl+S", "Save the current project")

    # Get a binding
    binding = KEYBINDINGS.get_binding("Save Project")

    # Update a binding
    KEYBINDINGS.set_binding("Save Project", "Ctrl+Shift+S")

    # Listen for changes
    KEYBINDINGS.changed.connect(on_keybinding_changed)
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass

try:
    from PySide6.QtCore import QObject, Signal, QSettings
    from PySide6.QtGui import QKeySequence

    HAS_PYSIDE = True
except ImportError:
    HAS_PYSIDE = False
    QObject = object
    Signal = None


LOGGER = logging.getLogger("keybinding_registry")


@dataclass
class KeybindingAction:
    """Represents a single keybinding action."""

    name: str
    default_shortcut: str
    description: str
    user_override: Optional[str] = None

    def get_current_shortcut(self) -> str:
        """Get currently active shortcut (user override or default)."""
        return self.user_override if self.user_override else self.default_shortcut

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "name": self.name,
            "default": self.default_shortcut,
            "description": self.description,
            "user_override": self.user_override,
        }

    @staticmethod
    def from_dict(d: dict) -> "KeybindingAction":
        """Deserialize from dict."""
        return KeybindingAction(
            name=d["name"],
            default_shortcut=d["default"],
            description=d.get("description", ""),
            user_override=d.get("user_override"),
        )


class KeybindingRegistry(QObject if HAS_PYSIDE else object):
    """
    Unified keybinding registry.

    Single source of truth for all keybindings in the application.
    Handles registration, persistence, conflict detection, and change notifications.
    """

    # Signal emitted when any keybinding changes
    if HAS_PYSIDE:
        binding_changed = Signal(str, str)  # (action_name, new_shortcut)
        registry_updated = Signal()  # emitted when registry structure changes

    def __init__(self, config_path: Optional[Path] = None):
        if HAS_PYSIDE:
            super().__init__()

        self._actions: Dict[str, KeybindingAction] = {}
        self._config_path = (
            config_path or Path.home() / ".efficientmanim" / "keybindings.json"
        )
        self._reserved_shortcuts: set = set()  # Track all shortcuts to detect conflicts

        # Create config directory if needed
        self._config_path.parent.mkdir(parents=True, exist_ok=True)

        # Load from disk
        self._load()

    def register_action(
        self, name: str, default_shortcut: str, description: str = ""
    ) -> None:
        """
        Register a new keybinding action.

        Args:
            name: Unique action name
            default_shortcut: Default shortcut string (e.g., "Ctrl+S")
            description: Human-readable description
        """
        if name in self._actions:
            LOGGER.warning(f"Action '{name}' already registered. Skipping.")
            return

        action = KeybindingAction(name, default_shortcut, description)
        self._actions[name] = action
        self._reserved_shortcuts.add(default_shortcut)

        if HAS_PYSIDE:
            self.registry_updated.emit()

        LOGGER.info(f"Registered action: {name} -> {default_shortcut}")

    def get_action(self, name: str) -> Optional[KeybindingAction]:
        """Get action by name."""
        return self._actions.get(name)

    def get_binding(self, action_name: str) -> str:
        """
        Get current shortcut for an action.

        Returns the user override if set, otherwise the default.
        """
        action = self._actions.get(action_name)
        if not action:
            return ""
        return action.get_current_shortcut()

    def set_binding(
        self, action_name: str, shortcut: str, check_conflicts: bool = True
    ) -> Tuple[bool, str]:
        """
        Set a keybinding for an action.

        Args:
            action_name: Name of the action
            shortcut: New shortcut string
            check_conflicts: If True, check for conflicts

        Returns:
            (success, message)
        """
        action = self._actions.get(action_name)
        if not action:
            return False, f"Action not found: {action_name}"

        # Check for conflicts
        if check_conflicts:
            conflict = self._find_conflicting_action(shortcut, action_name)
            if conflict:
                msg = f"Conflict: '{shortcut}' is already bound to '{conflict}'"
                LOGGER.warning(msg)
                return False, msg

        # Set the override
        action.user_override = shortcut if shortcut != action.default_shortcut else None

        # Persist
        self._save()

        # Notify
        if HAS_PYSIDE:
            self.binding_changed.emit(action_name, shortcut)

        LOGGER.info(f"Updated binding: {action_name} -> {shortcut}")
        return True, f"Updated: {action_name} -> {shortcut}"

    def reset_binding(self, action_name: str) -> bool:
        """Reset a binding to its default."""
        action = self._actions.get(action_name)
        if not action:
            return False

        action.user_override = None
        self._save()

        if HAS_PYSIDE:
            self.binding_changed.emit(action_name, action.default_shortcut)

        LOGGER.info(f"Reset binding: {action_name} -> {action.default_shortcut}")
        return True

    def reset_all(self) -> None:
        """Reset all bindings to defaults."""
        for action in self._actions.values():
            action.user_override = None
        self._save()

        if HAS_PYSIDE:
            self.registry_updated.emit()

        LOGGER.info("Reset all bindings to defaults")

    def get_all_actions(self) -> List[KeybindingAction]:
        """Get all registered actions."""
        return list(self._actions.values())

    def get_all_action_names(self) -> List[str]:
        """Get names of all registered actions."""
        return sorted(self._actions.keys())

    def _find_conflicting_action(
        self, shortcut: str, exclude_action: str = ""
    ) -> Optional[str]:
        """Find which action is using a shortcut (if any, excluding specified action)."""
        for name, action in self._actions.items():
            if name == exclude_action:
                continue
            if action.get_current_shortcut() == shortcut and shortcut:
                return name
        return None

    def validate_shortcuts(self) -> List[str]:
        """
        Validate all shortcuts for conflicts.

        Returns:
            List of conflict messages (empty if all valid)
        """
        conflicts = []
        seen = {}

        for name, action in self._actions.items():
            shortcut = action.get_current_shortcut()
            if not shortcut:
                continue

            if shortcut in seen:
                msg = f"Conflict: '{shortcut}' assigned to both '{name}' and '{seen[shortcut]}'"
                conflicts.append(msg)
            else:
                seen[shortcut] = name

        return conflicts

    def _save(self) -> None:
        """Persist registry to config file."""
        try:
            data = {
                "actions": {
                    name: action.to_dict() for name, action in self._actions.items()
                }
            }
            with open(self._config_path, "w") as f:
                json.dump(data, f, indent=2)
            LOGGER.debug(f"Saved keybindings to {self._config_path}")
        except Exception as e:
            LOGGER.error(f"Failed to save keybindings: {e}")

    def _load(self) -> None:
        """Load registry from config file."""
        if not self._config_path.exists():
            LOGGER.debug(f"No keybindings config found at {self._config_path}")
            return

        try:
            with open(self._config_path, "r") as f:
                data = json.load(f)

            for action_data in data.get("actions", {}).values():
                action = KeybindingAction.from_dict(action_data)
                self._actions[action.name] = action
                shortcut = action.get_current_shortcut()
                if shortcut:
                    self._reserved_shortcuts.add(shortcut)

            LOGGER.debug(
                f"Loaded {len(self._actions)} keybindings from {self._config_path}"
            )
        except Exception as e:
            LOGGER.error(f"Failed to load keybindings: {e}")

    def to_dict(self) -> dict:
        """Export registry as dict."""
        return {
            "actions": {
                name: action.to_dict() for name, action in self._actions.items()
            }
        }

    def export_json(self, indent: int = 2) -> str:
        """Export registry as JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


# Global registry instance
KEYBINDINGS = KeybindingRegistry()


# STANDARD DEFAULT BINDINGS (unified from both old systems)
DEFAULT_KEYBINDINGS = [
    ("New Project", "Ctrl+N", "Create a new project"),
    ("Open Project", "Ctrl+O", "Open an existing project"),
    ("Save Project", "Ctrl+S", "Save the current project"),
    ("Save As", "Ctrl+Shift+S", "Save project with a new name"),
    ("Exit", "Ctrl+Q", "Exit the application"),
    ("Undo", "Ctrl+Z", "Undo last action"),
    ("Redo", "Ctrl+Y", "Redo last action"),
    ("Delete Selected", "Del", "Delete selected nodes/wires"),
    ("Zoom In", "Ctrl+=", "Zoom in on canvas"),
    ("Zoom Out", "Ctrl+-", "Zoom out on canvas"),
    ("Fit View", "Ctrl+0", "Fit scene to view"),
    ("Clear All", "Ctrl+Alt+Del", "Clear all nodes (requires confirmation)"),
    ("Auto-Layout", "Ctrl+L", "Auto-layout node graph"),
    ("Export Code", "Ctrl+E", "Export generated code"),
    ("Copy Code", "Ctrl+Shift+C", "Copy generated code to clipboard"),
    ("Keybindings", "Ctrl+K", "Open keybindings editor"),
    ("Settings", "Ctrl+,", "Open settings dialog"),
    ("AI Generate", "Ctrl+G", "Generate content with AI"),
    ("Render Video", "Ctrl+R", "Render the video"),
    ("Switch to Editor", "Ctrl+1", "Switch to editor screen"),
    ("Switch to Timeline", "Ctrl+2", "Switch to timeline screen"),
    ("Next Tab", "Ctrl+Tab", "Switch to next tab"),
    ("Previous Tab", "Ctrl+Shift+Tab", "Switch to previous tab"),
    ("Duplicate Node", "Ctrl+D", "Duplicate selected node"),
    ("Lock Timing", "Ctrl+Shift+L", "Lock node timing (timeline)"),
    ("Unlock Timing", "Ctrl+Shift+U", "Unlock node timing (timeline)"),
]


def initialize_default_keybindings() -> None:
    """Register all default keybindings in the global registry."""
    for name, shortcut, description in DEFAULT_KEYBINDINGS:
        KEYBINDINGS.register_action(name, shortcut, description)
    LOGGER.info(f"Initialized {len(DEFAULT_KEYBINDINGS)} default keybindings")
