# ruff: noqa: E402
from PySide6.QtWidgets import QMainWindow

"""
extension_registry.py — Central registry for extension-provided panels

Deferred panel registration: Extensions declare panels during module import,
but they are only realized (added to UI) after the main window is created.
"""

import logging
from typing import Callable, Dict, List, Any
from dataclasses import dataclass

LOGGER = logging.getLogger("extension_registry")


@dataclass
class PanelDefinition:
    """Describes a UI panel provided by an extension."""

    panel_name: str
    widget_class_path: str  # "module.ClassName" for lazy import
    position: str  # "left", "right", "bottom", "floating"
    extension_id: str
    description: str = ""

    def __repr__(self):
        return f"PanelDef({self.panel_name} @ {self.position})"


class ExtensionRegistry:
    """
    Central registry for all extension-provided functionality.

    Panels are registered during module import (via ExtensionAPI calls),
    but are only realized (actually added to UI) after the main window exists.
    """

    def __init__(self):
        self._panels: Dict[str, PanelDefinition] = {}
        self._realize_callbacks: List[Callable] = []

    def register_panel(
        self,
        panel_name: str,
        widget_class_path: str,
        position: str,
        extension_id: str,
        description: str = "",
    ) -> None:
        """
        Register a UI panel definition (called during module import).

        Does NOT add the panel to UI yet — stores the definition.
        Actual panel creation happens in realize_panels() after main window exists.
        """
        if panel_name in self._panels:
            LOGGER.warning(f"Panel '{panel_name}' already registered, skipping")
            return

        panel_def = PanelDefinition(
            panel_name=panel_name,
            widget_class_path=widget_class_path,
            position=position,
            extension_id=extension_id,
            description=description,
        )
        self._panels[panel_name] = panel_def
        LOGGER.info(f"📦 Registered panel: {panel_name} ({extension_id}) @ {position}")

    def get_panels(self) -> Dict[str, PanelDefinition]:
        """Get all registered panel definitions."""
        return dict(self._panels)

    def get_panels_for_position(self, position: str) -> List[PanelDefinition]:
        """Get all panels for a specific position."""
        return [p for p in self._panels.values() if p.position == position]

    def clear(self) -> None:
        """Clear all registered panels (useful for testing / re-init)."""
        self._panels.clear()

    def realize_panels(self, main_window: "QMainWindow") -> Dict[str, Any]:
        """
        Create and add all registered panels to the main window.

        Called once after __import__('ui.main_window').main_window.EfficientManimWindow.__init__() completes.

        Returns:
            {panel_name: widget_instance} for all successfully created panels
        """
        from importlib import import_module
        from PySide6.QtWidgets import QDockWidget
        from PySide6.QtCore import Qt

        created = {}

        for panel_name, panel_def in self._panels.items():
            try:
                module_path, class_name = panel_def.widget_class_path.rsplit(".", 1)
                module = import_module(module_path)
                widget_class = getattr(module, class_name)

                widget = widget_class()
                LOGGER.info(f"✓ Created widget: {class_name}")

                dock = QDockWidget(panel_name, main_window)
                dock.setWidget(widget)
                dock.setObjectName(f"dock_{panel_name.lower().replace(' ', '_')}")

                position_map = {
                    "left": Qt.DockWidgetArea.LeftDockWidgetArea,
                    "right": Qt.DockWidgetArea.RightDockWidgetArea,
                    "bottom": Qt.DockWidgetArea.BottomDockWidgetArea,
                    "floating": None,
                }

                dock_position = position_map.get(
                    panel_def.position, Qt.DockWidgetArea.RightDockWidgetArea
                )

                if dock_position:
                    main_window.addDockWidget(dock_position, dock)
                else:
                    dock.setFloating(True)
                    main_window.addDockWidget(
                        Qt.DockWidgetArea.RightDockWidgetArea, dock
                    )

                created[panel_name] = widget
                LOGGER.info(f"✅ Panel realized: {panel_name} @ {panel_def.position}")

            except Exception as e:
                LOGGER.error(
                    f"❌ Failed to realize panel '{panel_name}': {e}", exc_info=True
                )

        return created


EXTENSION_REGISTRY = ExtensionRegistry()
