"""
extension_api.py — Safe API for extensions to interact with the host

Extensions do NOT have direct access to host objects.
All interactions go through this controlled API.
Enforces permissions and maintains a deterministic contract.
"""

import logging
from typing import Callable, Optional, Dict, Any

from .extension_manager import EXTENSION_MANAGER, PermissionType
from .extension_registry import EXTENSION_REGISTRY
from .node_registry import NODE_REGISTRY

LOGGER = logging.getLogger("extension_api")


class ExtensionAPI:
    """
    Safe API for extensions.

    All extension operations go through this class.
    Enforces permissions and prevents direct host access.
    """

    def __init__(self, extension_id: str):
        self.extension_id = extension_id
        self._permission_manager = EXTENSION_MANAGER._permission_manager
        self._extension_manager = EXTENSION_MANAGER

    # ── Node registration ─────────────────────────────────────────────────────

    def register_node(
        self,
        node_name: str,
        class_path: str,
        category: str = "Custom",
        description: str = "",
    ) -> None:
        """
        Register a deterministic node class.

        Requires: REGISTER_NODES permission

        Args:
            node_name:   Display name (e.g. "Integral Symbol")
            class_path:  Module path  (e.g. "app.extensions.math_symbols.IntegralSymbol")
            category:    Category in node browser
            description: Human-readable description
        """
        self._permission_manager.check_permission(
            self.extension_id, PermissionType.REGISTER_NODES
        )

        NODE_REGISTRY.register_node(
            node_name=node_name,
            class_path=class_path,
            category=category,
            description=description,
            extension_id=self.extension_id,
        )

        LOGGER.info(f"Extension {self.extension_id} registered node: {node_name}")

    # ── UI panel registration ─────────────────────────────────────────────────

    def register_ui_panel(
        self, panel_name: str, widget_class: str, position: str = "right"
    ) -> None:
        """
        Register a custom UI panel.

        Requires: REGISTER_UI_PANEL permission

        Args:
            panel_name:   Display name (e.g. "Color Palettes")
            widget_class: QWidget subclass path
                          (e.g. "app.extensions.color_palette.ColorPalettePanel")
            position:     "left", "right", "bottom", or "floating"
        """
        self._permission_manager.check_permission(
            self.extension_id, PermissionType.REGISTER_UI_PANEL
        )

        EXTENSION_REGISTRY.register_panel(
            panel_name=panel_name,
            widget_class_path=widget_class,
            position=position,
            extension_id=self.extension_id,
        )

        LOGGER.info(
            f"Extension {self.extension_id} registered UI panel: {panel_name} @ {position}"
        )

    # ── MCP hook registration ─────────────────────────────────────────────────

    def register_mcp_hook(self, hook_name: str, callback: Callable) -> None:
        """
        Register an MCP hook for AI integration.

        Requires: REGISTER_MCP_HOOK permission

        Supported hooks: pre_render, post_render, node_created, node_deleted
        """
        self._permission_manager.check_permission(
            self.extension_id, PermissionType.REGISTER_MCP_HOOK
        )

        self._extension_manager.register_hook(hook_name, callback)
        LOGGER.info(f"Extension {self.extension_id} registered hook: {hook_name}")

    # ── Read-only graph access ────────────────────────────────────────────────

    def get_graph(self) -> Dict[str, Any]:
        """Get read-only snapshot of current node graph."""
        return {"nodes": {}, "edges": []}

    # ── Logging ───────────────────────────────────────────────────────────────

    def log(self, message: str, level: str = "info") -> None:
        """Log a message from an extension."""
        logger = LOGGER.getChild(self.extension_id)
        getattr(
            logger, level if level in ("debug", "info", "warning", "error") else "info"
        )(message)

    # ── Filesystem (sandboxed) ────────────────────────────────────────────────

    def filesystem_read(self, path: str) -> Optional[bytes]:
        """
        Read file from extension directory (sandboxed).

        Requires: FILESYSTEM_ACCESS permission
        """
        self._permission_manager.check_permission(
            self.extension_id, PermissionType.FILESYSTEM_ACCESS
        )
        if ".." in path or path.startswith("/"):
            raise PermissionError("Invalid path")
        try:
            from pathlib import Path

            ext_dir = Path.home() / ".efficientmanim" / "ext" / self.extension_id
            file_path = ext_dir / path
            if not file_path.is_relative_to(ext_dir):
                raise PermissionError(f"Access denied: {path}")
            return file_path.read_bytes()
        except Exception as e:
            LOGGER.error(f"Read failed: {e}")
            return None

    def filesystem_write(self, path: str, data: bytes) -> bool:
        """
        Write file to extension directory (sandboxed).

        Requires: FILESYSTEM_ACCESS permission
        """
        self._permission_manager.check_permission(
            self.extension_id, PermissionType.FILESYSTEM_ACCESS
        )
        if ".." in path or path.startswith("/"):
            raise PermissionError("Invalid path")
        try:
            from pathlib import Path

            ext_dir = Path.home() / ".efficientmanim" / "ext" / self.extension_id
            file_path = ext_dir / path
            if not file_path.is_relative_to(ext_dir):
                raise PermissionError(f"Access denied: {path}")
            file_path.write_bytes(data)
            return True
        except Exception as e:
            LOGGER.error(f"Write failed: {e}")
            return False


def get_extension_api(extension_id: str) -> ExtensionAPI:
    """Factory: get API instance for an extension."""
    return ExtensionAPI(extension_id)
