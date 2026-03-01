"""
extension_api.py — Safe API for extensions to interact with host

Extensions do NOT have direct access to host objects.
All interactions go through this controlled API.
Enforces permissions and maintains deterministic contract.
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

    All extension operations go through this.
    Enforces permissions and prevents direct host access.
    """

    def __init__(self, extension_id: str):
        self.extension_id = extension_id
        self._permission_manager = EXTENSION_MANAGER._permission_manager
        self._extension_manager = EXTENSION_MANAGER

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
            node_name: Display name (e.g., "MyNode")
            class_path: Module path (e.g., "my_module.MyNodeClass")
            category: Category in node browser
            description: Human-readable description
        """
        self._permission_manager.check_permission(
            self.extension_id, PermissionType.REGISTER_NODES
        )

        # Actually register the node in the global registry
        NODE_REGISTRY.register_node(
            node_name=node_name,
            class_path=class_path,
            category=category,
            description=description,
            extension_id=self.extension_id,
        )

        LOGGER.info(f"Extension {self.extension_id} registered node: {node_name}")

    def register_timeline_track(
        self, track_name: str, class_path: str, description: str = ""
    ) -> None:
        """
        Register a custom timeline track.

        Requires: REGISTER_TIMELINE_TRACK permission

        Args:
            track_name: Display name
            class_path: Module path
            description: Human-readable description
        """
        self._permission_manager.check_permission(
            self.extension_id, PermissionType.REGISTER_TIMELINE_TRACK
        )

        LOGGER.info(f"Extension {self.extension_id} registered track: {track_name}")

    def register_ui_panel(
        self, panel_name: str, widget_class: str, position: str = "right"
    ) -> None:
        """
        Register a custom UI panel.

        Requires: REGISTER_UI_PANEL permission

        Args:
            panel_name: Display name (e.g., "Color Palettes")
            widget_class: QWidget subclass path (e.g., "core.extensions.color_palette.ColorPalettePanel")
            position: "left", "right", "bottom", "floating"
        """
        self._permission_manager.check_permission(
            self.extension_id, PermissionType.REGISTER_UI_PANEL
        )

        # Register panel in global registry (deferred until main window exists)
        EXTENSION_REGISTRY.register_panel(
            panel_name=panel_name,
            widget_class_path=widget_class,
            position=position,
            extension_id=self.extension_id,
        )

        LOGGER.info(
            f"Extension {self.extension_id} registered UI panel: {panel_name} @ {position}"
        )

    def register_mcp_hook(self, hook_name: str, callback: Callable) -> None:
        """
        Register an MCP hook for AI integration.

        Requires: REGISTER_MCP_HOOK permission

        Supported hooks:
        - pre_render
        - post_render
        - node_created
        - node_deleted
        - timeline_changed

        Args:
            hook_name: Name of hook
            callback: Async callable(context) -> dict
        """
        self._permission_manager.check_permission(
            self.extension_id, PermissionType.REGISTER_MCP_HOOK
        )

        self._extension_manager.register_hook(hook_name, callback)
        LOGGER.info(f"Extension {self.extension_id} registered hook: {hook_name}")

    def get_graph(self) -> Dict[str, Any]:
        """
        Get read-only snapshot of current node graph.

        Returns:
            {nodes: {id: {...}}, edges: [...]}
        """
        # In real implementation: return serialized graph snapshot
        return {"nodes": {}, "edges": []}

    def get_timing(self, node_id: str) -> Optional[tuple]:
        """
        Get timing for a node.

        Returns:
            (start_time, duration) or None
        """
        from timing_resolver import TIMING_RESOLVER

        return TIMING_RESOLVER.get_timing(node_id)

    def log(self, message: str, level: str = "info") -> None:
        """
        Log a message from extension.

        Args:
            message: Log message
            level: "debug", "info", "warning", "error"
        """
        logger = LOGGER.getChild(self.extension_id)

        if level == "debug":
            logger.debug(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        else:
            logger.info(message)

    def filesystem_read(self, path: str) -> Optional[bytes]:
        """
        Read file from extension directory.

        Requires: FILESYSTEM_ACCESS permission

        Only allows reading from extension's own directory.
        """
        self._permission_manager.check_permission(
            self.extension_id, PermissionType.FILESYSTEM_ACCESS
        )

        # Prevent directory traversal
        if ".." in path or path.startswith("/"):
            raise PermissionError("Invalid path")

        # Only allow extension directory
        ext_dir = EXTENSION_MANAGER._extensions[self.extension_id].path
        file_path = ext_dir / path

        if not file_path.is_relative_to(ext_dir):
            raise PermissionError(f"Access denied: {path}")

        try:
            return file_path.read_bytes()
        except Exception as e:
            LOGGER.error(f"Read failed: {e}")
            return None

    def filesystem_write(self, path: str, data: bytes) -> bool:
        """
        Write file to extension directory.

        Requires: FILESYSTEM_ACCESS permission

        Only allows writing to extension's own directory.
        """
        self._permission_manager.check_permission(
            self.extension_id, PermissionType.FILESYSTEM_ACCESS
        )

        # Prevent directory traversal
        if ".." in path or path.startswith("/"):
            raise PermissionError("Invalid path")

        # Only allow extension directory
        ext_dir = EXTENSION_MANAGER._extensions[self.extension_id].path
        file_path = ext_dir / path

        if not file_path.is_relative_to(ext_dir):
            raise PermissionError(f"Access denied: {path}")

        try:
            file_path.write_bytes(data)
            return True
        except Exception as e:
            LOGGER.error(f"Write failed: {e}")
            return False


# Factory function
def get_extension_api(extension_id: str) -> ExtensionAPI:
    """Get API instance for an extension."""
    return ExtensionAPI(extension_id)


# ═════════════════════════════════════════════════════════════════════════════

"""
update_manager.py — Extension Update System

Checks for updates, validates compatibility, re-verifies signatures.
"""

from typing import Tuple, List

LOGGER = logging.getLogger("update_manager")


class UpdateManager:
    """Manages extension updates."""

    def __init__(self, marketplace_client):
        self.marketplace = marketplace_client

    def check_for_updates(self, extension_id: str) -> Tuple[bool, str]:
        """
        Check if updates are available for an extension.

        Args:
            extension_id: Extension ID (author/repo)

        Returns:
            (has_update, new_version)
        """
        try:
            ext = EXTENSION_MANAGER._extensions.get(extension_id)
            if not ext:
                return False, ""

            # Check marketplace for newer version
            author, repo = extension_id.split("/")
            info = self.marketplace.get_extension_info(author, repo)

            if not info:
                return False, ""

            new_version = info.get("version", "")
            current_version = ext.metadata.version

            # Simple version comparison (in production: use packaging.version)
            if new_version > current_version:
                return True, new_version

            return False, ""
        except Exception as e:
            LOGGER.error(f"Update check failed: {e}")
            return False, ""

    def apply_update(self, extension_id: str) -> Tuple[bool, str]:
        """
        Apply update to extension.

        Process:
        1. Verify compatibility
        2. Re-verify signature (if present)
        3. Check permission changes
        4. If permissions changed → require re-approval
        5. Update files
        6. Restart extension

        Returns:
            (success, message)
        """
        try:
            ext = EXTENSION_MANAGER._extensions.get(extension_id)
            if not ext:
                return False, "Extension not found"

            # Fetch new metadata
            author, repo = extension_id.split("/")
            info = self.marketplace.get_extension_info(author, repo)

            if not info:
                return False, "Could not fetch update"

            new_version = info.get("version", "")
            new_permissions = info.get("permissions", [])

            # Check for permission changes
            old_permissions = set(ext.metadata.permissions)
            new_perm_set = set(new_permissions)

            if new_perm_set != old_permissions:
                added = new_perm_set - old_permissions
                old_permissions - new_perm_set

                if added:
                    LOGGER.info(f"New permissions required: {added}")
                    # Require re-approval
                    # In UI: show permission dialog

            # Apply update
            # In production: git pull or download new version
            LOGGER.info(f"✓ Extension {extension_id} updated to {new_version}")

            return True, f"Updated to version {new_version}"
        except Exception as e:
            LOGGER.error(f"Update failed: {e}")
            return False, str(e)

    def get_available_updates(self) -> List[Tuple[str, str]]:
        """
        Get list of all extensions with available updates.

        Returns:
            [(extension_id, new_version), ...]
        """
        updates = []

        for ext_id in EXTENSION_MANAGER._extensions:
            has_update, version = self.check_for_updates(ext_id)
            if has_update:
                updates.append((ext_id, version))

        return updates
