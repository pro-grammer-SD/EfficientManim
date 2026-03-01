"""
extension_mcp.py — MCP Integration for Extension Management

Allows AI agents to:
  - Discover extensions
  - Install extensions
  - Manage permissions
  - Query extension info
  - Trigger hooks

All operations governed and logged.
"""

import logging
from typing import Dict, Optional

from .extension_manager import EXTENSION_MANAGER, PermissionType
from .github_installer import GitHubInstaller, MarketplaceClient
from .sdk_generator import SDKGenerator

LOGGER = logging.getLogger("extension_mcp")


class ExtensionMCP:
    """MCP commands for extension management."""

    def __init__(self):
        self.marketplace = MarketplaceClient()

    def list_extensions(self) -> Dict:
        """
        List all installed extensions with their state.

        MCP Command: extension_list

        Returns:
        {
            "extensions": [
                {
                    "id": "author/repo",
                    "name": "Extension Name",
                    "version": "1.0.0",
                    "state": "ENABLED",
                    "permissions_approved": ["register_nodes"],
                    "installed_at": "2024-01-15T10:00:00"
                }
            ]
        }
        """
        extensions = []

        for ext_id, ext in EXTENSION_MANAGER._extensions.items():
            extensions.append(ext.to_dict())

        return {"extensions": extensions}

    def search_marketplace(self, query: str) -> Dict:
        """
        Search marketplace for extensions.

        MCP Command: extension_search

        Args:
            query: Search term (name, author, description)

        Returns:
        {
            "results": [
                {
                    "name": "...",
                    "author": "...",
                    "version": "...",
                    "verified": true,
                    "permissions": [...],
                    "install_command": "efficientmanim install author/repo"
                }
            ]
        }
        """
        results = self.marketplace.search_extensions(query)

        if not results:
            return {"results": []}

        return {"results": results}

    def get_extension_info(self, author: str, repo: str) -> Dict:
        """
        Get detailed information about an extension.

        MCP Command: extension_info

        Args:
            author: Extension author (GitHub username)
            repo: Repository name

        Returns:
        {
            "name": "...",
            "author": "...",
            "version": "...",
            "description": "...",
            "permissions": [...],
            "dependencies": [...],
            "verified": true,
            "engine_version": ">=2.0.3",
            "changelog": "...",
            "screenshots": [...]
        }
        """
        ext_id = f"{author}/{repo}"

        # Check if installed locally
        if ext_id in EXTENSION_MANAGER._extensions:
            ext = EXTENSION_MANAGER._extensions[ext_id]
            return {
                "name": ext.metadata.name,
                "author": ext.metadata.author,
                "version": ext.metadata.version,
                "description": ext.metadata.description,
                "permissions": ext.metadata.permissions,
                "dependencies": ext.metadata.dependencies,
                "verified": ext.metadata.verified,
                "engine_version": ext.metadata.engine_version,
                "installed": True,
                "state": ext.state.name,
            }

        # Check marketplace
        info = self.marketplace.get_extension_info(author, repo)
        if info:
            return {**info, "installed": False}

        return {"error": f"Extension not found: {author}/{repo}"}

    def install_extension(self, url: str) -> Dict:
        """
        Install extension from GitHub URL.

        MCP Command: extension_install

        Args:
            url: GitHub URL or shorthand (owner/repo)

        Returns:
        {
            "success": true,
            "extension_id": "owner/repo",
            "message": "...",
            "permissions_requested": ["register_nodes"],
            "next_step": "approve_permissions"
        }
        """
        # Install
        result = GitHubInstaller.install_from_github(url)

        if not result.success:
            return {
                "success": False,
                "message": result.message,
            }

        ext_id = result.extension_id
        ext = EXTENSION_MANAGER._extensions.get(ext_id)

        if not ext:
            return {
                "success": False,
                "message": "Installation succeeded but extension not found in registry",
            }

        return {
            "success": True,
            "extension_id": ext_id,
            "name": ext.metadata.name,
            "version": ext.metadata.version,
            "permissions_requested": ext.metadata.permissions,
            "message": f"Extension {ext_id} installed. Permissions pending approval.",
            "next_step": "approve_permissions",
        }

    def approve_permission(self, extension_id: str, permission: str) -> Dict:
        """
        Approve a permission for an extension.

        MCP Command: extension_approve_permission

        Args:
            extension_id: Extension ID (author/repo)
            permission: Permission name (e.g., "register_nodes")

        Returns:
        {
            "success": true,
            "extension_id": "...",
            "permission": "register_nodes",
            "message": "...",
            "permissions_approved": ["register_nodes"],
            "ready_to_enable": true
        }
        """
        ext = EXTENSION_MANAGER._extensions.get(extension_id)

        if not ext:
            return {"success": False, "error": f"Extension not found: {extension_id}"}

        try:
            perm_type = PermissionType(permission)
        except ValueError:
            return {"success": False, "error": f"Invalid permission: {permission}"}

        EXTENSION_MANAGER._permission_manager.approve_permission(
            extension_id, perm_type
        )

        # Check if all permissions approved
        all_approved = all(
            EXTENSION_MANAGER._permission_manager.is_permitted(
                extension_id, PermissionType(p)
            )
            for p in ext.metadata.permissions
        )

        return {
            "success": True,
            "extension_id": extension_id,
            "permission": permission,
            "message": f"Permission approved: {permission}",
            "permissions_approved": ext.metadata.permissions,
            "ready_to_enable": all_approved,
        }

    def enable_extension(self, extension_id: str) -> Dict:
        """
        Enable an extension (load and initialize).

        MCP Command: extension_enable

        Requires: All permissions approved

        Args:
            extension_id: Extension ID (author/repo)

        Returns:
        {
            "success": true,
            "extension_id": "...",
            "message": "Extension enabled",
            "state": "ENABLED",
            "nodes_registered": [...]
        }
        """
        ext = EXTENSION_MANAGER._extensions.get(extension_id)

        if not ext:
            return {"success": False, "error": f"Extension not found: {extension_id}"}

        success, message = EXTENSION_MANAGER.enable_extension(extension_id)

        return {
            "success": success,
            "extension_id": extension_id,
            "message": message,
            "state": ext.state.name,
        }

    def disable_extension(self, extension_id: str) -> Dict:
        """
        Disable an extension (unload).

        MCP Command: extension_disable

        Args:
            extension_id: Extension ID (author/repo)

        Returns:
        {
            "success": true,
            "extension_id": "...",
            "message": "Extension disabled",
            "state": "DISABLED"
        }
        """
        success = EXTENSION_MANAGER.disable_extension(extension_id)

        ext = EXTENSION_MANAGER._extensions.get(extension_id)
        state = ext.state.name if ext else "UNKNOWN"

        return {
            "success": success,
            "extension_id": extension_id,
            "message": "Extension disabled",
            "state": state,
        }

    def uninstall_extension(self, extension_id: str, confirm: bool = False) -> Dict:
        """
        Uninstall an extension (remove files, disable).

        MCP Command: extension_uninstall

        Destructive operation requires confirm=True.

        Args:
            extension_id: Extension ID (author/repo)
            confirm: Must be True to proceed

        Returns:
        {
            "success": true,
            "extension_id": "...",
            "message": "Extension uninstalled"
        }
        """
        if not confirm:
            return {
                "success": False,
                "error": "Destructive operation. Set confirm=True to proceed.",
            }

        ext = EXTENSION_MANAGER._extensions.get(extension_id)

        if not ext:
            return {"success": False, "error": f"Extension not found: {extension_id}"}

        try:
            # Disable first
            EXTENSION_MANAGER.disable_extension(extension_id)

            # Delete files
            import shutil

            shutil.rmtree(ext.path)

            # Remove from registry
            del EXTENSION_MANAGER._extensions[extension_id]

            LOGGER.info(f"Extension uninstalled: {extension_id}")

            return {
                "success": True,
                "extension_id": extension_id,
                "message": "Extension uninstalled",
            }
        except Exception as e:
            LOGGER.error(f"Uninstall failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def check_updates(self, extension_id: Optional[str] = None) -> Dict:
        """
        Check for available updates.

        MCP Command: extension_check_updates

        Args:
            extension_id: Specific extension, or None for all

        Returns:
        {
            "updates": [
                {
                    "extension_id": "author/repo",
                    "current_version": "1.0.0",
                    "new_version": "1.1.0",
                    "compatible": true
                }
            ]
        }
        """
        from github_installer import MarketplaceClient
        from extension_api import UpdateManager

        marketplace = MarketplaceClient()
        update_mgr = UpdateManager(marketplace)

        available = update_mgr.get_available_updates()

        updates = [
            {
                "extension_id": ext_id,
                "new_version": new_version,
                "current_version": EXTENSION_MANAGER._extensions[
                    ext_id
                ].metadata.version,
                "compatible": True,  # Could check engine version
            }
            for ext_id, new_version in available
        ]

        return {"updates": updates}

    def create_extension_template(self, name: str, author: str) -> Dict:
        """
        Create extension template project.

        MCP Command: extension_create_template

        Args:
            name: Extension name (e.g., "MyExtension")
            author: Author name (e.g., "github_username")

        Returns:
        {
            "success": true,
            "extension_name": "MyExtension",
            "path": "/path/to/MyExtension",
            "message": "Template created, ready to customize"
        }
        """
        try:
            path = SDKGenerator.create_extension(name, author)

            return {
                "success": True,
                "extension_name": name,
                "path": str(path),
                "message": f"Extension template created at {path}",
                "next_steps": [
                    "Customize lib.py with your nodes",
                    "Add dependencies to requirements.txt",
                    "Test locally",
                    "Push to GitHub",
                    "Submit to marketplace",
                ],
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def trigger_hook(self, hook_name: str, context: Dict = None) -> Dict:
        """
        Trigger hook for all registered extensions.

        MCP Command: extension_trigger_hook

        Supported hooks:
        - pre_render
        - post_render
        - node_created
        - node_deleted
        - timeline_changed

        Args:
            hook_name: Hook name
            context: Hook context data

        Returns:
        {
            "hook": "pre_render",
            "triggered_extensions": ["author/repo", ...],
            "results": [{extension_id: result}, ...]
        }
        """
        if context is None:
            context = {}

        results = EXTENSION_MANAGER.trigger_hook(hook_name, context)

        return {
            "hook": hook_name,
            "triggered_count": len(results),
            "results": results,
        }


# Integration into main.py MCP system
def setup_extension_mcp(mcp_server):
    """Setup extension MCP commands in existing MCP server."""

    ext_mcp = ExtensionMCP()

    @mcp_server._register("extension_list")
    def _(payload: dict):
        result = ext_mcp.list_extensions()
        return {"success": True, "data": result}

    @mcp_server._register("extension_search")
    def _(payload: dict):
        query = payload.get("query", "")
        result = ext_mcp.search_marketplace(query)
        return {"success": True, "data": result}

    @mcp_server._register("extension_info")
    def _(payload: dict):
        author = payload.get("author", "")
        repo = payload.get("repo", "")
        result = ext_mcp.get_extension_info(author, repo)
        return {"success": True, "data": result}

    @mcp_server._register("extension_install")
    def _(payload: dict):
        url = payload.get("url", "")
        result = ext_mcp.install_extension(url)
        return {"success": result.get("success", False), "data": result}

    @mcp_server._register("extension_approve_permission")
    def _(payload: dict):
        ext_id = payload.get("extension_id", "")
        permission = payload.get("permission", "")
        result = ext_mcp.approve_permission(ext_id, permission)
        return {"success": result.get("success", False), "data": result}

    @mcp_server._register("extension_enable")
    def _(payload: dict):
        ext_id = payload.get("extension_id", "")
        result = ext_mcp.enable_extension(ext_id)
        return {"success": result.get("success", False), "data": result}

    @mcp_server._register("extension_disable")
    def _(payload: dict):
        ext_id = payload.get("extension_id", "")
        result = ext_mcp.disable_extension(ext_id)
        return {"success": result.get("success", False), "data": result}

    @mcp_server._register("extension_uninstall")
    def _(payload: dict):
        ext_id = payload.get("extension_id", "")
        confirm = payload.get("confirm", False)
        result = ext_mcp.uninstall_extension(ext_id, confirm)
        return {"success": result.get("success", False), "data": result}

    @mcp_server._register("extension_check_updates")
    def _(payload: dict):
        ext_id = payload.get("extension_id")
        result = ext_mcp.check_updates(ext_id)
        return {"success": True, "data": result}

    @mcp_server._register("extension_create_template")
    def _(payload: dict):
        name = payload.get("name", "")
        author = payload.get("author", "")
        result = ext_mcp.create_extension_template(name, author)
        return {"success": result.get("success", False), "data": result}

    @mcp_server._register("extension_trigger_hook")
    def _(payload: dict):
        hook_name = payload.get("hook_name", "")
        context = payload.get("context", {})
        result = ext_mcp.trigger_hook(hook_name, context)
        return {"success": True, "data": result}

    LOGGER.info("✓ Extension MCP commands registered")
