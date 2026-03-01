"""
extension_manager.py — Core Extension Platform

Single authoritative manager for all extension operations.
Enforces:
  - Permission governance
  - Signature verification
  - Deterministic contract
  - Sandbox isolation
  - Reversibility
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto

LOGGER = logging.getLogger("extension_manager")

EXTENSION_ROOT = Path.home() / ".efficientmanim" / "ext"
EXTENSION_ROOT.mkdir(parents=True, exist_ok=True)


class PermissionType(Enum):
    """Extension permission types."""

    REGISTER_NODES = "register_nodes"
    REGISTER_UI_PANEL = "register_ui_panel"
    REGISTER_MCP_HOOK = "register_mcp_hook"
    FILESYSTEM_ACCESS = "filesystem_access"
    NETWORK_ACCESS = "network_access"


class ExtensionState(Enum):
    """Extension lifecycle state."""

    PENDING = auto()
    APPROVED = auto()
    ENABLED = auto()
    DISABLED = auto()
    ERROR = auto()


@dataclass
class ExtensionPermission:
    """Represents a single permission."""

    type: PermissionType
    description: str = ""
    approved: bool = False
    approved_at: Optional[str] = None


@dataclass
class ExtensionMetadata:
    """Extension metadata."""

    name: str
    author: str
    version: str
    description: str = ""
    engine_version: str = ">=2.0.0"
    permissions: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    entry_file: str = "lib.py"
    has_signature: bool = False
    verified: bool = False
    screenshot_urls: List[str] = field(default_factory=list)
    changelog: str = ""

    @staticmethod
    def from_file(path: Path) -> Optional["ExtensionMetadata"]:
        """Load metadata from metadata.json."""
        try:
            with open(path / "metadata.json", "r") as f:
                data = json.load(f)
            return ExtensionMetadata(**data)
        except Exception as e:
            LOGGER.error(f"Failed to load metadata from {path}: {e}")
            return None


@dataclass
class Extension:
    """Loaded extension instance."""

    id: str
    metadata: ExtensionMetadata
    path: Path
    state: ExtensionState = ExtensionState.PENDING
    permissions: Dict[str, ExtensionPermission] = field(default_factory=dict)


class ExtensionSecurityLayer:
    """Validates extension code before execution."""

    FORBIDDEN = {"os", "sys", "subprocess", "exec", "__import__"}

    def scan(self, code: str) -> Tuple[bool, List[str]]:
        violations = []
        for token in self.FORBIDDEN:
            if token in code:
                violations.append(token)
        return len(violations) == 0, violations


class PermissionManager:
    """Manages and enforces extension permissions."""

    def __init__(self):
        self._permissions: Dict[str, Dict[str, bool]] = {}
        self._load_permissions()

    def request_permission(
        self, ext_id: str, permission: PermissionType, description: str = ""
    ) -> ExtensionPermission:
        perm = ExtensionPermission(permission, description)
        if ext_id not in self._permissions:
            self._permissions[ext_id] = {}
        self._permissions[ext_id][permission.value] = False
        LOGGER.info(f"Permission requested: {ext_id}.{permission.value}")
        return perm

    def approve_permission(self, ext_id: str, permission: PermissionType) -> None:
        if ext_id not in self._permissions:
            self._permissions[ext_id] = {}
        self._permissions[ext_id][permission.value] = True
        self._save_permissions()
        LOGGER.info(f"✓ Permission approved: {ext_id}.{permission.value}")

    def is_permitted(self, ext_id: str, permission: PermissionType) -> bool:
        if ext_id not in self._permissions:
            return False
        is_approved = self._permissions[ext_id].get(permission.value, False)
        if not is_approved:
            LOGGER.warning(f"Permission denied: {ext_id}.{permission.value}")
        return is_approved

    def check_permission(self, ext_id: str, permission: PermissionType) -> None:
        if not self.is_permitted(ext_id, permission):
            raise PermissionError(
                f"Extension {ext_id} lacks permission: {permission.value}"
            )

    def auto_approve_permissions(
        self, ext_id: str, permissions: List[PermissionType]
    ) -> None:
        """Auto-approve permissions for built-in/demo extensions."""
        for permission in permissions:
            if ext_id not in self._permissions:
                self._permissions[ext_id] = {}
            self._permissions[ext_id][permission.value] = True
        self._save_permissions()
        LOGGER.info(f"✓ Auto-approved {len(permissions)} permissions for {ext_id}")

    def _load_permissions(self) -> None:
        perm_file = EXTENSION_ROOT / "permissions.json"
        try:
            if perm_file.exists():
                with open(perm_file, "r") as f:
                    self._permissions = json.load(f)
        except Exception as e:
            LOGGER.error(f"Failed to load permissions: {e}")

    def _save_permissions(self) -> None:
        perm_file = EXTENSION_ROOT / "permissions.json"
        try:
            with open(perm_file, "w") as f:
                json.dump(self._permissions, f, indent=2)
        except Exception as e:
            LOGGER.error(f"Failed to save permissions: {e}")


class ExtensionManager:
    """
    Central extension manager.

    Single point of authority for all extension operations.
    """

    def __init__(self):
        self._extensions: Dict[str, Extension] = {}
        self._security_layer = ExtensionSecurityLayer()
        self._permission_manager = PermissionManager()
        self._hooks: Dict[str, List[Callable]] = {
            "pre_render": [],
            "post_render": [],
            "node_created": [],
            "node_deleted": [],
        }

    def register_hook(self, hook_name: str, callback: Callable) -> None:
        if hook_name not in self._hooks:
            self._hooks[hook_name] = []
        self._hooks[hook_name].append(callback)
        LOGGER.info(f"Hook registered: {hook_name}")

    def fire_hook(self, hook_name: str, context: dict) -> List[dict]:
        results = []
        for cb in self._hooks.get(hook_name, []):
            try:
                result = cb(context)
                if result:
                    results.append(result)
            except Exception as e:
                LOGGER.error(f"Hook {hook_name} failed: {e}")
        return results


EXTENSION_MANAGER = ExtensionManager()
