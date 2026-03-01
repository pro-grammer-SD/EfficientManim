"""
app.api — Extension API and registry layer.
"""

from .extension_api import ExtensionAPI
from .extension_manager import EXTENSION_MANAGER, PermissionType
from .extension_registry import EXTENSION_REGISTRY
from .node_registry import NODE_REGISTRY

__all__ = [
    "ExtensionAPI",
    "EXTENSION_MANAGER",
    "PermissionType",
    "EXTENSION_REGISTRY",
    "NODE_REGISTRY",
]
