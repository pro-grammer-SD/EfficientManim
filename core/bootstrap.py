"""
bootstrap.py — Initialize built-in demo extensions with pre-approved permissions

Handles registration and permission setup for extensions bundled with EfficientManim.
"""

import logging
from .extension_manager import EXTENSION_MANAGER, PermissionType

LOGGER = logging.getLogger("bootstrap")


def bootstrap_builtin_extensions():
    """
    Initialize built-in demo extensions.

    Pre-approves permissions for extensions that ship with EfficientManim.
    This is safe because these extensions are part of the codebase and verified.
    """

    # Built-in extension definitions
    # Each tuple: (extension_id, required_permissions)
    builtin_extensions = [
        ("builtin/math-symbols", [PermissionType.REGISTER_NODES]),
        ("builtin/color-palette", [PermissionType.REGISTER_UI_PANEL]),
        ("builtin/timeline-templates", [PermissionType.REGISTER_TIMELINE_TRACK]),
    ]

    for ext_id, permissions in builtin_extensions:
        try:
            # Pre-approve permissions for built-in extensions
            EXTENSION_MANAGER._permission_manager.auto_approve_permissions(
                ext_id, permissions
            )
            LOGGER.info(f"✓ Initialized: {ext_id}")
        except Exception as e:
            LOGGER.error(f"Failed to bootstrap {ext_id}: {e}")


# Auto-bootstrap on import
try:
    bootstrap_builtin_extensions()
except Exception as e:
    LOGGER.error(f"Bootstrap failed: {e}")
