"""
EfficientManim Core Package

Core modules for the application including:
- Extension platform
- UI components
- Data management
- Timing system
- MCP agent
"""

__version__ = "2.0.4"
# Initialize built-in extensions with pre-approved permissions
try:
    from . import bootstrap as bootstrap
except Exception:
    pass  # Bootstrap is optional
