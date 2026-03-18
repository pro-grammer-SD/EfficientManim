# 🔌 EfficientManim — Plugin / Extension API

## Overview

EfficientManim has a permission-gated extension platform. Extensions live in `~/.efficientmanim/ext/<author>/<repo_name>/` and are loaded automatically on startup. They can register new node types, UI panels, timeline tracks, and MCP command hooks.

---

## Extension Directory Structure

```
~/.efficientmanim/ext/
└── your_name/
    └── my_extension/
        ├── metadata.json    Required: extension manifest
        ├── lib.py           Required: extension entry point
        └── ...              Any other Python modules or assets
```

---

## metadata.json

Every extension must have a `metadata.json` at its root:

```json
{
  "name": "My Extension",
  "author": "your_name",
  "version": "1.0.0",
  "description": "Adds custom geometry nodes for fractal rendering.",
  "engine_version": ">=2.0.0",
  "permissions": [
    "register_nodes",
    "register_ui_panel"
  ],
  "dependencies": [],
  "entry_file": "lib.py",
  "changelog": "Initial release"
}
```

### Fields

| Field | Required | Description |
|---|---|---|
| `name` | ✅ | Human-readable extension name |
| `author` | ✅ | Your handle or organization |
| `version` | ✅ | Semantic version string |
| `description` | ✅ | One-line description |
| `engine_version` | ✅ | Minimum EfficientManim version (e.g. `>=2.0.0`) |
| `permissions` | ✅ | List of requested permissions (see below) |
| `dependencies` | ❌ | Python packages required (e.g. `["scipy>=1.0"]`) |
| `entry_file` | ❌ | Python file containing `on_load(api)` (default: `lib.py`) |

### Available Permissions

| Permission | What It Allows |
|---|---|
| `register_nodes` | Add new node types to the canvas class browser |
| `register_timeline_track` | Add custom timeline track types |
| `register_ui_panel` | Add new panels to the sidebar |
| `register_mcp_hook` | Register additional MCP commands |
| `filesystem_access` | Read/write files outside the extension directory |
| `network_access` | Make network requests |

Permissions must be approved by the user on first load.

---

## lib.py — Entry Point

The entry file must define an `on_load(api)` function:

```python
"""
My Extension — lib.py
"""

def on_load(api):
    """
    Called once on startup after the extension is approved and enabled.
    
    Args:
        api: ExtensionAPI instance scoped to this extension.
    """
    # Register custom Mobject nodes
    api.register_node(
        node_name="KochSnowflake",
        class_path="my_extension.nodes.KochSnowflake",
        category="🔬 Fractals",
        description="Koch snowflake fractal curve",
    )
    
    # Register a sidebar panel
    api.register_ui_panel(
        panel_name="Fractal Controls",
        class_path="my_extension.panel.FractalPanel",
        dock_area="right",
    )
    
    print("[my_extension] Loaded successfully.")
```

---

## ExtensionAPI Reference

The `api` object passed to `on_load` is an `ExtensionAPI` instance. All operations are gated by the permissions declared in `metadata.json`.

### `api.register_node(node_name, class_path, category, description)`

Register a Manim subclass as a new node type. The class must be importable from the given `class_path`.

```python
api.register_node(
    node_name="GoldenSpiral",
    class_path="my_extension.shapes.GoldenSpiral",
    category="🌀 Custom Shapes",
    description="Logarithmic golden spiral",
)
```

The class must be a valid Manim `Mobject` or `Animation` subclass — EfficientManim will call `inspect.signature` on it to populate the Properties panel automatically.

### `api.register_timeline_track(track_name, class_path, description)`

Add a custom track type to the timeline system.

```python
api.register_timeline_track(
    track_name="Morph Track",
    class_path="my_extension.tracks.MorphTrack",
)
```

### `api.register_ui_panel(panel_name, class_path, dock_area)`

Add a new panel to the sidebar. `class_path` must point to a `QWidget` subclass.

```python
api.register_ui_panel(
    panel_name="Color Harmony",
    class_path="my_extension.panel.ColorHarmonyPanel",
    dock_area="right",
)
```

### `api.register_mcp_hook(command_name, handler_path)`

Register a new command with the MCP agent so Gemini can call it.

```python
api.register_mcp_hook(
    command_name="apply_fractal",
    handler_path="my_extension.mcp_hooks.apply_fractal_handler",
)
```

### `api.log(level, message)`

Write to the application log panel.

```python
api.log("info", "Extension loaded OK")
api.log("warn", "Feature X requires scipy — falling back to numpy")
api.log("error", "Failed to import optional dependency")
```

---

## Full Skeleton Example

```
~/.efficientmanim/ext/
└── yourname/
    └── color_tools/
        ├── metadata.json
        ├── lib.py
        └── panels.py
```

**metadata.json:**
```json
{
  "name": "Color Tools",
  "author": "yourname",
  "version": "0.1.0",
  "description": "Adds a color harmony generator panel.",
  "engine_version": ">=2.0.0",
  "permissions": ["register_ui_panel"],
  "dependencies": [],
  "entry_file": "lib.py"
}
```

**lib.py:**
```python
from .panels import ColorHarmonyPanel

def on_load(api):
    api.register_ui_panel(
        panel_name="🎨 Color Harmony",
        class_path="color_tools.panels.ColorHarmonyPanel",
        dock_area="right",
    )
    api.log("info", "Color Tools extension loaded.")
```

**panels.py:**
```python
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel

class ColorHarmonyPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("🎨 Color Harmony"))
        # ... your UI code here
```

---

## Extension Lifecycle

```
Discovered → Permission review → Approved → Enabled → on_load(api) called
                                                ↓
                                         Extension active
                                                ↓
                              App shutdown / disable → extension deregistered
```

Extensions that error during `on_load` are set to `ERROR` state and skipped on subsequent launches until you fix them.

---

## Accessing the Extension Manager

For advanced use, the global `EXTENSION_MANAGER` singleton is accessible:

```python
from core.extension_manager import EXTENSION_MANAGER

# List all loaded extensions
for ext in EXTENSION_MANAGER.get_all_extensions():
    print(ext.id, ext.state.name)
```

---

## Node Registry

Custom nodes are stored in `core.node_registry.NODE_REGISTRY`. You can introspect it:

```python
from core.node_registry import NODE_REGISTRY

for node in NODE_REGISTRY.get_all_nodes():
    print(node.name, node.category, node.extension_id)
```
