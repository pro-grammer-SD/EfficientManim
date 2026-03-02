# EfficientManim Extension Platform

**Version**: 2.0.4 
**Status**: Production-Ready  
**Document Version**: 1.0.0

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Extension Lifecycle](#extension-lifecycle)
3. [Permission System](#permission-system)
4. [How to Build an Extension](#how-to-build-an-extension)
5. [Security Model](#security-model)
6. [Example Extension](#example-extension)
7. [API Reference](#api-reference)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

### System Components

The EfficientManim extension platform consists of five core components:

#### 1. **ExtensionManager** (`extension_manager.py`)
Central authority for all extension operations. Responsibilities:
- Extension discovery and loading
- Permission governance enforcement
- Extension lifecycle management (enable/disable)
- Hook registration and triggering
- Security validation before loading

#### 2. **ExtensionAPI** (`extension_api.py`)
Safe, sandboxed interface for extensions to interact with the host application. Provides:
- Node registration
- Timeline track registration
- UI panel registration
- MCP hook registration
- Graph introspection (read-only)
- Timing queries

Extensions **never** have direct access to application internals. All interactions are mediated through this API.

#### 3. **PermissionManager** (`extension_manager.py`)
Enforces permission-based access control. Maintains:
- Permission grants per extension
- Permission approval/denial state
- Persistence of permissions to disk

#### 4. **ExtensionSecurityLayer** (`extension_manager.py`)
Validates extension integrity before activation. Checks:
- Metadata completeness and validity
- Engine version compatibility
- Cryptographic signatures (if present)
- Dependency safety (forbidden packages)

#### 5. **ExtensionPermissions** (Enum)
Defines all possible permissions:
```python
class PermissionType(Enum):
    REGISTER_NODES = "register_nodes"              # Define custom node types
    REGISTER_TIMELINE_TRACK = "register_timeline_track"  # Timeline track templates
    REGISTER_UI_PANEL = "register_ui_panel"        # UI panel widgets
    REGISTER_MCP_HOOK = "register_mcp_hook"        # AI integration hooks
    FILESYSTEM_ACCESS = "filesystem_access"        # File I/O permissions
    NETWORK_ACCESS = "network_access"              # Network access permissions
```

---

## Extension Lifecycle

Every extension follows a defined lifecycle:

```
DISCOVERY → VALIDATION → PERMISSION APPROVAL → ENABLED → DISABLED
                            ↓
                      (Permission denied)
                            ↓
                         ERROR
```

### Stages

1. **Discovery**
   - ExtensionManager scans extension directories
   - Metadata.json parsed and validated
   - Extension object created in PENDING state

2. **Validation**
   - Security layer validates extension integrity
   - Metadata checked for completeness
   - Dependencies verified as safe
   - Signature verified (if present)

3. **Permission Approval**
   - Required permissions declared in metadata.json
   - ExtensionManager checks permission grants
   - If any permission denied → ERROR state
   - If all approved → proceeds to enable

4. **Loading & Execution**
   - Extension module dynamically imported
   - setup() function called with ExtensionAPI instance
   - Extension registers its components (nodes, panels, etc.)
   - Extension transitioned to ENABLED state

5. **Disable / Unload**
   - Extension module unloaded
   - Hooks deregistered
   - State changed to DISABLED

---

## Permission System

The permission system is the core security mechanism. Extensions must explicitly declare required permissions, and all permissions must be approved before the extension can load.

### Why Permissions Exist

**Security Boundary**: Permissions prevent malicious code from:
- Registering unauthorized node types with sensitive operations
- Accessing the filesystem without user knowledge
- Making unauthorized network requests
- Hooking into AI operations inappropriately
- Accessing application internals

**Transparency**: Users and administrators can see exactly what each extension is doing:
- What types it registers
- What it hooks into
- What resources it accesses

**Auditability**: Permission changes are logged and persisted.

### Permission Workflow

#### Step 1: Declare Permissions in metadata.json
```json
{
  "name": "My Extension",
  "author": "author_name",
  "version": "1.0.0",
  "permissions": [
    "register_nodes",
    "register_ui_panel"
  ]
}
```

#### Step 2: ExtensionManager Validates
```python
ext = Extension(..., metadata=metadata)
enable_extension(ext_id)  # Checks permissions
```

#### Step 3: PermissionManager Enforces
Before each operation, ExtensionAPI calls:
```python
self._permission_manager.check_permission(ext_id, PermissionType.REGISTER_NODES)
# Raises PermissionError if not approved
```

#### Step 4: User/Admin Approval
User explicitly approves permissions in UI (future version), or:
- Built-in extensions are pre-approved
- Third-party extensions require user approval

### Permission Approval States

| State | Meaning | Action |
|-------|---------|--------|
| **Approved** | Extension can use this permission | Extension loads and functions |
| **Denied** | Extension blocked from using permission | Extension fails to load; PermissionError raised |
| **Not Requested** | Extension did not declare this permission | Permission check passes (not needed) |

---

## How to Build an Extension

### Step 1: Create Extension Directory Structure

```
~/.efficientmanim/ext/
  author_name/
    extension_name/
      metadata.json         ← Extension metadata
      lib.py               ← Entry point (must contain setup())
      requirements.txt     ← Optional: Python dependencies
      README.md            ← Optional: Documentation
```

### Step 2: Create metadata.json

Declare your extension and required permissions:

```json
{
  "name": "My Color Picker",
  "author": "your_name",
  "version": "1.0.0",
  "description": "A custom color picker panel for quick color selection",
  "engine_version": ">=2.0.0",
  "permissions": [
    "register_ui_panel"
  ],
  "dependencies": [],
  "entry_file": "lib.py",
  "has_signature": false,
  "verified": false,
  "changelog": "Initial release"
}
```

### Step 3: Create lib.py with setup() Function

The entry point must define a `setup(api: ExtensionAPI)` function:

```python
from core.extension_api import ExtensionAPI

def setup(api: ExtensionAPI) -> bool:
    """
    Initialize extension.
    
    Called by ExtensionManager when extension is loaded.
    Must complete without raising exceptions.
    
    Args:
        api: ExtensionAPI instance for interacting with host
    
    Returns:
        bool: True if setup succeeded, False otherwise
    """
    
    # Register your components
    api.register_ui_panel(
        panel_name="My Color Picker",
        widget_class="my_extension.ColorPickerPanel",
        position="right"
    )
    
    return True


# Define your custom classes
class ColorPickerPanel:
    """A custom color picker panel."""
    
    def __init__(self):
        self.selected_color = "#FFFFFF"
    
    def get_color(self):
        return self.selected_color
    
    def set_color(self, hex_color: str):
        self.selected_color = hex_color
```

### Step 4: Declare Minimum Permissions

Only request permissions you actually use. Examples:

**For registering nodes:**
```json
"permissions": ["register_nodes"]
```

**For custom UI panels:**
```json
"permissions": ["register_ui_panel"]
```

**For timeline tracks:**
```json
"permissions": ["register_timeline_track"]
```

**For multiple features:**
```json
"permissions": [
  "register_nodes",
  "register_ui_panel",
  "register_timeline_track"
]
```

---

## Security Model

### Design Principles

1. **Principle of Least Privilege**
   - Extensions request only permissions they need
   - Permissions are explicit and minimal
   - Defaults are restrictive (deny by default)

2. **Sandboxing via API**
   - Extensions access only ExtensionAPI surface
   - No direct access to internal objects
   - All interactions mediated and logged

3. **Signature Verification**
   - Extensions can be cryptographically signed
   - Signature verified before loading
   - Prevents tampering and unauthorized modification

4. **Deterministic Execution**
   - Extensions register static components (nodes, panels)
   - No dynamic code generation
   - No shell execution or subprocess spawning
   - No arbitrary file I/O (controlled via FILESYSTEM_ACCESS permission)

### Attack Scenarios Mitigated

| Attack | Mitigation | Mechanism |
|--------|-----------|-----------|
| Unauthorized node registration | REGISTER_NODES permission | Must declare in metadata.json |
| Malicious keybindings | Not directly supported | No permission for keybinding registration |
| Network exfiltration | NETWORK_ACCESS permission | Must request explicitly |
| File system tampering | FILESYSTEM_ACCESS permission | Controlled & logged |
| Code injection via hooks | Hook validation | Hooks execute in isolated context |
| Privilege escalation | No capability transfer | API is least-privileged |

### User Trust Model

```
User Approval Flow:

1. Extension discovered in ~/.efficientmanim/ext/
2. Metadata validated by SecurityLayer
3. Required permissions displayed to user
4. User approves/denies specific permissions
5. If approved: Extension loads
   If denied: Extension fails to load (gracefully)
```

---

## Example Extension

### Minimal Extension: Echo Node

A simple extension that registers a custom node with minimal code.

**Directory Structure:**
```
~/.efficientmanim/ext/
  examples/
    echo-node/
      metadata.json
      lib.py
```

**metadata.json:**
```json
{
  "name": "Echo Node",
  "author": "EfficientManim",
  "version": "1.0.0",
  "description": "A simple echo node for testing",
  "engine_version": ">=2.0.0",
  "permissions": ["register_nodes"],
  "entry_file": "lib.py"
}
```

**lib.py:**
```python
"""Echo Node Extension - Minimal example."""

from core.extension_api import ExtensionAPI


def setup(api: ExtensionAPI) -> bool:
    """Register the echo node."""
    api.register_node(
        node_name="Echo",
        class_path="lib.EchoNode",
        category="Examples",
        description="Echoes input text back")
    return True


class EchoNode:
    """A simple echo node for demonstration."""
    
    def __init__(self, text: str = ""):
        self.text = text
    
    def echo(self) -> str:
        """Return the echoed text."""
        return f"Echo: {self.text}"
```

---

## API Reference

### ExtensionAPI Methods

#### `register_node(node_name, class_path, category, description)`
Register a custom node type.

**Parameters:**
- `node_name` (str): Display name of the node
- `class_path` (str): Module path to node class (e.g., "my_ext.MyNode")
- `category` (str): Category in node browser (default: "Custom")
- `description` (str): Human-readable description

**Requires:** REGISTER_NODES permission

**Example:**
```python
api.register_node(
    node_name="My Node",
    class_path="ext.CustomNode",
    category="Custom",
    description="My custom node"
)
```

---

#### `register_ui_panel(panel_name, widget_class, position)`
Register a custom UI panel.

**Parameters:**
- `panel_name` (str): Display name of the panel
- `widget_class` (str): QWidget subclass path
- `position` (str): Position - "left", "right", "bottom", or "floating"

**Requires:** REGISTER_UI_PANEL permission

**Example:**
```python
api.register_ui_panel(
    panel_name="My Panel",
    widget_class="ext.MyPanel",
    position="right"
)
```

---

#### `register_timeline_track(track_name, class_path, description)`
Register a timeline track template.

**Parameters:**
- `track_name` (str): Display name
- `class_path` (str): Class path to track implementation
- `description` (str): Description

**Requires:** REGISTER_TIMELINE_TRACK permission

**Example:**
```python
api.register_timeline_track(
    track_name="Fade Effect",
    class_path="ext.FadeTrack",
    description="Fade in/out animation"
)
```

---

#### `register_mcp_hook(hook_name, callback)`
Register a hook for AI integration.

**Parameters:**
- `hook_name` (str): Hook name (e.g., "pre_render", "node_created")
- `callback` (Callable): Async function(context) → dict

**Requires:** REGISTER_MCP_HOOK permission

**Example:**
```python
async def on_node_created(context):
    return {"processed": True}

api.register_mcp_hook("node_created", on_node_created)
```

---

#### `get_graph()`
Get read-only snapshot of current node graph.

**Returns:** Dict with "nodes" and "edges" keys

**Example:**
```python
graph = api.get_graph()
nodes = graph["nodes"]  # {node_id: {...}}
edges = graph["edges"]  # List of connections
```

---

#### `get_timing(node_id)`
Get timing information for a node.

**Parameters:**
- `node_id` (str): Node identifier

**Returns:** Tuple of (start_time, duration) or None

**Example:**
```python
timing = api.get_timing("node_123")
if timing:
    start, duration = timing
```

---

## Best Practices

### 1. Declare Minimum Permissions
Only declare permissions your extension actually uses:
```json
// GOOD: Only what's needed
"permissions": ["register_nodes"]

// BAD: Over-privileging
"permissions": [
  "register_nodes",
  "register_ui_panel", 
  "register_timeline_track",
  "network_access"
]
```

### 2. Validate User Input
Always validate parameters in setup():
```python
def setup(api: ExtensionAPI) -> bool:
    # Validate before registration
    try:
        api.register_node(...)
        return True
    except Exception as e:
        return False  # Fail gracefully
```

### 3. Use Descriptive Names
Help users understand what your extension does:
```python
# GOOD
node_name="Gaussian Blur"
description="Apply Gaussian blur filter to images"

# BAD
node_name="Node1"
description="Blur"
```

### 4. Follow Naming Conventions
- Use PascalCase for class names: `MyNode`, `ColorPicker`
- Use snake_case for module names: `my_extension`, `color_picker`
- Use descriptive, unique names: `GaussianBlur` not `Node`

### 5. Include Documentation
Always include a README.md explaining:
- What your extension does
- How to use it
- Required permissions and why
- Configuration options

### 6. Test with Explicit Permissions
Always verify your extension loads with only declared permissions:
```bash
# Should work
python main.py  # Extension loads with correct permissions

# Would fail (if permissions were missing)
# → PermissionError: Extension lacks permission
```

### 7. Handle Permission Errors Gracefully
Your setup() function should return False on permission errors:
```python
def setup(api: ExtensionAPI) -> bool:
    try:
        # This may raise PermissionError
        api.register_ui_panel(...)
        return True
    except PermissionError as e:
        # Log and fail gracefully
        return False
```

---

## Troubleshooting

### "Permission Denied" Error

**Error:**
```
PermissionError: Extension color-palette lacks permission: register_ui_panel
```

**Cause:** Permission not declared in metadata.json or not approved

**Solution:**
1. Check metadata.json contains permission:
   ```json
   "permissions": ["register_ui_panel"]
   ```
2. Restart application to load metadata
3. Verify permission is approved in permissions.json

---

### Extension Not Loading

**Error:**
```
Failed to enable extension: No such file or directory
```

**Cause:** Directory structure incorrect or entry file missing

**Solution:**
1. Verify directory structure:
   ```
   ~/.efficientmanim/ext/author/extension_name/
     ├── metadata.json
     ├── lib.py
     └── requirements.txt (optional)
   ```
2. Verify lib.py contains setup() function
3. Check metadata.json points to correct entry_file

---

### setup() Function Failing

**Error:**
```
Extension failed: [error message]
```

**Cause:** Exception in setup() function

**Solution:**
1. Add try/catch in setup():
   ```python
   def setup(api: ExtensionAPI) -> bool:
       try:
           api.register_node(...)
           return True
       except Exception as e:
           # Log error
           return False
   ```
2. Return False on error instead of raising
3. Check class_path values are correct

---

### Circular Import Error

**Error:**
```
ImportError: circular import detected
```

**Cause:** Extension imports from core in setup()

**Solution:**
Delay imports until setup() is called:
```python
# WRONG: Imports at module level
from core.extension_manager import EXTENSION_MANAGER

def setup(api):
    ...

# CORRECT: Import inside function
def setup(api):
    from core.extension_manager import EXTENSION_MANAGER
    ...
```

---

## Summary

The EfficientManim extension platform provides:

✓ **Safety**: Permission-based security gates prevent malicious code  
✓ **Transparency**: All required permissions explicitly declared  
✓ **Simplicity**: Minimal API surface, easy to learn  
✓ **Flexibility**: Support for nodes, panels, tracks, and hooks  
✓ **Auditability**: All extension operations logged and persisted  

Extensions are first-class citizens in EfficientManim, with secure access to application internals through a controlled, well-documented interface.

---

**For questions or contributions, see [CONTRIBUTING.md](../docs/CONTRIBUTING.md)**
