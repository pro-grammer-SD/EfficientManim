# EXTENSION PLATFORM INTEGRATION GUIDE

**Status**: COMPLETE PLATFORM, READY FOR INTEGRATION  
**Files**: 5 core modules + extension_mcp.py  
**Lines**: 1900+ platform code  

---

## INTEGRATION CHECKLIST

### Pre-Integration (Understanding)
- [ ] Read EXTENSION_PLATFORM_SPEC.md completely
- [ ] Review all 5 platform modules (understand responsibilities)
- [ ] Review MCP integration in extension_mcp.py
- [ ] Review marketplace protocol format
- [ ] Review security model and sandbox rules

### Step 1: Add Platform Modules to main.py

Add imports:
```python
# Extension platform (v1.0)
try:
    from extension_manager import EXTENSION_MANAGER, PermissionType
    from extension_api import get_extension_api
    from github_installer import GitHubInstaller
    from sdk_generator import SDKGenerator
    from extension_mcp import setup_extension_mcp
    EXTENSION_PLATFORM_AVAILABLE = True
except ImportError:
    EXTENSION_PLATFORM_AVAILABLE = False
    print("WARNING: Extension platform not available")
```

### Step 2: Initialize Platform in __init__

In `EfficientManimWindow.__init__()`, after existing initializations:

```python
# Initialize extension platform
if EXTENSION_PLATFORM_AVAILABLE:
    # Load installed extensions
    EXTENSION_MANAGER.discover_installed()
    LOGGER.info(f"✓ Extension platform initialized, {len(EXTENSION_MANAGER._extensions)} extensions installed")
```

### Step 3: Setup MCP Integration

In the MCP server setup, add:

```python
if EXTENSION_PLATFORM_AVAILABLE:
    setup_extension_mcp(mcp_server)
    LOGGER.info("✓ Extension MCP commands registered")
```

### Step 4: Add Extension Management UI

Create menu item or panel:

```python
# In setup_menu():
if EXTENSION_PLATFORM_AVAILABLE:
    tools_menu = bar.addMenu("Tools")
    
    extensions_action = QAction("Manage Extensions", self)
    extensions_action.triggered.connect(self.show_extensions_dialog)
    tools_menu.addAction(extensions_action)
    
    sdk_action = QAction("Create Extension", self)
    sdk_action.triggered.connect(self.show_create_extension_dialog)
    tools_menu.addAction(sdk_action)
```

### Step 5: Add Extension Dialogs

Implement dialogs:

```python
def show_extensions_dialog(self):
    """Show installed extensions management UI."""
    from PySide6.QtWidgets import QDialog, QVBoxLayout, QTableWidget, QPushButton
    
    dialog = QDialog(self)
    dialog.setWindowTitle("Extensions")
    dialog.setGeometry(100, 100, 800, 600)
    
    layout = QVBoxLayout()
    
    # List installed extensions
    table = QTableWidget()
    table.setColumnCount(4)
    table.setHorizontalHeaderLabels(["Name", "Version", "State", "Actions"])
    
    for ext_id, ext in EXTENSION_MANAGER._extensions.items():
        row = table.rowCount()
        table.insertRow(row)
        
        table.setItem(row, 0, QTableWidgetItem(ext.metadata.name))
        table.setItem(row, 1, QTableWidgetItem(ext.metadata.version))
        table.setItem(row, 2, QTableWidgetItem(ext.state.name))
        
        # Action buttons
        action_widget = QWidget()
        action_layout = QHBoxLayout(action_widget)
        
        if ext.state == ExtensionState.ENABLED:
            disable_btn = QPushButton("Disable")
            disable_btn.clicked.connect(lambda checked, eid=ext_id: self.disable_extension(eid))
            action_layout.addWidget(disable_btn)
        else:
            enable_btn = QPushButton("Enable")
            enable_btn.clicked.connect(lambda checked, eid=ext_id: self.enable_extension(eid))
            action_layout.addWidget(enable_btn)
        
        table.setCellWidget(row, 3, action_widget)
    
    layout.addWidget(table)
    
    # Install button
    install_btn = QPushButton("Install from GitHub")
    install_btn.clicked.connect(self.show_install_extension_dialog)
    layout.addWidget(install_btn)
    
    dialog.setLayout(layout)
    dialog.exec()

def show_create_extension_dialog(self):
    """Show create extension template dialog."""
    from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton
    
    dialog = QDialog(self)
    dialog.setWindowTitle("Create Extension")
    
    layout = QVBoxLayout()
    
    # Extension name
    layout.addWidget(QLabel("Extension Name:"))
    name_input = QLineEdit()
    name_input.setPlaceholderText("MyExtension")
    layout.addWidget(name_input)
    
    # Author
    layout.addWidget(QLabel("Author (GitHub username):"))
    author_input = QLineEdit()
    author_input.setPlaceholderText("github_username")
    layout.addWidget(author_input)
    
    # Create button
    create_btn = QPushButton("Create Template")
    create_btn.clicked.connect(
        lambda: self.create_extension_template(
            name_input.text(),
            author_input.text()
        )
    )
    layout.addWidget(create_btn)
    
    dialog.setLayout(layout)
    dialog.exec()

def show_install_extension_dialog(self):
    """Show install extension dialog."""
    from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton
    from PySide6.QtCore import Qt
    
    dialog = QDialog(self)
    dialog.setWindowTitle("Install Extension")
    
    layout = QVBoxLayout()
    
    layout.addWidget(QLabel("GitHub URL or shorthand (owner/repo):"))
    url_input = QLineEdit()
    url_input.setPlaceholderText("octocat/SuperCameraPack")
    layout.addWidget(url_input)
    
    install_btn = QPushButton("Install")
    install_btn.clicked.connect(
        lambda: self.install_extension_from_url(url_input.text())
    )
    layout.addWidget(install_btn)
    
    dialog.setLayout(layout)
    dialog.exec()

def install_extension_from_url(self, url: str):
    """Install extension from GitHub URL."""
    if not url:
        return
    
    result = GitHubInstaller.install_from_github(url)
    
    if not result.success:
        self.show_error_dialog(f"Installation failed: {result.message}")
        return
    
    ext_id = result.extension_id
    ext = EXTENSION_MANAGER._extensions.get(ext_id)
    
    if not ext:
        self.show_error_dialog("Installation succeeded but extension not found")
        return
    
    # Show permission approval dialog
    self.show_permission_approval_dialog(ext_id, ext.metadata.permissions)

def show_permission_approval_dialog(self, ext_id: str, permissions: list):
    """Show permission approval dialog."""
    from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QCheckBox, QPushButton
    
    dialog = QDialog(self)
    dialog.setWindowTitle(f"Approve Permissions - {ext_id}")
    
    layout = QVBoxLayout()
    layout.addWidget(QLabel(f"Extension {ext_id} requests these permissions:"))
    
    checkboxes = {}
    for perm in permissions:
        checkbox = QCheckBox(perm)
        checkbox.setChecked(False)
        checkboxes[perm] = checkbox
        layout.addWidget(checkbox)
    
    approve_btn = QPushButton("Approve Selected")
    approve_btn.clicked.connect(
        lambda: self.approve_permissions(ext_id, checkboxes)
    )
    layout.addWidget(approve_btn)
    
    dialog.setLayout(layout)
    dialog.exec()

def approve_permissions(self, ext_id: str, checkboxes: dict):
    """Approve selected permissions."""
    for perm, checkbox in checkboxes.items():
        if checkbox.isChecked():
            try:
                perm_type = PermissionType(perm)
                EXTENSION_MANAGER._permission_manager.approve_permission(ext_id, perm_type)
            except ValueError:
                LOGGER.error(f"Invalid permission: {perm}")
    
    # Check if all permissions approved
    ext = EXTENSION_MANAGER._extensions.get(ext_id)
    if ext:
        all_approved = all(
            EXTENSION_MANAGER._permission_manager.is_permitted(ext_id, PermissionType(p))
            for p in ext.metadata.permissions
        )
        
        if all_approved:
            self.enable_extension(ext_id)

def enable_extension(self, ext_id: str):
    """Enable an extension."""
    success, message = EXTENSION_MANAGER.enable_extension(ext_id)
    
    if success:
        self.show_info_dialog(f"Extension enabled: {message}")
    else:
        self.show_error_dialog(f"Failed to enable extension: {message}")

def disable_extension(self, ext_id: str):
    """Disable an extension."""
    EXTENSION_MANAGER.disable_extension(ext_id)
    self.show_info_dialog(f"Extension disabled: {ext_id}")

def create_extension_template(self, name: str, author: str):
    """Create extension template."""
    if not name or not author:
        self.show_error_dialog("Name and author are required")
        return
    
    try:
        path = SDKGenerator.create_extension(name, author)
        self.show_info_dialog(f"Template created at {path}")
    except Exception as e:
        self.show_error_dialog(f"Failed to create template: {e}")
```

### Step 6: Hook Extension Lifecycle

When rendering, trigger pre/post render hooks:

```python
def render_video(self):
    """Render video (with extension hooks)."""
    
    # Trigger pre-render hook
    if EXTENSION_PLATFORM_AVAILABLE:
        EXTENSION_MANAGER.trigger_hook("pre_render", {
            "graph": self.graph,
            "output_path": self.output_path,
        })
    
    # Do actual rendering
    # ... existing render code ...
    
    # Trigger post-render hook
    if EXTENSION_PLATFORM_AVAILABLE:
        EXTENSION_MANAGER.trigger_hook("post_render", {
            "output_path": self.output_path,
            "success": True,
        })
```

### Step 7: Hook Node Lifecycle

When nodes are created/deleted, trigger hooks:

```python
def _add_node_by_class(self, cls, x, y):
    """Add node (with extension hooks)."""
    
    node_item = self._add_node_item(cls, x, y)
    
    # Trigger node_created hook
    if EXTENSION_PLATFORM_AVAILABLE:
        EXTENSION_MANAGER.trigger_hook("node_created", {
            "node_id": node_item.data.id,
            "node_type": node_item.data.type,
            "position": (x, y),
        })
    
    return node_item

def delete_node(self, node_id: str):
    """Delete node (with extension hooks)."""
    
    node_item = self.nodes.get(node_id)
    if node_item:
        # Trigger node_deleted hook
        if EXTENSION_PLATFORM_AVAILABLE:
            EXTENSION_MANAGER.trigger_hook("node_deleted", {
                "node_id": node_id,
                "node_type": node_item.data.type,
            })
    
    # Do actual deletion
    # ... existing delete code ...
```

### Step 8: CLI Commands

Add CLI commands to `__main__` if using click:

```python
import click

@click.group()
def cli():
    """EfficientManim CLI."""
    pass

@cli.command()
@click.argument("name")
@click.option("--author", required=True)
def create_extension(name, author):
    """Create extension template."""
    try:
        path = SDKGenerator.create_extension(name, author)
        click.echo(f"✓ Extension created at {path}")
    except Exception as e:
        click.echo(f"✗ Failed: {e}", err=True)

@cli.command()
@click.argument("url")
def install_extension(url):
    """Install extension from GitHub."""
    result = GitHubInstaller.install_from_github(url)
    
    if result.success:
        click.echo(f"✓ Extension installed: {result.extension_id}")
    else:
        click.echo(f"✗ Failed: {result.message}", err=True)

if __name__ == "__main__":
    cli()
```

---

## TESTING CHECKLIST

After integration, verify:

### Extension Installation
- [ ] Can install from GitHub URL (https://github.com/owner/repo)
- [ ] Can install from shorthand (owner/repo)
- [ ] Metadata.json validation works
- [ ] Entry file validation works
- [ ] Virtual environment created
- [ ] requirements.txt installed in venv
- [ ] Signature verification works

### Permission System
- [ ] Permission approval dialog shown
- [ ] Permissions stored persistently
- [ ] Permission enforcement works
- [ ] Denied permissions raise PermissionError
- [ ] No silent failures

### Extension Management
- [ ] Extensions list UI shows all extensions
- [ ] Can enable extension
- [ ] Can disable extension
- [ ] Can uninstall extension
- [ ] Extension state persisted

### SDK & Templates
- [ ] Can create extension template
- [ ] Template is complete and documented
- [ ] Example node works
- [ ] Example track works
- [ ] MCP hooks template included
- [ ] Signing instructions included

### MCP Integration
- [ ] extension_list command works
- [ ] extension_search command works
- [ ] extension_install command works
- [ ] extension_approve_permission command works
- [ ] extension_enable command works
- [ ] extension_trigger_hook command works
- [ ] All commands return correct format

### Security
- [ ] No arbitrary pip installs
- [ ] No shell execution
- [ ] No writing outside extension directory
- [ ] Filesystem sandbox enforced
- [ ] Permissions required for all operations
- [ ] Tampered extensions disabled

### Determinism
- [ ] Extension nodes are deterministic
- [ ] Extensions don't break render
- [ ] Extensions don't mutate core
- [ ] Same graph → same output (with extension)

---

## FILE STRUCTURE AFTER INTEGRATION

```
efficientmanim/
├── main.py (UPDATED with extension platform init)
├── mcp.py (UPDATED with extension_mcp setup)
├── extension_manager.py (NEW)
├── extension_api.py (NEW)
├── github_installer.py (NEW)
├── sdk_generator.py (NEW)
├── extension_mcp.py (NEW)
└── (existing files)

~/.efficientmanim/
├── ext/
│   ├── author1/
│   │   ├── extension1/
│   │   │   ├── metadata.json
│   │   │   ├── lib.py
│   │   │   ├── venv/
│   │   │   ├── requirements.txt
│   │   │   └── signature.sig (optional)
│   │   └── extension2/
│   └── author2/
├── permissions.json (NEW)
├── layout.json
└── keybindings.json
```

---

## TROUBLESHOOTING

### Issue: Extension installation fails
**Cause**: metadata.json invalid  
**Fix**: Verify metadata.json exists and is valid JSON

### Issue: Permission enforcement not working
**Cause**: PermissionManager not initialized  
**Fix**: Ensure EXTENSION_MANAGER is created before extensions load

### Issue: Virtual environment not created
**Cause**: python -m venv failed  
**Fix**: Ensure python and venv module available

### Issue: Marketplace client not working
**Cause**: requests library not installed  
**Fix**: Install: pip install requests

### Issue: Extensions can access outside their directory
**Cause**: Path validation missing  
**Fix**: Ensure filesystem_read/write enforce extension dir path

---

## SUCCESS CRITERIA

Platform integration is complete only if:

✅ Extensions install from GitHub  
✅ Marketplace protocol works  
✅ Signatures verified  
✅ Permissions enforced  
✅ UI dialogs functional  
✅ MCP commands working  
✅ Hooks trigger correctly  
✅ Determinism preserved  
✅ No security bypasses  
✅ All tests pass  

---

## DEPLOYMENT

1. Copy platform modules to EfficientManim directory
2. Update main.py with integration code
3. Update mcp.py with extension_mcp setup
4. Test using checklist above
5. Deploy with full documentation

---

**Status**: ✅ READY FOR INTEGRATION  
**Next**: Follow integration steps in order  

