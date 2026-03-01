# EFFICIENTMANIM EXTENSION PLATFORM SPECIFICATION

**Version**: 1.0  
**Status**: Complete Architecture  
**Compliance**: Deterministic, Sandboxed, Governed  

---

## EXECUTIVE SUMMARY

EfficientManim is no longer just an editor. It is a **platform** with an extension ecosystem.

This specification defines:

1. **Extension Marketplace Protocol** — Discoverable, installable extensions
2. **Remote GitHub Installation** — One-command extension installation
3. **Signed Extension Verification** — Cryptographic signature validation
4. **Permission Approval UI** — User-controlled permission grants
5. **Community SDK Template System** — Extension scaffolding generator
6. **Extension Update System** — Version management with compatibility checking
7. **Deterministic Contract Preservation** — Extensions cannot break reproducibility

---

## ARCHITECTURE OVERVIEW

### Platform Layer Hierarchy

```
EfficientManimWindow (host application)
    ↓
ExtensionManager (central authority)
    ├── ExtensionSecurityLayer (validation)
    ├── PermissionManager (governance)
    ├── GitHubInstaller (remote installation)
    ├── MarketplaceClient (marketplace communication)
    ├── SDKGenerator (template scaffolding)
    ├── UpdateManager (version management)
    │
    └── Extension Registry
        ├── Extension (metadata, permissions, state)
        ├── Extension (metadata, permissions, state)
        └── Extension (metadata, permissions, state)
```

### Data Flow

```
User Action (e.g., install from GitHub)
    ↓
ExtensionManager.install_from_github(url)
    ↓
GitHubInstaller.install_from_github()
    - Clone repo
    - Validate metadata.json
    - Create venv
    - Install requirements.txt
    ↓
ExtensionSecurityLayer.validate_extension()
    - Check metadata
    - Verify engine compatibility
    - Verify signature (if present)
    - Check dependencies
    ↓
PermissionManager.request_permission()
    - Show UI to user
    - User approves/denies
    ↓
ExtensionManager.enable_extension()
    - Load module
    - Call setup(api)
    - Register hooks/nodes/tracks
    ↓
Extension Ready (deterministic contract preserved)
```

---

## COMPONENT SPECIFICATIONS

### 1. ExtensionManager

**Responsibilities**:
- Central authority for all extension operations
- Discover installed extensions
- Enable/disable extensions
- Manage hooks
- Enforce governance rules

**API**:
```python
EXTENSION_MANAGER.discover_installed()        # List extensions
EXTENSION_MANAGER.enable_extension(ext_id)    # Enable with checks
EXTENSION_MANAGER.disable_extension(ext_id)   # Disable
EXTENSION_MANAGER.register_hook(name, cb)     # Register hook
EXTENSION_MANAGER.trigger_hook(name, ...)     # Trigger hooks
```

**Critical Rule**: All extension operations must go through this manager.

### 2. ExtensionSecurityLayer

**Validates**:
- Metadata completeness (name, author)
- Engine version compatibility
- Signature validity (if present)
- Dependency safety (no forbidden packages)

**Returns**: (valid: bool, message: str)

**Critical Rule**: Extensions rejected if security validation fails.

### 3. PermissionManager

**Permission Types**:
```
REGISTER_NODES
REGISTER_TIMELINE_TRACK
REGISTER_UI_PANEL
REGISTER_MCP_HOOK
FILESYSTEM_ACCESS
NETWORK_ACCESS
```

**Enforcement**:
```python
# Extension tries to register node
api.register_node(...)  # Calls:
permission_manager.check_permission(ext_id, REGISTER_NODES)
# If not approved: PermissionError
```

**Critical Rule**: No silent failures. Denied permission raises exception.

### 4. GitHubInstaller

**Installation Flow**:
1. Parse GitHub URL (full URL or shorthand)
2. Clone into `~/.efficientmanim/ext/author/repo`
3. Validate metadata.json exists and is valid JSON
4. Validate entry file exists
5. Create virtual environment (isolated)
6. Install requirements.txt (in venv only)
7. Return InstallationResult

**Security Rules**:
- No arbitrary pip installs outside venv
- No shell execution
- No writing outside allowed directories
- Clone timeout: 30 seconds
- pip timeout: 5 minutes

### 5. MarketplaceClient

**Marketplace Index Format**:
```json
{
  "extensions": [
    {
      "name": "ExtensionName",
      "author": "github_username",
      "repo": "extension_repo",
      "version": "1.0.0",
      "engine_version": ">=2.0.3",
      "description": "...",
      "permissions": ["register_nodes", "register_timeline_track"],
      "verified": true,
      "signature_url": "https://..."
    }
  ]
}
```

**Operations**:
```python
marketplace.get_index()                  # Fetch full index
marketplace.search_extensions(query)     # Search
marketplace.get_extension_info(a, r)     # Detailed info
marketplace.verify_extension(a, r, v)    # Verify with marketplace
```

### 6. SDKGenerator

**Creates Extension Template**:
```
extension_name/
├── metadata.json
├── lib.py (with setup() function)
├── requirements.txt
├── README.md
├── example_node.py
├── example_timeline_track.py
├── mcp_hooks.py
├── test_extension.py
├── SIGNING.md
└── .gitignore
```

**Usage**:
```bash
efficientmanim create-extension MyExtension --author github_username
```

**Output**: Complete, documented, ready-to-customize template.

### 7. UpdateManager

**Operations**:
```python
manager.check_for_updates(ext_id)        # Check for new version
manager.apply_update(ext_id)             # Apply update
manager.get_available_updates()          # List all updatable
```

**Update Process**:
1. Check marketplace for newer version
2. Verify engine compatibility
3. Re-verify signature
4. If permissions changed → require re-approval
5. Apply update files
6. Restart extension

### 8. ExtensionAPI

**Safe Interface** for extensions:
```python
# Extensions never touch host directly
# All access goes through API

api.register_node(...)
api.register_timeline_track(...)
api.register_ui_panel(...)
api.register_mcp_hook(...)
api.get_graph()                  # Read-only snapshot
api.get_timing(node_id)
api.log(message, level)
api.filesystem_read(path)        # Extension dir only
api.filesystem_write(path, data) # Extension dir only
```

**Permission Enforcement**:
Every API call checks permissions internally.

```python
api.register_node(...)  # Checks REGISTER_NODES internally
# If not approved: PermissionError("Extension X lacks permission: register_nodes")
```

---

## SECURITY MODEL

### Signature Verification

**Standard**: RSA-2048 signatures

**Files Signed**:
- metadata.json
- lib.py
- requirements.txt
- declared_files

**Verification States** (shown in UI):
- 🟢 **Verified** — Signature matches public key, author is verified
- 🟡 **Signed** — Signature valid but author unverified
- 🔴 **Unsigned** — No signature present
- ❌ **Tampered** — Signature invalid (disabled automatically)

**Critical Rule**: Tampered extensions are disabled immediately.

### Permission System

**User Approves Permissions During Install**:
```
Install Extension: SuperCameraPack
Author: github_dev
Version: 1.2.0

Permissions Requested:
  ✓ register_nodes
  ✓ register_timeline_track

[Deny] [Approve]
```

**Permissions Persistent**:
- Stored in `~/.efficientmanim/permissions.json`
- Survive app restart
- Can be revoked in UI

### Sandbox Isolation

**Virtual Environments**:
- Each extension has its own venv
- Dependencies isolated
- No cross-extension conflicts

**Filesystem Sandbox**:
- Extensions can only access their own directory
- Read/write via API (enforces permissions)
- No directory traversal allowed

---

## DETERMINISM GUARANTEE

### Contract

**Extensions CANNOT break determinism.**

Before enabling any extension:
1. Security validation passes
2. All permissions approved
3. Signature verified (if present)
4. No core mutation capability
5. Read-only graph access

**Proof**:
```python
# Extension can register nodes, but:
# - Cannot modify existing nodes
# - Cannot access host internals
# - Cannot break timing contract
# - Cannot modify render logic

# Same graph + same timing = same output
# Extensions don't change this guarantee
```

---

## EXTENSION DEVELOPMENT WORKFLOW

### Step 1: Create Template
```bash
efficientmanim create-extension MyExtension --author myusername
cd MyExtension
```

### Step 2: Customize
- Edit `lib.py` to register nodes/tracks
- Create custom node classes in `example_node.py`
- Add MCP hooks in `mcp_hooks.py`
- Add dependencies to `requirements.txt`

### Step 3: Test
```bash
python -m pytest tests/
```

### Step 4: Sign (Optional)
```bash
# Generate keys (one-time)
openssl genrsa -out private_key.pem 2048
openssl rsa -in private_key.pem -pubout -out public_key.pem

# Sign extension
efficientmanim sign-extension \
  --extension MyExtension \
  --private-key private_key.pem \
  --output signature.sig

# Add signature.sig and public_key.pem to repo
git add signature.sig public_key.pem
```

### Step 5: Publish
```bash
git push origin main
# Submit via: https://marketplace.efficientmanim.dev/publish
```

### Step 6: Users Install
```bash
# From GitHub
efficientmanim install myusername/MyExtension

# From Marketplace
# Search and click "Install"
```

---

## MARKETPLACE PROTOCOL

### Discovery
```
User searches marketplace
↓
Returns matching extensions with metadata
↓
User clicks "Install"
↓
Shows permissions, author verification
↓
GitHub installer clones repo
```

### Metadata Requirements

**For Marketplace Listing**:
```json
{
  "name": "...",
  "author": "...",
  "version": "...",
  "description": "...",
  "engine_version": ">=2.0.3",
  "permissions": [...],
  "verified": true/false,
  "screenshot_urls": [...]
}
```

### Compatibility Filtering

- Only show compatible versions
- Hide extensions for older engines
- Warn if engine too old
- Block installation if incompatible

---

## GOVERNANCE RULES

All extension operations must:

✅ Go through ExtensionManager  
✅ Have security validation  
✅ Have permission checks  
✅ Be logged for audit trail  
✅ Be reversible (can disable)  
✅ Not mutate graph directly  
✅ Respect deterministic contract  
✅ Use isolated venv  

**Forbidden**:
❌ Direct file writes outside extension directory  
❌ Shell execution  
❌ Arbitrary pip installs  
❌ Access to host internals  
❌ Breaking determinism  

---

## TESTING INFRASTRUCTURE

### Extension Testing Template

```python
# test_extension.py (provided in template)

class MockAPI:
    """Mock ExtensionAPI for unit testing"""
    def register_node(self, ...): ...
    def register_timeline_track(self, ...): ...

def test_setup():
    """Test extension initialization"""
    api = MockAPI()
    setup(api)
    assert len(api.registered_nodes) > 0

def test_deterministic():
    """Test extension nodes are deterministic"""
    node1 = MyNodeClass()
    node2 = MyNodeClass()
    assert type(node1) == type(node2)
```

### CI/CD Integration

```yaml
# .github/workflows/test.yml
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install -r requirements.txt
      - run: pytest tests/
      - run: python -m efficientmanim sign-extension ...
```

---

## API DOCUMENTATION

### register_node()
```python
api.register_node(
    node_name: str,          # "MyNode"
    class_path: str,         # "module.ClassName"
    category: str = "Custom",# Node browser category
    description: str = ""    # Help text
)
```

Requires: `REGISTER_NODES` permission

### register_timeline_track()
```python
api.register_timeline_track(
    track_name: str,
    class_path: str,
    description: str = ""
)
```

Requires: `REGISTER_TIMELINE_TRACK` permission

### register_ui_panel()
```python
api.register_ui_panel(
    panel_name: str,
    widget_class: str,
    position: str = "right"  # "left", "right", "bottom", "floating"
)
```

Requires: `REGISTER_UI_PANEL` permission

### register_mcp_hook()
```python
api.register_mcp_hook(
    hook_name: str,          # "pre_render", "post_render", etc.
    callback: Callable       # async callable(context) -> dict
)
```

Requires: `REGISTER_MCP_HOOK` permission

Supported hooks:
- `pre_render` — Before rendering starts
- `post_render` — After rendering completes
- `node_created` — When node added
- `node_deleted` — When node removed
- `timeline_changed` — When timeline modified

### filesystem_read() / filesystem_write()
```python
data = api.filesystem_read("config.json")      # Read from extension dir
api.filesystem_write("output.json", data)      # Write to extension dir
```

Requires: `FILESYSTEM_ACCESS` permission

---

## SUCCESS CRITERIA

Implementation is complete only if:

✅ Extensions install from GitHub  
✅ Marketplace protocol works  
✅ Signature verification functional  
✅ Permission approval enforced  
✅ SDK generator creates templates  
✅ Updates validated and applied  
✅ No core mutation possible  
✅ Determinism preserved  
✅ Hooks work for AI integration  
✅ Filesystem sandbox enforced  

---

## FAILURE CONDITIONS

Implementation is **INCOMPLETE** if:

❌ Extensions can bypass ExtensionManager  
❌ Permissions not enforced  
❌ Tampered extensions not disabled  
❌ Extensions can mutate core  
❌ Determinism broken  
❌ Signature verification missing  
❌ SDK doesn't generate working templates  
❌ Updates don't re-verify  

---

## SUMMARY

The extension platform provides:

✅ **Discoverable** extensions via marketplace  
✅ **Installable** from GitHub in one command  
✅ **Secure** via signatures and sandboxing  
✅ **Governed** via permission system  
✅ **Updatable** with compatibility checking  
✅ **Deterministic** (extensions can't break it)  
✅ **Scalable** (marketplace ecosystem)  

**Status**: ✅ **COMPLETE**  
**Compliance**: **FULL SPECIFICATION**  

---

End of Extension Platform Specification
