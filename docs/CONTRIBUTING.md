# 🛠️ EfficientManim — Contributing Guide

## Setting Up a Dev Environment

```bash
# 1. Fork and clone
git clone https://github.com/your-fork/EfficientManim.git
cd EfficientManim

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Install dev extras (optional but recommended)
pip install pytest black isort pyflakes

# 5. Launch
python main.py
```

---

## Code Style & Conventions

### Formatting
- **Black** with default settings (line length 88).
- **isort** for import ordering.

```bash
black main.py core/ collab/ app/
isort main.py core/ collab/ app/
```

### Naming
- Classes: `UpperCamelCase`
- Functions and methods: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_single_leading_underscore`

### Qt Patterns
- **Never call blocking I/O on the main thread.** Use `QThread` subclasses or background `asyncio` loops.
- Use `QTimer.singleShot(0, fn)` to defer UI updates from worker callbacks back to the main thread.
- Prefer `Signal` + `connect` over direct method calls across thread boundaries.
- Always `deleteLater()` Qt objects instead of `del` when crossing thread boundaries.

### Error Handling
- All file I/O must be wrapped in `try/except`.
- Render workers must never propagate exceptions — catch and emit `error` signal.
- Log to `LOGGER` (not `print`) using appropriate level: `info`, `warn`, `error`.

---

## Module Ownership Map

| Module / File | Owner Area | Notes |
|---|---|---|
| `main.py` | Core app | All UI classes, compiler, renderer, project I/O. Monolithic by design. |
| `home.py` | Home screen | Recent projects, launch logic |
| `collab/` | Live collaboration | WebSocket server/client, delta sync, PIN registry |
| `collab/server.py` | WS server | `CollabServer` — asyncio in background thread |
| `collab/client.py` | WS client | `CollabClient` — asyncio in background thread |
| `collab/manager.py` | High-level collab | `CollaborationManager` — Qt signals, session lifecycle |
| `collab/delta.py` | Delta logic | `make_delta`, `serialize_graph`, `apply_delta` |
| `collab/conflict.py` | Node locking | `NodeLockManager`, `LockInfo`, TTL expiry |
| `collab/pin_registry.py` | PIN persistence | `~/.efficientmanim/collab_sessions.json` |
| `core/mcp.py` | MCP agent | `MCPAgent`, `MCPRegistry`, command handlers |
| `core/keybinding_registry.py` | Keybindings | Single-source-of-truth registry |
| `core/keybindings_panel.py` | Keybindings UI | `UnifiedKeybindingsPanel` |
| `core/themes.py` | Theme system | `THEME_MANAGER`, QSS generation |
| `core/extension_manager.py` | Extension platform | Permission governance, lifecycle |
| `core/extension_api.py` | Extension API | Safe sandboxed API for plugins |
| `core/node_registry.py` | Node registry | Custom node type registration |
| `app/` | Legacy app layer | Older versions of core modules, kept for compat |

---

## Key Data Flow for New Features

### Adding a New Delta Action (for collaboration)

1. Add the action string to `collab/delta.py` → `apply_delta()` with an `if action == "..."` block.
2. Emit the delta from the relevant UI event in `main.py` by calling `self.collab.send_delta(action, payload)` — guard with `if not self._collab_applying`.
3. Document the new action in `docs/live_collaboration.md`.

### Adding a New Node Type

1. Add an entry to `NodeType` enum in `main.py`.
2. Add a header color in `NodeItem.paint()`.
3. Add connection rules to `GraphScene.try_connect()`.
4. Add code generation logic to `EfficientManimWindow.compile_graph()`.
5. Add a type badge in `PropertiesPanel.set_node()`.
6. Update `docs/node_reference.md`.

### Adding a New Panel / Tab

1. Create a `QWidget` subclass.
2. Instantiate it in `EfficientManimWindow.setup_ui()`.
3. Add it to `self.tabs_top.addTab(...)`.
4. If it needs graph change notifications, connect `self.scene.graph_changed_signal`.

---

## Submitting Pull Requests

1. **Branch** from `main`: `git checkout -b feature/my-feature`.
2. **One PR per feature** — keep diffs focused.
3. **No syntax errors.** Run `python -c "import ast; ast.parse(open('main.py').read())"` before pushing.
4. **Update docs** — if you change behaviour, update the relevant file in `docs/`.
5. **Update README.md** if the feature is user-visible.
6. **Describe the change** in the PR body: what it does, why, and how you tested it.

### PR Title Convention

```
feat: Add timeline scrubbing to canvas bottom bar
fix: Correct load_graph_from_json for collab sync
docs: Add plugin_api.md and contributing guide
refactor: Split compile_graph into smaller helpers
```

---

## Running Tests

There is currently no automated test suite. Testing is done manually:

1. Launch the app and verify the feature works end-to-end.
2. Test edge cases: empty scene, no API key, missing assets, corrupt project file.
3. For collaboration features: open two windows, start a session, verify deltas sync.
4. For rendering: produce at least one MP4 with your changes applied.

If you add automated tests, place them in `tests/` and use `pytest`.

---

## Reporting Bugs

Open a GitHub issue with:
- EfficientManim version (shown in title bar)
- Python version (`python --version`)
- OS and version
- Steps to reproduce
- Expected vs actual behaviour
- Relevant section from `~/.efficientmanim/session.log`
