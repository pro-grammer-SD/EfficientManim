# EfficientManim — Contributing Guide

## Setting Up a Dev Environment

```bash
# 1. Fork and clone
git clone https://github.com/your-fork/EfficientManim.git
cd EfficientManim

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install dev extras (optional)
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
black app/ ui/ graph/ rendering/ core/ utils/
isort app/ ui/ graph/ rendering/ core/ utils/
```

### Naming
- Classes: `UpperCamelCase`
- Functions and methods: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_single_leading_underscore`

### Qt Patterns
- Never call blocking I/O on the main thread. Use `QThread` or background tasks.
- Use `Signal` + `connect` for cross-thread communication.
- Prefer `deleteLater()` for Qt object cleanup.

### Error Handling
- Wrap all file I/O in `try/except`.
- Render workers must emit `error` and never raise across threads.
- Log via `LOGGER` using the correct level.

---

## Module Ownership Map

| Module / File | Owner Area | Notes |
|---|---|---|
| `app/` | App startup | Bootstrapping and window creation |
| `ui/` | UI layer | Main window, panels, dialogs, menus, toolbars |
| `graph/` | Graph system | Nodes, edges, layout, scene/view |
| `rendering/` | Rendering | Preview/video workers and helpers |
| `core/` | Core services | Config, project helpers, history, assets |
| `utils/` | Utilities | Tooltips, logging, shortcuts, parsing |

---

## Key Data Flow For New Features

### Adding a New Node Type
1. Add the enum in `graph/node.py`.
2. Update validation rules in `graph/graph_editor.py`.
3. Update code generation in `ui/main_window.py`.
4. Update UI in `ui/panels/node_panel.py`.
5. Document in `docs/node_reference.md`.

### Adding a New Panel / Tab
1. Create a `QWidget` in `ui/panels/`.
2. Instantiate it in `ui/main_window.py`.
3. Add it to the top tab widget.
4. Connect to `scene.graph_changed_signal` if needed.

---

## Submitting Pull Requests

1. Branch from `main`: `git checkout -b feature/my-feature`.
2. One PR per feature; keep diffs focused.
3. Run basic checks before pushing.
4. Update `docs/` and `README.md` if behavior changes.

### PR Title Convention
```
feat: Add timeline scrubbing to video panel
fix: Correct graph load error handling
docs: Expand node reference
refactor: Extract render helpers
```

---

## Running Tests

There is currently no automated test suite. Testing is manual:
1. Launch the app and verify the feature end-to-end.
2. Test edge cases: empty scene, missing assets, invalid project file.
3. Render at least one MP4 with changes applied.

If you add tests, place them in `tests/` and use `pytest`.

---

## Reporting Bugs

Open an issue with:
- EfficientManim version (shown in title bar)
- Python version (`python --version`)
- OS and version
- Steps to reproduce
- Expected vs actual behavior
- Relevant lines from `~/.efficientmanim/app.log`
