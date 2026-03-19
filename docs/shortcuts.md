# ⌨️ EfficientManim — Keyboard Shortcuts

## Full Shortcut Reference

### File

| Action | Default Shortcut | Description |
|---|---|---|
| New Project | `Ctrl+N` | Clear canvas and start fresh |
| Open Project | `Ctrl+O` | Open a `.efp` project file |
| Save Project | `Ctrl+S` | Save to current file (prompts if unsaved) |
| Save As | `Ctrl+Shift+S` | Save to a new filename |
| Exit | `Ctrl+Q` | Quit the application |

### Edit

| Action | Default Shortcut | Description |
|---|---|---|
| Undo | `Ctrl+Z` | Undo last node add/remove/parameter change |
| Redo | `Ctrl+Y` | Redo last undone action |
| Delete Selected | `Del` | Delete selected nodes and wires |
| Select All | `Ctrl+A` | Select all nodes on canvas |
| Duplicate Node | `Ctrl+D` | Duplicate selected node |

### View & Navigation

| Action | Default Shortcut | Description |
|---|---|---|
| Fit View | `Ctrl+0` | Fit all nodes to the visible viewport |
| Zoom In | `Ctrl+=` | Zoom into canvas |
| Zoom Out | `Ctrl+-` | Zoom out of canvas |
| Zoom (mouse) | `Ctrl+Scroll` | Zoom centered on cursor |
| Pan | `Middle Mouse` | Pan the canvas by dragging |
| Auto-Layout | `Ctrl+L` | Automatically arrange nodes in columns |
| Clear All | `Ctrl+Alt+Del` | Delete everything (requires confirmation) |

### Tools & Nodes

| Action | Default Shortcut | Description |
|---|---|---|
| Export Code (.py) | `Ctrl+E` | Save generated scene code to a Python file |
| Copy Code | `Ctrl+Shift+C` | Copy generated code to clipboard |
| Render Video | `Ctrl+R` | Start full scene video render |
| Add Play Node | `Ctrl+Shift+P` | Add an explicit `self.play()` node |
| Add Wait Node | `Ctrl+Shift+W` | Add an explicit `self.wait()` node |
| Create VGroup | `Ctrl+G` | Create VGroup from canvas selection |

### AI

| Action | Default Shortcut | Description |
|---|---|---|
| AI Generate | `Ctrl+G` | Generate with AI (in AI panel) |

### Help & Settings

| Action | Default Shortcut | Description |
|---|---|---|
| Edit Keybindings | `Ctrl+,` | Open keybindings editor |
| Keyboard Shortcuts | `Ctrl+?` | Show this shortcuts reference |
| Settings | `Ctrl+,` | Open settings dialog |

### Tabs

| Action | Default Shortcut | Description |
|---|---|---|
| Next Tab | `Ctrl+Tab` | Switch to next sidebar tab |
| Previous Tab | `Ctrl+Shift+Tab` | Switch to previous sidebar tab |
| Switch to Editor | `Ctrl+1` | Switch to editor view |
| Switch to Timeline | `Ctrl+2` | Switch to timeline view |

---

## Editing Keybindings

1. Open **Help → Edit Keybindings…** (or press `Ctrl+,`).
2. The keybindings panel shows all registered actions with their current shortcut.
3. Double-click any row to edit the shortcut — press the desired key combination.
4. If there is a conflict, a warning is shown and the duplicate is highlighted.
5. Changes are applied **immediately** to all corresponding `QAction` objects — no restart required.
6. Keybindings are persisted to `QSettings` (Windows Registry or `~/.config/Gemini/EfficientManim.ini`).

### Resetting to Defaults

Click **Reset All to Defaults** in the keybindings panel. This restores every action to the default from the table above.

### Shortcut Conflict Resolution

The registry tracks all reserved shortcuts. If you attempt to assign a shortcut that is already in use, the editor warns you and shows which action currently owns it. You must either reassign the conflicting action first or choose a different key.

---

## Canvas Mouse Controls

| Action | Input |
|---|---|
| Add wire | Click output socket (red) → drag → click input socket (green) |
| Pan | Hold middle mouse button and drag |
| Zoom | `Ctrl+Scroll` |
| Select one node | Left click |
| Select multiple | Rubber-band drag (left click empty space) |
| Select all | `Ctrl+A` |
| Move node | Left click and drag |
| Delete selected | `Del` |

---

## Tips

- The canvas filter bar (`🔍 Filter nodes…` above the canvas) dims non-matching nodes so you can quickly locate what you're looking for.
- `Ctrl+L` auto-layout separates Mobjects (column 1) and Animations (column 2), then calls `fit_view` automatically.
