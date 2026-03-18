# 🌿 EfficientManim — The Ultimate Node-Based Manim IDE

![Icon](icon/icon.ico)

**🌈 Create mathematical animations visually with the power of Python, AI, and real-time collaboration.**

---

## 🚀 Key Features

### 🎬 Node-Based Visual Workflow
- **Visual Editor:** Drag-and-drop Mobjects and Animations with intuitive wiring
- **Infinite Canvas:** Pan and zoom freely to manage large node graphs
- **Live Preview:** Real-time static previews of individual nodes
- **Smart Connections:** Automatic wire validation and scene synchronization

### 🎬 Multiple Scenes
- Manage multiple scenes per project in the **Scenes** tab
- Create, rename, delete, and switch between scenes — each with its own node graph
- Scene state auto-saved and restored when switching

### 📦 VGroup Utility
- Select Mobject nodes → click **Create VGroup** in the **VGroups** tab
- VGroup code automatically generated: `group_1 = VGroup(circle_1, square_1)`
- Groups shown in expandable tree view with source badges (canvas / AI / snippet / GitHub)
- Full member management: add, remove, highlight on canvas, copy code

---

## 🤝 Live Collaboration

EfficientManim supports **real-time multi-user editing** over a local network or on the same machine — no cloud accounts, no third-party services.

### What It Is
Two or more EfficientManim windows share the same node graph live. Every node move, property change, wire add/delete, and scene switch is broadcast as a JSON delta and applied to all connected instances within milliseconds. A 6-digit **PIN** identifies the session — easy to read aloud or paste into chat.

### Starting a Session
1. Open your project in EfficientManim.
2. Click **Collaboration → Start Live Collaboration**.
3. A dialog shows your PIN in large, bold text. Share it with collaborators.
4. The toolbar shows a green ● and the active PIN.

### Joining a Session
1. Click **Collaboration → Join Collaboration**.
2. Enter the 6-digit PIN.
3. Click **Connect** — your canvas loads the host's full graph and live sync begins.

### Multi-Window (Same Machine)
Run `python main.py` a second time, join via the same PIN — both windows stay in sync.

### Network Requirements
- **Same machine:** No setup needed.
- **LAN:** Host firewall must allow the ephemeral TCP port shown in the session dialog.
- **Internet:** Not supported out-of-the-box; use a reverse proxy (e.g. `ngrok`) on the host.

📖 Full details: [docs/live_collaboration.md](docs/live_collaboration.md)

---

## 🤖 Gemini AI Integration

### Code Generation
- Describe animations in plain English — AI generates a complete `Scene.construct()` block
- AI code parsed into typed, editable nodes with correct wiring
- Streaming responses with real-time feedback in the AI Assistant dock

### MCP Agent Mode
- Gemini reads the live scene state as JSON and issues typed commands executed directly against the running app
- No merge step — Gemini edits the graph the same way a human would
- Full action log: **Help → MCP Agent → Show Action Log**

### Auto Voiceover Agent
- Analyzes all nodes, writes per-node scripts, generates TTS, attaches audio automatically
- At render time, all segments merge into a synchronized voiceover track

📖 Full details: [docs/ai_features.md](docs/ai_features.md)

---

## 🎙️ AI Voiceover Studio
- Gemini TTS with six voices: Puck, Charon, Kore, Fenrir, Aoede, Zephyr
- Built-in audio player with play/pause/stop/seek and time display
- Attach audio to any node — duration auto-syncs `run_time=`

---

## 🐙 GitHub Snippet Loader
- Clone any GitHub repository and browse its `.py` files
- Double-click to load into the AI panel as a snippet
- VGroup definitions auto-detected and registered

---

## ⭐ Recents Panel
- Top-5 most-used Mobjects and Animations by actual insertion count
- Persisted to `~/.efficientmanim/usage.json`
- Double-click to instantly add to canvas

---

## ⌨️ Editable Keybindings
- **Help → Edit Keybindings…** — changes apply instantly, no restart
- Duplicate detection; persisted to `QSettings`

---

## 📦 Portable Project Format (.efp)
- ZIP-based: graph JSON + compiled code + all bundled media
- Cross-platform — one file contains everything
📖 Full details: [docs/efp_format.md](docs/efp_format.md)

---

## 🎨 Manim Class Browser
- 60+ Manim classes in 8 categories with real-time search filter
- Double-click or drag to add node to canvas

---

## 🎬 Professional Video Rendering
- Full scene export to MP4/WebM, up to 4K, 15–60 FPS, four quality presets
- Integrated video player for instant review

---

## ✒️ LaTeX Studio
- Live LaTeX preview via MathPad API
- One-click apply to any `MathTex` or `Tex` node

---

## 🔌 Plugin / Extension System
- Extensions in `~/.efficientmanim/ext/<author>/<name>/`
- Permission-gated: nodes, UI panels, timeline tracks, MCP hooks
📖 Full details: [docs/plugin_api.md](docs/plugin_api.md)

---

## 🏠 Home Screen

```bash
python home.py   # Home screen with recent projects
python main.py   # Open editor directly
```

---

## ⌨️ Keyboard Shortcuts

| Action | Shortcut | Action | Shortcut |
|--------|----------|--------|----------|
| New Project | `Ctrl+N` | Fit View | `Ctrl+0` |
| Open Project | `Ctrl+O` | Zoom In | `Ctrl+=` |
| Save Project | `Ctrl+S` | Zoom Out | `Ctrl+-` |
| Save As | `Ctrl+Shift+S` | Auto-Layout | `Ctrl+L` |
| Exit | `Ctrl+Q` | Export Code | `Ctrl+E` |
| Undo | `Ctrl+Z` | Copy Code | `Ctrl+Shift+C` |
| Redo | `Ctrl+Y` | Render Video | `Ctrl+R` |
| Delete Selected | `Del` | Add Play Node | `Ctrl+Shift+P` |
| Select All | `Ctrl+A` | Add Wait Node | `Ctrl+Shift+W` |
| Create VGroup | `Ctrl+G` | Edit Keybindings | `Ctrl+,` |

📖 Full reference: [docs/shortcuts.md](docs/shortcuts.md)

---

## 📚 Documentation

| Document | Contents |
|---|---|
| [docs/overview.md](docs/overview.md) | Architecture, modules, data flow |
| [docs/setup.md](docs/setup.md) | Installation, requirements, first launch |
| [docs/live_collaboration.md](docs/live_collaboration.md) | PIN sessions, delta sync, network setup |
| [docs/node_reference.md](docs/node_reference.md) | All node types and properties |
| [docs/ai_features.md](docs/ai_features.md) | Code gen, MCP agent, voiceover, LaTeX |
| [docs/shortcuts.md](docs/shortcuts.md) | Full keyboard shortcut reference |
| [docs/efp_format.md](docs/efp_format.md) | Project file format internals |
| [docs/plugin_api.md](docs/plugin_api.md) | Extension / plugin development guide |
| [docs/contributing.md](docs/contributing.md) | Dev setup, code style, PR guide |

---

## 🛠️ Installation

```bash
git clone https://github.com/pro-grammer-SD/EfficientManim.git
cd EfficientManim
pip install -r requirements.txt
python main.py
```

**Requirements:** Python 3.10+, FFmpeg (on PATH), Git, LaTeX (for MathTex)

Full guide: [docs/setup.md](docs/setup.md)

---

## 🌿 About

© 2026 — Soumalya Das (@pro-grammer-SD) · Co-authored with Bailey Beber  
Built with PySide6 · Manim · Google Gemini
