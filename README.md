# 🌿 EfficientManim

<p align="center">
  <img src="icon/icon.ico" alt="EfficientManim Icon" width="80"/>
</p>

<p align="center">
  <a href="https://github.com/pro-grammer-SD/EfficientManim"><img src="https://img.shields.io/badge/Python-3.10+-3776ab?style=for-the-badge&logo=python&logoColor=white" alt="Python"/></a>
  <a href="https://ideacred.com/submissions" target="_blank" rel="noopener noreferrer">
  <img src="https://ideacred.com/api/badge/pro-grammer-SD/EfficientManim?style=for-the-badge" 
       alt="IdeaCred"></a>
  <a href="https://www.manim.community/"><img src="https://img.shields.io/badge/Manim-Community-4caf50?style=for-the-badge" alt="Manim"/></a>
  <a href="https://github.com/pro-grammer-SD/EfficientManim/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-f59e0b?style=for-the-badge" alt="MIT License"/></a>
  <a href="https://www.reddit.com/r/manim/comments/1qck0ji/i_built_a_nodebased_manim_ide_with_ai_assistance/"><img src="https://img.shields.io/badge/Reddit-Discussion-ff4500?style=for-the-badge&logo=reddit&logoColor=white" alt="Reddit"/></a>
</p>

<p align="center">
  <strong>✨🌿 A node-based visual IDE for mathematical animations — with AI tooling, a structured history system, and a built-in explanation engine for students and teachers. 🎊🎯</strong>
</p>

---

## ✨ What It Is

EfficientManim turns [Manim](https://www.manim.community/) from a code-only tool into a visual, AI-augmented workspace. Wire Mobjects and animations together on an infinite canvas, generate scenes from plain English or PDF slides, and get structured explanations for every move — all without leaving the editor.

---

## 🚀 Features

### 🎬 Node-Based Visual Workflow

Drag-and-drop Mobjects and Animations onto an **infinite canvas**, connect them with smart wires, and watch your scene graph come alive. Wire validation runs automatically, and live static previews update as you build.

- **60+ Manim classes** across 8 categories (Geometry, Text, Graphs, 3D, Animations…), searchable in real time via the **Manim Class Browser**
- **Multiple Scenes** per project — create, rename, delete, and switch between them in the Scenes tab; each scene carries its own node graph and auto-saves state on switch
- **VGroup Utility** — select Mobject nodes → click **Create VGroup** → get auto-generated `VGroup(...)` code displayed in an expandable tree
- **Recents Panel** — top-5 Mobjects and top-5 Animations tracked per session, persisted to `~/.efficientmanim/usage.json`; double-click to add instantly

---

### 🤖 AI Features

#### Prompt → Manim
Describe an animation in plain English. Get runnable Manim code.

#### 📄 PDF Slide Animation
Upload one or more PDFs. The AI parses each slide, builds an animation plan, and produces both Manim code and a ready-to-edit node graph — turning a static lecture deck into a structured animated scene.

#### 🎙️ AI Voiceover Studio
TTS generation with multi-voice support and automatic duration syncing to animation timing.

#### 🐙 GitHub Snippet Loader
Clone any GitHub repo into `~/.efficientmanim/snippets/`, browse `.py` files, and double-click to load a snippet directly into the AI panel. All repos repopulate on startup automatically.

---

### 🧠 Explain This Animation

Select a scene, a group of nodes, or a single animation — then hit **🧠 Explain** in the main toolbar. The system produces a clear, student-friendly explanation of what the animation is doing and why.

| Mode | What You Get |
|------|-------------|
| **Simple** | Short, intuition-first summary |
| **Detailed** | Full step breakdown with conceptual reasoning |

Copy or regenerate instantly from the Explain Panel. Explanations can also be triggered programmatically via MCP (`explain.scene`, `explain.selected_nodes`).

---

### 🎓 Learning Mode

Learning Mode watches what you build and fires micro-explanations as you go — turning the editor into a live tutor.

**Triggers automatically when you:**
- Add axes, graphs, or equations
- Apply transformations
- Attach tangent lines, slopes, or area visuals
- Hit a checkpoint or land a large batch addition

**Default:** OFF (to keep the workspace quiet for experienced users).  
**Enable:** Settings → **Learning Mode** → Enable Learning Mode.

---

### 👩🏫 Teacher Mode

Generate a full structured lesson from the current scene in one click.

**Output includes:**
- Concept explanation
- Visual explanation tied to the animation
- Step-by-step teaching script
- Student-friendly notes
- Key takeaways

**Export as:** Markdown (`.md`), plain text (`.txt`), or copy to clipboard.  
**Enable:** Settings → **Teacher Mode** → Enable Teacher Mode.

---

### 🕰️ History-Powered Explanations

The rewritten `HistoryManager` captures grouped atomic operations — AI merges, property edits, motion, wiring — with per-node undo/redo stacks and project/scene/node checkpoints.

The explanation engine hooks directly into history:
- **Explain a checkpoint state** — what did the scene look like at this moment?
- **Explain a diff** — what changed between two checkpoints?
- **Explain an undo/redo action** — in plain student language, not code

See `docs/history_system.md` for the full data model and checkpoint API.

---

### 🧭 MCP Command Interface

Every action in EfficientManim — node edits, wires, VGroups, scenes, rendering, assets, AI workflows, themes, keybindings, TTS, explanations, Learning Mode, Teacher Mode — is scriptable via structured JSON through the MCP command layer.

```json
// Example
{ "command": "explain.scene", "mode": "detailed" }
{ "command": "teacher_mode.generate_lesson", "export": "markdown" }
{ "command": "history.restore_checkpoint", "id": "chk_42" }
```

Full reference in `docs/mcp_commands.md`.

---

### 🎨 Workspace & Polish

| Feature | Detail |
|---------|--------|
| **Dark / Light Themes** | Full dark mode, QSS-based with a ColorToken system, toggled via `Ctrl+T` or Settings |
| **Tooltips Everywhere** | Every button, panel, and action has a tooltip — the UI is self-explanatory |
| **Editable Keybindings** | Help → Edit Keybindings… Double-click any shortcut to rebind; duplicate detection included |
| **Editable Project Name** | Rename the `.efp` file from a textbox in the top-right corner — no file manager required |
| **Portable `.efp` Format** | ZIP-based project bundle: nodes, wires, images, sounds, and video assets in one file |
| **Professional Rendering** | Full scene export to MP4/WebM — up to 4K, 15–60 FPS, multiple quality presets |

---

## ⌨️ Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| New Project | `Ctrl+N` |
| Open Project | `Ctrl+O` |
| Save Project | `Ctrl+S` |
| Save As | `Ctrl+Shift+S` |
| Exit | `Ctrl+Q` |
| Undo | `Ctrl+Z` |
| Redo | `Ctrl+Y` |
| Delete Selected | `Del` |
| Fit View | `Ctrl+0` |
| Auto-Layout | `Ctrl+L` |
| Export Code | `Ctrl+E` |
| Copy Code | `Ctrl+Shift+C` |
| Toggle Theme | `Ctrl+T` |
| Edit Keybindings | `Ctrl+,` |

All shortcuts are fully editable via **Help → Edit Keybindings…**

---

## 🛠️ Prerequisites

- **Python 3.10+**
- **FFmpeg** — required for video rendering (must be in `PATH`)
- **Git** — optional, for the GitHub Snippet Loader
- **LaTeX** — optional, for local LaTeX rendering

---

## 📦 Installation

```bash
git clone https://github.com/pro-grammer-SD/EfficientManim.git
cd EfficientManim
pip install -r requirements.txt
python main.py
```

**Manual install:**

```bash
pip install manim PySide6 google-genai pydub requests numpy pdfplumber regex
```

---

## 📸 Gallery

| | |
|---|---|
| ![Starting up](gallery/1.png) | ![Blank node canvas](gallery/2.png) |
| *Launch and initialize the workspace* | *A clean canvas ready for composition* |
| ![Recents Menu](gallery/3.png) | ![Tinkering with Nodes](gallery/4.png) |
| *Recents panel — jump back in fast* | *Manipulate nodes interactively* |
| ![Searching for elements](gallery/5.png) | ![GitHub snippets panel](gallery/6.png) |
| *Real-time element search* | *GitHub snippets at your fingertips* |
| ![Selecting a GitHub snippet](gallery/7.png) | ![Loading the snippet](gallery/8.png) |
| *Selecting a snippet from a cloned repo* | *Loading external code into the workspace* |
| ![Quickly previewing](gallery/9.png) | ![Rendered output](gallery/10.png) |
| *Instant preview for fast iteration* | *Final rendered output in the node canvas* |

---

## 📚 Documentation

| File | Contents |
|------|----------|
| `docs/explain_system.md` | Explain Panel architecture and extension guide |
| `docs/history_system.md` | History data model, checkpoint API, and history-powered explanations |
| `docs/mcp_commands.md` | Full MCP command reference — every command, payload, and sample response |
| `docs/README.md` | Architectural overview, project structure, and design decisions |

---

## 💬 Community

Discuss the project on Reddit: [r/manim — I built a node-based Manim IDE with AI assistance](https://www.reddit.com/r/manim/comments/1qck0ji/i_built_a_nodebased_manim_ide_with_ai_assistance/)

---

<p align="center">Made with ❤️💚💙 by Soumalya · <a href="https://github.com/pro-grammer-SD">@pro-grammer-SD</a></p>
