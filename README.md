# 🌿 EfficientManim — The Ultimate Node-Based Manim IDE

![Icon](icon/icon.ico)

**🌈 Create mathematical animations visually with the power of Python, AI, and Production-Grade Type Safety.**

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
- Groups shown in expandable tree view

### 🤖 Gemini AI Code Generation
- Describe animations in plain English — AI generates Manim code
- AI code parsed into editable nodes with correct wiring
- Streaming responses with real-time feedback

### 🎙️ AI-Powered Voiceover Studio
- Gemini TTS Integration with multi-voice support (Zephyr, Puck, Fenrir, etc.)
- Auto-sync animation duration to audio length

### 🐙 GitHub Snippet Loader
- Clone any GitHub repository into `~/.efficientmanim/snippets/`
- Browse `.py` files; double-click to load into AI panel as snippet
- All cloned repos automatically repopulated on startup

### ⭐ Recents Panel
- Top-5 most-used Mobjects + top-5 most-used Animations tracked per session
- Persisted to `~/.efficientmanim/usage.json`
- Double-click to instantly add to canvas

### ⌨️ Editable Keybindings
- **Help → Edit Keybindings…** opens the keybindings editor
- Double-click any shortcut to change it
- Duplicate detection; changes persisted to QSettings

### 📁 Editable Project Name
- Project name textbox in the top-right corner of the window
- Rename your `.efp` file without leaving the editor

### 📦 Portable Project Format (.efp)
- Bundled images, sounds, and videos included
- ZIP-based, cross-platform, easy to share

### 🌓 Dark / Light Themes
- Full dark mode with no white bleed-through
- Switch via the theme button in the corner or Settings
- QSS-based styling with ColorToken system

### 🎨 Manim Class Browser
- 60+ Manim classes in 8 categories (Geometry, Text, Graphs, 3D, Animations, etc.)
- Search bar filters in real-time
- Double-click or drag to add node to canvas

### 🎬 Professional Video Rendering
- Full scene export to MP4/WebM
- Up to 4K resolution, 15–60 FPS, quality presets

---

## 🏠 Home Screen

Run `python home.py` to see the home screen with recent projects.  
Run `python main.py` to skip the home screen and open the editor directly.

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

All shortcuts are editable via **Help → Edit Keybindings…**

---

## 🛠️ Prerequisites

1. **Python 3.10+**
2. **FFmpeg** — for video rendering (must be in PATH)
3. **Git** — optional, for GitHub Snippet Loader
4. **LaTeX** — optional, for local LaTeX rendering

## 📦 Installation

```bash
git clone https://github.com/pro-grammer-SD/EfficientManim.git
cd EfficientManim
pip install -r requirements.txt
python home.py    # Full experience with home screen
# or
python main.py    # Open editor directly
```

Manual install:
```bash
pip install manim PySide6 google-genai pydub requests numpy
```

---

## 📸 Screenshots

![Starting up](gallery/1.png)
![Inserting elements](gallery/2.png)
![Customizing animations](gallery/3.png)
![Rendered output](gallery/4.png)

---

Made with lots of ❤️💚💙 by Soumalya a.k.a. @pro-grammer-SD

Discussions: https://www.reddit.com/r/manim/comments/1qck0ji/i_built_a_nodebased_manim_ide_with_ai_assistance/
