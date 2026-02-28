# EfficientManim v2.0.1

A visual, node-based Manim IDE with AI-powered code generation via Google Gemini.

---

## Features

- **Node Graph Canvas** — Visually compose Manim scenes by connecting Mobject and Animation nodes
- **AI Code Generation** — Generate Manim code from natural language prompts using Gemini AI
- **Live Mobject Preview** — Automatically renders a PNG preview when you select and modify a node
- **Video Rendering** — Render full scenes to MP4/WebM via Manim with one click
- **Manim Class Browser** — Searchable palette of all built-in Manim classes
- **Code Snippet Library** — Load reusable templates directly into the editor
- **GitHub Snippets** — Clone and browse `.py` files from any GitHub repository
- **LaTeX Studio** — Live LaTeX preview with API-based rendering
- **AI Voiceover (TTS)** — Generate narration audio via Gemini TTS and sync it to animation nodes
- **VGroups** — Group Mobjects visually; VGroup code is auto-generated
- **Usage Tracking** — Recents pane shows your most-used Mobjects and Animations
- **Custom Keybindings** — Every action is remappable

---

## Requirements

- Python 3.10+
- [Manim Community Edition](https://www.manim.community/)
- PySide6
- `google-genai` (for AI features)
- `pydub` (optional, for audio duration sync)

Install dependencies:

```bash
pip install manim PySide6 google-genai pydub
```

---

## Running the Application

```bash
# Launch home screen
python home.py

# Or launch the editor directly
python main.py
```

---

## AI Configuration

1. Go to **File → Settings**
2. Enter your **Google Gemini API Key**
3. Select the desired code generation and TTS models

---

## Rendering

Any valid Manim `Scene` subclass renders correctly. The renderer dynamically detects the scene class in generated code — **no forced renaming, no hidden rewriting**.

When you export code to `.py`, the class name is preserved exactly as defined. When rendering via the Video panel, the correct class is detected and rendered transparently.

---

## Project Files

Projects are saved as `.efp` files (ZIP archives containing graph data, assets, and code).

---

## Themes

EfficientManim uses a clean **light mode** design. No theme switching; one consistent, professional interface.

---

## License

© 2026 Soumalya Das (@pro-grammer-SD)
