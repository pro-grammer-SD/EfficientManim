# EfficientManim

![Icon](icon/icon.ico)

EfficientManim is a node-based visual IDE for building mathematical animations. It combines a drag-and-drop scene graph with AI-assisted tooling, a structured history system, and a learning-first explanation engine.

---

## What You Can Do
- Build Manim scenes visually with nodes, wiring, and live previews.
- Generate animations from natural-language prompts or PDF slides.
- Produce AI voiceovers synced to animation timing.
- Inspect and control everything through MCP commands.

---

## 🧠 Explain This Animation
**What it does**
- Converts the current animation into a clear, student-friendly explanation.
- Works for full scenes, selected nodes, or single animations.

**How it helps**
- Students get intuitive explanations and step-by-step reasoning.
- Teachers can reuse explanations as lesson scripts.
- Beginners learn the “why” behind each visual move.

**Explain Panel workflow**
- Open the panel with the **🧠 Explain** button in the main toolbar.
- Choose **Simple** or **Detailed** mode.
- Regenerate or copy the explanation instantly.

**Simple vs Detailed**
- **Simple**: short, intuition-first summaries.
- **Detailed**: full step breakdown with conceptual reasoning.

---

## 🎓 Learning Mode
**What it is**
Learning Mode automatically generates micro-explanations while students build animations.

**When it triggers**
- Adding axes, graphs, or equations
- Creating transformations
- Adding tangent/slope/area visuals
- Major structural changes
- Checkpoints or large batch additions

**Why it’s useful**
It turns the editor into a live tutor, explaining what just changed and why it matters.

**Default state**
Learning Mode is **OFF** by default to avoid interruptions.

**How to enable**
Settings → **Learning Mode** → Enable Learning Mode.

---

## 👩🏫 Teacher Mode
**What it does**
Teacher Mode generates structured lesson notes from the current scene.

**What you get**
- Concept explanation
- Visual explanation tied to the animation
- Step-by-step teaching script
- Student-friendly notes
- Key takeaways

**Export options**
- Markdown (`.md`)
- Plain text (`.txt`)
- Copy to clipboard

**How to enable**
Settings → **Teacher Mode** → Enable Teacher Mode.

---

## 🕰 History-Powered Explanations
EfficientManim can explain how a scene evolved over time:
- Explain any checkpoint state
- Explain what changed between two checkpoints
- Explain undo/redo actions in student language

These explanations are powered by the same analysis engine as the Explain Panel.

---

## 🤖 MCP Command Support
The explanation system is fully scriptable:
- Trigger explanations programmatically (`explain.scene`, `explain.selected_nodes`)
- Control Learning Mode and Teacher Mode via MCP
- Generate lesson notes through `teacher_mode.generate_lesson`

See `docs/mcp_commands.md` for detailed request/response examples.

---

## Getting Started
Run the app:
```bash
python main.py
```

---

## Installation
```bash
git clone https://github.com/pro-grammer-SD/EfficientManim.git
cd EfficientManim
pip install -r requirements.txt
python main.py
```

Manual install:
```bash
pip install manim PySide6 google-genai pydub requests numpy pdfplumber regex
```

---

## Documentation
- `docs/explain_system.md` — Explain system architecture and extension guide
- `docs/history_system.md` — History model + history-powered explanations
- `docs/mcp_commands.md` — MCP command reference (including Explain, Learning, Teacher)

---

## License
MIT (see `LICENSE`).
