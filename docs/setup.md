# 🛠️ EfficientManim — Setup Guide

## System Requirements

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | Required for match statements and modern type hints |
| FFmpeg | Any recent | Must be on system `PATH`; required for Manim video rendering |
| Git | Any | Required for GitHub Snippet Loader |
| LaTeX | TeX Live / MiKTeX | Required for `MathTex` / `Tex` nodes |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/programmersd21/EfficientManim.git
cd EfficientManim
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs: `manim`, `PySide6`, `google-genai`, `pydub`, `requests`, `numpy`.

### 4. Install system dependencies

**FFmpeg** (required for video rendering and audio):
- **Windows:** Download from https://ffmpeg.org/download.html and add the `bin/` folder to `PATH`.
- **macOS:** `brew install ffmpeg`
- **Linux:** `sudo apt install ffmpeg`

**LaTeX** (required for `MathTex`/`Tex` nodes):
- **Windows:** Install [MiKTeX](https://miktex.org/)
- **macOS:** `brew install --cask mactex`
- **Linux:** `sudo apt install texlive-full`

### 5. Configure Gemini API Key

Go to **File → Settings** and paste your [Google AI Studio](https://aistudio.google.com/) API key into the API Key field.

Alternatively, set the environment variable before launching:
```bash
export GEMINI_API_KEY=your_key_here
```

### 6. Launch the application

```bash
python main.py
```

---

## First Launch Walkthrough

1. The home screen shows recent projects. Click **New Project** or just wait — it opens the editor automatically if you run `main.py`.
2. In the editor, expand the **Elements** tab on the right. Double-click `Circle` to add a Mobject node.
3. Double-click `FadeIn` to add an Animation node.
4. Double-click `▶ Play` (at the top of the Elements tab) to add a Play node.
5. Connect: `Circle → FadeIn`, then `FadeIn → Play`.
6. Check the **Code** tab at the bottom — you should see `self.play(FadeIn(m_xxxxxx))`.
7. Click **Render Scene** in the Video tab to produce an MP4.

---

## Common Installation Issues

**`manim` command not found after install**
Add your Python `Scripts` folder (Windows) or `bin` folder (Linux/macOS) to `PATH`.

**PySide6 crashes on startup (Windows)**
Make sure you are not mixing 32-bit and 64-bit Python installations.

**`ModuleNotFoundError: No module named 'google.genai'`**
Run `pip install google-genai>=0.3.0`.

**LaTeX-related Manim errors**
Install the full TeX Live distribution. On Windows, run MiKTeX's package manager and install `standalone`, `amsmath`, and `preview`.

**"No audio data received" from TTS**
Check your Gemini API key. TTS requires the `gemini-2.5-flash-preview-tts` model which is in preview — make sure your key has access.

---

## Environment Variables

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Gemini API key (overrides Settings dialog) |
| `QT_LOGGING_RULES` | Set to `qt.multimedia.*=false` (auto-set by app) |

---

## Platform Notes

**Windows:** The app sets `AppUserModelID` on startup for correct taskbar grouping. Windows Defender may flag `manim` subprocess calls — add your Python/Manim folders to the exclusion list.

**Linux / Wayland:** Use `QT_QPA_PLATFORM=xcb` if you see rendering issues.

**macOS:** If you see font issues with Manim output, install `TeX Live` via Homebrew and ensure `pdflatex` is on `PATH`.
