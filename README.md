# EfficientManim - Production-Ready Version

**The Ultimate Node-Based Manim IDE**  
*Create mathematical animations visually with the power of Python, AI, and Production-Grade Type Safety.*

---

## 🚀 Key Features

### 🎬 Node-Based Visual Workflow
- **Visual Editor:** Drag-and-drop Mobjects and Animations with intuitive wiring
- **Infinite Canvas:** Pan and zoom freely to manage large node graphs
- **Live Preview:** Real-time static previews of individual nodes
- **Smart Connections:** Automatic wire validation and scene synchronization
- **Robust Resource Cleanup:** Proper memory management prevents leaks

### 🎙️ AI-Powered Voiceover Studio
- **Gemini TTS Integration:** Generate realistic voiceovers using Google's Gemini 2.5 models
- **Auto-Sync:** Automatically synchronizes animation duration to audio length
- **Multi-Voice Support:** Choose from diverse voices (Zephyr, Puck, Fenrir, etc.)
- **Stream Processing:** Real-time response from Gemini API with streaming support

### 📦 Portable Project Format (.efp)
- **Bundled Assets:** Images, sounds, and videos automatically included in projects
- **Cross-Platform:** Projects work seamlessly on Windows, Linux, and macOS
- **Smart Path Handling:** Automatic path conversion for asset management
- **ZIP-Based Format:** Easy to share and version control

### 🛡️ Enterprise-Grade Type Safety  
- **Smart Type Parsing:** Automatically distinguishes numeric, color, vector, and asset types
- **Crash Prevention:** Validates all inputs before renderer processing
- **Font Validation:** Safe font initialization with automatic size clamping (8-24pt)
- **Utilities Module:** Comprehensive type validation framework in `utils.py`
- **Color Safety:** Type-safe color conversion and validation system

### 🤖 Gemini AI Code Generation
- **Text-to-Animation:** Describe animations in plain English for AI generation
- **Node Extraction:** AI code automatically parsed into editable nodes
- **Scene Integration:** Seamlessly merge AI-generated content with existing graphs
- **Streaming Responses:** Real-time code generation with visual feedback

### 🎨 Modern Theme System
- **Light & Dark Themes:** Professional themes for both light and dark environments
- **QSS-Based Styling:** 550+ lines of comprehensive stylesheets in `themes.py`
- **ColorToken System:** Centralized color management for consistency
- **Modern UI Elements:** Rounded buttons, smooth transitions, proper contrast ratios
- **Theme Switching:** Dynamically change themes without restarting

### 🎬 Professional Video Rendering
- **Full Scene Export:** Render complete node graphs to MP4/WebM
- **Quality Settings:** Control Resolution (up to 4K), Framerate (15-60 FPS), Render Quality
- **Thread-Safe Rendering:** Background rendering with proper thread management
- **Progress Tracking:** Visual feedback during long render operations

---

## 💻 Recent Production Improvements

### ✅ Critical Fixes Applied (v2.0.0)
1. **Edge Connection Registration** - Wires now properly register with scene graph
2. **Properties Panel Rendering** - All node parameters display correctly  
3. **QFont Validation** - Font point sizes auto-clamped (8-24pt) - Zero font errors
4. **Preview Resource Cleanup** - No memory leaks on node deletion/close
5. **PySide6 API Updates** - Updated to current QMessageBox.StandardButton API

### 🆕 New Type Safety System
- **utils.py Module (200+ lines)** - FontManager, ColorValidator, PointValidator
- **themes.py Module (550+ lines)** - Professional light/dark themes with comprehensive QSS
- **Smart Validation** - All inputs type-checked before processing
- **Font Manager** - Safe font creation with automatic size constraints

---

## ⚙️ System Architecture

### Core Components

**main.py (4651 lines)**
- `EfficientManimWindow`: Main application window
- `GraphScene`: Qt graphics scene with validation
- `NodeItem`: Visual nodes with socket management
- `PropertiesPanel`: Dynamic parameter inspector
- `RenderWorker`: Background preview rendering  
- Threading & resource management

**utils.py (NEW - Type Safety)**
- `FontManager`: Safe QFont creation
- `ColorValidator`: Type-safe colors
- `PointValidator`: 2D point validation
- `StateValidator`: State verification

**themes.py (NEW - Professional Styling)**
- Light and Dark themes
- 500+ lines of QSS styling
- ColorToken system
- Modern component styling
*   **Enabled Checkbox:** Toggle parameters on/off.
*   **Escape Checkbox:** Removes quotes from string values (for variables).

### 3. Voiceover & Assets
*   **Voiceover Tab:** Generate TTS audio and attach it directly to specific animation nodes.
*   **Asset Manager:** Drag & Drop images/sounds. Assets are auto-linked by ID.

---

## ⚙️ Example Workflow

1.  **Import Assets:** Go to the Assets tab and import a `.png` file.
2.  **Add Nodes:** Add an `ImageMobject` and a `FadeIn` animation.
3.  **Voiceover:** 
    *   Go to the **Voiceover** tab.
    *   Type a script and click **Generate Audio**.
    *   Select the `FadeIn` node in the "Sync" dropdown.
    *   The animation will now last exactly as long as the spoken text.
4.  **Connect:** Wire the nodes together.
5.  **Render:** Go to the Video tab and export your video.

---

## ⌨️ Keyboard Shortcuts

### 📁 File Operations
| Action | Shortcut | Description |
| :--- | :--- | :--- |
| **New Project** | `Ctrl` + `N` | Clears the current project. |
| **Open Project** | `Ctrl` + `O` | Open an existing `.efp` project. |
| **Save Project** | `Ctrl` + `S` | Save the current project. |
| **Exit** | `Ctrl` + `Q` | Quit (Prompts to save if modified). |

### ✏️ Editing
| Action | Shortcut | Description |
| :--- | :--- | :--- |
| **Undo** | `Ctrl` + `Z` | Undo the last action. |
| **Redo** | `Ctrl` + `Y` | Redo the last undone action. |
| **Delete** | `Delete` | Delete the selected nodes/wires. |

### 👁️ View & Canvas
| Action | Shortcut | Description |
| :--- | :--- | :--- |
| **Zoom In** | `Ctrl` + `+` | Zoom into the canvas. |
| **Zoom Out** | `Ctrl` + `-` | Zoom out of the canvas. |
| **Mouse Zoom** | `Ctrl` + `Scroll` | Smooth zoom at mouse position. |
| **Pan** | `Middle Mouse` | Drag canvas to pan. |
| **Fit View** | `Ctrl` + `0` | Fit all nodes on screen. |
| **Clear All** | `Ctrl` + `Alt` + `Del` | Deletes **all** nodes and wires. |

---

## 🚀 Getting Started

### 🛠️ Prerequisites

Before running the app, ensure you have the following installed on your system:

1.  **Python 3.10+**: [Download Python](https://www.python.org/downloads/)
2.  **FFmpeg**: Required for video rendering and audio processing.
    *   [Download FFmpeg](https://ffmpeg.org/download.html)
    *   *Important:* Ensure `ffmpeg` is added to your system's PATH variable.
3.  **LaTeX** (Optional): Required only if you want to render LaTeX locally instead of using the online API.
    *   Windows: [MiKTeX](https://miktex.org/)
    *   Mac: [MacTeX](https://www.tug.org/mactex/)
    *   Linux: `texlive-full`

### 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/pro-grammer-SD/EfficientManim.git
    cd EfficientManim
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

    *If you prefer installing manually:*
    ```bash
    pip install manim PySide6 google-genai pydub requests numpy
    ```

3.  **Run the Application:**
    ```bash
    python main.py
    ```
    
## 📸 Screenshots

![Image 1](gallery/1.png "Starting up a basic project from scratch...")
![Image 2](gallery/2.png "Inserting elements and search for new animations...")
![Image 3](gallery/3.png "Getting better with cool animations, now customizing...")
![Image 4](gallery/4.png "Render working, a beautiful animation is generated...")

---

Made with lots of ❤️💚💙 by Soumalya a.k.a. @pro-grammer-SD.

Discussions at this subreddit: https://www.reddit.com/r/manim/comments/1qck0ji/i_built_a_nodebased_manim_ide_with_ai_assistance/
