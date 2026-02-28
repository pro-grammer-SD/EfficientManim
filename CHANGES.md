# EfficientManim — Changelog

---

## v2.0.1 — Structural Cleanup & Rendering Integrity Update

**Date:** February 28, 2026

This release removes architectural hacks and replaces them with honest, transparent behavior.

---

### Removed: Geist Font System

The Geist custom font has been completely removed.

- All `QFontDatabase` font-loading logic removed
- All Geist font references removed from codebase
- All fallback checks and font utility helpers removed
- `utils.py` cleaned; no font-loading logs remain
- Application uses the system default font cleanly via `QFont()` with point size 10
- No font warnings; no residual references

---

### Removed: Scenes Tab

The **Scenes** tab has been fully removed from the UI.

- `MultiSceneManagerPanel` no longer mounted as a tab
- All signal connections (`scene_switched`, `scene_added`, `scene_deleted`) disconnected from the main window
- Tab entry `"🎬 Scenes"` removed from `QTabWidget`
- No dead imports; no dangling signal handlers in the active UI path
- Application runs cleanly without the Scenes tab

---

### Fixed: Scene Class Renaming — Removed

**Previously**, the graph compiler hardcoded all generated code to use:

```python
class EfficientScene(Scene):
```

This silently renamed every AI-generated or user-defined class to `EfficientScene`, which was a trust violation.

**Now**, the compiler generates a neutral class name:

```python
class GeneratedScene(Scene):
```

And the video renderer uses `detect_scene_class()` to dynamically identify the actual scene class in any code — whether AI-generated or user-written.

**Behavior now:**

- Any class inheriting from `Scene` renders correctly
- AI-generated class names (e.g., `PythagoreanTheorem`, `MyScene`) are **never renamed**
- No injected wrapper classes
- No hidden code mutation before rendering
- Rendering is transparent and deterministic

---

### Fixed: Rendering System

The `VideoRenderWorker` now:

- Accepts `scene_class` as an explicit parameter
- Passes the detected class name directly to the Manim CLI
- Never assumes `EfficientScene` or any fixed name

The `detect_scene_class(code)` function scans for `class X(Scene):` patterns and returns the actual class name, falling back to `"Scene"` only if none is found.

---

### Fixed: SettingsDialog

- Removed broken `self.theme.currentText()` reference (widget was never created)
- Settings dialog now saves cleanly without attempting theme switching
- Light-mode-only mode confirmed stable

---

### Fixed: Theme System

- Removed undefined `ThemeMode` type annotation from `_on_theme_changed()`
- Theme change handler signature cleaned up to be safe and forward-compatible

---

### Code Snippet Templates

All built-in snippet templates updated:

- `class EfficientScene(Scene)` → `class MyScene(Scene)` in all SnippetLibrary entries
- Templates are now neutral and don't impose a naming convention on users

---

## v2.0.1 — Major Feature Release

Initial production release including:

- Node graph canvas with visual wiring
- AI code generation via Google Gemini
- Live Mobject preview rendering
- Video export to MP4/WebM
- Manim class browser
- Code snippet library
- GitHub snippet loader
- LaTeX live preview
- AI voiceover (TTS) with node sync
- VGroup creation and code generation
- Usage tracking and Recents pane
- Custom keybindings
- Light mode design system
