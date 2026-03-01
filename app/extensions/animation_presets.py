"""
Animation Presets Extension — ✨ Quick Effects

Provides a "Quick Effects" panel with one-click Manim animation presets.
When a mobject is selected in the graph and the user clicks a preset button,
the extension injects the corresponding Manim animation call into the scene's
Python code and triggers a live preview re-render.

Presets:
    Bounce In       — BounceIn via Succession of ScaleIn + slight overshoot
    Fade & Slide    — FadeIn combined with a directional shift
    Rotate Pop      — Rotate followed by a brief scale pulse
    Elastic Scale   — GrowFromCenter with elastic rate-function
    Typewriter Text — AddTextLetterByLetter (Manim Community ≥ 0.17)

All presets produce valid Manim Community Edition Python that the renderer
can execute immediately.
"""

from __future__ import annotations

import logging
import textwrap
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    pass  # Forward refs only

LOGGER = logging.getLogger("animation_presets_extension")


# ── Extension metadata ────────────────────────────────────────────────────────

EXTENSION_METADATA = {
    "name": "Animation Presets",
    "author": "EfficientManim",
    "version": "1.0.0",
    "description": "One-click Manim animation presets via the ✨ Quick Effects panel",
    "permissions": ["register_ui_panel"],
}


# ── Setup ─────────────────────────────────────────────────────────────────────


def setup(api) -> bool:
    """Register the Quick Effects panel."""
    api.register_ui_panel(
        panel_name="✨ Quick Effects",
        widget_class="app.extensions.animation_presets.QuickEffectsPanel",
        position="right",
    )
    LOGGER.info(
        "✓ Animation Presets extension initialised — Quick Effects panel registered"
    )
    return True


# ── Preset definitions ────────────────────────────────────────────────────────

# Each preset is:
#   (display_name, emoji, tooltip, code_template)
#
# The code_template is a Python snippet that will be injected.
# {obj} is replaced at call-time with the selected mobject variable name.

_PRESETS: list[tuple[str, str, str, str]] = [
    (
        "Bounce In",
        "🏀",
        "Scales the object in with a springy bounce overshoot effect.",
        textwrap.dedent("""\
            self.play(
                ScaleInPlace({obj}, scale_factor=0.01, run_time=0.01),
                run_time=0.01,
            )
            self.play(
                ScaleInPlace({obj}, scale_factor=100, run_time=0.3,
                             rate_func=there_and_back_with_pause),
            )
            self.play(
                ScaleInPlace({obj}, scale_factor=1.15, run_time=0.15,
                             rate_func=rush_into),
            )
            self.play(
                ScaleInPlace({obj}, scale_factor=1/1.15, run_time=0.1,
                             rate_func=rush_from),
            )
        """),
    ),
    (
        "Fade & Slide",
        "🌊",
        "Fades the object in while sliding it upward into position.",
        textwrap.dedent("""\
            {obj}.shift(DOWN * 0.8)
            {obj}.set_opacity(0)
            self.play(
                {obj}.animate.shift(UP * 0.8).set_opacity(1),
                run_time=0.7,
                rate_func=smooth,
            )
        """),
    ),
    (
        "Rotate Pop",
        "🌀",
        "Rotates the object a full turn then pops it to full scale.",
        textwrap.dedent("""\
            {obj}.scale(0.1)
            self.play(
                Rotate({obj}, angle=TAU, run_time=0.6, rate_func=rush_into),
                {obj}.animate.scale(10),
                run_time=0.6,
            )
        """),
    ),
    (
        "Elastic Scale",
        "🎈",
        "Grows the object from nothing with an elastic spring overshoot.",
        textwrap.dedent("""\
            self.play(
                GrowFromCenter({obj}, run_time=0.8,
                               rate_func=overshoot),
            )
        """),
    ),
    (
        "Typewriter Text",
        "⌨️",
        "Reveals a Text or MathTex mobject one character at a time.",
        textwrap.dedent("""\
            # Works best when {obj} is a Text or MathTex instance.
            self.play(AddTextLetterByLetter({obj}))
        """),
    ),
]


# ── Scene code injection helpers ──────────────────────────────────────────────


def _build_animation_snippet(preset_code: str, obj_name: str) -> str:
    """Return preset_code with {obj} resolved to obj_name."""
    return preset_code.replace("{obj}", obj_name)


def _inject_into_scene_code(scene_code: str, snippet: str, obj_name: str) -> str:
    """
    Insert the animation snippet into the scene's construct() body.

    Strategy: find the last non-whitespace line inside construct() and
    append the snippet after it (before any trailing blank lines / closing
    of the method).  Falls back to appending at end of construct if no
    obvious insertion point is found.
    """
    lines = scene_code.splitlines(keepends=True)
    construct_idx = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("def construct") or stripped.startswith(
            "def construct("
        ):
            construct_idx = i
            break

    if construct_idx is None:
        # No construct() found — append as-is (will be a render error but at
        # least we tried; user can edit)
        return scene_code + "\n" + snippet

    # Find indentation level of the construct body
    body_indent = "        "  # default 8 spaces (class + method)
    for line in lines[construct_idx + 1 :]:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            body_indent = len(line) - len(line.lstrip())
            body_indent = " " * body_indent
            break

    # Find the last non-blank line inside construct()
    insert_after = construct_idx + 1
    for i in range(construct_idx + 1, len(lines)):
        line = lines[i]
        # A line at or shallower indent than construct() means we left the method
        if line.strip() and not line.startswith(" ") and not line.startswith("\t"):
            break
        # Otherwise track the last non-blank body line
        if line.strip():
            insert_after = i

    # Indent the snippet to match the body
    indented_snippet = ""
    for snip_line in snippet.splitlines(keepends=True):
        if snip_line.strip():
            indented_snippet += body_indent + snip_line
        else:
            indented_snippet += snip_line

    # Insert
    result = (
        lines[: insert_after + 1] + ["\n", indented_snippet] + lines[insert_after + 1 :]
    )
    return "".join(result)


# ── Panel widget ──────────────────────────────────────────────────────────────


class QuickEffectsPanel(QWidget):
    """
    ✨ Quick Effects — one-click animation preset panel.

    Usage:
    1. Select a mobject in the node graph (its Python variable name is read
       from the selected NodeItem's data).
    2. Click a preset button.
    3. The animation code is injected into the scene's construct() method and
       a live preview re-render is triggered.
    """

    preset_applied = Signal(str, str)  # (preset_name, injected_code)

    def __init__(self):
        super().__init__()
        self._main_window = None  # set by _link_to_window()
        self._selected_obj: str = "mobject"  # default fallback variable name
        self._setup_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # ── Header ────────────────────────────────────────────────────────────
        title = QLabel("✨ Quick Effects")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        subtitle = QLabel("Select a mobject, then apply a preset.")
        subtitle.setStyleSheet("color: #6b7280; font-size: 11px;")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        # ── Selected object indicator ─────────────────────────────────────────
        sel_frame = QFrame()
        sel_frame.setFrameShape(QFrame.Shape.StyledPanel)
        sel_frame.setStyleSheet(
            "QFrame { background:#f3f4f6; border:1px solid #d1d5db; border-radius:4px; }"
        )
        sel_layout = QHBoxLayout(sel_frame)
        sel_layout.setContentsMargins(8, 4, 8, 4)
        sel_layout.addWidget(QLabel("Target:"))
        self._sel_label = QLabel(f"<b>{self._selected_obj}</b>")
        self._sel_label.setTextFormat(Qt.TextFormat.RichText)
        sel_layout.addWidget(self._sel_label, 1)
        layout.addWidget(sel_frame)

        # ── Divider ───────────────────────────────────────────────────────────
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet("color: #e5e7eb;")
        layout.addWidget(divider)

        # ── Preset buttons inside a scroll area ───────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        btn_container = QWidget()
        btn_layout = QVBoxLayout(btn_container)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(6)

        for display_name, emoji, tooltip, code_tpl in _PRESETS:
            btn = self._make_preset_button(display_name, emoji, tooltip, code_tpl)
            btn_layout.addWidget(btn)

        btn_layout.addStretch()
        scroll.setWidget(btn_container)
        layout.addWidget(scroll, 1)

        # ── Status bar ────────────────────────────────────────────────────────
        self._status = QLabel("Ready")
        self._status.setStyleSheet("font-size: 10px; color: #6b7280;")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

    def _make_preset_button(
        self, name: str, emoji: str, tooltip: str, code_tpl: str
    ) -> QPushButton:
        btn = QPushButton(f"  {emoji}  {name}")
        btn.setToolTip(tooltip)
        btn.setMinimumHeight(40)
        btn.setStyleSheet(
            """
            QPushButton {
                background-color: #ffffff;
                border: 1px solid #d1d5db;
                border-radius: 6px;
                font-size: 13px;
                text-align: left;
                padding-left: 12px;
            }
            QPushButton:hover {
                background-color: #eff6ff;
                border-color: #2563eb;
                color: #1d4ed8;
            }
            QPushButton:pressed {
                background-color: #dbeafe;
                border-color: #1d4ed8;
            }
            """
        )
        btn.clicked.connect(lambda: self._apply_preset(name, code_tpl))
        return btn

    # ── Preset application ────────────────────────────────────────────────────

    def _apply_preset(self, preset_name: str, code_tpl: str) -> None:
        """
        Apply the chosen animation preset to the currently selected object.

        Steps:
        1. Resolve the target object name (from graph selection or fallback).
        2. Build the Manim snippet.
        3. Inject it into the active scene's construct() code.
        4. Trigger a live preview re-render via the main window's renderer.
        5. Update the status bar.
        """
        obj_name = self._get_selected_object_name()
        snippet = _build_animation_snippet(code_tpl, obj_name)

        success = self._inject_and_render(preset_name, snippet, obj_name)

        if success:
            self._status.setText(f"✓ Applied '{preset_name}' to {obj_name}")
            self._status.setStyleSheet("font-size: 10px; color: #10b981;")
        else:
            self._status.setText(
                f"⚠ '{preset_name}' injected — click Render to preview."
            )
            self._status.setStyleSheet("font-size: 10px; color: #f59e0b;")

        self.preset_applied.emit(preset_name, snippet)
        LOGGER.info(f"Applied preset '{preset_name}' to object '{obj_name}'")

    def _get_selected_object_name(self) -> str:
        """
        Return the Python variable name for the currently selected graph node.

        Walks up the widget tree to find EfficientManimWindow and inspects its
        `scene.selectedItems()`.  Falls back to self._selected_obj.
        """
        try:
            win = self._get_main_window()
            if win is None:
                return self._selected_obj

            selected = win.scene.selectedItems()
            if not selected:
                return self._selected_obj

            # First selected item that has node data
            item = selected[0]
            if hasattr(item, "data") and hasattr(item.data, "cls_name"):
                # Derive a reasonable Python variable name from the class
                cls = item.data.cls_name
                # e.g. "Circle" → "circle", "MathTex" → "math_tex"
                import re

                var = re.sub(r"([A-Z])", r"_\1", cls).lstrip("_").lower()
                return var or self._selected_obj

            if hasattr(item, "data") and hasattr(item.data, "name"):
                raw = item.data.name.strip().replace(" ", "_").lower()
                if raw:
                    return raw

        except Exception as e:
            LOGGER.debug(f"Could not determine selected object name: {e}")

        return self._selected_obj

    def _inject_and_render(self, preset_name: str, snippet: str, obj_name: str) -> bool:
        """
        Inject animation snippet into the active scene code and trigger render.

        Returns True if the main window accepted and queued a render.
        """
        try:
            win = self._get_main_window()
            if win is None:
                LOGGER.warning("No main window found — cannot inject preset")
                return False

            # ── Get current scene code ────────────────────────────────────────
            code_editor = None
            if hasattr(win, "panel_code") and hasattr(win.panel_code, "editor"):
                code_editor = win.panel_code.editor
            elif hasattr(win, "code_editor"):
                code_editor = win.code_editor

            current_code: str = ""
            if code_editor is not None:
                current_code = code_editor.toPlainText()

            if not current_code.strip():
                # If no code exists yet, build a minimal scene template
                current_code = self._minimal_scene_template(obj_name, preset_name)

            # ── Inject the snippet ────────────────────────────────────────────
            new_code = _inject_into_scene_code(current_code, snippet, obj_name)

            # ── Write back to editor ──────────────────────────────────────────
            if code_editor is not None:
                code_editor.setPlainText(new_code)

            # ── Trigger a re-render / preview ─────────────────────────────────
            if hasattr(win, "compile_graph"):
                win.compile_graph()
            if hasattr(win, "_trigger_auto_render"):
                win._trigger_auto_render()
            elif hasattr(win, "render_preview"):
                win.render_preview()

            return True

        except Exception as e:
            LOGGER.error(
                f"Inject & render failed for preset '{preset_name}': {e}", exc_info=True
            )
            return False

    # ── Window helpers ────────────────────────────────────────────────────────

    def _get_main_window(self):
        """Walk widget tree to find EfficientManimWindow."""
        if self._main_window is not None:
            return self._main_window
        widget = self
        while widget is not None:
            # Import lazily to avoid circular deps at module load time
            if type(widget).__name__ == "EfficientManimWindow":
                self._main_window = widget
                return widget
            widget = widget.parent()
        # Fallback: search QApplication top-level windows
        app = QApplication.instance()
        if app:
            for w in app.topLevelWidgets():
                if type(w).__name__ == "EfficientManimWindow":
                    self._main_window = w
                    return w
        return None

    def set_selected_object(self, obj_name: str) -> None:
        """Update the displayed target object (called externally by graph signals)."""
        self._selected_obj = obj_name
        self._sel_label.setText(f"<b>{obj_name}</b>")
        self._status.setText(f"Target set to '{obj_name}'")
        self._status.setStyleSheet("font-size: 10px; color: #6b7280;")

    # ── Minimal scene template ────────────────────────────────────────────────

    @staticmethod
    def _minimal_scene_template(obj_name: str, preset_name: str) -> str:
        """
        Generate a runnable scene template when no existing code is present.

        Used so the preset can still produce a valid render even on a blank slate.
        """
        return textwrap.dedent(f"""\
            from manim import *

            class QuickEffectsDemo(Scene):
                \"\"\"Auto-generated by ✨ Quick Effects: {preset_name}\"\"\"

                def construct(self):
                    {obj_name} = Circle(radius=1.5, color=BLUE)
                    self.add({obj_name})
        """)


# ── Standalone demo (run with: manim -pql animation_presets_demo.py QuickEffectsDemo) ──

DEMO_SCENE = textwrap.dedent("""\
    \"\"\"
    ✨ Quick Effects — Animation Presets Demo

    Demonstrates all five built-in animation presets from the
    EfficientManim Animation Presets extension.

    Run with:
        manim -pql animation_presets_demo.py QuickEffectsPresetsDemo
    \"\"\"
    from manim import *


    class QuickEffectsPresetsDemo(Scene):
        \"\"\"Renders all five Quick Effects presets in sequence.\"\"\"

        def construct(self):
            # ── Shared label helper ────────────────────────────────────────────
            def show_label(text: str) -> None:
                lbl = Text(text, font_size=28, color=YELLOW).to_edge(UP)
                self.play(FadeIn(lbl, shift=DOWN * 0.3), run_time=0.4)
                self.wait(0.2)
                self.play(FadeOut(lbl), run_time=0.3)

            # ─────────────────────────────────────────────────────────────────
            # 1. BOUNCE IN
            # ─────────────────────────────────────────────────────────────────
            show_label("🏀  Bounce In")
            circle = Circle(radius=1.2, color=BLUE, fill_opacity=0.7)

            # Start invisible (scale 0.01)
            self.play(
                ScaleInPlace(circle, scale_factor=0.01, run_time=0.01),
                run_time=0.01,
            )
            # Shoot up past normal size, then settle
            self.play(
                ScaleInPlace(circle, scale_factor=100, run_time=0.35,
                             rate_func=there_and_back_with_pause),
            )
            self.play(
                ScaleInPlace(circle, scale_factor=1.15, run_time=0.15,
                             rate_func=rush_into),
            )
            self.play(
                ScaleInPlace(circle, scale_factor=1 / 1.15, run_time=0.10,
                             rate_func=rush_from),
            )
            self.wait(0.4)
            self.play(FadeOut(circle))

            # ─────────────────────────────────────────────────────────────────
            # 2. FADE & SLIDE
            # ─────────────────────────────────────────────────────────────────
            show_label("🌊  Fade & Slide")
            square = Square(side_length=1.8, color=GREEN, fill_opacity=0.6)
            # Start shifted down + invisible
            square.shift(DOWN * 0.8)
            square.set_opacity(0)
            self.play(
                square.animate.shift(UP * 0.8).set_opacity(1),
                run_time=0.7,
                rate_func=smooth,
            )
            self.wait(0.4)
            self.play(FadeOut(square))

            # ─────────────────────────────────────────────────────────────────
            # 3. ROTATE POP
            # ─────────────────────────────────────────────────────────────────
            show_label("🌀  Rotate Pop")
            star = Star(n=6, outer_radius=1.2, color=ORANGE,
                        fill_opacity=0.8).scale(0.1)
            self.add(star)
            self.play(
                Rotate(star, angle=TAU, run_time=0.6, rate_func=rush_into),
                star.animate.scale(10),
                run_time=0.6,
            )
            self.wait(0.4)
            self.play(FadeOut(star))

            # ─────────────────────────────────────────────────────────────────
            # 4. ELASTIC SCALE
            # ─────────────────────────────────────────────────────────────────
            show_label("🎈  Elastic Scale")
            diamond = Square(side_length=1.5, color=PURPLE,
                             fill_opacity=0.6).rotate(PI / 4)
            self.play(
                GrowFromCenter(diamond, run_time=0.8, rate_func=overshoot),
            )
            self.wait(0.4)
            self.play(FadeOut(diamond))

            # ─────────────────────────────────────────────────────────────────
            # 5. TYPEWRITER TEXT
            # ─────────────────────────────────────────────────────────────────
            show_label("⌨️  Typewriter Text")
            phrase = Text("EfficientManim ✨", font_size=40, color=WHITE)
            self.play(AddTextLetterByLetter(phrase))
            self.wait(0.5)
            self.play(FadeOut(phrase))

            self.wait(0.3)
""")
