"""
✨ Quick Effects — Animation Presets Demo

Demonstrates all five built-in animation presets from the
EfficientManim Animation Presets extension.

Run with:
    manim -pql animation_presets_demo.py QuickEffectsPresetsDemo
"""

from manim import *


class QuickEffectsPresetsDemo(Scene):
    """Renders all five Quick Effects presets in sequence."""

    def construct(self):

        def show_label(text: str) -> None:
            lbl = Text(text, font_size=28, color=YELLOW).to_edge(UP)
            self.play(FadeIn(lbl, shift=DOWN * 0.3), run_time=0.4)
            self.wait(0.2)
            self.play(FadeOut(lbl), run_time=0.3)

        # ── 1. BOUNCE IN ──────────────────────────────────────────────────────
        show_label("🏀  Bounce In")
        circle = Circle(radius=1.2, color=BLUE, fill_opacity=0.7)
        self.play(
            ScaleInPlace(circle, scale_factor=0.01, run_time=0.01),
            run_time=0.01,
        )
        self.play(
            ScaleInPlace(
                circle,
                scale_factor=100,
                run_time=0.35,
                rate_func=there_and_back_with_pause,
            ),
        )
        self.play(
            ScaleInPlace(circle, scale_factor=1.15, run_time=0.15, rate_func=rush_into),
        )
        self.play(
            ScaleInPlace(
                circle, scale_factor=1 / 1.15, run_time=0.10, rate_func=rush_from
            ),
        )
        self.wait(0.4)
        self.play(FadeOut(circle))

        # ── 2. FADE & SLIDE ───────────────────────────────────────────────────
        show_label("🌊  Fade & Slide")
        square = Square(side_length=1.8, color=GREEN, fill_opacity=0.6)
        square.shift(DOWN * 0.8)
        square.set_opacity(0)
        self.play(
            square.animate.shift(UP * 0.8).set_opacity(1),
            run_time=0.7,
            rate_func=smooth,
        )
        self.wait(0.4)
        self.play(FadeOut(square))

        # ── 3. ROTATE POP ─────────────────────────────────────────────────────
        show_label("🌀  Rotate Pop")
        star = Star(n=6, outer_radius=1.2, color=ORANGE, fill_opacity=0.8).scale(0.1)
        self.add(star)
        self.play(
            Rotate(star, angle=TAU, run_time=0.6, rate_func=rush_into),
            star.animate.scale(10),
            run_time=0.6,
        )
        self.wait(0.4)
        self.play(FadeOut(star))

        # ── 4. ELASTIC SCALE ──────────────────────────────────────────────────
        show_label("🎈  Elastic Scale")
        diamond = Square(side_length=1.5, color=PURPLE, fill_opacity=0.6).rotate(PI / 4)
        self.play(
            GrowFromCenter(diamond, run_time=0.8, rate_func=overshoot),
        )
        self.wait(0.4)
        self.play(FadeOut(diamond))

        # ── 5. TYPEWRITER TEXT ────────────────────────────────────────────────
        show_label("⌨️  Typewriter Text")
        phrase = Text("EfficientManim ✨", font_size=40, color=WHITE)
        self.play(AddTextLetterByLetter(phrase))
        self.wait(0.5)
        self.play(FadeOut(phrase))

        self.wait(0.3)
