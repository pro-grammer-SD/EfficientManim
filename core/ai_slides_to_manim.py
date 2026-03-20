from __future__ import annotations
# -*- coding: utf-8 -*-

import re
from typing import Any


def sanitize_latex(eq: str) -> str:
    s = "" if eq is None else str(eq)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace('"""', "").replace("'''", "")
    s = s.strip()
    if s.startswith("$$") and s.endswith("$$") and len(s) >= 4:
        s = s[2:-2].strip()
    elif s.startswith("$") and s.endswith("$") and len(s) >= 2:
        s = s[1:-1].strip()
    s = s.replace("\\[", "").replace("\\]", "")
    s = s.replace("\n", " ")
    s = re.sub(r"\\begin\s*\{[^}]*\}", "", s)
    s = re.sub(r"\\end\s*\{[^}]*\}", "", s)
    s = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\s*\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathbf\s*\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\displaystyle\b", "", s)
    s = re.sub(r"\\left\b", "", s)
    s = re.sub(r"\\right\b", "", s)
    s = s.replace("\\,", " ")
    s = s.replace("\\;", " ")
    s = s.replace("\\:", " ")
    s = s.replace("\\!", " ")
    s = s.replace("\\times", "\\cdot")
    s = s.replace("\\ldots", "\\cdots")
    s = s.replace("\\dfrac", "\\frac")
    s = s.replace("\\tfrac", "\\frac")
    s = re.sub(r"\\\\+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    while s.endswith("\\"):
        s = s[:-1].rstrip()
    return s


def _is_safe_latex(eq: str) -> bool:
    if not eq:
        return False
    if len(eq) > 200:
        return False
    lowered = eq.lower()
    banned_tokens = [
        "\\begin",
        "\\end",
        "\\text",
        "\\label",
        "\\tag",
        "\\verb",
        "\\href",
        "\\url",
        "\\include",
        "\\input",
        "\\write",
        "\\def",
        "\\newcommand",
        "\\newenvironment",
        "\\usepackage",
        "\\require",
        "\\matrix",
        "\\array",
        "\\align",
        "\\cases",
    ]
    for token in banned_tokens:
        if token in lowered:
            return False
    if "$" in eq or "\n" in eq:
        return False
    if eq.count("{") != eq.count("}"):
        return False
    if eq.count("(") != eq.count(")"):
        return False
    if eq.count("[") != eq.count("]"):
        return False

    allowed_commands = {
        "frac",
        "cdot",
        "cdots",
        "binom",
        "sqrt",
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "zeta",
        "eta",
        "theta",
        "iota",
        "kappa",
        "lambda",
        "mu",
        "nu",
        "xi",
        "pi",
        "rho",
        "sigma",
        "tau",
        "phi",
        "chi",
        "psi",
        "omega",
        "Gamma",
        "Delta",
        "Theta",
        "Lambda",
        "Xi",
        "Pi",
        "Sigma",
        "Phi",
        "Psi",
        "Omega",
    }
    commands = re.findall(r"\\[A-Za-z]+", eq)
    for cmd in commands:
        if cmd[1:] not in allowed_commands:
            return False
    if re.search(r"\\[^A-Za-z]", eq):
        return False

    allowed_chars = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 +-*/=()[]{}.,:;|!<>^_"
    )
    for ch in eq:
        if ch == "\\":
            continue
        if ch not in allowed_chars:
            return False

    return True


def _latex_raw(text: str) -> str:
    safe = text.replace("\r\n", "\n").replace("\r", "\n")
    safe = safe.replace("\n", " ")
    safe = safe.replace('"', '\\"')
    safe = safe.rstrip("\\")
    return f'r"{safe}"'


def _py_str(text: str) -> str:
    return repr(text)


def _equation_code(eq: str, font_size: int = 34) -> str:
    sanitized = sanitize_latex(eq)
    if _is_safe_latex(sanitized):
        return f"MathTex({_latex_raw(sanitized)}, font_size={font_size})"
    display = f"Equation: {sanitized}".strip()
    if not display:
        display = "Equation"
    return f"Text({_py_str(display)}, font_size={font_size})"


def _normalize_derivation(deriv: Any) -> list[dict]:
    if isinstance(deriv, dict):
        deriv = [deriv]
    if not isinstance(deriv, list):
        return []
    steps = []
    for item in deriv:
        if isinstance(item, dict):
            eq = str(item.get("equation") or "").strip()
            exp = str(item.get("explanation") or "").strip()
            if eq:
                steps.append({"equation": eq, "explanation": exp})
        else:
            eq = str(item).strip()
            if eq:
                steps.append({"equation": eq, "explanation": ""})
    return steps


class SlidesToManim:
    """Generates runnable Manim code from structured slide JSON."""

    @staticmethod
    def generate_code(
        slide_deck: dict | list, scene_name: str = "GeneratedScene"
    ) -> str:
        if isinstance(slide_deck, dict):
            slides = slide_deck.get("slides") or []
        else:
            slides = slide_deck

        lines: list[str] = []
        lines.append("from manim import *")
        lines.append("")
        lines.append("class GeneratedScene(Scene):")
        lines.append("    def construct(self):")
        lines.append('        self.camera.background_color = "#ffffff"')
        lines.append("        self.wait(0.1)")
        lines.append("")

        for idx, slide in enumerate(slides, start=1):
            if not isinstance(slide, dict):
                continue
            title = str(slide.get("title") or "").strip()
            bullets = slide.get("bullets") or []
            equations = slide.get("equations") or []
            deriv = slide.get("derivation_steps") or []

            bullets = [str(b).strip() for b in bullets if str(b).strip()]
            equations = [str(e).strip() for e in equations if str(e).strip()]
            deriv = _normalize_derivation(deriv)

            lines.append(f"        # Slide {idx}")

            slide_objects: list[str] = []

            if title:
                lines.append(f"        title = Text({_py_str(title)}, font_size=46)")
                lines.append("        title.to_edge(UP)")
                lines.append("        self.play(FadeIn(title))")
                slide_objects.append("title")

            if bullets:
                bullet_items = [
                    f"Text({_py_str('- ' + b)}, font_size=30)" for b in bullets
                ]
                lines.append(f"        bullets = VGroup({', '.join(bullet_items)})")
                lines.append(
                    "        bullets.arrange(DOWN, aligned_edge=LEFT, buff=0.25)"
                )
                if title:
                    lines.append(
                        "        bullets.next_to(title, DOWN, aligned_edge=LEFT, buff=0.5)"
                    )
                else:
                    lines.append("        bullets.to_edge(LEFT)")
                lines.append("        self.play(FadeIn(bullets))")
                slide_objects.append("bullets")

            if equations:
                eq_items = [_equation_code(eq, font_size=34) for eq in equations]
                lines.append(f"        equations = VGroup({', '.join(eq_items)})")
                lines.append("        equations.arrange(DOWN, buff=0.4)")
                if bullets:
                    lines.append("        equations.next_to(bullets, RIGHT, buff=1.0)")
                elif title:
                    lines.append("        equations.next_to(title, DOWN, buff=0.6)")
                lines.append("        equations.to_edge(RIGHT)")
                lines.append("        self.play(FadeIn(equations))")
                slide_objects.append("equations")

            if deriv:
                deriv_items = []
                for step in deriv:
                    eq = step.get("equation") or ""
                    if str(eq).strip():
                        deriv_items.append(_equation_code(str(eq), font_size=34))
                if deriv_items:
                    lines.append(f"        deriv_steps = [{', '.join(deriv_items)}]")
                    if title:
                        lines.append(
                            "        deriv_steps[0].next_to(title, DOWN, buff=0.8)"
                        )
                    elif bullets:
                        lines.append(
                            "        deriv_steps[0].next_to(bullets, DOWN, buff=0.8)"
                        )
                    else:
                        lines.append("        deriv_steps[0].move_to(ORIGIN)")
                    lines.append("        current_step = deriv_steps[0]")
                    lines.append("        self.play(FadeIn(current_step))")
                    lines.append("        for next_step in deriv_steps[1:]:")
                    lines.append("            next_step.move_to(current_step)")
                    lines.append(
                        "            self.play(FadeOut(current_step), FadeIn(next_step))"
                    )
                    lines.append("            current_step = next_step")
                    slide_objects.append("current_step")

            lines.append("        self.wait(0.5)")

            if slide_objects:
                lines.append(
                    f"        self.play(FadeOut(VGroup({', '.join(slide_objects)})))"
                )
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"
