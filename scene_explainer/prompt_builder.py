from __future__ import annotations
# -*- coding: utf-8 -*-

from typing import List, Tuple

from scene_explainer.explanation_models import SceneAnalysis, HistoryDiff


class PromptBuilder:
    """Constructs high-quality prompts for the explanation engine."""

    def build_explain_prompt(self, analysis: SceneAnalysis, mode: str) -> str:
        mode = (mode or "detailed").strip().lower()
        if mode not in ("simple", "detailed"):
            mode = "detailed"

        focus_lines = self._build_focus_lines(analysis)
        objects_block = self._format_objects(analysis, mode=mode)
        steps_block = self._format_steps(analysis, mode=mode)
        relationships_block = self._format_relationships(analysis)
        hints = self._merge_hints(analysis)

        detail_note = (
            "Use short, intuitive sentences." if mode == "simple" else "Provide a complete, step-by-step explanation."
        )

        return (
            "You are a math teacher explaining a visual animation to a student.\n"
            "Do not mention programming, code, nodes, or internal implementation details.\n"
            "Do not mention any library names.\n"
            "Use plain English and a teaching tone.\n"
            "\n"
            f"Mode: {mode}\n"
            f"Instruction: {detail_note}\n"
            "\n"
            "Return ONLY valid JSON with exactly these keys:\n"
            "concept_explanation, step_by_step, visual_reasoning, simple_explanation, key_takeaways, mode_used.\n"
            "- key_takeaways must be a JSON array of short strings.\n"
            "- mode_used must be either 'simple' or 'detailed'.\n"
            "\n"
            f"Scene name: {analysis.scene_name}\n"
            f"Object count: {analysis.object_count}\n"
            "\n"
            "Objects:\n"
            f"{objects_block}\n"
            "\n"
            "Text elements:\n"
            f"{self._format_list(analysis.text_elements)}\n"
            "\n"
            "Math expressions:\n"
            f"{self._format_list(analysis.math_expressions)}\n"
            "\n"
            "Animation steps:\n"
            f"{steps_block}\n"
            "\n"
            "Object relationships:\n"
            f"{relationships_block}\n"
            "\n"
            "Concept hints:\n"
            f"{self._format_list(hints)}\n"
            "\n"
            "Focus guidance:\n"
            f"{self._format_list(focus_lines)}\n"
        ).strip()

    def build_learning_prompt(self, analysis: SceneAnalysis, what_happened: str) -> str:
        hints = self._merge_hints(analysis)
        return (
            "You are a supportive math tutor.\n"
            "Write a short auto-explanation for a student.\n"
            "Do not mention programming, code, nodes, or library names.\n"
            "Return ONLY valid JSON with keys: what_happened, why_it_matters.\n"
            "Keep it brief: one sentence for what happened, one or two for why it matters.\n"
            "\n"
            f"Event summary: {what_happened}\n"
            "\n"
            "Scene hints:\n"
            f"{self._format_list(hints)}\n"
        ).strip()

    def build_lesson_prompt(self, analysis: SceneAnalysis) -> str:
        hints = self._merge_hints(analysis)
        objects_block = self._format_objects(analysis, mode="detailed")
        steps_block = self._format_steps(analysis, mode="detailed")

        return (
            "You are an expert math teacher creating lesson notes for an animation.\n"
            "Do not mention programming, code, nodes, or library names.\n"
            "Return ONLY valid JSON with keys: lesson_title, concept_explanation, visual_explanation, step_by_step_teaching, student_notes, key_takeaways.\n"
            "- key_takeaways must be a JSON array of 3-6 short bullet strings.\n"
            "\n"
            f"Scene name: {analysis.scene_name}\n"
            f"Object count: {analysis.object_count}\n"
            "Objects:\n"
            f"{objects_block}\n"
            "\n"
            "Animation steps:\n"
            f"{steps_block}\n"
            "\n"
            "Concept hints:\n"
            f"{self._format_list(hints)}\n"
        ).strip()

    def build_history_change_prompt(
        self,
        diff: HistoryDiff,
        before: SceneAnalysis,
        after: SceneAnalysis,
        mode: str,
    ) -> str:
        mode = (mode or "detailed").strip().lower()
        if mode not in ("simple", "detailed"):
            mode = "detailed"
        return (
            "You are a math teacher describing how an animation changed between two checkpoints.\n"
            "Do not mention programming, code, nodes, or library names.\n"
            "Return ONLY valid JSON with keys: concept_change_summary, educational_significance.\n"
            "\n"
            f"Mode: {mode}\n"
            f"From checkpoint: {diff.from_checkpoint}\n"
            f"To checkpoint: {diff.to_checkpoint}\n"
            f"Objects added: {', '.join(diff.objects_added) or 'None'}\n"
            f"Objects removed: {', '.join(diff.objects_removed) or 'None'}\n"
            f"Animations added: {', '.join(diff.animations_added) or 'None'}\n"
            f"Animations removed: {', '.join(diff.animations_removed) or 'None'}\n"
            "\n"
            "Before hints:\n"
            f"{self._format_list(self._merge_hints(before))}\n"
            "After hints:\n"
            f"{self._format_list(self._merge_hints(after))}\n"
        ).strip()

    # ──────────────────────────────────────────────────────────────────
    # Formatting helpers
    # ──────────────────────────────────────────────────────────────────

    def _format_list(self, items: List[str]) -> str:
        if not items:
            return "- None"
        return "\n".join(f"- {i}" for i in items)

    def _format_objects(self, analysis: SceneAnalysis, mode: str) -> str:
        objects = analysis.objects
        if not objects:
            return "- None"
        max_items = 18 if mode == "simple" else 30
        if len(objects) <= max_items:
            return "\n".join(self._format_object_line(o) for o in objects)

        # Summarize large scenes
        counts = {}
        for o in objects:
            counts[o.type] = counts.get(o.type, 0) + 1
        summary = "Summary by type: " + ", ".join(
            f"{k}={v}" for k, v in sorted(counts.items())
        )
        preview = "\n".join(self._format_object_line(o) for o in objects[:max_items])
        return f"- {summary}\n{preview}\n- ... ({len(objects) - max_items} more objects)"

    def _format_object_line(self, obj) -> str:
        label = f" label='{obj.label}'" if obj.label else ""
        return f"- {obj.type} ({obj.role}){label}"

    def _format_steps(self, analysis: SceneAnalysis, mode: str) -> str:
        steps = analysis.animation_steps
        if not steps:
            return "- None"
        max_items = 20 if mode == "simple" else 35
        lines = []
        for step in steps[:max_items]:
            tgt = ", ".join(step.targets) if step.targets else "(no targets)"
            lines.append(
                f"- Step {step.step_index}: {step.animation_type} | targets: {tgt} | duration: {step.duration} | lag: {step.lag_ratio}"
            )
        if len(steps) > max_items:
            lines.append(f"- ... ({len(steps) - max_items} more steps)")
        return "\n".join(lines)

    def _format_relationships(self, analysis: SceneAnalysis) -> str:
        rels = analysis.object_relationships
        if not rels:
            return "- None"
        lines = []
        for r in rels:
            lines.append(f"- {r.from_id} {r.relationship} {r.to_id}")
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────
    # Concept inference and focus guidance
    # ──────────────────────────────────────────────────────────────────

    def _merge_hints(self, analysis: SceneAnalysis) -> List[str]:
        hints = set(analysis.concept_hints)
        inferred = self._infer_additional_hints(analysis)
        for h in inferred:
            hints.add(h)
        return sorted(hints)

    def _infer_additional_hints(self, analysis: SceneAnalysis) -> List[str]:
        hints: List[str] = []
        types = {o.type for o in analysis.objects}
        has_axes = any(t in ("Axes", "NumberPlane", "ThreeDAxes") for t in types)
        has_graph = any("Graph" in t or "Function" in t for t in types)
        has_math = any(t in ("MathTex", "Tex") for t in types)
        has_text_only = bool(analysis.objects) and all(
            t in ("Text", "Paragraph", "MarkupText") for t in types
        )
        has_shapes = any(
            t in ("Circle", "Square", "Rectangle", "Triangle", "Polygon", "Ellipse")
            for t in types
        )
        has_transforms = any(
            step.animation_type in ("Transform", "ReplacementTransform", "TransformMatchingTex")
            for step in analysis.animation_steps
        )

        if has_axes and has_graph:
            hints.append("function behavior and slope")
        if has_math:
            hints.append("meaning of the equation")
        if has_transforms:
            hints.append("what the transformation demonstrates")
        if has_shapes:
            hints.append("geometric properties")
        if has_text_only:
            hints.append("conceptual meaning of the text")
        if has_axes and has_graph:
            # Tangent / derivative inference
            for t in types:
                if "Line" in t or "Arrow" in t:
                    hints.append("derivative or slope")
                    break
        if has_axes and has_graph:
            # Area under curve inference
            for t in types:
                if t in ("Polygon", "Rectangle"):
                    hints.append("area under the curve")
                    break
        if has_math and any(step.animation_type == "Write" for step in analysis.animation_steps):
            hints.append("algebraic derivation")
        if has_shapes and has_transforms:
            hints.append("geometric transformation")
        return sorted(set(hints))

    def _build_focus_lines(self, analysis: SceneAnalysis) -> List[str]:
        lines: List[str] = []
        types = {o.type for o in analysis.objects}
        has_axes = any(t in ("Axes", "NumberPlane", "ThreeDAxes") for t in types)
        has_graph = any("Graph" in t or "Function" in t for t in types)
        has_math = any(t in ("MathTex", "Tex") for t in types)
        has_text_only = bool(analysis.objects) and all(
            t in ("Text", "Paragraph", "MarkupText") for t in types
        )
        has_shapes = any(
            t in ("Circle", "Square", "Rectangle", "Triangle", "Polygon", "Ellipse")
            for t in types
        )
        has_transforms = any(
            step.animation_type in ("Transform", "ReplacementTransform", "TransformMatchingTex")
            for step in analysis.animation_steps
        )

        if has_axes and has_graph:
            lines.append("Focus on function behavior, domain/range, and slope.")
        if has_math:
            lines.append("Explain what the equation or expression represents.")
        if has_transforms:
            lines.append("Describe what the transformation demonstrates." )
        if has_shapes:
            lines.append("Highlight geometric properties and spatial reasoning.")
        if has_text_only:
            lines.append("Explain the core idea behind the text content.")
        if not lines:
            lines.append("Provide a clear, student-friendly summary of the animation.")

        if len(lines) > 1:
            lines.append("Synthesize these ideas into one coherent explanation.")
        return lines
