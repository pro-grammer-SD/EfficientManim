from __future__ import annotations
# -*- coding: utf-8 -*-

import json
import os
import re
from typing import Any, Callable, Dict, Optional

from PySide6.QtCore import QObject, QThread, Signal, QEventLoop

from core.config import SETTINGS

from scene_explainer.analyzer import SceneAnalyzer
from scene_explainer.explanation_models import (
    ExplainResponse,
    LessonNotes,
    SceneAnalysis,
)
from scene_explainer.prompt_builder import PromptBuilder


class _AIWorker(QThread):
    completed = Signal(object, int)
    failed = Signal(str, int)

    def __init__(
        self,
        prompt: str,
        model: str,
        parser: Callable[[str], object],
        request_id: int,
    ) -> None:
        super().__init__()
        self.prompt = prompt
        self.model = model
        self.parser = parser
        self.request_id = request_id
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True
        self.requestInterruption()

    def run(self) -> None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            self.failed.emit(
                "Gemini API key not configured. Open Settings to add it.",
                self.request_id,
            )
            return
        try:
            from google import genai

            client = genai.Client(api_key=api_key)
            chunks: list[str] = []
            for chunk in client.models.generate_content_stream(
                model=self.model,
                contents=self.prompt,
            ):
                if self.isInterruptionRequested() or self._cancelled:
                    return
                text = getattr(chunk, "text", None)
                if text:
                    chunks.append(text)
            if self.isInterruptionRequested() or self._cancelled:
                return
            full_text = "".join(chunks).strip()
            if not full_text:
                self.failed.emit("Empty AI response.", self.request_id)
                return
            parsed = self.parser(full_text)
            self.completed.emit(parsed, self.request_id)
        except Exception as exc:
            self.failed.emit(str(exc), self.request_id)


class AIExplainer(QObject):
    """Runs AI explanations asynchronously with cancellation support."""

    explanation_ready = Signal(object)
    lesson_ready = Signal(object)
    learning_ready = Signal(object)
    history_ready = Signal(object)
    error = Signal(str)

    def __init__(
        self, prompt_builder: PromptBuilder | None = None, parent=None
    ) -> None:
        super().__init__(parent)
        self.prompt_builder = prompt_builder or PromptBuilder()
        self._current_worker: Optional[_AIWorker] = None
        self._request_id = 0

    def cancel_current(self) -> None:
        if self._current_worker is not None:
            try:
                self._current_worker.cancel()
            except Exception:
                pass

    def request_explanation(
        self, analysis: SceneAnalysis, mode: str = "detailed"
    ) -> int:
        prompt = self.prompt_builder.build_explain_prompt(analysis, mode)
        return self._start_worker(
            prompt, self._parse_explain_response, self.explanation_ready
        )

    def request_lesson_notes(self, analysis: SceneAnalysis) -> int:
        prompt = self.prompt_builder.build_lesson_prompt(analysis)
        return self._start_worker(prompt, self._parse_lesson_notes, self.lesson_ready)

    def request_learning_explanation(
        self, analysis: SceneAnalysis, what_happened: str
    ) -> int:
        prompt = self.prompt_builder.build_learning_prompt(analysis, what_happened)
        return self._start_worker(
            prompt, self._parse_learning_response, self.learning_ready
        )

    def request_history_change(self, prompt: str) -> int:
        return self._start_worker(
            prompt, self._parse_history_change, self.history_ready
        )

    def request_history_checkpoint(
        self, analysis: SceneAnalysis, mode: str = "detailed"
    ) -> int:
        prompt = self.prompt_builder.build_explain_prompt(analysis, mode)
        return self._start_worker(
            prompt, self._parse_explain_response, self.history_ready
        )

    def run_blocking(self, prompt: str, parser: Callable[[str], object]) -> object:
        """Run AI call in background thread and wait (UI-safe via nested event loop)."""
        result: dict[str, object] = {"value": None, "error": None}
        request_id = self._next_request_id()
        worker = self._create_worker(prompt, parser, request_id)

        loop = QEventLoop()

        def _on_done(payload, rid):
            if rid != request_id:
                return
            result["value"] = payload
            loop.quit()

        def _on_err(msg, rid):
            if rid != request_id:
                return
            result["error"] = msg
            loop.quit()

        worker.completed.connect(_on_done)
        worker.failed.connect(_on_err)
        worker.start()
        loop.exec()

        if result["error"]:
            raise RuntimeError(str(result["error"]))
        return result["value"]

    # ──────────────────────────────────────────────────────────────
    # Worker helpers
    # ──────────────────────────────────────────────────────────────

    def _start_worker(
        self, prompt: str, parser: Callable[[str], object], signal: Signal
    ) -> int:
        self.cancel_current()
        request_id = self._next_request_id()
        worker = self._create_worker(prompt, parser, request_id)
        self._current_worker = worker

        def _on_done(payload, rid):
            if rid != request_id:
                return
            signal.emit(payload)

        def _on_err(msg, rid):
            if rid != request_id:
                return
            self.error.emit(msg)

        worker.completed.connect(_on_done)
        worker.failed.connect(_on_err)
        worker.start()
        return request_id

    def _create_worker(
        self, prompt: str, parser: Callable[[str], object], request_id: int
    ) -> _AIWorker:
        model = str(SETTINGS.get("GEMINI_MODEL", "gemini-3-flash-preview"))
        return _AIWorker(
            prompt=prompt, model=model, parser=parser, request_id=request_id
        )

    def _next_request_id(self) -> int:
        self._request_id += 1
        return self._request_id

    # ──────────────────────────────────────────────────────────────
    # Parsing & sanitization
    # ──────────────────────────────────────────────────────────────

    def _parse_explain_response(self, text: str) -> ExplainResponse:
        data = self._extract_json(text)
        concept = self._sanitize_text(str(data.get("concept_explanation", "")).strip())
        step_by_step = self._sanitize_text(str(data.get("step_by_step", "")).strip())
        visual = self._sanitize_text(str(data.get("visual_reasoning", "")).strip())
        simple = self._sanitize_text(str(data.get("simple_explanation", "")).strip())
        key_takeaways = data.get("key_takeaways") or []
        if not isinstance(key_takeaways, list):
            key_takeaways = [str(key_takeaways)]
        key_takeaways = [
            self._sanitize_text(str(k).strip()) for k in key_takeaways if str(k).strip()
        ]
        mode_used = str(data.get("mode_used", "detailed")).strip().lower()
        if mode_used not in ("simple", "detailed"):
            mode_used = "detailed"
        return ExplainResponse(
            concept_explanation=concept,
            step_by_step=step_by_step,
            visual_reasoning=visual,
            simple_explanation=simple,
            key_takeaways=key_takeaways,
            mode_used=mode_used,
        )

    def _parse_learning_response(self, text: str) -> dict:
        data = self._extract_json(text)
        what = self._sanitize_text(str(data.get("what_happened", "")).strip())
        why = self._sanitize_text(str(data.get("why_it_matters", "")).strip())
        return {"what_happened": what, "why_it_matters": why}

    def _parse_lesson_notes(self, text: str) -> LessonNotes:
        data = self._extract_json(text)
        title = (
            self._sanitize_text(str(data.get("lesson_title", "Lesson"))).strip()
            or "Lesson"
        )
        concept = self._sanitize_text(str(data.get("concept_explanation", "")).strip())
        visual = self._sanitize_text(str(data.get("visual_explanation", "")).strip())
        step = self._sanitize_text(str(data.get("step_by_step_teaching", "")).strip())
        student = self._sanitize_text(str(data.get("student_notes", "")).strip())
        key_takeaways = data.get("key_takeaways") or []
        if not isinstance(key_takeaways, list):
            key_takeaways = [str(key_takeaways)]
        key_takeaways = [
            self._sanitize_text(str(k).strip()) for k in key_takeaways if str(k).strip()
        ]
        return LessonNotes(
            lesson_title=title,
            concept_explanation=concept,
            visual_explanation=visual,
            step_by_step_teaching=step,
            student_notes=student,
            key_takeaways=key_takeaways,
        )

    def _parse_history_change(self, text: str) -> dict:
        data = self._extract_json(text)
        concept_change = self._sanitize_text(
            str(data.get("concept_change_summary", "")).strip()
        )
        educational = self._sanitize_text(
            str(data.get("educational_significance", "")).strip()
        )
        return {
            "concept_change_summary": concept_change,
            "educational_significance": educational,
        }

    def _extract_json(self, text: str) -> Dict[str, Any]:
        cleaned = self._strip_fences(text)
        try:
            return json.loads(cleaned)
        except Exception:
            pass
        # Try to find first JSON object
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        raise ValueError("Failed to parse AI response as JSON.")

    def _strip_fences(self, text: str) -> str:
        if "```" not in text:
            return text.strip()
        # Remove markdown fences
        text = re.sub(r"```[a-zA-Z0-9]*", "", text)
        return text.replace("```", "").strip()

    def _sanitize_text(self, text: str) -> str:
        if not text:
            return ""
        banned = [
            "python",
            "manim",
            "node",
            "nodes",
            "node graph",
            "node-based",
            "class",
            "method",
        ]
        replacements = {
            "fadein": "fade in",
            "fadeout": "fade out",
            "write": "writing",
            "unwrite": "erasing",
            "drawborderthenfill": "outline then fill",
            "create": "draw",
            "uncreate": "erase",
            "transform": "transformation",
            "replacementtransform": "transformation",
            "transformmatchingtex": "transformation",
            "moveto": "move",
            "shift": "shift",
            "applyfunction": "change",
            "rotate": "rotation",
            "scale": "scaling",
            "laggedstart": "staggered sequence",
            "laggedstartmap": "staggered sequence",
            "animationgroup": "grouped sequence",
            "succession": "sequence",
            "flash": "highlight",
            "growfrompoint": "grow effect",
            "growarrow": "grow effect",
        }
        lowered = text
        for bad in banned:
            lowered = re.sub(rf"\b{re.escape(bad)}\b", "", lowered, flags=re.IGNORECASE)
        for key, value in replacements.items():
            lowered = re.sub(
                rf"\b{re.escape(key)}\b", value, lowered, flags=re.IGNORECASE
            )
        lowered = re.sub(r"\s{2,}", " ", lowered)
        return lowered.strip()


class ExplainService(QObject):
    """Shared explanation engine used by GUI and MCP commands."""

    def __init__(self, main_window, parent=None) -> None:
        super().__init__(parent)
        self.main_window = main_window
        self.analyzer = SceneAnalyzer()
        self.prompt_builder = PromptBuilder()
        self.ai = AIExplainer(self.prompt_builder)

    def analyze_scene(
        self,
        node_ids: Optional[list[str]] = None,
        animation_id: Optional[str] = None,
        objects_only: bool = False,
    ) -> SceneAnalysis:
        return self.analyzer.analyze_window(
            self.main_window,
            node_ids=node_ids,
            animation_id=animation_id,
            objects_only=objects_only,
        )

    def explain_scene(
        self, analysis: SceneAnalysis, mode: str = "detailed"
    ) -> ExplainResponse:
        prompt = self.prompt_builder.build_explain_prompt(analysis, mode)
        result = self.ai.run_blocking(prompt, self.ai._parse_explain_response)
        if isinstance(result, ExplainResponse):
            return result
        raise RuntimeError("Invalid explanation response")

    def lesson_notes(self, analysis: SceneAnalysis) -> LessonNotes:
        prompt = self.prompt_builder.build_lesson_prompt(analysis)
        result = self.ai.run_blocking(prompt, self.ai._parse_lesson_notes)
        if isinstance(result, LessonNotes):
            return result
        raise RuntimeError("Invalid lesson response")

    def learning_explanation(self, analysis: SceneAnalysis, what_happened: str) -> dict:
        prompt = self.prompt_builder.build_learning_prompt(analysis, what_happened)
        result = self.ai.run_blocking(prompt, self.ai._parse_learning_response)
        if isinstance(result, dict):
            return result
        raise RuntimeError("Invalid learning response")

    def history_change_explanation(self, prompt: str) -> dict:
        result = self.ai.run_blocking(prompt, self.ai._parse_history_change)
        if isinstance(result, dict):
            return result
        raise RuntimeError("Invalid history change response")

    def cancel_all(self) -> None:
        try:
            self.ai.cancel_current()
        except Exception:
            pass
