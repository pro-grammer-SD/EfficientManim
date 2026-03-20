from __future__ import annotations
# -*- coding: utf-8 -*-

import json
import os
from typing import Any, Callable

from core.config import SETTINGS
from utils.logger import LOGGER


class AISlideAnimator:
    """Calls the AI model to convert slide context into structured animation JSON."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.2,
        max_output_tokens: int = 8192,
    ) -> None:
        if model is None or not str(model).strip():
            model = (
                SETTINGS.get("GEMINI_MODEL", "gemini-3-flash-preview")
                or "gemini-3-flash-preview"
            )
        self.model = str(model)
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def generate_slides(
        self,
        context_text: str,
        user_prompt: str,
        on_chunk: Callable[[str], None] | None = None,
    ) -> dict:
        api_key = SETTINGS.get("GEMINI_API_KEY", "") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set. Open Settings to add it.")

        system_prompt = self._system_prompt()
        prompt = (
            f"{system_prompt}\n\n"
            f"USER_PROMPT:\n{user_prompt}\n\n"
            f"PDF_CONTEXT:\n{context_text}\n"
        )

        try:
            from google import genai
            from google.genai import types
        except Exception as exc:
            raise RuntimeError(f"google-genai not available: {exc}") from exc

        client = genai.Client(api_key=api_key)
        config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            response_mime_type="application/json",
        )

        full_text = ""
        for chunk in client.models.generate_content_stream(
            model=self.model, contents=prompt, config=config
        ):
            text = self._extract_text(chunk)
            if not text:
                continue
            full_text += text
            if on_chunk:
                on_chunk(text)

        data = self._parse_json(full_text)
        return self._normalize(data)

    def _extract_text(self, chunk: Any) -> str:
        try:
            if getattr(chunk, "text", None):
                return str(chunk.text)
            if chunk.candidates and chunk.candidates[0].content:
                parts = chunk.candidates[0].content.parts or []
                if parts and getattr(parts[0], "text", None):
                    return str(parts[0].text)
        except Exception:
            pass
        return ""

    def _parse_json(self, text: str) -> Any:
        if not text:
            raise ValueError("Empty AI response.")
        raw = text.strip()
        json_text = self._extract_json_block(raw)
        try:
            return json.loads(json_text)
        except Exception as exc:
            LOGGER.error(f"Failed to parse JSON: {exc}")
            raise

    def _extract_json_block(self, text: str) -> str:
        if text.startswith("{") or text.startswith("["):
            return text
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
        return text

    def _normalize(self, data: Any) -> dict:
        if isinstance(data, dict):
            deck_title = data.get("deck_title") or data.get("title") or ""
            slides = data.get("slides") or data.get("pages") or []
        elif isinstance(data, list):
            deck_title = ""
            slides = data
        else:
            deck_title = ""
            slides = []

        normalized: list[dict] = []
        for slide in slides:
            if not isinstance(slide, dict):
                continue
            title = slide.get("title") or slide.get("heading") or ""
            bullets = (
                slide.get("bullets")
                or slide.get("bullet_points")
                or slide.get("points")
                or []
            )
            equations = (
                slide.get("equations")
                or slide.get("math")
                or slide.get("formulas")
                or []
            )
            deriv = (
                slide.get("derivation_steps")
                or slide.get("derivations")
                or slide.get("steps")
                or []
            )

            normalized.append(
                {
                    "title": self._to_str(title),
                    "bullets": self._to_list(bullets),
                    "equations": self._to_list(equations),
                    "derivation_steps": self._normalize_derivations(deriv),
                }
            )

        return {"deck_title": self._to_str(deck_title), "slides": normalized}

    def _normalize_derivations(self, deriv: Any) -> list[dict]:
        steps = []
        if isinstance(deriv, dict):
            deriv = [deriv]
        if not isinstance(deriv, list):
            return steps
        for item in deriv:
            if isinstance(item, dict):
                eq = item.get("equation") or item.get("eq") or item.get("math") or ""
                exp = item.get("explanation") or item.get("note") or ""
                if eq or exp:
                    steps.append(
                        {"equation": self._to_str(eq), "explanation": self._to_str(exp)}
                    )
            else:
                steps.append({"equation": self._to_str(item), "explanation": ""})
        return steps

    def _to_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [self._to_str(v) for v in value if str(v).strip()]
        return [self._to_str(value)] if str(value).strip() else []

    def _to_str(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    def _system_prompt(self) -> str:
        return (
            "You are a high-precision math slide content generator for a Manim slide engine.\n"
            "Your content will be animated automatically, so your structure must be clean and readable.\n"
            "You generate ONLY structured slide content. You are NOT allowed to generate code.\n"
            "\n"
            "OUTPUT FORMAT (MANDATORY):\n"
            "Return ONLY valid JSON.\n"
            "Do NOT write explanations.\n"
            "Do NOT use markdown.\n"
            "Do NOT wrap in code blocks.\n"
            "\n"
            "JSON SCHEMA:\n"
            "{\n"
            '  "deck_title": "optional overall title",\n'
            '  "slides": [\n'
            "    {\n"
            '      "title": "slide title",\n'
            '      "bullets": ["bullet 1", "bullet 2"],\n'
            '      "equations": ["latex equation"],\n'
            '      "derivation_steps": [\n'
            '        {"equation": "latex step", "explanation": "optional short note"}\n'
            "      ]\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "\n"
            "ABSOLUTE RESTRICTIONS:\n"
            "- Do NOT generate Python code.\n"
            "- Do NOT generate Manim code.\n"
            "- Do NOT generate styling instructions.\n"
            "- Do NOT generate colors.\n"
            "- Do NOT generate layout commands.\n"
            "- Do NOT generate animation commands.\n"
            "- Generate content ONLY (titles, bullets, equations).\n"
            "\n"
            "CRITICAL JSON RULES:\n"
            "- Use double quotes only.\n"
            "- No trailing commas.\n"
            "- No comments.\n"
            "- No text outside JSON.\n"
            "- Always return at least 4 slides.\n"
            "\n"
            "LAYOUT SAFETY RULES (VERY IMPORTANT):\n"
            "- Slides must NEVER contain long text.\n"
            "- Bullets must stay short (max 10–12 words).\n"
            "- Titles must stay short (max 5–6 words).\n"
            "- Avoid long equations.\n"
            "- Avoid multi-line explanations.\n"
            "- Avoid large blocks of text.\n"
            "- Avoid repeating the same idea in multiple bullets.\n"
            "\n"
            "ANTI-OVERLAP RULES:\n"
            "- Do NOT create slides with too much content.\n"
            "- Maximum 3 bullets per slide.\n"
            "- Maximum 2 equations per slide.\n"
            "- Use short equations that fit on one line.\n"
            "- Never generate very long derivation steps.\n"
            "- Each slide must remain visually simple.\n"
            "\n"
            "ANIMATION-FRIENDLY STRUCTURE:\n"
            "- Bullets must be independent (so they can animate one-by-one).\n"
            "- Do NOT merge multiple ideas into one bullet.\n"
            "- Use clear mathematical progression across slides.\n"
            "- Prefer step-by-step explanation rather than large slides.\n"
            "\n"
            "LATEX SAFETY RULES:\n"
            "- Equations must work inside MathTex.\n"
            "- Never use \\begin or \\end.\n"
            "- Never use \\left or \\right.\n"
            "- Never generate multi-line math.\n"
            "- Never generate aligned equations.\n"
            "- Never generate text inside LaTeX.\n"
            "- Always use simple valid math.\n"
            "\n"
            "ALLOWED LATEX ONLY:\n"
            "- n!\n"
            "- n^k\n"
            "- x^2\n"
            "- \\frac{a}{b}\n"
            "- \\binom{n}{k}\n"
            "- m + n\n"
            "- m \\cdot n\n"
            "- P(n, k)\n"
            "- Basic algebra and probability formulas\n"
            "\n"
            "DERIVATION RULES:\n"
            "- Only include derivation_steps if the slide is about a formula.\n"
            "- Maximum 3 steps.\n"
            "- Each step must be short and valid standalone LaTeX.\n"
            "\n"
            "QUALITY RULES:\n"
            "- Slides must be visually clean.\n"
            "- Prefer clarity over complexity.\n"
            "- Prefer more slides instead of crowded slides.\n"
            "- Never output broken LaTeX.\n"
            "\n"
            "Return ONLY the JSON."
        )
