from __future__ import annotations
# -*- coding: utf-8 -*-

import os
import re
import struct
from typing import Any

import numpy as np
from PySide6.QtGui import QColor, QFont

from utils.logger import LOGGER

try:
    import manim
    from manim.utils.color import ManimColor

    MANIM_AVAILABLE = True
except Exception:
    manim = None
    ManimColor = None
    MANIM_AVAILABLE = False


def generate(prompt: str, on_chunk, model: str = "gemini-3-flash-preview") -> None:
    """Gemini AI integration hook."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        on_chunk("Gemini API key not configured. Open Settings to add it.")
        return

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=prompt,
        ):
            if chunk.text:
                on_chunk(chunk.text)
    except Exception as exc:
        on_chunk(f"Gemini error: {exc}")


def detect_scene_class(code: str) -> str:
    """Detect Scene class name in Python code."""
    pattern = r"class\s+(\w+)\s*\(\s*Scene\s*\)"
    matches = re.findall(pattern, code)
    return matches[0] if matches else "Scene"


_BACKGROUND_SETTING_LINE_PATTERNS = (
    re.compile(r"^[ \t]*(?:self\.)?camera\.background_color\s*=(?!=)"),
    re.compile(r"^[ \t]*(?:manim\.)?config\.background_color\s*=(?!=)"),
    re.compile(
        r"^[ \t]*(?:return\s+)?(?:[A-Za-z_][A-Za-z0-9_]*\s*=\s*)?Scene\s*\(.*\bbackground_color\s*=(?!=)"
    ),
)


def sanitize_background_settings(code: str) -> str:
    """Remove lines that set background color in AI-generated code."""
    if not code:
        return code
    lines = code.splitlines(keepends=True)
    cleaned: list[str] = []
    for line in lines:
        if any(p.search(line) for p in _BACKGROUND_SETTING_LINE_PATTERNS):
            continue
        cleaned.append(line)
    return "".join(cleaned)


def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    """Generate WAV header for Gemini PCM output."""

    def parse_audio_mime_type(mt: str) -> dict:
        bits_per_sample = 16
        rate = 24000
        parts = mt.split(";")
        for param in parts:
            param = param.strip()
            if param.lower().startswith("rate="):
                try:
                    rate = int(param.split("=", 1)[1])
                except Exception:
                    pass
            elif param.startswith("audio/L"):
                try:
                    bits_per_sample = int(param.split("L", 1)[1])
                except Exception:
                    pass
        return {"bits_per_sample": bits_per_sample, "rate": rate}

    parameters = parse_audio_mime_type(mime_type)
    bits_per_sample = parameters["bits_per_sample"]
    sample_rate = parameters["rate"]
    num_channels = 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + audio_data


def bold_font(point_size: int = 10) -> QFont:
    f = QFont()
    f.setPointSize(point_size)
    f.setBold(True)
    return f


class TypeSafeParser:
    """Comprehensive type validation and safe parsing for Manim parameters."""

    NUMERIC_KEYWORDS = {
        "radius",
        "width",
        "height",
        "scale",
        "factor",
        "size",
        "thickness",
        "stroke_width",
        "font_size",
        "length",
        "rate",
        "opacity",
        "alpha",
        "x",
        "y",
        "z",
        "angle",
        "degrees",
        "radians",
    }

    COLOR_KEYWORDS = {
        "color",
        "fill_color",
        "stroke_color",
        "background_color",
        "fg_color",
        "bg_color",
    }

    POINT_KEYWORDS = {
        "point",
        "points",
        "center",
        "pos",
        "position",
        "start",
        "end",
        "direction",
    }

    @staticmethod
    def is_asset_param(param_name: str) -> bool:
        n = param_name.lower()
        if "filename" in n:
            return True
        if "file" in n or "image" in n or "sound" in n or "svg" in n:
            if "fill" in n or "profile" in n:
                return False
            return True
        return False

    @staticmethod
    def is_numeric_param(param_name: str) -> bool:
        if TypeSafeParser.is_asset_param(param_name):
            return False
        
        name_lower = param_name.lower()
        # EXCEPTION: 'text' and 'tex' parameters must never be treated as numeric
        # even if they contain 'x' or other keywords.
        if name_lower in ("text", "tex_strings", "tex"):
            return False
            
        return any(kw in name_lower for kw in TypeSafeParser.NUMERIC_KEYWORDS)

    @staticmethod
    def is_color_param(param_name: str) -> bool:
        return any(kw in param_name.lower() for kw in TypeSafeParser.COLOR_KEYWORDS)

    @staticmethod
    def is_point_param(param_name: str) -> bool:
        return any(kw in param_name.lower() for kw in TypeSafeParser.POINT_KEYWORDS)

    @staticmethod
    def parse_numeric(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                val = float(value.strip())
                if -1e10 < val < 1e10:
                    return val
                LOGGER.warn(f"Numeric value out of range: {value}, default {default}")
                return default
            return float(value) if value else default
        except (ValueError, TypeError) as exc:
            LOGGER.warn(
                f"Invalid numeric value '{value}': {type(exc).__name__}, default {default}"
            )
            return default

    @staticmethod
    def parse_color(value: Any, default_hex: str = "#FFFFFF") -> str:
        try:
            if value is None:
                return default_hex

            if hasattr(value, "to_hex") or (
                hasattr(value, "__class__") and "ManimColor" in type(value).__name__
            ):
                return value.to_hex() if hasattr(value, "to_hex") else str(value)

            if isinstance(value, str):
                value_clean = value.strip()
                try:
                    if manim is not None and hasattr(manim, value_clean.upper()):
                        manim_color = getattr(manim, value_clean.upper())
                        if (
                            hasattr(manim_color, "to_hex")
                            or hasattr(manim_color, "_internal_value")
                            or "ManimColor" in type(manim_color).__name__
                        ):
                            return str(manim_color)
                except Exception:
                    pass

                if value_clean.startswith("#") and len(value_clean) == 7:
                    try:
                        int(value_clean[1:], 16)
                        return value_clean
                    except ValueError:
                        pass

                qc = QColor(value_clean)
                if qc.isValid():
                    return qc.name()

                return default_hex

            if isinstance(value, (tuple, list)) and len(value) >= 3:
                r, g, b = value[:3]
                if isinstance(r, float) and 0 <= r <= 1:
                    r, g, b = int(r * 255), int(g * 255), int(b * 255)
                return f"#{int(r):02x}{int(g):02x}{int(b):02x}"

            if isinstance(value, QColor) and value.isValid():
                return value.name()

        except Exception as exc:
            LOGGER.warn(f"Color parse error for '{value}': {type(exc).__name__}")

        return default_hex

    @staticmethod
    def to_manim_color(hex_color: Any):
        try:
            if hasattr(hex_color, "to_hex") or "ManimColor" in type(hex_color).__name__:
                return hex_color
            if isinstance(hex_color, str):
                return ManimColor(hex_color) if MANIM_AVAILABLE else hex_color
            return ManimColor("#FFFFFF") if MANIM_AVAILABLE else "#FFFFFF"
        except Exception:
            return ManimColor("#FFFFFF") if MANIM_AVAILABLE else "#FFFFFF"

    @staticmethod
    def validate_point_safe(point: Any, default: Any = None):
        if default is None:
            default = np.array([0.0, 0.0, 0.0])

        try:
            if point is None:
                return default
            if isinstance(point, np.ndarray):
                return point.astype(float)
            arr = np.array(point, dtype=float)
            if arr.shape[0] in [2, 3]:
                if arr.shape[0] == 2:
                    return np.array([arr[0], arr[1], 0.0])
                return arr
            return default
        except Exception:
            return default


class ColorNormalizer:
    """Centralized color normalization utility."""

    @staticmethod
    def normalize_to_hex(value: Any) -> str:
        return TypeSafeParser.parse_color(value)
