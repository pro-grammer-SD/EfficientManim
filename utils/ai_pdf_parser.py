from __future__ import annotations
# -*- coding: utf-8 -*-

from pathlib import Path
from statistics import median

try:
    import pdfplumber  # type: ignore
except Exception as _exc:
    pdfplumber = None  # type: ignore

try:
    import regex as re  # type: ignore
except Exception:
    import re  # type: ignore

from utils.logger import LOGGER


class PDFParser:
    """Extracts structured slide data from PDFs."""

    def __init__(self) -> None:
        self._bullet_re = re.compile(
            r"^\s*(?:[\u2022\u2023\u25E6\u2043\u2219•\-\*]|"
            r"\d+\s*[.)]|[a-zA-Z]\s*[.)])\s+"
        )
        self._math_re = re.compile(
            r"(\\[a-zA-Z]+|[=+\-*/^_]|[∑∫√∞≤≥≠≈±]|"
            r"\b(?:sin|cos|tan|log|ln|lim|sum|int)\b)",
            re.IGNORECASE,
        )

    def parse_pdfs(self, paths: list[str | Path]) -> list[dict]:
        slides: list[dict] = []
        for p in paths:
            try:
                slides.extend(self.parse_pdf(p))
            except Exception as exc:
                LOGGER.error(f"PDF parse failed: {p} -> {exc}")
        return slides

    def parse_pdf(self, path: str | Path) -> list[dict]:
        if pdfplumber is None:
            raise RuntimeError(
                "pdfplumber is not installed. Install it to enable PDF parsing."
            )

        pdf_path = Path(path).expanduser().resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(str(pdf_path))

        slides: list[dict] = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_index, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                text = self._normalize_text(text)

                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                heading = self._extract_heading(page, lines)
                bullets = self._extract_bullets(lines)
                equations = self._extract_equations(lines)

                slides.append(
                    {
                        "source": str(pdf_path),
                        "page": page_index,
                        "heading": heading,
                        "bullets": bullets,
                        "equations": equations,
                        "raw_text": text[:4000],
                    }
                )
        return slides

    def _normalize_text(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def _extract_heading(self, page, lines: list[str]) -> str:
        heading = ""
        try:
            words = page.extract_words(extra_attrs=["size", "top", "x0"])
        except Exception:
            words = []

        line_info = self._group_words_into_lines(words)
        sizes = [li["size"] for li in line_info if li["size"] > 0]
        med = median(sizes) if sizes else 0.0
        candidates = []

        for li in line_info:
            txt = li["text"].strip()
            if not txt:
                continue
            if self._bullet_re.match(txt):
                continue
            if med > 0 and li["size"] >= med * 1.2 and len(txt) <= 120:
                candidates.append(txt)

        if candidates:
            heading = candidates[0]
        elif lines:
            for ln in lines[:3]:
                if not self._bullet_re.match(ln):
                    heading = ln
                    break
        return heading.strip()

    def _group_words_into_lines(self, words: list[dict]) -> list[dict]:
        if not words:
            return []

        lines: dict[float, list[dict]] = {}
        for w in words:
            try:
                top = round(float(w.get("top", 0.0)), 1)
            except Exception:
                top = 0.0
            lines.setdefault(top, []).append(w)

        line_info = []
        for top in sorted(lines.keys()):
            line_words = sorted(lines[top], key=lambda x: float(x.get("x0", 0.0)))
            text = " ".join(w.get("text", "") for w in line_words).strip()
            sizes = [float(w.get("size", 0.0)) for w in line_words if w.get("size")]
            avg_size = sum(sizes) / len(sizes) if sizes else 0.0
            line_info.append({"text": text, "size": avg_size})
        return line_info

    def _extract_bullets(self, lines: list[str]) -> list[str]:
        bullets = []
        for ln in lines:
            m = self._bullet_re.match(ln)
            if m:
                cleaned = self._bullet_re.sub("", ln).strip()
                if cleaned:
                    bullets.append(cleaned)
        return bullets

    def _extract_equations(self, lines: list[str]) -> list[str]:
        equations = []
        for ln in lines:
            if self._looks_like_equation(ln):
                equations.append(ln.strip())
        return equations

    def _looks_like_equation(self, line: str) -> bool:
        if not line:
            return False
        if self._bullet_re.match(line):
            return False
        if self._math_re.search(line):
            has_alpha = re.search(r"[A-Za-z0-9]", line) is not None
            return has_alpha
        return False
