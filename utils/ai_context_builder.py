from __future__ import annotations
# -*- coding: utf-8 -*-

from pathlib import Path


class AIContextBuilder:
    """Builds structured context for AI slide generation."""

    def __init__(
        self,
        max_chars: int = 14000,
        max_bullets: int = 8,
        max_equations: int = 6,
        max_raw_chars: int = 800,
    ) -> None:
        self.max_chars = max_chars
        self.max_bullets = max_bullets
        self.max_equations = max_equations
        self.max_raw_chars = max_raw_chars

    def build_context(self, slides: list[dict], user_prompt: str) -> str:
        base = self._build_context(
            slides,
            include_raw=True,
            bullet_limit=self.max_bullets,
            equation_limit=self.max_equations,
        )
        if len(base) > self.max_chars:
            base = self._build_context(
                slides,
                include_raw=False,
                bullet_limit=self.max_bullets,
                equation_limit=self.max_equations,
            )
        if len(base) > self.max_chars:
            base = self._build_context(
                slides,
                include_raw=False,
                bullet_limit=max(3, self.max_bullets // 2),
                equation_limit=max(2, self.max_equations // 2),
            )
        if len(base) > self.max_chars:
            base = base[: self.max_chars] + "\n[TRUNCATED]"

        prompt = user_prompt.strip() if user_prompt else ""
        return f"{base}\n\nUSER_PROMPT:\n{prompt}".strip()

    def _build_context(
        self,
        slides: list[dict],
        include_raw: bool,
        bullet_limit: int,
        equation_limit: int,
    ) -> str:
        grouped: dict[str, list[dict]] = {}
        for slide in slides:
            src = slide.get("source", "unknown")
            grouped.setdefault(src, []).append(slide)

        parts: list[str] = []
        parts.append("PDF_CONTEXT:")
        for src, pages in grouped.items():
            pdf_name = Path(src).name
            parts.append(f"\n[PDF] {pdf_name}")
            for page in pages:
                page_num = page.get("page", "?")
                parts.append(f"Page {page_num}:")

                heading = (page.get("heading") or "").strip()
                if heading:
                    parts.append(f"Title: {self._trim(heading, 160)}")

                bullets = page.get("bullets") or []
                if bullets:
                    parts.append("Bullets:")
                    for b in bullets[:bullet_limit]:
                        parts.append(f"- {self._trim(str(b), 200)}")
                    if len(bullets) > bullet_limit:
                        parts.append(
                            f"- [omitted {len(bullets) - bullet_limit} more bullets]"
                        )

                equations = page.get("equations") or []
                if equations:
                    parts.append("Equations:")
                    for e in equations[:equation_limit]:
                        parts.append(f"= {self._trim(str(e), 200)}")
                    if len(equations) > equation_limit:
                        parts.append(
                            f"= [omitted {len(equations) - equation_limit} more equations]"
                        )

                if include_raw:
                    raw = (page.get("raw_text") or "").strip()
                    if raw:
                        raw = self._trim(raw, self.max_raw_chars)
                        parts.append(f"Text: {raw}")
        return "\n".join(parts).strip()

    def _trim(self, text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip() + "..."
