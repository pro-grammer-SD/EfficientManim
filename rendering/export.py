from __future__ import annotations
# -*- coding: utf-8 -*-

from pathlib import Path

from utils.logger import LOGGER


def export_python(code: str, path: str) -> bool:
    try:
        Path(path).write_text(code, encoding="utf-8")
        return True
    except Exception as exc:
        LOGGER.error(f"Export failed: {exc}")
        return False
