from __future__ import annotations
# -*- coding: utf-8 -*-

from PySide6.QtWidgets import QApplication

from rendering.render_manager import ImageLoaderWorker
from utils.logger import LOGGER


def load_preview_async(path: str, max_width: int, on_loaded):
    """Load and scale a preview image asynchronously."""
    worker = ImageLoaderWorker(path, max_width)
    worker.image_loaded.connect(on_loaded)
    worker.start()
    return worker


def copy_to_clipboard(text: str) -> None:
    try:
        QApplication.clipboard().setText(text)
    except Exception as exc:
        LOGGER.warn(f"Clipboard copy failed: {exc}")
