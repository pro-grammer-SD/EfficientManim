from __future__ import annotations
# -*- coding: utf-8 -*-

from pathlib import Path

from utils.logger import LOGGER


class PDFAttachmentManager:
    """Manages ordered PDF attachments for the AI pipeline."""

    def __init__(self) -> None:
        self._files: list[str] = []
        self._file_set: set[str] = set()

    def add_file(self, path: str | Path) -> bool:
        try:
            p = Path(path).expanduser().resolve()
        except Exception:
            return False

        if not p.exists() or not p.is_file():
            return False
        if p.suffix.lower() != ".pdf":
            return False

        ps = str(p)
        if ps in self._file_set:
            return False

        self._files.append(ps)
        self._file_set.add(ps)
        LOGGER.info(f"PDF attached: {ps}")
        return True

    def add_files(self, paths: list[str | Path]) -> list[str]:
        added = []
        for p in paths:
            if self.add_file(p):
                added.append(str(Path(p).expanduser().resolve()))
        return added

    def remove_file(self, path: str | Path) -> bool:
        try:
            p = str(Path(path).expanduser().resolve())
        except Exception:
            return False

        if p not in self._file_set:
            return False
        self._file_set.remove(p)
        try:
            self._files.remove(p)
        except ValueError:
            pass
        LOGGER.info(f"PDF removed: {p}")
        return True

    def clear(self) -> None:
        self._files.clear()
        self._file_set.clear()
        LOGGER.info("PDF attachments cleared")

    def list_files(self) -> list[str]:
        return list(self._files)

    def count(self) -> int:
        return len(self._files)
