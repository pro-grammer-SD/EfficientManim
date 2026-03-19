from __future__ import annotations
# -*- coding: utf-8 -*-

import collections
import traceback
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, Signal


class LogManager(QObject):
    log_signal = Signal(str, str)  # Level, Message

    def __init__(self):
        super().__init__()

    def info(self, msg):
        self.log_signal.emit("INFO", str(msg))

    def warn(self, msg):
        self.log_signal.emit("WARN", str(msg))

    def error(self, msg):
        self.log_signal.emit("ERROR", str(msg))

    def manim(self, msg):
        self.log_signal.emit("MANIM", str(msg))

    def ai(self, msg):
        self.log_signal.emit("AI", str(msg))


class EnhancedLogManager(LogManager):
    """Enhanced logging with file persistence, categories, and ring buffer.

    Categories: GLOBAL, PREVIEW, RENDER, EXTENSION, UI, VALIDATION
    Levels: DEBUG, INFO, WARN, ERROR, CRITICAL, MANIM, AI
    """

    MAX_LOG_ENTRIES = 5000

    def __init__(self):
        super().__init__()
        log_dir = None
        try:
            from core.config import AppPaths

            log_dir = AppPaths.USER_DATA
        except Exception:
            log_dir = Path.home() / ".efficientmanim"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = log_dir / "app.log"
        self.log_entries = collections.deque(maxlen=self.MAX_LOG_ENTRIES)

    def _write_log(self, level, msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] {level}: {msg}"
        self.log_entries.append(log_entry)
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_entry + "\n")
        except Exception:
            pass

    def log(self, level, category, msg):
        full_msg = f"[{category}] {msg}"
        self._write_log(level, full_msg)
        self.log_signal.emit(level, full_msg)

    def debug(self, msg):
        self._write_log("DEBUG", str(msg))

    def info(self, msg):
        self._write_log("INFO", str(msg))
        self.log_signal.emit("INFO", str(msg))

    def warn(self, msg):
        self._write_log("WARN", str(msg))
        self.log_signal.emit("WARN", str(msg))

    def error(self, msg):
        self._write_log("ERROR", str(msg))
        self.log_signal.emit("ERROR", str(msg))

    def critical(self, msg):
        tb = traceback.format_exc()
        full_msg = f"{msg}\n{tb}" if tb and "NoneType" not in tb else str(msg)
        self._write_log("CRITICAL", full_msg)
        self.log_signal.emit("ERROR", full_msg)

    def manim(self, msg):
        self._write_log("MANIM", str(msg))
        self.log_signal.emit("MANIM", str(msg))

    def ai(self, msg):
        self._write_log("AI", str(msg))
        self.log_signal.emit("AI", str(msg))


LOGGER = EnhancedLogManager()
