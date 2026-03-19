from __future__ import annotations
# -*- coding: utf-8 -*-

import ctypes
import logging
import os
import sys
import threading
import traceback
from pathlib import Path

from PySide6.QtCore import QLoggingCategory
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from core.config import AppPaths
from core.themes import THEME_MANAGER
from utils.logger import LOGGER


def configure_app_id() -> None:
    if sys.platform == "win32":
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "com.programmersd.efficientmanim"
        )


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def configure_qt_logging() -> None:
    QLoggingCategory.setFilterRules("qt.multimedia.*=false")
    os.environ["QT_LOGGING_RULES"] = "qt.multimedia.*=false"


def install_exception_hooks() -> None:
    def _global_exception_hook(exctype, value, tb):
        try:
            tb_str = "".join(traceback.format_exception(exctype, value, tb))
            LOGGER.log(
                "ERROR", "GLOBAL", f"Uncaught {exctype.__name__}: {value}\n{tb_str}"
            )
        except Exception:
            print(
                f"[FATAL] Uncaught: {exctype.__name__}: {value}",
                file=sys.stderr,
            )

    def _thread_exception_hook(args):
        try:
            tb_str = "".join(
                traceback.format_exception(
                    args.exc_type, args.exc_value, args.exc_traceback
                )
            )
            thread_name = args.thread.name if args.thread else "unknown"
            LOGGER.log(
                "ERROR",
                "GLOBAL",
                f"Uncaught in thread '{thread_name}': {args.exc_type.__name__}: {args.exc_value}\n{tb_str}",
            )
        except Exception:
            print(
                f"[FATAL] Thread exception: {args.exc_type}: {args.exc_value}",
                file=sys.stderr,
            )

    sys.excepthook = _global_exception_hook
    threading.excepthook = _thread_exception_hook


def create_application() -> QApplication:
    configure_app_id()
    configure_logging()
    configure_qt_logging()
    AppPaths.ensure_dirs()
    app = QApplication(sys.argv)

    icon_path = Path(__file__).resolve().parent.parent / "icon" / "icon.ico"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    app.setStyleSheet(THEME_MANAGER.get_stylesheet())
    install_exception_hooks()
    return app
