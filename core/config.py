from __future__ import annotations
# -*- coding: utf-8 -*-

import os
import shutil
import tempfile
from pathlib import Path

from PySide6.QtCore import QObject, QSettings, QStandardPaths, Signal

APP_NAME = "EfficientManim"
APP_VERSION = "2.0.5"
PROJECT_EXT = ".efp"  # EfficientManim Project (Zip)
LIGHT_MODE = True


class AppPaths:
    """Centralized path management."""

    USER_DATA = (
        Path(
            QStandardPaths.writableLocation(
                QStandardPaths.StandardLocation.AppDataLocation
            )
        )
        / APP_NAME
    )
    TEMP_DIR = Path(tempfile.gettempdir()) / "EfficientManim_Temp"
    FONTS_DIR = Path("fonts")

    @staticmethod
    def ensure_dirs():
        """Ensure all required directories exist, with force cleanup of temp."""
        AppPaths.USER_DATA.mkdir(parents=True, exist_ok=True)

        if AppPaths.TEMP_DIR.exists():
            try:
                shutil.rmtree(AppPaths.TEMP_DIR)
            except PermissionError:
                try:
                    for item in AppPaths.TEMP_DIR.rglob("*"):
                        try:
                            if item.is_file():
                                item.unlink(missing_ok=True)
                            elif item.is_dir():
                                item.rmdir()
                        except (PermissionError, OSError):
                            pass
                    try:
                        AppPaths.TEMP_DIR.rmdir()
                    except (PermissionError, OSError):
                        pass
                except Exception as exc:
                    try:
                        from utils.logger import LOGGER

                        LOGGER.warn(f"Temp cleanup partial (locked files): {exc}")
                    except Exception:
                        pass
            except Exception as exc:
                try:
                    from utils.logger import LOGGER

                    LOGGER.warn(f"Temp cleanup failed: {exc}")
                except Exception:
                    pass

        AppPaths.TEMP_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def force_cleanup_old_files(age_seconds=3600):
        """Remove old files from temp directory (older than age_seconds)."""
        try:
            import time

            current_time = time.time()
            for item in AppPaths.TEMP_DIR.rglob("*"):
                try:
                    if item.is_file():
                        mtime = item.stat().st_mtime
                        if current_time - mtime > age_seconds:
                            item.unlink(missing_ok=True)
                except (PermissionError, OSError):
                    pass
        except Exception:
            pass


class SettingsManager(QObject):
    """Persistent settings via QSettings."""

    settings_changed = Signal()

    def __init__(self):
        super().__init__()
        self._store = QSettings("Gemini", APP_NAME)
        self.apply_env()

    def get(self, key, default=None, type=None):
        val = self._store.value(key, default)
        if type is bool:
            return str(val).lower() == "true"
        if type is int:
            try:
                return int(val)
            except (TypeError, ValueError):
                return default
        if type is float:
            try:
                return float(val)
            except (TypeError, ValueError):
                return default
        return val

    def set(self, key, value):
        self._store.setValue(key, str(value) if value is not None else "")
        if key == "GEMINI_API_KEY":
            self.apply_env()
        self.settings_changed.emit()

    def apply_env(self):
        key = self.get("GEMINI_API_KEY", "")
        if key:
            os.environ["GEMINI_API_KEY"] = str(key)


SETTINGS = SettingsManager()
