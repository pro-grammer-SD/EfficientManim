# Register this as a proper app using the classic ctypes.windll workaround
import ctypes
import sys
from urllib.parse import urlparse

if sys.platform == "win32":
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
        "com.programmersd.efficientmanim"
    )

import struct

# Audio handling
try:
    from pydub import AudioSegment

    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print(
        "WARNING: pydub not found. Audio duration features will be disabled. 'pip install pydub'"
    )

import os
from google import genai
from google.genai import types


def generate(prompt, on_chunk, model="gemini-3-flash-preview"):
    """
    Gemini AI Integration Hook.
    Retrieves API Key from Settings and uses selected model.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        on_chunk("⚠️ Gemini API Key not configured. Please go to File > Settings.")
        return

    try:
        client = genai.Client(api_key=api_key)
        # Stream content using the simple text API
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=prompt,
        ):
            if chunk.text:
                on_chunk(chunk.text)
    except Exception as e:
        on_chunk(f"❌ Gemini Error: {str(e)}")


# ==============================================================================
# EFFICIENT MANIM ULTIMATE - MONOLITHIC SOURCE
# ==============================================================================

import requests
import urllib.parse
import shutil
import json
import uuid
import tempfile
import subprocess
import inspect
import traceback
import re
import zipfile
import platform
import numpy as np
from enum import Enum, auto
from pathlib import Path
from datetime import datetime

# Unified keybinding system
try:
    from app.keybinding_registry import KEYBINDINGS, initialize_default_keybindings
    from app.keybindings_panel import UnifiedKeybindingsPanel

    KEYBINDINGS_AVAILABLE = True
except ImportError as e:
    KEYBINDINGS_AVAILABLE = False
    print(f"WARNING: Keybinding modules not found: {e}")

    # Fallback stubs
    class KEYBINDINGS:
        @staticmethod
        def get_binding(name):
            return ""

        def binding_changed():
            pass

        def registry_updated():
            pass

    class UnifiedKeybindingsPanel:
        pass

    def initialize_default_keybindings():
        pass


# PySide6 Imports
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QPlainTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsItem,
    QGraphicsPathItem,
    QLabel,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QPushButton,
    QDialog,
    QFormLayout,
    QComboBox,
    QColorDialog,
    QScrollArea,
    QFrame,
    QMessageBox,
    QMenu,
    QFileDialog,
    QListWidget,
    QListWidgetItem,
    QStyle,
    QSlider,
    QGroupBox,
    QInputDialog,
    QDockWidget,
    QProgressBar,
)
from PySide6.QtCore import (
    Qt,
    Signal,
    QObject,
    QThread,
    QPointF,
    QRectF,
    QTimer,
    QSettings,
    QSize,
    QMimeData,
    QStandardPaths,
    QUrl,
    QLoggingCategory,
)

# Suppress Qt Multimedia / FFmpeg debug and warning messages
QLoggingCategory.setFilterRules("qt.multimedia.*=false")
os.environ["QT_LOGGING_RULES"] = "qt.multimedia.*=false"
from PySide6.QtGui import (
    QAction,
    QColor,
    QPen,
    QBrush,
    QFont,
    QPainter,
    QPixmap,
    QKeySequence,
    QIcon,
    QPainterPath,
    QDrag,
    QTextCursor,
    QImage,
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget


# Manim Import (Safe)
try:
    import manim
    from manim import *
    from manim.utils.color import ManimColor, ParsableManimColor

    MANIM_AVAILABLE = True
except ImportError:
    MANIM_AVAILABLE = False
    print("CRITICAL: Manim library not found. Rendering will be disabled.")

# MCP (Model Context Protocol) — safe lazy import; MCPAgent is instantiated
# after the main window is fully constructed in EfficientManimWindow.__init__.
try:
    from app.mcp import MCPAgent as _MCPAgent

    MCP_AVAILABLE = True
except ImportError:
    _MCPAgent = None  # type: ignore[assignment,misc]
    MCP_AVAILABLE = False
    print("WARNING: mcp.py not found. MCP Agent Mode will be disabled.")

# ==============================================================================
# 1. CORE CONFIGURATION & UTILS
# ==============================================================================

import logging

APP_NAME = "EfficientManim"
APP_VERSION = "2.0.4"
PROJECT_EXT = ".efp"  # EfficientManim Project (Zip)


class AppPaths:
    """Centralized path management."""

    USER_DATA = (
        Path(
            QStandardPaths.writableLocation(
                QStandardPaths.StandardLocation.AppDataLocation
            )
        )
        / "EfficientManim"
    )
    TEMP_DIR = Path(tempfile.gettempdir()) / "EfficientManim_Session"
    FONTS_DIR = Path("fonts")

    @staticmethod
    def ensure_dirs():
        """Ensure all required directories exist, with force cleanup of temp."""
        AppPaths.USER_DATA.mkdir(parents=True, exist_ok=True)

        # Force re-create temp session dir on startup (handles locked files)
        if AppPaths.TEMP_DIR.exists():
            try:
                # Try normal removal first
                shutil.rmtree(AppPaths.TEMP_DIR)
            except PermissionError:
                # If normal removal fails, try to remove individual files and directories
                try:
                    for item in AppPaths.TEMP_DIR.rglob("*"):
                        try:
                            if item.is_file():
                                item.unlink(missing_ok=True)
                            elif item.is_dir():
                                item.rmdir()
                        except (PermissionError, OSError):
                            pass  # Skip locked files
                    # Try to remove main dir if empty
                    try:
                        AppPaths.TEMP_DIR.rmdir()
                    except (PermissionError, OSError):
                        pass
                except Exception as e:
                    print(
                        f"⚠️ Warning: Could not fully clear temp dir (locked files): {e}"
                    )
            except Exception as e:
                print(f"⚠️ Warning: Could not clear temp dir: {e}")

        # Recreate clean temp directory
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
            pass  # Silently ignore cleanup errors


def detect_scene_class(code: str) -> str:
    """
    Detect Scene class name in Python code.

    Args:
        code: Python source code

    Returns:
        Class name of the first Scene subclass found, or "Scene" if none found
    """
    # Try to find class definitions that inherit from Scene
    # Pattern: class ClassName(Scene):
    pattern = r"class\s+(\w+)\s*\(\s*Scene\s*\)"
    matches = re.findall(pattern, code)

    if matches:
        return matches[0]  # Return first matching Scene subclass

    # Fallback: return "Scene" if no subclass found
    return "Scene"


# Theme system - Light mode only, no switching
LIGHT_MODE = True  # Always true


# ==============================================================================
# 2. LOGGING SYSTEM
# ==============================================================================


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


# ==============================================================================
# USER DATA TRACKING (Usage Stats, Recents, Keybindings)
# ==============================================================================


class UserDataManager:
    """Tracks usage statistics, recent projects, and persists user preferences."""

    _BASE_DIR = Path.home() / ".efficientmanim"
    _RECENTS_FILE = _BASE_DIR / "recents.json"
    _USAGE_FILE = _BASE_DIR / "usage.json"
    _KEYBINDINGS_FILE = _BASE_DIR / "keybindings.json"

    def __init__(self):
        self._BASE_DIR.mkdir(parents=True, exist_ok=True)
        self._recents: list = self._load_json(self._RECENTS_FILE, [])
        self._usage: dict = self._load_json(self._USAGE_FILE, {})
        self._keybindings: dict = self._load_json(self._KEYBINDINGS_FILE, {})

    @staticmethod
    def _load_json(path: Path, default):
        try:
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
        return default

    @staticmethod
    def _save_json(path: Path, data) -> None:
        try:
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def add_recent(self, path: str) -> None:
        if path in self._recents:
            self._recents.remove(path)
        self._recents.insert(0, path)
        self._recents = self._recents[:10]
        self._save_json(self._RECENTS_FILE, self._recents)

    def get_recents(self) -> list:
        self._recents = [p for p in self._recents if Path(p).exists()]
        return list(self._recents)

    def record_use(self, class_name: str, node_type: str = "mobject") -> None:
        """Increment counter for class_name under typed bucket (mobject/animation)."""
        bucket = (
            "animation"
            if str(node_type).lower() in ("animation", "anim")
            else "mobject"
        )
        if not isinstance(self._usage, dict):
            self._usage = {}
        typed = self._usage.setdefault(bucket, {})
        typed[class_name] = typed.get(class_name, 0) + 1
        self._save_json(self._USAGE_FILE, self._usage)

    def top_by_type(self, node_type: str, n: int = 5) -> list:
        """Return [(class_name, count), ...] for the top-n in given type bucket."""
        bucket = (
            "animation"
            if str(node_type).lower() in ("animation", "anim")
            else "mobject"
        )
        typed = self._usage.get(bucket, {}) if isinstance(self._usage, dict) else {}
        return sorted(typed.items(), key=lambda kv: kv[1], reverse=True)[:n]

    def top_used(self, n: int = 5) -> list:
        """Legacy: flat list of top-n class names across all types."""
        combined = {}
        if isinstance(self._usage, dict):
            for bucket in ("mobject", "animation"):
                for k, v in self._usage.get(bucket, {}).items():
                    combined[k] = combined.get(k, 0) + v
        return sorted(combined, key=lambda k: combined[k], reverse=True)[:n]

    def get_keybinding(self, action: str, default: str = "") -> str:
        return self._keybindings.get(action, default)

    def set_keybinding(self, action: str, shortcut: str) -> None:
        self._keybindings[action] = shortcut
        self._save_json(self._KEYBINDINGS_FILE, self._keybindings)

    def get_all_keybindings(self) -> dict:
        return dict(self._keybindings)


USER_DATA = UserDataManager()

LOGGER = LogManager()

# ==============================================================================
# 3. SETTINGS MANAGER
# ==============================================================================


class SettingsManager(QObject):
    """Persistent Settings via QSettings."""

    settings_changed = Signal()

    def __init__(self):
        super().__init__()
        self._store = QSettings("Gemini", "EfficientManim")
        self.apply_env()

    # FIX: Added 'type' parameter to handle booleans correctly
    def get(self, key, default=None, type=None):
        val = self._store.value(key, default)
        if type == bool:
            return str(val).lower() == "true"
        if type == int:
            try:
                return int(val)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return default
        if type == float:
            try:
                return float(val)  # type: ignore[arg-type]
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

# ==============================================================================
# 3.5 UNDO/REDO SYSTEM
# ==============================================================================


class UndoRedoAction:
    """Base class for undo/redo actions."""

    def __init__(self, description):
        self.description = description

    def undo(self):
        raise NotImplementedError

    def redo(self):
        raise NotImplementedError


class NodeAddAction(UndoRedoAction):
    """Action for adding a node."""

    def __init__(self, window, node_item, node_data):
        super().__init__(f"Add {node_data.cls_name}")
        self.window = window
        self.node_item = node_item
        self.node_data = node_data

    def undo(self):
        if self.node_data.id in self.window.nodes:
            self.window.remove_node(self.window.nodes[self.node_data.id])

    def redo(self):
        if self.node_data.id not in self.window.nodes:
            self.window.scene.addItem(self.node_item)
            self.window.nodes[self.node_data.id] = self.node_item


class NodeParamChangeAction(UndoRedoAction):
    """Action for changing node parameters."""

    def __init__(self, node_data, key, old_val, new_val):
        super().__init__(f"Change {key}")
        self.node_data = node_data
        self.key = key
        self.old_val = old_val
        self.new_val = new_val

    def undo(self):
        self.node_data.params[self.key] = self.old_val

    def redo(self):
        self.node_data.params[self.key] = self.new_val


class UndoRedoManager:
    """Manages undo/redo history."""

    def __init__(self, max_history=50):
        self.history = []
        self.current_index = -1
        self.max_history = max_history

    def push(self, action):
        """Add action and truncate redo stack."""
        self.history = self.history[: self.current_index + 1]
        self.history.append(action)
        self.current_index += 1
        if len(self.history) > self.max_history:
            self.history.pop(0)
            self.current_index -= 1

    def undo(self):
        if self.current_index >= 0:
            self.history[self.current_index].undo()
            self.current_index -= 1
            return True
        return False

    def redo(self):
        if self.current_index < len(self.history) - 1:
            self.current_index += 1
            self.history[self.current_index].redo()
            return True
        return False

    def can_undo(self):
        return self.current_index >= 0

    def can_redo(self):
        return self.current_index < len(self.history) - 1


# ==============================================================================
# 3.6 ENHANCED LOGGING WITH TIMESTAMPS
# ==============================================================================


class EnhancedLogManager(LogManager):
    """Enhanced logging with file persistence."""

    def __init__(self):
        super().__init__()
        self.log_file = AppPaths.USER_DATA / "session.log"
        self.session_logs = []

    def _write_log(self, level, msg):
        """Write to both memory and file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] {level}: {msg}"
        self.session_logs.append(log_entry)

        try:
            with open(self.log_file, "a") as f:
                f.write(log_entry + "\n")
        except:
            pass

    def info(self, msg):
        self._write_log("INFO", str(msg))
        self.log_signal.emit("INFO", str(msg))

    def warn(self, msg):
        self._write_log("WARN", str(msg))
        self.log_signal.emit("WARN", str(msg))

    def error(self, msg):
        self._write_log("ERROR", str(msg))
        self.log_signal.emit("ERROR", str(msg))

    def manim(self, msg):
        self._write_log("MANIM", str(msg))
        self.log_signal.emit("MANIM", str(msg))

    def ai(self, msg):
        self._write_log("AI", str(msg))
        self.log_signal.emit("AI", str(msg))


# Update LOGGER to use enhanced version
LOGGER = EnhancedLogManager()

# ==============================================================================
# 3.7 THEME MANAGER
# ==============================================================================

# Import unified theme manager - Light mode only
from app.theme.themes import THEME_MANAGER

# Extension system imports
from app.api.extension_api import ExtensionAPI
from app.api.extension_registry import EXTENSION_REGISTRY
from app.api.node_registry import NODE_REGISTRY
from app.api.extension_manager import EXTENSION_MANAGER, PermissionType

# ==============================================================================
# 3.8 KEYBOARD SHORTCUTS REGISTRY
# ==============================================================================


# ══════════════════════════════════════════════════════════════════════════════
# KEYBINDINGS REGISTRY (Unified System)
# Replaced by keybinding_registry.py module — single source of truth
# See keybinding_registry.py for details
# ══════════════════════════════════════════════════════════════════════════════


# ==============================================================================
# 3. SETTINGS MANAGER (continuation)
# ==============================================================================

# ==============================================================================
# 4a. WORKER THREADS
# ==============================================================================


class ImageLoaderWorker(QThread):
    """Loads and scales image in background to keep UI smooth."""

    image_loaded = Signal(QPixmap)

    def __init__(self, path, max_width):
        super().__init__()
        self.path = path
        self.max_width = max_width

    def run(self):
        if not os.path.exists(self.path):
            return

        image = QImage(self.path)
        if not image.isNull():
            # Scale here to save memory on main thread
            pixmap = QPixmap.fromImage(image)
            if pixmap.width() > self.max_width:
                pixmap = pixmap.scaledToWidth(
                    self.max_width, Qt.TransformationMode.SmoothTransformation
                )
            self.image_loaded.emit(pixmap)


class AIWorker(QThread):
    chunk_received = Signal(str)
    finished_signal = Signal()
    error_signal = Signal(str)

    def __init__(self, prompt, model="gemini-3-flash-preview"):
        super().__init__()
        self.prompt = prompt
        self.model = model

    def run(self):
        generate(self.prompt, self.handle_chunk, model=self.model)
        self.finished_signal.emit()

    def handle_chunk(self, text):
        self.chunk_received.emit(text)


# 4b. LaTeX Worker Threads
class LatexApiWorker(QThread):
    """Fetches rendered LaTeX PNG from external API in background."""

    success = Signal(bytes)
    error = Signal(str)

    def __init__(self, latex_str):
        super().__init__()
        self.latex_str = latex_str

    def run(self):
        if not self.latex_str.strip():
            return

        try:
            # Using mathpad.ai as requested
            base = "https://mathpad.ai/api/v1/latex2image"
            params = {
                "latex": self.latex_str,
                "format": "png",
                "scale": 4,  # High quality
            }
            url = f"{base}?{urllib.parse.urlencode(params)}"

            # Timeout is important to prevent hanging
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            self.success.emit(response.content)

        except Exception as e:
            self.error.emit(str(e))


# ==============================================================================
# 4c. TTS WORKER (NEW)
# ==============================================================================


def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    """Helper to generate WAV header for Gemini PCM output."""

    def parse_audio_mime_type(mime_type: str) -> dict:
        bits_per_sample = 16
        rate = 24000
        parts = mime_type.split(";")
        for param in parts:
            param = param.strip()
            if param.lower().startswith("rate="):
                try:
                    rate = int(param.split("=", 1)[1])
                except:
                    pass
            elif param.startswith("audio/L"):
                try:
                    bits_per_sample = int(param.split("L", 1)[1])
                except:
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


class TTSWorker(QThread):
    finished_signal = Signal(str)  # Path to saved file
    error_signal = Signal(str)

    def __init__(self, text, voice_name, model_name):
        super().__init__()
        self.text = text
        self.voice_name = voice_name
        self.model_name = model_name
        self.api_key = os.environ.get("GEMINI_API_KEY")

    def run(self):
        if not self.api_key:
            self.error_signal.emit("API Key missing")
            return

        try:
            client = genai.Client(api_key=self.api_key)

            # Construct Prompt
            contents = self.text  # Simple string prompt for TTS

            # Config
            generate_content_config = types.GenerateContentConfig(
                response_modalities=["audio"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=self.voice_name
                        )
                    )
                ),
            )

            # Stream & Collect
            full_audio = b""
            mime_type = "audio/wav"  # Default

            for chunk in client.models.generate_content_stream(
                model=self.model_name,
                contents=contents,
                config=generate_content_config,
            ):
                try:
                    if (
                        chunk.candidates
                        and chunk.candidates[0].content
                        and chunk.candidates[0].content.parts
                    ):
                        part = chunk.candidates[0].content.parts[0]
                        if part.inline_data and part.inline_data.data is not None:
                            full_audio += bytes(part.inline_data.data)
                            if part.inline_data.mime_type:
                                mime_type = part.inline_data.mime_type
                except Exception:
                    pass

            if not full_audio:
                self.error_signal.emit("No audio data received from Gemini.")
                return

            # Convert if necessary
            final_data = full_audio
            if "wav" not in mime_type:
                final_data = convert_to_wav(full_audio, mime_type)

            # Save to temp
            filename = f"tts_{uuid.uuid4().hex[:8]}.wav"
            save_path = AppPaths.TEMP_DIR / filename

            with open(save_path, "wb") as f:
                f.write(final_data)

            self.finished_signal.emit(str(save_path))

        except Exception as e:
            self.error_signal.emit(str(e))


class RenderWorker(QThread):
    """Fast, single-frame renderer for Node Previews."""

    success = Signal(str, str)  # node_id, absolute_path
    error = Signal(str)

    def __init__(self, script_path, node_id, output_dir, fps, quality):
        super().__init__()
        self.script_path = script_path
        self.node_id = node_id
        self.output_dir = output_dir
        self.quality = quality

    def run(self):
        try:
            print(f"[Preview] Starting render for {self.node_id}...")

            # Ensure output dir exists
            if not self.output_dir.exists():
                self.output_dir.mkdir(parents=True)

            flags = ["-s", "--format=png", "--disable_caching", "-r", "400,300"]
            flags.append(f"-q{self.quality}")

            cmd = ["manim"] + flags + [str(self.script_path), "PreviewScene"]
            print(f"[Preview] Command: {' '.join(cmd)}")

            startupinfo = None
            if platform.system() == "Windows":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.output_dir),
                startupinfo=startupinfo,
            )

            if process.returncode != 0:
                err = process.stderr if process.stderr else process.stdout
                print(f"[Preview] FAILED. Error: {err}")
                self.error.emit(f"Manim Failed: {err[:50]}...")
                return

            print("[Preview] Manim process finished.")

            # Search for the output file
            # Manim structure: /media/images/PreviewScene/PreviewScene_ManimCE_v0.18.1.png
            # Or simplified: /media/images/PreviewScene.png

            media_dir = self.output_dir / "media"
            print(f"[Preview] Searching in: {media_dir}")

            pngs = list(media_dir.rglob("*.png"))
            print(f"[Preview] Found {len(pngs)} PNGs.")

            if pngs:
                latest = max(pngs, key=os.path.getmtime)
                abs_path = str(latest.absolute())
                print(f"[Preview] Success! Path: {abs_path}")
                self.success.emit(self.node_id, abs_path)
            else:
                print("[Preview] Error: No PNGs found after render.")
                self.error.emit("No PNG generated.")

        except Exception as e:
            print(f"[Preview] Exception: {e}")
            self.error.emit(str(e))


class VideoRenderWorker(QThread):
    """Renders full scenes to MP4/WebM video using Manim."""

    progress = Signal(str)  # Status message
    success = Signal(str)  # Output video path
    error = Signal(str)  # Error message

    def __init__(
        self,
        script_path,
        output_dir,
        fps,
        resolution,
        quality,
        scene_class: str = "Scene",
    ):
        super().__init__()
        self.script_path = script_path
        self.output_dir = output_dir
        self.fps = fps
        self.resolution = resolution  # (width, height)
        self.quality = quality  # l, m, h, k
        self.scene_class = scene_class  # Scene class name to render
        self.is_running = True

    def stop_render(self):
        """Request graceful stop and force kill process."""
        self.is_running = False
        # Force kill the external process if it's stuck
        if hasattr(self, "process") and self.process:
            try:
                self.process.kill()
            except:
                pass

    def run(self):
        """Execute video rendering in background thread."""
        try:
            if not self.is_running:
                self.error.emit("Render cancelled.")
                return

            self.progress.emit("Building Manim command...")

            # Copy the flags construction lines from your original file:
            flags = ["--disable_caching", f"-q{self.quality}", f"--fps={self.fps}"]
            if self.resolution:
                w, h = self.resolution
                flags.append(f"--resolution={w},{h}")
            cmd = ["manim"] + flags + [str(self.script_path), self.scene_class]

            self.progress.emit("Starting render...")

            # Windows: hide console
            startupinfo = None
            if platform.system() == "Windows":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            # --- FIX: Assign to self.process so we can kill it later ---
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(self.output_dir),
                startupinfo=startupinfo,
            )

            # Monitor process
            stdout, stderr = self.process.communicate()

            if not self.is_running:
                self.error.emit("Render cancelled by user.")
                return

            if self.process.returncode != 0:
                err_msg = stderr if stderr else stdout
                self.error.emit(f"Manim render failed:\n{err_msg}")
                return

            self.progress.emit("Render complete. Locating output...")
            media_dir = self.output_dir / "media" / "videos"
            video_formats = ["*.mp4", "*.webm"]
            videos = []
            for fmt in video_formats:
                videos.extend(media_dir.rglob(fmt))

            if videos:
                latest_video = max(videos, key=os.path.getmtime)
                self.success.emit(str(latest_video))
            else:
                self.error.emit("Render finished but no video file found.")

        except Exception as e:
            self.error.emit(f"Render error: {str(e)}")


# ==============================================================================
# 5. DATA MODELS
# ==============================================================================


class NodeType(Enum):
    MOBJECT = auto()
    ANIMATION = auto()


class NodeData:
    """Enhanced node data with type safety, parameter metadata, and AI support."""

    def __init__(self, name, n_type, cls_name):
        self.id = str(uuid.uuid4())
        self.name = name
        self.type = n_type
        self.cls_name = cls_name
        self.params = {}  # Key-Value store
        self.param_metadata = {}  # {"param_name": {"enabled": bool, "escape": bool}}
        self.pos_x = 0
        self.pos_y = 0
        self.preview_path = None
        # Voiceover support
        self.audio_asset_id = None
        self.voiceover_transcript: str | None = None
        self.voiceover_duration: float = 0.0
        # AI metadata
        self.is_ai_generated = False
        self.ai_source: str | None = None  # Original Manim class name
        self.ai_code_snippet: str | None = None  # Original code from AI

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.name,
            "cls_name": self.cls_name,
            "params": self.params,
            "param_metadata": self.param_metadata,
            "pos": (self.pos_x, self.pos_y),
            "preview_path": self.preview_path,
            "audio_asset_id": self.audio_asset_id,
            "voiceover_transcript": self.voiceover_transcript,
            "voiceover_duration": self.voiceover_duration,
            "is_ai_generated": self.is_ai_generated,
            "ai_source": self.ai_source,
            "ai_code_snippet": self.ai_code_snippet,
        }

    @staticmethod
    def from_dict(d):
        n = NodeData(d["name"], NodeType[d["type"]], d["cls_name"])
        n.id = d["id"]
        n.params = d["params"]
        n.param_metadata = d.get("param_metadata", {})
        n.pos_x, n.pos_y = d["pos"]
        n.preview_path = d.get("preview_path")
        n.audio_asset_id = d.get("audio_asset_id")
        n.voiceover_transcript = d.get("voiceover_transcript")
        n.voiceover_duration = d.get("voiceover_duration", 0.0)
        n.is_ai_generated = d.get("is_ai_generated", False)
        n.ai_source = d.get("ai_source")
        n.ai_code_snippet = d.get("ai_code_snippet")
        return n

    def is_param_enabled(self, param_name):
        """Check if parameter is enabled (default: True if not set)."""
        return self.param_metadata.get(param_name, {}).get("enabled", True)

    def set_param_enabled(self, param_name, enabled):
        """Enable or disable parameter."""
        if param_name not in self.param_metadata:
            self.param_metadata[param_name] = {}
        self.param_metadata[param_name]["enabled"] = enabled

    def should_escape_string(self, param_name):
        """Check if string should be escaped (default: False)."""
        return self.param_metadata.get(param_name, {}).get("escape", False)

    def set_escape_string(self, param_name, escape):
        """Mark parameter to escape strings."""
        if param_name not in self.param_metadata:
            self.param_metadata[param_name] = {}
        self.param_metadata[param_name]["escape"] = escape


class Asset:
    def __init__(self, name, path, kind):
        self.id = str(uuid.uuid4())
        self.name = name
        # original_path: Where it came from initially (for reference)
        self.original_path = str(Path(path).as_posix())
        # current_path: Where it exists NOW (could be original, or temp/extracted)
        self.current_path = str(Path(path).as_posix())
        self.kind = kind  # "image", "video", "audio"
        self.local_file = ""  # Filename in .efp assets/ folder

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "original": self.original_path,
            # We don't save current_path, because on load we calculate a new one
            "kind": self.kind,
            "local": self.local_file,
        }

    @staticmethod
    def from_dict(d):
        """
        Reconstruct Asset from saved dict.

        CRITICAL FIX: Recalculate current_path on load.

        The saved original_path might be from a different machine/session.
        We need to validate it exists, otherwise look for local_file or fail safely.
        """
        a = Asset(d["name"], d["original"], d["kind"])
        a.id = d["id"]
        a.local_file = d.get("local", "")

        # ═══════════════════════════════════════════════════════════════
        # CRITICAL VALIDATION: Revalidate current_path on deserialization
        # ═══════════════════════════════════════════════════════════════
        original = Path(d["original"])

        # Attempt 1: Original path exists as-is
        if original.exists():
            a.current_path = original.as_posix()
            LOGGER.info(f"Asset '{a.name}' (id={a.id[:8]}): original path valid")
            return a

        # Attempt 2: Local file was extracted to temp
        if a.local_file and (AppPaths.TEMP_DIR / a.local_file).exists():
            a.current_path = (AppPaths.TEMP_DIR / a.local_file).as_posix()
            LOGGER.warn(
                f"Asset '{a.name}' (id={a.id[:8]}): original missing, using temp: {a.current_path}"
            )
            return a

        # Attempt 3: User data assets folder
        user_assets = AppPaths.USER_DATA / "assets" / a.local_file
        if a.local_file and user_assets.exists():
            a.current_path = user_assets.as_posix()
            LOGGER.warn(
                f"Asset '{a.name}' (id={a.id[:8]}): found in user assets: {a.current_path}"
            )
            return a

        # FAILED: No valid path found
        LOGGER.error(
            f"Asset '{a.name}' (id={a.id[:8]}): MISSING - original={a.original_path}, "
            f"local={a.local_file}. Render may fail."
        )
        a.current_path = a.original_path  # Keep original in hopes it reappears
        return a


# ==============================================================================
# 6. GRAPHICS SYSTEM
# ==============================================================================


class SocketItem(QGraphicsPathItem):
    """Port for connecting nodes."""

    def __init__(self, parent, is_output):
        super().__init__(parent)
        self.is_output = is_output
        self.radius = 6
        self.links = []

        # Geometry
        path = QPainterPath()
        path.addEllipse(-self.radius, -self.radius, self.radius * 2, self.radius * 2)
        self.setPath(path)

        # Style
        color = QColor("#2ecc71") if not is_output else QColor("#e74c3c")
        self.setBrush(QBrush(color))
        self.setPen(QPen(Qt.GlobalColor.black, 1))

    def get_scene_pos(self):
        return self.scenePos()


class WireItem(QGraphicsPathItem):
    """Connection between sockets with robust path management."""

    def __init__(self, start_socket, end_socket):
        super().__init__()
        self.start_socket = start_socket
        self.end_socket = end_socket
        if not self.start_socket or not self.end_socket:
            raise ValueError("Both start_socket and end_socket must be valid")

        self.setZValue(-1)

        # Style: curved bezier path
        pen = QPen(QColor("#7f8c8d"), 2.5)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        self.setPen(pen)
        self.update_path()

    def update_path(self):
        """Update wire path based on socket positions."""
        try:
            if not self.start_socket or not self.end_socket:
                return

            p1 = self.start_socket.get_scene_pos()
            p2 = self.end_socket.get_scene_pos()

            # Validate positions are valid QPointF objects
            if not isinstance(p1, QPointF) or not isinstance(p2, QPointF):
                LOGGER.warn("Invalid socket positions for wire update")
                return

            path = QPainterPath()
            path.moveTo(p1)

            dx = p2.x() - p1.x()

            # Create smooth bezier curve
            ctrl1 = QPointF(p1.x() + dx * 0.5, p1.y())
            ctrl2 = QPointF(p2.x() - dx * 0.5, p2.y())

            path.cubicTo(ctrl1, ctrl2, p2)
            self.setPath(path)
        except Exception as e:
            LOGGER.warn(f"Error updating wire path: {e}")


class NodeItem(QGraphicsItem):
    """Visual representation of NodeData."""

    def __init__(self, data: NodeData):
        super().__init__()
        self.node_data = data  # Internal storage
        self._init_geometry()

    # ── Transparent proxy so that node.data.X works everywhere ──────────────
    @property
    def data(self) -> NodeData:  # type: ignore[override]
        return self.node_data

    @data.setter
    def data(self, value: NodeData):
        self.node_data = value

    def _init_geometry(self):
        """Initialize geometry, sockets, and flags (called at end of __init__)."""
        self.width = 180
        self.height = 90

        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setPos(self.node_data.pos_x, self.node_data.pos_y)

        # Sockets
        self.in_socket = SocketItem(self, False)
        self.in_socket.setPos(0, self.height / 2)

        self.out_socket = SocketItem(self, True)
        self.out_socket.setPos(self.width, self.height / 2)

    def boundingRect(self):
        return QRectF(0, 0, self.width, self.height)

    def paint(self, painter, option, widget):
        # Shadow
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 30))
        painter.drawRoundedRect(4, 4, self.width, self.height, 8, 8)

        # Body
        painter.setBrush(QColor("white"))
        pen = QPen(QColor("#bdc3c7"), 1.5)
        if self.isSelected():
            pen = QPen(QColor("#3498db"), 2.5)
        painter.setPen(pen)
        painter.drawRoundedRect(0, 0, self.width, self.height, 8, 8)

        # Header
        header_h = 28
        header_color = (
            QColor("#3498db")
            if self.node_data.type == NodeType.MOBJECT
            else QColor("#9b59b6")
        )

        path = QPainterPath()
        path.addRoundedRect(0, 0, self.width, header_h, 8, 8)
        # Clip
        painter.setClipPath(path)
        painter.fillPath(path, header_color)
        painter.setClipping(False)

        # Text
        painter.setPen(Qt.GlobalColor.white)
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        name = self.node_data.name
        painter.drawText(
            QRectF(8, 0, self.width - 16, header_h),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            name,
        )

        # Class Name
        painter.setPen(QColor("#7f8c8d"))
        painter.setFont(QFont("Segoe UI", 9))
        cls_name = self.node_data.cls_name
        painter.drawText(
            QRectF(8, 35, self.width - 16, 20), Qt.AlignmentFlag.AlignLeft, cls_name
        )

        # Indicator
        preview_path = self.node_data.preview_path
        if preview_path:
            painter.setBrush(QColor("#2ecc71"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(self.width - 16, self.height - 16, 8, 8)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            self.node_data.pos_x = value.x()
            self.node_data.pos_y = value.y()
            if hasattr(self, "in_socket") and hasattr(self, "out_socket"):
                for w in self.in_socket.links + self.out_socket.links:
                    w.update_path()
            if self.scene():
                s = self.scene()
                s and hasattr(s, "notify_change") and s.notify_change()  # type: ignore[union-attr]
        return super().itemChange(change, value)


class GraphScene(QGraphicsScene):
    """Manages node connections and logic enforcement."""

    selection_changed_signal = Signal()
    graph_changed_signal = Signal()  # Structure changed

    def __init__(self):
        super().__init__()
        self.drag_wire = None
        self.start_socket = None
        self.setBackgroundBrush(QBrush(QColor("#f4f6f7")))

    def notify_change(self):
        self.graph_changed_signal.emit()

    def mousePressEvent(self, event):
        item = self.itemAt(event.scenePos(), self.views()[0].transform())

        if isinstance(item, SocketItem):
            self.start_socket = item
            self.drag_wire = QGraphicsPathItem()
            self.drag_wire.setPen(QPen(Qt.GlobalColor.black, 2, Qt.PenStyle.DashLine))
            self.addItem(self.drag_wire)
            return

        super().mousePressEvent(event)
        self.selection_changed_signal.emit()

    def mouseMoveEvent(self, event):
        if self.drag_wire and self.start_socket:
            p1 = self.start_socket.get_scene_pos()
            p2 = event.scenePos()
            path = QPainterPath()
            path.moveTo(p1)
            path.lineTo(p2)
            self.drag_wire.setPath(path)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.drag_wire:
            self.removeItem(self.drag_wire)
            self.drag_wire = None

            end_item = self.itemAt(event.scenePos(), self.views()[0].transform())

            if isinstance(end_item, SocketItem) and end_item != self.start_socket:
                self.try_connect(self.start_socket, end_item)

            self.start_socket = None

        super().mouseReleaseEvent(event)

    def try_connect(self, s1, s2):
        """Attempt to create a connection between two sockets with validation.

        Args:
            s1: Source socket
            s2: Target socket

        Returns:
            WireItem or None: Created wire if successful, None if validation failed
        """
        # Validation: Cannot connect socket to itself
        if s1 == s2:
            self.show_warning(
                "Invalid Connection", "Cannot connect a socket to itself."
            )
            return None

        # Validation: Must connect output to input
        if s1.is_output == s2.is_output:
            direction = "both outputs" if s1.is_output else "both inputs"
            self.show_warning(
                "Invalid Connection",
                f"Cannot connect {direction}. Please connect output to input.",
            )
            return None

        if s1.is_output:
            out_sock, in_sock = s1, s2
        else:
            out_sock, in_sock = s2, s1

        node_src = out_sock.parentItem()
        node_dst = in_sock.parentItem()

        # Validation: Check node types for valid connections
        # 1. Mobject -> Mobject (INVALID)
        if (
            node_src.node_data.type == NodeType.MOBJECT
            and node_dst.node_data.type == NodeType.MOBJECT
        ):
            self.show_warning(
                "Invalid Connection",
                "Directly connecting Mobjects is not allowed.\nPlease insert an Animation node in between.",
            )
            return None

        # 2. Animation -> Animation (INVALID for now)
        if (
            node_src.node_data.type == NodeType.ANIMATION
            and node_dst.node_data.type == NodeType.ANIMATION
        ):
            self.show_warning(
                "Invalid Connection",
                "Chaining animations directly is not supported.\nTarget a Mobject instead.",
            )
            return None

        # Create Wire with error handling
        try:
            wire = WireItem(out_sock, in_sock)
            self.addItem(wire)
            out_sock.links.append(wire)
            in_sock.links.append(wire)

            self.notify_change()
            LOGGER.info(
                f"Connected {node_src.node_data.name} -> {node_dst.node_data.name}"
            )
            return wire
        except Exception as e:
            LOGGER.error(f"Failed to create connection: {e}")
            self.show_warning(
                "Connection Error", f"Failed to create connection: {str(e)}"
            )
            return None

    def show_warning(self, title, msg):
        views = self.views()
        if views:
            QMessageBox.warning(views[0], title, msg)


# ==============================================================================
# 7. ASSET MANAGEMENT
# ==============================================================================


class AssetManager(QObject):
    assets_changed = Signal()

    def __init__(self):
        super().__init__()
        self.assets = {}  # id -> Asset

    def add_asset(self, path):
        path_obj = Path(path)
        if not path_obj.exists():
            return

        # Normalize path for Manim (Forward Slashes)
        clean_path = path_obj.resolve().as_posix()

        kind = "unknown"
        s = path_obj.suffix.lower()
        if s in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
            kind = "image"
        elif s in [".svg"]:
            kind = "image"  # SVG is treated as image/vector
        elif s in [".mp4", ".mov", ".avi", ".webm"]:
            kind = "video"
        elif s in [".mp3", ".wav", ".ogg"]:
            kind = "audio"

        if kind == "unknown":
            LOGGER.error(f"Unsupported asset type: {s}")
            return

        asset = Asset(path_obj.name, clean_path, kind)
        self.assets[asset.id] = asset
        self.assets_changed.emit()
        LOGGER.info(f"Asset added: {asset.name}")
        return asset

    def clear(self):
        self.assets.clear()
        self.assets_changed.emit()

    def get_list(self):
        return list(self.assets.values())

    def get_asset_path(self, asset_id):
        """
        Safely retrieve the absolute POSIX path for Manim.

        CRITICAL FIX: Validate file exists before returning.
        Logs error if asset file is missing so user knows render will fail.
        """
        if asset_id in self.assets:
            asset = self.assets[asset_id]

            # ══════════════════════════════════════════════════════════════
            # VALIDATION: Confirm file exists before returning
            # This prevents silent render failures
            # ══════════════════════════════════════════════════════════════
            path_str = asset.current_path
            path_obj = Path(path_str)

            if not path_obj.exists():
                LOGGER.error(
                    f"Asset file missing: '{asset.name}' (id={asset_id[:8]}) → {path_str}"
                )
                return None

            return path_str

        LOGGER.error(f"Unknown asset ID: {asset_id}")
        return None


ASSETS = AssetManager()

# ==============================================================================
# 8. PANELS & UI MODULES
# ==============================================================================


class TypeSafeParser:
    """Comprehensive type validation and safe parsing for Manim parameters."""

    # Parameter category mapping
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
    def is_asset_param(param_name):
        """Check if parameter expects a file/asset."""
        n = param_name.lower()
        # Specific fix for ImageMobject
        if "filename" in n:
            return True

        # General file keywords
        if "file" in n or "image" in n or "sound" in n or "svg" in n:
            # Exclude false positives like "fill_opacity" or "profile"
            if "fill" in n or "profile" in n:
                return False
            return True
        return False

    @staticmethod
    def is_numeric_param(param_name):
        """Check if parameter should be numeric."""
        if TypeSafeParser.is_asset_param(param_name):
            return False
        return any(kw in param_name.lower() for kw in TypeSafeParser.NUMERIC_KEYWORDS)

    @staticmethod
    def is_color_param(param_name):
        """Check if parameter should be a color."""
        return any(kw in param_name.lower() for kw in TypeSafeParser.COLOR_KEYWORDS)

    @staticmethod
    def is_point_param(param_name):
        """Check if parameter should be points/coordinates."""
        return any(kw in param_name.lower() for kw in TypeSafeParser.POINT_KEYWORDS)

    @staticmethod
    def parse_numeric(value, default=0.0):
        """Safely parse value as number with robust validation.

        Args:
            value: Value to parse (int, float, str, or other)
            default: Default value if parsing fails

        Returns:
            float: Parsed numeric value or default
        """
        try:
            if value is None:
                return default
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                # Try parsing
                val = float(value.strip())
                if -1e10 < val < 1e10:  # Sanity check
                    return val
                LOGGER.warn(
                    f"Numeric value out of range: {value}, using default {default}"
                )
                return default
            # Fallback
            return float(value) if value else default
        except (ValueError, TypeError) as e:
            LOGGER.warn(
                f"Invalid numeric value '{value}': {type(e).__name__}, using default {default}"
            )
            return default

    @staticmethod
    def parse_color(value, default_hex="#FFFFFF"):
        """Safely parse value as ManimColor with comprehensive validation.

        Supports: ManimColor objects, hex strings, Manim constants, RGB tuples, and QColor objects.

        Args:
            value: Color value in various formats
            default_hex: Default hex color if parsing fails

        Returns:
            str: Hex color string (#RRGGBB format)
        """
        try:
            if value is None:
                return default_hex

            # Already ManimColor
            if hasattr(value, "to_hex") or (
                hasattr(value, "__class__") and "ManimColor" in type(value).__name__
            ):
                return value.to_hex() if hasattr(value, "to_hex") else str(value)

            # String handling
            if isinstance(value, str):
                value_clean = value.strip()

                # Try Manim color constants first (e.g., "BLUE", "RED", "CYAN")
                try:
                    if hasattr(manim, value_clean.upper()):
                        manim_color = getattr(manim, value_clean.upper())
                        if (
                            hasattr(manim_color, "to_hex")
                            or hasattr(manim_color, "_internal_value")
                            or "ManimColor" in type(manim_color).__name__
                        ):
                            return str(manim_color)
                except:
                    pass

                # Hex string
                if value_clean.startswith("#") and len(value_clean) == 7:
                    try:
                        int(value_clean[1:], 16)  # Validate hex
                        return value_clean
                    except ValueError:
                        pass

                # Try QColor parsing
                qc = QColor(value_clean)
                if qc.isValid():
                    return qc.name()

                return default_hex

            # RGB tuple/list
            if isinstance(value, (tuple, list)) and len(value) >= 3:
                r, g, b = value[:3]
                # Validate range
                if isinstance(r, float) and 0 <= r <= 1:
                    r, g, b = int(r * 255), int(g * 255), int(b * 255)
                return f"#{int(r):02x}{int(g):02x}{int(b):02x}"

            # QColor object
            if isinstance(value, QColor) and value.isValid():
                return value.name()

        except Exception as e:
            LOGGER.warn(f"Color parsing error for '{value}': {type(e).__name__}")

        return default_hex

    @staticmethod
    def to_manim_color(hex_color):
        """Convert hex color to ManimColor for rendering."""
        try:
            if hasattr(hex_color, "to_hex") or "ManimColor" in type(hex_color).__name__:
                return hex_color
            if isinstance(hex_color, str):
                return ManimColor(hex_color) if MANIM_AVAILABLE else hex_color
            return ManimColor("#FFFFFF") if MANIM_AVAILABLE else "#FFFFFF"
        except:
            return ManimColor("#FFFFFF") if MANIM_AVAILABLE else "#FFFFFF"

    @staticmethod
    def validate_point_safe(point, default=None):
        """Validate point for safe multiplication."""
        if default is None:
            default = np.array([0.0, 0.0, 0.0])

        try:
            if point is None:
                return default

            # If it's already ndarray, return
            if isinstance(point, np.ndarray):
                return point.astype(float)

            # Convert to numpy array
            arr = np.array(point, dtype=float)
            if arr.shape[0] in [2, 3]:
                if arr.shape[0] == 2:
                    return np.array([arr[0], arr[1], 0.0])
                return arr
            return default
        except:
            return default


class ColorNormalizer:
    """Centralized color normalization utility."""

    @staticmethod
    def normalize_to_hex(value):
        """Convert any color representation to hex string."""
        return TypeSafeParser.parse_color(value)


class SceneOutlinerPanel(QWidget):
    """Lists all nodes in the scene for easy selection and management."""

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setup_ui()

        # Connect to scene changes to auto-refresh
        self.main_window.scene.graph_changed_signal.connect(self.refresh_list)

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Search Bar
        self.search = QLineEdit()
        self.search.setPlaceholderText("🔍 Filter nodes...")
        self.search.textChanged.connect(self.refresh_list)
        layout.addWidget(self.search)

        # The List
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.list_widget.itemClicked.connect(self.on_item_clicked)
        # Context menu for deleting
        self.list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self.show_context_menu)
        layout.addWidget(self.list_widget)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_del = QPushButton("🗑️ Delete")
        btn_del.clicked.connect(self.delete_selected)
        btn_layout.addWidget(btn_del)

        btn_refresh = QPushButton("🔄 Refresh")
        btn_refresh.clicked.connect(self.refresh_list)
        btn_layout.addWidget(btn_refresh)

        layout.addLayout(btn_layout)

    def refresh_list(self):
        """Rebuild the list based on current nodes."""
        self.list_widget.clear()
        filter_text = self.search.text().lower()

        # Sort nodes by creation order (roughly) or name
        nodes = list(self.main_window.nodes.values())

        for node in nodes:
            # Filter
            if filter_text and filter_text not in node.node_data.name.lower():
                continue

            # Create Item
            icon_char = "📦" if node.node_data.type == NodeType.MOBJECT else "🎬"
            display_text = (
                f"{icon_char} {node.node_data.name} ({node.node_data.cls_name})"
            )

            item = QListWidgetItem(display_text)
            item.setData(Qt.ItemDataRole.UserRole, node.node_data.id)

            # Highlight if selected in scene
            if node.isSelected():
                item.setBackground(QColor("#d6eaf8"))  # Light blue

            self.list_widget.addItem(item)

    def on_item_clicked(self, item):
        """Select the node in the scene when clicked in the list."""
        node_id = item.data(Qt.ItemDataRole.UserRole)
        if node_id in self.main_window.nodes:
            # Deselect all first
            for n in self.main_window.nodes.values():
                n.setSelected(False)

            # Select target
            node = self.main_window.nodes[node_id]
            node.setSelected(True)

            # Focus view on node
            self.main_window.view.centerOn(node)
            self.main_window.on_selection()  # Trigger property panel update

    def delete_selected(self):
        """Delete nodes selected in the list."""
        ids_to_delete = []
        for item in self.list_widget.selectedItems():
            ids_to_delete.append(item.data(Qt.ItemDataRole.UserRole))

        if not ids_to_delete:
            return

        reply = QMessageBox.question(
            self,
            "Delete",
            f"Delete {len(ids_to_delete)} items?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            for nid in ids_to_delete:
                if nid in self.main_window.nodes:
                    self.main_window.remove_node(self.main_window.nodes[nid])
            self.refresh_list()
            self.main_window.compile_graph()

    def show_context_menu(self, pos):
        menu = QMenu()
        del_act = menu.addAction("Delete")
        action = menu.exec(self.list_widget.mapToGlobal(pos))
        if action == del_act:
            self.delete_selected()


class LatexEditorPanel(QWidget):
    """Panel for Live LaTeX editing via API."""

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.temp_preview_path = AppPaths.TEMP_DIR / "latex_preview.png"

        # FIX: Keep track of all active threads to prevent Garbage Collection crashes
        self._active_workers = set()

        self.setup_ui()

        # Debouncer timer (Auto-update logic)
        self.debouncer = QTimer()
        self.debouncer.setSingleShot(True)
        self.debouncer.setInterval(800)  # Wait 800ms after typing stops
        self.debouncer.timeout.connect(self.trigger_render)

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Header
        header = QLabel("✒️ LaTeX Studio (Online)")
        header.setStyleSheet("font-weight: bold; font-size: 14px; color: #2c3e50;")
        layout.addWidget(header)

        # Editor
        self.editor = QPlainTextEdit()
        self.editor.setPlaceholderText("Enter LaTeX here... e.g. E = mc^2")
        self.editor.setStyleSheet("font-family: Consolas; font-size: 12pt;")
        self.editor.setMaximumHeight(100)
        self.editor.textChanged.connect(self.on_text_changed)
        layout.addWidget(self.editor)

        # Preview Area
        self.preview_lbl = QLabel("Preview will appear here...")
        self.preview_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_lbl.setStyleSheet(
            "background: white; border: 2px dashed #bdc3c7; border-radius: 4px;"
        )
        self.preview_lbl.setMinimumHeight(150)

        # Scroll area
        scroll = QScrollArea()
        scroll.setWidget(self.preview_lbl)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        # Controls Group
        ctrl_group = QFrame()
        ctrl_group.setStyleSheet(
            "background: #ecf0f1; border-radius: 4px; padding: 4px;"
        )
        ctrl_layout = QVBoxLayout(ctrl_group)

        # Target Node Selector
        self.node_combo = QComboBox()
        self.node_combo.setPlaceholderText("Select Target Node...")
        ctrl_layout.addWidget(QLabel("Apply to Node:"))
        ctrl_layout.addWidget(self.node_combo)

        # Refresh Button
        btn_refresh = QPushButton("🔄 Refresh Node List")
        btn_refresh.clicked.connect(self.refresh_nodes)
        ctrl_layout.addWidget(btn_refresh)

        # Apply Button
        self.btn_apply = QPushButton("✅ Apply to Node")
        self.btn_apply.setStyleSheet(
            "background-color: #27ae60; color: white; font-weight: bold; padding: 6px;"
        )
        self.btn_apply.clicked.connect(self.apply_to_node)
        ctrl_layout.addWidget(self.btn_apply)

        layout.addWidget(ctrl_group)

        # Status
        self.status_lbl = QLabel("Ready")
        self.status_lbl.setStyleSheet("color: gray;")
        layout.addWidget(self.status_lbl)

    def on_text_changed(self):
        """Called on every keystroke."""
        self.status_lbl.setText("Typing...")
        self.debouncer.start()  # Reset timer

    def trigger_render(self):
        """Called by timer to start API call."""
        tex = self.editor.toPlainText().strip()
        if not tex:
            return

        self.status_lbl.setText("Fetching render from API...")

        # FIX: Create a new worker and track it
        worker = LatexApiWorker(tex)
        worker.success.connect(self.on_render_success)
        worker.error.connect(self.on_render_error)

        # FIX: Clean up reference when done
        worker.finished.connect(lambda: self._cleanup_worker(worker))

        # Add to active set so Python doesn't delete it while it's running
        self._active_workers.add(worker)
        worker.start()

    def _cleanup_worker(self, worker):
        """Remove worker from active set to allow Garbage Collection."""
        if worker in self._active_workers:
            self._active_workers.remove(worker)
        worker.deleteLater()

    def on_render_success(self, image_data):
        self.status_lbl.setText("Render received.")

        try:
            with open(self.temp_preview_path, "wb") as f:
                f.write(image_data)
        except Exception as e:
            LOGGER.warn(f"Could not cache LaTeX png: {e}")

        pixmap = QPixmap()
        pixmap.loadFromData(image_data)

        if pixmap.width() > self.preview_lbl.width():
            pixmap = pixmap.scaledToWidth(
                self.preview_lbl.width() - 20,
                Qt.TransformationMode.SmoothTransformation,
            )

        self.preview_lbl.setPixmap(pixmap)
        self.preview_lbl.setText("")

    def on_render_error(self, err_msg):
        self.status_lbl.setText("API Error.")
        # Only show error if it's not just a cancellation
        if "Network" in err_msg or "400" in err_msg:
            self.preview_lbl.setText(f"❌ Error:\n{err_msg}")
            self.preview_lbl.setPixmap(QPixmap())

    def refresh_nodes(self):
        """Populate combo with nodes that look like they accept text."""
        current = self.node_combo.currentData()
        self.node_combo.clear()

        count = 0
        for nid, node in self.main_window.nodes.items():
            cname = node.data.cls_name.lower()
            if "tex" in cname or "text" in cname or "label" in cname:
                # FIX: Format as "Name (First 6 chars of ID)"
                short_id = nid[:6]
                display_text = f"{node.data.name} ({short_id})"
                self.node_combo.addItem(display_text, nid)
                count += 1

        if current:
            idx = self.node_combo.findData(current)
            if idx >= 0:
                self.node_combo.setCurrentIndex(idx)

        self.status_lbl.setText(f"Found {count} text-compatible nodes.")

    def apply_to_node(self):
        """Inject the LaTeX code into the selected node."""
        node_id = self.node_combo.currentData()
        if not node_id or node_id not in self.main_window.nodes:
            QMessageBox.warning(self, "Error", "Please select a valid target node.")
            return

        tex_code = self.editor.toPlainText().strip()
        if not tex_code:
            return

        # Balance parentheses
        open_count = 0
        balanced_tex = ""
        for char in tex_code:
            if char == "(":
                open_count += 1
                balanced_tex += char
            elif char == ")":
                if open_count > 0:
                    open_count -= 1
                    balanced_tex += char
            else:
                balanced_tex += char
        balanced_tex += ")" * open_count

        node = self.main_window.nodes[node_id]

        # Cleanup conflicting params
        for k in ("tex_strings", "arg0", "arg1", "t"):
            node.data.params.pop(k, None)

        # Escape backslashes
        safe_tex = balanced_tex.replace("\\", "\\\\")

        # Use raw string literal format
        formatted_code = f'r"""{safe_tex}"""'

        target_param = "text"
        if node.data.cls_name == "MathTex":
            target_param = "arg0"
        elif node.data.cls_name == "Text":
            target_param = "text"

        node.data.params[target_param] = formatted_code

        # Configure Metadata
        node.data.set_escape_string(target_param, True)
        node.data.set_param_enabled(target_param, True)

        node.update()
        self.main_window.compile_graph()
        self.status_lbl.setText(f"Applied to {node.data.name}!")

        node.setSelected(True)
        self.main_window.on_selection()

        # Trigger re-render
        self.main_window.queue_render(node)


class VideoOutputPanel(QWidget):
    """Integrated Video Player with Seek and Play/Pause controls."""

    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.duration = 0

        # Init Player
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)

        self.player.setVideoOutput(self.video_widget)

        # Connections
        self.player.positionChanged.connect(self.on_position_changed)
        self.player.durationChanged.connect(self.on_duration_changed)
        self.player.mediaStatusChanged.connect(self.on_media_status_changed)

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header_bar = QFrame()
        header_bar.setStyleSheet(
            "background: #2c3e50; border-bottom: 1px solid #34495e;"
        )
        header_layout = QHBoxLayout(header_bar)
        header_layout.setContentsMargins(10, 5, 10, 5)

        lbl = QLabel("🎥 Output Monitor")
        lbl.setStyleSheet("color: white; font-weight: bold;")
        header_layout.addWidget(lbl)
        layout.addWidget(header_bar)

        # Video Area
        self.video_widget = QVideoWidget()
        self.video_widget.setStyleSheet("background-color: black;")
        layout.addWidget(self.video_widget, 1)  # Expandable

        # Controls Area
        controls = QFrame()
        controls.setStyleSheet("background: #ecf0f1; border-top: 1px solid #bdc3c7;")
        ctrl_layout = QHBoxLayout(controls)
        ctrl_layout.setContentsMargins(10, 5, 10, 5)

        # Play/Pause Button
        self.btn_play = QPushButton("▶")
        self.btn_play.setFixedSize(30, 30)
        self.btn_play.clicked.connect(self.toggle_play)
        ctrl_layout.addWidget(self.btn_play)

        # Slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self.set_position)
        ctrl_layout.addWidget(self.slider)

        # Time Label
        self.lbl_time = QLabel("00:00 / 00:00")
        self.lbl_time.setStyleSheet("font-family: monospace;")
        ctrl_layout.addWidget(self.lbl_time)

        layout.addWidget(controls)

    def load_video(self, file_path, autoplay=True):
        """Load a video file and optionally start playing."""
        self.player.setSource(QUrl.fromLocalFile(file_path))
        self.audio_output.setVolume(1.0)  # 100% volume
        if autoplay:
            self.player.play()
            self.btn_play.setText("⏸")

    def toggle_play(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.Playing:
            self.player.pause()
            self.btn_play.setText("▶")
        else:
            self.player.play()
            self.btn_play.setText("⏸")

    def on_position_changed(self, position):
        """Update slider as video plays."""
        if not self.slider.isSliderDown():
            self.slider.setValue(position)
        self.update_time_label(position)

    def on_duration_changed(self, duration):
        """Update slider range when video loads."""
        self.duration = duration
        self.slider.setRange(0, duration)

    def set_position(self, position):
        """User dragged slider."""
        self.player.setPosition(position)

    def update_time_label(self, current_ms):
        def fmt(ms):
            seconds = (ms // 1000) % 60
            minutes = ms // 60000
            return f"{minutes:02}:{seconds:02}"

        self.lbl_time.setText(f"{fmt(current_ms)} / {fmt(self.duration)}")

    def on_media_status_changed(self, status):
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.btn_play.setText("▶")


class GraphView(QGraphicsView):
    """Custom View with Zoom (Ctrl+Wheel) and Pan (Middle Mouse) support."""

    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._is_panning = False

    def wheelEvent(self, event):
        """Handle Zoom with Ctrl + Scroll."""
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            zoom_factor = 1.15
            if event.angleDelta().y() > 0:
                self.scale(zoom_factor, zoom_factor)
            else:
                self.scale(1 / zoom_factor, 1 / zoom_factor)
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event):
        """Handle Panning with Middle Mouse Button."""
        if event.button() == Qt.MouseButton.MiddleButton:
            self._is_panning = True
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            # Create a dummy event to initiate the drag immediately
            dummy = type(event)(event)
            super().mousePressEvent(dummy)
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._is_panning = False
            self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts including Ctrl+A for select all."""
        if (
            event.key() == Qt.Key.Key_A
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            # Select all nodes in the scene
            if hasattr(self.scene(), "selectAllItems"):
                self.scene().selectAllItems()
            else:
                # Fallback: manually select all items
                for item in self.scene().items():
                    if hasattr(item, "setSelected"):
                        item.setSelected(True)
            event.accept()
        else:
            super().keyPressEvent(event)


class PropertiesPanel(QWidget):
    """Enhanced inspector with type safety, parameter validation, and metadata columns."""

    node_updated = Signal()

    def __init__(self):
        super().__init__()
        self.current_node = None
        self._vbox_layout = QVBoxLayout(self)

        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.form_widget = QWidget()
        self.form = QFormLayout(self.form_widget)
        self._scroll_area.setWidget(self.form_widget)

        self._vbox_layout.addWidget(self._scroll_area)

        # Debouncer for update signals
        self.debouncer = QTimer()
        self.debouncer.setSingleShot(True)
        self.debouncer.setInterval(500)
        self.debouncer.timeout.connect(self.node_updated.emit)

        # Track active widgets for safe cleanup
        self.active_widgets = {}

    def set_node(self, node_item: "NodeItem | None"):
        """Load node properties into inspector with full type safety."""
        self.current_node = node_item

        # Clean up previous widgets
        for widget in self.active_widgets.values():
            try:
                widget.deleteLater()
            except:
                pass
        self.active_widgets.clear()

        while self.form.count():
            child = self.form.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        if not node_item:
            return

        # FIX: Validate types, but SKIP asset parameters to prevent UUID->0.0 corruption
        for k, v in list(node_item.node_data.params.items()):
            if TypeSafeParser.is_asset_param(k):
                continue  # Do not touch asset strings/UUIDs
            elif TypeSafeParser.is_color_param(k):
                node_item.node_data.params[k] = TypeSafeParser.parse_color(v)
            elif TypeSafeParser.is_numeric_param(k):
                node_item.node_data.params[k] = TypeSafeParser.parse_numeric(v)

        # ===== AI GENERATED INDICATOR =====
        if node_item.node_data.is_ai_generated:
            ai_label = QLabel("✨ AI GENERATED NODE ✨")
            font = QFont()
            font.setPointSize(10)
            ai_label.setFont(font)
            ai_label.setStyleSheet(
                "background: linear-gradient(90deg, #e3f2fd, #f3e5f5); "
                "color: #1565c0; padding: 6px; border-radius: 3px; "
                "border-left: 3px solid #1565c0;"
            )
            ai_label.setToolTip(
                f"This node was generated by Gemini AI.\n"
                f"Class: {node_item.node_data.ai_source or node_item.node_data.cls_name}\n"
                f"All parameters are available for editing."
            )
            self.form.addRow(ai_label)

        # Meta Information
        self.form.addRow(QLabel("<b>Properties</b>"))
        id_lbl = QLabel(node_item.node_data.id[:8])
        id_lbl.setStyleSheet("color: gray;")
        self.form.addRow("ID", id_lbl)

        name_edit = QLineEdit(node_item.node_data.name)
        name_edit.textChanged.connect(lambda t: self.update_param("_name", t))
        self.form.addRow("Name", name_edit)

        if not MANIM_AVAILABLE:
            self.form.addRow(QLabel("Manim not loaded."))
            return

        try:
            cls = getattr(manim, node_item.node_data.cls_name, None)
            if not cls:
                return

            sig = inspect.signature(cls.__init__)

            # Auto-load missing parameters
            for param_name, param in sig.parameters.items():
                if param_name in ("self", "args", "kwargs", "mobject"):
                    continue

                if param_name not in node_item.node_data.params:
                    if param.default is inspect.Parameter.empty:
                        default_val = param.default
                        if TypeSafeParser.is_color_param(param_name):
                            default_val = TypeSafeParser.parse_color(default_val)
                        elif TypeSafeParser.is_numeric_param(param_name):
                            default_val = TypeSafeParser.parse_numeric(default_val)
                        node_item.node_data.params[param_name] = default_val

            # Create Rows
            for name, param in sig.parameters.items():
                if name in ["self", "args", "kwargs", "mobject"]:
                    continue

                # Check default value
                val = node_item.node_data.params.get(name, param.default)
                if val is inspect.Parameter.empty:
                    val = None

                # FIX: Disable 'tex_strings' by default if not explicitly set
                if name == "tex_strings":
                    # Only disable if it wasn't manually enabled by user previously
                    if name not in node_item.node_data.param_metadata:
                        node_item.node_data.set_param_enabled(name, False)

                row_widget = self.create_parameter_row(name, val, param.annotation)
                # FIX: Actually add the row to the form layout
                if row_widget:
                    self.form.addRow(name, row_widget)

        except Exception as e:
            LOGGER.error(f"Inspector Error for {node_item.node_data.cls_name}: {e}")
            traceback.print_exc()

    def create_parameter_row(self, key, value, annotation):
        """Create a parameter row with main value, State checkbox, and Escape String checkbox."""
        try:
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)

            # Main value widget
            value_widget = self.create_typed_widget(key, value, annotation)
            if not value_widget:
                return None

            row_layout.addWidget(value_widget, 3)

            # State checkbox (Enable/Disable parameter)
            state_chk = QCheckBox("Enabled")
            state_chk.setToolTip(
                "Check to include in code generation (Ctrl+E to toggle)"
            )
            is_enabled = self.current_node.node_data.is_param_enabled(key)
            state_chk.setChecked(is_enabled)
            state_chk.stateChanged.connect(
                lambda s: (
                    self.current_node.node_data.set_param_enabled(key, s == 2)
                    if self.current_node
                    else None
                )
            )
            row_layout.addWidget(state_chk, 1)

            # Escape String checkbox
            escape_chk = QCheckBox("Escape")
            escape_chk.setToolTip("Check to escape strings (remove quotes)")
            should_escape = self.current_node.data.should_escape_string(key)
            escape_chk.setChecked(should_escape)
            escape_chk.stateChanged.connect(
                lambda s: (
                    self.current_node.data.set_escape_string(key, s == 2)
                    if self.current_node
                    else None
                )
            )
            row_layout.addWidget(escape_chk, 1)

            return row_widget

        except Exception as e:
            LOGGER.error(f"Error creating parameter row for {key}: {e}")
            return None

    def create_typed_widget(self, key, value, annotation):
        """Create typed widget with safe type enforcement and validation."""

        def on_change(v):
            if not self.current_node:
                return
            try:
                if key == "_name":
                    self.current_node.data.name = str(v)
                else:
                    # FIX: Strict order of operations. Check Asset FIRST.
                    if TypeSafeParser.is_asset_param(key):
                        # Store the UUID string directly. Do NOT parse as number.
                        self.current_node.data.params[key] = v
                    elif TypeSafeParser.is_color_param(key):
                        v = TypeSafeParser.parse_color(v)
                        self.current_node.data.params[key] = v
                    elif TypeSafeParser.is_numeric_param(key):
                        v = TypeSafeParser.parse_numeric(v)
                        self.current_node.data.params[key] = v
                    else:
                        self.current_node.data.params[key] = v

                self.current_node.update()
                self.debouncer.start()
            except Exception as e:
                LOGGER.warn(f"Value change error for {key}: {e}")

        try:
            #  1. SPECIAL: Target Mobject Selector
            # Check for keys like "mobject", "vmobject", "mobjects"
            key_lower = key.lower()
            target_keywords = ["mobject", "vmobject", "target", "object"]

            # Logic: If it's an Animation node AND key contains one of the keywords
            is_anim = self.current_node.data.type == NodeType.ANIMATION
            is_target_param = any(k in key_lower for k in target_keywords)

            if is_anim and is_target_param:
                combo = QComboBox()
                combo.addItem("-- Select Target --", None)

                # Scan main window nodes for Mobjects
                # Note: We need a way to access main_window nodes.
                # Ideally pass main_window to PropertiesPanel or access via scene items.
                scene = self.current_node.scene()
                if scene:
                    for item in scene.items():
                        if (
                            isinstance(item, NodeItem)
                            and item.data.type == NodeType.MOBJECT
                        ):
                            # Add Mobject to dropdown
                            display_name = f"📦 {item.data.name} ({item.data.id[:4]})"
                            combo.addItem(display_name, item.data.id)

                # Set current value
                if value:
                    idx = combo.findData(value)
                    if idx >= 0:
                        combo.setCurrentIndex(idx)

                # AUTO-CONFIGURE: Enable and Escape by default
                self.current_node.data.set_param_enabled(key, True)
                self.current_node.data.set_escape_string(key, True)

                combo.currentIndexChanged.connect(
                    lambda i: on_change(combo.itemData(i))
                )
                return combo

            # 2. ASSET PATHS
            if TypeSafeParser.is_asset_param(key):
                combo = QComboBox()
                combo.addItem("(None)", None)
                for asset in ASSETS.get_list():
                    # Show Emoji + Name
                    combo.addItem(f"📄 {asset.name}", asset.id)

                if value:
                    idx = combo.findData(value)
                    if idx >= 0:
                        combo.setCurrentIndex(idx)

                combo.currentIndexChanged.connect(
                    lambda i: on_change(combo.itemData(i))
                )
                return combo

            # 3. COLORS
            if TypeSafeParser.is_color_param(key):
                btn = QPushButton("Pick Color")
                _hex_val = str(TypeSafeParser.parse_color(value))
                btn.setStyleSheet(f"background-color: {_hex_val}; color: white;")

                def _make_color_picker(_btn, _initial_hex):
                    def pick_color():
                        col = QColorDialog.getColor(
                            QColor(_initial_hex), None, "Select Color"
                        )
                        if col.isValid():
                            new_hex = col.name()
                            _btn.setStyleSheet(
                                f"background-color: {new_hex}; color: white;"
                            )
                            on_change(new_hex)

                    return pick_color

                btn.clicked.connect(_make_color_picker(btn, _hex_val))
                return btn

            # 4. NUMERIC
            if TypeSafeParser.is_numeric_param(key) or annotation in (float, int):
                # ... (Keep existing numeric logic) ...
                if annotation == float or isinstance(value, float):
                    sb = QDoubleSpinBox()
                    sb.setRange(-10000.0, 10000.0)
                    sb.setSingleStep(0.1)
                    sb.setValue(TypeSafeParser.parse_numeric(value))
                    sb.valueChanged.connect(on_change)
                    return sb
                else:
                    sb = QSpinBox()
                    sb.setRange(-10000, 10000)
                    sb.setValue(int(TypeSafeParser.parse_numeric(value)))
                    sb.valueChanged.connect(on_change)
                    return sb

            # 5. BOOLEAN
            if annotation == bool or isinstance(value, bool):
                chk = QCheckBox()
                chk.setChecked(bool(value))
                chk.stateChanged.connect(lambda s: on_change(s == 2))
                return chk

            # 6. STRING / FALLBACK
            str_val = str(value) if value is not None else ""
            le = QLineEdit(str_val)
            le.textChanged.connect(on_change)
            return le

        except Exception as e:
            LOGGER.error(f"Widget creation error for '{key}': {e}")
            return None

    def update_param(self, key, val):
        """Update a parameter safely."""
        if self.current_node:
            try:
                if key == "_name":
                    self.current_node.data.name = str(val)
                self.current_node.update()
            except Exception as e:
                LOGGER.warn(f"update_param error: {e}")


class ElementsPanel(QWidget):
    add_requested = Signal(str, str)  # Type, Class

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search...")
        self.search.textChanged.connect(self.filter)
        layout.addWidget(self.search)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemDoubleClicked.connect(self.on_dbl_click)
        layout.addWidget(self.tree)
        self.populate()

    def populate(self):
        if not MANIM_AVAILABLE:
            return
        self.tree.clear()
        mob_root = QTreeWidgetItem(self.tree, ["Mobjects"])
        anim_root = QTreeWidgetItem(self.tree, ["Animations"])

        # Add built-in Manim elements
        for name in dir(manim):
            if name.startswith("_"):
                continue
            obj = getattr(manim, name)
            if inspect.isclass(obj):
                try:
                    if issubclass(obj, manim.Mobject) and obj is not manim.Mobject:
                        QTreeWidgetItem(mob_root, [name])
                    elif (
                        issubclass(obj, manim.Animation) and obj is not manim.Animation
                    ):
                        QTreeWidgetItem(anim_root, [name])
                except:
                    pass
        
        mob_root.setExpanded(True)

        # Extension nodes from NODE_REGISTRY
        try:
            from app.api.node_registry import NODE_REGISTRY
            from PySide6.QtCore import Qt
            extension_nodes = NODE_REGISTRY.get_nodes()
            for category, node_defs in extension_nodes.items():
                cat_root = QTreeWidgetItem(self.tree, [f"📦 {category}"])
                cat_root.setToolTip(0, f"Extension category: {category}")
                for node_def in node_defs:
                    item = QTreeWidgetItem(cat_root, [node_def.node_name])
                    item.setToolTip(0, node_def.description)
                    item.setData(0, Qt.ItemDataRole.UserRole, {
                        "type": "extension_node",
                        "class_path": node_def.class_path,
                        "category": category,
                    })
                cat_root.setExpanded(True)
        except Exception as _e:
            import logging
            logging.getLogger("elements_panel").warning(f"Could not load extension nodes: {_e}")

    def filter(self, txt):
        root = self.tree.invisibleRootItem()
        txt = txt.lower()
        for i in range(root.childCount()):
            cat = root.child(i)
            hide = True
            for j in range(cat.childCount()):
                item = cat.child(j)
                if txt in item.text(0).lower():
                    item.setHidden(False)
                    hide = False
                else:
                    item.setHidden(True)
            cat.setHidden(hide)

    def on_dbl_click(self, item, col):
        if item.childCount() == 0:
            from PySide6.QtCore import Qt
            p = item.parent()
            parent_text = p.text(0) if p else ""

            # Check if this is an extension node
            ext_data = item.data(0, Qt.ItemDataRole.UserRole)
            if isinstance(ext_data, dict) and ext_data.get("type") == "extension_node":
                # Extension node: emit with class_path so the graph knows how to instantiate
                self.add_requested.emit("ExtensionNode", ext_data["class_path"])
                return

            # Determine type based on parent category
            if parent_text == "Mobjects":
                t = "Mobject"
            elif parent_text == "Animations":
                t = "Animation"
            else:
                # Default to trying as Mobject
                t = "Mobject"
            
            self.add_requested.emit(t, item.text(0))


class VideoRenderPanel(QWidget):
    """Panel for rendering scenes to video."""

    render_requested = Signal(dict)

    def __init__(self):
        super().__init__()
        self.render_worker = None
        self.setup_ui()

    def setup_ui(self):
        # NOTE: This is a QWidget. It uses addLayout/addWidget.
        layout = QVBoxLayout(self)

        title = QLabel("🎬 Video Render")
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)

        form = QFormLayout()

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(15, 60)
        self.fps_spin.setValue(30)
        form.addRow("Frame Rate:", self.fps_spin)

        res_layout = QHBoxLayout()
        self.width_spin = QSpinBox()
        self.width_spin.setRange(320, 3840)
        self.width_spin.setValue(1280)
        self.width_spin.setSingleStep(160)
        res_layout.addWidget(QLabel("Width:"))
        res_layout.addWidget(self.width_spin)

        self.height_spin = QSpinBox()
        self.height_spin.setRange(180, 2160)
        self.height_spin.setValue(720)
        self.height_spin.setSingleStep(90)
        res_layout.addWidget(QLabel("Height:"))
        res_layout.addWidget(self.height_spin)
        form.addRow("Resolution:", res_layout)

        self.quality_combo = QComboBox()
        self.quality_combo.addItems(
            ["Low (ql)", "Medium (qm)", "High (qh)", "Ultra (qk)"]
        )
        self.quality_combo.setCurrentIndex(1)
        form.addRow("Quality:", self.quality_combo)

        layout.addLayout(form)

        path_layout = QHBoxLayout()
        self.output_path_lbl = QLineEdit()
        self.output_path_lbl.setReadOnly(True)
        self.output_path_lbl.setText(str(AppPaths.TEMP_DIR))
        path_layout.addWidget(QLabel("Output:"))
        path_layout.addWidget(self.output_path_lbl)

        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.browse_output)
        path_layout.addWidget(self.browse_btn)
        layout.addLayout(path_layout)

        ctrl_layout = QHBoxLayout()
        self.render_scene_btn = QPushButton("📽 Render Full Scene")
        self.render_scene_btn.setStyleSheet(
            "background-color: #27ae60; color: white; padding: 8px;"
        )
        self.render_scene_btn.clicked.connect(self.render_full_scene)
        ctrl_layout.addWidget(self.render_scene_btn)

        self.cancel_btn = QPushButton("⏹ Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setStyleSheet(
            "background-color: #e74c3c; color: white; padding: 8px;"
        )
        self.cancel_btn.clicked.connect(self.cancel_render)
        ctrl_layout.addWidget(self.cancel_btn)

        layout.addLayout(ctrl_layout)

        self.status_display = QTextEdit()
        self.status_display.setReadOnly(True)
        self.status_display.setMaximumHeight(120)
        layout.addWidget(QLabel("Status:"))
        layout.addWidget(self.status_display)

        layout.addStretch()

    # ... (Keep existing browse_output, render_full_scene, update_status, etc.) ...
    def browse_output(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select Output", str(AppPaths.TEMP_DIR)
        )
        if path:
            self.output_path_lbl.setText(path)

    def render_full_scene(self):
        output_path = Path(self.output_path_lbl.text())
        if not output_path.exists():
            return
        qual = ["l", "m", "h", "k"][self.quality_combo.currentIndex()]
        config = {
            "fps": self.fps_spin.value(),
            "resolution": (self.width_spin.value(), self.height_spin.value()),
            "quality": qual,
            "output_path": str(output_path),
        }
        self.render_requested.emit(config)

    def cancel_render(self):
        if self.render_worker:
            self.render_worker.stop_render()

    def start_rendering(self, worker):
        self.render_worker = worker
        self.render_scene_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        worker.progress.connect(lambda m: self.update_status(m, "blue"))
        worker.success.connect(self.on_render_success)
        worker.error.connect(self.on_render_error)

    def on_render_success(self, path):
        self.render_scene_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.update_status(f"Done: {Path(path).name}", "green")

    def on_render_error(self, err):
        self.render_scene_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.update_status(f"Error: {err}", "red")

    def update_status(self, msg, col):
        self.status_display.append(f"<span style='color:{col}'>{msg}</span>")


class AIPanel(QWidget):
    """Enhanced AI Panel with visual distinction and node generation."""

    merge_requested = Signal(str)
    nodes_generated = Signal(dict)  # Emits dict of generated node info

    def __init__(self):
        super().__init__()
        self.worker = None
        self.last_code = None
        self.extracted_nodes = []  # Track AI-generated nodes
        self._mcp_agent = None  # Set by EfficientManimWindow after construction

        # Create main layout with visual distinction
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # ===== HEADER =====
        header = self._create_header()
        main_layout.addWidget(header)

        # ===== CONTENT AREA =====
        content_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left side: Prompt + Response
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)

        # Prompt section
        prompt_group = self._create_prompt_section()
        left_layout.addWidget(prompt_group)

        # Response section (scrollable)
        response_group = self._create_response_section()
        left_layout.addWidget(response_group)

        # Control buttons
        button_layout = self._create_button_layout()
        left_layout.addLayout(button_layout)

        content_splitter.addWidget(left_panel)
        content_splitter.setCollapsible(0, False)

        # Right side: Generated Nodes Preview
        right_panel = self._create_nodes_preview()
        content_splitter.addWidget(right_panel)

        content_splitter.setSizes([600, 300])
        content_splitter.setCollapsible(1, True)

        main_layout.addWidget(content_splitter)

    def _create_header(self) -> QWidget:
        """Create visually distinct header."""
        header = QFrame()
        header.setStyleSheet(
            "QFrame { background: linear-gradient(90deg, #1e88e5 0%, #1565c0 100%); "
            "border-bottom: 2px solid #0d47a1; }"
        )
        header.setMaximumHeight(50)

        layout = QHBoxLayout(header)
        layout.setContentsMargins(12, 8, 12, 8)

        title = QLabel("🤖 AI Code Generator")
        font = QFont()
        font.setPointSize(11)
        font.setBold(True)
        title.setFont(font)
        title.setStyleSheet("color: white; font-weight: bold;")

        subtitle = QLabel("Generate Manim code with Gemini AI")
        subtitle.setStyleSheet("color: rgba(255,255,255,0.7); font-size: 9pt;")

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addStretch()

        status_label = QLabel("Status: Ready")
        status_label.setStyleSheet("color: rgba(255,255,255,0.8);")
        self.status_label = status_label
        layout.addWidget(status_label)

        return header

    def _create_prompt_section(self) -> QFrame:
        """Create prompt input section with styling."""
        frame = QFrame()
        frame.setStyleSheet(
            "QFrame { border: 1px solid #e0e0e0; border-radius: 4px; "
            "background: #f5f5f5; }"
        )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 8, 8, 8)

        label = QLabel("📝 Your Prompt")
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        label.setFont(font)
        label.setStyleSheet("color: #1565c0;")
        layout.addWidget(label)

        self.input = QPlainTextEdit()
        self.input.setPlaceholderText(
            "Describe your animation...\n\nExample: Create a blue rectangle that smoothly rotates 100 degrees."
        )
        self.input.setMaximumHeight(90)
        self.input.setStyleSheet(
            "QPlainTextEdit { border: 1px solid #bdbdbd; border-radius: 3px; "
            "background: white; padding: 6px; }"
        )
        layout.addWidget(self.input)

        # ── Auto Voiceover toggle ──────────────────────────────────
        self.chk_auto_voiceover = QCheckBox("🎙️ Enable Auto Voiceover")
        self.chk_auto_voiceover.setToolTip(
            "When enabled, Gemini will act as an autonomous agent:\n"
            "it will analyze all animation nodes, extract meaningful text,\n"
            "generate optimized voiceover scripts, generate TTS audio per\n"
            "animation, and automatically attach voiceovers to nodes.\n"
            "The result is a fully voiceover-synced project ready to render."
        )
        self.chk_auto_voiceover.setStyleSheet("font-weight: bold; color: #8e44ad;")
        layout.addWidget(self.chk_auto_voiceover)

        # ── MCP Agent Mode toggle ──────────────────────────────────
        self.chk_mcp_mode = QCheckBox("🔌 MCP Agent Mode")
        self.chk_mcp_mode.setToolTip(
            "Instead of generating Manim code, Gemini reads the live scene\n"
            "state and issues MCP commands to directly create/modify/delete\n"
            "nodes in the graph — no code generation, no merge step needed.\n\n"
            "Gemini sees all current nodes, params, and assets before acting."
        )
        self.chk_mcp_mode.setStyleSheet("font-weight: bold; color: #1a73e8;")
        # Only one mode can be active at once
        self.chk_mcp_mode.toggled.connect(
            lambda on: self.chk_auto_voiceover.setEnabled(not on)
        )
        self.chk_auto_voiceover.toggled.connect(
            lambda on: self.chk_mcp_mode.setEnabled(not on)
        )
        layout.addWidget(self.chk_mcp_mode)

        return frame

    def _create_response_section(self) -> QFrame:
        """Create response display section with scrolling."""
        frame = QFrame()
        frame.setStyleSheet(
            "QFrame { border: 1px solid #e0e0e0; border-radius: 4px; "
            "background: #fafafa; }"
        )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 8, 8, 8)

        label = QLabel("💬 AI Response")
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        label.setFont(font)
        label.setStyleSheet("color: #1565c0;")
        layout.addWidget(label)

        # Scrollable output area
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setStyleSheet(
            "QTextEdit { border: 1px solid #bdbdbd; border-radius: 3px; "
            "background: white; padding: 6px; }"
        )
        layout.addWidget(self.output)

        return frame

    def _create_button_layout(self) -> QHBoxLayout:
        """Create control buttons with visual distinction."""
        layout = QHBoxLayout()

        # Generate button (primary)
        self.btn_gen = QPushButton("⚡ Generate Code")
        self.btn_gen.setStyleSheet(
            "QPushButton { background: #2196f3; color: white; border: none; "
            "border-radius: 4px; padding: 8px 16px; font-weight: bold; }"
            "QPushButton:hover { background: #1976d2; }"
            "QPushButton:pressed { background: #1565c0; }"
        )
        self.btn_gen.clicked.connect(self.generate)
        layout.addWidget(self.btn_gen, 1)

        # Merge button (success)
        self.btn_merge = QPushButton("✅ Merge to Scene")
        self.btn_merge.setEnabled(False)
        self.btn_merge.setStyleSheet(
            "QPushButton { background: #4caf50; color: white; border: none; "
            "border-radius: 4px; padding: 8px 16px; font-weight: bold; }"
            "QPushButton:hover { background: #45a049; }"
            "QPushButton:disabled { background: #bdbdbd; }"
        )
        self.btn_merge.clicked.connect(self.merge)
        layout.addWidget(self.btn_merge, 1)

        # Reject button (danger)
        self.btn_reject = QPushButton("❌ Reject")
        self.btn_reject.setEnabled(False)
        self.btn_reject.setStyleSheet(
            "QPushButton { background: #f44336; color: white; border: none; "
            "border-radius: 4px; padding: 8px 16px; font-weight: bold; }"
            "QPushButton:hover { background: #da190b; }"
            "QPushButton:disabled { background: #bdbdbd; }"
        )
        self.btn_reject.clicked.connect(self.reject)
        layout.addWidget(self.btn_reject, 1)

        # Clear button (secondary)
        self.btn_clear = QPushButton("🗑️ Clear")
        self.btn_clear.setStyleSheet(
            "QPushButton { background: #9e9e9e; color: white; border: none; "
            "border-radius: 4px; padding: 8px 16px; }"
            "QPushButton:hover { background: #757575; }"
        )
        self.btn_clear.clicked.connect(lambda: self.output.clear())
        layout.addWidget(self.btn_clear, 1)

        return layout

    def _create_nodes_preview(self) -> QFrame:
        """Create preview panel for generated nodes."""
        frame = QFrame()
        frame.setStyleSheet(
            "QFrame { border-left: 2px solid #1565c0; background: #f0f4ff; }"
        )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 8, 8, 8)

        title = QLabel("🎬 Generated Nodes")
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        title.setFont(font)
        title.setStyleSheet("color: #1565c0;")
        layout.addWidget(title)

        # Nodes list (scrollable)
        self.nodes_list = QListWidget()
        self.nodes_list.setStyleSheet(
            "QListWidget { border: 1px solid #e0e0e0; border-radius: 3px; "
            "background: white; }"
            "QListWidget::item { padding: 6px; border-bottom: 1px solid #f0f0f0; }"
            "QListWidget::item:hover { background: #e3f2fd; }"
        )
        layout.addWidget(self.nodes_list)

        self.extracted_nodes = []

        return frame

    def generate(self):
        """Generate AI code, run Auto Voiceover agent, or execute MCP Agent Mode."""
        if self.chk_auto_voiceover.isChecked():
            self._run_auto_voiceover_agent()
            return

        if self.chk_mcp_mode.isChecked():
            self._run_mcp_agent_mode()
            return

        txt = self.input.toPlainText().strip()
        if not txt:
            self.status_label.setText("Status: Empty prompt")
            return

        self.output.clear()
        self.nodes_list.clear()
        self.extracted_nodes = []

        self.output.append(f"<b style='color: #1565c0;'>👤 USER:</b> {txt}\n")
        self.btn_gen.setEnabled(False)
        self.status_label.setText("Status: Generating...")
        self.input.clear()

        # Get selected model from settings
        selected_model = str(
            SETTINGS.get("GEMINI_MODEL", "gemini-3-flash-preview")
            or "gemini-3-flash-preview"
        )
        self.output.append(f"<b style='color: #666;'>🤖 Model:</b> {selected_model}\n")

        sys_prompt = (
            f"You are a Manim expert. Generate production-ready Python Manim code following these STRICT rules:\n\n"
            f"USER REQUEST: {txt}\n\n"
            f"MANDATORY RULES:\n"
            f"1. Preserve ALL node properties exactly (color, fill_opacity, stroke_width, etc.)\n"
            f"2. Generate proper animations using self.play() - NEVER bare self.add() for animated content\n"
            f"3. Include animations: FadeIn, FadeOut, Rotate, Transform, MoveTo, Scale as appropriate\n"
            f"4. Use readable variable names: triangle_1, circle_2 (NOT generic m_xxxxx)\n"
            f"5. Import only what's needed: from manim import *\n"
            f"6. Add comments explaining each animation step in natural language\n"
            f"7. Output ONLY Python code, no explanations\n"
            f"8. Ensure correct order: FadeIn nodes → animate transformations → FadeOut nodes\n"
            f"9. Make every action explicit - no summaries or abstractions\n"
            f"10. Output fully runnable code with NO syntax errors\n\n"
            f"OUTPUT FORMAT:\n"
            f"```python\n"
            f"from manim import *\n\n"
            f"class MyScene(Scene):\n"
            f"    def construct(self):\n"
            f"        # Create objects with exact properties\n"
            f"        obj_1 = ClassName(property=value, ...)\n"
            f"        # Animate in\n"
            f"        self.play(FadeIn(obj_1))\n"
            f"        # Transform/animate\n"
            f"        self.play(Animation(obj_1))\n"
            f"        # Animate out\n"
            f"        self.play(FadeOut(obj_1))\n"
            f"```\n\n"
            f"Generate ONLY the code block. No text before or after.\n"
        )

        # Get selected model and create worker
        selected_model = str(SETTINGS.get("GEMINI_MODEL", "gemini-3-flash-preview"))
        self.worker = AIWorker(sys_prompt, model=selected_model)
        self.worker.chunk_received.connect(self.on_chunk)
        self.worker.finished_signal.connect(self.on_finish)
        self.worker.start()

    def on_chunk(self, txt):
        """Handle streamed AI response."""
        self.output.moveCursor(QTextCursor.MoveOperation.End)
        self.output.insertPlainText(txt)

    def on_finish(self):
        """Parse generated code and extract nodes."""
        self.btn_gen.setEnabled(True)
        full = self.output.toPlainText()
        match = re.findall(r"```python(.*?)```", full, re.DOTALL)

        if match:
            self.last_code = match[-1].strip()
            self._extract_nodes_from_code(self.last_code)

            if self.extracted_nodes:
                self.btn_merge.setEnabled(True)
                self.btn_reject.setEnabled(True)
                self.status_label.setText(
                    f"Status: Ready to merge ({len(self.extracted_nodes)} nodes)"
                )
                LOGGER.ai(
                    f"Code block ready. Extracted {len(self.extracted_nodes)} nodes."
                )
            else:
                self.status_label.setText("Status: Code ready (no nodes detected)")
        else:
            self.status_label.setText("Status: No code block found")

    def _extract_nodes_from_code(self, code: str):
        """Extract BOTH mobjects and animations from AI-generated code."""
        self.extracted_nodes = []
        self.nodes_list.clear()

        # Extract all object definitions
        pattern = r"(\w+)\s*=\s*([A-Z][a-zA-Z0-9]*)\s*\((.*?)\)(?:\s|$)"
        matches = re.finditer(pattern, code, re.DOTALL)

        for match in matches:
            var_name, class_name, params_str = match.groups()

            # Skip if not a Manim class
            if not hasattr(manim, class_name):
                continue

            # Extract and parse ALL parameters
            params = self._parse_node_parameters(params_str)
            param_count = len(params)

            # Determine if it's an animation
            is_animation = False
            try:
                if hasattr(manim, class_name):
                    cls = getattr(manim, class_name)
                    is_animation = (
                        issubclass(cls, manim.Animation)
                        if hasattr(manim, "Animation")
                        else "Animation" in class_name
                    )
            except:
                pass

            self.extracted_nodes.append(
                {
                    "var_name": var_name,
                    "class_name": class_name,
                    "params": params,
                    "params_str": params_str,
                    "source": "ai",
                    "type": "animation" if is_animation else "mobject",
                }
            )

            # Add to list with icon indicating type
            node_icon = "🎬" if is_animation else "📦"
            node_type_label = "Animation" if is_animation else "Mobject"
            item = QListWidgetItem(
                f"{node_icon} {var_name}: {class_name} ({param_count} params) [{node_type_label}]"
            )

            # Color code: blue for mobjects, purple for animations
            if is_animation:
                item.setBackground(QColor("#f3e5f5"))
                item.setForeground(QColor("#7b1fa2"))
            else:
                item.setBackground(QColor("#e3f2fd"))
                item.setForeground(QColor("#1565c0"))

            # Build detailed tooltip with parameter list
            param_list = "\n".join(
                [f"  • {k}={v[:40]}" for k, v in list(params.items())[:5]]
            )
            if len(params) > 5:
                param_list += f"\n  ... and {len(params) - 5} more"

            item.setToolTip(
                f"AI Generated {node_type_label}\n"
                f"Variable: {var_name}\n"
                f"Type: {class_name}\n"
                f"Parameters ({param_count}):\n{param_list}\n\n"
                f"All parameters are captured and ready to use."
            )
            self.nodes_list.addItem(item)

        # Also show animations from self.play() calls
        pattern_play = r"self\.play\((.*?)\)(?=\s|$)"
        for match in re.finditer(pattern_play, code, re.DOTALL):
            play_content = match.group(1)
            anim_pattern = r"([A-Z][a-zA-Z0-9]*)\((.*?)\)"

            for anim_match in re.finditer(anim_pattern, play_content):
                anim_class = anim_match.group(1)

                # Skip if not a Manim animation class
                if not hasattr(manim, anim_class):
                    continue

                try:
                    cls = getattr(manim, anim_class)
                    is_anim = (
                        issubclass(cls, manim.Animation)
                        if hasattr(manim, "Animation")
                        else "Animation" in anim_class
                    )

                    if is_anim and not any(
                        n["class_name"] == anim_class and n["type"] == "animation"
                        for n in self.extracted_nodes
                    ):
                        # Add this animation if not already added
                        self.extracted_nodes.append(
                            {
                                "var_name": f"{anim_class.lower()}_1",
                                "class_name": anim_class,
                                "params": {},
                                "source": "ai",
                                "type": "animation",
                            }
                        )

                        item = QListWidgetItem(
                            f"🎬 {anim_class.lower()}_1: {anim_class} (from self.play)"
                        )
                        item.setBackground(QColor("#f3e5f5"))
                        item.setForeground(QColor("#7b1fa2"))
                        item.setToolTip(
                            f"Animation: {anim_class}\nExtracted from self.play() call"
                        )
                        self.nodes_list.addItem(item)
                except:
                    pass

    def _parse_node_parameters(self, params_str: str) -> dict:
        """Parse parameters from node definition string, handling nested structures."""
        params = {}

        # Split by comma, but respect parentheses/brackets nesting
        depth = 0
        current_item = ""

        for char in params_str:
            if char in "([{":
                depth += 1
                current_item += char
            elif char in ")]}":
                depth -= 1
                current_item += char
            elif char == "," and depth == 0:
                # End of parameter
                if "=" in current_item:
                    try:
                        key, value = current_item.split("=", 1)
                        key = key.strip()
                        value = value.strip()
                        # Store raw value (will be cleaned during merge)
                        params[key] = value
                    except ValueError:
                        pass
                current_item = ""
            else:
                current_item += char

        # Don't forget last item
        if current_item and "=" in current_item:
            try:
                key, value = current_item.split("=", 1)
                key = key.strip()
                value = value.strip()
                params[key] = value
            except ValueError:
                pass

        return params

    def _run_auto_voiceover_agent(self):
        """Gemini auto-voiceover agent: analyze all nodes, generate and attach voiceovers.

        This is the 'Enable Auto Voiceover' mode. Gemini acts as an autonomous
        agent that:
          1. Inspects all animation nodes in the scene.
          2. Extracts meaningful text / context from each node.
          3. Generates an optimized voiceover script per node.
          4. Generates TTS audio for each script via TTSWorker.
          5. Attaches audio to each node and syncs durations.
        """
        main_window = self._get_main_window()
        if main_window is None:
            QMessageBox.critical(self, "Error", "Cannot find main window reference.")
            return

        nodes: dict = main_window.nodes
        if not nodes:
            QMessageBox.warning(
                self, "No Nodes", "There are no nodes in the scene to voiceover."
            )
            return

        self.output.clear()
        self.btn_gen.setEnabled(False)
        self.status_label.setText("Status: Auto Voiceover Agent running…")
        self.output.append("<b style='color:#8e44ad;'>🤖 Auto Voiceover Agent</b><br>")
        self.output.append(f"Analyzing <b>{len(nodes)}</b> nodes…<br>")

        # Build node context for Gemini
        node_summaries = []
        for nid, node_item in nodes.items():
            d = node_item.data
            params_str = (
                ", ".join(f"{k}={v}" for k, v in list(d.params.items())[:5])
                if d.params
                else "no params"
            )
            node_summaries.append(
                f"  - [{d.type.name}] {d.name} ({d.cls_name}): {params_str}"
            )

        node_context = "\n".join(node_summaries)

        agent_prompt = (
            "You are an expert animation narrator and voiceover script writer.\n\n"
            "Here is a list of animation nodes in a Manim scene:\n\n"
            f"{node_context}\n\n"
            "Your job:\n"
            "For EACH node listed above, write a short, engaging voiceover script (1-2 sentences max).\n"
            "The script should naturally describe what the animation is doing.\n"
            "Use plain, spoken English — no markdown, no code.\n\n"
            "Output ONLY a JSON array, with one object per node, in this exact format:\n"
            "[\n"
            '  {"node_name": "<exact node name>", "script": "<voiceover script>"},\n'
            "  ...\n"
            "]\n\n"
            "Output ONLY the JSON array. No explanation, no markdown fences."
        )

        selected_model = str(SETTINGS.get("GEMINI_MODEL", "gemini-3-flash-preview"))
        self._auto_vo_nodes = nodes
        self._auto_vo_worker = AIWorker(agent_prompt, model=selected_model)
        self._auto_vo_buffer = ""
        self._auto_vo_worker.chunk_received.connect(self._on_auto_vo_chunk)
        self._auto_vo_worker.finished_signal.connect(self._on_auto_vo_scripts_ready)
        self._auto_vo_worker.start()

    def _on_auto_vo_chunk(self, txt: str):
        self._auto_vo_buffer += txt
        self.output.moveCursor(QTextCursor.MoveOperation.End)
        self.output.insertPlainText(txt)

    def _on_auto_vo_scripts_ready(self):
        """Parse Gemini's JSON response and kick off TTS generation per node."""
        raw = self._auto_vo_buffer.strip()

        # Strip markdown fences if present
        raw = re.sub(r"```json|```", "", raw).strip()

        try:
            scripts: list[dict] = json.loads(raw)
        except json.JSONDecodeError as e:
            self.btn_gen.setEnabled(True)
            self.status_label.setText("Status: Agent parse error.")
            LOGGER.error(f"Auto Voiceover agent parse error: {e}\nRaw: {raw[:200]}")
            self.output.append(
                "<br><span style='color:red'>❌ Failed to parse scripts from Gemini. "
                "Check Logs for details.</span>"
            )
            return

        if not scripts:
            self.btn_gen.setEnabled(True)
            self.status_label.setText("Status: Agent returned empty scripts.")
            return

        self.output.append(
            f"<br><b>✅ Scripts generated for {len(scripts)} nodes.</b> Starting TTS…<br>"
        )
        self._auto_vo_queue = list(scripts)
        self._auto_vo_node_map = {
            node_item.data.name: node_item for node_item in self._auto_vo_nodes.values()
        }
        self._auto_vo_voice = SETTINGS.get("DEFAULT_VOICE", "Zephyr")
        self._auto_vo_model = SETTINGS.get("TTS_MODEL", "gemini-2.5-flash-preview-tts")
        self._auto_vo_active_worker = None
        self._auto_vo_index = 0
        self._process_next_auto_vo()

    def _process_next_auto_vo(self):
        """Process TTS generation for the next item in the queue."""
        if not self._auto_vo_queue:
            self._finish_auto_voiceover()
            return

        item = self._auto_vo_queue.pop(0)
        node_name = item.get("node_name", "")
        script = item.get("script", "")
        self._auto_vo_index += 1

        if not script or node_name not in self._auto_vo_node_map:
            self.output.append(
                f"<span style='color:orange'>⚠️ Skipping '{node_name}' — "
                f"{'no script' if not script else 'node not found'}</span><br>"
            )
            QTimer.singleShot(100, self._process_next_auto_vo)
            return

        self.output.append(
            f"<span style='color:#1a73e8'>🎙 [{self._auto_vo_index}] {node_name}</span>: "
            f"{script}<br>"
        )
        self.status_label.setText(f"Status: TTS for '{node_name}'…")

        worker = TTSWorker(script, self._auto_vo_voice, self._auto_vo_model)
        # Store metadata on the worker for the callback to access
        worker._target_node_name = node_name
        worker._script = script
        worker.finished_signal.connect(
            lambda path, n=node_name, s=script: self._on_auto_vo_tts_done(path, n, s)
        )
        worker.error_signal.connect(
            lambda err, n=node_name: self._on_auto_vo_tts_error(err, n)
        )
        self._auto_vo_active_worker = worker
        worker.start()

    def _on_auto_vo_tts_done(self, file_path: str, node_name: str, script: str):
        """Attach generated audio to target node and process the next one."""
        asset = ASSETS.add_asset(file_path)
        node_item = self._auto_vo_node_map.get(node_name)

        if asset and node_item:
            node_item.data.audio_asset_id = asset.id
            node_item.data.voiceover_transcript = script
            # Estimate duration from pydub if available
            if PYDUB_AVAILABLE:
                try:
                    from pydub import AudioSegment as _AS

                    seg = _AS.from_file(file_path)
                    node_item.data.voiceover_duration = len(seg) / 1000.0
                except Exception:
                    pass
            node_item.update()
            self.output.append(
                f"<span style='color:#27ae60'>✅ Attached to '{node_name}'</span><br>"
            )
        else:
            self.output.append(
                f"<span style='color:red'>❌ Failed to attach to '{node_name}'</span><br>"
            )

        QTimer.singleShot(200, self._process_next_auto_vo)

    def _on_auto_vo_tts_error(self, err: str, node_name: str):
        self.output.append(
            f"<span style='color:red'>❌ TTS error for '{node_name}': {err}</span><br>"
        )
        LOGGER.error(f"Auto Voiceover TTS error for {node_name}: {err}")
        QTimer.singleShot(200, self._process_next_auto_vo)

    def _finish_auto_voiceover(self):
        """Finalize auto-voiceover run."""
        main_window = self._get_main_window()
        if main_window:
            main_window.compile_graph()
            main_window.mark_modified()

        self.btn_gen.setEnabled(True)
        self.status_label.setText("Status: Auto Voiceover complete ✅")
        self.output.append(
            "<br><b style='color:#27ae60;font-size:13px;'>"
            "🎬 Auto Voiceover complete! All nodes have been synced and the project is render-ready."
            "</b><br>"
        )
        QMessageBox.information(
            self,
            "Auto Voiceover Complete",
            "All animation nodes have been voiceover-synced.\n"
            "The project is now render-ready with synchronized audio.",
        )

    # ══════════════════════════════════════════════════════════════
    # MCP AGENT INTEGRATION
    # ══════════════════════════════════════════════════════════════

    def set_mcp_agent(self, agent) -> None:
        """Called by EfficientManimWindow after construction to inject the live agent."""
        self._mcp_agent = agent
        LOGGER.info("AIPanel: MCP Agent connected.")
        # Enable MCP mode checkbox now that an agent is available
        if hasattr(self, "chk_mcp_mode"):
            self.chk_mcp_mode.setEnabled(True)

    def _get_mcp_agent(self):
        """Return the MCP agent, trying to lazy-init if not yet injected."""
        if self._mcp_agent is not None:
            return self._mcp_agent
        # Fallback: walk widget tree and grab agent from window
        win = self._get_main_window()
        if win is not None and hasattr(win, "mcp") and win.mcp is not None:
            self._mcp_agent = win.mcp
            return self._mcp_agent
        return None

    def _run_mcp_agent_mode(self) -> None:
        """
        MCP Agent Mode: Gemini reads the live scene state (via get_context),
        then outputs a JSON list of MCP commands which are executed directly
        against the running application — no code generation, no merge step.
        """
        txt = self.input.toPlainText().strip()
        if not txt:
            self.status_label.setText("Status: Empty prompt")
            return

        agent = self._get_mcp_agent()
        if agent is None:
            if not MCP_AVAILABLE:
                QMessageBox.critical(
                    self,
                    "MCP Not Available",
                    "mcp.py was not found next to main.py.\n"
                    "Make sure mcp.py is in the same directory and restart.",
                )
            else:
                QMessageBox.critical(
                    self,
                    "MCP Error",
                    "MCP Agent is not initialised yet.\n"
                    "Please wait for the application to finish loading.",
                )
            self.chk_mcp_mode.setChecked(False)
            return

        self.output.clear()
        self.nodes_list.clear()
        self.btn_gen.setEnabled(False)
        self.status_label.setText("Status: MCP Agent running…")
        self.output.append(
            "<b style='color:#1a73e8;font-size:13px;'>🔌 MCP Agent Mode</b><br>"
            f"<b>Instruction:</b> {txt}<br><br>"
        )
        self.input.clear()

        # ── 1. Capture live scene state ────────────────────────────────────
        ctx_result = agent.execute("get_context")
        ctx_json = (
            json.dumps(ctx_result.data, indent=2, default=str)
            if ctx_result.success
            else "{}"
        )

        cmds_result = agent.execute("list_commands")
        available_commands: list = cmds_result.data if cmds_result.success else []

        # Commands Gemini is allowed to use (exclude destructive ops unless asked)
        safe_commands = [c for c in available_commands if c not in ("clear_scene",)]

        self._mcp_ctx_before = ctx_result.data  # save for diff display later

        # ── 2. Build the Gemini prompt ─────────────────────────────────────
        selected_model = str(SETTINGS.get("GEMINI_MODEL", "gemini-3-flash-preview"))

        system_prompt = (
            "You are an autonomous animation editing agent for EfficientManim.\n\n"
            f"Available MCP commands:\n{json.dumps(safe_commands, indent=2)}\n\n"
            "Command payload reference:\n"
            "  create_node:      {cls_name, name, node_type (ANIMATION|MOBJECT), params: {}, x, y}\n"
            "  set_node_param:   {node_id, key, value}\n"
            "  rename_node:      {node_id, name}\n"
            "  delete_node:      {node_id, confirm: true}  ← only if user explicitly asks to delete\n"
            "  attach_voiceover: {node_id, audio_path, transcript, duration}\n"
            "  remove_voiceover: {node_id}\n"
            "  select_node:      {node_id}\n"
            "  switch_tab:       {tab}  ← partial name ok, e.g. 'Properties'\n"
            "  compile_graph:    {}\n"
            "  trigger_render:   {node_id (optional)}\n"
            "  save_project:     {}\n"
            "  switch_scene:     {scene_name}\n\n"
            "Rules:\n"
            "1. Output ONLY a valid JSON array. No explanation. No markdown. No fences.\n"
            '2. Each element: {"command": "...", "payload": {...}}\n'
            "3. For nodes you CREATE in this session, use 'node_name' key instead of "
            "'node_id' in subsequent commands — it will be resolved automatically.\n"
            "4. Always end with compile_graph if you modified the scene.\n"
            "5. Use node IDs from the current state for existing nodes.\n"
            "6. NEVER issue clear_scene or delete_node unless the user explicitly asked.\n\n"
            f"Current project state:\n{ctx_json}\n\n"
            f"USER INSTRUCTION: {txt}\n\n"
            "Output the JSON array now. Start with [ and end with ]. Nothing else."
        )

        # ── 3. Stream Gemini response ──────────────────────────────────────
        self._mcp_buffer = ""

        def on_chunk(text: str) -> None:
            self._mcp_buffer += text
            self.output.moveCursor(QTextCursor.MoveOperation.End)
            self.output.insertPlainText(text)

        def on_finish() -> None:
            self._execute_mcp_commands_from_buffer()
            self.btn_gen.setEnabled(True)

        self.worker = AIWorker(system_prompt, model=selected_model)
        self.worker.chunk_received.connect(on_chunk)
        self.worker.finished_signal.connect(on_finish)
        self.worker.start()

    def _execute_mcp_commands_from_buffer(self) -> None:
        """
        Parse Gemini's JSON command list and execute each one through MCPAgent.
        Handles node name → ID resolution for nodes created in the same batch.
        """
        agent = self._get_mcp_agent()
        if agent is None:
            self.status_label.setText("Status: MCP agent lost.")
            return

        raw = self._mcp_buffer.strip()

        # Strip any markdown fences Gemini occasionally adds
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()

        # Try to extract a JSON array substring as a fallback
        def _try_parse(text: str):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                start = text.find("[")
                end = text.rfind("]") + 1
                if start != -1 and end > start:
                    return json.loads(text[start:end])
                raise

        try:
            commands: list = _try_parse(raw)
        except json.JSONDecodeError as e:
            self.status_label.setText("Status: ❌ Parse error.")
            self.output.append(
                f"<br><span style='color:red;'>❌ Could not parse Gemini's command list: {e}</span><br>"
                "<span style='color:#888;'>Raw output logged to console.</span>"
            )
            LOGGER.error(f"MCP agent parse error: {e}\nRaw:\n{raw[:800]}")
            return

        if not isinstance(commands, list) or len(commands) == 0:
            self.output.append(
                "<br><span style='color:orange;'>⚠️ Gemini returned no commands.</span>"
            )
            self.status_label.setText("Status: No commands to execute.")
            return

        self.output.append(
            f"<br><b style='color:#1a73e8;'>Executing {len(commands)} command(s):</b><br>"
        )

        # ── Node name → ID resolution table ───────────────────────────────
        # Populated as create_node results come in; used to patch node_name refs.
        name_to_id: dict[str, str] = {}

        success_count = 0
        fail_count = 0

        for cmd_obj in commands:
            if not isinstance(cmd_obj, dict):
                continue

            command: str = cmd_obj.get("command", "")
            payload: dict = dict(cmd_obj.get("payload", {}))

            if not command:
                continue

            # Resolve node_name → node_id if Gemini used a name for a just-created node
            if "node_name" in payload and "node_id" not in payload:
                resolved_id = name_to_id.get(payload.pop("node_name"))
                if resolved_id:
                    payload["node_id"] = resolved_id
                else:
                    self.output.append(
                        f"<span style='color:orange;'>⚠️ {command} — "
                        f"could not resolve node_name to an id, skipping.</span><br>"
                    )
                    fail_count += 1
                    continue

            result = agent.execute(command, payload)

            if result.success:
                # If we just created a node, remember its id by name
                if command == "create_node" and isinstance(result.data, dict):
                    created_name = payload.get("name", "")
                    created_id = result.data.get("id", "")
                    if created_name and created_id:
                        name_to_id[created_name] = created_id

                data_str = str(result.data) if result.data else ""
                self.output.append(
                    f"<span style='color:#27ae60;'>✅ {command}</span>"
                    + (
                        f" <span style='color:#555;font-size:10px;'>→ {data_str[:80]}</span>"
                        if data_str
                        else ""
                    )
                    + "<br>"
                )
                success_count += 1
            else:
                self.output.append(
                    f"<span style='color:red;'>❌ {command} — {result.error}</span><br>"
                )
                fail_count += 1

        # ── Show node count delta ──────────────────────────────────────────
        try:
            after = agent.execute("get_context")
            if (
                after.success
                and hasattr(self, "_mcp_ctx_before")
                and self._mcp_ctx_before
            ):
                before_count = self._mcp_ctx_before.get("node_count", 0)
                after_count = after.data.get("node_count", 0)
                delta = after_count - before_count
                if delta > 0:
                    self.output.append(
                        f"<span style='color:#27ae60;'>📊 {delta} new node(s) added.</span><br>"
                    )
                elif delta < 0:
                    self.output.append(
                        f"<span style='color:orange;'>📊 {abs(delta)} node(s) removed.</span><br>"
                    )
        except Exception:
            pass

        status_icon = "✅" if fail_count == 0 else "⚠️"
        self.status_label.setText(
            f"Status: {status_icon} Done — {success_count} ok, {fail_count} failed."
        )
        self.output.append(
            f"<br><b style='color:#1a73e8;'>"
            f"🔌 MCP Agent finished — {success_count}/{success_count + fail_count} commands succeeded."
            f"</b>"
        )

    def _get_main_window(self):
        """Walk up the widget tree to find the EfficientManimWindow."""
        widget = self.parent()
        while widget is not None:
            if isinstance(widget, EfficientManimWindow):
                return widget
            widget = widget.parent()
        return None

    def merge(self):
        """Emit merge signal with code."""
        if self.last_code:
            self.merge_requested.emit(self.last_code)
            # Signal node generation for UI update
            if self.extracted_nodes:
                self.nodes_generated.emit(
                    {"code": self.last_code, "nodes": self.extracted_nodes}
                )

    def reject(self):
        """Reject AI code and reset."""
        self.last_code = None
        self.output.clear()
        self.nodes_list.clear()
        self.extracted_nodes = []
        self.btn_merge.setEnabled(False)
        self.btn_reject.setEnabled(False)
        self.status_label.setText("Status: Code rejected")
        LOGGER.ai("AI Code rejected.")


# ==============================================================================
# 8B. AI NODE INTEGRATION
# ==============================================================================


class AINodeIntegrator:
    """Handles integration of AI-generated nodes into the scene graph."""

    @staticmethod
    def parse_ai_code(code: str) -> tuple:
        """
        Parse AI-generated code and extract BOTH node and animation definitions.
        Handles:
          - var = ClassName(...) definitions
          - self.play(AnimationClass(target, ...)) inline calls
          - self.play(target.animate.method(...)) calls

        Returns: (mobjects, animations, play_sequence)
        Where play_sequence = list of {anim_class, anim_var, target_var, params, raw}
        """
        mobjects = []
        animations = []
        play_sequence = []
        mobject_vars = {}  # var_name -> class_name

        KNOWN_ANIMS = {
            "FadeIn",
            "FadeOut",
            "Write",
            "DrawBorderThenFill",
            "Create",
            "Transform",
            "ReplacementTransform",
            "Rotate",
            "Scale",
            "ScaleInPlace",
            "MoveTo",
            "ApplyMethod",
            "Indicate",
            "FocusOn",
            "Circumscribe",
            "ShowCreation",
            "Uncreate",
            "GrowFromCenter",
            "ShrinkToCenter",
            "Succession",
            "AnimationGroup",
        }

        def _balanced_paren_end(s, start):
            """Return index after the matching closing paren starting at start."""
            depth = 1
            i = start
            while i < len(s) and depth > 0:
                if s[i] == "(":
                    depth += 1
                elif s[i] == ")":
                    depth -= 1
                i += 1
            return i

        def _is_animation_class(class_name):
            if class_name in KNOWN_ANIMS:
                return True
            try:
                cls = getattr(manim, class_name, None)
                if cls is None:
                    return False
                return issubclass(cls, manim.Animation)
            except Exception:
                return False

        # ── Step 1: Variable assignments ──────────────────────────────────
        for m in re.finditer(
            r"^[ \t]*(\w+)\s*=\s*([A-Z][a-zA-Z0-9]*)\s*\(", code, re.MULTILINE
        ):
            var_name, class_name = m.groups()
            if not hasattr(manim, class_name):
                continue
            end_idx = _balanced_paren_end(code, m.end())
            params_str = code[m.end() : end_idx - 1]
            params = AINodeIntegrator._parse_params(params_str)

            is_anim = _is_animation_class(class_name)
            entry = {
                "var_name": var_name,
                "class_name": class_name,
                "params": params,
                "source": "ai",
                "code_snippet": code[m.start() : end_idx],
            }
            if is_anim:
                animations.append(entry)
            else:
                mobjects.append(entry)
                mobject_vars[var_name] = class_name

        # ── Step 2: self.play() calls in order ────────────────────────────
        for m in re.finditer(r"self\.play\(", code):
            end_idx = _balanced_paren_end(code, m.end())
            raw_play = code[m.end() : end_idx - 1].strip()

            # .animate chain: var.animate.method(args)
            animate_m = re.match(
                r"(\w+)\.animate\.([\w]+)\((.*)\)$", raw_play, re.DOTALL
            )
            if animate_m:
                target_var, method, method_args = animate_m.groups()
                play_sequence.append(
                    {
                        "anim_class": f"animate.{method}",
                        "anim_var": f"{target_var}_animate_{method}_{len(play_sequence) + 1}",
                        "target_var": target_var,
                        "params": AINodeIntegrator._parse_params(method_args),
                        "raw": raw_play,
                        "is_animate_chain": True,
                    }
                )
                continue

            # Standard AnimClass(args)
            inner_m = re.match(r"([A-Z][a-zA-Z0-9]*)\s*\((.*)\)$", raw_play, re.DOTALL)
            if inner_m:
                anim_class, anim_args = inner_m.groups()
                if not hasattr(manim, anim_class):
                    continue
                if not _is_animation_class(anim_class):
                    continue

                target_var = None
                for mob_var in mobject_vars:
                    if re.search(r"\b" + re.escape(mob_var) + r"\b", anim_args):
                        target_var = mob_var
                        break

                params = AINodeIntegrator._parse_params(anim_args)
                if target_var and target_var in params:
                    del params[target_var]

                play_sequence.append(
                    {
                        "anim_class": anim_class,
                        "anim_var": f"{anim_class.lower()}_{len(play_sequence) + 1}",
                        "target_var": target_var,
                        "params": params,
                        "raw": raw_play,
                        "is_animate_chain": False,
                    }
                )

        return mobjects, animations, play_sequence

    @staticmethod
    def _parse_params(params_str: str) -> dict:
        """Parse parameter string into dict with proper quote/constant handling."""
        params = {}

        # Enhanced key=value extraction
        for item in params_str.split(","):
            item = item.strip()
            if "=" in item:
                try:
                    key, value = item.split("=", 1)
                    key = key.strip()
                    value = value.strip()

                    # Clean quotes from strings
                    if (value.startswith('"') and value.endswith('"')) or (
                        value.startswith("'") and value.endswith("'")
                    ):
                        value = value[1:-1]

                    # Store cleaned value
                    params[key] = value
                except ValueError:
                    pass

        return params

    @staticmethod
    def create_node_from_ai(
        var_name: str,
        class_name: str,
        params: dict,
        scene_graph,
        node_type=NodeType.MOBJECT,
        pos: "tuple[int, int]" = (50, 50),
        override_cls_name: str = None,
    ) -> "NodeItem":
        """
        Create a node in the scene graph from AI definition.

        Args:
            var_name: Variable name (e.g., 'circle')
            class_name: Manim class name (e.g., 'Circle')
            params: Parameter dict
            scene_graph: The SceneGraph instance
            node_type: NodeType.MOBJECT or NodeType.ANIMATION
            pos: (x, y) position in scene
            override_cls_name: Display class name override (e.g. 'animate.shift')

        Returns:
            NodeItem: The created node
        """
        # Create NodeData with AI metadata
        display_cls = override_cls_name if override_cls_name else class_name
        node_data = NodeData(var_name, node_type, display_cls)
        node_data.pos_x, node_data.pos_y = pos

        # Apply parameters with type safety
        for param_name, param_value in params.items():
            param_value = str(param_value).strip()
            if TypeSafeParser.is_color_param(param_name):
                clean_value = param_value.strip("'\"")
                node_data.params[param_name] = TypeSafeParser.parse_color(clean_value)
            elif TypeSafeParser.is_numeric_param(param_name):
                clean_value = param_value.strip("'\"")
                node_data.params[param_name] = TypeSafeParser.parse_numeric(clean_value)
            else:
                clean_value = param_value.strip("'\"")
                node_data.params[param_name] = clean_value

        # Mark as AI-generated
        node_data.is_ai_generated = True
        node_data.ai_source = class_name

        # Create NodeItem
        item = NodeItem(node_data)

        # Add to scene graph
        scene_graph.scene.addItem(item)
        scene_graph.nodes[item.data.id] = item

        # Auto-detect and load all class parameters (for real Manim classes)
        if hasattr(manim, class_name):
            AINodeIntegrator._load_class_parameters(node_data, class_name)

        return item

    @staticmethod
    def _load_class_parameters(node_data: "NodeData", class_name: str):
        """
        Only validate explicitly provided parameters.
        Do NOT auto-load unused function parameters or defaults.
        This prevents inspect._empty and unused parameter injection.
        """
        try:
            cls = getattr(manim, class_name, None)
            if not cls:
                return

            sig = inspect.signature(cls.__init__)

            # Track which parameters were explicitly provided by AI
            for param_name in list(node_data.params.keys()):
                param = sig.parameters.get(param_name)
                if not param:
                    # Parameter doesn't exist in class, remove it
                    del node_data.params[param_name]
                    continue

                # Skip special parameters
                if param_name in ("self", "args", "kwargs", "mobject"):
                    del node_data.params[param_name]
                    continue

                # Validate and clean the value
                value = node_data.params[param_name]

                # Remove None and empty values
                if value is None or str(value) == "<class 'inspect._empty'>":
                    del node_data.params[param_name]
                    continue

                # Check for string representations of inspect._empty
                if isinstance(value, str):
                    if value.strip() in (
                        "inspect._empty",
                        "<class 'inspect._empty'>",
                        "_empty",
                    ):
                        del node_data.params[param_name]
                        continue

        except Exception as e:
            LOGGER.error(f"Failed to validate parameters for {class_name}: {e}")

    @staticmethod
    def validate_ai_nodes(nodes: list) -> tuple:
        """
        Validate AI-generated nodes.

        Returns: (valid_nodes, errors)
        """
        valid = []
        errors = []

        for node in nodes:
            if not hasattr(manim, node["class_name"]):
                errors.append(f"Invalid class: {node['class_name']}")
                continue
            valid.append(node)

        return valid, errors

    @staticmethod
    def merge_ai_code_to_scene(code: str, scene_graph) -> dict:
        """
        Merge AI-generated code into scene with animations and connections.
        Creates a fully connected graph with proper node positions.

        Returns: {
            'success': bool,
            'nodes_added': int,
            'nodes': list of created GraphicsItems,
            'errors': list of error messages
        }
        """
        try:
            mobjects, animations, play_sequence = AINodeIntegrator.parse_ai_code(code)

            valid_mobjects, mob_errors = AINodeIntegrator.validate_ai_nodes(mobjects)
            errors = list(mob_errors)

            created_nodes = []
            mobject_items = {}  # var_name -> NodeItem

            # Layout constants
            COL_WIDTH = 220
            ROW_HEIGHT = 120
            START_X = 50
            START_Y = 50

            # ── Create Mobject Nodes (column 0) ───────────────────────────
            for row_idx, node_def in enumerate(valid_mobjects):
                try:
                    item = AINodeIntegrator.create_node_from_ai(
                        node_def["var_name"],
                        node_def["class_name"],
                        node_def["params"],
                        scene_graph,
                        node_type=NodeType.MOBJECT,
                        pos=(START_X, START_Y + row_idx * ROW_HEIGHT),
                    )
                    created_nodes.append(item)
                    mobject_items[node_def["var_name"]] = item
                except Exception as e:
                    errors.append(
                        f"Failed to create mobject {node_def['var_name']}: {e}"
                    )

            # ── Create Animation Nodes (per play_sequence entry) ──────────
            anim_col = 1
            prev_anim_item = None

            for seq_idx, play_entry in enumerate(play_sequence):
                anim_class = play_entry["anim_class"]
                anim_var = play_entry["anim_var"]
                target_var = play_entry["target_var"]
                params = play_entry.get("params", {})
                is_chain = play_entry.get("is_animate_chain", False)

                # For .animate chains use a generic label
                display_class = (
                    anim_class.split(".")[-1] if "." in anim_class else anim_class
                )
                if is_chain:
                    display_class = f"animate.{display_class}"

                # Position: column based on animation index
                pos_x = START_X + anim_col * COL_WIDTH
                pos_y = START_Y + seq_idx * ROW_HEIGHT

                try:
                    item = AINodeIntegrator.create_node_from_ai(
                        anim_var,
                        anim_class if not is_chain else "ApplyMethod",
                        params,
                        scene_graph,
                        node_type=NodeType.ANIMATION,
                        pos=(pos_x, pos_y),
                        override_cls_name=display_class,
                    )
                    created_nodes.append(item)

                    # Connect animation to target mobject
                    if target_var and target_var in mobject_items:
                        mob_item = mobject_items[target_var]
                        try:
                            wire = WireItem(mob_item.out_socket, item.in_socket)
                            scene_graph.scene.addItem(wire)
                            mob_item.out_socket.links.append(wire)
                            item.in_socket.links.append(wire)
                        except Exception as we:
                            LOGGER.warn(
                                f"Wire creation failed ({target_var} -> {anim_var}): {we}"
                            )

                    # Chain animations sequentially via out -> in
                    if prev_anim_item is not None:
                        try:
                            chain_wire = WireItem(
                                prev_anim_item.out_socket, item.in_socket
                            )
                            scene_graph.scene.addItem(chain_wire)
                            prev_anim_item.out_socket.links.append(chain_wire)
                            item.in_socket.links.append(chain_wire)
                        except Exception:
                            pass

                    prev_anim_item = item

                except Exception as e:
                    errors.append(f"Failed to create animation node {anim_var}: {e}")

            scene_graph.scene.notify_change()

            return {
                "success": len(created_nodes) > 0,
                "nodes_added": len(created_nodes),
                "nodes": created_nodes,
                "errors": errors,
            }

        except Exception as e:
            LOGGER.error(f"merge_ai_code_to_scene error: {e}")
            traceback.print_exc()
            return {
                "success": False,
                "nodes_added": 0,
                "nodes": [],
                "errors": [f"Fatal error: {str(e)}"],
            }


class AssetsPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        bar = QHBoxLayout()
        btn_imp = QPushButton("Import Asset")
        btn_imp.clicked.connect(self.do_import)
        bar.addWidget(btn_imp)
        layout.addLayout(bar)

        self.list = QListWidget()
        self.list.setIconSize(QSize(48, 48))
        self.list.setDragEnabled(True)
        layout.addWidget(self.list)
        ASSETS.assets_changed.connect(self.refresh)

    def do_import(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import", "", "Media (*.png *.jpg *.mp4 *.mp3)"
        )
        for p in paths:
            ASSETS.add_asset(p)

    def refresh(self):
        self.list.clear()
        for asset in ASSETS.get_list():
            item = QListWidgetItem(asset.name)
            if asset.kind == "image":
                item.setIcon(QIcon(asset.original_path))
            else:
                item.setIcon(
                    self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon)
                )
            item.setData(Qt.ItemDataRole.UserRole, asset.id)
            self.list.addItem(item)

    def startDrag(self, actions):
        item = self.list.currentItem()
        if not item:
            return
        mime = QMimeData()
        mime.setText(f"ASSET:{item.data(Qt.ItemDataRole.UserRole)}")
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.CopyAction)


class VoiceoverPanel(QWidget):
    """Panel for AI TTS generation, audio preview, and node synchronization.

    Supports voiceover attachment to ALL animation node types.
    Includes full playback controls: play, pause, stop, seek, duration display.
    """

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.tts_worker = None
        self._current_audio_path: str | None = None
        self._player = QMediaPlayer()
        self._audio_out = QAudioOutput()
        self._player.setAudioOutput(self._audio_out)
        self._audio_out.setVolume(1.0)
        self._player_duration = 0

        # Wire player signals
        self._player.positionChanged.connect(self._on_position_changed)
        self._player.durationChanged.connect(self._on_duration_changed)
        self._player.mediaStatusChanged.connect(self._on_media_status)
        self._player.playbackStateChanged.connect(self._on_playback_state_changed)
        # NEW: Error handler for media load failures
        self._player.errorChanged.connect(self._on_player_error)

        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        # ── Header ────────────────────────────────────────────────
        header = QLabel("🎙️ AI Voiceover Studio")
        header.setStyleSheet("font-size: 14px; font-weight: bold; color: #2c3e50;")
        layout.addWidget(header)

        # ── Settings Grid ─────────────────────────────────────────
        form = QFormLayout()
        self.voice_combo = QComboBox()
        voices = ["Puck", "Charon", "Kore", "Fenrir", "Aoede", "Zephyr"]
        self.voice_combo.addItems(voices)
        self.voice_combo.setCurrentText("Zephyr")
        form.addRow("Voice:", self.voice_combo)
        layout.addLayout(form)

        # ── Script Input ──────────────────────────────────────────
        layout.addWidget(QLabel("Script:"))
        self.text_input = QPlainTextEdit()
        self.text_input.setPlaceholderText("Enter text to speak here...")
        self.text_input.setMaximumHeight(90)
        layout.addWidget(self.text_input)

        # ── Generate Button ───────────────────────────────────────
        self.btn_gen = QPushButton("⚡ Generate Audio")
        self.btn_gen.setStyleSheet(
            "background-color: #8e44ad; color: white; padding: 8px; font-weight: bold;"
        )
        self.btn_gen.clicked.connect(self.generate_audio)
        layout.addWidget(self.btn_gen)

        # ── Generation Progress ───────────────────────────────────
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # indeterminate
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumHeight(6)
        layout.addWidget(self.progress_bar)

        # ── Audio Preview ─────────────────────────────────────────
        preview_frame = QFrame()
        preview_frame.setFrameShape(QFrame.Shape.StyledPanel)
        preview_frame.setStyleSheet(
            "QFrame { background: #1a1a2e; border-radius: 6px; border: 1px solid #444; }"
        )
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(8, 6, 8, 6)
        preview_layout.setSpacing(4)

        preview_lbl = QLabel("🎵 Audio Preview")
        preview_lbl.setStyleSheet(
            "color: #a0a0c0; font-size: 11px; font-weight: bold; border: none;"
        )
        preview_layout.addWidget(preview_lbl)

        # Seek slider
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 0)
        self.seek_slider.setEnabled(False)
        self.seek_slider.sliderMoved.connect(self._on_seek)
        self.seek_slider.setStyleSheet(
            "QSlider::groove:horizontal { height: 4px; background: #444; border-radius: 2px; }"
            "QSlider::sub-page:horizontal { background: #8e44ad; border-radius: 2px; }"
            "QSlider::handle:horizontal { width: 12px; height: 12px; margin: -4px 0; "
            "background: white; border-radius: 6px; }"
        )
        preview_layout.addWidget(self.seek_slider)

        # Time label
        self.lbl_time = QLabel("00:00 / 00:00")
        self.lbl_time.setStyleSheet(
            "color: #808090; font-family: monospace; font-size: 10px; border: none;"
        )
        self.lbl_time.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(self.lbl_time)

        # Playback controls row
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(4)

        self.btn_play = QPushButton("▶")
        self.btn_play.setFixedSize(32, 32)
        self.btn_play.setEnabled(False)
        self.btn_play.setToolTip("Play / Pause")
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_play.setStyleSheet(
            "QPushButton { background: #8e44ad; color: white; border-radius: 16px; font-size: 12px; border: none; }"
            "QPushButton:disabled { background: #555; }"
        )

        self.btn_stop = QPushButton("⏹")
        self.btn_stop.setFixedSize(32, 32)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setToolTip("Stop")
        self.btn_stop.clicked.connect(self._stop_audio)
        self.btn_stop.setStyleSheet(
            "QPushButton { background: #555; color: white; border-radius: 16px; font-size: 12px; border: none; }"
            "QPushButton:disabled { background: #444; color: #888; }"
        )

        ctrl_row.addStretch()
        ctrl_row.addWidget(self.btn_play)
        ctrl_row.addWidget(self.btn_stop)
        ctrl_row.addStretch()
        preview_layout.addLayout(ctrl_row)

        layout.addWidget(preview_frame)

        # ── Separator ─────────────────────────────────────────────
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #bdc3c7;")
        layout.addWidget(line)

        # ── Node Sync Section ─────────────────────────────────────
        sync_lbl = QLabel("🔗 Attach to Animation Node")
        sync_lbl.setStyleSheet("font-weight: bold;")
        layout.addWidget(sync_lbl)

        node_row = QHBoxLayout()
        self.node_combo = QComboBox()
        self.node_combo.setPlaceholderText("Select an Animation Node...")
        node_row.addWidget(self.node_combo, 1)

        btn_refresh = QPushButton("🔄")
        btn_refresh.setFixedSize(28, 28)
        btn_refresh.setToolTip("Refresh Node List")
        btn_refresh.clicked.connect(self.refresh_nodes)
        node_row.addWidget(btn_refresh)
        layout.addLayout(node_row)

        # "Add to Animation Node" button
        self.btn_attach = QPushButton("📎 Add to Animation Node")
        self.btn_attach.setStyleSheet(
            "background-color: #27ae60; color: white; padding: 7px; font-weight: bold;"
        )
        self.btn_attach.setEnabled(False)
        self.btn_attach.setToolTip(
            "Attach the generated audio to the selected animation node"
        )
        self.btn_attach.clicked.connect(self._attach_to_node)
        layout.addWidget(self.btn_attach)

        self.status_lbl = QLabel("Ready")
        self.status_lbl.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(self.status_lbl)

        layout.addStretch()

        # Initial refresh
        QTimer.singleShot(1000, self.refresh_nodes)

    # ── Node population ────────────────────────────────────────────

    def refresh_nodes(self):
        """Populate combo with ALL animation nodes (any type)."""
        current = self.node_combo.currentData()
        self.node_combo.clear()

        count = 0
        for nid, node in self.main_window.nodes.items():
            short_id = nid[:6]
            type_tag = "🎬" if node.data.type == NodeType.ANIMATION else "🔷"
            display_text = f"{type_tag} {node.data.name} ({short_id})"
            self.node_combo.addItem(display_text, nid)
            count += 1

        if current:
            idx = self.node_combo.findData(current)
            if idx >= 0:
                self.node_combo.setCurrentIndex(idx)

        self.status_lbl.setText(f"Found {count} nodes.")

    # ── Audio generation ───────────────────────────────────────────

    def generate_audio(self):
        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Input Error", "Please enter some text.")
            return

        self.btn_gen.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.status_lbl.setText("Generating audio via Gemini TTS…")

        voice = self.voice_combo.currentText()
        model = SETTINGS.get("TTS_MODEL", "gemini-2.5-flash-preview-tts")

        self.tts_worker = TTSWorker(text, voice, model)
        self.tts_worker.finished_signal.connect(self.on_tts_success)
        self.tts_worker.error_signal.connect(self.on_tts_error)
        self.tts_worker.start()

    def on_tts_success(self, file_path):
        self.btn_gen.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_lbl.setText("✅ Audio generated — ready for preview.")

        asset = ASSETS.add_asset(file_path)
        if not asset:
            self.status_lbl.setText("⚠️ Error registering asset.")
            return

        self._current_audio_path = file_path
        self._load_preview(file_path)
        self.btn_attach.setEnabled(True)

        # Store transcript on the worker's text for later node attachment
        self._last_transcript = self.text_input.toPlainText().strip()
        self._last_asset = asset

    def on_tts_error(self, err):
        self.btn_gen.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_lbl.setText("❌ Generation failed.")
        LOGGER.error(f"TTS Error: {err}")
        QMessageBox.critical(self, "TTS Error", err)

    # ── Audio preview ──────────────────────────────────────────────

    def _load_preview(self, file_path: str):
        """
        Load audio file into the preview player.

        CRITICAL FIX: Proper QUrl conversion with error handling.
        - Convert path to QUrl with proper escaping
        - Validate file exists before loading
        - Clear any previous errors
        - Log load status
        """
        try:
            # ═════════════════════════════════════════════════════════════
            # VALIDATION: File must exist
            # ═════════════════════════════════════════════════════════════
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                self.status_lbl.setText(f"❌ File not found: {file_path}")
                LOGGER.error(f"Audio preview: file missing at {file_path}")
                self.btn_play.setEnabled(False)
                self.btn_stop.setEnabled(False)
                self.seek_slider.setEnabled(False)
                return

            # ═════════════════════════════════════════════════════════════
            # FIX: Use QUrl.fromLocalFile() for proper path escaping
            # Windows paths with backslashes must be converted safely
            # ═════════════════════════════════════════════════════════════
            from PySide6.QtCore import QUrl

            # Convert to absolute path to prevent relative path issues
            abs_path = file_path_obj.resolve()

            # Create QUrl with proper local file encoding
            # This handles spaces, special characters, and backslashes correctly
            media_url = QUrl.fromLocalFile(str(abs_path))

            if not media_url.isValid():
                self.status_lbl.setText("❌ Invalid file path or unsupported format")
                LOGGER.error(f"Audio preview: invalid QUrl for {abs_path}")
                self.btn_play.setEnabled(False)
                self.btn_stop.setEnabled(False)
                self.seek_slider.setEnabled(False)
                return

            # ═════════════════════════════════════════════════════════════
            # LOAD: Set media source and enable controls
            # ═════════════════════════════════════════════════════════════
            self._player.setSource(media_url)

            # Enable playback controls
            self.seek_slider.setEnabled(True)
            self.btn_play.setEnabled(True)
            self.btn_stop.setEnabled(True)

            self.status_lbl.setText("✅ Audio loaded. Press ▶ to preview.")
            LOGGER.info(f"Audio preview loaded: {abs_path}")

        except Exception as e:
            self.status_lbl.setText(f"❌ Error loading audio: {type(e).__name__}")
            LOGGER.error(f"Audio preview load error: {e}", exc_info=True)
            self.btn_play.setEnabled(False)
            self.btn_stop.setEnabled(False)
            self.seek_slider.setEnabled(False)

    def _toggle_play(self):
        state = self._player.playbackState()
        if state == QMediaPlayer.PlaybackState.Playing:
            self._player.pause()
        else:
            self._player.play()

    def _stop_audio(self):
        self._player.stop()
        self.seek_slider.setValue(0)
        self.lbl_time.setText("00:00 / 00:00")

    def _on_seek(self, position: int):
        self._player.setPosition(position)

    def _on_position_changed(self, position: int):
        if not self.seek_slider.isSliderDown():
            self.seek_slider.setValue(position)
        self._update_time_label(position)

    def _on_duration_changed(self, duration: int):
        self._player_duration = duration
        self.seek_slider.setRange(0, duration)
        self._update_time_label(0)

    def _on_media_status(self, status):
        """
        Handle media status changes.

        Catches load errors, end-of-media, and other status changes.
        """
        try:
            # Handle end of media
            if status == QMediaPlayer.MediaStatus.EndOfMedia:
                self.btn_play.setText("▶")
                LOGGER.info("Audio playback: end of media reached")

            # Handle load errors
            elif status == QMediaPlayer.MediaStatus.InvalidMedia:
                self.status_lbl.setText("❌ Invalid audio format or corrupted file")
                LOGGER.error("Audio preview: invalid media format")
                self.btn_play.setEnabled(False)
                self.btn_stop.setEnabled(False)

            elif status == QMediaPlayer.MediaStatus.NoMedia:
                # No media loaded (expected after clear or init)
                pass

            elif status == QMediaPlayer.MediaStatus.LoadedMedia:
                # Media successfully loaded
                LOGGER.info("Audio preview: media loaded successfully")

            elif status == QMediaPlayer.MediaStatus.LoadingMedia:
                # Media is being loaded
                self.status_lbl.setText("⏳ Loading audio...")

        except Exception as e:
            LOGGER.error(f"Error in _on_media_status: {e}")

    def _on_player_error(self):
        """
        Handle QMediaPlayer errors.

        Called when an error occurs during playback or loading.
        """
        try:
            error = self._player.error()
            error_string = self._player.errorString()

            if error != QMediaPlayer.Error.NoError:
                self.status_lbl.setText(f"❌ Playback error: {error_string}")
                LOGGER.error(f"Audio preview error: {error} - {error_string}")
                self.btn_play.setEnabled(False)
                self.btn_stop.setEnabled(False)

        except Exception as e:
            LOGGER.error(f"Error in _on_player_error: {e}")

    def _on_playback_state_changed(self, state):
        if state == QMediaPlayer.PlaybackState.Playing:
            self.btn_play.setText("⏸")
        else:
            self.btn_play.setText("▶")

    def _update_time_label(self, current_ms: int):
        def fmt(ms):
            s = (ms // 1000) % 60
            m = ms // 60000
            return f"{m:02}:{s:02}"

        self.lbl_time.setText(f"{fmt(current_ms)} / {fmt(self._player_duration)}")

    # ── Node attachment ────────────────────────────────────────────

    def _attach_to_node(self):
        """Attach generated audio to the selected animation node."""
        node_id = self.node_combo.currentData()
        if not node_id:
            QMessageBox.warning(
                self, "No Node Selected", "Please select an animation node first."
            )
            return

        if node_id not in self.main_window.nodes:
            QMessageBox.warning(self, "Invalid Node", "Selected node no longer exists.")
            return

        if not hasattr(self, "_last_asset") or self._last_asset is None:
            QMessageBox.warning(
                self, "No Audio", "Generate audio first before attaching."
            )
            return

        node = self.main_window.nodes[node_id]
        node.data.audio_asset_id = self._last_asset.id
        node.data.voiceover_transcript = getattr(self, "_last_transcript", "")
        if self._player_duration > 0:
            node.data.voiceover_duration = self._player_duration / 1000.0

        node.update()
        self.main_window.compile_graph()
        self.main_window.mark_modified()

        self.status_lbl.setText(f"✅ Attached to '{node.data.name}'")
        LOGGER.info(f"Voiceover attached to node {node.data.name} ({node_id[:6]})")

        QMessageBox.information(
            self,
            "Voiceover Attached",
            f"Audio successfully attached to '{node.data.name}'.\n"
            f"Duration: {node.data.voiceover_duration:.2f}s\n"
            "The node will use this audio during render.",
        )


class KeyboardShortcutsDialog(QDialog):
    """Display keyboard shortcuts and help."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.resize(500, 400)
        layout = QVBoxLayout(self)

        text_display = QTextEdit()
        text_display.setReadOnly(True)
        text_display.setPlainText(KeyboardShortcuts.describe_shortcuts())  # pyright: ignore[reportUndefinedVariable]
        layout.addWidget(text_display)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)


class SettingsDialog(QDialog):
    theme_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(500, 450)
        layout = QVBoxLayout(self)

        # --- Group 1: General & Performance ---
        grp_gen = QGroupBox("General & Performance")
        form_gen = QFormLayout(grp_gen)

        # Theme - Light mode only
        theme_label = QLabel("Light Mode Only")
        form_gen.addRow("Theme:", theme_label)

        # Live Preview Toggle
        self.chk_preview = QCheckBox("Enable Live Mobject Preview")
        self.chk_preview.setToolTip(
            "Renders a small PNG preview when properties change."
        )
        # Default to True (enabled)
        self.chk_preview.setChecked(
            bool(SETTINGS.get("ENABLE_PREVIEW", True, type=bool))
        )
        form_gen.addRow("Live Preview:", self.chk_preview)

        # FPS
        self.fps = QSpinBox()
        self.fps.setRange(15, 60)
        self.fps.setValue(int(SETTINGS.get("FPS", 15) or 15))
        form_gen.addRow("Preview FPS:", self.fps)

        # Quality
        self.quality = QComboBox()
        self.quality.addItems(["Low (ql)", "Medium (qm)", "High (qh)"])
        self.quality.setCurrentText(
            str(SETTINGS.get("QUALITY", "Low (ql)") or "Low (ql)")
        )
        form_gen.addRow("Quality:", self.quality)

        layout.addWidget(grp_gen)

        # --- Group 2: AI Configuration ---
        grp_ai = QGroupBox("Google Gemini AI")
        form_ai = QFormLayout(grp_ai)

        self.api_key = QLineEdit(str(SETTINGS.get("GEMINI_API_KEY", "") or ""))
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key.setPlaceholderText("Paste API Key Here")
        form_ai.addRow("API Key:", self.api_key)

        # Code Gen Model
        self.gemini_model = QComboBox()
        self.gemini_model.addItems(["gemini-3-flash-preview", "gemini-3-pro-preview"])
        self.gemini_model.setCurrentText(
            str(SETTINGS.get("GEMINI_MODEL", "gemini-3-flash-preview") or "")
        )
        form_ai.addRow("Code Model:", self.gemini_model)

        # TTS Model
        self.tts_model = QComboBox()
        self.tts_model.addItems(
            ["gemini-2.5-flash-preview-tts", "gemini-2.5-pro-preview-tts"]
        )
        self.tts_model.setCurrentText(
            str(SETTINGS.get("GEMINI_MODEL", "gemini-2.5-flash-preview-tts") or "")
        )
        form_ai.addRow("TTS Model:", self.tts_model)

        layout.addWidget(grp_ai)

        # --- Buttons ---
        btns = QHBoxLayout()
        b_save = QPushButton("Save and Close")
        b_save.setStyleSheet(
            "background-color: #2ecc71; color: white; font-weight: bold;"
        )
        b_save.clicked.connect(self.save)

        b_cancel = QPushButton("Cancel")
        b_cancel.clicked.connect(self.reject)

        btns.addStretch()
        btns.addWidget(b_save)
        btns.addWidget(b_cancel)
        layout.addLayout(btns)

    def save(self):
        SETTINGS.set("GEMINI_API_KEY", self.api_key.text())
        SETTINGS.set("FPS", self.fps.value())
        SETTINGS.set("QUALITY", self.quality.currentText())
        SETTINGS.set("GEMINI_MODEL", self.gemini_model.currentText())
        SETTINGS.set("TTS_MODEL", self.tts_model.currentText())
        SETTINGS.set("ENABLE_PREVIEW", self.chk_preview.isChecked())

        # Light mode only - no theme switching

        self.accept()


# ==============================================================================
# 8C. POWERFUL NEW PANELS & FEATURES
# ==============================================================================


# ── Manim Class Browser ───────────────────────────────────────────────────────
class ManimClassBrowser(QWidget):
    """Searchable palette of all Manim mobjects and animations."""

    node_requested = Signal(str, str)  # class_name, node_type

    CATEGORIES = {
        "📐 Geometry": [
            "Square",
            "Rectangle",
            "Circle",
            "Ellipse",
            "Triangle",
            "Arrow",
            "Line",
            "DashedLine",
            "DoubleArrow",
            "Polygon",
            "RegularPolygon",
            "Dot",
            "Cross",
            "Star",
            "Arc",
        ],
        "📝 Text": [
            "Text",
            "Tex",
            "MathTex",
            "MarkupText",
            "Title",
            "Paragraph",
            "BulletedList",
        ],
        "📊 Graphs & Plots": [
            "Axes",
            "NumberPlane",
            "PolarPlane",
            "NumberLine",
            "CoordinateSystem",
            "BarChart",
            "LineGraph",
        ],
        "🎭 3D Objects": [
            "Sphere",
            "Cube",
            "Cylinder",
            "Cone",
            "Torus",
            "Surface",
            "ParametricSurface",
        ],
        "🎬 Animations (In)": [
            "FadeIn",
            "Write",
            "DrawBorderThenFill",
            "Create",
            "GrowFromCenter",
            "GrowArrow",
            "SpinInFromNothing",
            "FadeInFromEdge",
            "Succession",
        ],
        "🎬 Animations (Out)": [
            "FadeOut",
            "Unwrite",
            "Uncreate",
            "ShrinkToCenter",
            "FadeOutToEdge",
        ],
        "🔄 Transforms": [
            "Transform",
            "ReplacementTransform",
            "TransformFromCopy",
            "MoveToTarget",
            "ApplyMethod",
            "ApplyMatrix",
            "Rotate",
            "Scale",
            "ScaleInPlace",
        ],
        "✨ Emphasis": [
            "Indicate",
            "FocusOn",
            "Circumscribe",
            "ShowPassingFlash",
            "Flash",
            "Wiggle",
            "ApplyWave",
            "Homotopy",
        ],
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Search bar
        self.search = QLineEdit()
        self.search.setPlaceholderText("🔍 Search Manim classes…")
        self.search.textChanged.connect(self._filter)
        layout.addWidget(self.search)

        # Tree
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setDragEnabled(True)
        self.tree.itemDoubleClicked.connect(self._on_double_click)
        self.tree.setToolTip("Double-click or drag to add node to canvas")
        layout.addWidget(self.tree)

        self._populate()

    def _populate(self, filter_text=""):
        self.tree.clear()
        ft = filter_text.lower()
        for category, items in self.CATEGORIES.items():
            filtered = [i for i in items if ft in i.lower()] if ft else items
            if not filtered:
                continue
            parent = QTreeWidgetItem([category])
            parent.setExpanded(bool(ft))
            self.tree.addTopLevelItem(parent)
            for cls_name in filtered:
                child = QTreeWidgetItem([cls_name])
                child.setToolTip(0, f"Double-click to add {cls_name} node to canvas")
                parent.addChild(child)
        if not ft:
            self.tree.expandAll() if len(self.CATEGORIES) <= 3 else None

    def _filter(self, text):
        self._populate(text)
        if text:
            self.tree.expandAll()

    def _on_double_click(self, item, _col):
        if item.parent() is not None:
            cls_name = item.text(0)
            # Determine type
            node_type = "animation"
            for cat, items in self.CATEGORIES.items():
                if (
                    cls_name in items
                    and "Animation" in cat
                    or "Transform" in cat
                    or "Emphasis" in cat
                ):
                    node_type = "animation"
                elif cls_name in items:
                    node_type = "mobject"
            self.node_requested.emit(cls_name, node_type)


# ── Code Snippet Library ──────────────────────────────────────────────────────
class SnippetLibrary(QWidget):
    """Reusable Manim code snippet templates."""

    snippet_requested = Signal(str)

    SNIPPETS = {
        "🎯 FadeIn + FadeOut": """\
from manim import *

class MyScene(Scene):
    def construct(self):
        circle = Circle(color=BLUE, fill_opacity=1.0)
        self.play(FadeIn(circle))
        self.wait(1)
        self.play(FadeOut(circle))
""",
        "🔄 Transform Shape": """\
from manim import *

class MyScene(Scene):
    def construct(self):
        square = Square(color=RED, fill_opacity=1.0)
        circle = Circle(color=BLUE, fill_opacity=1.0)
        self.play(FadeIn(square))
        self.play(ReplacementTransform(square, circle))
        self.play(FadeOut(circle))
""",
        "📝 Animated Text": """\
from manim import *

class MyScene(Scene):
    def construct(self):
        title = Text("Hello, Manim!", font_size=48)
        subtitle = Text("Visual Mathematics", font_size=32, color=BLUE)
        subtitle.next_to(title, DOWN)
        self.play(Write(title))
        self.play(FadeIn(subtitle))
        self.wait(2)
        self.play(FadeOut(title), FadeOut(subtitle))
""",
        "📐 Geometry Showcase": """\
from manim import *

class MyScene(Scene):
    def construct(self):
        shapes = VGroup(
            Square(color=RED),
            Circle(color=BLUE),
            Triangle(color=GREEN),
        ).arrange(RIGHT, buff=0.5)
        self.play(Create(shapes))
        self.play(shapes.animate.scale(1.5))
        self.wait(1)
        self.play(FadeOut(shapes))
""",
        "📊 Number Line": """\
from manim import *

class MyScene(Scene):
    def construct(self):
        nl = NumberLine(x_range=[-5, 5, 1], include_numbers=True)
        dot = Dot(color=YELLOW).move_to(nl.n2p(0))
        self.play(Create(nl))
        self.play(FadeIn(dot))
        self.play(dot.animate.move_to(nl.n2p(3)))
        self.wait(1)
        self.play(FadeOut(dot), FadeOut(nl))
""",
        "✨ Emphasis & Highlight": """\
from manim import *

class MyScene(Scene):
    def construct(self):
        eq = MathTex(r"E = mc^2", font_size=64)
        self.play(Write(eq))
        self.play(Indicate(eq, color=YELLOW, scale_factor=1.3))
        self.play(Circumscribe(eq, color=RED))
        self.wait(1)
        self.play(FadeOut(eq))
""",
        "🔢 Axes & Plot": """\
from manim import *

class MyScene(Scene):
    def construct(self):
        axes = Axes(x_range=[-3, 3], y_range=[-2, 2])
        curve = axes.plot(lambda x: x**2, color=YELLOW)
        label = axes.get_graph_label(curve, "x^2")
        self.play(Create(axes))
        self.play(Create(curve), Write(label))
        self.wait(2)
        self.play(FadeOut(axes), FadeOut(curve), FadeOut(label))
""",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        title = QLabel("📚 Code Snippets")
        font = QFont()
        font.setPointSize(10)
        title.setFont(font)
        layout.addWidget(title)

        self.list = QListWidget()
        for name in self.SNIPPETS:
            self.list.addItem(name)
        self.list.itemDoubleClicked.connect(self._on_double_click)
        self.list.setToolTip("Double-click to load snippet into AI panel")
        layout.addWidget(self.list)

        btn = QPushButton("📋 Load Snippet into AI")
        btn.clicked.connect(self._load_selected)
        layout.addWidget(btn)

    def _on_double_click(self, item):
        self.snippet_requested.emit(self.SNIPPETS[item.text()])

    def _load_selected(self):
        item = self.list.currentItem()
        if item:
            self.snippet_requested.emit(self.SNIPPETS[item.text()])


# ── Node Search / Filter Bar ──────────────────────────────────────────────────
class NodeSearchBar(QWidget):
    """Toolbar for searching and filtering nodes on the canvas."""

    filter_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        self.search = QLineEdit()
        self.search.setPlaceholderText("🔍 Filter nodes…")
        self.search.textChanged.connect(self.filter_changed)
        layout.addWidget(self.search)

        btn_clear = QPushButton("✕")
        btn_clear.setFixedWidth(28)
        btn_clear.clicked.connect(self.search.clear)
        btn_clear.setToolTip("Clear filter")
        layout.addWidget(btn_clear)


# ── Quick Export Toolbar ───────────────────────────────────────────────────────
class QuickExportBar(QWidget):
    """One-click export actions."""

    export_requested = Signal(str)  # format

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        for label, fmt in [
            ("📄 .py", "py"),
            ("📋 Copy", "copy"),
            ("🎬 Render MP4", "mp4"),
        ]:
            btn = QPushButton(label)
            btn.setFixedHeight(28)
            btn.clicked.connect(lambda _=False, f=fmt: self.export_requested.emit(f))
            layout.addWidget(btn)
        layout.addStretch()


# ==============================================================================
# 8. PANELS & WIDGETS (WITHOUT SCENES TAB)
# ==============================================================================


class VGroupPanel(QWidget):
    """Panel for creating, viewing, and managing VGroups from all sources.

    Supports canvas-created groups (with live NodeItem references) and
    code-origin groups parsed from AI output, local snippets, and GitHub
    snippets. All groups show their members in the tree.
    """

    vgroup_created = Signal(str, list)

    # Source labels and icons
    _SOURCE_ICON = {"canvas": "🎨", "ai": "🤖", "snippet": "📄", "github": "🐙"}
    _SOURCE_COLOR = {
        "canvas": "#4f46e5",
        "ai": "#7c3aed",
        "snippet": "#0891b2",
        "github": "#059669",
    }

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        # name → list of node IDs (canvas) or [] (code-origin)
        self._groups: dict[str, list] = {}
        # name → {"source": str, "members": [str]}
        self._meta: dict[str, dict] = {}
        self._build()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # ── Create section ──────────────────────────────────────────────────
        create_box = QGroupBox("Create VGroup")
        create_box.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid #d1d5db; "
            "border-radius: 6px; margin-top: 8px; padding-top: 4px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; }"
        )
        create_layout = QVBoxLayout(create_box)
        create_layout.setSpacing(4)

        name_row = QHBoxLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Name (e.g. my_group)")
        self.name_edit.setToolTip("Valid Python identifier for this VGroup")
        name_row.addWidget(self.name_edit)
        create_layout.addLayout(name_row)

        btn_create = QPushButton("📦  Create from Canvas Selection")
        btn_create.clicked.connect(self._create_vgroup)
        btn_create.setToolTip("Select Mobject nodes on the canvas first, then click")
        btn_create.setStyleSheet(
            "QPushButton { background-color: #4f46e5; color: white; font-weight: bold;"
            " border-radius: 5px; padding: 7px 10px; }"
            "QPushButton:hover { background-color: #4338ca; }"
            "QPushButton:pressed { background-color: #3730a3; }"
        )
        create_layout.addWidget(btn_create)

        hint = QLabel("Select Mobject nodes on the canvas first.")
        hint.setStyleSheet("color: #9ca3af; font-size: 10px;")
        create_layout.addWidget(hint)
        layout.addWidget(create_box)

        # ── Search ──────────────────────────────────────────────────────────
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("🔍  Filter groups…")
        self.search_edit.textChanged.connect(self._filter_tree)
        self.search_edit.setStyleSheet(
            "QLineEdit { border: 1px solid #d1d5db; border-radius: 5px; padding: 5px; }"
        )
        layout.addWidget(self.search_edit)

        # ── Group count label ────────────────────────────────────────────────
        self.count_lbl = QLabel("No groups yet")
        self.count_lbl.setStyleSheet("color: #6b7280; font-size: 10px;")
        layout.addWidget(self.count_lbl)

        # ── Tree ────────────────────────────────────────────────────────────
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        self.tree.setAlternatingRowColors(True)
        self.tree.setStyleSheet(
            "QTreeWidget { border: 1px solid #e5e7eb; border-radius: 5px; }"
            "QTreeWidget::item { padding: 3px 2px; }"
            "QTreeWidget::item:selected { background: #ede9fe; color: #4f46e5; }"
        )
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.tree, stretch=1)

        # ── Action toolbar ──────────────────────────────────────────────────
        toolbar_box = QGroupBox("Actions")
        toolbar_box.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid #d1d5db; "
            "border-radius: 6px; margin-top: 8px; padding-top: 4px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; }"
        )
        tb_layout = QVBoxLayout(toolbar_box)
        tb_layout.setSpacing(4)

        # Row 1: group-level actions
        row1 = QHBoxLayout()
        row1.setSpacing(4)

        self.btn_rename = self._make_btn(
            "✏️ Rename", "#0891b2", self._rename_vgroup, "Rename the selected VGroup"
        )
        self.btn_duplicate = self._make_btn(
            "⧉ Duplicate",
            "#0891b2",
            self._duplicate_vgroup,
            "Create a copy of this VGroup with a new name",
        )
        self.btn_copy_code = self._make_btn(
            "📋 Copy Code",
            "#059669",
            self._copy_code,
            "Copy  name = VGroup(members…)  to clipboard",
        )
        row1.addWidget(self.btn_rename)
        row1.addWidget(self.btn_duplicate)
        row1.addWidget(self.btn_copy_code)
        tb_layout.addLayout(row1)

        # Row 2: canvas-interaction actions
        row2 = QHBoxLayout()
        row2.setSpacing(4)

        self.btn_highlight = self._make_btn(
            "🎯 Highlight",
            "#7c3aed",
            self._highlight_on_canvas,
            "Select all member nodes on the canvas",
        )
        self.btn_add_members = self._make_btn(
            "＋ Add Nodes",
            "#059669",
            self._add_members,
            "Add currently selected canvas nodes to this group",
        )
        self.btn_remove_member = self._make_btn(
            "－ Remove Member",
            "#f59e0b",
            self._remove_member,
            "Remove the selected member from this group",
        )
        row2.addWidget(self.btn_highlight)
        row2.addWidget(self.btn_add_members)
        row2.addWidget(self.btn_remove_member)
        tb_layout.addLayout(row2)

        # Row 3: destructive
        row3 = QHBoxLayout()
        row3.addStretch()
        self.btn_delete = self._make_btn(
            "🗑 Delete Group",
            "#dc2626",
            self._delete_vgroup,
            "Permanently remove this VGroup (does not delete nodes)",
        )
        self.btn_delete.setMinimumWidth(130)
        row3.addWidget(self.btn_delete)
        tb_layout.addLayout(row3)

        layout.addWidget(toolbar_box)

        # Initial button state
        self._set_buttons_enabled(False)

    @staticmethod
    def _make_btn(label: str, color: str, slot, tooltip: str) -> QPushButton:
        btn = QPushButton(label)
        btn.setToolTip(tooltip)
        btn.clicked.connect(slot)
        btn.setStyleSheet(
            f"QPushButton {{ background-color: {color}; color: white; font-size: 11px;"
            f" border-radius: 4px; padding: 5px 6px; }}"
            f"QPushButton:hover {{ opacity: 0.85; }}"
            f"QPushButton:disabled {{ background-color: #d1d5db; color: #9ca3af; }}"
        )
        return btn

    # ── Button state management ──────────────────────────────────────────────

    def _set_buttons_enabled(self, group_selected: bool, member_selected: bool = False):
        for btn in (
            self.btn_rename,
            self.btn_duplicate,
            self.btn_copy_code,
            self.btn_highlight,
            self.btn_add_members,
            self.btn_delete,
        ):
            btn.setEnabled(group_selected)
        self.btn_remove_member.setEnabled(member_selected)

    def _on_selection_changed(self):
        item = self.tree.currentItem()
        if item is None:
            self._set_buttons_enabled(False)
        elif item.parent() is None:
            # Top-level = group row
            self._set_buttons_enabled(True, False)
        else:
            # Child = member row
            self._set_buttons_enabled(True, True)

    # ── Group resolution helpers ─────────────────────────────────────────────

    def _current_group_name(self) -> str | None:
        """Return the group name from the currently selected tree item."""
        item = self.tree.currentItem()
        if item is None:
            return None
        if item.parent() is None:
            return item.data(0, Qt.ItemDataRole.UserRole)
        return item.parent().data(0, Qt.ItemDataRole.UserRole)

    def _member_display_names(self, name: str) -> list[str]:
        """Return the display names for members of group `name`."""
        ids = self._groups.get(name, [])
        meta = self._meta.get(name, {})
        # Canvas group: resolve node names from live NodeItems
        if ids:
            names = []
            for nid in ids:
                node = self.main_window.nodes.get(nid)
                names.append(node.data.name if node else f"<missing:{nid[:6]}>")
            return names
        # Code-origin group: use parsed member names
        return meta.get("members", [])

    # ── Tree refresh ─────────────────────────────────────────────────────────

    def _refresh_tree(self):
        filter_txt = (
            self.search_edit.text().lower() if hasattr(self, "search_edit") else ""
        )
        self.tree.clear()
        for gname, ids in self._groups.items():
            if filter_txt and filter_txt not in gname.lower():
                continue
            meta = self._meta.get(gname, {})
            source = meta.get("source", "canvas")
            icon = self._SOURCE_ICON.get(source, "📦")
            color = self._SOURCE_COLOR.get(source, "#4f46e5")
            members = self._member_display_names(gname)
            count_str = f"{len(members)} member{'s' if len(members) != 1 else ''}"
            root = QTreeWidgetItem(self.tree)
            root.setData(0, Qt.ItemDataRole.UserRole, gname)
            # Header text: icon + name + source badge + count
            root.setText(0, f"{icon}  {gname}  ·  {source}  ·  {count_str}")
            root.setForeground(0, QBrush(QColor(color)))
            root.setFont(0, QFont("Segoe UI", 10, QFont.Weight.Bold))
            root.setToolTip(
                0,
                f"Source: {source}\nMembers: {', '.join(members) if members else 'none'}",
            )

            if members:
                for mname in members:
                    child = QTreeWidgetItem(root)
                    child.setText(0, f"    └  {mname}")
                    child.setForeground(0, QBrush(QColor("#374151")))
                    child.setData(0, Qt.ItemDataRole.UserRole + 1, mname)
            else:
                empty = QTreeWidgetItem(root)
                empty.setText(0, "    (no members resolved)")
                empty.setForeground(0, QBrush(QColor("#9ca3af")))
                f = empty.font(0)
                f.setItalic(True)
                empty.setFont(0, f)

            root.setExpanded(True)

        n = len(self._groups)
        self.count_lbl.setText(
            f"{n} group{'s' if n != 1 else ''}" if n else "No groups yet"
        )
        self._set_buttons_enabled(False)

    def _filter_tree(self, _txt: str):
        self._refresh_tree()

    # ── Create ───────────────────────────────────────────────────────────────

    def _create_vgroup(self):
        sel = self.main_window.scene.selectedItems()
        members = [
            item
            for item in sel
            if isinstance(item, NodeItem) and item.data.type == NodeType.MOBJECT
        ]
        if not members:
            QMessageBox.warning(
                self,
                "No Selection",
                "Select at least one Mobject node on the canvas first.",
            )
            return
        name = self.name_edit.text().strip() or f"vgroup_{len(self._groups) + 1}"
        if not name.isidentifier():
            QMessageBox.warning(
                self, "Invalid Name", "VGroup name must be a valid Python identifier."
            )
            return
        if name in self._groups:
            QMessageBox.warning(
                self, "Duplicate Name", f"A VGroup named '{name}' already exists."
            )
            return
        ids = [m.data.id for m in members]
        self._groups[name] = ids
        self._meta[name] = {
            "source": "canvas",
            "members": [m.data.name for m in members],
        }
        self.vgroup_created.emit(name, ids)
        self._refresh_tree()
        self.name_edit.clear()
        # Select the new group in tree
        for i in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(i)
            if it.data(0, Qt.ItemDataRole.UserRole) == name:
                self.tree.setCurrentItem(it)
                break

    # ── Rename ───────────────────────────────────────────────────────────────

    def _rename_vgroup(self):
        name = self._current_group_name()
        if not name:
            return
        new_name, ok = QInputDialog.getText(
            self, "Rename VGroup", "New name:", QLineEdit.EchoMode.Normal, name
        )
        if not ok:
            return
        new_name = new_name.strip()
        if not new_name or new_name == name:
            return
        if not new_name.isidentifier():
            QMessageBox.warning(
                self, "Invalid Name", "VGroup name must be a valid Python identifier."
            )
            return
        if new_name in self._groups:
            QMessageBox.warning(
                self, "Duplicate Name", f"A VGroup named '{new_name}' already exists."
            )
            return
        # Re-insert preserving order
        self._groups = {
            (new_name if k == name else k): v for k, v in self._groups.items()
        }
        self._meta = {(new_name if k == name else k): v for k, v in self._meta.items()}
        self._refresh_tree()

    # ── Duplicate ────────────────────────────────────────────────────────────

    def _duplicate_vgroup(self):
        name = self._current_group_name()
        if not name:
            return
        base = f"{name}_copy"
        candidate = base
        i = 2
        while candidate in self._groups:
            candidate = f"{base}_{i}"
            i += 1
        self._groups[candidate] = list(self._groups[name])
        self._meta[candidate] = dict(self._meta.get(name, {}))
        self._meta[candidate]["source"] = self._meta.get(name, {}).get(
            "source", "canvas"
        )
        self._refresh_tree()

    # ── Copy Code ────────────────────────────────────────────────────────────

    def _copy_code(self):
        name = self._current_group_name()
        if not name:
            return
        members = self._member_display_names(name)
        args = ", ".join(members) if members else "# no members"
        code = f"{name} = VGroup({args})"
        QApplication.clipboard().setText(code)
        # Brief visual feedback via tooltip on button
        self.btn_copy_code.setToolTip(f"Copied: {code}")
        QTimer.singleShot(
            2500,
            lambda: self.btn_copy_code.setToolTip(
                "Copy  name = VGroup(members…)  to clipboard"
            ),
        )

    # ── Highlight on Canvas ───────────────────────────────────────────────────

    def _highlight_on_canvas(self):
        name = self._current_group_name()
        if not name:
            return
        ids = self._groups.get(name, [])
        if not ids:
            QMessageBox.information(
                self,
                "Canvas Highlight",
                "This group was created from code — member nodes are not "
                "tracked on the canvas.\nAdd them manually to enable highlighting.",
            )
            return
        # Deselect all, then select members
        self.main_window.scene.clearSelection()
        found = 0
        for nid in ids:
            node = self.main_window.nodes.get(nid)
            if node:
                node.setSelected(True)
                found += 1
        if found == 0:
            QMessageBox.information(
                self,
                "Canvas Highlight",
                "None of the member nodes were found on the canvas.",
            )

    # ── Add Members from Selection ────────────────────────────────────────────

    def _add_members(self):
        name = self._current_group_name()
        if not name:
            return
        sel = self.main_window.scene.selectedItems()
        new_nodes = [
            item
            for item in sel
            if isinstance(item, NodeItem)
            and item.data.type == NodeType.MOBJECT
            and item.data.id not in self._groups.get(name, [])
        ]
        if not new_nodes:
            QMessageBox.information(
                self,
                "Add Members",
                "Select new Mobject nodes on the canvas first.\n"
                "(Already-added nodes are ignored.)",
            )
            return
        for node in new_nodes:
            self._groups.setdefault(name, []).append(node.data.id)
            meta = self._meta.setdefault(name, {"source": "canvas", "members": []})
            if node.data.name not in meta.get("members", []):
                meta.setdefault("members", []).append(node.data.name)
        self._refresh_tree()

    # ── Remove Member ─────────────────────────────────────────────────────────

    def _remove_member(self):
        item = self.tree.currentItem()
        if item is None or item.parent() is None:
            return
        group_name = item.parent().data(0, Qt.ItemDataRole.UserRole)
        member_name = item.data(0, Qt.ItemDataRole.UserRole + 1)
        if not group_name or not member_name:
            return
        # Remove from ID list (canvas groups)
        ids = self._groups.get(group_name, [])
        new_ids = []
        for nid in ids:
            node = self.main_window.nodes.get(nid)
            if node and node.data.name == member_name:
                continue  # skip this one
            new_ids.append(nid)
        self._groups[group_name] = new_ids
        # Remove from meta members list (code-origin groups)
        meta_members = self._meta.get(group_name, {}).get("members", [])
        if member_name in meta_members:
            meta_members.remove(member_name)
        self._refresh_tree()

    # ── Delete Group ─────────────────────────────────────────────────────────

    def _delete_vgroup(self):
        name = self._current_group_name()
        if not name:
            return
        if (
            QMessageBox.question(
                self,
                "Delete VGroup",
                f"Delete VGroup '{name}'?\n\nThis does NOT delete the canvas nodes.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        self._groups.pop(name, None)
        self._meta.pop(name, None)
        self._refresh_tree()

    # ── Public API ────────────────────────────────────────────────────────────

    def get_groups(self) -> dict:
        return dict(self._groups)

    def set_groups(self, groups: dict) -> None:
        self._groups = dict(groups)
        # Rebuild meta for any groups that don't have it
        for name in self._groups:
            if name not in self._meta:
                self._meta[name] = {"source": "canvas", "members": []}
        self._refresh_tree()

    def register_snippet_vgroups(self, code: str, source: str = "snippet") -> int:
        """Parse code for VGroup assignments and register any new ones.

        Parses patterns like:  my_group = VGroup(circle, square, text)
        Extracts member variable names so they appear in the tree.
        Returns the count of newly registered VGroups.
        """
        import re

        # Capture name and the full argument list (handles multi-line with re.DOTALL)
        pattern = re.compile(
            r"^[ \t]*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*VGroup\s*\(([^)]*)\)",
            re.MULTILINE,
        )
        _SKIP = frozenset(("self", "cls", "None", "True", "False", "return"))
        new_count = 0
        for match in pattern.finditer(code):
            name = match.group(1)
            if name.startswith("__") or name in _SKIP:
                continue
            if name in self._groups:
                continue
            # Parse member names from argument list
            raw_args = match.group(2)
            members = []
            for arg in raw_args.split(","):
                arg = arg.strip()
                # Accept simple identifiers only (skip *args, keyword=val, literals)
                if re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", arg) and arg not in _SKIP:
                    members.append(arg)
            self._groups[name] = []  # no node IDs for code-origin groups
            self._meta[name] = {"source": source, "members": members}
            new_count += 1
        if new_count:
            self._refresh_tree()
        return new_count


# ── Keybindings Panel ─────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# OLD KeybindingsPanel REMOVED
# Replaced by UnifiedKeybindingsPanel from keybindings_panel.py
# which uses the unified KeybindingRegistry for single source of truth
# ──────────────────────────────────────────────────────────────────────────────


# ── Recents & Usage helpers (thin wrappers around UserDataManager) ─────────────


def _add_to_recents(path: str) -> None:
    """Register a project path in the recents list."""
    USER_DATA.add_recent(path)


class _UsageTracker(QObject):
    """Central usage tracker — counts Mobject/Animation insertions per class.

    Emits `updated` after every record() call so subscribed panels
    can refresh without polling or manual refresh buttons.
    """

    updated = Signal()

    def __init__(self):
        super().__init__()

    def record(self, class_name: str, node_type: str = "mobject") -> None:
        """Record one insertion of class_name (pass type_str from the caller)."""
        USER_DATA.record_use(class_name, node_type)
        self.updated.emit()

    def top_mobjects(self, n: int = 5) -> list:
        """[(name, count), ...] for the top-n most-used Mobjects."""
        return USER_DATA.top_by_type("mobject", n)

    def top_animations(self, n: int = 5) -> list:
        """[(name, count), ...] for the top-n most-used Animations."""
        return USER_DATA.top_by_type("animation", n)

    def top(self, n: int = 5) -> list:
        """Legacy flat list."""
        return USER_DATA.top_used(n)


USAGE_TRACKER = _UsageTracker()


class RecentsPanel(QWidget):
    """⭐ Recents — shows the top 5 Mobjects and top 5 Animations by actual
    insertion frequency.  Double-click any item to insert it on the canvas.

    Updates live: no manual refresh required.
    """

    add_requested = Signal(str, str)  # (type_str, class_name)

    # How many to show per category
    TOP_N = 5

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()
        # Auto-refresh whenever a node is inserted anywhere
        USAGE_TRACKER.updated.connect(self._refresh)
        self._refresh()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Header
        hdr = QLabel("⭐  Most-Used Elements")
        hdr.setStyleSheet("font-weight: bold; font-size: 13px; color: #1f2937;")
        layout.addWidget(hdr)

        sub = QLabel(
            f"Top {self.TOP_N} Mobjects and Animations ranked by how often "
            "you insert them.  Double-click to add to canvas."
        )
        sub.setWordWrap(True)
        sub.setStyleSheet("color: #6b7280; font-size: 10px;")
        layout.addWidget(sub)

        # Tree
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        self.tree.setStyleSheet(
            "QTreeWidget { border: 1px solid #e5e7eb; border-radius: 5px; }"
            "QTreeWidget::item { padding: 4px 2px; }"
            "QTreeWidget::item:selected { background: #fef9c3; color: #92400e; }"
        )
        self.tree.itemDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self.tree, stretch=1)

        # Footer hint
        footer = QLabel("💡 Insert nodes from Elements or Classes tab to populate.")
        footer.setWordWrap(True)
        footer.setStyleSheet("color: #9ca3af; font-size: 10px; font-style: italic;")
        self.footer_lbl = footer
        layout.addWidget(footer)

    def _refresh(self):
        """Rebuild the tree from live usage statistics."""
        self.tree.clear()

        mob_data = USAGE_TRACKER.top_mobjects(self.TOP_N)
        anim_data = USAGE_TRACKER.top_animations(self.TOP_N)

        self._populate_section("📦  Mobjects", mob_data, "#4f46e5", "mobject")
        self._populate_section("🎬  Animations", anim_data, "#7c3aed", "animation")

        # Show/hide hint depending on whether there is any data
        has_data = bool(mob_data or anim_data)
        self.footer_lbl.setVisible(not has_data)

    def _populate_section(
        self,
        label: str,
        data: list,  # [(class_name, count), ...]
        color: str,
        type_str: str,
    ):
        section = QTreeWidgetItem(self.tree)
        section.setText(0, label)
        section.setForeground(0, QBrush(QColor(color)))
        section.setFont(0, _bold_font())
        section.setFlags(section.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        section.setToolTip(
            0, f"Double-click a {type_str} below to add it to the canvas"
        )

        if not data:
            empty = QTreeWidgetItem(section)
            empty.setText(0, "   (none yet)")
            empty.setForeground(0, QBrush(QColor("#9ca3af")))
            f = empty.font(0)
            f.setItalic(True)
            empty.setFont(0, f)
            empty.setFlags(empty.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        else:
            max_count = data[0][1] if data else 1
            for rank, (cls_name, count) in enumerate(data, 1):
                child = QTreeWidgetItem(section)
                # Bar width proportional to usage (1–10 chars)
                bar_len = max(1, round(count / max_count * 10))
                bar = "█" * bar_len
                child.setText(0, f"  {rank}. {cls_name}")
                plural = "s" if count != 1 else ""
                tip = f"{cls_name}\nInserted {count} time{plural}\nDouble-click to add to canvas"
                child.setToolTip(0, tip)
                # Store metadata for insertion
                child.setData(0, Qt.ItemDataRole.UserRole, cls_name)
                child.setData(0, Qt.ItemDataRole.UserRole + 1, type_str)
                # Usage bar as second column label embedded in text
                count_badge = QLabel(
                    f"<span style='color:{color};font-size:10px'>{bar} {count}×</span>"
                )
                count_badge.setContentsMargins(0, 0, 4, 0)
                self.tree.setItemWidget(child, 0, None)  # ensure text side is used
                child.setText(0, f"  {rank}.  {cls_name}  ·  {count}×  {bar}")
                child.setForeground(0, QBrush(QColor("#374151")))

        section.setExpanded(True)

    def _on_double_click(self, item: QTreeWidgetItem, _col: int):
        cls_name = item.data(0, Qt.ItemDataRole.UserRole)
        type_str = item.data(0, Qt.ItemDataRole.UserRole + 1)
        if cls_name and type_str:
            self.add_requested.emit(type_str, cls_name)


def _bold_font() -> QFont:
    f = QFont()
    f.setBold(True)
    return f


# ── GitHub Snippet Loader ─────────────────────────────────────────────────────
_SNIPPETS_DIR: Path = AppPaths.USER_DATA / "github_snippets"


class GitHubSnippetLoader(QWidget):
    snippet_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()
        self._scan_existing()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        hdr = QLabel("🐙 GitHub Snippets")
        hdr.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(hdr)
        url_row = QHBoxLayout()
        self.url_edit = QLineEdit()
        self.url_edit.setPlaceholderText("https://github.com/user/repo")
        url_row.addWidget(self.url_edit)
        btn_clone = QPushButton("⬇ Clone")
        btn_clone.clicked.connect(self._clone)
        btn_clone.setStyleSheet(
            "background-color: #1a7f37; color: white; padding: 4px 8px;"
        )
        url_row.addWidget(btn_clone)
        layout.addLayout(url_row)
        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet("color: #9ca3af; font-size: 11px;")
        layout.addWidget(self.status_lbl)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemDoubleClicked.connect(self._on_select)
        layout.addWidget(self.tree)
        btn_del = QPushButton("🗑 Remove Repo")
        btn_del.clicked.connect(self._remove_repo)
        layout.addWidget(btn_del)

    def _clone(self):
        url = self.url_edit.text().strip()
        if not url:
            return
        try:
            # Support HTTPS/HTTP GitHub URLs and SSH-style GitHub URLs.
            dev_name = None
            repo_name = None

            # SSH-style: git@github.com:user/repo.git
            if url.startswith("git@github.com:"):
                path_part = url[len("git@github.com:") :].rstrip("/")
                parts = path_part.split("/")
                if len(parts) >= 2:
                    dev_name = parts[0]
                    repo_name = parts[1]
                else:
                    raise ValueError("Incomplete SSH GitHub URL")
            else:
                parsed = urlparse(url)
                # Require a proper HTTP(S) URL with github.com as hostname
                if parsed.scheme not in ("http", "https") or parsed.hostname != "github.com":
                    raise ValueError("Not a GitHub HTTPS/HTTP URL")
                path = parsed.path.lstrip("/").rstrip("/")
                parts = path.split("/")
                if len(parts) < 2:
                    raise ValueError("Incomplete GitHub repository path")
                dev_name = parts[0]
                repo_name = parts[1]

            # Normalize repository name (strip optional .git suffix)
            if repo_name.endswith(".git"):
                repo_name = repo_name[: -len(".git")]

        except Exception:
            self.status_lbl.setText("❌ Invalid GitHub URL")
            return
        dest = _SNIPPETS_DIR / dev_name / repo_name
        if dest.exists():
            self.status_lbl.setText(f"⚠️ Already cloned: {dev_name}/{repo_name}")
            return
        self.status_lbl.setText(f"⏳ Cloning {dev_name}/{repo_name}…")
        QApplication.processEvents()
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            result = subprocess.run(
                ["git", "clone", "--depth", "1", url, str(dest)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                self.status_lbl.setText(f"✅ Cloned {dev_name}/{repo_name}")
                self._add_repo_to_tree(dev_name, repo_name, dest)
            else:
                self.status_lbl.setText(f"❌ {result.stderr[:150]}")
        except FileNotFoundError:
            self.status_lbl.setText("❌ 'git' not found — install Git and add to PATH")
        except Exception as e:
            self.status_lbl.setText(f"❌ {str(e)[:100]}")

    def _scan_existing(self):
        self.tree.clear()
        if not _SNIPPETS_DIR.exists():
            return
        for dev_dir in sorted(_SNIPPETS_DIR.iterdir()):
            if not dev_dir.is_dir():
                continue
            for repo_dir in sorted(dev_dir.iterdir()):
                if not repo_dir.is_dir():
                    continue
                self._add_repo_to_tree(dev_dir.name, repo_dir.name, repo_dir)

    def _add_repo_to_tree(self, dev, repo, path):
        root = QTreeWidgetItem(self.tree, [f"{dev}/{repo}"])
        root.setData(0, Qt.ItemDataRole.UserRole, str(path))
        for pf in sorted(path.rglob("*.py")):
            rel = str(pf.relative_to(path))
            child = QTreeWidgetItem(root, [rel])
            child.setData(0, Qt.ItemDataRole.UserRole, str(pf))
        root.setExpanded(True)

    def _on_select(self, item, _col):
        if not item.parent():
            return
        fp = item.data(0, Qt.ItemDataRole.UserRole)
        if fp and Path(fp).exists():
            try:
                self.snippet_selected.emit(Path(fp).read_text(encoding="utf-8"))
                self.status_lbl.setText(f"✅ {Path(fp).name}")
            except Exception as e:
                self.status_lbl.setText(f"❌ {e}")

    def _remove_repo(self):
        item = self.tree.currentItem()
        if not item:
            return
        root = item if not item.parent() else item.parent()
        path = root.data(0, Qt.ItemDataRole.UserRole)
        if not path:
            return
        r = QMessageBox.question(
            self,
            "Remove Repo",
            f"Delete '{root.text(0)}' from disk?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if r == QMessageBox.StandardButton.Yes:
            import shutil
            import stat
            import time

            try:
                # ═══════════════════════════════════════════════════════════════
                # WINDOWS-SAFE RECURSIVE DELETION
                #
                # Fixes WinError 5: Access Denied
                # - chmod files to remove read-only flags
                # - Retry with exponential backoff
                # - OS-specific force-close handling
                # ═══════════════════════════════════════════════════════════════

                def handle_remove_readonly(func, path, exc_info):
                    """
                    Error handler for shutil.rmtree() on Windows.

                    When a file is locked or read-only:
                    1. Make it writable with os.chmod()
                    2. Retry the operation
                    """
                    if not os.access(path, os.W_OK):
                        # File is read-only or locked
                        os.chmod(path, stat.S_IWUSR | stat.S_IRUSR)
                        func(path)
                    else:
                        raise

                # Attempt 1: Standard deletion
                try:
                    shutil.rmtree(path, onerror=handle_remove_readonly)
                    LOGGER.info(f"Successfully deleted GitHub repo: {path}")
                except Exception as e1:
                    # Attempt 2: Force-chmod all files first, then retry
                    LOGGER.warn(
                        f"First deletion attempt failed: {e1}, retrying with force-chmod..."
                    )
                    try:
                        for root_dir, dirs, files in os.walk(path):
                            for d in dirs:
                                os.chmod(
                                    os.path.join(root_dir, d),
                                    stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR,
                                )
                            for f in files:
                                os.chmod(
                                    os.path.join(root_dir, f),
                                    stat.S_IWUSR | stat.S_IRUSR,
                                )
                        time.sleep(0.5)  # Brief delay for file handles to release
                        shutil.rmtree(path)
                        LOGGER.info(f"Successfully deleted GitHub repo (retry): {path}")
                    except Exception as e2:
                        # Attempt 3: Windows-specific: try moving to temp first
                        if sys.platform == "win32":
                            LOGGER.warn(
                                f"Second deletion failed: {e2}, attempting temp relocation..."
                            )
                            try:
                                temp_loc = (
                                    Path(tempfile.gettempdir())
                                    / f"efm_del_{uuid.uuid4().hex[:8]}"
                                )
                                shutil.move(path, str(temp_loc))
                                shutil.rmtree(
                                    str(temp_loc), onerror=handle_remove_readonly
                                )
                                LOGGER.info(
                                    f"Successfully deleted via temp relocation: {path}"
                                )
                            except Exception as e3:
                                raise Exception(
                                    f"Could not delete after 3 attempts. Last error: {e3}"
                                )
                        else:
                            raise Exception(f"Could not delete. Last error: {e2}")

                # Remove from tree on success
                idx = self.tree.indexOfTopLevelItem(root)
                self.tree.takeTopLevelItem(idx)
                self.status_lbl.setText(f"✅ Deleted '{root.text(0)}'")

            except Exception as e:
                LOGGER.error(f"Failed to delete repo: {e}")
                QMessageBox.warning(
                    self, "Error", f"Could not delete repository:\n{str(e)[:100]}"
                )


# ── Editable Project Name Widget ──────────────────────────────────────────────
class ProjectNameWidget(QWidget):
    name_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        lbl = QLabel("📁")
        lbl.setStyleSheet("font-size: 14px;")
        layout.addWidget(lbl)
        self.edit = QLineEdit("Untitled")
        self.edit.setFixedWidth(160)
        self.edit.setPlaceholderText("Project name…")
        self.edit.setToolTip(
            "Editable project name. Press Enter to rename the .efp file."
        )
        self.edit.returnPressed.connect(self._on_commit)
        self.edit.editingFinished.connect(self._on_commit)
        layout.addWidget(self.edit)
        sfx = QLabel(".efp")
        sfx.setStyleSheet("color: #6b7280;")
        layout.addWidget(sfx)

    def _on_commit(self):
        name = self.edit.text().strip()
        if name:
            self.name_changed.emit(name)

    def set_name(self, stem: str):
        self.edit.blockSignals(True)
        self.edit.setText(stem)
        self.edit.blockSignals(False)

    def get_name(self) -> str:
        return self.edit.text().strip() or "Untitled"


# ==============================================================================
# 9. MAIN WINDOW
# ==============================================================================


class EfficientManimWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.resize(1600, 1000)
        self.setStyleSheet(THEME_MANAGER.get_stylesheet())

        # CRITICAL FIX: Set icon with absolute path before showing window
        icon_path = Path(__file__).parent / "icon" / "icon.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path.absolute())))
        else:
            # Fallback to relative path if absolute doesn't work
            self.setWindowIcon(QIcon("icon/icon.ico"))

        self.nodes = {}
        self.project_path = None
        self.undo_manager = UndoRedoManager()
        self.project_modified = False
        self.is_ai_generated_code = False

        # ═════════════════════════════════════════════════════════════════════
        # Initialize unified keybinding registry
        # ═════════════════════════════════════════════════════════════════════
        if KEYBINDINGS_AVAILABLE:
            initialize_default_keybindings()
            KEYBINDINGS.binding_changed.connect(self._on_keybinding_changed)
            KEYBINDINGS.registry_updated.connect(self._refresh_keybindings)

        # Multi-scene storage: scene_name -> {nodes_serialized, wires_serialized}
        self._all_scenes: dict = {"Scene 1": {"nodes": {}, "wires": []}}
        self._current_scene_name = "Scene 1"

        # Keybindings (unified registry with single UI panel)
        self._keybindings = UnifiedKeybindingsPanel(self)

        AppPaths.ensure_dirs()
        self.init_font()

        # Connect theme change signal for dynamic switching

        self.setup_ui()
        self.setup_menu()
        self.apply_theme()

        # Initialize built-in extensions and realize their panels
        self._initialize_extensions()

        # Refresh Elements panel to show extension nodes
        if hasattr(self, "panel_elems"):
            self.panel_elems.populate()
        
        # FIX: Ensure window is destroyed on close to trigger destroyed signal in home.py
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        self.render_queue = []
        self.render_timer = QTimer()
        self.render_timer.timeout.connect(self.process_render_queue)
        self.render_timer.start(500)

        # ═══════════════════════════════════════════════════════════════════════
        # AUTO-RELOAD SYSTEM
        #
        # Provides 3-second periodic preview refresh + immediate reload on changes
        # - Detects code changes via content hash
        # - Detects node graph changes via structure hash
        # - Detects asset changes via modification timestamps
        # - Prevents render flooding with debounce + in-progress flag
        # - Respectsuser setting: ENABLE_PREVIEW
        # ═══════════════════════════════════════════════════════════════════════

        self.auto_reload_enabled = True  # Toggleable via Settings UI
        self.auto_reload_timer = QTimer()
        self.auto_reload_timer.timeout.connect(self._auto_reload_tick)
        self.auto_reload_timer.start(3000)  # Fire every 3 seconds

        # State tracking for change detection
        self._last_code_hash = ""
        self._last_graph_hash = ""
        self._last_assets_hash = ""

        # Debounce: pending render from changes
        self._pending_auto_render = False
        self._auto_render_debounce = QTimer()
        self._auto_render_debounce.timeout.connect(self._trigger_auto_render)
        self._auto_render_debounce.setSingleShot(True)

        # In-progress tracking: prevent concurrent renders
        self._render_in_progress = False

        LOGGER.info(
            "Auto-reload system initialized: "
            "3s timer + change detection + debounce + render-in-progress guard"
        )

        # ── MCP Agent — wired to this window, available app-wide as self.mcp ──
        if MCP_AVAILABLE and _MCPAgent is not None:
            self.mcp: "_MCPAgent | None" = _MCPAgent(self)  # pyright: ignore[reportInvalidTypeForm]
            LOGGER.info(
                "MCP Agent initialised. Use self.mcp.execute(command, payload)."
            )
        else:
            self.mcp = None
            LOGGER.info("MCP Agent not available (mcp.py missing).")

        # Notify the AI Panel about the live mcp instance now that window is ready
        if hasattr(self, "panel_ai") and self.mcp is not None:
            self.panel_ai.set_mcp_agent(self.mcp)

        LOGGER.info("System Ready.")

    def init_font(self):
        """Initialize system fonts cleanup."""
        default_font = QFont()
        default_font.setPointSize(10)
        self.setFont(default_font)

    def setup_ui(self):
        # NOTE: This is QMainWindow. It MUST use setCentralWidget.
        main = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(main)

        # --- LEFT SIDE (Graph + Video Splitter) ---
        left_splitter = QSplitter(Qt.Orientation.Vertical)

        # 1. Top: Graph Scene
        self.scene = GraphScene()
        self.scene.selection_changed_signal.connect(self.on_selection)
        self.scene.graph_changed_signal.connect(self.mark_modified)
        self.scene.graph_changed_signal.connect(self.compile_graph)

        # Node search/filter bar + canvas
        canvas_widget = QWidget()
        canvas_layout = QVBoxLayout(canvas_widget)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.setSpacing(2)

        self.node_search_bar = NodeSearchBar()
        self.node_search_bar.filter_changed.connect(self._filter_nodes)
        canvas_layout.addWidget(self.node_search_bar)

        self.view = GraphView(self.scene)
        canvas_layout.addWidget(self.view)

        self.quick_export_bar = QuickExportBar()
        self.quick_export_bar.export_requested.connect(self._quick_export)
        canvas_layout.addWidget(self.quick_export_bar)

        left_splitter.addWidget(canvas_widget)

        # 2. Bottom: Video Output Panel (NEW)
        self.panel_output = VideoOutputPanel()
        left_splitter.addWidget(self.panel_output)

        left_splitter.setSizes([700, 300])
        main.addWidget(left_splitter)

        # --- RIGHT SIDE (Tabs) ---
        right = QSplitter(Qt.Orientation.Vertical)
        self.tabs_top = QTabWidget()

        # Initialize Panels
        self.panel_props = PropertiesPanel()
        self.panel_props.node_updated.connect(self.mark_modified)
        self.panel_props.node_updated.connect(self.on_node_changed)

        self.panel_outliner = SceneOutlinerPanel(self)

        self.panel_elems = ElementsPanel()
        self.panel_elems.add_requested.connect(self.add_node_center)

        # NEW: Manim Class Browser
        self.panel_class_browser = ManimClassBrowser()
        self.panel_class_browser.node_requested.connect(
            self._add_node_from_class_browser
        )

        self.panel_assets = AssetsPanel()

        self.panel_video = VideoRenderPanel()
        self.panel_video.render_requested.connect(self.render_to_video)

        self.panel_ai = AIPanel()
        self.panel_ai.merge_requested.connect(self.merge_ai_code)

        # NEW: Snippet Library
        self.panel_snippets = SnippetLibrary()
        self.panel_snippets.snippet_requested.connect(self._load_snippet_to_ai)

        self.panel_voice = VoiceoverPanel(self)

        # NEW: Initialize LaTeX Panel
        self.panel_latex = LatexEditorPanel(self)

        # Add Tabs
        # New panels: Scene Manager, VGroup, GitHub Snippets, Recents
        self.panel_vgroup = VGroupPanel(self)
        self.panel_vgroup.vgroup_created.connect(self.mark_modified)

        self.panel_github = GitHubSnippetLoader()
        self.panel_github.snippet_selected.connect(self._load_github_snippet_to_ai)

        # Recents panel — top-5 Mobjects and Animations by insertion frequency
        self.panel_recents = RecentsPanel(self)
        self.panel_recents.add_requested.connect(self.add_node_center)

        self.tabs_top.addTab(self.panel_elems, "📦 Elements")
        self.tabs_top.addTab(self.panel_recents, "⭐ Recents")
        self.tabs_top.addTab(self.panel_class_browser, "🎨 Classes")
        self.tabs_top.addTab(self.panel_vgroup, "🔗 VGroups")
        self.tabs_top.addTab(self.panel_outliner, "📑 Outliner")
        self.tabs_top.addTab(self.panel_props, "🧩 Properties")
        self.tabs_top.addTab(self.panel_assets, "🗂 Assets")
        self.tabs_top.addTab(self.panel_latex, "✒️ LaTeX")
        self.tabs_top.addTab(self.panel_snippets, "📚 Snippets")
        self.tabs_top.addTab(self.panel_github, "🐙 GitHub")
        self.tabs_top.addTab(self.panel_voice, "🎙️ Voiceover")
        self.tabs_top.addTab(self.panel_video, "🎬 Video")
        right.addWidget(self.tabs_top)

        # ── AI Panel: permanent left dock (cannot float or be undocked) ──
        self.ai_dock = QDockWidget("🤖 AI Assistant", self)
        self.ai_dock.setObjectName("AIDockWidget")
        # Lock the dock: no floating, no closing, no moving
        self.ai_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.ai_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea)
        self.ai_dock.setWidget(self.panel_ai)
        self.ai_dock.setMinimumWidth(320)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.ai_dock)

        self.tabs_bot = QTabWidget()

        # Preview Area
        prev_widget = QWidget()
        prev_layout = QVBoxLayout(prev_widget)
        self.preview_lbl = QLabel("Select a node to preview")
        self.preview_lbl.setObjectName("PreviewLabel")
        self.preview_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        prev_layout.addWidget(self.preview_lbl)

        self.code_view = QTextEdit()
        self.code_view.setReadOnly(True)
        self.code_view.setStyleSheet("font-family: Consolas;")

        self.logs = QTextEdit()
        self.logs.setReadOnly(True)
        LOGGER.log_signal.connect(self.append_log)

        self.tabs_bot.addTab(prev_widget, "🖼 Preview")
        self.tabs_bot.addTab(self.code_view, "🧠 Code")
        self.tabs_bot.addTab(self.logs, "📜 Logs")
        right.addWidget(self.tabs_bot)

        main.addWidget(right)
        main.setSizes([1000, 600])

    def setup_menu(self):
        """Setup menu bar with all actions."""
        bar = self.menuBar()

        # --- FIX: Use self.corner_container to prevent Garbage Collection crash ---
        self.corner_container = QWidget()
        corner_layout = QHBoxLayout(self.corner_container)
        corner_layout.setContentsMargins(0, 0, 15, 0)  # 15px right margin

        # Editable project name field in toolbar
        self.project_name_edit = QLineEdit("Untitled Project")
        self.project_name_edit.setFixedWidth(200)
        self.project_name_edit.setPlaceholderText("Project name…")
        self.project_name_edit.setToolTip("Edit project name (renames the .efp file)")
        self.project_name_edit.returnPressed.connect(self._rename_project)
        self.project_name_edit.editingFinished.connect(self._rename_project)
        corner_layout.addWidget(self.project_name_edit)

        bar.setCornerWidget(self.corner_container, Qt.Corner.TopRightCorner)

        # File Menu
        file_menu = bar.addMenu("File")
        file_menu.addAction(
            "New Project", self.new_project, QKeySequence.StandardKey.New
        )
        file_menu.addAction(
            "Open Project", self.open_project, QKeySequence.StandardKey.Open
        )
        file_menu.addAction(
            "Save Project", self.save_project, QKeySequence.StandardKey.Save
        )
        file_menu.addAction("Save As...", self.save_project_as)
        file_menu.addSeparator()
        file_menu.addAction("Settings", self.open_settings)
        file_menu.addSeparator()

        # UNIFIED KEYBINDINGS: Use registry instead of hardcoded shortcuts
        self._quit_action = QAction("Exit", self)
        self._quit_action.setShortcut(KEYBINDINGS.get_binding("Exit") or "Ctrl+Q")
        self._quit_action.triggered.connect(self.close)
        self._quit_action.setObjectName("_quit_action")
        file_menu.addAction(self._quit_action)

        # Edit Menu
        edit_menu = bar.addMenu("Edit")
        self._undo_action = QAction("Undo", self)
        self._undo_action.setShortcut(KEYBINDINGS.get_binding("Undo") or "Ctrl+Z")
        self._undo_action.triggered.connect(self.undo_action)
        self._undo_action.setObjectName("_undo_action")
        edit_menu.addAction(self._undo_action)

        self._redo_action = QAction("Redo", self)
        self._redo_action.setShortcut(KEYBINDINGS.get_binding("Redo") or "Ctrl+Y")
        self._redo_action.triggered.connect(self.redo_action)
        self._redo_action.setObjectName("_redo_action")
        edit_menu.addAction(self._redo_action)

        edit_menu.addSeparator()
        self._delete_action = QAction("Delete Selected", self)
        self._delete_action.setShortcut(
            KEYBINDINGS.get_binding("Delete Selected") or "Del"
        )
        self._delete_action.triggered.connect(self.delete_selected)
        self._delete_action.setObjectName("_delete_action")
        edit_menu.addAction(self._delete_action)

        # View Menu
        view_menu = bar.addMenu("View")
        view_menu.addAction("Fit to View", self.fit_view, QKeySequence("Ctrl+0"))

        # Zoom Shortcuts
        view_menu.addSeparator()
        self._zoom_in_action = QAction("Zoom In", self)
        self._zoom_in_action.setShortcut(KEYBINDINGS.get_binding("Zoom In") or "Ctrl+=")
        self._zoom_in_action.triggered.connect(lambda: self.view.scale(1.15, 1.15))
        self._zoom_in_action.setObjectName("_zoom_in_action")
        view_menu.addAction(self._zoom_in_action)

        self._zoom_out_action = QAction("Zoom Out", self)
        self._zoom_out_action.setShortcut(
            KEYBINDINGS.get_binding("Zoom Out") or "Ctrl+-"
        )
        self._zoom_out_action.triggered.connect(
            lambda: self.view.scale(1 / 1.15, 1 / 1.15)
        )
        self._zoom_out_action.setObjectName("_zoom_out_action")
        view_menu.addAction(self._zoom_out_action)

        view_menu.addSeparator()
        view_menu.addAction(
            "Auto-Layout Nodes", self.auto_layout_nodes, QKeySequence("Ctrl+L")
        )
        view_menu.addAction(
            "Clear All",
            self.clear_scene,
            QKeySequence("Ctrl+Alt+Delete"),
        )

        # Tools Menu (NEW)
        tools_menu = bar.addMenu("Tools")
        tools_menu.addAction(
            "Export Code (.py)",
            lambda: self._quick_export("py"),
            QKeySequence("Ctrl+E"),
        )
        tools_menu.addAction(
            "Copy Code to Clipboard",
            lambda: self._quick_export("copy"),
            QKeySequence("Ctrl+Shift+C"),
        )
        tools_menu.addSeparator()

        # Render Video action (unified keybinding)
        self._render_video_action = QAction("Render Video", self)
        self._render_video_action.setShortcut(
            KEYBINDINGS.get_binding("Render Video") or "Ctrl+R"
        )
        self._render_video_action.triggered.connect(lambda: self.render_to_video({}))
        self._render_video_action.setObjectName("_render_video_action")
        tools_menu.addAction(self._render_video_action)
        tools_menu.addSeparator()

        tools_menu.addAction(
            "Create VGroup from Selection",
            self.create_vgroup_from_selection,
            QKeySequence("Ctrl+G"),
        )
        tools_menu.addSeparator()
        tools_menu.addAction("Open Manim Documentation", self._open_manim_docs)
        tools_menu.addAction("Open Gallery / Examples", self._open_manim_gallery)

        # Help Menu
        help_menu = bar.addMenu("Help")
        help_menu.addAction(
            "Keyboard Shortcuts",
            self.show_shortcuts,
            QKeySequence("Ctrl+?"),
        )
        help_menu.addAction(
            "Edit Keybindings…", self.show_keybindings, QKeySequence("Ctrl+,")
        )
        help_menu.addAction("About", self.show_about)
        help_menu.addSeparator()

        # ── MCP submenu ────────────────────────────────────────────────────
        mcp_menu = help_menu.addMenu("🔌 MCP Agent")

        mcp_status_action = QAction("MCP Status / Ping", self)
        mcp_status_action.setToolTip("Ping the MCP agent and show current node count.")
        mcp_status_action.triggered.connect(self._mcp_ping)
        mcp_menu.addAction(mcp_status_action)

        mcp_context_action = QAction("Inspect Scene Context (JSON)", self)
        mcp_context_action.setToolTip(
            "Show the full MCPContext JSON for the current scene."
        )
        mcp_context_action.triggered.connect(self._mcp_show_context)
        mcp_menu.addAction(mcp_context_action)

        mcp_list_action = QAction("List All Commands", self)
        mcp_list_action.setToolTip("Show all registered MCP command names.")
        mcp_list_action.triggered.connect(self._mcp_list_commands)
        mcp_menu.addAction(mcp_list_action)

        mcp_log_action = QAction("Show Action Log", self)
        mcp_log_action.setToolTip("Show every MCP command executed this session.")
        mcp_log_action.triggered.connect(self._mcp_show_log)
        mcp_menu.addAction(mcp_log_action)

        mcp_menu.addSeparator()
        mcp_compile_action = QAction("Force Compile Graph via MCP", self)
        mcp_compile_action.triggered.connect(
            lambda: self._mcp_exec_and_notify("compile_graph", {})
        )
        mcp_menu.addAction(mcp_compile_action)

        mcp_save_action = QAction("Save Project via MCP", self)
        mcp_save_action.triggered.connect(
            lambda: self._mcp_exec_and_notify("save_project", {})
        )
        mcp_menu.addAction(mcp_save_action)

    # --- MENU ACTIONS ---

    def new_project(self):
        """Create a new project."""
        reply = QMessageBox.question(
            self,
            "New Project",
            "Clear current project?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.clear_scene()

    def save_project_as(self):
        """Save project with new name (Save As)."""
        # Get default filename from project name textbox
        default_filename = self.project_name_edit.text().strip() or "Untitled Project"
        if not default_filename.endswith(PROJECT_EXT):
            default_filename += PROJECT_EXT

        # Use last project path directory if available, otherwise Documents
        last_dir = (
            str(Path(self.project_path).parent)
            if self.project_path
            else str(Path.home() / "Documents")
        )

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project As",
            str(Path(last_dir) / default_filename),
            f"EfficientManim (*{PROJECT_EXT})",
        )
        if not path:
            return

        # Update project path and name
        self.project_path = path
        self.project_name_edit.setText(Path(path).stem)
        self.save_project()

    # ══════════════════════════════════════════════════════════════════════
    # MCP MENU ACTIONS  (Help → MCP Agent submenu)
    # ══════════════════════════════════════════════════════════════════════

    def _mcp_ping(self) -> None:
        """Ping the MCP agent and display status in a message box."""
        if self.mcp is None:
            QMessageBox.warning(
                self,
                "MCP Unavailable",
                "MCP Agent is not initialised.\n"
                "Make sure mcp.py is in the same directory as main.py.",
            )
            return
        result = self.mcp.execute("ping")
        if result.success:
            node_count = result.data.get("node_count", "?")
            QMessageBox.information(
                self,
                "MCP Agent — OK",
                f"✅ MCP Agent is alive.\n\n"
                f"Nodes in current scene: {node_count}\n"
                f"Registered commands: {len(self.mcp.list_commands())}",
            )
        else:
            QMessageBox.critical(self, "MCP Error", f"Ping failed:\n{result.error}")

    def _mcp_show_context(self) -> None:
        """Open a dialog showing the full MCPContext JSON for the current scene."""
        if self.mcp is None:
            QMessageBox.warning(
                self, "MCP Unavailable", "MCP Agent is not initialised."
            )
            return
        result = self.mcp.execute("get_context")
        if not result.success:
            QMessageBox.critical(self, "MCP Error", result.error)
            return
        import json as _json

        ctx_text = _json.dumps(result.data, indent=2, default=str)

        dlg = QDialog(self)
        dlg.setWindowTitle("MCP — Scene Context JSON")
        dlg.resize(700, 550)
        layout = QVBoxLayout(dlg)

        lbl = QLabel(
            f"Current scene: <b>{result.data.get('current_scene', '?')}</b>  |  "
            f"Nodes: <b>{result.data.get('node_count', 0)}</b>  |  "
            f"Assets: <b>{result.data.get('asset_count', 0)}</b>"
        )
        layout.addWidget(lbl)

        txt = QTextEdit()
        txt.setReadOnly(True)
        txt.setFont(QFont("Consolas", 9))
        txt.setPlainText(ctx_text)
        layout.addWidget(txt)

        btn_row = QHBoxLayout()
        btn_copy = QPushButton("📋 Copy JSON")
        btn_copy.clicked.connect(
            lambda: (
                QApplication.clipboard().setText(ctx_text),
                btn_copy.setText("✅ Copied!"),
            )
        )
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dlg.accept)
        btn_row.addWidget(btn_copy)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)
        dlg.exec()

    def _mcp_list_commands(self) -> None:
        """Show all registered MCP command names in a dialog."""
        if self.mcp is None:
            QMessageBox.warning(
                self, "MCP Unavailable", "MCP Agent is not initialised."
            )
            return
        commands = self.mcp.list_commands()
        text = "\n".join(f"  • {c}" for c in commands)
        dlg = QDialog(self)
        dlg.setWindowTitle(f"MCP — {len(commands)} Registered Commands")
        dlg.resize(340, 500)
        layout = QVBoxLayout(dlg)
        lbl = QLabel(f"<b>{len(commands)} commands available:</b>")
        layout.addWidget(lbl)
        txt = QTextEdit()
        txt.setReadOnly(True)
        txt.setFont(QFont("Consolas", 9))
        txt.setPlainText(text)
        layout.addWidget(txt)
        btn = QPushButton("Close")
        btn.clicked.connect(dlg.accept)
        layout.addWidget(btn)
        dlg.exec()

    def _mcp_show_log(self) -> None:
        """Show every MCP command executed this session."""
        if self.mcp is None:
            QMessageBox.warning(
                self, "MCP Unavailable", "MCP Agent is not initialised."
            )
            return
        log = self.mcp.get_action_log()
        if not log:
            QMessageBox.information(
                self, "MCP Action Log", "No commands have been executed yet."
            )
            return
        import json as _json

        lines = []
        for i, entry in enumerate(log, 1):
            payload_str = _json.dumps(entry.get("payload", {}), default=str)
            lines.append(f"[{i:03}] {entry['command']}  {payload_str}")
        text = "\n".join(lines)

        dlg = QDialog(self)
        dlg.setWindowTitle(f"MCP Action Log — {len(log)} entries")
        dlg.resize(680, 480)
        layout = QVBoxLayout(dlg)
        txt = QTextEdit()
        txt.setReadOnly(True)
        txt.setFont(QFont("Consolas", 9))
        txt.setPlainText(text)
        layout.addWidget(txt)
        btn_row = QHBoxLayout()
        btn_copy = QPushButton("📋 Copy Log")
        btn_copy.clicked.connect(lambda: QApplication.clipboard().setText(text))
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dlg.accept)
        btn_row.addWidget(btn_copy)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)
        dlg.exec()

    def _mcp_exec_and_notify(self, command: str, payload: dict) -> None:
        """Execute a single MCP command and show a toast-style result notification."""
        if self.mcp is None:
            QMessageBox.warning(
                self, "MCP Unavailable", "MCP Agent is not initialised."
            )
            return
        result = self.mcp.execute(command, payload)
        if result.success:
            LOGGER.info(f"MCP [{command}] OK: {result.data}")
            QMessageBox.information(
                self, f"MCP — {command}", f"✅ Command succeeded.\n\n{result.data}"
            )
        else:
            LOGGER.error(f"MCP [{command}] FAILED: {result.error}")
            QMessageBox.critical(self, f"MCP — {command} failed", f"❌ {result.error}")

    def show_shortcuts(self):
        """Show keyboard shortcuts dialog."""
        KeyboardShortcutsDialog(self).exec()

    def show_about(self):
        """Show about dialog."""
        QMessageBox.information(
            self,
            "About",
            f"{APP_NAME} v{APP_VERSION}\n\n"
            "A visual node-based Manim IDE with AI integration.\n\n"
            "Features:\n"
            "  • Node graph canvas\n"
            "  • AI code generation (Gemini)\n"
            "  • Live preview rendering\n"
            "  • Code snippet library\n"
            "  • Manim class browser\n"
            "  • One-click export\n\n"
            "© 2026 - Soumalya Das (@pro-grammer-SD)",
        )

    def _open_manim_docs(self):
        """Open Manim documentation in browser."""
        import webbrowser

        webbrowser.open("https://docs.manim.community/en/stable/")

    def _open_manim_gallery(self):
        """Open Manim example gallery in browser."""
        import webbrowser

        webbrowser.open("https://docs.manim.community/en/stable/examples.html")

    def apply_theme(self) -> None:
        """Apply light-mode stylesheet to all widgets."""
        self.setStyleSheet(THEME_MANAGER.get_stylesheet())
        if hasattr(self, "scene"):
            self.scene.setBackgroundBrush(QBrush(QColor("#f4f6f7")))

    def _initialize_extensions(self) -> None:
        """
        Initialize built-in extensions and realize their panels into the main window.

        Called during __init__ after UI setup is complete.
        Built-in extensions:
          • Color Palette   — theme colorizer panel
          • Math Symbols    — Manim node types in the Elements panel
          • Animation Presets — ✨ Quick Effects one-click preset panel
        """
        import logging
        logger = logging.getLogger("extensions")

        # ── Pre-approve permissions for all built-in (first-party) extensions ──
        builtin_permissions = {
            "color-palette":       [PermissionType.REGISTER_UI_PANEL],
            "math-symbols":        [PermissionType.REGISTER_NODES],
            "animation-presets":   [PermissionType.REGISTER_UI_PANEL],
        }
        for ext_id, perms in builtin_permissions.items():
            EXTENSION_MANAGER._permission_manager.auto_approve_permissions(ext_id, perms)
        logger.info("✓ Permissions pre-approved for built-in extensions")

        try:
            from app.extensions.color_palette import (
                setup as setup_color_palette,
                set_main_window as set_color_main_window,
            )
            from app.extensions.math_symbols import setup as setup_math_symbols
            from app.extensions.animation_presets import setup as setup_animation_presets

            # Color Palette — needs a main window reference for live theme propagation
            set_color_main_window(self)
            api_color = ExtensionAPI("color-palette")
            setup_color_palette(api_color)
            logger.info("✓ Color Palette extension initialised")

            # Math Symbols — registers custom Manim node types in the Elements panel
            api_math = ExtensionAPI("math-symbols")
            setup_math_symbols(api_math)
            logger.info("✓ Math Symbols extension initialised")

            # Animation Presets — registers the ✨ Quick Effects panel
            api_presets = ExtensionAPI("animation-presets")
            setup_animation_presets(api_presets)
            logger.info("✓ Animation Presets extension initialised")

            logger.info("✓ Built-in extensions initialised (3 total)")
        except Exception as e:
            logger.error(f"Failed to initialise extensions: {e}", exc_info=True)

        try:
            # Materialise all registered panels into the main window as dock widgets
            realized_panels = EXTENSION_REGISTRY.realize_panels(self)

            for panel_name, widget in realized_panels.items():
                try:
                    widget.setStyleSheet(THEME_MANAGER.get_stylesheet())
                except Exception as e:
                    logger.warning(f"Could not apply theme to panel '{panel_name}': {e}")

            logger.info(f"✅ Realised {len(realized_panels)} extension panels into main window")
            for panel_name, widget in realized_panels.items():
                logger.info(f"   • {panel_name}: {type(widget).__name__}")
        except Exception as e:
            logger.error(f"Failed to realise extension panels: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Failed to realize extension panels: {e}", exc_info=True)

    # ── Project Naming ─────────────────────────────────────────────────────────
    def _rename_project(self):
        """Rename the current project file based on the editable name field."""
        new_name = self.project_name_edit.text().strip()
        if not new_name:
            return
        if not new_name.endswith(".efp"):
            new_name_efp = new_name + ".efp"
        else:
            new_name_efp = new_name
            new_name = new_name[:-4]

        if self.project_path:
            old_path = Path(self.project_path)
            new_path = old_path.parent / new_name_efp
            try:
                if old_path != new_path:
                    old_path.rename(new_path)
                    self.project_path = str(new_path)
                    _add_to_recents(str(new_path))
                    self.mark_modified()
                    self.statusBar().showMessage(f"Renamed to {new_name_efp}", 2000)
            except Exception as e:
                self.statusBar().showMessage(f"Rename failed: {e}", 3000)
        else:
            # No file yet – just update the window title
            self.setWindowTitle(f"{APP_NAME} v{APP_VERSION} - {new_name}")

    def open_project_from_path(self, path: str):
        """Open a project from a given path (used by home screen)."""
        if Path(path).exists():
            self._do_open_project(path)

    # ── Scene Management ───────────────────────────────────────────────────────
    def _save_current_scene_state(self):
        """Serialize current canvas state into _all_scenes[current_scene_name]."""
        try:
            nodes_data = {}
            for nid, node in self.nodes.items():
                nodes_data[nid] = {
                    "name": node.data.name,
                    "cls_name": node.data.cls_name,
                    "var_name": node.data.var_name,
                    "type": node.data.type.name,
                    "params": node.data.params,
                    "x": node.x(),
                    "y": node.y(),
                }
            self._all_scenes[self._current_scene_name] = {
                "nodes": nodes_data,
                "wires": [],  # wire serialization handled elsewhere
            }
        except Exception as e:
            LOGGER.warn(f"Could not save scene state: {e}")

    def _load_scene_state(self, name: str):
        """Clear canvas and load scene state from _all_scenes[name]."""
        try:
            # Clear current canvas
            for nid, node in list(self.nodes.items()):
                self.scene.removeItem(node)
            self.nodes.clear()
            self.scene.clear()

            state = self._all_scenes.get(name, {})
            for nid, ndata in state.get("nodes", {}).items():
                nt = NodeType[ndata.get("type", "MOBJECT")]
                data = NodeData(ndata["cls_name"], nt, ndata["var_name"])
                data.params = ndata.get("params", {})
                node = NodeItem(data)
                node.setPos(ndata.get("x", 0), ndata.get("y", 0))
                self.scene.addItem(node)
                self.nodes[data.id] = node
            self._current_scene_name = name
        except Exception as e:
            LOGGER.warn(f"Could not load scene '{name}': {e}")

    def _on_scene_switch(self, name: str):
        """User switched to a different scene."""
        if name == self._current_scene_name:
            return
        self._save_current_scene_state()
        if name not in self._all_scenes:
            self._all_scenes[name] = {"nodes": {}, "wires": []}
        self._load_scene_state(name)
        self.statusBar().showMessage(f"Scene: {name}", 2000)
        self.compile_graph()

    def _on_scene_added(self, name: str):
        """A new scene was added."""
        self._all_scenes[name] = {"nodes": {}, "wires": []}
        self.statusBar().showMessage(f"New scene: {name}", 1500)

    def _on_scene_deleted(self, name: str):
        """A scene was deleted."""
        if name in self._all_scenes:
            del self._all_scenes[name]
        if name == self._current_scene_name and self._all_scenes:
            # Switch to first remaining scene
            first = next(iter(self._all_scenes))
            self._load_scene_state(first)
        self.statusBar().showMessage(f"Scene '{name}' deleted", 1500)

    # ── Add Node by Class Name (from Recents pane) ─────────────────────────────
    def _add_node_by_class(self, class_name: str, node_type_str: str = "mobject"):
        """Add a node to the canvas by class name."""
        try:
            from main import NodeType, NodeData, NodeItem
        except ImportError:
            pass

        try:
            nt = NodeType.MOBJECT if node_type_str == "mobject" else NodeType.ANIMATION
            data = NodeData(class_name, nt, class_name)
            node = NodeItem(data)
            node.setPos(50, 50 + len(self.nodes) * 120)
            self.scene.addItem(node)
            self.nodes[data.id] = node
            USAGE_TRACKER.record(class_name, node_type_str)
            self.mark_modified()
        except Exception as e:
            LOGGER.warn(f"Could not add node {class_name}: {e}")

    def show_keybindings(self):
        """Show the keybindings dialog."""
        self._keybindings.exec()

    def create_vgroup_from_selection(self):
        """Create a VGroup from selected nodes."""
        selected = [
            item for item in self.scene.selectedItems() if isinstance(item, NodeItem)
        ]
        if selected:
            ids = [item.data.id for item in selected]
            self.panel_vgroup.create_group_from_selection(ids)
            self.statusBar().showMessage(f"VGroup created with {len(ids)} nodes", 2000)

    def mark_modified(self):
        """Mark project as modified and update window title."""
        self.project_modified = True
        title = f"{APP_NAME} v{APP_VERSION}"
        if self.project_path:
            title += f" - {Path(self.project_path).name}"
        title += " *"  # Star indicates unsaved changes
        self.setWindowTitle(title)

    def reset_modified(self):
        """Reset modified flag after save."""
        self.project_modified = False
        title = f"{APP_NAME} v{APP_VERSION}"
        if self.project_path:
            title += f" - {Path(self.project_path).name}"
        self.setWindowTitle(title)

    # ═══════════════════════════════════════════════════════════════════════════
    # KEYBINDING HANDLERS — manage dynamic keybinding changes at runtime
    # ═══════════════════════════════════════════════════════════════════════════

    def _on_keybinding_changed(self, action_name: str, new_shortcut: str) -> None:
        """
        Called when a keybinding is changed in the registry.
        Updates the corresponding QAction immediately (no restart required).
        """
        try:
            # Dictionary mapping action names to their QAction objects
            # These are created in setup_menu()
            action_map = {
                "Exit": getattr(self, "_quit_action", None),
                "Undo": getattr(self, "_undo_action", None),
                "Redo": getattr(self, "_redo_action", None),
                "Delete Selected": getattr(self, "_delete_action", None),
                "Save Project": getattr(self, "_save_action", None),
                "Zoom In": getattr(self, "_zoom_in_action", None),
                "Zoom Out": getattr(self, "_zoom_out_action", None),
                "Render Video": getattr(self, "_render_video_action", None),
            }

            action = action_map.get(action_name)
            if action:
                action.setShortcut(new_shortcut)
                LOGGER.info(f"✓ Rebound '{action_name}' to '{new_shortcut}'")
        except Exception as e:
            LOGGER.error(f"Failed to rebind keybinding '{action_name}': {e}")

    def _refresh_keybindings(self) -> None:
        """
        Refresh all keybindings from registry.
        Called when registry structure changes (e.g., reset to defaults).
        """
        try:
            if not KEYBINDINGS_AVAILABLE:
                return

            # Re-apply all keybindings from registry to their actions
            action_map = {
                "Exit": ("_quit_action", "Ctrl+Q"),
                "Undo": ("_undo_action", "Ctrl+Z"),
                "Redo": ("_redo_action", "Ctrl+Y"),
                "Delete Selected": ("_delete_action", "Del"),
                "Save Project": ("_save_action", "Ctrl+S"),
                "Zoom In": ("_zoom_in_action", "Ctrl+="),
                "Zoom Out": ("_zoom_out_action", "Ctrl+-"),
                "Render Video": ("_render_video_action", "Ctrl+R"),
            }

            for action_name, (attr_name, default) in action_map.items():
                action = getattr(self, attr_name, None)
                if action:
                    shortcut = KEYBINDINGS.get_binding(action_name) or default
                    action.setShortcut(shortcut)

            LOGGER.info("✓ Refreshed all keybindings from registry")
        except Exception as e:
            LOGGER.error(f"Failed to refresh keybindings: {e}")

    def closeEvent(self, event):
        """Intercept close event to check for unsaved changes and cleanup resources."""
        # FIX: Clean up all preview workers before closing
        for node_id in list(self.nodes.keys()):
            self._cleanup_preview_worker(node_id)

        # Clear preview display
        self.preview_lbl.clear()
        self.preview_lbl.setPixmap(QPixmap())

        # Clean up temp files
        AppPaths.force_cleanup_old_files(age_seconds=0)

        if self.project_modified and self.nodes:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Do you want to save before quitting?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
            )

            if reply == QMessageBox.StandardButton.Save:
                self.save_project()
                # If save was cancelled inside save_project, ignore close
                if self.project_modified:
                    event.ignore()
                else:
                    event.accept()
            elif reply == QMessageBox.StandardButton.Discard:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    # ═══════════════════════════════════════════════════════════════════════════
    # AUTO-RELOAD SYSTEM IMPLEMENTATION
    # ═══════════════════════════════════════════════════════════════════════════

    def _compute_code_hash(self) -> str:
        """Compute SHA256 hash of current code view content."""
        try:
            import hashlib

            code = self.code_view.toPlainText()
            return hashlib.sha256(code.encode()).hexdigest()
        except Exception as e:
            LOGGER.error(f"Error computing code hash: {e}")
            return ""

    def _compute_graph_hash(self) -> str:
        """Compute hash of node graph structure (node IDs, connections)."""
        try:
            import hashlib

            # Hash all node IDs + their parameters
            node_data = []
            for nid in sorted(self.nodes.keys()):
                node = self.nodes[nid]
                node_data.append(
                    f"{nid}:{node.data.cls_name}:{sorted(node.data.params.items())}"
                )

            # Hash all edges (connections)
            edge_data = []
            for node in self.nodes.values():
                for wire in node.out_socket.links:
                    edge_data.append(f"{id(wire)}")

            combined = "".join(node_data) + "".join(edge_data)
            return hashlib.sha256(combined.encode()).hexdigest()
        except Exception as e:
            LOGGER.error(f"Error computing graph hash: {e}")
            return ""

    def _compute_assets_hash(self) -> str:
        """Compute hash of asset timestamps and IDs."""
        try:
            import hashlib

            asset_data = []
            for asset in ASSETS.get_list():
                try:
                    mtime = os.path.getmtime(asset.current_path)
                    asset_data.append(f"{asset.id}:{mtime}")
                except Exception:
                    asset_data.append(f"{asset.id}:missing")

            combined = "".join(asset_data)
            return hashlib.sha256(combined.encode()).hexdigest()
        except Exception as e:
            LOGGER.error(f"Error computing assets hash: {e}")
            return ""

    def _auto_reload_tick(self):
        """
        Called every 3 seconds by auto_reload_timer.

        Detects changes in code, graph, or assets.
        Queues re-render if state changed (with debounce).
        """
        try:
            # Skip if disabled or AI-generated code
            if not self.auto_reload_enabled or self.is_ai_generated_code:
                return

            # Skip if render already in progress (prevent flooding)
            if self._render_in_progress:
                return

            # Compute current state
            code_hash = self._compute_code_hash()
            graph_hash = self._compute_graph_hash()
            assets_hash = self._compute_assets_hash()

            # Check for changes
            code_changed = code_hash != self._last_code_hash
            graph_changed = graph_hash != self._last_graph_hash
            assets_changed = assets_hash != self._last_assets_hash

            # Update hashes for next iteration
            self._last_code_hash = code_hash
            self._last_graph_hash = graph_hash
            self._last_assets_hash = assets_hash

            # If anything changed, queue a render with debounce
            if code_changed or graph_changed or assets_changed:
                LOGGER.debug(
                    f"Auto-reload: change detected "
                    f"(code={code_changed}, graph={graph_changed}, assets={assets_changed})"
                )
                self._pending_auto_render = True

                # Debounce: wait 500ms before rendering to avoid flooding
                # if user is making rapid edits
                self._auto_render_debounce.stop()
                self._auto_render_debounce.start(500)

        except Exception as e:
            LOGGER.error(f"Error in _auto_reload_tick: {e}")

    def _trigger_auto_render(self):
        """
        Called after debounce period expires.

        Triggers re-render if there's a pending change.
        """
        try:
            if not self._pending_auto_render:
                return

            self._pending_auto_render = False

            # Check if preview is enabled
            if not SETTINGS.get("ENABLE_PREVIEW", True, type=bool):
                return

            LOGGER.info("Auto-reload: triggering render due to detected changes")

            # Queue all mobject nodes for re-render
            for node in self.nodes.values():
                if node.data.type == NodeType.MOBJECT:
                    self.queue_render(node)

        except Exception as e:
            LOGGER.error(f"Error in _trigger_auto_render: {e}")

    def undo_action(self):
        """Undo last action."""
        if self.undo_manager.undo():
            self.compile_graph()
            LOGGER.info("Undo executed")
        else:
            LOGGER.warn("Nothing to undo")

    def redo_action(self):
        """Redo last undone action."""
        if self.undo_manager.redo():
            self.compile_graph()
            LOGGER.info("Redo executed")
        else:
            LOGGER.warn("Nothing to redo")

    def fit_view(self):
        """Fit scene to view."""
        rect = self.scene.itemsBoundingRect()
        if not rect.isEmpty():
            self.view.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
            self.view.scale(0.9, 0.9)
            LOGGER.info("View fitted")

    def clear_scene(self):
        """Clear all nodes and wires with proper resource cleanup."""
        reply = QMessageBox.question(
            self,
            "Clear Scene",
            "Delete all nodes and wires?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            # FIX: Clean up render workers before clearing nodes
            for node_id in list(self.nodes.keys()):
                self._cleanup_preview_worker(node_id)

            self.nodes.clear()
            self.scene.clear()
            # FIX: Clear preview display
            self.preview_lbl.clear()
            self.preview_lbl.setPixmap(QPixmap())  # Release pixmap memory
            self.preview_lbl.setText("No Selection")
            self.undo_manager = UndoRedoManager()
            self.compile_graph()
            LOGGER.info("Scene cleared")

    # --- GRAPH LOGIC ---

    def add_node_center(self, type_str, cls_name):
        self.is_ai_generated_code = False  # Reset flag - manual node added
        USAGE_TRACKER.record(cls_name, type_str)
        if self.code_view.toPlainText().strip() == "":
            self.compile_graph()
        center = self.view.mapToScene(self.view.rect().center())
        
        # Handle extension nodes by treating them as custom mobjects
        actual_type = "Mobject" if type_str == "ExtensionNode" else type_str
        
        self.add_node(actual_type, cls_name, pos=(center.x(), center.y()))

    # ── New Feature Helpers ───────────────────────────────────────────────────

    def _add_node_from_class_browser(self, cls_name: str, node_type_hint: str):
        """Add a node from the Manim class browser double-click."""
        type_str = "animation" if node_type_hint == "animation" else "mobject"
        # Better detection using issubclass
        try:
            cls = getattr(manim, cls_name, None)
            if cls and issubclass(cls, manim.Animation):
                type_str = "animation"
            elif cls:
                type_str = "mobject"
        except Exception:
            pass
        self.add_node_center(type_str, cls_name)
        LOGGER.info(f"Added {cls_name} from class browser")

    def _load_snippet_to_ai(self, code: str) -> None:
        """Load snippet code into the AI panel input and switch to AI tab."""
        self._load_code_to_ai(code, source="snippet")

    def _load_github_snippet_to_ai(self, code: str) -> None:
        """Load GitHub snippet code into the AI panel."""
        self._load_code_to_ai(code, source="github")

    def _load_code_to_ai(self, code: str, source: str = "snippet") -> None:
        """Common loader: registers VGroups with correct source, then opens AI tab."""
        # ── VGroup auto-registration ──────────────────────────────────────
        if hasattr(self, "panel_vgroup") and "VGroup" in code:
            n = self.panel_vgroup.register_snippet_vgroups(code, source=source)
            if n:
                LOGGER.info(f"{source}: auto-registered {n} VGroup(s) in VGroup tab")
        # ── Switch to AI tab ───────────────────────────────────────────────
        for i in range(self.tabs_top.count()):
            if "AI" in self.tabs_top.tabText(i):
                self.tabs_top.setCurrentIndex(i)
                break
        # Put code in AI panel output for review
        self.panel_ai.output.clear()
        self.panel_ai.output.setPlainText(code)
        self.panel_ai.last_code = code
        self.panel_ai._extract_nodes_from_code(code)
        if self.panel_ai.extracted_nodes:
            self.panel_ai.btn_merge.setEnabled(True)
            self.panel_ai.btn_reject.setEnabled(True)
            self.panel_ai.status_label.setText(
                f"Status: Snippet ready ({len(self.panel_ai.extracted_nodes)} nodes detected)"
            )
        LOGGER.info(f"{source} snippet loaded into AI panel")

    def _filter_nodes(self, text: str):
        """Highlight/dim canvas nodes matching search text."""
        text = text.lower()
        for node_item in self.nodes.values():
            match = (
                (not text)
                or text in node_item.node_data.name.lower()
                or text in node_item.node_data.cls_name.lower()
            )
            node_item.setOpacity(1.0 if match else 0.25)

    def _quick_export(self, fmt: str):
        """Handle quick export bar actions."""
        code = self.code_view.toPlainText()
        if not code.strip():
            QMessageBox.warning(
                self, "No Code", "Nothing to export. Build a scene first."
            )
            return

        if fmt == "py":
            path, _ = QFileDialog.getSaveFileName(
                self, "Export Python File", "GeneratedScene.py", "Python Files (*.py)"
            )
            if path:
                try:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(code)
                    LOGGER.info(f"Exported to {path}")
                    QMessageBox.information(self, "Exported", f"Saved to:\n{path}")
                except Exception as e:
                    QMessageBox.critical(self, "Export Error", str(e))

        elif fmt == "copy":
            QApplication.clipboard().setText(code)
            LOGGER.info("Code copied to clipboard")
            # Brief status feedback
            self.statusBar().showMessage("✅ Code copied to clipboard!", 2000)

        elif fmt == "mp4":
            # Switch to video render tab and start render
            for i in range(self.tabs_top.count()):
                if "Video" in self.tabs_top.tabText(i):
                    self.tabs_top.setCurrentIndex(i)
                    break
            self.panel_video.render_scene_btn.click()

    def auto_layout_nodes(self):
        """Auto-arrange nodes in a clean left-to-right flow layout."""
        if not self.nodes:
            return

        nodes_list = list(self.nodes.values())
        mobjects = [n for n in nodes_list if n.data.type == NodeType.MOBJECT]
        animations = [n for n in nodes_list if n.data.type == NodeType.ANIMATION]

        COL_W = 220
        ROW_H = 120
        START_X, START_Y = 50, 50

        for i, node in enumerate(mobjects):
            node.setPos(START_X, START_Y + i * ROW_H)
            node.data.pos_x = START_X
            node.data.pos_y = START_Y + i * ROW_H

        for i, node in enumerate(animations):
            node.setPos(START_X + COL_W, START_Y + i * ROW_H)
            node.data.pos_x = START_X + COL_W
            node.data.pos_y = START_Y + i * ROW_H

        # Update all wires
        for item in self.scene.items():
            if isinstance(item, WireItem):
                item.update_path()

        self.scene.notify_change()
        self.fit_view()
        LOGGER.info(f"Auto-layout applied to {len(nodes_list)} nodes")

    def add_node(
        self, type_str, cls_name, params=None, pos=(0, 0), nid=None, name=None
    ):
        # Normalize type string to handle "Mobject" / "MOBJECT" / "mobject" variants
        is_mobject = str(type_str).upper() == "MOBJECT"
        ntype = NodeType.MOBJECT if is_mobject else NodeType.ANIMATION

        # name defaults to cls_name when not provided (preserves all existing callers)
        display_name = name if name else cls_name
        data = NodeData(display_name, ntype, cls_name)
        if nid:
            data.id = nid
        if params:
            data.params = dict(params)
        data.pos_x, data.pos_y = pos

        item = NodeItem(data)
        self.scene.addItem(item)
        self.nodes[data.id] = item
        LOGGER.info(f"Created {cls_name} as '{display_name}' id={data.id[:8]}")
        self.compile_graph()  # Auto-Refresh
        return item

    def delete_selected(self):
        for item in self.scene.selectedItems():
            if isinstance(item, NodeItem):
                self.remove_node(item)
            elif isinstance(item, WireItem):
                self.remove_wire(item)
        self.compile_graph()  # Auto-Refresh

    def remove_node(self, node):
        # FIX: Clean up preview worker for this node
        self._cleanup_preview_worker(node.data.id)

        wires = node.in_socket.links + node.out_socket.links
        for w in wires:
            self.remove_wire(w)
        if node.data.id in self.nodes:
            del self.nodes[node.data.id]
        self.scene.removeItem(node)

        # FIX: Clear preview if this was the selected node
        if self.panel_props.current_node == node:
            self.preview_lbl.clear()
            self.preview_lbl.setPixmap(QPixmap())
            self.preview_lbl.setText("No Selection")

    def remove_wire(self, wire):
        if wire in wire.start_socket.links:
            wire.start_socket.links.remove(wire)
        if wire in wire.end_socket.links:
            wire.end_socket.links.remove(wire)
        self.scene.removeItem(wire)
        # Graph Changed (Handled by delete_selected, but good for singular removals)

    def on_selection(self):
        sel = self.scene.selectedItems()
        if len(sel) == 1 and isinstance(sel[0], NodeItem):
            self.panel_props.set_node(sel[0])
            self.show_preview(sel[0])
        else:
            self.panel_props.set_node(None)

    def on_node_changed(self):
        # 1. Update Project Modified State
        self.mark_modified()

        # 2. Re-compile the full code (for the code view tab)
        if not self.is_ai_generated_code:
            self.compile_graph()

        # 3. Trigger Preview Render (Only for Mobjects)
        node = self.panel_props.current_node
        if node and node.data.type == NodeType.MOBJECT:
            self.preview_lbl.setText("Queueing...")  # visual feedback
            self.queue_render(node)

    # --- COMPILER & RENDERER ---

    def compile_graph(self):
        if self.is_ai_generated_code:
            return self.code_view.toPlainText()

        code = "from manim import *\nimport numpy as np\n"
        if PYDUB_AVAILABLE:
            code += "from pydub import AudioSegment\n"

        code += "\nclass GeneratedScene(Scene):\n    def construct(self):\n"

        # 1. Instantiate Mobjects
        mobjects = [n for n in self.nodes.values() if n.data.type == NodeType.MOBJECT]
        m_vars = {}
        for m in mobjects:
            args = []

            # HANDLE POSITIONAL ARGS (arg0, arg1...)
            # Collect keys like arg0, arg1, sort them, and add as positional
            pos_args = {}
            named_args = {}

            for k, v in m.data.params.items():
                if k.startswith("_"):
                    continue
                if not m.data.is_param_enabled(k):
                    continue

                if k.startswith("arg") and k[3:].isdigit():
                    idx = int(k[3:])
                    v_clean = self._format_param_value(k, v, m.data)
                    pos_args[idx] = v_clean
                else:
                    named_args[k] = v

            # Add positional args in order
            for i in sorted(pos_args.keys()):
                args.append(pos_args[i])

            # Add named args (filter out None values which indicate inspect._empty)
            for k, v in named_args.items():
                v_clean = self._format_param_value(k, v, m.data)
                if v_clean is not None:  # Skip inspect._empty and None values
                    args.append(f"{k}={v_clean}")

            var = f"m_{m.data.id[:6]}"
            m_vars[m.data.id] = var
            code += f"        {var} = {m.data.cls_name}({', '.join(args)})\n"
            code += f"        self.add({var})\n"

        # 1b. VGroup code generation
        if hasattr(self, "panel_vgroup"):
            for gname, ids in self.panel_vgroup.get_groups().items():
                member_vars = [m_vars[nid] for nid in ids if nid in m_vars]
                if member_vars:
                    code += f"        {gname} = VGroup({', '.join(member_vars)})\n"

        # 2. Group animations
        animations = [
            n for n in self.nodes.values() if n.data.type == NodeType.ANIMATION
        ]
        played = set()

        # Voiceover precision tracking: (offset_ms: int, AudioSegment) pairs
        # built up across all animation batches, then merged into one file.
        _vo_entries: list[tuple[int, object]] = []  # (offset_ms, pydub.AudioSegment)
        _timeline_ms: int = 0  # running cursor through the scene timeline

        while animations:
            ready_anims = []
            for anim in animations:
                targets = []
                for link in anim.in_socket.links:
                    src_node = link.start_socket.parentItem()
                    if src_node.data.id in m_vars:
                        targets.append(m_vars[src_node.data.id])

                if targets:
                    ready_anims.append((anim, targets))
                elif not anim.in_socket.links:
                    pass

            if not ready_anims:
                break

            play_lines = []
            # Per-batch maximum duration in ms — drives timeline advance
            _batch_max_ms: int = 0

            # Process this batch of animations
            for anim, targets in ready_anims:
                anim_args = targets.copy()

                # Validate missing mobject param
                if "mobject" in anim.data.params or "mobjects" in anim.data.params:
                    val = anim.data.params.get("mobject") or anim.data.params.get(
                        "mobjects"
                    )
                    if not val or (isinstance(val, str) and len(val) != 36):
                        LOGGER.warn(
                            f"Animation '{anim.data.name}' has a 'mobject' parameter but no target selected!"
                        )

                # ── Voiceover precision sync ───────────────────────────────
                # We do NOT mutate anim.data.params here.
                # Instead we track an override for run_time separately.
                _rt_override_s: float | None = None

                if anim.data.audio_asset_id:
                    _vo_path = ASSETS.get_asset_path(anim.data.audio_asset_id)

                    # ═════════════════════════════════════════════════════
                    # CRITICAL VALIDATION: Check file actually exists before
                    # attempting to load it. Prevents silent render failure.
                    # ═════════════════════════════════════════════════════
                    if _vo_path is None:
                        LOGGER.error(
                            f"Voiceover for '{anim.data.name}': asset file missing or invalid. "
                            f"Render will fail unless fixed."
                        )
                    elif not os.path.exists(_vo_path):
                        LOGGER.error(
                            f"Voiceover for '{anim.data.name}': file not found at {_vo_path}. "
                            f"Skipping this voiceover."
                        )
                    elif not PYDUB_AVAILABLE:
                        # pydub not installed — fall back to add_sound without merge
                        _clean = _vo_path.replace("\\", "/")
                        code += f"        # Voiceover for {anim.data.name} (pydub unavailable — no precision sync)\n"
                        code += f"        self.add_sound(r'{_clean}')\n"
                        LOGGER.warn(
                            f"Voiceover for '{anim.data.name}': pydub unavailable, using fallback add_sound()"
                        )
                    else:
                        # ═════════════════════════════════════════════════════
                        # PYDUB AVAILABLE + FILE EXISTS: Proceed with precision
                        # ═════════════════════════════════════════════════════
                        try:
                            from pydub import AudioSegment as _AS

                            _seg = _AS.from_file(_vo_path)
                            _duration_ms = len(_seg)  # pydub uses ms

                            # Validate: audio must have positive duration
                            if _duration_ms <= 0:
                                LOGGER.error(
                                    f"Voiceover for '{anim.data.name}' has zero duration — skipping."
                                )
                            else:
                                # Validate: store if this node already has a duration
                                # recorded and detect drift (> 50ms = warning)
                                recorded_s = anim.data.voiceover_duration
                                if recorded_s > 0:
                                    drift_ms = abs(
                                        _duration_ms - int(recorded_s * 1000)
                                    )
                                    if drift_ms > 50:
                                        LOGGER.warn(
                                            f"Voiceover drift detected for '{anim.data.name}': "
                                            f"recorded={recorded_s:.3f}s, "
                                            f"actual={_duration_ms / 1000:.3f}s, "
                                            f"drift={drift_ms}ms"
                                        )

                                # Update stored duration to the exact measured value
                                anim.data.voiceover_duration = round(
                                    _duration_ms / 1000.0, 3
                                )

                                # Record for merged-track builder
                                _vo_entries.append((_timeline_ms, _seg))
                                LOGGER.info(
                                    f"Voiceover '{anim.data.name}': "
                                    f"offset={_timeline_ms}ms duration={_duration_ms}ms file={_vo_path}"
                                )

                                # run_time override: match animation duration to audio exactly
                                _rt_override_s = round(_duration_ms / 1000.0, 3)
                                _batch_max_ms = max(_batch_max_ms, _duration_ms)

                        except Exception as _e:
                            LOGGER.error(
                                f"pydub failed loading voiceover for '{anim.data.name}' from {_vo_path}: {_e}"
                            )

                # ── Format parameters ─────────────────────────────────────
                _used_run_time = False
                for k, v in anim.data.params.items():
                    if k.startswith("_") or not anim.data.is_param_enabled(k):
                        continue
                    if k == "run_time":
                        _used_run_time = True
                        if _rt_override_s is not None:
                            # Use the pydub-computed exact duration
                            anim_args.append(f"run_time={_rt_override_s}")
                        elif isinstance(v, str) and "duration_seconds" in v:
                            # Legacy stale expression from old compile: reset it
                            anim_args.append("run_time=1.0")
                        else:
                            v_clean = self._format_param_value(k, v, anim.data)
                            if v_clean is not None:
                                anim_args.append(f"{k}={v_clean}")
                        continue
                    v_clean = self._format_param_value(k, v, anim.data)
                    if v_clean is not None:
                        anim_args.append(f"{k}={v_clean}")

                # Inject run_time from override even if not already in params
                if _rt_override_s is not None and not _used_run_time:
                    anim_args.append(f"run_time={_rt_override_s}")

                # Track batch max for non-voiceover animations too
                if _rt_override_s is None:
                    try:
                        _rt_raw = anim.data.params.get("run_time", "1.0")
                        if not (
                            isinstance(_rt_raw, str) and "duration_seconds" in _rt_raw
                        ):
                            _batch_max_ms = max(
                                _batch_max_ms, int(float(_rt_raw) * 1000)
                            )
                        else:
                            _batch_max_ms = max(_batch_max_ms, 1000)
                    except Exception:
                        _batch_max_ms = max(_batch_max_ms, 1000)

                play_lines.append(f"{anim.data.cls_name}({', '.join(anim_args)})")
                played.add(anim)

            if play_lines:
                code += f"        self.play({', '.join(play_lines)})\n"
                code += "        self.wait(0.5)\n"

            # Advance timeline cursor: batch duration + 500ms wait
            _timeline_ms += _batch_max_ms + 500

            animations = [a for a in animations if a not in played]

        # ── Merged voiceover track injection ──────────────────────────────
        # If any animation nodes had voiceovers, build a single merged WAV
        # with millisecond-accurate silence padding and inject it once at
        # scene start. This prevents drift and clipping across all voiceovers.
        if _vo_entries and PYDUB_AVAILABLE:
            try:
                from pydub import AudioSegment as _AS

                # Total merged track length = last offset + its segment duration + 500ms tail
                _last_offset_ms, _last_seg = max(_vo_entries, key=lambda x: x[0])
                _total_ms = _last_offset_ms + len(_last_seg) + 500

                # Create silence canvas of the full required length
                _merged = _AS.silent(
                    duration=_total_ms, frame_rate=_last_seg.frame_rate
                )

                # Overlay each segment at its exact ms offset
                for _offset_ms, _seg in _vo_entries:
                    # Unify frame rates to prevent sample rate mismatch
                    if _seg.frame_rate != _merged.frame_rate:
                        _seg = _seg.set_frame_rate(_merged.frame_rate)
                    _merged = _merged.overlay(_seg, position=_offset_ms)

                # Export to a session-scoped temp file
                _merged_path = (
                    AppPaths.TEMP_DIR / f"merged_vo_{uuid.uuid4().hex[:8]}.wav"
                )
                _merged.export(str(_merged_path), format="wav")
                _merged_posix = _merged_path.as_posix()

                # Inject a single add_sound call at the very start of construct()
                _inject = f"        self.add_sound(r'{_merged_posix}')  # merged voiceover track\n"
                code = code.replace(
                    "class GeneratedScene(Scene):\n    def construct(self):\n",
                    f"class GeneratedScene(Scene):\n    def construct(self):\n{_inject}",
                    1,
                )
                LOGGER.info(
                    f"Merged voiceover track: {len(_vo_entries)} segment(s), "
                    f"total={_total_ms}ms → {_merged_path.name}"
                )
            except Exception as _e:
                LOGGER.error(f"Merged voiceover build failed: {_e}")

        self.code_view.setText(code)
        return code

    def _format_param_value(self, param_name, value, node_data):
        """Safely format parameter value with type enforcement and string escaping."""
        try:
            # CRITICAL FIX: Filter out inspect._empty and invalid default values
            # These should NEVER appear in generated code
            if value is None or str(value) == "<class 'inspect._empty'>":
                return None  # Signal to skip this parameter

            # Also check string representations of inject._empty
            if isinstance(value, str):
                value_stripped = value.strip()
                if value_stripped in (
                    "inspect._empty",
                    "<class 'inspect._empty'>",
                    "_empty",
                ):
                    return None  # Signal to skip this parameter

            # 0. MOBJECT REFERENCE (UUID Detection)
            # If value is a string looking like a UUID (36 chars) and exists in our node list
            if isinstance(value, str) and len(value) == 36 and value in self.nodes:
                # It is a reference to another node!
                # Return the variable name: m_123456
                return f"m_{value[:6]}"

            # 1. ASSET HANDLING
            if isinstance(value, str) and value in ASSETS.assets:
                abs_path = ASSETS.get_asset_path(value)
                if abs_path:
                    return f'r"{abs_path}"'

            # FIXED: Check if this is an already-formatted raw string (from LaTeX panel)
            if (
                isinstance(value, str)
                and value.startswith('r"""')
                and value.endswith('"""')
            ):
                # Return as-is - it's already a complete raw string literal
                return value

            # 2. Type-safe formatting
            if TypeSafeParser.is_color_param(param_name):
                color_val = str(value).strip()
                if color_val.upper() in dir(manim) and hasattr(
                    manim, color_val.upper()
                ):
                    return color_val.upper()
                return repr(TypeSafeParser.parse_color(color_val))

            elif TypeSafeParser.is_numeric_param(param_name):
                num_val = TypeSafeParser.parse_numeric(value)
                return repr(num_val)

            elif TypeSafeParser.is_point_param(param_name):
                point_val = TypeSafeParser.validate_point_safe(value)
                return f"np.array({repr(point_val.tolist())})"

            # 3. String escaping
            elif isinstance(value, str):
                # If marked to escape (don't add quotes), return raw value
                if node_data.should_escape_string(param_name):
                    return value.strip("'\"")
                return repr(value)

            else:
                return repr(value)

        except Exception as e:
            LOGGER.warn(f"Error formatting {param_name}={value}: {e}")
            return repr(value)

    def queue_render(self, node):
        # FIX: Check Setting first. If disabled, do nothing.
        if not SETTINGS.get("ENABLE_PREVIEW", True, type=bool):
            self.preview_lbl.setText("Preview Disabled\n(Enable in Settings)")
            return

        # Allow re-adding the same node ID to queue if parameters changed
        if node.data.id in self.render_queue:
            return

        self.render_queue.append(node.data.id)

        # Cleanup old files
        AppPaths.force_cleanup_old_files(age_seconds=300)

    def process_render_queue(self):
        """
        Generates a dedicated, isolated script for Mobject previewing.

        CRITICAL FIX: Respect render-in-progress flag to prevent concurrent renders.
        """
        try:
            # ══════════════════════════════════════════════════════════════════
            # RENDER FLOODING PROTECTION: Don't start another render if one is
            # already in progress. This prevents UI freeze and memory leaks.
            # ══════════════════════════════════════════════════════════════════
            if self._render_in_progress:
                LOGGER.debug("Skipping queue processing: render already in progress")
                return

            if not self.render_queue:
                return

            nid = self.render_queue.pop(0)
            if nid not in self.nodes:
                return
            node = self.nodes[nid]

            # 1. SKIP ANIMATIONS
            if node.data.type != NodeType.MOBJECT:
                return

            LOGGER.info(f"Rendering preview for {node.data.name}")

            # 2. Build Independent Script
            script = "from manim import *\nimport numpy as np\n"
            script += "config.background_color = ManimColor((0, 0, 0, 0))\n\n"
            script += "class PreviewScene(Scene):\n    def construct(self):\n"

            # 3. SPLIT POSITIONAL AND NAMED ARGS
            pos_args = {}  # {0: "val", 1: "val"}
            named_args = {}  # {"color": "RED"}

            for k, v in node.data.params.items():
                if k.startswith("_"):
                    continue

                # Check Enabled Status
                if not node.data.is_param_enabled(k):
                    continue

                # Format the value safely
                v_clean = self._format_param_value(k, v, node.data)

                # Check for arg0, arg1, arg2...
                if k.startswith("arg") and k[3:].isdigit():
                    try:
                        idx = int(k[3:])
                        pos_args[idx] = v_clean
                    except ValueError:
                        named_args[k] = v_clean
                else:
                    named_args[k] = v_clean

            # 4. Reconstruct Argument List
            final_args = []

            # Add Positional Args first (sorted by index)
            if pos_args:
                for i in sorted(pos_args.keys()):
                    final_args.append(str(pos_args[i]))

            # Add Named Args
            for k, v in named_args.items():
                final_args.append(f"{k}={v}")

            # 5. Instantiate with Exception Block for Safety
            script += "        # Target Mobject\n"
            script += f"        obj = {node.data.cls_name}({', '.join(final_args)})\n"
            script += "        obj.move_to(ORIGIN)\n"
            script += "        if obj.width > config.frame_width: obj.scale_to_fit_width(config.frame_width * 0.9)\n"
            script += "        self.add(obj)\n"

            # 6. Write File
            s_path = AppPaths.TEMP_DIR / f"preview_{nid}.py"
            with open(s_path, "w", encoding="utf-8") as f:
                f.write(script)

            # 7. Start Worker
            worker = RenderWorker(s_path, nid, AppPaths.TEMP_DIR, 15, "l")
            worker.success.connect(self.on_render_ok)

            # Show error in UI instead of crashing/quitting
            def handle_error(err):
                LOGGER.warn(f"Preview Render Error for {node.data.name}: {err}")
                if self.panel_props.current_node == node:
                    self.preview_lbl.setText(f"❌ Render Failed\n{str(err)[:30]}...")
                    self.preview_lbl.setStyleSheet("color: red;")

            worker.error.connect(handle_error)
            worker.finished.connect(lambda: self._cleanup_preview_worker(nid))
            worker.start()

            setattr(self, f"rw_{nid}", worker)

        except Exception as e:
            LOGGER.error(f"Queue Processing Error: {e}")
            traceback.print_exc()

    def on_render_ok(self, nid, path):
        """Called when RenderWorker successfully creates a PNG."""
        try:
            if nid not in self.nodes:
                return

            node = self.nodes[nid]
            node.data.preview_path = path

            # Force update of visual item (the green dot)
            node.update()

            # If this node is currently selected, show the image immediately
            sel = self.scene.selectedItems()
            if sel and isinstance(sel[0], NodeItem) and sel[0] == node:
                self.show_preview(node)
        finally:
            # Clean up worker reference regardless of success/failure
            self._cleanup_preview_worker(nid)

    def _cleanup_preview_worker(self, node_id):
        """Cleanup render worker thread for a specific node."""
        try:
            worker_attr = f"rw_{node_id}"
            if hasattr(self, worker_attr):
                worker = getattr(self, worker_attr)
                # Wait for thread to finish if it's still running
                if hasattr(worker, "isRunning") and worker.isRunning():
                    worker.quit()
                    worker.wait(3000)  # Wait max 3 seconds
                # Delete the worker
                delattr(self, worker_attr)
        except Exception as e:
            LOGGER.warn(f"Error cleaning up preview worker: {e}")

    def show_preview(self, node):
        """Display preview for selected node with safe resource management."""
        self.preview_lbl.clear()

        # 1. Check data
        if not node:
            self.preview_lbl.setPixmap(QPixmap())  # Release any cached pixmap
            self.preview_lbl.setText("No Selection")
            return

        if not node.data.preview_path:
            # FIX: Check toggle state
            if not SETTINGS.get("ENABLE_PREVIEW", True, type=bool):
                self.preview_lbl.setText("Preview Disabled\n(Enable in Settings)")
            else:
                self.preview_lbl.setText("Waiting for Render...")
                # Force a render if enabled but missing
                self.queue_render(node)
            return

        path = node.data.preview_path

        # 2. Check file
        if not os.path.exists(path):
            self.preview_lbl.setText("File Missing\nRe-queueing...")
            self.queue_render(node)
            return

        # 3. Load Image with errors safely handled
        try:
            pix = QPixmap(path)
            if pix.isNull():
                self.preview_lbl.setText(
                    "❌ Invalid Image\n(Corrupted or unsupported format)"
                )
                return

            # Scale to fit available space
            available_size = self.preview_lbl.size()
            if available_size.width() <= 1 or available_size.height() <= 1:
                # Size not yet initialized, use default
                available_size = QSize(400, 400)

            scaled = pix.scaledToWidth(
                available_size.width() - 20, Qt.TransformationMode.SmoothTransformation
            )
            self.preview_lbl.setPixmap(scaled)
        except Exception as e:
            self.preview_lbl.setText(f"❌ Error loading preview:\n{type(e).__name__}")
            LOGGER.warn(f"Preview load error: {e}")

    # --- VIDEO RENDERING ---

    def render_to_video(self, config):
        """Render full scene to video with specified config."""
        try:
            # Validate we have a compilable scene
            if not self.code_view.toPlainText().strip():
                QMessageBox.warning(
                    self, "Error", "No scene code to render. Create some nodes first."
                )
                return

            output_dir = Path(config["output_path"])
            if not output_dir.exists():
                QMessageBox.warning(
                    self, "Error", f"Output directory does not exist: {output_dir}"
                )
                return

            LOGGER.info("Building video render script...")

            # Generate full scene code
            scene_code = self.code_view.toPlainText()

            # Detect scene class name from code
            scene_class = detect_scene_class(scene_code)

            # If no Scene subclass found, create a basic one
            if "class " not in scene_code or "Scene" not in scene_code:
                scene_code = "from manim import *\n\nclass MyScene(Scene):\n    def construct(self):\n        pass\n"
                scene_class = "MyScene"

            # Write scene to temporary file
            script_path = output_dir / "video_render_scene.py"
            with open(script_path, "w") as f:
                f.write(scene_code)

            LOGGER.info(f"Scene script written to {script_path}")

            # Start video render worker
            fps = config["fps"]
            resolution = config["resolution"]
            quality = config["quality"]

            worker = VideoRenderWorker(
                script_path, output_dir, fps, resolution, quality, scene_class
            )

            # Connect signals
            worker.progress.connect(lambda msg: LOGGER.info(msg))
            worker.success.connect(self.on_video_render_success)
            worker.error.connect(self.on_video_render_error)

            # Store reference and start
            self.video_render_worker = worker
            self.panel_video.start_rendering(worker)
            worker.start()

            LOGGER.info(
                f"Video render started: {fps}fps, {resolution}, quality {quality}"
            )

        except Exception as e:
            LOGGER.error(f"Video render setup failed: {e}")
            QMessageBox.critical(self, "Render Error", f"Failed to start render:\n{e}")

    def on_video_render_success(self, video_path):
        """Called when video render completes successfully."""
        LOGGER.info(f"✓ Video rendered successfully: {video_path}")
        self.panel_video.on_render_success(video_path)

        # --- FIX: Auto-play in the new panel ---
        if os.path.exists(video_path):
            self.panel_output.load_video(video_path, autoplay=True)

        # Optionally show file dialog to open the video
        reply = QMessageBox.information(
            self,
            "Render Complete",
            f"Video saved to:\n{video_path}\n\nOpen video file?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            try:
                import subprocess
                import sys

                if sys.platform == "win32":
                    os.startfile(str(Path(video_path).parent))
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", str(Path(video_path).parent)])
                else:
                    subprocess.Popen(["xdg-open", str(Path(video_path).parent)])
            except Exception as e:
                LOGGER.warn(f"Could not open file location: {e}")

    def on_video_render_error(self, error_msg):
        """Called when video render encounters an error."""
        LOGGER.error(f"Video render failed: {error_msg}")

    # --- AI MERGE ---

    def merge_ai_code(self, code):
        """Merge AI-generated code into the scene using AINodeIntegrator.

        Completely replaces ALL previous nodes with AI-generated nodes.
        The AI-generated code is used DIRECTLY without regeneration.
        """
        LOGGER.ai("Merging AI-generated code...")

        try:
            # DELETE ALL EXISTING NODES
            LOGGER.ai(f"Removing {len(self.nodes)} previous node(s)...")
            nodes_to_remove = list(self.nodes.values())
            for node_item in nodes_to_remove:
                self.scene.removeItem(node_item)
            self.nodes.clear()

            # Also remove all WireItems
            for item in list(self.scene.items()):
                if isinstance(item, WireItem):
                    self.scene.removeItem(item)

            # Clear render queue
            self.render_queue.clear()

            # Use AINodeIntegrator for robust parsing and node creation
            result = AINodeIntegrator.merge_ai_code_to_scene(code, self)

            if result["success"]:
                LOGGER.ai(f"✅ Successfully added {result['nodes_added']} node(s)")

                # Set the code view to the AI-generated code directly
                self.code_view.setText(code)
                self.is_ai_generated_code = True
                LOGGER.ai("Code view updated with AI-generated code")
                # ── VGroup auto-registration from AI-generated code ────────
                if hasattr(self, "panel_vgroup") and "VGroup" in code:
                    n = self.panel_vgroup.register_snippet_vgroups(code, source="ai")
                    if n:
                        LOGGER.ai(f"AI merge: auto-registered {n} VGroup(s)")

                # Update properties panel
                if result["nodes"]:
                    first_node = result["nodes"][0]
                    self.panel_props.set_node(first_node)
                    LOGGER.ai(
                        f"Inspector updated - showing {len(first_node.data.params)} parameters"
                    )

                # Trigger render preview for mobject nodes
                for node in result["nodes"]:
                    if node.data.type == NodeType.MOBJECT:
                        self.queue_render(node)

                if result["errors"]:
                    LOGGER.warn(
                        f"Merge completed with {len(result['errors'])} warning(s):"
                    )
                    for err in result["errors"]:
                        LOGGER.warn(f"  - {err}")
            else:
                error_msg = "\n".join(result["errors"][:5])
                LOGGER.error(f"Failed to merge AI code:\n{error_msg}")
                QMessageBox.critical(
                    self,
                    "AI Merge Failed",
                    f"Could not create nodes from AI code:\n\n{error_msg}\n\n"
                    "Check that Manim is installed and the code is valid.",
                )

        except Exception as e:
            LOGGER.error(f"AI merge error: {str(e)}")
            traceback.print_exc()
            QMessageBox.critical(
                self, "AI Merge Error", f"Unexpected error during merge:\n{str(e)}"
            )

    # --- PROJECT I/O ---

    def save_project(self):
        # Get default filename from project name textbox
        default_filename = self.project_name_edit.text().strip() or "Untitled Project"
        if not default_filename.endswith(PROJECT_EXT):
            default_filename += PROJECT_EXT

        # Use last project path directory if available, otherwise Documents
        last_dir = (
            str(Path(self.project_path).parent)
            if self.project_path
            else str(Path.home() / "Documents")
        )

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project",
            str(Path(last_dir) / default_filename),
            f"EfficientManim (*{PROJECT_EXT})",
        )
        if not path:
            return

        # Update project name from saved path
        self.project_name_edit.setText(Path(path).stem)

        meta = {
            "name": Path(path).stem,
            "created": str(datetime.now()),
            "version": APP_VERSION,
        }

        graph_data = {
            "nodes": [n.data.to_dict() for n in self.nodes.values()],
            "wires": [
                {
                    "start": w.start_socket.parentItem().data.id,
                    "end": w.end_socket.parentItem().data.id,
                }
                for w in self.scene.items()
                if isinstance(w, WireItem)
            ],
        }

        try:
            with tempfile.TemporaryDirectory() as td:
                t_path = Path(td)

                # Write JSONs
                with open(t_path / "metadata.json", "w") as f:
                    json.dump(meta, f, indent=2)
                with open(t_path / "graph.json", "w") as f:
                    json.dump(graph_data, f, indent=2)
                with open(t_path / "code.py", "w") as f:
                    f.write(self.code_view.toPlainText())

                # Handle Assets
                assets_dir = t_path / "assets"
                assets_dir.mkdir()

                asset_manifest = []
                for a in ASSETS.get_list():
                    # Calculate safe local filename
                    safe_suffix = Path(a.current_path).suffix
                    dst_name = f"{a.id}{safe_suffix}"

                    source_path = Path(a.current_path)

                    if source_path.exists():
                        shutil.copy2(source_path, assets_dir / dst_name)

                        # Save manifest entry
                        d = a.to_dict()
                        d["local"] = dst_name  # Important: Store local ref
                        asset_manifest.append(d)
                    else:
                        LOGGER.warn(f"Could not find asset to save: {source_path}")

                with open(t_path / "assets.json", "w") as f:
                    json.dump(asset_manifest, f, indent=2)

                # Zip it up
                shutil.make_archive(
                    str(Path(path).parent / Path(path).stem), "zip", t_path
                )

                # Rename .zip to .efp
                final_zip = Path(path).parent / f"{Path(path).stem}.zip"
                final_efp = Path(path).with_suffix(PROJECT_EXT)

                if final_efp.exists():
                    final_efp.unlink()
                shutil.move(str(final_zip), final_efp)

                self.project_path = str(final_efp)
                _add_to_recents(str(final_efp))

                # Update project name field
                if hasattr(self, "project_name_edit"):
                    self.project_name_edit.setText(final_efp.stem)

                self.reset_modified()
                LOGGER.info(f"Project saved to {final_efp}")

        except Exception as e:
            LOGGER.error(f"Save Failed: {e}")
            traceback.print_exc()

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open", "", f"EfficientManim (*{PROJECT_EXT})"
        )
        if not path:
            return
        self._do_open_project(path)

    def _do_open_project(self, path: str):
        """Core project loading logic."""
        try:
            self.nodes.clear()
            self.scene.clear()
            ASSETS.clear()

            # Create a clean extraction folder for this project session
            dest = AppPaths.TEMP_DIR / "Project_Assets"
            if dest.exists():
                shutil.rmtree(dest)
            dest.mkdir(parents=True)

            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(dest)

            # 1. Load Assets and Re-link Paths
            assets_json = dest / "assets.json"
            if assets_json.exists():
                with open(assets_json) as f:
                    asset_list = json.load(f)
                    for ad in asset_list:
                        # Construct the path to the extracted file
                        local_name = ad.get("local", "")
                        extracted_path = dest / "assets" / local_name

                        # Re-create asset object
                        a = Asset(
                            ad["name"], str(extracted_path.as_posix()), ad["kind"]
                        )
                        a.id = ad["id"]
                        a.original_path = ad["original"]

                        # CRITICAL: Set current_path to the extracted temp file
                        if extracted_path.exists():
                            a.current_path = str(extracted_path.resolve().as_posix())
                        else:
                            # Fallback if extraction failed (shouldn't happen)
                            LOGGER.warn(f"Missing asset file: {local_name}")
                            a.current_path = ad["original"]

                        ASSETS.assets[a.id] = a
                    ASSETS.assets_changed.emit()

            # 2. Load Graph
            graph_json = dest / "graph.json"
            if graph_json.exists():
                with open(graph_json) as f:
                    g = json.load(f)
                    node_map = {}
                    for nd in g["nodes"]:
                        # Fix for Mobject casing
                        type_str = (
                            nd["type"].capitalize()
                            if nd["type"].upper() == "MOBJECT"
                            else "Animation"
                        )

                        node = self.add_node(
                            type_str,
                            nd["cls_name"],
                            nd["params"],
                            nd["pos"],
                            nd["id"],
                            name=nd.get("name", nd["cls_name"]),
                        )

                        # Restore metadata
                        if "param_metadata" in nd:
                            node.data.param_metadata = nd["param_metadata"]
                        if "is_ai_generated" in nd:
                            node.data.is_ai_generated = nd["is_ai_generated"]

                        node_map[nd["id"]] = node

                    for w in g["wires"]:
                        n1, n2 = node_map.get(w["start"]), node_map.get(w["end"])
                        if n1 and n2:
                            self.scene.try_connect(n1.out_socket, n2.in_socket)

            # 3. Load Saved Code (if any)
            code_py = dest / "code.py"
            if code_py.exists():
                with open(code_py, "r") as f:
                    self.code_view.setText(f.read())

            self.compile_graph()
            self.project_path = path
            _add_to_recents(path)
            if hasattr(self, "project_name_edit"):
                self.project_name_edit.setText(Path(path).stem)
            self.reset_modified()
            LOGGER.info("Project Loaded Successfully.")

        except Exception as e:
            LOGGER.error(f"Open Failed: {e}")
            traceback.print_exc()

    def open_settings(self):
        SettingsDialog(self).exec()

    def append_log(self, level, msg):
        c = "black"
        if level == "ERROR":
            c = "red"
        elif level == "WARN":
            c = "orange"
        elif level == "AI":
            c = "blue"
        elif level == "MANIM":
            c = "purple"
        ts = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"<span style='color:{c}'><b>[{ts}] {level}:</b> {msg}</span>")


# ==============================================================================
# 10. ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Set application icon at QApplication level BEFORE any window is created
    _icon_path = Path(__file__).parent / "icon" / "icon.ico"
    if _icon_path.exists():
        app.setWindowIcon(QIcon(str(_icon_path.absolute())))

    # ── Apply Global Light-Mode Stylesheet ────────────────────────────────
    app.setStyleSheet(THEME_MANAGER.get_stylesheet())

    _original_excepthook = sys.excepthook

    def exception_hook(exctype, value, traceback_obj):
        LOGGER.error(f"CRITICAL: {value}")
        traceback.print_tb(traceback_obj)
        _original_excepthook(exctype, value, traceback_obj)

    sys.excepthook = exception_hook

    win = EfficientManimWindow()
    win.show()
    sys.exit(app.exec())
