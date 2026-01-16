# Register this as a proper app using the classic ctypes.windll workaround
import ctypes
import sys

if sys.platform == "win32":
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("com.programmersd.efficientmanim")

import struct

# Audio handling
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("WARNING: pydub not found. Audio duration features will be disabled. 'pip install pydub'")

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
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            ),
        ]
        generate_content_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
        )

        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            if chunk.text:
                on_chunk(chunk.text)
    except Exception as e:
        on_chunk(f"❌ Gemini Error: {str(e)}")

# ==============================================================================
# EFFICIENT MANIM ULTIMATE - MONOLITHIC SOURCE
# ==============================================================================

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

# PySide6 Imports
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTabWidget, QTextEdit, QPlainTextEdit, QTreeWidget, QTreeWidgetItem,
    QGraphicsView, QGraphicsScene, QGraphicsItem, QGraphicsPathItem,
    QLabel, QLineEdit, QSpinBox, QDoubleSpinBox,
    QCheckBox, QPushButton, QDialog, QFormLayout, QComboBox, QColorDialog,
    QScrollArea, QFrame, QMessageBox,
    QFileDialog, QListWidget, QListWidgetItem,
    QStyle
)
from PySide6.QtCore import (
    Qt, Signal, QObject, QThread, QPointF, QRectF, QTimer,
    QSettings, QSize, QMimeData, QStandardPaths
)
from PySide6.QtGui import (
    QAction, QColor, QPen, QBrush, QFont, QPainter, QPixmap, QKeySequence, QFontDatabase, QIcon, QPainterPath,
    QDrag, QTextCursor
)

# Manim Import (Safe)
try:
    import manim
    from manim import *
    from manim.utils.color import ManimColor, ParsableManimColor
    MANIM_AVAILABLE = True
except ImportError:
    MANIM_AVAILABLE = False
    print("CRITICAL: Manim library not found. Rendering will be disabled.")

# ==============================================================================
# 1. CORE CONFIGURATION & UTILS
# ==============================================================================

APP_NAME = "EfficientManim"
APP_VERSION = "0.1.1"
PROJECT_EXT = ".efp" # EfficientManim Project (Zip)

class AppPaths:
    """Centralized path management."""
    USER_DATA = Path(QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)) / "EfficientManim"
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
                    for item in AppPaths.TEMP_DIR.rglob('*'):
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
                    print(f"⚠️ Warning: Could not fully clear temp dir (locked files): {e}")
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
            for item in AppPaths.TEMP_DIR.rglob('*'):
                try:
                    if item.is_file():
                        mtime = item.stat().st_mtime
                        if current_time - mtime > age_seconds:
                            item.unlink(missing_ok=True)
                except (PermissionError, OSError):
                    pass
        except Exception as e:
            pass  # Silently ignore cleanup errors

class Theme:
    """Application Styling."""
    PRIMARY = "#3498db"
    SECONDARY = "#2c3e50"
    ACCENT = "#e67e22"
    BG_LIGHT = "#ecf0f1"
    BG_DARK = "#bdc3c7"
    TEXT = "#2c3e50"
    
    STYLESHEET = f"""
    QMainWindow {{ background-color: {BG_LIGHT}; }}
    QWidget {{ font-family: "Geist", "Segoe UI", sans-serif; font-size: 13px; color: {TEXT}; }}
    QSplitter::handle {{ background-color: {BG_DARK}; width: 2px; }}
    QTabWidget::pane {{ border: 1px solid {BG_DARK}; background: white; border-radius: 4px; }}
    QTabBar::tab {{ background: {BG_LIGHT}; border: 1px solid {BG_DARK}; padding: 6px 12px; margin-right: 2px; border-top-left-radius: 4px; border-top-right-radius: 4px; }}
    QTabBar::tab:selected {{ background: white; border-bottom: 2px solid {PRIMARY}; font-weight: bold; color: {PRIMARY}; }}
    QPushButton {{ background-color: white; border: 1px solid {BG_DARK}; padding: 6px 12px; border-radius: 4px; }}
    QPushButton:hover {{ background-color: #f7f9f9; border-color: {PRIMARY}; }}
    QPushButton:pressed {{ background-color: {BG_DARK}; }}
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ padding: 4px; border: 1px solid {BG_DARK}; border-radius: 4px; background: white; }}
    QTextEdit, QPlainTextEdit, QListWidget, QTreeWidget {{ border: 1px solid {BG_DARK}; border-radius: 4px; background: white; }}
    QLabel#PreviewLabel {{ border: 2px dashed #bdc3c7; background-color: #ecf0f1; color: #7f8c8d; }}
    """

# ==============================================================================
# 2. LOGGING SYSTEM
# ==============================================================================

class LogManager(QObject):
    log_signal = Signal(str, str) # Level, Message

    def __init__(self):
        super().__init__()

    def info(self, msg): self.log_signal.emit("INFO", str(msg))
    def warn(self, msg): self.log_signal.emit("WARN", str(msg))
    def error(self, msg): self.log_signal.emit("ERROR", str(msg))
    def manim(self, msg): self.log_signal.emit("MANIM", str(msg))
    def ai(self, msg): self.log_signal.emit("AI", str(msg))

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

    def get(self, key, default=None):
        return self._store.value(key, default)

    def set(self, key, value):
        self._store.setValue(key, value)
        if key == "GEMINI_API_KEY":
            self.apply_env()
        self.settings_changed.emit()

    def apply_env(self):
        key = self.get("GEMINI_API_KEY", "")
        if key:
            os.environ["GEMINI_API_KEY"] = key

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
        self.history = self.history[:self.current_index + 1]
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

class ThemeManager:
    """Manages application themes."""
    LIGHT_THEME = "light"
    DARK_THEME = "dark"
    
    def __init__(self):
        self.current_theme = SETTINGS.get("THEME", self.LIGHT_THEME)
    
    def get_stylesheet(self):
        if self.current_theme == self.DARK_THEME:
            return self.get_dark_stylesheet()
        return Theme.STYLESHEET
    
    @staticmethod
    def get_dark_stylesheet():
        return """
        QMainWindow { background-color: #1e1e1e; }
        QWidget { font-family: "Geist", "Segoe UI", sans-serif; font-size: 13px; color: #e0e0e0; }
        QSplitter::handle { background-color: #333333; width: 2px; }
        QTabWidget::pane { border: 1px solid #333333; background: #2d2d2d; border-radius: 4px; }
        QTabBar::tab { background: #1e1e1e; border: 1px solid #333333; padding: 6px 12px; margin-right: 2px; }
        QTabBar::tab:selected { background: #2d2d2d; border-bottom: 2px solid #0d7377; color: #0d7377; font-weight: bold; }
        QPushButton { background-color: #2d2d2d; border: 1px solid #404040; color: #e0e0e0; padding: 6px 12px; border-radius: 4px; }
        QPushButton:hover { background-color: #3d3d3d; border-color: #0d7377; }
        QPushButton:pressed { background-color: #1e1e1e; }
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { padding: 4px; border: 1px solid #404040; border-radius: 4px; background: #2d2d2d; color: #e0e0e0; }
        QTextEdit, QPlainTextEdit, QListWidget, QTreeWidget { border: 1px solid #404040; border-radius: 4px; background: #2d2d2d; color: #e0e0e0; }
        QLabel#PreviewLabel { border: 2px dashed #404040; background-color: #1e1e1e; color: #606060; }
        QScrollBar:vertical { background: #1e1e1e; width: 12px; }
        QScrollBar::handle:vertical { background: #404040; border-radius: 6px; }
        QScrollBar::handle:vertical:hover { background: #505050; }
        """
    
    def set_theme(self, theme):
        self.current_theme = theme
        SETTINGS.set("THEME", theme)

THEME_MANAGER = ThemeManager()

# ==============================================================================
# 3.8 KEYBOARD SHORTCUTS REGISTRY
# ==============================================================================

class KeyboardShortcuts:
    """Centralized keyboard shortcuts."""
    SHORTCUTS = {
        "Delete": (QKeySequence.Delete, "Delete selected nodes/wires"),
        "Undo": (QKeySequence.Undo, "Undo last action"),
        "Redo": (QKeySequence.Redo, "Redo last action"),
        "Save": (QKeySequence.Save, "Save project"),
        "Open": (QKeySequence.Open, "Open project"),
        "Fit View": (QKeySequence(Qt.CTRL | Qt.Key_0), "Fit scene to view"),
        "Clear": (QKeySequence(Qt.CTRL| Qt.Key_Delete), "Clear all nodes"),
    }
    
    @classmethod
    def get_shortcut(cls, action_name):
        return cls.SHORTCUTS.get(action_name, (None, ""))[0]
    
    @classmethod
    def describe_shortcuts(cls):
        """Return formatted help text."""
        lines = ["=== Keyboard Shortcuts ==="]
        for name, (seq, desc) in cls.SHORTCUTS.items():
            lines.append(f"{str(seq):20} {desc}")
        return "\n".join(lines)

# ==============================================================================
# 3. SETTINGS MANAGER (continuation)
# ==============================================================================

# ==============================================================================
# 4. WORKER THREADS
# ==============================================================================

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

# ==============================================================================
# 4b. TTS WORKER (NEW)
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
                except: pass
            elif param.startswith("audio/L"):
                try:
                    bits_per_sample = int(param.split("L", 1)[1])
                except: pass
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
        b"RIFF", chunk_size, b"WAVE", b"fmt ", 16, 1, num_channels,
        sample_rate, byte_rate, block_align, bits_per_sample,
        b"data", data_size
    )
    return header + audio_data

class TTSWorker(QThread):
    finished_signal = Signal(str) # Path to saved file
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
            contents = [types.Content(role="user", parts=[types.Part.from_text(text=self.text)])]
            
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
            mime_type = "audio/wav" # Default

            for chunk in client.models.generate_content_stream(
                model=self.model_name,
                contents=contents,
                config=generate_content_config,
            ):
                if chunk.candidates and chunk.candidates[0].content.parts:
                    part = chunk.candidates[0].content.parts[0]
                    if part.inline_data:
                        full_audio += part.inline_data.data
                        if part.inline_data.mime_type:
                            mime_type = part.inline_data.mime_type

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
    """Manim CLI Runner."""
    success = Signal(str, str) # node_id, path
    error = Signal(str)
    
    def __init__(self, script_path, node_id, output_dir, fps, quality):
        super().__init__()
        self.script_path = script_path
        self.node_id = node_id
        self.output_dir = output_dir
        self.fps = fps
        self.quality = quality # l, m, h, k

    def run(self):
        try:
            # Construct Command
            # manim -s -ql --format=png --disable_caching script.py Scene
            
            flags = ["-s", "--format=png", "--disable_caching"]
            flags.append(f"-q{self.quality}")
            flags.append(f"--fps={self.fps}")
            
            cmd = ["manim"] + flags + [str(self.script_path), "PreviewScene"]
            
            # Windows: hide console window
            startupinfo = None
            if platform.system() == "Windows":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.output_dir),
                startupinfo=startupinfo
            )
            
            if process.returncode != 0:
                err = process.stderr if process.stderr else process.stdout
                self.error.emit(f"Manim Error:\n{err}")
                return

            # Locate Output
            media_dir = self.output_dir / "media"
            pngs = list(media_dir.rglob("*.png"))
            
            if pngs:
                # Get latest
                latest = max(pngs, key=os.path.getmtime)
                self.success.emit(self.node_id, str(latest))
            else:
                self.error.emit("Render finished but no PNG output found.")

        except Exception as e:
            self.error.emit(f"Worker Exception: {e}")

class VideoRenderWorker(QThread):
    """Renders full scenes to MP4/WebM video using Manim."""
    progress = Signal(str)  # Status message
    success = Signal(str)   # Output video path
    error = Signal(str)     # Error message
    
    def __init__(self, script_path, output_dir, fps, resolution, quality):
        super().__init__()
        self.script_path = script_path
        self.output_dir = output_dir
        self.fps = fps
        self.resolution = resolution  # (width, height)
        self.quality = quality  # l, m, h, k
        self.is_running = True

    def stop_render(self):
        """Request graceful stop and force kill process."""
        self.is_running = False
        # Force kill the external process if it's stuck
        if hasattr(self, 'process') and self.process:
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
            flags.insert(0, "-p")
            cmd = ["manim"] + flags + [str(self.script_path), "EfficientScene"]
            
            self.progress.emit(f"Starting render...")
            
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
                startupinfo=startupinfo
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
        # NEW: Voiceover support
        self.audio_asset_id = None 
        # AI metadata
        self.is_ai_generated = False
        self.ai_source = None  # Original Manim class name
        self.ai_code_snippet = None  # Original code from AI

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
            "audio_asset_id": self.audio_asset_id, # NEW
            "is_ai_generated": self.is_ai_generated,
            "ai_source": self.ai_source,
            "ai_code_snippet": self.ai_code_snippet
        }

    @staticmethod
    def from_dict(d):
        n = NodeData(d["name"], NodeType[d["type"]], d["cls_name"])
        n.id = d["id"]
        n.params = d["params"]
        n.param_metadata = d.get("param_metadata", {})
        n.pos_x, n.pos_y = d["pos"]
        n.preview_path = d.get("preview_path")
        n.audio_asset_id = d.get("audio_asset_id") # NEW
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
        self.kind = kind # "image", "video", "audio"
        self.local_file = "" # Filename in .efp assets/ folder

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "original": self.original_path,
            # We don't save current_path, because on load we calculate a new one
            "kind": self.kind,
            "local": self.local_file
        }
    
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
        path.addEllipse(-self.radius, -self.radius, self.radius*2, self.radius*2)
        self.setPath(path)
        
        # Style
        color = QColor("#2ecc71") if not is_output else QColor("#e74c3c")
        self.setBrush(QBrush(color))
        self.setPen(QPen(Qt.black, 1))

    def get_scene_pos(self):
        return self.scenePos()

class WireItem(QGraphicsPathItem):
    """Connection between sockets."""
    def __init__(self, start_socket, end_socket):
        super().__init__()
        self.start_socket = start_socket
        self.end_socket = end_socket
        self.setZValue(-1)
        
        pen = QPen(QColor("#7f8c8d"), 2.5)
        pen.setCapStyle(Qt.RoundCap)
        self.setPen(pen)
        self.update_path()

    def update_path(self):
        if not self.start_socket or not self.end_socket: return
        
        p1 = self.start_socket.get_scene_pos()
        p2 = self.end_socket.get_scene_pos()
        
        path = QPainterPath()
        path.moveTo(p1)
        
        dx = p2.x() - p1.x()
        
        ctrl1 = QPointF(p1.x() + dx * 0.5, p1.y())
        ctrl2 = QPointF(p2.x() - dx * 0.5, p2.y())
        
        path.cubicTo(ctrl1, ctrl2, p2)
        self.setPath(path)

class NodeItem(QGraphicsItem):
    """Visual representation of NodeData."""
    def __init__(self, data: NodeData):
        super().__init__()
        self.data = data
        self.width = 180
        self.height = 90
        
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemSendsGeometryChanges)
        self.setPos(data.pos_x, data.pos_y)
        
        # Sockets
        self.in_socket = SocketItem(self, False)
        self.in_socket.setPos(0, self.height/2)
        
        self.out_socket = SocketItem(self, True)
        self.out_socket.setPos(self.width, self.height/2)
        
    def boundingRect(self):
        return QRectF(0, 0, self.width, self.height)

    def paint(self, painter, option, widget):
        # Shadow
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0,0,0,30))
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
        header_color = QColor("#3498db") if self.data.type == NodeType.MOBJECT else QColor("#9b59b6")
        
        path = QPainterPath()
        path.addRoundedRect(0, 0, self.width, header_h, 8, 8)
        # Clip
        painter.setClipPath(path)
        painter.fillPath(path, header_color)
        painter.setClipping(False)
        
        # Text
        painter.setPen(Qt.white)
        painter.setFont(QFont("Geist", 10, QFont.Bold))
        painter.drawText(QRectF(8, 0, self.width-16, header_h), Qt.AlignLeft | Qt.AlignVCenter, self.data.name)
        
        # Class Name
        painter.setPen(QColor("#7f8c8d"))
        painter.setFont(QFont("Geist", 9))
        painter.drawText(QRectF(8, 35, self.width-16, 20), Qt.AlignLeft, self.data.cls_name)
        
        # Indicator
        if self.data.preview_path:
            painter.setBrush(QColor("#2ecc71"))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(self.width-16, self.height-16, 8, 8)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            self.data.pos_x = value.x()
            self.data.pos_y = value.y()
            if hasattr(self, "in_socket") and hasattr(self, "out_socket"):
                for w in self.in_socket.links + self.out_socket.links:
                    w.update_path()
            if self.scene():
                self.scene().notify_change()
        return super().itemChange(change, value)
    
class GraphScene(QGraphicsScene):
    """Manages node connections and logic enforcement."""
    selection_changed_signal = Signal()
    graph_changed_signal = Signal() # Structure changed
    
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
            self.drag_wire.setPen(QPen(Qt.black, 2, Qt.DashLine))
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
        if s1.is_output == s2.is_output: return
        
        if s1.is_output: out_sock, in_sock = s1, s2
        else: out_sock, in_sock = s2, s1
            
        node_src = out_sock.parentItem()
        node_dst = in_sock.parentItem()
        
        # 1. Mobject -> Mobject (INVALID)
        if node_src.data.type == NodeType.MOBJECT and node_dst.data.type == NodeType.MOBJECT:
            self.show_warning("Invalid Connection", "Directly connecting Mobjects is not allowed.\nPlease insert an Animation node in between.")
            return

        # 2. Animation -> Animation (INVALID for now)
        if node_src.data.type == NodeType.ANIMATION and node_dst.data.type == NodeType.ANIMATION:
             self.show_warning("Invalid Connection", "Chaining animations directly is not supported.\nTarget a Mobject instead.")
             return
             
        # Create Wire
        wire = WireItem(out_sock, in_sock)
        self.addItem(wire)
        out_sock.links.append(wire)
        in_sock.links.append(wire)
        
        self.notify_change()

    def show_warning(self, title, msg):
        views = self.views()
        if views: QMessageBox.warning(views[0], title, msg)

# ==============================================================================
# 7. ASSET MANAGEMENT
# ==============================================================================

class AssetManager(QObject):
    assets_changed = Signal()
    
    def __init__(self):
        super().__init__()
        self.assets = {} # id -> Asset

    def add_asset(self, path):
        path_obj = Path(path)
        if not path_obj.exists(): return
        
        # Normalize path for Manim (Forward Slashes)
        clean_path = path_obj.resolve().as_posix()
        
        kind = "unknown"
        s = path_obj.suffix.lower()
        if s in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]: kind = "image"
        elif s in [".svg"]: kind = "image" # SVG is treated as image/vector
        elif s in [".mp4", ".mov", ".avi", ".webm"]: kind = "video"
        elif s in [".mp3", ".wav", ".ogg"]: kind = "audio"
        
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
        """Safely retrieve the absolute POSIX path for Manim."""
        if asset_id in self.assets:
            return self.assets[asset_id].current_path
        return None

ASSETS = AssetManager()

# ==============================================================================
# 8. PANELS & UI MODULES
# ==============================================================================

class TypeSafeParser:
    """Comprehensive type validation and safe parsing for Manim parameters."""
    
    # Parameter category mapping
    NUMERIC_KEYWORDS = {'radius', 'width', 'height', 'scale', 'factor', 'size', 'thickness', 
                        'stroke_width', 'font_size', 'length', 'rate', 'opacity', 'alpha',
                        'x', 'y', 'z', 'angle', 'degrees', 'radians'}
    
    COLOR_KEYWORDS = {'color', 'fill_color', 'stroke_color', 'background_color', 'fg_color', 'bg_color'}
    
    POINT_KEYWORDS = {'point', 'points', 'center', 'pos', 'position', 'start', 'end', 'direction'}
    
    @staticmethod
    def is_asset_param(param_name):
        """Check if parameter expects a file/asset."""
        n = param_name.lower()
        # Specific fix for ImageMobject
        if "filename" in n: return True
        
        # General file keywords
        if "file" in n or "image" in n or "sound" in n or "svg" in n:
             # Exclude false positives like "fill_opacity" or "profile"
             if "fill" in n or "profile" in n: return False
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
        """Safely parse value as number with validation."""
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
                return default
            # Fallback
            return float(value) if value else default
        except (ValueError, TypeError):
            LOGGER.warn(f"Invalid numeric value: {value}, using default {default}")
            return default
    
    @staticmethod
    def parse_color(value, default_hex="#FFFFFF"):
        """Safely parse value as ManimColor with validation, including Manim constants."""
        try:
            if value is None:
                return default_hex
            
            # Already ManimColor
            if isinstance(value, (ManimColor, ParsableManimColor)):
                return value.to_hex() if hasattr(value, 'to_hex') else str(value)
            
            # String handling
            if isinstance(value, str):
                value_clean = value.strip()
                
                # Try Manim color constants first (e.g., "BLUE", "RED", "CYAN")
                try:
                    if hasattr(manim, value_clean.upper()):
                        manim_color = getattr(manim, value_clean.upper())
                        if isinstance(manim_color, (ManimColor, ParsableManimColor)) or hasattr(manim_color, '_internal_value'):
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
                    r, g, b = int(r*255), int(g*255), int(b*255)
                return f"#{int(r):02x}{int(g):02x}{int(b):02x}"
            
            # QColor object
            if isinstance(value, QColor) and value.isValid():
                return value.name()
        
        except Exception as e:
            LOGGER.warn(f"Color parsing failed for {value}: {e}")
        
        return default_hex
    
    @staticmethod
    def to_manim_color(hex_color):
        """Convert hex color to ManimColor for rendering."""
        try:
            if isinstance(hex_color, (ManimColor, ParsableManimColor)):
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
    
class GraphView(QGraphicsView):
    """Custom View with Zoom (Ctrl+Wheel) and Pan (Middle Mouse) support."""
    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._is_panning = False

    def wheelEvent(self, event):
        """Handle Zoom with Ctrl + Scroll."""
        if event.modifiers() & Qt.ControlModifier:
            zoom_factor = 1.15
            if event.angleDelta().y() > 0:
                self.scale(zoom_factor, zoom_factor)
            else:
                self.scale(1 / zoom_factor, 1 / zoom_factor)
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event):
        """Handle Panning with Middle Mouse Button."""
        if event.button() == Qt.MiddleButton:
            self._is_panning = True
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            # Create a dummy event to initiate the drag immediately
            dummy = type(event)(event)
            super().mousePressEvent(dummy)
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._is_panning = False
            self.setDragMode(QGraphicsView.RubberBandDrag)
        super().mouseReleaseEvent(event)

class PropertiesPanel(QWidget):
    """Enhanced inspector with type safety, parameter validation, and metadata columns."""
    node_updated = Signal()

    def __init__(self):
        super().__init__()
        self.current_node = None
        self.layout = QVBoxLayout(self)
        
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.form_widget = QWidget()
        self.form = QFormLayout(self.form_widget)
        self.scroll.setWidget(self.form_widget)
        
        self.layout.addWidget(self.scroll)
        
        # Debouncer for update signals
        self.debouncer = QTimer()
        self.debouncer.setSingleShot(True)
        self.debouncer.setInterval(500)
        self.debouncer.timeout.connect(self.node_updated.emit)
        
        # Track active widgets for safe cleanup
        self.active_widgets = {}

    def set_node(self, node_item: NodeItem):
        """Load node properties into inspector with full type safety."""
        self.current_node = node_item
        
        # Clean up previous widgets
        for widget in self.active_widgets.values():
            try: widget.deleteLater()
            except: pass
        self.active_widgets.clear()
        
        while self.form.count():
            child = self.form.takeAt(0)
            if child.widget(): child.widget().deleteLater()

        if not node_item: return

        # FIX: Validate types, but SKIP asset parameters to prevent UUID->0.0 corruption
        for k, v in list(node_item.data.params.items()):
            if TypeSafeParser.is_asset_param(k):
                continue # Do not touch asset strings/UUIDs
            elif TypeSafeParser.is_color_param(k):
                node_item.data.params[k] = TypeSafeParser.parse_color(v)
            elif TypeSafeParser.is_numeric_param(k):
                node_item.data.params[k] = TypeSafeParser.parse_numeric(v)
        
        # ===== AI GENERATED INDICATOR =====
        if node_item.data.is_ai_generated:
            ai_label = QLabel("✨ AI GENERATED NODE ✨")
            ai_label_font = QFont()
            ai_label_font.setBold(True)
            ai_label.setFont(ai_label_font)
            ai_label.setStyleSheet(
                "background: linear-gradient(90deg, #e3f2fd, #f3e5f5); "
                "color: #1565c0; padding: 6px; border-radius: 3px; "
                "border-left: 3px solid #1565c0;"
            )
            ai_label.setToolTip(
                f"This node was generated by Gemini AI.\n"
                f"Class: {node_item.data.ai_source or node_item.data.cls_name}\n"
                f"All parameters are available for editing."
            )
            self.form.addRow(ai_label)
            
        # Meta Information
        self.form.addRow(QLabel("<b>Properties</b>"))
        id_lbl = QLabel(node_item.data.id[:8])
        id_lbl.setStyleSheet("color: gray;")
        self.form.addRow("ID", id_lbl)
        
        name_edit = QLineEdit(node_item.data.name)
        name_edit.textChanged.connect(lambda t: self.update_param("_name", t))
        self.form.addRow("Name", name_edit)
        
        if not MANIM_AVAILABLE:
            self.form.addRow(QLabel("Manim not loaded."))
            return

        try:
            cls = getattr(manim, node_item.data.cls_name, None)
            if not cls: return
            
            sig = inspect.signature(cls.__init__)
            
            # Auto-load missing parameters
            for param_name, param in sig.parameters.items():
                if param_name in ('self', 'args', 'kwargs', 'mobject'): continue
                
                if param_name not in node_item.data.params:
                    if param.default is inspect.Parameter.empty:
                        default_val = param.default
                        if TypeSafeParser.is_color_param(param_name):
                            default_val = TypeSafeParser.parse_color(default_val)
                        elif TypeSafeParser.is_numeric_param(param_name):
                            default_val = TypeSafeParser.parse_numeric(default_val)
                        node_item.data.params[param_name] = default_val
            
            # Create Rows
            for name, param in sig.parameters.items():
                if name in ['self', 'args', 'kwargs', 'mobject']: continue
                
                val = node_item.data.params.get(name, param.default)
                if val is inspect.Parameter.empty: val = None
                
                row_widget = self.create_parameter_row(name, val, param.annotation)
                if row_widget:
                    self.form.addRow(name, row_widget)
                    self.active_widgets[name] = row_widget
                    
        except Exception as e:
            LOGGER.error(f"Inspector Error for {node_item.data.cls_name}: {e}")
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
            state_chk.setToolTip("Check to include in code generation (Ctrl+E to toggle)")
            is_enabled = self.current_node.data.is_param_enabled(key)
            state_chk.setChecked(is_enabled)
            state_chk.stateChanged.connect(
                lambda s: self.current_node.data.set_param_enabled(key, s == 2) if self.current_node else None
            )
            row_layout.addWidget(state_chk, 1)
            
            # Escape String checkbox
            escape_chk = QCheckBox("Escape")
            escape_chk.setToolTip("Check to escape strings (remove quotes)")
            should_escape = self.current_node.data.should_escape_string(key)
            escape_chk.setChecked(should_escape)
            escape_chk.stateChanged.connect(
                lambda s: self.current_node.data.set_escape_string(key, s == 2) if self.current_node else None
            )
            row_layout.addWidget(escape_chk, 1)
            
            return row_widget
        
        except Exception as e:
            LOGGER.error(f"Error creating parameter row for {key}: {e}")
            return None

    def create_typed_widget(self, key, value, annotation):
        """Create typed widget with safe type enforcement and validation."""
        def on_change(v):
            if not self.current_node: return
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
            # 1. ASSET PATHS (Corrected detection)
            if TypeSafeParser.is_asset_param(key):
                combo = QComboBox()
                combo.addItem("(None)", None)
                for asset in ASSETS.get_list():
                    # Show Emoji + Name
                    combo.addItem(f"📄 {asset.name}", asset.id)
                
                if value:
                    idx = combo.findData(value)
                    if idx >= 0: combo.setCurrentIndex(idx)
                
                combo.currentIndexChanged.connect(lambda i: on_change(combo.itemData(i)))
                return combo

            # 2. COLORS
            if TypeSafeParser.is_color_param(key):
                # ... (Keep existing color logic) ...
                btn = QPushButton("Pick Color")
                hex_val = TypeSafeParser.parse_color(value)
                btn.setStyleSheet(f"background-color: {hex_val}; color: white;")
                
                def pick_color():
                    col = QColorDialog.getColor(QColor(hex_val), None, "Select Color")
                    if col.isValid():
                        new_hex = col.name()
                        btn.setStyleSheet(f"background-color: {new_hex}; color: white;")
                        on_change(new_hex)

                btn.clicked.connect(pick_color)
                return btn

             # 3. NUMERIC
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

            # 4. BOOLEAN
            if annotation == bool or isinstance(value, bool):
                chk = QCheckBox()
                chk.setChecked(bool(value))
                chk.stateChanged.connect(lambda s: on_change(s == 2))
                return chk

            # 5. STRING / FALLBACK
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
    add_requested = Signal(str, str) # Type, Class

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
        if not MANIM_AVAILABLE: return
        self.tree.clear()
        mob_root = QTreeWidgetItem(self.tree, ["Mobjects"])
        anim_root = QTreeWidgetItem(self.tree, ["Animations"])
        
        for name in dir(manim):
            if name.startswith("_"): continue
            obj = getattr(manim, name)
            if inspect.isclass(obj):
                try:
                    if issubclass(obj, manim.Mobject) and obj is not manim.Mobject:
                        QTreeWidgetItem(mob_root, [name])
                    elif issubclass(obj, manim.Animation) and obj is not manim.Animation:
                        QTreeWidgetItem(anim_root, [name])
                except: pass
        mob_root.setExpanded(True)

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
                else: item.setHidden(True)
            cat.setHidden(hide)

    def on_dbl_click(self, item, col):
        if item.childCount() == 0:
            p = item.parent()
            t = "Mobject" if p.text(0) == "Mobjects" else "Animation"
            self.add_requested.emit(t, item.text(0))

class VideoRenderPanel(QWidget):
    """Panel for rendering scenes to video."""
    render_requested = Signal(dict)  # Config dict: fps, resolution, quality, output_path

    def __init__(self):
        super().__init__()
        self.render_worker = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("🎬 Video Render")
        title_font = title.font()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Settings Form
        form = QFormLayout()
        
        # FPS
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(15, 60)
        self.fps_spin.setValue(30)
        self.fps_spin.setSuffix(" FPS")
        form.addRow("Frame Rate:", self.fps_spin)
        
        # Resolution
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
        
        # Quality
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["Low (ql) - Fast", "Medium (qm)", "High (qh) - Slow", "Ultra (qk) - Very Slow"])
        self.quality_combo.setCurrentIndex(1)  # Medium
        form.addRow("Quality:", self.quality_combo)
        
        layout.addLayout(form)
        
        # Output Path
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
        
        # Render Controls
        ctrl_layout = QHBoxLayout()
        
        self.render_scene_btn = QPushButton("📽 Render Full Scene")
        self.render_scene_btn.setStyleSheet("background-color: #27ae60; color: white; padding: 8px; font-weight: bold;")
        self.render_scene_btn.clicked.connect(self.render_full_scene)
        ctrl_layout.addWidget(self.render_scene_btn)
        
        self.cancel_btn = QPushButton("⏹ Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setStyleSheet("background-color: #e74c3c; color: white; padding: 8px;")
        self.cancel_btn.clicked.connect(self.cancel_render)
        ctrl_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(ctrl_layout)
        
        # Status Display
        self.status_display = QTextEdit()
        self.status_display.setReadOnly(True)
        self.status_display.setMaximumHeight(120)
        self.status_display.setStyleSheet("background: #ecf0f1; color: #2c3e50;")
        layout.addWidget(QLabel("Status:"))
        layout.addWidget(self.status_display)
        
        layout.addStretch()

    def browse_output(self):
        """Let user select output directory."""
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            str(AppPaths.TEMP_DIR)
        )
        if path:
            self.output_path_lbl.setText(path)

    def render_full_scene(self):
        """Render the full compiled scene to video."""
        # This will be called from main window with the scene code
        output_path = Path(self.output_path_lbl.text())
        if not output_path.exists():
            QMessageBox.warning(self, "Error", "Output directory does not exist.")
            return
        
        # Quality map
        quality_map = {0: "l", 1: "m", 2: "h", 3: "k"}
        quality = quality_map.get(self.quality_combo.currentIndex(), "m")
        
        config = {
            "fps": self.fps_spin.value(),
            "resolution": (self.width_spin.value(), self.height_spin.value()),
            "quality": quality,
            "output_path": str(output_path)
        }
        
        self.render_requested.emit(config)

    def cancel_render(self):
        """Cancel ongoing render."""
        if self.render_worker:
            self.render_worker.stop_render()
            self.update_status("Render cancelled.", "orange")

    def start_rendering(self, worker):
        """Called by main window when render starts."""
        self.render_worker = worker
        self.render_scene_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.update_status("Rendering in progress...", "blue")
        
        worker.progress.connect(lambda msg: self.update_status(msg, "blue"))
        worker.success.connect(self.on_render_success)
        # --- FIX: Connect to a proper handler that resets buttons ---
        worker.error.connect(self.on_render_error)

    def on_render_error(self, err_msg):
        """Called when render fails/cancels to ensure buttons unlock."""
        self.render_scene_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.update_status(f"❌ Error: {err_msg}", "red")
        LOGGER.error(f"Render process stopped: {err_msg}")

    def on_render_success(self, video_path):
        """Called when render completes successfully."""
        self.render_scene_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.update_status(f"✓ Video saved!\n{Path(video_path).name}", "green")
        LOGGER.info(f"Video rendered: {video_path}")

    def update_status(self, message, color="black"):
        """Update status display with colored text."""
        self.status_display.append(f"<span style='color:{color}'><b>[{datetime.now().strftime('%H:%M:%S')}]</b> {message}</span>")
        # Auto-scroll to bottom
        self.status_display.verticalScrollBar().setValue(
            self.status_display.verticalScrollBar().maximum()
        )

class AIPanel(QWidget):
    """Enhanced AI Panel with visual distinction and node generation."""
    merge_requested = Signal(str)
    nodes_generated = Signal(dict)  # Emits dict of generated node info

    def __init__(self):
        super().__init__()
        self.worker = None
        self.last_code = None
        self.extracted_nodes = []  # Track AI-generated nodes
        
        # Create main layout with visual distinction
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # ===== HEADER =====
        header = self._create_header()
        main_layout.addWidget(header)
        
        # ===== CONTENT AREA =====
        content_splitter = QSplitter(Qt.Horizontal)
        
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
        title_font = QFont()
        title_font.setPointSize(11)
        title_font.setBold(True)
        title.setFont(title_font)
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
        label_font = QFont()
        label_font.setBold(True)
        label.setFont(label_font)
        label.setStyleSheet("color: #1565c0;")
        layout.addWidget(label)
        
        self.input = QPlainTextEdit()
        self.input.setPlaceholderText("Describe your animation...\n\nExample: Create a blue rectangle that smoothly rotates 100 degrees.")
        self.input.setMaximumHeight(90)
        self.input.setStyleSheet(
            "QPlainTextEdit { border: 1px solid #bdbdbd; border-radius: 3px; "
            "background: white; padding: 6px; }"
        )
        layout.addWidget(self.input)
        
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
        label_font = QFont()
        label_font.setBold(True)
        label.setFont(label_font)
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
        title_font = QFont()
        title_font.setBold(True)
        title.setFont(title_font)
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
        """Generate AI code."""
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
        selected_model = SETTINGS.get("GEMINI_MODEL", "gemini-3-flash-preview")
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
            f"class EfficientScene(Scene):\n"
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
        selected_model = SETTINGS.get("GEMINI_MODEL", "gemini-3-flash-preview")
        self.worker = AIWorker(sys_prompt, model=selected_model)
        self.worker.chunk_received.connect(self.on_chunk)
        self.worker.finished_signal.connect(self.on_finish)
        self.worker.start()
        
    def on_chunk(self, txt):
        """Handle streamed AI response."""
        self.output.moveCursor(QTextCursor.End)
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
                self.status_label.setText(f"Status: Ready to merge ({len(self.extracted_nodes)} nodes)")
                LOGGER.ai(f"Code block ready. Extracted {len(self.extracted_nodes)} nodes.")
            else:
                self.status_label.setText("Status: Code ready (no nodes detected)")
        else:
            self.status_label.setText("Status: No code block found")
            
    def _extract_nodes_from_code(self, code: str):
        """Extract BOTH mobjects and animations from AI-generated code."""
        self.extracted_nodes = []
        self.nodes_list.clear()
        
        # Extract all object definitions
        pattern = r'(\w+)\s*=\s*([A-Z][a-zA-Z0-9]*)\s*\((.*?)\)(?:\s|$)'
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
                    is_animation = issubclass(cls, manim.Animation) if hasattr(manim, 'Animation') else 'Animation' in class_name
            except:
                pass
                
            self.extracted_nodes.append({
                'var_name': var_name,
                'class_name': class_name,
                'params': params,
                'params_str': params_str,
                'source': 'ai',
                'type': 'animation' if is_animation else 'mobject'
            })
            
            # Add to list with icon indicating type
            node_icon = "🎬" if is_animation else "📦"
            node_type_label = "Animation" if is_animation else "Mobject"
            item = QListWidgetItem(f"{node_icon} {var_name}: {class_name} ({param_count} params) [{node_type_label}]")
            
            # Color code: blue for mobjects, purple for animations
            if is_animation:
                item.setBackground(QColor("#f3e5f5"))
                item.setForeground(QColor("#7b1fa2"))
            else:
                item.setBackground(QColor("#e3f2fd"))
                item.setForeground(QColor("#1565c0"))
            
            # Build detailed tooltip with parameter list
            param_list = "\n".join([f"  • {k}={v[:40]}" for k, v in list(params.items())[:5]])
            if len(params) > 5:
                param_list += f"\n  ... and {len(params)-5} more"
            
            item.setToolTip(
                f"AI Generated {node_type_label}\n"
                f"Variable: {var_name}\n"
                f"Type: {class_name}\n"
                f"Parameters ({param_count}):\n{param_list}\n\n"
                f"All parameters are captured and ready to use."
            )
            self.nodes_list.addItem(item)
        
        # Also show animations from self.play() calls
        pattern_play = r'self\.play\((.*?)\)(?=\s|$)'
        for match in re.finditer(pattern_play, code, re.DOTALL):
            play_content = match.group(1)
            anim_pattern = r'([A-Z][a-zA-Z0-9]*)\((.*?)\)'
            
            for anim_match in re.finditer(anim_pattern, play_content):
                anim_class = anim_match.group(1)
                
                # Skip if not a Manim animation class
                if not hasattr(manim, anim_class):
                    continue
                
                try:
                    cls = getattr(manim, anim_class)
                    is_anim = issubclass(cls, manim.Animation) if hasattr(manim, 'Animation') else 'Animation' in anim_class
                    
                    if is_anim and not any(n['class_name'] == anim_class and n['type'] == 'animation' for n in self.extracted_nodes):
                        # Add this animation if not already added
                        self.extracted_nodes.append({
                            'var_name': f"{anim_class.lower()}_1",
                            'class_name': anim_class,
                            'params': {},
                            'source': 'ai',
                            'type': 'animation'
                        })
                        
                        item = QListWidgetItem(f"🎬 {anim_class.lower()}_1: {anim_class} (from self.play)")
                        item.setBackground(QColor("#f3e5f5"))
                        item.setForeground(QColor("#7b1fa2"))
                        item.setToolTip(f"Animation: {anim_class}\nExtracted from self.play() call")
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
            if char in '([{':
                depth += 1
                current_item += char
            elif char in ')]}':
                depth -= 1
                current_item += char
            elif char == ',' and depth == 0:
                # End of parameter
                if '=' in current_item:
                    try:
                        key, value = current_item.split('=', 1)
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
        if current_item and '=' in current_item:
            try:
                key, value = current_item.split('=', 1)
                key = key.strip()
                value = value.strip()
                params[key] = value
            except ValueError:
                pass
        
        return params
            
    def merge(self):
        """Emit merge signal with code."""
        if self.last_code:
            self.merge_requested.emit(self.last_code)
            # Signal node generation for UI update
            if self.extracted_nodes:
                self.nodes_generated.emit({
                    'code': self.last_code,
                    'nodes': self.extracted_nodes
                })
                
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
        
        Returns: (mobjects, animations, connections)
        Where connections = [(animation_var, mobject_var), ...]
        """
        mobjects = []
        animations = []
        connections = []
        
        # Extract mobject definitions: var = ClassName(params...)
        # Exclude Animation classes
        pattern_mobjects = r'(\w+)\s*=\s*([A-Z][a-zA-Z0-9]*)\s*\((.*?)\)'
        
        for match in re.finditer(pattern_mobjects, code, re.DOTALL):
            var_name, class_name, params_str = match.groups()
            
            # Skip non-Manim classes
            if not hasattr(manim, class_name):
                continue
            
            # Check if it's an Animation class
            is_animation = False
            try:
                if hasattr(manim, class_name):
                    cls = getattr(manim, class_name)
                    is_animation = issubclass(cls, manim.Animation) if hasattr(manim, 'Animation') else 'Animation' in class_name
            except:
                pass
            
            # Parse parameters
            params = AINodeIntegrator._parse_params(params_str)
            
            if is_animation:
                animations.append({
                    'var_name': var_name,
                    'class_name': class_name,
                    'params': params,
                    'source': 'ai',
                    'code_snippet': match.group(0)
                })
            else:
                mobjects.append({
                    'var_name': var_name,
                    'class_name': class_name,
                    'params': params,
                    'source': 'ai',
                    'code_snippet': match.group(0)
                })
        
        # Extract animations from self.play() calls
        # Pattern: self.play(AnimClass(obj, ...))
        pattern_play = r'self\.play\((.*?)\)(?=\s|$)'
        
        for match in re.finditer(pattern_play, code, re.DOTALL):
            play_content = match.group(1)
            
            # Extract individual animations from play call
            anim_pattern = r'([A-Z][a-zA-Z0-9]*)\((.*?(?:\([^)]*\))?[^)]*?)\)'
            for anim_match in re.finditer(anim_pattern, play_content, re.DOTALL):
                anim_class, anim_args = anim_match.groups()
                
                # Skip if not an animation
                if not hasattr(manim, anim_class):
                    continue
                
                # Find which mobject this animation applies to
                for mobject in mobjects:
                    if mobject['var_name'] in anim_args:
                        # Create connection: animation -> mobject
                        connections.append((anim_class, mobject['var_name']))
                        break
        
        return mobjects, animations, connections
        
    @staticmethod
    def _parse_params(params_str: str) -> dict:
        """Parse parameter string into dict with proper quote/constant handling."""
        params = {}
        
        # Enhanced key=value extraction
        for item in params_str.split(','):
            item = item.strip()
            if '=' in item:
                try:
                    key, value = item.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Clean quotes from strings
                    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    
                    # Store cleaned value
                    params[key] = value
                except ValueError:
                    pass
                    
        return params
        
    @staticmethod
    @staticmethod
    def create_node_from_ai(var_name: str, class_name: str, 
                            params: dict, scene_graph, node_type=NodeType.MOBJECT) -> 'NodeItem':
        """
        Create a node in the scene graph from AI definition.
        
        Args:
            var_name: Variable name (e.g., 'circle')
            class_name: Manim class name (e.g., 'Circle')
            params: Parameter dict
            scene_graph: The SceneGraph instance
            node_type: NodeType.MOBJECT or NodeType.ANIMATION
            
        Returns:
            NodeItem: The created node
        """
        # Create NodeData with AI metadata
        node_data = NodeData(var_name, node_type, class_name)
        
        # Apply parameters with type safety
        for param_name, param_value in params.items():
            param_value = str(param_value).strip()
            
            # Use TypeSafeParser for type safety
            if TypeSafeParser.is_color_param(param_name):
                # Strip quotes from color constants
                clean_value = param_value.strip("'\"")
                node_data.params[param_name] = TypeSafeParser.parse_color(clean_value)
            elif TypeSafeParser.is_numeric_param(param_name):
                clean_value = param_value.strip("'\"")
                node_data.params[param_name] = TypeSafeParser.parse_numeric(clean_value)
            else:
                # Strip quotes for all string values
                clean_value = param_value.strip("'\"")
                node_data.params[param_name] = clean_value
                
        # Mark as AI-generated
        node_data.is_ai_generated = True
        node_data.ai_source = class_name
        
        # Create NodeItem
        item = NodeItem(node_data)
        item.setFlag(QGraphicsItem.ItemIsMovable, True)
        item.setFlag(QGraphicsItem.ItemIsSelectable, True)
        
        # Add to scene graph
        scene_graph.scene.addItem(item)
        scene_graph.nodes[item.data.id] = item
        
        # Auto-detect and load all class parameters
        AINodeIntegrator._load_class_parameters(node_data, class_name)
        
        return item
        
    @staticmethod
    def _load_class_parameters(node_data: 'NodeData', class_name: str):
        """
        Auto-detect all parameters from Manim class and load into node_data.
        """
        try:
            cls = getattr(manim, class_name, None)
            if not cls:
                return
                
            sig = inspect.signature(cls.__init__)
            
            for param_name, param in sig.parameters.items():
                # Skip self and special parameters
                if param_name in ('self', 'args', 'kwargs', 'mobject'):
                    continue
                    
                # Skip if already in params
                if param_name in node_data.params:
                    continue
                    
                # Set default value
                if param.default != inspect.Parameter.empty:
                    default_value = param.default
                    
                    # Apply type safety to defaults
                    if TypeSafeParser.is_color_param(param_name):
                        default_value = TypeSafeParser.parse_color(default_value)
                    elif TypeSafeParser.is_numeric_param(param_name):
                        default_value = TypeSafeParser.parse_numeric(default_value)
                        
                    node_data.params[param_name] = default_value
                    
        except Exception as e:
            LOGGER.error(f"Failed to load parameters for {class_name}: {e}")
            
    @staticmethod
    def validate_ai_nodes(nodes: list) -> tuple:
        """
        Validate AI-generated nodes.
        
        Returns: (valid_nodes, errors)
        """
        valid = []
        errors = []
        
        for node in nodes:
            if not hasattr(manim, node['class_name']):
                errors.append(f"Invalid class: {node['class_name']}")
                continue
            valid.append(node)
            
        return valid, errors
        
    @staticmethod
    def merge_ai_code_to_scene(code: str, scene_graph) -> dict:
        """
        Merge AI-generated code into scene with animations and connections.
        
        Returns: {
            'success': bool,
            'nodes_added': int,
            'nodes': list of created GraphicsItems,
            'errors': list of error messages
        }
        """
        try:
            # Parse code - now returns (mobjects, animations, connections)
            mobjects, animations, connections = AINodeIntegrator.parse_ai_code(code)
            
            # Validate all nodes
            valid_mobjects, mob_errors = AINodeIntegrator.validate_ai_nodes(mobjects)
            valid_animations, anim_errors = AINodeIntegrator.validate_ai_nodes(animations)
            
            errors = mob_errors + anim_errors
            
            # Create mobject nodes
            created_nodes = []
            mobject_items = {}  # Track var_name -> NodeItem mapping
            
            for node_def in valid_mobjects:
                try:
                    item = AINodeIntegrator.create_node_from_ai(
                        node_def['var_name'],
                        node_def['class_name'],
                        node_def['params'],
                        scene_graph,
                        node_type=NodeType.MOBJECT
                    )
                    created_nodes.append(item)
                    mobject_items[node_def['var_name']] = item
                except Exception as e:
                    errors.append(f"Failed to create mobject {node_def['var_name']}: {str(e)}")
            
            # Create animation nodes and connect them
            for node_def in valid_animations:
                try:
                    # Generate unique name for animation node
                    anim_var = f"{node_def['class_name'].lower()}_1"
                    
                    item = AINodeIntegrator.create_node_from_ai(
                        anim_var,
                        node_def['class_name'],
                        node_def['params'],
                        scene_graph,
                        node_type=NodeType.ANIMATION
                    )
                    created_nodes.append(item)
                except Exception as e:
                    errors.append(f"Failed to create animation {node_def['class_name']}: {str(e)}")
            
            # Create connections between animations and mobjects
            for anim_class, mobobj_var in connections:
                if mobobj_var in mobject_items:
                    # Find animation node that applies to this mobject
                    for node in created_nodes:
                        if node.data.type == NodeType.ANIMATION and node.data.cls_name == anim_class:
                            mob_node = mobject_items[mobobj_var]
                            # Create wire: animation output -> mobject input
                            try:
                                wire = WireItem(node.out_socket, mob_node.in_socket)
                                scene_graph.scene.addItem(wire)
                                node.out_socket.links.append(wire)
                                mob_node.in_socket.links.append(wire)
                            except Exception as e:
                                pass
                            break
                    
            return {
                'success': len(created_nodes) > 0,
                'nodes_added': len(created_nodes),
                'nodes': created_nodes,
                'errors': errors
            }
            
        except Exception as e:
            LOGGER.error(f"merge_ai_code_to_scene error: {e}")
            return {
                'success': False,
                'nodes_added': 0,
                'nodes': [],
                'errors': [f"Fatal error: {str(e)}"]
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
        paths, _ = QFileDialog.getOpenFileNames(self, "Import", "", "Media (*.png *.jpg *.mp4 *.mp3)")
        for p in paths: ASSETS.add_asset(p)

    def refresh(self):
        self.list.clear()
        for asset in ASSETS.get_list():
            item = QListWidgetItem(asset.name)
            if asset.kind == "image": item.setIcon(QIcon(asset.original_path))
            else: item.setIcon(self.style().standardIcon(QStyle.SP_FileIcon))
            item.setData(Qt.UserRole, asset.id)
            self.list.addItem(item)
            
    def startDrag(self, actions):
        item = self.list.currentItem()
        if not item: return
        mime = QMimeData()
        mime.setText(f"ASSET:{item.data(Qt.UserRole)}")
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(Qt.CopyAction)

class VoiceoverPanel(QWidget):
    """Panel for AI TTS generation and Node synchronization."""
    
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window # Reference to access nodes
        self.tts_worker = None
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("🎙️ AI Voiceover Studio")
        header.setStyleSheet("font-size: 14px; font-weight: bold; color: #2c3e50;")
        layout.addWidget(header)
        
        # Settings Grid
        form = QFormLayout()
        
        # Model Selector
        self.model_combo = QComboBox()
        self.model_combo.addItems(["gemini-2.5-pro-preview-tts", "gemini-2.5-flash-preview-tts"])
        # Load from settings
        saved_model = SETTINGS.get("TTS_MODEL", "gemini-2.5-flash-preview-tts")
        self.model_combo.setCurrentText(saved_model)
        self.model_combo.currentTextChanged.connect(lambda t: SETTINGS.set("TTS_MODEL", t))
        form.addRow("Model:", self.model_combo)
        
        # Voice Selector
        self.voice_combo = QComboBox()
        # Gemini Voices (Standard set)
        voices = ["Puck", "Charon", "Kore", "Fenrir", "Aoede", "Zephyr"] 
        self.voice_combo.addItems(voices)
        self.voice_combo.setCurrentText("Zephyr")
        form.addRow("Voice:", self.voice_combo)
        
        layout.addLayout(form)
        
        # Text Input
        layout.addWidget(QLabel("Script:"))
        self.text_input = QPlainTextEdit()
        self.text_input.setPlaceholderText("Enter text to speak here...")
        layout.addWidget(self.text_input)
        
        # Generate Button
        self.btn_gen = QPushButton("⚡ Generate Audio")
        self.btn_gen.setStyleSheet("background-color: #8e44ad; color: white; padding: 8px; font-weight: bold;")
        self.btn_gen.clicked.connect(self.generate_audio)
        layout.addWidget(self.btn_gen)
        
        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #bdc3c7;")
        layout.addWidget(line)
        
        # Node Sync Section
        sync_lbl = QLabel("🔗 Sync to Animation")
        sync_lbl.setStyleSheet("font-weight: bold;")
        layout.addWidget(sync_lbl)
        
        self.node_combo = QComboBox()
        self.node_combo.setPlaceholderText("Select an Animation Node...")
        layout.addWidget(self.node_combo)
        
        # Refresh Nodes Button (in case new ones added)
        btn_refresh = QPushButton("🔄 Refresh Node List")
        btn_refresh.clicked.connect(self.refresh_nodes)
        layout.addWidget(btn_refresh)
        
        self.status_lbl = QLabel("Ready")
        self.status_lbl.setStyleSheet("color: gray;")
        layout.addWidget(self.status_lbl)
        
        # Initial refresh
        QTimer.singleShot(1000, self.refresh_nodes)

    def refresh_nodes(self):
        """Populate combo box with Animation nodes only."""
        current_id = self.node_combo.currentData()
        self.node_combo.clear()
        
        self.node_combo.addItem("-- Select Animation --", None)
        
        count = 0
        for nid, node in self.main_window.nodes.items():
            if node.data.type == NodeType.ANIMATION:
                # Show Name and Class
                name = f"{node.data.name} ({node.data.cls_name})"
                self.node_combo.addItem(name, nid)
                count += 1
                
        if current_id:
            idx = self.node_combo.findData(current_id)
            if idx >= 0: self.node_combo.setCurrentIndex(idx)
            
        self.status_lbl.setText(f"Found {count} animation nodes.")

    def generate_audio(self):
        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Input Error", "Please enter some text.")
            return
            
        self.btn_gen.setEnabled(False)
        self.status_lbl.setText("Generating audio via Gemini...")
        
        voice = self.voice_combo.currentText()
        model = self.model_combo.currentText()
        
        self.tts_worker = TTSWorker(text, voice, model)
        self.tts_worker.finished_signal.connect(self.on_tts_success)
        self.tts_worker.error_signal.connect(self.on_tts_error)
        self.tts_worker.start()

    def on_tts_success(self, file_path):
        self.btn_gen.setEnabled(True)
        self.status_lbl.setText("Audio generated!")
        
        # 1. Register as Asset
        asset = ASSETS.add_asset(file_path)
        if not asset:
            self.status_lbl.setText("Error registering asset.")
            return

        # 2. Assign to Node (if selected)
        node_id = self.node_combo.currentData()
        if node_id and node_id in self.main_window.nodes:
            node = self.main_window.nodes[node_id]
            node.data.audio_asset_id = asset.id
            node.update() # Refresh graph
            self.status_lbl.setText(f"Saved & Synced to '{node.data.name}'")
            
            # Trigger graph re-compile to add the duration logic
            self.main_window.compile_graph()
            
            QMessageBox.information(self, "Success", f"Audio generated and attached to {node.data.name}.\nRun Time will auto-fit.")
        else:
             QMessageBox.information(self, "Success", "Audio generated and added to Assets.\n(No node selected for sync)")

    def on_tts_error(self, err):
        self.btn_gen.setEnabled(True)
        self.status_lbl.setText("Generation Failed.")
        LOGGER.error(f"TTS Error: {err}")
        QMessageBox.critical(self, "TTS Error", err)

class KeyboardShortcutsDialog(QDialog):
    """Display keyboard shortcuts and help."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.resize(500, 400)
        layout = QVBoxLayout(self)
        
        text_display = QTextEdit()
        text_display.setReadOnly(True)
        text_display.setPlainText(KeyboardShortcuts.describe_shortcuts())
        layout.addWidget(text_display)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

class SettingsDialog(QDialog):
    theme_changed = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(400, 400)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        
        self.api_key = QLineEdit(SETTINGS.get("GEMINI_API_KEY", ""))
        self.api_key.setEchoMode(QLineEdit.Password)
        form.addRow("Gemini API Key:", self.api_key)
        
        self.fps = QSpinBox()
        self.fps.setRange(15, 60)
        self.fps.setValue(int(SETTINGS.get("FPS", 15)))
        form.addRow("Preview FPS:", self.fps)
        
        self.quality = QComboBox()
        self.quality.addItems(["Low (ql)", "Medium (qm)", "High (qh)"])
        self.quality.setCurrentText(SETTINGS.get("QUALITY", "Low (ql)"))
        form.addRow("Quality:", self.quality)
        
        # Theme selector
        self.theme = QComboBox()
        self.theme.addItems([ThemeManager.LIGHT_THEME.capitalize(), ThemeManager.DARK_THEME.capitalize()])
        self.theme.setCurrentText(THEME_MANAGER.current_theme.capitalize())
        form.addRow("Theme:", self.theme)
        
        # Gemini Model selector
        self.gemini_model = QComboBox()
        self.gemini_model.addItems(["gemini-3-flash-preview", "gemini-3-pro-preview"])
        current_model = SETTINGS.get("GEMINI_MODEL", "gemini-3-flash-preview")
        self.gemini_model.setCurrentText(current_model)
        form.addRow("Gemini Model:", self.gemini_model)
        
        layout.addLayout(form)
        btns = QHBoxLayout()
        b_save = QPushButton("Save")
        b_save.clicked.connect(self.save)
        b_cancel = QPushButton("Cancel")
        b_cancel.clicked.connect(self.reject)
        btns.addWidget(b_save)
        btns.addWidget(b_cancel)
        layout.addLayout(btns)

    def save(self):
        SETTINGS.set("GEMINI_API_KEY", self.api_key.text())
        SETTINGS.set("FPS", self.fps.value())
        SETTINGS.set("QUALITY", self.quality.currentText())
        SETTINGS.set("GEMINI_MODEL", self.gemini_model.currentText())
        
        selected_theme = self.theme.currentText().lower()
        THEME_MANAGER.set_theme(selected_theme)
        self.theme_changed.emit(selected_theme)
        
        self.accept()

# ==============================================================================
# 9. MAIN WINDOW
# ==============================================================================

class EfficientManimWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.resize(1600, 1000)
        self.setStyleSheet(THEME_MANAGER.get_stylesheet())
        self.setWindowIcon(QIcon("icon/icon.ico"))
        
        self.nodes = {} 
        self.project_path = None
        self.undo_manager = UndoRedoManager()
        self.project_modified = False # NEW: Track changes
        self.is_ai_generated_code = False  # Track if code came from AI (don't regenerate)
        
        AppPaths.ensure_dirs()
        self.init_font()
        self.setup_ui()
        self.setup_menu()

        # --- NEW: Apply initial theme state ---
        self.apply_theme()
        
        self.render_queue = []
        self.render_timer = QTimer()
        self.render_timer.timeout.connect(self.process_render_queue)
        self.render_timer.start(500)
        
        LOGGER.info("System Ready.")

    def init_font(self):
        font_path = AppPaths.FONTS_DIR / "Geist-Regular.ttf"
        if font_path.exists():
            fid = QFontDatabase.addApplicationFont(str(font_path.absolute()))
            if fid != -1:
                fam = QFontDatabase.applicationFontFamilies(fid)[0]
                self.setFont(QFont(fam, 10))

    def setup_ui(self):
        main = QSplitter(Qt.Horizontal)
        self.setCentralWidget(main)
        
        # LEFT
        self.scene = GraphScene()
        self.scene.selection_changed_signal.connect(self.on_selection)
        
        # NEW: Connect graph changes to modified tracker
        self.scene.graph_changed_signal.connect(self.mark_modified) 
        self.scene.graph_changed_signal.connect(self.compile_graph)
        
        # NEW: Use custom GraphView
        self.view = GraphView(self.scene) 
        main.addWidget(self.view)
        
        # RIGHT
        right = QSplitter(Qt.Vertical)
        self.tabs_top = QTabWidget()
        
       # NEW: Connect property changes to modified tracker
        self.panel_props = PropertiesPanel()
        self.panel_props.node_updated.connect(self.mark_modified) # Track prop changes
        self.panel_props.node_updated.connect(self.on_node_changed)

        self.panel_elems = ElementsPanel()
        self.panel_elems.add_requested.connect(self.add_node_center)
        
        self.panel_assets = AssetsPanel()
        
        self.panel_video = VideoRenderPanel()
        self.panel_video.render_requested.connect(self.render_to_video)
        
        self.panel_ai = AIPanel()
        self.panel_ai.merge_requested.connect(self.merge_ai_code)
        
        # NEW: Init Voiceover Panel
        self.panel_voice = VoiceoverPanel(self)
        
        self.tabs_top.addTab(self.panel_elems, "📦 Elements")
        self.tabs_top.addTab(self.panel_props, "🧩 Properties")
        self.tabs_top.addTab(self.panel_assets, "🗂 Assets")
        
        # AI Integration
        self.tabs_top.addTab(self.panel_ai, "🤖 AI")
        # NEW: Add Voiceover Tab
        self.tabs_top.addTab(self.panel_voice, "🎙️ Voiceover")
        
        # Video Rendering Tab
        self.tabs_top.addTab(self.panel_video, "🎬 Video")

        right.addWidget(self.tabs_top)
        
        self.tabs_bot = QTabWidget()
        
        # Preview Area with Toolbar
        prev_widget = QWidget()
        prev_layout = QVBoxLayout(prev_widget)
        prev_tb = QHBoxLayout()
        prev_tb.addWidget(QLabel("Preview"))
        prev_tb.addStretch()
        prev_layout.addLayout(prev_tb)
        
        self.preview_lbl = QLabel("No Preview")
        self.preview_lbl.setObjectName("PreviewLabel")
        self.preview_lbl.setAlignment(Qt.AlignCenter)
        scr = QScrollArea()
        scr.setWidget(self.preview_lbl)
        scr.setWidgetResizable(True)
        scr.setAlignment(Qt.AlignCenter)
        prev_layout.addWidget(scr)
        
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
        corner_layout.setContentsMargins(0, 0, 15, 0) # 15px right margin
        
        self.theme_btn = QPushButton()
        self.theme_btn.setCursor(Qt.PointingHandCursor)
        # REMOVED: self.theme_btn.setFixedSize(30, 30) <--- Allow it to expand
        self.theme_btn.clicked.connect(self.toggle_theme)
        
        corner_layout.addWidget(self.theme_btn)
        bar.setCornerWidget(self.corner_container, Qt.TopRightCorner)
        # -------------------------------------------------------------------------

        # File Menu
        file_menu = bar.addMenu("File")
        file_menu.addAction("New Project", self.new_project, QKeySequence.New)
        file_menu.addAction("Open Project", self.open_project, QKeySequence.Open)
        file_menu.addAction("Save Project", self.save_project, QKeySequence.Save)
        file_menu.addAction("Save As...", self.save_project_as)
        file_menu.addSeparator()
        file_menu.addAction("Settings", self.open_settings)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close, QKeySequence.Quit)
        
        quit_act = QAction("Exit", self)
        quit_act.setShortcut(QKeySequence("Ctrl+Q"))
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        # Edit Menu
        edit_menu = bar.addMenu("Edit")
        undo_action = QAction("Undo", self)
        undo_action.setShortcut(QKeySequence.Undo)
        undo_action.triggered.connect(self.undo_action)
        edit_menu.addAction(undo_action)
        
        redo_action = QAction("Redo", self)
        redo_action.setShortcut(QKeySequence.Redo)
        redo_action.triggered.connect(self.redo_action)
        edit_menu.addAction(redo_action)
        
        edit_menu.addSeparator()
        delete_action = QAction("Delete Selected", self)
        delete_action.setShortcut(QKeySequence.Delete)
        delete_action.triggered.connect(self.delete_selected)
        edit_menu.addAction(delete_action)
        
        # View Menu
        view_menu = bar.addMenu("View")
        view_menu.addAction("Fit to View", self.fit_view, QKeySequence("Ctrl+0"))
        
        # NEW: Zoom Shortcuts
        view_menu.addSeparator()
        zoom_in_act = QAction("Zoom In", self)
        zoom_in_act.setShortcut(QKeySequence("Ctrl+=")) # Standard Ctrl +
        zoom_in_act.triggered.connect(lambda: self.view.scale(1.15, 1.15))
        view_menu.addAction(zoom_in_act)
        
        zoom_out_act = QAction("Zoom Out", self)
        zoom_out_act.setShortcut(QKeySequence("Ctrl+-"))
        zoom_out_act.triggered.connect(lambda: self.view.scale(1/1.15, 1/1.15))
        view_menu.addAction(zoom_out_act)
        
        view_menu.addSeparator()
        view_menu.addAction("Clear All", self.clear_scene, QKeySequence(Qt.CTRL | Qt.ALT | Qt.Key_Delete))

        # Help Menu
        help_menu = bar.addMenu("Help")
        help_menu.addAction("Keyboard Shortcuts", self.show_shortcuts, QKeySequence(Qt.CTRL | Qt.Key_Question))
        help_menu.addAction("About", self.show_about)

    # --- MENU ACTIONS ---

    def new_project(self):
        """Create a new project."""
        reply = QMessageBox.question(
            self, "New Project", "Clear current project?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.clear_scene()

    def save_project_as(self):
        """Save project with new name."""
        path, _ = QFileDialog.getSaveFileName(self, "Save Project As", "", f"EfficientManim (*{PROJECT_EXT})")
        if path:
            self.project_path = path
            self.save_project()

    def show_shortcuts(self):
        """Show keyboard shortcuts dialog."""
        KeyboardShortcutsDialog(self).exec()

    def show_about(self):
        """Show about dialog."""
        QMessageBox.information(
            self, "About", 
            f"{APP_NAME} v{APP_VERSION}\n\n"
            "A visual node-based Manim IDE with AI integration.\n\n"
            "© 2024 - Efficient Manim Team"
        )

    def toggle_theme(self):
        """Switch between light and dark themes."""
        current = THEME_MANAGER.current_theme
        new_theme = ThemeManager.DARK_THEME if current == ThemeManager.LIGHT_THEME else ThemeManager.LIGHT_THEME
        
        THEME_MANAGER.set_theme(new_theme)
        self.apply_theme()
    
    def apply_theme(self):
        """Apply the current theme stylesheet and update UI elements."""
        is_dark = THEME_MANAGER.current_theme == ThemeManager.DARK_THEME
        
        # 1. Apply Stylesheet
        self.setStyleSheet(THEME_MANAGER.get_stylesheet())
        
        # 2. Update Toggle Button Visuals (With Safety Check)
        if hasattr(self, 'theme_btn'):
            try:
                # Text indicates what you will switch TO
                btn_text = "  ☀ Light Mode  " if is_dark else "  🌙 Dark Mode  "
                self.theme_btn.setText(btn_text)
                self.theme_btn.setToolTip("Toggle Theme")
                
                # Dynamic Colors for high visibility
                if is_dark:
                    # Dark Mode Styling (Button looks bright/white to indicate Light option)
                    btn_style = """
                        QPushButton { 
                            border: 1px solid #555; 
                            border-radius: 4px;
                            background-color: #333; 
                            color: #fff;
                            padding: 4px 8px;
                            font-weight: bold;
                        }
                        QPushButton:hover { background-color: #444; border-color: #fff; }
                    """
                else:
                    # Light Mode Styling (Button looks dark to indicate Dark option)
                    btn_style = """
                        QPushButton { 
                            border: 1px solid #ccc; 
                            border-radius: 4px;
                            background-color: #f0f0f0; 
                            color: #333;
                            padding: 4px 8px;
                            font-weight: bold;
                        }
                        QPushButton:hover { background-color: #e0e0e0; border-color: #333; }
                    """
                
                self.theme_btn.setStyleSheet(btn_style)
            except RuntimeError:
                pass # Widget might be deleted during shutdown or reload, ignore safely

        # 3. Update Canvas Background
        if hasattr(self, 'scene'):
            bg_color = QColor("#2d2d2d") if is_dark else QColor("#f4f6f7")
            self.scene.setBackgroundBrush(QBrush(bg_color))

    def mark_modified(self):
        """Mark project as modified and update window title."""
        self.project_modified = True
        title = f"{APP_NAME} v{APP_VERSION}"
        if self.project_path:
            title += f" - {Path(self.project_path).name}"
        title += " *" # Star indicates unsaved changes
        self.setWindowTitle(title)

    def reset_modified(self):
        """Reset modified flag after save."""
        self.project_modified = False
        title = f"{APP_NAME} v{APP_VERSION}"
        if self.project_path:
            title += f" - {Path(self.project_path).name}"
        self.setWindowTitle(title)

    def closeEvent(self, event):
        """Intercept close event to check for unsaved changes."""
        if self.project_modified and self.nodes:
            reply = QMessageBox.question(
                self, 
                "Unsaved Changes", 
                "You have unsaved changes. Do you want to save before quitting?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )

            if reply == QMessageBox.Save:
                self.save_project()
                # If save was cancelled inside save_project, ignore close
                if self.project_modified: 
                    event.ignore()
                else:
                    event.accept()
            elif reply == QMessageBox.Discard:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

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
            self.view.fitInView(rect, Qt.KeepAspectRatio)
            self.view.scale(0.9, 0.9)
            LOGGER.info("View fitted")

    def clear_scene(self):
        """Clear all nodes and wires."""
        reply = QMessageBox.question(
            self, "Clear Scene", "Delete all nodes and wires?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.nodes.clear()
            self.scene.clear()
            self.undo_manager = UndoRedoManager()
            self.compile_graph()
            LOGGER.info("Scene cleared")

    # --- GRAPH LOGIC ---

    def add_node_center(self, type_str, cls_name):
        self.is_ai_generated_code = False  # Reset flag - manual node added
        if self.code_view.toPlainText().strip() == "":
             self.compile_graph()
        center = self.view.mapToScene(self.view.rect().center())
        self.add_node(type_str, cls_name, pos=(center.x(), center.y()))

    def add_node(self, type_str, cls_name, params=None, pos=(0,0), nid=None):
        # FIX: Normalize string to uppercase to handle "Mobject" (from UI) and "MOBJECT" (from JSON)
        is_mobject = str(type_str).upper() == "MOBJECT"
        ntype = NodeType.MOBJECT if is_mobject else NodeType.ANIMATION
        
        data = NodeData(cls_name, ntype, cls_name)
        if nid: data.id = nid
        if params: data.params = params
        data.pos_x, data.pos_y = pos
        
        item = NodeItem(data)
        self.scene.addItem(item)
        self.nodes[data.id] = item
        LOGGER.info(f"Created {cls_name}")
        self.compile_graph() # Auto-Refresh
        return item
    
    def delete_selected(self):
        for item in self.scene.selectedItems():
            if isinstance(item, NodeItem): self.remove_node(item)
            elif isinstance(item, WireItem): self.remove_wire(item)
        self.compile_graph() # Auto-Refresh

    def remove_node(self, node):
        wires = node.in_socket.links + node.out_socket.links
        for w in wires: self.remove_wire(w)
        if node.data.id in self.nodes: del self.nodes[node.data.id]
        self.scene.removeItem(node)

    def remove_wire(self, wire):
        if wire in wire.start_socket.links: wire.start_socket.links.remove(wire)
        if wire in wire.end_socket.links: wire.end_socket.links.remove(wire)
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
        # Only regenerate code if it's NOT AI-generated
        if not self.is_ai_generated_code:
            self.compile_graph() # Auto-Refresh on param change
        
        # We still want to see the visual preview update in the bottom left
        # even if we don't update the Python code text.
        node = self.panel_props.current_node
        if node and node.data.type == NodeType.MOBJECT:
            self.queue_render(node)

    # --- COMPILER & RENDERER ---

    def compile_graph(self):
        if self.is_ai_generated_code:
            return self.code_view.toPlainText()
        
        code = "from manim import *\nimport numpy as np\n"
        if PYDUB_AVAILABLE:
            code += "from pydub import AudioSegment\n"
        
        code += "\nclass EfficientScene(Scene):\n    def construct(self):\n"

        # 1. Instantiate Mobjects
        mobjects = [n for n in self.nodes.values() if n.data.type == NodeType.MOBJECT]
        m_vars = {}
        for m in mobjects:
            args = []
            for k, v in m.data.params.items():
                if k.startswith("_"): continue
                if not m.data.is_param_enabled(k): continue
                v_clean = self._format_param_value(k, v, m.data)
                args.append(f'{k}={v_clean}')
                
            var = f"m_{m.data.id[:6]}"
            m_vars[m.data.id] = var
            code += f"        {var} = {m.data.cls_name}({', '.join(args)})\n"
            code += f"        self.add({var})\n"

        # 2. Group animations
        animations = [n for n in self.nodes.values() if n.data.type == NodeType.ANIMATION]
        played = set()
        
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
            
            # Process this batch of animations
            for anim, targets in ready_anims:
                anim_args = targets.copy()
                
                # Check for Voiceover Sync
                audio_path = None
                duration_var = None
                
                if anim.data.audio_asset_id:
                    path = ASSETS.get_asset_path(anim.data.audio_asset_id)
                    if path and PYDUB_AVAILABLE:
                        # Clean path for Python string
                        clean_path = path.replace("\\", "/")
                        
                        # Inject Audio Logic
                        audio_var = f"audio_{anim.data.id[:6]}"
                        code += f"        # Voiceover for {anim.data.name}\n"
                        code += f"        self.add_sound(r'{clean_path}')\n"
                        code += f"        {audio_var} = AudioSegment.from_file(r'{clean_path}')\n"
                        
                        # Override run_time parameter
                        anim.data.params['run_time'] = f"{audio_var}.duration_seconds"
                
                # Format parameters
                for k, v in anim.data.params.items():
                    if not k.startswith("_") and anim.data.is_param_enabled(k):
                        if k == 'run_time' and isinstance(v, str) and "duration_seconds" in v:
                             # It's a variable reference we just injected, don't format it as a string
                             anim_args.append(f"{k}={v}")
                        else:
                             v_clean = self._format_param_value(k, v, anim.data)
                             anim_args.append(f"{k}={v_clean}")
                
                play_lines.append(f"{anim.data.cls_name}({', '.join(anim_args)})")
                played.add(anim)

            if play_lines:
                code += f"        self.play({', '.join(play_lines)})\n"
                # Add a small wait after animations to prevent audio cutoff
                code += f"        self.wait(0.5)\n"

            animations = [a for a in animations if a not in played]

        self.code_view.setText(code)
        return code

    def _format_param_value(self, param_name, value, node_data):
        """Safely format parameter value with type enforcement and string escaping."""
        try:
            # 1. ASSET HANDLING
            # If the value matches an Asset ID, return the absolute path string
            if isinstance(value, str) and value in ASSETS.assets:
                abs_path = ASSETS.get_asset_path(value)
                if abs_path:
                    # Manim needs forward slashes even on Windows
                    return f'r"{abs_path}"' 
            
            # 2. Type-safe formatting
            if TypeSafeParser.is_color_param(param_name):
                color_val = str(value).strip()
                if color_val.upper() in dir(manim) and hasattr(manim, color_val.upper()):
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
                if node_data.should_escape_string(param_name):
                    return value.strip("'\"")
                return repr(value)
            
            else:
                return repr(value)
        
        except Exception as e:
            LOGGER.warn(f"Error formatting {param_name}={value}: {e}")
            return repr(value)

    def queue_render(self, node):
        if node.data.id not in self.render_queue:
            self.render_queue.append(node.data.id)
        # Periodically cleanup old temp files to prevent handle leaks
        AppPaths.force_cleanup_old_files(age_seconds=300)

    def process_render_queue(self):
        """Render preview with safe type checking to prevent ufunc errors."""
        if not self.render_queue: return
        
        nid = self.render_queue.pop(0)
        if nid not in self.nodes: return
        node = self.nodes[nid]
        
        # Build Snippet with type-safe parameter formatting
        args = []
        for k, v in node.data.params.items():
            if k.startswith("_"): continue
            if not node.data.is_param_enabled(k): continue
            
            v_clean = self._format_param_value(k, v, node.data)
            args.append(f'{k}={v_clean}')

        script = "from manim import *\nimport numpy as np\n\nclass PreviewScene(Scene):\n    def construct(self):\n"
        script += f"        obj = {node.data.cls_name}({', '.join(args)})\n"
        script += "        self.add(obj)\n"
        
        s_path = AppPaths.TEMP_DIR / f"preview_{nid}.py"
        try:
            with open(s_path, "w") as f: f.write(script)
        except Exception as e:
            LOGGER.error(f"Failed to write preview script: {e}")
            return
        
        fps = int(SETTINGS.get("FPS", 15))
        qual_map = {"Low (ql)": "l", "Medium (qm)": "m", "High (qh)": "h"}
        q = qual_map.get(SETTINGS.get("QUALITY", "Low (ql)"), "l")
        
        worker = RenderWorker(s_path, nid, AppPaths.TEMP_DIR, fps, q)
        worker.success.connect(self.on_render_ok)
        worker.error.connect(lambda e: LOGGER.manim(e))
        worker.start()
        setattr(self, f"rw_{nid}", worker)

    def on_render_ok(self, nid, path):
        if nid in self.nodes:
            node = self.nodes[nid]
            node.data.preview_path = path
            node.update()
            sel = self.scene.selectedItems()
            if sel and sel[0] == node:
                self.show_preview(node)

    def show_preview(self, node):
        """Display preview for selected node only."""
        # Clear previous preview to release file handles
        self.preview_lbl.clear()
        
        # If node has a preview, show it
        if node and hasattr(node, 'data') and node.data.preview_path:
            if os.path.exists(node.data.preview_path):
                try:
                    pix = QPixmap(node.data.preview_path)
                    if not pix.isNull():
                        # Create a copy to avoid holding file handle
                        scaled_pix = pix.scaledToWidth(300, Qt.SmoothTransformation)
                        self.preview_lbl.setPixmap(scaled_pix)
                        return
                except Exception as e:
                    LOGGER.warn(f"Failed to load preview: {e}")
        
        # If no preview or not found, show status
        self.preview_lbl.setText("No preview")
        self.preview_lbl.setAlignment(Qt.AlignCenter)
        self.preview_lbl.setStyleSheet("color: gray; font-size: 10pt;")

    # --- VIDEO RENDERING ---

    def render_to_video(self, config):
        """Render full scene to video with specified config."""
        try:
            # Validate we have a compilable scene
            if not self.code_view.toPlainText().strip():
                QMessageBox.warning(self, "Error", "No scene code to render. Create some nodes first.")
                return
            
            output_dir = Path(config["output_path"])
            if not output_dir.exists():
                QMessageBox.warning(self, "Error", f"Output directory does not exist: {output_dir}")
                return
            
            LOGGER.info("Building video render script...")
            
            # Generate full scene code
            scene_code = self.code_view.toPlainText()
            
            # Ensure it has EfficientScene class (or add wrapper)
            if "class EfficientScene" not in scene_code:
                scene_code = "from manim import *\n\nclass EfficientScene(Scene):\n    def construct(self):\n        pass\n"
            
            # Write scene to temporary file
            script_path = output_dir / "video_render_scene.py"
            with open(script_path, "w") as f:
                f.write(scene_code)
            
            LOGGER.info(f"Scene script written to {script_path}")
            
            # Start video render worker
            fps = config["fps"]
            resolution = config["resolution"]
            quality = config["quality"]
            
            worker = VideoRenderWorker(script_path, output_dir, fps, resolution, quality)
            
            # Connect signals
            worker.progress.connect(lambda msg: LOGGER.info(msg))
            worker.success.connect(self.on_video_render_success)
            worker.error.connect(self.on_video_render_error)
            
            # Store reference and start
            self.video_render_worker = worker
            self.panel_video.start_rendering(worker)
            worker.start()
            
            LOGGER.info(f"Video render started: {fps}fps, {resolution}, quality {quality}")
            
        except Exception as e:
            LOGGER.error(f"Video render setup failed: {e}")
            QMessageBox.critical(self, "Render Error", f"Failed to start render:\n{e}")

    def on_video_render_success(self, video_path):
        """Called when video render completes successfully."""
        LOGGER.info(f"✓ Video rendered successfully: {video_path}")
        self.panel_video.on_render_success(video_path)
        
        # Optionally show file dialog to open the video
        reply = QMessageBox.information(
            self,
            "Render Complete",
            f"Video saved to:\n{video_path}\n\nOpen video file?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
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
        
        IMPORTANT: Completely replaces ALL previous nodes with AI-generated nodes.
        The AI-generated code is used DIRECTLY without regeneration.
        """
        LOGGER.ai("Merging AI-generated code...")
        
        try:
            # DELETE ALL EXISTING NODES - AI code completely replaces everything
            LOGGER.ai(f"Removing {len(self.nodes)} previous node(s)...")
            nodes_to_remove = list(self.nodes.values())
            for node_item in nodes_to_remove:
                self.scene.removeItem(node_item)
                del self.nodes[node_item.data.id]
            
            # Clear render queue
            self.render_queue.clear()
            
            # Use AINodeIntegrator for robust parsing and node creation
            result = AINodeIntegrator.merge_ai_code_to_scene(code, self)
            
            if result['success']:
                LOGGER.ai(f"✅ Successfully added {result['nodes_added']} node(s)")
                
                # IMPORTANT: Set the code view to the AI-generated code directly
                # This ensures the exact AI output is preserved without regeneration
                self.code_view.setText(code)
                self.is_ai_generated_code = True  # Mark as AI code - don't regenerate
                LOGGER.ai("Code view updated with AI-generated code (locked from regeneration)")
                
                # Update properties panel if there are created nodes
                if result['nodes']:
                    first_node = result['nodes'][0]
                    self.panel_props.set_node(first_node)
                    # Ensure all parameters are loaded
                    LOGGER.ai(f"Inspector updated - showing {len(first_node.data.params)} parameters")
                    
                # Trigger render preview for new nodes
                for node in result['nodes']:
                    self.queue_render(node)
                    
            else:
                LOGGER.error("Failed to merge AI code:")
                for error in result['errors']:
                    LOGGER.error(f"  - {error}")
                    
        except Exception as e:
            LOGGER.error(f"AI merge error: {str(e)}")
            traceback.print_exc()

    # --- PROJECT I/O ---

    def save_project(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Project", "", f"EfficientManim (*{PROJECT_EXT})")
        if not path: return
        
        meta = {"name": Path(path).stem, "created": str(datetime.now()), "version": APP_VERSION}
        
        graph_data = {
            "nodes": [n.data.to_dict() for n in self.nodes.values()],
            "wires": [{"start": w.start_socket.parentItem().data.id, "end": w.end_socket.parentItem().data.id} 
                      for w in self.scene.items() if isinstance(w, WireItem)]
        }
        
        try:
            with tempfile.TemporaryDirectory() as td:
                t_path = Path(td)
                
                # Write JSONs
                with open(t_path / "metadata.json", "w") as f: json.dump(meta, f, indent=2)
                with open(t_path / "graph.json", "w") as f: json.dump(graph_data, f, indent=2)
                with open(t_path / "code.py", "w") as f: f.write(self.code_view.toPlainText())
                
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
                        d["local"] = dst_name # Important: Store local ref
                        asset_manifest.append(d)
                    else:
                        LOGGER.warn(f"Could not find asset to save: {source_path}")
                
                with open(t_path / "assets.json", "w") as f: json.dump(asset_manifest, f, indent=2)
                
                # Zip it up
                shutil.make_archive(str(Path(path).parent / Path(path).stem), 'zip', t_path)
                
                # Rename .zip to .efp
                final_zip = Path(path).parent / f"{Path(path).stem}.zip"
                final_efp = Path(path).with_suffix(PROJECT_EXT)
                
                if final_efp.exists(): final_efp.unlink()
                shutil.move(str(final_zip), final_efp)
                
                LOGGER.info(f"Project saved to {final_efp}")
                
        except Exception as e:
            LOGGER.error(f"Save Failed: {e}")
            traceback.print_exc()

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open", "", f"EfficientManim (*{PROJECT_EXT})")
        if not path: return
        
        try:
            self.nodes.clear(); self.scene.clear(); ASSETS.clear()
            
            # Create a clean extraction folder for this project session
            dest = AppPaths.TEMP_DIR / "Project_Assets"
            if dest.exists(): shutil.rmtree(dest)
            dest.mkdir(parents=True)
            
            with zipfile.ZipFile(path, 'r') as zf: zf.extractall(dest)
            
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
                        a = Asset(ad["name"], str(extracted_path.as_posix()), ad["kind"])
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
                        type_str = nd["type"].capitalize() if nd["type"].upper() == "MOBJECT" else "Animation"
                        
                        node = self.add_node(type_str, nd["cls_name"], nd["params"], nd["pos"], nd["id"])
                        
                        # Restore metadata
                        if "param_metadata" in nd:
                            node.data.param_metadata = nd["param_metadata"]
                        if "is_ai_generated" in nd:
                            node.data.is_ai_generated = nd["is_ai_generated"]
                        
                        node_map[nd["id"]] = node
                        
                    for w in g["wires"]:
                        n1, n2 = node_map.get(w["start"]), node_map.get(w["end"])
                        if n1 and n2: self.scene.try_connect(n1.out_socket, n2.in_socket)
            
            # 3. Load Saved Code (if any)
            code_py = dest / "code.py"
            if code_py.exists():
                with open(code_py, "r") as f:
                    self.code_view.setText(f.read())
            
            self.compile_graph()
            LOGGER.info("Project Loaded Successfully.")
            
        except Exception as e:
            LOGGER.error(f"Open Failed: {e}")
            traceback.print_exc()

    def open_settings(self):
        SettingsDialog(self).exec()

    def append_log(self, level, msg):
        c = "black"
        if level == "ERROR": c = "red"
        elif level == "WARN": c = "orange"
        elif level == "AI": c = "blue"
        elif level == "MANIM": c = "purple"
        ts = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"<span style='color:{c}'><b>[{ts}] {level}:</b> {msg}</span>")

# ==============================================================================
# 10. ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    sys._excepthook = sys.excepthook
    def exception_hook(exctype, value, traceback_obj):
        LOGGER.error(f"CRITICAL: {value}")
        traceback.print_tb(traceback_obj)
        sys._excepthook(exctype, value, traceback_obj)
    sys.excepthook = exception_hook
    
    win = EfficientManimWindow()
    win.show()
    sys.exit(app.exec())
    