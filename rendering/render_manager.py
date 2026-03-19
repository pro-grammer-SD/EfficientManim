from __future__ import annotations
# -*- coding: utf-8 -*-

import os
import subprocess
import traceback
import uuid
import urllib.parse

import requests
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QImage, QPixmap

from core.config import AppPaths
from utils.helpers import convert_to_wav, generate
from utils.logger import LOGGER


class ImageLoaderWorker(QThread):
    """Loads and scales image in background to keep UI smooth."""

    image_loaded = Signal(QPixmap)

    def __init__(self, path: str, max_width: int):
        super().__init__()
        self.path = path
        self.max_width = max_width

    def run(self):
        if not os.path.exists(self.path):
            return

        image = QImage(self.path)
        if not image.isNull():
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

    def __init__(self, prompt: str, model: str = "gemini-3-flash-preview"):
        super().__init__()
        self.prompt = prompt
        self.model = model

    def run(self):
        try:
            generate(self.prompt, self.handle_chunk, model=self.model)
            self.finished_signal.emit()
        except Exception as exc:
            self.error_signal.emit(str(exc))

    def handle_chunk(self, text: str):
        self.chunk_received.emit(text)


class LatexApiWorker(QThread):
    """Fetches rendered LaTeX PNG from external API in background."""

    success = Signal(bytes)
    error = Signal(str)

    def __init__(self, latex_str: str):
        super().__init__()
        self.latex_str = latex_str

    def run(self):
        if not self.latex_str.strip():
            return

        try:
            base = "https://mathpad.ai/api/v1/latex2image"
            params = {
                "latex": self.latex_str,
                "format": "png",
                "scale": 4,
            }
            url = f"{base}?{urllib.parse.urlencode(params)}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            self.success.emit(response.content)
        except Exception as exc:
            self.error.emit(str(exc))


class TTSWorker(QThread):
    finished_signal = Signal(str)
    error_signal = Signal(str)

    def __init__(self, text: str, voice_name: str, model_name: str):
        super().__init__()
        self.text = text
        self.voice_name = voice_name
        self.model_name = model_name
        self.api_key = os.environ.get("GEMINI_API_KEY")

    def run(self):
        if not self.api_key:
            self.error_signal.emit("API key missing")
            return

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)
            contents = self.text

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

            full_audio = b""
            mime_type = "audio/wav"

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

            final_data = full_audio
            if "wav" not in mime_type:
                final_data = convert_to_wav(full_audio, mime_type)

            filename = f"tts_{uuid.uuid4().hex[:8]}.wav"
            save_path = AppPaths.TEMP_DIR / filename

            with open(save_path, "wb") as f:
                f.write(final_data)

            self.finished_signal.emit(str(save_path))

        except Exception as exc:
            self.error_signal.emit(str(exc))


class RenderWorker(QThread):
    """Fast, single-frame renderer for node previews."""

    success = Signal(str, str)
    error = Signal(str)

    def __init__(self, script_path, node_id, output_dir, fps, quality):
        super().__init__()
        self.script_path = script_path
        self.node_id = node_id
        self.output_dir = output_dir
        self.quality = quality
        self._subprocess = None
        self._cancelled = False

    def cancel(self):
        self._cancelled = True
        proc = self._subprocess
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass

    def run(self):
        try:
            if self._cancelled:
                self.error.emit("Cancelled before start.")
                return

            if not self.output_dir.exists():
                self.output_dir.mkdir(parents=True)

            flags = ["-s", "--format=png", "--disable_caching", "-r", "400,300"]
            flags.append(f"-q{self.quality}")
            cmd = ["manim"] + flags + [str(self.script_path), "PreviewScene"]

            LOGGER.log("INFO", "RENDER", f"Preview render: {self.node_id[:8]}")

            self._subprocess = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(self.output_dir),
            )
            stdout, stderr = self._subprocess.communicate()
            returncode = self._subprocess.returncode
            self._subprocess = None

            if self._cancelled:
                self.error.emit("Cancelled.")
                return

            if returncode != 0:
                err = (stderr or stdout or "Unknown error")[:200]
                LOGGER.log(
                    "WARN", "RENDER", f"Preview failed [{self.node_id[:8]}]: {err[:80]}"
                )
                self.error.emit(f"Manim failed: {err[:80]}")
                return

            media_dir = self.output_dir / "media"
            pngs = list(media_dir.rglob("*.png")) if media_dir.exists() else []

            if pngs:
                latest = max(pngs, key=os.path.getmtime)
                self.success.emit(self.node_id, str(latest.absolute()))
            else:
                self.error.emit("No PNG generated.")

        except Exception as exc:
            tb = traceback.format_exc()
            LOGGER.log(
                "ERROR",
                "RENDER",
                f"RenderWorker exception [{self.node_id[:8]}]: {exc}\n{tb}",
            )
            self.error.emit(str(exc))


class VideoRenderWorker(QThread):
    """Renders full scenes to video using Manim."""

    progress = Signal(str)
    success = Signal(str)
    error = Signal(str)

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
        self.resolution = resolution
        self.quality = quality
        self.scene_class = scene_class
        self.is_running = True

    def stop_render(self):
        self.is_running = False
        proc = getattr(self, "process", None)
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass

    def run(self):
        try:
            if not self.is_running:
                self.error.emit("Render cancelled.")
                return

            self.progress.emit("Building render command...")
            flags = ["--disable_caching", f"-q{self.quality}", f"--fps={self.fps}"]
            if self.resolution:
                w, h = self.resolution
                flags.append(f"--resolution={w},{h}")
            cmd = ["manim"] + flags + [str(self.script_path), self.scene_class]

            self.progress.emit("Starting render...")

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(self.output_dir),
            )

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

        except Exception as exc:
            self.error.emit(f"Render error: {exc}")
