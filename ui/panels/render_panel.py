from __future__ import annotations
# -*- coding: utf-8 -*-

from pathlib import Path

from PySide6.QtCore import Qt, Signal, QUrl, QTimer
from PySide6.QtGui import QFont
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.config import AppPaths
from utils.tooltips import apply_tooltip


class VideoOutputPanel(QWidget):
    """Integrated Video Player with Seek and Play/Pause controls."""

    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.duration = 0
        self._autoplay_pending = False
        self._has_media = False
        self._was_playing = False

        # Init Player
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)

        self.player.setVideoOutput(self.video_widget)

        # Connections
        self.player.positionChanged.connect(self.on_position_changed)
        self.player.durationChanged.connect(self.on_duration_changed)
        self.player.mediaStatusChanged.connect(self.on_media_status_changed)
        self.player.playbackStateChanged.connect(self.on_playback_state_changed)
        if hasattr(self.player, "errorOccurred"):
            self.player.errorOccurred.connect(self.on_player_error)

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header_bar = QFrame()
        header_bar.setStyleSheet(
            "background: #ffffff; border-bottom: 1px solid #34495e;"
        )
        header_layout = QHBoxLayout(header_bar)
        header_layout.setContentsMargins(10, 5, 10, 5)

        lbl = QLabel("Output Monitor")
        lbl.setStyleSheet("color: #000000; font-weight: bold;")
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

        # Play Button
        self.btn_play = QPushButton("Play")
        self.btn_play.setFixedSize(60, 30)
        self.btn_play.clicked.connect(self.play)
        apply_tooltip(
            self.btn_play,
            "Play or pause preview",
            "Toggle video playback",
            "Space",
            "Play/Pause",
        )
        ctrl_layout.addWidget(self.btn_play)

        # Pause Button
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setFixedSize(60, 30)
        self.btn_pause.clicked.connect(self.pause)
        apply_tooltip(
            self.btn_pause,
            "Pause playback",
            "Pause video without resetting",
            None,
            "Pause",
        )
        ctrl_layout.addWidget(self.btn_pause)

        # Slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self.set_position)
        self.slider.sliderPressed.connect(self.on_slider_pressed)
        self.slider.sliderReleased.connect(self.on_slider_released)
        self.slider.valueChanged.connect(self.on_slider_value_changed)
        self.slider.setEnabled(False)
        ctrl_layout.addWidget(self.slider)

        # Time Label
        self.lbl_time = QLabel("00:00 / 00:00")
        self.lbl_time.setStyleSheet("font-family: monospace;")
        ctrl_layout.addWidget(self.lbl_time)

        layout.addWidget(controls)

    def load_video(self, file_path, autoplay=True):
        """Load a video file and optionally start playing.

        IMPORTANT: Never call play() immediately after setSource() — Qt hasn't
        decoded the first frame yet and QVideoWidget will stay black.  Instead,
        arm a one-shot slot that fires once the media reaches LoadedMedia status.
        """
        self._autoplay_pending = autoplay
        self._has_media = False
        self.btn_play.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.slider.setEnabled(False)
        # Disconnect any previous one-shot connection to avoid stacking
        try:
            self.player.mediaStatusChanged.disconnect(self._on_load_ready)
        except RuntimeError:
            pass
        self.player.mediaStatusChanged.connect(self._on_load_ready)

        # Reset player to avoid stale state
        try:
            self.player.stop()
        except Exception:
            pass

        self.player.setSource(QUrl.fromLocalFile(file_path))
        self.audio_output.setVolume(1.0)

    def _on_load_ready(self, status):
        """Fire once when media is loaded; then disconnect the one-shot."""
        loaded = (
            QMediaPlayer.MediaStatus.LoadedMedia,
            QMediaPlayer.MediaStatus.BufferedMedia,
        )
        if status in loaded:
            # Disconnect immediately — one-shot behaviour
            try:
                self.player.mediaStatusChanged.disconnect(self._on_load_ready)
            except RuntimeError:
                pass
            self._has_media = True
            self.slider.setEnabled(True)
            self.btn_play.setEnabled(True)
            self.btn_pause.setEnabled(True)
            if self._autoplay_pending:
                self.player.play()
            else:
                # Force first frame to display
                self.player.play()
                QTimer.singleShot(60, self.player.pause)
            self._autoplay_pending = False

    def play(self):
        if not self._has_media:
            self.lbl_time.setText("No video loaded")
            return
        self.player.play()

    def pause(self):
        if not self._has_media:
            return
        self.player.pause()

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
        if self._has_media:
            self.player.setPosition(position)
            self.update_time_label(position)

    def update_time_label(self, current_ms):
        def fmt(ms):
            seconds = (ms // 1000) % 60
            minutes = ms // 60000
            return f"{minutes:02}:{seconds:02}"

        self.lbl_time.setText(f"{fmt(current_ms)} / {fmt(self.duration)}")

    def on_media_status_changed(self, status):
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.player.pause()
            self.player.setPosition(0)

    def on_playback_state_changed(self, state):
        playing = state == QMediaPlayer.PlaybackState.Playing
        self.btn_play.setEnabled(not playing)
        self.btn_pause.setEnabled(playing)

    def on_slider_pressed(self):
        self._was_playing = (
            self.player.playbackState() == QMediaPlayer.PlaybackState.Playing
        )
        if self._was_playing:
            self.player.pause()

    def on_slider_released(self):
        if self._has_media:
            self.player.setPosition(self.slider.value())
            if self._was_playing:
                self.player.play()

    def on_slider_value_changed(self, value):
        if self.slider.isSliderDown():
            self.update_time_label(value)

    def on_player_error(self, *args):
        self.btn_play.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.slider.setEnabled(False)
        self.lbl_time.setText("Playback error")


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

        title = QLabel("Video Render")
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
        apply_tooltip(
            self.browse_btn,
            "Choose output folder",
            "Select where rendered videos are saved",
            None,
            "Browse Output",
        )
        path_layout.addWidget(self.browse_btn)
        layout.addLayout(path_layout)

        ctrl_layout = QHBoxLayout()
        self.render_scene_btn = QPushButton("Render Scene")
        self.render_scene_btn.setStyleSheet(
            "background-color: #27ae60; color: white; padding: 8px;"
        )
        self.render_scene_btn.clicked.connect(self.render_full_scene)
        apply_tooltip(
            self.render_scene_btn,
            "Render the current scene",
            "Uses the settings above",
            "Ctrl+R",
            "Render",
        )
        ctrl_layout.addWidget(self.render_scene_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setStyleSheet(
            "background-color: #e74c3c; color: white; padding: 8px;"
        )
        self.cancel_btn.clicked.connect(self.cancel_render)
        apply_tooltip(
            self.cancel_btn,
            "Stop the current render",
            "Ends the render process",
            None,
            "Cancel Render",
        )
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
