"""
Timeline Templates Extension - Pre-built Animation Track Templates

Provides commonly-used timeline track templates for quick animation setup.
Includes a fully functional Timeline Panel for managing animation tracks.

Example usage:
    from core.extension_api import ExtensionAPI
    api = ExtensionAPI("timeline-templates")
    setup(api)
"""

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QDoubleSpinBox,
    QListWidget,
    QListWidgetItem,
    QGridLayout,
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont
import logging

LOGGER = logging.getLogger("timeline_templates_extension")


# Extension metadata (declares required permissions)
EXTENSION_METADATA = {
    "name": "Timeline Templates",
    "author": "EfficientManim",
    "version": "1.0.0",
    "description": "Pre-built timeline track templates and timeline manager panel",
    "permissions": ["register_ui_panel", "register_timeline_track"],
}


def setup(api):
    """
    Initialize timeline templates extension.

    Registers pre-built timeline track generators for common animation patterns
    and registers the Timeline Panel for managing animations.
    """
    # Register UI panel for timeline management
    api.register_ui_panel(
        panel_name="Timeline Manager",
        widget_class="core.extensions.timeline_templates.TimelineManagerPanel",
        position="bottom",
    )

    # Register fade-in/out transition track
    api.register_timeline_track(
        track_name="Fade Transition",
        class_path="core.extensions.timeline_templates.FadeTransitionTrack",
        description="Pre-configured fade in/out animation track",
    )

    # Register pan/zoom track
    api.register_timeline_track(
        track_name="Pan & Zoom",
        class_path="core.extensions.timeline_templates.PanZoomTrack",
        description="Camera pan and zoom animation track template",
    )

    # Register particle effect track
    api.register_timeline_track(
        track_name="Particle Effects",
        class_path="core.extensions.timeline_templates.ParticleEffectTrack",
        description="Particle system animation track with presets",
    )

    LOGGER.info("✓ Timeline Templates extension initialized")
    return True


class TimelineManagerPanel(QWidget):
    """
    Timeline manager panel for creating and managing animation tracks.

    Features:
    - Create animation tracks from templates
    - Manage track parameters (duration, timing, etc.)
    - Preview animations
    - Drag-and-drop track organization
    """

    track_created = Signal(str)  # Emits track name

    def __init__(self):
        super().__init__()
        self.tracks = []
        self.current_duration = 5.0  # seconds
        self._setup_ui()

    def _setup_ui(self):
        """Build the timeline panel UI."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Title
        title = QLabel("⏱️ Timeline Manager")
        title_font = QFont()
        title_font.setPointSize(11)
        title_font.setBold(True)
        title.setFont(title_font)
        main_layout.addWidget(title)

        # Total Duration Control
        duration_layout = QHBoxLayout()
        duration_layout.addWidget(QLabel("Total Duration (s):"))
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setMinimum(0.1)
        self.duration_spin.setMaximum(600.0)
        self.duration_spin.setValue(self.current_duration)
        self.duration_spin.setSingleStep(0.5)
        self.duration_spin.valueChanged.connect(self._on_duration_changed)
        duration_layout.addWidget(self.duration_spin)
        main_layout.addLayout(duration_layout)

        # Timeline Slider (visual scrubber)
        playback_layout = QHBoxLayout()
        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(100)
        self.timeline_slider.setValue(0)
        self.timeline_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 8px;
                background: #e5e7eb;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #2563eb;
                width: 14px;
                margin: -3px 0;
                border-radius: 7px;
            }
            QSlider::handle:horizontal:hover {
                background: #1d4ed8;
            }
        """)
        playback_layout.addWidget(QLabel("Timeline:"))
        playback_layout.addWidget(self.timeline_slider, 1)

        # Time display
        self.time_label = QLabel("0.0s / 5.0s")
        self.time_label.setStyleSheet("font-family: monospace; min-width: 80px;")
        playback_layout.addWidget(self.time_label)
        main_layout.addLayout(playback_layout)

        # Playback controls
        controls_layout = QHBoxLayout()
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.setMaximumWidth(80)
        self.play_btn.clicked.connect(self._on_play)
        controls_layout.addWidget(self.play_btn)

        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setMaximumWidth(80)
        self.stop_btn.clicked.connect(self._on_stop)
        controls_layout.addWidget(self.stop_btn)

        controls_layout.addStretch()
        main_layout.addLayout(controls_layout)

        # Track Templates
        template_label = QLabel("Track Templates:")
        template_label.setStyleSheet("font-weight: bold;")
        main_layout.addWidget(template_label)

        # Template buttons
        template_layout = QGridLayout()
        templates = [
            ("Fade In/Out", "FadeTransitionTrack"),
            ("Pan/Zoom", "PanZoomTrack"),
            ("Particles", "ParticleEffectTrack"),
        ]

        for idx, (name, class_name) in enumerate(templates):
            btn = QPushButton(f"+ {name}")
            btn.clicked.connect(
                lambda checked, cn=class_name: self._add_track(cn, name)
            )
            template_layout.addWidget(btn, idx // 2, idx % 2)

        main_layout.addLayout(template_layout)

        # Active Tracks
        tracks_label = QLabel("Active Tracks:")
        tracks_label.setStyleSheet("font-weight: bold;")
        main_layout.addWidget(tracks_label)

        self.tracks_list = QListWidget()
        self.tracks_list.setMaximumHeight(120)
        main_layout.addWidget(self.tracks_list)

        # Track control
        track_control_layout = QHBoxLayout()
        self.remove_track_btn = QPushButton("Remove Selected")
        self.remove_track_btn.clicked.connect(self._remove_selected_track)
        track_control_layout.addWidget(self.remove_track_btn)

        self.clear_tracks_btn = QPushButton("Clear All")
        self.clear_tracks_btn.clicked.connect(self._clear_all_tracks)
        track_control_layout.addWidget(self.clear_tracks_btn)

        main_layout.addLayout(track_control_layout)

        # Status
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-size: 10px; color: #666;")
        main_layout.addWidget(self.status_label)

        main_layout.addStretch()

        # Playback timer
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self._update_playback)
        self.is_playing = False
        self.playback_position = 0.0

    def _on_duration_changed(self, value):
        """Handle duration change."""
        self.current_duration = value
        self.time_label.setText(f"0.0s / {value:.1f}s")
        self.status_label.setText(f"Duration: {value:.1f}s")

    def _add_track(self, class_name: str, display_name: str):
        """Add a track from template."""
        track_item = f"{display_name} (0.0s - {self.current_duration:.1f}s)"
        item = QListWidgetItem(track_item)
        item.setData(
            Qt.ItemDataRole.UserRole,
            {"class": class_name, "duration": self.current_duration},
        )
        self.tracks_list.addItem(item)
        self.tracks.append(
            {
                "name": display_name,
                "class": class_name,
                "duration": self.current_duration,
            }
        )
        self.status_label.setText(f"✓ Added {display_name}")
        self.track_created.emit(display_name)
        LOGGER.info(f"Added track: {display_name}")

    def _remove_selected_track(self):
        """Remove selected track from list."""
        current_row = self.tracks_list.currentRow()
        if current_row >= 0:
            self.tracks_list.takeItem(current_row)
            if current_row < len(self.tracks):
                removed = self.tracks.pop(current_row)
                self.status_label.setText(f"✓ Removed {removed['name']}")
                LOGGER.info(f"Removed track: {removed['name']}")

    def _clear_all_tracks(self):
        """Clear all tracks."""
        self.tracks_list.clear()
        self.tracks = []
        self.status_label.setText("✓ Cleared all tracks")
        LOGGER.info("Cleared all tracks")

    def _on_play(self):
        """Start playback."""
        if not self.is_playing:
            self.is_playing = True
            self.playback_position = 0.0
            self.playback_timer.start(50)  # 50ms update interval
            self.play_btn.setText("⏸ Pause")
            self.status_label.setText("▶ Playing...")
        else:
            self.is_playing = False
            self.playback_timer.stop()
            self.play_btn.setText("▶ Play")
            self.status_label.setText("⏸ Paused")

    def _on_stop(self):
        """Stop playback."""
        self.is_playing = False
        self.playback_timer.stop()
        self.playback_position = 0.0
        self.timeline_slider.setValue(0)
        self.time_label.setText(f"0.0s / {self.current_duration:.1f}s")
        self.play_btn.setText("▶ Play")
        self.status_label.setText("⏹ Stopped")

    def _update_playback(self):
        """Update playback position."""
        if self.is_playing:
            self.playback_position += 0.05  # 50ms increment

            if self.playback_position >= self.current_duration:
                self._on_stop()
                return

            # Update slider
            progress = (self.playback_position / self.current_duration) * 100
            self.timeline_slider.setValue(int(progress))

            # Update time label
            self.time_label.setText(
                f"{self.playback_position:.1f}s / {self.current_duration:.1f}s"
            )


class FadeTransitionTrack:
    """
    Fade in/out transition timeline track template.

    Provides a standard fade animation pattern:
    - Fade in over N seconds
    - Hold at full opacity for M seconds
    - Fade out over K seconds
    """

    def __init__(
        self,
        fade_in_duration: float = 0.5,
        hold_duration: float = 1.0,
        fade_out_duration: float = 0.5,
    ):
        self.fade_in_duration = fade_in_duration
        self.fade_out_duration = fade_out_duration
        self.hold_duration = hold_duration
        self.total_duration = fade_in_duration + hold_duration + fade_out_duration

    def get_keyframes(self) -> list:
        """Get keyframe data for fade animation."""
        return [
            {"time": 0.0, "opacity": 0.0},
            {"time": self.fade_in_duration, "opacity": 1.0},
            {"time": self.fade_in_duration + self.hold_duration, "opacity": 1.0},
            {"time": self.total_duration, "opacity": 0.0},
        ]


class PanZoomTrack:
    """
    Pan and zoom camera animation track template.

    Provides smooth camera movement and zoom over time:
    - Pan across X/Y axes
    - Zoom in/out for focus effects
    - Smooth keyframe interpolation
    """

    def __init__(
        self, pan_distance: float = 2.0, zoom_factor: float = 2.0, duration: float = 3.0
    ):
        self.pan_distance = pan_distance
        self.zoom_factor = zoom_factor
        self.duration = duration

    def get_keyframes(self) -> list:
        """Get keyframe data for pan/zoom animation."""
        half_duration = self.duration / 2
        return [
            {"time": 0.0, "x": 0.0, "y": 0.0, "zoom": 1.0},
            {
                "time": half_duration,
                "x": self.pan_distance / 2,
                "y": self.pan_distance / 4,
                "zoom": self.zoom_factor / 1.5,
            },
            {
                "time": self.duration,
                "x": self.pan_distance,
                "y": self.pan_distance / 2,
                "zoom": self.zoom_factor,
            },
        ]


class ParticleEffectTrack:
    """
    Particle system animation track template.

    Provides a configurable particle emitter with:
    - Customizable particle count
    - Emission rate control
    - Lifetime and despawn behavior
    - Various spawn shapes
    """

    def __init__(
        self,
        particle_count: int = 100,
        emission_rate: int = 50,
        lifetime: float = 2.0,
        spawn_shape: str = "circle",
    ):
        self.particle_count = particle_count
        self.emission_rate = emission_rate  # particles/second
        self.lifetime = lifetime  # seconds before particle despawns
        self.spawn_shape = spawn_shape  # "circle", "square", "line", "random"

    def get_config(self) -> dict:
        """Get particle system configuration."""
        return {
            "count": self.particle_count,
            "rate": self.emission_rate,
            "lifetime": self.lifetime,
            "shape": self.spawn_shape,
            "physics": {"gravity": -9.8, "damping": 0.95},
        }

    def get_preset(self, preset_name: str) -> dict:
        """Get particle effect preset."""
        presets = {
            "rain": {"emission_rate": 200, "lifetime": 4.0, "spawn_shape": "area"},
            "fireworks": {
                "emission_rate": 500,
                "lifetime": 1.0,
                "spawn_shape": "circle",
            },
            "snow": {"emission_rate": 50, "lifetime": 3.0, "spawn_shape": "area"},
        }
        return presets.get(preset_name, {})
