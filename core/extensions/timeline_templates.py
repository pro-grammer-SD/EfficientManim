"""
Timeline Templates Extension - Timeline Track Templates for EfficientManim

Provides pre-built timeline track templates for common animation patterns.

Example usage:
    from core.extension_api import ExtensionAPI
    from core.extensions.timeline_templates import setup
    api = ExtensionAPI("timeline-templates")
    setup(api)
"""

# Extension metadata (declares required permissions)
EXTENSION_METADATA = {
    "name": "Timeline Templates",
    "author": "EfficientManim",
    "version": "1.0.0",
    "description": "Pre-built timeline track templates for common animation patterns",
    "permissions": ["register_timeline_track"],
}


def setup(api):
    """
    Initialize timeline templates extension.

    Registers timeline track template generators.
    """
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

    return True


class FadeTransitionTrack:
    """Fade in/out transition timeline track template."""

    def __init__(self):
        self.fade_in_duration = 0.5
        self.fade_out_duration = 0.5
        self.hold_duration = 1.0
        self.total_duration = 2.0

    def get_keyframes(self) -> list:
        """Get keyframe data for fade animation."""
        return [
            {"time": 0.0, "opacity": 0.0},
            {"time": self.fade_in_duration, "opacity": 1.0},
            {"time": self.fade_in_duration + self.hold_duration, "opacity": 1.0},
            {"time": self.total_duration, "opacity": 0.0},
        ]


class PanZoomTrack:
    """Pan and zoom camera animation track template."""

    def __init__(self):
        self.pan_speed = 1.0
        self.zoom_speed = 0.5
        self.duration = 3.0

    def get_keyframes(self) -> list:
        """Get keyframe data for pan/zoom animation."""
        return [
            {"time": 0.0, "x": 0.0, "y": 0.0, "zoom": 1.0},
            {"time": 1.5, "x": 1.0, "y": 0.5, "zoom": 1.5},
            {"time": 3.0, "x": 2.0, "y": 1.0, "zoom": 2.0},
        ]


class ParticleEffectTrack:
    """Particle system animation track template."""

    def __init__(self):
        self.particle_count = 100
        self.emission_rate = 50
        self.lifetime = 2.0
        self.spawn_shape = "circle"

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
