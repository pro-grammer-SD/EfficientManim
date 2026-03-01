"""
Color Palette Extension - UI Panel Extension for EfficientManim

Provides custom color palette panel for quick color selection.

Example usage:
    from core.extension_api import ExtensionAPI
    from core.extensions.color_palette import setup
    api = ExtensionAPI("color-palette")
    setup(api)
"""

# Extension metadata (declares required permissions)
EXTENSION_METADATA = {
    "name": "Color Palette",
    "author": "EfficientManim",
    "version": "1.0.0",
    "description": "Custom color palette manager with preset themes",
    "permissions": ["register_ui_panel"],
}


def setup(api):
    """
    Initialize color palette extension.

    Registers a custom UI panel for color selection.
    """
    # Register the custom color palette UI panel
    api.register_ui_panel(
        panel_name="Color Palettes",
        widget_class="core.extensions.color_palette.ColorPalettePanel",
        position="right",
    )

    return True


class ColorPalettePanel:
    """Custom UI panel for managing color palettes."""

    def __init__(self):
        self.palettes = {
            "Material": [
                "#f3f4f6",
                "#e5e7eb",
                "#d1d5db",
                "#9ca3af",
                "#6b7280",
                "#4b5563",
            ],
            "Dracula": [
                "#282a36",
                "#44475a",
                "#6272a4",
                "#8be9fd",
                "#50fa7b",
                "#ffb86c",
            ],
            "Solarized": [
                "#eee8d5",
                "#fdf6e3",
                "#839496",
                "#0087af",
                "#229999",
                "#cb4b16",
            ],
            "Nord": ["#2e3440", "#3b4252", "#434c5e", "#4c566a", "#d8dee9", "#eceff4"],
        }
        self.current_palette = "Material"

    def get_palette(self, name: str) -> list:
        """Get colors from palette by name."""
        return self.palettes.get(name, self.palettes["Material"])

    def add_palette(self, name: str, colors: list) -> bool:
        """Add custom palette."""
        if name not in self.palettes:
            self.palettes[name] = colors
            return True
        return False
