"""
Color Palette Extension - Dynamic Theme Colorizer for EfficientManim

Provides a custom color palette panel that dynamically updates the app theme.
Users can select colors and the entire app theme is instantly updated.

Example usage:
    from app.api.extension_api import ExtensionAPI
    api = ExtensionAPI("color-palette")
    setup(api)

The ColorPalettePanel widget will be added to the UI automatically
after the main application window is created, and will have the ability
to dynamically update the app theme in real-time.
"""

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QGridLayout,
    QComboBox,
    QScrollArea,
    QFrame,
    QApplication,
)
from PySide6.QtCore import Signal
from PySide6.QtGui import QColor
import logging

LOGGER = logging.getLogger("color_palette_extension")


# Extension metadata (declares required permissions)
EXTENSION_METADATA = {
    "name": "Color Palette",
    "author": "EfficientManim",
    "version": "1.0.0",
    "description": "Dynamic color palette manager with preset themes and real-time app theme updates",
    "permissions": ["register_ui_panel"],
}


# Global reference to the main window (set during initialization)
_main_window_ref = None


def set_main_window(main_window):
    """
    Set the main window reference for theme updates.
    Called during extension initialization.
    """
    global _main_window_ref
    _main_window_ref = main_window
    LOGGER.info("✓ Color Palette extension linked to main window")


def setup(api):
    """
    Initialize color palette extension.

    Registers the custom color palette UI panel with the main application.
    The panel becomes available immediately after the main window is created
    and has full ability to update the app theme dynamically.
    """
    # Register the widget class path (lazy-loaded later when main window exists)
    api.register_ui_panel(
        panel_name="Color Palettes",
        widget_class="app.extensions.color_palette.ColorPalettePanel",
        position="right",
    )

    return True


class ColorPalettePanel(QWidget):
    """
    Custom UI panel for managing color palettes and dynamic theme updates.

    Features:
    - Multiple color palettes (Material, Dracula, Solarized, Nord)
    - Quick color selection
    - Real-time app theme updates when colors are selected
    - Custom palette creation
    - Copy color hex codes
    """

    color_selected = Signal(str)  # Emits hex color code

    def __init__(self):
        super().__init__()
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
        self._setup_ui()

    def _setup_ui(self):
        """Build the panel UI."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Title
        title = QLabel("🎨 Color Palettes")
        title.setStyleSheet("font-size: 12px; font-weight: bold; color: #2c3e50;")
        main_layout.addWidget(title)

        # Palette selector
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Theme:"))
        self.combo = QComboBox()
        self.combo.addItems(list(self.palettes.keys()))
        self.combo.setCurrentText(self.current_palette)
        self.combo.currentTextChanged.connect(self._on_palette_changed)
        selector_layout.addWidget(self.combo)
        main_layout.addLayout(selector_layout)

        # Scrollable color grid
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("border: 1px solid #e0e0e0; border-radius: 4px;")

        grid_widget = QWidget()
        self.grid = QGridLayout(grid_widget)
        self.grid.setSpacing(6)
        self.grid.setContentsMargins(8, 8, 8, 8)

        scroll_area.setWidget(grid_widget)
        main_layout.addWidget(scroll_area, 1)

        # Populate initial colors
        self._populate_grid()

        # Info label
        self.info_label = QLabel("Select a color")
        self.info_label.setStyleSheet("font-size: 10px; color: #666;")
        main_layout.addWidget(self.info_label)

    def _populate_grid(self):
        """Populate the color grid based on current palette."""
        # Clear existing buttons
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Add color buttons
        colors = self.palettes[self.current_palette]
        cols = 3
        for idx, color_hex in enumerate(colors):
            row = idx // cols
            col = idx % cols

            btn = self._create_color_button(color_hex)
            self.grid.addWidget(btn, row, col)

        # Fill remaining cells
        total_cells = ((len(colors) - 1) // cols + 1) * cols
        for idx in range(len(colors), total_cells):
            row = idx // cols
            col = idx % cols
            spacer = QFrame()
            self.grid.addWidget(spacer, row, col)

    def _create_color_button(self, color_hex: str) -> QPushButton:
        """Create a clickable color button."""
        btn = QPushButton()
        btn.setFixedSize(60, 60)

        # Set background to the color
        btn.setStyleSheet(
            f"QPushButton {{"
            f"  background-color: {color_hex};"
            f"  border: 2px solid #ccc;"
            f"  border-radius: 4px;"
            f"  padding: 0px;"
            f"}}"
            f"QPushButton:hover {{"
            f"  border: 2px solid #333;"
            f"}}"
            f"QPushButton:pressed {{"
            f"  border: 3px solid #000;"
            f"}}"
        )

        btn.setToolTip(color_hex)
        btn.clicked.connect(lambda: self._on_color_selected(color_hex))

        return btn

    def _on_palette_changed(self, palette_name: str):
        """Handle palette selection change."""
        self.current_palette = palette_name
        self._populate_grid()
        self.info_label.setText(f"Switched to {palette_name} palette")

    def _on_color_selected(self, color_hex: str):
        """
        Handle color button click.
        Updates the app theme with the selected color as the primary color.
        Propagates to: QApplication stylesheet, main window, and all children.
        """
        # Copy to clipboard
        try:
            import subprocess

            process = subprocess.Popen(
                ["powershell", "-Command", f'Set-Clipboard -Value "{color_hex}"'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            process.communicate(timeout=2)
        except Exception:
            pass  # Clipboard copy failed, but color is still selected

        # ── UPDATE THE APP THEME ─────────────────────────────────────────────
        try:
            from app.theme.themes import THEME_MANAGER, LightTheme

            # 1. Mutate LightTheme class-level color attributes
            LightTheme.PRIMARY = color_hex
            LightTheme.PRIMARY_DARK = self._darken_color(color_hex)
            LightTheme.PRIMARY_LIGHT = self._lighten_color(color_hex)

            # 2. Bust the stylesheet cache so get_stylesheet() rebuilds with new colors
            THEME_MANAGER.reload_stylesheet()

            # 3. Retrieve freshly-generated stylesheet
            new_stylesheet = THEME_MANAGER.get_stylesheet()

            # 4a. Apply to whole application (catches all top-level windows)
            app = QApplication.instance()
            if app:
                app.setStyleSheet(new_stylesheet)

            # 4b. Also apply directly to main window reference for immediate effect
            global _main_window_ref
            if _main_window_ref is not None:
                _main_window_ref.setStyleSheet(new_stylesheet)

            LOGGER.info(f"✓ App theme updated with primary color: {color_hex}")
        except Exception as e:
            LOGGER.error(f"Failed to update theme: {e}", exc_info=True)

        self.color_selected.emit(color_hex)
        self.info_label.setText(f"✓ Theme Updated: {color_hex}")

    @staticmethod
    def _darken_color(hex_color: str, factor: float = 0.8) -> str:
        """Darken a color by the given factor (0-1)."""
        try:
            color = QColor(hex_color)
            color.setHsv(
                color.hue(),
                color.saturation(),
                int(color.value() * factor),
                color.alpha(),
            )
            return color.name()
        except Exception:
            return "#1d4ed8"  # Default dark blue

    @staticmethod
    def _lighten_color(hex_color: str, factor: float = 0.3) -> str:
        """Lighten a color by the given factor (0-1)."""
        try:
            color = QColor(hex_color)
            color.setHsv(
                color.hue(),
                int(color.saturation() * (1 - factor)),
                int(color.value() + (255 - color.value()) * factor),
                color.alpha(),
            )
            return color.name()
        except Exception:
            return "#dbeafe"  # Default light blue

    def get_palette(self, name: str) -> list:
        """Get colors from palette by name."""
        return self.palettes.get(name, self.palettes["Material"])

    def add_palette(self, name: str, colors: list) -> bool:
        """Add custom palette."""
        if name not in self.palettes:
            self.palettes[name] = colors
            self.combo.addItem(name)
            return True
        return False
