"""
EfficientManim - Light Mode Only Theme System
Simplified, deterministic styling for light mode.
No theme switching. No complexity. One clean design.
"""

from PySide6.QtCore import Signal, QObject


class LightTheme:
    """Light mode color palette - Single source of truth."""

    # Primary Colors
    PRIMARY = "#2563eb"
    PRIMARY_DARK = "#1d4ed8"
    PRIMARY_LIGHT = "#dbeafe"

    # Semantic Colors
    SUCCESS = "#10b981"
    SUCCESS_DARK = "#059669"
    WARNING = "#f59e0b"
    WARNING_DARK = "#d97706"
    ERROR = "#ef4444"
    ERROR_DARK = "#dc2626"

    # Grayscale
    BG_PRIMARY = "#ffffff"  # Main background
    BG_SECONDARY = "#f9fafb"  # Card backgrounds
    BG_TERTIARY = "#f3f4f6"  # Tertiary backgrounds
    BG_HOVER = "#e5e7eb"  # Hover state backgrounds

    # Text Colors
    TEXT_PRIMARY = "#1f2937"
    TEXT_SECONDARY = "#6b7280"
    TEXT_TERTIARY = "#9ca3af"

    # Borders
    BORDER_LIGHT = "#e5e7eb"
    BORDER_DEFAULT = "#d1d5db"
    BORDER_DARK = "#9ca3af"

    # Component Specific
    CANVAS_BG = "#f5f5f5"
    NODE_BG = "#ffffff"
    NODE_BORDER = "#d1d5db"
    SCROLLBAR = "#d1d5db"
    SCROLLBAR_HOVER = "#9ca3af"

    @staticmethod
    def get_color(color_name: str) -> str:
        """Get color by name. Returns as hex string."""
        if hasattr(LightTheme, color_name):
            return getattr(LightTheme, color_name)
        return LightTheme.BG_PRIMARY


class ThemeManager(QObject):
    """
    Unified light-mode theme manager.
    Provides stylesheet generation and color access.
    NO THEME SWITCHING - Light mode only.
    """

    theme_changed = Signal()  # Emitted on stylesheet reload

    def __init__(self):
        super().__init__()
        self.current_mode = "light"  # Always light
        self._stylesheet_cache = None

    def get_stylesheet(self) -> str:
        """Get the complete light-mode stylesheet."""
        if self._stylesheet_cache is not None:
            return self._stylesheet_cache

        self._stylesheet_cache = LIGHT_STYLESHEET
        return self._stylesheet_cache

    def get_color(self, color_name: str) -> str:
        """Get color from light theme palette."""
        return LightTheme.get_color(color_name)

    def reload_stylesheet(self):
        """Reload stylesheet (clear cache)."""
        self._stylesheet_cache = None
        self.theme_changed.emit()

    # Backwards compatibility attributes
    @property
    def current_theme(self):
        """Always returns light for backwards compatibility."""
        return "light"

    def set_theme(self, mode: object = None) -> None:
        """No-op — application is strictly light mode."""
        pass


# ═══════════════════════════════════════════════════════════════════
# SINGLETON THEME MANAGER INSTANCE
# ═══════════════════════════════════════════════════════════════════
# This is the global instance used by all modules
THEME_MANAGER = ThemeManager()


# ═══════════════════════════════════════════════════════════════════
# LIGHT MODE STYLESHEET - COMPREHENSIVE & CLEAN
# Complete styling for all widgets in professional light design
# ═══════════════════════════════════════════════════════════════════

LIGHT_STYLESHEET = """
/* ═══════════════════════════════════════════════════════════════════
   EFFICIENT MANIM - LIGHT MODE STYLESHEET
   Clean, minimal, professional light design for all UI elements
   ═══════════════════════════════════════════════════════════════════ */

/* ──────────────────────────────────────────────────────────────────
   BASE STYLING - Window, Widget, Frame
   ────────────────────────────────────────────────────────────────── */
QMainWindow, QMainWindow * {
    background-color: #ffffff;
    color: #1f2937;
}

QWidget {
    background-color: transparent;
    color: #1f2937;
    font-family: "Segoe UI", "San Francisco", sans-serif;
    font-size: 13px;
}

QLabel {
    color: #1f2937;
    background-color: transparent;
}

QFrame {
    background-color: #ffffff;
    color: #1f2937;
    border: none;
}

QFrame#card, QFrame#node {
    background-color: #f9fafb;
    border: 1px solid #d1d5db;
    border-radius: 6px;
}

QGraphicsView {
    background-color: #f5f5f5;
    border: 1px solid #d1d5db;
    border-radius: 4px;
    outline: none;
}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QPlainTextEdit {
    background-color: #ffffff;
    color: #1f2937;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    padding: 6px 8px;
    selection-background-color: #dbeafe;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, 
QTextEdit:focus, QPlainTextEdit:focus {
    border: 1px solid #2563eb;
    outline: none;
    background-color: #f0f9ff;
}

QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {
    border: 1px solid #9ca3af;
}

QComboBox::drop-down {
    background-color: #ffffff;
    border: none;
    width: 20px;
}

QComboBox::down-arrow {
    image: none;
    background-color: #dbeafe;
    border-radius: 3px;
}

QComboBox QAbstractItemView {
    background-color: #ffffff;
    color: #1f2937;
    border: 1px solid #d1d5db;
    border-radius: 4px;
    selection-background-color: #dbeafe;
    selection-color: #1f2937;
    padding: 2px;
}

QPushButton {
    background-color: #2563eb;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 500;
    outline: none;
}

QPushButton:hover {
    background-color: #1d4ed8;
}

QPushButton:pressed {
    background-color: #1e40af;
}

QPushButton:disabled {
    background-color: #e5e7eb;
    color: #9ca3af;
}

QPushButton#secondary {
    background-color: #f3f4f6;
    color: #1f2937;
    border: 1px solid #d1d5db;
}

QPushButton#secondary:hover {
    background-color: #e5e7eb;
    border: 1px solid #9ca3af;
}

QPushButton#danger {
    background-color: #ef4444;
}

QPushButton#danger:hover {
    background-color: #dc2626;
}

QPushButton#success {
    background-color: #10b981;
}

QPushButton#success:hover {
    background-color: #059669;
}

QTreeWidget, QListWidget, QTableWidget {
    background-color: #ffffff;
    color: #1f2937;
    border: 1px solid #d1d5db;
    border-radius: 6px;
}

QTreeWidget::item:selected, QListWidget::item:selected {
    background-color: #dbeafe;
    color: #1f2937;
}

QTreeWidget::item:hover, QListWidget::item:hover {
    background-color: #f3f4f6;
}

QScrollBar:vertical {
    background-color: #f9fafb;
    width: 10px;
    border: none;
}

QScrollBar::handle:vertical {
    background-color: #d1d5db;
    border-radius: 5px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background-color: #9ca3af;
}

QScrollBar:horizontal {
    background-color: #f9fafb;
    height: 10px;
    border: none;
}

QScrollBar::handle:horizontal {
    background-color: #d1d5db;
    border-radius: 5px;
    min-width: 20px;
}

QScrollBar::handle:horizontal:hover {
    background-color: #9ca3af;
}

QTabWidget::pane {
    border: 1px solid #d1d5db;
    background-color: #ffffff;
    border-radius: 6px;
}

QTabBar::tab {
    background-color: #f3f4f6;
    border: 1px solid #d1d5db;
    padding: 8px 12px;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    color: #6b7280;
    font-weight: 500;
}

QTabBar::tab:selected {
    background-color: #ffffff;
    border-bottom: 2px solid #2563eb;
    color: #2563eb;
}

QTabBar::tab:hover:!selected {
    background-color: #e5e7eb;
}

QMenuBar {
    background-color: #ffffff;
    color: #1f2937;
    border-bottom: 1px solid #d1d5db;
}

QMenuBar::item:selected {
    background-color: #f3f4f6;
}

QMenu {
    background-color: #ffffff;
    color: #1f2937;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    padding: 4px 0;
}

QMenu::item:selected {
    background-color: #dbeafe;
}

QCheckBox, QRadioButton {
    color: #1f2937;
    spacing: 6px;
}

QCheckBox::indicator, QRadioButton::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #d1d5db;
    border-radius: 3px;
    background-color: #ffffff;
}

QCheckBox::indicator:checked, QRadioButton::indicator:checked {
    background-color: #2563eb;
    border: 1px solid #2563eb;
}

QSlider::groove:horizontal {
    height: 4px;
    background: #e5e7eb;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #2563eb;
    width: 16px;
    margin: -6px 0;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background: #1d4ed8;
}

QSplitter::handle {
    background: #e5e7eb;
    width: 3px;
    height: 3px;
}

QSplitter::handle:hover {
    background: #9ca3af;
}

QGroupBox {
    border: 1px solid #d1d5db;
    border-radius: 6px;
    margin-top: 8px;
    padding-top: 8px;
    font-weight: 600;
    color: #1f2937;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
}

QHeaderView::section {
    background-color: #f3f4f6;
    color: #1f2937;
    padding: 4px;
    border: none;
    border-bottom: 1px solid #d1d5db;
    font-weight: 500;
}

QDockWidget {
    background-color: #ffffff;
    color: #1f2937;
    titlebar-close-icon: url(none);
    border: none;
}

QDockWidget::title {
    background-color: #f3f4f6;
    padding: 6px;
    border-bottom: 1px solid #d1d5db;
    color: #1f2937;
}

/* Output Monitor, Progress Bar, Timer, and Status Components */
QProgressBar {
    background-color: #e5e7eb;
    border: 1px solid #d1d5db;
    border-radius: 4px;
    text-align: center;
    color: #1f2937;
    height: 20px;
}

QProgressBar::chunk {
    background-color: #2563eb;
    border-radius: 3px;
}

QLabel#timer, QLabel#status {
    color: #1f2937;
    background-color: transparent;
}

QPlainTextEdit#outputMonitor, QTextEdit#outputMonitor {
    background-color: #ffffff;
    color: #1f2937;
    border: 1px solid #d1d5db;
    border-radius: 4px;
    padding: 4px;
}

/* AI Tab Components - No White Backgrounds in Light Mode */
QWidget#AIPanel {
    background-color: #ffffff;
}

QFrame#AIHeader {
    background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%);
    border-bottom: 2px solid #0d47a1;
    border-radius: 4px;
}

QFrame#AIPrompt, QFrame#AIResponse, QFrame#AINodePreview {
    background-color: #f9fafb;
    border: 1px solid #d1d5db;
    border-radius: 4px;
}

QLabel#AppTitle {
    font-size: 18px;
    font-weight: bold;
    color: #1f2937;
}

QLabel#AppSubtitle {
    font-size: 12px;
    color: #6b7280;
}

QLabel#VersionLabel {
    font-size: 10px;
    color: #9ca3af;
}

QLabel#GreetLabel {
    font-size: 22px;
    font-weight: bold;
    color: #1f2937;
}

QLabel#SectionTitle {
    font-size: 13px;
    font-weight: bold;
    color: #1f2937;
}

QLabel#FeatureTitle {
    font-size: 11px;
    font-weight: 600;
    color: #1f2937;
}

QLabel#FeatureDesc {
    font-size: 10px;
    color: #6b7280;
}

QLabel#EmptyLabel {
    font-size: 11px;
    color: #6b7280;
}

QFrame#Sidebar {
    background-color: #ffffff;
    border-right: 1px solid #e5e7eb;
}

QFrame#ContentArea {
    background-color: #ffffff;
}

QFrame#FeatureCard {
    background-color: #f9fafb;
    border: 1px solid #d1d5db;
    border-radius: 8px;
}

QListWidget#RecentsList {
    background-color: #ffffff;
    border: 1px solid #d1d5db;
    border-radius: 6px;
}

QPushButton#SidebarBtn {
    background-color: transparent;
    color: #1f2937;
    border: none;
    border-radius: 6px;
    padding: 10px 12px;
    text-align: left;
}

QPushButton#SidebarBtn:hover {
    background-color: #f3f4f6;
}

QPushButton#SidebarBtn:pressed {
    background-color: #e5e7eb;
}

/* Dialog and Popup */
QDialog {
    background-color: #ffffff;
    color: #1f2937;
}

/* Scrollarea */
QScrollArea {
    background-color: transparent;
    border: none;
}

QScrollArea > QWidget > QWidget {
    background-color: transparent;
}

QScrollArea QWidget {
    background-color: transparent;
}

/* Status bar and Toolbar */
QStatusBar {
    background-color: #f9fafb;
    color: #6b7280;
    border-top: 1px solid #d1d5db;
}

QToolBar {
    background-color: #ffffff;
    border: none;
    border-bottom: 1px solid #d1d5db;
}

/* End of Light Mode Stylesheet */
"""


