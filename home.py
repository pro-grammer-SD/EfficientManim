"""
EfficientManim - Home Screen
Entry point for the application with clean light-mode UI.
Rebuilt with proper layout system: no overlapping, no clipping, responsive.
"""

import sys
import json
from pathlib import Path

# ── Qt imports ────────────────────────────────────────────────────────────────
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QFrame,
    QFileDialog,
    QGridLayout,
)
from PySide6.QtCore import Qt, QSize, QTimer
from PySide6.QtGui import QIcon, QFont

# ── Constants ─────────────────────────────────────────────────────────────────
APP_DATA_DIR = Path.home() / ".efficientmanim"
RECENTS_FILE = APP_DATA_DIR / "recents.json"
USER_DATA_FILE = APP_DATA_DIR / "userdata.json"
ICON_PATH = Path(__file__).parent / "icon" / "icon.ico"

APP_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ── Recent Projects ────────────────────────────────────────────────────────────
def load_recents() -> list[str]:
    """Load recent project paths from disk."""
    try:
        if RECENTS_FILE.exists():
            data = json.loads(RECENTS_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                # Filter out projects that no longer exist
                return [p for p in data if Path(p).exists()]
    except Exception:
        pass
    return []


def save_recents(paths: list[str]) -> None:
    """Save recent project paths to disk (max 10)."""
    try:
        RECENTS_FILE.write_text(json.dumps(paths[:10], indent=2), encoding="utf-8")
    except Exception:
        pass


def add_recent(path: str) -> None:
    """Add a project to the recents list."""
    recents = load_recents()
    if path in recents:
        recents.remove(path)
    recents.insert(0, path)
    save_recents(recents[:10])


# ── Home Screen ────────────────────────────────────────────────────────────────
class HomeScreen(QMainWindow):
    """Modern home screen for EfficientManim."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("EfficientManim")
        self.setMinimumSize(900, 600)
        self.resize(1000, 680)

        # Set window icon
        if ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(ICON_PATH)))

        self._build_ui()
        self._apply_theme()
        self._refresh_recents()

    # ── UI Construction ────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Left Sidebar ───────────────────────────────────────────────────────
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(280)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(24, 32, 24, 24)
        sidebar_layout.setSpacing(16)

        # Logo / title using FontManager
        logo_lbl = QLabel("⚡")
        font = QFont()
        font.setPointSize(40)
        font.setBold(True)
        logo_lbl.setFont(font)
        logo_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sidebar_layout.addWidget(logo_lbl)

        title_lbl = QLabel("EfficientManim")
        font = QFont()
        font.setPointSize(18)
        font.setBold(True)
        title_lbl.setFont(font)
        title_lbl.setObjectName("AppTitle")
        sidebar_layout.addWidget(title_lbl)

        sub_lbl = QLabel("The Node-Based Manim IDE")
        font = QFont()
        font.setPointSize(11)
        sub_lbl.setFont(font)
        sub_lbl.setObjectName("AppSubtitle")
        sidebar_layout.addWidget(sub_lbl)

        sidebar_layout.addSpacing(24)

        # Action buttons
        for label, icon, slot in [
            ("New Project", "📄", self.new_project),
            ("Open Project…", "📂", self.open_project),
            ("Open Editor", "🎨", self.open_editor),
        ]:
            btn = QPushButton(f"  {icon}  {label}")
            btn.setObjectName("SidebarBtn")
            btn.setMinimumHeight(44)
            # system default font
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(slot)
            sidebar_layout.addWidget(btn)

        sidebar_layout.addStretch()

        # Version info
        ver_lbl = QLabel("v2.0.3  •  Production")
        font = QFont()
        font.setPointSize(9)
        ver_lbl.setFont(font)
        ver_lbl.setObjectName("VersionLabel")
        sidebar_layout.addWidget(ver_lbl)

        outer.addWidget(sidebar)

        # ── Right Content Area ─────────────────────────────────────────────────
        content = QFrame()
        content.setObjectName("ContentArea")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(32, 32, 32, 32)
        content_layout.setSpacing(20)

        # Greeting
        greet_lbl = QLabel("Welcome back 👋")
        font = QFont()
        font.setPointSize(22)
        font.setBold(True)
        greet_lbl.setFont(font)
        greet_lbl.setObjectName("GreetLabel")
        content_layout.addWidget(greet_lbl)

        # Recent projects section
        recent_title = QLabel("Recent Projects")
        font = QFont()
        font.setPointSize(13)
        font.setBold(True)
        recent_title.setFont(font)
        recent_title.setObjectName("SectionTitle")
        content_layout.addWidget(recent_title)

        self.recents_list = QListWidget()
        self.recents_list.setObjectName("RecentsList")
        self.recents_list.setSpacing(4)
        self.recents_list.setIconSize(QSize(32, 32))
        self.recents_list.itemDoubleClicked.connect(self._open_recent)
        self.recents_list.setMinimumHeight(200)
        self.recents_list.setMaximumHeight(260)
        content_layout.addWidget(self.recents_list)

        # Empty state label (shown when no recents)
        self.empty_lbl = QLabel(
            "No recent projects.\nClick 'New Project' or 'Open Project' to get started."
        )
        pass  # system default
        self.empty_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_lbl.setObjectName("EmptyLabel")
        self.empty_lbl.hide()
        content_layout.addWidget(self.empty_lbl)

        # Features row
        features_title = QLabel("What's New in v2.0")
        font = QFont()
        font.setPointSize(13)
        font.setBold(True)
        features_title.setFont(font)
        features_title.setObjectName("SectionTitle")
        content_layout.addWidget(features_title)

        features_grid = QGridLayout()
        features_grid.setSpacing(12)

        features = [
            (
                "🌿 Freshly Crafted Theme",
                "A refined visual theme designed for clarity, balance, and modern aesthetics.",
            ),
            ("🔖 VGroup Utility", "Group mobjects visually in the Elements pane"),
            ("📊 Usage Tracking", "Recents pane shows your most-used objects"),
            (
                "⌨️ Custom Keybindings",
                "Every action is remappable via the Keybindings panel",
            ),
            ("🐙 GitHub Snippets", "Load code snippets directly from any GitHub repo"),
            (
                "💡 Light Mode Design",
                "Clean, minimal, professional interface optimized for productivity",
            ),
            (
                "🤖 MCP Agent",
                "AI-powered Model Context Protocol agent for intelligent assistance",
            ),
            (
                "🎙️ Auto Voiceover",
                "Automatically generate voiceovers for your animations",
            ),
            (
                "🎨 Revamped Icon",
                "Fresh, professional icon design for the application",
            ),
            (
                "🛸 Extension System",
                "Build and integrate powerful extensions with EfficientManim's modular, permission-based architecture.",
            ),
            (
                "📘 Documentation Upgrade",
                "EfficientManim now includes comprehensive, production-ready documentation across core systems and extensions.",
            ),
            (
                "🧹 Codebase Refactor",
                "The project structure has been streamlined, redundant code removed, and architecture hardened for production stability.",
            ),
        ]

        for idx, (feat_title, feat_desc) in enumerate(features):
            card = QFrame()
            card.setObjectName("FeatureCard")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(14, 12, 14, 12)
            card_layout.setSpacing(8)

            t = QLabel(feat_title)
            font = QFont()
            font.setPointSize(11)
            font.setBold(True)
            t.setFont(font)
            t.setObjectName("FeatureTitle")
            card_layout.addWidget(t)

            d = QLabel(feat_desc)
            font = QFont()
            font.setPointSize(10)
            d.setFont(font)
            d.setWordWrap(True)
            d.setObjectName("FeatureDesc")
            card_layout.addWidget(d)

            features_grid.addWidget(card, idx // 3, idx % 3)

        content_layout.addLayout(features_grid)
        content_layout.addStretch()

        outer.addWidget(content, 1)

    # ── Theme ──────────────────────────────────────────────────────────────────
    def _apply_theme(self):
        """Apply light-mode stylesheet to home screen."""
        from app.theme.themes import THEME_MANAGER

        self.setStyleSheet(THEME_MANAGER.get_stylesheet())

    # ── Recent Projects ────────────────────────────────────────────────────────
    def _refresh_recents(self):
        self.recents_list.clear()
        recents = load_recents()

        if not recents:
            self.recents_list.hide()
            self.empty_lbl.show()
            return

        self.recents_list.show()
        self.empty_lbl.hide()

        for path in recents:
            p = Path(path)
            item = QListWidgetItem(f"  📁  {p.stem}  —  {str(p.parent)}")
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setToolTip(path)
            self.recents_list.addItem(item)

    def _open_recent(self, item: QListWidgetItem):
        path = item.data(Qt.ItemDataRole.UserRole)
        if path and Path(path).exists():
            self._launch_editor(path)
        else:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.warning(
                self, "File Not Found", f"Project file no longer exists:\n{path}"
            )
            # Remove from recents
            recents = load_recents()
            if path in recents:
                recents.remove(path)
                save_recents(recents)
            self._refresh_recents()

    # ── Actions ────────────────────────────────────────────────────────────────
    def new_project(self):
        """Create a new empty project and open the editor."""
        self._launch_editor(None)

    def open_project(self):
        """Show file dialog to open an existing .efp project."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open EfficientManim Project",
            str(Path.home()),
            "EfficientManim Project (*.efp);;All Files (*)",
        )
        if path:
            add_recent(path)
            self._launch_editor(path)

    def open_editor(self):
        """Open the editor without loading a project."""
        self._launch_editor(None)

    def _launch_editor(self, project_path: str | None):
        """Import and launch the main editor window."""
        try:
            # FIX: Use standard import to avoid re-running main.py multiple times
            try:
                import main
            except ImportError:
                # Fallback to dynamic load if not in path (rare)
                import importlib.util

                spec = importlib.util.spec_from_file_location(
                    "main", Path(__file__).parent / "main.py"
                )
                if spec is None or spec.loader is None:
                    raise ImportError("Cannot load main.py")
                if "main" not in sys.modules:
                    main = importlib.util.module_from_spec(spec)
                    sys.modules["main"] = main
                    spec.loader.exec_module(main)
                else:
                    main = sys.modules["main"]

            WindowClass = getattr(main, "EfficientManimWindow", None)
            if WindowClass is None:
                raise AttributeError("EfficientManimWindow not found in main.py")

            self._editor_window = WindowClass()

            # Re-show home screen when editor is destroyed
            def on_editor_close():
                self._refresh_recents()
                self.show()
                self._editor_window = None

            self._editor_window.destroyed.connect(on_editor_close)
            self._editor_window.show()

            # If a project path was given, open it after the window is shown
            if project_path and hasattr(self._editor_window, "open_project_from_path"):
                QTimer.singleShot(
                    200,
                    lambda: self._editor_window.open_project_from_path(project_path),
                )

            # Hide (not close) the home screen
            self.hide()

        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            import traceback

            traceback.print_exc()
            QMessageBox.critical(
                self,
                "Launch Error",
                f"Failed to launch editor:\n{type(e).__name__}: {e}",
            )


# ── Entry Point ────────────────────────────────────────────────────────────────
def main():
    app = QApplication.instance() or QApplication(sys.argv)

    # ── Apply Global Light-Mode Stylesheet ────────────────────────────────
    from app.theme.themes import THEME_MANAGER

    app.setStyleSheet(THEME_MANAGER.get_stylesheet())

    app.setApplicationName("EfficientManim")
    app.setOrganizationName("EfficientManim")

    if ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(ICON_PATH)))

    window = HomeScreen()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
