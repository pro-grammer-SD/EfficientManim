from __future__ import annotations
from PySide6.QtWidgets import QMainWindow
# -*- coding: utf-8 -*-

import json
import os
import shutil
import tempfile
import traceback
import uuid
import zipfile
import webbrowser
from datetime import datetime
from contextlib import nullcontext
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtGui import QBrush, QColor, QFont, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.config import APP_NAME, APP_VERSION, PROJECT_EXT, AppPaths, SETTINGS
from core.file_manager import ASSETS, USAGE_TRACKER, Asset, add_recent
from core.history_manager import HistoryManager, WireState
from core.themes import THEME_MANAGER
from graph.edge import WireItem
from graph.graph_editor import GraphScene, GraphView
from graph.layout import auto_layout_nodes as apply_auto_layout
from graph.node import NodeData, NodeItem, NodeType
from graph.node_factory import NodeFactory
from rendering.render_manager import RenderWorker, VideoRenderWorker
from ui.dialogs import KeyboardShortcutsDialog, SettingsDialog
from ui.menus import build_menus
from ui.panels.node_panel import (
    AINodeIntegrator,
    AIPanel,
    AssetsPanel,
    ElementsPanel,
    GitHubSnippetLoader,
    LatexEditorPanel,
    ManimClassBrowser,
    NodeSearchBar,
    PropertiesPanel,
    RecentsPanel,
    SceneOutlinerPanel,
    SnippetLibrary,
    VGroupPanel,
    VoiceoverPanel,
)
from ui.panels.history_panel import HistoryPanel
from ui.panels.render_panel import VideoOutputPanel, VideoRenderPanel
from ui.toolbar import QuickExportBar
from utils.helpers import TypeSafeParser, detect_scene_class, manim
from utils.logger import LOGGER
from utils.shortcuts import (
    KEYBINDINGS,
    KEYBINDINGS_AVAILABLE,
    initialize_default_keybindings,
)
from scene_explainer.ai_explainer import ExplainService
from scene_explainer.ui_panel import (
    ExplainPanel,
    LearningModeController,
    TeacherModeController,
)

try:
    from core.keybindings_panel import UnifiedKeybindingsPanel

    UNIFIED_KEYBINDINGS_AVAILABLE = True
except Exception:
    UNIFIED_KEYBINDINGS_AVAILABLE = False

    class UnifiedKeybindingsPanel(QWidget):
        pass


try:
    from core.mcp import MCPAgent as _MCPAgent

    MCP_AVAILABLE = True
except Exception:
    _MCPAgent = None  # type: ignore[assignment]
    MCP_AVAILABLE = False

# Audio handling
try:
    from pydub import AudioSegment

    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    AudioSegment = None


class EfficientManimWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.resize(1600, 1000)
        self.setStyleSheet(THEME_MANAGER.get_stylesheet())

        # CRITICAL FIX: Set icon with absolute path before showing window
        icon_path = Path(__file__).resolve().parent.parent / "icon" / "icon.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path.absolute())))
        else:
            # Fallback to relative path if absolute doesn't work
            self.setWindowIcon(QIcon("icon/icon.ico"))

        self.nodes = {}
        self.project_path = None
        self.history_manager: HistoryManager | None = None
        self.project_modified = False
        self.is_ai_generated_code = False
        self._skip_quit_prompt = False

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
        build_menus(self)
        self._init_history()
        self._init_explain_panel()
        self.apply_theme()

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
        # Strong reference set — prevents Python GC from destroying active workers
        self._active_render_workers = set()

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
        self.scene.main_window = self
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
        self.quick_export_bar.undo_requested.connect(self.undo_action)
        self.quick_export_bar.redo_requested.connect(self.redo_action)
        self.quick_export_bar.explain_requested.connect(self.explain_current_context)
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
        self.panel_props = PropertiesPanel(self)
        self.panel_props.node_updated.connect(self.mark_modified)
        self.panel_props.node_updated.connect(self.on_node_changed)

        self.panel_outliner = SceneOutlinerPanel(self)
        self.panel_history = HistoryPanel()

        self.panel_elems = ElementsPanel()
        self.panel_elems.add_requested.connect(self.add_node_center)
        self.panel_elems.add_structural_requested.connect(self._add_structural_node)

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

        self.tabs_top.addTab(self.panel_elems, "Elements")
        self.tabs_top.addTab(self.panel_recents, "Recents")
        self.tabs_top.addTab(self.panel_class_browser, "Classes")
        self.tabs_top.addTab(self.panel_vgroup, "VGroups")
        self.tabs_top.addTab(self.panel_outliner, "Outliner")
        self.tabs_top.addTab(self.panel_history, "History")
        self.tabs_top.addTab(self.panel_props, "Properties")
        self.tabs_top.addTab(self.panel_assets, "Assets")
        self.tabs_top.addTab(self.panel_latex, "LaTeX")
        self.tabs_top.addTab(self.panel_snippets, "Snippets")
        self.tabs_top.addTab(self.panel_github, "GitHub")
        self.tabs_top.addTab(self.panel_voice, "Voiceover")
        self.tabs_top.addTab(self.panel_video, "Video")
        right.addWidget(self.tabs_top)

        # ── AI Panel: permanent left dock (cannot float or be undocked) ──
        self.ai_dock = QDockWidget("AI Assistant", self)
        self.ai_dock.setObjectName("AIDockWidget")
        # Lock the dock: no floating, no closing, no moving
        self.ai_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.ai_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea)
        self.ai_dock.setWidget(self.panel_ai)
        self.ai_dock.setMinimumWidth(240)
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

        self.tabs_bot.addTab(prev_widget, "Preview")
        self.tabs_bot.addTab(self.code_view, "Code")
        self.tabs_bot.addTab(self.logs, "Logs")
        right.addWidget(self.tabs_bot)

        main.addWidget(right)
        main.setSizes([1000, 600])

    # ── History System ───────────────────────────────────────────────────────

    def _init_history(self) -> None:
        """Initialize history manager, hooks, and UI bindings."""
        self.history_manager = HistoryManager(
            snapshot_provider=self._history_snapshot_provider,
            apply_snapshot=self._history_apply_snapshot,
            apply_node_state=self._history_apply_node_state,
            scene_name_provider=lambda: self._current_scene_name,
            max_history=None,
        )
        self.history_manager.state_changed.connect(self._update_history_actions)
        self.history_manager.history_changed.connect(self._refresh_history_panel)
        self.history_manager.snapshot_applied.connect(self._on_history_applied)
        if hasattr(self, "panel_history"):
            try:
                self.panel_history.set_manager(self.history_manager)
            except Exception:
                pass

        # Move history debounce
        self._move_debounce = QTimer()
        self._move_debounce.setSingleShot(True)
        self._move_debounce.timeout.connect(self._commit_move_history)
        self._pending_move_nodes: set[str] = set()

        # Initialize root snapshot
        self.history_manager.reset(description="Initial State")

    def _init_explain_panel(self) -> None:
        """Initialize Explain Panel and learning/teacher mode controllers."""
        try:
            self.explain_service = ExplainService(self)
            self.panel_explain = ExplainPanel(self)
            self.explain_dock = QDockWidget("Explain Panel", self)
            self.explain_dock.setObjectName("ExplainDockWidget")
            self.explain_dock.setFeatures(
                QDockWidget.DockWidgetFeature.NoDockWidgetFeatures
            )
            self.explain_dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
            self.explain_dock.setWidget(self.panel_explain)
            self.explain_dock.setMinimumWidth(240)
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.explain_dock)
            self.learning_mode_controller = LearningModeController(
                self, self.panel_explain
            )
            self.teacher_mode_controller = TeacherModeController(
                self, self.panel_explain
            )
        except Exception as exc:
            LOGGER.error(f"Explain panel init failed: {exc}")

    def _history_snapshot_provider(self):
        """Return raw node/wire state for snapshotting."""
        nodes_data: dict[str, dict] = {}
        for nid, node in self.nodes.items():
            # Ensure live position is captured
            try:
                node.data.pos_x = float(node.x())
                node.data.pos_y = float(node.y())
            except Exception:
                pass
            nodes_data[nid] = node.data.to_dict()

        wires: list[WireState] = []
        for item in list(self.scene.items()):
            if isinstance(item, WireItem):
                try:
                    from_id = item.start_socket.parentItem().data.id
                    to_id = item.end_socket.parentItem().data.id
                    wires.append(WireState(item.wire_id, from_id, to_id))
                except Exception:
                    pass
        return nodes_data, wires

    def _history_apply_snapshot(self, snapshot):
        """Restore full graph from a snapshot (no history capture)."""
        if self.history_manager is None:
            return
        with self.history_manager.suspend():
            # Clean existing nodes and workers
            for node_id in list(self.nodes.keys()):
                self._cleanup_preview_worker(node_id)
            self.nodes.clear()
            self.scene.clear()

            # Rebuild nodes
            for nid, node_state in snapshot.nodes.items():
                data = NodeData.from_dict(node_state.data)
                node = NodeItem(data)
                try:
                    node._window = self
                except Exception:
                    pass
                node.setPos(data.pos_x, data.pos_y)
                self.scene.addItem(node)
                self.nodes[data.id] = node

            # Rebuild wires
            for w in snapshot.wires:
                self.add_wire_by_ids(w.from_node, w.to_node, wire_id=w.wire_id)

            # Reset panels
            self.panel_props.set_node(None)
            if hasattr(self, "panel_outliner"):
                try:
                    self.panel_outliner.refresh_list()
                except Exception:
                    pass

            # Update preview
            self.preview_lbl.clear()
            self.preview_lbl.setPixmap(QPixmap())
            self.preview_lbl.setText("No Selection")

    def _history_apply_node_state(self, node_id: str, node_data: dict | None, wires):
        """Restore a single node and its wires (no history capture)."""
        if self.history_manager is None:
            return
        with self.history_manager.suspend():
            # Remove existing wires connected to the node
            for item in list(self.scene.items()):
                if isinstance(item, WireItem):
                    try:
                        a = item.start_socket.parentItem().data.id
                        b = item.end_socket.parentItem().data.id
                        if node_id in (a, b):
                            self.remove_wire(item, record_history=False)
                    except Exception:
                        pass

            # Delete node if no data
            if node_data is None:
                node_item = self.nodes.get(node_id)
                if node_item:
                    self.remove_node(node_item, record_history=False)
                return

            # Create or update node
            node_item = self.nodes.get(node_id)
            new_data = NodeData.from_dict(node_data)
            if node_item is None:
                node_item = NodeItem(new_data)
                try:
                    node_item._window = self
                except Exception:
                    pass
                node_item.setPos(new_data.pos_x, new_data.pos_y)
                self.scene.addItem(node_item)
                self.nodes[new_data.id] = node_item
            else:
                node_item.data = new_data
                node_item.setPos(new_data.pos_x, new_data.pos_y)
                node_item.update()

            # Rebuild wires for this node
            for w in wires:
                self.add_wire_by_ids(w.from_node, w.to_node, wire_id=w.wire_id)

    def _on_history_applied(self, snapshot):
        """Post-undo/redo UI refresh."""
        self.is_ai_generated_code = False
        self.compile_graph()
        self.mark_modified()

    def _update_history_actions(self, can_undo: bool, can_redo: bool):
        """Enable/disable undo/redo UI actions."""
        if hasattr(self, "_undo_action"):
            self._undo_action.setEnabled(can_undo)
        if hasattr(self, "_redo_action"):
            self._redo_action.setEnabled(can_redo)
        if hasattr(self, "quick_export_bar"):
            try:
                self.quick_export_bar.set_history_enabled(can_undo, can_redo)
            except Exception:
                pass

    def _refresh_history_panel(self):
        if hasattr(self, "panel_history"):
            try:
                self.panel_history.refresh()
            except Exception:
                pass

    def _commit_move_history(self):
        if not self._pending_move_nodes:
            return
        if self.history_manager is None:
            return
        ids = sorted(self._pending_move_nodes)
        self._pending_move_nodes.clear()
        desc = "Move Node" if len(ids) == 1 else "Move Nodes"
        merge_key = f"move:{','.join(ids)}" if len(ids) == 1 else "move:multi"
        self.history_manager.capture(
            desc, merge_key=merge_key, metadata={"affected_nodes": ids}
        )

    def new_project(self, force: bool = False):
        """Create a new project."""
        if not force:
            reply = QMessageBox.question(
                self,
                "New Project",
                "Clear current project?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        self.clear_scene(force=True)

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
        self.save_project_to(path)

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
                f"MCP Agent is alive.\n\n"
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
        btn_copy = QPushButton("Copy JSON")
        btn_copy.clicked.connect(
            lambda: (
                QApplication.clipboard().setText(ctx_text),
                btn_copy.setText("Copied"),
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
        """Show every MCP command executed in this run."""
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
        btn_copy = QPushButton("Copy Log")
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
                self, f"MCP — {command}", f"Command succeeded.\n\n{result.data}"
            )
        else:
            LOGGER.error(f"MCP [{command}] FAILED: {result.error}")
            QMessageBox.critical(self, f"MCP — {command} failed", f"{result.error}")

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

        webbrowser.open("https://docs.manim.community/en/stable/")

    def _open_manim_gallery(self):
        """Open Manim example gallery in browser."""

        webbrowser.open("https://docs.manim.community/en/stable/examples.html")

    def apply_theme(self) -> None:
        """Apply light-mode stylesheet to all widgets."""
        self.setStyleSheet(THEME_MANAGER.get_stylesheet())
        if hasattr(self, "scene"):
            self.scene.setBackgroundBrush(QBrush(QColor("#f4f6f7")))

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
                    add_recent(str(new_path))
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
                try:
                    node._window = self
                except Exception:
                    pass
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
        if self.history_manager:
            self.history_manager.reset(description=f"Scene {name}")

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
            type_upper = node_type_str.upper()
            type_str = (
                type_upper
                if type_upper in ("PLAY", "WAIT", "VGROUP")
                else ("MOBJECT" if type_upper in ("MOBJECT", "MOB") else "ANIMATION")
            )
            pos = (50, 50 + len(self.nodes) * 120)
            self.add_node(type_str, class_name, pos=pos, name=class_name)
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
        if not selected:
            return
        if self.history_manager:
            self.history_manager.begin_group("Create VGroup")
        ids = [item.data.id for item in selected]
        # Use VGroupPanel's internal create logic
        if hasattr(self, "panel_vgroup"):
            self.panel_vgroup._groups[
                f"vgroup_{len(self.panel_vgroup._groups) + 1}"
            ] = ids
            name = f"vgroup_{len(self.panel_vgroup._groups)}"
            self.panel_vgroup._meta[name] = {
                "source": "canvas",
                "members": [self.nodes[i].data.name for i in ids if i in self.nodes],
            }
            self.panel_vgroup._refresh_tree()
            self.add_vgroup_node(group_name=name, member_ids=ids)
        if self.history_manager:
            self.history_manager.end_group()
        self.statusBar().showMessage(f"VGroup created with {len(ids)} nodes", 2000)

    # ── Explain Panel Integration ──────────────────────────────────────────

    def _show_explain_panel(self) -> None:
        if hasattr(self, "explain_dock"):
            try:
                self.explain_dock.show()
            except Exception:
                pass

    def explain_current_context(self) -> None:
        if not hasattr(self, "panel_explain"):
            return
        selected = [
            item for item in self.scene.selectedItems() if isinstance(item, NodeItem)
        ]
        if selected:
            anim = next(
                (n for n in selected if n.data.type == NodeType.ANIMATION), None
            )
            if anim is not None:
                self.panel_explain.explain_selected_animation(anim.data.id)
            else:
                self.panel_explain.explain_selected_nodes([n.data.id for n in selected])
        else:
            self.panel_explain.explain_scene()
        self._show_explain_panel()

    def explain_selected_objects(self) -> None:
        if not hasattr(self, "panel_explain"):
            return
        selected = [
            item
            for item in self.scene.selectedItems()
            if isinstance(item, NodeItem) and item.data.type == NodeType.MOBJECT
        ]
        if not selected:
            return
        self.panel_explain.explain_selected_objects([n.data.id for n in selected])
        self._show_explain_panel()

    def explain_selected_nodes(self) -> None:
        if not hasattr(self, "panel_explain"):
            return
        selected = [
            item for item in self.scene.selectedItems() if isinstance(item, NodeItem)
        ]
        if not selected:
            return
        self.panel_explain.explain_selected_nodes([n.data.id for n in selected])
        self._show_explain_panel()

    def explain_selected_animation(self) -> None:
        if not hasattr(self, "panel_explain"):
            return
        selected = [
            item
            for item in self.scene.selectedItems()
            if isinstance(item, NodeItem) and item.data.type == NodeType.ANIMATION
        ]
        if not selected:
            return
        self.panel_explain.explain_selected_animation(selected[0].data.id)
        self._show_explain_panel()

    def generate_lesson_notes(self) -> None:
        if not hasattr(self, "panel_explain"):
            return
        self.panel_explain.generate_lesson_notes()
        self._show_explain_panel()

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
            # These are created in build_menus()
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
        if getattr(self, "_skip_quit_prompt", False):
            event.accept()
            return
        should_close = self.request_app_quit()
        if should_close:
            event.accept()
        else:
            event.ignore()

    def request_app_quit(self) -> bool:
        """Attempt to close the application with the standard save prompt."""
        # FIX: Clean up all preview workers before closing
        for node_id in list(self.nodes.keys()):
            self._cleanup_preview_worker(node_id)

        # Clear preview display
        self.preview_lbl.clear()
        self.preview_lbl.setPixmap(QPixmap())

        # Clean up temp files
        AppPaths.force_cleanup_old_files(age_seconds=0)

        should_close = False
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
                    return False
                should_close = True
            elif reply == QMessageBox.StandardButton.Discard:
                should_close = True
            else:
                return False
        else:
            should_close = True

        if should_close:
            try:
                if (
                    hasattr(self, "explain_service")
                    and self.explain_service is not None
                ):
                    self.explain_service.cancel_all()
            except Exception:
                pass
            try:
                panel = getattr(self, "panel_ai", None)
                if panel is not None:
                    for attr in ("worker", "_auto_vo_worker"):
                        w = getattr(panel, attr, None)
                        if w is not None and hasattr(w, "isRunning") and w.isRunning():
                            try:
                                w.requestInterruption()
                                w.quit()
                                w.wait(500)
                            except Exception:
                                pass
            except Exception:
                pass
        return should_close

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
        if self.history_manager and self.history_manager.undo():
            LOGGER.info("Undo executed")
        else:
            LOGGER.warn("Nothing to undo")

    def redo_action(self):
        """Redo last undone action."""
        if self.history_manager and self.history_manager.redo():
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

    def clear_scene(self, force: bool = False):
        """Clear all nodes and wires with proper resource cleanup."""
        if not force:
            reply = QMessageBox.question(
                self,
                "Clear Scene",
                "Delete all nodes and wires?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        if self.history_manager:
            self.history_manager.begin_group("Clear Scene")
        # FIX: Clean up render workers before clearing nodes
        for node_id in list(self.nodes.keys()):
            self._cleanup_preview_worker(node_id)

        self.nodes.clear()
        self.scene.clear()
        # FIX: Clear preview display
        self.preview_lbl.clear()
        self.preview_lbl.setPixmap(QPixmap())  # Release pixmap memory
        self.preview_lbl.setText("No Selection")
        if self.history_manager:
            self.history_manager.mark_group_dirty()
            self.history_manager.end_group()
        self.compile_graph()
        LOGGER.info("Scene cleared")

    # --- GRAPH LOGIC ---

    def add_node_center(self, type_str, cls_name):
        self.is_ai_generated_code = False  # Reset flag - manual node added
        USAGE_TRACKER.record(cls_name, type_str)
        if self.code_view.toPlainText().strip() == "":
            self.compile_graph()
        center = self.view.mapToScene(self.view.rect().center())

        self.add_node(type_str, cls_name, pos=(center.x(), center.y()))

    def _add_structural_node(self, node_type: str):
        """Dispatcher for structural node creation from ElementsPanel buttons."""
        if node_type == "play":
            self.add_play_node()
        elif node_type == "wait":
            self.add_wait_node()
        elif node_type == "vgroup":
            self.add_vgroup_node()

    def add_play_node(self):
        """Add an explicit PLAY node to the canvas at center."""
        self.is_ai_generated_code = False
        center = self.view.mapToScene(self.view.rect().center())
        count = sum(1 for n in self.nodes.values() if n.data.type == NodeType.PLAY)
        name = f"play_{count + 1}"
        node = self.add_node(
            "PLAY", "play()", params={}, pos=(center.x(), center.y()), name=name
        )
        LOGGER.info(f"Added PLAY node: {name}")
        return node

    def add_wait_node(self):
        """Add an explicit WAIT node to the canvas at center."""
        self.is_ai_generated_code = False
        center = self.view.mapToScene(self.view.rect().center())
        count = sum(1 for n in self.nodes.values() if n.data.type == NodeType.WAIT)
        name = f"wait_{count + 1}"
        node = self.add_node(
            "WAIT",
            "wait()",
            params={"duration": 1.0},
            pos=(center.x() + 220, center.y()),
            name=name,
        )
        LOGGER.info(f"Added WAIT node: {name}")
        return node

    def add_vgroup_node(self, group_name: str = "", member_ids: list = None):
        """Add a VGROUP canvas node. Optionally pre-populated with member_ids."""
        self.is_ai_generated_code = False
        if member_ids and self.history_manager:
            self.history_manager.begin_group("Add VGroup Node")
        center = self.view.mapToScene(self.view.rect().center())
        count = sum(1 for n in self.nodes.values() if n.data.type == NodeType.VGROUP)
        var_name = (
            group_name
            if (group_name and group_name.isidentifier())
            else f"vgroup_{count + 1}"
        )
        node = self.add_node(
            "VGROUP",
            var_name,
            params={},
            pos=(center.x(), center.y() + 120),
            name=var_name,
        )
        LOGGER.info(f"Added VGROUP node: {var_name}")
        # Auto-wire member mobjects if provided
        if member_ids:
            for mid in member_ids:
                member_item = self.nodes.get(mid)
                if member_item and member_item.data.type == NodeType.MOBJECT:
                    self.scene.try_connect(member_item.out_socket, node.in_socket)
        if member_ids and self.history_manager:
            self.history_manager.end_group()
        return node

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
            self.statusBar().showMessage("Code copied to clipboard.", 2000)

        elif fmt == "mp4":
            # Switch to video render tab and start render
            for i in range(self.tabs_top.count()):
                if "Video" in self.tabs_top.tabText(i):
                    self.tabs_top.setCurrentIndex(i)
                    break
            self.panel_video.render_scene_btn.click()

    def auto_layout_nodes(self):
        """Auto-arrange nodes in a clean left-to-right flow layout."""
        if self.history_manager:
            self.history_manager.begin_group("Auto Layout")
        apply_auto_layout(self.nodes, self.scene, self.fit_view)
        if self.history_manager:
            self.history_manager.end_group()

    def add_node(
        self, type_str, cls_name, params=None, pos=(0, 0), nid=None, name=None
    ):
        data, item = NodeFactory.create_node(
            type_str, cls_name, params=params, pos=pos, nid=nid, name=name
        )
        try:
            item._window = self
        except Exception:
            pass
        self.scene.addItem(item)
        self.nodes[data.id] = item
        LOGGER.info(
            f"Created {cls_name} ({data.type.name}) as '{data.name}' id={data.id[:8]}"
        )
        self.compile_graph()  # Auto-Refresh
        if self.history_manager:
            self.history_manager.capture(f"Add {data.cls_name}", merge_key=None)
        return item

    def delete_selected(self):
        if self.history_manager:
            self.history_manager.begin_group("Delete Selection")
        for item in self.scene.selectedItems():
            if isinstance(item, NodeItem):
                self.remove_node(item, record_history=False)
            elif isinstance(item, WireItem):
                self.remove_wire(item, record_history=False)
        if self.history_manager:
            self.history_manager.mark_group_dirty()
            self.history_manager.end_group()
        self.compile_graph()  # Auto-Refresh

    def remove_node(self, node, record_history: bool = True):
        # FIX: Clean up preview worker for this node
        self._cleanup_preview_worker(node.data.id)

        wires = node.in_socket.links + node.out_socket.links
        for w in wires:
            self.remove_wire(w, record_history=False)
        if node.data.id in self.nodes:
            del self.nodes[node.data.id]
        self.scene.removeItem(node)

        # FIX: Clear preview if this was the selected node
        if self.panel_props.current_node == node:
            self.preview_lbl.clear()
            self.preview_lbl.setPixmap(QPixmap())
            self.preview_lbl.setText("No Selection")

        if record_history and self.history_manager:
            self.history_manager.capture(f"Delete {node.data.cls_name}")

    def remove_wire(self, wire, record_history: bool = True):
        if wire in wire.start_socket.links:
            wire.start_socket.links.remove(wire)
        if wire in wire.end_socket.links:
            wire.end_socket.links.remove(wire)
        self.scene.removeItem(wire)
        # Graph Changed (Handled by delete_selected, but good for singular removals)
        if record_history and self.history_manager:
            try:
                src = wire.start_socket.parentItem().data.name
                dst = wire.end_socket.parentItem().data.name
                desc = f"Disconnect {src} -> {dst}"
            except Exception:
                desc = "Disconnect"
            self.history_manager.capture(desc, merge_key="wire")

    def on_node_property_changed(self, node, key, value):
        """Handle node property edits from the Properties panel."""
        self.mark_modified()
        if key == "_name":
            try:
                self.panel_outliner.refresh_list()
            except Exception:
                pass
        if self.history_manager and not self.history_manager.is_restoring:
            try:
                nname = node.data.name
            except Exception:
                nname = "Node"
            if key == "_name":
                desc = f"Rename {nname}"
            else:
                desc = f"Edit {nname}.{key}"
            merge_key = f"prop:{node.data.id}:{key}"
            self.history_manager.capture(desc, merge_key=merge_key)

    def add_wire_by_ids(self, from_node_id: str, to_node_id: str, wire_id=None):
        if from_node_id not in self.nodes or to_node_id not in self.nodes:
            return None
        src = self.nodes[from_node_id].out_socket
        dst = self.nodes[to_node_id].in_socket
        return self.scene.try_connect(src, dst, wire_id=wire_id)

    def on_wire_added(self, wire: WireItem) -> None:
        """History hook when a new wire is created."""
        if self.history_manager is None or self.history_manager.is_restoring:
            return
        try:
            src = wire.start_socket.parentItem().data.name
            dst = wire.end_socket.parentItem().data.name
            desc = f"Connect {src} -> {dst}"
        except Exception:
            desc = "Connect"
        self.history_manager.capture(desc, merge_key="wire")

    def remove_wire_by_id(self, wire_id: str):
        if not wire_id:
            return
        for item in list(self.scene.items()):
            if isinstance(item, WireItem) and getattr(item, "wire_id", None) == wire_id:
                self.remove_wire(item)
                break

    def load_graph_from_json(self, graph_json: dict):
        """Clear canvas and load graph JSON into the editor."""
        ctx = self.history_manager.suspend() if self.history_manager else nullcontext()
        try:
            with ctx:
                # Clean existing nodes
                for node_id in list(self.nodes.keys()):
                    self._cleanup_preview_worker(node_id)
                self.nodes.clear()
                self.scene.clear()

                node_map = {}
                for nd in graph_json.get("nodes", []):
                    node = self.add_node(
                        nd.get("type", "MOBJECT"),
                        nd.get("cls_name", ""),
                        params=nd.get("params", {}),
                        pos=tuple(nd.get("pos", (0, 0))),
                        nid=nd.get("id"),
                        name=nd.get("name"),
                    )
                    # Restore metadata
                    node.data.param_metadata = nd.get("param_metadata", {})
                    node.data.audio_asset_id = nd.get("audio_asset_id")
                    node.data.voiceover_transcript = nd.get("voiceover_transcript")
                    node.data.voiceover_duration = float(
                        nd.get("voiceover_duration", 0.0)
                    )
                    node.data.is_ai_generated = bool(nd.get("is_ai_generated", False))
                    node_map[node.data.id] = node

                for w in graph_json.get("wires", []):
                    self.add_wire_by_ids(
                        w.get("from_node"),
                        w.get("to_node"),
                        wire_id=w.get("wire_id"),
                    )
                self.compile_graph()
            if self.history_manager:
                self.history_manager.reset(description="Load Graph")
        except Exception as e:
            LOGGER.warn(f"Failed to load graph JSON: {e}")

    def on_selection(self):
        sel = self.scene.selectedItems()
        if len(sel) == 1 and isinstance(sel[0], NodeItem):
            self.panel_props.set_node(sel[0])
            self.show_preview(sel[0])
        else:
            self.panel_props.set_node(None)

    def on_node_moved(self, node_item: NodeItem) -> None:
        """Track node movement with debounce for history."""
        if self.history_manager is None or self.history_manager.is_restoring:
            return
        try:
            self._pending_move_nodes.add(node_item.data.id)
            # Debounce to avoid capturing every frame
            self._move_debounce.start(220)
        except Exception:
            pass

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

        # ── STEP 2: VGROUP canvas nodes (first-class graph objects) ─────────
        vgroup_nodes = [
            n for n in self.nodes.values() if n.data.type == NodeType.VGROUP
        ]
        vg_vars = {}  # vgroup_node_id → python variable name
        for _vg in vgroup_nodes:
            _vg_members = []
            for _lnk in _vg.in_socket.links:
                _vsrc = _lnk.start_socket.parentItem()
                if _vsrc.data.id in m_vars:
                    _vg_members.append(m_vars[_vsrc.data.id])
            if not _vg_members:
                LOGGER.log(
                    "WARN",
                    "VALIDATION",
                    f"VGroup node '{_vg.data.name}' has no connected Mobject members — skipped.",
                )
                continue
            _vg_var = (
                _vg.data.cls_name
                if (_vg.data.cls_name and _vg.data.cls_name.isidentifier())
                else f"vg_{_vg.data.id[:6]}"
            )
            vg_vars[_vg.data.id] = _vg_var
            code += (
                f"        {_vg_var} = VGroup({chr(44) + chr(32)}.join(_vg_members))\n"
            )
            code += f"        self.add({_vg_var})\n"

        # Patch: re-emit VGroup lines correctly (chr join was wrong above — redo cleanly)
        # Strip bad lines and redo
        _vg_bad_prefix = "        vg_"
        # simpler: just rebuild code from scratch for vgroups by tracking what was added
        # NOTE: The chr() join trick doesn't work in f-strings cleanly.
        # Correct approach using direct string concat:
        code_lines = code.split("\n")
        # Remove any malformed VGroup lines
        code_lines = [line for line in code_lines if "chr(44)" not in line]
        code = "\n".join(code_lines)
        # Now re-emit VGroup definitions cleanly
        for _vg_id, _vg_var in vg_vars.items():
            _vg_node = self.nodes.get(_vg_id)
            if _vg_node:
                _vg_members = []
                for _lnk in _vg_node.in_socket.links:
                    _vsrc = _lnk.start_socket.parentItem()
                    if _vsrc.data.id in m_vars:
                        _vg_members.append(m_vars[_vsrc.data.id])
                if _vg_members:
                    code += f"        {_vg_var} = VGroup({', '.join(_vg_members)})\n"
                    code += f"        self.add({_vg_var})\n"

        # ── STEP 2b: Legacy VGroup panel groups (backward compat) ────────────
        if hasattr(self, "panel_vgroup"):
            for gname, ids in self.panel_vgroup.get_groups().items():
                member_vars = [m_vars[nid] for nid in ids if nid in m_vars]
                if member_vars:
                    already = any(
                        self.nodes.get(vid)
                        and self.nodes[vid].data.type == NodeType.VGROUP
                        for vid in ids
                    )
                    if not already:
                        code += f"        {gname} = VGroup({', '.join(member_vars)})\n"

        # ── STEP 3: Topology-sort PLAY / WAIT execution chain ─────────────
        play_nodes = [n for n in self.nodes.values() if n.data.type == NodeType.PLAY]
        wait_nodes = [n for n in self.nodes.values() if n.data.type == NodeType.WAIT]
        anim_nodes = [
            n for n in self.nodes.values() if n.data.type == NodeType.ANIMATION
        ]

        def _struct_upstream(node):
            return [
                lnk.start_socket.parentItem()
                for lnk in node.in_socket.links
                if lnk.start_socket.parentItem().data.type
                in (NodeType.PLAY, NodeType.WAIT)
            ]

        def _struct_downstream(node):
            return [
                lnk.end_socket.parentItem()
                for lnk in node.out_socket.links
                if lnk.end_socket.parentItem().data.type
                in (NodeType.PLAY, NodeType.WAIT)
            ]

        structural_roots = [
            n for n in play_nodes + wait_nodes if not _struct_upstream(n)
        ]
        ordered_structural = []
        visited_struct = set()

        def _topo_visit(node):
            if node.data.id in visited_struct:
                return
            visited_struct.add(node.data.id)
            ordered_structural.append(node)
            for child in _struct_downstream(node):
                _topo_visit(child)

        for root in structural_roots:
            _topo_visit(root)
        for n in play_nodes + wait_nodes:
            if n.data.id not in visited_struct:
                ordered_structural.append(n)

        # ── STEP 4: Voiceover tracking ────────────────────────────────────
        _vo_entries: list[tuple[int, object]] = []
        _timeline_ms: int = 0

        def _format_anim_call(anim, target_vars):
            nonlocal _timeline_ms
            anim_args = list(target_vars)
            _rt_override_s = None
            _batch_ms = 0

            if anim.data.audio_asset_id:
                _vo_path = ASSETS.get_asset_path(anim.data.audio_asset_id)
                if _vo_path is None:
                    LOGGER.error(f"Voiceover '{anim.data.name}': asset missing")
                elif not os.path.exists(_vo_path):
                    LOGGER.error(f"Voiceover '{anim.data.name}': file not found")
                elif PYDUB_AVAILABLE:
                    try:
                        from pydub import AudioSegment as _AS

                        _seg = _AS.from_file(_vo_path)
                        _dur_ms = len(_seg)
                        if _dur_ms > 0:
                            if anim.data.voiceover_duration > 0:
                                drift = abs(
                                    _dur_ms - int(anim.data.voiceover_duration * 1000)
                                )
                                if drift > 50:
                                    LOGGER.warn(
                                        f"Voiceover drift '{anim.data.name}': {drift}ms"
                                    )
                            anim.data.voiceover_duration = round(_dur_ms / 1000.0, 3)
                            _vo_entries.append((_timeline_ms, _seg))
                            _rt_override_s = round(_dur_ms / 1000.0, 3)
                            _batch_ms = _dur_ms
                    except Exception as _ve:
                        LOGGER.error(f"pydub failed for '{anim.data.name}': {_ve}")

            _used_rt = False
            for k, v in anim.data.params.items():
                if k.startswith("_") or not anim.data.is_param_enabled(k):
                    continue
                if k == "run_time":
                    _used_rt = True
                    if _rt_override_s is not None:
                        anim_args.append(f"run_time={_rt_override_s}")
                    elif isinstance(v, str) and "duration_seconds" in v:
                        anim_args.append("run_time=1.0")
                    else:
                        v_c = self._format_param_value(k, v, anim.data)
                        if v_c is not None:
                            anim_args.append(f"run_time={v_c}")
                    continue
                v_c = self._format_param_value(k, v, anim.data)
                if v_c is not None:
                    anim_args.append(f"{k}={v_c}")

            if _rt_override_s is not None and not _used_rt:
                anim_args.append(f"run_time={_rt_override_s}")

            if _batch_ms == 0:
                try:
                    rt_raw = anim.data.params.get("run_time", "1.0")
                    _batch_ms = (
                        1000
                        if (isinstance(rt_raw, str) and "duration_seconds" in rt_raw)
                        else max(1000, int(float(rt_raw) * 1000))
                    )
                except Exception:
                    _batch_ms = 1000

            return f"{anim.data.cls_name}({', '.join(anim_args)})", _batch_ms

        # ── STEP 5: Generate explicit PLAY / WAIT node code ───────────────
        anim_claimed = set()

        for node in ordered_structural:
            if node.data.type == NodeType.PLAY:
                play_calls = []
                batch_ms = 0
                for lnk in node.in_socket.links:
                    src = lnk.start_socket.parentItem()
                    if src.data.type == NodeType.ANIMATION:
                        tgt_vars = [
                            m_vars[al.start_socket.parentItem().data.id]
                            for al in src.in_socket.links
                            if al.start_socket.parentItem().data.id in m_vars
                        ]
                        if not tgt_vars:
                            LOGGER.log(
                                "WARN",
                                "VALIDATION",
                                f"Animation '{src.data.name}' in PLAY '{node.data.name}' has no Mobject target.",
                            )
                            continue
                        call, bms = _format_anim_call(src, tgt_vars)
                        play_calls.append(call)
                        batch_ms = max(batch_ms, bms)
                        anim_claimed.add(src.data.id)
                    elif src.data.type == NodeType.VGROUP and src.data.id in vg_vars:
                        anim_cls = node.data.params.get("anim_class", "FadeIn")
                        play_calls.append(f"{anim_cls}({vg_vars[src.data.id]})")

                if not play_calls:
                    LOGGER.log(
                        "WARN",
                        "VALIDATION",
                        f"PLAY node '{node.data.name}' has no connected animations — emitting comment.",
                    )
                    code += (
                        f"        # PLAY '{node.data.name}': no animations connected\n"
                    )
                else:
                    code += f"        self.play({', '.join(play_calls)})\n"
                    _timeline_ms += batch_ms

            elif node.data.type == NodeType.WAIT:
                raw_dur = node.data.params.get("duration", 1.0)
                try:
                    dur = float(raw_dur)
                    if dur < 0.01:
                        LOGGER.log(
                            "WARN",
                            "VALIDATION",
                            f"WAIT node '{node.data.name}' duration {dur} < 0.01 — clamped.",
                        )
                        dur = 0.01
                except (TypeError, ValueError):
                    LOGGER.log(
                        "WARN",
                        "VALIDATION",
                        f"WAIT node '{node.data.name}' invalid duration '{raw_dur}' — using 1.0",
                    )
                    dur = 1.0
                code += f"        self.wait({dur})\n"
                _timeline_ms += int(dur * 1000)

        # ── STEP 6: Legacy animations not connected to any PLAY node ──────
        legacy_anims = [n for n in anim_nodes if n.data.id not in anim_claimed]
        if legacy_anims:
            if play_nodes:
                LOGGER.log(
                    "WARN",
                    "VALIDATION",
                    f"{len(legacy_anims)} animation(s) not connected to any PLAY node. "
                    "Add PLAY nodes (Tools → Add Play Node) for deterministic ordering.",
                )
            code += "\n        # ── Legacy animation batch (connect to PLAY nodes) ──\n"
            _played_legacy = set()
            _remaining_legacy = list(legacy_anims)
            while _remaining_legacy:
                _ready = []
                for _an in _remaining_legacy:
                    _tgts = [
                        m_vars[_lk.start_socket.parentItem().data.id]
                        for _lk in _an.in_socket.links
                        if _lk.start_socket.parentItem().data.id in m_vars
                    ]
                    if _tgts:
                        _ready.append((_an, _tgts))
                if not _ready:
                    break
                _play_lines, _bms = [], 0
                for _an, _tgts in _ready:
                    _call, _b = _format_anim_call(_an, _tgts)
                    _play_lines.append(_call)
                    _bms = max(_bms, _b)
                    _played_legacy.add(_an)
                if _play_lines:
                    code += f"        self.play({', '.join(_play_lines)})\n"
                    code += "        self.wait(0.5)\n"
                    _timeline_ms += _bms + 500
                _remaining_legacy = [
                    a for a in _remaining_legacy if a not in _played_legacy
                ]

        # ── STEP 7: Merged voiceover track ────────────────────────────────
        if _vo_entries and PYDUB_AVAILABLE:
            try:
                from pydub import AudioSegment as _AS

                _last_off, _last_seg = max(_vo_entries, key=lambda x: x[0])
                _total_ms = _last_off + len(_last_seg) + 500
                _merged = _AS.silent(
                    duration=_total_ms, frame_rate=_last_seg.frame_rate
                )
                for _off, _seg in _vo_entries:
                    if _seg.frame_rate != _merged.frame_rate:
                        _seg = _seg.set_frame_rate(_merged.frame_rate)
                    _merged = _merged.overlay(_seg, position=_off)
                _mpath = AppPaths.TEMP_DIR / f"merged_vo_{uuid.uuid4().hex[:8]}.wav"
                _merged.export(str(_mpath), format="wav")
                _inject = f"        self.add_sound(r'{_mpath.as_posix()}')  # merged voiceover\n"
                code = code.replace(
                    "class GeneratedScene(Scene):\n    def construct(self):\n",
                    f"class GeneratedScene(Scene):\n    def construct(self):\n{_inject}",
                    1,
                )
                LOGGER.info(
                    f"Merged voiceover: {len(_vo_entries)} segments, total={_total_ms}ms"
                )
            except Exception as _e:
                LOGGER.error(f"Merged voiceover build failed: {_e}")

        self.code_view.setText(code)
        return code

    def _format_param_value(self, param_name, value, node_data):
        """Safely format parameter value with type enforcement and string escaping.
        Includes parameter sanitization: clamps invalid values, replaces None with
        safe defaults, strips inspect._empty — no raw user data ever reaches Manim.
        """
        try:
            # ── PARAMETER SANITIZATION LAYER ─────────────────────────────────
            # items=None → replace with empty list (BulletedList, etc.)
            if param_name in ("items", "strings", "labels") and value is None:
                LOGGER.log(
                    "WARN", "VALIDATION", f"Replaced None for '{param_name}' with []"
                )
                return "[]"

            # font_size / font_size <= 0 → clamp to 1
            if param_name in ("font_size", "font_size") and value is not None:
                try:
                    fs = float(value)
                    if fs <= 0:
                        LOGGER.log(
                            "WARN", "VALIDATION", f"Clamped {param_name}={fs} to 1"
                        )
                        return "1"
                except (TypeError, ValueError):
                    pass

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

            # 5. Instantiate with Exception Block for Safety — wrap entire construct()
            script += "        try:\n"
            script += "            # Target Mobject\n"
            script += (
                "            obj = "
                + node.data.cls_name
                + "("
                + ", ".join(final_args)
                + ")\n"
            )
            script += "            obj.move_to(ORIGIN)\n"
            script += "            if obj.width > config.frame_width: obj.scale_to_fit_width(config.frame_width * 0.9)\n"
            script += "            self.add(obj)\n"
            script += "        except Exception as _e:\n"
            script += "            import traceback as _tb, sys\n"
            script += "            print(f'[PREVIEW ERROR] {_e}', file=sys.stderr)\n"
            script += "            _tb.print_exc(file=sys.stderr)\n"

            # 6. Write File
            s_path = AppPaths.TEMP_DIR / f"preview_{nid}.py"
            with open(s_path, "w", encoding="utf-8") as f:
                f.write(script)

            # 7. Start Worker — set in-progress flag and maintain strong reference
            self._render_in_progress = True
            worker = RenderWorker(s_path, nid, AppPaths.TEMP_DIR, 15, "l")
            self._active_render_workers.add(worker)
            worker.success.connect(self.on_render_ok)

            # Show error in UI instead of crashing/quitting
            _node_ref = node  # Capture for closure

            def handle_error(err, _n=_node_ref):
                LOGGER.log(
                    "WARN", "RENDER", f"Preview Render Error for {_n.data.name}: {err}"
                )
                if self.panel_props.current_node == _n:
                    self.preview_lbl.setText(f"Render Failed\n{str(err)[:30]}...")
                    self.preview_lbl.setStyleSheet("color: red;")

            worker.error.connect(handle_error)
            worker.finished.connect(
                lambda _nid=nid, _w=worker: self._cleanup_preview_worker(_nid, _w)
            )
            worker.start()

            setattr(self, f"rw_{nid}", worker)

        except Exception as e:
            LOGGER.error(f"Queue Processing Error: {e}")
            traceback.print_exc()

    def on_render_ok(self, nid, path):
        """Called when RenderWorker successfully creates a PNG.
        NOTE: Cleanup is handled exclusively by the worker.finished signal
        to avoid double-cleanup race conditions.
        """
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
        except Exception as e:
            LOGGER.log("ERROR", "RENDER", f"on_render_ok error: {e}")

    def _cleanup_preview_worker(self, node_id, worker=None):
        """Cleanup render worker thread for a specific node.
        Called exclusively from worker.finished signal — never directly from
        on_render_ok — to prevent double-cleanup race conditions.
        """
        try:
            # Reset render-in-progress state so next item can be processed
            self._render_in_progress = False

            # Remove from strong-reference set (allows GC once thread is done)
            if worker is not None:
                self._active_render_workers.discard(worker)

            worker_attr = f"rw_{node_id}"
            if hasattr(self, worker_attr):
                w = getattr(self, worker_attr)
                # Use cancel() to kill subprocess, then wait for thread exit
                if hasattr(w, "cancel"):
                    w.cancel()
                if hasattr(w, "isRunning") and w.isRunning():
                    w.wait(3000)  # Wait max 3 seconds
                delattr(self, worker_attr)

            # Schedule next item in queue if any
            QTimer.singleShot(50, self.process_render_queue)
        except Exception as e:
            LOGGER.log("WARN", "RENDER", f"Error cleaning up preview worker: {e}")

    def show_preview(self, node):
        """Display preview for selected node with safe resource management."""
        self.preview_lbl.clear()

        # 1. Check data
        if not node:
            self.preview_lbl.setPixmap(QPixmap())  # Release any cached pixmap
            self.preview_lbl.setText("No Selection")
            return

        if node.data.type == NodeType.ANIMATION:
            self.preview_lbl.setPixmap(QPixmap())
            self.preview_lbl.setText("Preview not supported\nfor Animation nodes.")
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
                    "Invalid Image\n(Corrupted or unsupported format)"
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
            self.preview_lbl.setText(f"Error loading preview:\n{type(e).__name__}")
            LOGGER.warn(f"Preview load error: {e}")

    # --- VIDEO RENDERING ---

    def validate_graph(self) -> list[str]:
        """Validate graph structural consistency. Returns list of error/warning messages.

        Enforces:
        - No empty PLAY nodes (warn, not block — they emit comments)
        - No empty VGROUP nodes (logged in compile_graph)
        - WAIT nodes have valid durations
        - ANIMATION nodes connected to PLAY nodes (warn if orphaned)
        - No missing mobject targets for animation nodes
        """
        issues = []
        for node in self.nodes.values():
            if node.data.type == NodeType.PLAY:
                anim_inputs = [
                    lnk
                    for lnk in node.in_socket.links
                    if lnk.start_socket.parentItem().data.type
                    in (NodeType.ANIMATION, NodeType.VGROUP)
                ]
                if not anim_inputs:
                    issues.append(
                        f"Warning: PLAY node '{node.data.name}' has no animation inputs."
                    )

            elif node.data.type == NodeType.WAIT:
                raw = node.data.params.get("duration", None)
                if raw is None:
                    issues.append(
                        f"Warning: WAIT node '{node.data.name}' has no duration set — will default to 1.0"
                    )
                else:
                    try:
                        d = float(raw)
                        if d < 0:
                            issues.append(
                                f"Warning: WAIT node '{node.data.name}' has negative duration {d}."
                            )
                    except (TypeError, ValueError):
                        issues.append(
                            f"Warning: WAIT node '{node.data.name}' has invalid duration '{raw}'."
                        )

            elif node.data.type == NodeType.VGROUP:
                mob_inputs = [
                    lnk
                    for lnk in node.in_socket.links
                    if lnk.start_socket.parentItem().data.type == NodeType.MOBJECT
                ]
                if not mob_inputs:
                    issues.append(
                        f"Warning: VGROUP node '{node.data.name}' has no member Mobjects — will be skipped."
                    )

            elif node.data.type == NodeType.ANIMATION:
                mob_inputs = [
                    lnk
                    for lnk in node.in_socket.links
                    if lnk.start_socket.parentItem().data.type == NodeType.MOBJECT
                ]
                if not mob_inputs:
                    issues.append(
                        f"Warning: Animation '{node.data.name}' has no connected Mobject target."
                    )
                play_outputs = [
                    lnk
                    for lnk in node.out_socket.links
                    if lnk.end_socket.parentItem().data.type == NodeType.PLAY
                ]
                if not play_outputs:
                    issues.append(
                        f"ℹ Animation '{node.data.name}' is not connected to a PLAY node — "
                        f"will use legacy implicit play order."
                    )

        return issues

    def render_to_video(self, config):
        """Render full scene to video with specified config."""
        try:
            # Normalize config and apply sane defaults (menu actions may pass {})
            config = self._normalize_render_config(config)

            # ── Graph validation before render ─────────────────────────────
            if not self.is_ai_generated_code:
                issues = self.validate_graph()
                if issues:
                    LOGGER.log("WARN", "GRAPH", "Pre-render validation issues:")
                    for issue in issues:
                        LOGGER.log("WARN", "GRAPH", f"  {issue}")
                    # Log count to status bar too
                    self.statusBar().showMessage(
                        f"Warning: {len(issues)} graph validation warning(s) — check Logs tab",
                        4000,
                    )

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
            with open(script_path, "w", encoding="utf-8") as f:
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

    def _normalize_render_config(self, config: dict | None) -> dict:
        cfg = dict(config or {})

        # Output path (menu actions can pass empty config)
        output_path = cfg.get("output_path") or cfg.get("output_dir")
        if not output_path:
            try:
                output_path = self.panel_video.output_path_lbl.text().strip()
            except Exception:
                output_path = ""
        if not output_path:
            output_path = str(AppPaths.TEMP_DIR)
        cfg["output_path"] = str(output_path)

        # FPS
        fps = cfg.get("fps")
        if fps is None or fps == "":
            try:
                fps = int(self.panel_video.fps_spin.value())
            except Exception:
                fps = 30
        cfg["fps"] = int(fps)

        # Resolution
        resolution = cfg.get("resolution") or cfg.get("res")
        if resolution is None:
            w = cfg.get("width") or cfg.get("w")
            h = cfg.get("height") or cfg.get("h")
            if w is not None and h is not None:
                resolution = (int(w), int(h))
            else:
                try:
                    resolution = (
                        int(self.panel_video.width_spin.value()),
                        int(self.panel_video.height_spin.value()),
                    )
                except Exception:
                    resolution = (1280, 720)
        else:
            try:
                resolution = (int(resolution[0]), int(resolution[1]))
            except Exception:
                resolution = (1280, 720)
        cfg["resolution"] = resolution

        # Quality
        quality = cfg.get("quality")
        if not quality:
            try:
                quality = ["l", "m", "h", "k"][
                    self.panel_video.quality_combo.currentIndex()
                ]
            except Exception:
                quality = "m"
        if isinstance(quality, str):
            ql = quality.strip().lower()
            if ql.startswith("q") and len(ql) >= 2:
                quality = ql[-1]
            elif "low" in ql:
                quality = "l"
            elif "medium" in ql:
                quality = "m"
            elif "high" in ql:
                quality = "h"
            elif "ultra" in ql or "4k" in ql:
                quality = "k"
        cfg["quality"] = str(quality)

        return cfg

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
                LOGGER.ai(f"Successfully added {result['nodes_added']} node(s)")

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

                # History capture (single atomic action)
                if self.history_manager:
                    try:
                        prompt_text = (
                            self.panel_ai.input.toPlainText()
                            if hasattr(self.panel_ai, "input")
                            else ""
                        )
                    except Exception:
                        prompt_text = ""
                    meta = {
                        "ai_prompt": prompt_text,
                        "ai_code": code[:1000],
                        "nodes_added": result.get("nodes_added", 0),
                    }
                    self.history_manager.capture("AI Merge", metadata=meta)
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

    def get_graph_json(self) -> dict:
        """Return current graph as a JSON-serializable dict."""
        return {
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

    def export_code_to(self, path: str) -> bool:
        """Export current code view to a file path."""
        try:
            code = self.code_view.toPlainText()
            if not code.strip():
                return False
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(code)
            return True
        except Exception as e:
            LOGGER.error(f"Export code failed: {e}")
            return False

    def export_code_to_clipboard(self) -> int:
        """Copy current code to clipboard. Returns chars copied."""
        code = self.code_view.toPlainText()
        QApplication.clipboard().setText(code)
        return len(code)

    def save_project_to(self, path: str) -> bool:
        """Save project directly to a path (no dialogs)."""
        if not path:
            return False
        if not str(path).endswith(PROJECT_EXT):
            path = f"{path}{PROJECT_EXT}"

        # Update project name from saved path
        if hasattr(self, "project_name_edit"):
            self.project_name_edit.setText(Path(path).stem)

        meta = {
            "name": Path(path).stem,
            "created": str(datetime.now()),
            "version": APP_VERSION,
        }

        graph_data = self.get_graph_json()

        try:
            with tempfile.TemporaryDirectory() as td:
                t_path = Path(td)

                # Write JSONs
                with open(t_path / "metadata.json", "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
                with open(t_path / "graph.json", "w", encoding="utf-8") as f:
                    json.dump(graph_data, f, indent=2)
                with open(t_path / "code.py", "w", encoding="utf-8") as f:
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

                with open(t_path / "assets.json", "w", encoding="utf-8") as f:
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
                add_recent(str(final_efp))

                # Update project name field
                if hasattr(self, "project_name_edit"):
                    self.project_name_edit.setText(final_efp.stem)

                self.reset_modified()
                LOGGER.info(f"Project saved to {final_efp}")
                return True

        except Exception as e:
            LOGGER.error(f"Save Failed: {e}")
            traceback.print_exc()
            return False

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

        self.save_project_to(path)

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
            ctx = (
                self.history_manager.suspend()
                if self.history_manager
                else nullcontext()
            )
            with ctx:
                self.nodes.clear()
                self.scene.clear()
                ASSETS.clear()

            # Create a clean extraction folder for this project workspace
            dest = AppPaths.TEMP_DIR / "Project_Assets"
            if dest.exists():
                shutil.rmtree(dest)
            dest.mkdir(parents=True)

            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(dest)

            # 1. Load Assets and Re-link Paths
            assets_json = dest / "assets.json"
            if assets_json.exists():
                with open(assets_json, encoding="utf-8") as f:
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
                    with open(graph_json, encoding="utf-8") as f:
                        g = json.load(f)
                        node_map = {}
                        for nd in g["nodes"]:
                            # Pass the raw type string directly — add_node handles all variants
                            type_str = nd[
                                "type"
                            ].upper()  # MOBJECT, ANIMATION, PLAY, WAIT, VGROUP

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
                with open(code_py, "r", encoding="utf-8") as f:
                    self.code_view.setText(f.read())

            self.compile_graph()
            if self.history_manager:
                self.history_manager.reset(
                    description="Open Project", clear_checkpoints=True
                )
            self.project_path = path
            add_recent(path)
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
