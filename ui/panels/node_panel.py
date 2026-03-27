from __future__ import annotations
# -*- coding: utf-8 -*-

import json
import os
import re
import traceback
import uuid
from pathlib import Path

from PySide6.QtCore import QMimeData, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QDrag,
    QFont,
    QIcon,
    QPixmap,
    QTextCursor,
)
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStyle,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.config import AppPaths, SETTINGS
from core.file_manager import ASSETS, USAGE_TRACKER
from graph.node import NodeData, NodeItem, NodeType
from rendering.render_manager import AIWorker, LatexApiWorker, TTSWorker
from ui.runtime_flags import MCP_AVAILABLE, PYDUB_AVAILABLE
from ui.ai_pdf_upload_widget import AIPDFUploadWidget
from utils.helpers import MANIM_AVAILABLE, TypeSafeParser, bold_font, manim
from utils.logger import LOGGER
from utils.tooltips import apply_tooltip

import inspect
import subprocess
import sys
import tempfile
from urllib.parse import urlparse


class SceneOutlinerPanel(QWidget):
    """Lists all nodes in the scene for easy selection and management."""

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setup_ui()

        # Connect to scene changes to auto-refresh
        self.main_window.scene.graph_changed_signal.connect(self.refresh_list)

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Search Bar
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter nodes...")
        self.search.textChanged.connect(self.refresh_list)
        layout.addWidget(self.search)

        # The List
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.list_widget.itemClicked.connect(self.on_item_clicked)
        # Context menu for deleting
        self.list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self.show_context_menu)
        layout.addWidget(self.list_widget)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_del = QPushButton("🗑️ Delete")
        btn_del.clicked.connect(self.delete_selected)
        btn_layout.addWidget(btn_del)

        btn_refresh = QPushButton("🔄 Refresh")
        btn_refresh.clicked.connect(self.refresh_list)
        btn_layout.addWidget(btn_refresh)

        layout.addLayout(btn_layout)

    def refresh_list(self):
        """Rebuild the list based on current nodes."""
        self.list_widget.clear()
        filter_text = self.search.text().lower()

        # Sort nodes by creation order (roughly) or name
        nodes = list(self.main_window.nodes.values())

        for node in nodes:
            # Filter
            if filter_text and filter_text not in node.node_data.name.lower():
                continue

            # Create Item
            _icon_map = {
                NodeType.MOBJECT: "📦",
                NodeType.ANIMATION: "🎬",
                NodeType.PLAY: "▶",
                NodeType.WAIT: "⏱",
                NodeType.VGROUP: "🔗",
            }
            icon_char = _icon_map.get(node.node_data.type, "🎬")
            display_text = (
                f"{icon_char} {node.node_data.name} ({node.node_data.cls_name})"
            )

            item = QListWidgetItem(display_text)
            item.setData(Qt.ItemDataRole.UserRole, node.node_data.id)

            # Highlight if selected in scene
            if node.isSelected():
                item.setBackground(QColor("#d6eaf8"))  # Light blue

            self.list_widget.addItem(item)

    def on_item_clicked(self, item):
        """Select the node in the scene when clicked in the list."""
        node_id = item.data(Qt.ItemDataRole.UserRole)
        if node_id in self.main_window.nodes:
            # Deselect all first
            for n in self.main_window.nodes.values():
                n.setSelected(False)

            # Select target
            node = self.main_window.nodes[node_id]
            node.setSelected(True)

            # Focus view on node
            self.main_window.view.centerOn(node)
            self.main_window.on_selection()  # Trigger property panel update

    def delete_selected(self):
        """Delete nodes selected in the list."""
        ids_to_delete = []
        for item in self.list_widget.selectedItems():
            ids_to_delete.append(item.data(Qt.ItemDataRole.UserRole))

        if not ids_to_delete:
            return

        reply = QMessageBox.question(
            self,
            "Delete",
            f"Delete {len(ids_to_delete)} items?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            for nid in ids_to_delete:
                if nid in self.main_window.nodes:
                    self.main_window.remove_node(self.main_window.nodes[nid])
            self.refresh_list()
            self.main_window.compile_graph()

    def show_context_menu(self, pos):
        menu = QMenu()
        del_act = menu.addAction("Delete")
        action = menu.exec(self.list_widget.mapToGlobal(pos))
        if action == del_act:
            self.delete_selected()


class LatexEditorPanel(QWidget):
    """Panel for Live LaTeX editing via API."""

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.temp_preview_path = AppPaths.TEMP_DIR / "latex_preview.png"

        # FIX: Keep track of all active threads to prevent Garbage Collection crashes
        self._active_workers = set()

        self.setup_ui()

        # Debouncer timer (Auto-update logic)
        self.debouncer = QTimer()
        self.debouncer.setSingleShot(True)
        self.debouncer.setInterval(800)  # Wait 800ms after typing stops
        self.debouncer.timeout.connect(self.trigger_render)

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Header
        header = QLabel("LaTeX Studio")
        header.setStyleSheet("font-weight: bold; font-size: 14px; color: #2c3e50;")
        layout.addWidget(header)

        # Editor
        self.editor = QPlainTextEdit()
        self.editor.setPlaceholderText("Enter LaTeX here... e.g. E = mc^2")
        self.editor.setStyleSheet("font-family: Consolas; font-size: 12pt;")
        self.editor.setMaximumHeight(100)
        self.editor.textChanged.connect(self.on_text_changed)
        layout.addWidget(self.editor)

        # Preview Area
        self.preview_lbl = QLabel("Preview will appear here...")
        self.preview_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_lbl.setStyleSheet(
            "background: white; border: 2px dashed #bdc3c7; border-radius: 4px;"
        )
        self.preview_lbl.setMinimumHeight(150)

        # Scroll area
        scroll = QScrollArea()
        scroll.setWidget(self.preview_lbl)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        # Controls Group
        ctrl_group = QFrame()
        ctrl_group.setStyleSheet(
            "background: #ecf0f1; border-radius: 4px; padding: 4px;"
        )
        ctrl_layout = QVBoxLayout(ctrl_group)

        # Target Node Selector
        self.node_combo = QComboBox()
        self.node_combo.setPlaceholderText("Select Target Node...")
        ctrl_layout.addWidget(QLabel("Apply to Node:"))
        ctrl_layout.addWidget(self.node_combo)

        # Refresh Button
        btn_refresh = QPushButton("🔄 Refresh Node List")
        btn_refresh.clicked.connect(self.refresh_nodes)
        ctrl_layout.addWidget(btn_refresh)

        # Apply Button
        self.btn_apply = QPushButton("Apply to Node")
        self.btn_apply.setStyleSheet(
            "background-color: #27ae60; color: white; font-weight: bold; padding: 6px;"
        )
        self.btn_apply.clicked.connect(self.apply_to_node)
        ctrl_layout.addWidget(self.btn_apply)

        layout.addWidget(ctrl_group)

        # Status
        self.status_lbl = QLabel("Ready")
        self.status_lbl.setStyleSheet("color: gray;")
        layout.addWidget(self.status_lbl)

    def on_text_changed(self):
        """Called on every keystroke."""
        self.status_lbl.setText("Typing...")
        self.debouncer.start()  # Reset timer

    def trigger_render(self):
        """Called by timer to start API call."""
        tex = self.editor.toPlainText().strip()
        if not tex:
            return

        self.status_lbl.setText("Fetching render from API...")

        # FIX: Create a new worker and track it
        worker = LatexApiWorker(tex)
        worker.success.connect(self.on_render_success)
        worker.error.connect(self.on_render_error)

        # FIX: Clean up reference when done
        worker.finished.connect(lambda: self._cleanup_worker(worker))

        # Add to active set so Python doesn't delete it while it's running
        self._active_workers.add(worker)
        worker.start()

    def _cleanup_worker(self, worker):
        """Remove worker from active set to allow Garbage Collection."""
        if worker in self._active_workers:
            self._active_workers.remove(worker)
        worker.deleteLater()

    def on_render_success(self, image_data):
        self.status_lbl.setText("Render received.")

        try:
            with open(self.temp_preview_path, "wb") as f:
                f.write(image_data)
        except Exception as e:
            LOGGER.warning(f"Could not cache LaTeX png: {e}")

        pixmap = QPixmap()
        pixmap.loadFromData(image_data)

        if pixmap.width() > self.preview_lbl.width():
            pixmap = pixmap.scaledToWidth(
                self.preview_lbl.width() - 20,
                Qt.TransformationMode.SmoothTransformation,
            )

        self.preview_lbl.setPixmap(pixmap)
        self.preview_lbl.setText("")

    def on_render_error(self, err_msg):
        self.status_lbl.setText("API Error.")
        # Only show error if it's not just a cancellation
        if "Network" in err_msg or "400" in err_msg:
            self.preview_lbl.setText(f"Error:\n{err_msg}")
            self.preview_lbl.setPixmap(QPixmap())

    def refresh_nodes(self):
        """Populate combo with nodes that look like they accept text."""
        current = self.node_combo.currentData()
        self.node_combo.clear()

        count = 0
        for nid, node in self.main_window.nodes.items():
            cname = node.data.cls_name.lower()
            if "tex" in cname or "text" in cname or "label" in cname:
                # FIX: Format as "Name (First 6 chars of ID)"
                short_id = nid[:6]
                display_text = f"{node.data.name} ({short_id})"
                self.node_combo.addItem(display_text, nid)
                count += 1

        if current:
            idx = self.node_combo.findData(current)
            if idx >= 0:
                self.node_combo.setCurrentIndex(idx)

        self.status_lbl.setText(f"Found {count} text-compatible nodes.")

    def apply_to_node(self):
        """Inject the LaTeX code into the selected node."""
        node_id = self.node_combo.currentData()
        if not node_id or node_id not in self.main_window.nodes:
            QMessageBox.warning(self, "Error", "Please select a valid target node.")
            return

        tex_code = self.editor.toPlainText().strip()
        if not tex_code:
            return

        # Balance parentheses
        open_count = 0
        balanced_tex = ""
        for char in tex_code:
            if char == "(":
                open_count += 1
                balanced_tex += char
            elif char == ")":
                if open_count > 0:
                    open_count -= 1
                    balanced_tex += char
            else:
                balanced_tex += char
        balanced_tex += ")" * open_count

        node = self.main_window.nodes[node_id]

        # Cleanup conflicting params
        for k in ("tex_strings", "arg0", "arg1", "t"):
            node.data.params.pop(k, None)

        # Escape backslashes
        safe_tex = balanced_tex.replace("\\", "\\\\")

        # Use raw string literal format
        formatted_code = f'r"""{safe_tex}"""'

        target_param = "text"
        if node.data.cls_name == "MathTex":
            target_param = "arg0"
        elif node.data.cls_name == "Text":
            target_param = "text"

        node.data.params[target_param] = formatted_code

        # Configure Metadata
        node.data.set_escape_string(target_param, False)
        node.data.set_param_enabled(target_param, True)

        node.update()
        self.main_window.compile_graph()
        self.status_lbl.setText(f"Applied to {node.data.name}!")

        node.setSelected(True)
        self.main_window.on_selection()

        # Trigger re-render
        self.main_window.queue_render(node)


class PropertiesPanel(QWidget):
    """Enhanced inspector with type safety, parameter validation, and metadata columns."""

    node_updated = Signal()

    def __init__(self, main_window=None):
        super().__init__()
        self.main_window = main_window
        self.current_node = None
        self._vbox_layout = QVBoxLayout(self)

        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.form_widget = QWidget()
        self.form = QFormLayout(self.form_widget)
        self._scroll_area.setWidget(self.form_widget)

        self._vbox_layout.addWidget(self._scroll_area)

        # Debouncer for update signals
        self.debouncer = QTimer()
        self.debouncer.setSingleShot(True)
        self.debouncer.setInterval(500)
        self.debouncer.timeout.connect(self.node_updated.emit)

        # Track active widgets for safe cleanup
        self.active_widgets = {}

    def set_node(self, node_item: "NodeItem | None"):
        """Load node properties into inspector with full type safety."""
        self.current_node = node_item

        # Clean up previous widgets
        for widget in self.active_widgets.values():
            try:
                widget.deleteLater()
            except Exception:
                pass
        self.active_widgets.clear()

        while self.form.count():
            child = self.form.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        if not node_item:
            return

        # FIX: Validate types, but SKIP asset parameters to prevent UUID->0.0 corruption
        for k, v in list(node_item.node_data.params.items()):
            if TypeSafeParser.is_asset_param(k):
                continue  # Do not touch asset strings/UUIDs
            elif TypeSafeParser.is_color_param(k):
                node_item.node_data.params[k] = TypeSafeParser.parse_color(v)
            elif TypeSafeParser.is_numeric_param(k):
                node_item.node_data.params[k] = TypeSafeParser.parse_numeric(v)

        # ===== AI GENERATED INDICATOR =====
        if node_item.node_data.is_ai_generated:
            ai_label = QLabel("✨ AI GENERATED NODE ✨")
            font = QFont()
            font.setPointSize(10)
            ai_label.setFont(font)
            ai_label.setStyleSheet(
                "background: linear-gradient(90deg, #e3f2fd, #f3e5f5); "
                "color: #1565c0; padding: 6px; border-radius: 3px; "
                "border-left: 3px solid #1565c0;"
            )
            ai_label.setToolTip(
                f"This node was generated by Gemini AI.\n"
                f"Class: {node_item.node_data.ai_source or node_item.node_data.cls_name}\n"
                f"All parameters are available for editing."
            )
            self.form.addRow(ai_label)

        # Meta Information
        self.form.addRow(QLabel("<b>Properties</b>"))
        id_lbl = QLabel(node_item.node_data.id[:8])
        id_lbl.setStyleSheet("color: gray;")
        self.form.addRow("ID", id_lbl)

        name_edit = QLineEdit(node_item.node_data.name)
        name_edit.textChanged.connect(lambda t: self.update_param("_name", t))
        self.form.addRow("Name", name_edit)

        # ── Structural nodes (PLAY/WAIT/VGROUP): show type-specific UI ───
        if node_item.node_data.type == NodeType.WAIT:
            type_lbl = QLabel("⏱ WAIT Node — explicit self.wait()")
            type_lbl.setStyleSheet(
                "background: #fef3c7; color: #92400e; padding: 4px 6px; "
                "border-radius: 4px; font-weight: bold; font-size: 11px;"
            )
            self.form.addRow(type_lbl)
            dur_spin = QDoubleSpinBox()
            dur_spin.setRange(0.01, 999.0)
            dur_spin.setDecimals(2)
            dur_spin.setSingleStep(0.5)
            raw_dur = node_item.node_data.params.get("duration", 1.0)
            try:
                dur_spin.setValue(float(raw_dur))
            except (TypeError, ValueError):
                dur_spin.setValue(1.0)
            dur_spin.valueChanged.connect(lambda v: self.update_param("duration", v))
            self.form.addRow("Duration (s)", dur_spin)
            self.active_widgets["duration"] = dur_spin
            return

        if node_item.node_data.type == NodeType.PLAY:
            type_lbl = QLabel("▶ PLAY Node — explicit self.play()")
            type_lbl.setStyleSheet(
                "background: #d1fae5; color: #064e3b; padding: 4px 6px; "
                "border-radius: 4px; font-weight: bold; font-size: 11px;"
            )
            self.form.addRow(type_lbl)
            hint = QLabel(
                "Connect Animation nodes to this node's\n"
                "input socket. Connect Play → Wait → Play\n"
                "to define execution sequence."
            )
            hint.setStyleSheet("color: #6b7280; font-size: 10px;")
            self.form.addRow(hint)
            return

        if node_item.node_data.type == NodeType.VGROUP:
            type_lbl = QLabel("🔗 VGROUP Node — explicit VGroup")
            type_lbl.setStyleSheet(
                "background: #ccfbf1; color: #134e4a; padding: 4px 6px; "
                "border-radius: 4px; font-weight: bold; font-size: 11px;"
            )
            self.form.addRow(type_lbl)
            hint = QLabel(
                "Connect Mobject nodes to this node's\n"
                "input socket to add them to the VGroup.\n"
                "Connect output to a PLAY node to animate."
            )
            hint.setStyleSheet("color: #6b7280; font-size: 10px;")
            self.form.addRow(hint)
            return

        if not MANIM_AVAILABLE:
            self.form.addRow(QLabel("Manim not loaded."))
            return

        try:
            cls = getattr(manim, node_item.node_data.cls_name, None)
            if not cls:
                return

            sig = inspect.signature(cls.__init__)

            # Auto-load missing parameters
            for param_name, param in sig.parameters.items():
                if param_name in ("self", "args", "kwargs", "mobject"):
                    continue

                if param_name not in node_item.node_data.params:
                    if param.default is not inspect.Parameter.empty:
                        default_val = param.default
                        if TypeSafeParser.is_color_param(param_name):
                            default_val = TypeSafeParser.parse_color(default_val)
                        elif TypeSafeParser.is_numeric_param(param_name):
                            default_val = TypeSafeParser.parse_numeric(default_val)
                        node_item.node_data.params[param_name] = default_val

            # Create Rows
            first_param = True
            for name, param in sig.parameters.items():
                if name in ["self", "args", "kwargs", "mobject"]:
                    continue

                # Check default value
                val = node_item.node_data.params.get(name, param.default)
                if val is inspect.Parameter.empty:
                    val = None

                # Handle default enablement and escaping
                if name not in node_item.node_data.param_metadata:
                    if name in ("tex_strings", "text"):
                        node_item.node_data.set_param_enabled(name, True)
                        # Ensure these are NOT escaped by default
                        node_item.node_data.set_escape_string(name, False)
                    else:
                        node_item.node_data.set_param_enabled(name, first_param)
                
                # Mark that we've processed the first valid parameter
                first_param = False

                row_widget = self.create_parameter_row(name, val, param.annotation)
                # FIX: Actually add the row to the form layout
                if row_widget:
                    self.form.addRow(name, row_widget)

        except Exception as e:
            LOGGER.error(f"Inspector Error for {node_item.node_data.cls_name}: {e}")
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
            state_chk.setToolTip(
                "Check to include in code generation (Ctrl+E to toggle)"
            )
            is_enabled = self.current_node.node_data.is_param_enabled(key)
            state_chk.setChecked(is_enabled)
            state_chk.stateChanged.connect(
                lambda s: (
                    self.current_node.node_data.set_param_enabled(key, s == 2)
                    if self.current_node
                    else None,
                    self.main_window.on_node_property_changed(
                        self.current_node, f"{key}_enabled", s == 2
                    )
                    if self.main_window and self.current_node
                    else None,
                )
            )
            row_layout.addWidget(state_chk, 1)

            # Escape String checkbox
            escape_chk = QCheckBox("Escape")
            escape_chk.setToolTip("Check to escape strings (remove quotes)")
            should_escape = self.current_node.node_data.should_escape_string(key)
            escape_chk.setChecked(should_escape)
            escape_chk.stateChanged.connect(
                lambda s: (
                    self.current_node.node_data.set_escape_string(key, s == 2)
                    if self.current_node
                    else None,
                    self.main_window.on_node_property_changed(
                        self.current_node, f"{key}_escape", s == 2
                    )
                    if self.main_window and self.current_node
                    else None,
                )
            )
            row_layout.addWidget(escape_chk, 1)

            return row_widget

        except Exception as e:
            LOGGER.error(f"Error creating parameter row for {key}: {e}")
            return None

    def create_typed_widget(self, key, value, annotation):
        """Create typed widget with safe type enforcement and validation."""

        def on_change(v):
            if not self.current_node:
                return
            try:
                if key == "_name":
                    self.current_node.node_data.name = str(v)
                else:
                    # FIX: Strict order of operations. Check Asset FIRST.
                    if TypeSafeParser.is_asset_param(key):
                        # Store the UUID string directly. Do NOT parse as number.
                        self.current_node.node_data.params[key] = v
                    elif TypeSafeParser.is_color_param(key):
                        v = TypeSafeParser.parse_color(v)
                        self.current_node.node_data.params[key] = v
                    elif TypeSafeParser.is_numeric_param(key):
                        v = TypeSafeParser.parse_numeric(v)
                        self.current_node.node_data.params[key] = v
                    else:
                        self.current_node.node_data.params[key] = v

                self.current_node.update()
                self.debouncer.start()
                if self.main_window is not None:
                    try:
                        self.main_window.on_node_property_changed(
                            self.current_node, key, v
                        )
                    except Exception:
                        pass
            except Exception as e:
                LOGGER.warning(f"Value change error for {key}: {e}")

        try:
            #  1. SPECIAL: Target Mobject Selector
            # Check for keys like "mobject", "vmobject", "mobjects"
            key_lower = key.lower()
            target_keywords = ["mobject", "vmobject", "target", "object"]

            # Logic: If it's an Animation node AND key contains one of the keywords
            is_anim = self.current_node.node_data.type == NodeType.ANIMATION
            is_target_param = any(k in key_lower for k in target_keywords)

            if is_anim and is_target_param:
                combo = QComboBox()
                combo.addItem("-- Select Target --", None)

                # Scan main window nodes for Mobjects
                # Note: We need a way to access main_window nodes.
                # Ideally pass main_window to PropertiesPanel or access via scene items.
                scene = self.current_node.scene()
                if scene:
                    for item in scene.items():
                        if (
                            isinstance(item, NodeItem)
                            and item.data.type == NodeType.MOBJECT
                        ):
                            # Add Mobject to dropdown
                            display_name = (
                                f"📦 {item.node_data.name} ({item.node_data.id[:4]})"
                            )
                            combo.addItem(display_name, item.data.id)

                # Set current value
                if value:
                    idx = combo.findData(value)
                    if idx >= 0:
                        combo.setCurrentIndex(idx)

                # AUTO-CONFIGURE: Enable and Escape by default
                self.current_node.node_data.set_param_enabled(key, True)
                self.current_node.node_data.set_escape_string(key, True)

                combo.currentIndexChanged.connect(
                    lambda i: on_change(combo.itemData(i))
                )
                return combo

            # 2. ASSET PATHS
            if TypeSafeParser.is_asset_param(key):
                combo = QComboBox()
                combo.addItem("(None)", None)
                for asset in ASSETS.get_list():
                    # Show Emoji + Name
                    combo.addItem(f"📄 {asset.name}", asset.id)

                if value:
                    idx = combo.findData(value)
                    if idx >= 0:
                        combo.setCurrentIndex(idx)

                combo.currentIndexChanged.connect(
                    lambda i: on_change(combo.itemData(i))
                )
                return combo

            # 3. COLORS
            if TypeSafeParser.is_color_param(key):
                btn = QPushButton("Pick Color")
                _hex_val = str(TypeSafeParser.parse_color(value))
                btn.setStyleSheet(f"background-color: {_hex_val}; color: white;")

                def _make_color_picker(_btn, _initial_hex):
                    def pick_color():
                        col = QColorDialog.getColor(
                            QColor(_initial_hex), None, "Select Color"
                        )
                        if col.isValid():
                            new_hex = col.name()
                            _btn.setStyleSheet(
                                f"background-color: {new_hex}; color: white;"
                            )
                            on_change(new_hex)

                    return pick_color

                btn.clicked.connect(_make_color_picker(btn, _hex_val))
                return btn

            # 4. NUMERIC
            if TypeSafeParser.is_numeric_param(key) or annotation in (float, int):
                # ... (Keep existing numeric logic) ...
                if annotation is float or isinstance(value, float):
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

            # 5. BOOLEAN
            if annotation is bool or isinstance(value, bool):
                chk = QCheckBox()
                chk.setChecked(bool(value))
                chk.stateChanged.connect(lambda s: on_change(s == 2))
                return chk

            # 6. STRING / FALLBACK
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
                    self.current_node.node_data.name = str(val)
                self.current_node.update()
            except Exception as e:
                LOGGER.warning(f"update_param error: {e}")


class ElementsPanel(QWidget):
    add_requested = Signal(str, str)  # Type, Class
    add_structural_requested = Signal(str)  # "play", "wait", "vgroup"

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        # ── Structural Node buttons ──────────────────────────────────────
        struct_box = QGroupBox("Scene Control Nodes")
        struct_box.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid #d1d5db; "
            "border-radius: 5px; margin-top: 8px; padding-top: 4px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; }"
        )
        struct_layout = QHBoxLayout(struct_box)
        struct_layout.setSpacing(4)

        btn_play = QPushButton("▶ Play")
        btn_play.setToolTip(
            "Add explicit play() node (Ctrl+Shift+P)\nConnect Animations to this node's input socket"
        )
        btn_play.setStyleSheet(
            "QPushButton { background: #27ae60; color: white; font-weight: bold; "
            "border-radius: 4px; padding: 5px 8px; }"
            "QPushButton:hover { background: #219a52; }"
        )
        btn_play.clicked.connect(lambda: self.add_structural_requested.emit("play"))
        struct_layout.addWidget(btn_play)

        btn_wait = QPushButton("⏱ Wait")
        btn_wait.setToolTip(
            "Add explicit wait() node (Ctrl+Shift+W)\nSet duration in Properties panel"
        )
        btn_wait.setStyleSheet(
            "QPushButton { background: #e67e22; color: white; font-weight: bold; "
            "border-radius: 4px; padding: 5px 8px; }"
            "QPushButton:hover { background: #ca6f1e; }"
        )
        btn_wait.clicked.connect(lambda: self.add_structural_requested.emit("wait"))
        struct_layout.addWidget(btn_wait)

        btn_vgroup = QPushButton("🔗 VGroup")
        btn_vgroup.setToolTip("Add VGroup node\nConnect Mobjects to group them")
        btn_vgroup.setStyleSheet(
            "QPushButton { background: #16a085; color: white; font-weight: bold; "
            "border-radius: 4px; padding: 5px 8px; }"
            "QPushButton:hover { background: #128678; }"
        )
        btn_vgroup.clicked.connect(lambda: self.add_structural_requested.emit("vgroup"))
        struct_layout.addWidget(btn_vgroup)

        layout.addWidget(struct_box)

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
        if not MANIM_AVAILABLE:
            return
        self.tree.clear()
        mob_root = QTreeWidgetItem(self.tree, ["Mobjects"])
        anim_root = QTreeWidgetItem(self.tree, ["Animations"])

        # Add built-in Manim elements
        for name in dir(manim):
            if name.startswith("_"):
                continue
            obj = getattr(manim, name)
            if inspect.isclass(obj):
                try:
                    if issubclass(obj, manim.Mobject) and obj is not manim.Mobject:
                        QTreeWidgetItem(mob_root, [name])
                    elif (
                        issubclass(obj, manim.Animation) and obj is not manim.Animation
                    ):
                        QTreeWidgetItem(anim_root, [name])
                except Exception:
                    pass

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
                else:
                    item.setHidden(True)
            cat.setHidden(hide)

    def on_dbl_click(self, item, col):
        if item.childCount() == 0:
            p = item.parent()
            parent_text = p.text(0)

            # Determine type based on parent category
            if parent_text == "Mobjects":
                t = "Mobject"
            elif parent_text == "Animations":
                t = "Animation"
            else:
                # Default to trying as Mobject
                t = "Mobject"

            self.add_requested.emit(t, item.text(0))


class AIPanel(QWidget):
    """Enhanced AI Panel with visual distinction and node generation."""

    merge_requested = Signal(str)
    nodes_generated = Signal(dict)  # Emits dict of generated node info

    def __init__(self):
        super().__init__()
        self.worker = None
        self.last_code = None
        self.extracted_nodes = []  # Track AI-generated nodes
        self._mcp_agent = None  # Set by __import__('ui.main_window').main_window.EfficientManimWindow after construction

        # Create main layout with visual distinction
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # ===== HEADER =====
        header = self._create_header()
        main_layout.addWidget(header)

        # ===== CONTENT AREA =====
        content_splitter = QSplitter(Qt.Orientation.Horizontal)

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

        # PDF upload section (inside AI panel)
        self.pdf_upload_widget = AIPDFUploadWidget(self.input, self.output, parent=self)
        left_layout.insertWidget(1, self.pdf_upload_widget)

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
            "QFrame { background-color: #ffffff; border-bottom: 2px solid #cccccc; }"
        )
        header.setMaximumHeight(50)

        layout = QHBoxLayout(header)
        layout.setContentsMargins(12, 8, 12, 8)

        title = QLabel("AI Code Generator")
        font = QFont()
        font.setPointSize(11)
        font.setBold(True)
        title.setFont(font)
        title.setStyleSheet("color: #000000; font-weight: bold;")  # text black

        subtitle = QLabel("Generate Manim code with Gemini AI")
        subtitle.setStyleSheet("color: #555555; font-size: 9pt;")  # dark grey

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addStretch()

        status_label = QLabel("Status: Ready")
        status_label.setStyleSheet("color: #333333;")  # dark grey
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

        label = QLabel("Your Prompt")
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        label.setFont(font)
        label.setStyleSheet("color: #1565c0;")
        layout.addWidget(label)

        self.input = QPlainTextEdit()
        self.input.setPlaceholderText(
            "Describe your animation...\n\nExample: Create a blue rectangle that smoothly rotates 100 degrees."
        )
        self.input.setMaximumHeight(90)
        self.input.setStyleSheet(
            "QPlainTextEdit { border: 1px solid #bdbdbd; border-radius: 3px; "
            "background: white; padding: 6px; }"
        )
        layout.addWidget(self.input)

        # ── Auto Voiceover toggle ──────────────────────────────────
        self.chk_auto_voiceover = QCheckBox("Enable Auto Voiceover")
        self.chk_auto_voiceover.setToolTip(
            "When enabled, Gemini will act as an autonomous agent:\n"
            "it will analyze all animation nodes, extract meaningful text,\n"
            "generate optimized voiceover scripts, generate TTS audio per\n"
            "animation, and automatically attach voiceovers to nodes.\n"
            "The result is a fully voiceover-synced project ready to render."
        )
        self.chk_auto_voiceover.setStyleSheet("font-weight: bold; color: #8e44ad;")
        layout.addWidget(self.chk_auto_voiceover)

        # ── MCP Agent Mode toggle ──────────────────────────────────
        self.chk_mcp_mode = QCheckBox("MCP Agent Mode")
        self.chk_mcp_mode.setToolTip(
            "Instead of generating Manim code, Gemini reads the live scene\n"
            "state and issues MCP commands to directly create/modify/delete\n"
            "nodes in the graph — no code generation, no merge step needed.\n\n"
            "Gemini sees all current nodes, params, and assets before acting."
        )
        self.chk_mcp_mode.setStyleSheet("font-weight: bold; color: #1a73e8;")
        # Only one mode can be active at once
        self.chk_mcp_mode.toggled.connect(
            lambda on: self.chk_auto_voiceover.setEnabled(not on)
        )
        self.chk_auto_voiceover.toggled.connect(
            lambda on: self.chk_mcp_mode.setEnabled(not on)
        )
        layout.addWidget(self.chk_mcp_mode)

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

        label = QLabel("AI Response")
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        label.setFont(font)
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
        self.btn_gen = QPushButton("Generate Code")
        self.btn_gen.setStyleSheet(
            "QPushButton { background: #2196f3; color: white; border: none; "
            "border-radius: 4px; padding: 8px 16px; font-weight: bold; }"
            "QPushButton:hover { background: #1976d2; }"
            "QPushButton:pressed { background: #1565c0; }"
        )
        self.btn_gen.clicked.connect(self.generate)
        apply_tooltip(
            self.btn_gen,
            "Generate code from your prompt",
            "Uses the selected Gemini model",
            None,
            "Generate",
        )
        layout.addWidget(self.btn_gen, 1)

        # Merge button (success)
        self.btn_merge = QPushButton("Merge to Scene")
        self.btn_merge.setEnabled(False)
        self.btn_merge.setStyleSheet(
            "QPushButton { background: #4caf50; color: white; border: none; "
            "border-radius: 4px; padding: 8px 16px; font-weight: bold; }"
            "QPushButton:hover { background: #45a049; }"
            "QPushButton:disabled { background: #bdbdbd; }"
        )
        self.btn_merge.clicked.connect(self.merge)
        apply_tooltip(
            self.btn_merge,
            "Merge generated nodes",
            "Adds AI-generated nodes to the scene",
            None,
            "Merge",
        )
        layout.addWidget(self.btn_merge, 1)

        # Reject button (danger)
        self.btn_reject = QPushButton("Reject")
        self.btn_reject.setEnabled(False)
        self.btn_reject.setStyleSheet(
            "QPushButton { background: #f44336; color: white; border: none; "
            "border-radius: 4px; padding: 8px 16px; font-weight: bold; }"
            "QPushButton:hover { background: #da190b; }"
            "QPushButton:disabled { background: #bdbdbd; }"
        )
        self.btn_reject.clicked.connect(self.reject)
        apply_tooltip(
            self.btn_reject,
            "Discard generated output",
            "Clears the current AI result",
            None,
            "Reject",
        )
        layout.addWidget(self.btn_reject, 1)

        # Clear button (secondary)
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.setStyleSheet(
            "QPushButton { background: #9e9e9e; color: white; border: none; "
            "border-radius: 4px; padding: 8px 16px; }"
            "QPushButton:hover { background: #757575; }"
        )
        self.btn_clear.clicked.connect(lambda: self.output.clear())
        apply_tooltip(
            self.btn_clear,
            "Clear AI output",
            "Removes the current response text",
            None,
            "Clear",
        )
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

        title = QLabel("Generated Nodes")
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        title.setFont(font)
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
        """Generate AI code, run Auto Voiceover agent, or execute MCP Agent Mode."""
        if self.chk_auto_voiceover.isChecked():
            self._run_auto_voiceover_agent()
            return

        if self.chk_mcp_mode.isChecked():
            self._run_mcp_agent_mode()
            return

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
        selected_model = str(
            SETTINGS.get("GEMINI_MODEL", "gemini-3-flash-preview")
            or "gemini-3-flash-preview"
        )
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
            f"class MyScene(Scene):\n"
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
        selected_model = str(SETTINGS.get("GEMINI_MODEL", "gemini-3-flash-preview"))
        self.worker = AIWorker(sys_prompt, model=selected_model)
        self.worker.chunk_received.connect(self.on_chunk)
        self.worker.finished_signal.connect(self.on_finish)
        self.worker.start()

    def on_chunk(self, txt):
        """Handle streamed AI response."""
        self.output.moveCursor(QTextCursor.MoveOperation.End)
        self.output.insertPlainText(txt)

    def on_finish(self):
        """Parse generated code and extract nodes."""
        self.btn_gen.setEnabled(True)
        full = self.output.toPlainText()
        match = re.findall(r"```python(.*?)```", full, re.DOTALL)

        if match:
            raw_code = match[-1].strip()
            self.last_code = self._normalize_scene_class_name(raw_code)
            self._extract_nodes_from_code(self.last_code)

            if self.extracted_nodes:
                self.btn_merge.setEnabled(True)
                self.btn_reject.setEnabled(True)
                self.status_label.setText(
                    f"Status: Ready to merge ({len(self.extracted_nodes)} nodes)"
                )
                LOGGER.ai(
                    f"Code block ready. Extracted {len(self.extracted_nodes)} nodes."
                )
            else:
                self.status_label.setText("Status: Code ready (no nodes detected)")
        else:
            self.status_label.setText("Status: No code block found")

    def _normalize_scene_class_name(self, code: str) -> str:
        """Force the scene class name to GeneratedScene(Scene) for AI output."""
        pattern = re.compile(
            r"^(\s*)class\s+([A-Za-z_]\w*)\s*\(([^)]*)\)\s*:",
            re.MULTILINE,
        )

        def repl(match):
            bases = match.group(3)
            if "Scene" not in bases:
                return match.group(0)
            indent = match.group(1)
            return f"{indent}class GeneratedScene(Scene):"

        updated, _ = pattern.subn(repl, code, count=1)
        return updated

    def _extract_nodes_from_code(self, code: str):
        """Extract BOTH mobjects and animations from AI-generated code."""
        self.extracted_nodes = []
        self.nodes_list.clear()

        # Extract all object definitions
        pattern = r"(\w+)\s*=\s*([A-Z][a-zA-Z0-9]*)\s*\((.*?)\)(?:\s|$)"
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
                    is_animation = (
                        issubclass(cls, manim.Animation)
                        if hasattr(manim, "Animation")
                        else "Animation" in class_name
                    )
            except Exception:
                pass

            self.extracted_nodes.append(
                {
                    "var_name": var_name,
                    "class_name": class_name,
                    "params": params,
                    "params_str": params_str,
                    "source": "ai",
                    "type": "animation" if is_animation else "mobject",
                }
            )

            # Add to list with icon indicating type
            node_icon = "🎬" if is_animation else "📦"
            node_type_label = "Animation" if is_animation else "Mobject"
            item = QListWidgetItem(
                f"{node_icon} {var_name}: {class_name} ({param_count} params) [{node_type_label}]"
            )

            # Color code: blue for mobjects, purple for animations
            if is_animation:
                item.setBackground(QColor("#f3e5f5"))
                item.setForeground(QColor("#7b1fa2"))
            else:
                item.setBackground(QColor("#e3f2fd"))
                item.setForeground(QColor("#1565c0"))

            # Build detailed tooltip with parameter list
            param_list = "\n".join(
                [f"  • {k}={v[:40]}" for k, v in list(params.items())[:5]]
            )
            if len(params) > 5:
                param_list += f"\n  ... and {len(params) - 5} more"

            item.setToolTip(
                f"AI Generated {node_type_label}\n"
                f"Variable: {var_name}\n"
                f"Type: {class_name}\n"
                f"Parameters ({param_count}):\n{param_list}\n\n"
                f"All parameters are captured and ready to use."
            )
            self.nodes_list.addItem(item)

        # Also show animations from self.play() calls
        pattern_play = r"self\.play\((.*?)\)(?=\s|$)"
        for match in re.finditer(pattern_play, code, re.DOTALL):
            play_content = match.group(1)
            anim_pattern = r"([A-Z][a-zA-Z0-9]*)\((.*?)\)"

            for anim_match in re.finditer(anim_pattern, play_content):
                anim_class = anim_match.group(1)

                # Skip if not a Manim animation class
                if not hasattr(manim, anim_class):
                    continue

                try:
                    cls = getattr(manim, anim_class)
                    is_anim = (
                        issubclass(cls, manim.Animation)
                        if hasattr(manim, "Animation")
                        else "Animation" in anim_class
                    )

                    if is_anim and not any(
                        n["class_name"] == anim_class and n["type"] == "animation"
                        for n in self.extracted_nodes
                    ):
                        # Add this animation if not already added
                        self.extracted_nodes.append(
                            {
                                "var_name": f"{anim_class.lower()}_1",
                                "class_name": anim_class,
                                "params": {},
                                "source": "ai",
                                "type": "animation",
                            }
                        )

                        item = QListWidgetItem(
                            f"🎬 {anim_class.lower()}_1: {anim_class} (from self.play)"
                        )
                        item.setBackground(QColor("#f3e5f5"))
                        item.setForeground(QColor("#7b1fa2"))
                        item.setToolTip(
                            f"Animation: {anim_class}\nExtracted from self.play() call"
                        )
                        self.nodes_list.addItem(item)
                except Exception:
                    pass

    def _parse_node_parameters(self, params_str: str) -> dict:
        """Parse parameters from node definition string, handling nested structures."""
        params = {}

        # Split by comma, but respect parentheses/brackets nesting
        depth = 0
        current_item = ""

        for char in params_str:
            if char in "([{":
                depth += 1
                current_item += char
            elif char in ")]}":
                depth -= 1
                current_item += char
            elif char == "," and depth == 0:
                # End of parameter
                if "=" in current_item:
                    try:
                        key, value = current_item.split("=", 1)
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
        if current_item and "=" in current_item:
            try:
                key, value = current_item.split("=", 1)
                key = key.strip()
                value = value.strip()
                params[key] = value
            except ValueError:
                pass

        return params

    def _run_auto_voiceover_agent(self):
        """Gemini auto-voiceover agent: analyze all nodes, generate and attach voiceovers.

        This is the 'Enable Auto Voiceover' mode. Gemini acts as an autonomous
        agent that:
          1. Inspects all animation nodes in the scene.
          2. Extracts meaningful text / context from each node.
          3. Generates an optimized voiceover script per node.
          4. Generates TTS audio for each script via TTSWorker.
          5. Attaches audio to each node and syncs durations.
        """
        main_window = self._get_main_window()
        if main_window is None:
            QMessageBox.critical(self, "Error", "Cannot find main window reference.")
            return

        nodes: dict = main_window.nodes
        if not nodes:
            QMessageBox.warning(
                self, "No Nodes", "There are no nodes in the scene to voiceover."
            )
            return

        self.output.clear()
        self.btn_gen.setEnabled(False)
        self.status_label.setText("Status: Auto Voiceover Agent running…")
        self.output.append("<b style='color:#8e44ad;'>🤖 Auto Voiceover Agent</b><br>")
        self.output.append(f"Analyzing <b>{len(nodes)}</b> nodes…<br>")

        # Build node context for Gemini
        node_summaries = []
        for nid, node_item in nodes.items():
            d = node_item.data
            params_str = (
                ", ".join(f"{k}={v}" for k, v in list(d.params.items())[:5])
                if d.params
                else "no params"
            )
            node_summaries.append(
                f"  - [{d.type.name}] {d.name} ({d.cls_name}): {params_str}"
            )

        node_context = "\n".join(node_summaries)

        agent_prompt = (
            "You are an expert animation narrator and voiceover script writer.\n\n"
            "Here is a list of animation nodes in a Manim scene:\n\n"
            f"{node_context}\n\n"
            "Your job:\n"
            "For EACH node listed above, write a short, engaging voiceover script (1-2 sentences max).\n"
            "The script should naturally describe what the animation is doing.\n"
            "Use plain, spoken English — no markdown, no code.\n\n"
            "Output ONLY a JSON array, with one object per node, in this exact format:\n"
            "[\n"
            '  {"node_name": "<exact node name>", "script": "<voiceover script>"},\n'
            "  ...\n"
            "]\n\n"
            "Output ONLY the JSON array. No explanation, no markdown fences."
        )

        selected_model = str(SETTINGS.get("GEMINI_MODEL", "gemini-3-flash-preview"))
        self._auto_vo_nodes = nodes
        self._auto_vo_worker = AIWorker(agent_prompt, model=selected_model)
        self._auto_vo_buffer = ""
        self._auto_vo_worker.chunk_received.connect(self._on_auto_vo_chunk)
        self._auto_vo_worker.finished_signal.connect(self._on_auto_vo_scripts_ready)
        self._auto_vo_worker.start()

    def _on_auto_vo_chunk(self, txt: str):
        self._auto_vo_buffer += txt
        self.output.moveCursor(QTextCursor.MoveOperation.End)
        self.output.insertPlainText(txt)

    def _on_auto_vo_scripts_ready(self):
        """Parse Gemini's JSON response and kick off TTS generation per node."""
        raw = self._auto_vo_buffer.strip()

        # Strip markdown fences if present
        raw = re.sub(r"```json|```", "", raw).strip()

        try:
            scripts: list[dict] = json.loads(raw)
        except json.JSONDecodeError as e:
            self.btn_gen.setEnabled(True)
            self.status_label.setText("Status: Agent parse error.")
            LOGGER.error(f"Auto Voiceover agent parse error: {e}\nRaw: {raw[:200]}")
            self.output.append(
                "<br><span style='color:red'>Failed to parse scripts from Gemini. "
                "Check Logs for details.</span>"
            )
            return

        if not scripts:
            self.btn_gen.setEnabled(True)
            self.status_label.setText("Status: Agent returned empty scripts.")
            return

        self.output.append(
            f"<br><b>Scripts generated for {len(scripts)} nodes.</b> Starting TTS…<br>"
        )
        self._auto_vo_queue = list(scripts)
        self._auto_vo_node_map = {
            node_item.data.name: node_item for node_item in self._auto_vo_nodes.values()
        }
        self._auto_vo_voice = SETTINGS.get("DEFAULT_VOICE", "Zephyr")
        self._auto_vo_model = SETTINGS.get("TTS_MODEL", "gemini-2.5-flash-preview-tts")
        self._auto_vo_active_worker = None
        self._auto_vo_index = 0
        self._process_next_auto_vo()

    def _process_next_auto_vo(self):
        """Process TTS generation for the next item in the queue."""
        if not self._auto_vo_queue:
            self._finish_auto_voiceover()
            return

        item = self._auto_vo_queue.pop(0)
        node_name = item.get("node_name", "")
        script = item.get("script", "")
        self._auto_vo_index += 1

        if not script or node_name not in self._auto_vo_node_map:
            self.output.append(
                f"<span style='color:orange'>Warning: Skipping '{node_name}' — "
                f"{'no script' if not script else 'node not found'}</span><br>"
            )
            QTimer.singleShot(100, self._process_next_auto_vo)
            return

        self.output.append(
            f"<span style='color:#1a73e8'>🎙 [{self._auto_vo_index}] {node_name}</span>: "
            f"{script}<br>"
        )
        self.status_label.setText(f"Status: TTS for '{node_name}'…")

        worker = TTSWorker(script, self._auto_vo_voice, self._auto_vo_model)
        # Store metadata on the worker for the callback to access
        worker._target_node_name = node_name
        worker._script = script
        worker.finished_signal.connect(
            lambda path, n=node_name, s=script: self._on_auto_vo_tts_done(path, n, s)
        )
        worker.error_signal.connect(
            lambda err, n=node_name: self._on_auto_vo_tts_error(err, n)
        )
        self._auto_vo_active_worker = worker
        worker.start()

    def _on_auto_vo_tts_done(self, file_path: str, node_name: str, script: str):
        """Attach generated audio to target node and process the next one."""
        asset = ASSETS.add_asset(file_path)
        node_item = self._auto_vo_node_map.get(node_name)

        if asset and node_item:
            node_item.data.audio_asset_id = asset.id
            node_item.data.voiceover_transcript = script
            # Estimate duration from pydub if available
            if PYDUB_AVAILABLE:
                try:
                    from pydub import AudioSegment as _AS

                    seg = _AS.from_file(file_path)
                    node_item.data.voiceover_duration = len(seg) / 1000.0
                except Exception:
                    pass
            node_item.update()
            self.output.append(
                f"<span style='color:#27ae60'>Attached to '{node_name}'</span><br>"
            )
        else:
            self.output.append(
                f"<span style='color:red'>Failed to attach to '{node_name}'</span><br>"
            )

        QTimer.singleShot(200, self._process_next_auto_vo)

    def _on_auto_vo_tts_error(self, err: str, node_name: str):
        self.output.append(
            f"<span style='color:red'>TTS error for '{node_name}': {err}</span><br>"
        )
        LOGGER.error(f"Auto Voiceover TTS error for {node_name}: {err}")
        QTimer.singleShot(200, self._process_next_auto_vo)

    def _finish_auto_voiceover(self):
        """Finalize auto-voiceover run."""
        main_window = self._get_main_window()
        if main_window:
            main_window.compile_graph()
            main_window.mark_modified()

        self.btn_gen.setEnabled(True)
        self.status_label.setText("Status: Auto Voiceover complete")
        self.output.append(
            "<br><b style='color:#27ae60;font-size:13px;'>"
            "🎬 Auto Voiceover complete! All nodes have been synced and the project is render-ready."
            "</b><br>"
        )
        QMessageBox.information(
            self,
            "Auto Voiceover Complete",
            "All animation nodes have been voiceover-synced.\n"
            "The project is now render-ready with synchronized audio.",
        )

    # ══════════════════════════════════════════════════════════════
    # MCP AGENT INTEGRATION
    # ══════════════════════════════════════════════════════════════

    def set_mcp_agent(self, agent) -> None:
        """Called by __import__('ui.main_window').main_window.EfficientManimWindow after construction to inject the live agent."""
        self._mcp_agent = agent
        LOGGER.info("AIPanel: MCP Agent connected.")
        # Enable MCP mode checkbox now that an agent is available
        if hasattr(self, "chk_mcp_mode"):
            self.chk_mcp_mode.setEnabled(True)

    def _get_mcp_agent(self):
        """Return the MCP agent, trying to lazy-init if not yet injected."""
        if self._mcp_agent is not None:
            return self._mcp_agent
        # Fallback: walk widget tree and grab agent from window
        win = self._get_main_window()
        if win is not None and hasattr(win, "mcp") and win.mcp is not None:
            self._mcp_agent = win.mcp
            return self._mcp_agent
        return None

    def _run_mcp_agent_mode(self) -> None:
        """
        MCP Agent Mode: Gemini reads the live scene state (via get_context),
        then outputs a JSON list of MCP commands which are executed directly
        against the running application — no code generation, no merge step.
        """
        txt = self.input.toPlainText().strip()
        if not txt:
            self.status_label.setText("Status: Empty prompt")
            return

        agent = self._get_mcp_agent()
        if agent is None:
            if not MCP_AVAILABLE:
                QMessageBox.critical(
                    self,
                    "MCP Not Available",
                    "mcp.py was not found next to main.py.\n"
                    "Make sure mcp.py is in the same directory and restart.",
                )
            else:
                QMessageBox.critical(
                    self,
                    "MCP Error",
                    "MCP Agent is not initialised yet.\n"
                    "Please wait for the application to finish loading.",
                )
            self.chk_mcp_mode.setChecked(False)
            return

        self.output.clear()
        self.nodes_list.clear()
        self.btn_gen.setEnabled(False)
        self.status_label.setText("Status: MCP Agent running…")
        self.output.append(
            "<b style='color:#1a73e8;font-size:13px;'>🔌 MCP Agent Mode</b><br>"
            f"<b>Instruction:</b> {txt}<br><br>"
        )
        self.input.clear()

        # ── 1. Capture live scene state ────────────────────────────────────
        ctx_result = agent.execute("get_context")
        ctx_json = (
            json.dumps(ctx_result.data, indent=2, default=str)
            if ctx_result.success
            else "{}"
        )

        cmds_result = agent.execute("list_commands")
        available_commands: list = cmds_result.data if cmds_result.success else []

        # Commands Gemini is allowed to use (exclude destructive ops unless asked)
        safe_commands = [c for c in available_commands if c not in ("clear_scene",)]

        self._mcp_ctx_before = ctx_result.data  # save for diff display later

        # ── 2. Build the Gemini prompt ─────────────────────────────────────
        selected_model = str(SETTINGS.get("GEMINI_MODEL", "gemini-3-flash-preview"))

        system_prompt = (
            "You are an autonomous animation editing agent for EfficientManim.\n\n"
            f"Available MCP commands:\n{json.dumps(safe_commands, indent=2)}\n\n"
            "Command payload reference:\n"
            "  create_node:      {cls_name, name, node_type (ANIMATION|MOBJECT), params: {}, x, y}\n"
            "  set_node_param:   {node_id, key, value}\n"
            "  rename_node:      {node_id, name}\n"
            "  delete_node:      {node_id, confirm: true}  ← only if user explicitly asks to delete\n"
            "  attach_voiceover: {node_id, audio_path, transcript, duration}\n"
            "  remove_voiceover: {node_id}\n"
            "  select_node:      {node_id}\n"
            "  switch_tab:       {tab}  ← partial name ok, e.g. 'Properties'\n"
            "  compile_graph:    {}\n"
            "  trigger_render:   {node_id (optional)}\n"
            "  save_project:     {}\n"
            "  switch_scene:     {scene_name}\n\n"
            "Rules:\n"
            "1. Output ONLY a valid JSON array. No explanation. No markdown. No fences.\n"
            '2. Each element: {"command": "...", "payload": {...}}\n'
            "3. For nodes you create in this project, use 'node_name' key instead of "
            "'node_id' in subsequent commands — it will be resolved automatically.\n"
            "4. Always end with compile_graph if you modified the scene.\n"
            "5. Use node IDs from the current state for existing nodes.\n"
            "6. NEVER issue clear_scene or delete_node unless the user explicitly asked.\n\n"
            f"Current project state:\n{ctx_json}\n\n"
            f"USER INSTRUCTION: {txt}\n\n"
            "Output the JSON array now. Start with [ and end with ]. Nothing else."
        )

        # ── 3. Stream Gemini response ──────────────────────────────────────
        self._mcp_buffer = ""

        def on_chunk(text: str) -> None:
            self._mcp_buffer += text
            self.output.moveCursor(QTextCursor.MoveOperation.End)
            self.output.insertPlainText(text)

        def on_finish() -> None:
            self._execute_mcp_commands_from_buffer()
            self.btn_gen.setEnabled(True)

        self.worker = AIWorker(system_prompt, model=selected_model)
        self.worker.chunk_received.connect(on_chunk)
        self.worker.finished_signal.connect(on_finish)
        self.worker.start()

    def _execute_mcp_commands_from_buffer(self) -> None:
        """
        Parse Gemini's JSON command list and execute each one through MCPAgent.
        Handles node name → ID resolution for nodes created in the same batch.
        """
        agent = self._get_mcp_agent()
        if agent is None:
            self.status_label.setText("Status: MCP agent lost.")
            return

        raw = self._mcp_buffer.strip()

        # Strip any markdown fences Gemini occasionally adds
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()

        # Try to extract a JSON array substring as a fallback
        def _try_parse(text: str):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                start = text.find("[")
                end = text.rfind("]") + 1
                if start != -1 and end > start:
                    return json.loads(text[start:end])
                raise

        try:
            commands: list = _try_parse(raw)
        except json.JSONDecodeError as e:
            self.status_label.setText("Status: Parse error.")
            self.output.append(
                f"<br><span style='color:red;'>Could not parse Gemini's command list: {e}</span><br>"
                "<span style='color:#888;'>Raw output logged to console.</span>"
            )
            LOGGER.error(f"MCP agent parse error: {e}\nRaw:\n{raw[:800]}")
            return

        if not isinstance(commands, list) or len(commands) == 0:
            self.output.append(
                "<br><span style='color:orange;'>Warning: Gemini returned no commands.</span>"
            )
            self.status_label.setText("Status: No commands to execute.")
            return

        self.output.append(
            f"<br><b style='color:#1a73e8;'>Executing {len(commands)} command(s):</b><br>"
        )

        # ── Node name → ID resolution table ───────────────────────────────
        # Populated as create_node results come in; used to patch node_name refs.
        name_to_id: dict[str, str] = {}

        success_count = 0
        fail_count = 0

        for cmd_obj in commands:
            if not isinstance(cmd_obj, dict):
                continue

            command: str = cmd_obj.get("command", "")
            payload: dict = dict(cmd_obj.get("payload", {}))

            if not command:
                continue

            # Resolve node_name → node_id if Gemini used a name for a just-created node
            if "node_name" in payload and "node_id" not in payload:
                resolved_id = name_to_id.get(payload.pop("node_name"))
                if resolved_id:
                    payload["node_id"] = resolved_id
                else:
                    self.output.append(
                        f"<span style='color:orange;'>Warning: {command} — "
                        f"could not resolve node_name to an id, skipping.</span><br>"
                    )
                    fail_count += 1
                    continue

            result = agent.execute(command, payload)

            if result.success:
                # If we just created a node, remember its id by name
                if command == "create_node" and isinstance(result.data, dict):
                    created_name = payload.get("name", "")
                    created_id = result.data.get("id", "")
                    if created_name and created_id:
                        name_to_id[created_name] = created_id

                data_str = str(result.data) if result.data else ""
                self.output.append(
                    f"<span style='color:#27ae60;'>{command}</span>"
                    + (
                        f" <span style='color:#555;font-size:10px;'>→ {data_str[:80]}</span>"
                        if data_str
                        else ""
                    )
                    + "<br>"
                )
                success_count += 1
            else:
                self.output.append(
                    f"<span style='color:red;'>{command} — {result.error}</span><br>"
                )
                fail_count += 1

        # ── Show node count delta ──────────────────────────────────────────
        try:
            after = agent.execute("get_context")
            if (
                after.success
                and hasattr(self, "_mcp_ctx_before")
                and self._mcp_ctx_before
            ):
                before_count = self._mcp_ctx_before.get("node_count", 0)
                after_count = after.data.get("node_count", 0)
                delta = after_count - before_count
                if delta > 0:
                    self.output.append(
                        f"<span style='color:#27ae60;'> {delta} new node(s) added.</span><br>"
                    )
                elif delta < 0:
                    self.output.append(
                        f"<span style='color:orange;'> {abs(delta)} node(s) removed.</span><br>"
                    )
        except Exception:
            pass

        status_icon = "OK" if fail_count == 0 else "Warning"
        self.status_label.setText(
            f"Status: {status_icon} Done — {success_count} ok, {fail_count} failed."
        )
        self.output.append(
            f"<br><b style='color:#1a73e8;'>"
            f"🔌 MCP Agent finished — {success_count}/{success_count + fail_count} commands succeeded."
            f"</b>"
        )

    def _get_main_window(self):
        """Walk up the widget tree to find the EfficientManimWindow."""
        widget = self.parent()
        while widget is not None:
            if type(widget).__name__ == "EfficientManimWindow":
                return widget
            widget = widget.parent()
        return None

    def merge(self):
        """Emit merge signal with code."""
        if self.last_code:
            self.merge_requested.emit(self.last_code)
            # Signal node generation for UI update
            if self.extracted_nodes:
                self.nodes_generated.emit(
                    {"code": self.last_code, "nodes": self.extracted_nodes}
                )

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
        Handles:
          - var = ClassName(...) definitions
          - self.play(AnimationClass(target, ...)) inline calls (multiple allowed)
          - self.play(target.animate.method(...)) calls
          - self.wait(...) calls

        Returns: (mobjects, animations, play_sequence)
        Where play_sequence = list of play/wait entries in execution order
        """
        mobjects = []
        animations = []
        play_sequence = []
        mobject_vars = {}  # var_name -> class_name
        anim_vars = {}  # var_name -> animation entry

        KNOWN_ANIMS = {
            "FadeIn",
            "FadeOut",
            "Write",
            "DrawBorderThenFill",
            "Create",
            "Transform",
            "ReplacementTransform",
            "Rotate",
            "Scale",
            "ScaleInPlace",
            "MoveTo",
            "ApplyMethod",
            "Indicate",
            "FocusOn",
            "Circumscribe",
            "ShowCreation",
            "Uncreate",
            "GrowFromCenter",
            "ShrinkToCenter",
            "Succession",
            "AnimationGroup",
        }

        def _balanced_paren_end(s, start):
            """Return index after the matching closing paren starting at start."""
            depth = 1
            i = start
            while i < len(s) and depth > 0:
                if s[i] == "(":
                    depth += 1
                elif s[i] == ")":
                    depth -= 1
                i += 1
            return i

        def _is_animation_class(class_name):
            if class_name in KNOWN_ANIMS:
                return True
            try:
                cls = getattr(manim, class_name, None)
                if cls is None:
                    return False
                return issubclass(cls, manim.Animation)
            except Exception:
                return False

        def _split_top_level_args(s: str) -> list[str]:
            args = []
            buf = []
            depth = 0
            in_str = False
            str_char = ""
            esc = False
            for ch in s:
                if in_str:
                    buf.append(ch)
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == str_char:
                        in_str = False
                    continue
                if ch in ("'", '"'):
                    in_str = True
                    str_char = ch
                    buf.append(ch)
                    continue
                if ch in "([{":
                    depth += 1
                elif ch in ")]}":
                    depth -= 1 if depth > 0 else 0
                if ch == "," and depth == 0:
                    arg = "".join(buf).strip()
                    if arg:
                        args.append(arg)
                    buf = []
                else:
                    buf.append(ch)
            tail = "".join(buf).strip()
            if tail:
                args.append(tail)
            return args

        def _is_kw_arg(token: str) -> bool:
            return bool(re.match(r"^[A-Za-z_]\w*\s*=", token))

        def _parse_kw_arg(token: str) -> tuple[str, str] | None:
            if not _is_kw_arg(token):
                return None
            key, value = token.split("=", 1)
            return key.strip(), value.strip()

        def _match_class_call(token: str) -> tuple[str, str] | None:
            m = re.match(r"([A-Z][a-zA-Z0-9]*)\s*\(", token)
            if not m:
                return None
            class_name = m.group(1)
            start = m.end()
            end = _balanced_paren_end(token, start)
            if end <= start:
                return None
            inner = token[start : end - 1]
            return class_name, inner

        # ── Step 1: Variable assignments ──────────────────────────────────
        for m in re.finditer(
            r"^[ \t]*(\w+)\s*=\s*([A-Z][a-zA-Z0-9]*)\s*\(", code, re.MULTILINE
        ):
            var_name, class_name = m.groups()
            if not hasattr(manim, class_name):
                continue
            end_idx = _balanced_paren_end(code, m.end())
            params_str = code[m.end() : end_idx - 1]
            params = AINodeIntegrator._parse_params(params_str)

            is_anim = _is_animation_class(class_name)
            entry = {
                "var_name": var_name,
                "class_name": class_name,
                "params": params,
                "raw_args": params_str,
                "source": "ai",
                "code_snippet": code[m.start() : end_idx],
            }
            if is_anim:
                animations.append(entry)
                anim_vars[var_name] = entry
            else:
                mobjects.append(entry)
                mobject_vars[var_name] = class_name

        inline_counter = 0
        anim_counter = 0

        def _create_inline_mobject(class_name: str, params_str: str) -> str:
            nonlocal inline_counter
            inline_counter += 1
            var_name = f"inline_{class_name.lower()}_{inline_counter}"
            entry = {
                "var_name": var_name,
                "class_name": class_name,
                "params": AINodeIntegrator._parse_params(params_str),
                "raw_args": params_str,
                "source": "ai",
                "code_snippet": f"{class_name}({params_str})",
            }
            mobjects.append(entry)
            mobject_vars[var_name] = class_name
            return var_name

        def _extract_target_vars(arg_str: str) -> list[str]:
            targets = []
            for tok in _split_top_level_args(arg_str):
                tok = tok.strip()
                if not tok:
                    continue
                tok = tok.lstrip("*").strip()
                if not tok:
                    continue
                if _is_kw_arg(tok):
                    continue
                if re.match(r"^[A-Za-z_]\w*$", tok) and tok in mobject_vars:
                    targets.append(tok)
                    continue
                cc = _match_class_call(tok)
                if cc:
                    cname, cargs = cc
                    if hasattr(manim, cname) and not _is_animation_class(cname):
                        targets.append(_create_inline_mobject(cname, cargs))
                        continue
            # Fallback: scan for any referenced mobject vars
            for mob_var in mobject_vars:
                if mob_var in targets:
                    continue
                if re.search(r"\b" + re.escape(mob_var) + r"\b", arg_str):
                    targets.append(mob_var)
            return targets

        def _parse_play_kwargs(args: list[str]) -> dict:
            play_kwargs = {}
            for tok in args:
                kw = _parse_kw_arg(tok)
                if not kw:
                    continue
                key, val = kw
                if key not in ("run_time", "rate_func", "lag_ratio"):
                    continue
                if key == "run_time":
                    try:
                        play_kwargs[key] = float(val)
                    except Exception:
                        play_kwargs[key] = val.strip("'\"")
                else:
                    play_kwargs[key] = val.strip()
            return play_kwargs

        def _make_anim_entry(
            anim_class: str,
            anim_var: str,
            anim_args: str,
            is_chain: bool,
            display_override: str | None = None,
        ) -> dict:
            params = AINodeIntegrator._parse_params(anim_args)
            targets = _extract_target_vars(anim_args)
            return {
                "anim_class": anim_class,
                "anim_var": anim_var,
                "target_vars": targets,
                "params": params,
                "raw": anim_args,
                "is_animate_chain": is_chain,
                "display_override": display_override,
            }

        # ── Step 2: self.play()/self.wait() calls in order ───────────────
        for m in re.finditer(r"self\.(play|wait)\(", code):
            call_type = m.group(1)
            end_idx = _balanced_paren_end(code, m.end())
            raw_args = code[m.end() : end_idx - 1].strip()

            if call_type == "wait":
                wait_args = _split_top_level_args(raw_args)
                duration = 1.0
                if wait_args:
                    first = wait_args[0].strip()
                    kw = _parse_kw_arg(first)
                    if kw and kw[0] == "duration":
                        first = kw[1]
                    try:
                        duration = float(first)
                    except Exception:
                        duration = first.strip("'\"")
                play_sequence.append(
                    {"kind": "wait", "duration": duration, "raw": raw_args}
                )
                continue

            # play(...)
            args = _split_top_level_args(raw_args)
            play_kwargs = _parse_play_kwargs(args)
            anim_entries = []

            for arg in args:
                if _is_kw_arg(arg):
                    # already handled in play_kwargs
                    continue
                arg_clean = arg.strip().lstrip("*").strip()
                if not arg_clean:
                    continue

                # .animate chain: var.animate.method(args)
                animate_m = re.match(
                    r"(\w+)\.animate\.([\w]+)\((.*)\)$", arg_clean, re.DOTALL
                )
                if animate_m:
                    target_var, method, method_args = animate_m.groups()
                    anim_counter += 1
                    anim_entries.append(
                        _make_anim_entry(
                            "ApplyMethod",
                            f"{target_var}_animate_{method}_{anim_counter}",
                            method_args,
                            True,
                            display_override=f"animate.{method}",
                        )
                    )
                    # Ensure target is explicitly connected
                    if anim_entries[-1]["target_vars"] == [] and target_var:
                        anim_entries[-1]["target_vars"] = [target_var]
                    continue

                # Animation variable reference (e.g., self.play(anim1))
                if re.match(r"^[A-Za-z_]\w*$", arg_clean) and arg_clean in anim_vars:
                    anim_def = anim_vars[arg_clean]
                    anim_class = anim_def["class_name"]
                    anim_args = anim_def.get("raw_args", "")
                    anim_entries.append(
                        _make_anim_entry(anim_class, arg_clean, anim_args, False, None)
                    )
                    continue

                # Standard AnimClass(args)
                cc = _match_class_call(arg_clean)
                if cc:
                    anim_class, anim_args = cc
                    if not hasattr(manim, anim_class):
                        continue
                    if not _is_animation_class(anim_class):
                        continue
                    anim_counter += 1
                    anim_entries.append(
                        _make_anim_entry(
                            anim_class,
                            f"{anim_class.lower()}_{anim_counter}",
                            anim_args,
                            False,
                            None,
                        )
                    )

            play_sequence.append(
                {
                    "kind": "play",
                    "animations": anim_entries,
                    "play_kwargs": play_kwargs,
                    "raw": raw_args,
                }
            )

        return mobjects, animations, play_sequence

    @staticmethod
    def _parse_params(params_str: str) -> dict:
        """Parse parameter string into dict with proper quote/constant handling."""
        params = {}

        # Enhanced key=value extraction
        for item in params_str.split(","):
            item = item.strip()
            if "=" in item:
                try:
                    key, value = item.split("=", 1)
                    key = key.strip()
                    value = value.strip()

                    # Clean quotes from strings
                    if (value.startswith('"') and value.endswith('"')) or (
                        value.startswith("'") and value.endswith("'")
                    ):
                        value = value[1:-1]

                    # Store cleaned value
                    params[key] = value
                except ValueError:
                    pass

        return params

    @staticmethod
    def create_node_from_ai(
        var_name: str,
        class_name: str,
        params: dict,
        scene_graph,
        node_type=NodeType.MOBJECT,
        pos: "tuple[int, int]" = (50, 50),
        override_cls_name: str = None,
    ) -> "NodeItem":
        """
        Create a node in the scene graph from AI definition.

        Args:
            var_name: Variable name (e.g., 'circle')
            class_name: Manim class name (e.g., 'Circle')
            params: Parameter dict
            scene_graph: The SceneGraph instance
            node_type: NodeType.MOBJECT or NodeType.ANIMATION
            pos: (x, y) position in scene
            override_cls_name: Display class name override (e.g. 'animate.shift')

        Returns:
            NodeItem: The created node
        """
        # Create NodeData with AI metadata
        display_cls = override_cls_name if override_cls_name else class_name
        node_data = NodeData(var_name, node_type, display_cls)
        node_data.pos_x, node_data.pos_y = pos

        # Apply parameters with type safety
        for param_name, param_value in params.items():
            param_value = str(param_value).strip()
            if TypeSafeParser.is_color_param(param_name):
                clean_value = param_value.strip("'\"")
                node_data.params[param_name] = TypeSafeParser.parse_color(clean_value)
            elif TypeSafeParser.is_numeric_param(param_name):
                clean_value = param_value.strip("'\"")
                node_data.params[param_name] = TypeSafeParser.parse_numeric(clean_value)
            else:
                clean_value = param_value.strip("'\"")
                node_data.params[param_name] = clean_value

        # Mark as AI-generated
        node_data.is_ai_generated = True
        node_data.ai_source = class_name

        # Create NodeItem
        item = NodeItem(node_data)
        try:
            item._window = scene_graph
        except Exception:
            pass

        # Add to scene graph
        scene_graph.scene.addItem(item)
        scene_graph.nodes[item.data.id] = item

        # Auto-detect and load all class parameters (for real Manim classes)
        if hasattr(manim, class_name):
            AINodeIntegrator._load_class_parameters(node_data, class_name)

        return item

    @staticmethod
    def create_structural_node_from_ai(
        name: str,
        node_type: NodeType,
        scene_graph,
        pos: "tuple[int, int]" = (50, 50),
        params: dict | None = None,
        cls_name: str | None = None,
    ) -> "NodeItem":
        """Create PLAY/WAIT nodes for AI-generated graphs."""
        display_cls = cls_name
        if display_cls is None:
            if node_type == NodeType.PLAY:
                display_cls = "play()"
            elif node_type == NodeType.WAIT:
                display_cls = "wait()"
            else:
                display_cls = "struct"

        node_data = NodeData(name, node_type, display_cls)
        node_data.pos_x, node_data.pos_y = pos
        if params is None:
            params = {"duration": 1.0} if node_type == NodeType.WAIT else {}
        node_data.params = dict(params)

        node_data.is_ai_generated = True
        node_data.ai_source = display_cls

        item = NodeItem(node_data)
        try:
            item._window = scene_graph
        except Exception:
            pass
        scene_graph.scene.addItem(item)
        scene_graph.nodes[item.data.id] = item
        return item

    @staticmethod
    def _load_class_parameters(node_data: "NodeData", class_name: str):
        """
        Only validate explicitly provided parameters.
        Do NOT auto-load unused function parameters or defaults.
        This prevents inspect._empty and unused parameter injection.
        """
        try:
            cls = getattr(manim, class_name, None)
            if not cls:
                return

            sig = inspect.signature(cls.__init__)

            # Track which parameters were explicitly provided by AI
            for param_name in list(node_data.params.keys()):
                param = sig.parameters.get(param_name)
                if not param:
                    # Parameter doesn't exist in class, remove it
                    del node_data.params[param_name]
                    continue

                # Skip special parameters
                if param_name in ("self", "args", "kwargs", "mobject"):
                    del node_data.params[param_name]
                    continue

                # Validate and clean the value
                value = node_data.params[param_name]

                # Remove None and empty values
                if value is None or str(value) == "<class 'inspect._empty'>":
                    del node_data.params[param_name]
                    continue

                # Check for string representations of inspect._empty
                if isinstance(value, str):
                    if value.strip() in (
                        "inspect._empty",
                        "<class 'inspect._empty'>",
                        "_empty",
                    ):
                        del node_data.params[param_name]
                        continue

        except Exception as e:
            LOGGER.error(f"Failed to validate parameters for {class_name}: {e}")

    @staticmethod
    def validate_ai_nodes(nodes: list) -> tuple:
        """
        Validate AI-generated nodes.

        Returns: (valid_nodes, errors)
        """
        valid = []
        errors = []

        for node in nodes:
            if not hasattr(manim, node["class_name"]):
                errors.append(f"Invalid class: {node['class_name']}")
                continue
            valid.append(node)

        return valid, errors

    @staticmethod
    def merge_ai_code_to_scene(code: str, scene_graph) -> dict:
        """
        Merge AI-generated code into scene with animations and connections.
        Creates a fully connected graph with proper node positions.

        Returns: {
            'success': bool,
            'nodes_added': int,
            'nodes': list of created GraphicsItems,
            'errors': list of error messages
        }
        """
        try:
            mobjects, animations, play_sequence = AINodeIntegrator.parse_ai_code(code)

            valid_mobjects, mob_errors = AINodeIntegrator.validate_ai_nodes(mobjects)
            errors = list(mob_errors)

            created_nodes = []
            mobject_items = {}  # var_name -> NodeItem

            # Layout constants
            COL_WIDTH = 220
            ROW_HEIGHT = 120
            START_X = 50
            START_Y = 50

            # ── Create Mobject Nodes (column 0) ───────────────────────────
            for row_idx, node_def in enumerate(valid_mobjects):
                try:
                    item = AINodeIntegrator.create_node_from_ai(
                        node_def["var_name"],
                        node_def["class_name"],
                        node_def["params"],
                        scene_graph,
                        node_type=NodeType.MOBJECT,
                        pos=(START_X, START_Y + row_idx * ROW_HEIGHT),
                    )
                    created_nodes.append(item)
                    mobject_items[node_def["var_name"]] = item
                except Exception as e:
                    errors.append(
                        f"Failed to create mobject {node_def['var_name']}: {e}"
                    )

            # ── Create Animation + Play/Wait Nodes (connected) ────────────
            anim_col = 1
            struct_col = 2
            play_index = 0
            wait_index = 0
            cursor_row = 0
            prev_struct_item = None
            struct_allowed = {
                (NodeType.PLAY, NodeType.WAIT),
                (NodeType.WAIT, NodeType.PLAY),
                (NodeType.PLAY, NodeType.PLAY),
            }

            def _connect_struct(prev_item, next_item):
                if not prev_item or not next_item:
                    return
                if (prev_item.data.type, next_item.data.type) not in struct_allowed:
                    return
                try:
                    scene_graph.scene.try_connect(
                        prev_item.out_socket, next_item.in_socket
                    )
                except Exception:
                    pass

            for entry in play_sequence:
                kind = entry.get("kind", "play")

                if kind == "wait":
                    wait_index += 1
                    pos_x = START_X + struct_col * COL_WIDTH
                    pos_y = START_Y + cursor_row * ROW_HEIGHT
                    wait_item = AINodeIntegrator.create_structural_node_from_ai(
                        f"wait_{wait_index}",
                        NodeType.WAIT,
                        scene_graph,
                        pos=(pos_x, pos_y),
                        params={"duration": entry.get("duration", 1.0)},
                        cls_name="wait()",
                    )
                    created_nodes.append(wait_item)
                    _connect_struct(prev_struct_item, wait_item)
                    prev_struct_item = wait_item
                    cursor_row += 1
                    continue

                # Play entries
                anim_entries = entry.get("animations", [])
                play_kwargs = entry.get("play_kwargs", {})
                if not anim_entries:
                    continue

                play_index += 1
                play_start_row = cursor_row
                anim_items = []

                for anim_entry in anim_entries:
                    anim_class = anim_entry["anim_class"]
                    anim_var = anim_entry["anim_var"]
                    is_chain = anim_entry.get("is_animate_chain", False)
                    display_override = anim_entry.get("display_override")

                    params = dict(anim_entry.get("params", {}))
                    for k, v in play_kwargs.items():
                        if k not in params:
                            params[k] = v

                    display_class = (
                        display_override
                        if display_override
                        else (
                            anim_class.split(".")[-1]
                            if "." in anim_class
                            else anim_class
                        )
                    )

                    pos_x = START_X + anim_col * COL_WIDTH
                    pos_y = START_Y + cursor_row * ROW_HEIGHT

                    try:
                        item = AINodeIntegrator.create_node_from_ai(
                            anim_var,
                            anim_class if not is_chain else "ApplyMethod",
                            params,
                            scene_graph,
                            node_type=NodeType.ANIMATION,
                            pos=(pos_x, pos_y),
                            override_cls_name=display_class,
                        )
                        created_nodes.append(item)
                        anim_items.append(item)

                        # Connect animation to target mobjects (all found)
                        targets = list(anim_entry.get("target_vars", []))
                        if not targets and len(mobject_items) == 1:
                            targets = list(mobject_items.keys())
                        for target_var in targets:
                            mob_item = mobject_items.get(target_var)
                            if not mob_item:
                                continue
                            try:
                                scene_graph.scene.try_connect(
                                    mob_item.out_socket, item.in_socket
                                )
                            except Exception:
                                pass

                    except Exception as e:
                        errors.append(
                            f"Failed to create animation node {anim_var}: {e}"
                        )

                    cursor_row += 1

                # Create PLAY node aligned to this batch
                play_row = play_start_row + max(0, len(anim_items) - 1) / 2
                pos_x = START_X + struct_col * COL_WIDTH
                pos_y = START_Y + play_row * ROW_HEIGHT
                play_item = AINodeIntegrator.create_structural_node_from_ai(
                    f"play_{play_index}",
                    NodeType.PLAY,
                    scene_graph,
                    pos=(pos_x, pos_y),
                    params={},
                    cls_name="play()",
                )
                created_nodes.append(play_item)

                # Connect anims into the PLAY node
                for anim_item in anim_items:
                    try:
                        scene_graph.scene.try_connect(
                            anim_item.out_socket, play_item.in_socket
                        )
                    except Exception:
                        pass

                # Chain structural flow (PLAY/WAIT)
                _connect_struct(prev_struct_item, play_item)
                prev_struct_item = play_item

            scene_graph.scene.notify_change()

            return {
                "success": len(created_nodes) > 0,
                "nodes_added": len(created_nodes),
                "nodes": created_nodes,
                "errors": errors,
            }

        except Exception as e:
            LOGGER.error(f"merge_ai_code_to_scene error: {e}")
            traceback.print_exc()
            return {
                "success": False,
                "nodes_added": 0,
                "nodes": [],
                "errors": [f"Fatal error: {str(e)}"],
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
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import", "", "Media (*.png *.jpg *.mp4 *.mp3)"
        )
        for p in paths:
            ASSETS.add_asset(p)

    def refresh(self):
        self.list.clear()
        for asset in ASSETS.get_list():
            item = QListWidgetItem(asset.name)
            if asset.kind == "image":
                item.setIcon(QIcon(asset.original_path))
            else:
                item.setIcon(
                    self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon)
                )
            item.setData(Qt.ItemDataRole.UserRole, asset.id)
            self.list.addItem(item)

    def startDrag(self, actions):
        item = self.list.currentItem()
        if not item:
            return
        mime = QMimeData()
        mime.setText(f"ASSET:{item.data(Qt.ItemDataRole.UserRole)}")
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.CopyAction)


class VoiceoverPanel(QWidget):
    """Panel for AI TTS generation, audio preview, and node synchronization.

    Supports voiceover attachment to ALL animation node types.
    Includes full playback controls: play, pause, stop, seek, duration display.
    """

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.tts_worker = None
        self._current_audio_path: str | None = None
        self._player = QMediaPlayer()
        self._audio_out = QAudioOutput()
        self._player.setAudioOutput(self._audio_out)
        self._audio_out.setVolume(1.0)
        self._player_duration = 0

        # Wire player signals
        self._player.positionChanged.connect(self._on_position_changed)
        self._player.durationChanged.connect(self._on_duration_changed)
        self._player.mediaStatusChanged.connect(self._on_media_status)
        self._player.playbackStateChanged.connect(self._on_playback_state_changed)
        # NEW: Error handler for media load failures
        self._player.errorChanged.connect(self._on_player_error)

        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        # ── Header ────────────────────────────────────────────────
        header = QLabel("🎙️ AI Voiceover Studio")
        header.setStyleSheet("font-size: 14px; font-weight: bold; color: #2c3e50;")
        layout.addWidget(header)

        # ── Settings Grid ─────────────────────────────────────────
        form = QFormLayout()
        self.voice_combo = QComboBox()
        voices = ["Puck", "Charon", "Kore", "Fenrir", "Aoede", "Zephyr"]
        self.voice_combo.addItems(voices)
        self.voice_combo.setCurrentText("Zephyr")
        form.addRow("Voice:", self.voice_combo)
        layout.addLayout(form)

        # ── Script Input ──────────────────────────────────────────
        layout.addWidget(QLabel("Script:"))
        self.text_input = QPlainTextEdit()
        self.text_input.setPlaceholderText("Enter text to speak here...")
        self.text_input.setMaximumHeight(90)
        layout.addWidget(self.text_input)

        # ── Generate Button ───────────────────────────────────────
        self.btn_gen = QPushButton("⚡ Generate Audio")
        self.btn_gen.setStyleSheet(
            "background-color: #8e44ad; color: white; padding: 8px; font-weight: bold;"
        )
        self.btn_gen.clicked.connect(self.generate_audio)
        layout.addWidget(self.btn_gen)

        # ── Generation Progress ───────────────────────────────────
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # indeterminate
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumHeight(6)
        layout.addWidget(self.progress_bar)

        # ── Audio Preview ─────────────────────────────────────────
        preview_frame = QFrame()
        preview_frame.setFrameShape(QFrame.Shape.StyledPanel)
        preview_frame.setStyleSheet(
            "QFrame { background: #1a1a2e; border-radius: 6px; border: 1px solid #444; }"
        )
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(8, 6, 8, 6)
        preview_layout.setSpacing(4)

        preview_lbl = QLabel("🎵 Audio Preview")
        preview_lbl.setStyleSheet(
            "color: #a0a0c0; font-size: 11px; font-weight: bold; border: none;"
        )
        preview_layout.addWidget(preview_lbl)

        # Seek slider
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 0)
        self.seek_slider.setEnabled(False)
        self.seek_slider.sliderMoved.connect(self._on_seek)
        self.seek_slider.setStyleSheet(
            "QSlider::groove:horizontal { height: 4px; background: #444; border-radius: 2px; }"
            "QSlider::sub-page:horizontal { background: #8e44ad; border-radius: 2px; }"
            "QSlider::handle:horizontal { width: 12px; height: 12px; margin: -4px 0; "
            "background: white; border-radius: 6px; }"
        )
        preview_layout.addWidget(self.seek_slider)

        # Time label
        self.lbl_time = QLabel("00:00 / 00:00")
        self.lbl_time.setStyleSheet(
            "color: #808090; font-family: monospace; font-size: 10px; border: none;"
        )
        self.lbl_time.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(self.lbl_time)

        # Playback controls row
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(4)

        self.btn_play = QPushButton("▶")
        self.btn_play.setFixedSize(32, 32)
        self.btn_play.setEnabled(False)
        self.btn_play.setToolTip("Play / Pause")
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_play.setStyleSheet(
            "QPushButton { background: #8e44ad; color: white; border-radius: 16px; font-size: 12px; border: none; }"
            "QPushButton:disabled { background: #555; }"
        )

        self.btn_stop = QPushButton("⏹")
        self.btn_stop.setFixedSize(32, 32)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setToolTip("Stop")
        self.btn_stop.clicked.connect(self._stop_audio)
        self.btn_stop.setStyleSheet(
            "QPushButton { background: #555; color: white; border-radius: 16px; font-size: 12px; border: none; }"
            "QPushButton:disabled { background: #444; color: #888; }"
        )

        ctrl_row.addStretch()
        ctrl_row.addWidget(self.btn_play)
        ctrl_row.addWidget(self.btn_stop)
        ctrl_row.addStretch()
        preview_layout.addLayout(ctrl_row)

        layout.addWidget(preview_frame)

        # ── Separator ─────────────────────────────────────────────
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #bdc3c7;")
        layout.addWidget(line)

        # ── Node Sync Section ─────────────────────────────────────
        sync_lbl = QLabel("🔗 Attach to Animation Node")
        sync_lbl.setStyleSheet("font-weight: bold;")
        layout.addWidget(sync_lbl)

        node_row = QHBoxLayout()
        self.node_combo = QComboBox()
        self.node_combo.setPlaceholderText("Select an Animation Node...")
        node_row.addWidget(self.node_combo, 1)

        btn_refresh = QPushButton("🔄")
        btn_refresh.setFixedSize(28, 28)
        btn_refresh.setToolTip("Refresh Node List")
        btn_refresh.clicked.connect(self.refresh_nodes)
        node_row.addWidget(btn_refresh)
        layout.addLayout(node_row)

        # "Add to Animation Node" button
        self.btn_attach = QPushButton("📎 Add to Animation Node")
        self.btn_attach.setStyleSheet(
            "background-color: #27ae60; color: white; padding: 7px; font-weight: bold;"
        )
        self.btn_attach.setEnabled(False)
        self.btn_attach.setToolTip(
            "Attach the generated audio to the selected animation node"
        )
        self.btn_attach.clicked.connect(self._attach_to_node)
        layout.addWidget(self.btn_attach)

        self.status_lbl = QLabel("Ready")
        self.status_lbl.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(self.status_lbl)

        layout.addStretch()

        # Initial refresh
        QTimer.singleShot(1000, self.refresh_nodes)

    # ── Node population ────────────────────────────────────────────

    def refresh_nodes(self):
        """Populate combo with ALL animation nodes (any type)."""
        current = self.node_combo.currentData()
        self.node_combo.clear()

        count = 0
        for nid, node in self.main_window.nodes.items():
            short_id = nid[:6]
            _vo_icon_map = {
                NodeType.ANIMATION: "🎬",
                NodeType.MOBJECT: "🔷",
                NodeType.PLAY: "▶",
                NodeType.WAIT: "⏱",
                NodeType.VGROUP: "🔗",
            }
            type_tag = _vo_icon_map.get(node.data.type, "🎬")
            display_text = f"{type_tag} {node.data.name} ({short_id})"
            self.node_combo.addItem(display_text, nid)
            count += 1

        if current:
            idx = self.node_combo.findData(current)
            if idx >= 0:
                self.node_combo.setCurrentIndex(idx)

        self.status_lbl.setText(f"Found {count} nodes.")

    # ── Audio generation ───────────────────────────────────────────

    def generate_audio(self):
        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Input Error", "Please enter some text.")
            return

        self.btn_gen.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.status_lbl.setText("Generating audio via Gemini TTS…")

        voice = self.voice_combo.currentText()
        model = SETTINGS.get("TTS_MODEL", "gemini-2.5-flash-preview-tts")

        self.tts_worker = TTSWorker(text, voice, model)
        self.tts_worker.finished_signal.connect(self.on_tts_success)
        self.tts_worker.error_signal.connect(self.on_tts_error)
        self.tts_worker.start()

    def on_tts_success(self, file_path):
        self.btn_gen.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_lbl.setText("Audio generated — ready for preview.")

        asset = ASSETS.add_asset(file_path)
        if not asset:
            self.status_lbl.setText("Warning: Error registering asset.")
            return

        self._current_audio_path = file_path
        self._load_preview(file_path)
        self.btn_attach.setEnabled(True)

        # Store transcript on the worker's text for later node attachment
        self._last_transcript = self.text_input.toPlainText().strip()
        self._last_asset = asset

    def on_tts_error(self, err):
        self.btn_gen.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_lbl.setText("Generation failed.")
        LOGGER.error(f"TTS Error: {err}")
        QMessageBox.critical(self, "TTS Error", err)

    # ── Audio preview ──────────────────────────────────────────────

    def _load_preview(self, file_path: str):
        """
        Load audio file into the preview player.

        CRITICAL FIX: Proper QUrl conversion with error handling.
        - Convert path to QUrl with proper escaping
        - Validate file exists before loading
        - Clear any previous errors
        - Log load status
        """
        try:
            # ═════════════════════════════════════════════════════════════
            # VALIDATION: File must exist
            # ═════════════════════════════════════════════════════════════
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                self.status_lbl.setText(f"File not found: {file_path}")
                LOGGER.error(f"Audio preview: file missing at {file_path}")
                self.btn_play.setEnabled(False)
                self.btn_stop.setEnabled(False)
                self.seek_slider.setEnabled(False)
                return

            # ═════════════════════════════════════════════════════════════
            # FIX: Use QUrl.fromLocalFile() for proper path escaping
            # Windows paths with backslashes must be converted safely
            # ═════════════════════════════════════════════════════════════
            from PySide6.QtCore import QUrl

            # Convert to absolute path to prevent relative path issues
            abs_path = file_path_obj.resolve()

            # Create QUrl with proper local file encoding
            # This handles spaces, special characters, and backslashes correctly
            media_url = QUrl.fromLocalFile(str(abs_path))

            if not media_url.isValid():
                self.status_lbl.setText("Invalid file path or unsupported format")
                LOGGER.error(f"Audio preview: invalid QUrl for {abs_path}")
                self.btn_play.setEnabled(False)
                self.btn_stop.setEnabled(False)
                self.seek_slider.setEnabled(False)
                return

            # ═════════════════════════════════════════════════════════════
            # LOAD: Set media source and enable controls
            # ═════════════════════════════════════════════════════════════
            self._player.setSource(media_url)

            # Enable playback controls
            self.seek_slider.setEnabled(True)
            self.btn_play.setEnabled(True)
            self.btn_stop.setEnabled(True)

            self.status_lbl.setText("Audio loaded. Press ▶ to preview.")
            LOGGER.info(f"Audio preview loaded: {abs_path}")

        except Exception as e:
            self.status_lbl.setText(f"Error loading audio: {type(e).__name__}")
            LOGGER.error(f"Audio preview load error: {e}", exc_info=True)
            self.btn_play.setEnabled(False)
            self.btn_stop.setEnabled(False)
            self.seek_slider.setEnabled(False)

    def _toggle_play(self):
        state = self._player.playbackState()
        if state == QMediaPlayer.PlaybackState.Playing:
            self._player.pause()
        else:
            self._player.play()

    def _stop_audio(self):
        self._player.stop()
        self.seek_slider.setValue(0)
        self.lbl_time.setText("00:00 / 00:00")

    def _on_seek(self, position: int):
        self._player.setPosition(position)

    def _on_position_changed(self, position: int):
        if not self.seek_slider.isSliderDown():
            self.seek_slider.setValue(position)
        self._update_time_label(position)

    def _on_duration_changed(self, duration: int):
        self._player_duration = duration
        self.seek_slider.setRange(0, duration)
        self._update_time_label(0)

    def _on_media_status(self, status):
        """
        Handle media status changes.

        Catches load errors, end-of-media, and other status changes.
        """
        try:
            # Handle end of media
            if status == QMediaPlayer.MediaStatus.EndOfMedia:
                self.btn_play.setText("▶")
                LOGGER.info("Audio playback: end of media reached")

            # Handle load errors
            elif status == QMediaPlayer.MediaStatus.InvalidMedia:
                self.status_lbl.setText("Invalid audio format or corrupted file")
                LOGGER.error("Audio preview: invalid media format")
                self.btn_play.setEnabled(False)
                self.btn_stop.setEnabled(False)

            elif status == QMediaPlayer.MediaStatus.NoMedia:
                # No media loaded (expected after clear or init)
                pass

            elif status == QMediaPlayer.MediaStatus.LoadedMedia:
                # Media successfully loaded
                LOGGER.info("Audio preview: media loaded successfully")

            elif status == QMediaPlayer.MediaStatus.LoadingMedia:
                # Media is being loaded
                self.status_lbl.setText("⏳ Loading audio...")

        except Exception as e:
            LOGGER.error(f"Error in _on_media_status: {e}")

    def _on_player_error(self):
        """
        Handle QMediaPlayer errors.

        Called when an error occurs during playback or loading.
        """
        try:
            error = self._player.error()
            error_string = self._player.errorString()

            if error != QMediaPlayer.Error.NoError:
                self.status_lbl.setText(f"Playback error: {error_string}")
                LOGGER.error(f"Audio preview error: {error} - {error_string}")
                self.btn_play.setEnabled(False)
                self.btn_stop.setEnabled(False)

        except Exception as e:
            LOGGER.error(f"Error in _on_player_error: {e}")

    def _on_playback_state_changed(self, state):
        if state == QMediaPlayer.PlaybackState.Playing:
            self.btn_play.setText("⏸")
        else:
            self.btn_play.setText("▶")

    def _update_time_label(self, current_ms: int):
        def fmt(ms):
            s = (ms // 1000) % 60
            m = ms // 60000
            return f"{m:02}:{s:02}"

        self.lbl_time.setText(f"{fmt(current_ms)} / {fmt(self._player_duration)}")

    # ── Node attachment ────────────────────────────────────────────

    def _attach_to_node(self):
        """Attach generated audio to the selected animation node."""
        node_id = self.node_combo.currentData()
        if not node_id:
            QMessageBox.warning(
                self, "No Node Selected", "Please select an animation node first."
            )
            return

        if node_id not in self.main_window.nodes:
            QMessageBox.warning(self, "Invalid Node", "Selected node no longer exists.")
            return

        if not hasattr(self, "_last_asset") or self._last_asset is None:
            QMessageBox.warning(
                self, "No Audio", "Generate audio first before attaching."
            )
            return

        node = self.main_window.nodes[node_id]
        node.data.audio_asset_id = self._last_asset.id
        node.data.voiceover_transcript = getattr(self, "_last_transcript", "")
        if self._player_duration > 0:
            node.data.voiceover_duration = self._player_duration / 1000.0

        node.update()
        self.main_window.compile_graph()
        self.main_window.mark_modified()

        self.status_lbl.setText(f"Attached to '{node.data.name}'")
        LOGGER.info(f"Voiceover attached to node {node.data.name} ({node_id[:6]})")

        QMessageBox.information(
            self,
            "Voiceover Attached",
            f"Audio successfully attached to '{node.data.name}'.\n"
            f"Duration: {node.data.voiceover_duration:.2f}s\n"
            "The node will use this audio during render.",
        )


class ManimClassBrowser(QWidget):
    """Searchable palette of all Manim mobjects and animations."""

    node_requested = Signal(str, str)  # class_name, node_type

    CATEGORIES = {
        "📐 Geometry": [
            "Square",
            "Rectangle",
            "Circle",
            "Ellipse",
            "Triangle",
            "Arrow",
            "Line",
            "DashedLine",
            "DoubleArrow",
            "Polygon",
            "RegularPolygon",
            "Dot",
            "Cross",
            "Star",
            "Arc",
        ],
        "📝 Text": [
            "Text",
            "Tex",
            "MathTex",
            "MarkupText",
            "Title",
            "Paragraph",
            "BulletedList",
        ],
        " Graphs & Plots": [
            "Axes",
            "NumberPlane",
            "PolarPlane",
            "NumberLine",
            "CoordinateSystem",
            "BarChart",
            "LineGraph",
        ],
        "🎭 3D Objects": [
            "Sphere",
            "Cube",
            "Cylinder",
            "Cone",
            "Torus",
            "Surface",
            "ParametricSurface",
        ],
        "🎬 Animations (In)": [
            "FadeIn",
            "Write",
            "DrawBorderThenFill",
            "Create",
            "GrowFromCenter",
            "GrowArrow",
            "SpinInFromNothing",
            "FadeInFromEdge",
            "Succession",
        ],
        "🎬 Animations (Out)": [
            "FadeOut",
            "Unwrite",
            "Uncreate",
            "ShrinkToCenter",
            "FadeOutToEdge",
        ],
        "🔄 Transforms": [
            "Transform",
            "ReplacementTransform",
            "TransformFromCopy",
            "MoveToTarget",
            "ApplyMethod",
            "ApplyMatrix",
            "Rotate",
            "Scale",
            "ScaleInPlace",
        ],
        "✨ Emphasis": [
            "Indicate",
            "FocusOn",
            "Circumscribe",
            "ShowPassingFlash",
            "Flash",
            "Wiggle",
            "ApplyWave",
            "Homotopy",
        ],
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Search bar
        self.search = QLineEdit()
        self.search.setPlaceholderText("🔍 Search Manim classes…")
        self.search.textChanged.connect(self._filter)
        layout.addWidget(self.search)

        # Tree
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setDragEnabled(True)
        self.tree.itemDoubleClicked.connect(self._on_double_click)
        self.tree.setToolTip("Double-click or drag to add node to canvas")
        layout.addWidget(self.tree)

        self._populate()

    def _populate(self, filter_text=""):
        self.tree.clear()
        ft = filter_text.lower()
        for category, items in self.CATEGORIES.items():
            filtered = [i for i in items if ft in i.lower()] if ft else items
            if not filtered:
                continue
            parent = QTreeWidgetItem([category])
            parent.setExpanded(bool(ft))
            self.tree.addTopLevelItem(parent)
            for cls_name in filtered:
                child = QTreeWidgetItem([cls_name])
                child.setToolTip(0, f"Double-click to add {cls_name} node to canvas")
                parent.addChild(child)
        if not ft:
            self.tree.expandAll() if len(self.CATEGORIES) <= 3 else None

    def _filter(self, text):
        self._populate(text)
        if text:
            self.tree.expandAll()

    def _on_double_click(self, item, _col):
        if item.parent() is not None:
            cls_name = item.text(0)
            # Determine type
            node_type = "animation"
            for cat, items in self.CATEGORIES.items():
                if cls_name in items:
                    if "Animation" in cat or "Transform" in cat or "Emphasis" in cat:
                        node_type = "animation"
                    else:
                        node_type = "mobject"
            self.node_requested.emit(cls_name, node_type)


# ── Code Snippet Library ──────────────────────────────────────────────────────


class SnippetLibrary(QWidget):
    """Reusable Manim code snippet templates."""

    snippet_requested = Signal(str)

    SNIPPETS = {
        "🎯 FadeIn + FadeOut": (
            "from manim import *\n\n"
            "class MyScene(Scene):\n"
            "    def construct(self):\n"
            "        circle = Circle(color=BLUE)\n"
            "        self.play(FadeIn(circle))\n"
            "        self.wait(1)\n"
            "        self.play(FadeOut(circle))\n"
        ),
        "🔄 Transform": (
            "from manim import *\n\n"
            "class MyScene(Scene):\n"
            "    def construct(self):\n"
            "        square = Square()\n"
            "        circle = Circle()\n"
            "        self.play(Create(square))\n"
            "        self.play(Transform(square, circle))\n"
            "        self.wait(1)\n"
        ),
        "📝 MathTex": (
            "from manim import *\n\n"
            "class MyScene(Scene):\n"
            "    def construct(self):\n"
            "        formula = MathTex(r'E = mc^2')\n"
            "        self.play(Write(formula))\n"
            "        self.wait(2)\n"
        ),
        "🎨 VGroup": (
            "from manim import *\n\n"
            "class MyScene(Scene):\n"
            "    def construct(self):\n"
            "        c = Circle(color=RED)\n"
            "        s = Square(color=BLUE)\n"
            "        group = VGroup(c, s).arrange(RIGHT)\n"
            "        self.play(Create(group))\n"
            "        self.wait(1)\n"
        ),
        "📈 Axes + Function Graph": (
            "from manim import *\n\n"
            "class MyScene(Scene):\n"
            "    def construct(self):\n"
            "        axes = Axes(\n"
            "            x_range=[-4, 4, 1],\n"
            "            y_range=[-2, 8, 2],\n"
            "            x_length=8,\n"
            "            y_length=5,\n"
            "            tips=False,\n"
            "        )\n"
            "        graph = axes.plot(lambda x: 0.5 * x**2, color=BLUE)\n"
            '        label = axes.get_graph_label(graph, label="y=0.5x^2")\n'
            "        self.play(Create(axes), Create(graph), FadeIn(label))\n"
            "        self.wait(1)\n"
        ),
        "🧮 Equation Steps (TransformMatchingTex)": (
            "from manim import *\n\n"
            "class MyScene(Scene):\n"
            "    def construct(self):\n"
            '        eq1 = MathTex(r"ax + b = c")\n'
            '        eq2 = MathTex(r"ax = c - b").move_to(eq1)\n'
            '        eq3 = MathTex(r"x = {c-b \\\\over a}").move_to(eq1)\n'
            "        self.play(Write(eq1))\n"
            "        self.play(TransformMatchingTex(eq1, eq2))\n"
            "        self.play(TransformMatchingTex(eq2, eq3))\n"
            "        self.wait(1)\n"
        ),
        "🧭 NumberPlane + Labeled Point": (
            "from manim import *\n\n"
            "class MyScene(Scene):\n"
            "    def construct(self):\n"
            "        plane = NumberPlane(\n"
            "            x_range=[-5, 5, 1],\n"
            "            y_range=[-3, 3, 1],\n"
            '            background_line_style={"stroke_opacity": 0.4},\n'
            "        )\n"
            "        dot = Dot(plane.c2p(2, 1), color=YELLOW)\n"
            '        label = MathTex("(2, 1)").next_to(dot, UR)\n'
            "        self.play(Create(plane), FadeIn(dot), Write(label))\n"
            "        self.wait(1)\n"
        ),
        "📉 Area Under Curve": (
            "from manim import *\n\n"
            "class MyScene(Scene):\n"
            "    def construct(self):\n"
            "        axes = Axes(\n"
            "            x_range=[-3, 3, 1],\n"
            "            y_range=[-1, 5, 1],\n"
            "            x_length=8,\n"
            "            y_length=5,\n"
            "            tips=False,\n"
            "        )\n"
            "        graph = axes.plot(lambda x: x**2 / 2, color=BLUE)\n"
            "        area = axes.get_area(graph, x_range=[-1, 2], color=BLUE, opacity=0.3)\n"
            "        self.play(Create(axes), Create(graph))\n"
            "        self.play(FadeIn(area))\n"
            "        self.wait(1)\n"
        ),
        "🧲 Vector Arrow + Label": (
            "from manim import *\n\n"
            "class MyScene(Scene):\n"
            "    def construct(self):\n"
            "        plane = NumberPlane(\n"
            "            x_range=[-4, 4, 1],\n"
            "            y_range=[-3, 3, 1],\n"
            '            background_line_style={"stroke_opacity": 0.4},\n'
            "        )\n"
            "        vec = Vector([3, 2], color=GREEN)\n"
            '        label = MathTex(r"\\\\vec{v}").next_to(vec.get_end(), UR)\n'
            "        self.play(Create(plane), GrowArrow(vec), Write(label))\n"
            "        self.wait(1)\n"
        ),
        "✨ Staggered Reveal (LaggedStartMap)": (
            "from manim import *\n\n"
            "class MyScene(Scene):\n"
            "    def construct(self):\n"
            "        dots = VGroup(*[Dot([x, 0, 0]) for x in range(-4, 5)])\n"
            "        self.play(LaggedStartMap(FadeIn, dots, lag_ratio=0.1))\n"
            "        self.wait(1)\n"
        ),
        "🌀 Parametric Curve": (
            "from manim import *\n"
            "import numpy as np\n\n"
            "class MyScene(Scene):\n"
            "    def construct(self):\n"
            "        curve = ParametricFunction(\n"
            "            lambda t: np.array([np.cos(t), np.sin(2 * t) / 2, 0]),\n"
            "            t_range=[0, TAU],\n"
            "            color=PURPLE,\n"
            "        )\n"
            "        self.play(Create(curve))\n"
            "        self.wait(1)\n"
        ),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        hdr = QLabel("📋 Snippet Library")
        hdr.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(hdr)

        self.search = QLineEdit()
        self.search.setPlaceholderText("🔍 Search snippets…")
        self.search.textChanged.connect(self._filter)
        layout.addWidget(self.search)

        self.list_widget = QListWidget()
        self.list_widget.itemDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self.list_widget)

        self._populate()

    def _populate(self, filter_text=""):
        self.list_widget.clear()
        ft = filter_text.lower()
        for name in self.SNIPPETS:
            if ft and ft not in name.lower():
                continue
            item = QListWidgetItem(name)
            self.list_widget.addItem(item)

    def _filter(self, text):
        self._populate(text)

    def _on_double_click(self, item):
        code = self.SNIPPETS.get(item.text(), "")
        if code:
            self.snippet_requested.emit(code)


class NodeSearchBar(QWidget):
    """Toolbar for searching and filtering nodes on the canvas."""

    filter_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter nodes…")
        self.search.textChanged.connect(self.filter_changed)
        layout.addWidget(self.search)

        btn_clear = QPushButton("✕")
        btn_clear.setFixedWidth(28)
        btn_clear.clicked.connect(self.search.clear)
        btn_clear.setToolTip("Clear filter")
        layout.addWidget(btn_clear)


# ── Quick Export Toolbar ───────────────────────────────────────────────────────


class VGroupPanel(QWidget):
    """Panel for creating, viewing, and managing VGroups from all sources.

    Supports canvas-created groups (with live NodeItem references) and
    code-origin groups parsed from AI output, local snippets, and GitHub
    snippets. All groups show their members in the tree.
    """

    vgroup_created = Signal(str, list)

    # Source labels and icons
    _SOURCE_ICON = {"canvas": "🎨", "ai": "🤖", "snippet": "📄", "github": "🐙"}
    _SOURCE_COLOR = {
        "canvas": "#4f46e5",
        "ai": "#7c3aed",
        "snippet": "#0891b2",
        "github": "#059669",
    }

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        # name → list of node IDs (canvas) or [] (code-origin)
        self._groups: dict[str, list] = {}
        # name → {"source": str, "members": [str]}
        self._meta: dict[str, dict] = {}
        self._build()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # ── Create section ──────────────────────────────────────────────────
        create_box = QGroupBox("Create VGroup")
        create_box.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid #d1d5db; "
            "border-radius: 6px; margin-top: 8px; padding-top: 4px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; }"
        )
        create_layout = QVBoxLayout(create_box)
        create_layout.setSpacing(4)

        name_row = QHBoxLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Name (e.g. my_group)")
        self.name_edit.setToolTip("Valid Python identifier for this VGroup")
        name_row.addWidget(self.name_edit)
        create_layout.addLayout(name_row)

        btn_create = QPushButton("📦  Create from Canvas Selection")
        btn_create.clicked.connect(self._create_vgroup)
        btn_create.setToolTip("Select Mobject nodes on the canvas first, then click")
        btn_create.setStyleSheet(
            "QPushButton { background-color: #4f46e5; color: white; font-weight: bold;"
            " border-radius: 5px; padding: 7px 10px; }"
            "QPushButton:hover { background-color: #4338ca; }"
            "QPushButton:pressed { background-color: #3730a3; }"
        )
        create_layout.addWidget(btn_create)

        hint = QLabel("Select Mobject nodes on the canvas first.")
        hint.setStyleSheet("color: #9ca3af; font-size: 10px;")
        create_layout.addWidget(hint)
        layout.addWidget(create_box)

        # ── Search ──────────────────────────────────────────────────────────
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("🔍  Filter groups…")
        self.search_edit.textChanged.connect(self._filter_tree)
        self.search_edit.setStyleSheet(
            "QLineEdit { border: 1px solid #d1d5db; border-radius: 5px; padding: 5px; }"
        )
        layout.addWidget(self.search_edit)

        # ── Group count label ────────────────────────────────────────────────
        self.count_lbl = QLabel("No groups yet")
        self.count_lbl.setStyleSheet("color: #6b7280; font-size: 10px;")
        layout.addWidget(self.count_lbl)

        # ── Tree ────────────────────────────────────────────────────────────
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        self.tree.setAlternatingRowColors(True)
        self.tree.setStyleSheet(
            "QTreeWidget { border: 1px solid #e5e7eb; border-radius: 5px; }"
            "QTreeWidget::item { padding: 3px 2px; }"
            "QTreeWidget::item:selected { background: #ede9fe; color: #4f46e5; }"
        )
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.tree, stretch=1)

        # ── Action toolbar ──────────────────────────────────────────────────
        toolbar_box = QGroupBox("Actions")
        toolbar_box.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid #d1d5db; "
            "border-radius: 6px; margin-top: 8px; padding-top: 4px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; }"
        )
        tb_layout = QVBoxLayout(toolbar_box)
        tb_layout.setSpacing(4)

        # Row 1: group-level actions
        row1 = QHBoxLayout()
        row1.setSpacing(4)

        self.btn_rename = self._make_btn(
            "✏️ Rename", "#0891b2", self._rename_vgroup, "Rename the selected VGroup"
        )
        self.btn_duplicate = self._make_btn(
            "⧉ Duplicate",
            "#0891b2",
            self._duplicate_vgroup,
            "Create a copy of this VGroup with a new name",
        )
        self.btn_copy_code = self._make_btn(
            "📋 Copy Code",
            "#059669",
            self._copy_code,
            "Copy  name = VGroup(members…)  to clipboard",
        )
        row1.addWidget(self.btn_rename)
        row1.addWidget(self.btn_duplicate)
        row1.addWidget(self.btn_copy_code)
        tb_layout.addLayout(row1)

        # Row 2: canvas-interaction actions
        row2 = QHBoxLayout()
        row2.setSpacing(4)

        self.btn_highlight = self._make_btn(
            "🎯 Highlight",
            "#7c3aed",
            self._highlight_on_canvas,
            "Select all member nodes on the canvas",
        )
        self.btn_add_members = self._make_btn(
            "＋ Add Nodes",
            "#059669",
            self._add_members,
            "Add currently selected canvas nodes to this group",
        )
        self.btn_remove_member = self._make_btn(
            "－ Remove Member",
            "#f59e0b",
            self._remove_member,
            "Remove the selected member from this group",
        )
        row2.addWidget(self.btn_highlight)
        row2.addWidget(self.btn_add_members)
        row2.addWidget(self.btn_remove_member)
        tb_layout.addLayout(row2)

        # Row 3: destructive
        row3 = QHBoxLayout()
        row3.addStretch()
        self.btn_delete = self._make_btn(
            "🗑 Delete Group",
            "#dc2626",
            self._delete_vgroup,
            "Permanently remove this VGroup (does not delete nodes)",
        )
        self.btn_delete.setMinimumWidth(130)
        row3.addWidget(self.btn_delete)
        tb_layout.addLayout(row3)

        layout.addWidget(toolbar_box)

        # Initial button state
        self._set_buttons_enabled(False)

    @staticmethod
    def _make_btn(label: str, color: str, slot, tooltip: str) -> QPushButton:
        btn = QPushButton(label)
        btn.setToolTip(tooltip)
        btn.clicked.connect(slot)
        btn.setStyleSheet(
            f"QPushButton {{ background-color: {color}; color: white; font-size: 11px;"
            f" border-radius: 4px; padding: 5px 6px; }}"
            f"QPushButton:hover {{ opacity: 0.85; }}"
            f"QPushButton:disabled {{ background-color: #d1d5db; color: #9ca3af; }}"
        )
        return btn

    # ── Button state management ──────────────────────────────────────────────

    def _set_buttons_enabled(self, group_selected: bool, member_selected: bool = False):
        for btn in (
            self.btn_rename,
            self.btn_duplicate,
            self.btn_copy_code,
            self.btn_highlight,
            self.btn_add_members,
            self.btn_delete,
        ):
            btn.setEnabled(group_selected)
        self.btn_remove_member.setEnabled(member_selected)

    def _on_selection_changed(self):
        item = self.tree.currentItem()
        if item is None:
            self._set_buttons_enabled(False)
        elif item.parent() is None:
            # Top-level = group row
            self._set_buttons_enabled(True, False)
        else:
            # Child = member row
            self._set_buttons_enabled(True, True)

    # ── Group resolution helpers ─────────────────────────────────────────────

    def _current_group_name(self) -> str | None:
        """Return the group name from the currently selected tree item."""
        item = self.tree.currentItem()
        if item is None:
            return None
        if item.parent() is None:
            return item.data(0, Qt.ItemDataRole.UserRole)
        return item.parent().data(0, Qt.ItemDataRole.UserRole)

    def _member_display_names(self, name: str) -> list[str]:
        """Return the display names for members of group `name`."""
        ids = self._groups.get(name, [])
        meta = self._meta.get(name, {})
        # Canvas group: resolve node names from live NodeItems
        if ids:
            names = []
            for nid in ids:
                node = self.main_window.nodes.get(nid)
                names.append(node.data.name if node else f"<missing:{nid[:6]}>")
            return names
        # Code-origin group: use parsed member names
        return meta.get("members", [])

    # ── Tree refresh ─────────────────────────────────────────────────────────

    def _refresh_tree(self):
        filter_txt = (
            self.search_edit.text().lower() if hasattr(self, "search_edit") else ""
        )
        self.tree.clear()
        for gname, ids in self._groups.items():
            if filter_txt and filter_txt not in gname.lower():
                continue
            meta = self._meta.get(gname, {})
            source = meta.get("source", "canvas")
            icon = self._SOURCE_ICON.get(source, "📦")
            color = self._SOURCE_COLOR.get(source, "#4f46e5")
            members = self._member_display_names(gname)
            count_str = f"{len(members)} member{'s' if len(members) != 1 else ''}"
            root = QTreeWidgetItem(self.tree)
            root.setData(0, Qt.ItemDataRole.UserRole, gname)
            # Header text: icon + name + source badge + count
            root.setText(0, f"{icon}  {gname}  ·  {source}  ·  {count_str}")
            root.setForeground(0, QBrush(QColor(color)))
            root.setFont(0, QFont("Segoe UI", 10, QFont.Weight.Bold))
            root.setToolTip(
                0,
                f"Source: {source}\nMembers: {', '.join(members) if members else 'none'}",
            )

            if members:
                for mname in members:
                    child = QTreeWidgetItem(root)
                    child.setText(0, f"    └  {mname}")
                    child.setForeground(0, QBrush(QColor("#374151")))
                    child.setData(0, Qt.ItemDataRole.UserRole + 1, mname)
            else:
                empty = QTreeWidgetItem(root)
                empty.setText(0, "    (no members resolved)")
                empty.setForeground(0, QBrush(QColor("#9ca3af")))
                f = empty.font(0)
                f.setItalic(True)
                empty.setFont(0, f)

            root.setExpanded(True)

        n = len(self._groups)
        self.count_lbl.setText(
            f"{n} group{'s' if n != 1 else ''}" if n else "No groups yet"
        )
        self._set_buttons_enabled(False)

    def _filter_tree(self, _txt: str):
        self._refresh_tree()

    # ── Create ───────────────────────────────────────────────────────────────

    def _create_vgroup(self):
        sel = self.main_window.scene.selectedItems()
        members = [
            item
            for item in sel
            if isinstance(item, NodeItem) and item.data.type == NodeType.MOBJECT
        ]
        if not members:
            QMessageBox.warning(
                self,
                "No Selection",
                "Select at least one Mobject node on the canvas first.",
            )
            return
        name = self.name_edit.text().strip() or f"vgroup_{len(self._groups) + 1}"
        if not name.isidentifier():
            QMessageBox.warning(
                self, "Invalid Name", "VGroup name must be a valid Python identifier."
            )
            return
        if name in self._groups:
            QMessageBox.warning(
                self, "Duplicate Name", f"A VGroup named '{name}' already exists."
            )
            return
        ids = [m.data.id for m in members]
        self._groups[name] = ids
        self._meta[name] = {
            "source": "canvas",
            "members": [m.data.name for m in members],
        }
        self.vgroup_created.emit(name, ids)
        self._refresh_tree()
        self.name_edit.clear()

        # ── Also create a VGROUP canvas node so it appears in the graph ───
        try:
            self.main_window.add_vgroup_node(group_name=name, member_ids=ids)
            LOGGER.info(
                f"VGROUP canvas node created for '{name}' with {len(ids)} members"
            )
        except Exception as _e:
            LOGGER.warning(f"Could not create VGROUP canvas node for '{name}': {_e}")

        # Select the new group in tree
        for i in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(i)
            if it.data(0, Qt.ItemDataRole.UserRole) == name:
                self.tree.setCurrentItem(it)
                break

    # ── Rename ───────────────────────────────────────────────────────────────

    def _rename_vgroup(self):
        name = self._current_group_name()
        if not name:
            return
        new_name, ok = QInputDialog.getText(
            self, "Rename VGroup", "New name:", QLineEdit.EchoMode.Normal, name
        )
        if not ok:
            return
        new_name = new_name.strip()
        if not new_name or new_name == name:
            return
        if not new_name.isidentifier():
            QMessageBox.warning(
                self, "Invalid Name", "VGroup name must be a valid Python identifier."
            )
            return
        if new_name in self._groups:
            QMessageBox.warning(
                self, "Duplicate Name", f"A VGroup named '{new_name}' already exists."
            )
            return
        # Re-insert preserving order
        self._groups = {
            (new_name if k == name else k): v for k, v in self._groups.items()
        }
        self._meta = {(new_name if k == name else k): v for k, v in self._meta.items()}
        self._refresh_tree()

    # ── Duplicate ────────────────────────────────────────────────────────────

    def _duplicate_vgroup(self):
        name = self._current_group_name()
        if not name:
            return
        base = f"{name}_copy"
        candidate = base
        i = 2
        while candidate in self._groups:
            candidate = f"{base}_{i}"
            i += 1
        self._groups[candidate] = list(self._groups[name])
        self._meta[candidate] = dict(self._meta.get(name, {}))
        self._meta[candidate]["source"] = self._meta.get(name, {}).get(
            "source", "canvas"
        )
        self._refresh_tree()

    # ── Copy Code ────────────────────────────────────────────────────────────

    def _copy_code(self):
        name = self._current_group_name()
        if not name:
            return
        members = self._member_display_names(name)
        args = ", ".join(members) if members else "# no members"
        code = f"{name} = VGroup({args})"
        QApplication.clipboard().setText(code)
        # Brief visual feedback via tooltip on button
        self.btn_copy_code.setToolTip(f"Copied: {code}")
        QTimer.singleShot(
            2500,
            lambda: self.btn_copy_code.setToolTip(
                "Copy  name = VGroup(members…)  to clipboard"
            ),
        )

    # ── Highlight on Canvas ───────────────────────────────────────────────────

    def _highlight_on_canvas(self):
        name = self._current_group_name()
        if not name:
            return
        ids = self._groups.get(name, [])
        if not ids:
            QMessageBox.information(
                self,
                "Canvas Highlight",
                "This group was created from code — member nodes are not "
                "tracked on the canvas.\nAdd them manually to enable highlighting.",
            )
            return
        # Deselect all, then select members
        self.main_window.scene.clearSelection()
        found = 0
        for nid in ids:
            node = self.main_window.nodes.get(nid)
            if node:
                node.setSelected(True)
                found += 1
        if found == 0:
            QMessageBox.information(
                self,
                "Canvas Highlight",
                "None of the member nodes were found on the canvas.",
            )

    # ── Add Members from Selection ────────────────────────────────────────────

    def _add_members(self):
        name = self._current_group_name()
        if not name:
            return
        sel = self.main_window.scene.selectedItems()
        new_nodes = [
            item
            for item in sel
            if isinstance(item, NodeItem)
            and item.data.type == NodeType.MOBJECT
            and item.data.id not in self._groups.get(name, [])
        ]
        if not new_nodes:
            QMessageBox.information(
                self,
                "Add Members",
                "Select new Mobject nodes on the canvas first.\n"
                "(Already-added nodes are ignored.)",
            )
            return
        for node in new_nodes:
            self._groups.setdefault(name, []).append(node.data.id)
            meta = self._meta.setdefault(name, {"source": "canvas", "members": []})
            if node.data.name not in meta.get("members", []):
                meta.setdefault("members", []).append(node.data.name)
        self._refresh_tree()

    # ── Remove Member ─────────────────────────────────────────────────────────

    def _remove_member(self):
        item = self.tree.currentItem()
        if item is None or item.parent() is None:
            return
        group_name = item.parent().data(0, Qt.ItemDataRole.UserRole)
        member_name = item.data(0, Qt.ItemDataRole.UserRole + 1)
        if not group_name or not member_name:
            return
        # Remove from ID list (canvas groups)
        ids = self._groups.get(group_name, [])
        new_ids = []
        for nid in ids:
            node = self.main_window.nodes.get(nid)
            if node and node.data.name == member_name:
                continue  # skip this one
            new_ids.append(nid)
        self._groups[group_name] = new_ids
        # Remove from meta members list (code-origin groups)
        meta_members = self._meta.get(group_name, {}).get("members", [])
        if member_name in meta_members:
            meta_members.remove(member_name)
        self._refresh_tree()

    # ── Delete Group ─────────────────────────────────────────────────────────

    def _delete_vgroup(self):
        name = self._current_group_name()
        if not name:
            return
        if (
            QMessageBox.question(
                self,
                "Delete VGroup",
                f"Delete VGroup '{name}'?\n\nThis does NOT delete the canvas nodes.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        self._groups.pop(name, None)
        self._meta.pop(name, None)
        self._refresh_tree()

    # ── Public API ────────────────────────────────────────────────────────────

    def get_groups(self) -> dict:
        return dict(self._groups)

    def set_groups(self, groups: dict) -> None:
        self._groups = dict(groups)
        # Rebuild meta for any groups that don't have it
        for name in self._groups:
            if name not in self._meta:
                self._meta[name] = {"source": "canvas", "members": []}
        self._refresh_tree()

    def register_snippet_vgroups(self, code: str, source: str = "snippet") -> int:
        """Parse code for VGroup assignments and register any new ones.

        Parses patterns like:  my_group = VGroup(circle, square, text)
        Extracts member variable names so they appear in the tree.
        Returns the count of newly registered VGroups.
        """
        import re

        # Capture name and the full argument list (handles multi-line with re.DOTALL)
        pattern = re.compile(
            r"^[ \t]*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*VGroup\s*\(([^)]*)\)",
            re.MULTILINE,
        )
        _SKIP = frozenset(("self", "cls", "None", "True", "False", "return"))
        new_count = 0
        for match in pattern.finditer(code):
            name = match.group(1)
            if name.startswith("__") or name in _SKIP:
                continue
            if name in self._groups:
                continue
            # Parse member names from argument list
            raw_args = match.group(2)
            members = []
            for arg in raw_args.split(","):
                arg = arg.strip()
                # Accept simple identifiers only (skip *args, keyword=val, literals)
                if re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", arg) and arg not in _SKIP:
                    members.append(arg)
            self._groups[name] = []  # no node IDs for code-origin groups
            self._meta[name] = {"source": source, "members": members}
            new_count += 1
        if new_count:
            self._refresh_tree()
        return new_count


class RecentsPanel(QWidget):
    """⭐ Recents — shows the top 5 Mobjects and top 5 Animations by actual
    insertion frequency.  Double-click any item to insert it on the canvas.

    Updates live: no manual refresh required.
    """

    add_requested = Signal(str, str)  # (type_str, class_name)

    # How many to show per category
    TOP_N = 5

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()
        # Auto-refresh whenever a node is inserted anywhere
        USAGE_TRACKER.updated.connect(self._refresh)
        self._refresh()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Header
        hdr = QLabel("Most-Used Elements")
        hdr.setStyleSheet("font-weight: bold; font-size: 13px; color: #1f2937;")
        layout.addWidget(hdr)

        sub = QLabel(
            f"Top {self.TOP_N} Mobjects and Animations ranked by how often "
            "you insert them.  Double-click to add to canvas."
        )
        sub.setWordWrap(True)
        sub.setStyleSheet("color: #6b7280; font-size: 10px;")
        layout.addWidget(sub)

        # Tree
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        self.tree.setStyleSheet(
            "QTreeWidget { border: 1px solid #e5e7eb; border-radius: 5px; }"
            "QTreeWidget::item { padding: 4px 2px; }"
            "QTreeWidget::item:selected { background: #fef9c3; color: #92400e; }"
        )
        self.tree.itemDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self.tree, stretch=1)

        # Footer hint
        footer = QLabel("💡 Insert nodes from Elements or Classes tab to populate.")
        footer.setWordWrap(True)
        footer.setStyleSheet("color: #9ca3af; font-size: 10px; font-style: italic;")
        self.footer_lbl = footer
        layout.addWidget(footer)

    def _refresh(self):
        """Rebuild the tree from live usage statistics."""
        self.tree.clear()

        mob_data = USAGE_TRACKER.top_mobjects(self.TOP_N)
        anim_data = USAGE_TRACKER.top_animations(self.TOP_N)

        self._populate_section("📦  Mobjects", mob_data, "#4f46e5", "mobject")
        self._populate_section("🎬  Animations", anim_data, "#7c3aed", "animation")

        # Show/hide hint depending on whether there is any data
        has_data = bool(mob_data or anim_data)
        self.footer_lbl.setVisible(not has_data)

    def _populate_section(
        self,
        label: str,
        data: list,  # [(class_name, count), ...]
        color: str,
        type_str: str,
    ):
        section = QTreeWidgetItem(self.tree)
        section.setText(0, label)
        section.setForeground(0, QBrush(QColor(color)))
        section.setFont(0, bold_font())
        section.setFlags(section.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        section.setToolTip(
            0, f"Double-click a {type_str} below to add it to the canvas"
        )

        if not data:
            empty = QTreeWidgetItem(section)
            empty.setText(0, "   (none yet)")
            empty.setForeground(0, QBrush(QColor("#9ca3af")))
            f = empty.font(0)
            f.setItalic(True)
            empty.setFont(0, f)
            empty.setFlags(empty.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        else:
            max_count = data[0][1] if data else 1
            for rank, (cls_name, count) in enumerate(data, 1):
                child = QTreeWidgetItem(section)
                # Bar width proportional to usage (1–10 chars)
                bar_len = max(1, round(count / max_count * 10))
                bar = "█" * bar_len
                child.setText(0, f"  {rank}. {cls_name}")
                plural = "s" if count != 1 else ""
                tip = f"{cls_name}\nInserted {count} time{plural}\nDouble-click to add to canvas"
                child.setToolTip(0, tip)
                # Store metadata for insertion
                child.setData(0, Qt.ItemDataRole.UserRole, cls_name)
                child.setData(0, Qt.ItemDataRole.UserRole + 1, type_str)
                # Usage bar as second column label embedded in text
                count_badge = QLabel(
                    f"<span style='color:{color};font-size:10px'>{bar} {count}×</span>"
                )
                count_badge.setContentsMargins(0, 0, 4, 0)
                self.tree.setItemWidget(child, 0, None)  # ensure text side is used
                child.setText(0, f"  {rank}.  {cls_name}  ·  {count}×  {bar}")
                child.setForeground(0, QBrush(QColor("#374151")))

        section.setExpanded(True)

    def _on_double_click(self, item: QTreeWidgetItem, _col: int):
        cls_name = item.data(0, Qt.ItemDataRole.UserRole)
        type_str = item.data(0, Qt.ItemDataRole.UserRole + 1)
        if cls_name and type_str:
            self.add_requested.emit(type_str, cls_name)


# ── GitHub Snippet Loader ─────────────────────────────────────────────────────
_SNIPPETS_DIR: Path = AppPaths.USER_DATA / "github_snippets"


class GitHubSnippetLoader(QWidget):
    snippet_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()
        self._scan_existing()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        hdr = QLabel("🐙 GitHub Snippets")
        hdr.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(hdr)
        url_row = QHBoxLayout()
        self.url_edit = QLineEdit()
        self.url_edit.setPlaceholderText("https://github.com/user/repo")
        url_row.addWidget(self.url_edit)
        btn_clone = QPushButton("⬇ Clone")
        btn_clone.clicked.connect(self._clone)
        btn_clone.setStyleSheet(
            "background-color: #1a7f37; color: white; padding: 4px 8px;"
        )
        url_row.addWidget(btn_clone)
        layout.addLayout(url_row)
        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet("color: #9ca3af; font-size: 11px;")
        layout.addWidget(self.status_lbl)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemDoubleClicked.connect(self._on_select)
        layout.addWidget(self.tree)
        btn_del = QPushButton("🗑 Remove Repo")
        btn_del.clicked.connect(self._remove_repo)
        layout.addWidget(btn_del)

    def _clone(self):
        url = self.url_edit.text().strip()
        if not url:
            return
        try:
            # Support HTTPS/HTTP GitHub URLs and SSH-style GitHub URLs.
            dev_name = None
            repo_name = None

            # SSH-style: git@github.com:user/repo.git
            if url.startswith("git@github.com:"):
                path_part = url[len("git@github.com:") :].rstrip("/")
                parts = path_part.split("/")
                if len(parts) >= 2:
                    dev_name = parts[0]
                    repo_name = parts[1]
                else:
                    raise ValueError("Incomplete SSH GitHub URL")
            else:
                parsed = urlparse(url)
                # Require a proper HTTP(S) URL with github.com as hostname
                if (
                    parsed.scheme not in ("http", "https")
                    or parsed.hostname != "github.com"
                ):
                    raise ValueError("Not a GitHub HTTPS/HTTP URL")
                path = parsed.path.lstrip("/").rstrip("/")
                parts = path.split("/")
                if len(parts) < 2:
                    raise ValueError("Incomplete GitHub repository path")
                dev_name = parts[0]
                repo_name = parts[1]

            # Normalize repository name (strip optional .git suffix)
            if repo_name.endswith(".git"):
                repo_name = repo_name[: -len(".git")]

        except Exception:
            self.status_lbl.setText("Invalid GitHub URL")
            return
        dest = _SNIPPETS_DIR / dev_name / repo_name
        if dest.exists():
            self.status_lbl.setText(f"Warning: Already cloned: {dev_name}/{repo_name}")
            return
        self.status_lbl.setText(f"⏳ Cloning {dev_name}/{repo_name}…")
        QApplication.processEvents()
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            result = subprocess.run(
                ["git", "clone", "--depth", "1", url, str(dest)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
            )
            if result.returncode == 0:
                self.status_lbl.setText(f"Cloned {dev_name}/{repo_name}")
                self._add_repo_to_tree(dev_name, repo_name, dest)
            else:
                self.status_lbl.setText(f"{result.stderr[:150]}")
        except FileNotFoundError:
            self.status_lbl.setText("'git' not found — install Git and add to PATH")
        except Exception as e:
            self.status_lbl.setText(f"{str(e)[:100]}")

    def _scan_existing(self):
        self.tree.clear()
        if not _SNIPPETS_DIR.exists():
            return
        for dev_dir in sorted(_SNIPPETS_DIR.iterdir()):
            if not dev_dir.is_dir():
                continue
            for repo_dir in sorted(dev_dir.iterdir()):
                if not repo_dir.is_dir():
                    continue
                self._add_repo_to_tree(dev_dir.name, repo_dir.name, repo_dir)

    def _add_repo_to_tree(self, dev, repo, path):
        root = QTreeWidgetItem(self.tree, [f"{dev}/{repo}"])
        root.setData(0, Qt.ItemDataRole.UserRole, str(path))
        for pf in sorted(path.rglob("*.py")):
            rel = str(pf.relative_to(path))
            child = QTreeWidgetItem(root, [rel])
            child.setData(0, Qt.ItemDataRole.UserRole, str(pf))
        root.setExpanded(True)

    def _on_select(self, item, _col):
        if not item.parent():
            return
        fp = item.data(0, Qt.ItemDataRole.UserRole)
        if fp and Path(fp).exists():
            try:
                self.snippet_selected.emit(Path(fp).read_text(encoding="utf-8"))
                self.status_lbl.setText(f"{Path(fp).name}")
            except Exception as e:
                self.status_lbl.setText(f"{e}")

    def _remove_repo(self):
        item = self.tree.currentItem()
        if not item:
            return
        root = item if not item.parent() else item.parent()
        path = root.data(0, Qt.ItemDataRole.UserRole)
        if not path:
            return
        r = QMessageBox.question(
            self,
            "Remove Repo",
            f"Delete '{root.text(0)}' from disk?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if r == QMessageBox.StandardButton.Yes:
            import shutil
            import stat
            import time

            try:
                # ═══════════════════════════════════════════════════════════════
                # WINDOWS-SAFE RECURSIVE DELETION
                #
                # Fixes WinError 5: Access Denied
                # - chmod files to remove read-only flags
                # - Retry with exponential backoff
                # - OS-specific force-close handling
                # ═══════════════════════════════════════════════════════════════

                def handle_remove_readonly(func, path, exc_info):
                    """
                    Error handler for shutil.rmtree() on Windows.

                    When a file is locked or read-only:
                    1. Make it writable with os.chmod()
                    2. Retry the operation
                    """
                    if not os.access(path, os.W_OK):
                        # File is read-only or locked
                        os.chmod(path, stat.S_IWUSR | stat.S_IRUSR)
                        func(path)
                    else:
                        raise

                # Attempt 1: Standard deletion
                try:
                    shutil.rmtree(path, onerror=handle_remove_readonly)
                    LOGGER.info(f"Successfully deleted GitHub repo: {path}")
                except Exception as e1:
                    # Attempt 2: Force-chmod all files first, then retry
                    LOGGER.warning(
                        f"First deletion attempt failed: {e1}, retrying with force-chmod..."
                    )
                    try:
                        for root_dir, dirs, files in os.walk(path):
                            for d in dirs:
                                os.chmod(
                                    os.path.join(root_dir, d),
                                    stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR,
                                )
                            for f in files:
                                os.chmod(
                                    os.path.join(root_dir, f),
                                    stat.S_IWUSR | stat.S_IRUSR,
                                )
                        time.sleep(0.5)  # Brief delay for file handles to release
                        shutil.rmtree(path)
                        LOGGER.info(f"Successfully deleted GitHub repo (retry): {path}")
                    except Exception as e2:
                        # Attempt 3: Windows-specific: try moving to temp first
                        if sys.platform == "win32":
                            LOGGER.warning(
                                f"Second deletion failed: {e2}, attempting temp relocation..."
                            )
                            try:
                                temp_loc = (
                                    Path(tempfile.gettempdir())
                                    / f"efm_del_{uuid.uuid4().hex[:8]}"
                                )
                                shutil.move(path, str(temp_loc))
                                shutil.rmtree(
                                    str(temp_loc), onerror=handle_remove_readonly
                                )
                                LOGGER.info(
                                    f"Successfully deleted via temp relocation: {path}"
                                )
                            except Exception as e3:
                                raise Exception(
                                    f"Could not delete after 3 attempts. Last error: {e3}"
                                )
                        else:
                            raise Exception(f"Could not delete. Last error: {e2}")

                # Remove from tree on success
                idx = self.tree.indexOfTopLevelItem(root)
                self.tree.takeTopLevelItem(idx)
                self.status_lbl.setText(f"Deleted '{root.text(0)}'")

            except Exception as e:
                LOGGER.error(f"Failed to delete repo: {e}")
                QMessageBox.warning(
                    self, "Error", f"Could not delete repository:\n{str(e)[:100]}"
                )


# ── Editable Project Name Widget ──────────────────────────────────────────────


class ProjectNameWidget(QWidget):
    name_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        lbl = QLabel("📁")
        lbl.setStyleSheet("font-size: 14px;")
        layout.addWidget(lbl)
        self.edit = QLineEdit("Untitled")
        self.edit.setFixedWidth(160)
        self.edit.setPlaceholderText("Project name…")
        self.edit.setToolTip(
            "Editable project name. Press Enter to rename the .efp file."
        )
        self.edit.returnPressed.connect(self._on_commit)
        self.edit.editingFinished.connect(self._on_commit)
        layout.addWidget(self.edit)
        sfx = QLabel(".efp")
        sfx.setStyleSheet("color: #6b7280;")
        layout.addWidget(sfx)

    def _on_commit(self):
        name = self.edit.text().strip()
        if name:
            self.name_changed.emit(name)

    def set_name(self, stem: str):
        self.edit.blockSignals(True)
        self.edit.setText(stem)
        self.edit.blockSignals(False)

    def get_name(self) -> str:
        return self.edit.text().strip() or "Untitled"


# ==============================================================================
# 9. MAIN WINDOW
# ==============================================================================
