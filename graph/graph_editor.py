from __future__ import annotations
# -*- coding: utf-8 -*-

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsView,
    QMenu,
    QMessageBox,
)

from graph.edge import WireItem
from graph.node import NodeType, SocketItem
from core.config import SETTINGS
from utils.logger import LOGGER


class GraphScene(QGraphicsScene):
    """Manages node connections and logic enforcement."""

    selection_changed_signal = Signal()
    graph_changed_signal = Signal()

    def __init__(self):
        super().__init__()
        self.drag_wire = None
        self.start_socket = None
        self.main_window = None
        self.setBackgroundBrush(QBrush(QColor("#f4f6f7")))

    def notify_change(self) -> None:
        self.graph_changed_signal.emit()

    def mousePressEvent(self, event):
        item = self.itemAt(event.scenePos(), self.views()[0].transform())

        if isinstance(item, SocketItem):
            self.start_socket = item
            self.drag_wire = QGraphicsPathItem()
            self.drag_wire.setPen(QPen(Qt.GlobalColor.black, 2, Qt.PenStyle.DashLine))
            self.addItem(self.drag_wire)
            return

        super().mousePressEvent(event)
        self.selection_changed_signal.emit()

    def mouseMoveEvent(self, event):
        if self.drag_wire and self.start_socket:
            p1 = self.start_socket.get_scene_pos()
            p2 = event.scenePos()
            path = QPainterPath()
            path.moveTo(p1)
            path.lineTo(p2)
            self.drag_wire.setPath(path)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.drag_wire:
            self.removeItem(self.drag_wire)
            self.drag_wire = None

            end_item = self.itemAt(event.scenePos(), self.views()[0].transform())

            if isinstance(end_item, SocketItem) and end_item != self.start_socket:
                self.try_connect(self.start_socket, end_item)

            self.start_socket = None

        super().mouseReleaseEvent(event)

    def try_connect(self, s1, s2, wire_id=None):
        if s1 == s2:
            self.show_warning(
                "Invalid Connection", "Cannot connect a socket to itself."
            )
            return None

        if s1.is_output == s2.is_output:
            direction = "both outputs" if s1.is_output else "both inputs"
            self.show_warning(
                "Invalid Connection",
                f"Cannot connect {direction}. Please connect output to input.",
            )
            return None

        if s1.is_output:
            out_sock, in_sock = s1, s2
        else:
            out_sock, in_sock = s2, s1

        node_src = out_sock.parentItem()
        node_dst = in_sock.parentItem()

        src_type = node_src.node_data.type
        dst_type = node_dst.node_data.type

        allowed = {
            (NodeType.MOBJECT, NodeType.ANIMATION),
            (NodeType.MOBJECT, NodeType.VGROUP),
            (NodeType.ANIMATION, NodeType.PLAY),
            (NodeType.VGROUP, NodeType.PLAY),
            (NodeType.PLAY, NodeType.WAIT),
            (NodeType.WAIT, NodeType.PLAY),
            (NodeType.PLAY, NodeType.PLAY),
        }
        if (src_type, dst_type) not in allowed:
            friendly = {
                NodeType.MOBJECT: "Mobject",
                NodeType.ANIMATION: "Animation",
                NodeType.PLAY: "Play",
                NodeType.WAIT: "Wait",
                NodeType.VGROUP: "VGroup",
            }
            self.show_warning(
                "Invalid Connection",
                f"Cannot connect {friendly.get(src_type, src_type.name)} -> "
                f"{friendly.get(dst_type, dst_type.name)}.\n\n"
                "Valid connections:\n"
                "  Mobject -> Animation\n"
                "  Mobject -> VGroup\n"
                "  Animation -> Play\n"
                "  VGroup -> Play\n"
                "  Play -> Wait\n"
                "  Wait -> Play\n"
                "  Play -> Play (ordering)",
            )
            return None

        try:
            for existing in out_sock.links:
                try:
                    if (
                        existing.start_socket == out_sock
                        and existing.end_socket == in_sock
                    ):
                        return existing
                except Exception:
                    pass

            wire = WireItem(out_sock, in_sock, wire_id=wire_id)
            self.addItem(wire)
            out_sock.links.append(wire)
            in_sock.links.append(wire)

            self.notify_change()
            LOGGER.info(
                f"Connected {node_src.node_data.name} -> {node_dst.node_data.name}"
            )
            if self.main_window is not None:
                try:
                    self.main_window.on_wire_added(wire)
                except Exception:
                    pass
            return wire
        except Exception as exc:
            LOGGER.error(f"Failed to create connection: {exc}")
            self.show_warning("Connection Error", f"Failed to create connection: {exc}")
            return None

    def show_warning(self, title, msg):
        views = self.views()
        if views:
            QMessageBox.warning(views[0], title, msg)


class GraphView(QGraphicsView):
    """Custom view with zoom and pan support."""

    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._is_panning = False

    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            zoom_factor = 1.15
            if event.angleDelta().y() > 0:
                self.scale(zoom_factor, zoom_factor)
            else:
                self.scale(1 / zoom_factor, 1 / zoom_factor)
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._is_panning = True
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            dummy = type(event)(event)
            super().mousePressEvent(dummy)
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._is_panning = False
            self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if (
            event.key() == Qt.Key.Key_A
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            if hasattr(self.scene(), "selectAllItems"):
                self.scene().selectAllItems()
            else:
                for item in self.scene().items():
                    if hasattr(item, "setSelected"):
                        item.setSelected(True)
            event.accept()
        else:
            super().keyPressEvent(event)

    def contextMenuEvent(self, event):
        win = getattr(self.scene(), "main_window", None)
        if win is None:
            return super().contextMenuEvent(event)

        selected_nodes = [
            item for item in self.scene().selectedItems() if hasattr(item, "data")
        ]
        has_selection = bool(selected_nodes)
        has_animation = any(
            getattr(item.data, "type", None) == NodeType.ANIMATION
            for item in selected_nodes
        )
        has_mobject = any(
            getattr(item.data, "type", None) == NodeType.MOBJECT
            for item in selected_nodes
        )

        menu = QMenu(self)
        act_scene = menu.addAction("Explain This Scene")
        act_nodes = menu.addAction("Explain Selected Nodes")
        act_anim = menu.addAction("Explain Selected Animation")
        act_objs = menu.addAction("Explain Selected Objects")

        act_nodes.setEnabled(has_selection)
        act_anim.setEnabled(has_animation)
        act_objs.setEnabled(has_mobject)

        teacher_on = bool(SETTINGS.get("TEACHER_MODE_ENABLED", False, type=bool))
        if teacher_on:
            menu.addSeparator()
            act_lesson = menu.addAction("Generate Lesson Notes")
            export_menu = menu.addMenu("Export Teaching Notes")
            export_md = export_menu.addAction("Markdown (.md)")
            export_txt = export_menu.addAction("Plain Text (.txt)")
            export_copy = export_menu.addAction("Copy to Clipboard")
        else:
            act_lesson = export_menu = export_md = export_txt = export_copy = None

        action = menu.exec(event.globalPos())

        if action == act_scene:
            win.explain_current_context()
        elif action == act_nodes:
            win.explain_selected_nodes()
        elif action == act_anim:
            win.explain_selected_animation()
        elif action == act_objs:
            win.explain_selected_objects()
        elif act_lesson and action == act_lesson:
            win.generate_lesson_notes()
        elif export_md and action == export_md:
            if hasattr(win, "panel_explain"):
                win.panel_explain._export_lesson("md")
        elif export_txt and action == export_txt:
            if hasattr(win, "panel_explain"):
                win.panel_explain._export_lesson("txt")
        elif export_copy and action == export_copy:
            if hasattr(win, "panel_explain"):
                win.panel_explain._export_lesson("copy")
