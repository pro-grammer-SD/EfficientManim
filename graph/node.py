from __future__ import annotations
# -*- coding: utf-8 -*-

import uuid
from enum import Enum, auto

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsItem, QGraphicsPathItem


class NodeType(Enum):
    MOBJECT = auto()
    ANIMATION = auto()
    PLAY = auto()
    WAIT = auto()
    VGROUP = auto()


class NodeData:
    """Enhanced node data with type safety, parameter metadata, and AI support."""

    def __init__(self, name: str, n_type: NodeType, cls_name: str):
        self.id = str(uuid.uuid4())
        self.name = name
        self.var_name = name
        self.type = n_type
        self.cls_name = cls_name
        self.params: dict = {}
        self.param_metadata: dict = {}
        self.pos_x = 0
        self.pos_y = 0
        self.preview_path: str | None = None
        self.audio_asset_id: str | None = None
        self.voiceover_transcript: str | None = None
        self.voiceover_duration: float = 0.0
        self.is_ai_generated = False
        self.ai_source: str | None = None
        self.ai_code_snippet: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "var_name": self.var_name,
            "type": self.type.name,
            "cls_name": self.cls_name,
            "params": self.params,
            "param_metadata": self.param_metadata,
            "pos": (self.pos_x, self.pos_y),
            "preview_path": self.preview_path,
            "audio_asset_id": self.audio_asset_id,
            "voiceover_transcript": self.voiceover_transcript,
            "voiceover_duration": self.voiceover_duration,
            "is_ai_generated": self.is_ai_generated,
            "ai_source": self.ai_source,
            "ai_code_snippet": self.ai_code_snippet,
        }

    @staticmethod
    def from_dict(d: dict) -> "NodeData":
        n = NodeData(d["name"], NodeType[d["type"]], d["cls_name"])
        n.id = d["id"]
        n.var_name = d.get("var_name", d["name"])
        n.params = d.get("params", {})
        n.param_metadata = d.get("param_metadata", {})
        n.pos_x, n.pos_y = d.get("pos", (0, 0))
        n.preview_path = d.get("preview_path")
        n.audio_asset_id = d.get("audio_asset_id")
        n.voiceover_transcript = d.get("voiceover_transcript")
        n.voiceover_duration = d.get("voiceover_duration", 0.0)
        n.is_ai_generated = d.get("is_ai_generated", False)
        n.ai_source = d.get("ai_source")
        n.ai_code_snippet = d.get("ai_code_snippet")
        return n

    def is_param_enabled(self, param_name: str) -> bool:
        return self.param_metadata.get(param_name, {}).get("enabled", True)

    def set_param_enabled(self, param_name: str, enabled: bool) -> None:
        if param_name not in self.param_metadata:
            self.param_metadata[param_name] = {}
        self.param_metadata[param_name]["enabled"] = enabled

    def should_escape_string(self, param_name: str) -> bool:
        return self.param_metadata.get(param_name, {}).get("escape", False)

    def set_escape_string(self, param_name: str, escape: bool) -> None:
        if param_name not in self.param_metadata:
            self.param_metadata[param_name] = {}
        self.param_metadata[param_name]["escape"] = escape


class SocketItem(QGraphicsPathItem):
    """Port for connecting nodes."""

    def __init__(self, parent, is_output: bool):
        super().__init__(parent)
        self.is_output = is_output
        self.radius = 6
        self.links: list = []

        path = QPainterPath()
        path.addEllipse(-self.radius, -self.radius, self.radius * 2, self.radius * 2)
        self.setPath(path)

        color = QColor("#2ecc71") if not is_output else QColor("#e74c3c")
        self.setBrush(QBrush(color))
        self.setPen(QPen(Qt.GlobalColor.black, 1))

    def get_scene_pos(self) -> QPointF:
        return self.scenePos()


class NodeItem(QGraphicsItem):
    """Visual representation of NodeData."""

    def __init__(self, data: NodeData):
        super().__init__()
        self.node_data = data
        self._window = None
        self._init_geometry()

    @property
    def data(self) -> NodeData:  # type: ignore[override]
        return self.node_data

    @data.setter
    def data(self, value: NodeData):
        self.node_data = value

    def _init_geometry(self) -> None:
        self.width = 180
        self.height = 90

        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.setPos(self.node_data.pos_x, self.node_data.pos_y)

        self.in_socket = SocketItem(self, False)
        self.in_socket.setPos(0, self.height / 2)

        self.out_socket = SocketItem(self, True)
        self.out_socket.setPos(self.width, self.height / 2)

    def boundingRect(self):
        return QRectF(0, 0, self.width, self.height)

    def paint(self, painter, option, widget=None):
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 30))
        painter.drawRoundedRect(4, 4, self.width, self.height, 8, 8)

        painter.setBrush(QColor("white"))
        pen = QPen(QColor("#bdc3c7"), 1.5)
        if self.isSelected():
            pen = QPen(QColor("#3498db"), 2.5)
        painter.setPen(pen)
        painter.drawRoundedRect(0, 0, self.width, self.height, 8, 8)

        header_h = 28
        header_colors = {
            NodeType.MOBJECT: "#3498db",
            NodeType.ANIMATION: "#9b59b6",
            NodeType.PLAY: "#27ae60",
            NodeType.WAIT: "#e67e22",
            NodeType.VGROUP: "#16a085",
        }
        header_color = QColor(header_colors.get(self.node_data.type, "#3498db"))

        path = QPainterPath()
        path.addRoundedRect(0, 0, self.width, header_h, 8, 8)
        painter.setClipPath(path)
        painter.fillPath(path, header_color)
        painter.setClipping(False)

        painter.setPen(Qt.GlobalColor.white)
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        name = self.node_data.name
        painter.drawText(
            QRectF(8, 0, self.width - 16, header_h),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            name,
        )

        painter.setPen(QColor("#7f8c8d"))
        painter.setFont(QFont("Segoe UI", 9))
        cls_name = self.node_data.cls_name
        painter.drawText(
            QRectF(8, 35, self.width - 16, 20),
            Qt.AlignmentFlag.AlignLeft,
            cls_name,
        )

        if self.node_data.type in (NodeType.PLAY, NodeType.WAIT, NodeType.VGROUP):
            badge_map = {
                NodeType.PLAY: "play()",
                NodeType.WAIT: "wait()",
                NodeType.VGROUP: "group",
            }
            badge = badge_map.get(self.node_data.type, "")
            painter.setPen(QColor(header_color).darker(130))
            painter.setFont(QFont("Segoe UI", 8))
            painter.drawText(
                QRectF(8, 55, self.width - 16, 20),
                Qt.AlignmentFlag.AlignLeft,
                badge,
            )

        if self.node_data.preview_path:
            painter.setBrush(QColor("#2ecc71"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(self.width - 16, self.height - 16, 8, 8)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            self.node_data.pos_x = value.x()
            self.node_data.pos_y = value.y()
            if hasattr(self, "in_socket") and hasattr(self, "out_socket"):
                for w in self.in_socket.links + self.out_socket.links:
                    w.update_path()
            if self.scene():
                s = self.scene()
                if hasattr(s, "notify_change"):
                    s.notify_change()
            if self._window is not None:
                try:
                    self._window.on_node_moved(self)
                except Exception:
                    pass
        return super().itemChange(change, value)
