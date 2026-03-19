from __future__ import annotations
# -*- coding: utf-8 -*-

import uuid

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsPathItem

from utils.logger import LOGGER


class WireItem(QGraphicsPathItem):
    """Connection between sockets with robust path management."""

    def __init__(self, start_socket, end_socket, wire_id: str | None = None):
        super().__init__()
        self.start_socket = start_socket
        self.end_socket = end_socket
        self.wire_id = wire_id or uuid.uuid4().hex
        if not self.start_socket or not self.end_socket:
            raise ValueError("Both start_socket and end_socket must be valid")

        self.setZValue(-1)
        pen = QPen(QColor("#7f8c8d"), 2.5)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        self.setPen(pen)
        self.update_path()

    def update_path(self) -> None:
        try:
            if not self.start_socket or not self.end_socket:
                return

            p1 = self.start_socket.get_scene_pos()
            p2 = self.end_socket.get_scene_pos()

            if not isinstance(p1, QPointF) or not isinstance(p2, QPointF):
                LOGGER.warn("Invalid socket positions for wire update")
                return

            path = QPainterPath()
            path.moveTo(p1)
            dx = p2.x() - p1.x()
            ctrl1 = QPointF(p1.x() + dx * 0.5, p1.y())
            ctrl2 = QPointF(p2.x() - dx * 0.5, p2.y())
            path.cubicTo(ctrl1, ctrl2, p2)
            self.setPath(path)
        except Exception as exc:
            LOGGER.warn(f"Error updating wire path: {exc}")
