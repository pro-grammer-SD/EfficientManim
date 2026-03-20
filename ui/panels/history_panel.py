from __future__ import annotations
# -*- coding: utf-8 -*-

from datetime import datetime

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QLabel, QListWidget, QListWidgetItem, QVBoxLayout, QWidget

from core.history_manager import HistoryManager


class HistoryPanel(QWidget):
    """Optional history timeline panel."""

    def __init__(self, history_manager: HistoryManager | None = None):
        super().__init__()
        self._history = history_manager

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        header = QLabel("History Timeline")
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        header.setFont(font)
        layout.addWidget(header)

        self.list = QListWidget()
        self.list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        layout.addWidget(self.list)

        if self._history is not None:
            self._history.history_changed.connect(self.refresh)
            self.refresh()

    def set_manager(self, history_manager: HistoryManager) -> None:
        self._history = history_manager
        self._history.history_changed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        if self._history is None:
            return
        snapshots, current_idx = self._history.get_timeline()
        self.list.clear()
        for i, snap in enumerate(snapshots):
            ts = (
                snap.timestamp.strftime("%H:%M:%S")
                if isinstance(snap.timestamp, datetime)
                else str(snap.timestamp)
            )
            label = f"{i:03d}  {snap.description}  •  {ts}"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, i)
            if i == current_idx:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
                item.setSelected(True)
            self.list.addItem(item)
