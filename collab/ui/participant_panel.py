"""
collab/ui/participant_panel.py — Live participant list dock panel.

Shows each collaborator's name, colour swatch, and which node they are
currently editing. Lock state is indicated per-row.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (
    QLabel, QListWidget, QListWidgetItem,
    QVBoxLayout, QWidget,
)


class ParticipantPanel(QWidget):
    """
    Displays all participants connected to the current session.
    Call update_participants(participants_dict) whenever the dict changes.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title = QLabel("🤝 Participants")
        title.setStyleSheet("font-weight: bold; font-size: 12px; color: #1f2937;")
        layout.addWidget(title)

        self.list_widget = QListWidget()
        self.list_widget.setAlternatingRowColors(True)
        self.list_widget.setStyleSheet(
            "QListWidget { border: 1px solid #e5e7eb; border-radius: 6px; }"
            "QListWidget::item { padding: 6px 8px; }"
            "QListWidget::item:selected { background: #eff6ff; color: #1e40af; }"
        )
        layout.addWidget(self.list_widget)

        self.status_label = QLabel("0 connected")
        self.status_label.setStyleSheet("color: #9ca3af; font-size: 10px;")
        layout.addWidget(self.status_label)

    # ── Public API ────────────────────────────────────────────────────

    def update_participants(self, participants: dict) -> None:
        """
        Rebuild the list from the participants dict:
        { client_id: {"id", "name", "color", "active_node"} }
        """
        self.list_widget.clear()

        for pid, info in participants.items():
            name = info.get("name") or pid[:8]
            color_hex = info.get("color") or "#e5e7eb"
            active_node = info.get("active_node")

            # Build display text
            if active_node:
                text = f"{name}  ✏ {active_node}"
            else:
                text = name

            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, pid)
            item.setToolTip(f"ID: {pid[:12]}\nColor: {color_hex}")

            # Colour swatch as background on left portion
            try:
                q_color = QColor(color_hex)
                if q_color.isValid():
                    # Lighten for readability
                    bg = QColor(q_color)
                    bg.setAlphaF(0.35)
                    item.setBackground(QBrush(bg))
                    # Use a dark text colour for contrast
                    item.setForeground(QBrush(QColor("#1f2937")))
            except Exception:
                pass

            # Bold font for the active editor
            if active_node:
                f = QFont()
                f.setBold(True)
                item.setFont(f)

            self.list_widget.addItem(item)

        n = len(participants)
        if n == 0:
            self.status_label.setText("No participants")
        elif n == 1:
            self.status_label.setText("1 participant")
        else:
            self.status_label.setText(f"{n} participants")
