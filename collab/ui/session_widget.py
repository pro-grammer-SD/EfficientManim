"""
collab/ui/session_widget.py — Persistent toolbar widget showing session state.

Shows:  ●  PIN: 482917   3 connected   [Copy PIN]  [End]
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QPushButton, QWidget,
)


class SessionWidget(QWidget):
    """
    Always visible in the top toolbar.
    When no session is active: shows a neutral "Not connected" indicator.
    When active: shows PIN, participant count, Copy and End buttons.
    """

    end_requested = Signal()   # emitted when user clicks End button

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pin_value = ""
        self._build_ui()
        self.set_active(False)

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(8)

        # Status indicator dot
        self.dot = QLabel("●")
        self.dot.setStyleSheet("color: #9ca3af; font-size: 14px;")
        layout.addWidget(self.dot)

        # Role badge (Host / Guest)
        self.role_badge = QLabel("")
        self.role_badge.setStyleSheet(
            "color: #374151; font-size: 10px; font-weight: bold; "
            "background: #f3f4f6; border-radius: 4px; padding: 1px 5px;"
        )
        self.role_badge.hide()
        layout.addWidget(self.role_badge)

        # PIN display
        self.pin_label = QLabel("No collaboration session")
        self.pin_label.setStyleSheet("color: #6b7280; font-size: 11px;")
        layout.addWidget(self.pin_label)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color: #e5e7eb;")
        sep.setFixedWidth(1)
        layout.addWidget(sep)

        # Participant count
        self.count_label = QLabel("")
        self.count_label.setStyleSheet("color: #6b7280; font-size: 11px;")
        self.count_label.hide()
        layout.addWidget(self.count_label)

        # Copy PIN button
        self.copy_btn = QPushButton("Copy PIN")
        self.copy_btn.setFixedHeight(22)
        self.copy_btn.setStyleSheet(
            "QPushButton { background: #ecfdf5; color: #065f46; border: 1px solid #6ee7b7; "
            "border-radius: 4px; padding: 0 8px; font-size: 11px; }"
            "QPushButton:hover { background: #d1fae5; }"
        )
        self.copy_btn.clicked.connect(self._copy_pin)
        self.copy_btn.hide()
        layout.addWidget(self.copy_btn)

        # End / Disconnect button
        self.end_btn = QPushButton("Disconnect")
        self.end_btn.setFixedHeight(22)
        self.end_btn.setStyleSheet(
            "QPushButton { background: #fef2f2; color: #991b1b; border: 1px solid #fca5a5; "
            "border-radius: 4px; padding: 0 8px; font-size: 11px; }"
            "QPushButton:hover { background: #fee2e2; }"
        )
        self.end_btn.clicked.connect(self.end_requested)
        self.end_btn.hide()
        layout.addWidget(self.end_btn)

    # ── Public API ────────────────────────────────────────────────────

    def set_active(self, active: bool) -> None:
        if active:
            self.dot.setStyleSheet("color: #10b981; font-size: 14px;")
            self.copy_btn.show()
            self.end_btn.show()
            self.count_label.show()
            self.role_badge.show()
        else:
            self.dot.setStyleSheet("color: #9ca3af; font-size: 14px;")
            self.pin_label.setText("No collaboration session")
            self.pin_label.setStyleSheet("color: #6b7280; font-size: 11px;")
            self.copy_btn.hide()
            self.end_btn.hide()
            self.count_label.hide()
            self.role_badge.hide()
            self._pin_value = ""

    def set_pin(self, pin: str) -> None:
        self._pin_value = pin
        if pin:
            formatted = f"{pin[:3]} {pin[3:]}" if len(pin) == 6 else pin
            self.pin_label.setText(f"PIN: {formatted}")
            self.pin_label.setStyleSheet(
                "color: #1f2937; font-size: 12px; font-weight: bold; font-family: Consolas;"
            )
        else:
            self.pin_label.setText("No collaboration session")
            self.pin_label.setStyleSheet("color: #6b7280; font-size: 11px;")

    def set_count(self, count: int) -> None:
        if count == 0:
            self.count_label.setText("(only you)")
        elif count == 1:
            self.count_label.setText("1 other")
        else:
            self.count_label.setText(f"{count} others")

    def set_role(self, role: str) -> None:
        """role is 'Host' or 'Guest'."""
        if role:
            self.role_badge.setText(role.upper())
            self.role_badge.show()
            color = "#1e40af" if role == "Host" else "#5b21b6"
            bg = "#dbeafe" if role == "Host" else "#ede9fe"
            self.role_badge.setStyleSheet(
                f"color: {color}; font-size: 10px; font-weight: bold; "
                f"background: {bg}; border-radius: 4px; padding: 1px 5px;"
            )
        else:
            self.role_badge.hide()

    # ── Internal ──────────────────────────────────────────────────────

    def _copy_pin(self) -> None:
        if not self._pin_value:
            return
        QApplication.clipboard().setText(self._pin_value)
        self.copy_btn.setText("✅ Copied!")
        QTimer.singleShot(2000, lambda: self.copy_btn.setText("Copy PIN"))
