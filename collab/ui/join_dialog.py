"""
collab/ui/join_dialog.py — "Join Collaboration" PIN-entry dialog.

The user enters a 6-digit PIN and clicks Connect. The dialog shows
connecting status and error feedback.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QIntValidator
from PySide6.QtWidgets import (
    QDialog, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QVBoxLayout,
)


class JoinCollabDialog(QDialog):
    """
    Non-blocking PIN input dialog.
    After exec() returns Accepted, call .pin() to retrieve the entered PIN.
    Connection attempt and error display is handled by the caller.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🤝 Join Live Collaboration")
        self.setModal(True)
        self.setMinimumWidth(360)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        # ── Header ────────────────────────────────────────────────────
        header = QLabel("Enter the 6-digit session PIN")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("color: #374151; font-size: 13px; font-weight: bold;")
        layout.addWidget(header)

        sub = QLabel("Ask the session host for the PIN shown in their toolbar.")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub.setStyleSheet("color: #9ca3af; font-size: 10px;")
        layout.addWidget(sub)

        # ── PIN input ─────────────────────────────────────────────────
        pin_frame = QFrame()
        pin_frame.setStyleSheet(
            "QFrame { background: #f9fafb; border: 1px solid #d1d5db; border-radius: 8px; }"
        )
        pin_fl = QVBoxLayout(pin_frame)
        pin_fl.setContentsMargins(16, 12, 16, 12)

        self.pin_input = QLineEdit()
        self.pin_input.setMaxLength(6)
        self.pin_input.setValidator(QIntValidator(100000, 999999))
        self.pin_input.setPlaceholderText("000000")
        self.pin_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        f = QFont("Consolas", 28, QFont.Weight.Bold)
        self.pin_input.setFont(f)
        self.pin_input.setStyleSheet(
            "QLineEdit { border: none; background: transparent; color: #1f2937; "
            "letter-spacing: 6px; }"
        )
        # Connect on Enter
        self.pin_input.returnPressed.connect(self._on_connect)
        pin_fl.addWidget(self.pin_input)
        layout.addWidget(pin_frame)

        # ── Status label ──────────────────────────────────────────────
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("font-size: 11px; color: #6b7280;")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # ── Buttons ───────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.connect_btn = QPushButton("🔗  Connect")
        self.connect_btn.setDefault(True)
        self.connect_btn.setStyleSheet(
            "QPushButton { background: #3b82f6; color: white; font-weight: bold; "
            "border-radius: 6px; padding: 8px 20px; }"
            "QPushButton:hover { background: #2563eb; }"
            "QPushButton:disabled { background: #9ca3af; }"
        )
        self.connect_btn.clicked.connect(self._on_connect)
        btn_row.addWidget(self.connect_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(
            "QPushButton { background: #f3f4f6; color: #374151; "
            "border-radius: 6px; padding: 8px 16px; }"
            "QPushButton:hover { background: #e5e7eb; }"
        )
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        # Auto-focus PIN input
        self.pin_input.setFocus()

    # ── Public API ────────────────────────────────────────────────────

    def pin(self) -> str:
        """Return the entered PIN string (digits only)."""
        return self.pin_input.text().strip()

    def set_status(self, text: str, error: bool = False, success: bool = False) -> None:
        if error:
            color = "#dc2626"
        elif success:
            color = "#10b981"
        else:
            color = "#6b7280"
        self.status_label.setStyleSheet(f"font-size: 11px; color: {color};")
        self.status_label.setText(text)

    def set_connecting(self, connecting: bool) -> None:
        self.connect_btn.setEnabled(not connecting)
        self.pin_input.setEnabled(not connecting)
        if connecting:
            self.set_status("Connecting…")

    # ── Internal ──────────────────────────────────────────────────────

    def _on_connect(self) -> None:
        pin = self.pin()
        if len(pin) != 6:
            self.set_status("Please enter all 6 digits.", error=True)
            self.pin_input.setFocus()
            return
        self.accept()
