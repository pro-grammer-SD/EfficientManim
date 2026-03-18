"""
collab/ui/start_dialog.py — "Start Collaboration" host dialog.

Shows the session PIN, host IP, participant count, and a Copy button.
Updates participant count live via set_participant_count().
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QColor
from PySide6.QtWidgets import (
    QApplication, QDialog, QFrame, QHBoxLayout,
    QLabel, QPushButton, QVBoxLayout,
)


class StartCollabDialog(QDialog):
    """
    Displayed when the host starts a session.
    Stays open as long as the host wants it; closing it does NOT end the session.
    """

    def __init__(self, pin: str, host_ip: str = "", port: int = 0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🤝 Live Collaboration — Active")
        self.setModal(False)          # non-modal so editor stays usable
        self.setMinimumWidth(380)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, False)

        self._pin = pin
        self._build_ui(pin, host_ip, port)

    def _build_ui(self, pin: str, host_ip: str, port: int) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        # ── Header ────────────────────────────────────────────────────
        header = QLabel("Share this PIN with collaborators")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("color: #6b7280; font-size: 12px;")
        layout.addWidget(header)

        # ── PIN display ───────────────────────────────────────────────
        pin_frame = QFrame()
        pin_frame.setStyleSheet(
            "QFrame { background: #f0fdf4; border: 2px solid #10b981; border-radius: 10px; }"
        )
        pin_layout = QVBoxLayout(pin_frame)
        pin_layout.setContentsMargins(16, 12, 16, 12)
        pin_layout.setSpacing(4)

        pin_label_title = QLabel("SESSION PIN")
        pin_label_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pin_label_title.setStyleSheet(
            "color: #065f46; font-size: 10px; font-weight: bold; letter-spacing: 2px; border: none;"
        )
        pin_layout.addWidget(pin_label_title)

        # Large spaced PIN digits
        self.pin_label = QLabel(self._format_pin(pin))
        self.pin_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        f = QFont("Consolas", 36, QFont.Weight.Bold)
        self.pin_label.setFont(f)
        self.pin_label.setStyleSheet("color: #065f46; letter-spacing: 8px; border: none;")
        pin_layout.addWidget(self.pin_label)

        layout.addWidget(pin_frame)

        # ── Network info ──────────────────────────────────────────────
        if host_ip or port:
            net_label = QLabel(f"ws://{host_ip}:{port}" if host_ip and port else "")
            net_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            net_label.setStyleSheet("color: #9ca3af; font-size: 10px; font-family: Consolas;")
            layout.addWidget(net_label)

        # ── Participant count ─────────────────────────────────────────
        self.participants_label = QLabel("● Session active — 0 participants connected")
        self.participants_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.participants_label.setStyleSheet("color: #10b981; font-size: 11px;")
        layout.addWidget(self.participants_label)

        # ── Buttons ───────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.copy_btn = QPushButton("📋  Copy PIN")
        self.copy_btn.setStyleSheet(
            "QPushButton { background: #10b981; color: white; font-weight: bold; "
            "border-radius: 6px; padding: 8px 16px; }"
            "QPushButton:hover { background: #059669; }"
        )
        self.copy_btn.clicked.connect(self._copy_pin)
        btn_row.addWidget(self.copy_btn)

        close_btn = QPushButton("Close (session stays active)")
        close_btn.setStyleSheet(
            "QPushButton { background: #f3f4f6; color: #374151; "
            "border-radius: 6px; padding: 8px 16px; }"
            "QPushButton:hover { background: #e5e7eb; }"
        )
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)

        layout.addLayout(btn_row)

        # ── Instructions ──────────────────────────────────────────────
        instructions = QLabel(
            "Collaborators: go to Collaboration → Join Collaboration\n"
            "and enter the PIN above."
        )
        instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        instructions.setStyleSheet("color: #9ca3af; font-size: 10px;")
        layout.addWidget(instructions)

    # ── Public API ────────────────────────────────────────────────────

    def set_participant_count(self, n: int) -> None:
        if n == 0:
            self.participants_label.setText("● Session active — 0 participants connected")
        elif n == 1:
            self.participants_label.setText("● 1 participant connected")
        else:
            self.participants_label.setText(f"● {n} participants connected")

    # ── Internal ──────────────────────────────────────────────────────

    @staticmethod
    def _format_pin(pin: str) -> str:
        """Format '482917' as '482 917' for readability."""
        if len(pin) == 6:
            return f"{pin[:3]} {pin[3:]}"
        return pin

    def _copy_pin(self) -> None:
        QApplication.clipboard().setText(self._pin)
        self.copy_btn.setText("✅  Copied!")
        QTimer.singleShot(2000, lambda: self.copy_btn.setText("📋  Copy PIN"))
