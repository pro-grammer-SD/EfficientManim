"""
autosave_manager.py — Production-grade autosave system

Implements debounced 3-second autosave with hash-based change detection.

Key Features:
- Hash-based change detection (code, graph, assets, keybindings)
- 3-second debounced timer (no disk thrashing)
- Automatic save triggers for specific events
- Silent operation (no dialogs)
- Respects user's last saved version as fallback
"""

import logging
from typing import Optional, Callable

try:
    from PySide6.QtCore import QTimer, QObject, Signal

    HAS_PYSIDE = True
except ImportError:
    HAS_PYSIDE = False
    QObject = object
    Signal = None


LOGGER = logging.getLogger("autosave")

# Autosave interval in milliseconds (3 seconds)
AUTOSAVE_INTERVAL_MS = 3000


class AutosaveManager(QObject if HAS_PYSIDE else object):
    """
    Manages automatic saving of projects with intelligent change detection.

    Never saves unnecessarily — only when state actually changes.
    Uses debouncing to prevent rapid successive saves.
    """

    if HAS_PYSIDE:
        autosave_triggered = Signal(str)  # (reason)

    def __init__(self):
        if HAS_PYSIDE:
            super().__init__()

        self._save_callback: Optional[Callable[[], None]] = None
        self._timer: Optional[QTimer] = None

        # Hash tracking
        self._last_code_hash = ""
        self._last_graph_hash = ""
        self._last_assets_hash = ""
        self._last_keybindings_hash = ""

        # Compute functions (will be set by EfficientManimWindow)
        self._compute_code_hash: Optional[Callable[[], str]] = None
        self._compute_graph_hash: Optional[Callable[[], str]] = None
        self._compute_assets_hash: Optional[Callable[[], str]] = None
        self._compute_keybindings_hash: Optional[Callable[[], str]] = None

        self._enabled = False

        if HAS_PYSIDE:
            self._setup_timer()

    def _setup_timer(self) -> None:
        """Initialize the autosave timer."""
        self._timer = QTimer()
        self._timer.setSingleShot(True)  # Debounce: only fire once per interval
        self._timer.setInterval(AUTOSAVE_INTERVAL_MS)
        self._timer.timeout.connect(self._on_timer_timeout)
        LOGGER.debug(f"Autosave timer configured: {AUTOSAVE_INTERVAL_MS}ms interval")

    def set_save_callback(self, callback: Callable[[], None]) -> None:
        """Register the save function to call on autosave."""
        self._save_callback = callback
        LOGGER.debug("Autosave callback registered")

    def set_hash_computers(
        self,
        code_fn: Callable[[], str],
        graph_fn: Callable[[], str],
        assets_fn: Callable[[], str],
        keybindings_fn: Callable[[], str],
    ) -> None:
        """Register hash computation functions."""
        self._compute_code_hash = code_fn
        self._compute_graph_hash = graph_fn
        self._compute_assets_hash = assets_fn
        self._compute_keybindings_hash = keybindings_fn
        LOGGER.debug("Hash computer functions registered")

    def enable(self) -> None:
        """Enable autosave."""
        self._enabled = True
        LOGGER.info("Autosave enabled")

    def disable(self) -> None:
        """Disable autosave."""
        self._enabled = False
        if HAS_PYSIDE and self._timer:
            self._timer.stop()
        LOGGER.info("Autosave disabled")

    def trigger_autosave(self, reason: str = "change detected") -> None:
        """
        Request autosave with debouncing.

        Won't save immediately — will wait for quiet period,
        then check if state actually changed before saving.
        """
        if not self._enabled:
            return

        if HAS_PYSIDE and self._timer:
            self._timer.stop()
            self._timer.start()  # Reset debounce timer
            LOGGER.debug(f"Autosave scheduled ({reason})")

    def _on_timer_timeout(self) -> None:
        """Timer fired — check if state changed, then save."""
        if not self._enabled or not self._save_callback:
            return

        try:
            # Compute current hashes
            code_hash = self._compute_code_hash() if self._compute_code_hash else ""
            graph_hash = self._compute_graph_hash() if self._compute_graph_hash else ""
            assets_hash = (
                self._compute_assets_hash() if self._compute_assets_hash else ""
            )
            keybindings_hash = (
                self._compute_keybindings_hash()
                if self._compute_keybindings_hash
                else ""
            )

            # Check if anything changed
            code_changed = code_hash != self._last_code_hash
            graph_changed = graph_hash != self._last_graph_hash
            assets_changed = assets_hash != self._last_assets_hash
            keybindings_changed = keybindings_hash != self._last_keybindings_hash

            if not (
                code_changed or graph_changed or assets_changed or keybindings_changed
            ):
                LOGGER.debug("No changes detected, skipping autosave")
                return

            # State changed — save now
            reason_parts = []
            if code_changed:
                reason_parts.append("code")
            if graph_changed:
                reason_parts.append("graph")
            if assets_changed:
                reason_parts.append("assets")
            if keybindings_changed:
                reason_parts.append("keybindings")

            reason = f"autosave ({', '.join(reason_parts)})"

            # Update hashes for next check
            self._last_code_hash = code_hash
            self._last_graph_hash = graph_hash
            self._last_assets_hash = assets_hash
            self._last_keybindings_hash = keybindings_hash

            # Perform the save
            self._save_callback()

            if HAS_PYSIDE:
                self.autosave_triggered.emit(reason)

            LOGGER.info(f"✓ Autosave completed ({', '.join(reason_parts)})")

        except Exception as e:
            LOGGER.error(f"Autosave failed: {e}")

    def force_save(self) -> None:
        """Force immediate save (bypass debounce)."""
        if HAS_PYSIDE and self._timer:
            self._timer.stop()
        self._on_timer_timeout()


# Global autosave manager instance
AUTOSAVE = AutosaveManager()
