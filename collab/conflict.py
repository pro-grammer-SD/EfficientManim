"""
collab/conflict.py — Node locking and conflict resolution.

NodeLockManager tracks which nodes are locked by which collaborator.
Locks have a TTL and expire automatically to prevent deadlocks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from PySide6.QtCore import QObject, QTimer, Signal


@dataclass
class LockInfo:
    node_id: str
    owner_id: str
    owner_name: str
    owner_color: str
    expires_at: float  # monotonic time


class NodeLockManager(QObject):
    """
    Thread-compatible (read from any thread, write only from Qt main thread via
    apply_lock / apply_unlock called through QTimer.singleShot).

    lock_changed(node_id) is emitted whenever a lock is acquired or released.
    """

    lock_changed = Signal(str)

    def __init__(self, ttl_seconds: float = 10.0, parent=None):
        super().__init__(parent)
        self.ttl_seconds = ttl_seconds
        self._locks: Dict[str, LockInfo] = {}

        # Cleanup timer — runs on Qt main thread every second
        self._cleanup_timer = QTimer(self)
        self._cleanup_timer.setInterval(1000)
        self._cleanup_timer.timeout.connect(self._expire_stale_locks)
        self._cleanup_timer.start()

    # ── Public API ────────────────────────────────────────────────────

    def lock(
        self,
        node_id: str,
        owner_id: str,
        owner_name: str,
        owner_color: str,
        expires_at: Optional[float] = None,
    ) -> None:
        """Acquire or refresh a lock. Call from Qt main thread."""
        exp = expires_at if expires_at is not None else time.monotonic() + self.ttl_seconds
        self._locks[node_id] = LockInfo(
            node_id=node_id,
            owner_id=owner_id or "",
            owner_name=owner_name or "Unknown",
            owner_color=owner_color or "#9ca3af",
            expires_at=exp,
        )
        self.lock_changed.emit(node_id)

    def unlock(self, node_id: str) -> None:
        """Release a lock. Call from Qt main thread."""
        if node_id in self._locks:
            del self._locks[node_id]
            self.lock_changed.emit(node_id)

    def get_lock(self, node_id: str) -> Optional[LockInfo]:
        """
        Return LockInfo if node is currently locked by anyone, else None.
        Expired locks are removed and None is returned.
        Safe to call from any thread (read-only).
        """
        info = self._locks.get(node_id)
        if info is None:
            return None
        if time.monotonic() > info.expires_at:
            # Will be cleaned up by the timer; pretend it's gone
            return None
        return info

    def is_locked_by(self, node_id: str, owner_id: str) -> bool:
        """Return True if node is locked specifically by owner_id."""
        info = self.get_lock(node_id)
        return info is not None and info.owner_id == owner_id

    def locked_nodes(self) -> List[str]:
        """Return list of all currently locked node IDs."""
        now = time.monotonic()
        return [nid for nid, info in self._locks.items() if info.expires_at > now]

    def as_dict(self) -> dict:
        """Snapshot of all active locks as plain dicts (for debugging)."""
        return {
            nid: {
                "owner_id": info.owner_id,
                "owner_name": info.owner_name,
                "owner_color": info.owner_color,
                "expires_in": max(0.0, info.expires_at - time.monotonic()),
            }
            for nid, info in self._locks.items()
            if time.monotonic() < info.expires_at
        }

    # ── Internal ──────────────────────────────────────────────────────

    def _expire_stale_locks(self) -> None:
        now = time.monotonic()
        expired = [nid for nid, info in self._locks.items() if info.expires_at <= now]
        for nid in expired:
            del self._locks[nid]
            self.lock_changed.emit(nid)
