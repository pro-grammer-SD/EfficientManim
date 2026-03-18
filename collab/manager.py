"""
collab/manager.py — High-level collaboration manager.

FIXES applied:
  BUG 1: Server binds to 0.0.0.0 (all interfaces). Registry stores "127.0.0.1"
          so same-machine connections always work.
  BUG 2: register_session() is called AFTER wait_ready() confirms the server
          is actually listening. No more dead PINs pointing at dead ports.
  BUG 3: wait_ready() now checks _start_failed — a false-positive "ready" when
          the server bind failed is caught and treated as failure.
  BUG 5: Server uses port=0, OS assigns a free port. actual_port is read back
          after bind. No TOCTOU race between _free_port() and serve().
"""

from __future__ import annotations

import json
import logging
import random
import socket
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QTimer, Signal

from .client import CollabClient
from .conflict import NodeLockManager
from .delta import apply_delta, make_delta, serialize_graph
from .pin_registry import (
    cleanup_expired,
    debug_dump,
    register_session,
    remove_session,
    resolve_pin,
)
from .server import CollabServer

LOGGER = logging.getLogger("collab.manager")

_ANIMALS = [
    "Otter", "Fox", "Koala", "Lynx", "Panda", "Heron",
    "Wren", "Badger", "Marten", "Ibis", "Gecko", "Tern",
    "Viper", "Crane", "Moose", "Raven", "Stoat", "Bison",
]
_COLORS = [
    "#fca5a5", "#fcd34d", "#86efac", "#67e8f9",
    "#c4b5fd", "#f9a8d4", "#fdba74", "#a5b4fc",
]


def _local_ip() -> str:
    """Best-effort LAN IP — used for display only, NOT for registry storage."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def _generate_pin() -> str:
    return str(random.randint(100000, 999999))


class CollaborationManager(QObject):
    """
    Lives on the Qt main thread.
    All background-thread callbacks are marshalled via QTimer.singleShot(0, …).
    """

    session_started = Signal(str)       # pin
    session_ended = Signal(str)         # reason
    participants_changed = Signal(dict)
    status_message = Signal(str)

    def __init__(self, window):
        super().__init__()
        self.window = window

        self.client_id: str = str(uuid.uuid4())
        self.display_name: str = random.choice(_ANIMALS)
        self.display_color: str = random.choice(_COLORS)

        self.is_hosting: bool = False
        self.is_connected: bool = False
        self.pin: str = ""

        # host_ip = LAN IP, for display in StartCollabDialog only
        # host_port = actual OS-assigned port after bind
        self.host_ip: str = ""
        self.host_port: int = 0

        self.server: Optional[CollabServer] = None
        self.client: Optional[CollabClient] = None
        self.participants: dict = {}

        self.lock_manager = NodeLockManager(ttl_seconds=10.0, parent=self)
        self.lock_manager.lock_changed.connect(self._on_lock_changed)

        # Thread-safe graph cache — serialize_graph() runs on main thread only
        self._graph_cache: dict = {"nodes": [], "wires": []}
        self._graph_lock = threading.Lock()

        self._snapshot_timer = QTimer(self)
        self._snapshot_timer.setInterval(5 * 60 * 1000)
        self._snapshot_timer.timeout.connect(self._save_snapshot)

        self._inactivity_timeout_min: int = 30

        # Clean up stale sessions from previous crashes (older than 3 hours)
        cleanup_expired(ttl_seconds=3 * 3600)

    # ── Session lifecycle ─────────────────────────────────────────────

    def start_host(self) -> Optional[str]:
        """
        Start a new session as host.
        Returns PIN on success, None on failure.

        FIXED ORDERING (BUG 2 + BUG 5):
          1. Generate PIN and LAN IP (display only)
          2. Serialize graph cache
          3. Create and start server with port=0 (OS picks port)
          4. wait_ready() — block until server is actually listening
          5. Read actual_port from server
          6. register_session(pin, "127.0.0.1", actual_port)  ← only here!
        """
        if self.is_connected:
            LOGGER.warning("start_host() called but already connected")
            return None

        pin = _generate_pin()
        # Avoid collisions with existing sessions (rare but possible)
        for _ in range(10):
            if resolve_pin(pin) is None:
                break
            pin = _generate_pin()

        # LAN IP for display only — server binds to 0.0.0.0 (BUG 1 FIX)
        self.host_ip = _local_ip()

        # Serialize graph on main thread BEFORE starting server thread
        self.update_graph_cache()

        # BUG 5 FIX: pass port=0, server asks OS for a free port, reads it back
        self.server = CollabServer(
            port=0,                            # OS picks; no TOCTOU race
            session_pin=pin,
            get_full_graph=self._get_graph_for_server,
            on_delta=self._on_server_received_delta,
            on_client_join=self._on_client_join_server,
            on_client_leave=self._on_client_leave_server,
            on_error=self._on_server_error,
            inactivity_timeout=self._inactivity_timeout_min * 60,
        )
        self.server.start()
        LOGGER.info(f"Server thread started for PIN {pin}; waiting for bind…")

        # BUG 3 FIX: wait_ready() checks _start_failed, returns False on error
        ready = self.server.wait_ready(timeout=5.0)
        if not ready:
            LOGGER.error(
                f"Server failed to start for PIN {pin}. "
                f"start_failed={self.server._start_failed}"
            )
            self.server.stop()
            self.server = None
            return None

        # BUG 5 FIX: read actual port after successful bind
        self.host_port = self.server.actual_port
        LOGGER.info(
            f"Server confirmed listening on 0.0.0.0:{self.host_port} (LAN: {self.host_ip})"
        )

        # BUG 2 FIX: register ONLY after server confirmed ready
        # Store "127.0.0.1" so same-machine clients always connect successfully.
        # The LAN IP is shown in StartCollabDialog for manual LAN sharing.
        register_session(pin, "127.0.0.1", self.host_port)
        LOGGER.info(f"Registry updated: PIN={pin} → 127.0.0.1:{self.host_port}")
        LOGGER.debug(debug_dump())

        self.is_hosting = True
        self.is_connected = True
        self.pin = pin
        self.display_name = "Host"

        self.participants = {self.client_id: self._my_participant_info()}
        self.participants_changed.emit(dict(self.participants))
        self.session_started.emit(pin)
        self._snapshot_timer.start()

        return pin

    def join_session(self, pin: str) -> bool:
        """
        Join a session by PIN. Returns True if PIN resolved (connection attempt
        started); connection result arrives via session_started / status_message.
        """
        if self.is_connected:
            LOGGER.warning("join_session() called but already connected")
            return False

        LOGGER.info(f"Attempting to join session PIN={pin}")
        LOGGER.debug(debug_dump())

        info = resolve_pin(pin)
        if info is None:
            LOGGER.error(
                f"join_session: PIN '{pin}' not in registry. {debug_dump()}"
            )
            return False

        if info.port == 0:
            LOGGER.error(f"Registry entry for PIN '{pin}' has port=0 — invalid")
            return False

        uri = f"ws://{info.host}:{info.port}"
        LOGGER.info(f"Connecting to {uri}")

        self.pin = pin
        self.host_ip = info.host
        self.host_port = info.port
        self.display_name = random.choice(_ANIMALS)

        self.client = CollabClient(
            on_message=self._on_client_received_message,
            on_connected=self._on_client_connected_cb,
            on_disconnected=self._on_client_disconnected_cb,
            on_error=self._on_client_error_cb,
        )
        self.client.connect(uri)
        return True

    def end_session(self, reason: str = "ended") -> None:
        """End or leave the current session."""
        if self.server:
            try:
                self.server.broadcast_system(
                    {"type": "session_ended", "reason": reason}
                )
            except Exception:
                pass
            self.server.stop()
            self.server = None

        if self.client:
            self.client.disconnect()
            self.client = None

        if self.is_hosting and self.pin:
            remove_session(self.pin)

        self.is_connected = False
        self.is_hosting = False
        self.pin = ""
        self.participants = {}
        self.participants_changed.emit({})
        self.session_ended.emit(reason)
        self._snapshot_timer.stop()

    # ── Graph cache ────────────────────────────────────────────────────

    def update_graph_cache(self) -> None:
        """Serialize current graph on main thread. Thread-safe for server reads."""
        try:
            graph = serialize_graph(self.window)
            with self._graph_lock:
                self._graph_cache = graph
        except Exception as exc:
            LOGGER.warning(f"update_graph_cache error: {exc}")

    def _get_graph_for_server(self) -> dict:
        """Return a snapshot of the cached graph. Safe from any thread."""
        with self._graph_lock:
            return dict(self._graph_cache)

    def broadcast_full_graph(self) -> None:
        """Force a full_graph_sync to all participants (e.g. after AI merge)."""
        if not self.is_connected:
            return
        self.update_graph_cache()
        graph = self._get_graph_for_server()
        delta = make_delta(
            "full_graph_sync",
            {"graph_json": graph},
            self.pin,
            self.client_id,
        )
        if self.server:
            self.server.broadcast_delta(delta, exclude_sender=None)
        elif self.client:
            self.client.send(delta)

    # ── Delta ─────────────────────────────────────────────────────────

    def send_delta(self, action: str, payload: dict) -> None:
        """Send a delta to all participants. Must call from Qt main thread."""
        if not self.is_connected:
            return
        delta = make_delta(action, payload, self.pin, self.client_id)
        if self.server:
            self.server.broadcast_delta(delta, exclude_sender=None)
        if self.client:
            self.client.send(delta)
        # Refresh graph cache after structural changes
        if action in (
            "node_added", "node_deleted", "wire_added", "wire_deleted",
            "vgroup_created", "vgroup_deleted",
        ):
            QTimer.singleShot(50, self.update_graph_cache)

    def _apply_remote_delta(self, delta: dict) -> None:
        """Schedule remote delta application on the Qt main thread."""
        sender = delta.get("sender_id")
        if sender and sender == self.client_id:
            return  # ignore own echo

        def _do():
            try:
                self.window._collab_applying = True
                apply_delta(self.window, delta)
            except Exception as exc:
                LOGGER.warning(f"apply_delta error: {exc}")
            finally:
                self.window._collab_applying = False
                try:
                    self.window.compile_graph()
                except Exception:
                    pass
                self.update_graph_cache()

        QTimer.singleShot(0, _do)

    # ── Locking ───────────────────────────────────────────────────────

    def lock_node(self, node_id: str) -> None:
        if not self.is_connected:
            return
        expires_at = time.monotonic() + self.lock_manager.ttl_seconds
        self.lock_manager.lock(
            node_id, self.client_id, self.display_name, self.display_color, expires_at
        )
        self.send_delta("node_lock", {
            "node_id": node_id,
            "owner_id": self.client_id,
            "owner_name": self.display_name,
            "owner_color": self.display_color,
            "expires_at": expires_at,
        })

    def unlock_node(self, node_id: str) -> None:
        if not self.is_connected:
            return
        self.lock_manager.unlock(node_id)
        self.send_delta("node_unlock", {"node_id": node_id})

    def _on_lock_changed(self, node_id: str) -> None:
        node = self.window.nodes.get(node_id)
        if node:
            node.update()

    # ── Server-side callbacks (called from server background thread) ──

    def _on_server_received_delta(self, delta: dict, sender_id: str) -> None:
        self._apply_remote_delta(delta)

    def _on_client_join_server(self, client_id: str) -> dict:
        name = random.choice(_ANIMALS)
        color = random.choice(_COLORS)
        info = {"id": client_id, "name": name, "color": color, "active_node": None}

        def _add():
            self.participants[client_id] = info
            self.participants_changed.emit(dict(self.participants))
            self.status_message.emit(f"🟢 {name} joined the session")

        QTimer.singleShot(0, _add)
        return info

    def _on_client_leave_server(self, client_id: str) -> None:
        def _remove():
            info = self.participants.pop(client_id, {})
            name = info.get("name", client_id[:6])
            self.participants_changed.emit(dict(self.participants))
            self.status_message.emit(f"⚪ {name} left the session")

        QTimer.singleShot(0, _remove)

    def _on_server_error(self, message: str) -> None:
        def _emit():
            self.status_message.emit(f"Server error: {message}")

        QTimer.singleShot(0, _emit)

    # ── Client-side callbacks (called from client background thread) ──

    def _on_client_connected_cb(self) -> None:
        def _on_main():
            self.is_connected = True
            self.participants = {self.client_id: self._my_participant_info()}
            self.participants_changed.emit(dict(self.participants))
            self.session_started.emit(self.pin)
            self.status_message.emit(f"✅ Connected to session PIN {self.pin}")

        QTimer.singleShot(0, _on_main)

    def _on_client_disconnected_cb(self, reason: str) -> None:
        def _on_main():
            was_connected = self.is_connected
            self.is_connected = False
            self.pin = ""
            self.participants = {}
            self.participants_changed.emit({})
            if was_connected:
                self.session_ended.emit(reason)

        QTimer.singleShot(0, _on_main)

    def _on_client_received_message(self, message: dict) -> None:
        msg_type = message.get("type")

        if msg_type == "delta":
            self._apply_remote_delta(message)

        elif msg_type == "participant_joined":
            info = message.get("client") or {}
            if info.get("id"):
                def _add(i=info):
                    self.participants[i["id"]] = i
                    self.participants_changed.emit(dict(self.participants))
                    self.status_message.emit(f"🟢 {i.get('name', '?')} joined")
                QTimer.singleShot(0, _add)

        elif msg_type == "participant_left":
            cid = message.get("client_id")
            if cid:
                def _remove(c=cid):
                    info = self.participants.pop(c, {})
                    self.participants_changed.emit(dict(self.participants))
                    self.status_message.emit(f"⚪ {info.get('name', c[:6])} left")
                QTimer.singleShot(0, _remove)

        elif msg_type == "session_ended":
            reason = message.get("reason", "ended")
            def _end(r=reason):
                self.end_session(r)
                self.status_message.emit(f"Session ended: {r}")
            QTimer.singleShot(0, _end)

    def _on_client_error_cb(self, message: str) -> None:
        LOGGER.error(f"Client connection error: {message}")
        def _emit():
            self.status_message.emit(f"❌ Connection error: {message}")
        QTimer.singleShot(0, _emit)

    # ── Helpers ───────────────────────────────────────────────────────

    def _my_participant_info(self) -> dict:
        return {
            "id": self.client_id,
            "name": self.display_name,
            "color": self.display_color,
            "active_node": None,
        }

    def update_active_node(self, node_id: Optional[str]) -> None:
        if not self.is_connected:
            return
        if self.client_id in self.participants:
            self.participants[self.client_id]["active_node"] = node_id
        self.send_delta("cursor_update", {
            "client_id": self.client_id,
            "active_node": node_id,
        })

    def _save_snapshot(self) -> None:
        if not self.is_hosting or not self.pin:
            return
        try:
            base = Path.home() / ".efficientmanim" / "collab_snapshots"
            base.mkdir(parents=True, exist_ok=True)
            stamp = int(time.time())
            path = base / f"{self.pin}_{stamp}.json"
            graph = serialize_graph(self.window)
            path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
            LOGGER.debug(f"Snapshot saved: {path.name}")
        except Exception as exc:
            LOGGER.warning(f"Snapshot failed: {exc}")
