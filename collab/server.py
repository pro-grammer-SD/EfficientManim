"""
collab/server.py — EfficientManim Live Collaboration WebSocket Server

FIXES applied:
  BUG 1: Always binds to "0.0.0.0" so both 127.0.0.1 and LAN IP work.
  BUG 3: _ready_event is ONLY set when the server is actually listening.
          A separate _start_failed flag is set on bind errors.
  BUG 5: port=0 is passed to websockets.serve; the actual OS-assigned port
          is read back and stored in self.actual_port.

Runs in a daemon background thread with its own asyncio event loop.
Never blocks the Qt main thread.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from typing import Callable, Dict, Optional

import websockets
import websockets.exceptions

LOGGER = logging.getLogger("collab.server")


class CollabServer:
    """
    WebSocket server for one collaboration session.
    All public methods are thread-safe.
    """

    def __init__(
        self,
        port: int,                             # desired port; 0 = let OS pick
        session_pin: str,
        get_full_graph: Callable[[], dict],
        on_delta: Callable[[dict, str], None],
        on_client_join: Optional[Callable[[str], dict]] = None,
        on_client_leave: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        inactivity_timeout: float = 1800.0,
    ):
        # BUG 1 FIX: always bind to 0.0.0.0 — accepts both 127.0.0.1 and LAN IP
        self._bind_host = "0.0.0.0"
        self._desired_port = port
        self.actual_port: int = 0          # set after bind succeeds

        self.session_pin = session_pin

        self._get_full_graph = get_full_graph
        self._on_delta = on_delta
        self._on_client_join = on_client_join
        self._on_client_leave = on_client_leave
        self._on_error = on_error
        self._inactivity_timeout = inactivity_timeout

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event: Optional[asyncio.Event] = None

        # BUG 3 FIX: _ready_event is set ONLY on success
        self._ready_event = threading.Event()
        self._start_failed = False          # True if bind/startup failed

        self._stopping = False
        self._clients: Dict[str, object] = {}
        self._clients_lock = threading.Lock()
        self._last_activity = time.monotonic()

    # ── Public API ────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the server in a daemon background thread. Non-blocking."""
        if self._thread and self._thread.is_alive():
            return
        self._stopping = False
        self._start_failed = False
        self._ready_event.clear()
        self._thread = threading.Thread(
            target=self._thread_main, name="CollabServer", daemon=True
        )
        self._thread.start()

    def wait_ready(self, timeout: float = 5.0) -> bool:
        """
        Block until the server is listening (or timeout / failure).
        Returns True ONLY if the server is actually accepting connections.

        BUG 3 FIX: returns False if startup failed, even if event was set.
        """
        signalled = self._ready_event.wait(timeout)
        if not signalled:
            LOGGER.error(f"Server did not become ready within {timeout}s")
            return False
        if self._start_failed:
            LOGGER.error("Server signalled ready but startup had failed")
            return False
        return True

    @property
    def client_count(self) -> int:
        with self._clients_lock:
            return len(self._clients)

    def broadcast_delta(self, delta: dict, exclude_sender: Optional[str] = None) -> None:
        self._schedule(self._broadcast(delta, exclude=exclude_sender))

    def broadcast_system(self, payload: dict) -> None:
        self._schedule(self._broadcast(payload, exclude=None))

    def stop(self) -> None:
        """Gracefully stop the server. Thread-safe."""
        if self._stopping:
            return
        self._stopping = True
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._trigger_stop)

    # ── Background thread ─────────────────────────────────────────────

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._server_main())
        except Exception as exc:
            LOGGER.error(f"CollabServer fatal: {exc}")
            self._start_failed = True
            self._ready_event.set()   # unblock wait_ready so caller isn't stuck
            if self._on_error:
                try:
                    self._on_error(str(exc))
                except Exception:
                    pass
        finally:
            try:
                pending = asyncio.all_tasks(loop)
                for t in pending:
                    t.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            try:
                loop.close()
            except Exception:
                pass

    async def _server_main(self) -> None:
        self._stop_event = asyncio.Event()
        try:
            # BUG 1 FIX: always bind to 0.0.0.0
            # BUG 5 FIX: pass desired port; OS picks if 0; read actual port back
            async with websockets.serve(
                self._connection_handler,
                self._bind_host,
                self._desired_port,
                ping_interval=20,
                ping_timeout=10,
            ) as server:
                # Read the actual bound port (handles port=0 case)
                sockets = getattr(server, "sockets", None)
                if sockets:
                    self.actual_port = sockets[0].getsockname()[1]
                else:
                    self.actual_port = self._desired_port

                LOGGER.info(
                    f"CollabServer listening on {self._bind_host}:{self.actual_port} "
                    f"(PIN {self.session_pin})"
                )

                # BUG 3 FIX: only set ready AFTER we confirm we are listening
                self._ready_event.set()

                watchdog = asyncio.create_task(self._inactivity_watchdog())
                await self._stop_event.wait()
                watchdog.cancel()

                # Gracefully close all client connections
                with self._clients_lock:
                    clients_snapshot = list(self._clients.values())
                for ws in clients_snapshot:
                    try:
                        await ws.close()
                    except Exception:
                        pass

        except OSError as exc:
            # BUG 3 FIX: set _start_failed before setting _ready_event
            LOGGER.error(
                f"CollabServer bind error {self._bind_host}:{self._desired_port}: {exc}"
            )
            self._start_failed = True
            self._ready_event.set()   # unblock wait_ready; caller checks _start_failed
            if self._on_error:
                self._on_error(f"Could not bind to port {self._desired_port}: {exc}")

        except Exception as exc:
            LOGGER.error(f"CollabServer._server_main error: {exc}")
            self._start_failed = True
            self._ready_event.set()
            if self._on_error:
                self._on_error(str(exc))

    def _trigger_stop(self) -> None:
        if self._stop_event and not self._stop_event.is_set():
            self._stop_event.set()

    # ── Per-connection handler ────────────────────────────────────────

    async def _connection_handler(self, websocket, *args) -> None:
        client_id = str(uuid.uuid4())
        with self._clients_lock:
            self._clients[client_id] = websocket
        self._last_activity = time.monotonic()
        LOGGER.info(f"Client connected: {client_id[:8]} ({self.client_count} total)")

        participant_info: Optional[dict] = None
        if self._on_client_join:
            try:
                participant_info = self._on_client_join(client_id)
            except Exception as exc:
                LOGGER.warning(f"on_client_join error: {exc}")

        # Send full graph state to the new joiner
        try:
            graph = self._get_full_graph()
            await websocket.send(json.dumps({
                "type": "delta",
                "session_pin": self.session_pin,
                "sender_id": "server",
                "timestamp": time.time(),
                "action": "full_graph_sync",
                "payload": {"graph_json": graph},
            }))
            LOGGER.debug(f"Sent full_graph_sync to {client_id[:8]}")
        except Exception as exc:
            LOGGER.warning(f"full_graph_sync send failed for {client_id[:8]}: {exc}")

        # Notify all other clients that someone joined
        if participant_info:
            await self._broadcast(
                {"type": "participant_joined", "client": participant_info},
                exclude=client_id,
            )

        # Main receive loop
        try:
            async for raw in websocket:
                self._last_activity = time.monotonic()
                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    LOGGER.debug(f"Malformed JSON from {client_id[:8]}")
                    continue

                if msg.get("type") == "delta":
                    try:
                        self._on_delta(msg, client_id)
                    except Exception as exc:
                        LOGGER.warning(f"on_delta error: {exc}")
                    await self._broadcast(msg, exclude=client_id)
                else:
                    await self._broadcast(msg, exclude=client_id)

        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as exc:
            LOGGER.debug(f"Connection handler error {client_id[:8]}: {exc}")
        finally:
            with self._clients_lock:
                self._clients.pop(client_id, None)
            if self._on_client_leave:
                try:
                    self._on_client_leave(client_id)
                except Exception:
                    pass
            await self._broadcast(
                {"type": "participant_left", "client_id": client_id}, exclude=None
            )
            LOGGER.info(
                f"Client disconnected: {client_id[:8]} ({self.client_count} remaining)"
            )

    # ── Helpers ───────────────────────────────────────────────────────

    async def _broadcast(self, payload: dict, exclude: Optional[str]) -> None:
        with self._clients_lock:
            targets = {
                cid: ws for cid, ws in self._clients.items()
                if exclude is None or cid != exclude
            }
        if not targets:
            return
        raw = json.dumps(payload)
        dead = []
        for cid, ws in targets.items():
            try:
                await ws.send(raw)
            except Exception:
                dead.append(cid)
        if dead:
            with self._clients_lock:
                for cid in dead:
                    self._clients.pop(cid, None)

    async def _inactivity_watchdog(self) -> None:
        if self._inactivity_timeout <= 0:
            return
        try:
            while not self._stopping:
                await asyncio.sleep(10)
                idle = time.monotonic() - self._last_activity
                if idle > self._inactivity_timeout:
                    LOGGER.info(
                        f"Session {self.session_pin} inactive for {idle:.0f}s — terminating"
                    )
                    await self._broadcast(
                        {"type": "session_ended", "reason": "inactivity"}, exclude=None
                    )
                    self.stop()
                    return
        except asyncio.CancelledError:
            pass

    def _schedule(self, coro) -> None:
        if self._loop and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(coro, self._loop)
