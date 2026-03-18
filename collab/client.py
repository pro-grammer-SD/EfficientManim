"""
collab/client.py — EfficientManim Live Collaboration WebSocket Client

Runs in a daemon background thread. All public methods are thread-safe.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Callable, Optional

import websockets
import websockets.exceptions

LOGGER = logging.getLogger("collab.client")


class CollabClient:
    """
    WebSocket client that connects to a CollabServer.
    Outgoing messages are queued from any thread; sending happens on the asyncio loop.
    """

    def __init__(
        self,
        on_message: Callable[[dict], None],
        on_connected: Optional[Callable[[], None]] = None,
        on_disconnected: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ):
        self._on_message = on_message
        self._on_connected = on_connected
        self._on_disconnected = on_disconnected
        self._on_error = on_error

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stopping = False

        # Thread-safe queue for outgoing messages.
        # We pre-create it as a regular list+lock so it's available immediately;
        # the asyncio.Queue is created inside the event loop thread.
        self._send_queue: Optional[asyncio.Queue] = None
        self._connected = threading.Event()

    # ── Public API ────────────────────────────────────────────────────

    def connect(self, uri: str) -> None:
        """Start the background connection thread. Non-blocking."""
        if self._thread and self._thread.is_alive():
            return
        self._stopping = False
        self._connected.clear()
        self._thread = threading.Thread(
            target=self._thread_main, args=(uri,), name="CollabClient", daemon=True
        )
        self._thread.start()

    def send(self, payload: dict) -> None:
        """Queue a message to be sent. Thread-safe. No-op if not connected."""
        if self._send_queue is None or self._loop is None or self._loop.is_closed():
            return
        try:
            asyncio.run_coroutine_threadsafe(
                self._send_queue.put(payload), self._loop
            )
        except Exception:
            pass

    def disconnect(self) -> None:
        """Request graceful disconnect. Thread-safe."""
        self._stopping = True
        if self._send_queue and self._loop and not self._loop.is_closed():
            # Enqueue sentinel to unblock _send_loop
            try:
                asyncio.run_coroutine_threadsafe(
                    self._send_queue.put(None), self._loop
                )
            except Exception:
                pass

    # ── Background thread ─────────────────────────────────────────────

    def _thread_main(self, uri: str) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        # Create the asyncio queue inside the loop thread
        self._send_queue = asyncio.Queue()
        try:
            loop.run_until_complete(self._client_main(uri))
        except Exception as exc:
            LOGGER.error(f"CollabClient fatal: {exc}")
            if self._on_error:
                try:
                    self._on_error(str(exc))
                except Exception:
                    pass
        finally:
            try:
                loop.close()
            except Exception:
                pass

    async def _client_main(self, uri: str) -> None:
        try:
            async with websockets.connect(
                uri,
                ping_interval=20,
                ping_timeout=10,
                open_timeout=10,
            ) as ws:
                LOGGER.info(f"Connected to {uri}")
                self._connected.set()
                if self._on_connected:
                    try:
                        self._on_connected()
                    except Exception as exc:
                        LOGGER.warning(f"on_connected error: {exc}")

                recv_task = asyncio.create_task(self._recv_loop(ws))
                send_task = asyncio.create_task(self._send_loop(ws))

                # Wait until either task finishes (disconnect / error)
                done, pending = await asyncio.wait(
                    [recv_task, send_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

        except websockets.exceptions.ConnectionClosed as exc:
            LOGGER.info(f"Connection closed: {exc}")
        except (OSError, websockets.exceptions.WebSocketException) as exc:
            LOGGER.error(f"Could not connect to {uri}: {exc}")
            if self._on_error:
                try:
                    self._on_error(f"Connection failed: {exc}")
                except Exception:
                    pass
        except Exception as exc:
            LOGGER.error(f"Client error: {exc}")
            if self._on_error:
                try:
                    self._on_error(str(exc))
                except Exception:
                    pass
        finally:
            reason = "stopped" if self._stopping else "disconnected"
            if self._on_disconnected:
                try:
                    self._on_disconnected(reason)
                except Exception:
                    pass

    async def _recv_loop(self, ws) -> None:
        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    LOGGER.debug("Invalid JSON received — skipped")
                    continue
                try:
                    self._on_message(msg)
                except Exception as exc:
                    LOGGER.warning(f"on_message error: {exc}")
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as exc:
            LOGGER.debug(f"recv_loop error: {exc}")

    async def _send_loop(self, ws) -> None:
        if self._send_queue is None:
            return
        try:
            while not self._stopping:
                try:
                    item = await asyncio.wait_for(self._send_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                if item is None:
                    # Sentinel — disconnect requested
                    break
                try:
                    await ws.send(json.dumps(item))
                except websockets.exceptions.ConnectionClosed:
                    break
                except Exception as exc:
                    LOGGER.warning(f"Send error: {exc}")
                    break
        except Exception as exc:
            LOGGER.debug(f"send_loop error: {exc}")
