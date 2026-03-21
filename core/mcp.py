# ruff: noqa: E402
from __future__ import annotations

"""
mcp.py — Model Context Protocol for EfficientManim

Exposes the full application context to Gemini as a structured,
programmatically accessible interface. Gemini can use this to:

  - Introspect all nodes, timeline, assets, and project state
  - Create, delete, and modify nodes and their parameters
  - Attach voiceovers to any node
  - Trigger rendering and export
  - Control the animation timeline
  - Switch tabs, trigger UI actions
  - Execute multi-step autonomous workflows
  - Log actions and receive structured error responses

Architecture:
  MCPContext       — immutable snapshot of the current application state
  MCPCommand       — a typed command to execute against the application
  MCPResult        — structured success/error response
  MCPAgent         — executes commands safely against the live app window
  MCPRegistry      — registers and routes commands to handlers

Usage (from AIPanel or other code):
    from mcp import MCPAgent
    agent = MCPAgent(main_window)
    result = agent.execute("create_node", {"cls_name": "Circle", "name": "my_circle"})
    if result.success:
        print("Created node:", result.data)

All destructive operations (delete_node, clear_scene) require explicit
`confirm=True` in the command payload to prevent accidental data loss.
"""

import json
import logging
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Optional

from graph.node import NodeData, NodeItem
from core.config import SETTINGS

LOGGER = logging.getLogger("mcp")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class MCPResult:
    """Structured result from an MCP command execution."""

    success: bool
    data: Any = None
    error: str = ""
    command: str = ""

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "command": self.command,
        }

    def __repr__(self) -> str:
        if self.success:
            return f"MCPResult(ok, data={self.data!r})"
        return f"MCPResult(FAIL, error={self.error!r})"


@dataclass
class MCPNodeInfo:
    """Serializable snapshot of a single node."""

    id: str
    name: str
    cls_name: str
    node_type: str
    params: dict
    has_voiceover: bool
    voiceover_transcript: Optional[str]
    voiceover_duration: float
    pos_x: float
    pos_y: float
    audio_asset_id: Optional[str]
    is_ai_generated: bool

    @staticmethod
    def from_node_item(node_item) -> "MCPNodeInfo":
        d = node_item.data
        return MCPNodeInfo(
            id=d.id,
            name=d.name,
            cls_name=d.cls_name,
            node_type=d.type.name,
            params=dict(d.params),
            has_voiceover=bool(d.audio_asset_id),
            voiceover_transcript=d.voiceover_transcript,
            voiceover_duration=d.voiceover_duration,
            pos_x=d.pos_x,
            pos_y=d.pos_y,
            audio_asset_id=d.audio_asset_id,
            is_ai_generated=d.is_ai_generated,
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "cls_name": self.cls_name,
            "node_type": self.node_type,
            "params": self.params,
            "has_voiceover": self.has_voiceover,
            "voiceover_transcript": self.voiceover_transcript,
            "voiceover_duration": self.voiceover_duration,
            "pos_x": self.pos_x,
            "pos_y": self.pos_y,
            "audio_asset_id": self.audio_asset_id,
            "is_ai_generated": self.is_ai_generated,
        }


@dataclass
class MCPContext:
    """Immutable snapshot of the full application state."""

    project_path: Optional[str]
    project_modified: bool
    current_scene: str
    all_scenes: list[str]
    node_count: int
    nodes: list[MCPNodeInfo]
    asset_count: int
    assets: list[dict]
    selected_node_ids: list[str]
    render_queue_length: int
    active_tab: str

    def to_dict(self) -> dict:
        return {
            "project_path": self.project_path,
            "project_modified": self.project_modified,
            "current_scene": self.current_scene,
            "all_scenes": self.all_scenes,
            "node_count": self.node_count,
            "nodes": [n.to_dict() for n in self.nodes],
            "asset_count": self.asset_count,
            "assets": self.assets,
            "selected_node_ids": self.selected_node_ids,
            "render_queue_length": self.render_queue_length,
            "active_tab": self.active_tab,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ═══════════════════════════════════════════════════════════════════════════════
# MCP AGENT
# ═══════════════════════════════════════════════════════════════════════════════


class MCPAgent:
    """
    Executes MCP commands against the live __import__('ui.main_window').main_window.EfficientManimWindow.

    All commands are dispatched through a registry. Each handler receives
    the agent's window reference and a payload dict, and returns MCPResult.

    Safety:
      - Destructive commands require confirm=True in payload.
      - All handlers are wrapped in try/except to prevent app crashes.
      - Every action is logged.
    """

    def __init__(self, main_window):
        self._win = main_window
        self._registry: dict[str, Callable[[dict], MCPResult]] = {}
        self._action_log: list[dict] = []
        self._register_all_handlers()

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_context(self) -> MCPContext:
        """Return a full snapshot of the current application state."""
        win = self._win
        try:
            nodes_snapshot = [MCPNodeInfo.from_node_item(n) for n in win.nodes.values()]
        except Exception:
            nodes_snapshot = []

        try:
            assets_list = [
                {"id": a.id, "name": a.name, "kind": a.kind, "path": a.current_path}
                for a in getattr(win, "ASSETS", _get_global_assets()).get_list()
            ]
        except Exception:
            assets_list = []

        try:
            selected_ids = [
                item.data.id
                for item in win.scene.selectedItems()
                if hasattr(item, "data")
            ]
        except Exception:
            selected_ids = []

        try:
            active_tab = win.tabs_top.tabText(win.tabs_top.currentIndex())
        except Exception:
            active_tab = ""

        try:
            all_scenes = list(win._all_scenes.keys())
            current_scene = win._current_scene_name
        except Exception:
            all_scenes = []
            current_scene = ""

        return MCPContext(
            project_path=str(win.project_path) if win.project_path else None,
            project_modified=win.project_modified,
            current_scene=current_scene,
            all_scenes=all_scenes,
            node_count=len(win.nodes),
            nodes=nodes_snapshot,
            asset_count=len(assets_list),
            assets=assets_list,
            selected_node_ids=selected_ids,
            render_queue_length=len(getattr(win, "render_queue", [])),
            active_tab=active_tab,
        )

    def execute(self, command: str, payload: Optional[dict] = None) -> MCPResult:
        """Execute a named MCP command with the given payload."""
        if payload is None:
            payload = {}

        LOGGER.info(f"MCP execute: {command!r} payload={payload}")
        self._action_log.append({"command": command, "payload": payload})

        handler = self._registry.get(command)
        if handler is None:
            return MCPResult(
                success=False,
                error=f"Unknown command: '{command}'. Available: {sorted(self._registry.keys())}",
                command=command,
            )

        def _run_handler():
            try:
                result = handler(payload)
                result.command = command
                return result
            except Exception as exc:
                tb = traceback.format_exc()
                LOGGER.error(f"MCP command '{command}' raised: {exc}\n{tb}")
                return MCPResult(
                    success=False,
                    error=f"Command '{command}' failed: {exc}",
                    command=command,
                )

        try:
            from PySide6.QtWidgets import QApplication
            from PySide6.QtCore import QThread, QEventLoop, QTimer

            app = QApplication.instance()
            if app is not None and QThread.currentThread() != app.thread():
                loop = QEventLoop()
                holder: dict = {}

                def _invoke():
                    holder["res"] = _run_handler()
                    loop.quit()

                QTimer.singleShot(0, self._win, _invoke)
                loop.exec()
                return holder.get("res") or MCPResult(
                    success=False, error="Command failed in GUI dispatch."
                )
            return _run_handler()
        except Exception:
            return _run_handler()

    def execute_batch(self, commands: list[dict]) -> list[MCPResult]:
        """Execute a list of {command, payload} dicts sequentially."""
        results = []
        history = getattr(self._win, "history_manager", None)
        if history is not None:
            history.begin_group("MCP Batch")
        try:
            for item in commands:
                cmd = item.get("command", "")
                payload = item.get("payload", {})
                results.append(self.execute(cmd, payload))
        finally:
            if history is not None:
                history.end_group()
        return results

    def get_action_log(self) -> list[dict]:
        """Return the history of all executed commands."""
        return list(self._action_log)

    def list_commands(self) -> list[str]:
        """Return all registered command names."""
        return sorted(self._registry.keys())

    # ── Handler registration ───────────────────────────────────────────────────

    def _register(self, name: str):
        """Decorator to register a method as an MCP command handler."""

        def decorator(fn: Callable[[dict], MCPResult]):
            self._registry[name] = fn
            return fn

        return decorator

    def _register_all_handlers(self):
        """Register all built-in command handlers."""

        win = self._win
        from datetime import datetime

        def _now_iso() -> str:
            return datetime.now().isoformat()

        def _payload(
            status: str,
            affected_nodes: Optional[list] = None,
            metadata: Optional[dict] = None,
            extra: Optional[dict] = None,
        ) -> dict:
            history = getattr(win, "history_manager", None)
            payload = {
                "status": status,
                "affected_nodes": affected_nodes or [],
                "history_pointer": history.history_pointer() if history else 0,
                "metadata": metadata or {},
                "timestamp": _now_iso(),
            }
            if extra:
                payload.update(extra)
            return payload

        def _node_by_id_or_name(node_id: str, name: str):
            if node_id:
                return win.nodes.get(node_id)
            if name:
                for ni in win.nodes.values():
                    if ni.data.name == name:
                        return ni
            return None

        # ── Context & inspection ────────────────────────────────────────────

        @self._register("get_context")
        def _(payload: dict) -> MCPResult:
            return MCPResult(success=True, data=self.get_context().to_dict())

        @self._register("list_nodes")
        def _(payload: dict) -> MCPResult:
            nodes = [
                MCPNodeInfo.from_node_item(n).to_dict() for n in win.nodes.values()
            ]
            return MCPResult(success=True, data=nodes)

        @self._register("get_node")
        def _(payload: dict) -> MCPResult:
            node_id = payload.get("node_id", "")
            node_name = payload.get("name", "")

            node_item = win.nodes.get(node_id)
            if node_item is None and node_name:
                # Try lookup by name
                for nid, ni in win.nodes.items():
                    if ni.data.name == node_name:
                        node_item = ni
                        break

            if node_item is None:
                return MCPResult(
                    success=False,
                    error=f"Node not found: id={node_id!r} name={node_name!r}",
                )
            return MCPResult(
                success=True, data=MCPNodeInfo.from_node_item(node_item).to_dict()
            )

        # ── Node creation ───────────────────────────────────────────────────

        @self._register("create_node")
        def _(payload: dict) -> MCPResult:
            cls_name = payload.get("cls_name", "").strip()
            name = payload.get("name", "").strip() or cls_name
            params = payload.get("params", {})
            x = float(payload.get("x", 0))
            y = float(payload.get("y", 0))

            # ── Validation ────────────────────────────────────────────────
            if not cls_name:
                return MCPResult(
                    success=False,
                    error="cls_name is required and must be a non-empty Manim class name.",
                )

            node_type_str = str(payload.get("node_type", "ANIMATION")).strip().upper()
            if node_type_str in ("MOB", "MOBJECTS"):
                node_type_str = "MOBJECT"
            if node_type_str in ("ANIM", "ANIMATION"):
                node_type_str = "ANIMATION"
            if node_type_str not in ("ANIMATION", "MOBJECT", "PLAY", "WAIT", "VGROUP"):
                LOGGER.warn(
                    f"MCP create_node: unknown node_type '{node_type_str}', defaulting to ANIMATION"
                )
                node_type_str = "ANIMATION"

            # UUID duplication guard
            existing_names = {n.data.name for n in win.nodes.values()}
            if name in existing_names:
                LOGGER.warn(
                    f"MCP create_node: a node named '{name}' already exists — proceeding (names are non-unique)"
                )

            try:
                # ── Use the OFFICIAL node creation API (same as GUI) ─────
                # win.add_node(type_str, cls_name, params, pos, nid, name)
                # This ensures full lifecycle: NodeData init, NodeItem creation,
                # scene.addItem, nodes dict registration, compile_graph trigger.
                item = win.add_node(
                    node_type_str,  # "MOBJECT" or "ANIMATION" — correct string type
                    cls_name,  # Manim class name string — correct string type
                    params=dict(params),  # copy to avoid mutation
                    pos=(x, y),
                    name=name,  # custom display name (new param)
                )
                win.mark_modified()

                # Integrity check: confirm the returned item is properly registered
                actual_id = item.data.id
                if actual_id not in win.nodes:
                    return MCPResult(
                        success=False,
                        error="Node was created but not registered in win.nodes — integrity violation.",
                    )
                if not item.data.cls_name:
                    return MCPResult(
                        success=False,
                        error="Node created with empty cls_name — integrity violation.",
                    )

                LOGGER.info(
                    f"MCP created node: cls={cls_name} name='{name}' type={node_type_str} id={actual_id[:8]}"
                )
                return MCPResult(
                    success=True,
                    data=_payload(
                        "success",
                        affected_nodes=[actual_id],
                        metadata={
                            "action": "create_node",
                            "node_type": node_type_str,
                            "cls_name": item.data.cls_name,
                        },
                        extra={
                            "id": actual_id,
                            "name": item.data.name,
                            "cls_name": item.data.cls_name,
                        },
                    ),
                )
            except Exception as e:
                return MCPResult(success=False, error=f"create_node failed: {e}")

        # ── Node modification ───────────────────────────────────────────────

        @self._register("set_node_param")
        def _(payload: dict) -> MCPResult:
            node_id = payload.get("node_id", "")
            param_key = payload.get("key", "")
            param_value = payload.get("value")

            node_item = win.nodes.get(node_id)
            if node_item is None:
                return MCPResult(success=False, error=f"Node not found: {node_id!r}")
            if not param_key:
                return MCPResult(success=False, error="key is required.")

            node_item.data.params[param_key] = param_value
            node_item.update()
            win.compile_graph()
            win.mark_modified()
            try:
                win.on_node_property_changed(node_item, param_key, param_value)
            except Exception:
                if getattr(win, "history_manager", None):
                    win.history_manager.capture(
                        f"MCP Edit {param_key}", merge_key=f"mcp:{node_id}:{param_key}"
                    )
            return MCPResult(
                success=True,
                data=_payload(
                    "success",
                    affected_nodes=[node_id],
                    metadata={
                        "action": "set_node_param",
                        "node_id": node_id,
                        "key": param_key,
                    },
                    extra={"node_id": node_id, "key": param_key, "value": param_value},
                ),
            )

        @self._register("rename_node")
        def _(payload: dict) -> MCPResult:
            node_id = payload.get("node_id", "")
            new_name = payload.get("name", "")
            node_item = win.nodes.get(node_id)
            if node_item is None:
                return MCPResult(success=False, error=f"Node not found: {node_id!r}")
            if not new_name:
                return MCPResult(success=False, error="name is required.")
            node_item.data.name = new_name
            node_item.update()
            win.mark_modified()
            try:
                win.on_node_property_changed(node_item, "_name", new_name)
            except Exception:
                if getattr(win, "history_manager", None):
                    win.history_manager.capture(
                        f"MCP Rename {node_id[:6]}", merge_key=f"mcp:{node_id}:name"
                    )
            return MCPResult(
                success=True,
                data=_payload(
                    "success",
                    affected_nodes=[node_id],
                    metadata={"action": "rename_node", "node_id": node_id},
                    extra={"node_id": node_id, "new_name": new_name},
                ),
            )

        @self._register("move_node")
        def _(payload: dict) -> MCPResult:
            node_id = payload.get("node_id", "")
            node_name = payload.get("name", "")
            x = payload.get("x")
            y = payload.get("y")
            node_item = _node_by_id_or_name(node_id, node_name)
            if node_item is None:
                return MCPResult(
                    success=False,
                    error=f"Node not found: id={node_id!r} name={node_name!r}",
                )
            if x is None or y is None:
                return MCPResult(success=False, error="x and y are required.")
            try:
                node_item.setPos(float(x), float(y))
                node_item.data.pos_x = float(x)
                node_item.data.pos_y = float(y)
                node_item.update()
                win.mark_modified()
                if getattr(win, "history_manager", None):
                    win.history_manager.capture(
                        f"Move {node_item.data.name}",
                        merge_key=f"move:{node_item.data.id}",
                        metadata={"affected_nodes": [node_item.data.id]},
                    )
                return MCPResult(
                    success=True,
                    data=_payload(
                        "success",
                        affected_nodes=[node_item.data.id],
                        metadata={
                            "action": "move_node",
                            "node_id": node_item.data.id,
                            "x": float(x),
                            "y": float(y),
                        },
                    ),
                )
            except Exception as e:
                return MCPResult(success=False, error=f"move_node failed: {e}")

        @self._register("set_node_params")
        def _(payload: dict) -> MCPResult:
            node_id = payload.get("node_id", "")
            node_name = payload.get("name", "")
            params = payload.get("params", {})
            node_item = _node_by_id_or_name(node_id, node_name)
            if node_item is None:
                return MCPResult(
                    success=False,
                    error=f"Node not found: id={node_id!r} name={node_name!r}",
                )
            if not isinstance(params, dict):
                return MCPResult(success=False, error="params must be a dict.")
            node_item.data.params.update(params)
            node_item.update()
            win.compile_graph()
            win.mark_modified()
            try:
                for k, v in params.items():
                    win.on_node_property_changed(node_item, k, v)
            except Exception:
                if getattr(win, "history_manager", None):
                    win.history_manager.capture(
                        f"MCP Edit {node_item.data.name}",
                        merge_key=f"mcp:{node_item.data.id}:params",
                    )
            return MCPResult(
                success=True,
                data=_payload(
                    "success",
                    affected_nodes=[node_item.data.id],
                    metadata={
                        "action": "set_node_params",
                        "node_id": node_item.data.id,
                        "keys": list(params.keys()),
                    },
                ),
            )

        @self._register("set_node_param_enabled")
        def _(payload: dict) -> MCPResult:
            node_id = payload.get("node_id", "")
            node_name = payload.get("name", "")
            param_key = payload.get("key", "")
            enabled = bool(payload.get("enabled", True))
            node_item = _node_by_id_or_name(node_id, node_name)
            if node_item is None:
                return MCPResult(
                    success=False,
                    error=f"Node not found: id={node_id!r} name={node_name!r}",
                )
            if not param_key:
                return MCPResult(success=False, error="key is required.")
            node_item.data.set_param_enabled(param_key, enabled)
            node_item.update()
            win.compile_graph()
            win.mark_modified()
            try:
                win.on_node_property_changed(
                    node_item, param_key, node_item.data.params.get(param_key)
                )
            except Exception:
                pass
            return MCPResult(
                success=True,
                data=_payload(
                    "success",
                    affected_nodes=[node_item.data.id],
                    metadata={
                        "action": "set_node_param_enabled",
                        "node_id": node_item.data.id,
                        "key": param_key,
                        "enabled": enabled,
                    },
                ),
            )

        @self._register("set_node_param_escape")
        def _(payload: dict) -> MCPResult:
            node_id = payload.get("node_id", "")
            node_name = payload.get("name", "")
            param_key = payload.get("key", "")
            escape = bool(payload.get("escape", False))
            node_item = _node_by_id_or_name(node_id, node_name)
            if node_item is None:
                return MCPResult(
                    success=False,
                    error=f"Node not found: id={node_id!r} name={node_name!r}",
                )
            if not param_key:
                return MCPResult(success=False, error="key is required.")
            node_item.data.set_escape_string(param_key, escape)
            node_item.update()
            win.compile_graph()
            win.mark_modified()
            return MCPResult(
                success=True,
                data=_payload(
                    "success",
                    affected_nodes=[node_item.data.id],
                    metadata={
                        "action": "set_node_param_escape",
                        "node_id": node_item.data.id,
                        "key": param_key,
                        "escape": escape,
                    },
                ),
            )

        # ── Node deletion ───────────────────────────────────────────────────

        @self._register("delete_node")
        def _(payload: dict) -> MCPResult:
            node_id = payload.get("node_id", "")
            confirm = payload.get("confirm", False)
            if not confirm:
                return MCPResult(
                    success=False,
                    error="Destructive operation. Set confirm=True to proceed.",
                )
            node_item = win.nodes.get(node_id)
            if node_item is None:
                return MCPResult(success=False, error=f"Node not found: {node_id!r}")
            try:
                # ═══════════════════════════════════════════════════════════════
                # CRITICAL FIX: Use the official remove_node() method to ensure
                # complete lifecycle: edge cleanup, preview worker cleanup, UI sync.
                #
                # DO NOT directly remove from scene/dict — that leaves orphaned
                # edges and breaks graph integrity.
                # ═══════════════════════════════════════════════════════════════
                win.remove_node(node_item)
                win.compile_graph()
                win.mark_modified()
                LOGGER.info(f"MCP deleted node: {node_id[:8]} via complete lifecycle")
                return MCPResult(
                    success=True,
                    data=_payload(
                        "success",
                        affected_nodes=[node_id],
                        metadata={"action": "delete_node", "node_id": node_id},
                        extra={"deleted_id": node_id},
                    ),
                )
            except Exception as e:
                LOGGER.error(f"delete_node failed: {e}")
                return MCPResult(success=False, error=f"delete_node failed: {e}")

        @self._register("clear_scene")
        def _(payload: dict) -> MCPResult:
            confirm = payload.get("confirm", False)
            if not confirm:
                return MCPResult(
                    success=False,
                    error="Destructive operation. Set confirm=True to proceed.",
                )
            try:
                win.clear_scene(force=True)
                return MCPResult(success=True, data={"message": "Scene cleared."})
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        # ── Wire management ────────────────────────────────────────────────

        @self._register("connect_nodes")
        def _(payload: dict) -> MCPResult:
            from_id = payload.get("from_node_id", "") or payload.get("from_id", "")
            to_id = payload.get("to_node_id", "") or payload.get("to_id", "")
            if not from_id or not to_id:
                return MCPResult(
                    success=False, error="from_node_id and to_node_id are required."
                )
            try:
                wire = win.add_wire_by_ids(from_id, to_id)
                if not wire:
                    return MCPResult(success=False, error="Failed to connect nodes.")
                wire_id = getattr(wire, "wire_id", "")
                return MCPResult(
                    success=True,
                    data=_payload(
                        "success",
                        affected_nodes=[from_id, to_id],
                        metadata={
                            "action": "connect_nodes",
                            "wire_id": wire_id,
                            "from_node_id": from_id,
                            "to_node_id": to_id,
                        },
                        extra={"wire_id": wire_id},
                    ),
                )
            except Exception as e:
                return MCPResult(success=False, error=f"connect_nodes failed: {e}")

        @self._register("disconnect_nodes")
        def _(payload: dict) -> MCPResult:
            wire_id = payload.get("wire_id", "")
            from_id = payload.get("from_node_id", "") or payload.get("from_id", "")
            to_id = payload.get("to_node_id", "") or payload.get("to_id", "")
            try:
                if wire_id:
                    win.remove_wire_by_id(wire_id)
                    return MCPResult(
                        success=True,
                        data=_payload(
                            "success",
                            affected_nodes=[from_id, to_id] if from_id or to_id else [],
                            metadata={"action": "disconnect_nodes", "wire_id": wire_id},
                        ),
                    )
                if not from_id or not to_id:
                    return MCPResult(
                        success=False,
                        error="Provide wire_id or both from_node_id and to_node_id.",
                    )
                removed = False
                for item in list(win.scene.items()):
                    if not hasattr(item, "start_socket") or not hasattr(
                        item, "end_socket"
                    ):
                        continue
                    try:
                        if (
                            item.start_socket.parentItem().data.id == from_id
                            and item.end_socket.parentItem().data.id == to_id
                        ):
                            win.remove_wire(item)
                            removed = True
                            break
                    except Exception:
                        continue
                if not removed:
                    return MCPResult(success=False, error="Wire not found.")
                return MCPResult(
                    success=True,
                    data=_payload(
                        "success",
                        affected_nodes=[from_id, to_id],
                        metadata={"action": "disconnect_nodes"},
                    ),
                )
            except Exception as e:
                return MCPResult(success=False, error=f"disconnect_nodes failed: {e}")

        @self._register("list_wires")
        def _(payload: dict) -> MCPResult:
            try:
                wires = []
                for item in list(win.scene.items()):
                    if not hasattr(item, "start_socket") or not hasattr(
                        item, "end_socket"
                    ):
                        continue
                    try:
                        from_id = item.start_socket.parentItem().data.id
                        to_id = item.end_socket.parentItem().data.id
                        wire_id = getattr(item, "wire_id", "")
                        wires.append(
                            {
                                "wire_id": wire_id,
                                "from_node_id": from_id,
                                "to_node_id": to_id,
                            }
                        )
                    except Exception:
                        continue
                return MCPResult(
                    success=True,
                    data=_payload(
                        "success",
                        affected_nodes=[],
                        metadata={"action": "list_wires", "count": len(wires)},
                        extra={"wires": wires},
                    ),
                )
            except Exception as e:
                return MCPResult(success=False, error=f"list_wires failed: {e}")

        # ── Voiceover attachment ────────────────────────────────────────────

        @self._register("attach_voiceover")
        def _(payload: dict) -> MCPResult:
            node_id = payload.get("node_id", "")
            audio_path = payload.get("audio_path", "")
            transcript = payload.get("transcript", "")
            duration = float(payload.get("duration", 0.0))

            node_item = win.nodes.get(node_id)
            if node_item is None:
                return MCPResult(success=False, error=f"Node not found: {node_id!r}")
            if not audio_path:
                return MCPResult(success=False, error="audio_path is required.")

            try:
                asset = _get_global_assets().add_asset(audio_path)
                if not asset:
                    return MCPResult(
                        success=False, error="Failed to register audio asset."
                    )

                node_item.data.audio_asset_id = asset.id
                node_item.data.voiceover_transcript = transcript
                node_item.data.voiceover_duration = duration
                node_item.update()
                win.compile_graph()
                win.mark_modified()
                if getattr(win, "history_manager", None):
                    win.history_manager.capture(
                        f"Attach Voiceover {node_id[:6]}",
                        merge_key=f"voice:{node_id}",
                    )
                return MCPResult(
                    success=True,
                    data={
                        "node_id": node_id,
                        "asset_id": asset.id,
                        "duration": duration,
                    },
                )
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        @self._register("remove_voiceover")
        def _(payload: dict) -> MCPResult:
            node_id = payload.get("node_id", "")
            node_item = win.nodes.get(node_id)
            if node_item is None:
                return MCPResult(success=False, error=f"Node not found: {node_id!r}")
            node_item.data.audio_asset_id = None
            node_item.data.voiceover_transcript = None
            node_item.data.voiceover_duration = 0.0
            node_item.update()
            win.mark_modified()
            if getattr(win, "history_manager", None):
                win.history_manager.capture(
                    f"Remove Voiceover {node_id[:6]}",
                    merge_key=f"voice:{node_id}",
                )
            return MCPResult(success=True, data={"node_id": node_id})

        # ── Selection ───────────────────────────────────────────────────────

        @self._register("select_node")
        def _(payload: dict) -> MCPResult:
            node_id = payload.get("node_id", "")
            node_item = win.nodes.get(node_id)
            if node_item is None:
                return MCPResult(success=False, error=f"Node not found: {node_id!r}")
            win.scene.clearSelection()
            node_item.setSelected(True)
            win.on_selection()
            return MCPResult(success=True, data={"selected": node_id})

        @self._register("select_nodes")
        def _(payload: dict) -> MCPResult:
            node_ids = payload.get("node_ids", [])
            if not isinstance(node_ids, list) or not node_ids:
                return MCPResult(success=False, error="node_ids list is required.")
            win.scene.clearSelection()
            found = []
            for nid in node_ids:
                node_item = win.nodes.get(str(nid))
                if node_item is not None:
                    node_item.setSelected(True)
                    found.append(str(nid))
            win.on_selection()
            return MCPResult(
                success=True,
                data=_payload(
                    "success",
                    affected_nodes=found,
                    metadata={"action": "select_nodes", "count": len(found)},
                ),
            )

        @self._register("deselect_all")
        def _(payload: dict) -> MCPResult:
            win.scene.clearSelection()
            return MCPResult(success=True)

        # ── Tab navigation ──────────────────────────────────────────────────

        @self._register("switch_tab")
        def _(payload: dict) -> MCPResult:
            tab_name = payload.get("tab", "")
            tabs = win.tabs_top
            for i in range(tabs.count()):
                if tab_name.lower() in tabs.tabText(i).lower():
                    tabs.setCurrentIndex(i)
                    return MCPResult(success=True, data={"tab": tabs.tabText(i)})
            return MCPResult(
                success=False,
                error=f"Tab not found: {tab_name!r}. Available: {[tabs.tabText(i) for i in range(tabs.count())]}",
            )

        # ── Render & export ─────────────────────────────────────────────────

        @self._register("trigger_render")
        def _(payload: dict) -> MCPResult:
            node_id = payload.get("node_id")
            try:
                if node_id:
                    node_item = win.nodes.get(node_id)
                    if node_item is None:
                        return MCPResult(
                            success=False, error=f"Node not found: {node_id!r}"
                        )
                    win.queue_render(node_item)
                    return MCPResult(success=True, data={"queued_node": node_id})
                else:
                    # Render all nodes in queue
                    import tempfile
                    from pathlib import Path

                    temp_dir = Path(tempfile.gettempdir()) / "EfficientManim_Temp"
                    config = {
                        "fps": 30,
                        "resolution": (1280, 720),
                        "quality": "m",
                        "output_path": str(temp_dir),
                    }
                    win.render_to_video(config)
                    return MCPResult(
                        success=True, data={"message": "Render triggered."}
                    )
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        @self._register("render_video")
        def _(payload: dict) -> MCPResult:
            """Render full scene video with explicit config."""
            try:
                output_path = payload.get("output_path") or payload.get("output_dir")
                if not output_path:
                    return MCPResult(success=False, error="output_path is required.")
                fps = int(payload.get("fps", 30))
                quality = payload.get("quality", "m")
                resolution = payload.get("resolution") or payload.get("res")
                if resolution is None:
                    resolution = (1280, 720)
                config = {
                    "fps": fps,
                    "resolution": tuple(resolution),
                    "quality": quality,
                    "output_path": str(output_path),
                }
                win.render_to_video(config)
                return MCPResult(
                    success=True,
                    data=_payload(
                        "success",
                        affected_nodes=[],
                        metadata={
                            "action": "render_video",
                            "fps": fps,
                            "quality": quality,
                            "resolution": resolution,
                            "output_path": str(output_path),
                        },
                    ),
                )
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        @self._register("cancel_video_render")
        def _(payload: dict) -> MCPResult:
            worker = getattr(win, "video_render_worker", None)
            if worker and hasattr(worker, "stop_render"):
                try:
                    worker.stop_render()
                    return MCPResult(
                        success=True,
                        data=_payload(
                            "success",
                            affected_nodes=[],
                            metadata={"action": "cancel_video_render"},
                        ),
                    )
                except Exception as e:
                    return MCPResult(success=False, error=str(e))
            return MCPResult(success=False, error="No active video render.")

        @self._register("cancel_preview_render")
        def _(payload: dict) -> MCPResult:
            try:
                # Cancel all active preview workers
                cancelled = 0
                for w in list(getattr(win, "_active_render_workers", set())):
                    if hasattr(w, "cancel"):
                        w.cancel()
                        cancelled += 1
                return MCPResult(
                    success=True,
                    data=_payload(
                        "success",
                        affected_nodes=[],
                        metadata={
                            "action": "cancel_preview_render",
                            "count": cancelled,
                        },
                    ),
                )
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        @self._register("compile_graph")
        def _(payload: dict) -> MCPResult:
            try:
                win.compile_graph()
                return MCPResult(success=True, data={"message": "Graph compiled."})
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        # ── Code & clipboard ────────────────────────────────────────────────

        @self._register("get_generated_code")
        def _(payload: dict) -> MCPResult:
            try:
                code = win.code_view.toPlainText()
                return MCPResult(success=True, data={"code": code})
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        @self._register("copy_code_to_clipboard")
        def _(payload: dict) -> MCPResult:
            try:
                count = win.export_code_to_clipboard()
                return MCPResult(success=True, data={"chars_copied": count})
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        @self._register("export_code")
        def _(payload: dict) -> MCPResult:
            path = payload.get("path") or payload.get("file_path")
            if not path:
                return MCPResult(success=False, error="path is required.")
            ok = win.export_code_to(str(path))
            if not ok:
                return MCPResult(success=False, error="Export failed.")
            return MCPResult(
                success=True,
                data=_payload(
                    "success",
                    affected_nodes=[],
                    metadata={"action": "export_code", "path": str(path)},
                ),
            )

        # ── Project state ───────────────────────────────────────────────────

        @self._register("save_project")
        def _(payload: dict) -> MCPResult:
            try:
                path = payload.get("path") or payload.get("file_path")
                if path:
                    ok = win.save_project_to(str(path))
                    if not ok:
                        return MCPResult(success=False, error="Save failed.")
                else:
                    if win.project_path:
                        ok = win.save_project_to(str(win.project_path))
                        if not ok:
                            return MCPResult(success=False, error="Save failed.")
                    else:
                        return MCPResult(
                            success=False,
                            error="No project path. Provide path.",
                        )
                return MCPResult(
                    success=True,
                    data=_payload(
                        "success",
                        affected_nodes=[],
                        metadata={
                            "action": "save_project",
                            "path": str(win.project_path),
                        },
                    ),
                )
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        @self._register("save_project_as")
        def _(payload: dict) -> MCPResult:
            path = payload.get("path") or payload.get("file_path")
            if not path:
                return MCPResult(success=False, error="path is required.")
            ok = win.save_project_to(str(path))
            if not ok:
                return MCPResult(success=False, error="Save failed.")
            return MCPResult(
                success=True,
                data=_payload(
                    "success",
                    affected_nodes=[],
                    metadata={"action": "save_project_as", "path": str(path)},
                ),
            )

        @self._register("open_project")
        def _(payload: dict) -> MCPResult:
            path = payload.get("path") or payload.get("file_path")
            if not path:
                return MCPResult(success=False, error="path is required.")
            try:
                win._do_open_project(str(path))
                return MCPResult(
                    success=True,
                    data=_payload(
                        "success",
                        affected_nodes=list(win.nodes.keys()),
                        metadata={"action": "open_project", "path": str(path)},
                    ),
                )
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        @self._register("new_project")
        def _(payload: dict) -> MCPResult:
            confirm = payload.get("confirm", False)
            if not confirm:
                return MCPResult(
                    success=False,
                    error="Destructive operation. Set confirm=True to proceed.",
                )
            win.new_project(force=True)
            return MCPResult(
                success=True,
                data=_payload(
                    "success", affected_nodes=[], metadata={"action": "new_project"}
                ),
            )

        @self._register("project_info")
        def _(payload: dict) -> MCPResult:
            info = {
                "path": str(win.project_path) if win.project_path else None,
                "name": win.project_name_edit.text()
                if hasattr(win, "project_name_edit")
                else "",
                "modified": bool(win.project_modified),
                "current_scene": getattr(win, "_current_scene_name", ""),
                "scenes": list(getattr(win, "_all_scenes", {}).keys()),
            }
            return MCPResult(success=True, data=info)

        @self._register("rename_project")
        def _(payload: dict) -> MCPResult:
            name = payload.get("name", "")
            if not name:
                return MCPResult(success=False, error="name is required.")
            try:
                if hasattr(win, "project_name_edit"):
                    win.project_name_edit.setText(name)
                win._rename_project()
                return MCPResult(
                    success=True,
                    data=_payload(
                        "success",
                        affected_nodes=[],
                        metadata={"action": "rename_project", "name": name},
                    ),
                )
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        @self._register("list_recents")
        def _(payload: dict) -> MCPResult:
            try:
                from core.file_manager import get_recents

                recents = get_recents()
                return MCPResult(
                    success=True,
                    data=_payload(
                        "success",
                        affected_nodes=[],
                        metadata={"action": "list_recents", "count": len(recents)},
                        extra={"recents": recents},
                    ),
                )
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        @self._register("open_recent")
        def _(payload: dict) -> MCPResult:
            try:
                from core.file_manager import get_recents

                idx = payload.get("index")
                path = payload.get("path")
                recents = get_recents()
                if path is None and idx is None:
                    return MCPResult(success=False, error="Provide index or path.")
                if path is None:
                    try:
                        path = recents[int(idx)]
                    except Exception:
                        return MCPResult(success=False, error="Invalid recent index.")
                win._do_open_project(str(path))
                return MCPResult(
                    success=True,
                    data=_payload(
                        "success",
                        affected_nodes=list(win.nodes.keys()),
                        metadata={"action": "open_recent", "path": str(path)},
                    ),
                )
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        @self._register("export_graph_json")
        def _(payload: dict) -> MCPResult:
            try:
                graph = win.get_graph_json()
                return MCPResult(
                    success=True,
                    data=_payload(
                        "success",
                        affected_nodes=list(win.nodes.keys()),
                        metadata={"action": "export_graph_json"},
                        extra={"graph": graph},
                    ),
                )
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        @self._register("import_graph_json")
        def _(payload: dict) -> MCPResult:
            graph = payload.get("graph")
            if not isinstance(graph, dict):
                return MCPResult(success=False, error="graph dict is required.")
            try:
                win.load_graph_from_json(graph)
                return MCPResult(
                    success=True,
                    data=_payload(
                        "success",
                        affected_nodes=list(win.nodes.keys()),
                        metadata={"action": "import_graph_json"},
                    ),
                )
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        # ── Scene management ───────────────────────────────────────────────

        @self._register("list_scenes")
        def _(payload: dict) -> MCPResult:
            scenes = list(getattr(win, "_all_scenes", {}).keys())
            current = getattr(win, "_current_scene_name", "")
            return MCPResult(
                success=True,
                data=_payload(
                    "success",
                    affected_nodes=[],
                    metadata={"action": "list_scenes", "current": current},
                    extra={"scenes": scenes, "current_scene": current},
                ),
            )

        @self._register("create_scene")
        def _(payload: dict) -> MCPResult:
            name = payload.get("scene_name") or payload.get("name")
            if not name:
                return MCPResult(success=False, error="scene_name is required.")
            if name in win._all_scenes:
                return MCPResult(success=False, error="Scene already exists.")
            try:
                win._on_scene_added(name)
                if payload.get("switch", False):
                    win._on_scene_switch(name)
                return MCPResult(
                    success=True,
                    data=_payload(
                        "success",
                        affected_nodes=list(win.nodes.keys()),
                        metadata={"action": "create_scene", "scene": name},
                    ),
                )
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        @self._register("delete_scene")
        def _(payload: dict) -> MCPResult:
            name = payload.get("scene_name") or payload.get("name")
            confirm = payload.get("confirm", False)
            if not confirm:
                return MCPResult(
                    success=False,
                    error="Destructive operation. Set confirm=True to proceed.",
                )
            if not name:
                return MCPResult(success=False, error="scene_name is required.")
            if name not in win._all_scenes:
                return MCPResult(success=False, error="Scene not found.")
            try:
                win._on_scene_deleted(name)
                return MCPResult(
                    success=True,
                    data=_payload(
                        "success",
                        affected_nodes=list(win.nodes.keys()),
                        metadata={"action": "delete_scene", "scene": name},
                    ),
                )
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        @self._register("rename_scene")
        def _(payload: dict) -> MCPResult:
            old = payload.get("old_name") or payload.get("scene_name")
            new = payload.get("new_name") or payload.get("name")
            if not old or not new:
                return MCPResult(
                    success=False, error="old_name and new_name are required."
                )
            if old not in win._all_scenes:
                return MCPResult(success=False, error="Scene not found.")
            if new in win._all_scenes:
                return MCPResult(success=False, error="Target name already exists.")
            try:
                win._all_scenes[new] = win._all_scenes.pop(old)
                if getattr(win, "_current_scene_name", "") == old:
                    win._current_scene_name = new
                return MCPResult(
                    success=True,
                    data=_payload(
                        "success",
                        affected_nodes=[],
                        metadata={"action": "rename_scene", "from": old, "to": new},
                    ),
                )
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        @self._register("switch_scene")
        def _(payload: dict) -> MCPResult:
            scene_name = payload.get("scene_name") or payload.get("name", "")
            if not scene_name:
                return MCPResult(success=False, error="scene_name is required.")
            if scene_name not in win._all_scenes:
                return MCPResult(
                    success=False,
                    error=f"Scene not found: {scene_name!r}. Available: {list(win._all_scenes.keys())}",
                )
            try:
                win._on_scene_switch(scene_name)
                return MCPResult(
                    success=True,
                    data=_payload(
                        "success",
                        affected_nodes=list(win.nodes.keys()),
                        metadata={"action": "switch_scene", "scene": scene_name},
                    ),
                )
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        # ── Node duplication ────────────────────────────────────────────────

        @self._register("duplicate_node")
        def _(payload: dict) -> MCPResult:
            """
            Duplicate an existing node (full clone).

            Payload:
                node_id: ID of node to duplicate
                offset_x: X offset for new node (optional, default: 100)
                offset_y: Y offset for new node (optional, default: 50)

            Returns:
                {new_node_id, new_name}
            """
            node_id = payload.get("node_id", "")
            offset_x = float(payload.get("offset_x", 100))
            offset_y = float(payload.get("offset_y", 50))

            node_item = win.nodes.get(node_id)
            if node_item is None:
                return MCPResult(success=False, error=f"Node not found: {node_id!r}")

            try:
                # Deep copy the node data
                original_data = node_item.data
                cloned_data = NodeData(
                    name=f"{original_data.name} (copy)",
                    n_type=original_data.type,
                    cls_name=original_data.cls_name,
                )

                # Copy all parameters
                cloned_data.params = dict(original_data.params)
                cloned_data.param_metadata = dict(original_data.param_metadata)

                # Copy voiceover info if present
                if original_data.audio_asset_id:
                    cloned_data.audio_asset_id = original_data.audio_asset_id
                    cloned_data.voiceover_transcript = (
                        original_data.voiceover_transcript
                    )
                    cloned_data.voiceover_duration = original_data.voiceover_duration

                # Position offset
                cloned_data.pos_x = original_data.pos_x + offset_x
                cloned_data.pos_y = original_data.pos_y + offset_y

                # Create visual item
                new_item = NodeItem(cloned_data)
                try:
                    new_item._window = win
                except Exception:
                    pass
                new_item.setPos(cloned_data.pos_x, cloned_data.pos_y)
                win.scene.addItem(new_item)
                win.nodes[cloned_data.id] = new_item

                win.compile_graph()
                win.mark_modified()
                if getattr(win, "history_manager", None):
                    win.history_manager.capture(
                        f"Duplicate {original_data.cls_name}", merge_key="duplicate"
                    )

                LOGGER.info(f"MCP duplicated node: {node_id} → {cloned_data.id}")

                return MCPResult(
                    success=True,
                    data=_payload(
                        "success",
                        affected_nodes=[cloned_data.id],
                        metadata={"action": "duplicate_node", "original_id": node_id},
                        extra={
                            "new_node_id": cloned_data.id,
                            "new_name": cloned_data.name,
                            "original_id": node_id,
                        },
                    ),
                )
            except Exception as e:
                LOGGER.error(f"duplicate_node failed: {e}")
                return MCPResult(success=False, error=f"Duplication failed: {e}")

        # ── VGroup management ──────────────────────────────────────────────

        @self._register("vgroup_list")
        def _(payload: dict) -> MCPResult:
            panel = getattr(win, "panel_vgroup", None)
            if panel is None:
                return MCPResult(success=False, error="VGroup panel not available.")
            groups = []
            for name, ids in panel.get_groups().items():
                meta = panel._meta.get(name, {}) if hasattr(panel, "_meta") else {}
                groups.append(
                    {
                        "name": name,
                        "member_ids": list(ids),
                        "source": meta.get("source", "canvas"),
                        "members": meta.get("members", []),
                    }
                )
            return MCPResult(
                success=True,
                data=_payload(
                    "success",
                    affected_nodes=[],
                    metadata={"action": "vgroup_list", "count": len(groups)},
                    extra={"vgroups": groups},
                ),
            )

        @self._register("vgroup_create")
        def _(payload: dict) -> MCPResult:
            panel = getattr(win, "panel_vgroup", None)
            if panel is None:
                return MCPResult(success=False, error="VGroup panel not available.")
            name = payload.get("name", "").strip()
            member_ids = payload.get("member_ids", []) or payload.get("members", [])
            if not name:
                name = f"vgroup_{len(panel.get_groups()) + 1}"
            if not name.isidentifier():
                return MCPResult(
                    success=False, error="name must be a valid identifier."
                )
            if name in panel.get_groups():
                return MCPResult(success=False, error="VGroup already exists.")
            ids = [mid for mid in member_ids if mid in win.nodes]
            panel._groups[name] = ids
            panel._meta[name] = {
                "source": payload.get("source", "api"),
                "members": [win.nodes[i].data.name for i in ids if i in win.nodes],
            }
            panel._refresh_tree()
            if payload.get("create_canvas_node", True):
                win.add_vgroup_node(group_name=name, member_ids=ids)
            return MCPResult(
                success=True,
                data=_payload(
                    "success",
                    affected_nodes=ids,
                    metadata={"action": "vgroup_create", "name": name},
                ),
            )

        @self._register("vgroup_rename")
        def _(payload: dict) -> MCPResult:
            panel = getattr(win, "panel_vgroup", None)
            if panel is None:
                return MCPResult(success=False, error="VGroup panel not available.")
            old = payload.get("old_name") or payload.get("name")
            new = payload.get("new_name")
            if not old or not new:
                return MCPResult(
                    success=False, error="old_name and new_name are required."
                )
            if old not in panel._groups:
                return MCPResult(success=False, error="VGroup not found.")
            if not new.isidentifier():
                return MCPResult(
                    success=False, error="new_name must be a valid identifier."
                )
            if new in panel._groups:
                return MCPResult(success=False, error="Target name already exists.")
            panel._groups = {
                (new if k == old else k): v for k, v in panel._groups.items()
            }
            panel._meta = {(new if k == old else k): v for k, v in panel._meta.items()}
            panel._refresh_tree()
            return MCPResult(
                success=True,
                data=_payload(
                    "success",
                    affected_nodes=[],
                    metadata={"action": "vgroup_rename", "from": old, "to": new},
                ),
            )

        @self._register("vgroup_duplicate")
        def _(payload: dict) -> MCPResult:
            panel = getattr(win, "panel_vgroup", None)
            if panel is None:
                return MCPResult(success=False, error="VGroup panel not available.")
            name = payload.get("name", "")
            if name not in panel._groups:
                return MCPResult(success=False, error="VGroup not found.")
            base = f"{name}_copy"
            candidate = base
            i = 2
            while candidate in panel._groups:
                candidate = f"{base}_{i}"
                i += 1
            panel._groups[candidate] = list(panel._groups[name])
            panel._meta[candidate] = dict(panel._meta.get(name, {}))
            panel._refresh_tree()
            return MCPResult(
                success=True,
                data=_payload(
                    "success",
                    affected_nodes=list(panel._groups[candidate]),
                    metadata={"action": "vgroup_duplicate", "name": candidate},
                ),
            )

        @self._register("vgroup_delete")
        def _(payload: dict) -> MCPResult:
            panel = getattr(win, "panel_vgroup", None)
            if panel is None:
                return MCPResult(success=False, error="VGroup panel not available.")
            name = payload.get("name", "")
            confirm = payload.get("confirm", False)
            if not confirm:
                return MCPResult(
                    success=False,
                    error="Destructive operation. Set confirm=True to proceed.",
                )
            if name not in panel._groups:
                return MCPResult(success=False, error="VGroup not found.")
            panel._groups.pop(name, None)
            panel._meta.pop(name, None)
            panel._refresh_tree()
            return MCPResult(
                success=True,
                data=_payload(
                    "success",
                    affected_nodes=[],
                    metadata={"action": "vgroup_delete", "name": name},
                ),
            )

        @self._register("vgroup_add_members")
        def _(payload: dict) -> MCPResult:
            panel = getattr(win, "panel_vgroup", None)
            if panel is None:
                return MCPResult(success=False, error="VGroup panel not available.")
            name = payload.get("name", "")
            member_ids = payload.get("member_ids", []) or []
            if name not in panel._groups:
                return MCPResult(success=False, error="VGroup not found.")
            added = []
            for mid in member_ids:
                if mid in win.nodes and mid not in panel._groups[name]:
                    panel._groups[name].append(mid)
                    added.append(mid)
            meta = panel._meta.setdefault(name, {"source": "canvas", "members": []})
            for mid in added:
                nm = win.nodes[mid].data.name
                if nm not in meta.get("members", []):
                    meta.setdefault("members", []).append(nm)
            panel._refresh_tree()
            return MCPResult(
                success=True,
                data=_payload(
                    "success",
                    affected_nodes=added,
                    metadata={"action": "vgroup_add_members", "name": name},
                ),
            )

        @self._register("vgroup_remove_member")
        def _(payload: dict) -> MCPResult:
            panel = getattr(win, "panel_vgroup", None)
            if panel is None:
                return MCPResult(success=False, error="VGroup panel not available.")
            name = payload.get("name", "")
            member_id = payload.get("member_id")
            member_name = payload.get("member_name")
            if name not in panel._groups:
                return MCPResult(success=False, error="VGroup not found.")
            removed = []
            if member_id:
                if member_id in panel._groups[name]:
                    panel._groups[name] = [
                        i for i in panel._groups[name] if i != member_id
                    ]
                    removed.append(member_id)
            elif member_name:
                new_ids = []
                for nid in panel._groups[name]:
                    node = win.nodes.get(nid)
                    if node and node.data.name == member_name:
                        removed.append(nid)
                        continue
                    new_ids.append(nid)
                panel._groups[name] = new_ids
                meta_members = panel._meta.get(name, {}).get("members", [])
                if member_name in meta_members:
                    meta_members.remove(member_name)
            else:
                return MCPResult(
                    success=False, error="member_id or member_name is required."
                )
            panel._refresh_tree()
            return MCPResult(
                success=True,
                data=_payload(
                    "success",
                    affected_nodes=removed,
                    metadata={"action": "vgroup_remove_member", "name": name},
                ),
            )

        @self._register("vgroup_highlight")
        def _(payload: dict) -> MCPResult:
            panel = getattr(win, "panel_vgroup", None)
            if panel is None:
                return MCPResult(success=False, error="VGroup panel not available.")
            name = payload.get("name", "")
            if name not in panel._groups:
                return MCPResult(success=False, error="VGroup not found.")
            ids = panel._groups[name]
            win.scene.clearSelection()
            found = []
            for nid in ids:
                node = win.nodes.get(nid)
                if node:
                    node.setSelected(True)
                    found.append(nid)
            win.on_selection()
            return MCPResult(
                success=True,
                data=_payload(
                    "success",
                    affected_nodes=found,
                    metadata={"action": "vgroup_highlight", "name": name},
                ),
            )

        # ── Asset management ────────────────────────────────────────────────

        @self._register("add_asset")
        def _(payload: dict) -> MCPResult:
            """
            Add a new asset (image, video, audio, etc.).

            Payload:
                file_path: Path to asset file
                asset_name: Name for asset (optional, auto-detected if not provided)
                asset_kind: Kind of asset ("image", "video", "audio", "data")

            Returns:
                {asset_id, asset_name, file_path}
            """
            file_path = payload.get("file_path", "")
            if not file_path:
                return MCPResult(success=False, error="file_path is required")

            try:
                asset_mgr = _get_global_assets()
                asset = asset_mgr.add_asset(file_path)

                if not asset:
                    return MCPResult(success=False, error="Failed to add asset")

                LOGGER.info(f"MCP added asset: {asset.name} ({asset.kind})")

                return MCPResult(
                    success=True,
                    data=_payload(
                        "success",
                        affected_nodes=[],
                        metadata={"action": "add_asset"},
                        extra={
                            "asset_id": asset.id,
                            "asset_name": asset.name,
                            "file_path": asset.current_path,
                            "asset_kind": asset.kind,
                        },
                    ),
                )
            except Exception as e:
                return MCPResult(success=False, error=f"Failed to add asset: {e}")

        @self._register("update_asset")
        def _(payload: dict) -> MCPResult:
            """
            Update an existing asset.

            Payload:
                asset_id: ID of asset to update
                new_file_path: New file path (optional)
                new_name: New name (optional)

            Returns:
                {asset_id, asset_name, file_path}
            """
            asset_id = payload.get("asset_id", "")
            new_file_path = payload.get("new_file_path", "")
            new_name = payload.get("new_name", "")

            if not asset_id:
                return MCPResult(success=False, error="asset_id is required")

            try:
                asset_mgr = _get_global_assets()
                asset = asset_mgr.update_asset(
                    asset_id,
                    new_path=new_file_path or None,
                    new_name=new_name or None,
                )

                if not asset:
                    return MCPResult(
                        success=False, error=f"Asset not found: {asset_id}"
                    )

                LOGGER.info(f"MCP updated asset: {asset_id}")

                return MCPResult(
                    success=True,
                    data=_payload(
                        "success",
                        affected_nodes=[],
                        metadata={"action": "update_asset", "asset_id": asset.id},
                        extra={
                            "asset_id": asset.id,
                            "asset_name": asset.name,
                            "file_path": asset.current_path,
                        },
                    ),
                )
            except Exception as e:
                return MCPResult(success=False, error=f"Failed to update asset: {e}")

        @self._register("delete_asset")
        def _(payload: dict) -> MCPResult:
            """
            Delete an asset.

            Payload:
                asset_id: ID of asset to delete
                confirm: Set to True to confirm deletion

            Returns:
                {deleted_id}
            """
            asset_id = payload.get("asset_id", "")
            confirm = payload.get("confirm", False)

            if not asset_id:
                return MCPResult(success=False, error="asset_id is required")

            if not confirm:
                return MCPResult(
                    success=False,
                    error="Destructive operation. Set confirm=True to proceed.",
                )

            try:
                asset_mgr = _get_global_assets()
                if not asset_mgr.delete_asset(asset_id):
                    return MCPResult(
                        success=False, error=f"Asset not found: {asset_id}"
                    )
                win.mark_modified()

                LOGGER.info(f"MCP deleted asset: {asset_id}")

                return MCPResult(
                    success=True,
                    data=_payload(
                        "success",
                        affected_nodes=[],
                        metadata={"action": "delete_asset", "asset_id": asset_id},
                        extra={"deleted_id": asset_id},
                    ),
                )
            except Exception as e:
                return MCPResult(success=False, error=f"Failed to delete asset: {e}")

        @self._register("list_assets")
        def _(payload: dict) -> MCPResult:
            """
            List all assets.

            Returns:
                {assets: [{id, name, kind, path}, ...]}
            """
            try:
                asset_mgr = _get_global_assets()
                assets_list = [
                    {
                        "id": a.id,
                        "name": a.name,
                        "kind": a.kind,
                        "path": a.current_path,
                    }
                    for a in asset_mgr.get_list()
                ]

                return MCPResult(
                    success=True,
                    data=_payload(
                        "success",
                        affected_nodes=[],
                        metadata={"action": "list_assets", "count": len(assets_list)},
                        extra={"assets": assets_list},
                    ),
                )
            except Exception as e:
                return MCPResult(success=False, error=f"Failed to list assets: {e}")

        @self._register("get_asset")
        def _(payload: dict) -> MCPResult:
            """
            Get details for a specific asset.

            Payload:
                asset_id: ID of asset to retrieve

            Returns:
                {id, name, kind, path}
            """
            asset_id = payload.get("asset_id", "")

            if not asset_id:
                return MCPResult(success=False, error="asset_id is required")

            try:
                asset_mgr = _get_global_assets()
                asset = asset_mgr.get_asset(asset_id)

                if not asset:
                    return MCPResult(
                        success=False, error=f"Asset not found: {asset_id}"
                    )

                return MCPResult(
                    success=True,
                    data=_payload(
                        "success",
                        affected_nodes=[],
                        metadata={"action": "get_asset", "asset_id": asset.id},
                        extra={
                            "id": asset.id,
                            "name": asset.name,
                            "kind": asset.kind,
                            "path": asset.current_path,
                        },
                    ),
                )
            except Exception as e:
                return MCPResult(success=False, error=f"Failed to get asset: {e}")

        # ── AI & Automation ────────────────────────────────────────────────

        @self._register("ai_generate_code")
        def _(payload: dict) -> MCPResult:
            prompt = payload.get("prompt", "").strip()
            if not prompt:
                return MCPResult(success=False, error="prompt is required.")
            panel = getattr(win, "panel_ai", None)
            if panel is None:
                return MCPResult(success=False, error="AI panel not available.")
            try:
                panel.input.setPlainText(prompt)
                panel.chk_mcp_mode.setChecked(False)
                panel.chk_auto_voiceover.setChecked(False)
                panel.generate()
                return MCPResult(
                    success=True,
                    data=_payload(
                        "success",
                        affected_nodes=[],
                        metadata={"action": "ai_generate_code", "prompt": prompt},
                    ),
                )
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        @self._register("ai_merge_code")
        def _(payload: dict) -> MCPResult:
            code = payload.get("code")
            panel = getattr(win, "panel_ai", None)
            if code is None and panel is not None:
                code = panel.last_code
            if not code:
                return MCPResult(success=False, error="code is required.")
            try:
                win.merge_ai_code(code)
                return MCPResult(
                    success=True,
                    data=_payload(
                        "success",
                        affected_nodes=list(win.nodes.keys()),
                        metadata={"action": "ai_merge_code", "code_chars": len(code)},
                    ),
                )
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        @self._register("ai_reject_code")
        def _(payload: dict) -> MCPResult:
            panel = getattr(win, "panel_ai", None)
            if panel is None:
                return MCPResult(success=False, error="AI panel not available.")
            try:
                panel.reject()
                return MCPResult(
                    success=True,
                    data=_payload(
                        "success",
                        affected_nodes=[],
                        metadata={"action": "ai_reject_code"},
                    ),
                )
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        @self._register("ai_get_last")
        def _(payload: dict) -> MCPResult:
            panel = getattr(win, "panel_ai", None)
            if panel is None:
                return MCPResult(success=False, error="AI panel not available.")
            data = {
                "last_code": panel.last_code,
                "extracted_nodes": list(panel.extracted_nodes or []),
            }
            return MCPResult(success=True, data=data)

        @self._register("ai_run_agent")
        def _(payload: dict) -> MCPResult:
            prompt = payload.get("prompt", "").strip()
            if not prompt:
                return MCPResult(success=False, error="prompt is required.")
            panel = getattr(win, "panel_ai", None)
            if panel is None:
                return MCPResult(success=False, error="AI panel not available.")
            try:
                if hasattr(panel, "chk_mcp_mode"):
                    panel.chk_mcp_mode.setChecked(True)
                panel.input.setPlainText(prompt)
                panel.generate()
                return MCPResult(
                    success=True,
                    data=_payload(
                        "success",
                        affected_nodes=[],
                        metadata={"action": "ai_run_agent", "prompt": prompt},
                    ),
                )
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        @self._register("ai_auto_voiceover")
        def _(payload: dict) -> MCPResult:
            panel = getattr(win, "panel_ai", None)
            if panel is None:
                return MCPResult(success=False, error="AI panel not available.")
            try:
                panel._run_auto_voiceover_agent()
                return MCPResult(
                    success=True,
                    data=_payload(
                        "success",
                        affected_nodes=[],
                        metadata={"action": "ai_auto_voiceover"},
                    ),
                )
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        @self._register("tts_generate")
        def _(payload: dict) -> MCPResult:
            text = payload.get("text", "").strip()
            if not text:
                return MCPResult(success=False, error="text is required.")
            try:
                from rendering.render_manager import TTSWorker
                from PySide6.QtCore import QEventLoop
                from core.config import SETTINGS

                voice = payload.get("voice") or SETTINGS.get("DEFAULT_VOICE", "Zephyr")
                model = payload.get("model") or SETTINGS.get(
                    "TTS_MODEL", "gemini-2.5-flash-preview-tts"
                )
                loop = QEventLoop()
                result = {"path": None, "error": None}

                worker = TTSWorker(text, voice, model)

                def _done(path):
                    result["path"] = path
                    loop.quit()

                def _err(msg):
                    result["error"] = msg
                    loop.quit()

                worker.finished_signal.connect(_done)
                worker.error_signal.connect(_err)
                worker.start()
                loop.exec()

                if result["error"]:
                    return MCPResult(success=False, error=str(result["error"]))

                audio_path = result["path"]
                asset = _get_global_assets().add_asset(audio_path)
                node_id = payload.get("node_id")
                if node_id and node_id in win.nodes and asset:
                    node_item = win.nodes[node_id]
                    node_item.data.audio_asset_id = asset.id
                    node_item.data.voiceover_transcript = text
                    node_item.update()
                return MCPResult(
                    success=True,
                    data=_payload(
                        "success",
                        affected_nodes=[node_id] if node_id else [],
                        metadata={
                            "action": "tts_generate",
                            "voice": voice,
                            "model": model,
                        },
                        extra={
                            "audio_path": audio_path,
                            "asset_id": asset.id if asset else None,
                            "attached_node": node_id if node_id else None,
                        },
                    ),
                )
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        @self._register("ai_load_snippet")
        def _(payload: dict) -> MCPResult:
            code = payload.get("code", "")
            source = payload.get("source", "snippet")
            if not code:
                return MCPResult(success=False, error="code is required.")
            try:
                win._load_code_to_ai(code, source=source)
                return MCPResult(
                    success=True,
                    data=_payload(
                        "success",
                        affected_nodes=[],
                        metadata={"action": "ai_load_snippet", "source": source},
                    ),
                )
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        @self._register("ai_pdf_import")
        def _(payload: dict) -> MCPResult:
            """Parse PDFs with AI and optionally apply to node graph."""
            paths = payload.get("paths") or payload.get("pdf_paths") or []
            prompt = payload.get("prompt", "")
            apply_nodes = bool(payload.get("apply_nodes", True))
            if not isinstance(paths, list) or not paths:
                return MCPResult(success=False, error="paths list is required.")
            try:
                from utils.ai_pdf_parser import PDFParser
                from utils.ai_context_builder import AIContextBuilder
                from utils.ai_slide_animator import AISlideAnimator
                from core.ai_slides_to_manim import SlidesToManim
                from graph.ai_slides_to_nodes import SlidesToNodes
                from utils.helpers import sanitize_background_settings

                parser = PDFParser()
                parsed = parser.parse_pdfs(paths)
                context = AIContextBuilder().build_context(parsed, prompt)
                animator = AISlideAnimator()
                slide_deck = animator.generate_slides(context, prompt)
                manim_code = SlidesToManim.generate_code(slide_deck)
                manim_code = sanitize_background_settings(manim_code)

                affected = []
                result = {}
                if apply_nodes:
                    result = SlidesToNodes.apply_to_scene(slide_deck, win, True)
                    affected = result.get("nodes", [])
                return MCPResult(
                    success=True,
                    data=_payload(
                        "success",
                        affected_nodes=[n.data.id for n in affected]
                        if affected
                        else [],
                        metadata={
                            "action": "ai_pdf_import",
                            "prompt": prompt,
                            "nodes_added": result.get("node_count", 0) if result else 0,
                        },
                        extra={
                            "slide_count": len(slide_deck.get("slides", []))
                            if isinstance(slide_deck, dict)
                            else len(slide_deck),
                            "manim_code": manim_code,
                        },
                    ),
                )
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        # ── Settings, Theme, Keybindings ───────────────────────────────────

        @self._register("settings_get")
        def _(payload: dict) -> MCPResult:
            key = payload.get("key", "")
            if not key:
                return MCPResult(success=False, error="key is required.")
            try:
                from core.config import SETTINGS

                type_hint = payload.get("type")
                type_map = {"bool": bool, "int": int, "float": float, "str": str}
                t = type_map.get(str(type_hint).lower()) if type_hint else None
                val = SETTINGS.get(key, payload.get("default"), type=t)
                return MCPResult(success=True, data={"key": key, "value": val})
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        @self._register("settings_set")
        def _(payload: dict) -> MCPResult:
            key = payload.get("key", "")
            if not key:
                return MCPResult(success=False, error="key is required.")
            try:
                from core.config import SETTINGS

                SETTINGS.set(key, payload.get("value"))
                return MCPResult(
                    success=True,
                    data=_payload(
                        "success",
                        affected_nodes=[],
                        metadata={"action": "settings_set", "key": key},
                    ),
                )
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        @self._register("settings_list")
        def _(payload: dict) -> MCPResult:
            try:
                from core.config import SETTINGS

                keys = SETTINGS._store.allKeys()
                return MCPResult(
                    success=True,
                    data=_payload(
                        "success",
                        affected_nodes=[],
                        metadata={"action": "settings_list", "count": len(keys)},
                        extra={"keys": keys},
                    ),
                )
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        @self._register("theme_get_palette")
        def _(payload: dict) -> MCPResult:
            try:
                from core.themes import LightTheme

                palette = {
                    k: getattr(LightTheme, k)
                    for k in dir(LightTheme)
                    if k.isupper() and isinstance(getattr(LightTheme, k), str)
                }
                return MCPResult(
                    success=True,
                    data=_payload(
                        "success",
                        affected_nodes=[],
                        metadata={"action": "theme_get_palette"},
                        extra={"palette": palette},
                    ),
                )
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        @self._register("theme_set_color")
        def _(payload: dict) -> MCPResult:
            color_name = payload.get("color_name", "")
            value = payload.get("value", "")
            if not color_name or not value:
                return MCPResult(
                    success=False, error="color_name and value are required."
                )
            try:
                from core.themes import LightTheme, THEME_MANAGER

                if not hasattr(LightTheme, color_name):
                    return MCPResult(success=False, error="Unknown color name.")
                setattr(LightTheme, color_name, str(value))
                THEME_MANAGER.reload_stylesheet()
                win.apply_theme()
                return MCPResult(
                    success=True,
                    data=_payload(
                        "success",
                        affected_nodes=[],
                        metadata={
                            "action": "theme_set_color",
                            "color_name": color_name,
                            "value": value,
                        },
                    ),
                )
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        @self._register("theme_reload")
        def _(payload: dict) -> MCPResult:
            try:
                from core.themes import THEME_MANAGER

                THEME_MANAGER.reload_stylesheet()
                win.apply_theme()
                return MCPResult(
                    success=True,
                    data=_payload(
                        "success",
                        affected_nodes=[],
                        metadata={"action": "theme_reload"},
                    ),
                )
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        @self._register("keybindings_list")
        def _(payload: dict) -> MCPResult:
            try:
                from core.keybinding_registry import KEYBINDINGS

                actions = [a.to_dict() for a in KEYBINDINGS.get_all_actions()]
                return MCPResult(
                    success=True,
                    data=_payload(
                        "success",
                        affected_nodes=[],
                        metadata={"action": "keybindings_list", "count": len(actions)},
                        extra={"actions": actions},
                    ),
                )
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        @self._register("keybindings_set")
        def _(payload: dict) -> MCPResult:
            action = payload.get("action", "")
            shortcut = payload.get("shortcut", "")
            if not action or not shortcut:
                return MCPResult(
                    success=False, error="action and shortcut are required."
                )
            try:
                from core.keybinding_registry import KEYBINDINGS

                ok, msg = KEYBINDINGS.set_binding(action, shortcut)
                if not ok:
                    return MCPResult(success=False, error=msg)
                return MCPResult(
                    success=True,
                    data=_payload(
                        "success",
                        affected_nodes=[],
                        metadata={"action": "keybindings_set", "action_name": action},
                    ),
                )
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        @self._register("keybindings_reset")
        def _(payload: dict) -> MCPResult:
            action = payload.get("action", "")
            if not action:
                return MCPResult(success=False, error="action is required.")
            try:
                from core.keybinding_registry import KEYBINDINGS

                ok = KEYBINDINGS.reset_binding(action)
                if not ok:
                    return MCPResult(success=False, error="Action not found.")
                return MCPResult(
                    success=True,
                    data=_payload(
                        "success",
                        affected_nodes=[],
                        metadata={"action": "keybindings_reset", "action_name": action},
                    ),
                )
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        @self._register("keybindings_reset_all")
        def _(payload: dict) -> MCPResult:
            try:
                from core.keybinding_registry import KEYBINDINGS

                KEYBINDINGS.reset_all()
                return MCPResult(
                    success=True,
                    data=_payload(
                        "success",
                        affected_nodes=[],
                        metadata={"action": "keybindings_reset_all"},
                    ),
                )
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        @self._register("usage_top")
        def _(payload: dict) -> MCPResult:
            try:
                from core.file_manager import USAGE_TRACKER

                n = int(payload.get("count", 5))
                return MCPResult(
                    success=True,
                    data=_payload(
                        "success",
                        affected_nodes=[],
                        metadata={"action": "usage_top", "count": n},
                        extra={"items": USAGE_TRACKER.top(n)},
                    ),
                )
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        @self._register("usage_top_mobjects")
        def _(payload: dict) -> MCPResult:
            try:
                from core.file_manager import USAGE_TRACKER

                n = int(payload.get("count", 5))
                return MCPResult(
                    success=True,
                    data=_payload(
                        "success",
                        affected_nodes=[],
                        metadata={"action": "usage_top_mobjects", "count": n},
                        extra={"items": USAGE_TRACKER.top_mobjects(n)},
                    ),
                )
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        @self._register("usage_top_animations")
        def _(payload: dict) -> MCPResult:
            try:
                from core.file_manager import USAGE_TRACKER

                n = int(payload.get("count", 5))
                return MCPResult(
                    success=True,
                    data=_payload(
                        "success",
                        affected_nodes=[],
                        metadata={"action": "usage_top_animations", "count": n},
                        extra={"items": USAGE_TRACKER.top_animations(n)},
                    ),
                )
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        # ── UI actions (menus/buttons) ─────────────────────────────────────

        @self._register("ui_list_actions")
        def _(payload: dict) -> MCPResult:
            try:
                from PySide6.QtGui import QAction

                actions = []
                for act in win.findChildren(QAction):
                    actions.append(
                        {
                            "text": act.text(),
                            "object_name": act.objectName(),
                            "shortcut": act.shortcut().toString()
                            if hasattr(act, "shortcut")
                            else "",
                        }
                    )
                return MCPResult(
                    success=True,
                    data=_payload(
                        "success",
                        affected_nodes=[],
                        metadata={"action": "ui_list_actions", "count": len(actions)},
                        extra={"actions": actions},
                    ),
                )
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        @self._register("ui_trigger_action")
        def _(payload: dict) -> MCPResult:
            name = (
                payload.get("name") or payload.get("text") or payload.get("object_name")
            )
            if not name:
                return MCPResult(success=False, error="name or text is required.")
            try:
                from PySide6.QtGui import QAction

                target = None
                for act in win.findChildren(QAction):
                    if act.objectName() == name or act.text() == name:
                        target = act
                        break
                    if name.lower() in act.text().lower():
                        target = act
                        break
                if target is None:
                    return MCPResult(success=False, error="Action not found.")
                target.trigger()
                return MCPResult(
                    success=True,
                    data=_payload(
                        "success",
                        affected_nodes=[],
                        metadata={"action": "ui_trigger_action", "name": name},
                    ),
                )
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        # ── History & Checkpoints ──────────────────────────────────────────

        def _history_payload(
            status: str,
            affected_nodes: Optional[list] = None,
            metadata: Optional[dict] = None,
            extra: Optional[dict] = None,
        ) -> dict:
            payload = {
                "status": status,
                "affected_nodes": affected_nodes or [],
                "history_pointer": getattr(win, "history_manager", None)
                and win.history_manager.history_pointer()
                or 0,
                "metadata": metadata or {},
            }
            if extra:
                payload.update(extra)
            return payload

        # ── Explain & Learning/Teacher Modes ──────────────────────────────

        @self._register("explain.scene")
        def _(payload: dict) -> MCPResult:
            mode = payload.get("mode", "detailed")
            service = getattr(win, "explain_service", None)
            if service is None:
                return MCPResult(success=False, error="Explain service not available.")
            try:
                analysis = service.analyze_scene()
                resp = service.explain_scene(analysis, mode)
                data = {
                    "scene_name": analysis.scene_name,
                    "concept_explanation": resp.concept_explanation,
                    "step_by_step": resp.step_by_step,
                    "visual_reasoning": resp.visual_reasoning,
                    "simple_explanation": resp.simple_explanation,
                    "key_takeaways": resp.key_takeaways,
                }
                return MCPResult(success=True, data=data)
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        @self._register("explain.selected_nodes")
        def _(payload: dict) -> MCPResult:
            node_ids = payload.get("node_ids", [])
            mode = payload.get("mode", "detailed")
            if not node_ids:
                return MCPResult(success=False, error="node_ids is required.")
            service = getattr(win, "explain_service", None)
            if service is None:
                return MCPResult(success=False, error="Explain service not available.")
            try:
                analysis = service.analyze_scene(node_ids=node_ids)
                resp = service.explain_scene(analysis, mode)
                data = {
                    "scene_name": analysis.scene_name,
                    "concept_explanation": resp.concept_explanation,
                    "step_by_step": resp.step_by_step,
                    "visual_reasoning": resp.visual_reasoning,
                    "simple_explanation": resp.simple_explanation,
                    "key_takeaways": resp.key_takeaways,
                }
                return MCPResult(success=True, data=data)
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        @self._register("explain.selected_animation")
        def _(payload: dict) -> MCPResult:
            animation_id = payload.get("animation_id", "")
            mode = payload.get("mode", "detailed")
            if not animation_id:
                return MCPResult(success=False, error="animation_id is required.")
            service = getattr(win, "explain_service", None)
            if service is None:
                return MCPResult(success=False, error="Explain service not available.")
            try:
                analysis = service.analyze_scene(animation_id=animation_id)
                resp = service.explain_scene(analysis, mode)
                data = {
                    "scene_name": analysis.scene_name,
                    "concept_explanation": resp.concept_explanation,
                    "step_by_step": resp.step_by_step,
                    "visual_reasoning": resp.visual_reasoning,
                    "simple_explanation": resp.simple_explanation,
                    "key_takeaways": resp.key_takeaways,
                }
                return MCPResult(success=True, data=data)
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        @self._register("explain.regenerate")
        def _(payload: dict) -> MCPResult:
            mode = payload.get("mode", "detailed")
            service = getattr(win, "explain_service", None)
            if service is None:
                return MCPResult(success=False, error="Explain service not available.")
            try:
                analysis = service.analyze_scene()
                resp = service.explain_scene(analysis, mode)
                data = {
                    "scene_name": analysis.scene_name,
                    "concept_explanation": resp.concept_explanation,
                    "step_by_step": resp.step_by_step,
                    "visual_reasoning": resp.visual_reasoning,
                    "simple_explanation": resp.simple_explanation,
                    "key_takeaways": resp.key_takeaways,
                }
                return MCPResult(success=True, data=data)
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        @self._register("explain.history_checkpoint")
        def _(payload: dict) -> MCPResult:
            checkpoint_id = payload.get("checkpoint_id", "")
            mode = payload.get("mode", "detailed")
            if not checkpoint_id:
                return MCPResult(success=False, error="checkpoint_id is required.")
            history = getattr(win, "history_manager", None)
            service = getattr(win, "explain_service", None)
            if history is None or service is None:
                return MCPResult(success=False, error="History or explain service not available.")
            try:
                from scene_explainer.history_explainer import HistoryExplainer

                explainer = HistoryExplainer(
                    history, service.analyzer, service.prompt_builder, service.ai
                )
                resp = explainer.explain_checkpoint(checkpoint_id, mode)
                data = {
                    "checkpoint_id": checkpoint_id,
                    "concept_explanation": resp.concept_explanation,
                    "step_by_step": resp.step_by_step,
                    "visual_reasoning": resp.visual_reasoning,
                    "simple_explanation": resp.simple_explanation,
                }
                return MCPResult(success=True, data=data)
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        @self._register("explain.history_change")
        def _(payload: dict) -> MCPResult:
            from_cp = payload.get("from_checkpoint", "")
            to_cp = payload.get("to_checkpoint", "")
            mode = payload.get("mode", "detailed")
            if not from_cp or not to_cp:
                return MCPResult(success=False, error="from_checkpoint and to_checkpoint are required.")
            history = getattr(win, "history_manager", None)
            service = getattr(win, "explain_service", None)
            if history is None or service is None:
                return MCPResult(success=False, error="History or explain service not available.")
            try:
                from scene_explainer.history_explainer import HistoryExplainer

                explainer = HistoryExplainer(
                    history, service.analyzer, service.prompt_builder, service.ai
                )
                resp = explainer.explain_history_change(from_cp, to_cp, mode)
                data = {
                    "objects_added": resp.objects_added,
                    "objects_removed": resp.objects_removed,
                    "animations_added": resp.animations_added,
                    "animations_removed": resp.animations_removed,
                    "concept_change_summary": resp.concept_change_summary,
                    "educational_significance": resp.educational_significance,
                }
                return MCPResult(success=True, data=data)
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        @self._register("explain.undo_action")
        def _(payload: dict) -> MCPResult:
            mode = payload.get("mode", "detailed")
            history = getattr(win, "history_manager", None)
            service = getattr(win, "explain_service", None)
            if history is None or service is None:
                return MCPResult(success=False, error="History or explain service not available.")
            try:
                from scene_explainer.history_explainer import HistoryExplainer

                explainer = HistoryExplainer(
                    history, service.analyzer, service.prompt_builder, service.ai
                )
                resp = explainer.explain_undo(mode)
                return MCPResult(success=True, data=resp)
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        @self._register("explain.redo_action")
        def _(payload: dict) -> MCPResult:
            mode = payload.get("mode", "detailed")
            history = getattr(win, "history_manager", None)
            service = getattr(win, "explain_service", None)
            if history is None or service is None:
                return MCPResult(success=False, error="History or explain service not available.")
            try:
                from scene_explainer.history_explainer import HistoryExplainer

                explainer = HistoryExplainer(
                    history, service.analyzer, service.prompt_builder, service.ai
                )
                resp = explainer.explain_redo(mode)
                return MCPResult(success=True, data=resp)
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        @self._register("learning_mode.enable")
        def _(payload: dict) -> MCPResult:
            SETTINGS.set("LEARNING_MODE_ENABLED", True)
            return MCPResult(success=True, data={"enabled": True, "status": "Learning Mode is now active"})

        @self._register("learning_mode.disable")
        def _(payload: dict) -> MCPResult:
            SETTINGS.set("LEARNING_MODE_ENABLED", False)
            return MCPResult(success=True, data={"enabled": False, "status": "Learning Mode has been disabled"})

        @self._register("learning_mode.status")
        def _(payload: dict) -> MCPResult:
            enabled = bool(SETTINGS.get("LEARNING_MODE_ENABLED", False, type=bool))
            return MCPResult(success=True, data={"enabled": enabled})

        @self._register("teacher_mode.enable")
        def _(payload: dict) -> MCPResult:
            SETTINGS.set("TEACHER_MODE_ENABLED", True)
            return MCPResult(success=True, data={"enabled": True, "status": "Teacher Mode is now active"})

        @self._register("teacher_mode.disable")
        def _(payload: dict) -> MCPResult:
            SETTINGS.set("TEACHER_MODE_ENABLED", False)
            return MCPResult(success=True, data={"enabled": False, "status": "Teacher Mode has been disabled"})

        @self._register("teacher_mode.generate_lesson")
        def _(payload: dict) -> MCPResult:
            service = getattr(win, "explain_service", None)
            if service is None:
                return MCPResult(success=False, error="Explain service not available.")
            try:
                analysis = service.analyze_scene()
                lesson = service.lesson_notes(analysis)
                data = {
                    "concept_explanation": lesson.concept_explanation,
                    "visual_explanation": lesson.visual_explanation,
                    "step_by_step_teaching": lesson.step_by_step_teaching,
                    "student_notes": lesson.student_notes,
                    "key_takeaways": lesson.key_takeaways,
                }
                return MCPResult(success=True, data=data)
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        @self._register("app.quit")
        def _(payload: dict) -> MCPResult:
            try:
                if hasattr(win, "explain_service") and win.explain_service is not None:
                    try:
                        win.explain_service.cancel_all()
                    except Exception:
                        pass
                if hasattr(win, "request_app_quit"):
                    should_close = bool(win.request_app_quit())
                    if not should_close:
                        return MCPResult(success=False, error="Quit cancelled by user.")
                    try:
                        setattr(win, "_skip_quit_prompt", True)
                    except Exception:
                        pass
                    win.close()
                    return MCPResult(success=True, data={"status": "success", "message": "Application closed safely"})
                win.close()
                return MCPResult(success=True, data={"status": "success", "message": "Application closed safely"})
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        @self._register("history.undo_project")
        def _(payload: dict) -> MCPResult:
            history = getattr(win, "history_manager", None)
            if history is None:
                return MCPResult(success=False, error="History manager not available.")
            diff = history.undo()
            if not diff:
                return MCPResult(
                    success=False,
                    data=_history_payload("error"),
                    error="Nothing to undo.",
                )
            return MCPResult(
                success=True,
                data=_history_payload(
                    "success",
                    affected_nodes=diff.get("affected_nodes", []),
                    metadata={"action": "undo_project"},
                ),
            )

        @self._register("history.undo_scene")
        def _(payload: dict) -> MCPResult:
            history = getattr(win, "history_manager", None)
            if history is None:
                return MCPResult(success=False, error="History manager not available.")
            scene = payload.get("scene_name") or payload.get("scene")
            current_scene = getattr(win, "_current_scene_name", "")
            if scene and scene != current_scene:
                return MCPResult(
                    success=False,
                    error="undo_scene operates on the active scene. Switch scene first.",
                )
            diff = history.undo()
            if not diff:
                return MCPResult(
                    success=False,
                    data=_history_payload("error"),
                    error="Nothing to undo for scene.",
                )
            return MCPResult(
                success=True,
                data=_history_payload(
                    "success",
                    affected_nodes=diff.get("affected_nodes", []),
                    metadata={"action": "undo_scene", "scene": current_scene},
                ),
            )

        @self._register("history.redo_project")
        def _(payload: dict) -> MCPResult:
            history = getattr(win, "history_manager", None)
            if history is None:
                return MCPResult(success=False, error="History manager not available.")
            diff = history.redo()
            if not diff:
                return MCPResult(
                    success=False,
                    data=_history_payload("error"),
                    error="Nothing to redo.",
                )
            return MCPResult(
                success=True,
                data=_history_payload(
                    "success",
                    affected_nodes=diff.get("affected_nodes", []),
                    metadata={"action": "redo_project"},
                ),
            )

        @self._register("history.redo_scene")
        def _(payload: dict) -> MCPResult:
            history = getattr(win, "history_manager", None)
            if history is None:
                return MCPResult(success=False, error="History manager not available.")
            scene = payload.get("scene_name") or payload.get("scene")
            current_scene = getattr(win, "_current_scene_name", "")
            if scene and scene != current_scene:
                return MCPResult(
                    success=False,
                    error="redo_scene operates on the active scene. Switch scene first.",
                )
            diff = history.redo()
            if not diff:
                return MCPResult(
                    success=False,
                    data=_history_payload("error"),
                    error="Nothing to redo for scene.",
                )
            return MCPResult(
                success=True,
                data=_history_payload(
                    "success",
                    affected_nodes=diff.get("affected_nodes", []),
                    metadata={"action": "redo_scene", "scene": current_scene},
                ),
            )

        @self._register("history.undo_node")
        def _(payload: dict) -> MCPResult:
            node_id = payload.get("node_id", "")
            history = getattr(win, "history_manager", None)
            if history is None:
                return MCPResult(success=False, error="History manager not available.")
            if not node_id:
                return MCPResult(success=False, error="node_id is required.")
            result = history.undo_node(node_id)
            if not result:
                return MCPResult(
                    success=False,
                    data=_history_payload("error"),
                    error="Nothing to undo for that node.",
                )
            return MCPResult(
                success=True,
                data=_history_payload(
                    "success",
                    affected_nodes=result.get("affected_nodes", []),
                    metadata={"action": "undo_node", "node_id": node_id},
                ),
            )

        @self._register("history.redo_node")
        def _(payload: dict) -> MCPResult:
            node_id = payload.get("node_id", "")
            history = getattr(win, "history_manager", None)
            if history is None:
                return MCPResult(success=False, error="History manager not available.")
            if not node_id:
                return MCPResult(success=False, error="node_id is required.")
            result = history.redo_node(node_id)
            if not result:
                return MCPResult(
                    success=False,
                    data=_history_payload("error"),
                    error="Nothing to redo for that node.",
                )
            return MCPResult(
                success=True,
                data=_history_payload(
                    "success",
                    affected_nodes=result.get("affected_nodes", []),
                    metadata={"action": "redo_node", "node_id": node_id},
                ),
            )

        @self._register("history.create_checkpoint")
        def _(payload: dict) -> MCPResult:
            history = getattr(win, "history_manager", None)
            if history is None:
                return MCPResult(success=False, error="History manager not available.")
            name = payload.get("name", "")
            scope = payload.get("scope", "scene")
            node_id = payload.get("node_id")
            description = payload.get("description", "")
            metadata = payload.get("metadata", {}) or {}
            if not name:
                return MCPResult(success=False, error="Checkpoint name is required.")
            try:
                cp = history.create_checkpoint(
                    name=name,
                    scope=scope,
                    node_id=node_id,
                    description=description,
                    metadata=metadata,
                )
                return MCPResult(
                    success=True,
                    data=_history_payload(
                        "success",
                        affected_nodes=cp.metadata.get("affected_nodes", []),
                        metadata=cp.to_dict(),
                    ),
                )
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        @self._register("history.restore_checkpoint")
        def _(payload: dict) -> MCPResult:
            history = getattr(win, "history_manager", None)
            if history is None:
                return MCPResult(success=False, error="History manager not available.")
            name = payload.get("checkpoint_name", "") or payload.get("name", "")
            if not name:
                return MCPResult(success=False, error="checkpoint_name is required.")
            result = history.restore_checkpoint(name)
            if not result:
                return MCPResult(
                    success=False,
                    data=_history_payload("error"),
                    error="Checkpoint not found or restore failed.",
                )
            return MCPResult(
                success=True,
                data=_history_payload(
                    "success",
                    affected_nodes=result.get("affected_nodes", []),
                    metadata={"checkpoint": name},
                ),
            )

        @self._register("history.list_checkpoints")
        def _(payload: dict) -> MCPResult:
            history = getattr(win, "history_manager", None)
            if history is None:
                return MCPResult(success=False, error="History manager not available.")
            scope = payload.get("scope")
            node_id = payload.get("node_id")
            data = history.list_checkpoints(scope=scope, node_id=node_id)
            return MCPResult(
                success=True,
                data=_history_payload(
                    "success",
                    affected_nodes=[],
                    metadata={"count": len(data)},
                    extra={"checkpoints": data},
                ),
            )

        @self._register("history.summarize_actions")
        def _(payload: dict) -> MCPResult:
            history = getattr(win, "history_manager", None)
            if history is None:
                return MCPResult(success=False, error="History manager not available.")
            count = int(payload.get("count", 10))
            between = payload.get("between")
            summary = history.summarize_actions(count=count, between=between)
            return MCPResult(
                success=True,
                data=_history_payload(
                    "success",
                    affected_nodes=[],
                    metadata={"summary": summary},
                ),
            )

        @self._register("history.timeline")
        def _(payload: dict) -> MCPResult:
            history = getattr(win, "history_manager", None)
            if history is None:
                return MCPResult(success=False, error="History manager not available.")
            snapshots, pointer = history.get_timeline()
            items = [
                {
                    "index": idx,
                    "scene": snap.scene,
                    "description": snap.description,
                    "timestamp": snap.timestamp.isoformat(),
                }
                for idx, snap in enumerate(snapshots)
            ]
            return MCPResult(
                success=True,
                data=_history_payload(
                    "success",
                    affected_nodes=[],
                    metadata={"pointer": pointer},
                    extra={"timeline": items},
                ),
            )

        @self._register("history.replay")
        def _(payload: dict) -> MCPResult:
            history = getattr(win, "history_manager", None)
            if history is None:
                return MCPResult(success=False, error="History manager not available.")
            direction = str(payload.get("direction", "redo")).lower()
            count = int(payload.get("count", 1))
            applied = 0
            last_diff = None
            for _ in range(max(0, count)):
                if direction == "undo":
                    last_diff = history.undo()
                else:
                    last_diff = history.redo()
                if not last_diff:
                    break
                applied += 1
            return MCPResult(
                success=True,
                data=_history_payload(
                    "success",
                    affected_nodes=last_diff.get("affected_nodes", [])
                    if last_diff
                    else [],
                    metadata={
                        "action": "history_replay",
                        "direction": direction,
                        "applied": applied,
                    },
                ),
            )

        @self._register("history.diff_between")
        def _(payload: dict) -> MCPResult:
            history = getattr(win, "history_manager", None)
            if history is None:
                return MCPResult(success=False, error="History manager not available.")
            a = payload.get("checkpoint_a", "")
            b = payload.get("checkpoint_b", "")
            if not a or not b:
                return MCPResult(
                    success=False,
                    error="checkpoint_a and checkpoint_b are required.",
                )
            diff = history.diff_between(a, b)
            if diff is None:
                return MCPResult(
                    success=False,
                    data=_history_payload("error"),
                    error="Diff failed (checkpoints not found).",
                )
            return MCPResult(
                success=True,
                data=_history_payload(
                    "success",
                    affected_nodes=diff.get("affected_nodes", []),
                    metadata=diff,
                ),
            )

        # ── Debugging ───────────────────────────────────────────────────────

        @self._register("get_action_log")
        def _(payload: dict) -> MCPResult:
            return MCPResult(success=True, data=self.get_action_log())

        @self._register("list_commands")
        def _(payload: dict) -> MCPResult:
            return MCPResult(success=True, data=self.list_commands())

        @self._register("ping")
        def _(payload: dict) -> MCPResult:
            return MCPResult(
                success=True, data={"message": "pong", "node_count": len(win.nodes)}
            )


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def _get_global_assets():
    """Safely retrieve the global ASSETS registry from the main module."""
    try:
        import __main__

        return __main__.ASSETS
    except AttributeError:
        # Fallback: walk QApplication windows to find main window
        from PySide6.QtWidgets import QApplication

        for widget in QApplication.topLevelWidgets():
            if hasattr(widget, "ASSETS"):
                return widget.ASSETS
        raise RuntimeError("Cannot locate global ASSETS registry.")
