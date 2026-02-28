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

from __future__ import annotations

import json
import logging
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Optional

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
    Executes MCP commands against the live EfficientManimWindow.

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

    def execute_batch(self, commands: list[dict]) -> list[MCPResult]:
        """Execute a list of {command, payload} dicts sequentially."""
        results = []
        for item in commands:
            cmd = item.get("command", "")
            payload = item.get("payload", {})
            results.append(self.execute(cmd, payload))
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

        @self._register("list_assets")
        def _(payload: dict) -> MCPResult:
            try:
                assets = _get_global_assets().get_list()
                data = [
                    {"id": a.id, "name": a.name, "kind": a.kind, "path": a.current_path}
                    for a in assets
                ]
                return MCPResult(success=True, data=data)
            except Exception as e:
                return MCPResult(success=False, error=str(e))

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
            if node_type_str not in ("ANIMATION", "MOBJECT"):
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
                    data={
                        "id": actual_id,
                        "name": item.data.name,
                        "cls_name": item.data.cls_name,
                    },
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
            return MCPResult(
                success=True,
                data={"node_id": node_id, "key": param_key, "value": param_value},
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
            return MCPResult(
                success=True, data={"node_id": node_id, "new_name": new_name}
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
                return MCPResult(success=True, data={"deleted_id": node_id})
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
                win.new_project()
                return MCPResult(success=True, data={"message": "Scene cleared."})
            except Exception as e:
                return MCPResult(success=False, error=str(e))

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
                    win.render_to_video()
                    return MCPResult(
                        success=True, data={"message": "Render triggered."}
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
                from PySide6.QtWidgets import QApplication

                code = win.code_view.toPlainText()
                QApplication.clipboard().setText(code)
                return MCPResult(success=True, data={"chars_copied": len(code)})
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        # ── Project state ───────────────────────────────────────────────────

        @self._register("save_project")
        def _(payload: dict) -> MCPResult:
            try:
                win.save_project()
                return MCPResult(success=True, data={"path": str(win.project_path)})
            except Exception as e:
                return MCPResult(success=False, error=str(e))

        @self._register("switch_scene")
        def _(payload: dict) -> MCPResult:
            scene_name = payload.get("scene_name", "")
            if not scene_name:
                return MCPResult(success=False, error="scene_name is required.")
            if scene_name not in win._all_scenes:
                return MCPResult(
                    success=False,
                    error=f"Scene not found: {scene_name!r}. Available: {list(win._all_scenes.keys())}",
                )
            try:
                win._switch_scene(scene_name)
                return MCPResult(success=True, data={"scene": scene_name})
            except Exception as e:
                return MCPResult(success=False, error=str(e))

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
