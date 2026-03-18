"""
collab/delta.py — Delta construction and application for collaboration.

IMPORTANT: serialize_graph() accesses Qt objects and MUST be called from the
Qt main thread only. apply_delta() is always called via QTimer.singleShot(0, ...)
so it also runs on the main thread.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

LOGGER = logging.getLogger("collab.delta")


# ──────────────────────────────────────────────────────────────────────────────
# Delta construction
# ──────────────────────────────────────────────────────────────────────────────

def make_delta(
    action: str,
    payload: dict,
    session_pin: str,
    sender_id: str,
) -> dict:
    """Build a well-formed delta message."""
    return {
        "type": "delta",
        "session_pin": session_pin,
        "sender_id": sender_id,
        "timestamp": time.time(),
        "action": action,
        "payload": payload,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Graph serialization  (MAIN THREAD ONLY)
# ──────────────────────────────────────────────────────────────────────────────

def serialize_graph(window) -> dict:
    """
    Serialize the full node graph to a plain dict.

    Must be called from the Qt main thread.
    Returns {"nodes": [...], "wires": [...]}.
    """
    nodes = []
    for node in list(window.nodes.values()):
        try:
            nodes.append(node.data.to_dict())
        except Exception as exc:
            LOGGER.warning(f"Could not serialize node: {exc}")

    wires = []
    for item in list(window.scene.items()):
        if item.__class__.__name__ != "WireItem":
            continue
        try:
            wires.append({
                "wire_id": getattr(item, "wire_id", None),
                "from_node": item.start_socket.parentItem().data.id,
                "from_port": "out",
                "to_node": item.end_socket.parentItem().data.id,
                "to_port": "in",
            })
        except Exception as exc:
            LOGGER.debug(f"Could not serialize wire: {exc}")

    return {"nodes": nodes, "wires": wires}


# ──────────────────────────────────────────────────────────────────────────────
# Delta application  (MAIN THREAD ONLY — called via QTimer.singleShot)
# ──────────────────────────────────────────────────────────────────────────────

def apply_delta(window, delta: Dict[str, Any]) -> None:
    """
    Apply a single incoming delta to the local graph.
    Must be called from the Qt main thread.
    window._collab_applying must be True when this is called (guards against re-emit).
    """
    action = delta.get("action")
    payload = delta.get("payload") or {}

    try:
        _HANDLERS[action](window, payload)
    except KeyError:
        LOGGER.debug(f"Unknown delta action: '{action}'")
    except Exception as exc:
        LOGGER.warning(f"apply_delta('{action}') error: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# Individual action handlers
# ──────────────────────────────────────────────────────────────────────────────

def _handle_full_graph_sync(window, payload: dict) -> None:
    graph_json = payload.get("graph_json")
    if graph_json:
        window.load_graph_from_json(graph_json)


def _handle_node_added(window, payload: dict) -> None:
    node_id = payload.get("node_id")
    if not node_id:
        return
    if node_id in window.nodes:
        # Already present — update position/params instead
        _handle_node_moved(window, payload)
        return

    node_type = payload.get("node_type", "MOBJECT")
    cls_name = payload.get("cls_name") or payload.get("node_type_class", "")
    name = payload.get("name") or cls_name
    x = float(payload.get("x", 0))
    y = float(payload.get("y", 0))
    params = dict(payload.get("params") or payload.get("properties") or {})

    item = window.add_node(
        node_type, cls_name, params=params, pos=(x, y), nid=node_id, name=name
    )
    # Restore optional metadata
    try:
        item.data.param_metadata = dict(payload.get("param_metadata") or {})
        item.data.audio_asset_id = payload.get("audio_asset_id")
        item.data.voiceover_transcript = payload.get("voiceover_transcript")
        item.data.voiceover_duration = float(payload.get("voiceover_duration") or 0.0)
        item.data.is_ai_generated = bool(payload.get("is_ai_generated", False))
    except Exception:
        pass


def _handle_node_deleted(window, payload: dict) -> None:
    node_id = payload.get("node_id")
    if node_id and node_id in window.nodes:
        window.remove_node(window.nodes[node_id])


def _handle_node_moved(window, payload: dict) -> None:
    node_id = payload.get("node_id")
    if not node_id or node_id not in window.nodes:
        return
    node = window.nodes[node_id]
    x = float(payload.get("x", node.x()))
    y = float(payload.get("y", node.y()))
    # Suppress position-change re-emit while we set position programmatically
    node.setPos(x, y)
    node.data.pos_x = x
    node.data.pos_y = y
    for wire in node.in_socket.links + node.out_socket.links:
        try:
            wire.update_path()
        except Exception:
            pass


def _handle_node_property_changed(window, payload: dict) -> None:
    node_id = payload.get("node_id")
    if not node_id or node_id not in window.nodes:
        return
    key = payload.get("property_key")
    if not key:
        return
    node = window.nodes[node_id]
    new_value = payload.get("new_value")

    # Name is a special field stored on node.data.name, not in params
    if key in ("__name__", "_name", "name"):
        node.data.name = str(new_value) if new_value is not None else node.data.name
    else:
        node.data.params[key] = new_value

    node.update()


def _handle_wire_added(window, payload: dict) -> None:
    wire_id = payload.get("wire_id")
    from_node = payload.get("from_node")
    to_node = payload.get("to_node")
    if not from_node or not to_node:
        return
    window.add_wire_by_ids(from_node, to_node, wire_id=wire_id)


def _handle_wire_deleted(window, payload: dict) -> None:
    wire_id = payload.get("wire_id")
    if wire_id:
        window.remove_wire_by_id(wire_id)


def _handle_scene_switched(window, payload: dict) -> None:
    scene_name = payload.get("scene_name") or payload.get("scene_id")
    if scene_name:
        window._on_scene_switch(scene_name)


def _handle_vgroup_created(window, payload: dict) -> None:
    name = payload.get("vgroup_id") or payload.get("name")
    members = payload.get("member_node_ids") or []
    if name and hasattr(window, "panel_vgroup"):
        window.panel_vgroup.add_group_from_collab(name, members)


def _handle_vgroup_deleted(window, payload: dict) -> None:
    name = payload.get("vgroup_id") or payload.get("name")
    if name and hasattr(window, "panel_vgroup"):
        window.panel_vgroup.delete_group_from_collab(name)


def _handle_node_lock(window, payload: dict) -> None:
    node_id = payload.get("node_id")
    if node_id:
        window.collab_lock_node(
            node_id,
            payload.get("owner_id"),
            payload.get("owner_name"),
            payload.get("owner_color"),
            payload.get("expires_at"),
        )


def _handle_node_unlock(window, payload: dict) -> None:
    node_id = payload.get("node_id")
    if node_id:
        window.collab_unlock_node(node_id)


# Action → handler dispatch table
_HANDLERS = {
    "full_graph_sync":         _handle_full_graph_sync,
    "node_added":              _handle_node_added,
    "node_deleted":            _handle_node_deleted,
    "node_moved":              _handle_node_moved,
    "node_property_changed":   _handle_node_property_changed,
    "wire_added":              _handle_wire_added,
    "wire_deleted":            _handle_wire_deleted,
    "scene_switched":          _handle_scene_switched,
    "vgroup_created":          _handle_vgroup_created,
    "vgroup_deleted":          _handle_vgroup_deleted,
    "node_lock":               _handle_node_lock,
    "node_unlock":             _handle_node_unlock,
}
