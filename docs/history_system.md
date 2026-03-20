# History System and MCP Commands

This document describes the new history, undo/redo, and checkpoint system, plus the MCP commands that expose it to AI agents.

**Overview**
- Full node-graph history with multi-level undo/redo.
- Atomic grouping for multi-step operations (AI merges, batch edits).
- Named checkpoints (project/scene/node scope).
- Per-node undo/redo using node-specific timelines.
- Diff and summary utilities for inspection.

**Core Classes**
- `HistoryManager` in `core/history_manager.py` manages snapshots, actions, and checkpoints.
- `GraphSnapshot`, `NodeState`, `WireState` model immutable history data.
- `NodeAction` and `SnapshotAction` record change intent.

**Hooking New Actions**
Call `history_manager.capture()` after any graph mutation.

Example:
```python
self.history_manager.capture(
    "Edit Circle.color",
    merge_key=f"prop:{node_id}:color",
    metadata={"node_id": node_id, "key": "color"}
)
```

Use grouping for atomic multi-step operations:
```python
self.history_manager.begin_group("AI Merge", metadata={"ai_prompt": prompt})
# ... create many nodes/wires ...
self.history_manager.end_group()
```

Suspend history during bulk loads:
```python
with self.history_manager.suspend():
    self.load_graph_from_json(payload)
```

**MCP Commands**
All history commands return a structured JSON payload with:
- `status`: `success` or `error`
- `affected_nodes`: list of node IDs
- `history_pointer`: current pointer index
- `metadata`: action or checkpoint metadata

Commands:
- `history.undo_project`
- `history.redo_project`
- `history.undo_scene` (payload: `{"scene_name": "Scene 1"}` — optional)
- `history.redo_scene` (payload: `{"scene_name": "Scene 1"}` — optional)
- `history.undo_node` (payload: `{"node_id": "..."}`)
- `history.redo_node` (payload: `{"node_id": "..."}`)
- `history.create_checkpoint` (payload: `{"name": "...", "scope": "scene|project|node", "node_id": "...", "description": "...", "metadata": {...}}`)
- `history.restore_checkpoint` (payload: `{"checkpoint_name": "..."}`)
- `history.list_checkpoints` (payload: `{"scope": "project|scene|node", "node_id": "..."}`)
- `history.summarize_actions` (payload: `{"count": 10, "between": {"start": 0, "end": 5}}`)
- `history.diff_between` (payload: `{"checkpoint_a": "...", "checkpoint_b": "..."}`)
- `history.timeline` (payload: `{}`)
- `history.replay` (payload: `{"direction": "undo|redo", "count": 3}`)

**Example MCP Calls**
Undo last project action:
```json
{"command": "history.undo_project", "payload": {}}
```

Undo a specific node action:
```json
{"command": "history.undo_node", "payload": {"node_id": "abc-123"}}
```

Restore a checkpoint:
```json
{"command": "history.restore_checkpoint", "payload": {"checkpoint_name": "Set circle color to aqua"}}
```

Summarize the last 5 actions:
```json
{"command": "history.summarize_actions", "payload": {"count": 5}}
```

**Notes**
- Project and scene scope currently operate on the active scene graph.
- Checkpoints include metadata such as `description`, `timestamp`, and `affected_nodes`.
- Per-node undo/redo is recorded as separate global actions to keep project history consistent.
