from __future__ import annotations
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager
import copy
import hashlib
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from PySide6.QtCore import QObject, Signal


# ─────────────────────────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WireState:
    wire_id: str
    from_node: str
    to_node: str

    def to_dict(self) -> dict:
        return {"wire_id": self.wire_id, "from": self.from_node, "to": self.to_node}


@dataclass(frozen=True)
class NodeState:
    data: dict
    fingerprint: str


@dataclass(frozen=True)
class GraphSnapshot:
    scene: str
    nodes: Dict[str, NodeState]
    wires: Tuple[WireState, ...]
    description: str
    timestamp: datetime
    metadata: dict
    fingerprint: str


@dataclass
class NodeAction:
    node_id: str
    before: Optional[NodeState]
    after: Optional[NodeState]
    before_wires: Tuple[WireState, ...]
    after_wires: Tuple[WireState, ...]
    description: str
    timestamp: datetime

    def affected_nodes(self) -> List[str]:
        ids = {self.node_id}
        for w in self.before_wires + self.after_wires:
            ids.add(w.from_node)
            ids.add(w.to_node)
        return sorted(ids)


@dataclass
class SnapshotAction:
    before: GraphSnapshot
    after: GraphSnapshot
    description: str
    timestamp: datetime
    merge_key: Optional[str] = None
    node_actions: Dict[str, NodeAction] = field(default_factory=dict)
    record_node_history: bool = True

    def can_merge(self, other: "SnapshotAction", window_ms: int) -> bool:
        if not self.merge_key or not other.merge_key:
            return False
        if self.merge_key != other.merge_key:
            return False
        delta = abs(int((other.timestamp - self.timestamp).total_seconds() * 1000.0))
        return delta <= window_ms


@dataclass
class NodeHistory:
    undo_stack: List[NodeAction] = field(default_factory=list)
    redo_stack: List[NodeAction] = field(default_factory=list)

    def push(self, action: NodeAction) -> None:
        self.undo_stack.append(action)
        self.redo_stack.clear()

    def can_undo(self) -> bool:
        return bool(self.undo_stack)

    def can_redo(self) -> bool:
        return bool(self.redo_stack)

    def pop_undo(self) -> Optional[NodeAction]:
        if not self.undo_stack:
            return None
        act = self.undo_stack.pop()
        self.redo_stack.append(act)
        return act

    def pop_redo(self) -> Optional[NodeAction]:
        if not self.redo_stack:
            return None
        act = self.redo_stack.pop()
        self.undo_stack.append(act)
        return act

    def undo_peek(self) -> Optional[NodeAction]:
        return self.undo_stack[-1] if self.undo_stack else None

    def redo_peek(self) -> Optional[NodeAction]:
        return self.redo_stack[-1] if self.redo_stack else None


@dataclass
class Checkpoint:
    name: str
    scope: str  # "project" | "scene" | "node"
    node_id: Optional[str]
    snapshot: Optional[GraphSnapshot]
    node_state: Optional[NodeState]
    node_wires: Tuple[WireState, ...]
    timestamp: datetime
    description: str
    metadata: dict

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "scope": self.scope,
            "node_id": self.node_id,
            "timestamp": self.timestamp.isoformat(),
            "description": self.description,
            "metadata": dict(self.metadata),
        }


# ─────────────────────────────────────────────────────────────────────────────
# History Manager
# ─────────────────────────────────────────────────────────────────────────────


class HistoryManager(QObject):
    history_changed = Signal()
    state_changed = Signal(bool, bool)
    snapshot_applied = Signal(object)
    diff_available = Signal(object)
    checkpoint_created = Signal(object)

    def __init__(
        self,
        snapshot_provider: Callable[[], Tuple[Dict[str, dict], List[WireState]]],
        apply_snapshot: Callable[[GraphSnapshot], None],
        apply_node_state: Callable[[str, Optional[dict], List[WireState]], None],
        scene_name_provider: Optional[Callable[[], str]] = None,
        max_history: Optional[int] = None,
        merge_window_ms: int = 750,
    ) -> None:
        super().__init__()
        self._snapshot_provider = snapshot_provider
        self._apply_snapshot_cb = apply_snapshot
        self._apply_node_state_cb = apply_node_state
        self._scene_name_provider = scene_name_provider or (lambda: "Scene 1")
        self._max_history = max_history
        self._merge_window_ms = merge_window_ms

        self._undo_stack: List[SnapshotAction] = []
        self._redo_stack: List[SnapshotAction] = []
        self._current_snapshot: Optional[GraphSnapshot] = None
        self._root_snapshot: Optional[GraphSnapshot] = None

        self._node_histories: Dict[str, NodeHistory] = {}
        self._checkpoints: Dict[str, Checkpoint] = {}

        self._group_depth = 0
        self._group_dirty = False
        self._group_description: str = "Grouped Action"
        self._group_metadata: dict = {}
        self._group_merge_key: Optional[str] = None

        self._suspend_count = 0
        self._is_restoring = False

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def is_restoring(self) -> bool:
        return self._is_restoring

    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    def history_pointer(self) -> int:
        return len(self._undo_stack)

    # ── Snapshot Helpers ───────────────────────────────────────────────────

    def _freeze(self, value):
        if isinstance(value, dict):
            return tuple(
                (str(k), self._freeze(v))
                for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
            )
        if isinstance(value, (list, tuple)):
            return tuple(self._freeze(v) for v in value)
        if isinstance(value, set):
            return tuple(sorted(self._freeze(v) for v in value))
        if isinstance(value, (int, float, str, bool)) or value is None:
            return value
        return repr(value)

    def _hash_obj(self, obj) -> str:
        frozen = self._freeze(obj)
        return hashlib.sha256(repr(frozen).encode("utf-8")).hexdigest()

    def _build_snapshot(
        self, description: str, metadata: Optional[dict] = None
    ) -> GraphSnapshot:
        nodes_raw, wires_raw = self._snapshot_provider()
        scene_name = self._scene_name_provider()
        ts = datetime.now()
        meta = dict(metadata or {})
        meta.setdefault("description", description)

        current_nodes = self._current_snapshot.nodes if self._current_snapshot else {}

        nodes: Dict[str, NodeState] = {}
        for nid, data in nodes_raw.items():
            fingerprint = self._hash_obj(data)
            if nid in current_nodes and current_nodes[nid].fingerprint == fingerprint:
                nodes[nid] = current_nodes[nid]
            else:
                nodes[nid] = NodeState(
                    data=copy.deepcopy(data), fingerprint=fingerprint
                )

        wires_sorted = tuple(sorted(wires_raw, key=lambda w: w.wire_id))
        wire_key = tuple((w.wire_id, w.from_node, w.to_node) for w in wires_sorted)
        node_key = tuple(sorted((nid, n.fingerprint) for nid, n in nodes.items()))
        snapshot_fingerprint = self._hash_obj((scene_name, node_key, wire_key))

        return GraphSnapshot(
            scene=scene_name,
            nodes=nodes,
            wires=wires_sorted,
            description=description,
            timestamp=ts,
            metadata=meta,
            fingerprint=snapshot_fingerprint,
        )

    def _wires_for_node(
        self, snapshot: GraphSnapshot, node_id: str
    ) -> Tuple[WireState, ...]:
        return tuple(
            w for w in snapshot.wires if w.from_node == node_id or w.to_node == node_id
        )

    def _diff_snapshots(self, before: GraphSnapshot, after: GraphSnapshot) -> dict:
        before_nodes = set(before.nodes.keys())
        after_nodes = set(after.nodes.keys())

        added_nodes = sorted(after_nodes - before_nodes)
        removed_nodes = sorted(before_nodes - after_nodes)

        changed_nodes = sorted(
            nid
            for nid in (before_nodes & after_nodes)
            if before.nodes[nid].fingerprint != after.nodes[nid].fingerprint
        )

        before_wires = {w.wire_id: w for w in before.wires}
        after_wires = {w.wire_id: w for w in after.wires}

        added_wires = [after_wires[wid] for wid in after_wires.keys() - before_wires]
        removed_wires = [before_wires[wid] for wid in before_wires.keys() - after_wires]

        affected_nodes = set(added_nodes + removed_nodes + changed_nodes)
        for w in added_wires + removed_wires:
            affected_nodes.add(w.from_node)
            affected_nodes.add(w.to_node)

        return {
            "added_nodes": added_nodes,
            "removed_nodes": removed_nodes,
            "changed_nodes": changed_nodes,
            "added_wires": [w.to_dict() for w in added_wires],
            "removed_wires": [w.to_dict() for w in removed_wires],
            "affected_nodes": sorted(affected_nodes),
        }

    def _build_node_actions(
        self, before: GraphSnapshot, after: GraphSnapshot, description: str
    ) -> Dict[str, NodeAction]:
        diff = self._diff_snapshots(before, after)
        affected = set(diff["affected_nodes"])

        actions: Dict[str, NodeAction] = {}
        for nid in affected:
            actions[nid] = NodeAction(
                node_id=nid,
                before=before.nodes.get(nid),
                after=after.nodes.get(nid),
                before_wires=self._wires_for_node(before, nid),
                after_wires=self._wires_for_node(after, nid),
                description=description,
                timestamp=after.timestamp,
            )
        return actions

    # ── Public API ─────────────────────────────────────────────────────────

    def reset(
        self,
        description: str = "Initial",
        metadata: Optional[dict] = None,
        clear_checkpoints: bool = True,
    ) -> None:
        snapshot = self._build_snapshot(description, metadata)
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._node_histories.clear()
        self._current_snapshot = snapshot
        self._root_snapshot = snapshot
        if clear_checkpoints:
            self._checkpoints.clear()
        self.history_changed.emit()
        self.state_changed.emit(False, False)

    def capture(
        self,
        description: str,
        merge_key: Optional[str] = None,
        metadata: Optional[dict] = None,
        record_node_history: bool = True,
    ) -> bool:
        if self._is_restoring or self._suspend_count > 0:
            return False
        if self._group_depth > 0:
            self._group_dirty = True
            # Store the first description in the group, ignore later
            if not self._group_description:
                self._group_description = description
            if metadata:
                self._group_metadata.update(metadata)
            if merge_key:
                self._group_merge_key = merge_key
            return False

        if self._current_snapshot is None:
            self.reset(description=description, metadata=metadata)
            return True

        snapshot = self._build_snapshot(description, metadata)
        if snapshot.fingerprint == self._current_snapshot.fingerprint:
            return False

        action = SnapshotAction(
            before=self._current_snapshot,
            after=snapshot,
            description=description,
            timestamp=snapshot.timestamp,
            merge_key=merge_key,
            record_node_history=record_node_history,
        )
        action.node_actions = self._build_node_actions(
            action.before, action.after, description
        )

        if (
            merge_key
            and self._undo_stack
            and self._undo_stack[-1].can_merge(action, self._merge_window_ms)
        ):
            last = self._undo_stack[-1]
            old_node_actions = dict(last.node_actions)
            last.after = snapshot
            last.timestamp = snapshot.timestamp
            last.node_actions = self._build_node_actions(
                last.before,
                last.after,
                last.description,
            )
            if record_node_history:
                for nid, nact in last.node_actions.items():
                    hist = self._node_histories.get(nid)
                    if hist and hist.undo_peek() is old_node_actions.get(nid):
                        hist.undo_stack[-1] = nact
        else:
            self._undo_stack.append(action)
            if (
                self._max_history is not None
                and len(self._undo_stack) > self._max_history
            ):
                self._undo_stack.pop(0)

        self._redo_stack.clear()
        self._current_snapshot = snapshot

        if record_node_history:
            for nid, nact in action.node_actions.items():
                self._node_histories.setdefault(nid, NodeHistory()).push(nact)

        self.history_changed.emit()
        self.state_changed.emit(self.can_undo(), self.can_redo())
        return True

    def begin_group(
        self,
        description: str,
        metadata: Optional[dict] = None,
        merge_key: Optional[str] = None,
    ) -> None:
        if self._group_depth == 0:
            self._group_description = description
            self._group_metadata = dict(metadata or {})
            self._group_merge_key = merge_key
            self._group_dirty = False
        self._group_depth += 1

    def end_group(self) -> bool:
        if self._group_depth == 0:
            return False
        self._group_depth -= 1
        if self._group_depth > 0:
            return False
        if not self._group_dirty:
            return False
        return self.capture(
            self._group_description,
            merge_key=self._group_merge_key,
            metadata=self._group_metadata,
        )

    def update_group_metadata(self, updates: dict) -> None:
        if self._group_depth > 0:
            self._group_metadata.update(updates)

    def mark_group_dirty(self) -> None:
        if self._group_depth > 0:
            self._group_dirty = True

    @contextmanager
    def group(
        self,
        description: str,
        metadata: Optional[dict] = None,
        merge_key: Optional[str] = None,
    ):
        self.begin_group(description, metadata=metadata, merge_key=merge_key)
        try:
            yield
        finally:
            self.end_group()

    @contextmanager
    def suspend(self):
        self._suspend_count += 1
        try:
            yield
        finally:
            self._suspend_count = max(0, self._suspend_count - 1)

    # ── Undo / Redo (Project/Scene) ─────────────────────────────────────────

    def undo(self) -> Optional[dict]:
        if not self._undo_stack:
            return None
        action = self._undo_stack.pop()
        self._redo_stack.append(action)

        self._apply_snapshot(action.before)
        self._current_snapshot = action.before

        if action.record_node_history:
            for nid, nact in action.node_actions.items():
                history = self._node_histories.get(nid)
                if history and history.undo_peek() is nact:
                    history.pop_undo()

        diff = self._diff_snapshots(action.after, action.before)
        self.history_changed.emit()
        self.state_changed.emit(self.can_undo(), self.can_redo())
        self.diff_available.emit(diff)
        return diff

    def redo(self) -> Optional[dict]:
        if not self._redo_stack:
            return None
        action = self._redo_stack.pop()
        self._undo_stack.append(action)

        self._apply_snapshot(action.after)
        self._current_snapshot = action.after

        if action.record_node_history:
            for nid, nact in action.node_actions.items():
                history = self._node_histories.get(nid)
                if history and history.redo_peek() is nact:
                    history.pop_redo()

        diff = self._diff_snapshots(action.before, action.after)
        self.history_changed.emit()
        self.state_changed.emit(self.can_undo(), self.can_redo())
        self.diff_available.emit(diff)
        return diff

    # ── Node-level Undo / Redo ─────────────────────────────────────────────

    def undo_node(self, node_id: str) -> Optional[dict]:
        history = self._node_histories.get(node_id)
        if not history or not history.can_undo():
            return None
        action = history.pop_undo()
        if action is None:
            return None
        self._apply_node_state(action.node_id, action.before, action.before_wires)
        # Record as a global action, but suppress node history (it already moved)
        self.capture(
            f"Undo Node {node_id[:6]}",
            merge_key=f"node:{node_id}",
            metadata={"node_id": node_id},
            record_node_history=False,
        )
        affected = action.affected_nodes()
        return {"affected_nodes": affected}

    def redo_node(self, node_id: str) -> Optional[dict]:
        history = self._node_histories.get(node_id)
        if not history or not history.can_redo():
            return None
        action = history.pop_redo()
        if action is None:
            return None
        self._apply_node_state(action.node_id, action.after, action.after_wires)
        self.capture(
            f"Redo Node {node_id[:6]}",
            merge_key=f"node:{node_id}",
            metadata={"node_id": node_id},
            record_node_history=False,
        )
        affected = action.affected_nodes()
        return {"affected_nodes": affected}

    # ── Checkpoints ─────────────────────────────────────────────────────────

    def create_checkpoint(
        self,
        name: str,
        scope: str = "scene",
        node_id: Optional[str] = None,
        description: str = "",
        metadata: Optional[dict] = None,
    ) -> Checkpoint:
        if not name:
            raise ValueError("Checkpoint name cannot be empty.")
        scope = scope.lower()
        meta = dict(metadata or {})
        meta.setdefault("description", description or f"Checkpoint {name}")
        meta.setdefault("timestamp", datetime.now().isoformat())

        if scope not in ("project", "scene", "node"):
            raise ValueError("scope must be one of: project, scene, node")

        snapshot = None
        node_state = None
        node_wires: Tuple[WireState, ...] = tuple()

        if scope in ("project", "scene"):
            snapshot = self._build_snapshot(meta["description"], meta)
            affected_nodes = sorted(snapshot.nodes.keys())
        else:
            if not node_id:
                raise ValueError("node_id is required for node checkpoints.")
            if self._current_snapshot is None:
                raise ValueError("No current snapshot to checkpoint.")
            node_state = self._current_snapshot.nodes.get(node_id)
            node_wires = self._wires_for_node(self._current_snapshot, node_id)
            affected_nodes = [node_id]

        meta.setdefault("affected_nodes", affected_nodes)

        cp = Checkpoint(
            name=name,
            scope=scope,
            node_id=node_id,
            snapshot=snapshot,
            node_state=node_state,
            node_wires=node_wires,
            timestamp=datetime.now(),
            description=meta["description"],
            metadata=meta,
        )
        self._checkpoints[name] = cp
        try:
            self.checkpoint_created.emit(cp)
        except Exception:
            pass
        return cp

    def restore_checkpoint(self, name: str) -> Optional[dict]:
        cp = self._checkpoints.get(name)
        if not cp:
            return None
        if cp.scope in ("project", "scene"):
            if cp.snapshot is None:
                return None
            if self._current_snapshot and (
                cp.snapshot.fingerprint == self._current_snapshot.fingerprint
            ):
                return {"affected_nodes": []}
            before = self._current_snapshot
            self._apply_snapshot(cp.snapshot)
            # Record as a new action explicitly
            if before is not None:
                action = SnapshotAction(
                    before=before,
                    after=cp.snapshot,
                    description=f"Restore Checkpoint {name}",
                    timestamp=datetime.now(),
                    merge_key=None,
                    record_node_history=True,
                )
                action.node_actions = self._build_node_actions(
                    action.before, action.after, action.description
                )
                self._undo_stack.append(action)
                self._redo_stack.clear()
                self._current_snapshot = cp.snapshot
                for nid, nact in action.node_actions.items():
                    self._node_histories.setdefault(nid, NodeHistory()).push(nact)
                self.history_changed.emit()
                self.state_changed.emit(self.can_undo(), self.can_redo())
                diff = self._diff_snapshots(action.before, action.after)
                self.diff_available.emit(diff)
                return {"affected_nodes": diff.get("affected_nodes", [])}
            self._current_snapshot = cp.snapshot
            return {"affected_nodes": sorted(cp.snapshot.nodes.keys())}
        # Node scope
        if cp.node_id is None:
            return None
        self._apply_node_state(cp.node_id, cp.node_state, cp.node_wires)
        self.capture(
            f"Restore Node Checkpoint {name}",
            merge_key=f"checkpoint:{cp.node_id}",
            metadata={"checkpoint": name, "node_id": cp.node_id, "scope": cp.scope},
            record_node_history=False,
        )
        affected = [cp.node_id]
        for w in cp.node_wires:
            affected.extend([w.from_node, w.to_node])
        return {"affected_nodes": sorted(set(affected))}

    def list_checkpoints(
        self, scope: Optional[str] = None, node_id: Optional[str] = None
    ) -> List[dict]:
        result = []
        for cp in self._checkpoints.values():
            if scope and cp.scope != scope:
                continue
            if node_id and cp.node_id != node_id:
                continue
            result.append(cp.to_dict())
        return sorted(result, key=lambda d: d.get("timestamp", ""))

    def diff_between(self, a: str, b: str) -> Optional[dict]:
        cp_a = self._checkpoints.get(a)
        cp_b = self._checkpoints.get(b)
        if not cp_a or not cp_b:
            return None
        if cp_a.snapshot and cp_b.snapshot:
            return self._diff_snapshots(cp_a.snapshot, cp_b.snapshot)
        if cp_a.node_state or cp_b.node_state:
            # Node-level diff
            data_a = cp_a.node_state.data if cp_a.node_state else {}
            data_b = cp_b.node_state.data if cp_b.node_state else {}
            affected = set()
            if cp_a.node_id:
                affected.add(cp_a.node_id)
            if cp_b.node_id:
                affected.add(cp_b.node_id)
            for w in cp_a.node_wires + cp_b.node_wires:
                affected.add(w.from_node)
                affected.add(w.to_node)
            return {
                "node_id": cp_a.node_id or cp_b.node_id,
                "data_changed": data_a != data_b,
                "wires_a": [w.to_dict() for w in cp_a.node_wires],
                "wires_b": [w.to_dict() for w in cp_b.node_wires],
                "affected_nodes": sorted(affected),
            }
        return None

    # ── Action Summary ─────────────────────────────────────────────────────

    def summarize_actions(
        self, count: int = 10, between: Optional[dict] = None
    ) -> dict:
        timeline = self._undo_stack + list(reversed(self._redo_stack))
        total = len(timeline)
        if between and isinstance(between, dict):
            start = max(0, int(between.get("start", 0)))
            end = min(total, int(between.get("end", total)))
            items = timeline[start:end]
            offset = start
        else:
            items = timeline[max(0, total - count) :]
            offset = max(0, total - count)

        summaries = []
        for idx, act in enumerate(items, start=offset):
            diff = self._diff_snapshots(act.before, act.after)
            summaries.append(
                {
                    "index": idx,
                    "description": act.description,
                    "timestamp": act.timestamp.isoformat(),
                    "affected_nodes": diff.get("affected_nodes", []),
                }
            )

        return {
            "total": total,
            "pointer": self.history_pointer(),
            "actions": summaries,
        }

    # ── Timeline (optional) ─────────────────────────────────────────────────

    def get_timeline(self) -> Tuple[List[GraphSnapshot], int]:
        if self._root_snapshot is None:
            return ([], 0)
        snapshots = [self._root_snapshot]
        snapshots += [a.after for a in self._undo_stack]
        snapshots += [a.after for a in reversed(self._redo_stack)]
        return snapshots, self.history_pointer()

    # ── Internal Apply Helpers ─────────────────────────────────────────────

    def _apply_snapshot(self, snapshot: GraphSnapshot) -> None:
        self._is_restoring = True
        try:
            self._apply_snapshot_cb(snapshot)
        finally:
            self._is_restoring = False
        self.snapshot_applied.emit(snapshot)

    def _apply_node_state(
        self, node_id: str, state: Optional[NodeState], wires: Iterable[WireState]
    ) -> None:
        self._is_restoring = True
        try:
            payload = state.data if state else None
            self._apply_node_state_cb(node_id, payload, list(wires))
        finally:
            self._is_restoring = False
        self.snapshot_applied.emit(self._current_snapshot)
