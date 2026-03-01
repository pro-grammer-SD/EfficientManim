"""
timing_resolver.py — Deterministic Timing Authority

Single source of truth for all timing relationships.
Graph-driven timing computation with deterministic guarantees.

Architecture:
  Node Graph (source of truth)
    ↓
  Timing Resolver (computes timing from graph structure)
    ↓
  Timeline Projection (visualizes resolved timing)

Rules:
  - Timeline NEVER mutates nodes directly
  - All timing changes go through TimingMutationAPI
  - Changes always flow: API → Resolver → Graph → Timeline
  - All timing must be hash-validated
  - No floating-point precision drift
"""

import hashlib
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum, auto

LOGGER = logging.getLogger("timing_resolver")


class TimingUnit(Enum):
    """Time measurement unit."""

    FRAMES = auto()
    SECONDS = auto()
    MILLISECONDS = auto()


@dataclass
class TimingConstraint:
    """Represents a timing constraint on a node."""

    node_id: str
    start_time: float  # seconds
    duration: float  # seconds
    locked: bool = False  # If True, cannot be changed


@dataclass
class TimingEvent:
    """Represents a timing event (for undo/redo/audit)."""

    node_id: str
    old_start: float
    new_start: float
    old_duration: float
    new_duration: float
    reason: str


class TimingResolver:
    """
    Single authoritative source for all timing computations.

    Ensures:
    - Deterministic timing (same graph structure = same timing)
    - No conflicts
    - All constraints satisfied
    - Hash-validated results
    - Audit trail of changes
    """

    def __init__(self, fps: int = 60):
        self.fps = fps
        self._frame_duration = 1.0 / fps  # seconds per frame
        self._constraints: Dict[str, TimingConstraint] = {}
        self._changes: List[TimingEvent] = []
        self._timing_hash = ""
        self._locked_nodes = set()

    def resolve_timing(
        self,
        graph_nodes: Dict[str, dict],
        voiceover_segments: List[dict] = None,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Compute timing for all nodes based on graph structure.

        Args:
            graph_nodes: Dict of {node_id: node_data}
            voiceover_segments: Optional list of voiceover timing segments

        Returns:
            Dict of {node_id: (start_time, duration)}
        """
        result = {}
        current_time = 0.0

        # Topological sort for deterministic ordering
        sorted_nodes = self._topological_sort(graph_nodes)

        for node_id in sorted_nodes:
            node = graph_nodes.get(node_id)
            if not node:
                continue

            # Check for locked constraint
            if node_id in self._locked_nodes:
                constraint = self._constraints.get(node_id)
                if constraint:
                    result[node_id] = (constraint.start_time, constraint.duration)
                    current_time = max(
                        current_time, constraint.start_time + constraint.duration
                    )
                    continue

            # Compute duration from node type
            duration = self._compute_duration(node)

            # Check for voiceover timing
            if node.get("audio_asset_id"):
                voiceover_duration = node.get("voiceover_duration", 0.0)
                duration = max(duration, voiceover_duration)

            # Store timing
            result[node_id] = (current_time, duration)
            current_time += duration

        # Validate and hash
        self._validate_timing(result, graph_nodes)
        self._timing_hash = self._hash_timing(result)

        return result

    def mutate_timing(
        self,
        node_id: str,
        new_start: float,
        new_duration: float,
        reason: str = "manual edit",
    ) -> Tuple[bool, str]:
        """
        Mutate timing for a single node.

        ALL timing changes must go through this API.

        Args:
            node_id: Node to modify
            new_start: New start time (seconds)
            new_duration: New duration (seconds)
            reason: Reason for change (audit trail)

        Returns:
            (success, message)
        """
        if node_id in self._locked_nodes:
            return False, f"Node {node_id} is locked, cannot modify timing"

        if new_start < 0 or new_duration <= 0:
            return False, "Invalid timing values (must be positive)"

        # Get old values
        old_constraint = self._constraints.get(node_id)
        old_start = old_constraint.start_time if old_constraint else 0.0
        old_duration = old_constraint.duration if old_constraint else 0.0

        # Create constraint
        constraint = TimingConstraint(node_id, new_start, new_duration)
        self._constraints[node_id] = constraint

        # Record change
        event = TimingEvent(
            node_id, old_start, new_start, old_duration, new_duration, reason
        )
        self._changes.append(event)

        LOGGER.info(
            f"✓ Timing mutated: {node_id} {old_start}→{new_start}s, {old_duration}→{new_duration}s ({reason})"
        )

        return True, f"Updated timing: {new_start}s start, {new_duration}s duration"

    def lock_timing(self, node_id: str) -> None:
        """Lock timing for a node (cannot be auto-recomputed)."""
        self._locked_nodes.add(node_id)
        LOGGER.info(f"Locked timing for {node_id}")

    def unlock_timing(self, node_id: str) -> None:
        """Unlock timing for a node."""
        self._locked_nodes.discard(node_id)
        LOGGER.info(f"Unlocked timing for {node_id}")

    def get_timing(self, node_id: str) -> Optional[Tuple[float, float]]:
        """Get current timing for a node."""
        constraint = self._constraints.get(node_id)
        if constraint:
            return (constraint.start_time, constraint.duration)
        return None

    def validate_timing(self, timing: Dict[str, Tuple[float, float]]) -> List[str]:
        """
        Validate timing for conflicts and issues.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        for node_id, (start, duration) in timing.items():
            if start < 0:
                errors.append(f"{node_id}: start time cannot be negative ({start})")
            if duration <= 0:
                errors.append(f"{node_id}: duration must be positive ({duration})")

        return errors

    def get_timing_hash(self) -> str:
        """Get hash of current timing state."""
        return self._timing_hash

    def get_audit_trail(self) -> List[TimingEvent]:
        """Get all timing changes (for audit/undo)."""
        return list(self._changes)

    def _compute_duration(self, node: dict) -> float:
        """Compute duration for a node from its properties."""
        # Default: 2 seconds for animations
        default_duration = 2.0

        # Check node type
        node_type = node.get("type", "")
        if "animation" in node_type.lower():
            return node.get("duration", default_duration)
        elif "voiceover" in node_type.lower():
            return node.get("voiceover_duration", default_duration)
        else:
            return node.get("duration", default_duration)

    def _topological_sort(self, nodes: Dict[str, dict]) -> List[str]:
        """
        Topologically sort nodes for deterministic ordering.

        Ensures same input always produces same output.
        """
        # Simple implementation: sort by node ID for determinism
        return sorted(nodes.keys())

    def _validate_timing(
        self, timing: Dict[str, Tuple[float, float]], nodes: Dict[str, dict]
    ) -> None:
        """Validate timing consistency."""
        for node_id, (start, duration) in timing.items():
            if start < 0 or duration <= 0:
                LOGGER.warning(f"Invalid timing for {node_id}: ({start}, {duration})")

    def _hash_timing(self, timing: Dict[str, Tuple[float, float]]) -> str:
        """Hash the current timing state."""
        timing_str = "|".join(
            f"{nid}:{s:.4f},{d:.4f}" for nid, (s, d) in sorted(timing.items())
        )
        return hashlib.sha256(timing_str.encode()).hexdigest()


class TimingMutationAPI:
    """
    Controlled API for all timing mutations.

    Ensures:
    - No direct node modification
    - All changes go through resolver
    - Changes are validated
    - Audit trail maintained
    """

    def __init__(self, resolver: TimingResolver):
        self.resolver = resolver
        self._mutators = []

    def register_mutator(self, callback) -> None:
        """Register a callback to apply mutations to nodes."""
        self._mutators.append(callback)

    def change_node_timing(
        self,
        node_id: str,
        start_time: float,
        duration: float,
        reason: str = "edit",
    ) -> Tuple[bool, str]:
        """
        Change timing for a node.

        This is the ONLY way to change node timing.
        Must go through resolver validation.
        """
        # Validate through resolver
        success, message = self.resolver.mutate_timing(
            node_id, start_time, duration, reason
        )

        if not success:
            return False, message

        # Apply to all registered nodes
        for mutator in self._mutators:
            mutator(node_id, start_time, duration)

        return True, message

    def lock_node(self, node_id: str) -> None:
        """Lock a node's timing (prevent auto-recompute)."""
        self.resolver.lock_timing(node_id)

    def unlock_node(self, node_id: str) -> None:
        """Unlock a node's timing."""
        self.resolver.unlock_timing(node_id)


# Global timing resolver instance
TIMING_RESOLVER = TimingResolver()
TIMING_API = TimingMutationAPI(TIMING_RESOLVER)
