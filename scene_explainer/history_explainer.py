from __future__ import annotations
# -*- coding: utf-8 -*-

from typing import Dict, List, Optional, Tuple

from core.history_manager import HistoryManager

from scene_explainer.analyzer import SceneAnalyzer
from scene_explainer.explanation_models import (
    ExplainResponse,
    HistoryDiff,
    HistoryExplainResponse,
    SceneAnalysis,
)
from scene_explainer.prompt_builder import PromptBuilder
from scene_explainer.ai_explainer import AIExplainer


class HistoryExplainer:
    """Explain checkpoints, diffs, and undo/redo using structured analysis."""

    def __init__(
        self,
        history_manager: HistoryManager,
        analyzer: SceneAnalyzer,
        prompt_builder: PromptBuilder,
        ai: AIExplainer,
    ) -> None:
        self.history = history_manager
        self.analyzer = analyzer
        self.prompt_builder = prompt_builder
        self.ai = ai
        self._checkpoint_cache: Dict[str, ExplainResponse] = {}
        self._scene_fingerprint: Optional[str] = None

    def invalidate_cache(self) -> None:
        self._checkpoint_cache.clear()
        self._scene_fingerprint = self._current_fingerprint()

    def explain_checkpoint(
        self, checkpoint_id: str, mode: str = "detailed"
    ) -> ExplainResponse:
        self._ensure_cache_valid()
        if checkpoint_id in self._checkpoint_cache:
            return self._checkpoint_cache[checkpoint_id]

        cp = self._get_checkpoint(checkpoint_id)
        if cp is None or cp.snapshot is None:
            raise ValueError("Checkpoint not found or has no snapshot.")

        analysis = self.analyzer.analyze_snapshot(
            cp.snapshot, scene_name=cp.snapshot.scene
        )
        prompt = self.prompt_builder.build_explain_prompt(analysis, mode)
        response = self.ai.run_blocking(prompt, self.ai._parse_explain_response)
        if isinstance(response, ExplainResponse):
            self._checkpoint_cache[checkpoint_id] = response
            return response
        raise RuntimeError("Invalid checkpoint explanation response")

    def explain_history_change(
        self, from_checkpoint: str, to_checkpoint: str, mode: str = "detailed"
    ) -> HistoryExplainResponse:
        before_cp = self._get_checkpoint(from_checkpoint)
        after_cp = self._get_checkpoint(to_checkpoint)
        if not before_cp or not after_cp:
            raise ValueError("Checkpoint not found.")
        if not before_cp.snapshot or not after_cp.snapshot:
            raise ValueError("Checkpoint snapshots unavailable.")

        before_analysis = self.analyzer.analyze_snapshot(before_cp.snapshot)
        after_analysis = self.analyzer.analyze_snapshot(after_cp.snapshot)
        diff = self._compute_diff(
            before_cp.snapshot, after_cp.snapshot, from_checkpoint, to_checkpoint
        )

        prompt = self.prompt_builder.build_history_change_prompt(
            diff, before_analysis, after_analysis, mode
        )
        ai_payload = self.ai.run_blocking(prompt, self.ai._parse_history_change)
        if not isinstance(ai_payload, dict):
            raise RuntimeError("Invalid history change response")

        return HistoryExplainResponse(
            checkpoint_id=None,
            objects_added=diff.objects_added,
            objects_removed=diff.objects_removed,
            animations_added=diff.animations_added,
            animations_removed=diff.animations_removed,
            concept_change_summary=ai_payload.get(
                "concept_change_summary", diff.concept_change_summary
            ),
            educational_significance=ai_payload.get("educational_significance", ""),
        )

    def explain_undo(self, mode: str = "detailed") -> dict:
        action = self._peek_last_undo_action()
        if action is None:
            raise ValueError("No undo action available.")
        diff = self._compute_diff(action.after, action.before, "current", "undo")
        return self._explain_action(diff, mode)

    def explain_redo(self, mode: str = "detailed") -> dict:
        action = self._peek_last_redo_action()
        if action is None:
            raise ValueError("No redo action available.")
        diff = self._compute_diff(action.before, action.after, "current", "redo")
        return self._explain_action(diff, mode)

    # ──────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────

    def _ensure_cache_valid(self) -> None:
        current = self._current_fingerprint()
        if self._scene_fingerprint is None:
            self._scene_fingerprint = current
            return
        if current and current != self._scene_fingerprint:
            self.invalidate_cache()

    def _current_fingerprint(self) -> Optional[str]:
        snapshot = getattr(self.history, "_current_snapshot", None)
        return getattr(snapshot, "fingerprint", None)

    def _get_checkpoint(self, checkpoint_id: str):
        return getattr(self.history, "_checkpoints", {}).get(checkpoint_id)

    def _compute_diff(
        self, before_snapshot, after_snapshot, from_id: str, to_id: str
    ) -> HistoryDiff:
        before_nodes = before_snapshot.nodes if before_snapshot else {}
        after_nodes = after_snapshot.nodes if after_snapshot else {}

        added = sorted(set(after_nodes.keys()) - set(before_nodes.keys()))
        removed = sorted(set(before_nodes.keys()) - set(after_nodes.keys()))

        def _node_label(snapshot_nodes, nid: str) -> str:
            data = snapshot_nodes[nid].data
            return data.get("name") or data.get("cls_name") or nid

        objects_added = [
            _node_label(after_nodes, nid)
            for nid in added
            if after_nodes[nid].data.get("type") == "MOBJECT"
        ]
        objects_removed = [
            _node_label(before_nodes, nid)
            for nid in removed
            if before_nodes[nid].data.get("type") == "MOBJECT"
        ]
        animations_added = [
            _node_label(after_nodes, nid)
            for nid in added
            if after_nodes[nid].data.get("type") == "ANIMATION"
        ]
        animations_removed = [
            _node_label(before_nodes, nid)
            for nid in removed
            if before_nodes[nid].data.get("type") == "ANIMATION"
        ]

        before_analysis = self.analyzer.analyze_snapshot(before_snapshot)
        after_analysis = self.analyzer.analyze_snapshot(after_snapshot)
        concept_change = self._concept_summary(before_analysis, after_analysis)

        # Add animation step differences
        added_steps, removed_steps = self._diff_steps(before_analysis, after_analysis)
        animations_added.extend(a for a in added_steps if a not in animations_added)
        animations_removed.extend(
            a for a in removed_steps if a not in animations_removed
        )

        return HistoryDiff(
            from_checkpoint=from_id,
            to_checkpoint=to_id,
            objects_added=sorted(set(objects_added)),
            objects_removed=sorted(set(objects_removed)),
            animations_added=sorted(set(animations_added)),
            animations_removed=sorted(set(animations_removed)),
            concept_change_summary=concept_change,
        )

    def _diff_steps(
        self, before: SceneAnalysis, after: SceneAnalysis
    ) -> Tuple[List[str], List[str]]:
        def _format_step(step) -> str:
            tgt = ", ".join(step.targets) if step.targets else ""
            return f"{step.animation_type}({tgt})".strip()

        before_steps = {_format_step(s) for s in before.animation_steps}
        after_steps = {_format_step(s) for s in after.animation_steps}
        added = sorted(after_steps - before_steps)
        removed = sorted(before_steps - after_steps)
        return added, removed

    def _concept_summary(self, before: SceneAnalysis, after: SceneAnalysis) -> str:
        before_hints = set(before.concept_hints)
        after_hints = set(after.concept_hints)
        added = sorted(after_hints - before_hints)
        removed = sorted(before_hints - after_hints)
        parts = []
        if added:
            parts.append(f"Introduced: {', '.join(added)}")
        if removed:
            parts.append(f"De-emphasized: {', '.join(removed)}")
        if not parts:
            return "No major concept shift detected."
        return "; ".join(parts)

    def _peek_last_undo_action(self):
        try:
            stack = getattr(self.history, "_redo_stack", [])
            return stack[-1] if stack else None
        except Exception:
            return None

    def _peek_last_redo_action(self):
        try:
            stack = getattr(self.history, "_undo_stack", [])
            return stack[-1] if stack else None
        except Exception:
            return None

    def _explain_action(self, diff: HistoryDiff, mode: str) -> dict:
        prompt = (
            "You are a math teacher explaining a recent undo/redo change to a student.\n"
            "Do not mention programming, code, nodes, or library names.\n"
            "Return ONLY valid JSON with keys: action_description, educational_impact.\n"
            "\n"
            f"Objects removed: {', '.join(diff.objects_removed) or 'None'}\n"
            f"Objects added: {', '.join(diff.objects_added) or 'None'}\n"
            f"Animations removed: {', '.join(diff.animations_removed) or 'None'}\n"
            f"Animations added: {', '.join(diff.animations_added) or 'None'}\n"
            f"Concept change: {diff.concept_change_summary}\n"
        )
        payload = self.ai.run_blocking(prompt, lambda t: self.ai._extract_json(t))
        if not isinstance(payload, dict):
            raise RuntimeError("Invalid undo/redo explanation response")
        return {
            "action_description": self.ai._sanitize_text(
                str(payload.get("action_description", "")).strip()
            ),
            "educational_impact": self.ai._sanitize_text(
                str(payload.get("educational_impact", "")).strip()
            ),
        }
