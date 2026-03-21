from __future__ import annotations
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field, asdict
from typing import Any, Optional, List, Dict


@dataclass
class SceneObject:
    id: str
    type: str
    label: Optional[str]
    role: str
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AnimationStep:
    step_index: int
    animation_type: str
    targets: List[str]
    duration: float
    lag_ratio: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ObjectRelationship:
    from_id: str
    to_id: str
    relationship: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SceneAnalysis:
    scene_name: str
    object_count: int
    objects: List[SceneObject] = field(default_factory=list)
    text_elements: List[str] = field(default_factory=list)
    math_expressions: List[str] = field(default_factory=list)
    animation_steps: List[AnimationStep] = field(default_factory=list)
    object_relationships: List[ObjectRelationship] = field(default_factory=list)
    concept_hints: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "scene_name": self.scene_name,
            "object_count": self.object_count,
            "objects": [o.to_dict() for o in self.objects],
            "text_elements": list(self.text_elements),
            "math_expressions": list(self.math_expressions),
            "animation_steps": [s.to_dict() for s in self.animation_steps],
            "object_relationships": [r.to_dict() for r in self.object_relationships],
            "concept_hints": list(self.concept_hints),
        }


@dataclass
class ExplainRequest:
    analysis: SceneAnalysis
    mode: str = "detailed"

    def to_dict(self) -> dict:
        return {"analysis": self.analysis.to_dict(), "mode": self.mode}


@dataclass
class ExplainResponse:
    concept_explanation: str
    step_by_step: str
    visual_reasoning: str
    simple_explanation: str
    key_takeaways: List[str]
    mode_used: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LessonNotes:
    lesson_title: str
    concept_explanation: str
    visual_explanation: str
    step_by_step_teaching: str
    student_notes: str
    key_takeaways: List[str]

    def to_dict(self) -> dict:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            f"# Lesson: {self.lesson_title}",
            "",
            "## Concept Being Explained",
            self.concept_explanation or "",
            "",
            "## Visual Explanation Using the Animation",
            self.visual_explanation or "",
            "",
            "## Step-by-Step Teaching Explanation",
            self.step_by_step_teaching or "",
            "",
            "## Student-Friendly Notes",
            self.student_notes or "",
            "",
            "## Key Takeaways",
        ]
        for item in self.key_takeaways:
            lines.append(f"- {item}")
        return "\n".join(lines).strip() + "\n"


@dataclass
class HistoryDiff:
    from_checkpoint: str
    to_checkpoint: str
    objects_added: List[str]
    objects_removed: List[str]
    animations_added: List[str]
    animations_removed: List[str]
    concept_change_summary: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HistoryExplainRequest:
    mode: str = "detailed"
    checkpoint_id: Optional[str] = None
    from_checkpoint: Optional[str] = None
    to_checkpoint: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HistoryExplainResponse:
    checkpoint_id: Optional[str] = None
    concept_explanation: str = ""
    step_by_step: str = ""
    visual_reasoning: str = ""
    simple_explanation: str = ""
    objects_added: List[str] = field(default_factory=list)
    objects_removed: List[str] = field(default_factory=list)
    animations_added: List[str] = field(default_factory=list)
    animations_removed: List[str] = field(default_factory=list)
    concept_change_summary: str = ""
    educational_significance: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
