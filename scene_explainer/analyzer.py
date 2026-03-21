from __future__ import annotations
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from core.history_manager import WireState
from graph.node import NodeData, NodeItem, NodeType
from utils.logger import LOGGER

from scene_explainer.explanation_models import (
    AnimationStep,
    ObjectRelationship,
    SceneAnalysis,
    SceneObject,
)


@dataclass(frozen=True)
class NormalizedNode:
    id: str
    name: str
    cls_name: str
    type: str
    params: Dict[str, Any]
    raw: Dict[str, Any]


class SceneAnalyzer:
    """Safely analyze a scene graph into structured, deterministic data."""

    AXES_TYPES = {"Axes", "NumberPlane", "ThreeDAxes"}
    GRAPH_TYPES = {
        "FunctionGraph",
        "ParametricFunction",
        "ImplicitFunction",
        "FunctionGraph",
        "ParametricCurve",
    }
    SHAPE_TYPES = {
        "Circle",
        "Square",
        "Rectangle",
        "Triangle",
        "Polygon",
        "Ellipse",
        "RegularPolygon",
        "RoundedRectangle",
    }
    TEXT_TYPES = {"Text", "Paragraph", "MarkupText"}
    MATH_TYPES = {"MathTex", "Tex"}
    LABEL_TYPES = {"BraceLabel", "DecimalNumber", "Variable"}
    POINT_TYPES = {"Dot", "SmallDot", "AnnotationDot"}
    LINE_TYPES = {
        "Line",
        "Arrow",
        "DoubleArrow",
        "Vector",
        "DashedLine",
        "TangentLine",
    }
    TRANSFORM_TYPES = {
        "Transform",
        "ReplacementTransform",
        "TransformMatchingTex",
        "TransformFromCopy",
    }

    def analyze_window(
        self,
        main_window,
        node_ids: Optional[Sequence[str]] = None,
        animation_id: Optional[str] = None,
        objects_only: bool = False,
    ) -> SceneAnalysis:
        """Analyze current live scene, optionally constrained to node_ids or animation_id."""
        try:
            if hasattr(main_window, "_history_snapshot_provider"):
                nodes_raw, wires_raw = main_window._history_snapshot_provider()
                scene_name = getattr(main_window, "_current_scene_name", "Scene")
            else:
                nodes_raw, wires_raw = self._snapshot_from_window(main_window)
                scene_name = getattr(main_window, "_current_scene_name", "Scene")
        except Exception as exc:
            LOGGER.warn(f"SceneAnalyzer: failed to read window scene: {exc}")
            return self._empty_analysis(getattr(main_window, "_current_scene_name", "Scene"))

        return self._analyze_raw(
            scene_name=scene_name,
            nodes_raw=nodes_raw,
            wires_raw=wires_raw,
            node_ids=node_ids,
            animation_id=animation_id,
            objects_only=objects_only,
        )

    def analyze_snapshot(
        self,
        snapshot,
        scene_name: Optional[str] = None,
    ) -> SceneAnalysis:
        """Analyze a HistoryManager GraphSnapshot safely."""
        if snapshot is None:
            return self._empty_analysis(scene_name or "Scene")
        try:
            nodes_raw = {
                nid: ns.data for nid, ns in getattr(snapshot, "nodes", {}).items()
            }
            wires_raw = list(getattr(snapshot, "wires", []))
            scene = scene_name or getattr(snapshot, "scene", "Scene")
            return self._analyze_raw(
                scene_name=scene,
                nodes_raw=nodes_raw,
                wires_raw=wires_raw,
                node_ids=None,
                animation_id=None,
                objects_only=False,
            )
        except Exception as exc:
            LOGGER.warn(f"SceneAnalyzer: failed to analyze snapshot: {exc}")
            return self._empty_analysis(scene_name or "Scene")

    # ──────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────

    def _snapshot_from_window(self, main_window) -> Tuple[Dict[str, dict], List[WireState]]:
        nodes_data: Dict[str, dict] = {}
        for nid, node in getattr(main_window, "nodes", {}).items():
            try:
                nodes_data[nid] = node.data.to_dict()
            except Exception:
                try:
                    nodes_data[nid] = NodeData.from_dict(node).to_dict()  # type: ignore[arg-type]
                except Exception:
                    pass
        wires: List[WireState] = []
        try:
            for item in list(main_window.scene.items()):
                if hasattr(item, "start_socket") and hasattr(item, "end_socket"):
                    try:
                        from_id = item.start_socket.parentItem().data.id
                        to_id = item.end_socket.parentItem().data.id
                        wire_id = getattr(item, "wire_id", f"wire_{from_id[:6]}_{to_id[:6]}")
                        wires.append(WireState(wire_id, from_id, to_id))
                    except Exception:
                        pass
        except Exception:
            pass
        return nodes_data, wires

    def _empty_analysis(self, scene_name: str) -> SceneAnalysis:
        return SceneAnalysis(
            scene_name=scene_name,
            object_count=0,
            objects=[],
            text_elements=[],
            math_expressions=[],
            animation_steps=[],
            object_relationships=[],
            concept_hints=[],
        )

    def _normalize_nodes(self, nodes_raw: Dict[str, Any]) -> Dict[str, NormalizedNode]:
        normalized: Dict[str, NormalizedNode] = {}
        for nid, raw in nodes_raw.items():
            try:
                if isinstance(raw, NodeItem):
                    data = raw.data
                    params = dict(data.params)
                    ntype = data.type.name if hasattr(data.type, "name") else str(data.type)
                    normalized[nid] = NormalizedNode(
                        id=data.id,
                        name=data.name,
                        cls_name=data.cls_name,
                        type=ntype,
                        params=params,
                        raw=data.to_dict(),
                    )
                    continue
                if isinstance(raw, NodeData):
                    params = dict(raw.params)
                    ntype = raw.type.name if hasattr(raw.type, "name") else str(raw.type)
                    normalized[nid] = NormalizedNode(
                        id=raw.id,
                        name=raw.name,
                        cls_name=raw.cls_name,
                        type=ntype,
                        params=params,
                        raw=raw.to_dict(),
                    )
                    continue
                if isinstance(raw, dict):
                    raw_id = raw.get("id") or nid
                    cls_name = raw.get("cls_name") or raw.get("class") or raw.get("name", "")
                    ntype = raw.get("type", "MOBJECT")
                    params = dict(raw.get("params", {}))
                    normalized[nid] = NormalizedNode(
                        id=str(raw_id),
                        name=str(raw.get("name", cls_name)),
                        cls_name=str(cls_name),
                        type=str(ntype),
                        params=params,
                        raw=raw,
                    )
                    continue
            except Exception:
                continue
        return normalized

    def _normalize_wires(self, wires_raw: Iterable[Any]) -> List[Tuple[str, str]]:
        wires: List[Tuple[str, str]] = []
        for w in wires_raw:
            try:
                if isinstance(w, WireState):
                    wires.append((w.from_node, w.to_node))
                    continue
                if isinstance(w, dict):
                    a = w.get("from") or w.get("from_node")
                    b = w.get("to") or w.get("to_node")
                    if a and b:
                        wires.append((str(a), str(b)))
                    continue
                if hasattr(w, "from_node") and hasattr(w, "to_node"):
                    wires.append((w.from_node, w.to_node))
                    continue
            except Exception:
                continue
        return wires

    def _analyze_raw(
        self,
        scene_name: str,
        nodes_raw: Dict[str, Any],
        wires_raw: Iterable[Any],
        node_ids: Optional[Sequence[str]],
        animation_id: Optional[str],
        objects_only: bool,
    ) -> SceneAnalysis:
        normalized = self._normalize_nodes(nodes_raw)
        wires = self._normalize_wires(wires_raw)

        if not normalized:
            return self._empty_analysis(scene_name)

        selected: Optional[set[str]] = None
        if node_ids:
            selected = {str(n) for n in node_ids if str(n) in normalized}
        if animation_id:
            selected = {str(animation_id)} if str(animation_id) in normalized else set()

        if selected is not None:
            selected = self._expand_selection(selected, normalized, wires)

        nodes = (
            {nid: n for nid, n in normalized.items() if nid in selected}
            if selected is not None
            else normalized
        )

        if not nodes:
            return self._empty_analysis(scene_name)

        out_map, in_map = self._build_maps(wires)

        objects: List[SceneObject] = []
        text_elements: List[str] = []
        math_expressions: List[str] = []

        # Gather target usage for role inference
        targets_in_anims = self._collect_animation_targets(nodes, in_map)

        for nid in sorted(nodes.keys()):
            node = nodes[nid]
            if node.type.upper() != NodeType.MOBJECT.name and node.type.upper() != NodeType.VGROUP.name:
                continue
            obj_type, category = self._classify(node.cls_name)
            label = self._extract_label(node)
            role = self._infer_role(nid, node, targets_in_anims, in_map, out_map)
            props = self._filter_properties(node)
            if category:
                props["category"] = category
            if node.cls_name:
                props["class_name"] = node.cls_name
            objects.append(
                SceneObject(
                    id=nid,
                    type=obj_type,
                    label=label,
                    role=role,
                    properties=props,
                )
            )
            if label:
                if node.cls_name in self.MATH_TYPES:
                    math_expressions.append(label)
                elif node.cls_name in self.TEXT_TYPES:
                    text_elements.append(label)
                elif node.cls_name in self.LABEL_TYPES:
                    text_elements.append(label)

        animation_steps = []
        if not objects_only:
            animation_steps = self._build_animation_steps(nodes, in_map, out_map)

        object_relationships = self._build_relationships(nodes, in_map, out_map)

        concept_hints = self._infer_concepts(nodes, animation_steps)

        return SceneAnalysis(
            scene_name=scene_name,
            object_count=len(objects),
            objects=objects,
            text_elements=sorted(set(text_elements)),
            math_expressions=sorted(set(math_expressions)),
            animation_steps=animation_steps,
            object_relationships=object_relationships,
            concept_hints=concept_hints,
        )

    def _build_maps(self, wires: List[Tuple[str, str]]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        out_map: Dict[str, List[str]] = {}
        in_map: Dict[str, List[str]] = {}
        for a, b in wires:
            out_map.setdefault(a, []).append(b)
            in_map.setdefault(b, []).append(a)
        for m in (out_map, in_map):
            for k in m:
                m[k] = sorted(set(m[k]))
        return out_map, in_map

    def _expand_selection(
        self,
        selected: set[str],
        nodes: Dict[str, NormalizedNode],
        wires: List[Tuple[str, str]],
    ) -> set[str]:
        """Include related targets for selected animations."""
        out_map, in_map = self._build_maps(wires)
        expanded = set(selected)
        for nid in list(selected):
            node = nodes.get(nid)
            if not node:
                continue
            if node.type.upper() == NodeType.ANIMATION.name:
                for target in in_map.get(nid, []):
                    expanded.add(target)
            if node.type.upper() == NodeType.PLAY.name:
                for anim in in_map.get(nid, []):
                    expanded.add(anim)
                    for target in in_map.get(anim, []):
                        expanded.add(target)
        return expanded

    def _classify(self, cls_name: str) -> Tuple[str, str]:
        if cls_name in self.AXES_TYPES:
            return cls_name, "Axes & Grids"
        if cls_name in self.GRAPH_TYPES or "Graph" in cls_name:
            return cls_name, "Graphs"
        if cls_name in self.SHAPE_TYPES:
            return cls_name, "Shapes"
        if cls_name in self.TEXT_TYPES:
            return cls_name, "Text"
        if cls_name in self.MATH_TYPES:
            return cls_name, "Math"
        if cls_name in self.LABEL_TYPES:
            return cls_name, "Labels"
        if cls_name in self.POINT_TYPES:
            return cls_name, "Points"
        if cls_name in self.LINE_TYPES or "Line" in cls_name:
            return cls_name, "Lines"
        return cls_name or "Object", "Other"

    def _extract_label(self, node: NormalizedNode) -> Optional[str]:
        if not node.params:
            return None
        candidates = []
        for key in ("text", "tex", "latex", "string", "label", "value", "arg0", "arg1"):
            if key in node.params:
                candidates.append(node.params[key])
        for c in candidates:
            if isinstance(c, str) and c.strip():
                return c.strip()
            if isinstance(c, (list, tuple)):
                joined = " ".join(str(x) for x in c if str(x).strip())
                if joined:
                    return joined
        return None

    def _filter_properties(self, node: NormalizedNode) -> Dict[str, Any]:
        props: Dict[str, Any] = {}
        for k, v in node.params.items():
            if str(k).startswith("_"):
                continue
            if isinstance(v, (str, int, float, bool)) or v is None:
                props[str(k)] = v
        return props

    def _infer_role(
        self,
        nid: str,
        node: NormalizedNode,
        targets_in_anims: set[str],
        in_map: Dict[str, List[str]],
        out_map: Dict[str, List[str]],
    ) -> str:
        if nid in targets_in_anims:
            return "primary"
        if in_map.get(nid) or out_map.get(nid):
            return "supporting"
        if node.cls_name in self.LABEL_TYPES or node.cls_name in self.POINT_TYPES:
            return "supporting"
        return "decorative"

    def _collect_animation_targets(
        self, nodes: Dict[str, NormalizedNode], in_map: Dict[str, List[str]]
    ) -> set[str]:
        targets: set[str] = set()
        for nid, node in nodes.items():
            if node.type.upper() == NodeType.ANIMATION.name:
                for tgt in in_map.get(nid, []):
                    if tgt in nodes:
                        targets.add(tgt)
        return targets

    def _build_animation_steps(
        self,
        nodes: Dict[str, NormalizedNode],
        in_map: Dict[str, List[str]],
        out_map: Dict[str, List[str]],
    ) -> List[AnimationStep]:
        play_nodes = [nid for nid, n in nodes.items() if n.type.upper() == NodeType.PLAY.name]
        wait_nodes = [nid for nid, n in nodes.items() if n.type.upper() == NodeType.WAIT.name]
        anim_nodes = [nid for nid, n in nodes.items() if n.type.upper() == NodeType.ANIMATION.name]

        def _struct_upstream(nid: str) -> List[str]:
            return [
                src
                for src in in_map.get(nid, [])
                if nodes.get(src) and nodes[src].type.upper() in (NodeType.PLAY.name, NodeType.WAIT.name)
            ]

        def _struct_downstream(nid: str) -> List[str]:
            return [
                dst
                for dst in out_map.get(nid, [])
                if nodes.get(dst) and nodes[dst].type.upper() in (NodeType.PLAY.name, NodeType.WAIT.name)
            ]

        structural_roots = [nid for nid in play_nodes + wait_nodes if not _struct_upstream(nid)]
        ordered_structural: List[str] = []
        visited: set[str] = set()

        def _visit(nid: str):
            if nid in visited:
                return
            visited.add(nid)
            ordered_structural.append(nid)
            for child in sorted(_struct_downstream(nid)):
                _visit(child)

        for root in sorted(structural_roots):
            _visit(root)
        for nid in sorted(play_nodes + wait_nodes):
            if nid not in visited:
                ordered_structural.append(nid)

        steps: List[AnimationStep] = []
        step_index = 0
        anim_claimed: set[str] = set()

        for nid in ordered_structural:
            node = nodes.get(nid)
            if not node:
                continue
            if node.type.upper() == NodeType.PLAY.name:
                incoming = [src for src in in_map.get(nid, []) if src in nodes]
                for src in sorted(incoming):
                    src_node = nodes[src]
                    if src_node.type.upper() == NodeType.ANIMATION.name:
                        targets = [
                            t
                            for t in in_map.get(src, [])
                            if t in nodes and nodes[t].type.upper() in (NodeType.MOBJECT.name, NodeType.VGROUP.name)
                        ]
                        step_index += 1
                        steps.append(
                            AnimationStep(
                                step_index=step_index,
                                animation_type=src_node.cls_name or "Animation",
                                targets=sorted(targets),
                                duration=self._parse_duration(src_node.params, default=1.0),
                                lag_ratio=self._parse_lag_ratio(src_node.params),
                            )
                        )
                        anim_claimed.add(src)
                    elif src_node.type.upper() == NodeType.VGROUP.name:
                        step_index += 1
                        anim_cls = node.params.get("anim_class", "FadeIn")
                        steps.append(
                            AnimationStep(
                                step_index=step_index,
                                animation_type=str(anim_cls),
                                targets=[src],
                                duration=1.0,
                                lag_ratio=0.0,
                            )
                        )
            elif node.type.upper() == NodeType.WAIT.name:
                step_index += 1
                steps.append(
                    AnimationStep(
                        step_index=step_index,
                        animation_type="Wait",
                        targets=[],
                        duration=self._parse_wait_duration(node.params),
                        lag_ratio=0.0,
                    )
                )

        legacy_anims = [nid for nid in anim_nodes if nid not in anim_claimed]
        for anim_id in sorted(legacy_anims):
            anim_node = nodes[anim_id]
            targets = [
                t
                for t in in_map.get(anim_id, [])
                if t in nodes and nodes[t].type.upper() in (NodeType.MOBJECT.name, NodeType.VGROUP.name)
            ]
            step_index += 1
            steps.append(
                AnimationStep(
                    step_index=step_index,
                    animation_type=anim_node.cls_name or "Animation",
                    targets=sorted(targets),
                    duration=self._parse_duration(anim_node.params, default=1.0),
                    lag_ratio=self._parse_lag_ratio(anim_node.params),
                )
            )

        return steps

    def _parse_duration(self, params: Dict[str, Any], default: float) -> float:
        raw = params.get("run_time", default)
        try:
            return float(raw)
        except Exception:
            return float(default)

    def _parse_wait_duration(self, params: Dict[str, Any]) -> float:
        raw = params.get("duration", 1.0)
        try:
            return float(raw)
        except Exception:
            return 1.0

    def _parse_lag_ratio(self, params: Dict[str, Any]) -> float:
        raw = params.get("lag_ratio", 0.0)
        try:
            return float(raw)
        except Exception:
            return 0.0

    def _build_relationships(
        self,
        nodes: Dict[str, NormalizedNode],
        in_map: Dict[str, List[str]],
        out_map: Dict[str, List[str]],
    ) -> List[ObjectRelationship]:
        relationships: List[ObjectRelationship] = []

        # Group relationships
        for nid, node in nodes.items():
            if node.type.upper() == NodeType.VGROUP.name:
                members = [
                    src
                    for src in in_map.get(nid, [])
                    if src in nodes and nodes[src].type.upper() == NodeType.MOBJECT.name
                ]
                for member in sorted(members):
                    relationships.append(
                        ObjectRelationship(
                            from_id=nid,
                            to_id=member,
                            relationship="groups_with",
                        )
                    )

        # Transform relationships
        for nid, node in nodes.items():
            if node.type.upper() == NodeType.ANIMATION.name and node.cls_name in self.TRANSFORM_TYPES:
                targets = [
                    t
                    for t in in_map.get(nid, [])
                    if t in nodes and nodes[t].type.upper() == NodeType.MOBJECT.name
                ]
                if len(targets) >= 2:
                    ordered = sorted(targets)
                    for i in range(len(ordered) - 1):
                        relationships.append(
                            ObjectRelationship(
                                from_id=ordered[i],
                                to_id=ordered[i + 1],
                                relationship="transforms_into",
                            )
                        )

        # Label relationships
        for nid, node in nodes.items():
            if node.type.upper() == NodeType.ANIMATION.name:
                targets = [
                    t
                    for t in in_map.get(nid, [])
                    if t in nodes and nodes[t].type.upper() == NodeType.MOBJECT.name
                ]
                labels = [t for t in targets if nodes[t].cls_name in self.LABEL_TYPES]
                others = [t for t in targets if nodes[t].cls_name not in self.LABEL_TYPES]
                for label in labels:
                    for other in others:
                        relationships.append(
                            ObjectRelationship(
                                from_id=label,
                                to_id=other,
                                relationship="labels",
                            )
                        )

        # Deduplicate deterministically
        uniq = {(r.from_id, r.to_id, r.relationship): r for r in relationships}
        result = [uniq[k] for k in sorted(uniq.keys())]
        return result

    def _infer_concepts(
        self,
        nodes: Dict[str, NormalizedNode],
        animation_steps: List[AnimationStep],
    ) -> List[str]:
        hints: List[str] = []
        cls_names = {n.cls_name for n in nodes.values()}

        if cls_names & self.AXES_TYPES:
            hints.append("coordinate plane")
        if cls_names & self.GRAPH_TYPES or any("Graph" in c for c in cls_names):
            hints.append("function graph")
        if cls_names & self.MATH_TYPES:
            hints.append("equation or expression")
        if cls_names & self.SHAPE_TYPES:
            hints.append("geometric shapes")
        if cls_names & self.LINE_TYPES:
            hints.append("lines and directions")

        for step in animation_steps:
            atype = step.animation_type
            if atype in self.TRANSFORM_TYPES:
                hints.append("transformation")
            if atype.lower().startswith("rotate"):
                hints.append("rotation")
            if atype.lower().startswith("scale"):
                hints.append("scaling")
            if atype.lower().startswith("shift") or atype.lower().startswith("move"):
                hints.append("translation")

        # Tangent/derivative hint
        if (cls_names & self.AXES_TYPES) and (cls_names & self.LINE_TYPES) and (
            cls_names & self.GRAPH_TYPES or any("Graph" in c for c in cls_names)
        ):
            hints.append("slope or derivative")

        # Area/integral hint
        if (cls_names & self.AXES_TYPES) and (cls_names & self.GRAPH_TYPES):
            for n in nodes.values():
                if "area" in n.name.lower() or "integral" in n.name.lower() or "shade" in n.name.lower():
                    hints.append("area under a curve")
                    break

        # Algebraic derivation hint
        math_count = sum(1 for c in cls_names if c in self.MATH_TYPES)
        if math_count >= 2:
            if any(step.animation_type in ("Write", "TransformMatchingTex") for step in animation_steps):
                hints.append("algebraic derivation")

        # Deduplicate, deterministic order
        return sorted(set(hints))
