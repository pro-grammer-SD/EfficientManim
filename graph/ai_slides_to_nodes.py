from __future__ import annotations
# -*- coding: utf-8 -*-

from typing import Any

from graph.edge import WireItem
from graph.node import NodeItem
from utils.logger import LOGGER


class SlidesToNodes:
    """Builds a node graph from structured slide JSON."""

    @staticmethod
    def apply_to_scene(
        slide_deck: dict | list,
        scene_graph: Any,
        clear_existing: bool = True,
    ) -> dict:
        if isinstance(slide_deck, dict):
            slides = slide_deck.get("slides") or []
        else:
            slides = slide_deck

        if clear_existing:
            SlidesToNodes._clear_scene(scene_graph)

        if hasattr(scene_graph, "is_ai_generated_code"):
            scene_graph.is_ai_generated_code = True

        created_nodes: list[NodeItem] = []
        created_wires: list[WireItem] = []

        start_x = 40
        start_y = 40
        col_w = 220
        row_h = 110
        slide_gap_x = 720

        previous_struct = None

        for idx, slide in enumerate(slides, start=1):
            if not isinstance(slide, dict):
                continue

            slide_x = start_x + (idx - 1) * slide_gap_x
            y = start_y

            title = str(slide.get("title") or "").strip()
            bullets = slide.get("bullets") or []
            equations = slide.get("equations") or []
            deriv = slide.get("derivation_steps") or []

            bullets = [str(b).strip() for b in bullets if str(b).strip()]
            equations = [str(e).strip() for e in equations if str(e).strip()]
            deriv_eqs = [
                str(d.get("equation")).strip()
                for d in deriv
                if isinstance(d, dict) and str(d.get("equation") or "").strip()
            ]

            mobject_nodes: list[NodeItem] = []
            anim_nodes: list[NodeItem] = []

            if title:
                node = scene_graph.add_node(
                    "MOBJECT",
                    "Text",
                    params={"text": title, "font_size": 48},
                    pos=(slide_x, y),
                    name=f"slide_{idx}_title",
                )
                node.data.is_ai_generated = True
                mobject_nodes.append(node)
                created_nodes.append(node)
                y += row_h

                anim = SlidesToNodes._create_anim(
                    scene_graph, node, "Write", (slide_x + col_w, y - row_h)
                )
                anim_nodes.append(anim)
                created_nodes.append(anim)

            for b in bullets:
                node = scene_graph.add_node(
                    "MOBJECT",
                    "Text",
                    params={"text": f"- {b}", "font_size": 30},
                    pos=(slide_x, y),
                    name=f"slide_{idx}_bullet_{len(mobject_nodes) + 1}",
                )
                node.data.is_ai_generated = True
                mobject_nodes.append(node)
                created_nodes.append(node)

                anim = SlidesToNodes._create_anim(
                    scene_graph, node, "FadeIn", (slide_x + col_w, y)
                )
                anim_nodes.append(anim)
                created_nodes.append(anim)
                y += row_h

            for eq in equations:
                node = scene_graph.add_node(
                    "MOBJECT",
                    "MathTex",
                    params={"arg0": SlidesToNodes._latex_str(eq)},
                    pos=(slide_x, y),
                    name=f"slide_{idx}_eq_{len(mobject_nodes) + 1}",
                )
                node.data.set_escape_string("arg0", True)
                node.data.is_ai_generated = True
                mobject_nodes.append(node)
                created_nodes.append(node)

                anim = SlidesToNodes._create_anim(
                    scene_graph, node, "Write", (slide_x + col_w, y)
                )
                anim_nodes.append(anim)
                created_nodes.append(anim)
                y += row_h

            for eq in deriv_eqs:
                node = scene_graph.add_node(
                    "MOBJECT",
                    "MathTex",
                    params={"arg0": SlidesToNodes._latex_str(eq)},
                    pos=(slide_x, y),
                    name=f"slide_{idx}_deriv_{len(mobject_nodes) + 1}",
                )
                node.data.set_escape_string("arg0", True)
                node.data.is_ai_generated = True
                mobject_nodes.append(node)
                created_nodes.append(node)

                anim = SlidesToNodes._create_anim(
                    scene_graph, node, "Write", (slide_x + col_w, y)
                )
                anim_nodes.append(anim)
                created_nodes.append(anim)
                y += row_h

            group_name = f"slide_{idx}_group"
            group = scene_graph.add_node(
                "VGROUP",
                group_name,
                params={},
                pos=(slide_x, y + 10),
                name=group_name,
            )
            group.data.is_ai_generated = True
            created_nodes.append(group)

            for mob in mobject_nodes:
                wire = scene_graph.add_wire_by_ids(mob.data.id, group.data.id)
                if wire:
                    created_wires.append(wire)

            play = scene_graph.add_node(
                "PLAY",
                "play()",
                params={},
                pos=(slide_x + 2 * col_w, start_y),
                name=f"slide_{idx}_play",
            )
            play.data.is_ai_generated = True
            created_nodes.append(play)

            for anim in anim_nodes:
                wire = scene_graph.add_wire_by_ids(anim.data.id, play.data.id)
                if wire:
                    created_wires.append(wire)

            if previous_struct is not None:
                wire = scene_graph.add_wire_by_ids(
                    previous_struct.data.id, play.data.id
                )
                if wire:
                    created_wires.append(wire)

            wait = scene_graph.add_node(
                "WAIT",
                "wait()",
                params={"duration": 0.5},
                pos=(slide_x + 3 * col_w, start_y),
                name=f"slide_{idx}_wait",
            )
            wait.data.is_ai_generated = True
            created_nodes.append(wait)

            wire = scene_graph.add_wire_by_ids(play.data.id, wait.data.id)
            if wire:
                created_wires.append(wire)
            previous_struct = wait

        if hasattr(scene_graph, "scene") and hasattr(
            scene_graph.scene, "notify_change"
        ):
            try:
                scene_graph.scene.notify_change()
            except Exception:
                pass

        return {
            "nodes": created_nodes,
            "wires": created_wires,
            "node_count": len(created_nodes),
            "wire_count": len(created_wires),
        }

    @staticmethod
    def _create_anim(scene_graph: Any, target: NodeItem, anim_cls: str, pos):
        anim = scene_graph.add_node(
            "ANIMATION",
            anim_cls,
            params={},
            pos=pos,
            name=f"{anim_cls.lower()}_{target.data.name}",
        )
        anim.data.is_ai_generated = True
        scene_graph.add_wire_by_ids(target.data.id, anim.data.id)
        return anim

    @staticmethod
    def _clear_scene(scene_graph: Any) -> None:
        try:
            nodes = list(scene_graph.nodes.values())
            for node in nodes:
                scene_graph.scene.removeItem(node)
            scene_graph.nodes.clear()

            for item in list(scene_graph.scene.items()):
                if isinstance(item, WireItem):
                    scene_graph.scene.removeItem(item)

            if hasattr(scene_graph, "render_queue"):
                scene_graph.render_queue.clear()
        except Exception as exc:
            LOGGER.error(f"Failed to clear scene: {exc}")

    @staticmethod
    def _latex_str(text: str) -> str:
        safe = text.replace("\r\n", "\n").replace("\r", "\n")
        if '"""' not in safe:
            return f'r"""{safe}"""'
        if "'''" not in safe:
            return f"r'''{safe}'''"
        return repr(safe)
