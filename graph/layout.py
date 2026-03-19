from __future__ import annotations
# -*- coding: utf-8 -*-

from graph.edge import WireItem
from graph.node import NodeType
from utils.logger import LOGGER


def auto_layout_nodes(nodes: dict, scene, fit_view=None) -> None:
    """Auto-arrange nodes in a clean left-to-right flow layout."""
    if not nodes:
        return

    nodes_list = list(nodes.values())
    mobjects = [n for n in nodes_list if n.data.type == NodeType.MOBJECT]
    animations = [n for n in nodes_list if n.data.type == NodeType.ANIMATION]

    col_w = 220
    row_h = 120
    start_x, start_y = 50, 50

    for i, node in enumerate(mobjects):
        node.setPos(start_x, start_y + i * row_h)
        node.data.pos_x = start_x
        node.data.pos_y = start_y + i * row_h

    for i, node in enumerate(animations):
        node.setPos(start_x + col_w, start_y + i * row_h)
        node.data.pos_x = start_x + col_w
        node.data.pos_y = start_y + i * row_h

    for item in scene.items():
        if isinstance(item, WireItem):
            item.update_path()

    if hasattr(scene, "notify_change"):
        scene.notify_change()

    if fit_view:
        try:
            fit_view()
        except Exception:
            pass

    LOGGER.info(f"Auto-layout applied to {len(nodes_list)} nodes")
