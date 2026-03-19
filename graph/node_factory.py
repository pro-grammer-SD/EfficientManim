from __future__ import annotations
# -*- coding: utf-8 -*-

from graph.node import NodeData, NodeItem, NodeType


class NodeFactory:
    @staticmethod
    def create_node(
        type_str: str,
        cls_name: str,
        params: dict | None = None,
        pos: tuple[float, float] = (0, 0),
        nid: str | None = None,
        name: str | None = None,
    ) -> tuple[NodeData, NodeItem]:
        type_upper = str(type_str).upper()
        if type_upper == "MOBJECT":
            ntype = NodeType.MOBJECT
        elif type_upper == "PLAY":
            ntype = NodeType.PLAY
        elif type_upper == "WAIT":
            ntype = NodeType.WAIT
        elif type_upper == "VGROUP":
            ntype = NodeType.VGROUP
        else:
            ntype = NodeType.ANIMATION

        display_name = name if name else cls_name
        data = NodeData(display_name, ntype, cls_name)
        if nid:
            data.id = nid

        if ntype == NodeType.WAIT and not params:
            data.params = {"duration": 1.0}
        elif ntype in (NodeType.PLAY, NodeType.VGROUP) and not params:
            data.params = {}

        if params:
            data.params = dict(params)
        data.pos_x, data.pos_y = pos

        item = NodeItem(data)
        return data, item
