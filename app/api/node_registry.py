"""
node_registry.py — Global registry for extension-provided nodes

Extensions register custom node types here via ExtensionAPI.
ElementsPanel polls this registry to display nodes in the UI.
"""

import logging
from typing import Dict, List
from dataclasses import dataclass

LOGGER = logging.getLogger("node_registry")


@dataclass
class NodeDefinition:
    """Describes a custom node provided by an extension."""

    node_name: str
    class_path: str
    category: str
    description: str
    extension_id: str


class NodeRegistry:
    """
    Global registry for all extension-provided nodes.

    Nodes registered during setup() are added here and then
    appear in the ElementsPanel for users to add to the graph.
    """

    def __init__(self):
        self._nodes: Dict[str, List[NodeDefinition]] = {}

    def register_node(
        self,
        node_name: str,
        class_path: str,
        category: str,
        description: str,
        extension_id: str,
    ) -> None:
        if category not in self._nodes:
            self._nodes[category] = []

        node_def = NodeDefinition(
            node_name=node_name,
            class_path=class_path,
            category=category,
            description=description,
            extension_id=extension_id,
        )
        self._nodes[category].append(node_def)
        LOGGER.info(f"Registered node: {node_name} -> {class_path} ({extension_id})")

    def get_nodes(self) -> Dict[str, List[NodeDefinition]]:
        """Get all registered nodes organised by category."""
        return dict(self._nodes)

    def get_nodes_by_category(self, category: str) -> List[NodeDefinition]:
        return self._nodes.get(category, [])

    def search_nodes(self, query: str) -> List[tuple]:
        """Search for nodes by name or description."""
        q = query.lower()
        return [
            (cat, node)
            for cat, nodes in self._nodes.items()
            for node in nodes
            if q in node.node_name.lower() or q in node.description.lower()
        ]

    def clear(self) -> None:
        self._nodes.clear()


NODE_REGISTRY = NodeRegistry()
