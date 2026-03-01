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
        # category -> [NodeDefinition]
        self._nodes: Dict[str, List[NodeDefinition]] = {}

    def register_node(
        self,
        node_name: str,
        class_path: str,
        category: str,
        description: str,
        extension_id: str,
    ) -> None:
        """
        Register a custom node.

        Args:
            node_name: Display name (e.g., "Integral Symbol")
            class_path: Module path (e.g., "core.extensions.math_symbols.IntegralSymbol")
            category: Category name (e.g., "Math Symbols")
            description: Human-readable description
            extension_id: Extension that registered this node
        """
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
        """Get all registered nodes organized by category."""
        return dict(self._nodes)

    def get_nodes_by_category(self, category: str) -> List[NodeDefinition]:
        """Get nodes in a specific category."""
        return self._nodes.get(category, [])

    def search_nodes(self, query: str) -> List[tuple]:
        """
        Search for nodes by name or description.

        Returns:
            List of (category, NodeDefinition) tuples
        """
        query_lower = query.lower()
        results = []

        for category, nodes in self._nodes.items():
            for node in nodes:
                if (
                    query_lower in node.node_name.lower()
                    or query_lower in node.description.lower()
                ):
                    results.append((category, node))

        return results


# Global node registry instance
NODE_REGISTRY = NodeRegistry()
