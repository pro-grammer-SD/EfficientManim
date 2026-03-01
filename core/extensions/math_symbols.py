"""
Math Symbol Extension - Demo Extension for EfficientManim

Registers custom mathematical symbol nodes for animation creation.

Example usage:
    from core.extension_api import ExtensionAPI
    from core.extensions.math_symbols import setup
    api = ExtensionAPI("math-symbols")
    setup(api)
"""

# Extension metadata (declares required permissions)
EXTENSION_METADATA = {
    "name": "Math Symbols",
    "author": "EfficientManim",
    "version": "1.0.0",
    "description": "Mathematical symbol nodes for equation animations",
    "permissions": ["register_nodes"],
}


def setup(api):
    """
    Initialize math symbol extension.

    Registers mathematical symbol nodes.
    """
    # Register integral symbol node
    api.register_node(
        node_name="Integral Symbol",
        class_path="core.extensions.math_symbols.IntegralSymbolNode",
        category="Mathematical Symbols",
        description="Animated integral symbol (∫) for calculus animations",
    )

    # Register summation node
    api.register_node(
        node_name="Summation Symbol",
        class_path="core.extensions.math_symbols.SummationSymbolNode",
        category="Mathematical Symbols",
        description="Animated summation symbol (Σ) for series animations",
    )

    # Register matrix node
    api.register_node(
        node_name="Matrix Grid",
        class_path="core.extensions.math_symbols.MatrixGridNode",
        category="Mathematical Symbols",
        description="Animated matrix grid with customizable dimensions",
    )

    return True


class IntegralSymbolNode:
    """Generates animated integral symbol."""

    def __init__(self):
        self.scale = 1.0
        self.color = "#2563eb"

    def render(self):
        """Render integral symbol."""
        pass


class SummationSymbolNode:
    """Generates animated summation symbol."""

    def __init__(self):
        self.scale = 1.0
        self.color = "#2563eb"
        self.range_start = 1
        self.range_end = 10

    def render(self):
        """Render summation symbol with range."""
        pass


class MatrixGridNode:
    """Generates animated matrix grid."""

    def __init__(self):
        self.rows = 3
        self.cols = 3
        self.cell_size = 0.5
        self.fill_color = "#f3f4f6"

    def render(self):
        """Render matrix grid."""
        pass
