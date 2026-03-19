"""
Math Symbols Extension - Custom Manim Symbol Nodes

Provides pre-configured mathematical symbols for easy animation.

Example usage:
    from core.extension_api import ExtensionAPI
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
    Initialize math symbols extension.

    Registers mathematical symbol node types that can be used
    to create custom equation components.
    """
    # Register integral symbol node
    api.register_node(
        node_name="Integral Symbol",
        class_path="core.extensions.math_symbols.IntegralSymbol",
        category="Math Symbols",
        description="Animated integral symbol (∫) for calculus",
    )

    # Register summation node
    api.register_node(
        node_name="Summation Symbol",
        class_path="core.extensions.math_symbols.SummationSymbol",
        category="Math Symbols",
        description="Animated summation symbol (Σ) for series",
    )

    # Register matrix node
    api.register_node(
        node_name="Matrix Grid",
        class_path="core.extensions.math_symbols.MatrixGrid",
        category="Math Symbols",
        description="Animated matrix grid with customizable size",
    )

    return True


try:
    from manim import Tex

    MANIM_AVAILABLE = True
except ImportError:
    MANIM_AVAILABLE = False


class IntegralSymbol(Tex if MANIM_AVAILABLE else object):
    """
    Animated integral symbol for calculus equations.

    Inherits from Manim's Tex class to provide native animation support.
    """

    def __init__(self, **kwargs):
        if MANIM_AVAILABLE:
            super().__init__(r"\int", **kwargs)
        else:
            self.scale = kwargs.get("scale", 1.0)
            self.color = kwargs.get("color", "#FFFFFF")


class SummationSymbol(Tex if MANIM_AVAILABLE else object):
    """
    Animated summation symbol for series equations.

    Inherits from Manim's Tex class to provide native animation support.
    """

    def __init__(self, **kwargs):
        if MANIM_AVAILABLE:
            super().__init__(r"\sum", **kwargs)
        else:
            self.scale = kwargs.get("scale", 1.0)
            self.color = kwargs.get("color", "#FFFFFF")
            self.range_start = kwargs.get("range_start", 1)
            self.range_end = kwargs.get("range_end", 10)


class MatrixGrid(Tex if MANIM_AVAILABLE else object):
    """
    Animated matrix grid for displaying matrix operations.

    Inherits from Manim's Tex class to provide native animation support.
    Displays a customizable matrix with proper LaTeX rendering or fallback styling.
    """

    def __init__(self, rows: int = 3, cols: int = 3, **kwargs):
        """
        Initialize matrix grid.

        Args:
            rows: Number of matrix rows (default: 3)
            cols: Number of matrix columns (default: 3)
            **kwargs: Additional arguments (scale, color, etc.)
        """
        if MANIM_AVAILABLE:
            # Generate matrix LaTeX
            matrix_latex = r"\begin{bmatrix} "
            matrix_latex += " & ".join(["0"] * cols).replace("0", r"\cdot")
            matrix_latex += r" \\ " * (rows - 1)
            matrix_latex += r" \end{bmatrix}"
            super().__init__(matrix_latex, **kwargs)
        else:
            self.rows = rows
            self.cols = cols
            self.cell_size = kwargs.get("cell_size", 0.5)
            self.fill_color = kwargs.get("fill_color", "#f3f4f6")
            self.scale = kwargs.get("scale", 1.0)
            self.color = kwargs.get("color", "#FFFFFF")

    def render(self):
        """Render matrix grid (stub for non-Manim fallback)."""
        pass
