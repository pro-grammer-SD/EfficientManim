"""
Utility module for type safety, validation, and font management
Provides robust handling of Qt objects and parameters
"""

from PySide6.QtGui import QFont, QColor, QPixmap
from PySide6.QtCore import QPointF, QSize
from typing import Optional, Union, Any
import logging

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# COLOR VALIDATION
# ═══════════════════════════════════════════════════════════════════


class ColorValidator:
    """Safe color validation and conversion."""

    @staticmethod
    def to_qcolor(value: Union[str, QColor, None]) -> Optional[QColor]:
        """
        Safely convert value to QColor.

        Args:
            value: Color string, QColor, or None

        Returns:
            QColor or None if invalid
        """
        if value is None:
            return None

        if isinstance(value, QColor):
            return value if value.isValid() else None

        if isinstance(value, str):
            color = QColor(value)
            return color if color.isValid() else None

        return None

    @staticmethod
    def to_hex(value: Union[str, QColor, None], default: str = "#FFFFFF") -> str:
        """Convert color to hex string."""
        color = ColorValidator.to_qcolor(value)
        return color.name() if color else default


class ImageValidator:
    """Safe image loading and validation."""

    @staticmethod
    def load_image(path: str, max_size: Optional[QSize] = None) -> Optional[QPixmap]:
        """
        Safely load and validate image.

        Args:
            path: Image file path
            max_size: Maximum size to scale to

        Returns:
            QPixmap or None if load fails
        """
        try:
            if not path or not isinstance(path, str):
                return None

            pixmap = QPixmap(path)

            if pixmap.isNull():
                logger.warning(f"Image is null: {path}")
                return None

            if max_size and (
                pixmap.width() > max_size.width() or pixmap.height() > max_size.height()
            ):
                pixmap = pixmap.scaledToWidth(max_size.width(), mode=0)  # pyright: ignore[reportArgumentType] # SmoothTransformation

            return pixmap
        except Exception as e:
            logger.error(f"Image load error: {e}")
            return None


class PointValidator:
    """Safe point validation for scene coordinates."""

    @staticmethod
    def validate_point(point: Any) -> Optional[QPointF]:
        """
        Validate point object.

        Args:
            point: Any point-like object

        Returns:
            QPointF or None if invalid
        """
        try:
            if point is None:
                return None

            if isinstance(point, QPointF):
                return point

            if hasattr(point, "x") and hasattr(point, "y"):
                return QPointF(float(point.x()), float(point.y()))

            return None
        except (ValueError, TypeError, AttributeError):
            logger.warning(f"Invalid point: {point}")
            return None

    @staticmethod
    def create_point(x: float, y: float) -> QPointF:
        """Safely create a point."""
        return QPointF(float(x), float(y))


class StateValidator:
    """Validate application state."""

    @staticmethod
    def is_valid_uuid(value: str) -> bool:
        """Check if value looks like a UUID."""
        if not isinstance(value, str):
            return False
        return len(value) == 36 and value.count("-") == 4

    @staticmethod
    def is_valid_hex_color(value: str) -> bool:
        """Check if value is a valid hex color."""
        if not isinstance(value, str):
            return False
        if not value.startswith("#") or len(value) != 7:
            return False
        try:
            int(value[1:], 16)
            return True
        except ValueError:
            return False

    @staticmethod
    def is_valid_numeric(value: Any) -> bool:
        """Check if value can be converted to number."""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False


class SafeDict(dict):
    """Dictionary with safe access."""

    def get_safe(
        self, key: str, default: Any = None, expected_type: type = None
    ) -> Any:  # pyright: ignore[reportArgumentType]
        """
        Safely get value with type checking.

        Args:
            key: Key to retrieve
            default: Default value if missing
            expected_type: Expected type (optional)

        Returns:
            Value or default
        """
        value = self.get(key, default)

        if expected_type and value is not None:
            if not isinstance(value, expected_type):
                logger.warning(
                    f"Type mismatch for {key}: expected {expected_type}, got {type(value)}"
                )
                return default

        return value
    