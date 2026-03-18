# 📦 EfficientManim — Node Reference

## Node Types

### Mobject Node (blue header)
Represents a Manim `Mobject` subclass. Instantiated in `construct()` with `self.add()`.

**Connection rules:**
- Output → Animation node
- Output → VGroup node

### Animation Node (purple header)
Represents a Manim `Animation` subclass. Wrapped in `self.play()`.

**Connection rules:**
- Input from Mobject (target)
- Output → Play node

### Play Node (green header, ▶)
Explicit `self.play()` call. Aggregates one or more Animation node outputs.

**Connection rules:**
- Input from Animation or VGroup
- Output → Wait or Play (for chaining)

### Wait Node (orange header, ⏱)
Explicit `self.wait(duration)` call.

**Properties:** `duration` (float, seconds)

**Connection rules:**
- Input from Play
- Output → Play

### VGroup Node (teal header, 🔗)
Groups multiple Mobjects into a `VGroup`.

**Connection rules:**
- Input from Mobject (members)
- Output → Play

---

## Common Mobjects

### Circle
| Property | Type | Default | Notes |
|---|---|---|---|
| `radius` | float | `1.0` | Radius in Manim units |
| `color` | color | `WHITE` | Stroke color |
| `fill_color` | color | transparent | Fill color |
| `fill_opacity` | float | `0.0` | 0.0–1.0 |
| `stroke_width` | float | `4.0` | |

### Rectangle / Square
| Property | Type | Default |
|---|---|---|
| `width` | float | `4.0` |
| `height` | float | `2.0` |
| `color` | color | `WHITE` |
| `fill_opacity` | float | `0.0` |

### Line / Arrow / DoubleArrow
| Property | Type | Notes |
|---|---|---|
| `start` | point | e.g. `LEFT` |
| `end` | point | e.g. `RIGHT` |
| `color` | color | |
| `stroke_width` | float | |

### Text
| Property | Type | Notes |
|---|---|---|
| `text` | string | The text content |
| `font_size` | float | Default 48 |
| `color` | color | |
| `font` | string | System font name |

### MathTex / Tex
| Property | Type | Notes |
|---|---|---|
| `arg0` | string (LaTeX) | Main LaTeX string, e.g. `r"E = mc^2"` |
| `color` | color | |
| `font_size` | float | |

Use the **LaTeX** tab to preview and apply LaTeX expressions to MathTex nodes.

### Axes
| Property | Type | Notes |
|---|---|---|
| `x_range` | list | `[-5, 5, 1]` |
| `y_range` | list | `[-3, 3, 1]` |

---

## Common Animations

### FadeIn / FadeOut
| Property | Type | Notes |
|---|---|---|
| (target) | node ref | Set via Properties panel |
| `run_time` | float | Seconds |
| `shift` | point | Optional shift direction |

### Write / Unwrite
Writes text character by character.

### Create / Uncreate
Draws border then fill.

### Transform / ReplacementTransform
| Property | Type | Notes |
|---|---|---|
| (target) | Mobject | Source |
| `mobject` | Mobject | Target shape |

### Rotate
| Property | Type | Notes |
|---|---|---|
| `angle` | float | Radians |
| `axis` | point | Default `OUT` (z-axis) |
| `run_time` | float | |

### Scale / ScaleInPlace
| Property | Type | Notes |
|---|---|---|
| `scale_factor` | float | e.g. `2.0` |
| `run_time` | float | |

---

## Parameter Panel Features

Every parameter row has two checkboxes:

- **Enabled** — when unchecked, the parameter is excluded from code generation entirely. Useful for optional parameters you don't want to specify.
- **Escape** — when checked, the string value is inserted without quotes (for Manim constants like `BLUE`, numpy expressions, or raw string literals).

For color parameters, clicking the colored button opens a color picker. The hex value is stored and formatted as a Manim-compatible string.

For asset parameters (file paths), a dropdown lists all assets registered in the Assets tab.
