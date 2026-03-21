# Explain System Architecture

## 1. Architecture Overview
The Explain system is a pipeline that turns a live scene into student-friendly language using a deterministic analysis layer and an AI explanation layer.

**Modules and flow**
- `scene_explainer/analyzer.py` — reads the scene graph (live or history snapshots) and produces a structured `SceneAnalysis` object.
- `scene_explainer/prompt_builder.py` — converts `SceneAnalysis` into a prompt tailored to the detected math concepts.
- `scene_explainer/ai_explainer.py` — sends the prompt to the configured AI model and validates a strict JSON response.
- `scene_explainer/ui_panel.py` — presents explanations in the dockable Explain Panel and manages Learning/Teacher Mode triggers.
- `core/mcp.py` — exposes the same engine through MCP commands (no divergence between UI and automation).

**Data flow**
```
Scene (nodes + wires)
   -> SceneAnalyzer
   -> SceneAnalysis (deterministic JSON-safe model)
   -> PromptBuilder
   -> AIExplainer
   -> Explain Panel / MCP response
```

## 2. Scene Analysis Pipeline
The analyzer uses the same snapshot provider as the history system, so it can inspect:
- Live scenes
- Cached scenes
- Selected node subsets
- AI-generated or PDF-imported scenes
- Empty scenes (graceful empty output)

**Example: Axes + function graph + tangent line**

Input scene (simplified):
- `Axes`
- `FunctionGraph`
- `Line` named "tangent"
- `ReplacementTransform` from the graph to a highlighted version

Structured analysis output (excerpt):
```json
{
  "scene_name": "Scene 1",
  "object_count": 3,
  "objects": [
    {"id": "a1", "type": "Axes", "label": null, "role": "primary", "properties": {"category": "Axes & Grids"}},
    {"id": "g1", "type": "FunctionGraph", "label": null, "role": "primary", "properties": {"category": "Graphs"}},
    {"id": "t1", "type": "Line", "label": null, "role": "supporting", "properties": {"category": "Lines"}}
  ],
  "animation_steps": [
    {"step_index": 1, "animation_type": "ReplacementTransform", "targets": ["g1"], "duration": 1.0, "lag_ratio": 0.0}
  ],
  "concept_hints": ["coordinate plane", "function graph", "slope or derivative"]
}
```

The analyzer is deterministic: the same scene state always yields the same `SceneAnalysis`.

## 3. AI Prompt Generation
The prompt builder adapts the prompt based on detected content. It also infers likely concepts when patterns match (for example: axes + curve + tangent line → slope/derivative).

**Example input (simplified)**
```json
{
  "object_count": 3,
  "objects": ["Axes", "FunctionGraph", "Line"],
  "animation_steps": ["ReplacementTransform"],
  "concept_hints": ["slope or derivative"]
}
```

**Generated prompt excerpt**
```
You are a math teacher explaining a visual animation to a student.
Do not mention programming, code, nodes, or library names.
Return ONLY valid JSON with keys:
concept_explanation, step_by_step, visual_reasoning, simple_explanation, key_takeaways, mode_used.

Objects:
- Axes (primary)
- FunctionGraph (primary)
- Line (supporting)

Animation steps:
- Step 1: ReplacementTransform | targets: g1 | duration: 1.0 | lag: 0.0

Concept hints:
- slope or derivative
```

The AI response is parsed and validated before being shown or returned through MCP.

## 4. Learning Mode Logic
Learning Mode provides automatic micro-explanations as students build an animation.

**Triggers**
- Axes, graphs, or equations added
- Transform animations added
- Tangent/slope/area indicators detected
- Major structural changes or large batch additions
- Checkpoints created

**Debouncing and queueing**
- Triggers are debounced at 1.5 seconds
- If an explanation is in progress, new triggers are queued
- Multiple triggers inside the debounce window are coalesced into one explanation

**Auto-explanation format**
```
What just happened: [one sentence]
Why it matters: [one or two sentences]
```

## 5. Teacher Mode Logic
Teacher Mode generates complete lesson notes from the current scene.

**Lesson structure**
```
# Lesson: [Concept Name]

## Concept Being Explained
...

## Visual Explanation Using the Animation
...

## Step-by-Step Teaching Explanation
...

## Student-Friendly Notes
...

## Key Takeaways
- ...
```

**Worked example (excerpt)**
```
# Lesson: Slope as a Derivative

## Concept Being Explained
The derivative gives the slope of a function at a specific point.

## Visual Explanation Using the Animation
The graph is drawn on axes, then a tangent line appears at one point to show the local slope.
```

## 6. History-Powered Explanations
History explanations use checkpoint snapshots to describe:
- What a past scene looked like
- What changed between two checkpoints
- The educational impact of undo/redo actions

The diff engine identifies object and animation changes, then the AI converts them into student-friendly language.

## 7. Extending the System

**Add a new object type**
1. Open `scene_explainer/analyzer.py`.
2. Add the class name to the appropriate set (e.g., `SHAPE_TYPES`, `LINE_TYPES`).
3. If needed, add a new category in `_classify`.

**Add a new explanation mode**
1. Add a new mode branch in `PromptBuilder.build_explain_prompt`.
2. Update the UI toggle in `scene_explainer/ui_panel.py`.
3. Ensure `ExplainResponse.mode_used` accepts the new mode.

**Add a new MCP command**
1. Add a handler in `core/mcp.py` using `@self._register("your_command")`.
2. Call the shared `ExplainService` so UI and MCP outputs match.
3. Update `docs/mcp_commands.md` with request/response examples.

**Customize the prompt for a subject area**
1. Edit `PromptBuilder._build_focus_lines` and `_infer_additional_hints`.
2. Add subject-specific hint rules and keep them grounded in detected objects.
3. Avoid references to code or implementation details.
