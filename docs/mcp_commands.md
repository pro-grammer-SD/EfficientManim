# MCP Commands Reference — Explain, Learning, Teacher, History

This document describes the new MCP commands added for the Explain system, Learning Mode, Teacher Mode, and History-powered explanations. Each command below includes:
1. Description
2. Input JSON
3. Output JSON
4. Example request
5. Example response
6. Recommended use case

All commands return an `MCPResult` wrapper. The payloads shown below are the **`data`** field of a successful response.

---

## explain.scene
**Description**: Explains the currently active full scene.

**Input JSON**
```json
{ "mode": "detailed | simple" }
```

**Output JSON**
```json
{
  "scene_name": "string",
  "concept_explanation": "string",
  "step_by_step": "string",
  "visual_reasoning": "string",
  "simple_explanation": "string",
  "key_takeaways": ["string"]
}
```

**Example request**
```json
{"command": "explain.scene", "payload": {"mode": "detailed"}}
```

**Example response**
```json
{
  "scene_name": "Scene 1",
  "concept_explanation": "The animation introduces the idea of slope by showing a curve on axes and highlighting a tangent line.",
  "step_by_step": "First the axes and curve appear, then the tangent line is drawn at a point on the graph.",
  "visual_reasoning": "The tangent line touches the curve at one point, making the slope visually obvious.",
  "simple_explanation": "It shows how steep the curve is at one spot.",
  "key_takeaways": ["A tangent line shows slope", "Slope is a local property", "Graphs make change visible"]
}
```

**Recommended use case**: When an AI agent needs a full conceptual summary of the current scene.

---

## explain.selected_nodes
**Description**: Explains a specific subset of nodes.

**Input JSON**
```json
{ "node_ids": ["string"], "mode": "detailed | simple" }
```

**Output JSON**: Same schema as `explain.scene`.

**Example request**
```json
{"command": "explain.selected_nodes", "payload": {"node_ids": ["n1", "n2"], "mode": "simple"}}
```

**Example response**
```json
{
  "scene_name": "Scene 1",
  "concept_explanation": "These elements introduce the axes and the curve we are studying.",
  "step_by_step": "The axes appear first, then the curve is drawn on them.",
  "visual_reasoning": "Placing the curve on axes shows how its values change across x.",
  "simple_explanation": "It sets up the graph so you can read it easily.",
  "key_takeaways": ["Axes give scale", "Graphs show change"]
}
```

**Recommended use case**: When analyzing a specific animation sub-sequence.

---

## explain.selected_animation
**Description**: Explains a single animation node or animation step by ID.

**Input JSON**
```json
{ "animation_id": "string", "mode": "detailed | simple" }
```

**Output JSON**: Same schema as `explain.scene`.

**Example request**
```json
{"command": "explain.selected_animation", "payload": {"animation_id": "a123", "mode": "detailed"}}
```

**Example response**
```json
{
  "scene_name": "Scene 1",
  "concept_explanation": "This step focuses attention on how one shape maps to another.",
  "step_by_step": "The first shape fades into the second at the same position.",
  "visual_reasoning": "The visual swap makes the relationship between the two shapes clear.",
  "simple_explanation": "One shape turns into another.",
  "key_takeaways": ["Transformations compare shapes", "Visual changes carry meaning"]
}
```

**Recommended use case**: When an agent needs to isolate and explain a single animation step.

---

## explain.regenerate
**Description**: Regenerates the explanation using the current scene state.

**Input JSON**
```json
{ "mode": "detailed | simple" }
```

**Output JSON**: Same schema as `explain.scene`.

**Example request**
```json
{"command": "explain.regenerate", "payload": {"mode": "detailed"}}
```

**Example response**
```json
{
  "scene_name": "Scene 1",
  "concept_explanation": "The updated animation now introduces area under a curve.",
  "step_by_step": "The curve is drawn, then a shaded region appears beneath it.",
  "visual_reasoning": "Shading the region makes area accumulation visible.",
  "simple_explanation": "It shows how much space is under the curve.",
  "key_takeaways": ["Area can be visualized", "Shading makes totals clear"]
}
```

**Recommended use case**: After edits, undo, or redo, to refresh explanations.

---

## explain.history_checkpoint
**Description**: Explains what the scene looked like at a specific history checkpoint.

**Input JSON**
```json
{ "checkpoint_id": "string", "mode": "detailed | simple" }
```

**Output JSON**
```json
{
  "checkpoint_id": "string",
  "concept_explanation": "string",
  "step_by_step": "string",
  "visual_reasoning": "string",
  "simple_explanation": "string"
}
```

**Example request**
```json
{"command": "explain.history_checkpoint", "payload": {"checkpoint_id": "Graph Added", "mode": "simple"}}
```

**Example response**
```json
{
  "checkpoint_id": "Graph Added",
  "concept_explanation": "The scene introduces the graph on axes to show how the function behaves.",
  "step_by_step": "Axes appear, then the curve is drawn.",
  "visual_reasoning": "The graph connects x-values to y-values visually.",
  "simple_explanation": "It shows what the function looks like."
}
```

**Recommended use case**: Explaining a past scene state for revision or lesson playback.

---

## explain.history_change
**Description**: Explains what changed between two checkpoints in student-friendly language.

**Input JSON**
```json
{ "from_checkpoint": "string", "to_checkpoint": "string", "mode": "detailed | simple" }
```

**Output JSON**
```json
{
  "objects_added": ["string"],
  "objects_removed": ["string"],
  "animations_added": ["string"],
  "animations_removed": ["string"],
  "concept_change_summary": "string",
  "educational_significance": "string"
}
```

**Example request**
```json
{"command": "explain.history_change", "payload": {"from_checkpoint": "Graph Added", "to_checkpoint": "Tangent Line Added", "mode": "detailed"}}
```

**Example response**
```json
{
  "objects_added": ["tangent_line"],
  "objects_removed": [],
  "animations_added": ["DrawBorderThenFill"],
  "animations_removed": [],
  "concept_change_summary": "Introduced: slope or derivative",
  "educational_significance": "The new tangent line makes the idea of slope at a point visible, which is the core of the derivative."
}
```

**Recommended use case**: Summarizing conceptual progress between two editing sessions.

---

## explain.undo_action
**Description**: Explains what the most recent undo operation removed.

**Input JSON**
```json
{ "mode": "detailed | simple" }
```

**Output JSON**
```json
{ "action_description": "string", "educational_impact": "string" }
```

**Example request**
```json
{"command": "explain.undo_action", "payload": {"mode": "simple"}}
```

**Example response**
```json
{
  "action_description": "The tangent line was removed from the graph.",
  "educational_impact": "Without the tangent line, the animation no longer highlights the slope at that point."
}
```

**Recommended use case**: Immediately after an undo event.

---

## explain.redo_action
**Description**: Explains what the most recent redo operation restored.

**Input JSON**
```json
{ "mode": "detailed | simple" }
```

**Output JSON**: Same as `explain.undo_action`.

**Example request**
```json
{"command": "explain.redo_action", "payload": {"mode": "simple"}}
```

**Example response**
```json
{
  "action_description": "The shaded region under the curve was restored.",
  "educational_impact": "This brings back the visual idea of area accumulation."
}
```

**Recommended use case**: Immediately after a redo event.

---

## learning_mode.enable
**Description**: Enables Learning Mode globally.

**Input JSON**
```json
{}
```

**Output JSON**
```json
{ "enabled": true, "status": "Learning Mode is now active" }
```

**Example request**
```json
{"command": "learning_mode.enable", "payload": {}}
```

**Example response**
```json
{ "enabled": true, "status": "Learning Mode is now active" }
```

**Recommended use case**: Turn on auto-explanations for student workflows.

---

## learning_mode.disable
**Description**: Disables Learning Mode globally.

**Input JSON**
```json
{}
```

**Output JSON**
```json
{ "enabled": false, "status": "Learning Mode has been disabled" }
```

**Example request**
```json
{"command": "learning_mode.disable", "payload": {}}
```

**Example response**
```json
{ "enabled": false, "status": "Learning Mode has been disabled" }
```

**Recommended use case**: Reduce distraction during focused editing.

---

## learning_mode.status
**Description**: Returns the current state of Learning Mode.

**Input JSON**
```json
{}
```

**Output JSON**
```json
{ "enabled": true }
```

**Example request**
```json
{"command": "learning_mode.status", "payload": {}}
```

**Example response**
```json
{ "enabled": true }
```

**Recommended use case**: Checking the UI state from an AI agent.

---

## teacher_mode.enable
**Description**: Enables Teacher Mode globally.

**Input JSON**
```json
{}
```

**Output JSON**
```json
{ "enabled": true, "status": "Teacher Mode is now active" }
```

**Example request**
```json
{"command": "teacher_mode.enable", "payload": {}}
```

**Example response**
```json
{ "enabled": true, "status": "Teacher Mode is now active" }
```

**Recommended use case**: Enable automatic lesson-note generation.

---

## teacher_mode.disable
**Description**: Disables Teacher Mode globally.

**Input JSON**
```json
{}
```

**Output JSON**
```json
{ "enabled": false, "status": "Teacher Mode has been disabled" }
```

**Example request**
```json
{"command": "teacher_mode.disable", "payload": {}}
```

**Example response**
```json
{ "enabled": false, "status": "Teacher Mode has been disabled" }
```

**Recommended use case**: Turn off auto lesson generation for performance.

---

## teacher_mode.generate_lesson
**Description**: Generates structured lesson notes from the current scene.

**Input JSON**
```json
{ "scene": "current", "format": "structured" }
```

**Output JSON**
```json
{
  "concept_explanation": "string",
  "visual_explanation": "string",
  "step_by_step_teaching": "string",
  "student_notes": "string",
  "key_takeaways": ["string"]
}
```

**Example request**
```json
{"command": "teacher_mode.generate_lesson", "payload": {"scene": "current", "format": "structured"}}
```

**Example response**
```json
{
  "concept_explanation": "The derivative gives the slope at a point on a curve.",
  "visual_explanation": "A tangent line appears at the chosen point to show the local slope.",
  "step_by_step_teaching": "Introduce the axes, draw the curve, then add the tangent line and explain its meaning.",
  "student_notes": "A tangent line touches the curve at one point and shows how steep it is.",
  "key_takeaways": ["Slope is local", "Tangent lines show derivatives", "Graphs reveal change"]
}
```

**Recommended use case**: Auto-generate teacher scripts and revision notes.

---

## app.quit
**Description**: Safely closes the application, honoring the unsaved-changes dialog.

**Input JSON**
```json
{}
```

**Output JSON**
```json
{ "status": "success", "message": "Application closed safely" }
```

**Example request**
```json
{"command": "app.quit", "payload": {}}
```

**Example response**
```json
{ "status": "success", "message": "Application closed safely" }
```

**Recommended use case**: Cleanly shutting down the app at the end of an automated workflow.
