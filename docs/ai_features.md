# 🤖 EfficientManim — AI Features Guide

## Overview

EfficientManim integrates Google Gemini across four major AI workflows: code generation, an autonomous MCP agent, a voiceover studio with Auto Voiceover, and a TTS engine with per-node transcript management. All AI calls run in background `QThread` workers so the UI stays responsive.

---

## Gemini Code Generation

### How It Works

1. Type a description in the **AI Assistant** dock on the left.
2. Click **Generate Code**.
3. Gemini streams back a complete `Scene.construct()` Python block.
4. When streaming finishes, click **Merge to Scene** to parse the code into nodes, or **Reject** to discard.

### Writing Good Prompts

- Be specific about shapes, colors, and sequences:
  > *"Create a blue circle that fades in, then transforms into a red square, then fades out. Add a white MathTex label 'E=mc²' that writes itself in below the shapes."*

- Name the animation sequence explicitly:
  > *"First FadeIn the circle, wait 1 second, then Rotate it 180 degrees, then FadeOut."*

- Specify timing if it matters:
  > *"Each animation should take exactly 1.5 seconds."*

### What Gets Generated

The AI produces a properly structured `class MyScene(Scene):` block with:
- Mobject instantiations with all properties
- `self.play()` calls in logical sequence
- `self.wait()` pauses between beats
- Inline comments explaining each step

### After Merging

`AINodeIntegrator.merge_ai_code_to_scene()` parses the code and creates:
- Typed **Mobject** nodes for every `var = ClassName(...)` assignment
- Typed **Animation** nodes for every `self.play(AnimClass(...))` call
- Wires connecting each animation to its target mobject
- Nodes laid out in columns: Mobjects left, Animations right

All AI-generated nodes are marked with a ✨ badge in the Properties panel and stored with `is_ai_generated = True`.

### Model Selection

Go to **File → Settings → Code Model** to choose between:
- `gemini-3-flash-preview` — fast, good for most scenes
- `gemini-3-pro-preview` — slower, better for complex multi-step scenes

---

## MCP Agent Mode

Enable the **MCP Agent Mode** checkbox in the AI panel before clicking Generate.

Instead of generating Python code, Gemini reads the live scene state and outputs a JSON array of typed commands executed directly against the running application.

### How It Works

1. The app captures the full current scene as `MCPContext` JSON (all nodes, params, assets, scene name).
2. Gemini receives: the scene context + available commands + your instruction.
3. Gemini responds with a JSON array like:
```json
[
  {"command": "create_node", "payload": {"cls_name": "Circle", "name": "my_circle", "node_type": "MOBJECT", "x": 100, "y": 100}},
  {"command": "set_node_param", "payload": {"node_name": "my_circle", "key": "color", "value": "#3498db"}},
  {"command": "compile_graph", "payload": {}}
]
```
4. Each command is executed immediately. Results stream into the AI response panel (✅ / ❌ per command).

### Available MCP Commands

| Command | Payload |
|---|---|
| `create_node` | `cls_name, name, node_type, params, x, y` |
| `set_node_param` | `node_id, key, value` |
| `rename_node` | `node_id, name` |
| `delete_node` | `node_id, confirm: true` |
| `attach_voiceover` | `node_id, audio_path, transcript, duration` |
| `remove_voiceover` | `node_id` |
| `select_node` | `node_id` |
| `switch_tab` | `tab` (partial name ok) |
| `compile_graph` | `{}` |
| `trigger_render` | `node_id` (optional) |
| `save_project` | `{}` |
| `switch_scene` | `scene_name` |
| `get_context` | `{}` → returns scene JSON |
| `list_commands` | `{}` → returns command list |
| `ping` | `{}` → returns node count |

### Inspecting MCP State

Use **Help → MCP Agent** submenu:
- **MCP Status / Ping** — shows node count and command count
- **Inspect Scene Context (JSON)** — full MCPContext as formatted JSON
- **List All Commands** — all registered command names
- **Show Action Log** — every command executed in this run

---

## Auto Voiceover Agent

Enable the **Enable Auto Voiceover** checkbox in the AI panel before clicking Generate.

### What It Does

1. Gemini receives a summary of every node in the scene.
2. It writes a 1–2 sentence voiceover script for each node.
3. TTS audio is generated for each script via `TTSWorker`.
4. Audio is attached to the corresponding node (`audio_asset_id`, `voiceover_transcript`, `voiceover_duration`).
5. At render time, the compiler merges all audio segments into a single WAV track and injects `self.add_sound(...)` into the generated code.

### Output

After completion:
- Each node shows a voiceover transcript in the Voiceover panel.
- The project is marked as modified.
- The generated code (when compiled) includes a merged voiceover track synchronized to the animation timeline.

---

## TTS Voiceover Studio

The **Voiceover** tab lets you generate, preview, and attach TTS audio to individual nodes.

### Workflow

1. Select a target node from the dropdown (any node type is supported).
2. Write your script in the text box.
3. Choose a voice: Puck, Charon, Kore, Fenrir, Aoede, or Zephyr.
4. Click **Generate Audio** — TTS streams via `gemini-2.5-flash-preview-tts`.
5. Preview the audio with the built-in player (play, pause, stop, seek).
6. Click **Add to Animation Node** to attach.

### Audio Player Controls

- ▶ / ⏸ — Play / Pause
- ⏹ — Stop and reset position
- Seek slider — drag to any position
- Time display — current position / total duration

### Per-Node Voiceover in Code

When a node has an attached voiceover, the compiler:
1. Reads the audio file duration via `pydub`.
2. Uses that duration as `run_time=` for the animation.
3. Places the audio segment at the correct timeline offset.
4. Merges all segments into `<temp>/merged_vo_XXXX.wav`.
5. Injects `self.add_sound(r'path/to/merged_vo.wav')` at the top of `construct()`.

### TTS Model Selection

Go to **File → Settings → TTS Model**:
- `gemini-2.5-flash-preview-tts` — fast, good quality
- `gemini-2.5-pro-preview-tts` — slower, highest quality

---

## LaTeX Studio

The **✒️ LaTeX** tab provides live LaTeX preview via the MathPad API and one-click application to any text node.

1. Type LaTeX in the editor (auto-renders after 800ms debounce).
2. Preview renders in the panel.
3. Select a `MathTex` or `Text` node from the dropdown.
4. Click **✅ Apply to Node** — the LaTeX string is injected with proper `r"""..."""` raw-string formatting and the `Escape` flag set so quotes are not added.
