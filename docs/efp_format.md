# 📦 EfficientManim — Project Format (.efp)

## What Is an .efp File?

A `.efp` file (EfficientManim Project) is a renamed ZIP archive containing everything needed to reproduce your project on any machine: the node graph, compiled code, all referenced media assets, and project metadata.

The file extension is `.efp` but it is a standard ZIP. Any ZIP tool (WinRAR, 7-Zip, `unzip`, `zipfile` in Python) can open it.

---

## Internal Structure

```
my_project.efp  (ZIP)
├── metadata.json      Project name, creation date, app version
├── graph.json         All nodes, wires, and their properties
├── code.py            Last compiled Scene.construct() code
├── assets/            All referenced media files (images, audio, video)
│   ├── <asset_id_1>.png
│   ├── <asset_id_2>.wav
│   └── ...
└── assets.json        Asset manifest (id → original path + local filename)
```

---

## File Descriptions

### `metadata.json`

```json
{
  "name": "My Animation",
  "created": "2026-03-18 14:30:00",
  "version": "2.0.4"
}
```

- `name` — project display name (matches the `.efp` stem)
- `created` — ISO timestamp of when Save was first invoked
- `version` — EfficientManim version that created the file

### `graph.json`

```json
{
  "nodes": [
    {
      "id": "a1b2c3d4-...",
      "name": "Circle",
      "var_name": "Circle",
      "type": "MOBJECT",
      "cls_name": "Circle",
      "params": {"radius": 1.5, "color": "#3498db", "fill_opacity": 1.0},
      "param_metadata": {"radius": {"enabled": true, "escape": false}},
      "pos": [120.0, 80.0],
      "preview_path": null,
      "audio_asset_id": null,
      "voiceover_transcript": null,
      "voiceover_duration": 0.0,
      "is_ai_generated": false,
      "ai_source": null,
      "ai_code_snippet": null
    }
  ],
  "wires": [
    {
      "start": "a1b2c3d4-...",
      "end": "e5f6g7h8-..."
    }
  ]
}
```

**Node fields:**

| Field | Type | Description |
|---|---|---|
| `id` | UUID string | Stable unique identifier |
| `name` | string | Display name shown on canvas |
| `var_name` | string | Python variable name in generated code |
| `type` | string | `MOBJECT \| ANIMATION \| PLAY \| WAIT \| VGROUP` |
| `cls_name` | string | Manim class name (e.g. `Circle`, `FadeIn`) |
| `params` | dict | Constructor arguments key → value |
| `param_metadata` | dict | Per-param `{enabled, escape}` flags |
| `pos` | `[x, y]` | Canvas position in scene coordinates |
| `audio_asset_id` | UUID or null | ID of attached voiceover asset |
| `voiceover_transcript` | string or null | TTS script text |
| `voiceover_duration` | float | Audio duration in seconds |
| `is_ai_generated` | bool | True if node came from AI merge |

**Wire fields:**

| Field | Description |
|---|---|
| `start` | Node ID of the output (source) socket's parent |
| `end` | Node ID of the input (destination) socket's parent |

### `assets.json`

```json
[
  {
    "id": "a1b2c3d4-e5f6-...",
    "name": "logo.png",
    "original": "/home/user/images/logo.png",
    "kind": "image",
    "local": "a1b2c3d4-e5f6-....png"
  }
]
```

- `original` — path on the machine that saved the project (may not exist elsewhere)
- `local` — filename inside the `assets/` folder within the ZIP
- `kind` — `image | video | audio`

On load, the app extracts assets to a temp directory and sets `current_path` to the extracted location. If `original` also exists on the loading machine, that is used preferentially.

### `code.py`

The last compiled Python scene code. This is regenerated each time the graph is compiled, but is stored here so a project can be rendered even if the loading machine does not have EfficientManim installed — just run `manim code.py GeneratedScene`.

---

## Manually Inspecting an .efp File

### Python

```python
import zipfile, json

with zipfile.ZipFile("my_project.efp", "r") as z:
    meta = json.loads(z.read("metadata.json"))
    graph = json.loads(z.read("graph.json"))
    code = z.read("code.py").decode("utf-8")
    print(f"Project: {meta['name']}, Nodes: {len(graph['nodes'])}")
```

### Command Line

```bash
unzip -p my_project.efp graph.json | python3 -m json.tool
```

---

## Adding New Data (Additive-Only Policy)

All new features that need to persist data **add new optional keys** to existing structures with safe defaults on load. The load path uses `.get()` with fallbacks everywhere, so old `.efp` files always load cleanly in newer app versions.

Never rename or remove existing keys — this would break backward compatibility for every existing project file.

---

## Backward Compatibility

| App Version | Notes |
|---|---|
| 2.0.5+ | Current format. `var_name`, `param_metadata`, and AI metadata are optional and backward-compatible. |
| Pre-2.0 | Earlier formats may lack `var_name`, `param_metadata`, `is_ai_generated`. All handled by `.get()` fallbacks in `NodeData.from_dict()`. |
