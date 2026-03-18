# 🌿 EfficientManim — Complete Feature Overview

## What is EfficientManim?

EfficientManim is a visual, node-based IDE for creating [Manim](https://www.manim.community/) mathematical animations. Instead of writing Python code by hand, you drag Mobject and Animation nodes onto an infinite canvas, connect them with wires, and the editor generates the correct `Scene.construct()` code automatically. An integrated Gemini AI assistant can generate entire scenes from a plain-English description, and a built-in video renderer turns your node graph into MP4/WebM output in one click.

---

## Architecture Overview

| Module | Responsibility |
|---|---|
| `main.py` | Monolithic entry point: all UI classes, graph logic, compiler, renderer, project I/O |
| `home.py` | Home screen with recent projects and session selection |
| `collab/` | Live collaboration package: WebSocket server/client, delta sync, PIN registry |
| `core/` | Supporting utilities: themes, keybinding registry, autosave, MCP agent, SDK generator |
| `app/` | Extension API, extension manager, plugin registry |

The application runs a single Qt event loop on the main thread. All blocking I/O (AI streaming, TTS generation, WebSocket server/client, video rendering) runs in `QThread` workers or background `asyncio` loops — never on the main thread.

---

## Node-Based Workflow

The canvas is an infinite `QGraphicsScene`. Users place nodes by double-clicking the Elements panel, by dragging from the Class Browser, or by asking the AI to generate them.

Every **NodeItem** wraps a **NodeData** object containing:
- `id` — stable UUID
- `type` — one of `MOBJECT | ANIMATION | PLAY | WAIT | VGROUP`
- `cls_name` — Manim class (e.g. `Circle`, `FadeIn`)
- `params` — key-value dict of constructor arguments
- `param_metadata` — per-parameter enabled/escape flags
- `pos_x`, `pos_y` — canvas position

Nodes are connected with **WireItem** bezier curves. The wire validation layer (`GraphScene.try_connect`) enforces the allowed connection matrix:

```
Mobject  →  Animation  →  Play  →  Wait  →  Play ...
Mobject  →  VGroup     →  Play
Play     →  Play
Wait     →  Play
```

---

## Scene Management

Projects store multiple scenes in `_all_scenes`, a dict mapping scene name → serialized graph state. Switching scenes saves the current graph and loads the target. Each scene is independently compiled to a `GeneratedScene(Scene)` class.

---

## VGroup System

VGroups are first-class canvas nodes (`NodeType.VGROUP`). Connect Mobject nodes to a VGroup node's input socket, then connect the VGroup output to a Play node. The compiler generates `vg_xyz = VGroup(m_aaa, m_bbb)` automatically.

The **VGroups** sidebar tab lets you create, rename, duplicate, and copy VGroup code, and also receives groups from AI-generated code and GitHub snippets.

---

## AI Features

- **Code Generation** — Gemini streams a complete `Scene.construct()` body. The `AINodeIntegrator` parses it into typed nodes with correct wiring.
- **Auto Voiceover** — Gemini analyzes all nodes, writes per-node scripts, generates TTS audio, and attaches it to each node automatically.
- **MCP Agent Mode** — Gemini reads the live scene state as JSON and issues typed commands (`create_node`, `set_node_param`, etc.) that are executed directly against the running app with no merge step.

---

## GitHub Snippet Loader

Repos are cloned with `git clone --depth 1` into `~/.efficientmanim/github_snippets/<user>/<repo>/`. Browsing the tree and double-clicking any `.py` file loads its content into the AI panel where it can be merged into the scene.

---

## Rendering Pipeline

Preview rendering uses `manim -s --format=png -ql` (single-frame, low quality) via `RenderWorker` (a `QThread`). Only Mobject nodes are previewed.

Full video rendering uses `VideoRenderWorker` which writes the compiled code to a temp `.py` file, invokes `manim -qm` (or your chosen quality), and streams progress messages.

---

## Theme System

Light mode only. The `THEME_MANAGER` singleton (in `core/themes.py`) returns a single QSS stylesheet applied globally. Node header colors are hard-coded per `NodeType`.

---

## Keybindings

The `core/keybinding_registry.py` module owns a registry of `action_name → QKeySequence`. Changes are persisted to `QSettings`. The `Help → Edit Keybindings…` dialog applies changes live without restart.

---

## Project Format (.efp)

`.efp` files are ZIP archives containing:
- `metadata.json` — project name, creation date, version
- `graph.json` — nodes and wires
- `code.py` — last compiled code
- `assets/` — copied media files
- `assets.json` — asset manifest with original paths and local filenames

---

## Home Screen

`python home.py` shows the home screen with recently opened projects. Double-clicking a project opens the editor via `EfficientManimWindow.open_project_from_path()`.
