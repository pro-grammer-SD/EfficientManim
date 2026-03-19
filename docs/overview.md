# EfficientManim — Overview

EfficientManim is a visual, node-based IDE for creating Manim animations. You build scenes by connecting Mobject and Animation nodes on a canvas; the editor generates clean `Scene.construct()` code, previews nodes, and renders video output.

## Architecture Summary
- `app/` handles startup and window creation.
- `ui/` contains the main window, home screen, menus, toolbars, dialogs, and panels.
- `graph/` contains node, edge, scene, and layout logic.
- `rendering/` handles preview and video render workers.
- `core/` stores configuration, history, assets, and project helpers.
- `utils/` provides tooltips, logging, shortcuts, and parsing helpers.

## Node-Based Workflow
- Nodes are placed on a `QGraphicsScene` and connected via wires.
- `GraphScene.try_connect()` enforces valid wiring rules.
- The compiler generates a `GeneratedScene(Scene)` from the graph.

## Scene Management
Multiple scenes are stored in `_all_scenes`. Switching scenes serializes the current graph and loads the target scene state.

## Rendering Pipeline
- Previews are single-frame renders handled by `RenderWorker`.
- Full video rendering uses `VideoRenderWorker` with selected quality and resolution.

## Home Screen
The home screen (`ui/home.py`) provides quick actions for new/open projects, recent projects, documentation, and an example project starter.
