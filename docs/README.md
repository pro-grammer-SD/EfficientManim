# EfficientManim Docs

## Project Structure
```
app/
  main.py            Entry point (creates app + home screen)
  app_window.py      Creates the main editor window
  startup.py         App bootstrapping (logging, icons, styles, hooks)

ui/
  main_window.py     Main editor window implementation
  home.py            Home screen (new/open/recent/docs/example)
  toolbar.py         Quick export toolbar
  menus.py           Menu bar builder
  dialogs.py         Settings and shortcuts dialogs
  panels/
    node_panel.py    Node, AI, assets, voiceover, outliner panels
    render_panel.py  Render + output panels
    settings_panel.py Settings form widget

graph/
  graph_editor.py    Graph scene/view and connection rules
  node.py            Node data model + node graphics
  edge.py            Wire graphics
  layout.py          Auto-layout helpers
  node_factory.py    Node creation helpers

rendering/
  render_manager.py  Worker threads for preview + video render
  preview.py         Preview helpers
  export.py          Export helpers

core/
  project_manager.py Project lifecycle coordinator
  file_manager.py    Recents + asset management
  history_manager.py Undo/redo stack
  config.py          App config + settings

utils/
  shortcuts.py       Keybinding helpers
  tooltips.py        Consistent tooltip builder
  helpers.py         Parsing + AI helpers
  logger.py          Application logging
```

## How The App Starts
1. `main.py` calls `app.main.main()`.
2. `app.startup.create_application()` configures logging, icons, Qt rules, and exception hooks.
3. The home screen is shown first. Selecting an action opens the main editor window.
4. The editor window is created by `app.app_window.create_main_window()`.

## Home Screen
`ui/home.py` provides the home screen UI with:
- New and Open project actions
- Recent projects list
- Documentation button
- Example project button
- Getting Started guidance

## Graph System
The graph system is defined in `graph/`:
- `node.py` contains `NodeData` and `NodeItem` (visual node).
- `edge.py` contains `WireItem` (connection).
- `graph_editor.py` provides `GraphScene` and `GraphView` plus validation rules.
- `node_factory.py` centralizes node creation.
- `layout.py` provides auto-layout helpers used by the main window.

## Rendering
Rendering is handled in `rendering/`:
- `render_manager.py` includes preview and video render worker threads.
- `preview.py` provides lightweight preview helpers.
- `export.py` contains code export helpers.
- `ui/panels/render_panel.py` provides the UI to control rendering and preview playback.

## Where To Add New Features
- UI widgets: add to `ui/panels/` or `ui/dialogs.py`.
- Graph logic: add to `graph/`.
- Rendering features: add to `rendering/`.
- App-level orchestration: add to `ui/main_window.py`.
- Shared helpers: add to `utils/`.

## Beginner Workflow
1. Start the app and create a new project.
2. Add Mobjects and Animations from the Elements tab.
3. Connect nodes to define ordering.
4. Preview nodes to validate changes.
5. Render the full scene in the Video tab.
6. Export Python code if needed.
