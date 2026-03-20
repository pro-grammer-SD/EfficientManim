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
  ai_pdf_upload_widget.py PDF attachment UI for the AI Assistant tab
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
  ai_slides_to_nodes.py Slide plan to node graph conversion

rendering/
  render_manager.py  Worker threads for preview + video render
  preview.py         Preview helpers
  export.py          Export helpers

core/
  project_manager.py Project lifecycle coordinator
  file_manager.py    Recents + asset management
  history_manager.py Undo/redo stack
  config.py          App config + settings
  ai_slides_to_manim.py Slide plan to Manim scene generator

utils/
  shortcuts.py       Keybinding helpers
  tooltips.py        Consistent tooltip builder
  helpers.py         Parsing + AI helpers
  logger.py          Application logging
  ai_pdf_attachment_manager.py PDF attachment state
  ai_pdf_parser.py   PDF text and structure extraction
  ai_context_builder.py Structured AI context builder
  ai_slide_animator.py AI JSON slide/animation planner
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

## AI PDF Animation Pipeline
1. `ui/ai_pdf_upload_widget.py` embeds PDF attachment controls in the AI Assistant tab and drives the pipeline.
2. `utils/ai_pdf_attachment_manager.py` stores ordered PDF paths and supports add/remove/clear.
3. `utils/ai_pdf_parser.py` reads PDFs with `pdfplumber`, extracting headings, bullets, equations, and page order.
4. `utils/ai_context_builder.py` compacts slide data and merges it with the user prompt into structured context.
5. `utils/ai_slide_animator.py` calls the AI API using `GEMINI_API_KEY` and returns a JSON slide plan.
6. `core/ai_slides_to_manim.py` converts the plan into runnable Manim scenes.
7. `graph/ai_slides_to_nodes.py` creates node groups, animations, and wiring per slide, then loads them into the editor.

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
