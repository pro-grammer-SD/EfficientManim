# MCP Commands Reference

This document lists the MCP commands exposed by EfficientManim, grouped by feature area. All commands return an `MCPResult` with:

```json
{
  "success": true,
  "data": { "status": "success", "affected_nodes": [], "history_pointer": 0, "metadata": {} },
  "error": "",
  "command": "..."
}
```

Legacy commands may return a simpler `data` payload (e.g., `{ "id": "...", "name": "..." }`). New commands follow the structured `status/metadata` schema.

## Project
- `project_info`
- `new_project` (payload: `{ "confirm": true }`)
- `open_project` (payload: `{ "path": "C:/path/file.efp" }`)
- `save_project` (payload: `{ "path": "C:/path/file.efp" }` or uses current project path)
- `save_project_as` (payload: `{ "path": "C:/path/file.efp" }`)
- `rename_project` (payload: `{ "name": "My Project" }`)
- `list_recents`
- `open_recent` (payload: `{ "index": 0 }` or `{ "path": "..." }`)
- `export_graph_json` / `import_graph_json`
- `export_code` (payload: `{ "path": "C:/path/GeneratedScene.py" }`)

## Scenes
- `list_scenes`
- `create_scene` (payload: `{ "scene_name": "Scene 2", "switch": true }`)
- `delete_scene` (payload: `{ "scene_name": "Scene 2", "confirm": true }`)
- `rename_scene` (payload: `{ "old_name": "Scene 1", "new_name": "Intro" }`)
- `switch_scene` (payload: `{ "scene_name": "Intro" }`)

## Nodes & Wires
- `create_node`
- `delete_node` (payload: `{ "node_id": "...", "confirm": true }`)
- `move_node` (payload: `{ "node_id": "...", "x": 100, "y": 200 }`)
- `set_node_param`
- `set_node_params` (batch update)
- `set_node_param_enabled`
- `set_node_param_escape`
- `rename_node`
- `duplicate_node`
- `connect_nodes` (payload: `{ "from_node_id": "...", "to_node_id": "..." }`)
- `disconnect_nodes` (payload: `{ "wire_id": "..." }` or `{ "from_node_id": "...", "to_node_id": "..." }`)
- `list_wires`
- `select_node`, `select_nodes`, `deselect_all`

## VGroups
- `vgroup_list`
- `vgroup_create` (payload: `{ "name": "my_group", "member_ids": ["..."], "create_canvas_node": true }`)
- `vgroup_rename`
- `vgroup_duplicate`
- `vgroup_delete` (payload: `{ "name": "...", "confirm": true }`)
- `vgroup_add_members`
- `vgroup_remove_member`
- `vgroup_highlight`

## AI & PDF
- `ai_generate_code` (payload: `{ "prompt": "..." }`)
- `ai_merge_code` (payload: `{ "code": "..." }`)
- `ai_reject_code`
- `ai_get_last`
- `ai_run_agent` (payload: `{ "prompt": "..." }`)
- `ai_auto_voiceover`
- `tts_generate` (payload: `{ "text": "...", "voice": "Zephyr", "model": "...", "node_id": "optional" }`)
- `ai_load_snippet` (payload: `{ "code": "...", "source": "snippet|github|ai" }`)
- `ai_pdf_import` (payload: `{ "paths": ["a.pdf"], "prompt": "...", "apply_nodes": true }`)

## Rendering
- `trigger_render` (preview render)
- `render_video` (payload: `{ "output_path": "C:/out", "fps": 30, "resolution": [1280,720], "quality": "m" }`)
- `cancel_video_render`
- `cancel_preview_render`
- `compile_graph`

## Assets
- `add_asset`, `update_asset`, `delete_asset`, `list_assets`, `get_asset`

## Settings / Theme / Keybindings
- `settings_get`, `settings_set`, `settings_list`
- `theme_get_palette`, `theme_set_color`, `theme_reload`
- `keybindings_list`, `keybindings_set`, `keybindings_reset`, `keybindings_reset_all`
- `usage_top`, `usage_top_mobjects`, `usage_top_animations`

## UI Actions
- `ui_list_actions` (lists QAction registry)
- `ui_trigger_action` (trigger by `text` or `object_name`)

## History
- `history.undo_project`, `history.redo_project`
- `history.undo_scene`, `history.redo_scene`
- `history.undo_node`, `history.redo_node`
- `history.create_checkpoint`, `history.restore_checkpoint`
- `history.list_checkpoints`, `history.summarize_actions`, `history.diff_between`
- `history.timeline`, `history.replay`

## Example MCP Calls

Undo last project action:
```json
{"command": "history.undo_project", "payload": {}}
```

Undo a specific node action:
```json
{"command": "history.undo_node", "payload": {"node_id": "abc-123"}}
```

Restore checkpoint “Set circle color to aqua”:
```json
{"command": "history.restore_checkpoint", "payload": {"checkpoint_name": "Set circle color to aqua"}}
```

Summarize last 5 actions:
```json
{"command": "history.summarize_actions", "payload": {"count": 5}}
```

Render a video:
```json
{"command": "render_video", "payload": {"output_path": "C:/renders", "fps": 30, "resolution": [1280, 720], "quality": "m"}}
```

## Hooking New Actions
To expose new functionality to MCP:

1. Add a handler in `core/mcp.py` using `@self._register("your_command")`.
2. Call the existing GUI or core method (e.g., `win.add_node`, `win.merge_ai_code`).
3. Return a structured payload via `_payload(...)` to include:
   - `status`
   - `affected_nodes`
   - `history_pointer`
   - `metadata`
4. For UI actions already bound to `QAction`, you can reuse `ui_trigger_action` instead of adding a new handler.
