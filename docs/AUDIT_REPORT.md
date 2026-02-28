# PHASE 1: MANDATORY SYSTEM AUDIT — EfficientManim

**Status:** CRITICAL GOVERNANCE VIOLATIONS DETECTED  
**Date:** 2025-02-28  
**Risk Level:** HIGH — Multiple architectural violations found  
**Regression Risk:** MEDIUM — Careful extraction and refactoring required  

---

## 1. NODE LIFECYCLE TRACE

### Current State
- **NodeData class (line 1042)**: Stores node state with params, metadata, voiceover support
- **Node Creation**: Via `_add_node_by_class()` (line 7080) or `_add_node_from_class_browser()` (line 7353)
- **Node Deletion**: Via `remove_node()` → calls `mark_modified()` 
- **Graph Persistence**: Serialized to `graph.json` in `save_project()`

### Issues Found
- ✅ **CLEAN**: Node creation respects type system
- ✅ **CLEAN**: Node deletion handles edge cleanup via `mark_modified()`
- ✅ **CLEAN**: Serialization/deserialization works via `to_dict()` / `from_dict()`
- ⚠️ **WARNING**: No transaction boundaries — concurrent modifications could cause orphaned edges

### Verdict
**ACCEPTABLE** — No critical lifecycle violations detected. Edge cleanup is delegated to render pipeline.

---

## 2. PARAMETER POPULATION SYSTEM TRACE

### Current State
- **Parameter Storage**: `NodeData.params` (dict) + `NodeData.param_metadata` (dict with "enabled" and "escape" flags)
- **Parameter Extraction**: `PropertiesPanel.set_node()` (line 2359)
  - Uses `inspect.signature()` to get __init__ parameters
  - Auto-detects missing parameters
  - Creates type-safe widgets via `create_parameter_row()` (line 2466)

### Issues Found
- ✅ **CLEAN**: Uses `inspect.signature()` properly — NOT dumping raw class objects
- ✅ **CLEAN**: Type-safe widget creation with validation
- ⚠️ **PROBLEM 1**: **String comparison to `<class 'inspect._empty'>`** (lines 4155, 7833)
  - Should use `inspect.Parameter.empty` constant instead
  - Current code: `if value is None or str(value) == "<class 'inspect._empty'>"`
  - Correct code: `if value is inspect.Parameter.empty`

- ⚠️ **PROBLEM 2**: **Parameter metadata not enforced as defaults**
  - Code shows `tex_strings` disabled by default (line 2451-2455)
  - But this is inconsistent — ALL Mobject options should follow same rule
  - Need centralized parameter default policy

### Verdict
**VIOLATIONS FOUND** — 
1. Need to replace hardcoded string comparisons with proper `inspect` module usage
2. Need to establish parameter default policy (first option enabled, others disabled)

---

## 3. AUTOSAVE SYSTEM TRACE

### Current State
- **Hash-based Change Detection**: Three hashes computed (lines 7171-7250)
  - `_compute_code_hash()` — SHA256 of code text
  - `_compute_graph_hash()` — Hash of node IDs + connections
  - `_compute_assets_hash()` — Hash of asset timestamps
- **Change Detection Logic**: Lines 7238-7250
  - Computes all three hashes
  - Updates instance variables `_last_code_hash`, `_last_graph_hash`, `_last_assets_hash`
  - But **NO TIMER IMPLEMENTATION FOUND**

### Issues Found
- ⚠️ **CRITICAL PROBLEM**: **No QTimer for autosave trigger**
  - Hash computation code exists but is NEVER CALLED
  - Autosave only happens on manual `save_project()` call
  - Specification requires 3-second debounced autosave timer
  
- ⚠️ **PROBLEM**: **Where are hash functions called?**
  - Searched for `_compute_code_hash\|_compute_graph_hash\|_compute_assets_hash` callers
  - Results: No calls found in current codebase (dead code)

- ⚠️ **PROBLEM**: **No keybinding change detection**
  - Autosave spec requires detecting keybinding changes
  - Current system has no hash for keybindings

### Verdict
**CRITICAL VIOLATION** — Autosave system is incomplete. Hash infrastructure exists but unused. Implementation needed:
1. Create `autosave_timer` QTimer (3000ms)
2. Wire hash functions into save trigger
3. Add keybinding hash tracking
4. Implement debounce + state change detection

---

## 4. RENDER PIPELINE TRACE

### Current State
- **RenderWorker** (line 862): Thread for single-node preview rendering
- **VideoRenderWorker** (line 934): Thread for full-scene video rendering
- **Render Queue**: `queue_render()` (line 7899) → `process_render_queue()` (line 7914)
- **Determinism Contract**: Hash-based state tracking exists

### Issues Found
- ✅ **CLEAN**: Uses worker threads to avoid UI blocking
- ✅ **CLEAN**: Queue-based rendering prevents concurrent renders
- ⚠️ **WARNING**: No `render_in_progress` flag found
  - Should prevent concurrent renders and queued renders

- ⚠️ **WARNING**: No topological sort guarantee for graph
  - Spec requires "Render Rules: Topological sorted graph"
  - Current implementation: nodes rendered in iteration order

- ⚠️ **WARNING**: No last-working-preview fallback on render failure
  - Spec requires "Preserve last working preview on failure"

### Verdict
**ACCEPTABLE WITH GAPS** — Core render pipeline is functional but lacks determinism guarantees. Need:
1. Explicit `render_in_progress` flag
2. Topological sort before render
3. Last-working-preview preservation on error

---

## 5. ASSET MANAGER TRACE

### Current State
- **AssetManager class** (line 1511): Manages file assets with UUID tracking
- **Asset class** (line 1119): Represents a single asset (name, path, kind)
- **Methods**: `add_asset()`, `remove_asset()`, `get_asset()`, `get_list()`
- **Voiceover Integration**: Assets can be TTS files referenced by node's `audio_asset_id`

### Issues Found
- ✅ **CLEAN**: Proper UUID-based asset tracking
- ✅ **CLEAN**: Asset paths resolved correctly
- ⚠️ **PROBLEM**: TTS files not consistently named
  - Spec requires: `assets/tts_(id).wav`
  - Current code doesn't enforce this naming convention

### Verdict
**ACCEPTABLE** — Need to enforce TTS naming convention in voiceover system.

---

## 6. VOICEOVER HANDLING TRACE

### Current State
- **VoiceoverNode**: Not a first-class node type (missing from NodeType enum)
- **Voiceover Storage**: Attached to regular nodes via:
  - `audio_asset_id` (UUID reference)
  - `voiceover_transcript` (text)
  - `voiceover_duration` (float)
- **TTS Generation**: Not integrated (spec requires pydub duration computation)
- **Code Generation**: No VoiceoverScene inheritance or context manager injection

### Issues Found
- ⚠️ **CRITICAL PROBLEM**: **VoiceoverNode is not a first-class node**
  - Spec requires: "Implement VoiceoverNode as first-class deterministic node"
  - Current system: voiceover attached to ANY node (hacky)
  - Need proper NodeType.VOICEOVER_NODE

- ⚠️ **CRITICAL PROBLEM**: **No code generation for voiceover**
  - Spec requires: `with self.voiceover(audio="PATH") as tracker: self.wait(tracker.duration)`
  - Current code generation doesn't inject this

- ⚠️ **PROBLEM**: **No TTS duration computation**
  - Spec requires: "Use pydub to compute exact duration"
  - Current system accepts duration input but doesn't validate

### Verdict
**CRITICAL VIOLATION** — Voiceover system is incomplete:
1. Need to promote voiceover to first-class node type
2. Need to generate VoiceoverScene context manager code
3. Need to implement TTS duration computation via pydub

---

## 7. TAB NAVIGATION SYSTEM TRACE

### Current State
- **Two QTabWidget instances**:
  - `tabs_top` (line 6495): 13 tabs (Elements, Recents, Classes, VGroups, Outliner, Properties, Assets, LaTeX, Snippets, GitHub, Voiceover, Video, etc.)
  - `tabs_bot` (line 6568): 3 tabs (Preview, Code, Logs)
- **Tab Creation**: Hardcoded in `__init__` (lines 6542-6588)
- **Tab Switching**: Manual via `setCurrentIndex()` or tab clicking
- **MCP Integration**: `switch_tab()` (line 551 in mcp.py)

### Issues Found
- ⚠️ **PROBLEM**: **No unified tab menu/switcher**
  - Spec requires: "A top-level tab switcher (menu or dropdown)"
  - Current: User must manually click tabs or use MCP
  - Need: Central Tab Navigator (keyboard shortcut, dropdown, or menu)

- ⚠️ **PROBLEM**: **No keyboard navigation for tabs**
  - Spec requires: "Keyboard navigation support"
  - Current: Only clicking supported
  - Suggestion: Ctrl+Tab to cycle, or Ctrl+1-9 for specific tabs

- ⚠️ **PROBLEM**: **No active tab highlighting**
  - Specification implies visual feedback
  - Current tab bar already has selection, but no dedicated indicator

- ✅ **CLEAN**: Tab list is accurate and reflects actual open tabs
- ✅ **CLEAN**: Tab updates dynamically (no desync)

### Verdict
**PARTIAL VIOLATION** — Tab system functional but missing unified navigator. Need:
1. Top-level tab menu/dropdown component
2. Keyboard shortcuts for tab switching (Ctrl+Tab, Ctrl+1-9)
3. Optional: Tab search filter for many tabs

---

## 8. KEYBINDING REGISTRY & STORAGE TRACE

### CRITICAL VIOLATION FOUND

#### Two Competing Systems

**System 1: KeyboardShortcuts class (line 598)**
```python
class KeyboardShortcuts:
    SHORTCUTS = {
        "Delete": (QKeySequence.StandardKey.Delete, "Delete selected nodes/wires"),
        "Undo": (QKeySequence.StandardKey.Undo, "Undo last action"),
        "Redo": (QKeySequence.StandardKey.Redo, "Redo last action"),
        "Save": (QKeySequence.StandardKey.Save, "Save project"),
        "Open": (QKeySequence.StandardKey.Open, "Open project"),
        "Fit View": (...),
        "Clear": (...),
    }
```
- **Count**: 7 actions
- **Storage**: In-memory static dict, not persisted
- **UI Representation**: `describe_shortcuts()` static method (line 626)

**System 2: KeybindingsPanel class (line 5834)**
```python
class KeybindingsPanel(QDialog):
    DEFAULT_BINDINGS = {
        "New Project": "Ctrl+N",
        "Open Project": "Ctrl+O",
        "Save Project": "Ctrl+S",
        "Save As": "Ctrl+Shift+S",
        "Exit": "Ctrl+Q",
        "Undo": "Ctrl+Z",
        "Redo": "Ctrl+Y",
        "Delete Selected": "Del",
        "Zoom In": "Ctrl+=",
        "Zoom Out": "Ctrl+-",
        "Fit View": "Ctrl+0",
        "Clear All": "Ctrl+Alt+Del",
        "Auto-Layout": "Ctrl+L",
        "Export Code": "Ctrl+E",
        "Copy Code": "Ctrl+Shift+C",
        "Keybindings": "Ctrl+K",
        "Settings": "Ctrl+,",
        "AI Generate": "Ctrl+G",
        "Render Video": "Ctrl+R",
    }
```
- **Count**: 19 actions
- **Storage**: SETTINGS config file (persisted) via `_load()` / `_save()`
- **UI Representation**: QTableWidget dialog with edit/save/reset buttons (line 5879-5937)

#### Governance Failure Details

1. **NO SINGLE SOURCE OF TRUTH**
   - System 1 has 7 shortcuts
   - System 2 has 19 shortcuts
   - No overlap in action names ("Delete" vs "Delete Selected", "Clear" vs "Clear All")

2. **INCONSISTENT ACTION LISTS**
   - Both panels display different actions
   - User edits in KeybindingsPanel don't affect KeyboardShortcuts
   - User edits in KeybindingsPanel might not affect actual QActions

3. **QAction BINDINGS NOT WIRED**
   - Main window creates QActions (lines 6632-6666)
   - Each QAction has `setShortcut()` called
   - These shortcuts are HARDCODED, not read from either system
   - User modifications in KeybindingsPanel do NOT rebind QActions

4. **MISSING ACTIONS**
   - "Render Video" defined in KeybindingsPanel (line 5854)
   - But NO corresponding QAction created in main window
   - No keybinding rebinding on change

5. **PERSISTENCE ISSUES**
   - KeybindingsPanel persists to SETTINGS
   - But on app restart, QActions reset to hardcoded shortcuts
   - User changes are loaded but not applied to QActions

6. **NO CONFLICT DETECTION ENFORCEMENT**
   - KeybindingsPanel checks for duplicates (line 5927-5935)
   - But doesn't PREVENT conflicting assignments
   - Doesn't show which action currently owns a shortcut

### Verdict
**CRITICAL GOVERNANCE VIOLATION** —  This is a **SYSTEM LAW ENFORCEMENT** failure:
1. Two separate registries with conflicting data
2. Changes don't apply to QActions
3. No reestablished persistent state
4. No unified UI representation
5. Missing "Render Video" action entirely

**Required Fix**: Implement unified KeybindingRegistry with:
- Central action registry
- Persistent storage
- Dynamic QAction rebinding
- Duplicate prevention
- UI sync guarantee

---

## SUMMARY OF ARCHITECTURAL VIOLATIONS

| System | Severity | Status | Root Cause |
|--------|----------|--------|-----------|
| Keybindings | **CRITICAL** | ❌ FAILED | Two competing registries, no sync |
| Autosave | **CRITICAL** | ❌ FAILED | Hash functions exist but never called |
| Voiceover | **CRITICAL** | ❌ FAILED | Not first-class node, no code gen |
| Render Pipeline | **MEDIUM** | ⚠️ PARTIAL | No determinism guarantees, missing flags |
| Tab Navigation | **MEDIUM** | ⚠️ PARTIAL | No unified menu, no keyboard support |
| Parameter System | **LOW** | ⚠️ WARNING | String comparison to `<class '...'>` |
| Asset Manager | **LOW** | ✅ CLEAN | Works correctly |
| Node Lifecycle | **LOW** | ✅ CLEAN | Works correctly |

---

## IMPLEMENTATION PLAN

### Phase 2A: Fix Keybindings (BLOCKING)
1. Create unified `KeybindingRegistry` class
2. Unify both systems into single source of truth
3. Implement persistent storage
4. Implement dynamic QAction rebinding
5. Create single UI panel (merge both)
6. Add "Render Video" action
7. Test Windows/Mac/Linux keyboard equivalents

### Phase 2B: Fix Autosave
1. Create `autosave_timer` QTimer(3000ms)
2. Connect hash function calls
3. Add keybinding hash tracking
4. Implement debounce + change detection
5. Test save/load cycle

### Phase 2C: Fix Voiceover (Production Grade)
1. Create NodeType.VOICEOVER_NODE
2. Implement first-class VoiceoverNode
3. Add pydub duration computation
4. Generate VoiceoverScene context manager code
5. Test code generation + rendering

### Phase 2D: Fix Tab Navigation
1. Create unified Tab Menu Navigator
2. Add keyboard shortcuts (Ctrl+Tab, Ctrl+1-9)
3. Add tab search/filter
4. Test all tabs accessible

### Phase 2E: Fix Render Pipeline Determinism
1. Add `render_in_progress` flag
2. Implement topological sort
3. Add last-working-preview fallback
4. Test determinism contract

### Phase 2F: Fix Parameter System
1. Replace hardcoded string comparisons
2. Establish parameter default policy
3. Test parameter serialization

### Phase 2G: Fix MCP Governance
1. Ensure all operations respect registries
2. Test node factory
3. Test asset manager
4. Test render manager

### Phase 2H: Final Testing
1. No regressions
2. Windows/Mac compatibility
3. UI stability
4. Determinism contract verified

---

## RISK ASSESSMENT

**HIGH RISK AREAS**:
- Keybindings refactoring (touches core action system)
- Autosave integration (touches save/load)
- Voiceover as first-class node (new concept)

**MITIGATION**:
- Comprehensive unit tests
- Integration tests
- Regression test suite
- Staged rollout

**ESTIMATED EFFORT**: 8-12 hours of careful engineering

---

End of PHASE 1 Audit Report
