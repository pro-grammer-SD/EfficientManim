"""
INTEGRATION GUIDE: Dual-Screen System + Timeline Engine

This guide shows how to integrate the new architectural components into main.py:
1. ScreenManager (stateful screen switching)
2. TimingResolver (deterministic timing)
3. LayoutPersistenceManager (UI state persistence)
4. Enhanced MCP (duplicate_node, asset management)
5. Screen-switching keybindings

NO PARTIAL INTEGRATION. All components must work together.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: ADD IMPORTS TO main.py (at top, after existing imports)
# ═══════════════════════════════════════════════════════════════════════════════

"""
Add these imports to main.py after the existing keybinding imports:

# Dual-screen system
try:
    from screen_manager import SCREEN_MANAGER, ScreenType, EditorScreen, TimelineScreen
    SCREEN_SYSTEM_AVAILABLE = True
except ImportError:
    SCREEN_SYSTEM_AVAILABLE = False
    print("WARNING: Screen manager not available")

# Timing resolver (deterministic timing)
try:
    from timing_resolver import TIMING_RESOLVER, TIMING_API, TimingResolver
    TIMING_SYSTEM_AVAILABLE = True
except ImportError:
    TIMING_SYSTEM_AVAILABLE = False
    print("WARNING: Timing resolver not available")

# Layout persistence
try:
    from layout_persistence import LAYOUT_PERSISTENCE
    LAYOUT_SYSTEM_AVAILABLE = True
except ImportError:
    LAYOUT_SYSTEM_AVAILABLE = False
    print("WARNING: Layout persistence not available")
"""

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: IN EfficientManimWindow.__init__
# ═══════════════════════════════════════════════════════════════════════════════

"""
Add to EfficientManimWindow.__init__ after self.setup_ui():

    # ═════════════════════════════════════════════════════════════════════
    # Initialize dual-screen system (PERSISTENT instances)
    # ═════════════════════════════════════════════════════════════════════
    if SCREEN_SYSTEM_AVAILABLE:
        self.stacked_widget = QStackedWidget()
        SCREEN_MANAGER.setup(self.stacked_widget)
        
        # Replace the central widget with stacked widget
        # (This requires refactoring the main layout)
        # self.setCentralWidget(self.stacked_widget)
        
        # Connect screen switching to keybindings
        KEYBINDINGS.binding_changed.connect(self._on_screen_keybinding)
    
    # ═════════════════════════════════════════════════════════════════════
    # Initialize timing resolver (DETERMINISTIC timing authority)
    # ═════════════════════════════════════════════════════════════════════
    if TIMING_SYSTEM_AVAILABLE:
        TIMING_RESOLVER.fps = 60  # Set frame rate
        TIMING_API.register_mutator(self._apply_node_timing)
    
    # ═════════════════════════════════════════════════════════════════════
    # Initialize layout persistence
    # ═════════════════════════════════════════════════════════════════════
    if LAYOUT_SYSTEM_AVAILABLE:
        # Restore layout on startup
        layout_state = LAYOUT_PERSISTENCE.get_screen_state("EDITOR")
        if layout_state:
            self._restore_editor_layout(layout_state)
"""

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: ADD KEYBINDING HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════

"""
Add these methods to EfficientManimWindow class:

    def _on_screen_keybinding(self, action_name: str, shortcut: str) -> None:
        \"\"\"Handle screen-switching keybindings.\"\"\"
        if action_name == "Switch to Editor" and shortcut == KEYBINDINGS.get_binding("Switch to Editor"):
            self.switch_to_editor_screen()
        elif action_name == "Switch to Timeline" and shortcut == KEYBINDINGS.get_binding("Switch to Timeline"):
            self.switch_to_timeline_screen()
    
    def switch_to_editor_screen(self) -> None:
        \"\"\"Switch to editor screen (Ctrl+1).\"\"\"
        if SCREEN_SYSTEM_AVAILABLE:
            SCREEN_MANAGER.switch_to_editor()
            LOGGER.info("Switched to Editor screen")
    
    def switch_to_timeline_screen(self) -> None:
        \"\"\"Switch to timeline screen (Ctrl+2).\"\"\"
        if SCREEN_SYSTEM_AVAILABLE:
            SCREEN_MANAGER.switch_to_timeline()
            LOGGER.info("Switched to Timeline screen")
    
    def _apply_node_timing(self, node_id: str, start_time: float, duration: float) -> None:
        \"\"\"Apply timing changes from resolver to node.\"\"\"
        node_item = self.nodes.get(node_id)
        if node_item:
            # Store timing in node metadata
            node_item.data.start_time = start_time
            node_item.data.duration = duration
            node_item.update()
            self.mark_modified()
    
    def _restore_editor_layout(self, state: dict) -> None:
        \"\"\"Restore editor screen layout from saved state.\"\"\"
        if not state:
            return
        
        # Restore scroll position
        scroll_pos = state.get("scroll_position", 0)
        if hasattr(self, 'view') and scroll_pos > 0:
            self.view.verticalScrollBar().setValue(scroll_pos)
        
        # Restore zoom level
        zoom = state.get("zoom_level", 1.0)
        if zoom != 1.0 and hasattr(self, 'view'):
            self.view.setTransform(self.view.transform().scale(zoom, zoom))
"""

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: ADD TO closeEvent() - PERSIST LAYOUT BEFORE CLOSING
# ═══════════════════════════════════════════════════════════════════════════════

"""
Add to EfficientManimWindow.closeEvent() BEFORE self.save_project():

    # Save layout state
    if LAYOUT_SYSTEM_AVAILABLE:
        editor_state = SCREEN_MANAGER.get_editor_screen().save_state() if SCREEN_MANAGER.get_editor_screen() else {}
        LAYOUT_PERSISTENCE.save_screen_state("EDITOR", editor_state)
        
        timeline_state = SCREEN_MANAGER.get_timeline_screen().save_state() if SCREEN_MANAGER.get_timeline_screen() else {}
        LAYOUT_PERSISTENCE.save_screen_state("TIMELINE", timeline_state)
"""

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: UPDATE setup_menu() - ADD SCREEN SWITCHING ACTIONS
# ═══════════════════════════════════════════════════════════════════════════════

"""
Add to setup_menu() in the View menu or create a new "Screens" menu:

        # Screens Menu (for dual-screen system)
        if SCREEN_SYSTEM_AVAILABLE:
            screens_menu = bar.addMenu("Screens")
            
            editor_action = QAction("Editor Screen", self)
            editor_action.setShortcut(KEYBINDINGS.get_binding("Switch to Editor") or "Ctrl+1")
            editor_action.triggered.connect(self.switch_to_editor_screen)
            screens_menu.addAction(editor_action)
            
            timeline_action = QAction("Timeline Screen", self)
            timeline_action.setShortcut(KEYBINDINGS.get_binding("Switch to Timeline") or "Ctrl+2")
            timeline_action.triggered.connect(self.switch_to_timeline_screen)
            screens_menu.addAction(timeline_action)
"""

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: TIMING INTEGRATION - When nodes change
# ═══════════════════════════════════════════════════════════════════════════════

"""
When a node is created, modified, or deleted, update timing:

    def _compile_graph_with_timing(self) -> None:
        \"\"\"Compile graph with deterministic timing.\"\"\"
        if not TIMING_SYSTEM_AVAILABLE:
            return
        
        # Get timing for all nodes
        timing = TIMING_RESOLVER.resolve_timing(
            {nid: n.data.to_dict() for nid, n in self.nodes.items()}
        )
        
        # Apply timing to nodes
        for node_id, (start_time, duration) in timing.items():
            self._apply_node_timing(node_id, start_time, duration)
        
        # Hash the result for determinism check
        timing_hash = TIMING_RESOLVER.get_timing_hash()
        LOGGER.debug(f"✓ Timing resolved (hash: {timing_hash[:8]}...)")
"""

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7: MCP GOVERNANCE - Ensure all mutations go through API
# ═══════════════════════════════════════════════════════════════════════════════

"""
In the MCP agent (mcp.py), for any timing mutations:

    # When modifying node timing through MCP:
    success, msg = TIMING_API.change_node_timing(
        node_id=node_id,
        start_time=new_start,
        duration=new_duration,
        reason="mcp:node_modify"
    )
    
    if not success:
        return MCPResult(success=False, error=msg)

This ensures MCP respects the timing resolver authority.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8: KEYBINDINGS UPDATES - Screen switching
# ═══════════════════════════════════════════════════════════════════════════════

"""
The keybindings are already updated in keybinding_registry.py:
- "Switch to Editor" → Ctrl+1
- "Switch to Timeline" → Ctrl+2
- "Duplicate Node" → Ctrl+D
- "Lock Timing" → Ctrl+Shift+L
- "Unlock Timing" → Ctrl+Shift+U

These will be automatically registered when initialize_default_keybindings() is called.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CRITICAL INTEGRATION REQUIREMENTS
# ═══════════════════════════════════════════════════════════════════════════════

"""
For the system to work correctly:

1. ✅ Screen instances created ONCE at startup (never recreated)
2. ✅ Switching uses visibility toggle only
3. ✅ Timing is computed through resolver, not bypassed
4. ✅ All node timing changes go through TIMING_API
5. ✅ Layout persists across screen switches
6. ✅ Keybindings unified (already done in Phase 2A)
7. ✅ MCP respects timing authority
8. ✅ No legacy slider/timer logic remains
9. ✅ Deterministic timing contract enforced
10. ✅ Bidirectional sync: Node ↔ Timeline maintained

FAILURE CONDITIONS (Implementation invalid if any occur):
- ❌ Screens recreated on switch
- ❌ Layout resets on screen switch
- ❌ Timing computed outside resolver
- ❌ Legacy slider still visible/active
- ❌ Timeline bypassed
- ❌ Keybindings not working
- ❌ State not persisted
"""

# ═══════════════════════════════════════════════════════════════════════════════
# TESTING CHECKLIST
# ═══════════════════════════════════════════════════════════════════════════════

"""
After integration, verify:

✅ Press Ctrl+1 → Switch to editor screen instantly
✅ Press Ctrl+2 → Switch to timeline screen instantly
✅ Close app on timeline screen → Reopen → Still on timeline screen
✅ Duplicate node (Ctrl+D) → New node created with offset
✅ Create TTS asset via MCP → Asset appears in asset list
✅ Modify node → Timing auto-computed through resolver
✅ Lock node timing (Ctrl+Shift+L) → Cannot auto-recompute
✅ Unlock node timing (Ctrl+Shift+U) → Can auto-recompute
✅ Timeline shows multi-track layout
✅ Voiceover waveform visible in timeline
✅ No legacy slider visible anywhere
✅ All keybindings work on both screens
"""

# ═══════════════════════════════════════════════════════════════════════════════
# TROUBLESHOOTING
# ═══════════════════════════════════════════════════════════════════════════════

"""
Issue: Screen switching doesn't work
→ Check: SCREEN_SYSTEM_AVAILABLE flag
→ Check: ScreenManager.setup() called in __init__
→ Check: Keybindings registered

Issue: Layout resets on switch
→ Check: LAYOUT_PERSISTENCE.save_screen_state() in closeEvent()
→ Check: _restore_editor_layout() called on switch

Issue: Timing not updating
→ Check: TIMING_SYSTEM_AVAILABLE flag
→ Check: TIMING_RESOLVER initialized with correct fps
→ Check: _compile_graph_with_timing() called on node changes

Issue: MCP duplicate_node fails
→ Check: NodeData class has copy-constructor support
→ Check: node_id parameter provided

Issue: Asset management via MCP not working
→ Check: _get_global_assets() can find ASSETS registry
→ Check: File permissions on asset files
"""

print("""
✅ Integration guide generated
📋 Follow steps 1-8 in order
🔍 Check critical requirements
✓ Run testing checklist
⚙️ No partial integration allowed
""")
