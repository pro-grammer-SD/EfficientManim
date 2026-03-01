"""
MAIN.PY CHANGES - DETAILED IMPLEMENTATION GUIDE

This document shows the exact changes made to main.py for full extension support.

=============================================================================
CHANGE 1: ElementsPanel - Add Extension Nodes Support
=============================================================================

LOCATION: Line ~2669 (ElementsPanel class definition)

OLD CODE:
```python
class ElementsPanel(QWidget):
    add_requested = Signal(str, str)  # Type, Class

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search...")
        self.search.textChanged.connect(self.filter)
        layout.addWidget(self.search)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemDoubleClicked.connect(self.on_dbl_click)
        layout.addWidget(self.tree)
        self.populate()

    def populate(self):
        if not MANIM_AVAILABLE:
            return
        self.tree.clear()
        mob_root = QTreeWidgetItem(self.tree, ["Mobjects"])
        anim_root = QTreeWidgetItem(self.tree, ["Animations"])

        for name in dir(manim):
            if name.startswith("_"):
                continue
            obj = getattr(manim, name)
            if inspect.isclass(obj):
                try:
                    if issubclass(obj, manim.Mobject) and obj is not manim.Mobject:
                        QTreeWidgetItem(mob_root, [name])
                    elif (
                        issubclass(obj, manim.Animation) and obj is not manim.Animation
                    ):
                        QTreeWidgetItem(anim_root, [name])
                except:
                    pass
        mob_root.setExpanded(True)

    def filter(self, txt):
        root = self.tree.invisibleRootItem()
        txt = txt.lower()
        for i in range(root.childCount()):
            cat = root.child(i)
            hide = True
            for j in range(cat.childCount()):
                item = cat.child(j)
                if txt in item.text(0).lower():
                    item.setHidden(False)
                    hide = False
                else:
                    item.setHidden(True)
            cat.setHidden(hide)

    def on_dbl_click(self, item, col):
        if item.childCount() == 0:
            p = item.parent()
            t = "Mobject" if p.text(0) == "Mobjects" else "Animation"
            self.add_requested.emit(t, item.text(0))
```

NEW CODE:
```python
class ElementsPanel(QWidget):
    add_requested = Signal(str, str)  # Type, Class

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search...")
        self.search.textChanged.connect(self.filter)
        layout.addWidget(self.search)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemDoubleClicked.connect(self.on_dbl_click)
        layout.addWidget(self.tree)
        self.populate()

    def populate(self):
        if not MANIM_AVAILABLE:
            return
        self.tree.clear()
        mob_root = QTreeWidgetItem(self.tree, ["Mobjects"])
        anim_root = QTreeWidgetItem(self.tree, ["Animations"])

        # Add built-in Manim elements
        for name in dir(manim):
            if name.startswith("_"):
                continue
            obj = getattr(manim, name)
            if inspect.isclass(obj):
                try:
                    if issubclass(obj, manim.Mobject) and obj is not manim.Mobject:
                        QTreeWidgetItem(mob_root, [name])
                    elif (
                        issubclass(obj, manim.Animation) and obj is not manim.Animation
                    ):
                        QTreeWidgetItem(anim_root, [name])
                except:
                    pass
        
        # ADD THIS SECTION: Add extension-provided nodes
        try:
            from core.node_registry import NODE_REGISTRY
            
            node_registry_root = QTreeWidgetItem(self.tree, ["Extensions"])
            extension_nodes = NODE_REGISTRY.get_nodes()
            
            for category, nodes in extension_nodes.items():
                category_item = QTreeWidgetItem(node_registry_root, [category])
                for node_def in nodes:
                    item = QTreeWidgetItem(category_item, [node_def.node_name])
                    # Store the class path for later loading
                    item.setData(Qt.ItemDataRole.UserRole, node_def.class_path)
            
            if extension_nodes:
                node_registry_root.setExpanded(True)
        except Exception as e:
            LOGGER.warning(f"Could not load extension nodes: {e}")
        # END NEW SECTION
        
        mob_root.setExpanded(True)

    def filter(self, txt):
        root = self.tree.invisibleRootItem()
        txt = txt.lower()
        for i in range(root.childCount()):
            cat = root.child(i)
            hide = True
            for j in range(cat.childCount()):
                item = cat.child(j)
                if txt in item.text(0).lower():
                    item.setHidden(False)
                    hide = False
                else:
                    item.setHidden(True)
            cat.setHidden(hide)

    def on_dbl_click(self, item, col):
        if item.childCount() == 0:
            p = item.parent()
            parent_text = p.text(0)
            
            # Determine type based on parent category
            if parent_text == "Mobjects":
                t = "Mobject"
            elif parent_text == "Animations":
                t = "Animation"
            elif parent_text in ["Extensions", "Custom"]:
                t = "ExtensionNode"
            else:
                # Default to trying as Mobject for extension nodes
                t = "ExtensionNode"
            
            self.add_requested.emit(t, item.text(0))
```

KEY CHANGES:
  - populate() now queries NODE_REGISTRY.get_nodes()
  - Extension nodes appear under "Extensions" category
  - on_dbl_click() detects ExtensionNode type
  - Proper error handling for missing registry

=============================================================================
CHANGE 2: add_node_center - Handle Extension Nodes
=============================================================================

LOCATION: Line ~7511 (add_node_center method)

OLD CODE:
```python
def add_node_center(self, type_str, cls_name):
    self.is_ai_generated_code = False  # Reset flag - manual node added
    USAGE_TRACKER.record(cls_name, type_str)
    if self.code_view.toPlainText().strip() == "":
        self.compile_graph()
    center = self.view.mapToScene(self.view.rect().center())
    self.add_node(type_str, cls_name, pos=(center.x(), center.y()))
```

NEW CODE:
```python
def add_node_center(self, type_str, cls_name):
    self.is_ai_generated_code = False  # Reset flag - manual node added
    USAGE_TRACKER.record(cls_name, type_str)
    if self.code_view.toPlainText().strip() == "":
        self.compile_graph()
    center = self.view.mapToScene(self.view.rect().center())
    
    # Handle extension nodes by treating them as custom mobjects
    actual_type = "Mobject" if type_str == "ExtensionNode" else type_str
    
    self.add_node(actual_type, cls_name, pos=(center.x(), center.y()))
```

KEY CHANGES:
  - Detects ExtensionNode type
  - Converts to "Mobject" for graph compatibility
  - Transparent to rest of system

=============================================================================
CHANGE 3: _initialize_extensions - Load All Extensions
=============================================================================

LOCATION: Line ~6998 (_initialize_extensions method)

OLD CODE:
```python
def _initialize_extensions(self) -> None:
    """
    Initialize built-in extensions and realize their panels into the main window.
    
    Called during __init__ after UI setup is complete.
    This is the correct time to realize panels since:
    1. The main window (self) fully exists
    2. All dock areas are available
    3. Extensions have been loaded via bootstrap
    """
    import logging
    logger = logging.getLogger("extensions")
    
    try:
        # Setup built-in demo extensions
        from core.extensions.color_palette import setup as setup_color_palette
        
        api = ExtensionAPI("color-palette")
        setup_color_palette(api)
        
        logger.info("✓ Built-in extensions initialized")
    except Exception as e:
        logger.error(f"Failed to initialize extensions: {e}", exc_info=True)
    
    try:
        # Realize all registered panels into the main window
        realized_panels = EXTENSION_REGISTRY.realize_panels(self)
        logger.info(f"✅ Realized {len(realized_panels)} extension panels")
        
        for panel_name, widget in realized_panels.items():
            logger.info(f"   • {panel_name}: {type(widget).__name__}")
    except Exception as e:
        logger.error(f"Failed to realize extension panels: {e}", exc_info=True)
```

NEW CODE:
```python
def _initialize_extensions(self) -> None:
    """
    Initialize built-in extensions and realize their panels into the main window.
    
    Called during __init__ after UI setup is complete.
    This is the correct time to realize panels since:
    1. The main window (self) fully exists
    2. All dock areas are available
    3. Extensions have been loaded via bootstrap
    
    ALL EXTENSIONS ARE LOADED: Color Palette, Math Symbols, Timeline Templates
    """
    import logging
    logger = logging.getLogger("extensions")
    
    try:
        # Setup all built-in demo extensions
        from core.extensions.color_palette import setup as setup_color_palette
        from core.extensions.color_palette import set_main_window as set_color_main_window
        from core.extensions.math_symbols import setup as setup_math_symbols
        from core.extensions.timeline_templates import setup as setup_timeline_templates
        
        # Initialize Color Palette with main window reference for theme updates
        set_color_main_window(self)
        api_color = ExtensionAPI("color-palette")
        setup_color_palette(api_color)
        logger.info("✓ Color Palette extension initialized (with theme updates)")
        
        # Initialize Math Symbols (provides nodes for Elements panel)
        api_math = ExtensionAPI("math-symbols")
        setup_math_symbols(api_math)
        logger.info("✓ Math Symbols extension initialized (nodes registered in Elements)")
        
        # Initialize Timeline Templates (provides timeline manager panel)
        api_timeline = ExtensionAPI("timeline-templates")
        setup_timeline_templates(api_timeline)
        logger.info("✓ Timeline Templates extension initialized (timeline panel available)")
        
        logger.info("✓ Built-in extensions initialized (3 total: Color Palette, Math Symbols, Timeline)")
    except Exception as e:
        logger.error(f"Failed to initialize extensions: {e}", exc_info=True)
    
    try:
        # Realize all registered panels into the main window
        realized_panels = EXTENSION_REGISTRY.realize_panels(self)
        
        # Apply theme to all realized panels to ensure consistent styling
        for panel_name, widget in realized_panels.items():
            try:
                # Apply the current theme stylesheet to the widget and its children
                widget.setStyleSheet(THEME_MANAGER.get_stylesheet())
            except Exception as e:
                logger.warning(f"Could not apply theme to panel '{panel_name}': {e}")
        
        logger.info(f"✅ Realized {len(realized_panels)} extension panels into main window")
        
        for panel_name, widget in realized_panels.items():
            logger.info(f"   • {panel_name}: {type(widget).__name__}")
    except Exception as e:
        logger.error(f"Failed to realize extension panels: {e}", exc_info=True)
```

KEY CHANGES:
  - Imports all 3 extension setup functions
  - Calls set_main_window(self) to link Color Palette to main window
  - Initializes each extension separately with detailed logging
  - Applies theme to all realized panels
  - Better error messages
  - Clearer logging about what each extension provides

=============================================================================
NOTES
=============================================================================

These are the ONLY changes needed to main.py. No other modifications required.

The changes are:
  1. Backward compatible
  2. Additive (no removals)
  3. Error-safe (try-except blocks)
  4. Well-logged (debugging info)
  5. Non-intrusive (isolated to specific methods)

All other functionality remains unchanged.
"""