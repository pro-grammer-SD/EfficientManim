from __future__ import annotations
# -*- coding: utf-8 -*-

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QHBoxLayout, QLineEdit, QWidget

from utils.shortcuts import apply_shortcut, shortcut_for
from utils.tooltips import apply_tooltip


def build_menus(window) -> None:
    """Build the main menu bar."""
    bar = window.menuBar()

    window.corner_container = QWidget()
    corner_layout = QHBoxLayout(window.corner_container)
    corner_layout.setContentsMargins(0, 0, 15, 0)

    window.project_name_edit = QLineEdit("Untitled Project")
    window.project_name_edit.setFixedWidth(220)
    window.project_name_edit.setPlaceholderText("Project name")
    window.project_name_edit.returnPressed.connect(window._rename_project)
    window.project_name_edit.editingFinished.connect(window._rename_project)
    apply_tooltip(
        window.project_name_edit,
        "Rename the project",
        "Press Enter to confirm",
        "Enter",
        "Project Name",
    )
    corner_layout.addWidget(window.project_name_edit)

    bar.setCornerWidget(window.corner_container, Qt.Corner.TopRightCorner)

    file_menu = bar.addMenu("File")
    new_action = file_menu.addAction(
        "New Project", window.new_project, QKeySequence.StandardKey.New
    )
    apply_tooltip(
        new_action,
        "Create a new project",
        "Start with a blank scene",
        "Ctrl+N",
        "New Project",
    )
    open_action = file_menu.addAction(
        "Open Project", window.open_project, QKeySequence.StandardKey.Open
    )
    apply_tooltip(
        open_action,
        "Open an existing project",
        "Choose a .efp file",
        "Ctrl+O",
        "Open Project",
    )
    save_action = file_menu.addAction(
        "Save Project", window.save_project, QKeySequence.StandardKey.Save
    )
    apply_tooltip(
        save_action,
        "Save the current project",
        "Write changes to disk",
        "Ctrl+S",
        "Save Project",
    )
    save_as_action = file_menu.addAction("Save As...", window.save_project_as)
    apply_tooltip(
        save_as_action,
        "Save project with a new name",
        "Choose a different file name",
        None,
        "Save As",
    )
    file_menu.addSeparator()
    settings_action = file_menu.addAction("Settings", window.open_settings)
    apply_tooltip(
        settings_action,
        "Open settings",
        "Configure preview, AI, and output",
        "Ctrl+,",
        "Settings",
    )
    file_menu.addSeparator()

    window._quit_action = QAction("Exit", window)
    window._quit_action.setShortcut(shortcut_for("Exit", "Ctrl+Q"))
    window._quit_action.triggered.connect(window.close)
    window._quit_action.setObjectName("_quit_action")
    apply_tooltip(
        window._quit_action,
        "Exit the application",
        "Closes the editor window",
        "Ctrl+Q",
        "Exit",
    )
    file_menu.addAction(window._quit_action)

    edit_menu = bar.addMenu("Edit")
    window._undo_action = QAction("Undo", window)
    apply_shortcut(window._undo_action, "Undo", "Ctrl+Z")
    window._undo_action.triggered.connect(window.undo_action)
    window._undo_action.setObjectName("_undo_action")
    apply_tooltip(
        window._undo_action,
        "Undo the last change",
        "Step back one action",
        "Ctrl+Z",
        "Undo",
    )
    edit_menu.addAction(window._undo_action)

    window._redo_action = QAction("Redo", window)
    apply_shortcut(window._redo_action, "Redo", "Ctrl+Y")
    window._redo_action.triggered.connect(window.redo_action)
    window._redo_action.setObjectName("_redo_action")
    apply_tooltip(
        window._redo_action,
        "Redo the last undone change",
        "Step forward one action",
        "Ctrl+Y",
        "Redo",
    )
    edit_menu.addAction(window._redo_action)

    edit_menu.addSeparator()
    window._delete_action = QAction("Delete Selected", window)
    apply_shortcut(window._delete_action, "Delete Selected", "Del")
    window._delete_action.triggered.connect(window.delete_selected)
    window._delete_action.setObjectName("_delete_action")
    apply_tooltip(
        window._delete_action,
        "Delete selected nodes",
        "Removes nodes from the canvas",
        "Del",
        "Delete Selected",
    )
    edit_menu.addAction(window._delete_action)

    view_menu = bar.addMenu("View")
    fit_action = view_menu.addAction(
        "Fit to View", window.fit_view, QKeySequence("Ctrl+0")
    )
    apply_tooltip(
        fit_action,
        "Fit the scene to the viewport",
        "Frames all nodes",
        "Ctrl+0",
        "Fit to View",
    )

    view_menu.addSeparator()
    window._zoom_in_action = QAction("Zoom In", window)
    window._zoom_in_action.setShortcut(shortcut_for("Zoom In", "Ctrl+="))
    window._zoom_in_action.triggered.connect(lambda: window.view.scale(1.15, 1.15))
    window._zoom_in_action.setObjectName("_zoom_in_action")
    apply_tooltip(
        window._zoom_in_action,
        "Zoom in",
        "Increase canvas scale",
        "Ctrl+=",
        "Zoom In",
    )
    view_menu.addAction(window._zoom_in_action)

    window._zoom_out_action = QAction("Zoom Out", window)
    window._zoom_out_action.setShortcut(shortcut_for("Zoom Out", "Ctrl+-"))
    window._zoom_out_action.triggered.connect(
        lambda: window.view.scale(1 / 1.15, 1 / 1.15)
    )
    window._zoom_out_action.setObjectName("_zoom_out_action")
    apply_tooltip(
        window._zoom_out_action,
        "Zoom out",
        "Decrease canvas scale",
        "Ctrl+-",
        "Zoom Out",
    )
    view_menu.addAction(window._zoom_out_action)

    view_menu.addSeparator()
    auto_layout_action = view_menu.addAction(
        "Auto-Layout Nodes", window.auto_layout_nodes, QKeySequence("Ctrl+L")
    )
    apply_tooltip(
        auto_layout_action,
        "Auto-arrange nodes",
        "Applies a clean left-to-right layout",
        "Ctrl+L",
        "Auto-Layout Nodes",
    )
    clear_action = view_menu.addAction(
        "Clear All", window.clear_scene, QKeySequence("Ctrl+Alt+Delete")
    )
    apply_tooltip(
        clear_action,
        "Clear the canvas",
        "Removes all nodes and wires",
        "Ctrl+Alt+Delete",
        "Clear All",
    )

    tools_menu = bar.addMenu("Tools")
    export_action = tools_menu.addAction(
        "Export Code (.py)",
        lambda: window._quick_export("py"),
        QKeySequence("Ctrl+E"),
    )
    apply_tooltip(
        export_action,
        "Export Python code",
        "Save generated scene code",
        "Ctrl+E",
        "Export Code",
    )
    copy_action = tools_menu.addAction(
        "Copy Code to Clipboard",
        lambda: window._quick_export("copy"),
        QKeySequence("Ctrl+Shift+C"),
    )
    apply_tooltip(
        copy_action,
        "Copy code to clipboard",
        "Paste into your editor",
        "Ctrl+Shift+C",
        "Copy Code",
    )
    tools_menu.addSeparator()

    window._render_video_action = QAction("Render Video", window)
    apply_shortcut(window._render_video_action, "Render Video", "Ctrl+R")
    window._render_video_action.triggered.connect(lambda: window.render_to_video({}))
    window._render_video_action.setObjectName("_render_video_action")
    apply_tooltip(
        window._render_video_action,
        "Render video output",
        "Opens the render panel and starts rendering",
        "Ctrl+R",
        "Render Video",
    )
    tools_menu.addAction(window._render_video_action)
    tools_menu.addSeparator()

    play_action = tools_menu.addAction(
        "Add Play Node", window.add_play_node, QKeySequence("Ctrl+Shift+P")
    )
    apply_tooltip(
        play_action,
        "Insert a play node",
        "Use to sequence animations",
        "Ctrl+Shift+P",
        "Add Play Node",
    )
    wait_action = tools_menu.addAction(
        "Add Wait Node", window.add_wait_node, QKeySequence("Ctrl+Shift+W")
    )
    apply_tooltip(
        wait_action,
        "Insert a wait node",
        "Add time gaps between plays",
        "Ctrl+Shift+W",
        "Add Wait Node",
    )
    tools_menu.addSeparator()

    vgroup_action = tools_menu.addAction(
        "Create VGroup from Selection",
        window.create_vgroup_from_selection,
        QKeySequence("Ctrl+G"),
    )
    apply_tooltip(
        vgroup_action,
        "Group selected mobjects",
        "Creates a VGroup node",
        "Ctrl+G",
        "Create VGroup",
    )
    tools_menu.addSeparator()
    manim_docs_action = tools_menu.addAction(
        "Open Manim Documentation", window._open_manim_docs
    )
    apply_tooltip(
        manim_docs_action,
        "Open Manim docs",
        "Launches the official documentation",
        None,
        "Manim Documentation",
    )
    gallery_action = tools_menu.addAction(
        "Open Gallery / Examples", window._open_manim_gallery
    )
    apply_tooltip(
        gallery_action,
        "Open the Manim gallery",
        "Browse official examples",
        None,
        "Manim Gallery",
    )

    help_menu = bar.addMenu("Help")
    shortcuts_action = help_menu.addAction(
        "Keyboard Shortcuts", window.show_shortcuts, QKeySequence("Ctrl+?")
    )
    apply_tooltip(
        shortcuts_action,
        "View keyboard shortcuts",
        "Shows the shortcut reference",
        "Ctrl+?",
        "Keyboard Shortcuts",
    )
    keybind_action = help_menu.addAction(
        "Edit Keybindings...", window.show_keybindings, QKeySequence("Ctrl+,")
    )
    apply_tooltip(
        keybind_action,
        "Edit keybindings",
        "Customize shortcuts",
        "Ctrl+,",
        "Edit Keybindings",
    )
    about_action = help_menu.addAction("About", window.show_about)
    apply_tooltip(
        about_action,
        "About EfficientManim",
        "Show version and credits",
        None,
        "About",
    )
    help_menu.addSeparator()

    mcp_menu = help_menu.addMenu("MCP Agent")

    mcp_status_action = QAction("MCP Status / Ping", window)
    mcp_status_action.setToolTip("Ping MCP and show current node count.")
    mcp_status_action.triggered.connect(window._mcp_ping)
    mcp_menu.addAction(mcp_status_action)

    mcp_context_action = QAction("Inspect Scene Context (JSON)", window)
    mcp_context_action.setToolTip("Show MCP context JSON for the current scene.")
    mcp_context_action.triggered.connect(window._mcp_show_context)
    mcp_menu.addAction(mcp_context_action)

    mcp_list_action = QAction("List All Commands", window)
    mcp_list_action.setToolTip("Show all registered MCP command names.")
    mcp_list_action.triggered.connect(window._mcp_list_commands)
    mcp_menu.addAction(mcp_list_action)

    mcp_log_action = QAction("Show Action Log", window)
    mcp_log_action.setToolTip("Show every MCP command executed in this run.")
    mcp_log_action.triggered.connect(window._mcp_show_log)
    mcp_menu.addAction(mcp_log_action)

    mcp_menu.addSeparator()
    mcp_compile_action = QAction("Force Compile Graph", window)
    mcp_compile_action.triggered.connect(
        lambda: window._mcp_exec_and_notify("compile_graph", {})
    )
    mcp_menu.addAction(mcp_compile_action)

    mcp_save_action = QAction("Save Project via MCP", window)
    mcp_save_action.triggered.connect(
        lambda: window._mcp_exec_and_notify("save_project", {})
    )
    mcp_menu.addAction(mcp_save_action)
