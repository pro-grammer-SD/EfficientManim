"""
Unified Keybindings Panel — Drop-in replacement for old dual-panel system

Uses KeybindingRegistry as single source of truth.
Displays all actions with current shortcuts.
Allows editing, with conflict detection and validation.
"""

try:
    from PySide6.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
        QTableWidgetItem, QLabel, QMessageBox, QLineEdit, QHeaderView,
    )
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor
except ImportError:
    pass

from keybinding_registry import KEYBINDINGS


class UnifiedKeybindingsPanel(QDialog):
    """
    Unified keybindings editor using KeybindingRegistry.
    
    Features:
    - Single source of truth (KeybindingRegistry)
    - All actions displayed (no duplication)
    - Conflict detection
    - Search/filter
    - Reset to defaults
    - Live persistence
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⌨️ Keybindings Editor")
        self.resize(600, 600)
        self._build_ui()
        self._populate_table()
    
    def _build_ui(self):
        """Build the UI."""
        layout = QVBoxLayout(self)
        
        # Header
        header_label = QLabel(
            "Edit keybindings below. Click a shortcut cell to modify. "
            "Duplicates are automatically detected."
        )
        header_label.setStyleSheet("color: #6b7280; font-size: 11px;")
        header_label.setWordWrap(True)
        layout.addWidget(header_label)
        
        # Search box
        search_label = QLabel("Search:")
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Type to filter actions...")
        self.search_box.textChanged.connect(self._on_search)
        search_layout = QHBoxLayout()
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_box)
        layout.addLayout(search_layout)
        
        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels([
            "Action",
            "Default",
            "Current",
            "Status"
        ])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.DoubleClicked)
        self.table.verticalHeader().setVisible(False)
        self.table.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.table)
        
        # Status message
        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: #10b981; font-size: 10px;")
        layout.addWidget(self.status_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        btn_defaults = QPushButton("↩ Reset to Defaults")
        btn_defaults.clicked.connect(self._reset_all)
        button_layout.addWidget(btn_defaults)
        
        button_layout.addStretch()
        
        btn_cancel = QPushButton("Close")
        btn_cancel.clicked.connect(self.reject)
        button_layout.addWidget(btn_cancel)
        
        layout.addLayout(button_layout)
    
    def _populate_table(self):
        """Populate table with all actions from registry."""
        actions = KEYBINDINGS.get_all_actions()
        self.table.setRowCount(len(actions))
        self._all_actions = actions  # Store for search
        
        for row, action in enumerate(actions):
            # Action name
            name_item = QTableWidgetItem(action.name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, name_item)
            
            # Default shortcut
            default_item = QTableWidgetItem(action.default_shortcut)
            default_item.setFlags(default_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 1, default_item)
            
            # Current shortcut (editable)
            current_item = QTableWidgetItem(action.get_current_shortcut())
            self.table.setItem(row, 2, current_item)
            
            # Status (custom, default, or conflict)
            status = self._get_status(action)
            status_item = QTableWidgetItem(status)
            status_item.setFlags(status_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if "✓" in status:
                status_item.setForeground(QColor("#10b981"))
            elif "!" in status:
                status_item.setForeground(QColor("#f97316"))
            elif "✗" in status:
                status_item.setForeground(QColor("#ef4444"))
            self.table.setItem(row, 3, status_item)
    
    def _get_status(self, action) -> str:
        """Get status string for an action."""
        if action.user_override:
            return "✓ Custom"
        elif action.default_shortcut:
            return "◆ Default"
        else:
            return "⊘ None"
    
    def _on_item_changed(self, item):
        """Handle table cell edit."""
        if item.column() != 2:  # Only current shortcut column
            return
        
        row = item.row()
        action_item = self.table.item(row, 0)
        action_name = action_item.text()
        new_shortcut = item.text().strip()
        
        # Try to set the binding
        success, message = KEYBINDINGS.set_binding(
            action_name,
            new_shortcut,
            check_conflicts=True
        )
        
        if not success:
            QMessageBox.warning(self, "Keybinding Error", message)
            # Revert to current value
            action = KEYBINDINGS.get_action(action_name)
            item.setText(action.get_current_shortcut())
        else:
            # Update status column
            action = KEYBINDINGS.get_action(action_name)
            status = self._get_status(action)
            status_item = self.table.item(row, 3)
            if status_item is not None:
                status_item.setText(status)
                if "✓" in status:
                    status_item.setForeground(QColor("#10b981"))
                elif "!" in status:
                    status_item.setForeground(QColor("#f97316"))
                elif "✗" in status:
                    status_item.setForeground(QColor("#ef4444"))
            
            self.status_label.setText(message)
    
    def _on_search(self, text):
        """Filter table based on search text."""
        text = text.lower()
        for row in range(self.table.rowCount()):
            action_item = self.table.item(row, 0)
            matches = text in action_item.text().lower()
            self.table.setRowHidden(row, not matches)
    
    def _reset_all(self):
        """Reset all bindings to defaults."""
        reply = QMessageBox.question(
            self,
            "Reset to Defaults",
            "Reset all keybindings to their default values?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            KEYBINDINGS.reset_all()
            self._populate_table()
            self.status_label.setText("✓ All bindings reset to defaults")
