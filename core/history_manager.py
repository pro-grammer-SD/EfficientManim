from __future__ import annotations
# -*- coding: utf-8 -*-


class UndoRedoAction:
    """Base class for undo/redo actions."""

    def __init__(self, description: str):
        self.description = description

    def undo(self):
        raise NotImplementedError

    def redo(self):
        raise NotImplementedError


class NodeAddAction(UndoRedoAction):
    """Action for adding a node."""

    def __init__(self, window, node_item, node_data):
        super().__init__(f"Add {node_data.cls_name}")
        self.window = window
        self.node_item = node_item
        self.node_data = node_data

    def undo(self):
        if self.node_data.id in self.window.nodes:
            self.window.remove_node(self.window.nodes[self.node_data.id])

    def redo(self):
        if self.node_data.id not in self.window.nodes:
            self.window.scene.addItem(self.node_item)
            self.window.nodes[self.node_data.id] = self.node_item


class NodeParamChangeAction(UndoRedoAction):
    """Action for changing node parameters."""

    def __init__(self, node_data, key, old_val, new_val):
        super().__init__(f"Change {key}")
        self.node_data = node_data
        self.key = key
        self.old_val = old_val
        self.new_val = new_val

    def undo(self):
        self.node_data.params[self.key] = self.old_val

    def redo(self):
        self.node_data.params[self.key] = self.new_val


class UndoRedoManager:
    """Manages undo/redo history."""

    def __init__(self, max_history: int = 50):
        self.history = []
        self.current_index = -1
        self.max_history = max_history

    def push(self, action):
        self.history = self.history[: self.current_index + 1]
        self.history.append(action)
        self.current_index += 1
        if len(self.history) > self.max_history:
            self.history.pop(0)
            self.current_index -= 1

    def undo(self) -> bool:
        if self.current_index >= 0:
            self.history[self.current_index].undo()
            self.current_index -= 1
            return True
        return False

    def redo(self) -> bool:
        if self.current_index < len(self.history) - 1:
            self.current_index += 1
            self.history[self.current_index].redo()
            return True
        return False

    def can_undo(self) -> bool:
        return self.current_index >= 0

    def can_redo(self) -> bool:
        return self.current_index < len(self.history) - 1
