from __future__ import annotations
# -*- coding: utf-8 -*-

try:
    from core.keybinding_registry import KEYBINDINGS, initialize_default_keybindings

    KEYBINDINGS_AVAILABLE = True
except Exception:
    KEYBINDINGS_AVAILABLE = False

    class KEYBINDINGS:  # type: ignore[no-redef]
        @staticmethod
        def get_binding(name):
            return ""

        def binding_changed():
            pass

        def registry_updated():
            pass

    def initialize_default_keybindings():
        pass


def shortcut_for(action_name: str, default: str = "") -> str:
    try:
        return KEYBINDINGS.get_binding(action_name) or default
    except Exception:
        return default


def apply_shortcut(qaction, action_name: str, default: str = "") -> None:
    shortcut = shortcut_for(action_name, default)
    if shortcut:
        qaction.setShortcut(shortcut)
