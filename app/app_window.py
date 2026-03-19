from __future__ import annotations
# -*- coding: utf-8 -*-

from core.project_manager import ProjectManager


def create_main_window():
    from ui.main_window import EfficientManimWindow

    window = EfficientManimWindow()
    window.project_manager = ProjectManager(window)
    return window
