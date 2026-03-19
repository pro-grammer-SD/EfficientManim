from __future__ import annotations
# -*- coding: utf-8 -*-

from typing import Optional

from core.file_manager import add_recent, get_recents


class ProjectManager:
    """Coordinator for project lifecycle actions."""

    def __init__(self, window=None):
        self.window = window

    def attach(self, window) -> None:
        self.window = window

    def new_project(self) -> None:
        if self.window:
            self.window.new_project()

    def open_project(self, path: Optional[str] = None) -> None:
        if not self.window:
            return
        if path:
            self.window._do_open_project(path)
            add_recent(path)
        else:
            self.window.open_project()

    def save_project(self) -> None:
        if self.window:
            self.window.save_project()

    def save_project_as(self) -> None:
        if self.window:
            self.window.save_project_as()

    def recent_projects(self) -> list[str]:
        return get_recents()
