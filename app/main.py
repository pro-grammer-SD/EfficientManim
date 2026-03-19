from __future__ import annotations
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices, QIcon
from PySide6.QtWidgets import QMainWindow

from app.app_window import create_main_window
from app.startup import create_application
from ui.home import HomeScreen
from utils.logger import LOGGER


def _open_docs():
    docs_path = Path(__file__).resolve().parent.parent / "docs" / "README.md"
    if docs_path.exists():
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(docs_path)))
    else:
        LOGGER.warn("Documentation not found.")


def _open_example(window):
    window.new_project()
    try:
        # Simple example: Circle -> FadeIn -> Play
        circle = window.add_node(
            "MOBJECT", "Circle", params={"color": "BLUE"}, pos=(80, 80), name="circle"
        )
        fade = window.add_node(
            "ANIMATION", "FadeIn", params={}, pos=(320, 80), name="fade_in"
        )
        play = window.add_play_node()
        if circle and fade and play:
            window.scene.try_connect(circle.out_socket, fade.in_socket)
            window.scene.try_connect(fade.out_socket, play.in_socket)
        window.compile_graph()
    except Exception as exc:
        LOGGER.warn(f"Example project setup failed: {exc}")


def main() -> None:
    app = create_application()

    editor_window = {"window": None}

    def show_editor(path: str | None = None, example: bool = False, new: bool = False):
        if editor_window["window"] is None:
            editor_window["window"] = create_main_window()
        win = editor_window["window"]
        if path:
            win._do_open_project(path)
        elif example:
            _open_example(win)
        elif new:
            win.new_project()
        win.show()
        home_window.close()

    home_window = QMainWindow()
    home_window.setStyleSheet("QMainWindow { background-color: #ffffff; }")
    home_screen = HomeScreen()
    home_window.setCentralWidget(home_screen)
    home_window.setWindowTitle("EfficientManim")
    home_window.resize(1000, 720)
    icon_path = Path(__file__).resolve().parent.parent / "icon" / "icon.ico"
    if icon_path.exists():
        home_window.setWindowIcon(QIcon(str(icon_path)))

    home_screen.new_project_requested.connect(lambda: show_editor(new=True))
    home_screen.open_project_requested.connect(
        lambda: (show_editor(), editor_window["window"].open_project())
    )
    home_screen.open_example_requested.connect(lambda: show_editor(example=True))
    home_screen.open_docs_requested.connect(_open_docs)
    home_screen.recent_project_requested.connect(lambda p: show_editor(path=p))

    home_window.show()

    try:
        sys.exit(app.exec())
    except SystemExit:
        pass


if __name__ == "__main__":
    main()
