from __future__ import annotations
# -*- coding: utf-8 -*-

from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QFileDialog,
)

from core.ai_slides_to_manim import SlidesToManim
from graph.ai_slides_to_nodes import SlidesToNodes
from graph.node import NodeType
from utils.ai_context_builder import AIContextBuilder
from utils.ai_pdf_attachment_manager import PDFAttachmentManager
from utils.ai_pdf_parser import PDFParser
from utils.ai_slide_animator import AISlideAnimator
from utils.helpers import sanitize_background_settings
from utils.logger import LOGGER


class PDFSlideWorker(QThread):
    chunk_received = Signal(str)
    stage_changed = Signal(str)
    finished_signal = Signal(dict, str)
    error_signal = Signal(str)

    def __init__(self, pdf_paths: list[str], user_prompt: str, parent=None) -> None:
        super().__init__(parent)
        self.pdf_paths = pdf_paths
        self.user_prompt = user_prompt

    def run(self) -> None:
        try:
            self.stage_changed.emit("Parsing PDFs...")
            parser = PDFParser()
            parsed = parser.parse_pdfs(self.pdf_paths)

            self.stage_changed.emit("Building context...")
            context = AIContextBuilder().build_context(parsed, self.user_prompt)

            self.stage_changed.emit("Generating slides with AI...")
            animator = AISlideAnimator()
            slide_deck = animator.generate_slides(
                context, self.user_prompt, on_chunk=self.chunk_received.emit
            )

            if not slide_deck.get("slides"):
                raise RuntimeError("AI returned an empty slide deck.")

            self.stage_changed.emit("Generating Manim code...")
            manim_code = SlidesToManim.generate_code(slide_deck)
            manim_code = sanitize_background_settings(manim_code)

            self.finished_signal.emit(slide_deck, manim_code)
        except Exception as exc:
            self.error_signal.emit(str(exc))


class AIPDFUploadWidget(QFrame):
    """PDF attachment UI and pipeline runner inside the AI panel."""

    def __init__(self, prompt_widget, output_widget, parent=None) -> None:
        super().__init__(parent)
        self.prompt_widget = prompt_widget
        self.output_widget = output_widget
        self.manager = PDFAttachmentManager()
        self.worker: PDFSlideWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        self.setStyleSheet(
            "QFrame { border: 1px solid #e0e0e0; border-radius: 4px; "
            "background: #fafafa; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        header = QLabel("Attach PDF  Generate Animated Math Video")
        header.setStyleSheet("font-weight: bold; color: #1565c0;")
        layout.addWidget(header)

        btn_row = QHBoxLayout()
        self.btn_attach = QPushButton("Attach PDF")
        self.btn_attach_multi = QPushButton("Attach Multiple PDFs")
        self.btn_attach.clicked.connect(self._attach_single)
        self.btn_attach_multi.clicked.connect(self._attach_multiple)
        btn_row.addWidget(self.btn_attach)
        btn_row.addWidget(self.btn_attach_multi)
        layout.addLayout(btn_row)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        layout.addWidget(self.list_widget)

        action_row = QHBoxLayout()
        self.btn_remove = QPushButton("Remove File")
        self.btn_remove.clicked.connect(self._remove_selected)
        self.btn_generate = QPushButton("Generate Animation From PDFs")
        self.btn_generate.setStyleSheet(
            "QPushButton { background: #27ae60; color: white; font-weight: bold; "
            "padding: 6px 10px; }"
            "QPushButton:disabled { background: #bdbdbd; }"
        )
        self.btn_generate.clicked.connect(self._generate)
        action_row.addWidget(self.btn_remove)
        action_row.addStretch()
        action_row.addWidget(self.btn_generate)
        layout.addLayout(action_row)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: gray;")
        layout.addWidget(self.status_label)

    def _attach_single(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Attach PDF", "", "PDF Files (*.pdf)"
        )
        if not path:
            return
        if self.manager.add_file(path):
            self._add_list_item(path)
        self._update_status()

    def _attach_multiple(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Attach PDFs", "", "PDF Files (*.pdf)"
        )
        if not paths:
            return
        added = self.manager.add_files(paths)
        for p in added:
            self._add_list_item(p)
        self._update_status()

    def _add_list_item(self, path: str) -> None:
        item = QListWidgetItem(Path(path).name)
        item.setToolTip(path)
        item.setData(Qt.ItemDataRole.UserRole, path)
        self.list_widget.addItem(item)

    def _remove_selected(self) -> None:
        items = self.list_widget.selectedItems()
        if not items:
            return
        for item in items:
            path = item.data(Qt.ItemDataRole.UserRole)
            if path:
                self.manager.remove_file(path)
            row = self.list_widget.row(item)
            self.list_widget.takeItem(row)
        self._update_status()

    def _generate(self) -> None:
        paths = self.manager.list_files()
        if not paths:
            self.status_label.setText("No PDFs attached.")
            return
        prompt = self.prompt_widget.toPlainText().strip()
        if self.worker and self.worker.isRunning():
            return

        self.output_widget.clear()
        self.status_label.setText("Starting...")
        self._set_busy(True)

        self.worker = PDFSlideWorker(paths, prompt, parent=self)
        self.worker.chunk_received.connect(self._append_output)
        self.worker.stage_changed.connect(self._set_status)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.error_signal.connect(self._on_error)
        self.worker.start()

    def _append_output(self, text: str) -> None:
        if not text:
            return
        cursor = self.output_widget.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.output_widget.setTextCursor(cursor)

    def _set_status(self, msg: str) -> None:
        self.status_label.setText(msg)

    def _on_error(self, msg: str) -> None:
        self._set_busy(False)
        self.status_label.setText(f"Error: {msg}")
        LOGGER.error(msg)

    def _on_finished(self, slide_deck: dict, manim_code: str) -> None:
        self._set_busy(False)
        self.status_label.setText("Applying to node editor...")

        main_window = self._find_main_window()
        if not main_window:
            self.status_label.setText("Could not locate main window.")
            return

        try:
            if hasattr(main_window, "code_view"):
                main_window.code_view.setText(manim_code)
            if hasattr(main_window, "is_ai_generated_code"):
                main_window.is_ai_generated_code = True

            result = SlidesToNodes.apply_to_scene(slide_deck, main_window, True)
            for node in main_window.nodes.values():
                if node.data.type == NodeType.MOBJECT:
                    main_window.queue_render(node)

            if result.get("nodes"):
                try:
                    main_window.panel_props.set_node(result["nodes"][0])
                except Exception:
                    pass

            self.status_label.setText(f"Generated {result.get('node_count', 0)} nodes.")
        except Exception as exc:
            self.status_label.setText(f"Apply failed: {exc}")
            LOGGER.error(f"Apply failed: {exc}")

    def _find_main_window(self):
        w = self.window()
        if w is None:
            return None
        if hasattr(w, "add_node") and hasattr(w, "scene"):
            return w
        parent = w.parent()
        while parent is not None:
            if hasattr(parent, "add_node") and hasattr(parent, "scene"):
                return parent
            parent = parent.parent()
        return None

    def _update_status(self) -> None:
        count = self.manager.count()
        self.status_label.setText(f"{count} PDF(s) attached.")

    def _set_busy(self, busy: bool) -> None:
        self.btn_attach.setEnabled(not busy)
        self.btn_attach_multi.setEnabled(not busy)
        self.btn_remove.setEnabled(not busy)
        self.btn_generate.setEnabled(not busy)
