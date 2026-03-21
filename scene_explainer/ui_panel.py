from __future__ import annotations
# -*- coding: utf-8 -*-

import json
from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QFileDialog,
)

from core.config import SETTINGS
from utils.logger import LOGGER
from scene_explainer.ai_explainer import ExplainService
from scene_explainer.explanation_models import ExplainResponse, LessonNotes
from scene_explainer.history_explainer import HistoryExplainer


@dataclass
class _ExplainRequestState:
    mode: str
    node_ids: Optional[list[str]] = None
    animation_id: Optional[str] = None
    objects_only: bool = False


class ExplainPanel(QWidget):
    """Dockable panel that displays AI explanations."""

    auto_explain_completed = Signal()

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.service = getattr(main_window, "explain_service", None) or ExplainService(main_window)
        self.history_explainer = None
        if getattr(main_window, "history_manager", None):
            self.history_explainer = HistoryExplainer(
                main_window.history_manager,
                self.service.analyzer,
                self.service.prompt_builder,
                self.service.ai,
            )

        self._last_request: Optional[_ExplainRequestState] = None
        self._current_explanation: Optional[ExplainResponse] = None
        self._current_lesson: Optional[LessonNotes] = None
        self._auto_mode_active = False

        self.setObjectName("ExplainPanel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self._build_ui()
        self._apply_styles()
        self._wire_signals()
        self._sync_teacher_controls()

    # ──────────────────────────────────────────────────────────────────
    # UI setup
    # ──────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        header = QLabel("Explain Panel")
        header.setStyleSheet("font-weight: bold; font-size: 14px; color: #2c3e50; background: transparent;")
        layout.addWidget(header)

        controls = QHBoxLayout()
        controls.setSpacing(6)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Detailed", "Simple"])
        self.mode_combo.setCurrentText("Detailed")
        controls.addWidget(QLabel("Mode:"))
        controls.addWidget(self.mode_combo)

        self.btn_regen = QPushButton("Regenerate")
        controls.addWidget(self.btn_regen)

        self.btn_copy = QPushButton("Copy")
        controls.addWidget(self.btn_copy)

        controls.addStretch(1)
        layout.addLayout(controls)

        self.loading_label = QLabel("Generating explanation...")
        self.loading_label.setStyleSheet("color: #1e88e5; font-weight: bold; background: transparent;")
        self.loading_label.setVisible(False)
        layout.addWidget(self.loading_label)

        # Auto-explain banner
        self.auto_frame = QFrame()
        self.auto_frame.setObjectName("ExplainAutoFrame")
        auto_layout = QVBoxLayout(self.auto_frame)
        auto_layout.setContentsMargins(8, 6, 8, 6)
        self.auto_title = QLabel("Learning Mode Insight")
        self.auto_title.setStyleSheet("font-weight: bold; color: #2e7d32; background: transparent;")
        self.auto_text = QLabel("")
        self.auto_text.setWordWrap(True)
        self.auto_text.setStyleSheet("background: transparent;")
        auto_layout.addWidget(self.auto_title)
        auto_layout.addWidget(self.auto_text)
        self.auto_frame.setVisible(False)
        layout.addWidget(self.auto_frame)

        # Scrollable content area
        scroll = QScrollArea()
        scroll.setObjectName("ExplainScrollArea")
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background-color: #ffffff;")
        scroll.viewport().setStyleSheet("background-color: #ffffff;")

        content = QWidget()
        content.setObjectName("ExplainContent")
        content.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.content_layout = QVBoxLayout(content)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(6)

        self.section_concept, self.txt_concept = self._create_section(
            "What This Animation Teaches"
        )
        self.section_steps, self.txt_steps = self._create_section(
            "Step-by-Step Explanation"
        )
        self.section_visual, self.txt_visual = self._create_section(
            "Why the Visuals Work"
        )
        self.section_simple, self.txt_simple = self._create_section(
            "Simple Explanation"
        )
        self.section_takeaways, self.txt_takeaways = self._create_section(
            "Key Takeaways"
        )
        self.section_lesson, self.txt_lesson = self._create_section(
            "Lesson Notes"
        )

        self.content_layout.addWidget(self.section_concept)
        self.content_layout.addWidget(self.section_steps)
        self.content_layout.addWidget(self.section_visual)
        self.content_layout.addWidget(self.section_simple)
        self.content_layout.addWidget(self.section_takeaways)
        self.content_layout.addWidget(self.section_lesson)
        self.content_layout.addStretch(1)

        scroll.setWidget(content)
        layout.addWidget(scroll)

        # Teacher controls
        self.teacher_controls = QHBoxLayout()
        self.btn_generate_lesson = QPushButton("Generate Lesson Notes")
        self.btn_export_lesson = QToolButton()
        self.btn_export_lesson.setText("Export")
        self.btn_export_lesson.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.btn_export_lesson.setMenu(self._build_export_menu())

        self.teacher_controls.addWidget(self.btn_generate_lesson)
        self.teacher_controls.addWidget(self.btn_export_lesson)
        self.teacher_controls.addStretch(1)
        layout.addLayout(self.teacher_controls)

    def _apply_styles(self) -> None:
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            "QWidget#ExplainPanel, QWidget#ExplainPanel QWidget { background-color: #ffffff; color: #1f2937; }"
            "QScrollArea#ExplainScrollArea { background-color: #ffffff; border: none; }"
            "QScrollArea#ExplainScrollArea QWidget#qt_scrollarea_viewport { background-color: #ffffff; }"
            "QAbstractScrollArea::viewport { background-color: #ffffff; }"
            "QScrollArea#ExplainScrollArea QWidget, QWidget#ExplainContent { background-color: #ffffff; border: none; }"
            "QWidget#ExplainPanel QLabel { color: #1f2937; background-color: transparent; }"
            "QWidget#ExplainPanel QGroupBox { background-color: #ffffff; border: 1px solid #e5e7eb; "
            "border-radius: 6px; margin-top: 8px; color: #1f2937; }"
            "QWidget#ExplainPanel QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; background-color: #ffffff; }"
            "QWidget#ExplainPanel QTextEdit { background-color: #ffffff; color: #111827; border: 1px solid #e5e7eb; "
            "border-radius: 4px; }"
            "QWidget#ExplainPanel QFrame { background-color: #ffffff; }"
            "QWidget#ExplainPanel QFrame#ExplainAutoFrame { background-color: #e8f5e9; border: 1px solid #c8e6c9; "
            "border-radius: 4px; }"
            "QWidget#ExplainPanel QPushButton { background-color: #f3f4f6; border: 1px solid #d1d5db; "
            "border-radius: 4px; padding: 4px 8px; color: #1f2937; }"
            "QWidget#ExplainPanel QPushButton:hover { background-color: #e5e7eb; }"
            "QWidget#ExplainPanel QComboBox { background-color: #ffffff; border: 1px solid #d1d5db; border-radius: 4px; color: #1f2937; }"
            "QScrollBar:vertical { background: #f9fafb; width: 10px; border: none; }"
            "QScrollBar::handle:vertical { background: #d1d5db; border-radius: 5px; min-height: 20px; }"
            "QScrollBar::handle:vertical:hover { background: #9ca3af; }"
        )

    def _create_section(self, title: str) -> tuple[QGroupBox, QTextEdit]:
        group = QGroupBox(title)
        group.setCheckable(True)
        group.setChecked(True)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setMinimumHeight(80)
        layout.addWidget(text)

        def _toggle(checked: bool):
            text.setVisible(checked)

        group.toggled.connect(_toggle)
        return group, text

    def _build_export_menu(self) -> QMenu:
        menu = QMenu(self)
        act_md = QAction("Markdown (.md)", self)
        act_txt = QAction("Plain Text (.txt)", self)
        act_copy = QAction("Copy to Clipboard", self)
        act_md.triggered.connect(lambda: self._export_lesson("md"))
        act_txt.triggered.connect(lambda: self._export_lesson("txt"))
        act_copy.triggered.connect(lambda: self._export_lesson("copy"))
        menu.addAction(act_md)
        menu.addAction(act_txt)
        menu.addAction(act_copy)
        return menu

    def _wire_signals(self) -> None:
        self.btn_regen.clicked.connect(self.regenerate)
        self.btn_copy.clicked.connect(self.copy_explanation)
        self.btn_generate_lesson.clicked.connect(self.generate_lesson_notes)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)

        self.service.ai.explanation_ready.connect(self._on_explanation_ready)
        self.service.ai.learning_ready.connect(self._on_learning_ready)
        self.service.ai.lesson_ready.connect(self._on_lesson_ready)
        self.service.ai.error.connect(self._on_error)

        SETTINGS.settings_changed.connect(self._sync_teacher_controls)

    def _sync_teacher_controls(self) -> None:
        enabled = bool(SETTINGS.get("TEACHER_MODE_ENABLED", False, type=bool))
        self.btn_generate_lesson.setVisible(enabled)
        self.btn_export_lesson.setVisible(enabled)
        self.section_lesson.setVisible(enabled)

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def explain_scene(self, mode: Optional[str] = None) -> None:
        self._request_explanation(mode=mode)

    def explain_selected_nodes(self, node_ids: list[str], mode: Optional[str] = None) -> None:
        self._request_explanation(mode=mode, node_ids=node_ids)

    def explain_selected_animation(self, animation_id: str, mode: Optional[str] = None) -> None:
        self._request_explanation(mode=mode, animation_id=animation_id)

    def explain_selected_objects(self, node_ids: list[str], mode: Optional[str] = None) -> None:
        self._request_explanation(mode=mode, node_ids=node_ids, objects_only=True)

    def show_auto_explanation(self, what: str, why: str) -> None:
        self.auto_text.setText(f"What just happened: {what}\nWhy it matters: {why}")
        self.auto_frame.setVisible(True)

    def request_auto_explanation(self, what_happened: str) -> None:
        analysis = self.service.analyze_scene()
        self._auto_mode_active = True
        self._set_loading(True)
        self.service.ai.request_learning_explanation(analysis, what_happened)

    def generate_lesson_notes(self) -> None:
        analysis = self.service.analyze_scene()
        self._set_loading(True)
        self.service.ai.request_lesson_notes(analysis)

    # ──────────────────────────────────────────────────────────────────
    # Internal flows
    # ──────────────────────────────────────────────────────────────────

    def regenerate(self) -> None:
        if self._last_request is None:
            self._request_explanation(mode=self._current_mode())
            return
        self._request_explanation(
            mode=self._current_mode(),
            node_ids=self._last_request.node_ids,
            animation_id=self._last_request.animation_id,
            objects_only=self._last_request.objects_only,
        )

    def copy_explanation(self) -> None:
        if not self._current_explanation:
            return
        text = self._format_explanation_text(self._current_explanation)
        QApplication.clipboard().setText(text)

    def _request_explanation(
        self,
        mode: Optional[str] = None,
        node_ids: Optional[list[str]] = None,
        animation_id: Optional[str] = None,
        objects_only: bool = False,
    ) -> None:
        mode = (mode or self._current_mode()).lower()
        analysis = self.service.analyze_scene(
            node_ids=node_ids,
            animation_id=animation_id,
            objects_only=objects_only,
        )
        self._last_request = _ExplainRequestState(
            mode=mode,
            node_ids=node_ids,
            animation_id=animation_id,
            objects_only=objects_only,
        )
        self._auto_mode_active = False
        self._set_loading(True)
        self.service.ai.request_explanation(analysis, mode)

    def _on_explanation_ready(self, response: ExplainResponse) -> None:
        self._set_loading(False)
        self._current_explanation = response
        self.auto_frame.setVisible(False)
        self.txt_concept.setPlainText(response.concept_explanation)
        self.txt_steps.setPlainText(response.step_by_step)
        self.txt_visual.setPlainText(response.visual_reasoning)
        self.txt_simple.setPlainText(response.simple_explanation)
        self.txt_takeaways.setPlainText("\n".join(f"- {k}" for k in response.key_takeaways))

    def _on_learning_ready(self, payload: dict) -> None:
        self._set_loading(False)
        what = payload.get("what_happened", "")
        why = payload.get("why_it_matters", "")
        if what or why:
            self.show_auto_explanation(what, why)
        self._auto_mode_active = False
        self.auto_explain_completed.emit()

    def _on_lesson_ready(self, notes: LessonNotes) -> None:
        self._set_loading(False)
        self._current_lesson = notes
        self.txt_lesson.setPlainText(notes.to_markdown())

    def _on_error(self, msg: str) -> None:
        self._set_loading(False)
        LOGGER.error(f"ExplainPanel error: {msg}")
        QMessageBox.warning(self, "Explain Error", msg)

    def _on_mode_changed(self) -> None:
        # Regenerate when toggled
        if self._last_request is None:
            return
        self.regenerate()

    def _set_loading(self, loading: bool) -> None:
        self.loading_label.setVisible(loading)
        self.btn_regen.setEnabled(not loading)
        self.btn_copy.setEnabled(not loading)

    def _current_mode(self) -> str:
        return "simple" if self.mode_combo.currentText().lower() == "simple" else "detailed"

    def _format_explanation_text(self, response: ExplainResponse) -> str:
        lines = [
            "What This Animation Teaches:",
            response.concept_explanation,
            "",
            "Step-by-Step Explanation:",
            response.step_by_step,
            "",
            "Why the Visuals Work:",
            response.visual_reasoning,
            "",
            "Simple Explanation:",
            response.simple_explanation,
            "",
            "Key Takeaways:",
        ]
        for k in response.key_takeaways:
            lines.append(f"- {k}")
        return "\n".join(lines).strip()

    def _export_lesson(self, mode: str) -> None:
        if not self._current_lesson:
            return
        if mode == "copy":
            QApplication.clipboard().setText(self._current_lesson.to_markdown())
            return

        ext = "md" if mode == "md" else "txt"
        filt = "Markdown (*.md)" if ext == "md" else "Text (*.txt)"
        path, _ = QFileDialog.getSaveFileName(self, "Export Lesson Notes", f"lesson.{ext}", filt)
        if not path:
            return
        content = (
            self._current_lesson.to_markdown()
            if ext == "md"
            else self._current_lesson.to_markdown().replace("# ", "").replace("## ", "")
        )
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as exc:
            QMessageBox.warning(self, "Export Failed", str(exc))


class LearningModeController(QObject):
    """Auto-explains significant changes while building animations."""

    def __init__(self, main_window, explain_panel: ExplainPanel, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.explain_panel = explain_panel
        self.enabled = bool(SETTINGS.get("LEARNING_MODE_ENABLED", False, type=bool))
        self._debounce = QTimer()
        self._debounce.setInterval(1500)
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self._flush_pending)
        self._pending_events: list[str] = []
        self._queue: list[str] = []
        self._in_progress = False
        self._last_snapshot = None

        SETTINGS.settings_changed.connect(self._sync_enabled)

        if getattr(main_window, "history_manager", None):
            main_window.history_manager.history_changed.connect(self._on_history_change)
            if hasattr(main_window.history_manager, "checkpoint_created"):
                main_window.history_manager.checkpoint_created.connect(self._on_checkpoint)

        self.explain_panel.auto_explain_completed.connect(self._on_auto_done)

    def _sync_enabled(self) -> None:
        self.enabled = bool(SETTINGS.get("LEARNING_MODE_ENABLED", False, type=bool))

    def _on_checkpoint(self, checkpoint):
        if not self.enabled:
            return
        self._enqueue_event("A checkpoint was created to save your progress")

    def _on_history_change(self):
        if not self.enabled:
            return
        history = getattr(self.main_window, "history_manager", None)
        snapshot = getattr(history, "_current_snapshot", None) if history else None
        if snapshot is None:
            return
        if self._last_snapshot is None:
            self._last_snapshot = snapshot
            return
        before_snapshot = self._last_snapshot
        diff = self._compute_diff(before_snapshot, snapshot)
        self._last_snapshot = snapshot
        if self._is_minor_diff(diff, before_snapshot, snapshot):
            return
        event = self._describe_trigger(diff, snapshot)
        if event:
            self._enqueue_event(event)

    def _enqueue_event(self, what_happened: str) -> None:
        if self._debounce.isActive():
            self._pending_events.append(what_happened)
            return
        self._pending_events = [what_happened]
        self._debounce.start()

    def _flush_pending(self) -> None:
        if not self._pending_events:
            return
        combined = self._combine_events(self._pending_events)
        self._pending_events = []
        if self._in_progress:
            self._queue.append(combined)
            return
        self._start_event(combined)

    def _start_event(self, what_happened: str) -> None:
        self._in_progress = True
        # Immediate heuristic explanation
        why = self._infer_why_it_matters()
        self.explain_panel.show_auto_explanation(what_happened, why)
        self.explain_panel.request_auto_explanation(what_happened)
        try:
            if hasattr(self.main_window, "explain_dock"):
                self.main_window.explain_dock.show()
        except Exception:
            pass

    def _on_auto_done(self) -> None:
        self._in_progress = False
        if self._queue:
            next_event = self._queue.pop(0)
            QTimer.singleShot(1500, lambda: self._start_event(next_event))

    def _combine_events(self, events: list[str]) -> str:
        if len(events) == 1:
            return events[0]
        trimmed = "; ".join(events[:3])
        return f"Several updates were made: {trimmed}."

    def _infer_why_it_matters(self) -> str:
        analysis = self.explain_panel.service.analyze_scene()
        hints = set(analysis.concept_hints)
        if "slope or derivative" in hints:
            return "This highlights the idea of slope at a point, which connects to derivatives."
        if "area under a curve" in hints:
            return "This connects the visuals to area under a curve, the core idea of integration."
        if "function graph" in hints:
            return "This helps relate the picture to how a function behaves."
        if "equation or expression" in hints:
            return "This links the visuals to the meaning of the equation."
        if "geometric shapes" in hints:
            return "This helps reason about shape properties and spatial relationships."
        return "This change adds a visual step that supports the concept being taught."

    def _compute_diff(self, before, after) -> dict:
        before_nodes = set(before.nodes.keys())
        after_nodes = set(after.nodes.keys())
        return {
            "added": sorted(after_nodes - before_nodes),
            "removed": sorted(before_nodes - after_nodes),
            "changed": sorted(
                nid
                for nid in (before_nodes & after_nodes)
                if before.nodes[nid].fingerprint != after.nodes[nid].fingerprint
            ),
        }

    def _is_minor_diff(self, diff: dict, before, after) -> bool:
        if diff["added"] or diff["removed"]:
            return False
        for nid in diff["changed"]:
            b = before.nodes[nid].data
            a = after.nodes[nid].data
            if not self._only_minor_changes(b, a):
                return False
        return True

    def _only_minor_changes(self, before_data: dict, after_data: dict) -> bool:
        ignore_keys = {"pos", "pos_x", "pos_y", "name"}
        style_keys = {"color", "stroke_width", "stroke_opacity", "fill_opacity", "opacity"}
        for key in set(before_data.keys()) | set(after_data.keys()):
            if key in ignore_keys:
                continue
            if key == "params":
                b_params = before_data.get("params", {})
                a_params = after_data.get("params", {})
                for pkey in set(b_params.keys()) | set(a_params.keys()):
                    if pkey in style_keys:
                        continue
                    if b_params.get(pkey) != a_params.get(pkey):
                        return False
                continue
            if before_data.get(key) != after_data.get(key):
                return False
        return True

    def _describe_trigger(self, diff: dict, snapshot) -> Optional[str]:
        added = diff["added"]
        if len(added) >= 5:
            return "You added a large batch of new elements to the scene."

        # Analyze added nodes
        nodes = snapshot.nodes
        for nid in added:
            data = nodes[nid].data
            cls_name = str(data.get("cls_name", ""))
            ntype = str(data.get("type", ""))
            name = str(data.get("name", ""))
            if cls_name in ("Axes", "NumberPlane", "ThreeDAxes"):
                return "You added axes to the scene."
            if cls_name in ("MathTex", "Tex"):
                return "You added a mathematical expression."
            if "Graph" in cls_name or "Function" in cls_name:
                return "You added a function graph."
            if ntype == "ANIMATION" and cls_name in (
                "Transform",
                "ReplacementTransform",
                "TransformMatchingTex",
            ):
                return "You added a transformation between objects."
            if "tangent" in name.lower() or "tangent" in cls_name.lower():
                return "You added a tangent line."
            if "slope" in name.lower() or "derivative" in name.lower():
                return "You added a slope or derivative indicator."
            if "area" in name.lower() or "integral" in name.lower():
                return "You added an area or integral visual."

        # Major structural change
        if len(diff["added"]) + len(diff["removed"]) >= 3:
            return "You made a major structural change to the animation."
        return None


class TeacherModeController(QObject):
    """Auto-generates lesson notes when key teaching milestones occur."""

    def __init__(self, main_window, explain_panel: ExplainPanel, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.explain_panel = explain_panel
        self.enabled = bool(SETTINGS.get("TEACHER_MODE_ENABLED", False, type=bool))
        self._last_fingerprint: Optional[str] = None

        SETTINGS.settings_changed.connect(self._sync_enabled)

        if getattr(main_window, "history_manager", None):
            main_window.history_manager.history_changed.connect(self._on_history_change)
            if hasattr(main_window.history_manager, "checkpoint_created"):
                main_window.history_manager.checkpoint_created.connect(self._on_checkpoint)

    def _sync_enabled(self) -> None:
        self.enabled = bool(SETTINGS.get("TEACHER_MODE_ENABLED", False, type=bool))

    def _on_checkpoint(self, checkpoint):
        if not self.enabled:
            return
        self._trigger_if_needed()

    def _on_history_change(self):
        if not self.enabled:
            return
        self._trigger_if_needed()

    def _trigger_if_needed(self):
        history = getattr(self.main_window, "history_manager", None)
        snapshot = getattr(history, "_current_snapshot", None) if history else None
        fingerprint = getattr(snapshot, "fingerprint", None)
        if fingerprint and fingerprint == self._last_fingerprint:
            return

        analysis = self.explain_panel.service.analyze_scene()
        if self._is_teaching_milestone(analysis):
            self._last_fingerprint = fingerprint
            self.explain_panel.generate_lesson_notes()

    def _is_teaching_milestone(self, analysis) -> bool:
        types = {o.type for o in analysis.objects}
        has_axes = any(t in ("Axes", "NumberPlane", "ThreeDAxes") for t in types)
        has_math = any(t in ("MathTex", "Tex") for t in types)
        has_graph = any("Graph" in t or "Function" in t for t in types)
        has_transform = any(
            step.animation_type in ("Transform", "ReplacementTransform", "TransformMatchingTex")
            for step in analysis.animation_steps
        )

        if has_axes and has_graph and has_math and has_transform:
            return True
        if len(analysis.animation_steps) >= 5:
            return True
        return False
    