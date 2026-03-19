from __future__ import annotations
# -*- coding: utf-8 -*-

from typing import Optional


def build_tooltip(
    title: str,
    description: str,
    hint: str,
    shortcut: Optional[str] = None,
) -> str:
    parts = [f"<b>{title}</b>", description, f"<i>{hint}</i>"]
    if shortcut:
        parts.append(f"Shortcut: <b>{shortcut}</b>")
    return "<br>".join(parts)


def apply_tooltip(
    widget,
    description: str,
    hint: str,
    shortcut: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    label = title or getattr(widget, "text", lambda: "Action")()
    tip = build_tooltip(label, description, hint, shortcut)
    if hasattr(widget, "setToolTip"):
        widget.setToolTip(tip)
    if hasattr(widget, "setStatusTip"):
        widget.setStatusTip(description)
