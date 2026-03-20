from __future__ import annotations
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal

from core.config import AppPaths
from utils.logger import LOGGER


class UserDataManager:
    """Tracks usage statistics, recent projects, and persists user preferences."""

    _BASE_DIR = Path.home() / ".efficientmanim"
    _RECENTS_FILE = _BASE_DIR / "recents.json"
    _USAGE_FILE = _BASE_DIR / "usage.json"
    _KEYBINDINGS_FILE = _BASE_DIR / "keybindings.json"

    def __init__(self):
        self._BASE_DIR.mkdir(parents=True, exist_ok=True)
        self._recents: list[str] = self._load_json(self._RECENTS_FILE, [])
        self._usage: dict[str, Any] = self._load_json(self._USAGE_FILE, {})
        self._keybindings: dict[str, str] = self._load_json(self._KEYBINDINGS_FILE, {})

    @staticmethod
    def _load_json(path: Path, default):
        try:
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
        return default

    @staticmethod
    def _save_json(path: Path, data) -> None:
        try:
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def add_recent(self, path: str) -> None:
        if path in self._recents:
            self._recents.remove(path)
        self._recents.insert(0, path)
        self._recents = self._recents[:10]
        self._save_json(self._RECENTS_FILE, self._recents)

    def get_recents(self) -> list[str]:
        self._recents = [p for p in self._recents if Path(p).exists()]
        return list(self._recents)

    def record_use(self, class_name: str, node_type: str = "mobject") -> None:
        bucket = (
            "animation"
            if str(node_type).lower() in ("animation", "anim")
            else "mobject"
        )
        if not isinstance(self._usage, dict):
            self._usage = {}
        typed = self._usage.setdefault(bucket, {})
        typed[class_name] = typed.get(class_name, 0) + 1
        self._save_json(self._USAGE_FILE, self._usage)

    def top_by_type(self, node_type: str, n: int = 5) -> list:
        bucket = (
            "animation"
            if str(node_type).lower() in ("animation", "anim")
            else "mobject"
        )
        typed = self._usage.get(bucket, {}) if isinstance(self._usage, dict) else {}
        return sorted(typed.items(), key=lambda kv: kv[1], reverse=True)[:n]

    def top_used(self, n: int = 5) -> list:
        combined: dict[str, int] = {}
        if isinstance(self._usage, dict):
            for bucket in ("mobject", "animation"):
                for k, v in self._usage.get(bucket, {}).items():
                    combined[k] = combined.get(k, 0) + v
        return sorted(combined, key=lambda k: combined[k], reverse=True)[:n]

    def get_keybinding(self, action: str, default: str = "") -> str:
        return self._keybindings.get(action, default)

    def set_keybinding(self, action: str, shortcut: str) -> None:
        self._keybindings[action] = shortcut
        self._save_json(self._KEYBINDINGS_FILE, self._keybindings)

    def get_all_keybindings(self) -> dict:
        return dict(self._keybindings)


USER_DATA = UserDataManager()


class UsageTracker(QObject):
    updated = Signal()

    def __init__(self):
        super().__init__()

    def record(self, class_name: str, node_type: str = "mobject") -> None:
        USER_DATA.record_use(class_name, node_type)
        self.updated.emit()

    def top_mobjects(self, n: int = 5) -> list:
        return USER_DATA.top_by_type("mobject", n)

    def top_animations(self, n: int = 5) -> list:
        return USER_DATA.top_by_type("animation", n)

    def top(self, n: int = 5) -> list:
        return USER_DATA.top_used(n)


USAGE_TRACKER = UsageTracker()


def add_recent(path: str) -> None:
    USER_DATA.add_recent(path)


def get_recents() -> list[str]:
    return USER_DATA.get_recents()


class Asset:
    def __init__(self, name: str, path: str, kind: str):
        self.id = str(__import__("uuid").uuid4())
        self.name = name
        self.original_path = str(Path(path).as_posix())
        self.current_path = str(Path(path).as_posix())
        self.kind = kind
        self.local_file = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "original": self.original_path,
            "kind": self.kind,
            "local": self.local_file,
        }

    @staticmethod
    def from_dict(d: dict) -> "Asset":
        a = Asset(d["name"], d["original"], d["kind"])
        a.id = d["id"]
        a.local_file = d.get("local", "")

        original = Path(d["original"])
        if original.exists():
            a.current_path = original.as_posix()
            LOGGER.info(f"Asset '{a.name}' (id={a.id[:8]}): original path valid")
            return a

        if a.local_file and (AppPaths.TEMP_DIR / a.local_file).exists():
            a.current_path = (AppPaths.TEMP_DIR / a.local_file).as_posix()
            LOGGER.warn(
                f"Asset '{a.name}' (id={a.id[:8]}): original missing, using temp: {a.current_path}"
            )
            return a

        user_assets = AppPaths.USER_DATA / "assets" / a.local_file
        if a.local_file and user_assets.exists():
            a.current_path = user_assets.as_posix()
            LOGGER.warn(
                f"Asset '{a.name}' (id={a.id[:8]}): found in user assets: {a.current_path}"
            )
            return a

        LOGGER.error(
            f"Asset '{a.name}' (id={a.id[:8]}): missing - original={a.original_path}, local={a.local_file}."
        )
        a.current_path = a.original_path
        return a


class AssetManager(QObject):
    assets_changed = Signal()

    def __init__(self):
        super().__init__()
        self.assets: dict[str, Asset] = {}

    def add_asset(self, path: str) -> Asset | None:
        path_obj = Path(path)
        if not path_obj.exists():
            return None

        clean_path = path_obj.resolve().as_posix()

        kind = "unknown"
        s = path_obj.suffix.lower()
        if s in [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".svg"]:
            kind = "image"
        elif s in [".mp4", ".mov", ".avi", ".webm"]:
            kind = "video"
        elif s in [".mp3", ".wav", ".ogg"]:
            kind = "audio"

        if kind == "unknown":
            LOGGER.error(f"Unsupported asset type: {s}")
            return None

        asset = Asset(path_obj.name, clean_path, kind)
        self.assets[asset.id] = asset
        self.assets_changed.emit()
        LOGGER.info(f"Asset added: {asset.name}")
        return asset

    def clear(self) -> None:
        self.assets.clear()
        self.assets_changed.emit()

    def get_list(self) -> list[Asset]:
        return list(self.assets.values())

    def get_asset(self, asset_id: str) -> Asset | None:
        return self.assets.get(asset_id)

    def delete_asset(self, asset_id: str) -> bool:
        if asset_id in self.assets:
            del self.assets[asset_id]
            self.assets_changed.emit()
            return True
        return False

    def update_asset(
        self, asset_id: str, new_path: str | None = None, new_name: str | None = None
    ) -> Asset | None:
        asset = self.assets.get(asset_id)
        if not asset:
            return None
        if new_path:
            asset.current_path = str(Path(new_path).resolve().as_posix())
        if new_name:
            asset.name = new_name
        self.assets_changed.emit()
        return asset

    def get_asset_path(self, asset_id: str) -> str | None:
        if asset_id in self.assets:
            asset = self.assets[asset_id]
            path_str = asset.current_path
            path_obj = Path(path_str)
            if not path_obj.exists():
                LOGGER.error(
                    f"Asset file missing: '{asset.name}' (id={asset_id[:8]}) -> {path_str}"
                )
                return None
            return path_str

        LOGGER.error(f"Unknown asset ID: {asset_id}")
        return None


ASSETS = AssetManager()
