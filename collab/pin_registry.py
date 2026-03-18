"""
collab/pin_registry.py — PIN registry for live collaboration sessions.

Stores { PIN → {host, port, created_at} } in a JSON file so that a second
EfficientManim window on the same machine can look up a session by PIN.

FIX (BUG 4): _save_registry now ensures the directory exists before every
write and logs errors instead of swallowing them silently.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

LOGGER = logging.getLogger("collab.registry")

# Canonical registry path — both windows share the same home directory
REGISTRY_PATH = Path.home() / ".efficientmanim" / "collab_sessions.json"


@dataclass
class SessionInfo:
    pin: str
    host: str       # "127.0.0.1" for same-machine sessions
    port: int
    created_at: float

    def to_dict(self) -> dict:
        return {
            "pin": self.pin,
            "host": self.host,
            "port": self.port,
            "created_at": self.created_at,
        }

    @staticmethod
    def from_dict(d: dict) -> "SessionInfo":
        return SessionInfo(
            pin=str(d.get("pin", "")),
            host=str(d.get("host", "127.0.0.1")),
            port=int(d.get("port", 0)),
            created_at=float(d.get("created_at", 0.0)),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_registry() -> Dict[str, dict]:
    """Load and return the registry dict. Returns {} on any failure."""
    if not REGISTRY_PATH.exists():
        LOGGER.debug(f"Registry file does not exist: {REGISTRY_PATH}")
        return {}
    try:
        content = REGISTRY_PATH.read_text(encoding="utf-8").strip()
        if not content:
            return {}
        return json.loads(content)
    except json.JSONDecodeError as exc:
        LOGGER.warning(f"Registry file corrupt ({exc}), treating as empty: {REGISTRY_PATH}")
        # Wipe the corrupt file so it doesn't block future writes
        try:
            REGISTRY_PATH.write_text("{}", encoding="utf-8")
        except Exception:
            pass
        return {}
    except Exception as exc:
        LOGGER.error(f"Failed to read registry: {exc}")
        return {}


def _save_registry(data: Dict[str, dict]) -> bool:
    """
    Write registry to disk. Returns True on success.

    FIX (BUG 4): explicitly ensures parent directory exists, logs all
    errors, and does NOT silently swallow failures.
    """
    try:
        # Ensure directory exists before every write (not just at import time)
        REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Atomic-ish write: write to temp then rename
        tmp = REGISTRY_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(REGISTRY_PATH)
        LOGGER.debug(f"Registry saved: {len(data)} session(s) → {REGISTRY_PATH}")
        return True
    except Exception as exc:
        LOGGER.error(f"Failed to save registry to {REGISTRY_PATH}: {exc}")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def register_session(pin: str, host: str, port: int) -> SessionInfo:
    """
    Register a live session in the registry.
    Should only be called AFTER the WebSocket server is confirmed listening.

    host should be "127.0.0.1" for same-machine sessions (the server binds
    to 0.0.0.0 so both 127.0.0.1 and LAN IP work, but 127.0.0.1 is what we
    store for guaranteed same-machine connectivity).
    """
    data = _load_registry()
    info = SessionInfo(pin=pin, host=host, port=port, created_at=time.time())
    data[pin] = info.to_dict()
    ok = _save_registry(data)
    if ok:
        LOGGER.info(f"Session registered: PIN={pin} at {host}:{port}")
    else:
        LOGGER.error(f"Session registration FAILED for PIN={pin}")
    return info


def remove_session(pin: str) -> None:
    """Remove a session from the registry."""
    data = _load_registry()
    if pin in data:
        del data[pin]
        _save_registry(data)
        LOGGER.info(f"Session removed: PIN={pin}")


def resolve_pin(pin: str) -> Optional[SessionInfo]:
    """
    Look up a PIN in the registry. Returns SessionInfo or None.
    Logs a clear message when not found.
    """
    data = _load_registry()
    LOGGER.debug(f"resolve_pin({pin}): registry has {len(data)} entries: {list(data.keys())}")
    raw = data.get(str(pin))
    if raw is None:
        LOGGER.warning(
            f"PIN '{pin}' not found. "
            f"Registry at {REGISTRY_PATH} has {len(data)} entry/entries: "
            f"{list(data.keys()) or 'empty'}"
        )
        return None
    try:
        info = SessionInfo.from_dict(raw)
        LOGGER.info(f"PIN '{pin}' resolved → {info.host}:{info.port}")
        return info
    except Exception as exc:
        LOGGER.error(f"PIN '{pin}' found but malformed: {exc}. Raw: {raw}")
        return None


def cleanup_expired(ttl_seconds: float) -> None:
    """Remove sessions older than ttl_seconds."""
    if ttl_seconds <= 0:
        return
    data = _load_registry()
    now = time.time()
    expired = [
        pin for pin, info in data.items()
        if now - float(info.get("created_at", 0)) > ttl_seconds
    ]
    if expired:
        for pin in expired:
            del data[pin]
        _save_registry(data)
        LOGGER.info(f"Cleaned up {len(expired)} expired session(s): {expired}")


def debug_dump() -> str:
    """Return a readable snapshot of the current registry (for logging)."""
    data = _load_registry()
    if not data:
        return f"Registry empty ({REGISTRY_PATH})"
    lines = [f"Registry ({REGISTRY_PATH}):"]
    for pin, info in data.items():
        age = time.time() - float(info.get("created_at", 0))
        lines.append(f"  PIN {pin} → {info.get('host')}:{info.get('port')} (age {age:.0f}s)")
    return "\n".join(lines)
