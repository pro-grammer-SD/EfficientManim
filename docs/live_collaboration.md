# 🤝 EfficientManim — Live Collaboration

## Concept

Live Collaboration lets two or more instances of EfficientManim edit the same node graph simultaneously, synchronized in real-time. A human-readable 6-digit PIN identifies the session — no cloud accounts, no third-party relay services, no configuration required. The host machine runs a WebSocket server; participants connect directly to it.

Changes to any node (position, property, wire, scene switch) are transmitted as lightweight JSON "deltas" and applied to all connected instances within milliseconds.

---

## Starting a Session (Host)

1. Open your project in EfficientManim.
2. Click **Collaboration → Start Live Collaboration** in the menu bar.
3. The app selects a free port automatically and starts a WebSocket server.
4. A dialog displays your **6-digit PIN** in large bold text. Share it verbally or paste it into chat.
5. The toolbar shows a green dot (●) and the active PIN.

The host's machine must remain running for the duration of the session.

---

## Joining a Session (Participant)

1. Click **Collaboration → Join Collaboration**.
2. Type the 6-digit PIN into the input dialog.
3. Click **Connect**.
4. The app resolves the PIN to a host IP and port, connects via WebSocket, and loads the full graph from the host.

If the session is on the same machine (multi-window), the PIN is resolved instantly from `~/.efficientmanim/collab_sessions.json`. For LAN connections, the host's local IP must be reachable from the participant's machine.

---

## Multi-Window Usage

You can join your own session from a second EfficientManim window:

1. Start a session in Window 1 — note the PIN.
2. Open a new EfficientManim window (run `python main.py` again).
3. Click **Collaboration → Join Collaboration** and enter the PIN.
4. Both windows now share the same live graph.

This is identical in protocol to a remote participant joining — no special-casing.

---

## Delta Synchronization

Every change broadcasts a typed JSON delta:

```json
{
  "type": "delta",
  "session_pin": "482917",
  "sender_id": "client-uuid-1234",
  "timestamp": 1720000000.123,
  "action": "node_moved",
  "payload": { "node_id": "circle_1", "x": 320.5, "y": 180.0 }
}
```

Supported actions:

| Action | What triggers it |
|---|---|
| `node_added` | Adding any node to the canvas |
| `node_deleted` | Removing a node |
| `node_moved` | Dragging a node |
| `node_property_changed` | Editing any parameter in the Properties panel |
| `wire_added` | Drawing a connection |
| `wire_deleted` | Removing a wire |
| `scene_switched` | Switching the active scene |
| `vgroup_created` | Creating a VGroup |
| `vgroup_deleted` | Deleting a VGroup |
| `node_lock` | Starting to edit a node (mouse press) |
| `node_unlock` | Releasing a node (mouse release) |
| `full_graph_sync` | Sent to every new joiner — full graph state |

Deltas are applied directly to the local graph without going through the undo/redo stack, so remote changes do not pollute your undo history.

---

## Conflict Resolution

**Last-Writer-Wins** is the default strategy. Every delta carries a `timestamp`; when two clients edit the same node near-simultaneously, the later timestamp wins.

**Node Locking** provides soft exclusion:
- When you click a node, a `node_lock` delta is broadcast. Other clients see a colored border around the node and a 🔒 icon.
- You cannot interact with a node locked by another user.
- Locks expire automatically after 10 seconds if no `node_unlock` delta arrives (prevents deadlock if a client crashes).
- When you release a node, `node_unlock` is broadcast and the lock is cleared.

Lock owner colors are assigned randomly per session participant.

---

## Session Lifecycle

**Starting** creates an entry in `~/.efficientmanim/collab_sessions.json`:
```json
{
  "482917": { "host": "192.168.1.10", "port": 49823, "created_at": 1720000000.0 }
}
```

**Ending** removes the entry and broadcasts `session_ended` to all clients.

**Inactivity timeout** — if no delta is sent or received for 30 minutes (configurable in Settings), the host server auto-terminates the session.

**Snapshots** — the host auto-saves the full graph to `~/.efficientmanim/collab_snapshots/<pin>_<timestamp>.json` every 5 minutes. These are plain JSON files you can inspect or restore manually.

**Participant disconnect** — the server removes the client from its registry. Other participants continue unaffected. A reconnecting participant receives a fresh `full_graph_sync`.

---

## Network Requirements

**Same machine (multi-window):** No network access required — uses `127.0.0.1`.

**Local Area Network:** The host machine's firewall must allow incoming TCP connections on the ephemeral port selected for the session. The port number is shown in the session info dialog.

**Internet:** Not supported out of the box. Both machines must be on the same LAN. For internet use, run a reverse proxy (e.g. `ngrok`) on the host machine.

---

## Known Limitations

- Asset files (images, audio) are not synced automatically. Each participant needs local copies at matching paths.
- AI code generation and TTS are per-window — generating AI code in one window does not broadcast it to others until you merge it.
- Render jobs are local — clicking Render Video only renders on the machine that clicked it.
- No end-to-end encryption. Do not share PINs for sensitive projects over untrusted networks.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────┐
│              HOST MACHINE                   │
│  ┌─────────────────┐   ┌─────────────────┐  │
│  │ EfficientManim  │   │ WebSocket Server│  │
│  │   Window 1      │──▶│  Port: 49823    │  │
│  └─────────────────┘   │  PIN:  482917   │  │
│                         └────────┬────────┘  │
└─────────────────────────────────│────────────┘
                                  │ WebSocket
              ┌───────────────────┼────────────────────┐
              ▼                   ▼                     ▼
   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
   │ EfficientManim  │  │ EfficientManim  │  │ EfficientManim  │
   │  Window 2       │  │  Window 3       │  │  Remote Client  │
   │ (same machine)  │  │ (same machine)  │  │ (LAN)           │
   └─────────────────┘  └─────────────────┘  └─────────────────┘
         All connected via PIN: 482917 — deltas broadcast in real-time
```

**Delta flow:**
```
User edits node → CollaborationManager.send_delta()
    → [if hosting] CollabServer.broadcast_delta() → all WebSocket clients
    → [if joining] CollabClient.send() → host server → all other clients
    
Incoming delta → apply_delta(window, delta) → QTimer.singleShot(0, _do)
    → window._collab_applying = True
    → modify graph directly (no undo stack)
    → window.compile_graph()
    → window._collab_applying = False
```
