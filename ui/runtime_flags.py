MCP_AVAILABLE = False
PYDUB_AVAILABLE = False

try:
    MCP_AVAILABLE = True
except Exception:
    pass

try:
    PYDUB_AVAILABLE = True
except Exception:
    pass
