"""
sdk_generator.py — Extension SDK Template Generator

Creates scaffolded extension projects.
Provides:
  - metadata.json template
  - lib.py template with API examples
  - requirements.txt
  - README.md with documentation
  - example_node.py
  - example_timeline_track.py
  - mcp_hooks.py
  - Testing template
  - Signing instructions
"""

from pathlib import Path
from typing import Optional

TEMPLATES = {
    "metadata.json": """{
  "name": "{extension_name}",
  "author": "{author}",
  "version": "0.1.0",
  "description": "A deterministic extension for EfficientManim",
  "engine_version": ">=2.0.4",
  "permissions": [
    "register_nodes",
    "register_timeline_track"
  ],
  "dependencies": [],
  "entry_file": "lib.py",
  "has_signature": false,
  "verified": false,
  "changelog": "Initial release"
}
""",
    "lib.py": """\"\"\"
{extension_name} — Main entry point for the extension.

This module is loaded by ExtensionManager and must contain
a setup() function that receives the ExtensionAPI instance.
\"\"\"

from extension_api import ExtensionAPI

# Initialize module flag
_initialized = False


def setup(api: ExtensionAPI) -> None:
    \"\"\"
    Initialize the extension.
    
    Called by ExtensionManager after permissions are approved.
    
    Args:
        api: ExtensionAPI instance for interacting with host
    \"\"\"
    global _initialized
    
    if _initialized:
        return
    
    _initialized = True
    
    # Register deterministic nodes
    register_nodes(api)
    
    # Register timeline tracks
    register_tracks(api)
    
    # Register MCP hooks
    register_hooks(api)
    
    print(f"✓ {__name__} initialized")


def register_nodes(api: ExtensionAPI) -> None:
    \"\"\"Register custom deterministic nodes.\"\"\"
    
    # Example: Register custom node
    api.register_node(
        "ExampleNode",
        "example_node.ExampleNodeClass",
        category="Custom",
        description="Example custom node"
    )


def register_tracks(api: ExtensionAPI) -> None:
    \"\"\"Register custom timeline tracks.\"\"\"
    
    # Example: Register custom timeline track
    api.register_timeline_track(
        "ExampleTrack",
        "example_timeline_track.ExampleTrack",
        description="Example custom timeline track"
    )


def register_hooks(api: ExtensionAPI) -> None:
    \"\"\"Register MCP hooks for AI integration.\"\"\"
    
    # Example: Register pre-render hook
    api.register_mcp_hook(
        "pre_render",
        "mcp_hooks.pre_render_hook"
    )


# Make setup discoverable
__all__ = ["setup"]
""",
    "example_node.py": """\"\"\"
Example deterministic node.
\"\"\"

from manim import *


class ExampleNodeClass(VMobject):
    \"\"\"
    Example custom node class.
    
    Deterministic contract:
    - Same parameters → same output always
    - No random initialization
    - Explicitly set all properties
    \"\"\"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # IMPORTANT: Never use random()
        # All initialization must be deterministic
        
        # Set explicit color
        self.set_color(BLUE)
        
        # Create geometry
        circle = Circle(radius=0.5)
        self.add(circle)


def create(params: dict) -> ExampleNodeClass:
    \"\"\"Factory function for creating the node.\"\"\"
    return ExampleNodeClass()
""",
    "example_timeline_track.py": """\"\"\"
Example timeline track.
\"\"\"


class ExampleTrack:
    \"\"\"Custom timeline track for organizing content.\"\"\"
    
    def __init__(self, name: str = "Example Track"):
        self.name = name
        self.blocks = []
        self.collapsed = False
    
    def add_block(self, start_time: float, duration: float, label: str = ""):
        \"\"\"Add a block to the track.\"\"\"
        self.blocks.append({
            "start": start_time,
            "duration": duration,
            "label": label
        })
    
    def to_dict(self) -> dict:
        \"\"\"Serialize track state.\"\"\"
        return {
            "name": self.name,
            "blocks": self.blocks,
            "collapsed": self.collapsed
        }
""",
    "mcp_hooks.py": """\"\"\"
MCP integration hooks.
Allow AI to interact with the extension.
\"\"\"


async def pre_render_hook(context):
    \"\"\"Called before rendering starts.\"\"\"
    print("Pre-render hook called")
    return {"status": "ready"}


async def post_render_hook(context):
    \"\"\"Called after rendering completes.\"\"\"
    print("Post-render hook called")
    return {"status": "complete"}
""",
    "requirements.txt": """# Add dependencies here
# Example:
# manim>=0.18.0
# numpy>=1.20.0
""",
    "README.md": """# {extension_name}

A deterministic extension for EfficientManim.

## Installation

### From GitHub

```bash
efficientmanim install {author}/{extension_name}
```

### From Marketplace

Search for "{extension_name}" in the EfficientManim marketplace.

## Features

- Deterministic nodes
- Custom timeline tracks
- MCP AI integration hooks

## Usage

After installation:

1. Open EfficientManim
2. Go to Tools → Extensions → {extension_name}
3. Approve permissions
4. New nodes/tracks appear in the editor

## Development

### Structure

- `metadata.json` — Extension metadata
- `lib.py` — Main entry point
- `example_node.py` — Custom node example
- `example_timeline_track.py` — Custom track example
- `mcp_hooks.py` — AI integration
- `requirements.txt` — Dependencies

### Testing

```bash
python -m pytest tests/
```

## Signing Your Extension

For publishing to marketplace:

```bash
efficientmanim sign-extension --private-key your_private.pem
```

## Publishing

```bash
git push origin main
# Then submit via marketplace: efficientmanim.dev/publish
```

## License

MIT
""",
    "test_extension.py": """\"\"\"
Test suite for extension.
\"\"\"

import pytest
from lib import setup


class MockAPI:
    \"\"\"Mock ExtensionAPI for testing.\"\"\"
    
    def __init__(self):
        self.registered_nodes = []
        self.registered_tracks = []
        self.registered_hooks = []
    
    def register_node(self, name, class_path, **kwargs):
        self.registered_nodes.append(name)
    
    def register_timeline_track(self, name, class_path, **kwargs):
        self.registered_tracks.append(name)
    
    def register_mcp_hook(self, hook_name, callback):
        self.registered_hooks.append(hook_name)


def test_setup():
    \"\"\"Test that setup initializes correctly.\"\"\"
    api = MockAPI()
    setup(api)
    
    # Should register nodes and tracks
    assert len(api.registered_nodes) > 0
    assert len(api.registered_tracks) > 0
    assert len(api.registered_hooks) > 0


def test_deterministic():
    \"\"\"Test that nodes are deterministic.\"\"\"
    from example_node import ExampleNodeClass
    
    # Same parameters should create identical objects
    node1 = ExampleNodeClass()
    node2 = ExampleNodeClass()
    
    # Both should be the same type
    assert type(node1) == type(node2)
""",
}


class SDKGenerator:
    """Generates extension project templates."""

    @staticmethod
    def create_extension(
        extension_name: str, author: str, output_dir: Optional[Path] = None
    ) -> Path:
        """
        Create extension template project.

        Creates:
        {extension_name}/
        ├── metadata.json
        ├── lib.py
        ├── requirements.txt
        ├── README.md
        ├── example_node.py
        ├── example_timeline_track.py
        ├── mcp_hooks.py
        ├── test_extension.py
        └── SIGNING.md

        Args:
            extension_name: Name of extension (e.g., "MyExtension")
            author: Author name (e.g., "github_username")
            output_dir: Where to create project (default: current directory)

        Returns:
            Path to created project
        """
        if output_dir is None:
            output_dir = Path.cwd()

        project_dir = output_dir / extension_name
        project_dir.mkdir(parents=True, exist_ok=True)

        # Create files from templates
        for filename, content in TEMPLATES.items():
            file_path = project_dir / filename

            # Substitute placeholders
            content = content.format(extension_name=extension_name, author=author)

            with open(file_path, "w") as f:
                f.write(content)

        # Create SIGNING.md
        signing_md = project_dir / "SIGNING.md"
        with open(signing_md, "w") as f:
            f.write(f"""# Signing {extension_name}

## Create Signing Keys

```bash
openssl genrsa -out private_key.pem 2048
openssl rsa -in private_key.pem -pubout -out public_key.pem
```

## Sign Your Extension

```bash
efficientmanim sign-extension \\
  --extension {extension_name} \\
  --private-key private_key.pem \\
  --output signature.sig
```

## Publish

1. Add `signature.sig` and `public_key.pem` to repo
2. Push to GitHub
3. Submit to marketplace at https://marketplace.efficientmanim.dev

## Verification

Users will see:
- 🟢 Verified (your signature matches your public key)
- 🟡 Signed (signature valid but author unverified)
- 🔴 Unsigned (no signature)
- ❌ Tampered (signature invalid)

Keep your private key secure!
""")

        # Create .gitignore
        gitignore = project_dir / ".gitignore"
        with open(gitignore, "w") as f:
            f.write("""# Virtual environment
venv/
.venv/

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Keys (never commit private keys!)
private_key.pem
*.pem

# Local testing
.pytest_cache/
.coverage
""")

        print(f"✓ Extension template created at {project_dir}")
        print("\nNext steps:")
        print(f"1. cd {extension_name}")
        print("2. Edit lib.py to customize your extension")
        print("3. Add dependencies to requirements.txt")
        print("4. Follow SIGNING.md to sign your extension")
        print("5. Push to GitHub and submit to marketplace")

        return project_dir


def create_extension_cli(name: str, author: str):
    """CLI command: efficientmanim create-extension MyExtension"""
    try:
        SDKGenerator.create_extension(name, author)
        print(f"✓ Extension '{name}' created successfully!")
    except Exception as e:
        print(f"✗ Failed to create extension: {e}")
