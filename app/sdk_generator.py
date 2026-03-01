"""
sdk_generator.py — Extension SDK Template Generator

Creates scaffolded extension projects.
"""

from pathlib import Path
from typing import Optional

TEMPLATES: dict = {
    "metadata.json": """{
  "name": "{extension_name}",
  "author": "{author}",
  "version": "0.1.0",
  "description": "A deterministic extension for EfficientManim",
  "engine_version": ">=2.0.3",
  "permissions": ["register_nodes", "register_ui_panel"],
  "dependencies": [],
  "entry_file": "lib.py",
  "has_signature": false,
  "verified": false,
  "changelog": "Initial release"
}
""",
}


class SDKGenerator:
    """Generates extension project templates."""

    @staticmethod
    def create_extension(
        extension_name: str, author: str, output_dir: Optional[Path] = None
    ) -> Path:
        """Create an extension template project directory."""
        if output_dir is None:
            output_dir = Path.cwd()
        project_dir = output_dir / extension_name
        project_dir.mkdir(parents=True, exist_ok=True)

        for filename, content in TEMPLATES.items():
            content = content.replace("{extension_name}", extension_name).replace(
                "{author}", author
            )
            with open(project_dir / filename, "w") as f:
                f.write(content)

        # lib.py
        with open(project_dir / "lib.py", "w") as f:
            f.write(
                f'"""{extension_name} extension entry point."""\n\n'
                "from app.api.extension_api import ExtensionAPI\n\n"
                "_initialized = False\n\n\n"
                "def setup(api: ExtensionAPI) -> None:\n"
                '    """Initialise the extension."""\n'
                "    global _initialized\n"
                "    if _initialized:\n"
                "        return\n"
                "    _initialized = True\n"
                '    print(f"\\u2713 {extension_name} initialized")\n\n\n'
                '__all__ = ["setup"]\n'
            )

        # example_node.py
        with open(project_dir / "example_node.py", "w") as f:
            f.write(
                '"""Example deterministic node."""\n\n'
                "from manim import *\n\n\n"
                "class ExampleNodeClass(VMobject):\n"
                '    """Example custom node — deterministic contract."""\n\n'
                "    def __init__(self, **kwargs):\n"
                "        super().__init__(**kwargs)\n"
                "        self.set_color(BLUE)\n"
                "        self.add(Circle(radius=0.5))\n\n\n"
                "def create(params: dict) -> ExampleNodeClass:\n"
                '    """Factory function."""\n'
                "    return ExampleNodeClass()\n"
            )

        # mcp_hooks.py
        with open(project_dir / "mcp_hooks.py", "w") as f:
            f.write(
                '"""MCP integration hooks."""\n\n\n'
                "async def pre_render_hook(context):\n"
                '    """Called before rendering."""\n'
                '    return {"status": "ready"}\n\n\n'
                "async def post_render_hook(context):\n"
                '    """Called after rendering."""\n'
                '    return {"status": "complete"}\n'
            )

        # requirements.txt
        with open(project_dir / "requirements.txt", "w") as f:
            f.write("# Add dependencies here\n# manim>=0.18.0\n")

        # SIGNING.md
        with open(project_dir / "SIGNING.md", "w") as f:
            f.write(
                f"# Signing {extension_name}\n\n"
                "## Create Signing Keys\n\n"
                "```bash\n"
                "openssl genrsa -out private_key.pem 2048\n"
                "openssl rsa -in private_key.pem -pubout -out public_key.pem\n"
                "```\n\n"
                "Keep your private key secure and never commit it to version control.\n"
            )

        # .gitignore
        with open(project_dir / ".gitignore", "w") as f:
            f.write("venv/\n__pycache__/\n*.pyc\n*.pem\n.pytest_cache/\n.coverage\n")

        print(f"\u2713 Extension template created at {project_dir}")
        return project_dir


def create_extension_cli(name: str, author: str) -> None:
    """CLI command: efficientmanim create-extension MyExtension"""
    try:
        SDKGenerator.create_extension(name, author)
        print(f"\u2713 Extension '{name}' created successfully!")
    except Exception as e:
        print(f"\u2717 Failed to create extension: {e}")
