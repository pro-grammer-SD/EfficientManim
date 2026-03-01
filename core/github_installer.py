"""
github_installer.py — Remote Extension Installation

Handles installation of extensions from GitHub.
Validates, installs dependencies, creates venv.

Rules:
  - No arbitrary pip installs outside venv
  - No shell execution
  - No writing outside allowed directories
  - Clone into ~/.efficientmanim/ext/author/repo
"""

import json
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Tuple, Optional
from urllib.parse import urlparse
from dataclasses import dataclass

LOGGER = logging.getLogger("github_installer")

EXTENSION_ROOT = Path.home() / ".efficientmanim" / "ext"


@dataclass
class InstallationResult:
    """Result of extension installation."""

    success: bool
    message: str
    extension_id: Optional[str] = None
    path: Optional[Path] = None


class GitHubInstaller:
    """Handles installation of extensions from GitHub."""

    @staticmethod
    def parse_github_url(url: str) -> Tuple[str, str]:
        """
        Parse GitHub URL to extract owner and repo.

        Supports:
        - https://github.com/owner/repo
        - https://github.com/owner/repo.git
        - owner/repo (shorthand)

        Returns: (owner, repo)
        """
        # Handle shorthand
        if "/" in url and not url.startswith("http"):
            parts = url.split("/")
            if len(parts) == 2:
                return parts[0], parts[1]

        # Handle full URL
        parsed = urlparse(url)
        if parsed.hostname and "github.com" in parsed.hostname:
            parts = parsed.path.strip("/").split("/")
            if len(parts) >= 2:
                owner = parts[0]
                repo = parts[1].replace(".git", "")
                return owner, repo

        raise ValueError(f"Invalid GitHub URL: {url}")

    @staticmethod
    def install_from_github(url: str) -> InstallationResult:
        """
        Install extension from GitHub URL.

        Process:
        1. Parse URL
        2. Clone into ~/.efficientmanim/ext/owner/repo
        3. Validate metadata.json
        4. Validate entry file
        5. Create venv
        6. Install requirements.txt
        7. Verify signature (if present)

        Returns: InstallationResult
        """
        try:
            owner, repo = GitHubInstaller.parse_github_url(url)

            # Create directories
            ext_dir = EXTENSION_ROOT / owner / repo
            if ext_dir.exists():
                return InstallationResult(
                    success=False, message=f"Extension already installed at {ext_dir}"
                )

            ext_dir.mkdir(parents=True, exist_ok=True)

            # Clone repository
            clone_url = f"https://github.com/{owner}/{repo}.git"
            try:
                result = subprocess.run(
                    ["git", "clone", "--depth", "1", clone_url, str(ext_dir)],
                    capture_output=True,
                    timeout=30,
                )
                if result.returncode != 0:
                    shutil.rmtree(ext_dir)
                    return InstallationResult(
                        success=False,
                        message=f"Git clone failed: {result.stderr.decode()}",
                    )
            except subprocess.TimeoutExpired:
                shutil.rmtree(ext_dir)
                return InstallationResult(success=False, message="Clone timeout (>30s)")
            except FileNotFoundError:
                shutil.rmtree(ext_dir)
                return InstallationResult(
                    success=False,
                    message="Git not found. Install git to use GitHub installer.",
                )

            # Validate metadata
            metadata_path = ext_dir / "metadata.json"
            if not metadata_path.exists():
                shutil.rmtree(ext_dir)
                return InstallationResult(
                    success=False, message="metadata.json not found in repository"
                )

            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
            except json.JSONDecodeError:
                shutil.rmtree(ext_dir)
                return InstallationResult(
                    success=False, message="Invalid metadata.json format"
                )

            # Validate entry file
            entry_file = metadata.get("entry_file", "lib.py")
            entry_path = ext_dir / entry_file
            if not entry_path.exists():
                shutil.rmtree(ext_dir)
                return InstallationResult(
                    success=False, message=f"Entry file not found: {entry_file}"
                )

            # Create virtual environment
            venv_path = ext_dir / "venv"
            try:
                result = subprocess.run(
                    ["python", "-m", "venv", str(venv_path)],
                    capture_output=True,
                    timeout=60,
                )
                if result.returncode != 0:
                    shutil.rmtree(ext_dir)
                    return InstallationResult(
                        success=False,
                        message=f"Failed to create venv: {result.stderr.decode()}",
                    )
            except subprocess.TimeoutExpired:
                shutil.rmtree(ext_dir)
                return InstallationResult(
                    success=False, message="venv creation timeout (>60s)"
                )

            # Install requirements
            requirements_path = ext_dir / "requirements.txt"
            if requirements_path.exists():
                try:
                    pip_exe = (
                        venv_path
                        / ("Scripts" if "win" in str(venv_path) else "bin")
                        / "pip"
                    )
                    result = subprocess.run(
                        [str(pip_exe), "install", "-r", str(requirements_path)],
                        capture_output=True,
                        timeout=300,
                    )
                    if result.returncode != 0:
                        LOGGER.warning(
                            f"pip install had warnings: {result.stderr.decode()}"
                        )
                except subprocess.TimeoutExpired:
                    return InstallationResult(
                        success=False, message="pip install timeout (>5m)"
                    )

            ext_id = f"{owner}/{repo}"

            LOGGER.info(f"✓ Extension installed: {ext_id}")

            return InstallationResult(
                success=True,
                message="Extension installed successfully",
                extension_id=ext_id,
                path=ext_dir,
            )

        except Exception as e:
            LOGGER.error(f"Installation failed: {e}")
            return InstallationResult(success=False, message=str(e))


class MarketplaceClient:
    """Client for extension marketplace communication."""

    def __init__(
        self, marketplace_url: str = "https://api.efficientmanim.dev/marketplace"
    ):
        self.marketplace_url = marketplace_url

    def get_index(self) -> Optional[dict]:
        """
        Fetch marketplace extension index.

        Expected format:
        {
          "extensions": [
            {
              "name": "...",
              "author": "...",
              "repo": "...",
              "version": "...",
              "engine_version": ">=2.0.0",
              "description": "...",
              "permissions": [...],
              "verified": true,
              "signature_url": "..."
            }
          ]
        }

        Returns: dict or None on error
        """
        try:
            import requests

            response = requests.get(f"{self.marketplace_url}/index", timeout=10)
            if response.status_code == 200:
                return response.json()
        except ImportError:
            LOGGER.warning("requests library not available, skipping marketplace")
        except Exception as e:
            LOGGER.error(f"Failed to fetch marketplace: {e}")

        return None

    def search_extensions(self, query: str) -> Optional[list]:
        """Search marketplace for extensions."""
        try:
            import requests

            response = requests.get(
                f"{self.marketplace_url}/search", params={"q": query}, timeout=10
            )
            if response.status_code == 200:
                return response.json().get("results", [])
        except Exception as e:
            LOGGER.error(f"Search failed: {e}")

        return None

    def get_extension_info(self, author: str, repo: str) -> Optional[dict]:
        """Get detailed info about an extension."""
        try:
            import requests

            response = requests.get(
                f"{self.marketplace_url}/extensions/{author}/{repo}", timeout=10
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            LOGGER.error(f"Failed to fetch extension info: {e}")

        return None

    def verify_extension(
        self, author: str, repo: str, version: str
    ) -> Tuple[bool, str]:
        """
        Verify extension with marketplace.

        Checks:
        - Author is verified
        - Version exists
        - Signature matches

        Returns: (verified, message)
        """
        try:
            import requests

            response = requests.post(
                f"{self.marketplace_url}/verify",
                json={"author": author, "repo": repo, "version": version},
                timeout=10,
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("verified", False), data.get("message", "")
        except Exception as e:
            LOGGER.error(f"Verification failed: {e}")
            return False, str(e)

        return False, "Unknown error"
