"""
Path resolution utilities for ToolBridge.

Handles XDG Base Directory Specification compliance and provides
consistent path resolution for state directories, archives, etc.
"""

import os
from functools import lru_cache
from pathlib import Path

import platformdirs


@lru_cache(maxsize=1)
def get_state_home() -> Path:
    """
    Get the ToolBridge state home directory.

    Resolution order:
    1. TOOLBRIDGE_STATE_HOME environment variable (if set)
    2. platformdirs user_state_dir (follows XDG spec on Linux)
       - Linux: ${XDG_STATE_HOME:-~/.local/state}/toolbridge
       - macOS: ~/Library/Application Support/toolbridge
       - Windows: %LOCALAPPDATA%/toolbridge

    Returns:
        Path to the state home directory (may not exist yet)
    """
    env_state_home = os.environ.get("TOOLBRIDGE_STATE_HOME")
    if env_state_home:
        return Path(env_state_home)

    return Path(platformdirs.user_state_dir("toolbridge"))


def get_archive_dir(override: str | None = None) -> Path:
    """
    Get the archive directory path.

    Resolution order:
    1. override parameter (from CLI --archive-dir)
    2. TOOLBRIDGE_ARCHIVE_DIR environment variable
    3. ${TOOLBRIDGE_STATE_HOME}/archives (default)

    Args:
        override: Optional explicit path from CLI argument

    Returns:
        Path to the archive directory (may not exist yet)
    """
    if override:
        return Path(override)

    env_archive_dir = os.environ.get("TOOLBRIDGE_ARCHIVE_DIR")
    if env_archive_dir:
        return Path(env_archive_dir)

    return get_state_home() / "archives"


def ensure_dir(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure exists

    Returns:
        The same path (for chaining)

    Raises:
        OSError: If directory cannot be created
    """
    path.mkdir(parents=True, exist_ok=True)
    return path
