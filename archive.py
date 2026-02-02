"""
Session archive management for ToolBridge.

Provides persistent storage for expired sessions with:
- Directory-per-day organization for efficient TTL cleanup
- Index file with file locking for fast lookups
- Async-compatible operations
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from filelock import FileLock, Timeout

from paths import ensure_dir, get_archive_dir

if TYPE_CHECKING:
    from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class ArchivedSessionSummary:
    """Summary information about an archived session."""

    session_id: str
    date: str  # YYYY-MM-DD
    archived_at: float
    created_at: float
    last_seen_at: float
    request_count: int
    tool_calls_total: int
    tool_calls_fixed: int
    tool_calls_failed: int
    client_ip: str | None
    prompt_tokens_total: int
    completion_tokens_total: int
    total_tokens_total: int
    # Agent detection
    detected_agent: str | None = None
    detected_agent_confidence: str | None = None


class ArchiveIndex:
    """
    Manages the archive index file with proper file locking.

    The index provides O(1) lookup from session_id to date directory,
    avoiding the need to search all directories.

    Index file format (index.json):
    {
        "version": 1,
        "sessions": {
            "session_id": {
                "date": "2026-02-02",
                "archived_at": 1738512000.0
            },
            ...
        }
    }
    """

    INDEX_VERSION = 1

    def __init__(self, archive_dir: Path):
        self.archive_dir = archive_dir
        self.index_path = archive_dir / "index.json"
        self.lock_path = archive_dir / "index.lock"
        self._lock = FileLock(str(self.lock_path), timeout=10)
        # In-process async lock to coordinate async operations
        self._async_lock = asyncio.Lock()

    def _read_index_sync(self) -> dict[str, Any]:
        """Read index file synchronously (call within file lock)."""
        if not self.index_path.exists():
            return {"version": self.INDEX_VERSION, "sessions": {}}

        try:
            with open(self.index_path, encoding="utf-8") as f:
                data = json.load(f)
                # Migrate if needed
                if data.get("version", 0) < self.INDEX_VERSION:
                    data["version"] = self.INDEX_VERSION
                return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to read index, starting fresh: {e}")
            return {"version": self.INDEX_VERSION, "sessions": {}}

    def _write_index_sync(self, data: dict[str, Any]) -> None:
        """Write index file synchronously (call within file lock)."""
        ensure_dir(self.archive_dir)
        # Write to temp file then rename for atomicity
        temp_path = self.index_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        temp_path.rename(self.index_path)

    async def add_session(
        self, session_id: str, date: str, archived_at: float
    ) -> None:
        """
        Add a session to the index.

        Args:
            session_id: The session ID
            date: Date string (YYYY-MM-DD)
            archived_at: Unix timestamp when archived
        """
        async with self._async_lock:
            # Run blocking file lock operation in thread pool
            await asyncio.to_thread(
                self._add_session_sync, session_id, date, archived_at
            )

    def _add_session_sync(
        self, session_id: str, date: str, archived_at: float
    ) -> None:
        """Synchronous version for thread pool execution."""
        try:
            with self._lock:
                data = self._read_index_sync()
                data["sessions"][session_id] = {
                    "date": date,
                    "archived_at": archived_at,
                }
                self._write_index_sync(data)
        except Timeout:
            logger.error("Timeout acquiring index lock for add_session")
            raise

    async def remove_session(self, session_id: str) -> bool:
        """
        Remove a session from the index.

        Returns:
            True if session was found and removed, False otherwise
        """
        async with self._async_lock:
            return await asyncio.to_thread(
                self._remove_session_sync, session_id
            )

    def _remove_session_sync(self, session_id: str) -> bool:
        """Synchronous version for thread pool execution."""
        try:
            with self._lock:
                data = self._read_index_sync()
                if session_id in data["sessions"]:
                    del data["sessions"][session_id]
                    self._write_index_sync(data)
                    return True
                return False
        except Timeout:
            logger.error("Timeout acquiring index lock for remove_session")
            raise

    async def get_session_location(self, session_id: str) -> str | None:
        """
        Get the date directory for a session.

        Returns:
            Date string (YYYY-MM-DD) if found, None otherwise
        """
        async with self._async_lock:
            return await asyncio.to_thread(
                self._get_session_location_sync, session_id
            )

    def _get_session_location_sync(self, session_id: str) -> str | None:
        """Synchronous version for thread pool execution."""
        try:
            with self._lock:
                data = self._read_index_sync()
                session_info = data["sessions"].get(session_id)
                if session_info:
                    return session_info["date"]
                return None
        except Timeout:
            logger.error("Timeout acquiring index lock for get_session_location")
            raise

    async def list_sessions(self) -> dict[str, dict[str, Any]]:
        """
        List all sessions in the index.

        Returns:
            Dict mapping session_id to {date, archived_at}
        """
        async with self._async_lock:
            return await asyncio.to_thread(self._list_sessions_sync)

    def _list_sessions_sync(self) -> dict[str, dict[str, Any]]:
        """Synchronous version for thread pool execution."""
        try:
            with self._lock:
                data = self._read_index_sync()
                return dict(data["sessions"])
        except Timeout:
            logger.error("Timeout acquiring index lock for list_sessions")
            raise

    async def remove_sessions_by_date(self, dates: list[str]) -> int:
        """
        Remove all sessions with the given dates from the index.

        Args:
            dates: List of date strings (YYYY-MM-DD) to remove

        Returns:
            Number of sessions removed
        """
        async with self._async_lock:
            return await asyncio.to_thread(
                self._remove_sessions_by_date_sync, dates
            )

    def _remove_sessions_by_date_sync(self, dates: list[str]) -> int:
        """Synchronous version for thread pool execution."""
        date_set = set(dates)
        try:
            with self._lock:
                data = self._read_index_sync()
                sessions = data["sessions"]
                to_remove = [
                    sid for sid, info in sessions.items()
                    if info.get("date") in date_set
                ]
                for sid in to_remove:
                    del sessions[sid]
                if to_remove:
                    self._write_index_sync(data)
                return len(to_remove)
        except Timeout:
            logger.error("Timeout acquiring index lock for remove_sessions_by_date")
            raise

    async def rebuild_from_disk(self) -> int:
        """
        Rebuild the index by scanning archive directories.

        Useful if index gets corrupted or out of sync.

        Returns:
            Number of sessions indexed
        """
        async with self._async_lock:
            return await asyncio.to_thread(self._rebuild_from_disk_sync)

    def _rebuild_from_disk_sync(self) -> int:
        """Synchronous version for thread pool execution."""
        sessions: dict[str, dict[str, Any]] = {}

        if not self.archive_dir.exists():
            return 0

        for date_dir in self.archive_dir.iterdir():
            if not date_dir.is_dir():
                continue

            # Skip if not a valid date directory name
            date_str = date_dir.name
            try:
                datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                continue

            # Scan JSON files in this directory
            for session_file in date_dir.glob("*.json"):
                session_id = session_file.stem
                try:
                    with open(session_file, encoding="utf-8") as f:
                        data = json.load(f)
                        archived_at = data.get("archived_at", session_file.stat().st_mtime)
                except (json.JSONDecodeError, OSError):
                    archived_at = session_file.stat().st_mtime

                sessions[session_id] = {
                    "date": date_str,
                    "archived_at": archived_at,
                }

        # Write the rebuilt index
        try:
            with self._lock:
                self._write_index_sync({
                    "version": self.INDEX_VERSION,
                    "sessions": sessions,
                })
        except Timeout:
            logger.error("Timeout acquiring index lock for rebuild")
            raise

        logger.info(f"Rebuilt archive index: {len(sessions)} sessions")
        return len(sessions)

    async def validate(self) -> tuple[bool, int]:
        """
        Validate the index and rebuild if necessary.

        Checks if:
        1. Index file exists
        2. Index file is valid JSON
        3. Index is not empty when archive directories exist

        Returns:
            Tuple of (was_rebuilt, session_count)
        """
        async with self._async_lock:
            return await asyncio.to_thread(self._validate_sync)

    def _validate_sync(self) -> tuple[bool, int]:
        """Synchronous version for thread pool execution."""
        needs_rebuild = False

        # Check if index file exists
        if not self.index_path.exists():
            # Check if there are any archive directories with sessions
            if self.archive_dir.exists():
                for date_dir in self.archive_dir.iterdir():
                    if date_dir.is_dir() and list(date_dir.glob("*.json")):
                        needs_rebuild = True
                        logger.warning("Index file missing but archive directories exist, rebuilding")
                        break
        else:
            # Try to read the index
            try:
                with self._lock:
                    with open(self.index_path, encoding="utf-8") as f:
                        data = json.load(f)
                        if "sessions" not in data:
                            needs_rebuild = True
                            logger.warning("Index file corrupted (missing sessions key), rebuilding")
            except (json.JSONDecodeError, OSError) as e:
                needs_rebuild = True
                logger.warning(f"Index file corrupted or unreadable: {e}, rebuilding")

        if needs_rebuild:
            count = self._rebuild_from_disk_sync()
            return (True, count)

        # Return current session count
        try:
            with self._lock:
                data = self._read_index_sync()
                return (False, len(data.get("sessions", {})))
        except Timeout:
            return (False, 0)


class SessionArchive:
    """
    Manages session archival with directory-per-day organization.

    Archive structure:
    archives/
    ├── 2026-02-02/
    │   ├── session_id_1.json
    │   └── session_id_2.json
    ├── 2026-02-03/
    │   └── session_id_3.json
    └── index.json
    """

    def __init__(
        self,
        archive_dir: Path | None = None,
        archive_ttl_hours: int = 168,  # 7 days default
    ):
        """
        Initialize the session archive.

        Args:
            archive_dir: Directory for archives (None uses default from paths module)
            archive_ttl_hours: Hours to retain archives (0 = forever)
        """
        self.archive_dir = archive_dir or get_archive_dir()
        self.archive_ttl_hours = archive_ttl_hours
        self.index = ArchiveIndex(self.archive_dir)

    async def validate_on_startup(self) -> dict[str, Any]:
        """
        Validate the archive on startup.

        Should be called when the application starts to ensure the index
        is consistent with files on disk. This will:
        1. Check if the index file exists and is valid
        2. Rebuild the index from disk if corrupted or missing
        3. Run TTL cleanup to remove expired archives

        Returns:
            Dict with validation results:
            - index_rebuilt: bool - Whether index was rebuilt
            - session_count: int - Number of archived sessions
            - expired_removed: int - Number of expired sessions cleaned up
        """
        # Validate and potentially rebuild index
        was_rebuilt, session_count = await self.index.validate()

        # Run TTL cleanup
        expired_removed = await self.cleanup_expired()

        # Update session count after cleanup
        if expired_removed > 0:
            session_count = await self.get_session_count()

        result = {
            "index_rebuilt": was_rebuilt,
            "session_count": session_count,
            "expired_removed": expired_removed,
        }

        if was_rebuilt:
            logger.info(f"Archive startup validation: index rebuilt with {session_count} sessions")
        if expired_removed > 0:
            logger.info(f"Archive startup validation: removed {expired_removed} expired sessions")

        return result

    async def archive_session(
        self,
        session_id: str,
        session_data: dict[str, Any],
    ) -> bool:
        """
        Archive a session to disk.

        Args:
            session_id: The session ID
            session_data: Full session data including messages

        Returns:
            True if successful, False otherwise
        """
        try:
            now = time.time()
            date_str = datetime.now().strftime("%Y-%m-%d")

            # Add archive metadata
            archive_data = {
                **session_data,
                "session_id": session_id,
                "archived_at": now,
            }

            # Ensure directory exists and write file
            date_dir = ensure_dir(self.archive_dir / date_str)
            session_file = date_dir / f"{session_id}.json"

            # Write to temp file then rename for atomicity
            await asyncio.to_thread(
                self._write_session_file, session_file, archive_data
            )

            # Update index
            await self.index.add_session(session_id, date_str, now)

            logger.info(f"Archived session {session_id} to {date_str}/")
            return True

        except Exception as e:
            logger.error(f"Failed to archive session {session_id}: {e}")
            return False

    def _write_session_file(
        self, session_file: Path, archive_data: dict[str, Any]
    ) -> None:
        """Write session file atomically."""
        temp_path = session_file.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(archive_data, f, indent=2)
        temp_path.rename(session_file)

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        """
        Retrieve an archived session.

        Args:
            session_id: The session ID to retrieve

        Returns:
            Session data dict if found, None otherwise
        """
        # First check index for location
        date_str = await self.index.get_session_location(session_id)
        if not date_str:
            return None

        session_file = self.archive_dir / date_str / f"{session_id}.json"

        if not session_file.exists():
            # Index out of sync, remove stale entry
            await self.index.remove_session(session_id)
            return None

        try:
            return await asyncio.to_thread(self._read_session_file, session_file)
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to read archived session {session_id}: {e}")
            return None

    def _read_session_file(self, session_file: Path) -> dict[str, Any]:
        """Read session file."""
        with open(session_file, encoding="utf-8") as f:
            return json.load(f)

    async def list_sessions(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ArchivedSessionSummary]:
        """
        List archived sessions.

        This method performs lazy cleanup: if a session is in the index but
        its file has been manually deleted, the stale index entry is removed
        and the session is excluded from results.

        Args:
            limit: Maximum number of sessions to return
            offset: Number of sessions to skip

        Returns:
            List of ArchivedSessionSummary objects (excludes stale entries)
        """
        index_sessions = await self.index.list_sessions()

        # Sort by archived_at descending (most recent first)
        sorted_sessions = sorted(
            index_sessions.items(),
            key=lambda x: x[1].get("archived_at", 0),
            reverse=True,
        )

        # Apply pagination
        paginated = sorted_sessions[offset:offset + limit]

        # Build summaries (requires reading each file for full data)
        # get_session() handles lazy cleanup: removes stale index entries
        # if the session file was manually deleted
        summaries: list[ArchivedSessionSummary] = []
        for session_id, info in paginated:
            session_data = await self.get_session(session_id)
            if session_data:
                summaries.append(ArchivedSessionSummary(
                    session_id=session_id,
                    date=info["date"],
                    archived_at=session_data.get("archived_at", info.get("archived_at", 0)),
                    created_at=session_data.get("created_at", 0),
                    last_seen_at=session_data.get("last_seen_at", 0),
                    request_count=session_data.get("request_count", 0),
                    tool_calls_total=session_data.get("tool_calls_total", 0),
                    tool_calls_fixed=session_data.get("tool_calls_fixed", 0),
                    tool_calls_failed=session_data.get("tool_calls_failed", 0),
                    client_ip=session_data.get("client_ip"),
                    prompt_tokens_total=session_data.get("prompt_tokens_total", 0),
                    completion_tokens_total=session_data.get("completion_tokens_total", 0),
                    total_tokens_total=session_data.get("total_tokens_total", 0),
                    detected_agent=session_data.get("detected_agent"),
                    detected_agent_confidence=session_data.get("detected_agent_confidence"),
                ))

        return summaries

    async def get_session_count(self) -> int:
        """Get total number of archived sessions."""
        sessions = await self.index.list_sessions()
        return len(sessions)

    async def cleanup_expired(self) -> int:
        """
        Remove archives older than archive_ttl_hours.

        Returns:
            Number of sessions removed
        """
        if self.archive_ttl_hours <= 0:
            return 0  # TTL disabled

        cutoff = datetime.now() - timedelta(hours=self.archive_ttl_hours)
        cutoff_str = cutoff.strftime("%Y-%m-%d")

        removed_count = 0
        dates_to_remove: list[str] = []

        if not self.archive_dir.exists():
            return 0

        # Find directories older than cutoff
        for date_dir in self.archive_dir.iterdir():
            if not date_dir.is_dir():
                continue

            date_str = date_dir.name
            try:
                datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                continue  # Not a date directory

            if date_str < cutoff_str:
                dates_to_remove.append(date_str)

        # Remove directories and update index
        for date_str in dates_to_remove:
            date_dir = self.archive_dir / date_str
            try:
                # Count files before removal
                file_count = len(list(date_dir.glob("*.json")))
                removed_count += file_count

                # Remove directory
                await asyncio.to_thread(shutil.rmtree, date_dir)
                logger.info(f"Removed expired archive directory: {date_str}/ ({file_count} sessions)")
            except OSError as e:
                logger.error(f"Failed to remove archive directory {date_str}: {e}")

        # Update index
        if dates_to_remove:
            await self.index.remove_sessions_by_date(dates_to_remove)

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} expired archived sessions")

        return removed_count

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete an archived session.

        Args:
            session_id: The session ID to delete

        Returns:
            True if session was found and deleted, False otherwise
        """
        date_str = await self.index.get_session_location(session_id)
        if not date_str:
            return False

        session_file = self.archive_dir / date_str / f"{session_id}.json"

        try:
            if session_file.exists():
                await asyncio.to_thread(session_file.unlink)
        except OSError as e:
            logger.error(f"Failed to delete session file {session_id}: {e}")
            return False

        await self.index.remove_session(session_id)
        logger.info(f"Deleted archived session {session_id}")
        return True


def session_stats_to_archive_dict(
    session_id: str,
    stats: Any,  # SessionStats type from toolbridge
) -> dict[str, Any]:
    """
    Convert a SessionStats object to a dict suitable for archiving.

    Args:
        session_id: The session ID
        stats: SessionStats object

    Returns:
        Dict with all session data serialized
    """
    # Serialize messages from deque
    messages: list[dict[str, Any]] = []
    for msg in stats.messages:
        messages.append({
            "timestamp": msg.timestamp,
            "direction": msg.direction,
            "role": msg.role,
            "content": msg.content,
            "tool_calls": msg.tool_calls,
            "raw_content": msg.raw_content,
            "debug": msg.debug,
            "prompt_tokens": msg.prompt_tokens,
        })

    return {
        "session_id": session_id,
        "created_at": stats.created_at,
        "last_seen_at": stats.last_seen_at,
        "request_count": stats.request_count,
        "tool_calls_total": stats.tool_calls_total,
        "tool_calls_fixed": stats.tool_calls_fixed,
        "tool_calls_failed": stats.tool_calls_failed,
        "client_ip": stats.client_ip,
        "last_prompt_tokens": stats.last_prompt_tokens,
        "last_completion_tokens": stats.last_completion_tokens,
        "last_total_tokens": stats.last_total_tokens,
        "prompt_tokens_total": stats.prompt_tokens_total,
        "completion_tokens_total": stats.completion_tokens_total,
        "total_tokens_total": stats.total_tokens_total,
        "detected_agent": stats.detected_agent,
        "detected_agent_confidence": stats.detected_agent_confidence,
        "secondary_fingerprint": stats.secondary_fingerprint,
        "messages": messages,
    }
