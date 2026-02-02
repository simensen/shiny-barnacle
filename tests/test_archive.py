"""Tests for session archive functionality."""

import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from archive import (
    ArchiveIndex,
    SessionArchive,
    session_stats_to_archive_dict,
)
from paths import ensure_dir, get_archive_dir, get_state_home

if TYPE_CHECKING:
    from toolbridge import SessionTracker


class TestPaths:
    """Tests for paths module."""

    def test_get_state_home_default(self) -> None:
        """Test get_state_home returns XDG-compliant path when env not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove any existing env vars
            os.environ.pop("TOOLBRIDGE_STATE_HOME", None)
            # Clear the lru_cache
            get_state_home.cache_clear()

            path = get_state_home()
            # Should contain 'toolbridge' in the path
            assert "toolbridge" in str(path)

    def test_get_state_home_from_env(self) -> None:
        """Test get_state_home uses TOOLBRIDGE_STATE_HOME when set."""
        get_state_home.cache_clear()
        with patch.dict(os.environ, {"TOOLBRIDGE_STATE_HOME": "/custom/state"}):
            path = get_state_home()
            assert path == Path("/custom/state")
        get_state_home.cache_clear()

    def test_get_archive_dir_default(self) -> None:
        """Test get_archive_dir returns state_home/archives by default."""
        get_state_home.cache_clear()
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("TOOLBRIDGE_ARCHIVE_DIR", None)
            os.environ.pop("TOOLBRIDGE_STATE_HOME", None)

            path = get_archive_dir()
            assert path.name == "archives"
        get_state_home.cache_clear()

    def test_get_archive_dir_override(self) -> None:
        """Test get_archive_dir uses override parameter."""
        path = get_archive_dir("/custom/archive")
        assert path == Path("/custom/archive")

    def test_get_archive_dir_from_env(self) -> None:
        """Test get_archive_dir uses TOOLBRIDGE_ARCHIVE_DIR when set."""
        with patch.dict(os.environ, {"TOOLBRIDGE_ARCHIVE_DIR": "/env/archive"}):
            path = get_archive_dir()
            assert path == Path("/env/archive")

    def test_ensure_dir_creates_directory(self) -> None:
        """Test ensure_dir creates directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "new" / "nested" / "dir"
            assert not new_dir.exists()

            result = ensure_dir(new_dir)

            assert new_dir.exists()
            assert new_dir.is_dir()
            assert result == new_dir


class TestArchiveIndex:
    """Tests for ArchiveIndex with file locking."""

    @pytest.fixture
    def archive_dir(self) -> Path:
        """Create a temporary archive directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            yield path

    @pytest.fixture
    def index(self, archive_dir: Path) -> ArchiveIndex:
        """Create an ArchiveIndex instance."""
        ensure_dir(archive_dir)
        return ArchiveIndex(archive_dir)

    @pytest.mark.asyncio
    async def test_add_and_get_session(self, index: ArchiveIndex) -> None:
        """Test adding and retrieving a session from the index."""
        await index.add_session("session123", "2026-02-02", 1738512000.0)

        location = await index.get_session_location("session123")

        assert location == "2026-02-02"

    @pytest.mark.asyncio
    async def test_get_nonexistent_session(self, index: ArchiveIndex) -> None:
        """Test getting a session that doesn't exist."""
        location = await index.get_session_location("nonexistent")

        assert location is None

    @pytest.mark.asyncio
    async def test_remove_session(self, index: ArchiveIndex) -> None:
        """Test removing a session from the index."""
        await index.add_session("session123", "2026-02-02", 1738512000.0)

        result = await index.remove_session("session123")

        assert result is True
        assert await index.get_session_location("session123") is None

    @pytest.mark.asyncio
    async def test_remove_nonexistent_session(self, index: ArchiveIndex) -> None:
        """Test removing a session that doesn't exist."""
        result = await index.remove_session("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_list_sessions(self, index: ArchiveIndex) -> None:
        """Test listing all sessions in the index."""
        await index.add_session("session1", "2026-02-02", 1738512000.0)
        await index.add_session("session2", "2026-02-03", 1738598400.0)

        sessions = await index.list_sessions()

        assert len(sessions) == 2
        assert "session1" in sessions
        assert "session2" in sessions
        assert sessions["session1"]["date"] == "2026-02-02"
        assert sessions["session2"]["date"] == "2026-02-03"

    @pytest.mark.asyncio
    async def test_remove_sessions_by_date(self, index: ArchiveIndex) -> None:
        """Test removing sessions by date."""
        await index.add_session("session1", "2026-02-02", 1738512000.0)
        await index.add_session("session2", "2026-02-02", 1738512100.0)
        await index.add_session("session3", "2026-02-03", 1738598400.0)

        removed = await index.remove_sessions_by_date(["2026-02-02"])

        assert removed == 2
        sessions = await index.list_sessions()
        assert len(sessions) == 1
        assert "session3" in sessions


class TestSessionArchive:
    """Tests for SessionArchive."""

    @pytest.fixture
    def archive_dir(self) -> Path:
        """Create a temporary archive directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def archive(self, archive_dir: Path) -> SessionArchive:
        """Create a SessionArchive instance."""
        return SessionArchive(archive_dir=archive_dir, archive_ttl_hours=168)

    @pytest.fixture
    def sample_session_data(self) -> dict:
        """Create sample session data."""
        return {
            "session_id": "test_session",
            "created_at": time.time() - 3600,
            "last_seen_at": time.time(),
            "request_count": 5,
            "tool_calls_total": 10,
            "tool_calls_fixed": 8,
            "tool_calls_failed": 2,
            "client_ip": "192.168.1.100",
            "last_prompt_tokens": 1000,
            "last_completion_tokens": 500,
            "last_total_tokens": 1500,
            "prompt_tokens_total": 5000,
            "completion_tokens_total": 2500,
            "total_tokens_total": 7500,
            "messages": [
                {
                    "timestamp": time.time(),
                    "direction": "request",
                    "role": "user",
                    "content": "Hello",
                    "tool_calls": None,
                    "raw_content": None,
                    "debug": None,
                    "prompt_tokens": 100,
                }
            ],
        }

    @pytest.mark.asyncio
    async def test_archive_session(
        self, archive: SessionArchive, sample_session_data: dict
    ) -> None:
        """Test archiving a session."""
        result = await archive.archive_session("test_session", sample_session_data)

        assert result is True
        # Check file was created
        location = await archive.index.get_session_location("test_session")
        assert location is not None

    @pytest.mark.asyncio
    async def test_get_session(
        self, archive: SessionArchive, sample_session_data: dict
    ) -> None:
        """Test retrieving an archived session."""
        await archive.archive_session("test_session", sample_session_data)

        retrieved = await archive.get_session("test_session")

        assert retrieved is not None
        assert retrieved["session_id"] == "test_session"
        assert retrieved["request_count"] == 5
        assert "archived_at" in retrieved

    @pytest.mark.asyncio
    async def test_get_nonexistent_session(self, archive: SessionArchive) -> None:
        """Test retrieving a session that doesn't exist."""
        retrieved = await archive.get_session("nonexistent")

        assert retrieved is None

    @pytest.mark.asyncio
    async def test_list_sessions(
        self, archive: SessionArchive, sample_session_data: dict
    ) -> None:
        """Test listing archived sessions."""
        await archive.archive_session("session1", sample_session_data)
        sample_session_data["session_id"] = "session2"
        await archive.archive_session("session2", sample_session_data)

        summaries = await archive.list_sessions()

        assert len(summaries) == 2
        session_ids = [s.session_id for s in summaries]
        assert "session1" in session_ids
        assert "session2" in session_ids

    @pytest.mark.asyncio
    async def test_get_session_count(
        self, archive: SessionArchive, sample_session_data: dict
    ) -> None:
        """Test getting the session count."""
        assert await archive.get_session_count() == 0

        await archive.archive_session("session1", sample_session_data)
        assert await archive.get_session_count() == 1

        sample_session_data["session_id"] = "session2"
        await archive.archive_session("session2", sample_session_data)
        assert await archive.get_session_count() == 2

    @pytest.mark.asyncio
    async def test_delete_session(
        self, archive: SessionArchive, sample_session_data: dict
    ) -> None:
        """Test deleting an archived session."""
        await archive.archive_session("test_session", sample_session_data)

        result = await archive.delete_session("test_session")

        assert result is True
        assert await archive.get_session("test_session") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_session(self, archive: SessionArchive) -> None:
        """Test deleting a session that doesn't exist."""
        result = await archive.delete_session("nonexistent")

        assert result is False


class TestSessionStatsToArchiveDict:
    """Tests for session_stats_to_archive_dict helper."""

    def test_converts_session_stats(self) -> None:
        """Test converting SessionStats to archive dict."""
        from collections import deque
        from dataclasses import dataclass

        # Create a mock SessionStats-like object
        @dataclass
        class MockMessage:
            timestamp: float
            direction: str
            role: str
            content: str
            tool_calls: list | None
            raw_content: str | None
            debug: dict | None
            prompt_tokens: int | None

        @dataclass
        class MockSessionStats:
            created_at: float
            last_seen_at: float
            request_count: int
            tool_calls_total: int
            tool_calls_fixed: int
            tool_calls_failed: int
            client_ip: str | None
            last_prompt_tokens: int
            last_completion_tokens: int
            last_total_tokens: int
            prompt_tokens_total: int
            completion_tokens_total: int
            total_tokens_total: int
            messages: deque
            detected_agent: str | None
            detected_agent_confidence: str | None
            secondary_fingerprint: str | None

        stats = MockSessionStats(
            created_at=1000.0,
            last_seen_at=2000.0,
            request_count=5,
            tool_calls_total=10,
            tool_calls_fixed=8,
            tool_calls_failed=2,
            client_ip="127.0.0.1",
            last_prompt_tokens=100,
            last_completion_tokens=50,
            last_total_tokens=150,
            prompt_tokens_total=500,
            completion_tokens_total=250,
            total_tokens_total=750,
            messages=deque([
                MockMessage(
                    timestamp=1500.0,
                    direction="request",
                    role="user",
                    content="Hello",
                    tool_calls=None,
                    raw_content=None,
                    debug=None,
                    prompt_tokens=50,
                )
            ]),
            detected_agent="Cline",
            detected_agent_confidence="high",
            secondary_fingerprint="abc123def456",
        )

        result = session_stats_to_archive_dict("session123", stats)

        assert result["session_id"] == "session123"
        assert result["created_at"] == 1000.0
        assert result["last_seen_at"] == 2000.0
        assert result["request_count"] == 5
        assert result["tool_calls_total"] == 10
        assert result["client_ip"] == "127.0.0.1"
        assert len(result["messages"]) == 1
        assert result["messages"][0]["content"] == "Hello"
        assert result["detected_agent"] == "Cline"
        assert result["detected_agent_confidence"] == "high"


class TestArchiveIndexValidation:
    """Tests for ArchiveIndex validation and rebuild."""

    @pytest.fixture
    def archive_dir(self) -> Path:
        """Create a temporary archive directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def index(self, archive_dir: Path) -> ArchiveIndex:
        """Create an ArchiveIndex instance."""
        ensure_dir(archive_dir)
        return ArchiveIndex(archive_dir)

    @pytest.mark.asyncio
    async def test_validate_empty_archive(self, index: ArchiveIndex) -> None:
        """Test validation on empty archive (no index, no files)."""
        was_rebuilt, count = await index.validate()

        # No rebuild needed for empty archive
        assert was_rebuilt is False
        assert count == 0

    @pytest.mark.asyncio
    async def test_validate_corrupted_index(
        self, archive_dir: Path, index: ArchiveIndex
    ) -> None:
        """Test validation rebuilds corrupted index file."""
        # Create a corrupted index file
        ensure_dir(archive_dir)
        with open(archive_dir / "index.json", "w") as f:
            f.write("not valid json {{{")

        was_rebuilt, count = await index.validate()

        assert was_rebuilt is True
        assert count == 0  # No session files to index

    @pytest.mark.asyncio
    async def test_validate_missing_index_with_sessions(
        self, archive_dir: Path, index: ArchiveIndex
    ) -> None:
        """Test validation rebuilds index when files exist but index is missing."""
        # Create a date directory with a session file but no index
        date_dir = archive_dir / "2026-02-02"
        ensure_dir(date_dir)
        with open(date_dir / "session123.json", "w") as f:
            f.write('{"session_id": "session123", "archived_at": 1738512000.0}')

        was_rebuilt, count = await index.validate()

        assert was_rebuilt is True
        assert count == 1

        # Verify the session is now in the index
        location = await index.get_session_location("session123")
        assert location == "2026-02-02"

    @pytest.mark.asyncio
    async def test_validate_valid_index(
        self, archive_dir: Path, index: ArchiveIndex
    ) -> None:
        """Test validation passes for valid index."""
        # Add a session normally
        await index.add_session("session123", "2026-02-02", 1738512000.0)

        was_rebuilt, count = await index.validate()

        assert was_rebuilt is False
        assert count == 1


class TestSessionArchiveValidation:
    """Tests for SessionArchive startup validation."""

    @pytest.fixture
    def archive_dir(self) -> Path:
        """Create a temporary archive directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def archive(self, archive_dir: Path) -> SessionArchive:
        """Create a SessionArchive instance."""
        return SessionArchive(archive_dir=archive_dir, archive_ttl_hours=168)

    @pytest.fixture
    def sample_session_data(self) -> dict:
        """Create sample session data."""
        return {
            "session_id": "test_session",
            "created_at": time.time() - 3600,
            "last_seen_at": time.time(),
            "request_count": 5,
            "tool_calls_total": 10,
            "tool_calls_fixed": 8,
            "tool_calls_failed": 2,
            "client_ip": "192.168.1.100",
            "last_prompt_tokens": 1000,
            "last_completion_tokens": 500,
            "last_total_tokens": 1500,
            "prompt_tokens_total": 5000,
            "completion_tokens_total": 2500,
            "total_tokens_total": 7500,
            "messages": [],
        }

    @pytest.mark.asyncio
    async def test_validate_on_startup_empty(self, archive: SessionArchive) -> None:
        """Test startup validation on empty archive."""
        result = await archive.validate_on_startup()

        assert result["index_rebuilt"] is False
        assert result["session_count"] == 0
        assert result["expired_removed"] == 0

    @pytest.mark.asyncio
    async def test_validate_on_startup_with_sessions(
        self, archive: SessionArchive, sample_session_data: dict
    ) -> None:
        """Test startup validation with existing sessions."""
        await archive.archive_session("session1", sample_session_data)

        result = await archive.validate_on_startup()

        assert result["index_rebuilt"] is False
        assert result["session_count"] == 1
        assert result["expired_removed"] == 0


class TestLazyCleanup:
    """Tests for lazy cleanup during list_sessions."""

    @pytest.fixture
    def archive_dir(self) -> Path:
        """Create a temporary archive directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def archive(self, archive_dir: Path) -> SessionArchive:
        """Create a SessionArchive instance."""
        return SessionArchive(archive_dir=archive_dir, archive_ttl_hours=168)

    @pytest.fixture
    def sample_session_data(self) -> dict:
        """Create sample session data."""
        return {
            "session_id": "test_session",
            "created_at": time.time() - 3600,
            "last_seen_at": time.time(),
            "request_count": 5,
            "tool_calls_total": 10,
            "tool_calls_fixed": 8,
            "tool_calls_failed": 2,
            "client_ip": "192.168.1.100",
            "prompt_tokens_total": 5000,
            "completion_tokens_total": 2500,
            "total_tokens_total": 7500,
            "messages": [],
        }

    @pytest.mark.asyncio
    async def test_list_sessions_removes_stale_entries(
        self, archive: SessionArchive, archive_dir: Path, sample_session_data: dict
    ) -> None:
        """Test that list_sessions removes stale index entries for deleted files."""
        # Archive two sessions
        await archive.archive_session("session1", sample_session_data)
        sample_session_data["session_id"] = "session2"
        await archive.archive_session("session2", sample_session_data)

        # Verify both are in the index
        sessions = await archive.index.list_sessions()
        assert len(sessions) == 2

        # Manually delete session1's file (simulating user deletion)
        date_str = sessions["session1"]["date"]
        session_file = archive_dir / date_str / "session1.json"
        session_file.unlink()

        # List sessions - should trigger lazy cleanup
        summaries = await archive.list_sessions()

        # Only session2 should be returned
        assert len(summaries) == 1
        assert summaries[0].session_id == "session2"

        # Verify session1 was removed from the index
        location = await archive.index.get_session_location("session1")
        assert location is None

    @pytest.mark.asyncio
    async def test_get_session_removes_stale_entry(
        self, archive: SessionArchive, archive_dir: Path, sample_session_data: dict
    ) -> None:
        """Test that get_session removes stale index entry for deleted file."""
        await archive.archive_session("session1", sample_session_data)

        # Verify session is in the index
        location = await archive.index.get_session_location("session1")
        assert location is not None

        # Manually delete the session file
        session_file = archive_dir / location / "session1.json"
        session_file.unlink()

        # Get session - should return None and remove stale entry
        result = await archive.get_session("session1")

        assert result is None
        # Verify it was removed from the index
        location = await archive.index.get_session_location("session1")
        assert location is None


class TestDebouncedArchiving:
    """Tests for debounced archive-on-change functionality."""

    @pytest.fixture
    def archive_dir(self) -> Path:
        """Create a temporary archive directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def archive(self, archive_dir: Path) -> SessionArchive:
        """Create a SessionArchive instance."""
        return SessionArchive(archive_dir=archive_dir, archive_ttl_hours=168)

    @pytest.fixture
    def tracker(self, archive: SessionArchive) -> "SessionTracker":
        """Create a SessionTracker with archive-on-change enabled."""
        from toolbridge import SessionTracker

        return SessionTracker(
            session_timeout=3600,
            message_buffer_size=100,
            archive=archive,
            archive_on_change=True,
            archive_debounce_seconds=0.5,  # Short debounce for testing
        )

    @pytest.mark.asyncio
    async def test_mark_dirty_on_track_request(self, tracker: "SessionTracker") -> None:
        """Test that track_request marks session as dirty."""
        request = {"messages": [{"role": "user", "content": "hello"}]}

        session_id = await tracker.track_request(request, client_ip="127.0.0.1")

        assert tracker.get_dirty_count() == 1
        assert session_id in tracker._dirty_sessions

    @pytest.mark.asyncio
    async def test_mark_dirty_on_add_message(self, tracker: "SessionTracker") -> None:
        """Test that add_message marks session as dirty."""
        request = {"messages": [{"role": "user", "content": "hello"}]}
        session_id = await tracker.track_request(request, client_ip="127.0.0.1")

        # Clear dirty state
        tracker._dirty_sessions.clear()
        assert tracker.get_dirty_count() == 0

        # Add a message
        await tracker.add_message(session_id, "response", "assistant", "Hi there!")

        assert tracker.get_dirty_count() == 1

    @pytest.mark.asyncio
    async def test_debounced_flush(
        self, tracker: "SessionTracker", archive: SessionArchive
    ) -> None:
        """Test that sessions are flushed after debounce delay."""
        import asyncio

        request = {"messages": [{"role": "user", "content": "hello"}]}
        session_id = await tracker.track_request(request, client_ip="127.0.0.1")

        # Session should be dirty but not yet archived
        assert tracker.get_dirty_count() == 1
        assert await archive.get_session(session_id) is None

        # Wait for debounce delay + some margin
        await asyncio.sleep(0.6)

        # Manually trigger flush check (simulating the background loop)
        flushed = await tracker._flush_ready_sessions()

        assert flushed == 1
        assert tracker.get_dirty_count() == 0

        # Session should now be in archive
        archived = await archive.get_session(session_id)
        assert archived is not None
        assert archived["client_ip"] == "127.0.0.1"

    @pytest.mark.asyncio
    async def test_flush_all_on_shutdown(
        self, tracker: "SessionTracker", archive: SessionArchive
    ) -> None:
        """Test that flush_all persists all dirty sessions immediately."""
        # Create multiple sessions
        for i in range(3):
            request = {"messages": [{"role": "user", "content": f"hello {i}"}]}
            await tracker.track_request(request, client_ip=f"127.0.0.{i}")

        assert tracker.get_dirty_count() == 3

        # Flush all immediately (as on shutdown)
        flushed = await tracker.flush_all()

        assert flushed == 3
        assert tracker.get_dirty_count() == 0

        # All sessions should be in archive
        summaries = await archive.list_sessions()
        assert len(summaries) == 3

    @pytest.mark.asyncio
    async def test_restore_from_archive(
        self, tracker: "SessionTracker", archive: SessionArchive
    ) -> None:
        """Test that recent sessions are restored from archive on startup."""
        # Create and archive a session
        request = {"messages": [{"role": "user", "content": "hello"}]}
        session_id = await tracker.track_request(request, client_ip="127.0.0.1")
        # Add request messages (simulates what the proxy does)
        await tracker.add_request_messages(session_id, request["messages"])
        await tracker.add_message(session_id, "response", "assistant", "Hi!")

        # Flush to archive
        await tracker.flush_all()

        # Create a new tracker (simulating restart)
        from toolbridge import SessionTracker

        new_tracker = SessionTracker(
            session_timeout=3600,
            message_buffer_size=100,
            archive=archive,
            archive_on_change=True,
            archive_debounce_seconds=0.5,
        )

        # Verify session is not in memory yet
        assert session_id not in new_tracker._sessions

        # Restore from archive
        restored = await new_tracker.restore_from_archive()

        assert restored == 1
        assert session_id in new_tracker._sessions

        # Verify restored session data
        stats = await new_tracker.get_session_stats(session_id)
        assert stats is not None
        assert stats.client_ip == "127.0.0.1"
        assert stats.request_count == 1
        assert len(stats.messages) == 2  # user message + assistant message

    @pytest.mark.asyncio
    async def test_restore_skips_expired_sessions(
        self, archive: SessionArchive
    ) -> None:
        """Test that restore_from_archive skips sessions older than timeout."""
        from toolbridge import SessionTracker

        # Archive a session with old last_seen_at
        old_session_data = {
            "session_id": "old_session",
            "created_at": time.time() - 7200,  # 2 hours ago
            "last_seen_at": time.time() - 7200,  # 2 hours ago (expired)
            "request_count": 1,
            "tool_calls_total": 0,
            "tool_calls_fixed": 0,
            "tool_calls_failed": 0,
            "client_ip": "127.0.0.1",
            "messages": [],
        }
        await archive.archive_session("old_session", old_session_data)

        # Create tracker with 1 hour timeout
        tracker = SessionTracker(
            session_timeout=3600,  # 1 hour
            message_buffer_size=100,
            archive=archive,
            archive_on_change=True,
            archive_debounce_seconds=0.5,
        )

        # Restore should skip the expired session
        restored = await tracker.restore_from_archive()

        assert restored == 0
        assert "old_session" not in tracker._sessions

    @pytest.mark.asyncio
    async def test_background_flush_loop(
        self, tracker: "SessionTracker", archive: SessionArchive
    ) -> None:
        """Test that background flush loop runs and flushes sessions."""
        import asyncio

        request = {"messages": [{"role": "user", "content": "hello"}]}
        session_id = await tracker.track_request(request, client_ip="127.0.0.1")

        # Start the flush loop
        await tracker.start_flush_loop()

        try:
            # Wait for debounce delay + flush check interval + margin
            await asyncio.sleep(1.2)

            # Session should have been flushed
            assert tracker.get_dirty_count() == 0
            archived = await archive.get_session(session_id)
            assert archived is not None
        finally:
            # Stop the flush loop
            await tracker.stop_flush_loop()

    @pytest.mark.asyncio
    async def test_no_archive_when_disabled(self) -> None:
        """Test that sessions are not archived when archive_on_change is False."""
        from toolbridge import SessionTracker

        tracker = SessionTracker(
            session_timeout=3600,
            message_buffer_size=100,
            archive=None,  # No archive
            archive_on_change=False,
        )

        request = {"messages": [{"role": "user", "content": "hello"}]}
        await tracker.track_request(request, client_ip="127.0.0.1")

        # Should not mark as dirty when archive_on_change is False
        assert tracker.get_dirty_count() == 0

    @pytest.mark.asyncio
    async def test_zero_tokens_preserve_previous_values(self) -> None:
        """Test that zero token values don't overwrite valid previous values.

        When the API returns 0/null tokens (e.g., during context compression),
        the session should preserve its previous token values rather than
        zeroing them out.
        """
        from toolbridge import SessionTracker

        tracker = SessionTracker(
            session_timeout=3600,
            message_buffer_size=100,
            archive=None,
            archive_on_change=False,
        )

        request = {"messages": [{"role": "user", "content": "hello"}]}

        # First request with valid token values
        session_id = await tracker.track_request(
            request,
            client_ip="127.0.0.1",
            prompt_tokens=1000,
            completion_tokens=200,
            total_tokens=1200,
        )

        stats = await tracker.get_session_stats(session_id)
        assert stats is not None
        assert stats.last_prompt_tokens == 1000
        assert stats.last_completion_tokens == 200
        assert stats.last_total_tokens == 1200

        # Second request with zero tokens (simulates context compression or API error)
        await tracker.track_request(
            request,
            client_ip="127.0.0.1",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )

        # Verify previous values are preserved
        stats = await tracker.get_session_stats(session_id)
        assert stats is not None
        assert stats.last_prompt_tokens == 1000  # Should NOT be 0
        assert stats.last_completion_tokens == 200  # Should NOT be 0
        assert stats.last_total_tokens == 1200  # Should NOT be 0

        # But request count should still increment
        assert stats.request_count == 2

        # Third request with new valid values should update
        await tracker.track_request(
            request,
            client_ip="127.0.0.1",
            prompt_tokens=2000,
            completion_tokens=300,
            total_tokens=2300,
        )

        stats = await tracker.get_session_stats(session_id)
        assert stats is not None
        assert stats.last_prompt_tokens == 2000
        assert stats.last_completion_tokens == 300
        assert stats.last_total_tokens == 2300
        assert stats.request_count == 3


class TestContinuationDetection:
    """Tests for context continuation detection in session ID generation."""

    @pytest.fixture
    def tracker(self) -> "SessionTracker":
        """Create a SessionTracker for testing."""
        from toolbridge import SessionTracker

        return SessionTracker(
            session_timeout=3600,
            message_buffer_size=100,
            archive=None,
            archive_on_change=False,
        )

    def test_is_continuation_message_cline_pattern(self) -> None:
        """Test detection of Cline's continuation marker."""
        from toolbridge import is_continuation_message

        # Exact Cline pattern
        assert is_continuation_message("[Continue assisting the user!]") is True
        # Without exclamation
        assert is_continuation_message("[Continue assisting the user]") is True
        # With extra content after
        assert is_continuation_message("[Continue assisting the user!] More text") is True
        # Leading whitespace
        assert is_continuation_message("  [Continue assisting the user!]") is True

    def test_is_continuation_message_multimodal_content(self) -> None:
        """Test continuation detection with multimodal content format."""
        from toolbridge import is_continuation_message

        # Cline sends content as list of blocks
        content = [
            {"type": "text", "text": "[Continue assisting the user!]"},
            {"type": "text", "text": "More instructions"},
        ]
        assert is_continuation_message(content) is True

    def test_is_continuation_message_not_continuation(self) -> None:
        """Test that normal messages are not detected as continuations."""
        from toolbridge import is_continuation_message

        assert is_continuation_message("Hello, please help me") is False
        assert is_continuation_message("Continue the task") is False  # No brackets
        assert is_continuation_message("") is False
        assert is_continuation_message([]) is False

    def test_compressed_sessions_have_consistent_ids(self, tracker: "SessionTracker") -> None:
        """Test that multiple compressed sessions with same system prompt get same ID.

        When Cline compresses context multiple times, each compressed session
        should get the same session ID (based on system prompt only).
        """
        system_prompt = "You are Cline, a helpful assistant."

        # First compression
        compressed_request_1 = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "[Continue assisting the user!]"},
                {"role": "assistant", "content": "Context note 1"},
                {"role": "user", "content": "Continue helping"},
                {"role": "assistant", "content": "I'll continue."},
                {"role": "user", "content": "Thanks"},
            ]
        }
        session_id_1 = tracker.get_session_id(compressed_request_1, client_ip="127.0.0.1")

        # Second compression (different conversation content, same system prompt)
        compressed_request_2 = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "[Continue assisting the user!]"},
                {"role": "assistant", "content": "Different context note"},
                {"role": "user", "content": "Different question"},
                {"role": "assistant", "content": "Different answer"},
                {"role": "user", "content": "More"},
                {"role": "assistant", "content": "More answer"},
            ]
        }
        session_id_2 = tracker.get_session_id(compressed_request_2, client_ip="127.0.0.1")

        # Both compressed sessions should have the same ID
        # (both use only system prompt for hashing)
        assert session_id_1 == session_id_2

    def test_original_vs_compressed_session_ids_differ(
        self, tracker: "SessionTracker"
    ) -> None:
        """Test that original and compressed sessions have different IDs.

        The original session ID includes the first user message, while the
        compressed session only uses the system prompt. This is unavoidable
        since we can't recover the original first user message after compression.
        """
        system_prompt = "You are Cline, a helpful assistant."

        # Original request (few messages, no continuation marker)
        original_request = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Help me with my project"},
                {"role": "assistant", "content": "I'll help you."},
            ]
        }
        original_session_id = tracker.get_session_id(original_request, client_ip="127.0.0.1")

        # Compressed request (many messages, continuation marker)
        compressed_request = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "[Continue assisting the user!]"},
                {"role": "assistant", "content": "Context note"},
                {"role": "user", "content": "Continue helping"},
                {"role": "assistant", "content": "I'll continue."},
                {"role": "user", "content": "Thanks"},
            ]
        }
        compressed_session_id = tracker.get_session_id(
            compressed_request, client_ip="127.0.0.1"
        )

        # Session IDs will differ because:
        # - Original: hash(system + "Help me with my project")
        # - Compressed: hash(system only)
        assert original_session_id != compressed_session_id

    def test_session_id_different_without_continuation(
        self, tracker: "SessionTracker"
    ) -> None:
        """Test that different first user messages create different session IDs.

        When there's no continuation marker, different first user messages
        should result in different session IDs.
        """
        system_prompt = "You are Cline, a helpful assistant."

        request1 = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Help me with project A"},
            ]
        }
        request2 = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Help me with project B"},
            ]
        }

        session_id_1 = tracker.get_session_id(request1, client_ip="127.0.0.1")
        session_id_2 = tracker.get_session_id(request2, client_ip="127.0.0.1")

        # Different first user messages = different session IDs
        assert session_id_1 != session_id_2

    def test_continuation_requires_many_messages(self, tracker: "SessionTracker") -> None:
        """Test that continuation detection only applies with many messages.

        A short conversation with a continuation marker should NOT be treated
        as a compressed context (to avoid false positives).
        """
        system_prompt = "You are Cline, a helpful assistant."

        # Short request with continuation marker (shouldn't trigger special handling)
        short_request = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "[Continue assisting the user!]"},
            ]
        }

        # Request without continuation marker (same message count)
        normal_request = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Normal first message"},
            ]
        }

        short_session_id = tracker.get_session_id(short_request, client_ip="127.0.0.1")
        normal_session_id = tracker.get_session_id(normal_request, client_ip="127.0.0.1")

        # With few messages, continuation detection doesn't apply
        # So these should be different (different first user messages hashed)
        assert short_session_id != normal_session_id

    @pytest.mark.asyncio
    async def test_compressed_session_attaches_to_active_session(self) -> None:
        """Test that compressed sessions attach to active original sessions.

        This is the key behavior: when Cline compresses context, if the original
        session is still active (in memory), the compressed session should attach
        to it rather than creating a new session.
        """
        from toolbridge import SessionTracker

        tracker = SessionTracker(
            session_timeout=3600,
            message_buffer_size=100,
            archive=None,
            archive_on_change=False,
        )

        system_prompt = "You are Cline, a helpful assistant."

        # Original request (before compression)
        original_request = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Help me with my project"},
                {"role": "assistant", "content": "I'll help you."},
            ]
        }

        # Track the original session (this creates the session and indexes it)
        original_session_id = await tracker.track_request(
            original_request, client_ip="127.0.0.1"
        )

        # Verify the session was created
        stats = await tracker.get_session_stats(original_session_id)
        assert stats is not None
        assert stats.request_count == 1

        # Compressed request (many messages, continuation marker)
        compressed_request = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "[Continue assisting the user!]"},
                {"role": "assistant", "content": "Context note"},
                {"role": "user", "content": "Continue helping"},
                {"role": "assistant", "content": "I'll continue."},
                {"role": "user", "content": "Thanks"},
            ]
        }

        # Track the compressed session - should attach to original
        compressed_session_id = await tracker.track_request(
            compressed_request, client_ip="127.0.0.1"
        )

        # Session IDs should match! The compressed session attached to the original.
        assert compressed_session_id == original_session_id

        # Verify request count incremented on the same session
        stats = await tracker.get_session_stats(original_session_id)
        assert stats is not None
        assert stats.request_count == 2

    @pytest.mark.asyncio
    async def test_compressed_session_creates_new_when_original_expired(self) -> None:
        """Test that compressed sessions create new session when original expired.

        If the original session has expired (no longer in memory), the compressed
        session should create a new session using the secondary fingerprint.
        """
        from toolbridge import SessionTracker

        # Use very short timeout for testing
        tracker = SessionTracker(
            session_timeout=1,  # 1 second timeout
            message_buffer_size=100,
            archive=None,
            archive_on_change=False,
        )

        system_prompt = "You are Cline, a helpful assistant."

        # Original request
        original_request = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Help me with my project"},
            ]
        }

        # Track the original session
        original_session_id = await tracker.track_request(
            original_request, client_ip="127.0.0.1"
        )

        # Wait for session to "expire" (become stale)
        import asyncio

        await asyncio.sleep(1.1)

        # Compressed request
        compressed_request = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "[Continue assisting the user!]"},
                {"role": "assistant", "content": "Context note"},
                {"role": "user", "content": "Continue helping"},
                {"role": "assistant", "content": "I'll continue."},
                {"role": "user", "content": "Thanks"},
            ]
        }

        # Get session ID (don't track yet) - should NOT match original because it's expired
        compressed_session_id = tracker.get_session_id(
            compressed_request, client_ip="127.0.0.1"
        )

        # Should be different from original (original expired, so no attachment)
        assert compressed_session_id != original_session_id

    @pytest.mark.asyncio
    async def test_multimodal_first_user_message_extracts_task_block(self) -> None:
        """Test that only the task block is used from multimodal first user message.

        Cline sends first user message with multiple text blocks:
        1. Task (e.g., "<task>Execute...</task>")
        2. task_progress template
        3. environment_details

        Only the first block should be used for session ID hashing.
        """
        from toolbridge import SessionTracker

        tracker = SessionTracker(
            session_timeout=3600,
            message_buffer_size=100,
            archive=None,
            archive_on_change=False,
        )

        system_prompt = "You are Cline, a helpful assistant."

        # Multimodal content (like Cline sends)
        multimodal_content = [
            {"type": "text", "text": "<task>Execute my-task.md</task>"},
            {"type": "text", "text": "# task_progress RECOMMENDED\n...template..."},
            {"type": "text", "text": "<environment_details>...</environment_details>"},
        ]

        # Request with multimodal content
        multimodal_request = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": multimodal_content},
            ]
        }

        # Request with just the task as plain string
        plain_request = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "<task>Execute my-task.md</task>"},
            ]
        }

        multimodal_session_id = tracker.get_session_id(
            multimodal_request, client_ip="127.0.0.1"
        )
        plain_session_id = tracker.get_session_id(plain_request, client_ip="127.0.0.1")

        # Should be the same - only the first text block matters
        assert multimodal_session_id == plain_session_id
