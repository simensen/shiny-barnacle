"""Tests for session archive functionality."""

import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from archive import (
    ArchiveIndex,
    SessionArchive,
    session_stats_to_archive_dict,
)
from paths import ensure_dir, get_archive_dir, get_state_home


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
