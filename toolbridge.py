#!/usr/bin/env python3
"""
LLM Response Transformer Proxy

An OpenAI-compatible proxy that transforms malformed tool calls into proper format.
Instead of retrying failed responses, it parses and reconstructs them.

Handles transformations like:
    <function=writeFile>                    →  tool_calls: [{
    <parameter=path>src/app.js</parameter>        function: {
    </function>                                      name: "writeFile",
                                                     arguments: '{"path":"src/app.js"}'
                                                   }
                                                 }]
"""

import argparse
import asyncio
import hashlib
import json
import logging
import re
import time
import uuid
from collections import deque
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any
from xml.sax.saxutils import unescape as xml_unescape

import httpx
from fastapi import FastAPI, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from archive import SessionArchive, session_stats_to_archive_dict
from paths import get_archive_dir

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# Stats Tracking
# =============================================================================


@dataclass
class ProxyStats:
    """Track proxy statistics for observability"""

    total_requests: int = 0
    passthrough_requests: int = 0  # Model output was already valid
    transformed_requests: int = 0  # Proxy fixed malformed output
    failed_transforms: int = 0  # Transformation attempted but failed
    backend_errors: int = 0  # Backend returned error

    def record_passthrough(self) -> None:
        self.total_requests += 1
        self.passthrough_requests += 1

    def record_transformed(self) -> None:
        self.total_requests += 1
        self.transformed_requests += 1

    def record_failed(self) -> None:
        self.total_requests += 1
        self.failed_transforms += 1

    def record_backend_error(self) -> None:
        self.total_requests += 1
        self.backend_errors += 1

    def to_dict(self) -> dict[str, Any]:
        total = self.total_requests or 1  # Avoid division by zero
        return {
            "total_requests": self.total_requests,
            "passthrough": {
                "count": self.passthrough_requests,
                "percent": round(100 * self.passthrough_requests / total, 1),
            },
            "transformed": {
                "count": self.transformed_requests,
                "percent": round(100 * self.transformed_requests / total, 1),
            },
            "failed": {
                "count": self.failed_transforms,
                "percent": round(100 * self.failed_transforms / total, 1),
            },
            "backend_errors": {
                "count": self.backend_errors,
                "percent": round(100 * self.backend_errors / total, 1),
            },
        }


# Global stats instance
stats = ProxyStats()


# =============================================================================
# Session Tracking
# =============================================================================


@dataclass
class ChatMessage:
    """A single chat message for logging purposes"""

    timestamp: float
    direction: str  # "request" or "response"
    role: str  # "user", "assistant", "system", "tool"
    content: str  # Full message content (after transformation if any)
    tool_calls: list[dict[str, Any]] | None = None  # If response contains tool calls
    raw_content: str | None = None  # Original content before transformation (if transformed)
    debug: dict[str, Any] | None = None  # Original JSON payload (parsed)
    prompt_tokens: int | None = None  # Snapshot of prompt_tokens at time of logging


@dataclass
class SessionStats:
    """Track per-session statistics"""

    created_at: float = field(default_factory=time.time)
    last_seen_at: float = field(default_factory=time.time)
    request_count: int = 0
    tool_calls_total: int = 0
    tool_calls_fixed: int = 0
    tool_calls_failed: int = 0
    client_ip: str | None = None
    messages: deque[ChatMessage] = field(default_factory=deque)  # Circular buffer of ChatMessage
    last_request_message_count: int = 0  # Track logged message count for deduplication
    # Token usage - last request (shows current context window size)
    last_prompt_tokens: int = 0
    last_completion_tokens: int = 0
    last_total_tokens: int = 0
    # Token usage - cumulative (shows total session cost)
    prompt_tokens_total: int = 0
    completion_tokens_total: int = 0
    total_tokens_total: int = 0


class SessionTracker:
    """
    Track sessions with automatic expiry.

    Sessions are identified by hashing the stable portion of the conversation
    (first 2 messages: typically system prompt + first user message).
    """

    def __init__(
        self,
        session_timeout: int = 3600,
        message_buffer_size: int = 1024,
        archive: SessionArchive | None = None,
    ):
        """
        Args:
            session_timeout: Seconds of inactivity before session expires (default 1 hour)
            message_buffer_size: Max messages per session circular buffer (default 1024)
            archive: Optional SessionArchive for persisting expired sessions
        """
        self._sessions: dict[str, SessionStats] = {}
        self._lock = asyncio.Lock()
        self.session_timeout = session_timeout
        self.message_buffer_size = message_buffer_size
        self.archive = archive

    def get_session_id(self, request: dict[str, Any], client_ip: str | None = None) -> str:
        """
        Generate a stable session ID from a chat completion request.

        Uses the first two messages (typically system + first user message)
        which remain constant throughout a conversation.
        """
        messages = request.get("messages", [])

        # Option 1: Use 'user' field if the client provides it (rare but ideal)
        if request.get("user"):
            return f"user_{request['user']}"

        # Option 2: Hash the stable prefix (first 2 messages)
        stable_messages = messages[:2]
        normalized = [
            {"role": msg.get("role"), "content": msg.get("content", "")}
            for msg in stable_messages
        ]

        # Include client IP for multi-user separation
        content = json.dumps(normalized, sort_keys=True, ensure_ascii=True)
        if client_ip:
            content += f"|ip:{client_ip}"

        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def track_request(
        self,
        request: dict[str, Any],
        client_ip: str | None = None,
        tool_calls_in_response: int = 0,
        tool_calls_fixed: int = 0,
        tool_calls_failed: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
    ) -> str:
        """
        Track a request and return its session ID.

        Call this after processing each request to update session stats.
        """
        session_id = self.get_session_id(request, client_ip)

        async with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionStats(
                    client_ip=client_ip,
                    messages=deque(maxlen=self.message_buffer_size),
                )

            session_stats = self._sessions[session_id]
            session_stats.last_seen_at = time.time()
            session_stats.request_count += 1
            session_stats.tool_calls_total += tool_calls_in_response
            session_stats.tool_calls_fixed += tool_calls_fixed
            session_stats.tool_calls_failed += tool_calls_failed

            # Update token usage - last request values
            session_stats.last_prompt_tokens = prompt_tokens
            session_stats.last_completion_tokens = completion_tokens
            session_stats.last_total_tokens = total_tokens

            # Update token usage - cumulative totals
            session_stats.prompt_tokens_total += prompt_tokens
            session_stats.completion_tokens_total += completion_tokens
            session_stats.total_tokens_total += total_tokens

            # Update client IP if not set
            if client_ip and not session_stats.client_ip:
                session_stats.client_ip = client_ip

        return session_id

    async def get_session_stats(self, session_id: str) -> SessionStats | None:
        """Get stats for a specific session."""
        async with self._lock:
            return self._sessions.get(session_id)

    async def add_message(
        self,
        session_id: str,
        direction: str,
        role: str,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
        raw_content: str | None = None,
        debug: dict[str, Any] | None = None,
        prompt_tokens: int | None = None,
    ) -> None:
        """
        Add a chat message to a session's message buffer.

        Args:
            session_id: The session ID to add the message to
            direction: "request" or "response"
            role: Message role ("user", "assistant", "system", "tool")
            content: The message content (after transformation if any)
            tool_calls: Optional list of tool calls (for assistant responses)
            raw_content: Original content before transformation (if transformed)
            debug: Original JSON payload (parsed)
            prompt_tokens: Snapshot of prompt_tokens at time of logging (for context size tracking)
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if session:
                message = ChatMessage(
                    timestamp=time.time(),
                    direction=direction,
                    role=role,
                    content=content,
                    tool_calls=tool_calls,
                    raw_content=raw_content,
                    debug=debug,
                    prompt_tokens=prompt_tokens,
                )
                session.messages.append(message)

    async def add_request_messages(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
        prompt_tokens: int | None = None,
    ) -> None:
        """
        Add only NEW request messages to the buffer.

        Tracks how many messages were in the previous request and only logs
        messages beyond that index, avoiding duplicate logging of the entire
        conversation history on each request.

        Args:
            session_id: The session ID to add messages to
            messages: The full messages array from the request body
            prompt_tokens: Snapshot of prompt_tokens at time of logging (for context size tracking)
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return

            # Only log messages beyond what we've already seen
            new_messages = messages[session.last_request_message_count:]

            for msg in new_messages:
                content = msg.get("content", "")
                # Handle content that may be a list (e.g., multi-modal messages)
                if isinstance(content, list):
                    content = str(content)
                message = ChatMessage(
                    timestamp=time.time(),
                    direction="request",
                    role=msg.get("role", "unknown"),
                    content=content,
                    tool_calls=msg.get("tool_calls"),  # Extract tool_calls consistently
                    debug=msg,  # Original JSON payload
                    prompt_tokens=prompt_tokens,  # Context size at this point in conversation
                )
                session.messages.append(message)

            # Update the count for next request
            session.last_request_message_count = len(messages)

    async def get_all_sessions(self) -> dict[str, SessionStats]:
        """Get all active sessions (excludes expired)."""
        await self._cleanup_expired()
        async with self._lock:
            return dict(self._sessions)

    async def _cleanup_expired(self) -> int:
        """Remove expired sessions, archiving them if enabled. Returns count of removed sessions."""
        now = time.time()
        removed = 0

        async with self._lock:
            expired_ids = [
                sid
                for sid, stats in self._sessions.items()
                if now - stats.last_seen_at > self.session_timeout
            ]

            for sid in expired_ids:
                # Archive before deletion if archive is enabled
                if self.archive is not None:
                    stats = self._sessions[sid]
                    archive_data = session_stats_to_archive_dict(sid, stats)
                    await self.archive.archive_session(sid, archive_data)

                del self._sessions[sid]
                removed += 1

        return removed


# Global session tracker instance
session_tracker = SessionTracker(session_timeout=3600)

# Global session archive instance (initialized in main() based on config)
session_archive: SessionArchive | None = None


class ProxyConfig(BaseSettings):
    """Proxy configuration settings.

    Configuration can be set via:
    1. CLI arguments (highest priority)
    2. Environment variables (TOOLBRIDGE_<SETTING_NAME>)
    3. .env file in the working directory
    4. Default values (lowest priority)
    """

    model_config = SettingsConfigDict(
        env_prefix="TOOLBRIDGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Connection settings
    backend_url: str = Field(
        default="http://localhost:8080",
        description="Backend llama.cpp server URL",
    )
    port: int = Field(
        default=4000,
        description="Port to listen on",
    )
    host: str = Field(
        default="0.0.0.0",
        description="Host to bind to",
    )

    # Transformation settings
    transform_enabled: bool = Field(
        default=True,
        description="Enable tool call transformation",
    )
    fallback_to_retry: bool = Field(
        default=True,
        description="If transform fails, try retry",
    )
    max_retries: int = Field(
        default=1,
        description="Max retries if transform fails",
    )

    # Streaming settings
    stream_buffer_timeout: float = Field(
        default=120.0,
        description="Timeout for streaming buffer in seconds",
    )

    # Sampling parameters (None means use client's value or backend default)
    temperature: float | None = Field(
        default=None,
        description="Override temperature (e.g., 0.7)",
    )
    top_p: float | None = Field(
        default=None,
        description="Override top_p (e.g., 0.9)",
    )
    top_k: int | None = Field(
        default=None,
        description="Override top_k (e.g., 40)",
    )
    min_p: float | None = Field(
        default=None,
        description="Override min_p (e.g., 0.05)",
    )
    repeat_penalty: float | None = Field(
        default=None,
        description="Override repeat_penalty (e.g., 1.0)",
    )
    presence_penalty: float | None = Field(
        default=None,
        description="Override presence_penalty",
    )
    frequency_penalty: float | None = Field(
        default=None,
        description="Override frequency_penalty",
    )

    # Debug settings
    debug: bool = Field(
        default=False,
        description="Enable debug logging",
    )
    log_sampling_params: bool = Field(
        default=True,
        description="Log sampling params for each request",
    )

    # Message logging settings
    log_messages: bool = Field(
        default=True,
        description="Enable message logging in sessions",
    )
    message_buffer_size: int = Field(
        default=1024,
        description="Max messages per session (circular buffer)",
    )

    # CORS settings
    cors_enabled: bool = Field(
        default=False,
        description="Enable CORS for admin endpoints",
    )
    cors_origins: list[str] | None = Field(
        default=None,
        description="Allowed CORS origins (None = allow all '*')",
    )
    cors_all_routes: bool = Field(
        default=False,
        description="Enable CORS for ALL routes including /v1/* proxy endpoints",
    )

    # Archive settings
    archive_enabled: bool = Field(
        default=True,
        description="Enable session archiving when sessions expire",
    )
    archive_dir: str | None = Field(
        default=None,
        description="Directory for session archives (default: XDG state dir)",
    )
    archive_ttl_hours: int = Field(
        default=168,
        description="Hours to retain archived sessions (0 = forever, default: 168 = 7 days)",
    )


def load_config() -> ProxyConfig:
    """Load configuration from environment variables and .env file."""
    return ProxyConfig()


config = load_config()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan context manager for startup/shutdown tasks."""
    # Startup: validate archive index if archiving is enabled
    if session_archive is not None:
        logger.info("Validating session archive on startup...")
        validation_result = await session_archive.validate_on_startup()
        if validation_result["index_rebuilt"]:
            logger.info(
                f"  Archive index was rebuilt: {validation_result['session_count']} sessions"
            )
        if validation_result["expired_removed"] > 0:
            logger.info(
                f"  Expired archives cleaned up: {validation_result['expired_removed']} sessions"
            )
        if not validation_result["index_rebuilt"] and validation_result["expired_removed"] == 0:
            logger.info(
                f"  Archive validated: {validation_result['session_count']} sessions"
            )

    yield  # App runs here

    # Shutdown: nothing to do currently
    pass


app = FastAPI(title="LLM Response Transformer Proxy", lifespan=lifespan)


def apply_sampling_params(body: dict[str, Any]) -> dict[str, Any]:
    """Apply proxy-level sampling parameters to request body.

    Proxy params override client params when set.
    Returns the modified body and logs the effective params.
    """
    # All known sampling-related parameter names (OpenAI + llama.cpp extensions)
    all_sampling_params = {
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repeat_penalty",
        "presence_penalty",
        "frequency_penalty",
        "mirostat",
        "mirostat_tau",
        "mirostat_eta",
        "seed",
        "stop",
        "max_tokens",
        "n",
        "typical_p",
        "tfs_z",
        "repeat_last_n",
        "penalize_nl",
        "dry_multiplier",
        "dry_base",
    }

    # Collect ALL sampling params the client sent (not just ones we override)
    client_params = {
        k: v for k, v in body.items() if k in all_sampling_params and v is not None
    }

    # Apply proxy overrides
    overrides = {}
    if config.temperature is not None:
        body["temperature"] = config.temperature
        overrides["temperature"] = config.temperature
    if config.top_p is not None:
        body["top_p"] = config.top_p
        overrides["top_p"] = config.top_p
    if config.top_k is not None:
        body["top_k"] = config.top_k
        overrides["top_k"] = config.top_k
    if config.min_p is not None:
        body["min_p"] = config.min_p
        overrides["min_p"] = config.min_p
    if config.repeat_penalty is not None:
        body["repeat_penalty"] = config.repeat_penalty
        overrides["repeat_penalty"] = config.repeat_penalty
    if config.presence_penalty is not None:
        body["presence_penalty"] = config.presence_penalty
        overrides["presence_penalty"] = config.presence_penalty
    if config.frequency_penalty is not None:
        body["frequency_penalty"] = config.frequency_penalty
        overrides["frequency_penalty"] = config.frequency_penalty

    # Log the parameters
    if config.log_sampling_params:
        # Also log non-sampling keys for debugging (excluding huge fields)
        skip_keys = all_sampling_params | {
            "messages",
            "tools",
            "tool_choice",
            "functions",
        }
        other_keys = {k: body.get(k) for k in body if k not in skip_keys}

        if overrides:
            # Calculate effective values for params we care about
            effective = {}
            for param in [
                "temperature",
                "top_p",
                "top_k",
                "min_p",
                "repeat_penalty",
                "presence_penalty",
                "frequency_penalty",
            ]:
                val = body.get(param)
                if val is not None:
                    effective[param] = val
            logger.info(
                f"Sampling params - Client sent: {client_params}, "
                f"Proxy overrides: {overrides}, Effective: {effective}"
            )
        else:
            logger.info(
                f"Sampling params - Client sent: {client_params} (no proxy overrides)"
            )

        # Log other interesting body keys
        if other_keys:
            logger.debug(f"Other request params: {other_keys}")

    return body


# =============================================================================
# Tool Call Parser
# =============================================================================


@dataclass
class ParsedToolCall:
    """Represents a parsed tool call"""

    function_name: str
    arguments: dict[str, Any]
    raw_text: str  # Original text that was parsed


@dataclass
class ParsedResponse:
    """Represents a fully parsed response"""

    preamble: str  # Text before tool calls
    tool_calls: list[ParsedToolCall]
    postamble: str  # Text after tool calls (rare)
    was_transformed: bool


class ToolCallParser:
    """
    Parser for various malformed tool call formats.

    Supports:
    - <function=name>...</function>
    - <function name="name">...</function>
    - <tool_call><function=name>...</function></tool_call> (passthrough)
    - Raw JSON tool calls
    """

    # Pattern to match various function call formats:
    # - <function=name>...</function>
    # - <function="name">...</function>
    # - <function name="name">...</function>
    # - <function name=name>...</function>
    FUNCTION_PATTERN = re.compile(
        r'<function(?:\s+name)?\s*[=:]\s*["\']?([^"\'>\s]+)["\']?\s*>(.*?)</function>',
        re.DOTALL | re.IGNORECASE,
    )

    # Pattern to match various parameter formats:
    # - <parameter=name>value</parameter>
    # - <parameter name="name">value</parameter>
    PARAMETER_PATTERN = re.compile(
        r'<parameter(?:\s+name)?\s*[=:]\s*["\']?([^"\'>\s]+)["\']?\s*>(.*?)</parameter>',
        re.DOTALL | re.IGNORECASE,
    )

    # Pattern for already-correct tool_call wrapper
    VALID_TOOL_CALL_PATTERN = re.compile(
        r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE
    )

    # Pattern for JSON-style tool calls (some models output this)
    JSON_TOOL_CALL_PATTERN = re.compile(
        r'\{[\s\n]*"name"[\s\n]*:[\s\n]*"([^"]+)"[\s\n]*,[\s\n]*"arguments"[\s\n]*:[\s\n]*(\{[^}]*\}|\[[^\]]*\]|"[^"]*")',
        re.DOTALL,
    )

    @classmethod
    def has_malformed_tool_call(cls, content: str) -> bool:
        """Check if content has tool calls that need transformation."""
        if not content:
            return False

        # Already valid - no transformation needed
        if cls.VALID_TOOL_CALL_PATTERN.search(content):
            # But check if there are ALSO malformed ones outside the wrapper
            # Remove valid ones and check what's left
            remaining = cls.VALID_TOOL_CALL_PATTERN.sub("", content)
            if not cls.FUNCTION_PATTERN.search(remaining):
                return False

        # Has function tags without proper wrapper
        if cls.FUNCTION_PATTERN.search(content):
            return True

        # Has raw JSON tool calls
        return bool(cls.JSON_TOOL_CALL_PATTERN.search(content))

    @classmethod
    def parse(cls, content: str) -> ParsedResponse:
        """
        Parse content and extract tool calls.

        Returns ParsedResponse with extracted tool calls and surrounding text.
        """
        if not content:
            return ParsedResponse(
                preamble="", tool_calls=[], postamble="", was_transformed=False
            )

        tool_calls = []

        # First, check for valid tool_call wrappers (passthrough)
        valid_matches = list(cls.VALID_TOOL_CALL_PATTERN.finditer(content))
        if valid_matches and not cls.FUNCTION_PATTERN.search(
            cls.VALID_TOOL_CALL_PATTERN.sub("", content)
        ):
            # All tool calls are properly wrapped, no transformation needed
            return ParsedResponse(
                preamble=content,  # Keep as-is
                tool_calls=[],
                postamble="",
                was_transformed=False,
            )

        # Find all function calls (malformed)
        function_matches = list(cls.FUNCTION_PATTERN.finditer(content))

        if not function_matches:
            # Try JSON format
            json_matches = list(cls.JSON_TOOL_CALL_PATTERN.finditer(content))
            if json_matches:
                return cls._parse_json_tool_calls(content, json_matches)

            # No tool calls found
            return ParsedResponse(
                preamble=content, tool_calls=[], postamble="", was_transformed=False
            )

        # Extract preamble (text before first function)
        first_match = function_matches[0]
        preamble = content[: first_match.start()].strip()

        # Extract postamble (text after last function)
        last_match = function_matches[-1]
        postamble = content[last_match.end() :].strip()

        # Parse each function call
        for match in function_matches:
            function_name = match.group(1)
            function_body = match.group(2)

            # Parse parameters from function body
            arguments = cls._parse_parameters(function_body)

            tool_calls.append(
                ParsedToolCall(
                    function_name=function_name,
                    arguments=arguments,
                    raw_text=match.group(0),
                )
            )

        return ParsedResponse(
            preamble=preamble,
            tool_calls=tool_calls,
            postamble=postamble,
            was_transformed=True,
        )

    @classmethod
    def _parse_parameters(cls, body: str) -> dict[str, Any]:
        """Parse parameters from function body."""
        arguments = {}

        for param_match in cls.PARAMETER_PATTERN.finditer(body):
            param_name = param_match.group(1)
            param_value = param_match.group(2)

            # Unescape XML entities
            param_value = xml_unescape(param_value)

            # Try to parse as JSON if it looks like JSON
            param_value = cls._maybe_parse_json(param_value)

            arguments[param_name] = param_value

        return arguments

    @classmethod
    def _parse_json_tool_calls(cls, content: str, matches: list[re.Match[str]]) -> ParsedResponse:
        """Parse JSON-style tool calls."""
        tool_calls = []

        first_match = matches[0]
        preamble = content[: first_match.start()].strip()

        last_match = matches[-1]
        postamble = content[last_match.end() :].strip()

        for match in matches:
            function_name = match.group(1)
            arguments_str = match.group(2)

            try:
                arguments = json.loads(arguments_str)
            except json.JSONDecodeError:
                arguments = {"raw": arguments_str}

            tool_calls.append(
                ParsedToolCall(
                    function_name=function_name,
                    arguments=arguments,
                    raw_text=match.group(0),
                )
            )

        return ParsedResponse(
            preamble=preamble,
            tool_calls=tool_calls,
            postamble=postamble,
            was_transformed=True,
        )

    @staticmethod
    def _maybe_parse_json(value: str) -> Any:
        """Try to parse value as JSON, return original if not JSON."""
        value = value.strip()

        if not value:
            return value

        # Check if it looks like JSON
        if value.startswith(("{", "[", '"')) or value in ("true", "false", "null"):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass

        # Try to parse as number
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        return value


# =============================================================================
# Response Transformer
# =============================================================================


class ResponseTransformer:
    """
    Transforms parsed tool calls into OpenAI-compatible format.
    """

    @staticmethod
    def generate_tool_call_id() -> str:
        """Generate a unique tool call ID."""
        return f"call_{uuid.uuid4().hex[:24]}"

    @classmethod
    def transform_response(
        cls, original_response: dict[str, Any], parsed: ParsedResponse
    ) -> dict[str, Any]:
        """
        Transform a response with malformed tool calls into proper format.

        Args:
            original_response: The original API response dict
            parsed: ParsedResponse from ToolCallParser

        Returns:
            Transformed response dict with proper tool_calls structure
        """
        if not parsed.was_transformed or not parsed.tool_calls:
            return original_response

        response: dict[str, Any] = json.loads(json.dumps(original_response))  # Deep copy

        if "choices" not in response or not response["choices"]:
            return original_response

        # Build tool_calls array
        tool_calls = []
        for tc in parsed.tool_calls:
            tool_calls.append(
                {
                    "id": cls.generate_tool_call_id(),
                    "type": "function",
                    "function": {
                        "name": tc.function_name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
            )

        # Update the message
        message = response["choices"][0].get("message", {})

        # Set content to preamble (or None if empty, per OpenAI spec)
        if parsed.preamble:
            message["content"] = parsed.preamble
        else:
            message["content"] = None

        # Add tool_calls
        message["tool_calls"] = tool_calls

        # Update finish_reason
        response["choices"][0]["finish_reason"] = "tool_calls"
        response["choices"][0]["message"] = message

        # Add transformation metadata (optional, for debugging)
        if "usage" not in response:
            response["usage"] = {}
        response["usage"]["_transformed"] = True
        response["usage"]["_tool_calls_extracted"] = len(tool_calls)

        logger.info(f"Transformed response: extracted {len(tool_calls)} tool call(s)")

        return response

    @classmethod
    def transform_streaming_content(
        cls, full_content: str, parsed: ParsedResponse, metadata: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Transform streaming content into proper SSE chunks.

        Returns list of chunk dicts ready for SSE serialization.
        """
        chunks = []
        chunk_id = metadata.get("id", f"chatcmpl-{uuid.uuid4().hex[:8]}")
        model = metadata.get("model", "unknown")
        created = int(time.time())

        # First chunk(s): stream the preamble content
        if parsed.preamble:
            # Split preamble into smaller chunks for realistic streaming
            chunk_size = 20
            for i in range(0, len(parsed.preamble), chunk_size):
                chunk_content = parsed.preamble[i : i + chunk_size]
                chunks.append(
                    {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": chunk_content},
                                "finish_reason": None,
                            }
                        ],
                    }
                )

        # Tool call chunks
        for idx, tc in enumerate(parsed.tool_calls):
            # Tool call start
            chunks.append(
                {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": idx,
                                        "id": cls.generate_tool_call_id(),
                                        "type": "function",
                                        "function": {
                                            "name": tc.function_name,
                                            "arguments": "",
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                }
            )

            # Arguments in chunks
            args_str = json.dumps(tc.arguments)
            for i in range(0, len(args_str), 50):
                chunk_args = args_str[i : i + 50]
                chunks.append(
                    {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": idx,
                                            "function": {"arguments": chunk_args},
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                )

        # Final chunk
        chunks.append(
            {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls" if parsed.tool_calls else "stop",
                    }
                ],
            }
        )

        return chunks


# =============================================================================
# Request Handlers
# =============================================================================


@dataclass
class RequestResult:
    """Result from processing a request, including session tracking stats."""

    response: dict[str, Any]
    tool_calls_total: int = 0
    tool_calls_fixed: int = 0
    tool_calls_failed: int = 0
    raw_content: str | None = None  # Original content before transformation (if transformed)
    # Token usage from response
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


async def handle_non_streaming_request(
    client: httpx.AsyncClient, body: dict[str, Any], session_id: str
) -> RequestResult:
    """Handle non-streaming request with transformation."""

    logger.info(f"[{session_id}] Processing non-streaming request")

    response = await client.post(
        f"{config.backend_url}/v1/chat/completions", json=body, timeout=120.0
    )
    response.raise_for_status()
    result = response.json()

    # Extract content
    content = ""
    if "choices" in result and result["choices"]:
        message = result["choices"][0].get("message", {})
        content = message.get("content", "")

    # Track tool call stats for session
    tool_calls_total = 0
    tool_calls_fixed = 0
    tool_calls_failed = 0
    raw_content = None  # Will be set if transformation happens

    # Check if transformation is needed
    if config.transform_enabled and ToolCallParser.has_malformed_tool_call(content):
        logger.info(f"[{session_id}] Detected malformed tool call, transforming...")

        try:
            parsed = ToolCallParser.parse(content)
            tool_calls_total = len(parsed.tool_calls)

            if parsed.was_transformed:
                raw_content = content  # Save original content before transformation
                result = ResponseTransformer.transform_response(result, parsed)
                tool_calls_fixed = len(parsed.tool_calls)
                logger.info(
                    f"[{session_id}] Transformation successful: "
                    f"{len(parsed.tool_calls)} tool call(s)"
                )
                stats.record_transformed()
            else:
                stats.record_passthrough()
        except Exception as e:
            logger.error(f"[{session_id}] Transformation failed: {e}")
            tool_calls_failed = tool_calls_total
            stats.record_failed()
            # Return original response if transformation fails
    else:
        # No transformation needed - model output was valid
        # Check if there are valid tool calls in the response
        if "choices" in result and result["choices"]:
            message = result["choices"][0].get("message", {})
            existing_tool_calls = message.get("tool_calls", [])
            tool_calls_total = len(existing_tool_calls)

        stats.record_passthrough()
        logger.debug(f"[{session_id}] Passthrough: no transformation needed")

    # Extract token usage from response
    usage = result.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)

    return RequestResult(
        response=result,
        tool_calls_total=tool_calls_total,
        tool_calls_fixed=tool_calls_fixed,
        tool_calls_failed=tool_calls_failed,
        raw_content=raw_content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


@dataclass
class StreamCollectionResult:
    """Result from collecting a streaming response."""

    content: str
    chunks: list[dict[str, Any]]
    metadata: dict[str, Any]
    tool_calls: list[dict[str, Any]]  # Reconstructed tool calls from stream
    # Token usage (if backend supports stream_options.include_usage)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


async def collect_stream(
    client: httpx.AsyncClient, body: dict[str, Any]
) -> StreamCollectionResult:
    """Collect streaming response into content, chunks, and tool calls."""

    full_content = ""
    chunks = []
    metadata = {}
    # Track tool calls being built from stream chunks
    # Key: tool call index, Value: dict with id, type, function (name, arguments)
    tool_calls_builder: dict[int, dict[str, Any]] = {}
    # Token usage (captured from final chunk if backend supports it)
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    try:
        async with client.stream(
            "POST",
            f"{config.backend_url}/v1/chat/completions",
            json=body,
            timeout=config.stream_buffer_timeout,
        ) as response:
            # Check for errors before trying to read stream
            if response.status_code >= 400:
                error_body = await response.aread()
                logger.error(
                    f"Backend returned {response.status_code}: {error_body.decode()}"
                )
                raise httpx.HTTPStatusError(
                    f"Backend error: {error_body.decode()}",
                    request=response.request,
                    response=response,
                )

            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue

                data = line[6:]
                if data == "[DONE]":
                    break

                try:
                    chunk = json.loads(data)
                    chunks.append(chunk)

                    if "choices" in chunk and chunk["choices"]:
                        delta = chunk["choices"][0].get("delta", {})
                        content_piece = delta.get("content")
                        if content_piece is not None:
                            full_content += content_piece

                        # Collect tool calls from stream
                        delta_tool_calls = delta.get("tool_calls", [])
                        for tc in delta_tool_calls:
                            idx = tc.get("index", 0)
                            if idx not in tool_calls_builder:
                                tool_calls_builder[idx] = {
                                    "id": tc.get("id", ""),
                                    "type": tc.get("type", "function"),
                                    "function": {"name": "", "arguments": ""},
                                }
                            # Update with any new data
                            if tc.get("id"):
                                tool_calls_builder[idx]["id"] = tc["id"]
                            if tc.get("type"):
                                tool_calls_builder[idx]["type"] = tc["type"]
                            func = tc.get("function", {})
                            if func.get("name"):
                                tool_calls_builder[idx]["function"]["name"] = func[
                                    "name"
                                ]
                            if func.get("arguments"):
                                tool_calls_builder[idx]["function"][
                                    "arguments"
                                ] += func["arguments"]

                        if chunk["choices"][0].get("finish_reason"):
                            metadata["finish_reason"] = chunk["choices"][0][
                                "finish_reason"
                            ]

                    if "model" in chunk:
                        metadata["model"] = chunk["model"]
                    if "id" in chunk:
                        metadata["id"] = chunk["id"]

                    # Capture usage if present (typically in final chunk)
                    if "usage" in chunk:
                        usage = chunk["usage"]
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)
                        total_tokens = usage.get("total_tokens", 0)

                except json.JSONDecodeError:
                    continue
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error from backend: {e}")
        raise

    # Convert tool_calls_builder to sorted list
    tool_calls = [tool_calls_builder[idx] for idx in sorted(tool_calls_builder.keys())]

    return StreamCollectionResult(
        content=full_content,
        chunks=chunks,
        metadata=metadata,
        tool_calls=tool_calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


async def handle_streaming_request(
    body: dict[str, Any], session_id: str, client_ip: str | None
) -> AsyncGenerator[str, None]:
    """Handle streaming request with transformation."""

    logger.info(f"[{session_id}] Processing streaming request")

    body["stream"] = True
    # Request usage information in streaming response (OpenAI-compatible APIs)
    body["stream_options"] = {"include_usage": True}

    # Track tool call stats for session
    tool_calls_total = 0
    tool_calls_fixed = 0
    tool_calls_failed = 0

    # Create client inside the generator so it stays open during iteration
    async with httpx.AsyncClient() as client:
        try:
            stream_result = await collect_stream(client, body)
            content = stream_result.content
            original_chunks = stream_result.chunks
            metadata = stream_result.metadata
            stream_tool_calls = stream_result.tool_calls
        except httpx.HTTPStatusError as e:
            # Return error as SSE event
            logger.error(f"[{session_id}] Backend error during streaming: {e}")
            stats.record_backend_error()
            # Track session even on error
            await session_tracker.track_request(
                request=body,
                client_ip=client_ip,
                tool_calls_in_response=0,
                tool_calls_fixed=0,
                tool_calls_failed=0,
            )
            # Log request messages if enabled (no response on error, no token count available)
            if config.log_messages:
                await session_tracker.add_request_messages(
                    session_id, body.get("messages", []), prompt_tokens=None
                )
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "backend_error",
                    "code": e.response.status_code if e.response else 500,
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            return
        except Exception as e:
            logger.error(f"[{session_id}] Unexpected error during streaming: {e}")
            stats.record_backend_error()
            # Track session even on error
            await session_tracker.track_request(
                request=body,
                client_ip=client_ip,
                tool_calls_in_response=0,
                tool_calls_fixed=0,
                tool_calls_failed=0,
            )
            # Log request messages if enabled (no response on error, no token count available)
            if config.log_messages:
                await session_tracker.add_request_messages(
                    session_id, body.get("messages", []), prompt_tokens=None
                )
            error_chunk = {"error": {"message": str(e), "type": "proxy_error"}}
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Log what we collected for debugging
        logger.info(
            f"[{session_id}] Stream collected: {len(original_chunks)} chunks, "
            f"{len(content)} chars content"
        )
        if content:
            # Show first 200 chars to help debug
            preview = content[:200].replace("\n", "\\n")
            logger.info(f"[{session_id}] Content preview: {preview}...")

        # Check if transformation needed
        has_malformed = (
            config.transform_enabled and ToolCallParser.has_malformed_tool_call(content)
        )
        logger.info(f"[{session_id}] Has malformed tool call: {has_malformed}")

        if has_malformed:
            logger.info(
                f"[{session_id}] Detected malformed tool call in stream, transforming..."
            )

            try:
                parsed = ToolCallParser.parse(content)
                tool_calls_total = len(parsed.tool_calls)

                if parsed.was_transformed:
                    tool_calls_fixed = len(parsed.tool_calls)

                    # Generate transformed chunks
                    transformed_chunks = (
                        ResponseTransformer.transform_streaming_content(
                            content, parsed, metadata
                        )
                    )

                    for chunk in transformed_chunks:
                        yield f"data: {json.dumps(chunk)}\n\n"
                        await asyncio.sleep(
                            0.005
                        )  # Small delay for realistic streaming

                    yield "data: [DONE]\n\n"
                    logger.info(
                        f"[{session_id}] Stream transformation successful: "
                        f"{len(parsed.tool_calls)} tool call(s)"
                    )
                    stats.record_transformed()

                    # Track session stats
                    await session_tracker.track_request(
                        request=body,
                        client_ip=client_ip,
                        tool_calls_in_response=tool_calls_total,
                        tool_calls_fixed=tool_calls_fixed,
                        tool_calls_failed=tool_calls_failed,
                        prompt_tokens=stream_result.prompt_tokens,
                        completion_tokens=stream_result.completion_tokens,
                        total_tokens=stream_result.total_tokens,
                    )

                    # Log chat messages if enabled
                    if config.log_messages:
                        await session_tracker.add_request_messages(
                            session_id,
                            body.get("messages", []),
                            prompt_tokens=stream_result.prompt_tokens,
                        )
                        # Log transformed response with tool calls
                        tool_calls_data = [
                            {
                                "id": ResponseTransformer.generate_tool_call_id(),
                                "type": "function",
                                "function": {
                                    "name": tc.function_name,
                                    "arguments": json.dumps(tc.arguments),
                                },
                            }
                            for tc in parsed.tool_calls
                        ]
                        await session_tracker.add_message(
                            session_id=session_id,
                            direction="response",
                            role="assistant",
                            content=parsed.preamble or "",
                            tool_calls=tool_calls_data,
                            raw_content=content,  # Original content before transformation
                            debug={"role": "assistant", "content": content},
                            prompt_tokens=stream_result.prompt_tokens,
                        )
                    return
                else:
                    logger.info(
                        f"[{session_id}] Parser returned was_transformed=False, passing through"
                    )
                    stats.record_passthrough()

            except Exception as e:
                logger.error(f"[{session_id}] Stream transformation failed: {e}")
                tool_calls_failed = tool_calls_total
                stats.record_failed()
                # Fall through to replay original
        else:
            # No transformation needed - model output was valid (or no tool calls)
            # Count any tool calls that were already properly formatted in the stream
            tool_calls_total = len(stream_tool_calls)
            stats.record_passthrough()
            logger.info(
                f"[{session_id}] Stream passthrough: no malformed tool calls detected, "
                f"{tool_calls_total} valid tool call(s)"
            )

        # No transformation needed or failed - replay original chunks
        for chunk in original_chunks:
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

        # Track session stats
        await session_tracker.track_request(
            request=body,
            client_ip=client_ip,
            tool_calls_in_response=tool_calls_total,
            tool_calls_fixed=tool_calls_fixed,
            tool_calls_failed=tool_calls_failed,
            prompt_tokens=stream_result.prompt_tokens,
            completion_tokens=stream_result.completion_tokens,
            total_tokens=stream_result.total_tokens,
        )

        # Log chat messages if enabled
        if config.log_messages:
            await session_tracker.add_request_messages(
                session_id,
                body.get("messages", []),
                prompt_tokens=stream_result.prompt_tokens,
            )
            # Log response (passthrough content with any tool calls from stream)
            debug_data: dict[str, Any] = {"role": "assistant", "content": content}
            if stream_tool_calls:
                debug_data["tool_calls"] = stream_tool_calls
            await session_tracker.add_message(
                session_id=session_id,
                direction="response",
                role="assistant",
                content=content,
                tool_calls=stream_tool_calls if stream_tool_calls else None,
                debug=debug_data,
                prompt_tokens=stream_result.prompt_tokens,
            )

        logger.info(
            f"[{session_id}] Stream complete. Stats: total={stats.total_requests}, "
            f"passthrough={stats.passthrough_requests}, transformed={stats.transformed_requests}, "
            f"prompt_tokens={stream_result.prompt_tokens}, "
            f"completion_tokens={stream_result.completion_tokens}"
        )


# =============================================================================
# API Endpoints
# =============================================================================


def get_client_ip(request: Request) -> str | None:
    """
    Extract client IP from request, handling proxies.

    Checks X-Forwarded-For header first (for clients behind load balancers),
    then falls back to direct client IP.
    """
    # Check X-Forwarded-For header (comma-separated list, first is original client)
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        # Take the first IP in the chain (original client)
        return forwarded_for.split(",")[0].strip()

    # Check X-Real-IP header (some proxies use this)
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()

    # Fall back to direct client IP
    if request.client:
        return request.client.host

    return None


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request) -> Response | dict[str, Any]:
    """OpenAI-compatible chat completions with transformation."""

    body = await request.json()
    is_streaming = body.get("stream", False)

    # Extract client IP and session ID for tracking
    client_ip = get_client_ip(request)
    session_id = session_tracker.get_session_id(body, client_ip)

    # Apply proxy-level sampling parameters (and log them)
    body = apply_sampling_params(body)

    logger.info(
        f"[{session_id}] Request: streaming={is_streaming}, client_ip={client_ip}"
    )

    if is_streaming:
        # For streaming, the generator manages its own client lifecycle and session tracking
        return StreamingResponse(
            handle_streaming_request(body, session_id, client_ip),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Toolbridge-Session": session_id,
            },
        )
    else:
        # For non-streaming, we can use context manager normally
        async with httpx.AsyncClient() as client:
            request_result = await handle_non_streaming_request(
                client, body, session_id
            )

            # Track session stats
            await session_tracker.track_request(
                request=body,
                client_ip=client_ip,
                tool_calls_in_response=request_result.tool_calls_total,
                tool_calls_fixed=request_result.tool_calls_fixed,
                tool_calls_failed=request_result.tool_calls_failed,
                prompt_tokens=request_result.prompt_tokens,
                completion_tokens=request_result.completion_tokens,
                total_tokens=request_result.total_tokens,
            )

            # Log chat messages if enabled
            if config.log_messages:
                # Log request messages
                await session_tracker.add_request_messages(
                    session_id,
                    body.get("messages", []),
                    prompt_tokens=request_result.prompt_tokens,
                )

                # Log response message
                response_data = request_result.response
                if "choices" in response_data and response_data["choices"]:
                    response_msg = response_data["choices"][0].get("message", {})
                    await session_tracker.add_message(
                        session_id=session_id,
                        direction="response",
                        role=response_msg.get("role", "assistant"),
                        content=response_msg.get("content", ""),
                        tool_calls=response_msg.get("tool_calls"),
                        raw_content=request_result.raw_content,  # Original content if transformed
                        debug=response_msg,  # Original JSON payload
                        prompt_tokens=request_result.prompt_tokens,
                    )

            logger.info(
                f"[{session_id}] Request complete. Stats: "
                f"total={stats.total_requests}, passthrough={stats.passthrough_requests}, "
                f"transformed={stats.transformed_requests}, "
                f"prompt_tokens={request_result.prompt_tokens}, "
                f"completion_tokens={request_result.completion_tokens}"
            )

            return request_result.response


@app.get("/v1/models")
async def list_models() -> object:
    """Proxy models endpoint."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{config.backend_url}/v1/models")
        return response.json()


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{config.backend_url}/health", timeout=5.0)
            backend_healthy = response.status_code == 200
    except Exception:
        backend_healthy = False

    return {
        "status": "healthy" if backend_healthy else "degraded",
        "backend_url": config.backend_url,
        "backend_healthy": backend_healthy,
        "transform_enabled": config.transform_enabled,
    }


@app.get("/stats")
async def get_stats() -> dict[str, Any]:
    """
    Get proxy statistics.

    Shows how many requests were:
    - passthrough: Model produced valid output, no transformation needed
    - transformed: Proxy fixed malformed tool calls
    - failed: Transformation was attempted but failed
    - backend_errors: Backend returned an error
    """
    return {
        "proxy_stats": stats.to_dict(),
        "raw_counts": {
            "total": stats.total_requests,
            "passthrough": stats.passthrough_requests,
            "transformed": stats.transformed_requests,
            "failed": stats.failed_transforms,
            "backend_errors": stats.backend_errors,
        },
        "sampling_overrides": {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "min_p": config.min_p,
            "repeat_penalty": config.repeat_penalty,
            "presence_penalty": config.presence_penalty,
            "frequency_penalty": config.frequency_penalty,
        },
        "interpretation": {
            "passthrough": "Model output was already valid - proxy just passed it through",
            "transformed": "Model output was malformed - proxy fixed it",
            "if_all_passthrough": "Great! The model is producing valid tool calls on its own",
            "if_zero_total": "No requests recorded yet - check if requests are reaching the proxy",
        },
    }


@app.get("/admin/sessions")
async def get_sessions(
    include_archived: bool = Query(
        default=False,
        description="Include archived sessions in the response",
    ),
    archived_only: bool = Query(
        default=False,
        description="Only return archived sessions (ignores active sessions)",
    ),
) -> dict[str, Any]:
    """
    Return sessions and their stats.

    Useful for debugging and monitoring per-session behavior.
    Active sessions expire after 1 hour of inactivity.

    Query parameters:
    - include_archived: Include archived sessions alongside active ones
    - archived_only: Only show archived sessions (ignores active sessions)
    """
    result: dict[str, Any] = {
        "session_timeout_seconds": session_tracker.session_timeout,
        "sessions": {},
    }

    # Get active sessions (unless archived_only)
    if not archived_only:
        sessions = await session_tracker.get_all_sessions()
        result["active_sessions"] = len(sessions)
        for sid, session_stats in sessions.items():
            result["sessions"][sid] = {
                "created_at": session_stats.created_at,
                "last_seen_at": session_stats.last_seen_at,
                "age_seconds": round(time.time() - session_stats.created_at, 1),
                "idle_seconds": round(time.time() - session_stats.last_seen_at, 1),
                "request_count": session_stats.request_count,
                "tool_calls_total": session_stats.tool_calls_total,
                "tool_calls_fixed": session_stats.tool_calls_fixed,
                "tool_calls_failed": session_stats.tool_calls_failed,
                "fix_rate": round(
                    session_stats.tool_calls_fixed
                    / max(session_stats.tool_calls_total, 1),
                    3,
                ),
                "client_ip": session_stats.client_ip,
                # Token usage - last request (shows current context window size)
                "last_prompt_tokens": session_stats.last_prompt_tokens,
                "last_completion_tokens": session_stats.last_completion_tokens,
                "last_total_tokens": session_stats.last_total_tokens,
                # Token usage - cumulative (shows total session cost)
                "prompt_tokens_total": session_stats.prompt_tokens_total,
                "completion_tokens_total": session_stats.completion_tokens_total,
                "total_tokens_total": session_stats.total_tokens_total,
                "is_archived": False,
            }
    else:
        result["active_sessions"] = 0

    # Get archived sessions if requested
    if (include_archived or archived_only) and session_archive is not None:
        archived_summaries = await session_archive.list_sessions(limit=1000)
        result["archived_sessions"] = len(archived_summaries)

        for summary in archived_summaries:
            # Skip if already in active sessions (shouldn't happen, but be safe)
            if summary.session_id in result["sessions"]:
                continue

            result["sessions"][summary.session_id] = {
                "created_at": summary.created_at,
                "last_seen_at": summary.last_seen_at,
                "archived_at": summary.archived_at,
                "age_seconds": round(time.time() - summary.created_at, 1),
                "idle_seconds": round(time.time() - summary.last_seen_at, 1),
                "request_count": summary.request_count,
                "tool_calls_total": summary.tool_calls_total,
                "tool_calls_fixed": summary.tool_calls_fixed,
                "tool_calls_failed": summary.tool_calls_failed,
                "fix_rate": round(
                    summary.tool_calls_fixed / max(summary.tool_calls_total, 1),
                    3,
                ),
                "client_ip": summary.client_ip,
                # Archived sessions don't have last_ token values
                "last_prompt_tokens": 0,
                "last_completion_tokens": 0,
                "last_total_tokens": 0,
                # Token usage - cumulative
                "prompt_tokens_total": summary.prompt_tokens_total,
                "completion_tokens_total": summary.completion_tokens_total,
                "total_tokens_total": summary.total_tokens_total,
                "is_archived": True,
            }
    elif include_archived or archived_only:
        result["archived_sessions"] = 0
        result["archive_note"] = "Archiving is disabled"

    return result


@app.get("/admin/sessions/{session_id}", response_model=None)
async def get_session(
    session_id: str,
    include_messages: bool = True,
    message_limit: int = 100,
) -> dict[str, Any] | JSONResponse:
    """
    Get stats for a specific session.

    Checks active sessions first, then falls back to archive if not found.

    Args:
        session_id: The session ID to look up
        include_messages: Whether to include chat messages (default: True)
        message_limit: Maximum number of recent messages to return (default: 100)
    """
    # First, check active sessions
    session_stats = await session_tracker.get_session_stats(session_id)

    if session_stats:
        # Active session found
        result: dict[str, Any] = {
            "session_id": session_id,
            "created_at": session_stats.created_at,
            "last_seen_at": session_stats.last_seen_at,
            "age_seconds": round(time.time() - session_stats.created_at, 1),
            "idle_seconds": round(time.time() - session_stats.last_seen_at, 1),
            "request_count": session_stats.request_count,
            "tool_calls_total": session_stats.tool_calls_total,
            "tool_calls_fixed": session_stats.tool_calls_fixed,
            "tool_calls_failed": session_stats.tool_calls_failed,
            "fix_rate": round(
                session_stats.tool_calls_fixed / max(session_stats.tool_calls_total, 1), 3
            ),
            "client_ip": session_stats.client_ip,
            # Token usage - last request (shows current context window size)
            "last_prompt_tokens": session_stats.last_prompt_tokens,
            "last_completion_tokens": session_stats.last_completion_tokens,
            "last_total_tokens": session_stats.last_total_tokens,
            # Token usage - cumulative (shows total session cost)
            "prompt_tokens_total": session_stats.prompt_tokens_total,
            "completion_tokens_total": session_stats.completion_tokens_total,
            "total_tokens_total": session_stats.total_tokens_total,
            "is_archived": False,
        }

        if include_messages:
            # Get the most recent messages (up to message_limit)
            messages = list(session_stats.messages)[-message_limit:]
            result["messages"] = [
                {
                    "timestamp": msg.timestamp,
                    "direction": msg.direction,
                    "role": msg.role,
                    "content": msg.content,
                    "tool_calls": msg.tool_calls,
                    "raw_content": msg.raw_content,
                    "debug": msg.debug,
                    "prompt_tokens": msg.prompt_tokens,
                }
                for msg in messages
            ]
            result["total_messages"] = len(session_stats.messages)
            result["message_buffer_size"] = config.message_buffer_size

        return result

    # Not in active sessions, check archive
    if session_archive is not None:
        archived_data = await session_archive.get_session(session_id)
        if archived_data:
            result = {
                "session_id": session_id,
                "created_at": archived_data.get("created_at", 0),
                "last_seen_at": archived_data.get("last_seen_at", 0),
                "archived_at": archived_data.get("archived_at", 0),
                "age_seconds": round(
                    time.time() - archived_data.get("created_at", time.time()), 1
                ),
                "idle_seconds": round(
                    time.time() - archived_data.get("last_seen_at", time.time()), 1
                ),
                "request_count": archived_data.get("request_count", 0),
                "tool_calls_total": archived_data.get("tool_calls_total", 0),
                "tool_calls_fixed": archived_data.get("tool_calls_fixed", 0),
                "tool_calls_failed": archived_data.get("tool_calls_failed", 0),
                "fix_rate": round(
                    archived_data.get("tool_calls_fixed", 0)
                    / max(archived_data.get("tool_calls_total", 1), 1),
                    3,
                ),
                "client_ip": archived_data.get("client_ip"),
                # Token usage - last request (archived sessions store these)
                "last_prompt_tokens": archived_data.get("last_prompt_tokens", 0),
                "last_completion_tokens": archived_data.get("last_completion_tokens", 0),
                "last_total_tokens": archived_data.get("last_total_tokens", 0),
                # Token usage - cumulative
                "prompt_tokens_total": archived_data.get("prompt_tokens_total", 0),
                "completion_tokens_total": archived_data.get("completion_tokens_total", 0),
                "total_tokens_total": archived_data.get("total_tokens_total", 0),
                "is_archived": True,
            }

            if include_messages:
                messages = archived_data.get("messages", [])[-message_limit:]
                result["messages"] = messages
                result["total_messages"] = len(archived_data.get("messages", []))
                result["message_buffer_size"] = config.message_buffer_size

            return result

    return JSONResponse(
        status_code=404,
        content={"error": "Session not found", "session_id": session_id}
    )


@app.get("/config")
async def get_config() -> dict[str, Any]:
    """Get current proxy configuration."""
    archive_config: dict[str, Any] = {
        "enabled": config.archive_enabled,
        "ttl_hours": config.archive_ttl_hours,
    }
    if session_archive is not None:
        archive_config["directory"] = str(session_archive.archive_dir)
    else:
        archive_config["directory"] = None

    return {
        "backend_url": config.backend_url,
        "transform_enabled": config.transform_enabled,
        "log_sampling_params": config.log_sampling_params,
        "message_logging": {
            "enabled": config.log_messages,
            "buffer_size": config.message_buffer_size,
        },
        "archive": archive_config,
        "sampling_overrides": {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "min_p": config.min_p,
            "repeat_penalty": config.repeat_penalty,
            "presence_penalty": config.presence_penalty,
            "frequency_penalty": config.frequency_penalty,
        },
        "note": "Sampling overrides: null means client passes through, set value overrides",
    }


@app.post("/stats/reset")
async def reset_stats() -> dict[str, Any]:
    """Reset statistics counters."""
    global stats
    stats = ProxyStats()
    return {"status": "reset", "stats": stats.to_dict()}


@app.get("/proxy/test-transform")
async def test_transform(content: str = "") -> dict[str, Any]:
    """
    Test the transformation logic.

    Usage: /proxy/test-transform?content=<url-encoded-content>
    """
    if not content:
        content = """I'll help you create that file.
<function=writeFile>
<parameter=path>src/app.js</parameter>
<parameter=content>console.log("hello")</parameter>
</function>"""

    has_malformed = ToolCallParser.has_malformed_tool_call(content)
    parsed = ToolCallParser.parse(content)

    mock_response = {
        "id": "test-123",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }

    if parsed.was_transformed:
        transformed = ResponseTransformer.transform_response(mock_response, parsed)
    else:
        transformed = mock_response

    return {
        "input_content": content,
        "has_malformed_tool_call": has_malformed,
        "parsed": {
            "preamble": parsed.preamble,
            "tool_calls": [
                {"function_name": tc.function_name, "arguments": tc.arguments}
                for tc in parsed.tool_calls
            ],
            "postamble": parsed.postamble,
            "was_transformed": parsed.was_transformed,
        },
        "transformed_response": transformed,
    }


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def passthrough(request: Request, path: str) -> Response:
    """Pass through other endpoints unchanged."""
    async with httpx.AsyncClient() as client:
        url = f"{config.backend_url}/{path}"
        response = await client.request(
            method=request.method,
            url=url,
            headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
            content=await request.body()
            if request.method in ["POST", "PUT", "PATCH"]
            else None,
            params=request.query_params,
            timeout=120.0,
        )
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=dict(response.headers),
        )


# =============================================================================
# Main
# =============================================================================


def _env_help(env_var: str, description: str, default: str | None = None) -> str:
    """Format help text with environment variable name."""
    if default is not None:
        return f"{description} [env: {env_var}, default: {default}]"
    return f"{description} [env: {env_var}]"


def main() -> None:
    # Load config from environment variables / .env file first
    global config
    config = load_config()

    parser = argparse.ArgumentParser(
        description="LLM Response Transformer Proxy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configuration Priority (highest to lowest):
  1. CLI arguments
  2. Environment variables (TOOLBRIDGE_*)
  3. .env file in working directory
  4. Default values

Environment Variables:
  TOOLBRIDGE_BACKEND_URL       Backend server URL
  TOOLBRIDGE_PORT              Port to listen on
  TOOLBRIDGE_HOST              Host to bind to
  TOOLBRIDGE_TRANSFORM_ENABLED Enable transformation (true/false)
  TOOLBRIDGE_DEBUG             Enable debug logging (true/false)
  TOOLBRIDGE_TEMPERATURE       Override temperature
  TOOLBRIDGE_TOP_P             Override top_p
  TOOLBRIDGE_TOP_K             Override top_k
  TOOLBRIDGE_MIN_P             Override min_p
  TOOLBRIDGE_REPEAT_PENALTY    Override repeat_penalty
  TOOLBRIDGE_PRESENCE_PENALTY  Override presence_penalty
  TOOLBRIDGE_FREQUENCY_PENALTY Override frequency_penalty
  TOOLBRIDGE_LOG_SAMPLING_PARAMS  Log sampling params (true/false)
  TOOLBRIDGE_LOG_MESSAGES      Enable message logging (true/false)
  TOOLBRIDGE_MESSAGE_BUFFER_SIZE  Max messages per session
  TOOLBRIDGE_CORS_ENABLED      Enable CORS (true/false)
  TOOLBRIDGE_CORS_ORIGINS      Comma-separated allowed origins (as JSON list)
  TOOLBRIDGE_CORS_ALL_ROUTES   Enable CORS for all routes (true/false)
  TOOLBRIDGE_ARCHIVE_ENABLED   Enable session archiving (true/false, default: true)
  TOOLBRIDGE_ARCHIVE_DIR       Archive directory (default: XDG state dir)
  TOOLBRIDGE_ARCHIVE_TTL_HOURS Hours to retain archives (default: 168 = 7 days)

Examples:
  # Basic usage (reads from env vars / .env if available)
  python toolbridge.py

  # Override backend URL via CLI
  python toolbridge.py --backend http://localhost:8080

  # With sampling overrides (force specific values)
  python toolbridge.py --temperature 0.7 --repeat-penalty 1.0

  # Using environment variables
  TOOLBRIDGE_BACKEND_URL=http://localhost:8080 python toolbridge.py
""",
    )

    # Connection settings - use None as default to detect if CLI was used
    parser.add_argument(
        "--backend",
        "-b",
        default=None,
        help=_env_help(
            "TOOLBRIDGE_BACKEND_URL",
            "Backend llama.cpp server URL",
            "http://localhost:8080",
        ),
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=None,
        help=_env_help("TOOLBRIDGE_PORT", "Port to listen on", "4000"),
    )
    parser.add_argument(
        "--host",
        default=None,
        help=_env_help("TOOLBRIDGE_HOST", "Host to bind to", "0.0.0.0"),
    )

    # Transform settings
    parser.add_argument(
        "--no-transform",
        action="store_true",
        help="Disable transformation (passthrough mode) "
        "[env: TOOLBRIDGE_TRANSFORM_ENABLED=false]",
    )

    # Sampling parameters (optional overrides)
    sampling = parser.add_argument_group("sampling parameters (optional overrides)")
    sampling.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=_env_help("TOOLBRIDGE_TEMPERATURE", "Override temperature (e.g., 0.7)"),
    )
    sampling.add_argument(
        "--top-p",
        type=float,
        default=None,
        help=_env_help("TOOLBRIDGE_TOP_P", "Override top_p (e.g., 0.9)"),
    )
    sampling.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=_env_help("TOOLBRIDGE_TOP_K", "Override top_k (e.g., 40)"),
    )
    sampling.add_argument(
        "--min-p",
        type=float,
        default=None,
        help=_env_help("TOOLBRIDGE_MIN_P", "Override min_p (e.g., 0.05)"),
    )
    sampling.add_argument(
        "--repeat-penalty",
        type=float,
        default=None,
        help=_env_help("TOOLBRIDGE_REPEAT_PENALTY", "Override repeat_penalty (e.g., 1.0)"),
    )
    sampling.add_argument(
        "--presence-penalty",
        type=float,
        default=None,
        help=_env_help("TOOLBRIDGE_PRESENCE_PENALTY", "Override presence_penalty"),
    )
    sampling.add_argument(
        "--frequency-penalty",
        type=float,
        default=None,
        help=_env_help("TOOLBRIDGE_FREQUENCY_PENALTY", "Override frequency_penalty"),
    )

    # Debug settings
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging [env: TOOLBRIDGE_DEBUG=true]",
    )
    parser.add_argument(
        "--no-log-params",
        action="store_true",
        help="Disable logging of sampling parameters "
        "[env: TOOLBRIDGE_LOG_SAMPLING_PARAMS=false]",
    )

    # Message logging settings
    parser.add_argument(
        "--no-log-messages",
        action="store_true",
        help="Disable chat message logging in sessions "
        "[env: TOOLBRIDGE_LOG_MESSAGES=false]",
    )
    parser.add_argument(
        "--message-buffer-size",
        type=int,
        default=None,
        help=_env_help(
            "TOOLBRIDGE_MESSAGE_BUFFER_SIZE",
            "Max messages per session in circular buffer",
            "1024",
        ),
    )

    # CORS settings
    cors_group = parser.add_argument_group(
        "CORS settings (for admin API access from browsers)"
    )
    cors_group.add_argument(
        "--cors",
        action="store_true",
        help="Enable CORS for /admin/* endpoints [env: TOOLBRIDGE_CORS_ENABLED=true]",
    )
    cors_group.add_argument(
        "--cors-origins",
        type=str,
        default=None,
        help="Comma-separated allowed origins (default: * if --cors enabled). "
        'Example: --cors-origins "http://localhost:5173,https://admin.example.com" '
        "[env: TOOLBRIDGE_CORS_ORIGINS as JSON array]",
    )
    cors_group.add_argument(
        "--cors-all",
        action="store_true",
        help="Enable CORS for ALL routes including /v1/* proxy endpoints "
        "(use with caution) [env: TOOLBRIDGE_CORS_ALL_ROUTES=true]",
    )

    # Archive settings
    archive_group = parser.add_argument_group(
        "archive settings (for expired session persistence)"
    )
    archive_group.add_argument(
        "--no-archive",
        action="store_true",
        help="Disable session archiving [env: TOOLBRIDGE_ARCHIVE_ENABLED=false]",
    )
    archive_group.add_argument(
        "--archive-dir",
        type=str,
        default=None,
        help=_env_help(
            "TOOLBRIDGE_ARCHIVE_DIR",
            "Directory for session archives",
            "${XDG_STATE_HOME:-~/.local/state}/toolbridge/archives",
        ),
    )
    archive_group.add_argument(
        "--archive-ttl",
        type=int,
        default=None,
        help=_env_help(
            "TOOLBRIDGE_ARCHIVE_TTL_HOURS",
            "Hours to retain archived sessions (0 = forever)",
            "168",
        ),
    )

    args = parser.parse_args()

    # Apply CLI overrides - only if explicitly provided (not None)
    # CLI takes precedence over environment variables
    if args.backend is not None:
        config.backend_url = args.backend
    if args.port is not None:
        config.port = args.port
    if args.host is not None:
        config.host = args.host

    # Boolean flags - CLI flags override env vars when explicitly set
    if args.no_transform:
        config.transform_enabled = False

    # Sampling parameters - only override if CLI provided
    if args.temperature is not None:
        config.temperature = args.temperature
    if args.top_p is not None:
        config.top_p = args.top_p
    if args.top_k is not None:
        config.top_k = args.top_k
    if args.min_p is not None:
        config.min_p = args.min_p
    if args.repeat_penalty is not None:
        config.repeat_penalty = args.repeat_penalty
    if args.presence_penalty is not None:
        config.presence_penalty = args.presence_penalty
    if args.frequency_penalty is not None:
        config.frequency_penalty = args.frequency_penalty

    # Debug/logging settings
    if args.no_log_params:
        config.log_sampling_params = False
    if args.no_log_messages:
        config.log_messages = False
    if args.message_buffer_size is not None:
        config.message_buffer_size = args.message_buffer_size

    # CORS settings - CLI flags override env vars
    if args.cors or args.cors_all:
        config.cors_enabled = True
    if args.cors_all:
        config.cors_all_routes = True
    if args.cors_origins:
        config.cors_origins = [o.strip() for o in args.cors_origins.split(",")]

    # Archive settings - CLI flags override env vars
    if args.no_archive:
        config.archive_enabled = False
    if args.archive_dir is not None:
        config.archive_dir = args.archive_dir
    if args.archive_ttl is not None:
        config.archive_ttl_hours = args.archive_ttl

    # Debug mode - can be set via CLI or env var
    if args.debug:
        config.debug = True

    # Initialize session archive if enabled
    global session_archive
    if config.archive_enabled:
        archive_dir = get_archive_dir(config.archive_dir)
        session_archive = SessionArchive(
            archive_dir=archive_dir,
            archive_ttl_hours=config.archive_ttl_hours,
        )
    else:
        session_archive = None

    # Reinitialize session tracker with configured buffer size and archive
    global session_tracker
    session_tracker = SessionTracker(
        session_timeout=3600,
        message_buffer_size=config.message_buffer_size,
        archive=session_archive,
    )

    # Apply debug logging level
    if config.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Setup CORS middleware if enabled
    if config.cors_enabled:
        from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
        from starlette.responses import Response as StarletteResponse

        origins = config.cors_origins if config.cors_origins else ["*"]

        if config.cors_all_routes:
            # CORS on all routes
            app.add_middleware(
                CORSMiddleware,
                allow_origins=origins,
                allow_credentials=True,
                allow_methods=["GET", "POST", "OPTIONS"],
                allow_headers=["*"],
            )
        else:
            # CORS only on /admin/* routes - use custom middleware
            class AdminOnlyCORSMiddleware(BaseHTTPMiddleware):
                async def dispatch(
                    self, request: Request, call_next: RequestResponseEndpoint
                ) -> StarletteResponse:
                    # Only apply CORS to admin endpoints
                    if not request.url.path.startswith("/admin"):
                        return await call_next(request)

                    # Handle preflight
                    if request.method == "OPTIONS":
                        origin = request.headers.get("origin", "")
                        if "*" in origins or origin in origins:
                            return StarletteResponse(
                                status_code=204,
                                headers={
                                    "Access-Control-Allow-Origin": origin or "*",
                                    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                                    "Access-Control-Allow-Headers": "*",
                                    "Access-Control-Allow-Credentials": "true",
                                },
                            )
                        return await call_next(request)

                    # Add CORS headers to response
                    response = await call_next(request)
                    origin = request.headers.get("origin", "")
                    if "*" in origins or origin in origins:
                        response.headers["Access-Control-Allow-Origin"] = origin or "*"
                        response.headers["Access-Control-Allow-Credentials"] = "true"
                    return response

            app.add_middleware(AdminOnlyCORSMiddleware)

    # Log startup configuration
    logger.info("Starting LLM Response Transformer Proxy")
    logger.info(f"  Backend: {config.backend_url}")
    logger.info(f"  Listening: {config.host}:{config.port}")
    logger.info(f"  Transform: {'enabled' if config.transform_enabled else 'disabled'}")
    if config.log_messages:
        logger.info(f"  Message logging: enabled (buffer size: {config.message_buffer_size})")
    else:
        logger.info("  Message logging: disabled")

    # Log sampling overrides if any are set
    overrides = []
    if config.temperature is not None:
        overrides.append(f"temperature={config.temperature}")
    if config.top_p is not None:
        overrides.append(f"top_p={config.top_p}")
    if config.top_k is not None:
        overrides.append(f"top_k={config.top_k}")
    if config.min_p is not None:
        overrides.append(f"min_p={config.min_p}")
    if config.repeat_penalty is not None:
        overrides.append(f"repeat_penalty={config.repeat_penalty}")
    if config.presence_penalty is not None:
        overrides.append(f"presence_penalty={config.presence_penalty}")
    if config.frequency_penalty is not None:
        overrides.append(f"frequency_penalty={config.frequency_penalty}")

    if overrides:
        logger.info(f"  Sampling overrides: {', '.join(overrides)}")
    else:
        logger.info("  Sampling overrides: none (using client values)")

    # Log CORS configuration
    if config.cors_enabled:
        origins_display = ", ".join(config.cors_origins) if config.cors_origins else "*"
        routes_scope = "all routes" if config.cors_all_routes else "admin endpoints only"
        logger.info(f"  CORS: enabled ({routes_scope})")
        logger.info(f"  CORS origins: {origins_display}")
    else:
        logger.info("  CORS: disabled")

    # Log archive configuration
    if config.archive_enabled and session_archive is not None:
        ttl_display = (
            f"{config.archive_ttl_hours} hours"
            if config.archive_ttl_hours > 0
            else "forever"
        )
        logger.info(f"  Session archive: enabled (TTL: {ttl_display})")
        logger.info(f"  Archive directory: {session_archive.archive_dir}")
    else:
        logger.info("  Session archive: disabled")

    import uvicorn

    uvicorn.run(app, host=config.host, port=config.port, log_level="info")


if __name__ == "__main__":
    main()
