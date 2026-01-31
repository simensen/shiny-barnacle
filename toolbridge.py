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
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Optional
from xml.sax.saxutils import unescape as xml_unescape

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

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

    def record_passthrough(self):
        self.total_requests += 1
        self.passthrough_requests += 1

    def record_transformed(self):
        self.total_requests += 1
        self.transformed_requests += 1

    def record_failed(self):
        self.total_requests += 1
        self.failed_transforms += 1

    def record_backend_error(self):
        self.total_requests += 1
        self.backend_errors += 1

    def to_dict(self) -> dict:
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
    tool_calls: list[dict] | None = None  # If response contains tool calls
    raw_content: str | None = None  # Original content before transformation (if transformed)


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
    messages: deque = field(default_factory=deque)  # Circular buffer of ChatMessage


class SessionTracker:
    """
    Track sessions with automatic expiry.

    Sessions are identified by hashing the stable portion of the conversation
    (first 2 messages: typically system prompt + first user message).
    """

    def __init__(self, session_timeout: int = 3600, message_buffer_size: int = 1024):
        """
        Args:
            session_timeout: Seconds of inactivity before session expires (default 1 hour)
            message_buffer_size: Max messages per session circular buffer (default 1024)
        """
        self._sessions: dict[str, SessionStats] = {}
        self._lock = asyncio.Lock()
        self.session_timeout = session_timeout
        self.message_buffer_size = message_buffer_size

    def get_session_id(self, request: dict, client_ip: str | None = None) -> str:
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
        request: dict,
        client_ip: str | None = None,
        tool_calls_in_response: int = 0,
        tool_calls_fixed: int = 0,
        tool_calls_failed: int = 0,
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
        tool_calls: list[dict] | None = None,
        raw_content: str | None = None,
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
                )
                session.messages.append(message)

    async def get_all_sessions(self) -> dict[str, SessionStats]:
        """Get all active sessions (excludes expired)."""
        await self._cleanup_expired()
        async with self._lock:
            return dict(self._sessions)

    async def _cleanup_expired(self) -> int:
        """Remove expired sessions. Returns count of removed sessions."""
        now = time.time()
        removed = 0

        async with self._lock:
            expired_ids = [
                sid
                for sid, stats in self._sessions.items()
                if now - stats.last_seen_at > self.session_timeout
            ]
            for sid in expired_ids:
                del self._sessions[sid]
                removed += 1

        return removed


# Global session tracker instance
session_tracker = SessionTracker(session_timeout=3600)


@dataclass
class ProxyConfig:
    """Proxy configuration settings"""

    backend_url: str = "http://localhost:8080"
    port: int = 4000
    host: str = "0.0.0.0"

    # Transformation settings
    transform_enabled: bool = True
    fallback_to_retry: bool = True  # If transform fails, try retry
    max_retries: int = 1  # Only used if transform fails

    # Streaming settings
    stream_buffer_timeout: float = 120.0

    # Sampling parameters (None means use client's value or backend default)
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repeat_penalty: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None

    # Debug settings
    log_sampling_params: bool = True  # Log sampling params for each request

    # Message logging settings
    log_messages: bool = True  # Enable message logging in sessions
    message_buffer_size: int = 1024  # Max messages per session (circular buffer)


config = ProxyConfig()
app = FastAPI(title="LLM Response Transformer Proxy")


def apply_sampling_params(body: dict) -> dict:
    """Apply proxy-level sampling parameters to request body.

    Proxy params override client params when set.
    Returns the modified body and logs the effective params.
    """
    # All known sampling-related parameter names (OpenAI + llama.cpp extensions)
    ALL_SAMPLING_PARAMS = {
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
        k: v for k, v in body.items() if k in ALL_SAMPLING_PARAMS and v is not None
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
        skip_keys = ALL_SAMPLING_PARAMS | {
            "messages",
            "tools",
            "tool_choice",
            "functions",
        }
        other_keys = {k: body.get(k) for k in body.keys() if k not in skip_keys}

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
                f"Sampling params - Client sent: {client_params}, Proxy overrides: {overrides}, Effective: {effective}"
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
    arguments: dict
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
        if cls.JSON_TOOL_CALL_PATTERN.search(content):
            return True

        return False

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
    def _parse_parameters(cls, body: str) -> dict:
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
    def _parse_json_tool_calls(cls, content: str, matches: list) -> ParsedResponse:
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
        cls, original_response: dict, parsed: ParsedResponse
    ) -> dict:
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

        response = json.loads(json.dumps(original_response))  # Deep copy

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
        cls, full_content: str, parsed: ParsedResponse, metadata: dict
    ) -> list[dict]:
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

    response: dict
    tool_calls_total: int = 0
    tool_calls_fixed: int = 0
    tool_calls_failed: int = 0
    raw_content: str | None = None  # Original content before transformation (if transformed)


async def handle_non_streaming_request(
    client: httpx.AsyncClient, body: dict, session_id: str
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
                    f"[{session_id}] Transformation successful: {len(parsed.tool_calls)} tool call(s)"
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

    return RequestResult(
        response=result,
        tool_calls_total=tool_calls_total,
        tool_calls_fixed=tool_calls_fixed,
        tool_calls_failed=tool_calls_failed,
        raw_content=raw_content,
    )


@dataclass
class StreamCollectionResult:
    """Result from collecting a streaming response."""

    content: str
    chunks: list[dict]
    metadata: dict
    tool_calls: list[dict]  # Reconstructed tool calls from stream


async def collect_stream(
    client: httpx.AsyncClient, body: dict
) -> StreamCollectionResult:
    """Collect streaming response into content, chunks, and tool calls."""

    full_content = ""
    chunks = []
    metadata = {}
    # Track tool calls being built from stream chunks
    # Key: tool call index, Value: dict with id, type, function (name, arguments)
    tool_calls_builder: dict[int, dict] = {}

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
    )


async def handle_streaming_request(
    body: dict, session_id: str, client_ip: str | None
) -> AsyncGenerator[str, None]:
    """Handle streaming request with transformation."""

    logger.info(f"[{session_id}] Processing streaming request")

    body["stream"] = True

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
            # Log request messages if enabled (no response on error)
            if config.log_messages:
                for msg in body.get("messages", []):
                    await session_tracker.add_message(
                        session_id=session_id,
                        direction="request",
                        role=msg.get("role", "unknown"),
                        content=msg.get("content", ""),
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
            # Log request messages if enabled (no response on error)
            if config.log_messages:
                for msg in body.get("messages", []):
                    await session_tracker.add_message(
                        session_id=session_id,
                        direction="request",
                        role=msg.get("role", "unknown"),
                        content=msg.get("content", ""),
                    )
            error_chunk = {"error": {"message": str(e), "type": "proxy_error"}}
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Log what we collected for debugging
        logger.info(
            f"[{session_id}] Stream collected: {len(original_chunks)} chunks, {len(content)} chars content"
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
                        f"[{session_id}] Stream transformation successful: {len(parsed.tool_calls)} tool call(s)"
                    )
                    stats.record_transformed()

                    # Track session stats
                    await session_tracker.track_request(
                        request=body,
                        client_ip=client_ip,
                        tool_calls_in_response=tool_calls_total,
                        tool_calls_fixed=tool_calls_fixed,
                        tool_calls_failed=tool_calls_failed,
                    )

                    # Log chat messages if enabled
                    if config.log_messages:
                        for msg in body.get("messages", []):
                            await session_tracker.add_message(
                                session_id=session_id,
                                direction="request",
                                role=msg.get("role", "unknown"),
                                content=msg.get("content", ""),
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
                f"[{session_id}] Stream passthrough: no malformed tool calls detected, {tool_calls_total} valid tool call(s)"
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
        )

        # Log chat messages if enabled
        if config.log_messages:
            for msg in body.get("messages", []):
                await session_tracker.add_message(
                    session_id=session_id,
                    direction="request",
                    role=msg.get("role", "unknown"),
                    content=msg.get("content", ""),
                )
            # Log response (passthrough content with any tool calls from stream)
            await session_tracker.add_message(
                session_id=session_id,
                direction="response",
                role="assistant",
                content=content,
                tool_calls=stream_tool_calls if stream_tool_calls else None,
            )

        logger.info(
            f"[{session_id}] Stream complete. Stats: total={stats.total_requests}, passthrough={stats.passthrough_requests}, transformed={stats.transformed_requests}"
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


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
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
            )

            # Log chat messages if enabled
            if config.log_messages:
                # Log request messages
                for msg in body.get("messages", []):
                    await session_tracker.add_message(
                        session_id=session_id,
                        direction="request",
                        role=msg.get("role", "unknown"),
                        content=msg.get("content", ""),
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
                    )

            return request_result.response


@app.get("/v1/models")
async def list_models():
    """Proxy models endpoint."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{config.backend_url}/v1/models")
        return response.json()


@app.get("/health")
async def health_check():
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
async def get_stats():
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
async def get_sessions():
    """
    Return all active sessions and their stats.

    Useful for debugging and monitoring per-session behavior.
    Sessions expire after 1 hour of inactivity.
    """
    sessions = await session_tracker.get_all_sessions()
    return {
        "active_sessions": len(sessions),
        "session_timeout_seconds": session_tracker.session_timeout,
        "sessions": {
            sid: {
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
            }
            for sid, session_stats in sessions.items()
        },
    }


@app.get("/admin/sessions/{session_id}")
async def get_session(
    session_id: str,
    include_messages: bool = True,
    message_limit: int = 100,
):
    """
    Get stats for a specific session.

    Args:
        session_id: The session ID to look up
        include_messages: Whether to include chat messages (default: True)
        message_limit: Maximum number of recent messages to return (default: 100)
    """
    session_stats = await session_tracker.get_session_stats(session_id)
    if not session_stats:
        return {"error": "Session not found", "session_id": session_id}

    result = {
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
                "raw_content": msg.raw_content,  # Original content before transformation
            }
            for msg in messages
        ]
        result["total_messages"] = len(session_stats.messages)
        result["message_buffer_size"] = config.message_buffer_size

    return result


@app.get("/config")
async def get_config():
    """Get current proxy configuration."""
    return {
        "backend_url": config.backend_url,
        "transform_enabled": config.transform_enabled,
        "log_sampling_params": config.log_sampling_params,
        "message_logging": {
            "enabled": config.log_messages,
            "buffer_size": config.message_buffer_size,
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
        "note": "Sampling overrides: null means client value passes through, set value overrides client",
    }


@app.post("/stats/reset")
async def reset_stats():
    """Reset statistics counters."""
    global stats
    stats = ProxyStats()
    return {"status": "reset", "stats": stats.to_dict()}


@app.get("/proxy/test-transform")
async def test_transform(content: str = ""):
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
async def passthrough(request: Request, path: str):
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


def main():
    parser = argparse.ArgumentParser(
        description="LLM Response Transformer Proxy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Sampling Parameters:
  When specified, these override whatever the client sends.
  If not specified, client values pass through to backend.

Examples:
  # Basic usage
  python toolbridge.py --backend http://localhost:8080

  # With sampling overrides (force specific values)
  python toolbridge.py --temperature 0.7 --repeat-penalty 1.0

  # Just log what client sends, don't override
  python toolbridge.py --debug
""",
    )

    # Connection settings
    parser.add_argument(
        "--backend",
        "-b",
        default="http://localhost:8080",
        help="Backend llama.cpp server URL (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--port", "-p", type=int, default=4000, help="Port to listen on (default: 4000)"
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )

    # Transform settings
    parser.add_argument(
        "--no-transform",
        action="store_true",
        help="Disable transformation (passthrough mode)",
    )

    # Sampling parameters (optional overrides)
    sampling = parser.add_argument_group("sampling parameters (optional overrides)")
    sampling.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override temperature (e.g., 0.7)",
    )
    sampling.add_argument(
        "--top-p", type=float, default=None, help="Override top_p (e.g., 0.9)"
    )
    sampling.add_argument(
        "--top-k", type=int, default=None, help="Override top_k (e.g., 40)"
    )
    sampling.add_argument(
        "--min-p", type=float, default=None, help="Override min_p (e.g., 0.05)"
    )
    sampling.add_argument(
        "--repeat-penalty",
        type=float,
        default=None,
        help="Override repeat_penalty (e.g., 1.0)",
    )
    sampling.add_argument(
        "--presence-penalty", type=float, default=None, help="Override presence_penalty"
    )
    sampling.add_argument(
        "--frequency-penalty",
        type=float,
        default=None,
        help="Override frequency_penalty",
    )

    # Debug settings
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--no-log-params",
        action="store_true",
        help="Disable logging of sampling parameters",
    )

    # Message logging settings
    parser.add_argument(
        "--no-log-messages",
        action="store_true",
        help="Disable chat message logging in sessions",
    )
    parser.add_argument(
        "--message-buffer-size",
        type=int,
        default=1024,
        help="Max messages per session in circular buffer (default: 1024)",
    )

    args = parser.parse_args()

    # Apply configuration
    config.backend_url = args.backend
    config.port = args.port
    config.host = args.host
    config.transform_enabled = not args.no_transform

    # Sampling parameters
    config.temperature = args.temperature
    config.top_p = args.top_p
    config.top_k = args.top_k
    config.min_p = args.min_p
    config.repeat_penalty = args.repeat_penalty
    config.presence_penalty = args.presence_penalty
    config.frequency_penalty = args.frequency_penalty
    config.log_sampling_params = not args.no_log_params

    # Message logging settings
    config.log_messages = not args.no_log_messages
    config.message_buffer_size = args.message_buffer_size

    # Reinitialize session tracker with new buffer size
    global session_tracker
    session_tracker = SessionTracker(
        session_timeout=3600,
        message_buffer_size=config.message_buffer_size,
    )

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Log startup configuration
    logger.info(f"Starting LLM Response Transformer Proxy")
    logger.info(f"  Backend: {config.backend_url}")
    logger.info(f"  Listening: {config.host}:{config.port}")
    logger.info(f"  Transform: {'enabled' if config.transform_enabled else 'disabled'}")
    if config.log_messages:
        logger.info(f"  Message logging: enabled (buffer size: {config.message_buffer_size})")
    else:
        logger.info(f"  Message logging: disabled")

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
        logger.info(f"  Sampling overrides: none (using client values)")

    import uvicorn

    uvicorn.run(app, host=config.host, port=config.port, log_level="info")


if __name__ == "__main__":
    main()
