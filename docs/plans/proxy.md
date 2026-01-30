# Toolbridge Session Tracking Implementation

## Overview

Toolbridge is a proxy that sits between LLM clients (Cursor, Zed, JetBrains AI, opencode, droid/factory.ai, etc.) and llama.cpp's OpenAI-compatible API. It transforms malformed XML-style tool calls into properly structured JSON `tool_calls` format.

This document describes how to implement session tracking to correlate requests from the same conversation, enabling per-session statistics, debugging, and adaptive behavior.

## The Problem

The OpenAI `/v1/chat/completions` API is stateless by design:
- No session ID field in requests
- No conversation ID field
- The optional `user` field is rarely populated by clients
- Clients send full conversation history with every request

We need to infer session identity from request content.

## Solution: Stable Prefix Hashing

Since clients send the full message history with each request, we can identify a session by hashing the **stable portion** of the conversation - typically the system prompt and first user message, which remain constant throughout a conversation.

### Why This Works

```
Request 1: [system, user1]
  → hash([system, user1]) → "xyz789"

Request 2: [system, user1, asst1, user2]  
  → hash([system, user1]) → "xyz789"  ✓ Same session

Request 3: [system, user1, asst1, user2, asst2, user3]
  → hash([system, user1]) → "xyz789"  ✓ Same session

Request 4: [system, user1, ..., user4]
  → hash([system, user1]) → "xyz789"  ✓ Same session
```

### Why NOT Hash the Entire Prefix

A previous approach considered hashing `messages[:-1]` (everything except the latest message). This fails because every request produces a different hash:

```
Request 1: hash([system]) → "aaa"
Request 2: hash([system, user1, asst1]) → "bbb"  ✗ Different!
Request 3: hash([system, user1, asst1, user2, asst2]) → "ccc"  ✗ Different!
```

## Implementation

### Core Session ID Function

```python
import hashlib
import json
from typing import Any

def get_session_id(request: dict[str, Any]) -> str:
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
    # These are typically [system_prompt, first_user_message]
    stable_messages = messages[:2]
    
    # Normalize for consistent hashing
    # Only include role and content, ignore other fields like 'name'
    normalized = [
        {"role": msg.get("role"), "content": msg.get("content", "")}
        for msg in stable_messages
    ]
    
    content = json.dumps(normalized, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]
```

### Session Tracker Class

```python
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock

@dataclass
class SessionStats:
    created_at: float = field(default_factory=time.time)
    last_seen_at: float = field(default_factory=time.time)
    request_count: int = 0
    tool_calls_total: int = 0
    tool_calls_fixed: int = 0
    tool_calls_failed: int = 0
    client_ip: str | None = None

class SessionTracker:
    def __init__(self, session_timeout: int = 3600):
        """
        Track sessions with automatic expiry.
        
        Args:
            session_timeout: Seconds of inactivity before session expires (default 1 hour)
        """
        self._sessions: dict[str, SessionStats] = {}
        self._lock = Lock()
        self.session_timeout = session_timeout
    
    def get_session_id(self, request: dict, client_ip: str | None = None) -> str:
        """Generate session ID from request."""
        messages = request.get("messages", [])
        
        if request.get("user"):
            return f"user_{request['user']}"
        
        stable_messages = messages[:2]
        normalized = [
            {"role": msg.get("role"), "content": msg.get("content", "")}
            for msg in stable_messages
        ]
        content = json.dumps(normalized, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def track_request(
        self,
        request: dict,
        client_ip: str | None = None,
        tool_calls_in_response: int = 0,
        tool_calls_fixed: int = 0,
        tool_calls_failed: int = 0
    ) -> str:
        """
        Track a request and return its session ID.
        
        Call this after processing each request to update session stats.
        """
        session_id = self.get_session_id(request, client_ip)
        
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionStats(client_ip=client_ip)
            
            stats = self._sessions[session_id]
            stats.last_seen_at = time.time()
            stats.request_count += 1
            stats.tool_calls_total += tool_calls_in_response
            stats.tool_calls_fixed += tool_calls_fixed
            stats.tool_calls_failed += tool_calls_failed
            
            # Update client IP if not set (or if it changed - unusual but possible)
            if client_ip and not stats.client_ip:
                stats.client_ip = client_ip
        
        return session_id
    
    def get_session_stats(self, session_id: str) -> SessionStats | None:
        """Get stats for a specific session."""
        with self._lock:
            return self._sessions.get(session_id)
    
    def get_all_sessions(self) -> dict[str, SessionStats]:
        """Get all active sessions (excludes expired)."""
        self._cleanup_expired()
        with self._lock:
            return dict(self._sessions)
    
    def _cleanup_expired(self) -> int:
        """Remove expired sessions. Returns count of removed sessions."""
        now = time.time()
        removed = 0
        
        with self._lock:
            expired_ids = [
                sid for sid, stats in self._sessions.items()
                if now - stats.last_seen_at > self.session_timeout
            ]
            for sid in expired_ids:
                del self._sessions[sid]
                removed += 1
        
        return removed
```

### Integration with Request Handler

```python
# Global tracker instance
session_tracker = SessionTracker(session_timeout=3600)

async def handle_chat_completion(request: dict, client_ip: str) -> dict:
    """Example request handler showing session tracking integration."""
    
    # Get session ID before processing (useful for logging)
    session_id = session_tracker.get_session_id(request, client_ip)
    
    # Process the request through llama.cpp
    response = await forward_to_llama_cpp(request)
    
    # Transform XML tool calls to JSON if needed
    fixed_count = 0
    failed_count = 0
    tool_calls = extract_tool_calls(response)
    
    if needs_transformation(response):
        transformed, fixed_count, failed_count = transform_tool_calls(response)
        response = transformed
    
    # Track the request
    session_tracker.track_request(
        request=request,
        client_ip=client_ip,
        tool_calls_in_response=len(tool_calls),
        tool_calls_fixed=fixed_count,
        tool_calls_failed=failed_count
    )
    
    # Optionally add session ID to response headers for debugging
    # response_headers["X-Toolbridge-Session"] = session_id
    
    return response
```

## Edge Cases and Considerations

### Potential Collisions

| Scenario | Impact | Mitigation |
|----------|--------|------------|
| Same system prompt + same first user message across different actual sessions | Sessions incorrectly merged | Acceptable for stats; consider adding client IP to hash if problematic |
| Dynamic system prompts (include timestamps, etc.) | Each request = new session | Hash only the user message, or extract stable portion of system prompt |
| No system prompt | Less uniqueness | Still works if first user message is unique |
| Very short first messages ("hi", "help") | Higher collision rate | Consider hashing first 3 messages, or accept the collision |

### Multi-User Considerations

When deployed for multiple users on a network:
- Sessions from different users with identical starting prompts will collide
- If this is problematic, include client IP in the hash:

```python
def get_session_id_with_ip(request: dict, client_ip: str) -> str:
    messages = request.get("messages", [])
    stable_messages = messages[:2]
    normalized = [
        {"role": msg.get("role"), "content": msg.get("content", "")}
        for msg in stable_messages
    ]
    # Include client IP for user separation
    content = json.dumps(normalized, sort_keys=True) + f"|ip:{client_ip}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]
```

### Recommended Defaults

- **Session timeout**: 1 hour (3600 seconds)
- **Hash length**: 16 hex characters (64 bits, sufficient for uniqueness)
- **Stable prefix**: First 2 messages
- **Include client IP**: Yes, for multi-user deployments

## Stats Endpoint (Optional)

Consider exposing session stats via an admin endpoint:

```python
@app.get("/admin/sessions")
async def get_sessions():
    """Return all active sessions and their stats."""
    sessions = session_tracker.get_all_sessions()
    return {
        "active_sessions": len(sessions),
        "sessions": {
            sid: {
                "created_at": stats.created_at,
                "last_seen_at": stats.last_seen_at,
                "request_count": stats.request_count,
                "tool_calls_total": stats.tool_calls_total,
                "tool_calls_fixed": stats.tool_calls_fixed,
                "tool_calls_failed": stats.tool_calls_failed,
                "fix_rate": stats.tool_calls_fixed / max(stats.tool_calls_total, 1),
                "client_ip": stats.client_ip
            }
            for sid, stats in sessions.items()
        }
    }
```

## Logging Recommendations

Include session ID in log messages for easier debugging:

```python
import logging

logger = logging.getLogger("toolbridge")

def log_request(session_id: str, request: dict, response: dict, fixes: int):
    logger.info(
        f"[{session_id}] "
        f"messages={len(request.get('messages', []))} "
        f"tool_calls={len(response.get('choices', [{}])[0].get('message', {}).get('tool_calls', []))} "
        f"fixed={fixes}"
    )
```

This allows filtering logs by session:
```bash
grep "abc123def456" toolbridge.log
```
