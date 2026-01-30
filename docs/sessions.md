# Toolbridge Session Tracking

Toolbridge automatically tracks sessions to correlate requests from the same conversation. This enables per-session statistics, debugging, and monitoring.

## How Sessions Work

### Session Identification

Sessions are identified by hashing the **stable portion** of each conversation - the first two messages (typically the system prompt and first user message). Since LLM clients send the full conversation history with every request, this stable prefix remains constant throughout a conversation:

```
Request 1: [system, user1]              → hash → "abc123..."
Request 2: [system, user1, asst1, user2] → hash → "abc123..." (same)
Request 3: [system, user1, ..., user5]   → hash → "abc123..." (same)
```

The session ID is a 16-character hex string derived from:
1. The `user` field if provided by the client (returned as `user_{value}`)
2. Otherwise, a SHA-256 hash of the first 2 messages + client IP

### Session Expiry

Sessions automatically expire after **1 hour of inactivity**. The `last_seen_at` timestamp updates with each request, so active conversations remain tracked indefinitely.

## API Endpoints

### List All Sessions

```
GET /admin/sessions
```

Returns all active sessions with their statistics.

**Example Response:**
```json
{
  "active_sessions": 3,
  "session_timeout_seconds": 3600,
  "sessions": {
    "a1b2c3d4e5f67890": {
      "created_at": 1706620800.123,
      "last_seen_at": 1706621400.456,
      "age_seconds": 600.3,
      "idle_seconds": 0.1,
      "request_count": 15,
      "tool_calls_total": 42,
      "tool_calls_fixed": 38,
      "tool_calls_failed": 0,
      "fix_rate": 0.905,
      "client_ip": "192.168.1.100"
    }
  }
}
```

**Field Descriptions:**

| Field | Description |
|-------|-------------|
| `active_sessions` | Total number of tracked sessions |
| `session_timeout_seconds` | Inactivity timeout (default 3600) |
| `created_at` | Unix timestamp when session started |
| `last_seen_at` | Unix timestamp of most recent request |
| `age_seconds` | Time since session creation |
| `idle_seconds` | Time since last request |
| `request_count` | Total requests in this session |
| `tool_calls_total` | Total tool calls seen in responses |
| `tool_calls_fixed` | Tool calls successfully transformed |
| `tool_calls_failed` | Tool calls that failed transformation |
| `fix_rate` | Ratio of fixed to total tool calls |
| `client_ip` | Client IP address (if detected) |

### Get Single Session

```
GET /admin/sessions/{session_id}
```

Returns statistics for a specific session.

**Example:**
```
GET /admin/sessions/a1b2c3d4e5f67890
```

**Response:**
```json
{
  "session_id": "a1b2c3d4e5f67890",
  "created_at": 1706620800.123,
  "last_seen_at": 1706621400.456,
  "age_seconds": 600.3,
  "idle_seconds": 0.1,
  "request_count": 15,
  "tool_calls_total": 42,
  "tool_calls_fixed": 38,
  "tool_calls_failed": 0,
  "fix_rate": 0.905,
  "client_ip": "192.168.1.100"
}
```

**Error Response (session not found):**
```json
{
  "error": "Session not found",
  "session_id": "nonexistent123"
}
```

## Response Headers

For streaming responses, Toolbridge includes the session ID in the response headers:

```
X-Toolbridge-Session: a1b2c3d4e5f67890
```

This allows clients to correlate their requests with session statistics.

## Filtering Logs by Session

All log messages include the session ID in brackets, enabling easy filtering:

```bash
# View all activity for a specific session
grep "\[a1b2c3d4e5f67890\]" toolbridge.log

# Watch live activity for a session
tail -f toolbridge.log | grep "\[a1b2c3d4e5f67890\]"
```

**Example log output:**
```
INFO: [a1b2c3d4e5f67890] Request: streaming=True, client_ip=192.168.1.100
INFO: [a1b2c3d4e5f67890] Detected malformed tool call in stream, transforming...
INFO: [a1b2c3d4e5f67890] Stream transformation successful: 2 tool call(s)
```

## Use Cases

### Debugging a Problematic Conversation

1. Get the session ID from logs or response headers
2. Query the session endpoint: `GET /admin/sessions/{session_id}`
3. Check `tool_calls_failed` to see if transformations are failing
4. Filter logs by session ID for detailed request/response flow

### Monitoring Transformation Effectiveness

1. Query all sessions: `GET /admin/sessions`
2. Look at aggregate `fix_rate` across sessions
3. Sessions with low fix rates may indicate new malformed patterns

### Identifying High-Volume Users

1. Query all sessions: `GET /admin/sessions`
2. Sort by `request_count` to find most active sessions
3. Use `client_ip` to identify the source

## Implementation Details

### Client IP Detection

Toolbridge extracts client IP from (in order of priority):
1. `X-Forwarded-For` header (for reverse proxy setups)
2. `X-Real-IP` header
3. Direct connection IP

### Session Storage

Sessions are stored in memory with automatic cleanup:
- New sessions are created on first request
- `last_seen_at` updates with each request
- Expired sessions are removed when `get_all_sessions()` is called
- No persistence across proxy restarts

### Thread Safety

Session tracking uses `asyncio.Lock` for thread-safe access in the async FastAPI environment.
