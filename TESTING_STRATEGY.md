# Testing Strategy Research: Toolbridge

## Executive Summary

This document outlines a comprehensive testing strategy for the Toolbridge project (LLM Response Transformer Proxy). The project currently has **no automated tests** despite having complex transformation logic that is critical to its function.

## Current State

- **Language/Framework**: Python 3.12 / FastAPI
- **Main Module**: `toolbridge.py` (1,889 lines)
- **Existing Test Infrastructure**: None
- **Dev Tools Available**: pytest, black, ruff, mypy (via flake.nix)

## What Needs Testing

### 1. Core Transformation Logic (Highest Priority)

The `ToolCallParser` and `ResponseTransformer` classes are the heart of this project and must be thoroughly tested.

| Component | Function | Test Priority |
|-----------|----------|---------------|
| `ToolCallParser.has_malformed_tool_call()` | Detects if content needs transformation | Critical |
| `ToolCallParser.parse()` | Extracts tool calls from malformed content | Critical |
| `ToolCallParser._parse_parameters()` | Parses parameter tags | Critical |
| `ResponseTransformer.transform_response()` | Converts parsed calls to OpenAI format | Critical |
| `ResponseTransformer.transform_streaming_content()` | Handles streaming transformation | Critical |

### 2. API Endpoints (High Priority)

| Endpoint | Method | Test Focus |
|----------|--------|------------|
| `/v1/chat/completions` | POST | Main proxy endpoint - streaming and non-streaming |
| `/v1/models` | GET | Passthrough to backend |
| `/admin/sessions` | GET | Session listing |
| `/admin/sessions/{id}` | GET | Session details with messages |
| `/health` | GET | Health check behavior |
| `/stats` | GET | Statistics reporting |
| `/config` | GET | Configuration exposure |
| `/proxy/test-transform` | GET | Transformation testing |

### 3. Session Management (Medium Priority)

- Session ID generation from request hashes
- Session expiry after timeout
- Message buffer (circular buffer behavior)
- Concurrent session access (async locks)

### 4. Configuration & Utilities (Medium Priority)

- `apply_sampling_params()` - Parameter override logic
- `get_client_ip()` - IP extraction from various headers
- CLI argument parsing

---

## Recommended Testing Stack

### Primary Tools

```
pytest              # Test framework
pytest-asyncio      # Async test support (required for FastAPI)
pytest-cov          # Coverage reporting
httpx               # Already a dependency - use for API testing
```

### Optional Enhancements

```
pytest-mock         # Mocking utilities
respx               # Mock httpx requests
hypothesis          # Property-based testing (good for parser edge cases)
```

### Updated requirements-dev.txt (suggested)

```
pytest>=8.0.0
pytest-asyncio>=0.23.0
pytest-cov>=4.1.0
respx>=0.20.0
```

---

## Proposed Test Structure

```
tests/
├── conftest.py                  # Shared fixtures
├── unit/
│   ├── test_parser.py           # ToolCallParser unit tests
│   ├── test_transformer.py      # ResponseTransformer unit tests
│   ├── test_stats.py            # ProxyStats unit tests
│   ├── test_session.py          # SessionTracker unit tests
│   └── test_config.py           # Configuration and utilities
├── integration/
│   ├── test_endpoints.py        # API endpoint tests
│   ├── test_streaming.py        # Streaming response tests
│   └── test_passthrough.py      # Backend passthrough tests
└── fixtures/
    ├── malformed_responses.py   # Sample malformed LLM outputs
    └── expected_transforms.py   # Expected transformation results
```

---

## Test Cases by Component

### ToolCallParser Tests

```python
# tests/unit/test_parser.py

class TestHasMalformedToolCall:
    """Test detection of malformed tool calls."""

    def test_empty_content_returns_false(self):
        assert ToolCallParser.has_malformed_tool_call("") is False
        assert ToolCallParser.has_malformed_tool_call(None) is False

    def test_plain_text_returns_false(self):
        assert ToolCallParser.has_malformed_tool_call("Hello world") is False

    def test_detects_function_equals_format(self):
        content = "<function=writeFile><parameter=path>test.txt</parameter></function>"
        assert ToolCallParser.has_malformed_tool_call(content) is True

    def test_detects_function_name_attribute_format(self):
        content = '<function name="writeFile"><parameter=path>test.txt</parameter></function>'
        assert ToolCallParser.has_malformed_tool_call(content) is True

    def test_valid_tool_call_wrapper_returns_false(self):
        content = "<tool_call>...</tool_call>"
        assert ToolCallParser.has_malformed_tool_call(content) is False

    def test_json_style_tool_call(self):
        content = '{"name": "writeFile", "arguments": {"path": "test.txt"}}'
        assert ToolCallParser.has_malformed_tool_call(content) is True


class TestParse:
    """Test parsing of malformed tool calls."""

    def test_parse_single_function_call(self):
        content = """<function=writeFile>
<parameter=path>src/app.js</parameter>
<parameter=content>console.log("hello")</parameter>
</function>"""
        result = ToolCallParser.parse(content)

        assert result.was_transformed is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function_name == "writeFile"
        assert result.tool_calls[0].arguments["path"] == "src/app.js"

    def test_parse_with_preamble(self):
        content = """I'll help you create that file.
<function=writeFile>
<parameter=path>test.txt</parameter>
</function>"""
        result = ToolCallParser.parse(content)

        assert result.preamble == "I'll help you create that file."
        assert len(result.tool_calls) == 1

    def test_parse_multiple_function_calls(self):
        content = """<function=readFile><parameter=path>a.txt</parameter></function>
<function=writeFile><parameter=path>b.txt</parameter></function>"""
        result = ToolCallParser.parse(content)

        assert len(result.tool_calls) == 2

    def test_parse_xml_escaped_content(self):
        content = """<function=writeFile>
<parameter=content>&lt;div&gt;Hello&lt;/div&gt;</parameter>
</function>"""
        result = ToolCallParser.parse(content)

        assert result.tool_calls[0].arguments["content"] == "<div>Hello</div>"

    def test_parse_json_parameter_values(self):
        content = """<function=test>
<parameter=config>{"key": "value"}</parameter>
<parameter=count>42</parameter>
<parameter=enabled>true</parameter>
</function>"""
        result = ToolCallParser.parse(content)

        assert result.tool_calls[0].arguments["config"] == {"key": "value"}
        assert result.tool_calls[0].arguments["count"] == 42
        # Note: "true" may be parsed as string or boolean depending on implementation
```

### ResponseTransformer Tests

```python
# tests/unit/test_transformer.py

class TestTransformResponse:
    """Test response transformation to OpenAI format."""

    def test_transform_adds_tool_calls_array(self):
        original = {
            "id": "test-123",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "..."},
                "finish_reason": "stop"
            }]
        }
        parsed = ParsedResponse(
            preamble="",
            tool_calls=[ParsedToolCall("writeFile", {"path": "test.txt"}, "...")],
            postamble="",
            was_transformed=True
        )

        result = ResponseTransformer.transform_response(original, parsed)

        assert "tool_calls" in result["choices"][0]["message"]
        assert len(result["choices"][0]["message"]["tool_calls"]) == 1
        assert result["choices"][0]["finish_reason"] == "tool_calls"

    def test_transform_sets_content_to_preamble(self):
        # ... test that preamble becomes content
        pass

    def test_transform_generates_unique_tool_call_ids(self):
        # ... test ID uniqueness
        pass

    def test_transform_preserves_original_on_no_transformation(self):
        # ... test passthrough behavior
        pass
```

### API Endpoint Tests

```python
# tests/integration/test_endpoints.py
import pytest
from httpx import AsyncClient, ASGITransport
from toolbridge import app

@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_status(self, client):
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "backend_healthy" in data


class TestStatsEndpoint:
    @pytest.mark.asyncio
    async def test_stats_returns_counts(self, client):
        response = await client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "proxy_stats" in data
        assert "raw_counts" in data


class TestAdminSessions:
    @pytest.mark.asyncio
    async def test_sessions_list_empty(self, client):
        response = await client.get("/admin/sessions")
        assert response.status_code == 200
        data = response.json()
        assert "active_sessions" in data
        assert "sessions" in data

    @pytest.mark.asyncio
    async def test_session_not_found(self, client):
        response = await client.get("/admin/sessions/nonexistent")
        assert response.status_code == 404


class TestTransformEndpoint:
    @pytest.mark.asyncio
    async def test_transform_with_default_content(self, client):
        response = await client.get("/proxy/test-transform")
        assert response.status_code == 200
        data = response.json()
        assert data["has_malformed_tool_call"] is True
        assert data["parsed"]["was_transformed"] is True
```

### Session Tracker Tests

```python
# tests/unit/test_session.py
import pytest
import asyncio
from toolbridge import SessionTracker, SessionStats

class TestSessionTracker:
    @pytest.fixture
    def tracker(self):
        return SessionTracker(session_timeout=60, message_buffer_size=10)

    @pytest.mark.asyncio
    async def test_generates_consistent_session_id(self, tracker):
        request = {"messages": [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"}
        ]}

        id1 = tracker.get_session_id(request)
        id2 = tracker.get_session_id(request)

        assert id1 == id2

    @pytest.mark.asyncio
    async def test_different_requests_get_different_ids(self, tracker):
        req1 = {"messages": [{"role": "user", "content": "Hello"}]}
        req2 = {"messages": [{"role": "user", "content": "Goodbye"}]}

        id1 = tracker.get_session_id(req1)
        id2 = tracker.get_session_id(req2)

        assert id1 != id2

    @pytest.mark.asyncio
    async def test_track_request_creates_session(self, tracker):
        request = {"messages": [{"role": "user", "content": "Test"}]}
        session_id = await tracker.track_request(request, client_ip="127.0.0.1")

        stats = await tracker.get_session_stats(session_id)
        assert stats is not None
        assert stats.request_count == 1

    @pytest.mark.asyncio
    async def test_message_buffer_is_circular(self, tracker):
        request = {"messages": [{"role": "user", "content": "Test"}]}
        session_id = await tracker.track_request(request)

        # Add more messages than buffer size
        for i in range(15):
            await tracker.add_message(session_id, "request", "user", f"Message {i}")

        stats = await tracker.get_session_stats(session_id)
        assert len(stats.messages) == 10  # Buffer size
```

---

## Configuration Files

### pytest.ini

```ini
[pytest]
asyncio_mode = auto
testpaths = tests
python_files = test_*.py
python_functions = test_*
addopts = -v --tb=short
filterwarnings =
    ignore::DeprecationWarning
```

### pyproject.toml (alternative)

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = "-v --tb=short"

[tool.coverage.run]
source = ["toolbridge"]
omit = ["tests/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "if __name__ == .__main__.:",
    "raise NotImplementedError",
]
```

---

## Mock Strategy for Backend Requests

Since the proxy communicates with a backend LLM server, tests need to mock these HTTP calls:

```python
# tests/conftest.py
import pytest
import respx
from httpx import Response

@pytest.fixture
def mock_backend():
    """Mock the backend LLM server."""
    with respx.mock(base_url="http://localhost:8080") as respx_mock:
        yield respx_mock


@pytest.fixture
def mock_chat_completion(mock_backend):
    """Setup a mock chat completion response."""
    def _mock(content: str, tool_calls: list = None):
        response_data = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop"
            }]
        }
        if tool_calls:
            response_data["choices"][0]["message"]["tool_calls"] = tool_calls
            response_data["choices"][0]["finish_reason"] = "tool_calls"

        mock_backend.post("/v1/chat/completions").respond(
            json=response_data
        )
    return _mock
```

---

## Test Data: Malformed Response Examples

```python
# tests/fixtures/malformed_responses.py

# Format 1: function=name (most common from Qwen3-Coder)
MALFORMED_WRITE_FILE = """I'll create that file for you.
<function=writeFile>
<parameter=path>src/components/Button.tsx</parameter>
<parameter=content>export const Button = () => <button>Click me</button></parameter>
</function>"""

# Format 2: function name="name"
MALFORMED_NAME_ATTRIBUTE = """<function name="readFile">
<parameter name="path">config.json</parameter>
</function>"""

# Format 3: Multiple tool calls
MALFORMED_MULTIPLE = """<function=readFile><parameter=path>a.txt</parameter></function>
<function=writeFile><parameter=path>b.txt</parameter><parameter=content>data</parameter></function>"""

# Format 4: JSON-style
MALFORMED_JSON = '{"name": "executeCommand", "arguments": {"command": "npm install"}}'

# Valid format (should pass through)
VALID_TOOL_CALL = """<tool_call>{"name": "writeFile", "arguments": {"path": "test.txt"}}</tool_call>"""
```

---

## Coverage Goals

| Component | Target Coverage |
|-----------|-----------------|
| `ToolCallParser` | 95%+ |
| `ResponseTransformer` | 95%+ |
| `SessionTracker` | 90%+ |
| API Endpoints | 85%+ |
| CLI/Config | 70%+ |
| **Overall** | **85%+** |

---

## Implementation Roadmap

### Phase 1: Foundation (Start Here)
1. Create test directory structure
2. Add pytest configuration (`pytest.ini` or `pyproject.toml`)
3. Add test dependencies to `requirements-dev.txt`
4. Create `conftest.py` with basic fixtures

### Phase 2: Unit Tests
1. `test_parser.py` - ToolCallParser tests
2. `test_transformer.py` - ResponseTransformer tests
3. `test_stats.py` - ProxyStats tests
4. `test_session.py` - SessionTracker tests

### Phase 3: Integration Tests
1. `test_endpoints.py` - API endpoint tests
2. `test_streaming.py` - Streaming response tests
3. Backend mocking with respx

### Phase 4: Edge Cases & Regression
1. Add malformed response fixtures
2. Property-based testing with hypothesis
3. Regression test suite for reported bugs

---

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=toolbridge --cov-report=html

# Run specific test file
pytest tests/unit/test_parser.py

# Run with verbose output
pytest -v

# Run only tests matching pattern
pytest -k "test_parse"

# Run tests in parallel (requires pytest-xdist)
pytest -n auto
```

---

## Integration with Nix

Update `flake.nix` to include test command:

```nix
# Add to shellHook
shellHook = ''
  echo "  pytest              - Run tests"
  echo "  pytest --cov        - Run tests with coverage"
'';

# Add test script to packages
packages.test = pkgs.writeShellScriptBin "run-tests" ''
  cd ${./.}
  ${pythonEnv}/bin/pytest "$@"
'';
```

---

## Notes

- The codebase uses `asyncio` extensively, so `pytest-asyncio` is essential
- Mock the backend HTTP calls to avoid needing a running LLM server
- Focus on transformation logic first - it's the core value proposition
- Consider adding snapshot/golden tests for transformation outputs
