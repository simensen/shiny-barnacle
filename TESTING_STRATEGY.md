# Testing Strategy: Toolbridge

## Overview

This document outlines the testing strategy for the Toolbridge project (LLM Response Transformer Proxy), including migration to `uv` for dependency management and standardized developer workflows.

### Current State
- **Language/Framework**: Python 3.12 / FastAPI
- **Main Module**: `toolbridge.py` (single-file application)
- **Existing Tests**: None
- **Package Management**: requirements.txt (to be migrated to uv/pyproject.toml)

### Goals
1. Migrate from `requirements.txt` to `uv` + `pyproject.toml`
2. Add comprehensive test coverage
3. Standardize developer workflows with Makefile
4. Update `flake.nix` to use uv instead of pure Nix Python packages

---

## Part 1: Project Infrastructure

### pyproject.toml

```toml
[project]
name = "toolbridge"
version = "0.1.0"
description = "LLM Response Transformer Proxy - OpenAI-compatible proxy for tool call transformation"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "httpx>=0.25.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.1.0",
    "pytest-mock>=3.12.0",
    "respx>=0.20.0",
    "mypy>=1.8.0",
    "ruff>=0.1.0",
]

[project.scripts]
toolbridge = "toolbridge:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.mypy]
python_version = "3.12"
strict = true
warn_return_any = true
warn_unused_ignores = true

[[tool.mypy.overrides]]
module = [
    "uvicorn",
    "fastapi",
    "fastapi.*",
    "httpx",
    "respx",
]
ignore_missing_imports = true

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP", "B", "C4", "SIM"]

[tool.coverage.run]
source = ["."]
omit = ["tests/*", ".direnv/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "if __name__ == .__main__.:",
    "raise NotImplementedError",
]
```

### Makefile

```makefile
.PHONY: help install dev test lint format fix typecheck clean run

# Default target
help:
	@echo "Toolbridge - Development Commands"
	@echo ""
	@echo "Development:"
	@echo "  make install    Install production dependencies"
	@echo "  make dev        Install development dependencies"
	@echo "  make run        Run the proxy locally"
	@echo ""
	@echo "Quality:"
	@echo "  make test       Run tests"
	@echo "  make coverage   Run tests with coverage report"
	@echo "  make lint       Run linter (ruff)"
	@echo "  make format     Format code (ruff)"
	@echo "  make fix        Format code with unsafe fixes"
	@echo "  make typecheck  Run type checker (mypy)"
	@echo "  make check      Run all checks (lint + typecheck + test)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean      Remove build artifacts"

# Install production dependencies
install:
	uv pip install -e .

# Install development dependencies
dev:
	uv pip install -e ".[dev]"

# Run the proxy locally
run:
	python toolbridge.py

# Run tests
test:
	python -m pytest tests/ -v

# Run tests with coverage
coverage:
	python -m pytest tests/ -v --cov=. --cov-report=html --cov-report=term

# Run linter
lint:
	ruff check toolbridge.py tests/

# Format code
format:
	ruff check --fix toolbridge.py tests/
	ruff format toolbridge.py tests/

# Format code with unsafe fixes
fix:
	ruff check --fix --unsafe-fixes toolbridge.py tests/
	ruff format toolbridge.py tests/

# Run type checker
typecheck:
	mypy toolbridge.py

# Run all checks
check: lint typecheck test

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
```

### flake.nix (uv-based)

```nix
{
  description = "LLM Response Transformer Proxy - OpenAI-compatible proxy for tool call transformation";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          name = "toolbridge-dev";

          buildInputs = with pkgs; [
            python312
            uv
            curl
            jq
          ];

          shellHook = ''
            export UV_PYTHON="python3.12"
            export UV_PYTHON_PREFERENCE="only-system"
            export UV_PROJECT_ENVIRONMENT=".direnv/.venv"
            export UV_CACHE_DIR=".direnv/.cache/uv"

            # Create virtual environment with uv if it doesn't exist
            if [ ! -d "$UV_PROJECT_ENVIRONMENT" ]; then
              echo "Creating virtual environment with uv..."
              uv venv $UV_PROJECT_ENVIRONMENT
            fi

            # Activate virtual environment
            source $UV_PROJECT_ENVIRONMENT/bin/activate

            # Install dependencies from pyproject.toml if it exists
            if [ -f "pyproject.toml" ]; then
              echo "Installing dependencies from pyproject.toml with uv..."
              uv pip install -e ".[dev]" 2>/dev/null || uv pip install -e .
            fi

            if [ -t 1 ]; then
              echo ""
              echo "Toolbridge Development Environment"
              echo ""
              echo "Python: $(python --version) ($(which python))"
              echo "UV: $(uv --version)"
              echo "Virtual environment: $VIRTUAL_ENV"
              echo ""
              echo "Quick start:"
              echo "  make run        Run the proxy"
              echo "  make test       Run tests"
              echo "  make check      Run all checks"
              echo "  make help       Show all commands"
              echo ""
            fi
          '';

          PYTHONDONTWRITEBYTECODE = "1";
          PYTHONHASHSEED = "0";
        };
      }
    );
}
```

---

## Part 2: Test Structure

### Directory Layout

```
tests/
├── conftest.py              # Shared fixtures
├── unit/
│   ├── test_parser.py       # ToolCallParser unit tests
│   ├── test_transformer.py  # ResponseTransformer unit tests
│   ├── test_stats.py        # ProxyStats unit tests
│   └── test_session.py      # SessionTracker unit tests
├── integration/
│   ├── test_endpoints.py    # API endpoint tests
│   └── test_streaming.py    # Streaming response tests
└── fixtures/
    └── malformed_responses.py  # Sample malformed LLM outputs
```

### What to Test (Priority Order)

| Priority | Component | Focus |
|----------|-----------|-------|
| Critical | `ToolCallParser.has_malformed_tool_call()` | Detection accuracy |
| Critical | `ToolCallParser.parse()` | Parsing all malformed formats |
| Critical | `ResponseTransformer.transform_response()` | OpenAI format compliance |
| High | API endpoints | `/v1/chat/completions`, `/health`, `/stats` |
| High | Streaming transformation | SSE event handling |
| Medium | `SessionTracker` | Session lifecycle, message buffers |
| Medium | Configuration | CLI args, `apply_sampling_params()` |

---

## Part 3: Test Implementation

### conftest.py

```python
import pytest
import respx
from httpx import AsyncClient, ASGITransport

# Import the app - works because toolbridge.py is in the root
from toolbridge import app


@pytest.fixture
async def client():
    """Async HTTP client for testing FastAPI endpoints."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def mock_backend():
    """Mock the backend LLM server."""
    with respx.mock(base_url="http://localhost:8080") as respx_mock:
        yield respx_mock


@pytest.fixture
def mock_chat_completion(mock_backend):
    """Setup a mock chat completion response."""
    def _mock(content: str, model: str = "test-model"):
        response_data = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }]
        }
        mock_backend.post("/v1/chat/completions").respond(json=response_data)
    return _mock
```

### Unit Tests: Parser

```python
# tests/unit/test_parser.py
from toolbridge import ToolCallParser


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

        assert "I'll help you create that file" in result.preamble
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
```

### Integration Tests: Endpoints

```python
# tests/integration/test_endpoints.py
import pytest


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_status(self, client):
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data


class TestStatsEndpoint:
    @pytest.mark.asyncio
    async def test_stats_returns_counts(self, client):
        response = await client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "proxy_stats" in data


class TestAdminSessions:
    @pytest.mark.asyncio
    async def test_sessions_list(self, client):
        response = await client.get("/admin/sessions")
        assert response.status_code == 200
        data = response.json()
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
        assert "has_malformed_tool_call" in data
        assert "parsed" in data
```

### Test Fixtures: Malformed Responses

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

# Format 4: JSON-style (bare JSON object)
MALFORMED_JSON = '{"name": "executeCommand", "arguments": {"command": "npm install"}}'

# Valid format (should pass through unchanged)
VALID_TOOL_CALL = """<tool_call>{"name": "writeFile", "arguments": {"path": "test.txt"}}</tool_call>"""
```

---

## Part 4: Migration Steps

### Phase 1: Infrastructure Setup

1. **Create `pyproject.toml`** from template above
2. **Create `Makefile`** from template above
3. **Update `flake.nix`** to uv-based version
4. **Delete `requirements.txt`** (dependencies now in pyproject.toml)
5. **Add to `.gitignore`**:
   ```
   .direnv/
   htmlcov/
   .coverage
   ```

### Phase 2: Test Foundation

1. Create `tests/` directory structure
2. Create `tests/conftest.py` with shared fixtures
3. Create `tests/fixtures/malformed_responses.py`

### Phase 3: Unit Tests

1. `tests/unit/test_parser.py` - ToolCallParser tests
2. `tests/unit/test_transformer.py` - ResponseTransformer tests
3. `tests/unit/test_session.py` - SessionTracker tests

### Phase 4: Integration Tests

1. `tests/integration/test_endpoints.py` - API tests
2. `tests/integration/test_streaming.py` - SSE streaming tests

---

## Part 5: Coverage Goals

| Component | Target |
|-----------|--------|
| `ToolCallParser` | 95%+ |
| `ResponseTransformer` | 95%+ |
| `SessionTracker` | 90%+ |
| API Endpoints | 85%+ |
| **Overall** | **85%+** |

---

## Quick Reference

```bash
# Enter dev environment (with Nix)
nix develop

# Or manually with uv
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Run tests
make test

# Run with coverage
make coverage

# Run all quality checks
make check

# Format code
make format
```
