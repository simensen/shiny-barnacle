# LLM Retry Proxy

An OpenAI-compatible proxy that sits between your AI coding tools (bolt.diy, Cursor, Zed, etc.) and llama.cpp. It intercepts responses, detects malformed tool calls, and **transforms them into proper format** without expensive retries.

## Two Approaches

This project includes two proxy implementations:

| File | Approach | Latency | Token Cost |
|------|----------|---------|------------|
| `transform_proxy.py` | **Parse & transform** malformed output | ~0ms | 0 extra |
| `proxy.py` | Detect & **retry** with hints | 2-3x | 2-3x |

**Recommendation:** Use `transform_proxy.py` - it's faster and cheaper.

## The Problem This Solves

When using Qwen3-Coder and similar models with agentic coding tools, the model sometimes produces malformed tool calls like:

```
I'll help you create that file.
<function=writeFile>
<parameter=path>src/app.js</parameter>
<parameter=content>console.log("hello")</parameter>
</function>
```

When it should produce a response with proper `tool_calls` JSON:

```json
{
  "choices": [{
    "message": {
      "content": "I'll help you create that file.",
      "tool_calls": [{
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "writeFile",
          "arguments": "{\"path\":\"src/app.js\",\"content\":\"console.log(\\\"hello\\\")\"}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

The **transform proxy** parses the malformed XML and reconstructs it as proper OpenAI-compatible tool calls.

---

## Quick Start

### Option 1: Using Nix (Recommended)

If you have Nix with flakes enabled:

```bash
# Enter the development shell
nix develop

# Or with direnv (automatic on cd)
direnv allow

# Run the proxy
python transform_proxy.py --backend http://localhost:8080 --port 4000
```

Or run directly without entering the shell:

```bash
# Run transform proxy
nix run .#transform -- --backend http://localhost:8080 --port 4000

# Run retry proxy
nix run .#retry -- --backend http://localhost:8080 --port 4000
```

### Option 2: Using pip

```bash
# Install dependencies
pip install -r requirements.txt

# Run the proxy
python transform_proxy.py --backend http://localhost:8080 --port 4000
```

### Then...

```bash
# Start llama.cpp server (in another terminal)
llama-server -m your-model.gguf -c 16384 --jinja --port 8080

# Configure your client to use http://localhost:4000 as the API base URL
```

---

## Project Structure

```
llm-retry-proxy/
├── flake.nix              # Nix flake for reproducible environment
├── .envrc                 # direnv configuration (auto-activates nix shell)
├── .gitignore
├── requirements.txt       # pip dependencies (alternative to nix)
├── transform_proxy.py     # ✅ Recommended: Parse & transform approach
├── proxy.py               # Alternative: Detect & retry approach
├── test_transform.py      # Test suite for transformation logic
├── llm-proxy.service      # Systemd unit (system-wide)
├── llm-proxy.user.service # Systemd unit (user-level, no root)
└── README.md
```

### Nix Flake Features

The `flake.nix` provides:

| Command | Description |
|---------|-------------|
| `nix develop` | Enter dev shell with Python + all deps |
| `nix run .#transform` | Run transform proxy (single worker) |
| `nix run .#retry` | Run retry proxy (single worker) |
| `nix run .#production` | Run with gunicorn (4 workers default) |
| `nix build` | Build the package |

Environment variables for production:
```bash
WORKERS=4 PORT=4000 HOST=0.0.0.0 nix run .#production
```

The dev shell includes:
- Python 3.12
- fastapi, uvicorn, httpx
- Development tools: pytest, black, ruff, mypy
- CLI utilities: curl, jq

---

## How Transformation Works

The proxy parses various malformed formats:

```
Input formats supported:
├── <function=name>...</function>           (most common)
├── <function="name">...</function>         (quoted)
├── <function name="name">...</function>    (attribute style)
└── {"name": "...", "arguments": {...}}     (raw JSON)

Output format:
└── OpenAI tool_calls JSON structure
```

### Transformation Example

**Input (malformed):**
```
I'll create that file for you.
<function=writeFile>
<parameter=path>src/app.js</parameter>
<parameter=content>console.log("Hello!");</parameter>
</function>
```

**Output (proper OpenAI format):**
```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "I'll create that file for you.",
      "tool_calls": [{
        "id": "call_9406505917684901935ce4ee",
        "type": "function",
        "function": {
          "name": "writeFile",
          "arguments": "{\"path\": \"src/app.js\", \"content\": \"console.log(\\\"Hello!\\\");\"}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

### Test the Transformation

The proxy includes a test endpoint:

```bash
curl "http://localhost:4000/proxy/test-transform"
```

Or with custom content:

```bash
curl "http://localhost:4000/proxy/test-transform?content=<function=test><parameter=x>1</parameter></function>"
```

---

## Concurrency & Scaling

### Default Mode (Single Worker, Async I/O)

The proxy uses Python's `asyncio` for concurrent request handling:

```bash
python transform_proxy.py --port 4000
```

This handles multiple concurrent requests via async I/O. While one request waits for the llama.cpp backend, others can proceed. The transformation logic is fast (~microseconds) and won't bottleneck.

**Good for:** Single user, development, or when llama.cpp is the bottleneck anyway.

### Multi-Worker Mode (Production)

For multiple concurrent users or higher throughput:

```bash
# Using uvicorn directly with multiple workers
python -m uvicorn transform_proxy:app --workers 4 --host 0.0.0.0 --port 4000

# Or using gunicorn (more production-ready)
gunicorn transform_proxy:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:4000
```

| Mode | Workers | Concurrent Requests | Use Case |
|------|---------|---------------------|----------|
| Default | 1 (async) | Many (I/O bound) | Dev, single user |
| `--workers 4` | 4 processes | 4x throughput | Multiple users |
| gunicorn | N processes | Production scale | Team/production |

### Thread Safety

The proxy has **no shared mutable state** - each request is independent. Multi-worker mode is safe with no additional configuration.

### Bottleneck Analysis

```
Request → Proxy (fast) → llama.cpp (slow) → Proxy (fast) → Response
              │                  │                │
          ~1-5ms            1-60 sec          ~1-5ms
```

In practice, llama.cpp is almost always the bottleneck. Multiple proxy workers only help if:
- llama.cpp has multiple slots (`-np 2` or higher)
- You're running multiple llama.cpp instances
- Transformation parsing becomes measurable (very large responses)

---

## Transform vs Retry: When to Use Each

### Transform Proxy (`transform_proxy.py`) ✅ Recommended

**Use when:**
- Model consistently produces the same malformed format
- You want zero additional latency
- You want to minimize token costs
- The malformed output is structurally parseable

**How it works:**
```
Request → Backend → Malformed Response → Parse → Transform → Client
                                            ↓
                                    (no retry needed)
```

### Retry Proxy (`proxy.py`)

**Use when:**
- Transformation fails (unparseable format)
- You want the model to learn the correct format via prompting
- Malformed patterns are inconsistent/unpredictable

**How it works:**
```
Request → Backend → Malformed Response → Detect → Retry with hints → Client
                                            ↓              ↓
                                    (2-3x latency)  (2-3x tokens)
```

---

## Understanding Streaming vs Non-Streaming

### What is Streaming?

When you make an API request to an LLM, you can choose between two modes:

| Mode | How it works | User experience |
|------|--------------|-----------------|
| **Non-streaming** | Server generates entire response, sends it all at once | User waits, then sees complete response |
| **Streaming** | Server sends tokens as they're generated via SSE | User sees response appear word-by-word |

### How is Streaming Controlled?

Streaming is controlled by the `stream` parameter in the API request:

```json
// Non-streaming request
{
  "model": "qwen3-coder",
  "messages": [{"role": "user", "content": "Hello"}],
  "stream": false
}

// Streaming request
{
  "model": "qwen3-coder", 
  "messages": [{"role": "user", "content": "Hello"}],
  "stream": true
}
```

### Who Controls Streaming?

**The client application controls streaming, not you directly.** Here's where different tools set this:

| Tool | Streaming Default | How to Change |
|------|-------------------|---------------|
| **bolt.diy** | `true` (streaming) | Check settings, may have toggle |
| **Cursor** | `true` (streaming) | Usually not configurable |
| **OpenCode** | `true` (streaming) | `--no-stream` flag or config |
| **Zed** | `true` (streaming) | Check assistant settings |
| **cURL/scripts** | Your choice | Set `"stream": false` in request |

### Why Streaming Complicates Retry Logic

With non-streaming:
```
Client → Proxy → Backend
                    ↓
             Generate full response
                    ↓
             Return to proxy
                    ↓
         Proxy validates response
                    ↓
    If malformed: retry with new request
                    ↓
         Return valid response to client
```

With streaming:
```
Client → Proxy → Backend
                    ↓
             Start generating tokens
                    ↓
    Token 1 → Token 2 → Token 3 → ... → Token N
         ↑
    Problem: Can't validate until we see the whole response!
```

### Proxy Streaming Strategies

This proxy implements **buffer-and-replay** for streaming:

```
1. Client requests streaming response
2. Proxy intercepts and collects ALL tokens from backend
3. Once complete, proxy validates the full response
4. If valid: replay collected tokens to client (appears as stream)
5. If invalid: retry with backend, then replay successful response
```

**Tradeoff:** This adds latency equal to the full generation time. The client won't see any tokens until the entire response is generated and validated.

#### Alternative Strategies (Not Implemented)

| Strategy | Description | Pros | Cons |
|----------|-------------|------|------|
| **Passthrough** | Stream directly, no validation | No latency | No retry capability |
| **Early abort** | Detect malformed pattern mid-stream, abort, retry | Some latency savings | Complex, may false positive |
| **Hybrid** | Stream until tool call detected, then buffer | Best of both | Very complex |

---

## Configuration

### Command Line Options

```bash
python proxy.py [OPTIONS]

Options:
  --backend, -b URL     Backend llama.cpp server URL (default: http://localhost:8080)
  --port, -p PORT       Port to listen on (default: 4000)
  --host HOST           Host to bind to (default: 0.0.0.0)
  --max-retries, -r N   Maximum retry attempts (default: 2)
  --no-hints            Disable retry hint injection
  --debug               Enable debug logging
```

### Examples

```bash
# Basic usage
python proxy.py

# Custom backend and port
python proxy.py --backend http://192.168.1.100:8080 --port 5000

# More retries, debug logging
python proxy.py --max-retries 3 --debug

# Disable hints (rely on temperature adjustment only)
python proxy.py --no-hints
```

### Configuring in Code

Edit `proxy.py` to modify the `ProxyConfig` class:

```python
@dataclass
class ProxyConfig:
    backend_url: str = "http://localhost:8080"
    port: int = 4000
    max_retries: int = 2
    retry_delay: float = 0.5
    
    # Add custom malformed patterns
    malformed_patterns: list = field(default_factory=lambda: [
        r'<function\s*=',
        r'<function\s+name\s*=',
        # Add your own patterns here
    ])
```

---

## Client Configuration

### bolt.diy

In your `.env` or settings:

```env
OPENAI_API_BASE_URL=http://localhost:4000/v1
# or
OLLAMA_API_BASE_URL=http://localhost:4000
```

Or in the UI, set the API endpoint to `http://localhost:4000`.

### Cursor

In Settings → Models → OpenAI API Base:

```
http://localhost:4000/v1
```

### Zed

In `settings.json`:

```json
{
  "language_models": {
    "openai": {
      "api_url": "http://localhost:4000"
    }
  }
}
```

### OpenCode

```bash
opencode --api-base http://localhost:4000/v1 --model local-model
```

### Direct API Calls

```bash
# Non-streaming (recommended for testing)
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-coder",
    "messages": [{"role": "user", "content": "Create a hello world file"}],
    "stream": false
  }'

# Streaming
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-coder", 
    "messages": [{"role": "user", "content": "Create a hello world file"}],
    "stream": true
  }'
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Your Machine                            │
│                                                                 │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────────┐ │
│  │   Client    │      │   Retry     │      │   llama.cpp     │ │
│  │  (bolt.diy) │ ───▶ │   Proxy     │ ───▶ │    Server       │ │
│  │             │      │  :4000      │      │    :8080        │ │
│  └─────────────┘      └─────────────┘      └─────────────────┘ │
│                              │                                  │
│                              ▼                                  │
│                     ┌───────────────┐                          │
│                     │   Validate    │                          │
│                     │   Response    │                          │
│                     └───────┬───────┘                          │
│                             │                                  │
│                    ┌────────┴────────┐                         │
│                    │                 │                         │
│                    ▼                 ▼                         │
│               [Valid]          [Malformed]                     │
│                  │                   │                         │
│                  ▼                   ▼                         │
│            Return to           Retry with                      │
│             client              hints                          │
│                                                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Main endpoint with retry logic |
| `/v1/models` | GET | Proxied to backend |
| `/health` | GET | Proxy health check |
| `/proxy/stats` | GET | Proxy statistics |
| `/*` | * | All other endpoints passed through |

---

## How Retry Logic Works

### Detection

The proxy looks for these malformed patterns:

1. `<function=` or `<function name=` without `<tool_call>` wrapper
2. `<function_call>` (wrong tag name)
3. Raw JSON tool calls like `{"name": "...", "arguments": ...}` without wrapper

### Retry Strategy

1. **Attempt 1**: Normal request
2. **Attempt 2** (if malformed):
   - Add formatting reminder to system prompt
   - Lower temperature by 0.1
3. **Attempt 3** (if still malformed):
   - Add explicit correction with snippet of failed output
   - Lower temperature by 0.2 total
4. **Give up**: Return last response (some response is better than none)

### Why This Helps

- **System prompt hints**: Remind model of correct format
- **Lower temperature**: More deterministic output, more likely to follow format
- **Showing the error**: On second retry, showing what went wrong helps model self-correct

---

## Monitoring

### Health Check

```bash
curl http://localhost:4000/health
```

Response:
```json
{
  "status": "healthy",
  "backend_url": "http://localhost:8080",
  "backend_healthy": true,
  "config": {
    "max_retries": 2,
    "add_retry_hints": true
  }
}
```

### Logs

The proxy logs all retry attempts:

```
2026-01-20 10:30:15 - INFO - Received request: streaming=True, messages=5
2026-01-20 10:30:15 - INFO - Stream attempt 1/3
2026-01-20 10:30:18 - WARNING - Stream attempt 1: Malformed - Found malformed pattern '<function\s*=' without valid wrapper
2026-01-20 10:30:18 - INFO - Retry 1: Adjusted temperature 0.7 -> 0.6
2026-01-20 10:30:18 - INFO - Stream attempt 2/3
2026-01-20 10:30:21 - INFO - Stream attempt 2: Valid, replaying to client
```

---

## Performance Considerations

### Latency Impact

| Scenario | Additional Latency |
|----------|-------------------|
| Valid response, non-streaming | ~0ms (just passthrough) |
| Valid response, streaming | Full generation time (buffer-and-replay) |
| Retry needed | 2-3x generation time |

### Memory Usage

For streaming with buffer-and-replay, the entire response is held in memory. For typical coding tasks (1-10KB responses), this is negligible.

### When to Use This vs Fix the Template

This proxy is a **workaround**, not a permanent solution. Consider:

- Use the proxy when you can't modify the model's chat template
- If you have control over the template, fix it at the source
- Our optimized chat template achieved 97-100% success rates, potentially making this proxy unnecessary

---

## Systemd Service Installation

### User Service (No Root Required)

This runs the proxy as your user, starts on login:

```bash
# Create user systemd directory
mkdir -p ~/.config/systemd/user

# Copy the service file
cp llm-proxy.user.service ~/.config/systemd/user/llm-proxy.service

# Edit to match your setup (paths, backend URL, etc.)
nano ~/.config/systemd/user/llm-proxy.service

# Reload systemd
systemctl --user daemon-reload

# Enable (start on login) and start
systemctl --user enable --now llm-proxy

# Check status
systemctl --user status llm-proxy

# View logs
journalctl --user -u llm-proxy -f
```

### System Service (Runs as Dedicated User)

For always-on servers or multi-user setups:

```bash
# Install the proxy files
sudo mkdir -p /opt/llm-proxy
sudo cp transform_proxy.py proxy.py /opt/llm-proxy/
sudo cp requirements.txt /opt/llm-proxy/

# Install Python deps (or use nix)
sudo pip install -r /opt/llm-proxy/requirements.txt

# Install the service
sudo cp llm-proxy.service /etc/systemd/system/

# Edit configuration
sudo nano /etc/systemd/system/llm-proxy.service

# Reload, enable, start
sudo systemctl daemon-reload
sudo systemctl enable --now llm-proxy

# Check status
sudo systemctl status llm-proxy

# View logs
sudo journalctl -u llm-proxy -f
```

### Configuration Options

Edit the `Environment=` lines in the service file:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROXY_BACKEND` | `http://localhost:8080` | llama.cpp server URL |
| `LLM_PROXY_PORT` | `4000` | Port to listen on |
| `LLM_PROXY_HOST` | `127.0.0.1` | Bind address (`0.0.0.0` for all interfaces) |

Or use an environment file:

```bash
# /etc/llm-proxy/config (system) or ~/.config/llm-proxy/config (user)
LLM_PROXY_BACKEND=http://localhost:8080
LLM_PROXY_PORT=4000
LLM_PROXY_HOST=127.0.0.1
```

Then uncomment `EnvironmentFile=` in the service file.

### Service Management Commands

```bash
# User service
systemctl --user start llm-proxy
systemctl --user stop llm-proxy
systemctl --user restart llm-proxy
systemctl --user status llm-proxy
journalctl --user -u llm-proxy -f

# System service (add sudo)
sudo systemctl start llm-proxy
sudo systemctl stop llm-proxy
sudo systemctl restart llm-proxy
sudo systemctl status llm-proxy
sudo journalctl -u llm-proxy -f
```

---

## Troubleshooting

### Proxy won't start

```bash
# Check if port is in use
lsof -i :4000

# Try a different port
python proxy.py --port 5000
```

### Backend connection refused

```bash
# Verify llama.cpp is running
curl http://localhost:8080/health

# Check backend URL
python proxy.py --backend http://127.0.0.1:8080
```

### Still getting malformed responses

1. Enable debug logging: `python proxy.py --debug`
2. Check if your malformed pattern is detected
3. Add custom patterns to `ProxyConfig.malformed_patterns`
4. Increase retries: `python proxy.py --max-retries 3`

### High latency with streaming

This is expected with buffer-and-replay. Options:
- Accept the latency (validation is worth it)
- Configure client to use non-streaming if possible
- Disable proxy for this client and rely on template fixes

---

## Extending the Proxy

### Adding Custom Detection Patterns

```python
config.malformed_patterns.append(r'your-pattern-here')
```

### Custom Retry Logic

Modify `augment_messages_for_retry()` to customize how messages are modified on retry.

### Metrics Collection

The `/proxy/stats` endpoint is a placeholder. Implement counters for:
- Total requests
- Retry rate
- Success rate by attempt number
- Average latency
